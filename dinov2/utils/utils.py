# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import re
import subprocess
import sys
from typing import Union
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import InterpolationMode, resize

logger = logging.getLogger("dinov2")


def exists(val):
    if isinstance(val, list):
        return all([exists(el) for el in val])
    return val is not None


def resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
    """Resize pos_embed weights.

    Resize pos_embed using bicubic interpolate method.
    Args:
        pos_embed (torch.Tensor): Position embedding weights.
        input_shpae (tuple): Tuple for (downsampled input image height,
            downsampled input image width).
        pos_shape (tuple): The resolution of downsampled origin training
            image.
        mode (str): Algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
    Return:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    assert pos_embed.ndim == 3, "shape of pos_embed must be [1, L, C]"
    if isinstance(pos_shape, tuple):
        pos_h, pos_w = pos_shape
    else:
        pos_h, pos_w = (pos_shape, pos_shape)

    if not isinstance(input_shape, tuple):
        input_shape = (input_shape, input_shape)

    # keep dim for easy deployment
    cls_token_weight = pos_embed[:, 0:1]
    pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
    # pos_embed_weight = pos_embed[:, 1:] # not compatible w registers
    pos_embed_weight = pos_embed_weight.reshape(
        1, pos_h, pos_w, pos_embed.shape[2]
    ).permute(0, 3, 1, 2)
    pos_embed_weight = resize(
        pos_embed_weight,
        size=input_shape,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
    return pos_embed


def match_pos_embeds(
    pos_embeds_ref: torch.Tensor,
    pos_embeds_loaded: torch.Tensor,
    img_shape: Union[tuple, int],
    loaded_img_shape: Union[tuple, int],
) -> torch.Tensor:
    if pos_embeds_loaded.flatten().shape == pos_embeds_ref.flatten().shape:
        pos_embeds_loaded = pos_embeds_loaded.reshape(pos_embeds_ref.shape)
    else:
        print(
            "Positional embeddings have different shapes, matching them...",
            end=" ",
        )
        print(
            "pos_embed_ref: ",
            pos_embeds_ref.shape,
            " pos_embed_loaded: ",
            pos_embeds_loaded.shape,
        )
        pos_embeds_loaded = resize_pos_embed(
            pos_embeds_loaded,
            input_shape=img_shape,
            pos_shape=loaded_img_shape,
            mode="bicubic",
        )
    return pos_embeds_loaded


def match_state_dict_keys(state_dict, keys_load, keys_model):
    print("Adapting keys")
    new_state_dict = dict()
    double_nb_ptn = "blocks.[0-9]{1,2}.([0-9]{1,2}.[a-z0-9_\.]+)"

    nb_double_ptn = len([k1 for k1 in keys_load if re.search(double_nb_ptn, k1)])
    if nb_double_ptn / len(keys_load) > 0.2:
        # replace doubly numbered blocks in loaded dict by single block number
        new_state_dict = {
            (
                ("blocks." + re.search(double_nb_ptn, k).group(1))
                if re.search(double_nb_ptn, k)
                else k
            ): v
            for k, v in state_dict.items()
        }
    else:  # inverse, replace single in load by double pttn
        for k1 in keys_model:
            match = re.search(double_nb_ptn, k1)
            if match:
                single_key = "blocks." + match.group(1)
                if single_key in keys_load:
                    new_state_dict[k1] = state_dict[single_key]
            else:
                new_state_dict[k1] = state_dict[k1]
    return new_state_dict


def load_pretrained_weights(
    model, pretrained_weights, checkpoint_key, teacher_student_key="teacher"
):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights, map_location="cpu"
        )
    else:
        chkpt = torch.load(pretrained_weights, map_location=torch.device("cpu"))
        if "model" in chkpt.keys():
            state_dict = chkpt["model"]
        else:
            state_dict = chkpt

    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    if teacher_student_key == "teacher":
        state_dict = {k: v for k, v in state_dict.items() if "student" not in k}
        state_dict = {
            k.replace("teacher.", ""): v for k, v in state_dict.items()
        }  # we take teacher for eval
    elif teacher_student_key == "student":
        state_dict = {k: v for k, v in state_dict.items() if "teacher" not in k}
        state_dict = {k.replace("student.", ""): v for k, v in state_dict.items()}
    else:
        print(
            f"Error: Key {teacher_student_key} not recognized, options are: 'student', 'teacher'"
        )
        sys.exit(1)

    keys_load = set(state_dict.keys())
    keys_model = set(model.state_dict().keys())
    if len(keys_load.intersection(keys_model)) / len(keys_model) < 0.6:
        state_dict = match_state_dict_keys(state_dict, keys_load, keys_model)

    if "pos_embed" in model.state_dict().keys() and "pos_embed" in state_dict.keys():
        loaded_img_shape = int(np.sqrt(state_dict["pos_embed"].shape[1] - 1))
        state_dict["pos_embed"] = match_pos_embeds(
            pos_embeds_ref=model.state_dict()["pos_embed"],
            pos_embeds_loaded=state_dict["pos_embed"],
            img_shape=model.img_size // model.patch_size,
            loaded_img_shape=loaded_img_shape,
        )

    # shape loaded state_dict like model state_dict
    state_dict = {
        k_c: (
            v_c.reshape(model.state_dict()[k_c].shape)
            if k_c in model.state_dict().keys()
            else v_c
        )
        for k_c, v_c in state_dict.items()
    }
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
