# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random

import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def collate_data_and_cast(
    samples,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    free_shapes=False,
    patch_size=14,
):
    # samples dict_keys(['global_crops', 'global_crops_teacher', 'local_crops', 'offsets'])
    # from b (nb_crops c h w) OR b (nb_crops c p nxp) to (b nc) c p np
    # nb crops goes with batch size ie: b nc -> (b nc)
    if isinstance(samples[0], dict):  # on gpu and with free_shapes
        nc, c, h, w = samples[0]["global_crops"].size()
        collated_global_crops = [
            rearrange(
                el["global_crops"],
                "nc c (p1 h1) (p2 w1) -> (h1 w1 p1) nc c p2",
                p1=patch_size,
                p2=patch_size,
                c=c,
            )
            for el in samples
        ]
        print("min shape b4 pad: ", min([el.shape[0] for el in collated_global_crops]))
        collated_global_crops = pad_sequence(collated_global_crops, batch_first=True)
        print(collated_global_crops.shape)  # 4, 4004, 2, 3, 14
        collated_global_crops = rearrange(
            collated_global_crops, "b np nc c p -> (b nc) c p np", p=patch_size
        )
        collated_local_crops = map(
            lambda t: rearrange(t["local_crops"], "c np p -> np c p", p=patch_size),
            samples,
        )  # np = (np nc), have to put the unequal dim (np) first for pad_sequence()
        collated_local_crops = pad_sequence(collated_local_crops, batch_first=True)
        collated_local_crops = rearrange(
            collated_local_crops, "b np c p -> b c p np", p=patch_size
        )
        print("glob, loc", collated_global_crops.shape, collated_local_crops.shape)
        B = collated_global_crops.size(0)

    else:  # on cpu
        n_global_crops = len(samples[0][0]["global_crops"])
        n_local_crops = len(samples[0][0]["local_crops"])

        collated_global_crops = torch.stack(
            [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples]
        ).to(dtype)
        collated_local_crops = torch.stack(
            [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples]
        ).to(dtype)

        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        if free_shapes:
            mask_generator.set_shape(
                collated_global_crops[i].shape[1] // patch_size,
                collated_global_crops[i].shape[2] // patch_size,
            )
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))
            )
        )
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # not possible if global crops of diff sizes

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
