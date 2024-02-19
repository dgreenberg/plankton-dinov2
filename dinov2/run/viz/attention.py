# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as pth_transforms

from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.models.vision_transformer import vit_small
from dinov2.utils.config import setup


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name for the wandb log",
        default="viz_attn_run",
    )
    return parser


def main(args):
    # image_size = (952, 952)
    output_dir = "."
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    imagenet = False

    if imagenet:
        # imagenet pretrained
        model = vit_small(patch_size=14, img_size=224, block_chunks=0)
        model.load_state_dict(
            torch.load(
                "/home/hgf_mdc/hgf_ysb1444/checkpoints/dinov2_vits14_pretrain.pth"
            )
        )

    else:
        # our model
        config = setup(args, do_eval=True)
        model = build_model_for_eval(config, args.pretrained_weights)

    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    orig_img = Image.open(
        # "/home/nkoreub/Documents/Projects/plankton-dinov2/data/raw/2011/Bidulphia/IFCB5_2011_318_031734_02892.png"
        "/home/nkoreub/Documents/Projects/plankton-dinov2/data/raw/2011/Pseudonitzschia/IFCB1_2011_232_170633_00210.png"
    )
    orig_img = orig_img.convert("RGB")
    image_size = (int(1800*(orig_img.size[1]/orig_img.size[0])), 1800)

    if imagenet:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    else:
        MEAN = [0.68622917, 0.68622917, 0.68622917]
        STD = [0.10176649, 0.10176649, 0.10176649]

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(MEAN, STD),
        ]
    )
    img = transform(orig_img)

    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1]  # number of heads

    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    attentions = np.append(
        attentions, np.mean(attentions, axis=0)[np.newaxis, :, :], axis=0
    )

    # save attention head heatmaps
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    titles = [
        "Image",
        "Head 0",
        "Head 1",
        "Head 2",
        "Head Average",
        "Head 3",
        "Head 4",
        "Head 5",
    ]

    images = [
        orig_img,
        attentions[0],
        attentions[1],
        attentions[2],
        attentions[-1],
        attentions[3],
        attentions[4],
        attentions[6],
    ]

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    print('saving to ', os.path.join(output_dir, "attn-viz_1" + ".png"))
    plt.savefig(os.path.join(output_dir, "attn-viz_1" + ".png"))


if __name__ == "__main__":
    description = "DINOv2 attention visualization"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
 