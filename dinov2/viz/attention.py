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

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as pth_transforms

from dinov2.models.vision_transformer import vit_small

if __name__ == "__main__":
    image_size = (952, 952)
    # image_size = (234, 492)
    output_dir = "."
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model = vit_small(patch_size=14, img_size=526, init_values=1.0, block_chunks=0)

    model.load_state_dict(
        torch.load("/home/hgf_mdc/hgf_ysb1444/checkpoints/dinov2_vits14_pretrain.pth")
    )
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    img = Image.open(
        "/home/hgf_mdc/hgf_ysb1444/data/2007/Tontonia_gracillima/IFCB1_2007_141_111713_00338.png"
    )
    img = img.convert("RGB")
    transform = pth_transforms.Compose(
        [
            # pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                [0.68622917, 0.68622917, 0.68622917],
                [0.10176649, 0.10176649, 0.10176649],
            ),
        ]
    )
    img = transform(img)
    print(img.shape)

    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # weird: one pixel gets high attention over all heads?
    # print(torch.max(attentions, dim=1))
    # attentions[:, 283] = 0

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format="png")
        print(f"{fname} saved.")
