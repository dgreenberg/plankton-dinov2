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
    attn_mask_lc = attn_mask_gc = None
    # samples dict_keys(['global_crops', 'global_crops_teacher', 'local_crops', 'offsets'])
    # from b (nb_crops c h w) OR b (nb_crops c p nxp) to (b nc) c p np
    # nb crops goes with batch size ie: b nc -> (b nc)
    if isinstance(samples, dict):
        B, nc, c, h, w = samples["global_crops"].size()
        collated_global_crops = rearrange(
            samples["global_crops"],
            "b nc c h w -> (b nc) c h w",
            c=c,
            b=B,
        )
        collated_local_crops = rearrange(
            samples["local_crops"],
            "b nc c h w -> (b nc) c h w",
            c=c,
            b=B,
        )
        B = nc * B  # we define a new pseudo batch size

    elif isinstance(samples[0], dict):  # on gpu and with free_shapes
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
        """
        gc_len_list = [
            el.shape[0] // patch_size + 1  # + 1 for cls token
            for el in collated_global_crops
            for _ in range(el.shape[1])
        ]  # len=B*nc
        """
        collated_global_crops = pad_sequence(collated_global_crops, batch_first=True)
        # gc_padded_len = collated_global_crops.shape[1] // patch_size + 1
        collated_global_crops = rearrange(
            collated_global_crops, "b np nc c p -> (b nc) c p np", p=patch_size
        )
        # LOCAL CROPS
        collated_local_crops = map(
            lambda t: rearrange(t["local_crops"], "c np p -> np c p", p=patch_size),
            samples,
        )  # np = (np nc), have to put the unequal dim (np) first for pad_sequence()
        collated_local_crops = pad_sequence(collated_local_crops, batch_first=True)
        # lc_padded_len = collated_local_crops.shape[1] // patch_size + 1
        # lc_len_list = [
        #    [el // patch_size + 1 for el in el["local_crop_len"]] for el in samples
        # ]
        collated_local_crops = rearrange(
            collated_local_crops, "b np c p -> b c p np", p=patch_size
        )
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

    """
    if free_shapes:  # masking attention to prohibit cross-attending across patches
        attn_mask_gc = [
            torch.block_diag(
                torch.ones((len, len)),
                torch.zeros((gc_padded_len - len, gc_padded_len - len)),
            )
            if gc_padded_len > len
            else torch.ones((len, len))
            for len in gc_len_list
        ]
        attn_mask_gc = torch.stack(attn_mask_gc).bool()

        attn_mask_lc = []
        for lc_lens in lc_len_list:
            A = [torch.ones((el, el)) for el in lc_lens]
            pad_len = lc_padded_len - sum(lc_lens)
            pad_border = torch.zeros((pad_len, pad_len))
            attn_mask_lc.append(torch.block_diag(*A, pad_border))
        attn_mask_lc = torch.stack(attn_mask_lc).bool()
        '''
        attn_mask = rearrange(batched_image_ids, "b i -> b 1 i 1") == rearrange(
            batched_image_ids, "b j -> b 1 1 j"
        )
        attn_mask = attn_mask & rearrange(key_pad_mask, "b j -> b 1 1 j")
        '''
    else:
        attn_mask_lc = attn_mask_gc = None
    """
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
        "attn_mask_gc": attn_mask_gc,
        "attn_mask_lc": attn_mask_lc,
    }
