# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random

import torch


def collate_data_and_cast(
    samples,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
):
    # dtype = torch.half  # TODO: Remove

    # samples dict_keys(['global_crops', 'global_crops_teacher', 'local_crops', 'offsets'])
    # from (nb_crops, b, c, h, w) to (nb_crops * b, c, h, w)
    if isinstance(samples, dict):  # on gpu
        collated_global_crops = torch.cat(samples["global_crops"], dim=0)
        collated_local_crops = torch.cat(samples["local_crops"], dim=0)
        collated_global_crops_teacher = torch.cat(
            samples["global_crops_teacher"], dim=0
        )

    else:  # on cpu
        n_global_crops = len(samples[0][0]["global_crops"])
        n_local_crops = len(samples[0][0]["local_crops"])
        n_global_crops_teacher = len(samples[0][0]["global_crops_teacher"])

        collated_global_crops = torch.stack(
            [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples]
        )
        collated_local_crops = torch.stack(
            [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples]
        )
        collated_global_crops_teacher = torch.stack(
            [
                s[0]["global_crops_teacher"][i]
                for i in range(n_global_crops_teacher)
                for s in samples
            ]
        )

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))
            )
        )
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

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
        "collated_global_crops_teacher": collated_global_crops_teacher.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
