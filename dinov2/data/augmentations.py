# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import numpy as np
import torch
from einops import rearrange
from kornia import augmentation
from kornia.augmentation.container import AugmentationSequential
from kornia.constants import Resample
from skimage.segmentation import felzenszwalb, quickshift, slic
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from .transforms import (
    GaussianBlur,
    KorniaGaussianBlur,
    MaybeToTensor,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")
MAX_PATCHES = 400


class SegmentationAlgo(Enum):
    FELZENSWALB = "felzenswalb"
    SLIC = "slic"


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        use_kornia=False,
        use_native_res=False,
        patch_size=14,
        do_seg_crops=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.use_kornia = use_kornia
        self.use_native_res = use_native_res
        if self.use_native_res:
            self.global_crops_area = self.global_crops_size * self.global_crops_size
        self.patch_size = patch_size
        self.do_seg_crops = do_seg_crops

        self.patch_maxpool_op = torch.nn.MaxPool2d(
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            dilation=1,
            return_indices=False,
        )

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        ######## Kornia
        if self.use_kornia:
            global_crops_size = (global_crops_size, global_crops_size)
            local_crops_size = (local_crops_size, local_crops_size)

            if self.use_native_res:
                self.random_crop = augmentation.RandomCrop(
                    global_crops_size,
                    resample=Resample.BILINEAR.name,
                    same_on_batch=False,
                    p=1.0,
                    keepdim=False,
                )
                self.geometric_augmentation_global = augmentation.RandomHorizontalFlip(
                    p=0.5, p_batch=1.0
                )
                self.geometric_augmentation_local = AugmentationSequential(
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                    data_keys=["input", "input"],
                )

                self.std_augmentation_local = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )
            else:
                self.geometric_augmentation_global = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        global_crops_size,
                        scale=global_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )
                self.geometric_augmentation_local = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )

            # color distorsions / blurring
            color_jittering = AugmentationSequential(
                augmentation.ColorJiggle(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                    same_on_batch=False,
                    p=0.8,
                ),
                augmentation.RandomGrayscale(p=0.2),
            )

            global_transfo1_extra = KorniaGaussianBlur(p=1.0)

            global_transfo2_extra = AugmentationSequential(
                KorniaGaussianBlur(p=0.1),
                augmentation.RandomSolarize(
                    thresholds=0.5, additions=0.1, same_on_batch=False, p=0.2
                ),
            )

            local_transfo_extra = KorniaGaussianBlur(p=1.0)

            # normalization
            self.normalize = make_normalize_transform(use_kornia=True)

            self.global_transfo1 = AugmentationSequential(
                color_jittering, global_transfo1_extra, self.normalize
            )
            self.global_transfo2 = AugmentationSequential(
                color_jittering, global_transfo2_extra, self.normalize
            )
            self.local_transfo = AugmentationSequential(
                color_jittering, local_transfo_extra, self.normalize
            )

        ######## TORCHVISION
        else:
            # random resized crop and flip
            self.geometric_augmentation_global = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        global_crops_size,
                        scale=global_crops_scale,
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )

            self.geometric_augmentation_local = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )

            # color distorsions / blurring
            color_jittering = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    v2.RandomGrayscale(p=0.2),
                ]
            )

            global_transfo1_extra = GaussianBlur(p=1.0)

            solarize_threshold = 0.5
            global_transfo2_extra = v2.Compose(
                [
                    GaussianBlur(p=0.1),
                    v2.RandomSolarize(threshold=solarize_threshold, p=0.2),
                ]
            )

            local_transfo_extra = GaussianBlur(p=0.5)

            # normalization
            self.normalize = v2.Compose(
                [
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )

            self.global_transfo1 = v2.Compose(
                [color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose(
                [color_jittering, local_transfo_extra, self.normalize]
            )

    def round_up_patch_size(self, crop_len: int):
        if crop_len % self.patch_size == 0:
            return crop_len
        return crop_len + (self.patch_size - crop_len % self.patch_size)

    def round_down_patch_size(self, crop_len: int):
        if crop_len % self.patch_size == 0:
            return crop_len
        return crop_len - crop_len % self.patch_size

    def make_rectangle_crop(self, image):
        img_global_list = []
        for single_image in image:
            if (
                single_image.size(-1) > self.global_crops_size
                and single_image.size(-2) > self.global_crops_size
            ):
                single_image = self.random_crop(single_image).squeeze()
            else:
                H, W = single_image.size(-2), single_image.size(-1)
                min_dim_idx = np.argmin((H, W))
                crop_0 = (H, W)[min_dim_idx]
                crop_1 = self.global_crops_area // crop_0

                if min_dim_idx == 0:
                    crop_y = crop_0
                    crop_x = crop_1
                else:
                    crop_y = crop_1
                    crop_x = crop_0
                crop_y = self.round_up_patch_size(crop_y)
                crop_x = self.round_up_patch_size(crop_x)

                if single_image.size(-2) - crop_x < 0:
                    crop_x = single_image.size(-2)
                if single_image.size(-1) - crop_y < 0:
                    crop_y = single_image.size(-1)

                offset_x, offset_y = 0, 0
                if single_image.size(-2) - crop_x:
                    offset_x = np.random.randint(single_image.size(-2) - crop_x)

                if single_image.size(-1) - crop_y:
                    offset_y = np.random.randint(single_image.size(-1) - crop_y)

                single_image = single_image[
                    offset_x : offset_x + crop_x,
                    offset_y : offset_y + crop_y,
                ]
            img_global_list.append(single_image)

        return torch.stack(img_global_list)

    def make_seg_crops(self, image, seg_algo):
        # image : C H W
        """
        orig_img_dims = image.shape
        resize_op = v2.Resize(
            orig_img_dims[1:],
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )
        """
        c, h, w = image.shape
        if seg_algo == SegmentationAlgo.FELZENSWALB:
            segments = felzenszwalb(
                image.permute((1, 2, 0)), scale=300, sigma=0.5, min_size=1000
            )
        elif seg_algo == SegmentationAlgo.SLIC:
            n_segments = max(int((h * w) / (98 * 98)), 3)
            segments = slic(
                image.permute((1, 2, 0)),
                n_segments=n_segments,
                compactness=10,
                sigma=1.0,
                start_label=1,
            )
        else:
            raise NotImplementedError(
                f"Specified segmentation algorithm {seg_algo} not implemented"
            )

        def seg_to_patched_seg(segments, image_gray):
            segments = segments / (segments.max() + 1e-5)
            conv_input = torch.tensor(
                segments[None, :, :]
            )  # on color images segments can be 3 H W?
            pooled_seg = self.patch_maxpool_op(conv_input)
            # resized_masks = resize_op(pooled_seg)
            resized_masks = pooled_seg.repeat_interleave(
                self.patch_size, dim=1
            ).repeat_interleave(self.patch_size, dim=2)
            # assert torch.all(resized_masks_alt == resized_masks)

            resized_masks = resized_masks[0, :, :]
            nb_seg = len(np.unique(resized_masks))
            resized_masks_int = (resized_masks * nb_seg).to(torch.int32)

            # map tensor to continuous int values
            unique_vals = torch.unique(resized_masks_int)
            lt = torch.full((int(resized_masks_int.max()) + 1,), -1)
            lt[unique_vals] = torch.arange(len(unique_vals))
            resized_masks_int = lt[resized_masks_int]

            # update nb_seg
            nb_seg = len(np.unique(resized_masks_int))
            # First, create a mask for each segment
            masks = []
            for mask_idx in range(nb_seg):
                matching_mask = resized_masks_int == mask_idx
                if (
                    torch.numel(matching_mask > 0) > self.patch_size * self.patch_size
                ):  # possible that some masks are 0
                    masks.append(resized_masks_int == mask_idx)

            masks = torch.stack(masks)

            # Then, find the bounding boxes for each segment
            bboxes = masks_to_boxes(masks)  # (x1, y1, x2, y2)

            bounded_images, bounded_masks = [], []
            for mask_idx in range(nb_seg):
                mask = masks[mask_idx, :, :]
                bbox = bboxes[mask_idx].to(torch.int32)

                # first bbox crop, then segment
                bounded_mask = mask[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
                bounded_image = image_gray[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
                """
                # first seg then bbox (less mem eff)
                fragment = torch.zeros(mask.shape, dtype=torch.float32)
                fragment[mask] = image_gray[mask]
                fragment = fragment[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                """
                bounded_images.append(bounded_image)
                bounded_masks.append(bounded_mask)

            return bounded_images, bounded_masks, resized_masks_int, bboxes

        return seg_to_patched_seg(segments, image[0, :, :])

    def pad_to_patch_mutiple(self, image):
        # Can also make a version that pads down
        pad_y = self.patch_size - image.shape[-1] % self.patch_size
        pad_x = self.patch_size - image.shape[-2] % self.patch_size

        pad_x_l, pad_x_r = pad_x // 2, pad_x // 2 + pad_x % 2
        pad_y_l, pad_y_r = pad_y // 2, pad_y // 2 + pad_y % 2

        return torch.nn.functional.pad(
            image,
            pad=(
                pad_y_l,
                pad_y_r,
                pad_x_l,
                pad_x_r,
            ),
            mode="replicate",
        )

    def __call__(self, image):
        # image : C H W
        output = {}
        image = self.pad_to_patch_mutiple(image)
        # global crops:
        if self.use_native_res:
            image_global = self.make_rectangle_crop(image)
        else:
            image_global = image

        im1_base = self.geometric_augmentation_global(image_global)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image_global)
        global_crop_2 = self.global_transfo2(im2_base)

        def crop_to_patches(crop):
            # B is usually 1 here ?
            b, c, h, w = crop.size()

            patches = crop.unfold(-1, self.patch_size, self.patch_size).unfold(
                -3, self.patch_size, self.patch_size
            )
            patches = rearrange(
                patches,
                "b c n1 n2 p1 p2 -> b c (n1 n2 p2) p1",
                b=b,
                c=c,
                p1=self.patch_size,
                p2=self.patch_size,
            )
            return patches

        output["global_crops"] = torch.cat(
            [crop_to_patches(global_crop_1), crop_to_patches(global_crop_2)], dim=0
        )
        # print('output["global_crops"]', output["global_crops"].shape)

        # global crops for teacher:
        # output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        if self.do_seg_crops:
            local_crops, masks, pooled_seg, bboxes = self.make_seg_crops(
                image, seg_algo=self.do_seg_crops
            )
            # masks = [mask[None, :, :].repeat((3, 1, 1)) for mask in masks]

            local_crops, masks = list(
                zip(
                    *[
                        self.geometric_augmentation_local(
                            crop[None, :, :].repeat((3, 1, 1)),
                            mask[None, :, :].repeat((3, 1, 1)).to(torch.float32),
                        )
                        for crop, mask in zip(local_crops, masks)
                    ]
                )
            )
            masks = [
                mask[0, :, :] > 0  # to bool
                for mask in masks
            ]

            fragments = []
            for crop, mask in zip(local_crops, masks):
                fragment = torch.zeros_like(crop, dtype=torch.float32)
                fragment[:, mask] = crop[:, mask]
                fragments.append(fragment)

            def select_and_concat_nonzero_patches(local_crops, image):
                # Select non-zero 14x14 patches from local_crops and flat concat
                # Input: list N x [C H W]
                # Output: C P NxP
                exit_loop = False
                tot_patches = 0
                list_flat_patches = []
                crop_len_list = []
                for local_crop in local_crops:
                    if len(local_crop.shape) > 3:
                        local_crop = local_crop[0]
                    C, H, W = local_crop.shape
                    tot_patches += (H // self.patch_size) * (W // self.patch_size)

                    if tot_patches >= MAX_PATCHES:
                        if (
                            len(crop_len_list) <= 1
                        ):  # if seg fails, revert to std patching
                            patches = [
                                self.local_transfo(
                                    self.std_augmentation_local(image)
                                ).squeeze()
                                for _ in range(8)  # 8 is std local crops nb
                            ]
                            patches = torch.cat(
                                patches, dim=1
                            )  # c (n_c lc_size) lc_size

                            exit_loop = True
                        else:
                            # otherwise, we collected enought patches and exit the loop
                            break
                        # PROBLEM, sometimes seg fails and one crop has more than max patches

                    patches = local_crop.unfold(
                        1, self.patch_size, self.patch_size
                    ).unfold(2, self.patch_size, self.patch_size)
                    unfold_shape = patches.size()

                    # c x1 x2 p p
                    patches = patches.reshape(
                        C,
                        unfold_shape[1] * unfold_shape[2] * self.patch_size,
                        self.patch_size,
                    )  # c (n p) p

                    selected_patches = []
                    patches_list = patches.chunk(
                        patches.shape[1] // self.patch_size, dim=1
                    )
                    for patch in patches_list:  # C P P
                        if (
                            abs(torch.mean(patch)) > 1e-4
                        ):  # <= 1e-4 is probably uninformative
                            selected_patches.append(patch.transpose(-1, -2))

                    curr_nb_patches = len(selected_patches)
                    tot_patches += curr_nb_patches

                    crop_len_list.append(curr_nb_patches)

                    selected_patches = torch.cat(selected_patches, dim=1)  # c (n_p p) p

                    list_flat_patches.append(selected_patches)

                    if exit_loop:
                        break
                # list_flat_patches n_crop (c (n_p p) p)
                flat_patches = torch.cat(list_flat_patches, dim=1)  # c (n_crop n_p p) p
                # print("crop_len_list", crop_len_list, np.sum(crop_len_list))
                if flat_patches.size(1) > 6000:
                    print(
                        "WARNING: flat_patches too big",
                        flat_patches.shape,
                        "#",
                        len(list_flat_patches),
                        len(crop_len_list),
                    )
                return flat_patches, crop_len_list

            flat_patches, crop_len_list = select_and_concat_nonzero_patches(
                fragments, image=image
            )
            # output["local_crops_vis"] = fragments  # for visualization
            output["local_crops"] = flat_patches
            output["local_crop_len"] = crop_len_list
            # output["pooled_seg"] = pooled_seg
            crop_dims = torch.cat(
                [bboxes[:, 1:2] - bboxes[:, :0], bboxes[:, 2:3] - bboxes[:, 0:1]], dim=1
            )  # N (W H)
            output["local_crop_dims"] = crop_dims
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image))
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = torch.cat(local_crops, dim=0)
        output["offsets"] = ()
        return output
