# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from kornia import augmentation
from torchvision import transforms
from kornia.constants import Resample
from kornia.augmentation.container import AugmentationSequential

from .transforms import (
    GaussianBlur,
    KorniaGaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        do_transform_on_gpu=False,
        use_kornia=False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.do_transform_on_gpu = do_transform_on_gpu
        self.use_kornia = use_kornia

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        ######## Kornia
        if self.use_kornia and self.do_transform_on_gpu:
            global_crops_size = (global_crops_size, global_crops_size)
            local_crops_size = (local_crops_size, local_crops_size)
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
                augmentation.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, same_on_batch=False, p=0.8
                ),
                augmentation.RandomGrayscale(p=0.2),
            )

            global_transfo1_extra = KorniaGaussianBlur(p=1.0)

            global_transfo2_extra = AugmentationSequential(
                KorniaGaussianBlur(p=0.1),
                augmentation.RandomSolarize(thresholds=0.5, additions=0.1, same_on_batch=False, p=0.2),
            )

            local_transfo_extra = KorniaGaussianBlur(p=1.0)

            # normalization
            self.normalize = make_normalize_transform(use_kornia=True)

            self.global_transfo1 = AugmentationSequential(color_jittering, global_transfo1_extra, self.normalize)
            self.global_transfo2 = AugmentationSequential(color_jittering, global_transfo2_extra, self.normalize)
            self.local_transfo = AugmentationSequential(color_jittering, local_transfo_extra, self.normalize)

        ########
        else:
            # random resized crop and flip
            self.geometric_augmentation_global = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        global_crops_size,
                        scale=global_crops_scale,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

            self.geometric_augmentation_local = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

            # color distorsions / blurring
            color_jittering = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

            global_transfo1_extra = GaussianBlur(p=1.0)

            if self.do_transform_on_gpu:
                solarize_threshold = 0.5
            else:
                solarize_threshold = 128
            global_transfo2_extra = transforms.Compose(
                [
                    GaussianBlur(p=0.1),
                    transforms.RandomSolarize(threshold=solarize_threshold, p=0.2),
                ]
            )

            local_transfo_extra = GaussianBlur(p=0.5)

            # normalization
            if not self.do_transform_on_gpu:  # if this is done on cpu, we have PIL Image, so to tensor
                self.normalize = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                    ]
                )
            else:  # if on gpu, the images are already tensors
                self.normalize = make_normalize_transform()

            self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
