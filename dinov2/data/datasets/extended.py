# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import sys
from typing import Any, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, decode_image

from .decoders import ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Any, Any], torch.Tensor, Image.Image]:
        img_bytes = self.get_image_data(index)
        try:
            # have to copy bc stream not writeable
            image = torch.frombuffer(np.copy(img_bytes), dtype=torch.uint8)

            # print(f"aa {image.shape} {image.dtype}")
            if "plankton" in str(self.root):
                image = decode_image(image, ImageReadMode.RGB)
            else:
                image_size = int(np.sqrt(image.shape[0] / 3))
                image = image.reshape(3, image_size, image_size)
            image = (image / 255.0).to(torch.float32)
        except Exception as e:
            print(e)
            print("Error: torch.frombuffer failed, trying PIL...", file=sys.stderr)
            try:
                image = ImageDataDecoder(img_bytes).decode()
            except Exception as e:
                raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
