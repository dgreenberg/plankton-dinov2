# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Union
import sys
from typing import Any, Tuple
import numpy as np
from PIL import Image
import torch

from torchvision.io import decode_image, ImageReadMode
from torchvision.datasets import VisionDataset

from .decoders import TargetDecoder, ImageDataDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], torch.Tensor, Image.Image]:
        img_bytes = self.get_image_data(index)
        try:
            # have to copy bc stream not writeable
            image = torch.frombuffer(np.copy(img_bytes), dtype=torch.uint8)
            image = decode_image(image, ImageReadMode.RGB)
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

        else:
            return image, target

    def __len__(self) -> int:
        raise NotImplementedError
