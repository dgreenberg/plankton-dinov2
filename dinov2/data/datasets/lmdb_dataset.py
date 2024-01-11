import glob
import json
import os
from enum import Enum
from typing import Union, Optional, Tuple

import h5py
import lmdb
import numpy as np

from dinov2.data.datasets import ImageNet

_TargetLMDBDataset = int


class _SplitLMDBDataset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split
    ALL = "all"


class HDF5Dataset(ImageNet):
    Target = _TargetLMDBDataset
    Split = _SplitLMDBDataset
    lmdb_handles = {}

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        entry = entries[index]
        print(entry.keys(), flush=True)
        image_relpath = entry["path"]
        hdf5_path = entry["hdf5_file"]
        hdf5_file = self.hdf5_handles[hdf5_path]
        image_data = hdf5_file[image_relpath][()]
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_id"]
        return None if self.split == _SplitLMDBDataset.TEST else int(class_index)

    def get_class_ids(self) -> np.ndarray:
        self._get_entries()
        return self._class_ids

    @property
    def _entries_path(self) -> str:
        if self.root.endswith("TRAIN") or self.root.endswith("VAL"):  # if we have a single file
            return self.root
        if self._split.value.upper() == "ALL":
            return "*"
        else:
            return f"-{self._split.value.upper()}"

    def _get_extra_full_path(self, extra_path: str) -> str:
        print(f"root: {self.root}, extra_root: {self._extra_root}, extra_path: {extra_path}")
        if extra_path is None:
            extra_path = ""
        if os.path.isfile(self.root):
            return self.root
        else:
            return os.path.join(self.root, self._extra_root + extra_path)

    def _get_entries(self) -> list:
        if self._entries is None:
            self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        file_list = glob.glob(extra_full_path)
        print("Datasets file list: ", file_list)

        accumulated = []
        class_ids = []

        if self.do_short_run:
            file_list = file_list[:1]
        for lmdb_path in file_list:
            lmdb_env = lmdb.open(lmdb_path)  # "/home/jluesch/Documents/data/plankton/lmdb/2007-TRAIN")

            lmdb_txn = lmdb_env.begin()
            lmdb_cursor = lmdb_txn.cursor()

            self.lmdb_handles[lmdb_cursor] = file

            # Add the HDF5 file name to each entry and accumulate the file entries
            for key, value in lmdb_cursor:
                print(int(key.decode()))
                entry = int(key.decode())
                entry["hdf5_file"] = hdf5_file
                accumulated.append(entry)

        print(f"#unique_class_ids: {self._split}, {len(len(set(accumulated)))}")

        self._entries = accumulated
        self._class_ids = class_ids

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.hdf5_handles.values():
            handle.close()
