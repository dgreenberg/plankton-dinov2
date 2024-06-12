import argparse
import json
import os
import sys
from typing import OrderedDict

import imageio as io
import lmdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops_table
from tqdm import tqdm

BASE_DIR = "/fast/AG_Kainmueller/data/pan_m"  # max cluster path
# base_dir = "X:/data/pan_m" # local path
MAP_SIZE = int(1e12)  # 1TB
NUM_IMGS_PER_LMDB_FILE = 100


def mibi_breast_naming_conv(fov_path):
    base_dir_ = os.path.join(BASE_DIR, "mibi_breast")
    fov_name = os.path.basename(fov_path)
    deepcell_output_dir = os.path.join(base_dir_, "segmentation_data")
    return os.path.join(
        deepcell_output_dir, "deepcell_output", fov_name + "_feature_0.tif"
    )


def mibi_decidua_naming_conv(fov_path):
    base_dir_ = os.path.join(BASE_DIR, "mibi_decidua")
    fov_name = os.path.basename(fov_path)
    deepcell_output_dir = os.path.join(base_dir_, "segmentation_data")
    return os.path.join(deepcell_output_dir, fov_name + "_segmentation_labels.tiff")


def vectra_colon_naming_conv(fname):
    return os.path.join(
        BASE_DIR, "vectra_colon", "segmentation", fname + "feature_0.ome.tif"
    )


def vectra_pancreas_naming_conv(fname):
    return os.path.join(
        BASE_DIR, "vectra_pancreas", "segmentation", fname + "feature_0.ome.tif"
    )


def codex_colon_naming_conv(fname):
    fov, reg = fname.split("_")[:2]
    fov_path = os.path.join(BASE_DIR, "codex_colon", "masks", fov)
    images = os.listdir(fov_path)
    labels = [img for img in images if "_labeled" in img]
    labels = [img for img in labels if reg in img]
    label_fname = labels[0]
    return os.path.join(os.path.normpath(fov_path), label_fname)


naming_convention_dict = {
    "mibi_breast": mibi_breast_naming_conv,
    "mibi_decidua": mibi_decidua_naming_conv,
    "vectra_colon": vectra_colon_naming_conv,
    "vectra_pancreas": vectra_pancreas_naming_conv,
    "codex_colon": codex_colon_naming_conv,
}

selected_channels = {
    "mibi_breast": [
        "Calprotectin.tiff",
        "CD11c.tiff",
        "CD14.tiff",
        "CD163.tiff",
        "CD20.tiff",
        "CD3.tiff",
        "CD31.tiff",
        "CD38.tiff",
        "CD4.tiff",
        "CD45.tiff",
        "CD45RB.tiff",
        "CD45RO.tiff",
        "CD56.tiff",
        "CD57.tiff",
        "CD68.tiff",
        "CD69.tiff",
        "CD8.tiff",
        "ChyTr.tiff",
        "CK17.tiff",
        "Collagen1.tiff",
        "ECAD.tiff",
        "FAP.tiff",
        "Fibronectin.tiff",
        "FOXP3.tiff",
        "GLUT1.tiff",
        "H3K27me3.tiff",
        "H3K9ac.tiff",
        "HLA1.tiff",
        "HLADR.tiff",
        "IDO.tiff",
        "Ki67.tiff",
        "LAG3.tiff",
        "PD1.tiff",
        "PDL1.tiff",
        "SMA.tiff",
        "TBET.tiff",
        "TCF1.tiff",
        "TIM3.tiff",
        "Vim.tiff",
    ],
    "mibi_decidua": [
        "CD11b.tif",
        "CD11c.tif",
        "CD14.tif",
        "CD16.tif",
        "CD163.tif",
        "CD20.tif",
        "CD206.tif",
        "CD3.tif",
        "CD31.tif",
        "CD4.tif",
        "CD44.tif",
        "CD45.tif",
        "CD56.tif",
        "CD57.tif",
        "CD68.tif",
        "CD8.tif",
        "CD80.tif",
        "CK7.tif",
        "DCSIGN.tif",
        "Ecad.tif",
        "FoxP3.tif",
        "Galectin9.tif",
        "GrB.tif",
        "H3.tif",
        "HLADR.tif",
        "HLAG.tif",
        "HO1.tif",
        "ICOS.tif",
        "IDO.tif",
        "iNOS.tif",
        "Ki67.tif",
        "PD1.tif",
        "PDL1.tif",
        "SMA.tif",
        "TIGIT.tif",
        "TIM3.tif",
        "Tryptase.tif",
        "VIM.tif",
    ],
    "vectra_colon": [
        "CD3.ome.tif",
        "CD8.ome.tif",
        "DAPI.ome.tif",
        "Foxp3.ome.tif",
        "ICOS.ome.tif",
        "panCK+CK7+CAM5.2.ome.tif",
        "PD-L1.ome.tif",
    ],
    "vectra_pancreas": [
        "CD40-L.ome.tif",
        "CD40.ome.tif",
        "CD8.ome.tif",
        "DAPI.ome.tif",
        "panCK.ome.tif",
        "PD-1.ome.tif",
        "PD-L1.ome.tif",
    ],
    "codex_colon": [
        "aDefensin5.ome.tif",
        "aSMA.ome.tif",
        "BCL2.ome.tif",
        "CD117.ome.tif",
        "CD11c.ome.tif",
        "CD123.ome.tif",
        "CD127.ome.tif",
        "CD138.ome.tif",
        "CD15.ome.tif",
        "CD16.ome.tif",
        "CD161.ome.tif",
        "CD163.ome.tif",
        "CD19.ome.tif",
        "CD206.ome.tif",
        "CD21.ome.tif",
        "CD25.ome.tif",
        "CD3.ome.tif",
        "CD31.ome.tif",
        "CD34.ome.tif",
        "CD36.ome.tif",
        "CD38.ome.tif",
        "CD4.ome.tif",
        "CD44.ome.tif",
        "CD45.ome.tif",
        "CD45RO.ome.tif",
        "CD49a.ome.tif",
        "CD49f.ome.tif",
        "CD56.ome.tif",
        "CD57.ome.tif",
        "CD66.ome.tif",
        "CD68.ome.tif",
        "CD69.ome.tif",
        "CD7.ome.tif",
        "CD8.ome.tif",
        "CD90.ome.tif",
        "CollIV.ome.tif",
        "Cytokeratin.ome.tif",
        "DRAQ5.ome.tif",
        "FAP.ome.tif",
        "HLADR.ome.tif",
        "Ki67.ome.tif",
        "MUC1.ome.tif",
        "MUC2.ome.tif",
        "MUC6.ome.tif",
        "NKG2D.ome.tif",
        "OLFM4.ome.tif",
        "Podoplanin.ome.tif",
        "SOX9.ome.tif",
        "Synaptophysin.ome.tif",
        "Vimentin.ome.tif",
    ],
}

dataset_paths = {
    "mibi_breast": os.path.join(BASE_DIR, "mibi_breast", "image_data", "samples"),
    "mibi_decidua": os.path.join(BASE_DIR, "mibi_decidua", "image_data"),
    "vectra_colon": os.path.join(BASE_DIR, "vectra_colon", "raw_structured"),
    "vectra_pancreas": os.path.join(BASE_DIR, "vectra_pancreas", "raw_structured"),
    "codex_colon": os.path.join(BASE_DIR, "codex_colon", "raw_structured"),
}


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def change_lmdb_envs(
    dataset_lmdb_dir, file_idx, env_imgs=None, env_labels=None, env_metadata=None
):
    if env_imgs is not None:
        env_imgs.close()
        env_labels.close()
        env_metadata.close()

    lmdb_imgs_path = os.path.join(dataset_lmdb_dir, "images" + str(file_idx))
    lmdb_labels_path = os.path.join(dataset_lmdb_dir, "labels" + str(file_idx))
    lmdb_metadata_path = os.path.join(dataset_lmdb_dir, "metadata" + str(file_idx))
    os.makedirs(lmdb_imgs_path, exist_ok=True)
    os.makedirs(lmdb_labels_path, exist_ok=True)
    os.makedirs(lmdb_metadata_path, exist_ok=True)

    env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE)
    env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE)
    env_metadata = lmdb.open(lmdb_metadata_path, map_size=MAP_SIZE)
    return env_imgs, env_labels, env_metadata


def load_channel(channel_path):
    channel_img = io.v2.imread(channel_path)  # (1024, 1024)
    channel_img = normalize(np.squeeze(channel_img))
    return channel_img


def fov_to_lmdb_crops(
    img_idx,
    fov,
    file_idx,
    dataset_lmdb_dir,
    dataset,
    path,
    surrounding_size,
    n_jobs=8,
    env_imgs=None,
    env_labels=None,
    env_metadata=None,
):
    fov_name_cleaned = "".join(e for e in str(fov) if e.isalnum())
    if img_idx % NUM_IMGS_PER_LMDB_FILE == 0:
        print(
            f'idx: {img_idx}, dataset: {dataset}, path: {path}, fov: "{fov_name_cleaned}"'
        )
        print("Switching context to new lmdb")
        env_imgs, env_labels, env_metadata = change_lmdb_envs(
            dataset_lmdb_dir,
            file_idx=file_idx,
            env_imgs=env_imgs,
            env_labels=env_labels,
            env_metadata=env_metadata,
        )
        file_idx += 1

        txn_meta, txn_imgs, txn_labels = (
            env_metadata.begin(write=True),
            env_imgs.begin(write=True),
            env_labels.begin(write=True),
        )

    img_idx = f"{dataset}_{img_idx:04d}"
    metadata_dict = {}

    fov_path = os.path.join(path, fov)
    channels = selected_channels[dataset]
    channel_imgs = []
    channel_names = [channel.split(".")[0] for channel in channels]
    channel_paths = [os.path.join(fov_path, channel) for channel in channels]
    # use joblib to parallelize loading of channels
    channel_imgs = [load_channel(channel_path) for channel_path in channel_paths]
    # concatenate channel images
    multiplex_img = np.stack(channel_imgs, axis=0)
    # get segmentation mask
    naming_convention = naming_convention_dict[dataset]
    segmentation_path = naming_convention(fov)
    segmentation_mask = io.v2.imread(segmentation_path).squeeze().astype(np.uint16)

    idx_bytes = img_idx.encode("utf-8")
    txn_labels.put(idx_bytes, segmentation_mask.tobytes())

    metadata_dict["fov"] = fov
    metadata_dict["channel_names"] = channel_names
    metadata_bytes = json.dumps(metadata_dict).encode("utf-8")
    txn_meta.put(idx_bytes, metadata_bytes)

    # get regionprops
    regionprops = pd.DataFrame(
        regionprops_table(segmentation_mask, properties=("label", "centroid"))
    )
    # mirrorpad multiplex image to avoid edge effects
    # @Jerome alternatively, one could just drop regions that are too close to the edge
    multiplex_img = np.pad(
        multiplex_img,
        (
            (0, 0),
            (surrounding_size, surrounding_size),
            (surrounding_size, surrounding_size),
        ),
        mode="reflect",
    )
    regionprops["centroid-0"] = regionprops["centroid-0"] + surrounding_size
    regionprops["centroid-1"] = regionprops["centroid-1"] + surrounding_size
    # iterate over regions and extract patches surrounding centroids from multiplex image
    patches = [
        multiplex_img[
            :,
            int(region["centroid-0"] - surrounding_size) : int(
                region["centroid-0"] + surrounding_size
            ),
            int(region["centroid-1"] - surrounding_size) : int(
                region["centroid-1"] + surrounding_size
            ),
        ]
        for _, region in regionprops.iterrows()
    ]

    # save patch, label, fov, dataset and channel_names for each training sample
    print(f"Saving patches {len(patches)} for img {img_idx}")
    for p_idx, patch in enumerate(patches):
        patch_bytes = patch.tobytes()
        full_idx = f"{img_idx}_{p_idx:03d}"

        idx_bytes = str(full_idx).encode("utf-8")
        txn_imgs.put(idx_bytes, patch_bytes)


def main(args):
    start_fov_dix = args.start_fov_dix
    end_fov_idx = args.end_fov_idx

    patch_size = args.patch_size
    n_jobs = args.n_jobs
    surrounding_size = patch_size // 2
    sel_dataset_paths = {k: dataset_paths[k] for k in args.dataset_keys}

    base_lmdb_dir = BASE_DIR + "_lmdb"
    os.makedirs(base_lmdb_dir, exist_ok=True)

    for dataset, path in sel_dataset_paths.items():
        dataset_lmdb_dir = os.path.join(base_lmdb_dir, dataset)
        file_idx = 0

        env_imgs, env_labels, env_metadata = None, None, None
        file_idx += 1

        fovs = os.listdir(path)
        print(f"TOTAL #FOVS {len(fovs)} FOR DATASET {dataset}")
        Parallel(n_jobs=n_jobs)(
            delayed(fov_to_lmdb_crops)(
                img_idx,
                fov,
                file_idx,
                dataset_lmdb_dir,
                dataset,
                path,
                surrounding_size,
                n_jobs=n_jobs,
                env_imgs=None,
                env_labels=None,
                env_metadata=None,
            )
            for img_idx, fov in tqdm(enumerate(sorted(fovs)[start_fov_dix:end_fov_idx]))
        )

        env_imgs.close()
        env_metadata.close()
        env_labels.close()
        print(f"FINISHED DATASET {dataset}, SAVED AT: {dataset_lmdb_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        help="Patch size",
        default=224,
    )
    parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        type=int,
        help="Number of jobs to run in parallel",
        default=8,
    )
    parser.add_argument(
        "--do_test_run",
        action=argparse.BooleanOptionalAction,
        help="Toggle test run with small subset of dataset",
        default=False,
    )
    parser.add_argument(
        "--dataset_keys",
        nargs="+",
        help="""Names of datasets to process. One or more of the follwowing:
        mibi_breast, mibi_decidua, vectra_colon, vectra_pancreas, codex_colon""",
    )
    parser.add_argument(
        "--start_fov_dix",
        type=int,
        help="Start index of FOVs to process",
        default=0,
    )
    parser.add_argument(
        "--end_fov_dix",
        type=int,
        help="End index of FOVs to process",
        default=-1,
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))


# channel_imgs = Parallel(n_jobs=n_jobs)(
#    delayed(load_channel)(channel_path) for channel_path in channel_paths
# )
