import argparse
import os
import sys

import imageio as io
import lmdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops_table
from tqdm import tqdm

BASE_DIR = "/fast/AG_Kainmueller/data/pan_m"  # max cluster path
# base_dir = "X:/data/pan_m" # local path


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


def main(args):
    # placeholder for normalization function, probably do channel-wise normalization?
    def normalize(x):
        return (x - x.min()) / (
            x.max() - x.min()
        )  # TODO: Per channel?? * 255 to int8 ??

    patch_size = args.patch_size
    n_jobs = args.n_jobs
    surrounding_size = patch_size // 2

    paths = {
        "mibi_breast": os.path.join(BASE_DIR, "mibi_breast", "image_data", "samples"),
        "mibi_decidua": os.path.join(BASE_DIR, "mibi_decidua", "image_data"),
        "vectra_colon": os.path.join(BASE_DIR, "vectra_colon", "raw_structured"),
        "vectra_pancreas": os.path.join(BASE_DIR, "vectra_pancreas", "raw_structured"),
        "codex_colon": os.path.join(BASE_DIR, "codex_colon", "raw_structured"),
    }

    def load_channel(channel_path):
        channel_img = io.imread(channel_path)
        print("channel_img.shape", channel_img.shape)
        channel_img = normalize(np.squeeze(channel_img))
        return channel_img

    def save_lmdb_crop(idx, img_bytes, metadata_dict, txn_imgs, txn_meta):
        idx_bytes = str(idx).encode("utf-8")
        txn_imgs.put(idx_bytes, img_bytes)
        txn_meta.put(idx_bytes, metadata_dict)

    base_lmdb_dir = BASE_DIR + "_lmdb"
    os.makedirs(base_lmdb_dir, exist_ok=True)
    map_size = int(1e11)

    for dataset, path in paths.items():
        dataset_name = path.split("/")[len(BASE_DIR.split("/"))]
        dataset_lmdb_dir = os.path.join(base_lmdb_dir, dataset_name)
        lmdb_imgs_path = os.path.join(dataset_lmdb_dir, "images")
        lmdb_metadata_path = os.path.join(dataset_lmdb_dir, "metadata")

        os.makedirs(lmdb_imgs_path, exist_ok=True)
        os.makedirs(lmdb_metadata_path, exist_ok=True)

        env_imgs = lmdb.open(lmdb_imgs_path, map_size=map_size)
        env_metadata = lmdb.open(lmdb_metadata_path, map_size=map_size)
        with env_metadata.begin(write=True) as txn_meta:
            with env_imgs.begin(write=True) as txn_imgs:
                fovs = os.listdir(path)
                for img_idx, fov in enumerate(fovs):
                    if img_idx % 1000 == 0:
                        print("idx, dataset, path, fov", img_idx, dataset, path, fov)

                    metadata_dict = {}

                    fov_path = os.path.join(path, fov)
                    channels = selected_channels[dataset]
                    channel_imgs = []
                    channel_names = [channel.split(".")[0] for channel in channels]
                    channel_paths = [
                        os.path.join(fov_path, channel) for channel in channels
                    ]
                    # use joblib to parallelize loading of channels
                    channel_imgs = Parallel(n_jobs=n_jobs)(
                        delayed(load_channel)(channel_path)
                        for channel_path in tqdm(channel_paths)
                    )
                    # concatenate channel images
                    multiplex_img = np.stack(channel_imgs, axis=0)
                    # get segmentation mask
                    naming_convention = naming_convention_dict[dataset]
                    segmentation_path = naming_convention(fov)
                    segmentation_mask = np.squeeze(
                        io.imread(segmentation_path).astype(np.uint16)
                    )

                    metadata_dict["fov"] = fov
                    metadata_dict["channel_names"] = channel_names
                    metadata_dict["segmentation_mask"] = segmentation_mask

                    # get regionprops
                    regionprops = pd.DataFrame(
                        regionprops_table(
                            segmentation_mask, properties=("label", "centroid")
                        )
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
                    regionprops["centroid-0"] = (
                        regionprops["centroid-0"] + surrounding_size
                    )
                    regionprops["centroid-1"] = (
                        regionprops["centroid-1"] + surrounding_size
                    )
                    # iterate over regions and extract patches surrounding centroids from multiplex image
                    patches = Parallel(n_jobs=n_jobs)(
                        delayed(
                            lambda idx, region: multiplex_img[
                                :,
                                int(region["centroid-0"] - surrounding_size) : int(
                                    region["centroid-0"] + surrounding_size
                                ),
                                int(region["centroid-1"] - surrounding_size) : int(
                                    region["centroid-1"] + surrounding_size
                                ),
                            ]
                        )(idx, region)
                        for idx, region in tqdm(regionprops.iterrows())
                    )
                    # save patch, label, fov, dataset and channel_names for each training sample
                    for p_idx, patch in enumerate(patches):
                        patch_bytes = patch.tobytes()
                        full_idx = f"{dataset_name}_{img_idx:04d}_{p_idx:03d}"
                        save_lmdb_crop(
                            full_idx, patch_bytes, metadata_dict, txn_imgs, txn_meta
                        )

        env_imgs.close()
        env_metadata.close()
        print(f"FINISHED {lmdb_imgs_path}")


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
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))
