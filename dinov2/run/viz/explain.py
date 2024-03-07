import argparse
import sys
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms as pth_transforms

from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.utils.config import setup


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name for the wandb log",
        default="explain_run",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="Image for the explanation",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Set number of nodes used."
    )
    parser.add_argument(
        "--xai_method",
        type=str,
        help="XAI method to apply. Choose from raw_attention, rollout, grad_sam, GAE, compare_all, ...",
    )
    return parser


def raw_attention(args, img, viz=True):
    pass


def rollout(args, img, viz=True):
    pass


def grad_sam(args, img, viz=True):
    pass


def gae(args, img, viz=True):
    pass


def compare_all(args, img):
    raw_attn = raw_attention(args, img, viz=False)
    rollout_attn = rollout(args, img, viz=False)
    grad_sam_attn = grad_sam(args, img, viz=False)
    gae_attn = gae(args, img, viz=False)


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = setup(args, do_eval=True)
    model = build_model_for_eval(config, args.pretrained_weights)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    original_image = Image.open(args.img_path).convert("RGB")
    width, height = original_image.size
    aspect_ratio = width / height if width > height else height / width
    if width > height:
        new_width = config.visualization.max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = config.visualization.max_size
        new_width = int(new_height / aspect_ratio)
    image_size = (new_height, new_width)

    MEAN = [0.68622917, 0.68622917, 0.68622917]
    STD = [0.10176649, 0.10176649, 0.10176649]

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(MEAN, STD),
        ]
    )
    img = transform(original_image)

    patch_size = config.student.patch_size

    # Make the image divisible by the patch size
    width, height = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :width, :height].unsqueeze(0)

    width_featmap = img.shape[-2] // patch_size
    height_featmap = img.shape[-1] // patch_size

    if args.xai_method == "raw_attention":
        raw_attention(args, img)
    elif args.xai_method == "compare_all":
        compare_all(args, img)
    elif args.xai_method == "grad_sam":
        grad_sam(args, img)
    elif args.xai_method == "rollout":
        rollout(args, img)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    description = "DINOv2 attention visualization"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
