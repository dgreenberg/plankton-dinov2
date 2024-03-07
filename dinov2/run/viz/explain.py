import argparse
import sys
from typing import List, Optional

import torch
from PIL import Image

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

    if args.xai_method == "raw_attention":
        raw_attention(args, original_image)
    elif args.xai_method == "compare_all":
        compare_all(args, original_image)
    elif args.xai_method == "grad_sam":
        grad_sam(args, original_image)
    elif args.xai_method == "rollout":
        rollout(args, original_image)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    description = "DINOv2 attention visualization"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
