# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.eval.knn import get_args_parser
from dinov2.eval.knn import main as knn_main

logger = logging.getLogger("dinov2")


def main(args):
    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    knn_main(args)

    return 0


if __name__ == "__main__":
    knn_args_parser = get_args_parser(add_help=False)
    args = knn_args_parser.parse_args()
    sys.exit(main(args))
