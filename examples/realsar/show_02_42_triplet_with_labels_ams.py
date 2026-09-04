#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from show_triplet_with_labels_ams import make_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose 02_42 noisy/residual/result in one labeled figure."
    )
    parser.add_argument(
        "--noisy",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "02_42_texture_4096_6656_y0_x0_noisy.png"
        ),
    )
    parser.add_argument(
        "--residual",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "02_42_texture_4096_6656_y0_x0_residual.png"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "02_42_texture_4096_6656_y0_x0_TransSARV2_DualFreqNG_Bottle.png"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "examples/realsar/"
            "02_42_texture_4096_6656_y0_x0_TransSARV2_DualFreqNG_Bottle_AMS_triplet_labeled.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_triplet(args.noisy, args.residual, args.result, args.output)


if __name__ == "__main__":
    main()
