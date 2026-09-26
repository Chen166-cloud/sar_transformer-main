"""Summarize the frozen Toronto showcase experiment with the prior metric code."""

from __future__ import annotations

import json
from pathlib import Path

import summarize_toronto_gt_benchmark as benchmark


ROOT = Path(r"D:\research\sar_transformer-main")
DATA_ROOT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\showcase_selected_20260926")
OUTPUT_ROOT = ROOT / "output/toronto_showcase_20260926"


def main() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    benchmark.DATA_ROOT = DATA_ROOT
    benchmark.OUTPUT_ROOT = OUTPUT_ROOT
    benchmark.SCENES = tuple((item["scene"], item["label_zh"]) for item in manifest["scenes"])
    benchmark.main()


if __name__ == "__main__":
    main()
