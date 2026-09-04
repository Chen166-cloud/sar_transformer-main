"""Run the canonical 6-preset x 3-seed supervised P0 experiment matrix.

Run this script inside the validated Docker image. Each child entry point creates
its own immutable output directory and records source/data/checkpoint hashes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ablation_config import ABLATION_PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="NWPU_RESISC45_SAR_global_L_v2")
    parser.add_argument(
        "--split-manifest",
        default="NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json",
    )
    parser.add_argument("--output-root", default="experiments_repro/nwpu_global_L_p0_ablation")
    parser.add_argument("--evaluation-root", default="test_results_repro/nwpu_global_L_p0_ablation")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--ablations", nargs="+", choices=sorted(ABLATION_PRESETS),
        default=list(ABLATION_PRESETS),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _complete(path: Path, required: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in required)


def _display(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    project_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root).resolve()
    split_manifest = Path(args.split_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    evaluation_root = Path(args.evaluation_root).resolve()

    jobs: list[dict] = []
    for ablation in args.ablations:
        for seed in args.seeds:
            name = f"{ablation}_seed{seed}"
            train_output = output_root / name
            evaluation_output = evaluation_root / name
            train_command = [
                sys.executable,
                str(project_root / "train_reproducible.py"),
                "--dataset-root", str(dataset_root),
                "--split-manifest", str(split_manifest),
                "--output-dir", str(train_output),
                "--ablation", ablation,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--crop", str(args.crop),
                "--workers", str(args.workers),
            ]
            evaluation_command = [
                sys.executable,
                str(project_root / "evaluate_synthetic_reproducible.py"),
                "--dataset-root", str(dataset_root),
                "--split-manifest", str(split_manifest),
                "--split", args.evaluation_split,
                "--checkpoint", str(train_output / "best_model.pth"),
                "--output-dir", str(evaluation_output),
                "--ablation", ablation,
                "--seed", str(seed),
                "--crop", str(args.crop),
            ]
            jobs.append(
                {
                    "name": name,
                    "train_output": train_output,
                    "evaluation_output": evaluation_output,
                    "train_command": train_command,
                    "evaluation_command": evaluation_command,
                }
            )

    if args.dry_run:
        print(json.dumps(
            {
                "jobs": len(jobs),
                "commands": [
                    {
                        "name": job["name"],
                        "train": _display(job["train_command"]),
                        "evaluate": _display(job["evaluation_command"]),
                    }
                    for job in jobs
                ],
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_runner_logs"
    log_root.mkdir(exist_ok=True)
    status_path = output_root / "matrix_status.json"
    status = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "epochs": args.epochs,
        "crop": args.crop,
        "dataset_root": str(dataset_root),
        "split_manifest": str(split_manifest),
        "evaluation_split": args.evaluation_split,
        "seeds": args.seeds,
        "ablations": args.ablations,
        "jobs": {},
    }
    _write_status(status_path, status)

    for job in jobs:
        name = job["name"]
        train_output = job["train_output"]
        evaluation_output = job["evaluation_output"]
        if _complete(train_output, ("best_model.pth", "last_model.pth", "epoch_metrics.csv")):
            train_state = "already_complete"
        elif train_output.exists():
            raise RuntimeError(
                f"Partial/non-empty output exists for {name}: {train_output}. "
                "The canonical trainer does not silently resume or overwrite it."
            )
        else:
            train_state = "running"
            status["jobs"][name] = {"train": train_state, "evaluation": "pending"}
            _write_status(status_path, status)
            with (log_root / f"{name}_train.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    job["train_command"], cwd=project_root, stdout=log,
                    stderr=subprocess.STDOUT, check=True,
                )
            train_state = "complete"

        if _complete(evaluation_output, ("metrics_per_image.csv", "summary.json")):
            evaluation_state = "already_complete"
        elif evaluation_output.exists():
            raise RuntimeError(
                f"Partial/non-empty evaluation exists for {name}: {evaluation_output}"
            )
        else:
            evaluation_state = "running"
            status["jobs"][name] = {
                "train": train_state, "evaluation": evaluation_state
            }
            _write_status(status_path, status)
            with (log_root / f"{name}_evaluation.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    job["evaluation_command"], cwd=project_root, stdout=log,
                    stderr=subprocess.STDOUT, check=True,
                )
            evaluation_state = "complete"

        status["jobs"][name] = {
            "train": train_state,
            "evaluation": evaluation_state,
        }
        _write_status(status_path, status)

    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
