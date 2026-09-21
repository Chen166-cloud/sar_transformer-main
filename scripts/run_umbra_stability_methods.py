"""Run the frozen Figure-3 seven-method protocol on one prepared Umbra ROI.

Methods: SAR-BM3D, SAR2SAR, SDUDNet, Trans-SAR, CL-SAR, MERLIN, and
MuLoG-DRUNet.  Existing completed method directories are skipped, so the
driver can be restarted after an interruption.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "cl_sar",
    "transsar",
    "sdudnet",
    "merlin",
    "sar2sar",
    "sarbm3d",
    "mulog_drunet",
)
COMPLETION = {
    "cl_sar": Path("cl_sar/run.json"),
    "transsar": Path("transsar/run.json"),
    "sdudnet": Path("sdudnet_db_display/run.json"),
    "merlin": Path("merlin_stride64_weighted/run.json"),
    "sar2sar": Path("sar2sar/summary.json"),
    "sarbm3d": Path("sarbm3d/summary.json"),
    "mulog_drunet": Path("mulog_drunet/run.json"),
}


def quote_matlab(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def commands(roi: Path, runs: Path, intensity_scale: float) -> dict[str, list[list[str]]]:
    noisy = roi / "noisy_intensity.npy"
    complex_roi = roi / "sicd_complex_roi.npy"
    parent = roi / "run.json"
    noisy_png = roi / "noisy.png"
    matlab_expression = (
        f"addpath('{quote_matlab(ROOT / 'matlab')}'); "
        f"run_sarbm3d_local('{quote_matlab(ROOT / 'external' / 'sarbm3d' / 'SARBM3D_v10_win64')}',"
        f"'{quote_matlab(runs / 'sarbm3d')}','{quote_matlab(roi / 'input.mat')}',"
        "1,'intensity','noisy','official');"
    )
    return {
        "cl_sar": [[
            str(ROOT / ".venv-cl-sar" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "cl_sar" / "run_local.py"),
            "--input", str(noisy), "--input-domain", "intensity", "--scale", "1",
            "--device", "cuda", "--output", str(runs / "cl_sar"),
        ]],
        "transsar": [[
            str(ROOT / ".venv-ours-legacy" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "transsar" / "run_umbra_official.py"),
            "--input", str(noisy), "--parent-run-json", str(parent),
            "--intensity-scale", repr(intensity_scale),
            "--output-dir", str(runs / "transsar"), "--device", "cuda",
        ]],
        "sdudnet": [[
            str(ROOT / ".venv-sdudnet" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "sdudnet" / "run_local.py"),
            "--input", str(noisy_png), "--model", "real", "--scale", "1",
            "--data-range", "1", "--device", "cuda",
            "--output", str(runs / "sdudnet_db_display"),
        ]],
        "merlin": [
            [
                str(ROOT / ".venv-merlin" / "Scripts" / "python.exe"),
                str(ROOT / "scripts" / "merlin" / "run_umbra_buenos_aires.py"),
                "--input", str(complex_roi), "--reference-intensity", str(noisy),
                "--parent-run-json", str(parent),
                "--output-dir", str(runs / "merlin_stride64"),
                "--patch-size", "256", "--stride-size", "64", "--skip-repeat",
            ],
            [
                str(ROOT / ".venv-merlin" / "Scripts" / "python.exe"),
                str(ROOT / "scripts" / "merlin" / "run_weighted_overlap.py"),
                "--input", str(complex_roi), "--reference-noisy", str(noisy),
                "--parent-run-json", str(parent),
                "--reference-uniform-stride64", str(runs / "merlin_stride64" / "denoised.npy"),
                "--output-dir", str(runs / "merlin_stride64_weighted"),
            ],
        ],
        "sar2sar": [[
            str(ROOT / ".venv-sar2sar" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "sar2sar" / "run_local.py"),
            "--input", str(noisy), "--input-domain", "intensity",
            "--output", str(runs / "sar2sar"), "--crop-size", "0",
            "--stride", "64", "--threads", "4",
        ]],
        "sarbm3d": [[
            str(Path(r"D:\Program Files\MATLAB\R2024a\bin\matlab.exe")),
            "-batch", matlab_expression,
        ]],
        "mulog_drunet": [[
            str(ROOT / ".venv-mulog-drunet" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "mulog_drunet" / "run_local.py"),
            "--input", str(noisy), "--input-domain", "intensity", "--looks", "1",
            "--scale", "1", "--iterations", "10", "--device", "cuda",
            "--output", str(runs / "mulog_drunet"),
        ]],
    }


def write_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    roi, experiment = args.roi.resolve(), args.output.resolve()
    runs = experiment / "runs"
    logs = experiment / "logs"
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    parent = json.loads((roi / "run.json").read_text(encoding="utf-8"))
    intensity_scale = float(parent["frozen_method_adapter"]["intensity_scale"])
    all_commands = commands(roi, runs, intensity_scale)
    state_path = experiment / "experiment_state.json"
    state = {
        "roi": str(roi),
        "protocol": "same seven released-method adapters and L=1 setting as Figure 3",
        "requested_methods": args.methods,
        "methods": {},
        "updated_utc": None,
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["methods"] = previous.get("methods", {})

    for method in args.methods:
        completion = runs / COMPLETION[method]
        if completion.is_file():
            print(f"[{method}] already complete: {completion}", flush=True)
            state["methods"][method] = {"status": "completed", "completion": str(completion)}
            continue
        state["methods"][method] = {"status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_state(state_path, state)
        log_path = logs / f"{method}.log"
        print(f"[{method}] starting; log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            for index, command in enumerate(all_commands[method], start=1):
                log.write(f"\n--- stage {index} command ---\n{json.dumps(command, ensure_ascii=False)}\n")
                log.flush()
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                if process.returncode != 0:
                    state["methods"][method] = {
                        "status": "failed",
                        "stage": index,
                        "returncode": process.returncode,
                        "log": str(log_path),
                    }
                    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    write_state(state_path, state)
                    raise RuntimeError(f"{method} stage {index} failed; see {log_path}")
        if not completion.is_file():
            raise RuntimeError(f"{method} returned successfully but completion artifact is missing: {completion}")
        state["methods"][method] = {
            "status": "completed",
            "completion": str(completion),
            "log": str(log_path),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_state(state_path, state)
        print(f"[{method}] complete", flush=True)
    print(state_path, flush=True)


if __name__ == "__main__":
    main()
