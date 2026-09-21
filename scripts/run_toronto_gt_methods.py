"""Run the scientifically compatible Figure-3 methods on one paired Toronto scene.

MERLIN is recorded as not applicable because the published Toronto benchmark is
GRD intensity data and contains no complex SLC phase samples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("cl_sar", "transsar", "sdudnet", "sar2sar", "sarbm3d", "mulog_drunet")
COMPLETION = {
    "cl_sar": Path("cl_sar/run.json"),
    "transsar": Path("transsar/run.json"),
    "sdudnet": Path("sdudnet/run.json"),
    "sar2sar": Path("sar2sar/summary.json"),
    "sarbm3d": Path("sarbm3d/summary.json"),
    "mulog_drunet": Path("mulog_drunet/run.json"),
}


def matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def method_commands(scene: Path, runs: Path) -> dict[str, list[str]]:
    noisy = scene / "noisy_intensity.npy"
    noisy_png = scene / "noisy.png"
    parent = scene / "run.json"
    sarbm3d = (
        f"addpath('{matlab_path(ROOT / 'matlab')}'); "
        f"run_sarbm3d_local('{matlab_path(ROOT / 'external' / 'sarbm3d' / 'SARBM3D_v10_win64')}',"
        f"'{matlab_path(runs / 'sarbm3d')}','{matlab_path(scene / 'input.mat')}',"
        "1,'intensity','noisy','official');"
    )
    return {
        "cl_sar": [str(ROOT / ".venv-cl-sar" / "Scripts" / "python.exe"),
                   str(ROOT / "scripts" / "cl_sar" / "run_local.py"),
                   "--input", str(noisy), "--input-domain", "intensity", "--scale", "255",
                   "--device", "cuda", "--output", str(runs / "cl_sar")],
        "transsar": [str(ROOT / ".venv-ours-legacy" / "Scripts" / "python.exe"),
                     str(ROOT / "scripts" / "transsar" / "run_umbra_official.py"),
                     "--input", str(noisy), "--parent-run-json", str(parent),
                     "--intensity-scale", "255", "--reflect-pad", "64",
                     "--output-dir", str(runs / "transsar"), "--device", "cuda"],
        "sdudnet": [str(ROOT / ".venv-sdudnet" / "Scripts" / "python.exe"),
                    str(ROOT / "scripts" / "sdudnet" / "run_local.py"),
                    "--input", str(noisy_png), "--model", "real", "--scale", "1",
                    "--data-range", "1", "--device", "cuda", "--output", str(runs / "sdudnet")],
        "sar2sar": [str(ROOT / ".venv-sar2sar" / "Scripts" / "python.exe"),
                    str(ROOT / "scripts" / "sar2sar" / "run_local.py"),
                    "--input", str(noisy), "--input-domain", "intensity", "--crop-size", "0",
                    "--stride", "64", "--threads", "4", "--output", str(runs / "sar2sar")],
        "sarbm3d": [str(Path(r"D:\Program Files\MATLAB\R2024a\bin\matlab.exe")), "-batch", sarbm3d],
        "mulog_drunet": [str(ROOT / ".venv-mulog-drunet" / "Scripts" / "python.exe"),
                         str(ROOT / "scripts" / "mulog_drunet" / "run_local.py"),
                         "--input", str(noisy), "--input-domain", "intensity", "--looks", "1",
                         "--scale", "255", "--iterations", "10", "--device", "cuda",
                         "--output", str(runs / "mulog_drunet")],
    }


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    scene, experiment = args.scene.resolve(), args.output.resolve()
    runs, logs = experiment / "runs", experiment / "logs"
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = experiment / "experiment_state.json"
    state = {
        "scene": str(scene),
        "protocol": "paired Sentinel-1 GRD intensity benchmark; L=1; published 8-bit units",
        "requested_methods": list(args.methods),
        "merlin": {"status": "not_applicable", "reason": "requires complex SLC; benchmark supplies GRD intensity only"},
        "methods": {},
    }
    if state_path.exists():
        state["methods"] = json.loads(state_path.read_text(encoding="utf-8")).get("methods", {})
    commands = method_commands(scene, runs)
    for method in args.methods:
        completion = runs / COMPLETION[method]
        if completion.is_file():
            state["methods"][method] = {"status": "completed", "completion": str(completion)}
            continue
        log_path = logs / f"{method}.log"
        state["methods"][method] = {"status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(state_path, state)
        print(f"[{method}] starting", flush=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            process = subprocess.run(commands[method], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", check=False)
        if process.returncode != 0 or not completion.is_file():
            state["methods"][method] = {"status": "failed", "returncode": process.returncode, "log": str(log_path)}
            write_json(state_path, state)
            raise RuntimeError(f"{method} failed; see {log_path}")
        state["methods"][method] = {"status": "completed", "completion": str(completion), "log": str(log_path)}
        write_json(state_path, state)
        print(f"[{method}] complete", flush=True)
    print(state_path)


if __name__ == "__main__":
    main()
