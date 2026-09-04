"""Reproducibility helpers shared by the canonical experiment entry points."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    # Required by CUDA >= 10.2 for deterministic cuBLAS GEMM operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    else:
        state_dict = checkpoint
        metadata = {}
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return metadata


def checkpoint_payload(
    model: torch.nn.Module,
    epoch: int,
    run_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {
        "state_dict": state_dict,
        "epoch": int(epoch),
        "run_config": dict(run_config),
        "metrics": dict(metrics),
    }


def prepare_run_directory(
    output_dir: str | os.PathLike[str],
    run_config: Mapping[str, Any],
    source_files: Iterable[str | os.PathLike[str]],
    checkpoint_path: Optional[str | os.PathLike[str]] = None,
    resume: bool = False,
) -> Path:
    """Create a traceable run directory without silently replacing old results."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing to write into non-empty run directory: {output}. "
            "Choose a new directory or pass --resume explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)

    source_hashes = {}
    for source in source_files:
        source_path = Path(source).resolve()
        source_hashes[str(source_path)] = sha256_file(source_path)

    metadata: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": dict(run_config),
        "source_sha256": source_hashes,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path is not None else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if resume:
        name = "resume_config_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    else:
        name = "run_config.json"
    config_path = output / name
    with open(config_path, "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return output
