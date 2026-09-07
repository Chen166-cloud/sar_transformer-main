"""Reproducibility helpers shared by the canonical experiment entry points."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
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
        # A formal run must fail loudly if an operation has no deterministic
        # implementation; warnings would otherwise permit irreproducible
        # weights while still labelling the run deterministic.
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def require_finite_tensor(tensor: torch.Tensor, label: str, context: str = "") -> None:
    """Fail before an invalid tensor can be recorded as a successful update."""

    if not bool(torch.isfinite(tensor).all().item()):
        suffix = f" ({context})" if context else ""
        raise FloatingPointError(f"Non-finite {label}{suffix}")


def require_finite_number(value: Any, label: str, context: str = "") -> float:
    """Return a finite scalar or raise with an auditable location."""

    converted = float(value)
    if not np.isfinite(converted):
        suffix = f" ({context})" if context else ""
        raise FloatingPointError(f"Non-finite {label}: {converted}{suffix}")
    return converted


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol metadata with a canonical encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """Capture every RNG used by the formal PyTorch experiment runners."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "resume on a CUDA-capable host to preserve the exact run."
            )
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Write a checkpoint atomically in the destination directory."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_dump(
    payload: Any,
    path: str | os.PathLike[str],
    *,
    exclusive: bool = True,
) -> None:
    """Durably publish JSON with a same-directory temporary and atomic replace."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not permit fsync on directories. The file
            # itself has still been flushed before the atomic replace.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def training_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    run_config: Mapping[str, Any],
    protocol_hashes: Mapping[str, str],
    best: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the resumable checkpoint required by ICSPS26-FROZEN-v2."""
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {
        "checkpoint_format": "icsps26-resumable-v1",
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "rng_state": capture_rng_state(),
        "run_config": dict(run_config),
        "protocol_hashes": dict(protocol_hashes),
        "best": dict(best),
        "metrics": dict(metrics),
    }


def load_training_checkpoint_strict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str | os.PathLike[str],
    expected_protocol_hashes: Mapping[str, str],
    expected_run_invariants: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Resume only when data, schedule, method, and frozen hyperparameters match."""
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_format") != "icsps26-resumable-v1":
        raise ValueError(
            "--resume requires an icsps26-resumable-v1 checkpoint; legacy weights "
            "may be used for evaluation but cannot reproduce an interrupted formal run."
        )
    actual_hashes = checkpoint.get("protocol_hashes", {})
    if dict(actual_hashes) != dict(expected_protocol_hashes):
        raise ValueError(
            "Protocol hash mismatch on resume: expected "
            f"{dict(expected_protocol_hashes)}, found {dict(actual_hashes)}"
        )
    saved_config = checkpoint.get("run_config", {})
    mismatches = {
        key: {"expected": value, "found": saved_config.get(key)}
        for key, value in expected_run_invariants.items()
        if saved_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen run invariant mismatch on resume: {mismatches}")
    load_checkpoint_strict(model, checkpoint_path, map_location=map_location)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
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


def validate_completed_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    protocol_id: str,
) -> Dict[str, Any]:
    """Require a checkpoint to be covered by its run's completion marker."""

    checkpoint = Path(checkpoint_path).resolve()
    completion_path = checkpoint.parent / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Formal checkpoint has no sibling completion marker: {completion_path}"
        )
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("protocol_id") != protocol_id or completion.get("formal_run") is not True:
        raise ValueError(
            f"Checkpoint completion marker has the wrong protocol/formal state: {completion_path}"
        )
    if (
        "completed_updates" in completion
        and int(completion["completed_updates"]) != int(completion.get("target_updates", -1))
    ):
        raise ValueError(f"Supervised run is incomplete: {completion_path}")
    actual_hash = sha256_file(checkpoint)
    declared_hashes = {
        str(completion[key]).lower()
        for key in ("checkpoint_best_sha256", "checkpoint_last_sha256")
        if completion.get(key)
    }
    if actual_hash not in declared_hashes:
        raise ValueError(
            f"Checkpoint SHA-256 is not covered by completion marker: {checkpoint}"
        )
    return completion


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
