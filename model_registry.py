"""Strict model construction for the ICSPS 2026 experiment protocol.

The registry deliberately separates the original TransSARV2 baseline from the
paper model.  It also loads SAR-CAM only from a clean, pinned checkout of the
authors' repository; no third-party source is copied into this project.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from ablation_config import ABLATION_PRESETS, FORMAL_VARIANTS
from numeric_domain import INTENSITY_DOMAIN
from transform_main import TransSARV2, TransSARV2_DualFreqNG_Bottle


SUPPORTED_METHODS = ("transsar_v2", "ours", "sar_cam")
SAR_CAM_ROOT_ENV = "SAR_CAM_ROOT"
SAR_CAM_REPOSITORY_URL = "https://github.com/JK-the-Ko/SAR-CAM.git"
# The official repository's main branch was frozen for the ICSPS 2026 runs.
SAR_CAM_PINNED_COMMIT = "ea5ee3bed00ab22735a7c87518fe5388c2d6c49a"
SAR_CAM_CONSTRUCTOR_KWARGS = {
    "scale": 2,
    "in_channels": 1,
    "channels": 128,
    "kernel_size": 3,
    "stride": 1,
    "dilation": 1,
    "bias": True,
}


_METHOD_ALIASES = {
    "transsar_v2": "transsar_v2",
    "transsarv2": "transsar_v2",
    "transsarv2-retrained": "transsar_v2",
    "ours": "ours",
    "transsarv2_dualfreqng_bottle": "ours",
    "sar_cam": "sar_cam",
    "sar-cam": "sar_cam",
    "sarcam": "sar_cam",
}


def normalize_method_name(name: str) -> str:
    token = str(name).strip()
    normalized = _METHOD_ALIASES.get(token.lower())
    if normalized is None:
        raise ValueError(
            "Unknown model {!r}; expected one of {}".format(
                name, list(SUPPORTED_METHODS)
            )
        )
    return normalized


def _normalize_variant(variant: Optional[str]) -> Optional[str]:
    if variant is None:
        return None
    token = str(variant).strip().lower().replace("-", "_")
    if token in ("", "none", "null"):
        return None
    # The historical preset is structurally identical to the formal Log-only
    # control, so old checkpoints can be verified without relabeling weights.
    if token == "wout_fusion":
        return "log_only"
    return token


def _resolve_ours_variant(variant: Optional[str]) -> str:
    normalized = _normalize_variant(variant) or "full"
    allowed = set(FORMAL_VARIANTS) | set(ABLATION_PRESETS)
    if normalized not in allowed:
        raise ValueError(
            "Unknown Ours variant {!r}; expected one of {}".format(
                variant, sorted(allowed)
            )
        )
    return normalized


def canonical_model_metadata(
    method: str,
    variant: Optional[str] = None,
    external_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Return the metadata that must be embedded in a formal checkpoint."""
    canonical = normalize_method_name(method)
    if canonical == "ours":
        canonical_variant = _resolve_ours_variant(variant)
        return {
            "registry_schema_version": 1,
            "method": canonical,
            "class_name": "TransSARV2_DualFreqNG_Bottle",
            "variant": canonical_variant,
        }
    if _normalize_variant(variant) is not None:
        raise ValueError(f"{canonical} does not accept an Ours ablation variant")
    if canonical == "transsar_v2":
        return {
            "registry_schema_version": 1,
            "method": canonical,
            "class_name": "TransSARV2",
            "variant": None,
        }
    commit = external_commit or SAR_CAM_PINNED_COMMIT
    return {
        "registry_schema_version": 1,
        "method": canonical,
        "class_name": "SAR_CAM",
        "variant": None,
        "external_repository": SAR_CAM_REPOSITORY_URL,
        "external_commit": commit,
        "constructor_kwargs": dict(SAR_CAM_CONSTRUCTOR_KWARGS),
    }


def extract_checkpoint_model_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract model identity from new or historical checkpoint metadata.

    A raw state dict is intentionally rejected: it contains no trustworthy
    method identity and therefore cannot be used in a formal comparison.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping with model metadata")
    explicit = checkpoint.get("model_metadata")
    run_config = checkpoint.get("run_config")
    if explicit is not None and not isinstance(explicit, Mapping):
        raise ValueError("checkpoint['model_metadata'] must be a mapping")
    if run_config is not None and not isinstance(run_config, Mapping):
        raise ValueError("checkpoint['run_config'] must be a mapping")
    metadata = dict(explicit or {})
    config = dict(run_config or {})

    method_token = (
        metadata.get("method")
        or metadata.get("model_name")
        or metadata.get("model")
        or metadata.get("class_name")
        or config.get("method")
        or config.get("model_name")
        or config.get("model")
        or config.get("class_name")
    )
    if method_token is None:
        raise ValueError(
            "Checkpoint has no model identity; expected model_metadata.method "
            "or run_config.model"
        )
    method = normalize_method_name(str(method_token))
    class_token = metadata.get("class_name") or config.get("class_name")
    if class_token is not None and normalize_method_name(str(class_token)) != method:
        raise ValueError(
            f"Checkpoint method/class metadata disagree: {method_token!r} vs {class_token!r}"
        )

    raw_variant = (
        metadata.get("variant")
        if "variant" in metadata else config.get("variant")
    )
    if raw_variant is None:
        raw_variant = metadata.get("ablation", config.get("ablation"))
    variant = _normalize_variant(raw_variant)
    if method == "ours":
        variant = _resolve_ours_variant(variant)
    elif variant is not None:
        raise ValueError(
            f"Checkpoint for {method} unexpectedly declares variant={variant!r}"
        )

    external = metadata.get("external_source")
    external_commit = metadata.get("external_commit") or config.get("external_commit")
    if external_commit is None and isinstance(external, Mapping):
        external_commit = external.get("commit")
    constructor_kwargs = metadata.get("constructor_kwargs") or config.get(
        "constructor_kwargs"
    )
    return {
        "method": method,
        "class_name": canonical_model_metadata(
            method, variant, external_commit=external_commit
        )["class_name"],
        "variant": variant,
        "external_commit": external_commit,
        "constructor_kwargs": constructor_kwargs,
    }


def validate_checkpoint_model_metadata(
    checkpoint: Mapping[str, Any],
    expected_method: str,
    expected_variant: Optional[str] = None,
    expected_external_commit: str = SAR_CAM_PINNED_COMMIT,
) -> dict[str, Any]:
    """Fail before loading weights when checkpoint/model identities differ."""
    actual = extract_checkpoint_model_metadata(checkpoint)
    expected = canonical_model_metadata(
        expected_method,
        expected_variant,
        external_commit=expected_external_commit,
    )
    if actual["method"] != expected["method"]:
        raise ValueError(
            "Checkpoint method mismatch: expected {!r}, found {!r}".format(
                expected["method"], actual["method"]
            )
        )
    if actual["variant"] != expected["variant"]:
        raise ValueError(
            "Checkpoint variant mismatch: expected {!r}, found {!r}".format(
                expected["variant"], actual["variant"]
            )
        )
    if expected["method"] == "sar_cam":
        if actual["external_commit"] is None:
            raise ValueError("SAR-CAM checkpoint does not record external_commit")
        if actual["external_commit"] != expected_external_commit:
            raise ValueError(
                "SAR-CAM checkpoint commit mismatch: expected {}, found {}".format(
                    expected_external_commit, actual["external_commit"]
                )
            )
        if actual["constructor_kwargs"] is None:
            raise ValueError("SAR-CAM checkpoint does not record constructor_kwargs")
        if dict(actual["constructor_kwargs"]) != expected["constructor_kwargs"]:
            raise ValueError(
                "SAR-CAM checkpoint constructor mismatch: expected {!r}, found {!r}".format(
                    expected["constructor_kwargs"], actual["constructor_kwargs"]
                )
            )
    return actual


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(
            f"Cannot inspect SAR-CAM checkout at {root}: {detail.strip()}"
        ) from error
    return completed.stdout.strip()


def verify_sar_cam_checkout(
    external_root: Optional[str | os.PathLike[str]] = None,
    expected_commit: str = SAR_CAM_PINNED_COMMIT,
) -> Path:
    """Validate the official, clean SAR-CAM checkout and return its root."""
    root_value = external_root or os.environ.get(SAR_CAM_ROOT_ENV)
    if not root_value:
        raise ValueError(
            "SAR-CAM requires external_root or the SAR_CAM_ROOT environment variable"
        )
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise FileNotFoundError(
            f"SAR-CAM external root is not a Git checkout: {root}"
        )
    for required in ("model.py", "model_parts.py"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"SAR-CAM checkout is missing {required}: {root}")

    actual_commit = _git_output(root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            "SAR-CAM commit mismatch at {}: expected {}, found {}. "
            "Checkout the pinned commit before running a formal experiment.".format(
                root, expected_commit, actual_commit
            )
        )
    origin = _git_output(root, "remote", "get-url", "origin")
    normalized_origin = origin.lower().replace(":", "/")
    if "github.com/jk-the-ko/sar-cam" not in normalized_origin:
        raise RuntimeError(
            f"SAR-CAM origin is not the authors' official repository: {origin}"
        )
    dirty = _git_output(
        root, "status", "--porcelain", "--untracked-files=no", "--", "model.py", "model_parts.py"
    )
    if dirty:
        raise RuntimeError(
            "SAR-CAM model sources differ from the pinned commit; restore model.py "
            "and model_parts.py before running"
        )
    return root


def _build_sar_cam(
    external_root: Optional[str | os.PathLike[str]], expected_commit: str
) -> torch.nn.Module:
    root = verify_sar_cam_checkout(external_root, expected_commit)
    module_name = "_icsps2026_official_sar_cam_model"
    model_parts_before = sys.modules.pop("model_parts", None)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, root / "model.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create an import spec for {root / 'model.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        factory = getattr(module, "Model", None)
        if not callable(factory):
            raise AttributeError("Official SAR-CAM model.py does not expose Model")
        model = factory(**SAR_CAM_CONSTRUCTOR_KWARGS)
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("model_parts", None)
        if model_parts_before is not None:
            sys.modules["model_parts"] = model_parts_before
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Official SAR-CAM Model factory did not return torch.nn.Module")
    return model


def build_model(
    method: str,
    variant: Optional[str] = None,
    numeric_domain: str = INTENSITY_DOMAIN,
    external_root: Optional[str | os.PathLike[str]] = None,
    expected_external_commit: str = SAR_CAM_PINNED_COMMIT,
    checkpoint_metadata: Optional[Mapping[str, Any]] = None,
) -> torch.nn.Module:
    """Build one registered method and optionally verify checkpoint identity."""
    canonical = normalize_method_name(method)
    if checkpoint_metadata is not None:
        validate_checkpoint_model_metadata(
            checkpoint_metadata,
            canonical,
            variant,
            expected_external_commit=expected_external_commit,
        )

    if canonical == "transsar_v2":
        canonical_model_metadata(canonical, variant)
        model = TransSARV2()
    elif canonical == "ours":
        ours_variant = _resolve_ours_variant(variant)
        if ours_variant in FORMAL_VARIANTS:
            model = TransSARV2_DualFreqNG_Bottle(
                variant=ours_variant,
                numeric_domain=numeric_domain,
            )
        else:
            model = TransSARV2_DualFreqNG_Bottle(
                ablation=ours_variant,
                numeric_domain=numeric_domain,
            )
    else:
        canonical_model_metadata(canonical, variant, expected_external_commit)
        model = _build_sar_cam(external_root, expected_external_commit)

    model.registry_metadata = canonical_model_metadata(
        canonical,
        variant,
        external_commit=(expected_external_commit if canonical == "sar_cam" else None),
    )
    return model


def build_model_from_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    numeric_domain: str = INTENSITY_DOMAIN,
    external_root: Optional[str | os.PathLike[str]] = None,
    expected_external_commit: str = SAR_CAM_PINNED_COMMIT,
) -> torch.nn.Module:
    """Construct the exact model declared by a metadata-bearing checkpoint."""
    metadata = extract_checkpoint_model_metadata(checkpoint)
    return build_model(
        metadata["method"],
        variant=metadata["variant"],
        numeric_domain=numeric_domain,
        external_root=external_root,
        expected_external_commit=expected_external_commit,
        checkpoint_metadata=checkpoint,
    )
