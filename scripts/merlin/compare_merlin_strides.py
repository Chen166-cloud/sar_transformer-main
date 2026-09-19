#!/usr/bin/env python3
"""Quantitatively compare MERLIN stride 254 and stride 64 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "output" / "umbra_buenos_aires_multimethod_1024" / "runs"
DEFAULT_DIR = RUNS / "merlin"
OVERLAP_DIR = RUNS / "merlin_stride64"
OUT_JSON = OVERLAP_DIR / "stride_comparison.json"
OUT_PNG = OVERLAP_DIR / "merlin_stride_comparison_600dpi.png"
DB_LIMITS = (-13.583520412445068, 27.24244294166567)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_db(intensity: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(intensity.astype(np.float64), 1e-12))


def edge_profiles(db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_profile = np.mean(np.abs(np.diff(db, axis=0)), axis=1)
    col_profile = np.mean(np.abs(np.diff(db, axis=1)), axis=0)
    return row_profile, col_profile


def periodic_energy(profile: np.ndarray, period: int, harmonics: int = 4) -> dict:
    centered = profile - np.mean(profile)
    denominator = float(len(centered) * np.sum(centered**2))
    per_harmonic = []
    indices = np.arange(len(centered), dtype=np.float64)
    for harmonic in range(1, harmonics + 1):
        coefficient = np.sum(
            centered * np.exp(-2j * np.pi * harmonic * indices / period)
        )
        fraction = 0.0 if denominator == 0 else float(abs(coefficient) ** 2 / denominator)
        per_harmonic.append(fraction)
    return {
        "fundamental_fraction": per_harmonic[0],
        "first_four_harmonics_fraction_sum": float(sum(per_harmonic)),
        "per_harmonic_fractions": per_harmonic,
    }


def boundary_metric(db: np.ndarray, period: int) -> dict:
    row_profile, col_profile = edge_profiles(db)
    boundaries = np.arange(period, db.shape[0], period, dtype=int)
    row_values = row_profile[boundaries - 1]
    col_values = col_profile[boundaries - 1]

    background_mask = np.ones(db.shape[0] - 1, dtype=bool)
    for boundary in boundaries:
        background_mask[max(0, boundary - 3) : min(len(background_mask), boundary + 2)] = False
    background_values = np.concatenate(
        (row_profile[background_mask], col_profile[background_mask])
    )
    boundary_values = np.concatenate((row_values, col_values))
    row_energy = periodic_energy(row_profile, period)
    col_energy = periodic_energy(col_profile, period)
    return {
        "period_pixels": period,
        "boundary_positions": boundaries.tolist(),
        "boundary_jump_db_mean": float(np.mean(boundary_values)),
        "background_jump_db_mean": float(np.mean(background_values)),
        "boundary_to_background_mean_ratio": float(
            np.mean(boundary_values) / np.mean(background_values)
        ),
        "boundary_jump_db_median": float(np.median(boundary_values)),
        "background_jump_db_median": float(np.median(background_values)),
        "boundary_to_background_median_ratio": float(
            np.median(boundary_values) / np.median(background_values)
        ),
        "row_grid_energy": row_energy,
        "col_grid_energy": col_energy,
        "mean_fundamental_grid_energy_fraction": float(
            0.5
            * (
                row_energy["fundamental_fraction"]
                + col_energy["fundamental_fraction"]
            )
        ),
        "mean_first_four_harmonics_grid_energy_fraction_sum": float(
            0.5
            * (
                row_energy["first_four_harmonics_fraction_sum"]
                + col_energy["first_four_harmonics_fraction_sum"]
            )
        ),
    }


def actual_transition_metric(db: np.ndarray, starts: list[int], patch_size: int) -> dict:
    row_profile, col_profile = edge_profiles(db)
    transitions = sorted(
        {
            value
            for start in starts
            for value in (start, start + patch_size)
            if 0 < value < db.shape[0]
        }
    )
    values = np.concatenate(
        (row_profile[np.array(transitions) - 1], col_profile[np.array(transitions) - 1])
    )
    background_mask = np.ones(db.shape[0] - 1, dtype=bool)
    for transition in transitions:
        background_mask[
            max(0, transition - 3) : min(len(background_mask), transition + 2)
        ] = False
    background = np.concatenate(
        (row_profile[background_mask], col_profile[background_mask])
    )
    return {
        "transition_positions": transitions,
        "transition_count": len(transitions),
        "transition_jump_db_mean": float(np.mean(values)),
        "background_jump_db_mean": float(np.mean(background)),
        "transition_to_background_mean_ratio": float(
            np.mean(values) / np.mean(background)
        ),
        "transition_jump_db_median": float(np.median(values)),
        "transition_jump_db_max": float(np.max(values)),
        "transition_jump_db_percentiles_25_50_75_90_95": [
            float(value) for value in np.percentile(values, [25, 50, 75, 90, 95])
        ],
        "background_jump_db_median": float(np.median(background)),
        "transition_to_background_median_ratio": float(
            np.median(values) / np.median(background)
        ),
    }


def run_metrics(path: Path) -> tuple[np.ndarray, dict, dict]:
    image = np.load(path / "denoised.npy", allow_pickle=False)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    db = to_db(image)
    metrics = {
        "stride_pixels": run["inference"]["stride_size"],
        "patch_size_pixels": run["inference"]["patch_size"],
        "patch_count": run["inference"]["patch_grid"]["patch_count"],
        "primary_elapsed_seconds": run["inference"]["primary_elapsed_seconds"],
        "repeat_elapsed_seconds": run["inference"]["repeat_validation"][
            "elapsed_seconds"
        ],
        "period_64": boundary_metric(db, 64),
        "period_254": boundary_metric(db, 254),
        "actual_patch_transitions": actual_transition_metric(
            db,
            run["inference"]["patch_grid"]["row_starts"],
            run["inference"]["patch_size"],
        ),
        "linear_intensity": {
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean(dtype=np.float64)),
        },
        "file_sha256": sha256_file(path / "denoised.npy"),
    }
    return image, db, metrics


def main() -> None:
    default, default_db, default_metrics = run_metrics(DEFAULT_DIR)
    overlap, overlap_db, overlap_metrics = run_metrics(OVERLAP_DIR)
    if default.shape != overlap.shape or default.shape != (1024, 1024):
        raise AssertionError((default.shape, overlap.shape))

    delta_db = overlap_db - default_db
    delta_limit = float(np.percentile(np.abs(delta_db), 99.5))
    comparison = {
        "comparison": "MERLIN Spotlight: official default stride 254 versus high-overlap stride 64",
        "common_contract": {
            "input": "same fixed 1024x1024 Umbra complex64 ROI",
            "model": "same official spotlight checkpoint",
            "patch_size": 256,
            "symetrise": True,
            "no_resize_minmax_scale_or_clip": True,
            "analysis_domain": "10*log10(linear intensity) without display clipping",
            "boundary_metric": "mean/median absolute adjacent-pixel dB jump at exact periodic boundaries divided by non-boundary background jump; 1 means no excess boundary jump",
            "grid_energy_metric": "normalized direct-DFT energy of row/column mean absolute dB-gradient profiles at 1/period and first four harmonics",
        },
        "stride254": default_metrics,
        "stride64": overlap_metrics,
        "pairwise_difference": {
            "stride64_minus_stride254_db_min": float(delta_db.min()),
            "stride64_minus_stride254_db_max": float(delta_db.max()),
            "stride64_minus_stride254_db_mean": float(delta_db.mean()),
            "stride64_minus_stride254_db_median": float(np.median(delta_db)),
            "stride64_minus_stride254_db_abs_percentiles_50_90_95_99_99p5": [
                float(value)
                for value in np.percentile(
                    np.abs(delta_db), [50.0, 90.0, 95.0, 99.0, 99.5]
                )
            ],
            "primary_runtime_ratio_stride64_over_stride254": float(
                overlap_metrics["primary_elapsed_seconds"]
                / default_metrics["primary_elapsed_seconds"]
            ),
            "patch_count_ratio_stride64_over_stride254": float(
                overlap_metrics["patch_count"] / default_metrics["patch_count"]
            ),
            "actual_transition_mean_jump_reduction_percent_with_stride64": float(
                100.0
                * (
                    1.0
                    - overlap_metrics["actual_patch_transitions"][
                        "transition_jump_db_mean"
                    ]
                    / default_metrics["actual_patch_transitions"][
                        "transition_jump_db_mean"
                    ]
                )
            ),
            "actual_transition_max_jump_reduction_percent_with_stride64": float(
                100.0
                * (
                    1.0
                    - overlap_metrics["actual_patch_transitions"][
                        "transition_jump_db_max"
                    ]
                    / default_metrics["actual_patch_transitions"][
                        "transition_jump_db_max"
                    ]
                )
            ),
        },
        "recommendation": {
            "preferred_stride": 64,
            "reason": "For the paper panel, stride 64 is visually preferable: it removes the dominant coarse 254-pixel block discontinuities and lowers both the mean and maximum jump at actual patch transitions. It does introduce a denser, finer 64-pixel grid and costs substantially more runtime, so this is a visual-quality choice rather than an efficiency choice; patch-wise symmetrisation still leaves visible grid structure.",
        },
    }
    OUT_JSON.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), constrained_layout=True)
    for axis, image, title in (
        (axes[0, 0], default_db, "MERLIN stride 254"),
        (axes[0, 1], overlap_db, "MERLIN stride 64"),
    ):
        axis.imshow(image, cmap="gray", vmin=DB_LIMITS[0], vmax=DB_LIMITS[1])
        axis.set_title(title)
        axis.axis("off")
    delta_artist = axes[0, 2].imshow(
        delta_db, cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit
    )
    axes[0, 2].set_title("stride 64 − 254 (dB)")
    axes[0, 2].axis("off")
    fig.colorbar(delta_artist, ax=axes[0, 2], fraction=0.046, pad=0.04)

    crop = np.s_[128:896, 128:896]
    for axis, image, title in (
        (axes[1, 0], default_db[crop], "Central crop, stride 254"),
        (axes[1, 1], overlap_db[crop], "Central crop, stride 64"),
    ):
        axis.imshow(image, cmap="gray", vmin=DB_LIMITS[0], vmax=DB_LIMITS[1])
        axis.set_title(title)
        axis.axis("off")
    delta_crop_artist = axes[1, 2].imshow(
        delta_db[crop], cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit
    )
    axes[1, 2].set_title("Central difference (dB)")
    axes[1, 2].axis("off")
    fig.colorbar(delta_crop_artist, ax=axes[1, 2], fraction=0.046, pad=0.04)
    fig.suptitle(
        "MERLIN overlap audit — identical input/model, shared dB display",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    plt.close(fig)

    comparison["outputs"] = {
        "json": {"path": str(OUT_JSON), "sha256": sha256_file(OUT_JSON)},
        "figure": {"path": str(OUT_PNG), "sha256": sha256_file(OUT_PNG)},
    }
    # Rewrite once so the JSON also records the figure hash.  It intentionally
    # omits a self-hash because self-referential hashes are not stable.
    comparison["outputs"]["json"].pop("sha256")
    OUT_JSON.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
