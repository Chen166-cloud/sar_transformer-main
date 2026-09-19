#!/usr/bin/env python3
"""Audit MERLIN uniform and positive-Hann overlap aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "output" / "umbra_buenos_aires_multimethod_1024" / "runs"
DIRS = {
    "stride254_uniform": RUNS / "merlin",
    "stride64_uniform": RUNS / "merlin_stride64",
    "stride64_positive_hann": RUNS / "merlin_stride64_weighted",
}
OUT_DIR = DIRS["stride64_positive_hann"]
OUT_JSON = OUT_DIR / "weighted_overlap_comparison.json"
OUT_PNG = OUT_DIR / "merlin_weighted_overlap_comparison_600dpi.png"
OUT_PREVIEW = OUT_DIR / "merlin_weighted_overlap_comparison_preview.png"
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
    return (
        np.mean(np.abs(np.diff(db, axis=0)), axis=1),
        np.mean(np.abs(np.diff(db, axis=1)), axis=0),
    )


def grid_energy(profile: np.ndarray, period: int) -> list[float]:
    centered = profile - profile.mean()
    denominator = float(len(centered) * np.sum(centered**2))
    positions = np.arange(len(centered), dtype=np.float64)
    if denominator == 0:
        return [0.0] * 4
    return [
        float(
            abs(
                np.sum(
                    centered
                    * np.exp(-2j * np.pi * harmonic * positions / period)
                )
            )
            ** 2
            / denominator
        )
        for harmonic in range(1, 5)
    ]


def period_metrics(db: np.ndarray, period: int) -> dict:
    row_profile, col_profile = edge_profiles(db)
    boundaries = np.arange(period, db.shape[0], period, dtype=int)
    boundary_values = np.concatenate(
        (row_profile[boundaries - 1], col_profile[boundaries - 1])
    )
    mask = np.ones(db.shape[0] - 1, dtype=bool)
    for boundary in boundaries:
        mask[max(0, boundary - 3) : min(mask.size, boundary + 2)] = False
    background = np.concatenate((row_profile[mask], col_profile[mask]))
    row_energy = grid_energy(row_profile, period)
    col_energy = grid_energy(col_profile, period)
    mean_energy = np.mean(np.array([row_energy, col_energy]), axis=0)
    return {
        "period_pixels": period,
        "boundary_jump_db_mean": float(boundary_values.mean()),
        "background_jump_db_mean": float(background.mean()),
        "boundary_to_background_mean_ratio": float(
            boundary_values.mean() / background.mean()
        ),
        "boundary_jump_db_median": float(np.median(boundary_values)),
        "background_jump_db_median": float(np.median(background)),
        "boundary_to_background_median_ratio": float(
            np.median(boundary_values) / np.median(background)
        ),
        "mean_grid_energy_per_harmonic": [float(value) for value in mean_energy],
        "mean_fundamental_grid_energy_fraction": float(mean_energy[0]),
        "mean_first_four_harmonics_grid_energy_fraction_sum": float(
            mean_energy.sum()
        ),
    }


def transition_metrics(db: np.ndarray, starts: list[int], patch_size: int) -> dict:
    row_profile, col_profile = edge_profiles(db)
    transitions = sorted(
        {
            value
            for start in starts
            for value in (start, start + patch_size)
            if 0 < value < db.shape[0]
        }
    )
    transition_values = np.concatenate(
        (
            row_profile[np.array(transitions) - 1],
            col_profile[np.array(transitions) - 1],
        )
    )
    mask = np.ones(db.shape[0] - 1, dtype=bool)
    for transition in transitions:
        mask[max(0, transition - 3) : min(mask.size, transition + 2)] = False
    background = np.concatenate((row_profile[mask], col_profile[mask]))
    return {
        "positions": transitions,
        "line_count": len(transitions),
        "jump_db_mean": float(transition_values.mean()),
        "jump_db_median": float(np.median(transition_values)),
        "jump_db_max": float(transition_values.max()),
        "jump_db_percentiles_25_50_75_90_95": [
            float(value)
            for value in np.percentile(transition_values, [25, 50, 75, 90, 95])
        ],
        "background_db_mean": float(background.mean()),
        "background_db_median": float(np.median(background)),
        "mean_ratio": float(transition_values.mean() / background.mean()),
        "median_ratio": float(
            np.median(transition_values) / np.median(background)
        ),
    }


def load_run(name: str, path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    intensity = np.load(path / "denoised.npy", allow_pickle=False)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    inference = run["inference"]
    patch_size = inference["patch_size"]
    if "patch_grid" in inference:
        starts = inference["patch_grid"]["row_starts"]
        patch_count = inference["patch_grid"]["patch_count"]
    else:
        starts = inference["row_starts"]
        patch_count = inference["patch_count"]
    db = to_db(intensity)
    metrics = {
        "name": name,
        "stride_pixels": inference["stride_size"],
        "patch_size_pixels": patch_size,
        "patch_count": patch_count,
        "primary_elapsed_seconds": inference["primary_elapsed_seconds"],
        "linear_intensity": {
            "min": float(intensity.min()),
            "max": float(intensity.max()),
            "mean": float(intensity.mean(dtype=np.float64)),
        },
        "actual_patch_transitions": transition_metrics(db, starts, patch_size),
        "period_64": period_metrics(db, 64),
        "period_254": period_metrics(db, 254),
        "denoised_sha256": sha256_file(path / "denoised.npy"),
    }
    return intensity, db, metrics


def reduction_percent(new: float, old: float) -> float:
    return float(100.0 * (1.0 - new / old))


def main() -> None:
    loaded = {name: load_run(name, path) for name, path in DIRS.items()}
    uniform_metrics = loaded["stride64_uniform"][2]
    weighted_metrics = loaded["stride64_positive_hann"][2]
    uniform_transition = uniform_metrics["actual_patch_transitions"]
    weighted_transition = weighted_metrics["actual_patch_transitions"]
    uniform_grid = uniform_metrics["period_64"]
    weighted_grid = weighted_metrics["period_64"]

    uniform_db = loaded["stride64_uniform"][1]
    weighted_db = loaded["stride64_positive_hann"][1]
    difference_db = weighted_db - uniform_db
    delta_limit = float(np.percentile(np.abs(difference_db), 99.5))
    comparison = {
        "comparison": "MERLIN patch aggregation audit on identical Spotlight stride-64 predictions",
        "scientific_status": {
            "model_or_patch_predictions_modified": False,
            "radiometric_gain_matching": False,
            "scene_wise_stretch_or_minmax": False,
            "postprocessing_change": "uniform overlap average replaced by positive-Hann weighted overlap-add",
            "publication_label_required": "MERLIN (weighted overlap)",
            "claim_as_unmodified_official_inference": False,
        },
        "metric_domain": "10*log10(linear intensity), without display clipping",
        "methods": {
            name: values[2] for name, values in loaded.items()
        },
        "weighted_vs_uniform_stride64": {
            "actual_transition_mean_jump_reduction_percent": reduction_percent(
                weighted_transition["jump_db_mean"], uniform_transition["jump_db_mean"]
            ),
            "actual_transition_median_jump_reduction_percent": reduction_percent(
                weighted_transition["jump_db_median"],
                uniform_transition["jump_db_median"],
            ),
            "actual_transition_max_jump_reduction_percent": reduction_percent(
                weighted_transition["jump_db_max"], uniform_transition["jump_db_max"]
            ),
            "period64_fundamental_grid_energy_reduction_percent": reduction_percent(
                weighted_grid["mean_fundamental_grid_energy_fraction"],
                uniform_grid["mean_fundamental_grid_energy_fraction"],
            ),
            "period64_first_four_harmonics_grid_energy_reduction_percent": reduction_percent(
                weighted_grid[
                    "mean_first_four_harmonics_grid_energy_fraction_sum"
                ],
                uniform_grid[
                    "mean_first_four_harmonics_grid_energy_fraction_sum"
                ],
            ),
            "weighted_minus_uniform_db_abs_percentiles_50_90_95_99_99p5": [
                float(value)
                for value in np.percentile(
                    np.abs(difference_db), [50, 90, 95, 99, 99.5]
                )
            ],
            "weighted_minus_uniform_db_mean": float(difference_db.mean()),
            "weighted_minus_uniform_db_median": float(np.median(difference_db)),
        },
        "recommendation": {
            "preferred_for_figure3": "stride64_positive_hann",
            "reason": "The positive-Hann overlap removes the visible 64-pixel grid: actual-transition jumps fall to background level and periodic grid energy drops sharply. It preserves the same official network patch predictions and applies no radiometric gain or scene-wise stretch.",
            "required_disclosure": "Label the panel and methods text as MERLIN with weighted-overlap inference; do not describe this adapter as the package's unmodified default aggregator.",
        },
    }

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.8), constrained_layout=True)
    titles = {
        "stride254_uniform": "Uniform, stride 254",
        "stride64_uniform": "Uniform, stride 64",
        "stride64_positive_hann": "Positive-Hann, stride 64",
    }
    for column, name in enumerate(DIRS):
        db = loaded[name][1]
        axes[0, column].imshow(
            db, cmap="gray", vmin=DB_LIMITS[0], vmax=DB_LIMITS[1]
        )
        axes[0, column].set_title(titles[name])
        axes[0, column].axis("off")
        axes[1, column].imshow(
            db[128:896, 128:896],
            cmap="gray",
            vmin=DB_LIMITS[0],
            vmax=DB_LIMITS[1],
        )
        axes[1, column].set_title("Central crop")
        axes[1, column].axis("off")
    artist = axes[0, 3].imshow(
        difference_db,
        cmap="coolwarm",
        vmin=-delta_limit,
        vmax=delta_limit,
    )
    axes[0, 3].set_title("Weighted − uniform 64 (dB)")
    axes[0, 3].axis("off")
    fig.colorbar(artist, ax=axes[0, 3], fraction=0.046, pad=0.04)
    crop_artist = axes[1, 3].imshow(
        difference_db[128:896, 128:896],
        cmap="coolwarm",
        vmin=-delta_limit,
        vmax=delta_limit,
    )
    axes[1, 3].set_title("Central difference (dB)")
    axes[1, 3].axis("off")
    fig.colorbar(crop_artist, ax=axes[1, 3], fraction=0.046, pad=0.04)
    fig.suptitle(
        "MERLIN overlap aggregation audit — identical complex input and patch predictions",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    plt.close(fig)

    from PIL import Image

    preview = Image.open(OUT_PNG)
    preview.thumbnail((2000, 1000), Image.Resampling.LANCZOS)
    preview.save(OUT_PREVIEW, optimize=True)
    comparison["outputs"] = {
        "figure_600dpi": {
            "path": str(OUT_PNG),
            "sha256": sha256_file(OUT_PNG),
        },
        "preview": {
            "path": str(OUT_PREVIEW),
            "sha256": sha256_file(OUT_PREVIEW),
        },
        "json": {"path": str(OUT_JSON)},
    }
    OUT_JSON.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "weighted_transition_mean_ratio": weighted_transition[
                    "mean_ratio"
                ],
                "uniform64_transition_mean_ratio": uniform_transition[
                    "mean_ratio"
                ],
                "mean_jump_reduction_percent": comparison[
                    "weighted_vs_uniform_stride64"
                ]["actual_transition_mean_jump_reduction_percent"],
                "max_jump_reduction_percent": comparison[
                    "weighted_vs_uniform_stride64"
                ]["actual_transition_max_jump_reduction_percent"],
                "period64_grid_energy_reduction_percent": comparison[
                    "weighted_vs_uniform_stride64"
                ]["period64_fundamental_grid_energy_reduction_percent"],
                "json": str(OUT_JSON),
                "figure": str(OUT_PNG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
