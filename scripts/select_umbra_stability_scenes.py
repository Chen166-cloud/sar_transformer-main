"""Select three repeat-geometry Umbra Open Data scene groups for stability tests.

The input is the normalized open-sar-triad GeoJSON catalogue.  Selection uses
only Umbra scenes with downloadable SICD products and groups acquisitions by
site, nominal resolution, polarization, orbit state, look direction, and a
one-degree incidence-angle bin.  The output records one representative test
scene and the remaining acquisitions as a temporal pseudo-ground-truth pool.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "umbra_scene_selection" / "catalog_cache" / "scenes.geojson"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "umbra_scene_selection" / "selected_candidates.json"

SPECS = (
    {
        "site": "Busan Port",
        "scene_type": "dense port, water, ships, and urban structures",
        "grid_lat": 35.10,
        "grid_lon": 129.10,
        "resolution_m": 0.35,
        "polarization": "['VV']",
        "orbit_state": "ascending",
        "look_dir": "left",
        "incidence_bin_deg": 44,
    },
    {
        "site": "Bangkok Suvarnabhumi Airport",
        "scene_type": "runways, terminals, grass, and peri-urban texture",
        "grid_lat": 13.70,
        "grid_lon": 100.75,
        "resolution_m": 0.25,
        "polarization": "['VV']",
        "orbit_state": "descending",
        "look_dir": "left",
        "incidence_bin_deg": 44,
    },
    {
        "site": "Newark New York",
        "scene_type": "dense urban, airport, port, and road network",
        "grid_lat": 40.70,
        "grid_lon": -74.15,
        "resolution_m": 0.35,
        "polarization": "['VV']",
        "orbit_state": "ascending",
        "look_dir": "left",
        "incidence_bin_deg": 44,
    },
)


def centroid_and_bbox(feature: dict) -> tuple[float, float, list[float]]:
    ring = feature["geometry"]["coordinates"][0]
    points = ring[:-1] if ring[0] == ring[-1] else ring
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return sum(ys) / len(ys), sum(xs) / len(xs), [min(xs), min(ys), max(xs), max(ys)]


def official_stac_url(date: str, item_id: str) -> str:
    return (
        "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/stac/"
        f"{date[:4]}/{date[:7]}/{date}/{item_id}/{item_id}.json"
    )


def provider_sidecar_url(sicd_url: str) -> str:
    directory, filename = sicd_url.rsplit("/", 1)
    base = filename.removesuffix("_SICD.nitf").removesuffix("_SICD_MM.nitf")
    return f"{directory}/{base}.stac.v2.json"


def select(features: list[dict], spec: dict) -> dict:
    rows: list[dict] = []
    for feature in features:
        properties = feature["properties"]
        products = properties.get("products") or {}
        if properties.get("provider") != "umbra" or "SICD" not in products:
            continue
        if properties.get("resolution") != spec["resolution_m"]:
            continue
        if properties.get("polarization") != spec["polarization"]:
            continue
        if properties.get("orbit_state") != spec["orbit_state"]:
            continue
        if properties.get("look_dir") != spec["look_dir"]:
            continue
        incidence = float(properties.get("incidence_angle") or -999.0)
        if round(incidence) != spec["incidence_bin_deg"]:
            continue
        latitude, longitude, bbox = centroid_and_bbox(feature)
        if abs(round(latitude / 0.05) * 0.05 - spec["grid_lat"]) > 1e-6:
            continue
        if abs(round(longitude / 0.05) * 0.05 - spec["grid_lon"]) > 1e-6:
            continue
        date = properties["date"]
        item_id = properties["id"]
        rows.append(
            {
                "id": item_id,
                "date": date,
                "incidence_angle_deg": incidence,
                "centroid": [longitude, latitude],
                "bbox": bbox,
                "stac": official_stac_url(date, item_id),
                "provider_metadata": provider_sidecar_url(products["SICD"]),
                "sicd": products["SICD"],
                "gec": products["GEC"],
            }
        )
    rows.sort(key=lambda row: row["date"])
    if len(rows) < 6:
        raise RuntimeError(f"{spec['site']}: only {len(rows)} matching acquisitions")

    intersection = [
        max(row["bbox"][0] for row in rows),
        max(row["bbox"][1] for row in rows),
        min(row["bbox"][2] for row in rows),
        min(row["bbox"][3] for row in rows),
    ]
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        raise RuntimeError(f"{spec['site']}: matching footprints have no common bbox")

    median_lon = statistics.median(row["centroid"][0] for row in rows)
    median_lat = statistics.median(row["centroid"][1] for row in rows)
    median_incidence = statistics.median(row["incidence_angle_deg"] for row in rows)
    middle_date_index = (len(rows) - 1) / 2.0
    scored = []
    for index, row in enumerate(rows):
        score = (
            abs(row["centroid"][0] - median_lon) / 0.01
            + abs(row["centroid"][1] - median_lat) / 0.01
            + abs(row["incidence_angle_deg"] - median_incidence)
            + abs(index - middle_date_index) * 0.02
        )
        scored.append((score, row))
    target = min(scored, key=lambda item: item[0])[1]
    reference_pool = [row for row in rows if row["id"] != target["id"]]

    return {
        **spec,
        "matching_acquisition_count": len(rows),
        "reference_acquisition_count": len(reference_pool),
        "common_bbox_lonlat": intersection,
        "target": target,
        "reference_pool": reference_pool,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalogue = json.loads(args.input.read_text(encoding="utf-8"))
    selections = [select(catalogue["features"], spec) for spec in SPECS]
    payload = {
        "source": "https://github.com/pmuguda/open-sar-triad/blob/main/data/scenes.geojson",
        "source_catalog": "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/stac/catalog.json",
        "selection_rule": (
            "same site, resolution, polarization, orbit state, look direction, "
            "and rounded incidence angle; one representative target held out"
        ),
        "ground_truth_status": (
            "No official speckle-free truth. Reference pool is for a co-registered "
            "temporal pseudo-ground-truth estimate after change masking."
        ),
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        selection["site"]: {
            "matches": selection["matching_acquisition_count"],
            "references": selection["reference_acquisition_count"],
            "target_date": selection["target"]["date"],
            "target_id": selection["target"]["id"],
            "target_sicd": selection["target"]["sicd"],
            "common_bbox_lonlat": selection["common_bbox_lonlat"],
        }
        for selection in selections
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
