#migrate_legacy_active_label_dataset.py
import argparse
import csv
import json
import os
from collections import Counter

from canonical_path import (
    CANONICAL_PATH_SCHEMA_VERSION,
    build_storage_layout,
    classify_legacy_row_migratability,
)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _canonicalize_legacy_row(row):
    frame_w = max(1.0, _safe_float(row.get("frame_w"), 1.0))
    frame_h = max(1.0, _safe_float(row.get("frame_h"), 1.0))
    rect_x = _safe_float(row.get("x"), 0.0)
    rect_y = _safe_float(row.get("y"), 0.0)
    rect_w = _safe_float(row.get("w"), 0.0)
    rect_h = _safe_float(row.get("h"), 0.0)
    center_x = rect_x + (rect_w * 0.5)
    center_y = rect_y + (rect_h * 0.5)
    payload = dict(row)
    payload["schema_version"] = CANONICAL_PATH_SCHEMA_VERSION
    payload["camera_name"] = row.get("camera_name", "")
    payload["source_camera_name"] = row.get("source_camera_name", row.get("camera_name", ""))
    payload["source_region"] = row.get("source_region", "")
    payload["crosses_seam"] = row.get("crosses_seam", "0")
    payload["global_frame_w"] = row.get("global_frame_w", row.get("frame_w", ""))
    payload["global_frame_h"] = row.get("global_frame_h", row.get("frame_h", ""))
    payload["x_src"] = f"{center_x:.6f}"
    payload["y_src"] = f"{center_y:.6f}"
    payload["x_norm"] = f"{center_x / frame_w:.6f}"
    payload["y_norm"] = f"{center_y / frame_h:.6f}"
    payload["horizon_y_at_x"] = row.get("horizon_y_at_x", "")
    payload["y_from_horizon_norm"] = row.get("y_from_horizon_norm", "")
    payload["bbox_w_norm"] = f"{rect_w / frame_w:.6f}"
    payload["bbox_h_norm"] = f"{rect_h / frame_h:.6f}"
    return payload


def migrate_legacy_dataset(project_dir, legacy_csv=None):
    layout = build_storage_layout(project_dir, CANONICAL_PATH_SCHEMA_VERSION)
    legacy_csv = legacy_csv or os.path.join(project_dir, "active_teach", "labels.csv")
    output_dir = layout["datasets"]["legacy_migrated_candidate"]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "labels.csv")
    summary_path = layout["metadata"]["legacy_to_canonical"]
    if not os.path.exists(legacy_csv):
        raise FileNotFoundError(f"Legacy labels not found: {legacy_csv}")

    counters = Counter()
    migrated_rows = []
    with open(legacy_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        for row in reader:
            status = classify_legacy_row_migratability(row)
            counters[status] += 1
            if status == "legacy_unmappable":
                continue
            migrated_rows.append(_canonicalize_legacy_row(row))

    canonical_header = [
        "schema_version",
        "timestamp",
        "label",
        "obj_id",
        "camera_name",
        "source_camera_name",
        "source_region",
        "crosses_seam",
        "global_frame_w",
        "global_frame_h",
        "frame_w",
        "frame_h",
        "x",
        "y",
        "w",
        "h",
        "x_src",
        "y_src",
        "x_norm",
        "y_norm",
        "horizon_y_at_x",
        "y_from_horizon_norm",
        "bbox_w_norm",
        "bbox_h_norm",
    ] + [f"f{i}" for i in range(16)] + ["image_path"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=canonical_header)
        writer.writeheader()
        for row in migrated_rows:
            writer.writerow({key: row.get(key, "") for key in canonical_header})

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = {
        "schema_version": CANONICAL_PATH_SCHEMA_VERSION,
        "legacy_csv": legacy_csv,
        "output_csv": output_csv,
        "counts": dict(counters),
        "legacy_header": headers,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy active label data to canonical v1 candidate dataset.")
    parser.add_argument("--project-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--legacy-csv", default=None)
    args = parser.parse_args()
    summary = migrate_legacy_dataset(args.project_dir, legacy_csv=args.legacy_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

