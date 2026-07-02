#canonical_path.py

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


CANONICAL_PATH_SCHEMA_VERSION = "canonical_v2"


def build_storage_layout(base_dir: str, schema_version: str = CANONICAL_PATH_SCHEMA_VERSION) -> Dict[str, Dict[str, str]]:
    datasets_root = os.path.join(base_dir, "datasets", "path")
    models_root = os.path.join(base_dir, "models", "path")
    runs_root = os.path.join(base_dir, "runs", "path")
    metadata_root = os.path.join(base_dir, "metadata")
    return {
        "datasets": {
            "legacy_old_train": os.path.join(datasets_root, "legacy_old_train"),
            "canonical_new_train": os.path.join(datasets_root, f"{schema_version}_train"),
            "canonical_new_val": os.path.join(datasets_root, f"{schema_version}_val"),
            "legacy_migrated_candidate": os.path.join(
                datasets_root, f"legacy_migrated_candidate_{schema_version}"
            ),
        },
        "models": {
            "legacy_last": os.path.join(models_root, "legacy_v_last"),
            "canonical_baseline": os.path.join(models_root, f"{schema_version}_baseline"),
            "canonical_plus_migrated": os.path.join(models_root, f"{schema_version}_plus_migrated"),
        },
        "runs": {
            "canonical_baseline_root": os.path.join(runs_root, f"{schema_version}_baseline"),
        },
        "metadata": {
            "path_schema_dir": os.path.join(metadata_root, "path_schema"),
            "path_schema": os.path.join(metadata_root, "path_schema", f"{schema_version}.json"),
            "migrations_dir": os.path.join(metadata_root, "migrations"),
            "legacy_to_canonical": os.path.join(
                metadata_root, "migrations", f"legacy_to_{schema_version}.json"
            ),
        },
    }


def ensure_storage_layout(base_dir: str, schema_version: str = CANONICAL_PATH_SCHEMA_VERSION) -> Dict[str, Dict[str, str]]:
    layout = build_storage_layout(base_dir, schema_version=schema_version)
    for group in layout.values():
        for path in group.values():
            os.makedirs(path if os.path.splitext(path)[1] == "" else os.path.dirname(path), exist_ok=True)
    schema_path = layout["metadata"]["path_schema"]
    if not os.path.exists(schema_path):
        payload = {
            "schema_version": schema_version,
            "space": "normalized_horizon_relative",
            "fields": [
                "camera_name",
                "source_camera_name",
                "source_region",
                "crosses_seam",
                "frame_w",
                "frame_h",
                "x_src",
                "y_src",
                "x_norm",
                "y_norm",
                "horizon_y_at_x",
                "y_from_horizon_norm",
                "bbox_w_norm",
                "bbox_h_norm",
            ],
            "notes": {
                "display_coordinates": "never used as training ground truth",
                "fov_phase": "optional future phase",
            },
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return layout


def build_path_ml_bundle(base_dir: str, bundle_id: str) -> Dict[str, str]:
    """
    Isolated train/test/models tree for Active Label path data (separate from canonical_v2_*).
    Layout: datasets/path/<bundle_id>/{train,test,models}/
    """
    root = os.path.join(base_dir, "datasets", "path", bundle_id)
    return {
        "root": root,
        "train": os.path.join(root, "train"),
        "test": os.path.join(root, "test"),
        "models": os.path.join(root, "models"),
    }


def ensure_path_ml_bundle(base_dir: str, bundle_id: str) -> Dict[str, str]:
    layout = build_path_ml_bundle(base_dir, bundle_id)
    for key in ("train", "test", "models"):
        os.makedirs(layout[key], exist_ok=True)
    return layout


def source_to_normalized(x_src: float, y_src: float, frame_w: int, frame_h: int) -> Tuple[float, float]:
    fw = float(max(1, int(frame_w)))
    fh = float(max(1, int(frame_h)))
    return float(x_src) / fw, float(y_src) / fh


def normalized_to_source(x_norm: float, y_norm: float, frame_w: int, frame_h: int) -> Tuple[int, int]:
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    x_src = int(round(float(x_norm) * fw))
    y_src = int(round(float(y_norm) * fh))
    return max(0, min(fw - 1, x_src)), max(0, min(fh - 1, y_src))


def sample_horizon_y(horizon_points: Optional[Sequence[Sequence[float]]], x_value: float) -> Optional[float]:
    if not horizon_points or len(horizon_points) < 2:
        return None
    pts = sorted((float(pt[0]), float(pt[1])) for pt in horizon_points if len(pt) >= 2)
    if len(pts) < 2:
        return None
    x = float(x_value)
    if x <= pts[0][0]:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    elif x >= pts[-1][0]:
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
    else:
        for idx in range(1, len(pts)):
            x1, y1 = pts[idx - 1]
            x2, y2 = pts[idx]
            if x1 <= x <= x2:
                break
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return float(y1)
    t = (x - x1) / dx
    return float(y1 + (y2 - y1) * t)


def _point_in_rect(x_value: float, y_value: float, rect: Optional[Dict[str, int]]) -> bool:
    if not rect:
        return False
    x = float(x_value)
    y = float(y_value)
    return (
        rect["x"] <= x <= (rect["x"] + rect["w"])
        and rect["y"] <= y <= (rect["y"] + rect["h"])
    )


def _rect_center(rect: Dict[str, int]) -> Tuple[float, float]:
    return rect["x"] + (rect["w"] * 0.5), rect["y"] + (rect["h"] * 0.5)


def _overlap_rect(left_rect: Optional[Dict[str, int]], right_rect: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if not left_rect or not right_rect:
        return None
    x1 = max(left_rect["x"], right_rect["x"])
    y1 = max(left_rect["y"], right_rect["y"])
    x2 = min(left_rect["x"] + left_rect["w"], right_rect["x"] + right_rect["w"])
    y2 = min(left_rect["y"] + left_rect["h"], right_rect["y"] + right_rect["h"])
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}


def stitched_point_to_source_camera(
    x_value: float,
    y_value: float,
    stitched_layout: Dict[str, Optional[Dict[str, int]]],
    source_camera_names: Optional[Dict[str, str]] = None,
    preferred_source: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    left_rect = stitched_layout.get("left")
    right_rect = stitched_layout.get("right")
    in_left = _point_in_rect(x_value, y_value, left_rect)
    in_right = _point_in_rect(x_value, y_value, right_rect)
    if not in_left and not in_right:
        return None

    source_region = "overlap" if (in_left and in_right) else ("left" if in_left else "right")
    resolved_region = source_region
    if source_region == "overlap":
        if preferred_source in ("left", "right"):
            resolved_region = preferred_source
        elif left_rect and right_rect:
            left_cx, _ = _rect_center(left_rect)
            right_cx, _ = _rect_center(right_rect)
            resolved_region = "left" if abs(float(x_value) - left_cx) <= abs(float(x_value) - right_cx) else "right"
        else:
            resolved_region = "left" if in_left else "right"

    region_rect = left_rect if resolved_region == "left" else right_rect
    if not region_rect:
        return None

    local_x = float(x_value) - float(region_rect["x"])
    local_y = float(y_value) - float(region_rect["y"])
    local_x = max(0.0, min(float(region_rect["w"]), local_x))
    local_y = max(0.0, min(float(region_rect["h"]), local_y))
    overlap_rect = _overlap_rect(left_rect, right_rect)
    source_camera_name = None
    if source_camera_names is not None:
        source_camera_name = source_camera_names.get(resolved_region)
    return {
        "source_region": source_region,
        "resolved_region": resolved_region,
        "source_camera_name": source_camera_name or resolved_region,
        "crosses_seam": source_region == "overlap",
        "local_x": local_x,
        "local_y": local_y,
        "region_rect": dict(region_rect),
        "source_frame_w": int(region_rect["w"]),
        "source_frame_h": int(region_rect["h"]),
        "overlap_rect": overlap_rect,
    }


def canonicalize_source_point(
    x_src: float,
    y_src: float,
    frame_shape: Sequence[int],
    horizon_points: Optional[Sequence[Sequence[float]]] = None,
    camera_name: Optional[str] = None,
    source_camera_name: Optional[str] = None,
    source_region: Optional[str] = None,
    crosses_seam: bool = False,
) -> Dict[str, object]:
    frame_h = int(frame_shape[0]) if len(frame_shape) >= 1 else 1
    frame_w = int(frame_shape[1]) if len(frame_shape) >= 2 else 1
    x_norm, y_norm = source_to_normalized(x_src, y_src, frame_w, frame_h)
    horizon_y = sample_horizon_y(horizon_points, x_src)
    y_from_horizon_norm = None
    if horizon_y is not None:
        y_from_horizon_norm = (float(y_src) - float(horizon_y)) / float(max(1, frame_h))
    return {
        "schema_version": CANONICAL_PATH_SCHEMA_VERSION,
        "camera_name": camera_name,
        "source_camera_name": source_camera_name or camera_name,
        "source_region": source_region or "single",
        "crosses_seam": bool(crosses_seam),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "x_src": float(x_src),
        "y_src": float(y_src),
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
        "horizon_y_at_x": None if horizon_y is None else float(horizon_y),
        "y_from_horizon_norm": None if y_from_horizon_norm is None else float(y_from_horizon_norm),
    }


def canonicalize_detection(
    rect: Sequence[float],
    frame_shape: Sequence[int],
    horizon_points: Optional[Sequence[Sequence[float]]] = None,
    camera_name: Optional[str] = None,
    point_resolver: Optional[Callable[[float, float], Optional[Dict[str, object]]]] = None,
) -> Dict[str, object]:
    x, y, w_box, h_box = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    center_x = x + (w_box * 0.5)
    center_y = y + (h_box * 0.5)
    resolved = point_resolver(center_x, center_y) if point_resolver is not None else None
    if resolved is not None:
        point_payload = canonicalize_source_point(
            x_src=resolved["local_x"],
            y_src=resolved["local_y"],
            frame_shape=(resolved["source_frame_h"], resolved["source_frame_w"]),
            horizon_points=None,
            camera_name=camera_name,
            source_camera_name=str(resolved.get("source_camera_name")),
            source_region=str(resolved.get("resolved_region") or resolved.get("source_region")),
            crosses_seam=bool(resolved.get("crosses_seam", False)),
        )
        horizon_global = sample_horizon_y(horizon_points, center_x)
        if horizon_global is not None:
            region_rect = resolved.get("region_rect") or {"y": 0}
            local_horizon = float(horizon_global) - float(region_rect.get("y", 0))
            point_payload["horizon_y_at_x"] = float(local_horizon)
            point_payload["y_from_horizon_norm"] = (
                float(resolved["local_y"]) - float(local_horizon)
            ) / float(max(1, int(resolved["source_frame_h"])))
    else:
        point_payload = canonicalize_source_point(
            x_src=center_x,
            y_src=center_y,
            frame_shape=frame_shape,
            horizon_points=horizon_points,
            camera_name=camera_name,
            source_camera_name=camera_name,
            source_region="single",
            crosses_seam=False,
        )

    bbox_w_norm = float(w_box) / float(max(1, int(point_payload["frame_w"])))
    bbox_h_norm = float(h_box) / float(max(1, int(point_payload["frame_h"])))
    point_payload.update(
        {
            "rect_x": float(x),
            "rect_y": float(y),
            "rect_w": float(w_box),
            "rect_h": float(h_box),
            "bbox_w_norm": bbox_w_norm,
            "bbox_h_norm": bbox_h_norm,
        }
    )
    return point_payload


def canonicalize_trajectory(
    points: Iterable[Sequence[float]],
    frame_shape: Sequence[int],
    horizon_points: Optional[Sequence[Sequence[float]]] = None,
    camera_name: Optional[str] = None,
    point_resolver: Optional[Callable[[float, float], Optional[Dict[str, object]]]] = None,
) -> List[Dict[str, object]]:
    canonical_points: List[Dict[str, object]] = []
    for point in points:
        if len(point) < 2:
            continue
        x_src = float(point[0])
        y_src = float(point[1])
        resolved = point_resolver(x_src, y_src) if point_resolver is not None else None
        if resolved is not None:
            payload = canonicalize_source_point(
                x_src=resolved["local_x"],
                y_src=resolved["local_y"],
                frame_shape=(resolved["source_frame_h"], resolved["source_frame_w"]),
                horizon_points=None,
                camera_name=camera_name,
                source_camera_name=str(resolved.get("source_camera_name")),
                source_region=str(resolved.get("resolved_region") or resolved.get("source_region")),
                crosses_seam=bool(resolved.get("crosses_seam", False)),
            )
            horizon_global = sample_horizon_y(horizon_points, x_src)
            if horizon_global is not None:
                region_rect = resolved.get("region_rect") or {"y": 0}
                local_horizon = float(horizon_global) - float(region_rect.get("y", 0))
                payload["horizon_y_at_x"] = float(local_horizon)
                payload["y_from_horizon_norm"] = (
                    float(resolved["local_y"]) - float(local_horizon)
                ) / float(max(1, int(resolved["source_frame_h"])))
        else:
            payload = canonicalize_source_point(
                x_src=x_src,
                y_src=y_src,
                frame_shape=frame_shape,
                horizon_points=horizon_points,
                camera_name=camera_name,
                source_camera_name=camera_name,
                source_region="single",
                crosses_seam=False,
            )
        canonical_points.append(payload)
    return canonical_points


def classify_legacy_row_migratability(row: Dict[str, object]) -> str:
    required = ("frame_w", "frame_h", "x", "y", "w", "h")
    if all(str(row.get(key, "")).strip() for key in required):
        if str(row.get("camera_name", "")).strip():
            return "canonical_ready"
        return "canonical_partial"
    return "legacy_unmappable"

