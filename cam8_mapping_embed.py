"""
Embedded cam8→arm mapping UI (test-only) for 22_gun_aim_assist_vector.py.
Reuses cam8_arm_grid_calibrator draw/state helpers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_cal = None
_active = False
_top_h = 0
_canvas_w = 0
_bottom_h = 0
_last_sync = 0.0
SYNC_INTERVAL_SEC = 0.3


def _module():
    global _cal
    if _cal is None:
        import cam8_arm_grid_calibrator as _cal_mod
        _cal = _cal_mod
    return _cal


def load_calib_json(json_path: Path) -> int:
    """Load cells, limits, and home reference from JSON into calibrator state."""
    cal = _module()
    n_cells = cal._load_cells_from_file(json_path)
    cal._state.limit_corners = []
    cal._state.reachable_poly_cam8 = None
    try:
        if json_path.is_file():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            corners = data.get("arm_limit_corners")
            if isinstance(corners, list):
                cal._state.limit_corners = list(corners)
            cal._build_reachable_polygon()
            ref = data.get("camera_reference") or data.get("home_reference")
            if isinstance(ref, dict) and "arm_pan" in ref and "arm_tilt" in ref:
                cal._state.home_arm_pan = float(ref["arm_pan"])
                cal._state.home_arm_tilt = float(ref["arm_tilt"])
                cal._state.home_confirmed = True
            else:
                cal._state.home_confirmed = False
    except Exception as exc:
        print(f"[MAP] load limits/home warning: {exc}")
    return n_cells


def enter(window_name: str, json_path: Path, canvas_w: int, top_h: int, bottom_h: int) -> int:
    """Activate mapping test mode and register mouse callback."""
    global _active, _top_h, _canvas_w, _bottom_h, _last_sync
    cal = _module()
    cal._state.phase = "calibration"
    cal._state.test_only_mode = True
    cal._state.test_go_pending = False
    cal._state.selected_cell = None
    cal._state.cam8_click = None
    cal._state.selected_source_xy = None
    cal._state.show_reachable_overlay = True
    n = load_calib_json(json_path)
    _active = True
    _top_h = int(top_h)
    _canvas_w = int(canvas_w)
    _bottom_h = int(bottom_h)
    _last_sync = 0.0
    cv2.setMouseCallback(window_name, cal._on_mouse, param={"cam8_top_h": _top_h})
    print(f"[MAP] Mapping mode ON — preloaded {n} cells from {json_path.name}")
    return n


def leave(window_name: str) -> None:
    """Deactivate mapping mode and clear mouse callback."""
    global _active
    cal = _module()
    cal._state.test_only_mode = False
    cal._state.test_go_pending = False
    _active = False
    try:
        cv2.setMouseCallback(window_name, lambda *args: None)
    except cv2.error:
        pass
    print("[MAP] Mapping mode OFF")


def is_active() -> bool:
    return _active


def tick_arm_sync(arm) -> None:
    global _last_sync
    now = time.time()
    if now - _last_sync < SYNC_INTERVAL_SEC:
        return
    _last_sync = now
    if arm is not None and hasattr(arm, "sync_position_from_grbl"):
        try:
            arm.sync_position_from_grbl()
        except Exception:
            pass


def tick_test_go(arm) -> None:
    cal = _module()
    if cal._state.test_only_mode and cal._state.test_go_pending:
        cal._state.test_go_pending = False
        cal._test_selected_cell(arm)


def confirm_cell(arm) -> None:
    _module()._confirm_calibration(arm)


def go_to_reference(arm, json_path: Path) -> bool:
    """Move arm to saved camera/home reference (same as calibrator H key)."""
    if arm is None:
        print("[MAP] need arm connected for go-to-ref")
        return False
    return bool(_module()._goto_reference(arm, json_path))


def save_json(arm, json_path: Path) -> bool:
    cal = _module()
    st = cal._state
    if st.cam8_w <= 0 or st.cam8_h <= 0:
        print("[MAP] No cam8 frame size yet — cannot save.")
        return False
    home_pan = float(st.home_arm_pan)
    home_tilt = float(st.home_arm_tilt)
    if not st.home_confirmed:
        ref = cal._get_ref_pan_tilt_from_file(json_path)
        if ref is not None:
            home_pan, home_tilt = ref
    return bool(
        cal.save_calibration(
            st.cell_to_pan_tilt,
            st.cam8_w,
            st.cam8_h,
            st.limit_corners,
            home_pan,
            home_tilt,
            output_path=json_path,
        )
    )


def _blank(h: int, w: int, msg: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, msg, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
    return img


def render_composite(
    frame8: Optional[np.ndarray],
    frame4: np.ndarray,
    cam8_status: str = "",
) -> np.ndarray:
    """Build cam8 top + cam4 bottom display (same layout as calibrator)."""
    cal = _module()
    canvas_w = _canvas_w
    top_h = _top_h
    bottom_h = _bottom_h

    if frame8 is not None and frame8.size > 0:
        cal._state.cam8_h, cal._state.cam8_w = frame8.shape[:2]
        cal._update_home_cam8_position()
        disp8 = cal._letterbox_cam8(frame8, canvas_w, top_h)
    else:
        msg = cam8_status.strip() if cam8_status else "cam8 no signal"
        disp8 = _blank(top_h, canvas_w, msg)

    cal._draw_grid_overlay(disp8, canvas_w, top_h)
    cal._draw_reachable_polygon(disp8)
    cal._draw_home_mark(disp8, canvas_w, top_h, cal._state.home_confirmed)
    cal._draw_confirmed_points(disp8, canvas_w, top_h)
    cal._draw_pending_click(disp8, canvas_w, top_h)

    f4 = frame4
    if f4.ndim == 2:
        f4 = cv2.cvtColor(f4, cv2.COLOR_GRAY2BGR)
    elif f4.shape[2] == 4:
        f4 = cv2.cvtColor(f4, cv2.COLOR_BGRA2BGR)
    disp4, cam4_map = cal._fit_frame_to_layout(f4, canvas_w, bottom_h)
    cal._draw_cam4_crosshair(disp4, canvas_w, bottom_h, cam4_map)

    display = np.vstack([disp8, disp4])
    return display


def mapped_cell_count() -> Tuple[int, int]:
    cal = _module()
    mapped = len(cal._state.cell_to_pan_tilt)
    total = cal.GRID_ROWS * cal.GRID_COLS
    return mapped, total


def selected_cell_label() -> Tuple[str, Tuple[int, int, int]]:
    """Return (label, BGR) for currently selected grid cell."""
    cell = _module()._state.selected_cell
    if cell is None:
        return "CELL:---", (120, 120, 120)
    row, col = cell
    key = f"{int(row)}_{int(col)}"
    mapped = key in _module()._state.cell_to_pan_tilt
    color = (0, 255, 180) if mapped else (0, 220, 255)
    return f"CELL:r{row}c{col}", color


def ref_hud_label_and_color() -> Tuple[str, Tuple[int, int, int]]:
    """Return (label, BGR) for home/camera reference status."""
    if _module()._state.home_confirmed:
        return "REF:OK", (0, 220, 0)
    return "REF:---", (0, 100, 255)
