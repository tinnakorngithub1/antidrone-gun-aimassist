"""
Embedded cam4 pixel/degree calibrator for 22_gun_aim_assist_vector.py.
Reuses cam4_arm_mouse_grid_calibrator state, mouse handler, and px/deg test flow.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

_cal = None
_active = False
_last_loop_time = 0.0
_last_sync_time = 0.0
_camera_name = "cam4"
SYNC_INTERVAL_SEC = 0.25
# Fine-tune via joystick: cap speed so moves stay precise (deg/s at full deflection)
PXDEG_JOY_FINETUNE_MAX_RATE_DEG = 6.0


def _module():
    global _cal
    if _cal is None:
        import cam4_arm_mouse_grid_calibrator as _cal_mod
        _cal = _cal_mod
    return _cal


def _set_paths(camera_name: str) -> None:
    cal = _module()
    cam = camera_name.strip() or "cam4"
    cal.PX_DEG_JSON_PATH = cal.CALIBRATION_DIR / f"{cam}_pixel_per_degree.json"
    cal.JSON_PATH = cal.CALIBRATION_DIR / f"{cam}_mouse_grid_lookup.json"
    cal.TEST_MOVE_LOG_PATH = cal.CALIBRATION_DIR / f"{cam}_test_move_log_homography.jsonl"
    cal.TEST_MOVE_LOG_PXDEG_PATH = cal.CALIBRATION_DIR / f"{cam}_test_move_log_pxdeg.jsonl"


def _ensure_stub_json(cal) -> None:
    if cal.JSON_PATH.is_file():
        return
    cal.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ow, oh = cal.REF_OUTPUT_W, cal.REF_OUTPUT_H
    data = {
        "output_width": ow,
        "output_height": oh,
        "crosshair": {"x": float(ow) / 2.0, "y": float(oh) / 2.0},
        "grid_cols": cal.GRID_COLS,
        "grid_rows": cal.GRID_ROWS,
        "cells": {},
        "center_fine_grid": {
            "fine_rows": cal.FINE_ROWS,
            "fine_cols": cal.FINE_COLS,
            "fine_cells": {},
        },
    }
    with open(cal.JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _reset_test_state(cal) -> None:
    st = cal._state
    st.mode = "test"
    st.use_pixel_per_degree = True
    st.test_continuous = False
    st.test_track_sim = False
    st.test_track_csrt = False
    st.test_teaching = False
    st.pending_test_move = None
    st.pending_test_capture = None
    st.test_reference_capture = None
    st.smooth_target = None
    st.smooth_start = None
    st.test_awaiting_home = False
    st.test_awaiting_confirm = False
    st.test_move_log_pending = False
    st.test_home_required_message_until = 0.0
    st.limit_clipped_message_until = 0.0
    st.verify_index = None
    st.verify_points = []
    st.continuous_target_pan = None
    st.continuous_target_tilt = None
    st.continuous_velocity_mm_s = 0.0


def _load_px_into_state(cal) -> None:
    st = cal._state
    px_deg_data = cal._load_pixel_per_degree_json()
    if px_deg_data is not None:
        try:
            st.px_per_deg_x = float(px_deg_data.get("pixel_per_degree_x", cal.PX_PER_DEG_X_DEFAULT))
            st.px_per_deg_y = float(px_deg_data.get("pixel_per_degree_y", cal.PX_PER_DEG_Y_DEFAULT))
        except (TypeError, ValueError):
            st.px_per_deg_x = cal.PX_PER_DEG_X_DEFAULT
            st.px_per_deg_y = cal.PX_PER_DEG_Y_DEFAULT
    else:
        st.px_per_deg_x = cal.PX_PER_DEG_X_DEFAULT
        st.px_per_deg_y = cal.PX_PER_DEG_Y_DEFAULT


def enter(window_name: str, arm: Any, camera_name: str) -> None:
    """Activate embedded px/deg test calibrator on the main 22 window."""
    global _active, _last_loop_time, _last_sync_time, _camera_name
    cal = _module()
    _camera_name = camera_name.strip() or "cam4"
    _set_paths(_camera_name)
    _ensure_stub_json(cal)
    _reset_test_state(cal)
    _load_px_into_state(cal)
    st = cal._state
    st.output_w = cal.REF_OUTPUT_W
    st.output_h = cal.REF_OUTPUT_H
    st.crosshair_x = st.output_w / 2.0
    st.crosshair_y = st.output_h / 2.0
    st.display_w = 0
    st.display_h = 0
    _active = True
    _last_loop_time = time.time()
    _last_sync_time = 0.0
    if arm is not None and hasattr(arm, "_clear_drive_verify"):
        try:
            arm._clear_drive_verify()
        except Exception:
            pass
    cv2.setMouseCallback(window_name, cal._on_mouse, param={"arm": arm})
    print(
        f"[PXDEG] Calibrator ON ({_camera_name}) — click→ENTER move | "
        f"joy/arrows fine-tune | Btn0=confirm | H=log+home | R=recal | G/Esc=exit"
    )


def leave(window_name: str) -> None:
    """Deactivate embedded calibrator and clear mouse callback."""
    global _active
    cal = _module()
    _active = False
    try:
        cv2.setMouseCallback(window_name, lambda *args: None)
    except cv2.error:
        pass
    print("[PXDEG] Calibrator OFF")


def is_active() -> bool:
    return _active


def set_display_size(display_w: int, display_h: int) -> None:
    cal = _module()
    cal._state.display_w = int(display_w)
    cal._state.display_h = int(display_h)


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize to calibrator reference resolution (3840x2160)."""
    cal = _module()
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    ow, oh = cal.REF_OUTPUT_W, cal.REF_OUTPUT_H
    if frame.shape[1] != ow or frame.shape[0] != oh:
        frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LINEAR)
    st = cal._state
    st.output_w = ow
    st.output_h = oh
    st.crosshair_x = ow / 2.0
    st.crosshair_y = oh / 2.0
    return frame


def _draw_crosshair_and_pending(cal, frame: np.ndarray) -> None:
    st = cal._state
    h, w = frame.shape[:2]
    cx = int(st.crosshair_x)
    cy = int(st.crosshair_y)
    color = cal.COLOR_CROSSHAIR
    r0 = cal.RETICLE_RADIUS_PX[0]
    cv2.line(frame, (0, cy), (max(0, cx - r0), cy), color, 2)
    cv2.line(frame, (min(w, cx + r0), cy), (w, cy), color, 2)
    cv2.line(frame, (cx, 0), (cx, max(0, cy - r0)), color, 2)
    cv2.line(frame, (cx, min(h, cy + r0)), (cx, h), color, 2)
    for r in cal.RETICLE_RADIUS_PX:
        cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)
    pending = getattr(st, "pending_test_move", None)
    if pending is not None:
        pt1 = (cx, cy)
        pt2 = (int(pending[2]), int(pending[3]))
        cv2.line(frame, pt1, pt2, cal.COLOR_PATH_YELLOW, 3)
        cv2.circle(frame, pt2, 8, cal.COLOR_PATH_YELLOW, 2)


def _step_message(cal, arm: Any) -> str:
    st = cal._state
    ax = getattr(arm, "pos_x", 0.0) if arm is not None else 0.0
    ay = getattr(arm, "pos_y", 0.0) if arm is not None else 0.0
    now = time.time()
    if getattr(st, "pending_test_move", None) is not None:
        c1, c2 = st.pending_test_move[0], st.pending_test_move[1]
        return f"Target X={c1:.1f} Y={c2:.1f} — Enter=move"
    if getattr(st, "test_awaiting_confirm", False):
        return "Joy/Arrows fine-tune | Btn0/Enter=confirm"
    if getattr(st, "test_awaiting_home", False):
        return "Homing — wait"
    if getattr(st, "test_home_required_message_until", 0.0) > now:
        return "Go home first — H"
    if abs(ax) > 1.0 or abs(ay) > 1.0:
        return "Not at home — H first"
    return "Click point on image"


def step_label(arm: Any = None) -> Tuple[str, Tuple[int, int, int]]:
    """Short step hint for bottom HUD status row."""
    if not _active:
        return "", (120, 120, 120)
    cal = _module()
    msg = _step_message(cal, arm)
    if len(msg) > 28:
        msg = msg[:25] + "..."
    return f"STEP:{msg}", (200, 200, 200)


def px_short_label() -> str:
    if not _active:
        return ""
    cal = _module()
    st = cal._state
    if st.px_per_deg_x is None or st.px_per_deg_y is None:
        return "PX:---"
    return f"PX:{st.px_per_deg_x:.0f},{st.px_per_deg_y:.0f}"


def _draw_hud(cal, frame: np.ndarray, arm: Any) -> None:
    """Top-left overlay — avoids overlap with bottom HUD bar."""
    st = cal._state
    lh = int(cal.HUD_LEFT_LINE_H)
    ax = getattr(arm, "pos_x", 0.0)
    ay = getattr(arm, "pos_y", 0.0)
    now = time.time()
    y0 = int(72 * cal.HUD_TEXT_SCALE)
    msg = _step_message(cal, arm)

    cv2.putText(
        frame,
        f"PX/DEG [{_camera_name}]",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        cal.FONT_HUD,
        (0, 220, 255),
        cal.THICKNESS_THIN,
    )
    cv2.putText(
        frame,
        msg,
        (10, y0 + lh),
        cv2.FONT_HERSHEY_SIMPLEX,
        cal.FONT_HUD,
        cal.COLOR_HUD_RED,
        cal.THICKNESS_THIN,
    )
    cv2.putText(
        frame,
        f"px=({st.px_per_deg_x:.1f},{st.px_per_deg_y:.1f})  arm X={ax:.1f} Y={ay:.1f}",
        (10, y0 + 2 * lh),
        cv2.FONT_HERSHEY_SIMPLEX,
        cal.FONT_HUD,
        cal.COLOR_HUD_RED,
        cal.THICKNESS_THIN,
    )
    if getattr(st, "recal_message_until", 0.0) > now and getattr(st, "recal_message_text", None):
        cv2.putText(
            frame,
            st.recal_message_text,
            (10, y0 + 3 * lh),
            cv2.FONT_HERSHEY_SIMPLEX,
            cal.FONT_HUD,
            (0, 255, 255),
            cal.THICKNESS_THIN,
        )


def tick(frame: np.ndarray, arm: Any) -> np.ndarray:
    """Per-frame physics + overlay. Returns the same frame (modified in place)."""
    global _last_loop_time, _last_sync_time
    if not _active or frame is None or frame.size == 0:
        return frame

    cal = _module()
    st = cal._state
    frame = prepare_frame(frame)
    h, w = frame.shape[:2]
    now = time.time()
    dt = now - _last_loop_time if _last_loop_time else 0.02
    _last_loop_time = now

    if (now - _last_sync_time) >= SYNC_INTERVAL_SEC:
        _last_sync_time = now
        if arm is not None and hasattr(arm, "sync_position_from_grbl"):
            try:
                arm.sync_position_from_grbl()
            except Exception:
                pass

    if getattr(st, "test_awaiting_home", False) and arm is not None:
        px_arm = getattr(arm, "pos_x", 0.0)
        py_arm = getattr(arm, "pos_y", 0.0)
        if abs(px_arm) < 1.0 and abs(py_arm) < 1.0:
            st.test_awaiting_home = False

    if st.smooth_target is not None and st.smooth_start is not None and arm is not None:
        elapsed = now - st.smooth_start_time
        t = min(1.0, elapsed / max(1e-6, st.smooth_duration))
        t_ease = t * t * (3.0 - 2.0 * t)
        pan = st.smooth_start[0] + t_ease * (st.smooth_target[0] - st.smooth_start[0])
        tilt = st.smooth_start[1] + t_ease * (st.smooth_target[1] - st.smooth_start[1])
        arm.move_absolute(pan, tilt, blocking=False)
        if t >= 1.0:
            try:
                arm.move_absolute(st.smooth_target[0], st.smooth_target[1], blocking=False)
            except Exception:
                pass
            st.smooth_target = None
            st.smooth_start = None
            st.last_test_crosshair_pixel = (st.crosshair_x, st.crosshair_y)
            st.last_test_actual_arm = None
            st.last_test_actual_pixel = None
            st.test_awaiting_confirm = True
            st.test_move_log_pending = True

    if (
        st.mode == "test"
        and st.pending_test_capture is not None
        and st.output_w > 0
        and st.output_h > 0
    ):
        px, py = st.pending_test_capture[0], st.pending_test_capture[1]
        half_w, half_h = cal.TEST_CAPTURE_HALF_W, cal.TEST_CAPTURE_HALF_H
        x0 = max(0, int(px) - half_w)
        y0 = max(0, int(py) - half_h)
        x1 = min(w, int(px) + half_w)
        y1 = min(h, int(py) + half_h)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1].copy()
            roi_h, roi_w = roi.shape[:2]
            cx = int(px - x0)
            cy = int(py - y0)
            cross_len = int(60 * cal.HUD_TEXT_SCALE)
            cross_thick = max(2, int(8 * cal.HUD_TEXT_SCALE))
            cv2.line(roi, (max(0, cx - cross_len), cy), (min(roi_w, cx + cross_len), cy), cal.COLOR_CLICK_MARKER, cross_thick)
            cv2.line(roi, (cx, max(0, cy - cross_len)), (cx, min(roi_h, cy + cross_len)), cal.COLOR_CLICK_MARKER, cross_thick)
            st.test_reference_capture = roi
        st.pending_test_capture = None

    _draw_crosshair_and_pending(cal, frame)
    _draw_hud(cal, frame, arm)
    return frame


def _save_current_px_deg(cal) -> bool:
    st = cal._state
    if st.px_per_deg_x is None or st.px_per_deg_y is None:
        return False
    ow = st.output_w or cal.REF_OUTPUT_W
    oh = st.output_h or cal.REF_OUTPUT_H
    data = {
        "output_width": ow,
        "output_height": oh,
        "crosshair": {"x": float(ow) / 2.0, "y": float(oh) / 2.0},
        "pixel_per_degree_x": float(st.px_per_deg_x),
        "pixel_per_degree_y": float(st.px_per_deg_y),
    }
    cal.PX_DEG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(cal.PX_DEG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[PXDEG] Saved {cal.PX_DEG_JSON_PATH.name}")
    return True


def handle_key(key: int, arm: Any) -> str:
    """
    Process calibrator keys while embedded.
    Returns: 'none', 'recalibrated', 'saved'.
    Exit (G/Esc) is handled by 22 itself.
    """
    if not _active or arm is None:
        return "none"

    cal = _module()
    st = cal._state
    key8 = key & 0xFF if key >= 0 else -1

    if getattr(st, "pending_test_move", None) is not None:
        if key8 in (cal.KEY_ENTER, cal.KEY_ENTER_LF):
            cmd1, cmd2, px_out, py_out = st.pending_test_move
            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            pos_x = getattr(arm, "pos_x", 0.0)
            pos_y = getattr(arm, "pos_y", 0.0)
            st.smooth_start = (pos_x, pos_y)
            st.smooth_target = (cmd1, cmd2)
            st.smooth_start_time = time.time()
            st.smooth_duration = cal.SMOOTH_MOVE_DURATION_SEC
            st.pending_test_move = None
            return "none"
        if key8 == cal.KEY_ESC:
            st.pending_test_move = None
            return "none"

    elif getattr(st, "test_awaiting_confirm", False) and st.mode == "test":
        if key8 in (cal.KEY_ENTER, cal.KEY_ENTER_LF):
            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            st.last_test_actual_arm = (getattr(arm, "pos_x", 0.0), getattr(arm, "pos_y", 0.0))
            st.last_test_crosshair_pixel = (st.crosshair_x, st.crosshair_y)
            st.test_awaiting_confirm = False
            return "none"

    if key8 == cal.KEY_RECALIB and st.mode == "test":
        if cal._recalibrate_from_log(st):
            return "recalibrated"
        return "none"

    if key8 == cal.KEY_HOME and st.mode == "test":
        if getattr(st, "test_move_log_pending", False) and st.last_test_pixel is not None and st.last_test_arm is not None:
            actual_arm = getattr(st, "last_test_actual_arm", None) or st.last_test_arm
            crosshair_px = getattr(st, "last_test_crosshair_pixel", None)
            err_px = err_py = err_dist = None
            if crosshair_px is not None and st.last_test_pixel is not None:
                px_click, py_click = st.last_test_pixel[0], st.last_test_pixel[1]
                err_px = px_click - crosshair_px[0]
                err_py = py_click - crosshair_px[1]
                err_dist = math.sqrt(err_px * err_px + err_py * err_py)
            cal._append_test_move_log(
                st.last_test_pixel,
                st.last_test_arm,
                actual_arm,
                cal.TEST_MOVE_LOG_PXDEG_PATH,
                output_w=st.output_w or cal.REF_OUTPUT_W,
                output_h=st.output_h or cal.REF_OUTPUT_H,
                crosshair_pixel=crosshair_px,
                error_pixel_x=err_px,
                error_pixel_y=err_py,
                error_pixel_dist=err_dist,
            )
            st.test_move_log_pending = False
            print(f"[PXDEG] Logged point → {cal.TEST_MOVE_LOG_PXDEG_PATH.name}")
        try:
            arm.go_home(blocking=False)
        except Exception:
            pass
        st.test_awaiting_home = True
        st.pending_test_move = None
        return "none"

    if key8 == cal.KEY_SAVE and st.mode == "test":
        if _save_current_px_deg(cal):
            return "saved"
        return "none"

    # Fine-tune only after smooth move completes (test_awaiting_confirm)
    if getattr(st, "test_awaiting_confirm", False) and st.mode == "test":
        if key in cal.ARROW_KEYS_LEFT:
            try:
                arm.move_relative(-cal.ARROW_STEP_DEG, 0.0, blocking=False)
            except Exception:
                pass
        elif key in cal.ARROW_KEYS_RIGHT:
            try:
                arm.move_relative(cal.ARROW_STEP_DEG, 0.0, blocking=False)
            except Exception:
                pass
        elif key in cal.ARROW_KEYS_UP:
            try:
                arm.move_relative(0.0, cal.ARROW_STEP_DEG, blocking=False)
            except Exception:
                pass
        elif key in cal.ARROW_KEYS_DOWN:
            try:
                arm.move_relative(0.0, -cal.ARROW_STEP_DEG, blocking=False)
            except Exception:
                pass

    return "none"


def is_awaiting_finetune() -> bool:
    """True when arm finished smooth move and user may fine-tune before confirm."""
    if not _active:
        return False
    st = _module()._state
    return bool(getattr(st, "test_awaiting_confirm", False) and st.mode == "test")


def apply_joystick_finetune(js_state: Any, dt: float, arm: Any, joystick_mapper: Any) -> None:
    """
    Drive arm with joystick during test_awaiting_confirm (same phase as arrow keys).
    Rate-capped for precision; axis-3 speed scale still applies via mapper.
    """
    if not is_awaiting_finetune() or arm is None or joystick_mapper is None or js_state is None:
        return
    if dt <= 0.0:
        return
    try:
        dx, dy = joystick_mapper.compute_delta(js_state, dt)
        max_step = PXDEG_JOY_FINETUNE_MAX_RATE_DEG * dt
        if max_step > 0.0:
            dx = max(-max_step, min(max_step, dx))
            dy = max(-max_step, min(max_step, dy))
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return
        arm.move_relative(dx, dy, blocking=False)
    except Exception:
        pass


def confirm_via_joystick(arm: Any) -> str:
    """Treat joystick button 0 as ENTER while embedded."""
    if not _active:
        return "none"
    return handle_key(_module().KEY_ENTER, arm)


def hud_label() -> Tuple[str, Tuple[int, int, int]]:
    cal = _module()
    st = cal._state
    if st.px_per_deg_x is not None and st.px_per_deg_y is not None:
        return (
            f"PXDEG:ON({st.px_per_deg_x:.0f},{st.px_per_deg_y:.0f})",
            (0, 220, 255),
        )
    return "PXDEG:ON", (0, 220, 255)


def get_px_per_deg() -> Tuple[Optional[float], Optional[float]]:
    cal = _module()
    st = cal._state
    return st.px_per_deg_x, st.px_per_deg_y
