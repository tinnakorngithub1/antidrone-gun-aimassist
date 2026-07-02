"""
Cam4 Mouse Distance HUD
-----------------------
Same behavior as Test + L in cam4_arm_mouse_grid_calibrator: camera, crosshair at center,
orange dot = mouse position, arm follows mouse. HUD shows only: Distance from crosshair: X.X px.
Press ESC to quit.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from gun_aim_assist import build_camera_from_config, ShooterConfig
except ImportError:
    build_camera_from_config = None
    ShooterConfig = None

try:
    from cam4_arm_controller import Cam4ArmController
except ImportError:
    Cam4ArmController = None

try:
    import config
except ImportError:
    config = None

# Reuse calibrator's variable-step arm drive (L mode)
try:
    from cam4_arm_mouse_grid_calibrator import (
        _variable_step_toward_target,
        _load_pixel_per_degree_json,
        PX_PER_DEG_X_DEFAULT,
        PX_PER_DEG_Y_DEFAULT,
        SWAP_PAN_TILT,
        CAMERA_ON_ARM,
        CONTINUOUS_THROTTLE_SEC,
        CONTINUOUS_P_ZONE_FAR_DEG,
        CONTINUOUS_P_VERY_FAR_DEG,
        CONTINUOUS_P_ZONE_NEAR_DEG,
        CONTINUOUS_P_THROTTLE_FAR_SEC,
        CONTINUOUS_P_THROTTLE_MID_SEC,
        CONTINUOUS_P_THROTTLE_NEAR_SEC,
    )
except ImportError:
    _variable_step_toward_target = None
    _load_pixel_per_degree_json = None
    PX_PER_DEG_X_DEFAULT = 50.0
    PX_PER_DEG_Y_DEFAULT = -50.0
    SWAP_PAN_TILT = True
    CAMERA_ON_ARM = True
    CONTINUOUS_THROTTLE_SEC = 0.01
    CONTINUOUS_P_ZONE_FAR_DEG = 5.0
    CONTINUOUS_P_VERY_FAR_DEG = 10.0
    CONTINUOUS_P_ZONE_NEAR_DEG = 0.5
    CONTINUOUS_P_THROTTLE_FAR_SEC = 0.02
    CONTINUOUS_P_THROTTLE_MID_SEC = 0.02
    CONTINUOUS_P_THROTTLE_NEAR_SEC = 0.02

# Same ref size as calibrator (4K)
REF_OUTPUT_W = 3840
REF_OUTPUT_H = 2160

HUD_TEXT_SCALE = max(1.0, REF_OUTPUT_W / 1920.0)
FONT_HUD = 0.65 * HUD_TEXT_SCALE
THICKNESS_THIN = max(1, int(1 * HUD_TEXT_SCALE))

COLOR_CROSSHAIR = (0, 255, 0)  # BGR green
RETICLE_RADIUS_PX = (12, 20, 32, 52)  # inner -> outer
COLOR_HUD_RED = (0, 0, 255)  # BGR red for HUD
KEY_ESC = 27

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
PX_DEG_JSON_PATH = CALIBRATION_DIR / "cam4_pixel_per_degree.json"


class _ArmDriveState:
    """Minimal state for _variable_step_toward_target (L mode)."""
    continuous_target_pan_prev = None
    continuous_target_tilt_prev = None
    continuous_target_time_prev = None
    continuous_velocity_mm_s = 0.0


def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
    state = param
    if event != cv2.EVENT_MOUSEMOVE:
        return
    dw = state.get("display_w", 0)
    dh = state.get("display_h", 0)
    ow = state.get("output_w", REF_OUTPUT_W)
    oh = state.get("output_h", REF_OUTPUT_H)
    if dw <= 0 or dh <= 0:
        state["mouse_px"] = max(0.0, min(ow - 1.0, float(x)))
        state["mouse_py"] = max(0.0, min(oh - 1.0, float(y)))
        return
    px_out = x * ow / dw
    py_out = y * oh / dh
    state["mouse_px"] = max(0.0, min(ow - 1.0, px_out))
    state["mouse_py"] = max(0.0, min(oh - 1.0, py_out))


def _draw_crosshair(frame: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
    r0 = RETICLE_RADIUS_PX[0]
    x_left = max(0, cx - r0)
    x_right = min(w, cx + r0)
    y_top = max(0, cy - r0)
    y_bottom = min(h, cy + r0)
    cv2.line(frame, (0, cy), (x_left, cy), COLOR_CROSSHAIR, 2)
    cv2.line(frame, (x_right, cy), (w, cy), COLOR_CROSSHAIR, 2)
    cv2.line(frame, (cx, 0), (cx, y_top), COLOR_CROSSHAIR, 2)
    cv2.line(frame, (cx, y_bottom), (cx, h), COLOR_CROSSHAIR, 2)
    for r in RETICLE_RADIUS_PX:
        cv2.circle(frame, (cx, cy), r, COLOR_CROSSHAIR, 2)
    cv2.circle(frame, (cx, cy), 3, COLOR_CROSSHAIR, -1)


def _load_px_per_deg():
    """Load pixel_per_degree from JSON; return (px_per_deg_x, px_per_deg_y) or (None, None)."""
    if _load_pixel_per_degree_json is not None:
        data = _load_pixel_per_degree_json()
    else:
        data = None
        if PX_DEG_JSON_PATH.is_file():
            try:
                with open(PX_DEG_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
    if not data or not isinstance(data, dict):
        return PX_PER_DEG_X_DEFAULT, PX_PER_DEG_Y_DEFAULT
    ppdx = data.get("pixel_per_degree_x")
    ppdy = data.get("pixel_per_degree_y")
    try:
        x = float(ppdx) if ppdx is not None else PX_PER_DEG_X_DEFAULT
        y = float(ppdy) if ppdy is not None else PX_PER_DEG_Y_DEFAULT
        return x, y
    except (TypeError, ValueError):
        return PX_PER_DEG_X_DEFAULT, PX_PER_DEG_Y_DEFAULT


def main() -> None:
    if build_camera_from_config is None or ShooterConfig is None:
        print("gun_aim_assist not available (build_camera_from_config, ShooterConfig).")
        return
    if Cam4ArmController is None:
        print("cam4_arm_controller not available.")
        return

    camera_name = "cam4"
    cam = build_camera_from_config(camera_name)
    cam.start()
    time.sleep(0.3)

    arm = Cam4ArmController()
    if not arm.connect():
        print("Arm connect failed.")
        cam.release()
        return

    px_per_deg_x, px_per_deg_y = _load_px_per_deg()
    margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
    xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
    yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
    x_lim = getattr(arm, "_effective_x_limits", None) or (xr[0] + margin, xr[1] - margin)
    y_lim = getattr(arm, "_effective_y_limits", None) or (yr[0] + margin, yr[1] - margin)
    x_lo, x_hi = x_lim[0], x_lim[1]
    y_lo, y_hi = y_lim[0], y_lim[1]

    state = {
        "output_w": REF_OUTPUT_W,
        "output_h": REF_OUTPUT_H,
        "crosshair_x": REF_OUTPUT_W / 2.0,
        "crosshair_y": REF_OUTPUT_H / 2.0,
        "mouse_px": None,
        "mouse_py": None,
        "display_w": 0,
        "display_h": 0,
    }
    arm_drive_state = _ArmDriveState()
    arm_drive_state.continuous_target_time_prev = time.time() - 0.05
    last_continuous_arm_move_time = 0.0

    win_name = "Cam4 Mouse Distance HUD"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, _on_mouse, param=state)

    try:
        from gun_aim_assist import get_screen_size
        screen_w, screen_h = get_screen_size()
    except Exception:
        screen_w, screen_h = 1920, 1080

    try:
        while True:
            now = time.time()
            active, frame, _ = cam.read()
            if not active or frame is None:
                time.sleep(0.02)
                continue
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.shape[1] != REF_OUTPUT_W or frame.shape[0] != REF_OUTPUT_H:
                frame = cv2.resize(
                    frame, (REF_OUTPUT_W, REF_OUTPUT_H), interpolation=cv2.INTER_LINEAR
                )
            h, w = frame.shape[:2]
            state["output_w"] = w
            state["output_h"] = h
            state["crosshair_x"] = w / 2.0
            state["crosshair_y"] = h / 2.0

            cx = int(state["crosshair_x"])
            cy = int(state["crosshair_y"])
            _draw_crosshair(frame, cx, cy, w, h)

            mouse_px = state.get("mouse_px")
            mouse_py = state.get("mouse_py")
            if mouse_px is not None and mouse_py is not None:
                pt1 = (cx, cy)
                pt2 = (int(mouse_px), int(mouse_py))
                if pt1 != pt2 and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.arrowedLine(frame, pt1, pt2, (0, 255, 255), 2, tipLength=0.15)

            # L mode: compute target pan/tilt from mouse and drive arm (variable step)
            if (
                _variable_step_toward_target is not None
                and mouse_px is not None
                and mouse_py is not None
                and px_per_deg_x is not None
                and px_per_deg_y is not None
            ):
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
                pan_cur = getattr(arm, "pos_x", 0.0)
                tilt_cur = getattr(arm, "pos_y", 0.0)
                ch_x = state["crosshair_x"]
                ch_y = state["crosshair_y"]
                dx_px = mouse_px - ch_x
                dy_px = mouse_py - ch_y
                delta_pan = dx_px / px_per_deg_x
                delta_tilt = dy_px / px_per_deg_y
                if CAMERA_ON_ARM:
                    target_pan = pan_cur + delta_pan
                    target_tilt = tilt_cur + delta_tilt
                else:
                    target_pan = delta_pan
                    target_tilt = delta_tilt
                if SWAP_PAN_TILT:
                    pan_tgt = float(np.clip(target_pan, y_lo, y_hi))
                    tilt_tgt = float(np.clip(target_tilt, x_lo, x_hi))
                else:
                    pan_tgt = float(np.clip(target_pan, x_lo, x_hi))
                    tilt_tgt = float(np.clip(target_tilt, y_lo, y_hi))

                current_pan = getattr(arm, "pos_x", 0.0)
                current_tilt = getattr(arm, "pos_y", 0.0)
                error_deg = math.hypot(pan_tgt - current_pan, tilt_tgt - current_tilt)
                if error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
                    throttle_sec = CONTINUOUS_THROTTLE_SEC
                elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
                    throttle_sec = CONTINUOUS_P_THROTTLE_MID_SEC
                else:
                    throttle_sec = CONTINUOUS_P_THROTTLE_NEAR_SEC
                min_interval = getattr(arm, "_min_move_interval_sec", 0.02)
                if error_deg > CONTINUOUS_P_VERY_FAR_DEG:
                    throttle_sec = max(throttle_sec, min(min_interval, 0.01))
                else:
                    throttle_sec = max(throttle_sec, min_interval)

                last_continuous_arm_move_time, _, _ = _variable_step_toward_target(
                    arm, pan_tgt, tilt_tgt, last_continuous_arm_move_time, throttle_sec, arm_drive_state
                )

            if mouse_px is not None and mouse_py is not None:
                dist_px = math.sqrt(
                    (mouse_px - state["crosshair_x"]) ** 2
                    + (mouse_py - state["crosshair_y"]) ** 2
                )
                hud_txt = f"Distance from crosshair: {dist_px:.1f} px"
            else:
                hud_txt = "Distance from crosshair: — px"
            y_hud = h - int(30 * HUD_TEXT_SCALE)
            cv2.putText(
                frame, hud_txt, (10, y_hud),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
            )

            scale = min(screen_w / w, screen_h / h, 1.0)
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            state["display_w"] = scaled_w
            state["display_h"] = scaled_h
            if (scaled_w, scaled_h) != (w, h):
                display_frame = cv2.resize(
                    frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                display_frame = frame
            cv2.imshow(win_name, display_frame)
            try:
                cv2.resizeWindow(win_name, scaled_w, scaled_h)
            except Exception:
                pass
            key = cv2.waitKey(1)
            key8 = key & 0xFF if key >= 0 else -1
            if key8 == KEY_ESC:
                break
    finally:
        cv2.destroyAllWindows()
        cam.release()
        if getattr(config, "CAM4_ARM_RETURN_TO_REF_ON_DISCONNECT", True):
            try:
                arm.go_home(blocking=True)
            except Exception:
                pass
        if hasattr(arm, "disconnect"):
            arm.disconnect()


if __name__ == "__main__":
    main()
