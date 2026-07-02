"""
Cam4 Mouse Click Vector HUD
cam4_mouse_click_vector_hud.py
---------------------------
Based on cam4_mouse_distance_hud: camera, crosshair at center.
Arm does NOT follow mouse. Click to set target → arm moves toward clicked point.
- White vector: crosshair → target (direction to go); shrinks as arm angle approaches target.
- Yellow vector: same direction, "ช้าไปทางไหน"; max 250 px at start, exponential decay near target (smooth stop).
Press ESC to quit.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Optional, Tuple

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

REF_OUTPUT_W = 3840
REF_OUTPUT_H = 2160

# Yellow vector: max length at start; exponential decay as we get close (smooth stop)
YELLOW_MAX_PX = 250
YELLOW_EXP_SCALE_PX = 80.0  # larger = yellow stays longer when near target

HUD_TEXT_SCALE = max(1.0, REF_OUTPUT_W / 1920.0)
FONT_HUD = 0.65 * HUD_TEXT_SCALE
THICKNESS_THIN = max(1, int(1 * HUD_TEXT_SCALE))

COLOR_CROSSHAIR = (0, 255, 0)
RETICLE_RADIUS_PX = (12, 20, 32, 52)
COLOR_HUD_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
KEY_ESC = 27

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
PX_DEG_JSON_PATH = CALIBRATION_DIR / "cam4_pixel_per_degree.json"


class _ArmDriveState:
    continuous_target_pan_prev = None
    continuous_target_tilt_prev = None
    continuous_target_time_prev = None
    continuous_velocity_mm_s = 0.0


def _px_display_to_output(x: int, y: int, state: dict) -> Tuple[float, float]:
    dw = state.get("display_w", 0)
    dh = state.get("display_h", 0)
    ow = state.get("output_w", REF_OUTPUT_W)
    oh = state.get("output_h", REF_OUTPUT_H)
    if dw <= 0 or dh <= 0:
        return max(0.0, min(ow - 1.0, float(x))), max(0.0, min(oh - 1.0, float(y)))
    px = x * ow / dw
    py = y * oh / dh
    return max(0.0, min(ow - 1.0, px)), max(0.0, min(oh - 1.0, py))


def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
    state = param
    px_out, py_out = _px_display_to_output(x, y, state)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["target_click_px"] = px_out
        state["target_click_py"] = py_out
        state["need_update_target_from_click"] = True
        return
    if event == cv2.EVENT_MOUSEMOVE:
        state["mouse_px"] = px_out
        state["mouse_py"] = py_out


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
        return (
            float(ppdx) if ppdx is not None else PX_PER_DEG_X_DEFAULT,
            float(ppdy) if ppdy is not None else PX_PER_DEG_Y_DEFAULT,
        )
    except (TypeError, ValueError):
        return PX_PER_DEG_X_DEFAULT, PX_PER_DEG_Y_DEFAULT


def main() -> None:
    if build_camera_from_config is None or ShooterConfig is None:
        print("gun_aim_assist not available (build_camera_from_config, ShooterConfig).")
        return
    if Cam4ArmController is None:
        print("cam4_arm_controller not available.")
        return

    cam = build_camera_from_config("cam4")
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
        "target_click_px": None,
        "target_click_py": None,
        "need_update_target_from_click": False,
        "target_pan": None,
        "target_tilt": None,
        "display_w": 0,
        "display_h": 0,
    }
    arm_drive_state = _ArmDriveState()
    arm_drive_state.continuous_target_time_prev = time.time() - 0.05
    last_continuous_arm_move_time = 0.0

    win_name = "Cam4 Mouse Click Vector HUD"
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

            ch_x = state["crosshair_x"]
            ch_y = state["crosshair_y"]
            cx = int(ch_x)
            cy = int(ch_y)
            _draw_crosshair(frame, cx, cy, w, h)

            target_click_px = state.get("target_click_px")
            target_click_py = state.get("target_click_py")

            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            pan_cur = getattr(arm, "pos_x", 0.0)
            tilt_cur = getattr(arm, "pos_y", 0.0)

            # Compute target pan/tilt from click once per click (so target is fixed until next click)
            if state.get("need_update_target_from_click") and target_click_px is not None and target_click_py is not None and px_per_deg_x and px_per_deg_y:
                dx_px = target_click_px - ch_x
                dy_px = target_click_py - ch_y
                delta_pan = dx_px / px_per_deg_x
                delta_tilt = dy_px / px_per_deg_y
                if CAMERA_ON_ARM:
                    tpan = pan_cur + delta_pan
                    ttilt = tilt_cur + delta_tilt
                else:
                    tpan = delta_pan
                    ttilt = delta_tilt
                if SWAP_PAN_TILT:
                    state["target_pan"] = float(np.clip(tpan, y_lo, y_hi))
                    state["target_tilt"] = float(np.clip(ttilt, x_lo, x_hi))
                else:
                    state["target_pan"] = float(np.clip(tpan, x_lo, x_hi))
                    state["target_tilt"] = float(np.clip(ttilt, y_lo, y_hi))
                state["need_update_target_from_click"] = False

            target_pan = state.get("target_pan")
            target_tilt = state.get("target_tilt")

            # White vector: from crosshair to target position in image (shrinks as arm approaches)
            # wx, wy = error in px (where target would be relative to center)
            if target_pan is not None and target_tilt is not None and px_per_deg_x and px_per_deg_y:
                err_pan = target_pan - pan_cur
                err_tilt = target_tilt - tilt_cur
                wx = err_pan * px_per_deg_x
                wy = err_tilt * px_per_deg_y
                white_len = math.hypot(wx, wy)

                if white_len >= 1e-6:
                    # Draw white vector (direction to target)
                    pt_white = (int(cx + wx), int(cy + wy))
                    if 0 <= pt_white[0] < w and 0 <= pt_white[1] < h and (cx, cy) != pt_white:
                        cv2.arrowedLine(frame, (cx, cy), pt_white, COLOR_WHITE, 2, tipLength=0.15)

                    # Yellow vector: same direction, length = exponential decay, max 250
                    ux = wx / white_len
                    uy = wy / white_len
                    yellow_len = YELLOW_MAX_PX * (1.0 - math.exp(-white_len / YELLOW_EXP_SCALE_PX))
                    yellow_len = min(YELLOW_MAX_PX, yellow_len, white_len)
                    if yellow_len >= 2.0:
                        guide_px = ch_x + ux * yellow_len
                        guide_py = ch_y + uy * yellow_len
                        pt_yellow = (int(guide_px), int(guide_py))
                        if 0 <= pt_yellow[0] < w and 0 <= pt_yellow[1] < h and (cx, cy) != pt_yellow:
                            cv2.arrowedLine(frame, (cx, cy), pt_yellow, COLOR_YELLOW, 2, tipLength=0.15)

            # Arm drive: only toward (target_pan, target_tilt) from click
            if (
                _variable_step_toward_target is not None
                and target_pan is not None
                and target_tilt is not None
                and px_per_deg_x is not None
                and px_per_deg_y is not None
            ):
                pan_tgt = target_pan
                tilt_tgt = target_tilt
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

            if target_click_px is not None and target_click_py is not None:
                dist_px = math.hypot(
                    target_click_px - ch_x,
                    target_click_py - ch_y
                )
                hud_txt = f"Click target: {dist_px:.1f} px  [Click]=new target  [ESC]=quit"
            else:
                hud_txt = "Click to set target  [ESC]=quit"
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

