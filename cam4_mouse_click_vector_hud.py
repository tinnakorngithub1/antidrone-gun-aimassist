"""
Cam4 Mouse Click Vector HUD
cam4_mouse_click_vector_hud.py
---------------------------
Based on cam4_mouse_distance_hud: camera, crosshair at center.
Arm does NOT follow mouse. Click to set target → arm moves toward clicked point.
- [C]=CSRT: drag bbox → arm follows bbox center (for retrain).
- [O]=ORB: drag bbox → ORB feature matching (no homography), center from median displacement, px/deg target.
- White vector: crosshair → target; Yellow vector: gradient 0–250, smoothed.
- [G]=graph [R]=retrain [ESC]=quit.
"""

import json
import math
import time
from collections import deque
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
        CONTINUOUS_DEADZONE_DEG,
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
    CONTINUOUS_DEADZONE_DEG = 0.02

REF_OUTPUT_W = 3840
REF_OUTPUT_H = 2160

# Yellow vector: gradient 0–250 by "speed" (remaining distance); temporal smoothing = EV-like, no jerk
YELLOW_MAX_PX = 250
YELLOW_GRADIENT_SCALE_PX = 120.0  # remaining px → 0..250 (larger = ramp to 250 more gradually)
YELLOW_SMOOTH_ALPHA = 0.90  # 0.85–0.95: higher = smoother, less jerk (EV feel)

HUD_TEXT_SCALE = max(1.0, REF_OUTPUT_W / 1920.0)
FONT_HUD = 0.65 * HUD_TEXT_SCALE
THICKNESS_THIN = max(1, int(1 * HUD_TEXT_SCALE))

COLOR_CROSSHAIR = (0, 255, 0)
RETICLE_RADIUS_PX = (12, 20, 32, 52)
COLOR_HUD_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
KEY_ESC = 27
KEY_G = ord("g")
KEY_V = ord("v")
KEY_R = ord("r")
KEY_C = ord("c")
KEY_O = ord("o")

ORB_SCALE = 0.25
ORB_NFEATURES = 500
ORB_UPDATE_EVERY_N = 1
ORB_MIN_MATCHES = 4
ORB_GOOD_MATCH_MAX_DIST = 60

TELEMETRY_MAXLEN = 3000
TRACK_SCALE = 0.25
CSRT_MIN_BBOX = 4
CSRT_UPDATE_EVERY_N_FRAMES = 2
TELEMETRY_SUBSAMPLE = 2  # append every N frames when driving

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
YELLOW_PARAMS_JSON = CALIBRATION_DIR / "cam4_yellow_vector_params.json"
PX_DEG_JSON_PATH = CALIBRATION_DIR / "cam4_pixel_per_degree.json"
YELLOW_MAX_CURVE_JSON = CALIBRATION_DIR / "cam4_yellow_max_curve.json"

# Yellow cap curve (far = small yellow, close = full): error_deg >= FAR_DEG -> yellow_max = YELLOW_NEAR_FAR; <= NEAR_DEG -> full
YELLOW_MAX_FAR_DEG = 3.0
YELLOW_MAX_NEAR_DEG = 0.5
YELLOW_NEAR_FAR_PX = 100.0
# When target (bbox/ORB) moves more than this (deg), update so arm follows
TRACK_TARGET_MOVE_THRESHOLD_DEG = 0.8
# Virtual center: aim this many px along white vector toward target; ramps to 0 when within this distance (ORB)
LEAD_OFFSET_PX = 75.0
# CSRT: virtual click on white vector at most this many px from crosshair; if vector shorter, use bbox center
VIRTUAL_CLICK_CAP_PX = 100.0


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
    mode = state.get("mode", "click")
    if mode == "csrt" or mode == "orb":
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (px_out, py_out)
            state["drag_current"] = None
            return
        if event == cv2.EVENT_LBUTTONUP:
            if state.get("drag_start") is not None and state.get("drag_current") is not None:
                x1, y1 = state["drag_start"]
                x2, y2 = state["drag_current"]
                x_min = max(0, min(x1, x2))
                x_max = min(state["output_w"], max(x1, x2))
                y_min = max(0, min(y1, y2))
                y_max = min(state["output_h"], max(y1, y2))
                bw = x_max - x_min
                bh = y_max - y_min
                if bw >= CSRT_MIN_BBOX and bh >= CSRT_MIN_BBOX:
                    state["pending_bbox"] = (int(x_min), int(y_min), int(bw), int(bh))
            state["drag_start"] = None
            state["drag_current"] = None
            return
        if event == cv2.EVENT_MOUSEMOVE and state.get("drag_start") is not None:
            state["drag_current"] = (px_out, py_out)
        return
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


def _show_telemetry_plot(state: dict) -> None:
    """Show matplotlib pop-up: Error vs t, Yellow vs t, Yellow vs white_len, Velocity vs t."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tel = state.get("telemetry")
    if not tel or len(tel) == 0:
        return
    # (t, white_len_px, error_deg, desired_yellow, yellow_smoothed, velocity_mm_s)
    t0 = tel[0][0]
    t = [row[0] - t0 for row in tel]
    white_len_px = [row[1] for row in tel]
    error_deg = [row[2] for row in tel]
    desired_yellow = [row[3] for row in tel]
    yellow_smoothed = [row[4] for row in tel]
    velocity_mm_s = [row[5] for row in tel]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    axes[0].plot(t, white_len_px, color="C0", label="white_len (px)")
    axes[0].set_ylabel("white_len (px)")
    axes[0].set_title("White vector length (Error) vs Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(t, desired_yellow, color="C1", linestyle="--", alpha=0.8, label="desired_yellow")
    axes[1].plot(t, yellow_smoothed, color="C2", label="yellow_smoothed")
    axes[1].set_ylabel("Yellow (px)")
    axes[1].set_title("Yellow vs Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    axes[2].scatter(white_len_px, yellow_smoothed, s=5, alpha=0.6, color="C3", label="yellow vs white_len")
    axes[2].set_xlabel("white_len (px)")
    axes[2].set_ylabel("yellow_smoothed (px)")
    axes[2].set_title("Yellow vs White length (for retrain)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)
    axes[3].plot(t, velocity_mm_s, color="C4")
    axes[3].set_ylabel("Velocity (mm/s)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Velocity vs Time")
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _load_yellow_params() -> Tuple[Optional[float], Optional[float]]:
    """Load yellow_scale_px, yellow_smooth_alpha from JSON. Return (scale_px, smooth_alpha) or (None, None)."""
    if not YELLOW_PARAMS_JSON.is_file():
        return None, None
    try:
        with open(YELLOW_PARAMS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale = data.get("yellow_scale_px")
        alpha = data.get("yellow_smooth_alpha")
        return (float(scale) if scale is not None else None, float(alpha) if alpha is not None else None)
    except Exception:
        return None, None


def _save_yellow_params(scale_px: float, smooth_alpha: Optional[float] = None) -> None:
    """Save yellow params to JSON."""
    try:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        data = {"yellow_scale_px": scale_px}
        if smooth_alpha is not None:
            data["yellow_smooth_alpha"] = smooth_alpha
        with open(YELLOW_PARAMS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _retrain_yellow_params(state: dict) -> bool:
    """Fit YELLOW_GRADIENT_SCALE_PX (and optionally YELLOW_SMOOTH_ALPHA) from telemetry (white_len, yellow_smoothed). Return True if updated."""
    tel = state.get("telemetry")
    if not tel or len(tel) < 10:
        return False
    data = [(row[1], row[4]) for row in tel if row[1] > 1e-6 and row[4] >= 0]
    if len(data) < 5:
        return False
    white_lens = np.array([d[0] for d in data], dtype=float)
    yellows = np.array([d[1] for d in data], dtype=float)

    def pred(scale: float) -> np.ndarray:
        out = YELLOW_MAX_PX * (1.0 - np.exp(-white_lens / max(scale, 1.0)))
        out = np.minimum(out, YELLOW_MAX_PX)
        out = np.minimum(out, white_lens)
        return out

    def loss(scale: float) -> float:
        p = pred(scale)
        return float(np.sum((p - yellows) ** 2))

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(loss, bounds=(20.0, 400.0), method="bounded")
        if res.success and res.fun < 1e12:
            new_scale = float(res.x)
            state["yellow_scale_px"] = new_scale
            _save_yellow_params(new_scale, state.get("yellow_smooth_alpha"))
            return True
    except Exception:
        pass
    return False


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


def _load_yellow_max_curve() -> Optional[dict]:
    """Load yellow_max curve from JSON if present. Returns dict with bins_deg, yellow_max_px or None."""
    if not YELLOW_MAX_CURVE_JSON.is_file():
        return None
    try:
        with open(YELLOW_MAX_CURVE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "bins_deg" in data and "yellow_max_px" in data:
            return data
    except Exception:
        pass
    return None


def _yellow_max_at_error_deg(error_deg: float, curve: Optional[dict]) -> float:
    """
    Return allowed yellow max (px) at this error_deg. Far -> small, close -> YELLOW_MAX_PX.
    If curve from JSON: use bins; else use default linear interpolation.
    """
    if curve is not None:
        bins = curve.get("bins_deg")
        vals = curve.get("yellow_max_px")
        if isinstance(bins, list) and isinstance(vals, list) and len(bins) == len(vals) and len(bins) >= 2:
            for i in range(len(bins) - 1):
                if error_deg >= bins[i + 1]:
                    continue
                lo, hi = float(bins[i]), float(bins[i + 1])
                vlo, vhi = float(vals[i]), float(vals[i + 1])
                if hi > lo:
                    t = (error_deg - lo) / (hi - lo)
                    return vlo + t * (vhi - vlo)
            return float(vals[-1]) if vals else YELLOW_MAX_PX
    # Default: error_deg >= FAR -> YELLOW_NEAR_FAR_PX; <= NEAR -> YELLOW_MAX_PX; else linear
    if error_deg <= YELLOW_MAX_NEAR_DEG:
        return YELLOW_MAX_PX
    if error_deg >= YELLOW_MAX_FAR_DEG:
        return YELLOW_NEAR_FAR_PX
    t = (error_deg - YELLOW_MAX_NEAR_DEG) / (YELLOW_MAX_FAR_DEG - YELLOW_MAX_NEAR_DEG)
    return YELLOW_NEAR_FAR_PX + t * (YELLOW_MAX_PX - YELLOW_NEAR_FAR_PX)


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

    loaded_scale, loaded_alpha = _load_yellow_params()
    state = {
        "output_w": REF_OUTPUT_W,
        "output_h": REF_OUTPUT_H,
        "crosshair_x": REF_OUTPUT_W / 2.0,
        "crosshair_y": REF_OUTPUT_H / 2.0,
        "mode": "click",
        "mouse_px": None,
        "mouse_py": None,
        "target_click_px": None,
        "target_click_py": None,
        "need_update_target_from_click": False,
        "target_pan": None,
        "target_tilt": None,
        "yellow_smoothed_px": None,
        "yellow_scale_px": loaded_scale,
        "yellow_smooth_alpha": loaded_alpha,
        "telemetry": deque(maxlen=TELEMETRY_MAXLEN),
        "_telemetry_frame_count": 0,
        "display_w": 0,
        "display_h": 0,
        "drag_start": None,
        "drag_current": None,
        "pending_bbox": None,
        "csrt_tracker": None,
        "csrt_initialized": False,
        "csrt_bbox": None,
        "csrt_smooth_px": None,
        "csrt_smooth_py": None,
        "csrt_lost": False,
        "_track_frame_count": 0,
        "orb_initialized": False,
        "orb_kp": None,
        "orb_des": None,
        "orb_bbox": None,
        "orb_template_wh_small": None,
        "orb_lost": False,
        "orb_center_px": None,
        "orb_center_py": None,
        "_orb_frame_count": 0,
        "yellow_max_curve": _load_yellow_max_curve(),
        "yellow_capped": None,
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

            mode = state.get("mode", "click")
            target_click_px = state.get("target_click_px")
            target_click_py = state.get("target_click_py")

            if mode == "csrt" or mode == "orb":
                if state.get("drag_start") is not None and state.get("drag_current") is not None:
                    x1, y1 = state["drag_start"]
                    x2, y2 = state["drag_current"]
                    ix1 = max(0, min(int(x1), w - 1))
                    iy1 = max(0, min(int(y1), h - 1))
                    ix2 = max(0, min(int(x2), w - 1))
                    iy2 = max(0, min(int(y2), h - 1))
                    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            pan_cur = getattr(arm, "pos_x", 0.0)
            tilt_cur = getattr(arm, "pos_y", 0.0)

            # ORB init: when mode orb and pending_bbox set, crop and extract template keypoints
            if mode == "orb" and state.get("pending_bbox") is not None:
                xb, yb, wb, hb = state["pending_bbox"]
                xb = max(0, min(int(xb), w - 1))
                yb = max(0, min(int(yb), h - 1))
                wb = max(CSRT_MIN_BBOX, min(int(wb), w - xb))
                hb = max(CSRT_MIN_BBOX, min(int(hb), h - yb))
                crop = frame[yb : yb + hb, xb : xb + wb]
                if crop.size > 0:
                    crop_small_w = max(1, int(wb * ORB_SCALE))
                    crop_small_h = max(1, int(hb * ORB_SCALE))
                    crop_small = cv2.resize(crop, (crop_small_w, crop_small_h), interpolation=cv2.INTER_LINEAR)
                    gray_small = cv2.cvtColor(crop_small, cv2.COLOR_BGR2GRAY)
                    orb_det = cv2.ORB_create(nfeatures=ORB_NFEATURES)
                    kp1, des1 = orb_det.detectAndCompute(gray_small, None)
                    if des1 is not None and len(kp1) >= ORB_MIN_MATCHES:
                        state["orb_kp"] = kp1
                        state["orb_des"] = des1
                        state["orb_bbox"] = (xb, yb, wb, hb)
                        state["orb_template_wh_small"] = (crop_small_w, crop_small_h)
                        state["orb_initialized"] = True
                        state["orb_lost"] = False
                        state["orb_center_px"] = float(xb + wb * 0.5)
                        state["orb_center_py"] = float(yb + hb * 0.5)
                        state["_orb_frame_count"] = 0
                    else:
                        state["orb_initialized"] = False
                        state["orb_lost"] = True
                        state["orb_center_px"] = None
                        state["orb_center_py"] = None
                state["pending_bbox"] = None

            # ORB update: every N frames match and get center from median displacement (no homography)
            if mode == "orb" and state.get("orb_initialized"):
                orb_count = state.get("_orb_frame_count", 0)
                if orb_count % ORB_UPDATE_EVERY_N == 0:
                    xb, yb, wb, hb = state["orb_bbox"]
                    kp1_list = state["orb_kp"]
                    des1 = state["orb_des"]
                    small_w = max(1, int(w * ORB_SCALE))
                    small_h = max(1, int(h * ORB_SCALE))
                    frame_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    gray_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    orb_det = cv2.ORB_create(nfeatures=ORB_NFEATURES)
                    kp2, des2 = orb_det.detectAndCompute(gray_frame, None)
                    if des2 is not None and len(kp2) >= ORB_MIN_MATCHES:
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = matcher.match(des1, des2)
                        good = [m for m in matches if m.distance < ORB_GOOD_MATCH_MAX_DIST]
                        if len(good) >= ORB_MIN_MATCHES:
                            inv_scale = 1.0 / ORB_SCALE
                            dxs, dys = [], []
                            for m in good:
                                pt1 = kp1_list[m.queryIdx].pt
                                pt2 = kp2[m.trainIdx].pt
                                src_x = xb + pt1[0] * inv_scale
                                src_y = yb + pt1[1] * inv_scale
                                dst_x = pt2[0] * inv_scale
                                dst_y = pt2[1] * inv_scale
                                dxs.append(dst_x - src_x)
                                dys.append(dst_y - src_y)
                            median_dx = float(np.median(dxs))
                            median_dy = float(np.median(dys))
                            state["orb_center_px"] = (xb + wb * 0.5) + median_dx
                            state["orb_center_py"] = (yb + hb * 0.5) + median_dy
                            state["orb_lost"] = False
                        else:
                            state["orb_lost"] = True
                            state["orb_center_px"] = None
                            state["orb_center_py"] = None
                    else:
                        state["orb_lost"] = True
                        state["orb_center_px"] = None
                        state["orb_center_py"] = None
                state["_orb_frame_count"] = orb_count + 1

            # Compute target pan/tilt: click mode = once per click; csrt mode = from bbox center every frame
            if mode == "click":
                if state.get("need_update_target_from_click") and target_click_px is not None and target_click_py is not None and px_per_deg_x and px_per_deg_y:
                    had_target = state.get("target_pan") is not None
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
                    if not had_target:
                        state["yellow_smoothed_px"] = 0.0  # ramp from 0→250 only on first target from idle
                    state["telemetry"].clear()
                    state["_telemetry_frame_count"] = 0

            if mode == "csrt":
                if state.get("pending_bbox") is not None:
                    if state.get("csrt_initialized"):
                        state["csrt_tracker"] = None
                        state["csrt_initialized"] = False
                        state["csrt_bbox"] = None
                        state["csrt_smooth_px"] = None
                        state["csrt_smooth_py"] = None
                        state["csrt_lost"] = False
                        state["_track_frame_count"] = 0
                    xb, yb, wb, hb = state["pending_bbox"]
                    xb = max(0, min(int(xb), w - 1))
                    yb = max(0, min(int(yb), h - 1))
                    wb = max(CSRT_MIN_BBOX, min(int(wb), w - xb))
                    hb = max(CSRT_MIN_BBOX, min(int(hb), h - yb))
                    tracker = None
                    try:
                        tracker = cv2.legacy.TrackerCSRT_create()
                    except Exception:
                        try:
                            tracker = cv2.TrackerCSRT_create()
                        except Exception:
                            pass
                    if tracker is not None:
                        small_w = max(1, int(w * TRACK_SCALE))
                        small_h = max(1, int(h * TRACK_SCALE))
                        frame_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                        xb_s = int(xb * TRACK_SCALE)
                        yb_s = int(yb * TRACK_SCALE)
                        wb_s = max(1, int(wb * TRACK_SCALE))
                        hb_s = max(1, int(hb * TRACK_SCALE))
                        try:
                            ok = tracker.init(frame_small, (xb_s, yb_s, wb_s, hb_s))
                            if ok:
                                state["csrt_tracker"] = tracker
                                state["csrt_initialized"] = True
                                state["csrt_bbox"] = (xb, yb, wb, hb)
                                state["csrt_smooth_px"] = xb + wb * 0.5
                                state["csrt_smooth_py"] = yb + hb * 0.5
                                state["csrt_lost"] = False
                                state["_track_frame_count"] = 0
                        except Exception:
                            pass
                    state["pending_bbox"] = None

                if state.get("csrt_initialized") and state.get("csrt_tracker") is not None:
                    count = state.get("_track_frame_count", 0)
                    if count % CSRT_UPDATE_EVERY_N_FRAMES == 0:
                        tr = state["csrt_tracker"]
                        small_w = max(1, int(w * TRACK_SCALE))
                        small_h = max(1, int(h * TRACK_SCALE))
                        frame_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                        try:
                            success, bbox_s = tr.update(frame_small)
                            if success and bbox_s is not None:
                                xb_s, yb_s, wb_s, hb_s = bbox_s
                                inv = 1.0 / TRACK_SCALE
                                xb = int(xb_s * inv)
                                yb = int(yb_s * inv)
                                wb = max(1, int(wb_s * inv))
                                hb = max(1, int(hb_s * inv))
                                state["csrt_smooth_px"] = float(xb + wb * 0.5)
                                state["csrt_smooth_py"] = float(yb + hb * 0.5)
                                state["csrt_bbox"] = (xb, yb, wb, hb)
                                state["csrt_lost"] = False
                            else:
                                state["csrt_bbox"] = None
                                state["csrt_lost"] = True
                                state["csrt_tracker"] = None
                                state["csrt_initialized"] = False
                                state["csrt_smooth_px"] = None
                                state["csrt_smooth_py"] = None
                        except Exception:
                            state["csrt_lost"] = True
                            state["csrt_tracker"] = None
                            state["csrt_initialized"] = False
                            state["csrt_smooth_px"] = None
                            state["csrt_smooth_py"] = None
                    state["_track_frame_count"] = count + 1

                if state.get("csrt_initialized") and not state.get("csrt_lost") and state.get("csrt_smooth_px") is not None and state.get("csrt_smooth_py") is not None and px_per_deg_x and px_per_deg_y:
                    bx = state["csrt_smooth_px"]
                    by = state["csrt_smooth_py"]
                    dx_px = bx - ch_x
                    dy_px = by - ch_y
                    white_len = math.hypot(dx_px, dy_px)
                    if white_len >= 1e-6:
                        cap_len = min(white_len, VIRTUAL_CLICK_CAP_PX)
                        ux = dx_px / white_len
                        uy = dy_px / white_len
                        effective_dx = cap_len * ux
                        effective_dy = cap_len * uy
                        state["_virtual_center_x"] = ch_x + effective_dx
                        state["_virtual_center_y"] = ch_y + effective_dy
                    else:
                        effective_dx = dx_px
                        effective_dy = dy_px
                        state["_virtual_center_x"] = ch_x
                        state["_virtual_center_y"] = ch_y
                    delta_pan = effective_dx / px_per_deg_x
                    delta_tilt = effective_dy / px_per_deg_y
                    if CAMERA_ON_ARM:
                        tpan = pan_cur + delta_pan
                        ttilt = tilt_cur + delta_tilt
                    else:
                        tpan = delta_pan
                        ttilt = delta_tilt
                    tpan_clip = float(np.clip(tpan, y_lo if SWAP_PAN_TILT else x_lo, y_hi if SWAP_PAN_TILT else x_hi))
                    ttilt_clip = float(np.clip(ttilt, x_lo if SWAP_PAN_TILT else y_lo, x_hi if SWAP_PAN_TILT else y_hi))
                    cur_tpan = state.get("target_pan")
                    cur_ttilt = state.get("target_tilt")
                    if cur_tpan is None or cur_ttilt is None:
                        state["target_pan"] = tpan_clip
                        state["target_tilt"] = ttilt_clip
                    else:
                        error_deg = math.hypot(cur_tpan - pan_cur, cur_ttilt - tilt_cur)
                        target_moved_deg = math.hypot(tpan_clip - cur_tpan, ttilt_clip - cur_ttilt)
                        if error_deg <= CONTINUOUS_P_ZONE_NEAR_DEG or target_moved_deg >= TRACK_TARGET_MOVE_THRESHOLD_DEG:
                            state["target_pan"] = tpan_clip
                            state["target_tilt"] = ttilt_clip
                else:
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    state["_virtual_center_x"] = None
                    state["_virtual_center_y"] = None

            if mode == "orb":
                if (
                    state.get("orb_initialized")
                    and not state.get("orb_lost")
                    and state.get("orb_center_px") is not None
                    and state.get("orb_center_py") is not None
                    and px_per_deg_x
                    and px_per_deg_y
                ):
                    ox = state["orb_center_px"]
                    oy = state["orb_center_py"]
                    dx_px = ox - ch_x
                    dy_px = oy - ch_y
                    white_len = math.hypot(dx_px, dy_px)
                    if white_len >= 1e-6:
                        lead_offset_px = min(LEAD_OFFSET_PX, white_len)
                        ux = dx_px / white_len
                        uy = dy_px / white_len
                        effective_dx = dx_px - lead_offset_px * ux
                        effective_dy = dy_px - lead_offset_px * uy
                        state["_virtual_center_x"] = ch_x + lead_offset_px * ux
                        state["_virtual_center_y"] = ch_y + lead_offset_px * uy
                    else:
                        effective_dx = dx_px
                        effective_dy = dy_px
                        state["_virtual_center_x"] = ch_x
                        state["_virtual_center_y"] = ch_y
                    delta_pan = effective_dx / px_per_deg_x
                    delta_tilt = effective_dy / px_per_deg_y
                    if CAMERA_ON_ARM:
                        tpan = pan_cur + delta_pan
                        ttilt = tilt_cur + delta_tilt
                    else:
                        tpan = delta_pan
                        ttilt = delta_tilt
                    tpan_clip = float(np.clip(tpan, y_lo if SWAP_PAN_TILT else x_lo, y_hi if SWAP_PAN_TILT else x_hi))
                    ttilt_clip = float(np.clip(ttilt, x_lo if SWAP_PAN_TILT else y_lo, x_hi if SWAP_PAN_TILT else y_hi))
                    cur_tpan = state.get("target_pan")
                    cur_ttilt = state.get("target_tilt")
                    if cur_tpan is None or cur_ttilt is None:
                        state["target_pan"] = tpan_clip
                        state["target_tilt"] = ttilt_clip
                    else:
                        error_deg = math.hypot(cur_tpan - pan_cur, cur_ttilt - tilt_cur)
                        target_moved_deg = math.hypot(tpan_clip - cur_tpan, ttilt_clip - cur_ttilt)
                        if error_deg <= CONTINUOUS_P_ZONE_NEAR_DEG or target_moved_deg >= TRACK_TARGET_MOVE_THRESHOLD_DEG:
                            state["target_pan"] = tpan_clip
                            state["target_tilt"] = ttilt_clip
                else:
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    state["_virtual_center_x"] = None
                    state["_virtual_center_y"] = None

            target_pan = state.get("target_pan")
            target_tilt = state.get("target_tilt")

            if mode == "csrt" and state.get("csrt_bbox") is not None and not state.get("csrt_lost"):
                xb, yb, wb, hb = state["csrt_bbox"]
                ix0 = max(0, min(int(xb), w - 1))
                iy0 = max(0, min(int(yb), h - 1))
                ix1 = max(0, min(int(xb + wb), w - 1))
                iy1 = max(0, min(int(yb + hb), h - 1))
                cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (0, 255, 0), 2)
                bx = state.get("csrt_smooth_px")
                by = state.get("csrt_smooth_py")
                if bx is not None and by is not None:
                    ix_red = max(0, min(int(bx), w - 1))
                    iy_red = max(0, min(int(by), h - 1))
                    cv2.circle(frame, (ix_red, iy_red), 10, (0, 0, 255), 3)
                    cv2.circle(frame, (ix_red, iy_red), 4, (0, 0, 255), -1)
                vcx = state.get("_virtual_center_x")
                vcy = state.get("_virtual_center_y")
                if vcx is not None and vcy is not None:
                    ivx = max(0, min(int(round(vcx)), w - 1))
                    ivy = max(0, min(int(round(vcy)), h - 1))
                    cv2.circle(frame, (ivx, ivy), 8, (255, 255, 0), 2)  # cyan = virtual center (75 px lead)
                    cv2.circle(frame, (ivx, ivy), 3, (255, 255, 0), -1)

            if mode == "orb" and state.get("orb_center_px") is not None and state.get("orb_center_py") is not None:
                ix_red = max(0, min(int(state["orb_center_px"]), w - 1))
                iy_red = max(0, min(int(state["orb_center_py"]), h - 1))
                cv2.circle(frame, (ix_red, iy_red), 10, (0, 0, 255), 3)
                cv2.circle(frame, (ix_red, iy_red), 4, (0, 0, 255), -1)
                vcx = state.get("_virtual_center_x")
                vcy = state.get("_virtual_center_y")
                if vcx is not None and vcy is not None:
                    ivx = max(0, min(int(round(vcx)), w - 1))
                    ivy = max(0, min(int(round(vcy)), h - 1))
                    cv2.circle(frame, (ivx, ivy), 8, (255, 255, 0), 2)  # cyan = virtual center (75 px lead)
                    cv2.circle(frame, (ivx, ivy), 3, (255, 255, 0), -1)

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
                    # In CSRT mode: white vector = crosshair → bbox center (red dot)
                    if mode == "csrt" and state.get("csrt_smooth_px") is not None and state.get("csrt_smooth_py") is not None and not state.get("csrt_lost"):
                        pt_white = (max(0, min(int(round(state["csrt_smooth_px"])), w - 1)), max(0, min(int(round(state["csrt_smooth_py"])), h - 1)))
                    else:
                        pt_white = (int(cx + wx), int(cy + wy))
                    if 0 <= pt_white[0] < w and 0 <= pt_white[1] < h and (cx, cy) != pt_white:
                        cv2.arrowedLine(frame, (cx, cy), pt_white, COLOR_WHITE, 2, tipLength=0.15)

                    # Yellow vector: gradient 0–250 by remaining distance, then temporal smoothing (EV-like)
                    ux = wx / white_len
                    uy = wy / white_len
                    scale_px = state.get("yellow_scale_px")
                    scale_px = YELLOW_GRADIENT_SCALE_PX if scale_px is None else scale_px
                    desired_yellow = YELLOW_MAX_PX * (1.0 - math.exp(-white_len / max(scale_px, 1.0)))
                    desired_yellow = min(YELLOW_MAX_PX, desired_yellow, white_len)
                else:
                    ux = uy = 0.0
                    white_len = 0.0
                    desired_yellow = 0.0

                # Temporal smoothing so yellow length ramps smoothly (no jerk)
                smooth_alpha = state.get("yellow_smooth_alpha")
                smooth_alpha = YELLOW_SMOOTH_ALPHA if smooth_alpha is None else smooth_alpha
                prev = state.get("yellow_smoothed_px")
                if prev is None:
                    state["yellow_smoothed_px"] = desired_yellow
                else:
                    state["yellow_smoothed_px"] = (
                        smooth_alpha * prev + (1.0 - smooth_alpha) * desired_yellow
                    )
                yellow_len = state["yellow_smoothed_px"]
                yellow_draw = min(yellow_len, white_len) if white_len >= 1e-6 else 0.0
                state["_telemetry_white_len"] = white_len
                state["_telemetry_desired_yellow"] = desired_yellow
                if mode in ("csrt", "orb"):
                    err_deg = math.hypot(target_pan - pan_cur, target_tilt - tilt_cur)
                    yellow_max_this = _yellow_max_at_error_deg(err_deg, state.get("yellow_max_curve"))
                    state["yellow_capped"] = min(yellow_draw, yellow_max_this)
                else:
                    state["yellow_capped"] = None
                yellow_draw_effective = (state.get("yellow_capped") if state.get("yellow_capped") is not None else yellow_draw) if mode in ("csrt", "orb") else yellow_draw
                if white_len >= 1e-6 and yellow_draw_effective >= 2.0:
                    guide_px = ch_x + ux * yellow_draw_effective
                    guide_py = ch_y + uy * yellow_draw_effective
                    pt_yellow = (int(guide_px), int(guide_py))
                    if 0 <= pt_yellow[0] < w and 0 <= pt_yellow[1] < h and (cx, cy) != pt_yellow:
                        cv2.arrowedLine(frame, (cx, cy), pt_yellow, COLOR_YELLOW, 2, tipLength=0.15)
            else:
                state["_telemetry_white_len"] = 0.0
                state["_telemetry_desired_yellow"] = 0.0

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

                step_scale = None
                if mode == "csrt":
                    # Command like real click at cyan point: full speed toward target (no yellow throttle)
                    step_scale = 1.0
                elif mode == "orb":
                    yc = state.get("yellow_capped")
                    if yc is not None and YELLOW_MAX_PX >= 1e-6:
                        step_scale = max(0.0, min(1.0, yc / YELLOW_MAX_PX))
                    else:
                        step_scale = 1.0
                last_continuous_arm_move_time, _, _ = _variable_step_toward_target(
                    arm, pan_tgt, tilt_tgt, last_continuous_arm_move_time, throttle_sec, arm_drive_state, step_scale=step_scale
                )
                if error_deg > CONTINUOUS_DEADZONE_DEG:
                    state["_telemetry_frame_count"] = state.get("_telemetry_frame_count", 0) + 1
                    if state["_telemetry_frame_count"] % TELEMETRY_SUBSAMPLE == 0:
                        wl = state.get("_telemetry_white_len", 0.0)
                        dy = state.get("_telemetry_desired_yellow", 0.0)
                        ys = state.get("yellow_smoothed_px") or 0.0
                        vel = getattr(arm_drive_state, "continuous_velocity_mm_s", 0.0)
                        state["telemetry"].append((now, wl, error_deg, dy, ys, vel))

            if mode == "csrt":
                if state.get("csrt_initialized") and not state.get("csrt_lost"):
                    hud_txt = "CSRT tracking  [C]=click [O]=ORB [G]=graph [R]=retrain [ESC]=quit"
                else:
                    hud_txt = "CSRT: drag bbox  [C]=click [O]=ORB [G]=graph [R]=retrain [ESC]=quit"
            elif mode == "orb":
                if state.get("orb_initialized") and not state.get("orb_lost"):
                    hud_txt = "ORB tracking  [O]=click mode [G]=graph [R]=retrain [ESC]=quit"
                else:
                    hud_txt = "ORB: drag bbox  [O]=click mode [G]=graph [R]=retrain [ESC]=quit"
            elif target_click_px is not None and target_click_py is not None:
                dist_px = math.hypot(
                    target_click_px - ch_x,
                    target_click_py - ch_y
                )
                hud_txt = f"Click: {dist_px:.1f} px  [C]=CSRT [O]=ORB [G]=graph [R]=retrain [ESC]=quit"
            else:
                hud_txt = "Click target  [C]=CSRT [O]=ORB [G]=graph [R]=retrain [ESC]=quit"
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
            if key8 == KEY_C:
                if mode == "click":
                    state["mode"] = "csrt"
                    state["target_click_px"] = None
                    state["target_click_py"] = None
                    state["need_update_target_from_click"] = False
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    state["drag_start"] = None
                    state["drag_current"] = None
                    state["pending_bbox"] = None
                    state["csrt_tracker"] = None
                    state["csrt_initialized"] = False
                    state["csrt_bbox"] = None
                    state["csrt_smooth_px"] = None
                    state["csrt_smooth_py"] = None
                    state["csrt_lost"] = False
                    state["_track_frame_count"] = 0
                    print("Mode: CSRT — drag to select bbox")
                else:
                    state["mode"] = "click"
                    state["csrt_tracker"] = None
                    state["csrt_initialized"] = False
                    state["csrt_bbox"] = None
                    state["csrt_smooth_px"] = None
                    state["csrt_smooth_py"] = None
                    state["csrt_lost"] = False
                    state["pending_bbox"] = None
                    state["drag_start"] = None
                    state["drag_current"] = None
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    state["_virtual_center_x"] = None
                    state["_virtual_center_y"] = None
                    print("Mode: Click")
            if key8 == KEY_O:
                if mode == "click":
                    state["mode"] = "orb"
                    state["target_click_px"] = None
                    state["target_click_py"] = None
                    state["need_update_target_from_click"] = False
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    state["drag_start"] = None
                    state["drag_current"] = None
                    state["pending_bbox"] = None
                    state["csrt_tracker"] = None
                    state["csrt_initialized"] = False
                    state["csrt_bbox"] = None
                    state["csrt_smooth_px"] = None
                    state["csrt_smooth_py"] = None
                    state["csrt_lost"] = False
                    state["_track_frame_count"] = 0
                    state["orb_initialized"] = False
                    state["orb_kp"] = None
                    state["orb_des"] = None
                    state["orb_bbox"] = None
                    state["orb_template_wh_small"] = None
                    state["orb_lost"] = False
                    state["orb_center_px"] = None
                    state["orb_center_py"] = None
                    state["_orb_frame_count"] = 0
                    print("Mode: ORB — drag to select bbox")
                elif mode == "orb":
                    state["mode"] = "click"
                    state["orb_initialized"] = False
                    state["orb_kp"] = None
                    state["orb_des"] = None
                    state["orb_bbox"] = None
                    state["orb_template_wh_small"] = None
                    state["orb_lost"] = False
                    state["orb_center_px"] = None
                    state["orb_center_py"] = None
                    state["_orb_frame_count"] = 0
                    state["pending_bbox"] = None
                    state["_virtual_center_x"] = None
                    state["_virtual_center_y"] = None
                    state["drag_start"] = None
                    state["drag_current"] = None
                    state["target_pan"] = None
                    state["target_tilt"] = None
                    print("Mode: Click")
            if key8 in (KEY_G, KEY_V) and state.get("telemetry") and len(state["telemetry"]) > 0:
                _show_telemetry_plot(state)
            if key8 == KEY_R:
                if _retrain_yellow_params(state):
                    scale_px = state.get("yellow_scale_px") or YELLOW_GRADIENT_SCALE_PX
                    print(f"Retrain OK: yellow_scale_px = {scale_px:.1f}")
                elif state.get("telemetry") and len(state["telemetry"]) > 0:
                    print("Retrain: not enough valid points or fit failed")
                else:
                    print("Retrain: no telemetry (click and move first)")
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
