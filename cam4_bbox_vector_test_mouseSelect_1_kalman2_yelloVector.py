"""
Cam4 Bbox Vector Test
cam4_bbox_vector_test.py
---------------------
Copy of cam4_mouse_distance_hud structure; arm does NOT rotate at start.
Draw bbox (drag) -> red dot = bbox center, white vector crosshair->center, HUD = vector length (px).
Press Enter to start arm rotation toward that position using guidance r = 5% of white vector length
(shrinks as vector shortens). CSRT tracks bbox so white vector shortens as arm approaches.
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
TRACK_SCALE = 0.25
CSRT_MIN_BBOX = 4
CSRT_SMOOTH_ALPHA = 0.28
# เวกเตอร์สีเหลือง: ขาว >= 75 px → r = 75; ขาว < 75 px → r = 40% ของความยาวขาว
GUIDANCE_R_MAX_PX = 75       # ความยาวเหลืองคงที่เมื่อเป้าไกล (ขาว >= 75 px)
GUIDANCE_R_NEAR_FRACTION = 0.05  # เมื่อขาว < 75 px ใช้ r = 40% ของความยาวขาว
CSRT_UPDATE_EVERY_N_FRAMES = 2  # รัน tracker ทุก N เฟรม; เฟรมอื่นใช้ Kalman predict (ปรับได้)

# Throttle เฉพาะสคริปต์นี้: ต่ำลง = ส่ง move บ่อยขึ้น = ลื่นขึ้น ไม่กระตุก (ค่าจาก calibrator 0.02)
BBOX_THROTTLE_FAR_SEC = 0.012
BBOX_THROTTLE_MID_SEC = 0.012
BBOX_THROTTLE_NEAR_SEC = 0.015

HUD_TEXT_SCALE = max(1.0, REF_OUTPUT_W / 1920.0)
FONT_HUD = 0.65 * HUD_TEXT_SCALE
THICKNESS_THIN = max(1, int(1 * HUD_TEXT_SCALE))

COLOR_CROSSHAIR = (0, 255, 0)
RETICLE_RADIUS_PX = (12, 20, 32, 52)
COLOR_HUD_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)  # BGR — vector to guidance point
KEY_ESC = 27
KEY_ENTER = 13
KEY_ENTER_LF = 10

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


def _create_kalman_for_center(cx: float, cy: float):
    """Kalman filter for 2D position (x, y) with constant-velocity model. State: [x, y, vx, vy]."""
    kf = cv2.KalmanFilter(4, 2)
    # x' = x + vx, y' = y + vy, vx' = vx, vy' = vy (dt=1)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    return kf


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
        print("gun_aim_assist not available.")
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
        "arm_drive_active": False,
        "lost_need_crop": False,
        "kalman": None,
        "_track_frame_count": 0,
    }
    arm_drive_state = _ArmDriveState()
    arm_drive_state.continuous_target_time_prev = time.time() - 0.05
    last_continuous_arm_move_time = 0.0

    win_name = "Cam4 Bbox Vector Test"
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
                frame = cv2.resize(frame, (REF_OUTPUT_W, REF_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
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

            scale_x = 1.0
            scale_y = 1.0

            # Draw drag rectangle while dragging
            if state.get("drag_start") is not None and state.get("drag_current") is not None:
                x1, y1 = state["drag_start"]
                x2, y2 = state["drag_current"]
                ix1 = int(x1 * scale_x)
                iy1 = int(y1 * scale_y)
                ix2 = int(x2 * scale_x)
                iy2 = int(y2 * scale_y)
                ix1 = max(0, min(ix1, w - 1))
                iy1 = max(0, min(iy1, h - 1))
                ix2 = max(0, min(ix2, w - 1))
                iy2 = max(0, min(iy2, h - 1))
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

            # Init CSRT from pending_bbox (or re-init when user crops a new target)
            if state.get("pending_bbox") is not None:
                if state.get("csrt_initialized"):
                    state["csrt_tracker"] = None
                    state["csrt_initialized"] = False
                    state["csrt_bbox"] = None
                    state["csrt_smooth_px"] = None
                    state["csrt_smooth_py"] = None
                    state["csrt_lost"] = False
                    state["arm_drive_active"] = False
                    state["lost_need_crop"] = False
                    state["kalman"] = None
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
                            cx0 = xb + wb * 0.5
                            cy0 = yb + hb * 0.5
                            state["csrt_tracker"] = tracker
                            state["csrt_initialized"] = True
                            state["csrt_bbox"] = (xb, yb, wb, hb)
                            state["csrt_smooth_px"] = cx0
                            state["csrt_smooth_py"] = cy0
                            state["csrt_lost"] = False
                            state["lost_need_crop"] = False
                            state["arm_drive_active"] = True
                            state["kalman"] = _create_kalman_for_center(cx0, cy0)
                            state["_track_frame_count"] = 0
                    except Exception:
                        pass
                state["pending_bbox"] = None

            # Update CSRT every N frames when initialized; every frame use Kalman for red dot / vector
            if state.get("csrt_initialized") and state.get("csrt_tracker") is not None:
                count = state.get("_track_frame_count", 0)
                kf = state.get("kalman")
                if kf is not None:
                    kf.predict()
                    state["csrt_smooth_px"] = float(kf.statePost[0, 0])
                    state["csrt_smooth_py"] = float(kf.statePost[1, 0])
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
                            cx_bbox = float(xb + wb * 0.5)
                            cy_bbox = float(yb + hb * 0.5)
                            kf = state.get("kalman")
                            if kf is None:
                                state["kalman"] = _create_kalman_for_center(cx_bbox, cy_bbox)
                                state["csrt_smooth_px"] = cx_bbox
                                state["csrt_smooth_py"] = cy_bbox
                            else:
                                est = kf.correct(np.array([[cx_bbox], [cy_bbox]], dtype=np.float32))
                                state["csrt_smooth_px"] = float(est[0, 0])
                                state["csrt_smooth_py"] = float(est[1, 0])
                            state["csrt_bbox"] = (xb, yb, wb, hb)
                            state["csrt_lost"] = False
                        else:
                            state["csrt_bbox"] = None
                            state["csrt_lost"] = True
                            state["csrt_tracker"] = None
                            state["csrt_initialized"] = False
                            state["csrt_smooth_px"] = None
                            state["csrt_smooth_py"] = None
                            state["arm_drive_active"] = False
                            state["lost_need_crop"] = True
                            state["kalman"] = None
                            state["_track_frame_count"] = 0
                    except Exception:
                        state["csrt_bbox"] = None
                        state["csrt_lost"] = True
                        state["csrt_tracker"] = None
                        state["csrt_initialized"] = False
                        state["csrt_smooth_px"] = None
                        state["csrt_smooth_py"] = None
                        state["arm_drive_active"] = False
                        state["lost_need_crop"] = True
                        state["kalman"] = None
                        state["_track_frame_count"] = 0
                state["_track_frame_count"] = count + 1

            # Draw tracking bbox (green) when we have it and not lost
            if state.get("csrt_bbox") is not None and not state.get("csrt_lost"):
                xb, yb, wb, hb = state["csrt_bbox"]
                ix0 = max(0, min(int(xb), w - 1))
                iy0 = max(0, min(int(yb), h - 1))
                ix1 = max(0, min(int(xb + wb), w - 1))
                iy1 = max(0, min(int(yb + hb), h - 1))
                cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (0, 255, 0), 2)

            # Draw red dot and white vector when CSRT has center and not lost
            vector_len_px = None
            if state.get("csrt_initialized") and not state.get("csrt_lost"):
                bx = state.get("csrt_smooth_px")
                by = state.get("csrt_smooth_py")
                if bx is not None and by is not None:
                    ix_red = int(bx * scale_x)
                    iy_red = int(by * scale_y)
                    ix_red = max(0, min(ix_red, w - 1))
                    iy_red = max(0, min(iy_red, h - 1))
                    cv2.circle(frame, (ix_red, iy_red), 10, (0, 0, 255), 3)
                    cv2.circle(frame, (ix_red, iy_red), 4, (0, 0, 255), -1)
                    pt1 = (cx, cy)
                    pt2 = (ix_red, iy_red)
                    if pt1 != pt2:
                        cv2.arrowedLine(frame, pt1, pt2, COLOR_WHITE, 2, tipLength=0.15)
                    vector_len_px = math.sqrt((bx - ch_x) ** 2 + (by - ch_y) ** 2)
                    # Yellow vector: white >= 75 px → r=75; white < 75 px → r=40% of white
                    if vector_len_px >= 1e-6 and state.get("arm_drive_active"):
                        if vector_len_px >= GUIDANCE_R_MAX_PX:
                            r = GUIDANCE_R_MAX_PX
                        else:
                            r = GUIDANCE_R_NEAR_FRACTION * vector_len_px
                        ux = (bx - ch_x) / vector_len_px
                        uy = (by - ch_y) / vector_len_px
                        guide_px = ch_x + ux * r
                        guide_py = ch_y + uy * r
                        pt_guide = (int(guide_px), int(guide_py))
                        if 0 <= pt_guide[0] < w and 0 <= pt_guide[1] < h and (cx, cy) != pt_guide:
                            cv2.arrowedLine(frame, (cx, cy), pt_guide, COLOR_YELLOW, 2, tipLength=0.15)

            # Arm drive: only when arm_drive_active and have bbox center (r = 5% of vector length)
            if (
                state.get("arm_drive_active")
                and state.get("csrt_initialized")
                and not state.get("csrt_lost")
                and state.get("csrt_smooth_px") is not None
                and state.get("csrt_smooth_py") is not None
                and _variable_step_toward_target is not None
                and px_per_deg_x is not None
                and px_per_deg_y is not None
            ):
                bx = state["csrt_smooth_px"]
                by = state["csrt_smooth_py"]
                vec_len = math.sqrt((bx - ch_x) ** 2 + (by - ch_y) ** 2)
                if vec_len < 1e-6:
                    guide_px = ch_x
                    guide_py = ch_y
                else:
                    ux = (bx - ch_x) / vec_len
                    uy = (by - ch_y) / vec_len
                    if vec_len >= GUIDANCE_R_MAX_PX:
                        r = GUIDANCE_R_MAX_PX
                    else:
                        r = GUIDANCE_R_NEAR_FRACTION * vec_len
                    guide_px = ch_x + ux * r
                    guide_py = ch_y + uy * r
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
                pan_cur = getattr(arm, "pos_x", 0.0)
                tilt_cur = getattr(arm, "pos_y", 0.0)
                dx_px = guide_px - ch_x
                dy_px = guide_py - ch_y
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
                    throttle_sec = BBOX_THROTTLE_FAR_SEC
                elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
                    throttle_sec = BBOX_THROTTLE_MID_SEC
                else:
                    throttle_sec = BBOX_THROTTLE_NEAR_SEC
                min_interval = getattr(arm, "_min_move_interval_sec", 0.02)
                if error_deg > CONTINUOUS_P_VERY_FAR_DEG:
                    throttle_sec = max(throttle_sec, min(min_interval, 0.01))
                else:
                    throttle_sec = max(throttle_sec, min_interval)
                last_continuous_arm_move_time, _, _ = _variable_step_toward_target(
                    arm, pan_tgt, tilt_tgt, last_continuous_arm_move_time, throttle_sec, arm_drive_state
                )

            if vector_len_px is not None:
                hud_txt = f"Vector length: {vector_len_px:.1f} px  [Enter]=start arm  [ESC]=quit"
            elif state.get("lost_need_crop"):
                hud_txt = "Target lost — drag to crop again  [ESC]=quit"
            else:
                hud_txt = "Drag to select bbox  [Enter]=start arm  [ESC]=quit"
            y_hud = h - int(30 * HUD_TEXT_SCALE)
            cv2.putText(frame, hud_txt, (10, y_hud), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)

            scale = min(screen_w / w, screen_h / h, 1.0)
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            state["display_w"] = scaled_w
            state["display_h"] = scaled_h
            if (scaled_w, scaled_h) != (w, h):
                display_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
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
            if key8 in (KEY_ENTER, KEY_ENTER_LF):
                state["arm_drive_active"] = True
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

