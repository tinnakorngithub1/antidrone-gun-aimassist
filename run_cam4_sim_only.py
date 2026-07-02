"""
run_cam4_sim_only.py — โหมด L (Sim track) เท่านั้น
------------------------------------------------
รันแล้วเข้าโหมดจำลองการบินโดรนทันที: เป้าสุ่ม pattern รอบศูนย์ แขนติดตามด้วย variable step
หรือจอยสติกจำลอง (กด J สลับ). ไม่มีโหมด Calibration / Test คลิก
ใช้ module เดียวกับ cam4_arm_mouse_grid_calibrator (กล้อง, แขน, grid/px_deg, variable step, Kalman)
คีย์: ] = สลับ pattern, K = speed lock, J = variable step / joystick, 0-9 = เลือก pattern, V = telemetry, Y = สลับเป้า Sim/YOLO, A = โหมด Manual (ลากเมาส์), T = โหมดสอน (ลากเมาส์ชี้วัตถุ แขนตาม บันทึก JSONL), R = เทรน/เทรนใหม่ MLP จาก teach log, M = โหมดใช้ model (replay log), O = โหมดติดตามวัตถุ (ลากเมาส์ครอบวัตถุเพื่อ lock แล้วแขนตาม), ESC = ออก
"""

import json
import math
import random
import time
from collections import deque
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

try:
    import config
except ImportError:
    config = None

# กล้อง + แขน + จอยสติก + grid + YOLO (เหมือน calibrator / main)
try:
    from gun_aim_assist import (
        build_camera_from_config,
        ShooterConfig,
        get_screen_size,
        load_yolo_for_aim_assist,
        crop_center_and_resize,
        detect_yolo_full_frame,
        map_detections_to_full_frame,
        pick_best_target,
        CENTER_CROP_RATIO,
        YOLO_INTERVAL,
        YOLO_CONF_MIN,
    )
except ImportError:
    build_camera_from_config = None
    ShooterConfig = None
    get_screen_size = lambda: (1920, 1080)
    load_yolo_for_aim_assist = None
    crop_center_and_resize = None
    detect_yolo_full_frame = None
    map_detections_to_full_frame = None
    pick_best_target = None
    CENTER_CROP_RATIO = 0.5
    YOLO_INTERVAL = 1
    YOLO_CONF_MIN = 0.1

try:
    from cam4_arm_controller import Cam4ArmController
except ImportError:
    Cam4ArmController = None

try:
    from joystick_cam4_controller import JoystickReader, JoystickArmMapper, SENSITIVITY_LOW
except ImportError:
    JoystickReader = None
    JoystickArmMapper = None

try:
    from cam4_arm_grid_lookup import load as load_grid_json, get_crosshair, pixel_to_arm_degrees
except ImportError:
    load_grid_json = None
    get_crosshair = None
    pixel_to_arm_degrees = None

# Throttle ตาม error: ห่าง = ทิ้งช่วง (ส่งคำสั่งไม่ถี่), ใกล้ = อัปเดตถี่ (min_interval)
THROTTLE_FAR_MAX_SEC = 0.12  # เมื่อ error ใหญ่ ใช้ interval สูงสุด (วินาที)
# โหมด YOLO: เมื่อไม่มี detection ใช้ Kalman predict-only ได้ไม่เกินช่วงนี้ (กล้องติดแขน — กรอบภาพเคลื่อนที่)
YOLO_EXTRAPOLATE_MAX_SEC = 0.4
# โหมด YOLO แบบ PTZ: ตั้งเป้าจาก error ต่อเฟรม (current + delta), ใช้ px/deg
YOLO_PTZ_DEADZONE_PX = 8  # ถ้า |error_px| < ค่านี้ ไม่สั่งขยับ (ลด jitter)
YOLO_PTZ_KP = 1.0  # gain: target = current + Kp * error_deg (1.0 = แก้เต็มที่ต่อเฟรม)

# โหมด L ทั้งหมดจาก calibrator
from cam4_arm_mouse_grid_calibrator import (
    CALIBRATION_DIR,
    CAMERA_ON_ARM,
    NEAR_CROSSHAIR_THRESHOLD_PX,
    CONTINUOUS_DEADZONE_DEG,
    CONTINUOUS_P_THROTTLE_FAR_SEC,
    CONTINUOUS_P_THROTTLE_MID_SEC,
    CONTINUOUS_P_THROTTLE_NEAR_SEC,
    CONTINUOUS_P_VERY_FAR_DEG,
    CONTINUOUS_P_ZONE_FAR_DEG,
    CONTINUOUS_P_ZONE_NEAR_DEG,
    CONTINUOUS_THROTTLE_SEC,
    DRONE_SIM_MAX_DEG,
    JSON_PATH,
    PX_DEG_JSON_PATH,
    PX_PER_DEG_X_DEFAULT,
    PX_PER_DEG_Y_DEFAULT,
    REF_OUTPUT_H,
    REF_OUTPUT_W,
    SIM_KALMAN_LOOKAHEAD_SEC,
    SIM_PATTERN_NAMES,
    SWAP_PAN_TILT,
    CalibratorState,
    KEY_ESC,
    KEY_J,
    KEY_NEXT_PATTERN,
    KEY_SPEED_LOCK,
    KEY_TELEMETRY_PLOT,
    COLOR_CROSSHAIR,
    COLOR_HUD_RED,
    FONT_HUD,
    HUD_LEFT_LINE_H,
    RETICLE_RADIUS_PX,
    THICKNESS_THIN,
    _drone_sim_target_deg,
    _kalman_update_and_predict,
    _l_joystick_style_toward_target,
    _load_pixel_per_degree_json,
    _variable_step_toward_target,
    _show_continuous_telemetry_plot,
    _initial_pan_tilt_from_fov,
)
try:
    from cam4_arm_mouse_grid_calibrator import _kalman_predict_only
except ImportError:
    _kalman_predict_only = None  # calibrator เก่าอาจยังไม่มีฟังก์ชันนี้


def predict_mlp(px: float, py: float, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> Tuple[float, float]:
    """Inference MLP 2→32→2: (px, py) -> (pan_deg, tilt_deg). ใช้ NumPy เท่านั้น."""
    x = np.array([[px, py]], dtype=np.float64)
    h = np.maximum(0, x @ W1 + b1)
    y = h @ W2 + b2
    return float(y[0, 0]), float(y[0, 1])


def main() -> None:
    if build_camera_from_config is None or ShooterConfig is None:
        print("gun_aim_assist not available.")
        return
    if Cam4ArmController is None:
        print("cam4_arm_controller not available.")
        return

    camera_name = "cam4"
    cam = build_camera_from_config(camera_name)
    cam.start()
    time.sleep(0.3)

    ShooterConfig().load()

    arm = Cam4ArmController()
    if not arm.connect():
        print("Arm connect failed.")
        cam.release()
        return

    joystick_reader = None
    joystick_mapper = None
    if JoystickReader and JoystickArmMapper:
        try:
            joystick_reader = JoystickReader()
            if getattr(joystick_reader, "enabled", False):
                joystick_mapper = JoystickArmMapper(arm, initial_sensitivity_mode=SENSITIVITY_LOW)
        except Exception:
            pass

    # State เฉพาะโหมด L
    state = CalibratorState()
    state.mode = "test"
    state.test_track_sim = True
    state.track_sim_arm_auto = True
    state.sim_drag_active = False
    state.output_w = REF_OUTPUT_W
    state.output_h = REF_OUTPUT_H
    state.crosshair_x = state.output_w / 2.0
    state.crosshair_y = state.output_h / 2.0
    state.sim_internal_time = 0.0
    state.last_sim_update_time = 0.0
    state.last_sim_detection_time = 0.0
    state.sim_pattern_index = 0
    state.sim_speed_lock = None
    state.sim_speed_state = "medium"
    state.sim_speed_switch_at = time.time() + random.uniform(2.0, 6.0)
    state.l_track_joystick_style = False
    state.last_continuous_arm_move_time = 0.0
    state.continuous_target_pan = None
    state.continuous_target_tilt = None
    state.continuous_target_pan_prev = None
    state.continuous_target_tilt_prev = None
    state.continuous_target_time_prev = 0.0
    state.continuous_velocity_mm_s = 0.0
    state.continuous_error_deg_smooth = None
    state.continuous_telemetry = deque(maxlen=3000)
    state.sim_kalman_x = None
    state.sim_kalman_P = None
    state.sim_kalman_last_time = 0.0
    state.sim_display_px = None
    state.sim_display_py = None
    state.yolo_last_detection_time = 0.0
    state.teach_mode = False
    state.teach_log_file = None
    state.model_control_mode = False
    state.mlp_weights = None
    state.replay_records = None
    state.replay_index = 0
    state.track_object_mode = False
    state.track_object_tracker = None
    state.track_object_initialized = False
    state.track_object_drag_start = None
    state.track_object_pending_bbox = None
    state.track_object_drag_current = None
    state.track_object_lost = False
    state.track_object_bbox = None  # (x, y, w, h) ล่าสุด จาก CSRT สำหรับวาด bbox ตลอดเมื่อ lock
    state.track_object_smooth_px = None  # EMA ลดกระตุกของจุดเป้า
    state.track_object_smooth_py = None

    # โหลด px_per_degree
    px_deg_data = _load_pixel_per_degree_json()
    if px_deg_data is not None:
        ppdx = px_deg_data.get("pixel_per_degree_x")
        ppdy = px_deg_data.get("pixel_per_degree_y")
        try:
            state.px_per_deg_x = float(ppdx) if ppdx is not None else PX_PER_DEG_X_DEFAULT
            state.px_per_deg_y = float(ppdy) if ppdy is not None else PX_PER_DEG_Y_DEFAULT
        except (TypeError, ValueError):
            state.px_per_deg_x = PX_PER_DEG_X_DEFAULT
            state.px_per_deg_y = PX_PER_DEG_Y_DEFAULT
    else:
        state.px_per_deg_x = PX_PER_DEG_X_DEFAULT
        state.px_per_deg_y = PX_PER_DEG_Y_DEFAULT

    # โหลด grid (ถ้ามี) สำหรับ crosshair จาก JSON
    if JSON_PATH.is_file() and load_grid_json and get_crosshair:
        state.test_data = load_grid_json(JSON_PATH)
        if state.test_data is not None:
            cx, cy = get_crosshair(state.test_data)
            ow = int(state.test_data.get("output_width", REF_OUTPUT_W))
            oh = int(state.test_data.get("output_height", REF_OUTPUT_H))
            if ow > 0 and oh > 0 and (ow != state.output_w or oh != state.output_h):
                cx = cx * state.output_w / ow
                cy = cy * state.output_h / oh
            state.crosshair_x = cx
            state.crosshair_y = cy

    # โหลด YOLO สำหรับโหมดเป้า YOLO (กด Y สลับ)
    yolo_model = None
    yolo_center_imgsz = 1280
    if load_yolo_for_aim_assist is not None:
        try:
            yolo_model, yolo_center_imgsz = load_yolo_for_aim_assist()
            if yolo_model is None:
                yolo_center_imgsz = 1280
        except Exception as e:
            print(f"YOLO load failed: {e}")
    state.use_yolo_target = False

    win_name = "Cam4 Sim Only (L-mode)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        screen_w, screen_h = get_screen_size()
    except Exception:
        screen_w, screen_h = 1920, 1080

    last_loop_time = time.time()
    frame_counter = 0
    print("Sim Only: [ ] = pattern, [K] = speed lock, [J] = track style, 0-9 = pattern, [V] = plot, [Y] = Sim/YOLO, [A] = เมาส์ลาก, [ESC] = quit")

    def _px_display_to_output(display_x, display_y, st):
        """แมปพิกัดจากหน้าต่างแสดงผลไป output frame."""
        if st.display_w <= 0 or st.display_h <= 0:
            return float(display_x), float(display_y)
        x = display_x * st.output_w / st.display_w
        y = display_y * st.output_h / st.display_h
        return max(0.0, min(st.output_w - 1.0, x)), max(0.0, min(st.output_h - 1.0, y))

    def _pixel_to_pan_tilt_p_style(px_out: float, py_out: float, st) -> Optional[Tuple[float, float]]:
        """แปลงพิกเซล → (pan_deg, tilt_deg) แบบเดียวกับกด P ใน Test: ใกล้ศูนย์ = px/deg, ไกล = homography."""
        if st.output_w <= 0 or st.output_h <= 0:
            return None
        cx = getattr(st, "crosshair_x", st.output_w / 2.0)
        cy = getattr(st, "crosshair_y", st.output_h / 2.0)
        dist = math.sqrt((px_out - cx) ** 2 + (py_out - cy) ** 2)
        near_thresh = NEAR_CROSSHAIR_THRESHOLD_PX * (st.output_w / 1920.0)
        if dist <= near_thresh and getattr(st, "px_per_deg_x", None) is not None and getattr(st, "px_per_deg_y", None) is not None:
            pan = (px_out - cx) / st.px_per_deg_x
            tilt = (py_out - cy) / st.px_per_deg_y
            return pan, tilt
        if getattr(st, "test_data", None) and pixel_to_arm_degrees is not None:
            res = pixel_to_arm_degrees(
                px_out, py_out, st.test_data, REF_OUTPUT_W, REF_OUTPUT_H, use_homography=True
            )
            if res is not None:
                return res[0], res[1]
        if getattr(st, "px_per_deg_x", None) is not None and getattr(st, "px_per_deg_y", None) is not None:
            pan = (px_out - cx) / st.px_per_deg_x
            tilt = (py_out - cy) / st.px_per_deg_y
            return pan, tilt
        pan, tilt = _initial_pan_tilt_from_fov(px_out, py_out, cx, cy, REF_OUTPUT_W, REF_OUTPUT_H)
        return pan, tilt

    TRACK_OBJECT_MIN_BBOX = 4  # w,h ขั้นต่ำก่อน init tracker

    def _on_mouse(event, x, y, _flags, param):
        if param is None or not isinstance(param, dict) or "state" not in param:
            return
        st = param["state"]
        if st.output_w <= 0 or st.output_h <= 0 or st.display_w <= 0 or st.display_h <= 0:
            return
        # โหมด O: ลากเมาส์ครอบวัตถุ (select ROI) เพื่อ lock และ track
        if getattr(st, "track_object_mode", False) and not getattr(st, "track_object_initialized", False):
            px_out, py_out = _px_display_to_output(x, y, st)
            if event == cv2.EVENT_LBUTTONDOWN:
                st.track_object_drag_start = (px_out, py_out)
                st.track_object_drag_current = None
                return
            if event == cv2.EVENT_MOUSEMOVE and getattr(st, "track_object_drag_start", None) is not None:
                st.track_object_drag_current = (px_out, py_out)
                return
            if event == cv2.EVENT_LBUTTONUP and getattr(st, "track_object_drag_start", None) is not None:
                x1, y1 = st.track_object_drag_start
                x2, y2 = px_out, py_out
                x_min = max(0, min(x1, x2))
                x_max = min(st.output_w, max(x1, x2))
                y_min = max(0, min(y1, y2))
                y_max = min(st.output_h, max(y1, y2))
                bw = x_max - x_min
                bh = y_max - y_min
                if bw >= TRACK_OBJECT_MIN_BBOX and bh >= TRACK_OBJECT_MIN_BBOX:
                    st.track_object_pending_bbox = (x_min, y_min, bw, bh)
                st.track_object_drag_start = None
                st.track_object_drag_current = None
                return
        if not getattr(st, "track_sim_arm_auto", True):
            if event == cv2.EVENT_LBUTTONUP:
                st.sim_drag_active = False
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                st.sim_drag_active = True
                px_out, py_out = _px_display_to_output(x, y, st)
                st.sim_display_px = px_out
                st.sim_display_py = py_out
            elif event == cv2.EVENT_MOUSEMOVE and getattr(st, "sim_drag_active", False):
                px_out, py_out = _px_display_to_output(x, y, st)
                st.sim_display_px = px_out
                st.sim_display_py = py_out

    cv2.setMouseCallback(win_name, _on_mouse, {"arm": arm, "state": state})

    while True:
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
        now = time.time()
        frame_counter += 1

        scale = min(screen_w / w, screen_h / h, 1.0)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        state.display_w = scaled_w
        state.display_h = scaled_h

        if not getattr(state, "track_sim_arm_auto", True):
            # โหมด A: เป้าอัปเดตเฉพาะตอนคลิกหรือลาก (จาก callback) ไม่ poll เมาส์ในลูป → จุดคงที่เมื่อปล่อยเมาส์ แขนไล่จน error 0
            state.crosshair_x = state.output_w / 2.0
            state.crosshair_y = state.output_h / 2.0
            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass

        x_lim = getattr(arm, "_effective_x_limits", None)
        y_lim = getattr(arm, "_effective_y_limits", None)
        if x_lim is None or y_lim is None:
            margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
            xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
            yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
            x_lim = (xr[0] + margin, xr[1] - margin)
            y_lim = (yr[0] + margin, yr[1] - margin)
        x_lo, x_hi = x_lim[0], x_lim[1]
        y_lo, y_hi = y_lim[0], y_lim[1]

        track_auto = getattr(state, "track_sim_arm_auto", True)
        use_yolo = getattr(state, "use_yolo_target", False)
        yolo_had_detection_this_frame = False
        yolo_ran_this_frame = False

        # โหมด O: init — ใช้ CSRT จำและติดตามวัตถุใน bbox ที่วาด
        pending_bbox = getattr(state, "track_object_pending_bbox", None)
        if pending_bbox is not None and getattr(state, "track_object_mode", False):
            xb, yb, wb, hb = pending_bbox
            xb = max(0, min(int(xb), w - 1))
            yb = max(0, min(int(yb), h - 1))
            wb = max(TRACK_OBJECT_MIN_BBOX, min(int(wb), w - xb))
            hb = max(TRACK_OBJECT_MIN_BBOX, min(int(hb), h - yb))
            tracker = None
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
            except Exception:
                try:
                    tracker = cv2.TrackerCSRT_create()
                except Exception:
                    pass
            if tracker is not None:
                try:
                    ok = tracker.init(frame, (xb, yb, wb, hb))
                    if ok:
                        state.track_object_tracker = tracker
                        state.track_object_bbox = (xb, yb, wb, hb)
                        state.track_object_initialized = True
                        state.track_object_lost = False
                except Exception:
                    pass
            state.track_object_pending_bbox = None

        if track_auto and use_yolo and yolo_model is not None and frame_counter % YOLO_INTERVAL == 0 and not getattr(state, "model_control_mode", False) and not getattr(state, "track_object_mode", False):
            yolo_ran_this_frame = True
            center_crop, x0, y0, cw, ch = crop_center_and_resize(
                frame, CENTER_CROP_RATIO, yolo_center_imgsz
            )
            detections_crop = detect_yolo_full_frame(
                yolo_model, center_crop, YOLO_CONF_MIN, imgsz=yolo_center_imgsz
            )
            detections = map_detections_to_full_frame(
                detections_crop, x0, y0, cw, ch, yolo_center_imgsz
            )
            cx_frame = w / 2.0
            cy_frame = h / 2.0
            min_side = min(w, h)
            target_det = pick_best_target(detections, cx_frame, cy_frame, min_side)
            if target_det is not None:
                x_d, y_d, bw, bh, _ = target_det
                raw_px = float(x_d + bw // 2)
                raw_py = float(y_d + bh // 2)
                state.sim_display_px = raw_px
                state.sim_display_py = raw_py
                # จุดสีส้ม: smooth ด้วย EMA ลดสั่นจาก FPS / detection แปลผัน (ได้น้อยเพราะบางเฟรมไม่มี detection)
                alpha = 0.35
                prev_px = getattr(state, "yolo_display_px", None)
                prev_py = getattr(state, "yolo_display_py", None)
                if prev_px is None or prev_py is None:
                    state.yolo_display_px = raw_px
                    state.yolo_display_py = raw_py
                else:
                    state.yolo_display_px = alpha * raw_px + (1.0 - alpha) * prev_px
                    state.yolo_display_py = alpha * raw_py + (1.0 - alpha) * prev_py
                yolo_had_detection_this_frame = True
                state.yolo_last_detection_time = now
            # บางเฟรมไม่มี detection — ไม่ล้างเป้า แขนหมุนไปเป้าเดิมต่อจนได้ detection ใหม่
        # (ลบการ clear continuous_target เมื่อไม่มี detection เพื่อให้แขนยังหมุนเข้าเป้าล่าสุดได้)
        elif track_auto and not use_yolo and not getattr(state, "model_control_mode", False) and not getattr(state, "track_object_mode", False):
            # ----- อัปเดตเป้าสิม (โหมด L Auto) -----
            lock = getattr(state, "sim_speed_lock", None)
            if lock is not None:
                state.sim_speed_state = lock
            elif now >= getattr(state, "sim_speed_switch_at", 0.0):
                state.sim_speed_switch_at = now + random.uniform(2.0, 6.0)
                state.sim_speed_state = random.choice(["stop", "slow", "medium", "fast"])
            speed_scale = {"stop": 0.0, "slow": 0.5, "medium": 1.5, "fast": 4.0}.get(
                getattr(state, "sim_speed_state", "medium"), 1.5
            )
            last_up = getattr(state, "last_sim_update_time", now)
            if last_up <= 0.0:
                last_up = now
            dt = max(0.0, now - last_up)
            state.sim_internal_time = getattr(state, "sim_internal_time", 0.0) + dt * speed_scale
            state.last_sim_update_time = now
            t_sec_effective = state.sim_internal_time
            pan_deg, tilt_deg = _drone_sim_target_deg(
                t_sec_effective,
                getattr(state, "sim_pattern_index", 0),
                state,
            )
            cx = getattr(state, "crosshair_x", state.output_w / 2.0)
            cy = getattr(state, "crosshair_y", state.output_h / 2.0)
            current_pan = getattr(arm, "pos_x", 0.0)
            current_tilt = getattr(arm, "pos_y", 0.0)
            if CAMERA_ON_ARM:
                d_pan = pan_deg - current_pan
                d_tilt = tilt_deg - current_tilt
            else:
                d_pan, d_tilt = pan_deg, tilt_deg
            if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                state.sim_display_px = cx + d_pan * state.px_per_deg_x
                state.sim_display_py = cy + d_tilt * state.px_per_deg_y
            else:
                scale_fallback = min(state.output_w, state.output_h) / 40.0
                state.sim_display_px = state.output_w / 2.0 + d_pan * scale_fallback
                state.sim_display_py = state.output_h / 2.0 + d_tilt * scale_fallback

        # โหมด O: อัปเดตตำแหน่งวัตถุจาก CSRT tracker + EMA ลดกระตุก
        TRACK_OBJECT_SMOOTH_ALPHA = 0.28  # เล็ก = เรียบมาก, ใหญ่ = ตามเร็วแต่สั่น
        if getattr(state, "track_object_mode", False) and getattr(state, "track_object_initialized", False):
            tr = getattr(state, "track_object_tracker", None)
            if tr is not None:
                try:
                    success, bbox = tr.update(frame)
                    if success and bbox is not None:
                        xb, yb, wb, hb = bbox
                        cx = float(xb + wb * 0.5)
                        cy = float(yb + hb * 0.5)
                        prev_sx = getattr(state, "track_object_smooth_px", None)
                        prev_sy = getattr(state, "track_object_smooth_py", None)
                        if prev_sx is None or prev_sy is None:
                            state.track_object_smooth_px = cx
                            state.track_object_smooth_py = cy
                        else:
                            a = TRACK_OBJECT_SMOOTH_ALPHA
                            state.track_object_smooth_px = a * cx + (1.0 - a) * prev_sx
                            state.track_object_smooth_py = a * cy + (1.0 - a) * prev_sy
                        state.sim_display_px = state.track_object_smooth_px
                        state.sim_display_py = state.track_object_smooth_py
                        state.track_object_bbox = (int(xb), int(yb), int(wb), int(hb))
                        state.track_object_lost = False
                    else:
                        state.track_object_bbox = None
                        state.track_object_lost = True
                except Exception:
                    state.track_object_bbox = None
                    state.track_object_lost = True
            else:
                state.track_object_bbox = None
                state.track_object_lost = True

        # อัปเดตเป้าแขน: โหมด Auto = Sim/YOLO; โหมด A = ใช้ sim_display_px/py จากเมาส์; โหมด O = จาก tracker (pipeline เดียวกัน)
        have_px_deg = state.px_per_deg_x is not None and state.px_per_deg_y is not None
        yolo_can_use_helper = use_yolo and yolo_had_detection_this_frame and have_px_deg
        a_mode_can_use_helper = not track_auto and (have_px_deg or getattr(state, "test_data", None))
        if (
            state.output_w > 0
            and state.output_h > 0
            and (state.sim_display_px is not None and state.sim_display_py is not None or getattr(state, "track_object_mode", False))
            and (have_px_deg or yolo_can_use_helper or a_mode_can_use_helper or getattr(state, "model_control_mode", False) or getattr(state, "track_object_mode", False))
            and (not track_auto or (not use_yolo or yolo_had_detection_this_frame) or getattr(state, "model_control_mode", False) or getattr(state, "track_object_mode", False))
        ):
            # โหมด O ติดตามวัตถุ (CSRT): เมื่อ Tracker หลุด → รักษาตำแหน่งล่าสุด (แขนหยุดนิ่ง)
            if getattr(state, "track_object_mode", False) and getattr(state, "track_object_lost", False):
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
                pan_cur = getattr(arm, "pos_x", 0.0)
                tilt_cur = getattr(arm, "pos_y", 0.0)
                state.continuous_target_pan = float(np.clip(pan_cur, x_lo, x_hi))
                state.continuous_target_tilt = float(np.clip(tilt_cur, y_lo, y_hi))
            # โหมด M (model): replay (px, py) จาก log → model → ตั้งเป้าแขน
            elif getattr(state, "model_control_mode", False) and getattr(state, "mlp_weights", None) and getattr(state, "replay_records", None):
                recs = state.replay_records
                idx = state.replay_index
                r = recs[idx]
                px_m, py_m = r["px"], r["py"]
                state.sim_display_px = px_m
                state.sim_display_py = py_m
                W1, b1, W2, b2 = state.mlp_weights["W1"], state.mlp_weights["b1"], state.mlp_weights["W2"], state.mlp_weights["b2"]
                pan_deg, tilt_deg = predict_mlp(px_m, py_m, W1, b1, W2, b2)
                if SWAP_PAN_TILT:
                    state.continuous_target_pan = float(np.clip(pan_deg, y_lo, y_hi))
                    state.continuous_target_tilt = float(np.clip(tilt_deg, x_lo, x_hi))
                else:
                    state.continuous_target_pan = float(np.clip(pan_deg, x_lo, x_hi))
                    state.continuous_target_tilt = float(np.clip(tilt_deg, y_lo, y_hi))
                state.replay_index = (idx + 1) % len(recs)
            else:
                # โหมด O (CSRT) ยังไม่มีพิกัดจาก tracker (เฟรมแรกหรือยังไม่อัปเดต) → hold ตำแหน่งปัจจุบัน
                if getattr(state, "track_object_mode", False) and (state.sim_display_px is None or state.sim_display_py is None):
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    pan_cur = getattr(arm, "pos_x", 0.0)
                    tilt_cur = getattr(arm, "pos_y", 0.0)
                    state.continuous_target_pan = float(np.clip(pan_cur, x_lo, x_hi))
                    state.continuous_target_tilt = float(np.clip(tilt_cur, y_lo, y_hi))
                else:
                    # Lookahead: โหมด YOLO / โหมด O (CSRT) ใช้ dynamic ตาม error — ทำนายตำแหน่งล่วงหน้า (SIM_KALMAN_LOOKAHEAD_SEC)
                    lookahead_sec = SIM_KALMAN_LOOKAHEAD_SEC
                    if use_yolo and yolo_had_detection_this_frame and have_px_deg:
                        cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                        cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                        dx_px = state.sim_display_px - cx
                        dy_px = state.sim_display_py - cy
                        dx_deg = dx_px / state.px_per_deg_x if state.px_per_deg_x else 0.0
                        dy_deg = dy_px / state.px_per_deg_y if state.px_per_deg_y else 0.0
                        error_deg_approx = math.sqrt(dx_deg * dx_deg + dy_deg * dy_deg)
                        extra = min(0.12, error_deg_approx / 50.0)
                        lookahead_sec = min(0.25, SIM_KALMAN_LOOKAHEAD_SEC + extra)
                    elif getattr(state, "track_object_mode", False) and not getattr(state, "track_object_lost", True) and have_px_deg and state.sim_display_px is not None and state.sim_display_py is not None:
                        # โหมด O (CSRT): lookahead ตาม error เช่นเดียวกับ Sim/YOLO
                        cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                        cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                        dx_px = state.sim_display_px - cx
                        dy_px = state.sim_display_py - cy
                        dx_deg = dx_px / state.px_per_deg_x if state.px_per_deg_x else 0.0
                        dy_deg = dy_px / state.px_per_deg_y if state.px_per_deg_y else 0.0
                        error_deg_approx = math.sqrt(dx_deg * dx_deg + dy_deg * dy_deg)
                        extra = min(0.12, error_deg_approx / 50.0)
                        lookahead_sec = min(0.25, SIM_KALMAN_LOOKAHEAD_SEC + extra)
                    px_target, py_target = _kalman_update_and_predict(
                        state,
                        state.sim_display_px,
                        state.sim_display_py,
                        now,
                        lookahead_sec,
                        state.output_w,
                        state.output_h,
                    )
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    if use_yolo and yolo_had_detection_this_frame:
                        # โหมด YOLO แบบ PTZ: ตั้งเป้าจาก error ต่อเฟรม (target = current + Kp * error_deg), ใช้ px/deg
                        pan_cur = getattr(arm, "pos_x", 0.0)
                        tilt_cur = getattr(arm, "pos_y", 0.0)
                        cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                        cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                        error_px_x = px_target - cx
                        error_px_y = py_target - cy
                        # desired = มุมที่ทำให้วัตถุอยู่กลาง; ได้จาก helper (px/deg ใกล้ศูนย์, homography ไกล) หรือ px/deg
                        res = _pixel_to_pan_tilt_p_style(px_target, py_target, state)
                        if res is not None:
                            desired_pan, desired_tilt = res
                        else:
                            delta_pan = error_px_x / state.px_per_deg_x if state.px_per_deg_x else 0.0
                            delta_tilt = error_px_y / state.px_per_deg_y if state.px_per_deg_y else 0.0
                            desired_pan = pan_cur + delta_pan if CAMERA_ON_ARM else delta_pan
                            desired_tilt = tilt_cur + delta_tilt if CAMERA_ON_ARM else delta_tilt
                        error_deg_pan = desired_pan - pan_cur
                        error_deg_tilt = desired_tilt - tilt_cur
                        # dead zone (pixel): ไม่ขยับถ้า error เล็ก
                        if abs(error_px_x) < YOLO_PTZ_DEADZONE_PX:
                            error_deg_pan = 0.0
                        if abs(error_px_y) < YOLO_PTZ_DEADZONE_PX:
                            error_deg_tilt = 0.0
                        target_pan = pan_cur + YOLO_PTZ_KP * error_deg_pan
                        target_tilt = tilt_cur + YOLO_PTZ_KP * error_deg_tilt
                        if SWAP_PAN_TILT:
                            state.continuous_target_pan = float(np.clip(target_pan, y_lo, y_hi))
                            state.continuous_target_tilt = float(np.clip(target_tilt, x_lo, x_hi))
                        else:
                            state.continuous_target_pan = float(np.clip(target_pan, x_lo, x_hi))
                            state.continuous_target_tilt = float(np.clip(target_tilt, y_lo, y_hi))
                        state.continuous_velocity_mm_s = 0.0
                        state.continuous_error_deg_smooth = None
                    else:
                        pan_cur = getattr(arm, "pos_x", 0.0)
                        tilt_cur = getattr(arm, "pos_y", 0.0)
                        cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                        cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                        dx_px = px_target - cx
                        dy_px = py_target - cy
                        delta_pan = dx_px / state.px_per_deg_x if state.px_per_deg_x else 0.0
                        delta_tilt = dy_px / state.px_per_deg_y if state.px_per_deg_y else 0.0
                        if not track_auto:
                            # โหมด A: logic เดียวกับ Sim ทั้งหมด — แค่จุดสีส้ม (sim_display) มาจากเมาส์ที่ลาก (ตั้งใน callback)
                            res = _pixel_to_pan_tilt_p_style(px_target, py_target, state)
                            if res is not None:
                                target_pan, target_tilt = res
                            elif state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                                target_pan = pan_cur + delta_pan
                                target_tilt = tilt_cur + delta_tilt
                            else:
                                target_pan, target_tilt = _initial_pan_tilt_from_fov(
                                    px_target, py_target, cx, cy, REF_OUTPUT_W, REF_OUTPUT_H
                                )
                                target_pan = target_pan if target_pan is not None else 0.0
                                target_tilt = target_tilt if target_tilt is not None else 0.0
                            if SWAP_PAN_TILT:
                                state.continuous_target_pan = float(np.clip(target_pan, y_lo, y_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, x_lo, x_hi))
                            else:
                                state.continuous_target_pan = float(np.clip(target_pan, x_lo, x_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, y_lo, y_hi))
                            state.continuous_velocity_mm_s = 0.0
                            state.continuous_error_deg_smooth = None
                            if getattr(state, "teach_mode", False) and getattr(state, "teach_log_file", None) is not None and state.sim_display_px is not None and state.sim_display_py is not None:
                                line = json.dumps({
                                    "px": state.sim_display_px,
                                    "py": state.sim_display_py,
                                    "pan_deg": state.continuous_target_pan,
                                    "tilt_deg": state.continuous_target_tilt,
                                    "pan_cur": pan_cur,
                                    "tilt_cur": tilt_cur,
                                    "t": now,
                                }) + "\n"
                                state.teach_log_file.write(line)
                                state.teach_log_file.flush()
                        elif CAMERA_ON_ARM:
                            target_pan = pan_cur + delta_pan
                            target_tilt = tilt_cur + delta_tilt
                            if SWAP_PAN_TILT:
                                state.continuous_target_pan = float(np.clip(target_pan, y_lo, y_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, x_lo, x_hi))
                            else:
                                state.continuous_target_pan = float(np.clip(target_pan, x_lo, x_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, y_lo, y_hi))
                            state.continuous_velocity_mm_s = 0.0
                            state.continuous_error_deg_smooth = None
                        else:
                            target_pan = delta_pan
                            target_tilt = delta_tilt
                            if SWAP_PAN_TILT:
                                state.continuous_target_pan = float(np.clip(target_pan, y_lo, y_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, x_lo, x_hi))
                            else:
                                state.continuous_target_pan = float(np.clip(target_pan, x_lo, x_hi))
                                state.continuous_target_tilt = float(np.clip(target_tilt, y_lo, y_hi))
                            state.continuous_velocity_mm_s = 0.0
                            state.continuous_error_deg_smooth = None
        else:
            # โหมด YOLO ไม่มี detection: ใช้ Kalman predict-only เป็นสะพาน (จำกัดเวลา เพราะกล้องติดแขน)
            if (
                use_yolo
                and not yolo_had_detection_this_frame
                and getattr(state, "sim_kalman_x", None) is not None
                and (now - getattr(state, "yolo_last_detection_time", 0.0)) <= YOLO_EXTRAPOLATE_MAX_SEC
                and _kalman_predict_only is not None
            ):
                lookahead_sec = SIM_KALMAN_LOOKAHEAD_SEC
                pred = _kalman_predict_only(
                    state, now, lookahead_sec, state.output_w, state.output_h
                )
                if pred is not None:
                    px_pred, py_pred = pred
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    pan_cur = getattr(arm, "pos_x", 0.0)
                    tilt_cur = getattr(arm, "pos_y", 0.0)
                    cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                    cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                    error_px_x = px_pred - cx
                    error_px_y = py_pred - cy
                    res = _pixel_to_pan_tilt_p_style(px_pred, py_pred, state)
                    if res is not None:
                        desired_pan, desired_tilt = res
                    else:
                        delta_pan = error_px_x / state.px_per_deg_x if state.px_per_deg_x else 0.0
                        delta_tilt = error_px_y / state.px_per_deg_y if state.px_per_deg_y else 0.0
                        desired_pan = pan_cur + delta_pan if CAMERA_ON_ARM else delta_pan
                        desired_tilt = tilt_cur + delta_tilt if CAMERA_ON_ARM else delta_tilt
                    error_deg_pan = desired_pan - pan_cur
                    error_deg_tilt = desired_tilt - tilt_cur
                    if abs(error_px_x) < YOLO_PTZ_DEADZONE_PX:
                        error_deg_pan = 0.0
                    if abs(error_px_y) < YOLO_PTZ_DEADZONE_PX:
                        error_deg_tilt = 0.0
                    target_pan = pan_cur + YOLO_PTZ_KP * error_deg_pan
                    target_tilt = tilt_cur + YOLO_PTZ_KP * error_deg_tilt
                    if SWAP_PAN_TILT:
                        state.continuous_target_pan = float(np.clip(target_pan, y_lo, y_hi))
                        state.continuous_target_tilt = float(np.clip(target_tilt, x_lo, x_hi))
                    else:
                        state.continuous_target_pan = float(np.clip(target_pan, x_lo, x_hi))
                        state.continuous_target_tilt = float(np.clip(target_tilt, y_lo, y_hi))
            elif (
                not use_yolo
                and now - getattr(state, "last_sim_detection_time", 0.0) >= getattr(state, "sim_detection_interval_sec", 0.06)
            ):
                state.last_sim_detection_time = now
                if SWAP_PAN_TILT:
                    state.continuous_target_pan = float(np.clip(pan_deg, y_lo, y_hi))
                    state.continuous_target_tilt = float(np.clip(tilt_deg, x_lo, x_hi))
                else:
                    state.continuous_target_pan = float(np.clip(pan_deg, x_lo, x_hi))
                    state.continuous_target_tilt = float(np.clip(tilt_deg, y_lo, y_hi))
                state.continuous_velocity_mm_s = 0.0
                state.continuous_error_deg_smooth = None

        # ----- ส่งแขนไปเป้า (variable step หรือ joystick style) -----
        if (
            state.continuous_target_pan is not None
            and state.continuous_target_tilt is not None
        ):
            pan_tgt = state.continuous_target_pan
            tilt_tgt = state.continuous_target_tilt
            current_pan = getattr(arm, "pos_x", 0.0)
            current_tilt = getattr(arm, "pos_y", 0.0)
            error_deg = (pan_tgt - current_pan) ** 2 + (tilt_tgt - current_tilt) ** 2
            error_deg = error_deg ** 0.5
            min_interval = getattr(arm, "_min_move_interval_sec", 0.02)
            # Throttle ตาม error: ห่าง = ทิ้งช่วง (throttle ใหญ่), ใกล้ = อัปเดตถี่ (min_interval)
            max_throttle = max(min_interval, THROTTLE_FAR_MAX_SEC)
            if use_yolo and state.sim_display_px is not None and state.sim_display_py is not None and state.px_per_deg_x and state.px_per_deg_y:
                cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                error_px = math.sqrt((state.sim_display_px - cx) ** 2 + (state.sim_display_py - cy) ** 2)
                px_per_deg = (abs(state.px_per_deg_x) + abs(state.px_per_deg_y)) / 2.0
                zone_far_px = max(1.0, CONTINUOUS_P_ZONE_FAR_DEG * px_per_deg)
                # linear: error 0 → min_interval, error >= zone_far_px → max_throttle
                t = min(1.0, error_px / zone_far_px)
                throttle_sec = min_interval + (max_throttle - min_interval) * t
                throttle_sec = max(min_interval, min(max_throttle, throttle_sec))
            else:
                # โหมด A และ Sim: throttle ตาม error_deg (linear เช่นกัน)
                zone_far_deg = max(0.1, CONTINUOUS_P_ZONE_FAR_DEG)
                t = min(1.0, error_deg / zone_far_deg)
                throttle_sec = min_interval + (max_throttle - min_interval) * t
                throttle_sec = max(min_interval, min(max_throttle, throttle_sec))

            if getattr(state, "l_track_joystick_style", False):
                new_time, _, _ = _l_joystick_style_toward_target(
                    arm, pan_tgt, tilt_tgt,
                    state.last_continuous_arm_move_time, throttle_sec, state,
                )
            else:
                new_time, _, _ = _variable_step_toward_target(
                    arm, pan_tgt, tilt_tgt,
                    state.last_continuous_arm_move_time, throttle_sec, state,
                )
            state.last_continuous_arm_move_time = new_time

        # ----- วาด crosshair + จุดสม + HUD -----
        cx_i = int(state.crosshair_x)
        cy_i = int(state.crosshair_y)
        r0 = RETICLE_RADIUS_PX[0]
        x_left = max(0, cx_i - r0)
        x_right = min(w, cx_i + r0)
        y_top = max(0, cy_i - r0)
        y_bottom = min(h, cy_i + r0)
        cv2.line(frame, (0, cy_i), (x_left, cy_i), COLOR_CROSSHAIR, 2)
        cv2.line(frame, (x_right, cy_i), (w, cy_i), COLOR_CROSSHAIR, 2)
        cv2.line(frame, (cx_i, 0), (cx_i, y_top), COLOR_CROSSHAIR, 2)
        cv2.line(frame, (cx_i, y_bottom), (cx_i, h), COLOR_CROSSHAIR, 2)
        for r in RETICLE_RADIUS_PX:
            cv2.circle(frame, (cx_i, cy_i), r, COLOR_CROSSHAIR, 2)
        cv2.circle(frame, (cx_i, cy_i), 3, COLOR_CROSSHAIR, -1)

        # โหมด YOLO: วาดจุดสีส้มที่ bbox เท่านั้น (yolo_display_px/py) ไม่ใช้ค่าที่อาจขยับตามแขน
        if getattr(state, "use_yolo_target", False) and getattr(state, "yolo_display_px", None) is not None and getattr(state, "yolo_display_py", None) is not None:
            px = state.yolo_display_px
            py = state.yolo_display_py
        else:
            px = getattr(state, "sim_display_px", None)
            py = getattr(state, "sim_display_py", None)
        if px is not None and py is not None:
            ix = int(px)
            iy = int(py)
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(frame, (ix, iy), 12, (0, 165, 255), 3)
                cv2.circle(frame, (ix, iy), 4, (0, 200, 255), -1)

        # โหมด O: วาด preview สี่เหลี่ยมขณะลากเลือก ROI
        ds = getattr(state, "track_object_drag_start", None)
        dc = getattr(state, "track_object_drag_current", None)
        if ds is not None and dc is not None:
            x1, y1 = int(ds[0]), int(ds[1])
            x2, y2 = int(dc[0]), int(dc[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # โหมด O lock แล้ว: วาด bbox ครอบวัตถุที่ติดตามตลอด
        tbbox = getattr(state, "track_object_bbox", None)
        if tbbox is not None and getattr(state, "track_object_initialized", False):
            xb, yb, wb, hb = tbbox
            if wb > 0 and hb > 0:
                cv2.rectangle(frame, (xb, yb), (xb + wb, yb + hb), (0, 255, 0), 2)

        y_msg = int(26 * (REF_OUTPUT_W / 1920.0))
        y_mode = y_msg + HUD_LEFT_LINE_H
        y_keys = y_mode + HUD_LEFT_LINE_H
        y_arm = y_keys + HUD_LEFT_LINE_H
        y_l1 = y_arm + HUD_LEFT_LINE_H
        y_l2 = y_l1 + HUD_LEFT_LINE_H
        y_l3 = y_l2 + HUD_LEFT_LINE_H
        target_src = "YOLO" if getattr(state, "use_yolo_target", False) else "Sim"
        if getattr(state, "track_object_mode", False):
            if getattr(state, "track_object_lost", False):
                auto_manual = "Track lost [O] drag again"
            elif getattr(state, "track_object_initialized", False):
                auto_manual = "Tracking object [O]"
            else:
                auto_manual = "Track object [O] drag to select ROI"
        elif getattr(state, "model_control_mode", False):
            auto_manual = "Model (replay)"
        elif getattr(state, "teach_mode", False):
            auto_manual = "Teach (log)"
        elif not getattr(state, "track_sim_arm_auto", True):
            auto_manual = "Manual (ลากเมาส์)"
        else:
            auto_manual = "Auto"
        cv2.putText(frame, f"Target: {target_src}  [Y]  |  {auto_manual}  [A] Teach [T] R M O  pattern 0-9 ]  K J V  ESC quit",
                    (10, y_msg), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
        pidx = getattr(state, "sim_pattern_index", 0) % len(SIM_PATTERN_NAMES)
        pname = SIM_PATTERN_NAMES[pidx]
        speed_lock = getattr(state, "sim_speed_lock", None)
        speed_label = "Fast" if speed_lock == "fast" else "Slow" if speed_lock == "slow" else "Stop" if speed_lock == "stop" else "Auto"
        track_style = "Joystick" if getattr(state, "l_track_joystick_style", False) else "Variable step"
        cv2.putText(frame, f"Pattern: [{pidx}] {pname}  Speed: {speed_label} [K]  Track: {track_style} [J]",
                    (10, y_l1), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
        ax = getattr(arm, "pos_x", 0.0)
        ay = getattr(arm, "pos_y", 0.0)
        cv2.putText(frame, f"Arm from (0,0): X={ax:.1f}  Y={ay:.1f}", (10, y_arm), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
        if px is not None and py is not None:
            dx = px - state.crosshair_x
            dy = py - state.crosshair_y
            err_px = (dx * dx + dy * dy) ** 0.5
            cv2.putText(frame, f"Sim err: dX={dx:.1f} dY={dy:.1f} dist={err_px:.1f} px",
                        (10, y_l2), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)

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
        key8 = (key & 0xFF) if key >= 0 else -1
        if key8 == KEY_ESC:
            break
        if key8 == KEY_NEXT_PATTERN:
            state.sim_pattern_index = (getattr(state, "sim_pattern_index", 0) + 1) % len(SIM_PATTERN_NAMES)
        if key8 == KEY_SPEED_LOCK:
            lock = getattr(state, "sim_speed_lock", None)
            state.sim_speed_lock = ("fast" if lock is None else "slow" if lock == "fast" else "stop" if lock == "slow" else None)
        if key8 == KEY_J:
            state.l_track_joystick_style = not getattr(state, "l_track_joystick_style", False)
        if ord("0") <= key8 <= ord("9"):
            state.sim_pattern_index = (key8 - ord("0")) % len(SIM_PATTERN_NAMES)
        if key8 == KEY_TELEMETRY_PLOT:
            _show_continuous_telemetry_plot(state, live=False)
        if key8 in (ord("y"), ord("Y")) and yolo_model is not None:
            state.use_yolo_target = not getattr(state, "use_yolo_target", False)
            if not state.use_yolo_target:
                state.yolo_display_px = None
                state.yolo_display_py = None
        if key8 in (ord("a"), ord("A")):
            state.track_sim_arm_auto = not getattr(state, "track_sim_arm_auto", True)
            state.sim_drag_active = False
        if key8 in (ord("t"), ord("T")):
            if not getattr(state, "teach_mode", False):
                state.teach_mode = True
                state.track_sim_arm_auto = False
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                path = CALIBRATION_DIR / f"cam4_teach_{ts}.jsonl"
                try:
                    state.teach_log_file = open(path, "a", encoding="utf-8")
                except Exception:
                    state.teach_log_file = None
            else:
                state.teach_mode = False
                if getattr(state, "teach_log_file", None) is not None:
                    state.teach_log_file.close()
                    state.teach_log_file = None
        if key8 in (ord("r"), ord("R")):
            try:
                import train_mlp
                if not train_mlp.train_and_save():
                    print("R: No teach log or no records — train skipped.")
            except Exception as e:
                print("R: Train failed:", e)
        if key8 in (ord("m"), ord("M")):
            if not getattr(state, "model_control_mode", False):
                npz_path = CALIBRATION_DIR / "cam4_arm_mlp.npz"
                if not npz_path.is_file():
                    print("M: No cam4_arm_mlp.npz — train with R first.")
                else:
                    try:
                        data = np.load(npz_path)
                        state.mlp_weights = {k: data[k] for k in ("W1", "b1", "W2", "b2")}
                    except Exception as e:
                        print("M: Failed to load model:", e)
                        state.mlp_weights = None
                    if state.mlp_weights is not None:
                        logs = sorted(CALIBRATION_DIR.glob("cam4_teach_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if not logs:
                            print("M: No cam4_teach_*.jsonl for replay.")
                            state.mlp_weights = None
                        else:
                            recs = []
                            with open(logs[0], "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        r = json.loads(line)
                                        if "px" in r and "py" in r:
                                            recs.append(r)
                                    except (json.JSONDecodeError, TypeError):
                                        continue
                            if not recs:
                                print("M: No valid records in latest teach log.")
                                state.mlp_weights = None
                            else:
                                state.replay_records = recs
                                state.replay_index = 0
                                state.model_control_mode = True
                                print("M: Model (replay) ON,", len(recs), "frames")
                    if not getattr(state, "model_control_mode", False):
                        state.mlp_weights = None
                        state.replay_records = None
            else:
                state.model_control_mode = False
                print("M: Model (replay) OFF")
        if key8 in (ord("o"), ord("O")):
            state.track_object_mode = not getattr(state, "track_object_mode", False)
            if not state.track_object_mode:
                state.track_object_tracker = None
                state.track_object_initialized = False
                state.track_object_drag_start = None
                state.track_object_pending_bbox = None
                state.track_object_drag_current = None
                state.track_object_lost = False
                state.track_object_bbox = None
                state.track_object_smooth_px = None
                state.track_object_smooth_py = None
                print("O: Track object OFF")
            else:
                # แขนกลับไป home (0,0) ก่อน แล้วค่อยลากเลือกวัตถุ
                state.continuous_target_pan = 0.0
                state.continuous_target_tilt = 0.0
                print("O: Track object ON — arm going home, then drag to select ROI")

    if getattr(state, "teach_log_file", None) is not None:
        state.teach_log_file.close()
        state.teach_log_file = None
    cam.release()
    cv2.destroyAllWindows()
    print("Sim Only: stopped.")


if __name__ == "__main__":
    main()
