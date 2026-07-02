"""
main.py: Gun Aim Assist with calibrator L-mode arm tracking
- ใช้โมดูลทั้งหมดจาก gun_aim_assist.py (กล้อง, YOLO, HUD, จอยสติก, โหมด, ยิง)
- การติดตามวัตถุโดยการหมุนแขนใช้ logic โหมด Test กด L จาก cam4_arm_mouse_grid_calibrator
- เลือกแบบหมุนแขน (variable step หรือ sam joystick) จากค่าตั้งค่าบนสุดด้านล่าง
"""
# -----------------------------------------------------------------------------
# ค่าตั้งค่าบนสุด: แบบหมุนแขนเมื่อติดตามเป้า (AUTO/LOCK)
# "variable_step" = พุ่งเข้าเป้าตามระยะ (feed×dt, แขนเร็วกว่าเป้า)
# "sam_joystick" = แบบจอยสติกจำลอง (error → virtual stick → rate×dt)
# -----------------------------------------------------------------------------
ARM_TRACK_STYLE = "variable_step"  # or "sam_joystick"

# YOLO confidence ขั้นต่ำ (0.0–1.0): detection ที่ conf ต่ำกว่านี้จะไม่นับ
YOLO_CONF_MIN = 0.01

# ใช้ Kalman ทำนายตำแหน่งล่วงหน้า (lookahead) เป็นเป้าแขน — ลดการสั่นกระตุก (เหมือน calibrator โหมด L)
USE_KALMAN_LOOKAHEAD = True

import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ใช้ gun_aim_assist ครบทุกอย่าง (ยกเว้นจะแทนที่บล็อก arm tracking)
import gun_aim_assist as gaa

# Grid lookup สำหรับแปลง pixel → องศาแขน (เหมือน calibrator โหมด L)
from cam4_arm_grid_lookup import (
    get_crosshair,
    load as load_grid_json,
    load_pixel_per_degree,
    pixel_to_arm_degrees,
)

# ฟังก์ชันส่งแขนแบบ variable step / joystick style จาก calibrator
from cam4_arm_mouse_grid_calibrator import (
    CONTINUOUS_P_THROTTLE_FAR_SEC,
    CONTINUOUS_P_THROTTLE_MID_SEC,
    CONTINUOUS_P_THROTTLE_NEAR_SEC,
    CONTINUOUS_P_VERY_FAR_DEG,
    CONTINUOUS_P_ZONE_FAR_DEG,
    CONTINUOUS_P_ZONE_NEAR_DEG,
    CONTINUOUS_THROTTLE_SEC,
    JSON_PATH,
    NEAR_CROSSHAIR_THRESHOLD_PX,
    PX_DEG_JSON_PATH,
    PX_PER_DEG_X_DEFAULT,
    PX_PER_DEG_Y_DEFAULT,
    REF_OUTPUT_H,
    REF_OUTPUT_W,
    SIM_KALMAN_LOOKAHEAD_SEC,
    SWAP_PAN_TILT,
    _initial_pan_tilt_from_fov,
    _kalman_update_and_predict,
    _l_joystick_style_toward_target,
    _variable_step_toward_target,
)

try:
    import config as app_config
except ImportError:
    app_config = None


class TrackState:
    """State object สำหรับ L-style arm move (ใช้กับ _variable_step_toward_target / _l_joystick_style_toward_target)."""

    def __init__(self):
        self.continuous_target_pan = None
        self.continuous_target_tilt = None
        self.continuous_target_pan_prev = None
        self.continuous_target_tilt_prev = None
        self.continuous_target_time_prev = 0.0
        self.last_continuous_arm_move_time = 0.0
        self.continuous_velocity_mm_s = 0.0
        self.continuous_telemetry = None
        # Kalman lookahead (ตรง calibrator โหมด L)
        self.sim_kalman_x: Optional[object] = None
        self.sim_kalman_P: Optional[object] = None
        self.sim_kalman_last_time: float = 0.0


def _get_arm_limits(arm):
    """ได้ (x_lo, x_hi), (y_lo, y_hi) สำหรับ clip เป้าแขน (องศา)."""
    x_lim = getattr(arm, "_effective_x_limits", None)
    y_lim = getattr(arm, "_effective_y_limits", None)
    if x_lim is not None and y_lim is not None:
        return x_lim, y_lim
    margin = getattr(app_config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if app_config else 2.0
    xr = getattr(app_config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if app_config else (-65.0, 65.0)
    yr = getattr(app_config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if app_config else (-35.0, 35.0)
    x_lim = (xr[0] + margin, xr[1] - margin)
    y_lim = (yr[0] + margin, yr[1] - margin)
    return x_lim, y_lim


def _throttle_sec_for_error(arm, pan_tgt, tilt_tgt):
    """เลือก throttle (วินาที) ตาม error องศา (ให้ตรงกับ calibrator โหมด L)."""
    current_pan = getattr(arm, "pos_x", 0.0)
    current_tilt = getattr(arm, "pos_y", 0.0)
    error_deg = math.hypot(pan_tgt - current_pan, tilt_tgt - current_tilt)
    if error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
        return CONTINUOUS_THROTTLE_SEC
    if error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
        return CONTINUOUS_P_THROTTLE_MID_SEC
    return CONTINUOUS_P_THROTTLE_NEAR_SEC


def main():
    camera_name = gaa.CAMERA_NAME if gaa.CAMERA_NAME is not None else gaa.ACTIVE_CAMERA
    cam_config = gaa.get_camera_config(camera_name)
    fov_h = cam_config.get("fov_horizontal", 60.0)
    fov_v = cam_config.get("fov_vertical", 36.0)
    print("main.py (Gun Aim Assist + calibrator L-style tracking): starting camera", camera_name)

    # ------------------------------------------------------------------
    # Optional: Initialize cam4 arm controller + tracker (tracker ใช้แค่ overlay ในโหมดจำลอง)
    # ------------------------------------------------------------------
    arm_controller = None
    arm_tracker = None
    arm_mode_manager = None
    joystick_reader = None
    joystick_mapper = None

    if gaa.Cam4ArmController is not None and gaa.Cam4ArmTracker is not None:
        use_arm = getattr(app_config, "CAM4_ARM_ENABLED", False) if app_config else False
        use_sim = getattr(app_config, "CAM4_ARM_SIMULATION_MODE", False) if app_config else False
        if use_arm and (camera_name == "cam4" or use_sim):
            try:
                if use_sim and gaa.SimCam4ArmController is not None:
                    print("main.py: initializing SimCam4ArmController (simulation mode)...")
                    arm_controller = gaa.SimCam4ArmController()
                else:
                    print("main.py: initializing Cam4ArmController...")
                    arm_controller = gaa.Cam4ArmController()
                if arm_controller.connect():
                    arm_tracker = gaa.Cam4ArmTracker(arm_controller, camera_name="cam4")
                else:
                    print("main.py: Cam4ArmController connect() failed, running without arm.")
                    arm_controller = None
            except Exception as e:
                print(f"main.py: failed to initialize Cam4 arm: {e}")
                arm_controller = None

    if gaa.ArmModeManager is not None:
        arm_mode_manager = gaa.ArmModeManager(has_arm=arm_controller is not None)
    if gaa.JoystickReader is not None and gaa.JoystickArmMapper is not None and arm_controller is not None:
        try:
            joystick_reader = gaa.JoystickReader()
            if getattr(joystick_reader, "enabled", False):
                max_pan = getattr(app_config, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0) if app_config else 80.0
                max_tilt = getattr(app_config, "CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0) if app_config else 60.0
                deadzone = getattr(app_config, "CAM4_ARM_JOYSTICK_DEADZONE", 0.05) if app_config else 0.05
                stick_exp = getattr(app_config, "CAM4_ARM_JOYSTICK_STICK_EXPONENT", 2.0) if app_config else 2.0
                scale_min = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MIN", 0.1) if app_config else 0.1
                scale_max = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MAX", 10.0) if app_config else 10.0
                speed_smooth = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SMOOTH_ALPHA", 0.82) if app_config else 0.82
                sens_str = getattr(app_config, "CAM4_ARM_JOYSTICK_SENSITIVITY", "medium") if app_config else "medium"
                initial_sens = gaa._parse_sensitivity_mode(sens_str)
                joystick_mapper = gaa.JoystickArmMapper(
                    arm_controller,
                    max_pan_rate_deg=max_pan,
                    max_tilt_rate_deg=max_tilt,
                    deadzone=deadzone,
                    stick_exponent=stick_exp,
                    speed_axis_scale_min=scale_min,
                    speed_axis_scale_max=scale_max,
                    speed_smooth_alpha=speed_smooth,
                    initial_sensitivity_mode=initial_sens,
                )
            else:
                joystick_reader = None
        except Exception as e:
            print(f"main.py: joystick init failed: {e}")
            joystick_reader = None
            joystick_mapper = None

    # โหลด grid JSON สำหรับแปลง pixel → องศา (เหมือน calibrator โหมด L)
    grid_data = load_grid_json(JSON_PATH) if JSON_PATH else None
    px_deg_data = load_pixel_per_degree(PX_DEG_JSON_PATH) if PX_DEG_JSON_PATH else None
    ref_ow = (
        float(px_deg_data["output_width"])
        if px_deg_data and px_deg_data.get("output_width") is not None
        else REF_OUTPUT_W
    )
    ref_oh = (
        float(px_deg_data["output_height"])
        if px_deg_data and px_deg_data.get("output_height") is not None
        else REF_OUTPUT_H
    )
    px_deg_cx, px_deg_cy = get_crosshair(px_deg_data) if px_deg_data else (ref_ow / 2.0, ref_oh / 2.0)
    px_per_deg_x = (
        float(px_deg_data["pixel_per_degree_x"])
        if px_deg_data and px_deg_data.get("pixel_per_degree_x") is not None
        else PX_PER_DEG_X_DEFAULT
    )
    px_per_deg_y = (
        float(px_deg_data["pixel_per_degree_y"])
        if px_deg_data and px_deg_data.get("pixel_per_degree_y") is not None
        else PX_PER_DEG_Y_DEFAULT
    )
    if (
        arm_controller is not None
        and grid_data is None
        and (px_per_deg_x is None or px_per_deg_y is None)
    ):
        print(
            "main.py: no grid JSON and no px/deg data - arm tracking will be skipped (no pixel→deg mapping)."
        )
    track_state = TrackState()

    cam = gaa.build_camera_from_config(camera_name)
    cam.start()

    print("Loading YOLO model...")
    yolo_model, yolo_center_imgsz = gaa.load_yolo_for_aim_assist()
    if yolo_model is None:
        print("YOLO model not available. Exiting.")
        cam.release()
        return

    cv2.namedWindow(gaa.WINDOW_NAME, cv2.WINDOW_NORMAL)
    screen_w, screen_h = gaa.get_screen_size()
    if gaa.DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(gaa.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    config = gaa.ShooterConfig()
    config.load()

    if gaa.USE_SHORT_BEEP:
        beep_path = str(Path(gaa.__file__).resolve().parent / gaa.SOUND_FILE_SHORT)
        gaa.create_short_beep_wav(beep_path, duration_sec=gaa.BEEP_SHORT_DURATION_SEC)

    app_mode = "normal"
    settings_selected_field = 0
    settings_step = (10, 50, 0.5, 0.05)
    settings_min_max = ((10, 500), (200, 1500), (1, 50), (0.1, 2.0))

    fps_times = deque(maxlen=30)
    frame_counter = 0
    loop_frames = 0
    GRACE_FRAMES = 60
    last_detections = []
    last_target_det = None
    target_missing_count = 0
    smoothed_aim_cx = None
    smoothed_aim_cy = None
    prev_ready_to_fire = False
    last_ready_sound_time = 0.0
    last_approach_beep_time = 0.0

    is_firing = False
    last_fire_time = 0.0
    last_fire_gcode_time = 0.0
    prev_sensitivity_cycle_pressed = False
    last_loop_time = time.time()

    use_joystick_style = ARM_TRACK_STYLE.strip().lower() == "sam_joystick"
    arm_track_state = {"source": "bbox", "mouse_drag": False, "mouse_x": 0.0, "mouse_y": 0.0}

    def _on_mouse(event, x, y, _flags, param):
        if param is None or not isinstance(param, dict):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if param.get("source") == "mouse":
                param["mouse_drag"] = True
                param["mouse_x"] = float(x)
                param["mouse_y"] = float(y)
        elif event == cv2.EVENT_LBUTTONUP:
            param["mouse_drag"] = False
        elif event == cv2.EVENT_MOUSEMOVE and param.get("mouse_drag"):
            param["mouse_x"] = float(x)
            param["mouse_y"] = float(y)

    try:
        cv2.setMouseCallback(gaa.WINDOW_NAME, _on_mouse, arm_track_state)
    except cv2.error:
        pass

    while True:
        t0 = time.time()
        active, frame, _ = cam.read()
        if not active or frame is None:
            time.sleep(0.01)
            continue

        loop_frames += 1

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        cx_frame, cy_frame = config.get_center(w, h)

        min_side = min(h, w)
        radius_px = int(min_side * gaa.RETICLE_RADIUS_RATIOS[0])
        if gaa.CENTER_RADIUS_PX > 0:
            radius_px = gaa.CENTER_RADIUS_PX
        radius_px = max(10, radius_px)

        if gaa.DISPLAY_FULLSCREEN:
            _dw, _dh = screen_w, screen_h
            _sc = min(_dw / w, _dh / h)
            _sw, _sh = int(w * _sc), int(h * _sc)
            _xo, _yo = (_dw - _sw) // 2, (_dh - _sh) // 2
        else:
            _dw = int(w * min(min(screen_w, gaa.DISPLAY_MAX_WIDTH) / w, min(screen_h, gaa.DISPLAY_MAX_HEIGHT) / h, 1.0))
            _dh = int(h * min(min(screen_w, gaa.DISPLAY_MAX_WIDTH) / w, min(screen_h, gaa.DISPLAY_MAX_HEIGHT) / h, 1.0))
            _sw, _sh = _dw, _dh
            _xo, _yo = 0, 0
        arm_track_state["display_w"] = _dw
        arm_track_state["display_h"] = _dh
        arm_track_state["x_off"] = _xo
        arm_track_state["y_off"] = _yo
        arm_track_state["scaled_w"] = _sw
        arm_track_state["scaled_h"] = _sh
        arm_track_state["frame_w"] = w
        arm_track_state["frame_h"] = h

        now_loop = time.time()
        dt_loop = now_loop - last_loop_time if now_loop > last_loop_time else 0.0
        last_loop_time = now_loop

        if app_mode == "normal":
            frame_counter += 1
            if frame_counter % gaa.YOLO_INTERVAL == 0:
                center_crop, x0, y0, cw, ch = gaa.crop_center_and_resize(
                    frame, gaa.CENTER_CROP_RATIO, yolo_center_imgsz
                )
                detections_crop = gaa.detect_yolo_full_frame(
                    yolo_model, center_crop, YOLO_CONF_MIN, imgsz=yolo_center_imgsz
                )
                last_detections = gaa.map_detections_to_full_frame(
                    detections_crop, x0, y0, cw, ch, yolo_center_imgsz
                )
                sticky_radius_px = max(40, int(min_side * gaa.TARGET_STICKY_RADIUS_RATIO))
                if last_target_det is None:
                    last_target_det = gaa.pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                    target_missing_count = 0
                else:
                    same_target = gaa.find_detection_near_reference(
                        last_target_det, last_detections, sticky_radius_px, min_side
                    )
                    if same_target is not None:
                        last_target_det = same_target
                        target_missing_count = 0
                    else:
                        target_missing_count += 1
                        if target_missing_count >= gaa.TARGET_GRACE_UPDATES:
                            last_target_det = gaa.pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                            target_missing_count = 0

            target_det = last_target_det
            ready_to_fire = False
            if target_det is None:
                track_state.sim_kalman_x = None
                track_state.sim_kalman_P = None
            if target_det is not None:
                x, y, w_d, h_d, _ = target_det
                cx_t = x + w_d // 2
                cy_t = y + h_d // 2
                alpha = gaa.AIM_CENTER_SMOOTH_ALPHA
                if smoothed_aim_cx is None:
                    smoothed_aim_cx, smoothed_aim_cy = float(cx_t), float(cy_t)
                else:
                    smoothed_aim_cx = alpha * smoothed_aim_cx + (1.0 - alpha) * cx_t
                    smoothed_aim_cy = alpha * smoothed_aim_cy + (1.0 - alpha) * cy_t
                d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                ready_to_fire = d <= radius_px

                # ----- การติดตามวัตถุโดยการหมุนแขน: ใช้ calibrator L-style (ไม่ใช้ arm_tracker.update_from_detection) -----
                pass

            # Arm track block: bbox or mouse drag, same L-style pipeline run when AUTO/LOCK and (bbox+target or mouse+drag)
            if (
                arm_controller is not None
                and arm_mode_manager is not None
                and (arm_mode_manager.mode == gaa.MODE_AUTO or arm_mode_manager.mode == gaa.MODE_LOCK)
            ):
                px_out = None
                py_out = None
                if arm_track_state["source"] == "bbox" and target_det is not None and smoothed_aim_cx is not None and smoothed_aim_cy is not None:
                    px_out = smoothed_aim_cx * ref_ow / w
                    py_out = smoothed_aim_cy * ref_oh / h
                elif arm_track_state["source"] == "mouse" and arm_track_state.get("mouse_drag"):
                    _mx = arm_track_state["mouse_x"]
                    _my = arm_track_state["mouse_y"]
                    _xo = arm_track_state.get("x_off", 0)
                    _yo = arm_track_state.get("y_off", 0)
                    _sw = max(arm_track_state.get("scaled_w", w), 1)
                    _sh = max(arm_track_state.get("scaled_h", h), 1)
                    _fw = arm_track_state.get("frame_w", w)
                    _fh = arm_track_state.get("frame_h", h)
                    px_frame = (_mx - _xo) * _fw / _sw
                    py_frame = (_my - _yo) * _fh / _sh
                    px_out = px_frame * ref_ow / w
                    py_out = py_frame * ref_oh / h
                if px_out is not None and py_out is not None:
                    cx, cy = px_deg_cx, px_deg_cy
                    res = None
                    use_kalman = (
                        USE_KALMAN_LOOKAHEAD
                        and arm_track_state["source"] == "bbox"
                        and px_per_deg_x is not None
                        and px_per_deg_y is not None
                    )
                    if use_kalman:
                        px_target, py_target = _kalman_update_and_predict(
                            track_state,
                            px_out,
                            py_out,
                            time.time(),
                            SIM_KALMAN_LOOKAHEAD_SEC,
                            int(ref_ow),
                            int(ref_oh),
                        )
                        pan_deg = (px_target - cx) / px_per_deg_x
                        tilt_deg = (py_target - cy) / px_per_deg_y
                        res = (pan_deg, tilt_deg)
                    if res is None:
                        dist = math.sqrt((px_out - cx) ** 2 + (py_out - cy) ** 2)
                        near_thresh = NEAR_CROSSHAIR_THRESHOLD_PX * (ref_ow / 1920.0)
                        if dist <= near_thresh and px_per_deg_x is not None and px_per_deg_y is not None:
                            pan_deg = (px_out - cx) / px_per_deg_x
                            tilt_deg = (py_out - cy) / px_per_deg_y
                            res = (pan_deg, tilt_deg)
                        if res is None and grid_data is not None:
                            res = pixel_to_arm_degrees(
                                px_out, py_out, grid_data, int(ref_ow), int(ref_oh), use_homography=True
                            )
                        if res is None and px_per_deg_x is not None and px_per_deg_y is not None:
                            pan_deg = (px_out - cx) / px_per_deg_x
                            tilt_deg = (py_out - cy) / px_per_deg_y
                            res = (pan_deg, tilt_deg)
                        if res is None:
                            pan_deg, tilt_deg = _initial_pan_tilt_from_fov(
                                px_out, py_out, cx, cy, int(ref_ow), int(ref_oh)
                            )
                            res = (pan_deg, tilt_deg)
                    if res is not None:
                        pan_deg, tilt_deg = res[0], res[1]
                        x_lim, y_lim = _get_arm_limits(arm_controller)
                        x_lo, x_hi = x_lim[0], x_lim[1]
                        y_lo, y_hi = y_lim[0], y_lim[1]
                        if SWAP_PAN_TILT:
                            cmd1 = float(np.clip(tilt_deg, y_lo, y_hi))
                            cmd2 = float(np.clip(pan_deg, x_lo, x_hi))
                        else:
                            cmd1 = float(np.clip(pan_deg, x_lo, x_hi))
                            cmd2 = float(np.clip(tilt_deg, y_lo, y_hi))
                        track_state.continuous_target_pan = cmd1
                        track_state.continuous_target_tilt = cmd2

                        throttle_sec = max(
                            _throttle_sec_for_error(arm_controller, cmd1, cmd2),
                            getattr(arm_controller, "_min_move_interval_sec", 0.02),
                        )
                        current_pan = getattr(arm_controller, "pos_x", 0.0)
                        current_tilt = getattr(arm_controller, "pos_y", 0.0)
                        error_deg = math.hypot(cmd1 - current_pan, cmd2 - current_tilt)
                        if error_deg > CONTINUOUS_P_VERY_FAR_DEG:
                            min_interval = getattr(arm_controller, "_min_move_interval_sec", 0.02)
                            throttle_sec = max(throttle_sec, min(min_interval, 0.01))
                        if use_joystick_style:
                            new_time, _, _ = _l_joystick_style_toward_target(
                                arm_controller,
                                cmd1,
                                cmd2,
                                track_state.last_continuous_arm_move_time,
                                throttle_sec,
                                track_state,
                            )
                        else:
                            new_time, _, _ = _variable_step_toward_target(
                                arm_controller,
                                cmd1,
                                cmd2,
                                track_state.last_continuous_arm_move_time,
                                throttle_sec,
                                track_state,
                            )
                        track_state.last_continuous_arm_move_time = new_time

            now = time.time()
            if ready_to_fire and gaa.SOUND_ON_READY:
                if not prev_ready_to_fire:
                    gaa._play_ready_sound()
                    last_ready_sound_time = now
                elif now - last_ready_sound_time >= gaa.SOUND_READY_INTERVAL:
                    gaa._play_ready_sound()
                    last_ready_sound_time = now
            elif target_det is not None and gaa.SOUND_APPROACH_ON and smoothed_aim_cx is not None:
                d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                interval = gaa.compute_approach_beep_interval(d, radius_px, min_side)
                if interval is not None and (now - last_approach_beep_time) >= interval:
                    gaa._play_ready_sound()
                    last_approach_beep_time = now
            prev_ready_to_fire = ready_to_fire

            guide_h, guide_v = gaa.compute_guide_direction(
                target_det, cx_frame, cy_frame, radius_px, ready_to_fire
            )
            distance_m = None
            if target_det is not None:
                x_t, y_t, w_d, h_d, _ = target_det
                distance_m = gaa.estimate_distance_m(w_d, h_d, w, h, fov_h, fov_v, config.target_size_m)
            gaa.draw_hud(
                frame, cx_frame, cy_frame, radius_px, target_det, ready_to_fire,
                guide_h, guide_v, is_day=gaa.is_daytime(), distance_m=distance_m, all_detections=last_detections
            )
            gaa.draw_hint_keys_normal(frame, w, h)
        elif app_mode == "adjust_center":
            gaa.draw_adjust_center_hud(frame, config, w, h)
        elif app_mode == "settings":
            gaa.draw_settings_overlay(frame, config, settings_selected_field)

        if (
            arm_controller is not None
            and joystick_reader is not None
            and joystick_mapper is not None
            and arm_mode_manager is not None
        ):
            js_state = joystick_reader.read()
            if js_state.sensitivity_cycle_pressed and not prev_sensitivity_cycle_pressed:
                joystick_mapper.cycle_sensitivity_mode()
            prev_sensitivity_cycle_pressed = js_state.sensitivity_cycle_pressed
            if js_state.mode_switch is not None:
                if js_state.mode_switch == gaa.MODE_LOCK and arm_mode_manager.mode == gaa.MODE_LOCK:
                    arm_mode_manager.set_mode(gaa.MODE_MANUAL)
                else:
                    arm_mode_manager.set_mode(js_state.mode_switch)

            if js_state.fire_pressed and arm_mode_manager.mode != gaa.MODE_SAFE:
                now_fire = time.time()
                if now_fire - last_fire_time >= gaa.FIRE_SOUND_INTERVAL:
                    gaa._play_fire_sound()
                    is_firing = True
                    last_fire_time = now_fire
                    fire_cooldown = getattr(app_config, "CAM4_ARM_FIRE_COOLDOWN_SEC", 0.5) if app_config else 0.5
                    if arm_controller is not None and hasattr(arm_controller, "fire"):
                        if now_fire - last_fire_gcode_time >= fire_cooldown:
                            arm_controller.fire()
                            last_fire_gcode_time = now_fire

            if arm_mode_manager.mode == gaa.MODE_MANUAL:
                joystick_mapper.apply(js_state, dt_loop)

        if arm_controller is not None and arm_mode_manager is not None:
            if hasattr(arm_controller, "is_healthy") and not arm_controller.is_healthy:
                arm_mode_manager.set_mode(gaa.MODE_SAFE)

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
        if is_firing and (time.time() - last_fire_time) > gaa.FIRE_FLASH_DURATION:
            is_firing = False

        fps_text = f"FPS: {fps:.1f}"
        (fps_tw, fps_th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(frame, fps_text, (w - fps_tw - 28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if arm_mode_manager is not None:
            mode_label, mode_color = arm_mode_manager.label_and_color()
            cv2.putText(frame, mode_label, (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
        if arm_controller is not None:
            src = arm_track_state.get("source", "bbox")
            arm_src_label = f"ARM: mouse (drag)" if src == "mouse" else "ARM: bbox"
            cv2.putText(frame, arm_src_label, (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        if joystick_mapper is not None:
            joy_label = f"JOY: {joystick_mapper.get_sensitivity_label()}"
            cv2.putText(frame, joy_label, (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if is_firing:
            fire_color = (255, 255, 0)
            cv2.circle(frame, (cx_frame, cy_frame), radius_px + 6, fire_color, 3)
            cv2.circle(frame, (cx_frame, cy_frame), 6, fire_color, -1)

        if arm_controller is not None and arm_tracker is not None and arm_mode_manager is not None:
            is_sim_arm = getattr(arm_controller, "is_simulation_mode", False)
            if is_sim_arm:
                try:
                    sim_px, sim_py = arm_tracker.arm_angles_to_pixel(
                        getattr(arm_controller, "pos_x", 0.0),
                        getattr(arm_controller, "pos_y", 0.0),
                        w, h,
                    )
                    center = (int(round(sim_px)), int(round(sim_py)))
                    radius = max(20, int(min(w, h) * 0.025))
                    cv2.circle(frame, center, radius, (255, 255, 255), 3)
                    cv2.circle(frame, center, 3, (255, 255, 255), -1)
                except Exception:
                    pass

        if gaa.DISPLAY_FULLSCREEN:
            display_w, display_h = screen_w, screen_h
            scale = min(display_w / w, display_h / h)
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            display_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            canvas[:] = (0, 0, 0)
            x_off = (display_w - scaled_w) // 2
            y_off = (display_h - scaled_h) // 2
            canvas[y_off : y_off + scaled_h, x_off : x_off + scaled_w] = display_frame
            display_frame = canvas
        else:
            max_w = min(screen_w, gaa.DISPLAY_MAX_WIDTH)
            max_h = min(screen_h, gaa.DISPLAY_MAX_HEIGHT)
            scale = min(max_w / w, max_h / h, 1.0)
            display_w = int(w * scale)
            display_h = int(h * scale)
            if (display_w, display_h) != (w, h):
                display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame
        try:
            cv2.resizeWindow(gaa.WINDOW_NAME, display_w, display_h)
        except cv2.error:
            pass
        cv2.imshow(gaa.WINDOW_NAME, display_frame)
        key = cv2.waitKey(1)
        if key == -1:
            pass
        elif key in (ord("q"), ord("Q")) and loop_frames >= GRACE_FRAMES:
            break
        elif app_mode == "normal":
            if key in (ord("c"), ord("C")):
                app_mode = "adjust_center"
            elif key in (ord("s"), ord("S")):
                app_mode = "settings"

            if arm_mode_manager is not None:
                if key in (ord("a"), ord("A")):
                    arm_mode_manager.set_mode(gaa.MODE_AUTO)
                elif key in (ord("m"), ord("M")):
                    arm_mode_manager.set_mode(gaa.MODE_MANUAL)
                elif key in (ord("f"), ord("F")):
                    arm_mode_manager.set_mode(gaa.MODE_SAFE)
                elif key in (ord("l"), ord("L")):
                    if arm_controller is not None:
                        arm_track_state["source"] = "bbox"
                        arm_track_state["mouse_drag"] = False
                    else:
                        arm_mode_manager.set_mode(gaa.MODE_LOCK)
            if key in (ord("p"), ord("P")) and arm_controller is not None:
                arm_track_state["source"] = "mouse"
                arm_track_state["mouse_drag"] = False
                track_state.sim_kalman_x = None
                track_state.sim_kalman_P = None

            if key in (32,) and arm_mode_manager is not None and arm_mode_manager.mode != gaa.MODE_SAFE:
                now_fire = time.time()
                if now_fire - last_fire_time >= gaa.FIRE_SOUND_INTERVAL:
                    gaa._play_fire_sound()
                    is_firing = True
                    last_fire_time = now_fire
                    fire_cooldown = getattr(app_config, "CAM4_ARM_FIRE_COOLDOWN_SEC", 0.5) if app_config else 0.5
                    if arm_controller is not None and hasattr(arm_controller, "fire"):
                        if now_fire - last_fire_gcode_time >= fire_cooldown:
                            arm_controller.fire()
                            last_fire_gcode_time = now_fire
        elif app_mode == "adjust_center":
            step = gaa.AIM_CENTER_STEP_PX
            if key in (83, 65363, ord("d")):
                config.move(step, 0)
            elif key in (81, 65361, ord("a")):
                config.move(-step, 0)
            elif key in (82, 65362, ord("w")):
                config.move(0, -step)
            elif key in (84, 65364, ord("s")):
                config.move(0, step)
            elif key in (ord("r"), ord("R")):
                config.reset_aim_center()
            elif key in (13, 10):
                config.save()
                app_mode = "normal"
        elif app_mode == "settings":
            if key in (ord("1"),):
                settings_selected_field = 0
            elif key in (ord("2"),):
                settings_selected_field = 1
            elif key in (ord("3"),):
                settings_selected_field = 2
            elif key in (ord("4"),):
                settings_selected_field = 3
            elif key in (82, 65362):
                idx = settings_selected_field
                step = settings_step[idx]
                _min, _max = settings_min_max[idx]
                if idx == 0:
                    config.effective_range_m = min(_max, config.effective_range_m + step)
                elif idx == 1:
                    config.muzzle_velocity_ms = min(_max, config.muzzle_velocity_ms + step)
                elif idx == 2:
                    config.bullet_weight_g = min(_max, config.bullet_weight_g + step)
                else:
                    config.target_size_m = min(_max, config.target_size_m + step)
            elif key in (84, 65364):
                idx = settings_selected_field
                step = settings_step[idx]
                _min, _max = settings_min_max[idx]
                if idx == 0:
                    config.effective_range_m = max(_min, config.effective_range_m - step)
                elif idx == 1:
                    config.muzzle_velocity_ms = max(_min, config.muzzle_velocity_ms - step)
                elif idx == 2:
                    config.bullet_weight_g = max(_min, config.bullet_weight_g - step)
                else:
                    config.target_size_m = max(_min, config.target_size_m - step)
            elif key in (13, 10):
                config.save()
                app_mode = "normal"

    cam.release()
    cv2.destroyAllWindows()
    print("main.py: stopped.")


if __name__ == "__main__":
    main()
