"""
gun_aim_assist_thermal_template_lock.py

Thermal aim-assist entrypoint:
- ใช้กล้อง thermal เป็นหลัก
- ใช้ YOLO .engine เพื่อหาเป้าเริ่มต้นก่อน lock เท่านั้น
- หลัง lock ใช้ template matching จาก bbox ที่เลือกเท่านั้น
- รองรับ click bbox / drag box / joystick lock-unlock
"""
import os
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import config as app_config
from smart_detection_yolo_only import detect_yolo_full_frame, load_yolo_model

from gun_aim_assist_vector import (
    ACTIVE_CAMERA,
    AIM_CENTER_STEP_PX,
    ArmModeManager,
    Cam4ArmController,
    Cam4ArmTracker,
    CENTER_RADIUS_PX,
    DISPLAY_MAX_HEIGHT,
    DISPLAY_MAX_WIDTH,
    JoystickArmMapper,
    JoystickReader,
    MODE_AUTO,
    MODE_LOCK,
    MODE_MANUAL,
    MODE_SAFE,
    RETICLE_RADIUS_RATIOS,
    ShooterConfig,
    SimCam4ArmController,
    SOUND_APPROACH_ON,
    SOUND_ON_READY,
    SOUND_READY_INTERVAL,
    USE_SHORT_BEEP,
    BEEP_SHORT_DURATION_SEC,
    SOUND_FILE_SHORT,
    _TargetKalman,
    _config_mod,
    _load_px_per_deg,
    _move_arm_to_target_center,
    _parse_sensitivity_mode,
    _play_ready_sound,
    _set_active_camera_for_calibration,
    _VECTOR_CALIB_DIR,
    build_camera_from_config,
    compute_approach_beep_interval,
    draw_adjust_center_hud,
    draw_hint_keys_normal,
    draw_settings_overlay,
    get_screen_size,
    load_grid_json,
    pixel_to_arm_degrees_grid,
)
from detect_thermal_lock_track import (
    compute_guide_direction,
    draw_hud,
    estimate_distance_m,
)

try:
    from ultralytics import YOLO as UltralyticsYOLO
    YOLO_AVAILABLE = True
except Exception:
    UltralyticsYOLO = None
    YOLO_AVAILABLE = False


CAMERA_NAME = "cam3_thermal"
WINDOW_NAME = "Thermal Template Lock Aim Assist"
DISPLAY_FULLSCREEN = True

YOLO_ENGINE = "last_imgsz640.engine"
YOLO_IMGSZ = 640
YOLO_CONF_MIN = 0.10
YOLO_INTERVAL = 2
YOLO_MAX_DET = 10

TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_MIN_SCORE = 0.50
TM_MIN_SCORE_STRICT = 0.65
MAX_JUMP_RATIO = 1.2
SEARCH_MARGIN = 180
LOCK_LOST_MAX_FRAMES = 6
LOCK_DRIVE_PREDICT_SEC = 0.10
LOCK_MOVE_THROTTLE_SEC = 0.02

DRAG_MIN_SIZE = 5
CLICK_MAX_MOVE = 8

KEY_LOCK = "L"
KEY_UNLOCK = "U"
KEY_ADJUST_CENTER = "C"
KEY_SETTINGS = "S"
KEY_QUIT = "Q"


def load_yolo_engine_only() -> Tuple[Optional[Any], int]:
    engine_path = os.path.join(os.path.dirname(__file__), YOLO_ENGINE)
    if YOLO_AVAILABLE and os.path.exists(engine_path):
        try:
            model = UltralyticsYOLO(engine_path, task="detect")
            dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False, device=0, imgsz=YOLO_IMGSZ)
            print(f"Thermal Template Lock: YOLO {YOLO_IMGSZ} loaded (engine)")
            return model, YOLO_IMGSZ
        except Exception as exc:
            print(f"Thermal Template Lock: YOLO engine load failed: {exc}")

    fallback = load_yolo_model()
    if isinstance(fallback, tuple):
        model, imgsz = fallback
    else:
        model, imgsz = fallback, None
    if model is not None:
        print(f"Thermal Template Lock: YOLO loaded via fallback (imgsz={imgsz or YOLO_IMGSZ})")
        return model, imgsz or YOLO_IMGSZ

    print("Thermal Template Lock: YOLO unavailable; manual drag lock only")
    return None, YOLO_IMGSZ


def init_runtime(camera_name: str) -> Dict[str, Any]:
    return {
        "camera_name": camera_name,
        "frame_w": 1,
        "frame_h": 1,
        "display_w": 1,
        "display_h": 1,
        "scaled_w": 1,
        "scaled_h": 1,
        "x_off": 0,
        "y_off": 0,
        "curr_gray": None,
        "drawing": False,
        "drag_start": None,
        "drag_end": None,
        "last_detections": [],
        "pending_lock_bbox": None,
        "locked": False,
        "lock_lost": False,
        "lock_fail_count": 0,
        "locked_bbox": None,
        "stored_template": None,
        "stored_tw": 0,
        "stored_th": 0,
        "lock_score": 0.0,
        "lock_kalman": _TargetKalman(),
        "last_lock_move_time": 0.0,
        "app_mode": "normal",
        "settings_selected_field": 0,
        "settings_step": (10, 50, 0.5, 0.05),
        "settings_min_max": ((10, 500), (200, 1500), (1, 50), (0.1, 2.0)),
        "last_ready_sound_time": 0.0,
        "last_approach_beep_time": 0.0,
        "prev_ready_to_fire": False,
        "ready_to_fire": False,
        "yolo_model": None,
        "yolo_imgsz": YOLO_IMGSZ,
        "yolo_frame_q": None,
        "yolo_result_q": None,
        "yolo_thread": None,
        "arm_controller": None,
        "arm_tracker": None,
        "arm_mode_manager": None,
        "joystick_reader": None,
        "joystick_mapper": None,
        "prev_sensitivity_cycle_pressed": False,
        "grid_data": None,
        "px_per_deg_x": None,
        "px_per_deg_y": None,
        "x_lo": None,
        "x_hi": None,
        "y_lo": None,
        "y_hi": None,
        "screen_w": DISPLAY_MAX_WIDTH,
        "screen_h": DISPLAY_MAX_HEIGHT,
        "cam_config": None,
        "config": None,
    }


def start_yolo_worker(runtime: Dict[str, Any]) -> None:
    frame_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=1)
    result_q: "queue.Queue[Tuple[list, float]]" = queue.Queue(maxsize=2)
    model = runtime["yolo_model"]
    imgsz = runtime["yolo_imgsz"]

    def _worker() -> None:
        while True:
            item = frame_q.get()
            if item is None:
                break
            frame, t_sent = item
            dets = detect_yolo_full_frame(
                model,
                frame,
                YOLO_CONF_MIN,
                imgsz=imgsz,
                max_det=YOLO_MAX_DET,
            )
            try:
                result_q.put_nowait((dets, t_sent))
            except queue.Full:
                try:
                    result_q.get_nowait()
                except queue.Empty:
                    pass
                result_q.put_nowait((dets, t_sent))

    runtime["yolo_frame_q"] = frame_q
    runtime["yolo_result_q"] = result_q
    runtime["yolo_thread"] = threading.Thread(target=_worker, daemon=True)
    runtime["yolo_thread"].start()


def stop_yolo_worker(runtime: Dict[str, Any]) -> None:
    frame_q = runtime.get("yolo_frame_q")
    worker = runtime.get("yolo_thread")
    if frame_q is not None:
        try:
            frame_q.put_nowait(None)
        except queue.Full:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_q.put_nowait(None)
            except queue.Full:
                pass
    if worker is not None and worker.is_alive():
        worker.join(timeout=1.0)


def update_display_mapping(runtime: Dict[str, Any], frame_w: int, frame_h: int) -> None:
    runtime["frame_w"] = frame_w
    runtime["frame_h"] = frame_h
    screen_w = runtime["screen_w"]
    screen_h = runtime["screen_h"]

    if DISPLAY_FULLSCREEN:
        scale = min(screen_w / frame_w, screen_h / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        runtime["display_w"] = screen_w
        runtime["display_h"] = screen_h
        runtime["scaled_w"] = scaled_w
        runtime["scaled_h"] = scaled_h
        runtime["x_off"] = (screen_w - scaled_w) // 2
        runtime["y_off"] = (screen_h - scaled_h) // 2
    else:
        display_w = int(min(screen_w, DISPLAY_MAX_WIDTH, frame_w))
        display_h = int(min(screen_h, DISPLAY_MAX_HEIGHT, frame_h))
        if display_w <= 0 or display_h <= 0:
            display_w, display_h = frame_w, frame_h
        scale = min(display_w / frame_w, display_h / frame_h, 1.0)
        runtime["display_w"] = int(frame_w * scale)
        runtime["display_h"] = int(frame_h * scale)
        runtime["scaled_w"] = runtime["display_w"]
        runtime["scaled_h"] = runtime["display_h"]
        runtime["x_off"] = 0
        runtime["y_off"] = 0


def screen_to_frame(x: int, y: int, runtime: Dict[str, Any]) -> Tuple[int, int]:
    fw = max(1, int(runtime["frame_w"]))
    fh = max(1, int(runtime["frame_h"]))
    sw = max(1, int(runtime["scaled_w"]))
    sh = max(1, int(runtime["scaled_h"]))
    x_off = int(runtime["x_off"])
    y_off = int(runtime["y_off"])
    fx = (x - x_off) * fw / sw
    fy = (y - y_off) * fh / sh
    fx = max(0.0, min(fw - 1.0, fx))
    fy = max(0.0, min(fh - 1.0, fy))
    return int(fx), int(fy)


def bbox_to_det(bbox: Optional[Tuple[int, int, int, int]], conf: float = 1.0):
    if bbox is None:
        return None
    x, y, w, h = bbox
    return (int(x), int(y), int(w), int(h), float(conf))


def select_best_detection_for_lock(detections, cx_frame: int, cy_frame: int):
    if not detections:
        return None
    return min(
        detections,
        key=lambda det: ((det[0] + det[2] * 0.5 - cx_frame) ** 2 + (det[1] + det[3] * 0.5 - cy_frame) ** 2, -det[4]),
    )


def clear_lock_state(runtime: Dict[str, Any]) -> None:
    runtime["locked"] = False
    runtime["lock_lost"] = False
    runtime["lock_fail_count"] = 0
    runtime["pending_lock_bbox"] = None
    runtime["locked_bbox"] = None
    runtime["stored_template"] = None
    runtime["stored_tw"] = 0
    runtime["stored_th"] = 0
    runtime["lock_score"] = 0.0
    lock_kalman = runtime.get("lock_kalman")
    if lock_kalman is not None:
        lock_kalman.reset()


def lock_from_bbox(gray: np.ndarray, bbox: Tuple[int, int, int, int], runtime: Dict[str, Any]) -> bool:
    x, y, w, h = bbox
    fh, fw = gray.shape[:2]
    if w < 2 or h < 2:
        return False
    x = max(0, int(x))
    y = max(0, int(y))
    x2 = min(fw, x + int(w))
    y2 = min(fh, y + int(h))
    if x2 <= x or y2 <= y:
        return False
    template = gray[y:y2, x:x2].copy()
    if template.size == 0:
        return False

    runtime["stored_template"] = template
    runtime["stored_tw"] = int(template.shape[1])
    runtime["stored_th"] = int(template.shape[0])
    runtime["locked_bbox"] = (x, y, runtime["stored_tw"], runtime["stored_th"])
    runtime["locked"] = True
    runtime["lock_lost"] = False
    runtime["lock_fail_count"] = 0
    runtime["lock_score"] = 1.0
    runtime["pending_lock_bbox"] = runtime["locked_bbox"]
    lock_kalman = runtime.get("lock_kalman")
    if lock_kalman is not None:
        lock_kalman.reset()
    return True


def compute_template_search_roi(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    tw: int,
    th: int,
) -> Tuple[int, int, int, int]:
    px, py, pw, ph = bbox
    margin = SEARCH_MARGIN
    rx1 = max(0, px - margin)
    ry1 = max(0, py - margin)
    rx2 = min(frame_w, px + pw + margin)
    ry2 = min(frame_h, py + ph + margin)
    if rx2 - rx1 < tw or ry2 - ry1 < th:
        rx1 = max(0, px - max(margin, tw))
        ry1 = max(0, py - max(margin, th))
        rx2 = min(frame_w, px + pw + max(margin, tw))
        ry2 = min(frame_h, py + ph + max(margin, th))
    return rx1, ry1, rx2, ry2


def match_locked_template(curr_gray: np.ndarray, runtime: Dict[str, Any]):
    template = runtime.get("stored_template")
    locked_bbox = runtime.get("locked_bbox")
    if template is None or locked_bbox is None:
        return False, None, 0.0

    tw = int(runtime["stored_tw"])
    th = int(runtime["stored_th"])
    fh, fw = curr_gray.shape[:2]
    rx1, ry1, rx2, ry2 = compute_template_search_roi(locked_bbox, fw, fh, tw, th)
    roi = curr_gray[ry1:ry2, rx1:rx2]
    if roi.shape[1] < tw or roi.shape[0] < th:
        return False, None, 0.0

    res = cv2.matchTemplate(roi, template, TM_METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < TM_MIN_SCORE:
        return False, None, float(max_val)

    nx = rx1 + int(max_loc[0])
    ny = ry1 + int(max_loc[1])
    px, py, pw, ph = locked_bbox
    dist = float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)
    template_diag = max(1.0, float((pw * pw + ph * ph) ** 0.5))
    if dist > MAX_JUMP_RATIO * template_diag and max_val < TM_MIN_SCORE_STRICT:
        return False, None, float(max_val)

    return True, (nx, ny, tw, th), float(max_val)


def update_template_lock(curr_gray: np.ndarray, runtime: Dict[str, Any]) -> None:
    if not runtime["locked"] or runtime["stored_template"] is None or runtime["locked_bbox"] is None:
        return

    ok, bbox, score = match_locked_template(curr_gray, runtime)
    runtime["lock_score"] = float(score)
    if ok and bbox is not None:
        runtime["locked_bbox"] = bbox
        runtime["lock_fail_count"] = 0
        runtime["lock_lost"] = False
    else:
        runtime["lock_fail_count"] += 1
        if runtime["lock_fail_count"] >= LOCK_LOST_MAX_FRAMES:
            runtime["lock_lost"] = True


def update_lock_kalman(runtime: Dict[str, Any]) -> None:
    bbox = runtime.get("locked_bbox")
    lock_kalman = runtime.get("lock_kalman")
    if bbox is None or lock_kalman is None or runtime.get("lock_lost"):
        return
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.5
    lock_kalman.update(cx, cy, 1.0)


def get_locked_target_point(runtime: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bbox = runtime.get("locked_bbox")
    if bbox is None or runtime.get("lock_lost"):
        return None
    lock_kalman = runtime.get("lock_kalman")
    if lock_kalman is not None and getattr(lock_kalman, "_initialized", False):
        return lock_kalman.predict_ahead(LOCK_DRIVE_PREDICT_SEC)
    x, y, w, h = bbox
    return (x + w * 0.5, y + h * 0.5)


def handle_lock_request(runtime: Dict[str, Any], recenter_only: bool = False) -> bool:
    config = runtime["config"]
    fw = runtime["frame_w"]
    fh = runtime["frame_h"]
    cx_frame, cy_frame = config.get_center(fw, fh)
    gray = runtime.get("curr_gray")

    if runtime.get("locked") and runtime.get("locked_bbox") is not None:
        runtime["lock_lost"] = False
        runtime["lock_fail_count"] = 0
        runtime["pending_lock_bbox"] = runtime["locked_bbox"]
        return True

    if recenter_only or gray is None:
        return False

    det = select_best_detection_for_lock(runtime.get("last_detections", []), cx_frame, cy_frame)
    if det is None:
        return False

    x, y, w, h, _ = det
    ok = lock_from_bbox(gray, (int(x), int(y), int(w), int(h)), runtime)
    if ok and runtime.get("arm_mode_manager") is not None:
        runtime["arm_mode_manager"].set_mode(MODE_LOCK)
    return ok


def on_mouse_lock(event, x, y, _flags, runtime: Dict[str, Any]):
    if runtime.get("app_mode") != "normal":
        return

    fx, fy = screen_to_frame(x, y, runtime)
    gray = runtime.get("curr_gray")
    dets = runtime.get("last_detections", [])

    if event == cv2.EVENT_LBUTTONDOWN:
        runtime["drawing"] = True
        runtime["drag_start"] = (fx, fy)
        runtime["drag_end"] = (fx, fy)
        return

    if event == cv2.EVENT_MOUSEMOVE and runtime.get("drawing"):
        runtime["drag_end"] = (fx, fy)
        return

    if event != cv2.EVENT_LBUTTONUP:
        return

    if not runtime.get("drawing") or runtime.get("drag_start") is None or gray is None:
        runtime["drawing"] = False
        runtime["drag_start"] = None
        runtime["drag_end"] = None
        return

    x1, y1 = runtime["drag_start"]
    x2, y2 = runtime["drag_end"] if runtime["drag_end"] is not None else (fx, fy)
    runtime["drawing"] = False
    runtime["drag_start"] = None
    runtime["drag_end"] = None

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx <= CLICK_MAX_MOVE and dy <= CLICK_MAX_MOVE and dets:
        for det in dets:
            bx, by, bw, bh, _ = det
            if bx <= x2 <= bx + bw and by <= y2 <= by + bh:
                if lock_from_bbox(gray, (int(bx), int(by), int(bw), int(bh)), runtime):
                    if runtime.get("arm_mode_manager") is not None:
                        runtime["arm_mode_manager"].set_mode(MODE_LOCK)
                return
        return

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w < DRAG_MIN_SIZE or h < DRAG_MIN_SIZE:
        return
    if lock_from_bbox(gray, (x_min, y_min, w, h), runtime):
        if runtime.get("arm_mode_manager") is not None:
            runtime["arm_mode_manager"].set_mode(MODE_LOCK)


def tick_manual(runtime: Dict[str, Any], js_state: Any, dt_loop: float) -> None:
    mapper = runtime.get("joystick_mapper")
    if mapper is not None and js_state is not None:
        mapper.apply(js_state, dt_loop)


def tick_lock(runtime: Dict[str, Any]) -> None:
    arm_controller = runtime.get("arm_controller")
    bbox = runtime.get("pending_lock_bbox")
    config = runtime.get("config")
    if arm_controller is None or bbox is None or config is None:
        return
    now = time.time()
    if now - runtime.get("last_lock_move_time", 0.0) < LOCK_MOVE_THROTTLE_SEC:
        return

    fw = runtime["frame_w"]
    fh = runtime["frame_h"]
    cx_frame, cy_frame = config.get_center(fw, fh)
    x, y, w_box, h_box = bbox
    target_px = (x + w_box * 0.5, y + h_box * 0.5)

    moved = _move_arm_to_target_center(
        arm_controller,
        bbox_to_det(bbox, runtime.get("lock_score", 1.0)),
        cx_frame,
        cy_frame,
        fw,
        fh,
        runtime.get("px_per_deg_x"),
        runtime.get("px_per_deg_y"),
        runtime.get("x_lo"),
        runtime.get("x_hi"),
        runtime.get("y_lo"),
        runtime.get("y_hi"),
        runtime.get("grid_data"),
        pixel_to_arm_degrees_grid,
        max_dist_px=None,
        target_px=target_px,
    )
    if moved:
        runtime["last_lock_move_time"] = now
        runtime["pending_lock_bbox"] = None


def handle_joystick(runtime: Dict[str, Any], js_state: Any, dt_loop: float) -> None:
    arm_mode_manager = runtime.get("arm_mode_manager")
    if js_state is None or arm_mode_manager is None:
        return

    prev_cycle = runtime.get("prev_sensitivity_cycle_pressed", False)
    cur_cycle = bool(getattr(js_state, "sensitivity_cycle_pressed", False))
    if cur_cycle and not prev_cycle and runtime.get("joystick_mapper") is not None:
        runtime["joystick_mapper"].cycle_sensitivity_mode()
    runtime["prev_sensitivity_cycle_pressed"] = cur_cycle

    if bool(getattr(js_state, "unlock_pressed", False)):
        clear_lock_state(runtime)
        arm_mode_manager.set_mode(MODE_MANUAL)
        return

    mode_switch = getattr(js_state, "mode_switch", None)
    if mode_switch == MODE_LOCK:
        if not runtime.get("locked"):
            handle_lock_request(runtime, recenter_only=False)
        else:
            arm_mode_manager.set_mode(MODE_LOCK)
            handle_lock_request(runtime, recenter_only=True)
        return

    if mode_switch == MODE_AUTO:
        arm_mode_manager.set_mode(MODE_MANUAL)
        return

    if mode_switch in (MODE_MANUAL, MODE_SAFE):
        clear_lock_state(runtime)
        arm_mode_manager.set_mode(mode_switch)
        return

    if arm_mode_manager.mode == MODE_MANUAL:
        tick_manual(runtime, js_state, dt_loop)


def draw_detection_candidates(frame: np.ndarray, detections) -> None:
    for det in detections:
        x, y, w, h, conf = det
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        cv2.putText(frame, f"{conf:.2f}", (int(x), int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def draw_drag_preview(frame: np.ndarray, runtime: Dict[str, Any]) -> None:
    if runtime.get("drawing") and runtime.get("drag_start") is not None and runtime.get("drag_end") is not None:
        cv2.rectangle(frame, runtime["drag_start"], runtime["drag_end"], (255, 255, 255), 2)


def draw_lock_status(frame: np.ndarray, runtime: Dict[str, Any]) -> None:
    status = "LOCK LOST" if runtime.get("lock_lost") else ("LOCKED" if runtime.get("locked") else "SEARCH")
    color = (0, 128, 255) if runtime.get("lock_lost") else ((0, 255, 255) if runtime.get("locked") else (255, 255, 255))
    cv2.putText(frame, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if runtime.get("locked_bbox") is not None:
        x, y, w, h = runtime["locked_bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    help_text = f"[{KEY_LOCK}] Lock/Recenter  [{KEY_UNLOCK}] Unlock"
    cv2.putText(frame, help_text, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)


def compose_display_frame(frame: np.ndarray, runtime: Dict[str, Any]) -> np.ndarray:
    display_w = runtime["display_w"]
    display_h = runtime["display_h"]
    scaled_w = runtime["scaled_w"]
    scaled_h = runtime["scaled_h"]
    x_off = runtime["x_off"]
    y_off = runtime["y_off"]

    if DISPLAY_FULLSCREEN:
        fitted = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        canvas[y_off : y_off + scaled_h, x_off : x_off + scaled_w] = fitted
        return canvas

    if (display_w, display_h) != frame.shape[1::-1]:
        return cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    return frame


def run_yolo_acquisition(frame: np.ndarray, runtime: Dict[str, Any], frame_counter: int) -> None:
    if runtime.get("locked"):
        runtime["last_detections"] = []
        return

    frame_q = runtime.get("yolo_frame_q")
    result_q = runtime.get("yolo_result_q")
    if frame_q is None or result_q is None or runtime.get("yolo_model") is None:
        return

    if frame_counter % YOLO_INTERVAL == 0:
        try:
            frame_q.put_nowait((frame.copy(), time.time()))
        except queue.Full:
            pass

    try:
        dets, _t_sent = result_q.get_nowait()
        runtime["last_detections"] = dets
    except queue.Empty:
        pass


def normalize_frame_channels(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def init_arm_and_joystick(runtime: Dict[str, Any]) -> None:
    camera_name = runtime["camera_name"]
    arm_controller = None
    arm_tracker = None

    if Cam4ArmController is not None and Cam4ArmTracker is not None and getattr(app_config, "CAM4_ARM_ENABLED", False):
        try:
            if getattr(app_config, "CAM4_ARM_SIMULATION_MODE", False) and SimCam4ArmController is not None:
                arm_controller = SimCam4ArmController()
            else:
                arm_controller = Cam4ArmController()
            if arm_controller.connect():
                arm_tracker = Cam4ArmTracker(arm_controller, camera_name=camera_name)
            else:
                arm_controller = None
        except Exception as exc:
            print(f"Thermal Template Lock: arm init failed: {exc}")
            arm_controller = None

    runtime["arm_controller"] = arm_controller
    runtime["arm_tracker"] = arm_tracker

    if ArmModeManager is not None:
        runtime["arm_mode_manager"] = ArmModeManager(has_arm=arm_controller is not None)
        runtime["arm_mode_manager"].set_mode(MODE_MANUAL)

    if JoystickReader is not None and JoystickArmMapper is not None and arm_controller is not None:
        try:
            joystick_reader = JoystickReader()
            if getattr(joystick_reader, "enabled", False):
                max_pan = getattr(app_config, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0)
                max_tilt = getattr(app_config, "CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0)
                deadzone = getattr(app_config, "CAM4_ARM_JOYSTICK_DEADZONE", 0.05)
                stick_exp = getattr(app_config, "CAM4_ARM_JOYSTICK_STICK_EXPONENT", 2.0)
                scale_min = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MIN", 0.1)
                scale_max = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MAX", 10.0)
                speed_smooth = getattr(app_config, "CAM4_ARM_JOYSTICK_SPEED_SMOOTH_ALPHA", 0.82)
                sens_str = getattr(app_config, "CAM4_ARM_JOYSTICK_SENSITIVITY", "medium")
                runtime["joystick_reader"] = joystick_reader
                runtime["joystick_mapper"] = JoystickArmMapper(
                    arm_controller,
                    max_pan_rate_deg=max_pan,
                    max_tilt_rate_deg=max_tilt,
                    deadzone=deadzone,
                    stick_exponent=stick_exp,
                    speed_axis_scale_min=scale_min,
                    speed_axis_scale_max=scale_max,
                    speed_smooth_alpha=speed_smooth,
                    initial_sensitivity_mode=_parse_sensitivity_mode(sens_str),
                )
        except Exception as exc:
            print(f"Thermal Template Lock: joystick init failed: {exc}")
            runtime["joystick_reader"] = None
            runtime["joystick_mapper"] = None

    if arm_controller is not None:
        runtime["px_per_deg_x"], runtime["px_per_deg_y"] = _load_px_per_deg()
        if load_grid_json is not None and pixel_to_arm_degrees_grid is not None:
            grid_path = _VECTOR_CALIB_DIR / f"{camera_name}_mouse_grid_lookup.json"
            if grid_path.is_file():
                runtime["grid_data"] = load_grid_json(grid_path)
        margin = getattr(_config_mod, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if _config_mod else 2.0
        xr = getattr(_config_mod, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if _config_mod else (-65.0, 65.0)
        yr = getattr(_config_mod, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if _config_mod else (-35.0, 35.0)
        x_lim = getattr(arm_controller, "_effective_x_limits", None) or (xr[0] + margin, xr[1] - margin)
        y_lim = getattr(arm_controller, "_effective_y_limits", None) or (yr[0] + margin, yr[1] - margin)
        runtime["x_lo"], runtime["x_hi"] = x_lim[0], x_lim[1]
        runtime["y_lo"], runtime["y_hi"] = y_lim[0], y_lim[1]


def update_ready_and_sound(runtime: Dict[str, Any], target_center, radius_px: int) -> None:
    config = runtime["config"]
    fw = runtime["frame_w"]
    fh = runtime["frame_h"]
    cx_frame, cy_frame = config.get_center(fw, fh)
    ready_to_fire = False
    if target_center is not None:
        d = float(((target_center[0] - cx_frame) ** 2 + (target_center[1] - cy_frame) ** 2) ** 0.5)
        ready_to_fire = d <= radius_px
    runtime["ready_to_fire"] = ready_to_fire

    now = time.time()
    if ready_to_fire and SOUND_ON_READY:
        if not runtime["prev_ready_to_fire"] or now - runtime["last_ready_sound_time"] >= SOUND_READY_INTERVAL:
            _play_ready_sound()
            runtime["last_ready_sound_time"] = now
    elif target_center is not None and SOUND_APPROACH_ON:
        min_side = min(fw, fh)
        d = float(((target_center[0] - cx_frame) ** 2 + (target_center[1] - cy_frame) ** 2) ** 0.5)
        interval = compute_approach_beep_interval(d, radius_px, min_side)
        if interval is not None and now - runtime["last_approach_beep_time"] >= interval:
            _play_ready_sound()
            runtime["last_approach_beep_time"] = now

    runtime["prev_ready_to_fire"] = ready_to_fire


def handle_keyboard(runtime: Dict[str, Any], key: int) -> bool:
    config = runtime["config"]
    app_mode = runtime["app_mode"]

    if key in (ord(KEY_QUIT.lower()), ord(KEY_QUIT.upper())):
        return False

    if app_mode == "normal":
        if key in (ord(KEY_ADJUST_CENTER.lower()), ord(KEY_ADJUST_CENTER.upper())):
            runtime["app_mode"] = "adjust_center"
        elif key in (ord(KEY_SETTINGS.lower()), ord(KEY_SETTINGS.upper())):
            runtime["app_mode"] = "settings"
        elif key in (ord(KEY_UNLOCK.lower()), ord(KEY_UNLOCK.upper())):
            clear_lock_state(runtime)
            if runtime.get("arm_mode_manager") is not None:
                runtime["arm_mode_manager"].set_mode(MODE_MANUAL)
        elif key in (ord(KEY_LOCK.lower()), ord(KEY_LOCK.upper())):
            if not handle_lock_request(runtime, recenter_only=runtime.get("locked", False)):
                print("Thermal Template Lock: no target to lock")
        return True

    if app_mode == "adjust_center":
        step = AIM_CENTER_STEP_PX
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
            runtime["app_mode"] = "normal"
        elif key == 27:
            runtime["app_mode"] = "normal"
        return True

    if app_mode == "settings":
        idx = runtime["settings_selected_field"]
        step, bounds = runtime["settings_step"], runtime["settings_min_max"]
        if key == ord("1"):
            runtime["settings_selected_field"] = 0
        elif key == ord("2"):
            runtime["settings_selected_field"] = 1
        elif key == ord("3"):
            runtime["settings_selected_field"] = 2
        elif key == ord("4"):
            runtime["settings_selected_field"] = 3
        elif key in (82, 65362):
            inc = step[idx]
            min_v, max_v = bounds[idx]
            if idx == 0:
                config.effective_range_m = min(max_v, config.effective_range_m + inc)
            elif idx == 1:
                config.muzzle_velocity_ms = min(max_v, config.muzzle_velocity_ms + inc)
            elif idx == 2:
                config.bullet_weight_g = min(max_v, config.bullet_weight_g + inc)
            else:
                config.target_size_m = min(max_v, config.target_size_m + inc)
        elif key in (84, 65364):
            dec = step[idx]
            min_v, max_v = bounds[idx]
            if idx == 0:
                config.effective_range_m = max(min_v, config.effective_range_m - dec)
            elif idx == 1:
                config.muzzle_velocity_ms = max(min_v, config.muzzle_velocity_ms - dec)
            elif idx == 2:
                config.bullet_weight_g = max(min_v, config.bullet_weight_g - dec)
            else:
                config.target_size_m = max(min_v, config.target_size_m - dec)
        elif key in (13, 10):
            config.save()
            runtime["app_mode"] = "normal"
        elif key == 27:
            runtime["app_mode"] = "normal"
        return True

    return True


def main():
    camera_name = CAMERA_NAME if CAMERA_NAME is not None else ACTIVE_CAMERA
    _set_active_camera_for_calibration(camera_name)
    runtime = init_runtime(camera_name)
    runtime["screen_w"], runtime["screen_h"] = get_screen_size()
    runtime["config"] = ShooterConfig(camera_name)
    runtime["config"].load()

    cam = build_camera_from_config(camera_name)
    runtime["cam_config"] = app_config.get_camera_config(camera_name)
    cam.start()
    runtime["cam"] = cam

    runtime["yolo_model"], runtime["yolo_imgsz"] = load_yolo_engine_only()
    if runtime["yolo_model"] is not None:
        start_yolo_worker(runtime)

    init_arm_and_joystick(runtime)

    if USE_SHORT_BEEP:
        beep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SOUND_FILE_SHORT)
        if not os.path.isfile(beep_path):
            from gun_aim_assist_vector import create_short_beep_wav
            create_short_beep_wav(beep_path, duration_sec=BEEP_SHORT_DURATION_SEC)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_lock, runtime)

    print("Thermal Template Lock: L=lock/recenter U=unlock C=aim center S=ballistics Q=quit")
    print("Mouse: click red bbox to lock, or drag box to lock manually")

    last_loop_time = time.time()
    frame_counter = 0

    try:
        while True:
            active, frame, _frame_ts = cam.read()
            if not active or frame is None:
                time.sleep(0.01)
                continue

            frame = normalize_frame_channels(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            runtime["curr_gray"] = curr_gray
            fh, fw = frame.shape[:2]
            update_display_mapping(runtime, fw, fh)

            now = time.time()
            dt_loop = max(0.0, now - last_loop_time)
            last_loop_time = now
            frame_counter += 1

            js_state = None
            if runtime.get("joystick_reader") is not None and getattr(runtime["joystick_reader"], "enabled", False):
                js_state = runtime["joystick_reader"].read()

            if runtime["app_mode"] == "normal":
                handle_joystick(runtime, js_state, dt_loop)
                run_yolo_acquisition(frame, runtime, frame_counter)
                if runtime.get("locked"):
                    update_template_lock(curr_gray, runtime)
                    update_lock_kalman(runtime)

                arm_mode_manager = runtime.get("arm_mode_manager")
                if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK:
                    tick_lock(runtime)
                elif arm_mode_manager is not None and arm_mode_manager.mode == MODE_MANUAL:
                    tick_manual(runtime, js_state, dt_loop)

            display = frame.copy()
            if not runtime.get("locked"):
                draw_detection_candidates(display, runtime.get("last_detections", []))
            draw_drag_preview(display, runtime)
            draw_lock_status(display, runtime)

            config = runtime["config"]
            cx_frame, cy_frame = config.get_center(fw, fh)
            min_side = min(fw, fh)
            radius_px = int(min_side * RETICLE_RADIUS_RATIOS[0])
            if CENTER_RADIUS_PX > 0:
                radius_px = CENTER_RADIUS_PX
            radius_px = max(10, radius_px)

            target_bbox = runtime.get("locked_bbox")
            target_center = None
            if target_bbox is not None and not runtime.get("lock_lost"):
                x, y, w_box, h_box = target_bbox
                target_center = (x + w_box / 2.0, y + h_box / 2.0)
            update_ready_and_sound(runtime, target_center, radius_px)
            guide_h, guide_v = compute_guide_direction(target_center, cx_frame, cy_frame, radius_px, runtime["ready_to_fire"])
            distance_m = None
            if target_bbox is not None:
                bx, by, bw, bh = target_bbox
                cam_cfg = runtime["cam_config"]
                distance_m = estimate_distance_m(
                    bw,
                    bh,
                    fw,
                    fh,
                    cam_cfg.get("fov_horizontal", 60.0),
                    cam_cfg.get("fov_vertical", 36.0),
                    config.target_size_m,
                )
            draw_hud(
                display,
                cx_frame,
                cy_frame,
                radius_px,
                target_bbox if not runtime.get("lock_lost") else None,
                target_center if not runtime.get("lock_lost") else None,
                runtime["ready_to_fire"],
                guide_h=guide_h,
                guide_v=guide_v,
                distance_m=distance_m,
                all_detections=(runtime.get("last_detections") if not runtime.get("locked") else None),
            )

            if runtime["app_mode"] == "normal":
                draw_hint_keys_normal(display, fw, fh)
            elif runtime["app_mode"] == "adjust_center":
                draw_adjust_center_hud(display, config, fw, fh)
            elif runtime["app_mode"] == "settings":
                draw_settings_overlay(display, config, runtime["settings_selected_field"])

            display_frame = compose_display_frame(display, runtime)
            try:
                cv2.resizeWindow(WINDOW_NAME, runtime["display_w"], runtime["display_h"])
            except cv2.error:
                pass
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if not handle_keyboard(runtime, key):
                break
    finally:
        stop_yolo_worker(runtime)
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

