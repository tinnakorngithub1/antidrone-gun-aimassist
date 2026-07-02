"""
Gun Aim Assist: ระบบช่วยเล็งกล้องติดปืน
- รับภาพจากกล้อง (ใช้ config ของโปรเจกต์)
- ตรวจจับเป้า (ดรอน/วัตถุ) ด้วย YOLO
- ถ้าศูนย์กลางเป้าอยู่ในวงรัศมีจากศูนย์กลางจอ = พร้อมยิง (ready to fire)
- แสดง HUD: crosshair, วงยิงได้, bbox เป้า, READY/AIM
"""
import cv2
import datetime
import json
import math
import numpy as np
import os
import struct
import time
import wave
from collections import deque

try:
    from fast_motion_sky import CameraStream
    from config import get_camera_config, ACTIVE_CAMERA
except ImportError as e:
    print(f"Error: {e}")
    raise

# Optional: Arm controller + tracker for cam4 (จะใช้เฉพาะเมื่อ config.CAM4_ARM_ENABLED = True)
try:
    from cam4_arm_controller import Cam4ArmController, SimCam4ArmController
    from cam4_arm_tracker import Cam4ArmTracker
except ImportError:
    Cam4ArmController = None
    SimCam4ArmController = None
    Cam4ArmTracker = None

# Optional: Joystick + arm mode manager (สำหรับ manual control)
try:
    from joystick_cam4_controller import (
        ArmModeManager,
        JoystickArmMapper,
        JoystickReader,
        MODE_AUTO,
        MODE_MANUAL,
        MODE_SAFE,
        MODE_LOCK,
        _parse_sensitivity_mode,
    )
except ImportError:
    ArmModeManager = None
    JoystickArmMapper = None
    JoystickReader = None
    MODE_AUTO = 0
    MODE_MANUAL = 1
    MODE_SAFE = 2
    MODE_LOCK = 3

from smart_detection_yolo_only import load_yolo_model, detect_yolo_full_frame

try:
    from ultralytics import YOLO as UltralyticsYOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    UltralyticsYOLO = None

# Sound (optional)
try:
    import pygame
    pygame.mixer.init()
    PYGAME_SOUND_AVAILABLE = True
except Exception:
    PYGAME_SOUND_AVAILABLE = False
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

_sound_ready = None
_sound_fire = None

def create_short_beep_wav(filepath, duration_sec=0.06, sample_rate=44100, freq=880):
    """สร้างไฟล์ WAV เสียงบีปสั้น (ความถี่ freq Hz, ยาว duration_sec วินาที) — ใช้ครั้งเดียวตอนสตาร์ท"""
    n_samples = int(sample_rate * duration_sec)
    buf = []
    for i in range(n_samples):
        t = i / sample_rate
        fade = 1.0
        if i < sample_rate * 0.005:
            fade = i / (sample_rate * 0.005)
        elif i > n_samples - sample_rate * 0.005:
            fade = (n_samples - i) / (sample_rate * 0.005)
        val = 0.3 * math.sin(2 * math.pi * freq * t) * fade
        buf.append(struct.pack("h", int(32767 * max(-1, min(1, val)))))
    try:
        with wave.open(filepath, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"".join(buf))
    except Exception:
        pass


def _play_ready_sound():
    """เล่นเสียงครั้งเดียวเมื่อเล็งตรง (transition)."""
    global _sound_ready
    if not SOUND_ON_READY:
        return
    filename = SOUND_FILE_SHORT if USE_SHORT_BEEP else SOUND_FILE
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.isfile(path):
        return
    try:
        if PYGAME_SOUND_AVAILABLE:
            if _sound_ready is None:
                _sound_ready = pygame.mixer.Sound(path)
            _sound_ready.play()
        elif PLAYSOUND_AVAILABLE:
            playsound(path, block=False)
    except Exception:
        pass


# เสียงปืน (ใช้ในโหมดจำลอง หรือจริงก็ได้ ถ้าต้องการ)
FIRE_SOUND_ENABLED = True
# ใช้ไฟล์เสียงปืนกลจากโฟลเดอร์โปรเจกต์
FIRE_SOUND_FILE = "49053354-machine-gun-307468.mp3"
FIRE_SOUND_INTERVAL = 0.1  # วินาทีระหว่างเสียงยิงอย่างน้อย
FIRE_FLASH_DURATION = 0.12  # ระยะเวลาที่ศูนย์เล็งเปลี่ยนสีหลังยิง


def _play_fire_sound():
    """เล่นเสียงปืนหนึ่งครั้ง (ใช้ร่วมกับ joystick / spacebar)."""
    global _sound_fire
    if not FIRE_SOUND_ENABLED:
        return
    path = os.path.join(os.path.dirname(__file__), FIRE_SOUND_FILE)
    if not os.path.isfile(path):
        return
    try:
        if PYGAME_SOUND_AVAILABLE:
            if _sound_fire is None:
                _sound_fire = pygame.mixer.Sound(path)
            _sound_fire.play()
        elif PLAYSOUND_AVAILABLE:
            playsound(path, block=False)
    except Exception:
        pass

# =============================================================================
# Parameters
# =============================================================================
# รัศมี "ตรงกลาง" เป็นอัตราส่วนของ min(W,H) — ภายในวงนี้ถือว่ายิงได้ (เช่น 0.05 = 5%)
CENTER_RADIUS_RATIO = 0.05
# ถ้ากำหนด CENTER_RADIUS_PX > 0 จะใช้ค่าพิกเซลแทน ratio
CENTER_RADIUS_PX = 0

# Confidence ขั้นต่ำของ detection ที่นับเป็นเป้า
YOLO_CONF_MIN = 0.1

# YOLO รันเฉพาะบริเวณกลาง: อัตราส่วนของความกว้าง/สูงที่ใช้เป็น crop กลาง (เช่น 0.5 = 50%)
CENTER_CROP_RATIO = 0.5
# ขนาดที่ส่งเข้า YOLO สำหรับ crop กลาง (1280 = แม่น, 640 = เร็ว) — ถูก override โดย load
YOLO_CENTER_IMGSZ = 1280

# ความเร็ว: รัน YOLO แค่ทุก N เฟรม; เฟรมอื่นใช้ผลล่าสุด
YOLO_INTERVAL = 1
# True = โหลด engine 640 (เร็ว), False = ใช้ 1280 (แม่น)
USE_FAST_YOLO = False
# Resize เฟรมก่อนแสดงผลถ้าใหญ่กว่านี้ (ลดเวลา imshow)
DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080
# เต็มจอไร้ขอบ (ไม่มีแถบชื่อ/ปุ่มปิด) — ภาพ fit ในจอ อาจมี letterbox
DISPLAY_FULLSCREEN = True

# วงเล็งหลายชั้น: อัตราส่วนรัศมี [วงใน=ยิงได้, วงกลาง, วงนอก, วงใหญ่ 2.5× วงนอก] ของ min(W,H)
RETICLE_RADIUS_RATIOS = (0.03, 0.05, 0.08, 0.13)  # 0.20 = 0.08 × 2.5
# Sticky target: เมื่อมีเป้าแล้ว ถ้า bbox หายไปแป๊บไม่สลับไปตัวอื่น — รัศมีถือว่า "ตัวเดิม" (อัตราส่วน min(W,H))
TARGET_STICKY_RADIUS_RATIO = 0.10
# bbox เล็กสุดที่ถือว่าเป็นเป้าได้ (ไม่ตามจุดเล็ก/noise): min(width, height) ต้องไม่ต่ำกว่านี้
TARGET_MIN_BOX_PX = 40  # พิกเซล — ถ้า min(w,h) < ค่านี้ไม่เลือกเป็นเป้า
TARGET_MIN_BOX_RATIO = 0.02  # อัตราส่วนของ min(frame_w, frame_h) — ใช้ค่าที่ใหญ่กว่าระหว่าง MIN_PX กับ min_side * RATIO
# จำนวนครั้งที่ YOLO อัปเดตที่ยอมให้เป้าหายก่อนยอมสลับเป้าใหม่ (ลดการกระโดดเมื่อ bbox หายชั่วคราว)
TARGET_GRACE_UPDATES = 5
# Smooth ตำแหน่งศูนย์เป้า (EMA) เพื่อลดกระตุกจาก bbox ที่เปลี่ยนขนาดทุกเฟรม — 0.93–0.95 นิ่งมาก
AIM_CENTER_SMOOTH_ALPHA = 0.94
# ลูกศรทิศทาง: ความยาวพิกเซล (ใหญ่ให้เห็นชัด เล็งง่าย)
ARROW_LEN = 120
ARROW_THICKNESS = 10
# เสียงเมื่อเล็งตรง: รัวเสียงทุก SOUND_READY_INTERVAL วินาที ขณะที่เล็งอยู่ในกรอบ (กระตุ้นให้ยิง)
SOUND_ON_READY = True
SOUND_FILE = "beep_2x.wav"
# บีปสั้น: สร้างด้วย code ตอนสตาร์ท — เวลาเร่งให้ถี่จะถี่ได้มากโดยไม่ทับกัน
USE_SHORT_BEEP = True
SOUND_FILE_SHORT = "beep_short.wav"
BEEP_SHORT_DURATION_SEC = 0.06  # วินาที — ยิ่งสั้นยิ่งถี่ได้มาก
# ตอนยิงได้: ถี่จนเหมือนเสียงยาว (interval สั้นกว่าความยาวบีป → บีปทับกันเป็นเสียงต่อเนื่อง)
SOUND_READY_INTERVAL = 0.05  # วินาที — ตอน ready เล่นถี่มากจนฟังเหมือนเสียงยาว

# เสียงเข้าใกล้เป้า: แบ่งระดับตามวงรัศมี — วงนอกสุดห่าง, ถัดมาถี่ขึ้น, วงในถี่มาก, ready = เสียงยาว
SOUND_APPROACH_ON = True
# ช่วงบีป (วินาที) ต่อระดับวง: วงนอกสุด → วงถัดมา → วงถัดมา (ใกล้ ready)
APPROACH_BEEP_INTERVAL_OUTER = 0.8   # วงนอกสุด — รู้ว่าเป้าเข้าสู่ระยะเล็งแล้ว
APPROACH_BEEP_INTERVAL_MID2 = 0.4    # วงถัดมา
APPROACH_BEEP_INTERVAL_MID1 = 0.25   # วงถัดมา (ใกล้ ready) ถี่มาก — ready ใช้ SOUND_READY_INTERVAL

# ระยะถึงเป้า: ขนาดโดรนจริง (เมตร) สำหรับคำนวณระยะคร่าวๆ จาก FOV + bbox
DRONE_SIZE_M = 0.30  # 35 cm

# สีเส้นเล็งตามเวลา: กลางวัน = แดง/cyan เห็นชัด, กลางคืน = ขาว/เหลือง
DAY_HOUR_START = 6   # 6:00 = เริ่มกลางวัน
DAY_HOUR_END = 18    # 18:00 = เริ่มกลางคืน

# ใช้กล้องจาก config ตัวไหน (None = ใช้ ACTIVE_CAMERA)
CAMERA_NAME = None

WINDOW_NAME = "Gun Aim Assist"


def get_screen_size():
    """Get screen width and height (pixels). Fallback to DISPLAY_MAX_* if unavailable."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT


# ปุ่มเข้าโหมดตั้งค่า (แสดงบน HUD)
KEY_ADJUST_CENTER = "C"
KEY_SETTINGS = "S"
KEY_QUIT = "Q"

# ขั้นตอนเลื่อนศูนย์เล็ง (พิกเซล)
AIM_CENTER_STEP_PX = 10


class ShooterConfig:
    """
    เก็บค่าศูนย์เล็งและค่ากระสุน (ระยะหวังผล, ความเร็ว, น้ำหนัก).
    บันทึก/โหลดจาก gun_aim_assist_config.json ข้างๆ สคริปต์
    """
    FILENAME = "gun_aim_assist_config.json"

    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.effective_range_m = 100
        self.muzzle_velocity_ms = 900
        self.bullet_weight_g = 9
        self.target_size_m = 0.30
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FILENAME)

    def load(self):
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.offset_x = int(d.get("offset_x", 0))
                self.offset_y = int(d.get("offset_y", 0))
                self.effective_range_m = float(d.get("effective_range_m", 100))
                self.muzzle_velocity_ms = float(d.get("muzzle_velocity_ms", 900))
                self.bullet_weight_g = float(d.get("bullet_weight_g", 9))
                self.target_size_m = float(d.get("target_size_m", 0.30))
            except Exception:
                pass

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({
                    "offset_x": self.offset_x,
                    "offset_y": self.offset_y,
                    "effective_range_m": self.effective_range_m,
                    "muzzle_velocity_ms": self.muzzle_velocity_ms,
                    "bullet_weight_g": self.bullet_weight_g,
                    "target_size_m": self.target_size_m,
                }, f, indent=2)
        except Exception:
            pass

    def get_center(self, frame_w, frame_h):
        cx = frame_w // 2 + self.offset_x
        cy = frame_h // 2 + self.offset_y
        cx = max(0, min(frame_w - 1, cx))
        cy = max(0, min(frame_h - 1, cy))
        return cx, cy

    def move(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

    def reset_aim_center(self):
        self.offset_x = 0
        self.offset_y = 0


def is_daytime():
    """ใช้เวลาในเครื่องกำหนดกลางวัน/กลางคืน."""
    h = datetime.datetime.now().hour
    return DAY_HOUR_START <= h < DAY_HOUR_END


def estimate_distance_m(w_px, h_px, frame_w, frame_h, fov_h_deg, fov_v_deg, size_m=0.35):
    """
    คำนวณระยะถึงเป้า (เมตร) คร่าวๆ จากขนาด bbox ในภาพ + FOV กล้อง + ขนาดจริงเป้า.
    ใช้โดรนขนาด size_m (เมตร). คืน None ถ้า bbox เล็กเกินไป (ไม่เสถียร).
    """
    if w_px < 3 or h_px < 3 or frame_w <= 0 or frame_h <= 0:
        return None
    if fov_h_deg <= 0 or fov_v_deg <= 0:
        return None
    deg2rad = math.pi / 180.0
    fov_h_rad = fov_h_deg * deg2rad
    fov_v_rad = fov_v_deg * deg2rad
    theta_h = (w_px / frame_w) * fov_h_rad
    theta_v = (h_px / frame_h) * fov_v_rad
    if theta_h < 1e-6 and theta_v < 1e-6:
        return None
    dist_h = size_m / (2.0 * math.tan(theta_h / 2.0)) if theta_h >= 1e-6 else None
    dist_v = size_m / (2.0 * math.tan(theta_v / 2.0)) if theta_v >= 1e-6 else None
    if dist_h is not None and dist_v is not None:
        return (dist_h + dist_v) / 2.0
    return dist_h if dist_h is not None else dist_v

# =============================================================================
# Build CameraStream from config
# =============================================================================
def build_camera_from_config(camera_name=None):
    """สร้าง CameraStream จาก get_camera_config(camera_name)."""
    name = camera_name if camera_name is not None else ACTIVE_CAMERA
    cfg = get_camera_config(name)
    use_video = cfg.get("use_video_file", False)
    source = cfg["video_filename"] if use_video else cfg["rtsp_url"]
    if use_video and source and not os.path.isabs(source):
        source = os.path.join(os.path.dirname(__file__), source)
    width = cfg["width"]
    height = cfg["height"]
    cam_name = cfg.get("name", name)
    # Optional UDP/stream params
    udp_ip = cfg.get("udp_ip")
    udp_port = cfg.get("udp_port")
    use_udp_direct = cfg.get("use_udp_direct")
    stream_format = cfg.get("stream_format")
    cam = CameraStream(
        source=source,
        width=width,
        height=height,
        use_video_file=use_video,
        camera_name=cam_name,
        udp_ip=udp_ip,
        udp_port=udp_port,
        use_udp_direct=use_udp_direct,
        stream_format=stream_format,
    )
    return cam


def load_yolo_for_aim_assist():
    """
    โหลด YOLO ตาม USE_FAST_YOLO: ถ้า True พยายามโหลด engine 640; ไม่ได้ใช้ load_yolo_model (1280).
    Returns: (yolo_model, imgsz) หรือ (None, None).
    """
    if USE_FAST_YOLO and YOLO_AVAILABLE:
        base_dir = os.path.dirname(__file__)
        path_640 = os.path.join(base_dir, "yolo_11n_day_night_200_2_imgsz640.engine")
        if os.path.exists(path_640):
            try:
                yolo_model = UltralyticsYOLO(path_640, task="detect")
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_model.predict(dummy, verbose=False, device=0, imgsz=640)
                print("Gun Aim Assist: YOLO 640 loaded (fast mode)")
                return yolo_model, 640
            except Exception as e:
                print(f"Gun Aim Assist: 640 load failed ({e}), falling back to 1280")
    model, imgsz = load_yolo_model()
    if imgsz is None:
        imgsz = YOLO_CENTER_IMGSZ
    return model, imgsz


def crop_center_and_resize(frame, ratio, target_size):
    """
    Crop บริเวณกลางเฟรมแล้ว resize เป็น target_size x target_size (สำหรับส่งเข้า YOLO).
    Returns: (crop_resized, x0, y0, cw, ch) เพื่อใช้ map พิกัดกลับ.
    """
    h, w = frame.shape[:2]
    cw = max(1, int(w * ratio))
    ch = max(1, int(h * ratio))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = frame[y0 : y0 + ch, x0 : x0 + cw]
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized, x0, y0, cw, ch


def map_detections_to_full_frame(detections, x0, y0, cw, ch, crop_size):
    """
    แปลง bbox จากพิกัด crop (crop_size x crop_size) กลับเป็นพิกัดเฟรมเต็ม.
    detections: [(x, y, w, h, conf), ...] ใน crop space.
    """
    if not detections:
        return []
    scale_x = cw / crop_size
    scale_y = ch / crop_size
    out = []
    for x, y, bw, bh, conf in detections:
        x_full = x0 + int(x * scale_x)
        y_full = y0 + int(y * scale_y)
        w_full = max(1, int(bw * scale_x))
        h_full = max(1, int(bh * scale_y))
        out.append((x_full, y_full, w_full, h_full, conf))
    return out


def _distance_to_crosshair(det, cx_frame, cy_frame):
    """ระยะจากศูนย์ bbox ถึงศูนย์เล็ง (พิกเซล)."""
    x, y, w, h, _ = det
    cx_t = x + w // 2
    cy_t = y + h // 2
    return math.sqrt((cx_t - cx_frame) ** 2 + (cy_t - cy_frame) ** 2)


def _distance_to_point(det, ref_cx, ref_cy):
    """ระยะจากศูนย์ bbox ถึงจุด (ref_cx, ref_cy) พิกเซล."""
    x, y, w, h, _ = det
    cx_t = x + w // 2
    cy_t = y + h // 2
    return math.sqrt((cx_t - ref_cx) ** 2 + (cy_t - ref_cy) ** 2)


def _min_box_size_px(min_side):
    """ขนาด bbox เล็กสุด (พิกเซล) ที่ถือว่าเป็นเป้าได้: ใช้ค่าที่ใหญ่กว่าระหว่าง MIN_PX กับ min_side * RATIO."""
    if min_side is None or min_side <= 0:
        return TARGET_MIN_BOX_PX
    by_ratio = max(4, int(min_side * TARGET_MIN_BOX_RATIO))
    return max(TARGET_MIN_BOX_PX, by_ratio)


def _filter_detections_by_min_size(detections, min_side):
    """กรองเฉพาะ detection ที่ min(w,h) >= ขนาดเล็กสุดที่กำหนด."""
    th = _min_box_size_px(min_side)
    return [d for d in detections if d[2] >= th and d[3] >= th]


def find_detection_near_reference(last_target_det, detections, max_dist_px, min_side=None):
    """
    หา bbox ใน detections ที่อยู่ใกล้ตำแหน่งศูนย์ของ last_target_det ที่สุด
    ถ้าระยะน้อยกว่า max_dist_px ถือว่าเป็นตัวเดิม (sticky). คืน det นั้น ไม่ใช่คืน None.
    กรอง bbox ที่เล็กเกิน TARGET_MIN_BOX_* ออกก่อน.
    """
    if not last_target_det or not detections:
        return None
    detections = _filter_detections_by_min_size(detections, min_side)
    if not detections:
        return None
    x, y, w, h, _ = last_target_det
    ref_cx = x + w // 2
    ref_cy = y + h // 2
    best = min(detections, key=lambda d: _distance_to_point(d, ref_cx, ref_cy))
    dist = _distance_to_point(best, ref_cx, ref_cy)
    return best if dist <= max_dist_px else None


def pick_best_target(detections, cx_frame, cy_frame, min_side=None):
    """
    เลือกเป้าหมายหลัก: bbox ที่ศูนย์กลางอยู่ใกล้ศูนย์เล็ง (crosshair) ที่สุด
    ไม่เลือก bbox ที่เล็กเกิน TARGET_MIN_BOX_PX / TARGET_MIN_BOX_RATIO (ไม่ตามจุดเล็ก/noise).
    Returns: (x, y, w, h, conf) หรือ None ถ้าไม่มี detection.
    """
    if not detections:
        return None
    detections = _filter_detections_by_min_size(detections, min_side)
    if not detections:
        return None
    best = min(detections, key=lambda det: _distance_to_crosshair(det, cx_frame, cy_frame))
    return best


def compute_guide_direction(target_det, cx_frame, cy_frame, radius_px, ready_to_fire):
    """
    คำนวณทิศทางแนะนำ: ขยับซ้าย/ขวา/ขึ้น/ลง เมื่อเป้าอยู่นอกวงยิงได้.
    Returns: (guide_h, guide_v) โดย guide_h เป็น "LEFT" | "RIGHT" | None, guide_v เป็น "UP" | "DOWN" | None.
    """
    if target_det is None or ready_to_fire:
        return None, None
    x, y, w_d, h_d, _ = target_det
    cx_t = x + w_d // 2
    cy_t = y + h_d // 2
    dx = cx_t - cx_frame
    dy = cy_t - cy_frame
    if abs(dx) <= radius_px and abs(dy) <= radius_px:
        return None, None
    guide_h = "RIGHT" if dx > 0 else "LEFT"
    guide_v = "DOWN" if dy > 0 else "UP"
    return guide_h, guide_v


def compute_approach_beep_interval(d, radius_px, min_side):
    """
    คำนวณช่วงเวลา (วินาที) ระหว่างบีปตามระดับวงรัศมี:
    - วงนอกสุด = บีปห่าง (รู้ว่าเป้าเข้าสู่ระยะเล็งแล้ว)
    - วงถัดมาแต่ละระดับ = ถี่ขึ้นเรื่อยๆ
    - วงใน (ready) = ใช้ SOUND_READY_INTERVAL ใน main (ถี่จนเหมือนเสียงยาว)
    d = ระยะจากศูนย์เล็งถึงศูนย์โดรน (พิกเซล), min_side = min(w,h)
    คืน None ถ้าไม่ควรเล่นบีป (ready หรืออยู่นอกวงทั้งหมด)
    """
    if d <= radius_px:
        return None  # อยู่ในวงยิงได้ = ใช้เสียง ready แทน (ถี่มากจนเสียงยาว)
    # รัศมีแต่ละวง (พิกเซล) ตาม RETICLE_RADIUS_RATIOS
    r0 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[0]))  # ready
    r1 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[1]))
    r2 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[2]))
    r3 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[3]))  # วงนอกสุด
    if d > r3:
        return None  # นอกวงทั้งหมด — ไม่บีป
    if d <= r1:
        return APPROACH_BEEP_INTERVAL_MID1   # วงถัดมา (ใกล้ ready) ถี่มาก
    if d <= r2:
        return APPROACH_BEEP_INTERVAL_MID2   # วงถัดมา
    return APPROACH_BEEP_INTERVAL_OUTER      # วงนอกสุด — รู้ว่าเข้าสู่ระยะเล็งแล้ว


def _det_eq(a, b):
    """เทียบว่า detection a กับ b เป็นตัวเดียวกัน (x,y,w,h)."""
    return (a[0], a[1], a[2], a[3]) == (b[0], b[1], b[2], b[3])


def draw_hud(frame, cx_frame, cy_frame, radius_px, target_det, ready_to_fire, guide_h=None, guide_v=None, is_day=None, distance_m=None, all_detections=None):
    """วาดเส้นระดับ, reticle หลายชั้น, จุดกลาง, bbox เป้า (conf สูงสุดเด่น), READY/AIM, ลูกศรทิศทาง. สีแยกกลางวัน/กลางคืน."""
    h, w = frame.shape[:2]
    if is_day is None:
        is_day = is_daytime()
    if is_day:
        color_ready = (0, 0, 255)   # BGR red — เล็งตรงแล้วแดงทั้งหมด
        color_aim = (0, 0, 0)       # BGR black — กลางวันตอนเล็งดำ
        color_arrow = (0, 0, 0)     # BGR black
    else:
        color_ready = (0, 0, 255)   # BGR red — เล็งตรงแล้วแดงทั้งหมด
        color_aim = (0, 255, 0)     # BGR green — กลางคืนตอนเล็งเขียว
        color_arrow = (0, 255, 0)   # BGR green

    # สีศูนย์เล็ง: แดงเมื่อยิงได้, ส้มเมื่อโดรนเข้าวงรัศมี (แต่ยังไม่ ready), ไม่ก็ตามโหมด (ดำ/เขียว)
    color_in_range = (0, 165, 255)  # BGR orange — โดรนเข้าวงใดวงหนึ่งแล้ว
    min_side = min(h, w)
    color_use = color_aim
    if ready_to_fire:
        color_use = color_ready
    elif target_det is not None:
        r3 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[3]))
        x_t, y_t, w_d, h_d, _ = target_det
        cx_t = x_t + w_d // 2
        cy_t = y_t + h_d // 2
        d = math.sqrt((cx_t - cx_frame) ** 2 + (cy_t - cy_frame) ** 2)
        if d <= r3:
            color_use = color_in_range

    # 1. เส้นระดับกล้อง (แนวนอน + แนวตั้ง) — ไม่ลากผ่านวงในสุด เพื่อไม่บังโดรน
    r0 = max(4, int(min_side * RETICLE_RADIUS_RATIOS[0]))
    x_left = max(0, cx_frame - r0)
    x_right = min(w, cx_frame + r0)
    y_top = max(0, cy_frame - r0)
    y_bottom = min(h, cy_frame + r0)
    cv2.line(frame, (0, cy_frame), (x_left, cy_frame), color_use, 3)
    cv2.line(frame, (x_right, cy_frame), (w, cy_frame), color_use, 3)
    cv2.line(frame, (cx_frame, 0), (cx_frame, y_top), color_use, 3)
    cv2.line(frame, (cx_frame, y_bottom), (cx_frame, h), color_use, 3)

    # 2. วงเล็งหลายชั้น
    for ratio in RETICLE_RADIUS_RATIOS:
        r = max(4, int(min_side * ratio))
        cv2.circle(frame, (cx_frame, cy_frame), r, color_use, 4)
    # จุดกลาง
    cv2.circle(frame, (cx_frame, cy_frame), 4, color_use, -1)

    # 3. bbox อื่น (ถ้ามี) — เส้นบาง สีเทา ให้เป้าหมายหลักเด่น
    color_other = (128, 128, 128)
    if all_detections:
        for det in all_detections:
            if target_det is not None and _det_eq(det, target_det):
                continue
            xo, yo, wo, ho, _ = det
            cv2.rectangle(frame, (xo, yo), (xo + wo, yo + ho), color_other, 1)

    # 4. เป้าหมายหลัก (conf สูงสุด) — bbox หนา ป้ายใหญ่ ตัวหนังสือเด่น
    if target_det is not None:
        x, y, w_box, h_box, conf = target_det
        color = color_use
        # กลางวันตอนเล็งไม่ตรง: ตัวหนังสือ HUD เป็นสีขาว ให้เห็นชัดบนพื้นหลังดำ
        color_text_hud = (255, 255, 255) if (color == (0, 0, 0)) else color
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 4)
        cx_t = x + w_box // 2
        cy_t = y + h_box // 2
        cv2.circle(frame, (cx_t, cy_t), 6, color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_label = 1.2
        th_label = 3
        pad = 10

        # ป้ายเหนือ bbox: "DRONE XX%" — ตัวหนังสือใหญ่ เด่นกว่าพื้นหลัง
        conf_pct = min(99, max(0, int(round(conf * 100))))
        label_drone = f"DRONE {conf_pct}%"
        (tw, th), _ = cv2.getTextSize(label_drone, font, scale_label, th_label)
        ty_baseline = max(th + pad, y - 6)
        r_y1 = max(0, ty_baseline - th - pad)
        r_x2 = min(w, x + tw + 24)
        cv2.rectangle(frame, (x, r_y1), (r_x2, ty_baseline + pad), (0, 0, 0), -1)
        tx, ty = x + pad, ty_baseline
        cv2.putText(frame, label_drone, (tx, ty), font, scale_label, (0, 0, 0), th_label + 2)
        cv2.putText(frame, label_drone, (tx, ty), font, scale_label, color_text_hud, th_label)

        # ระยะห่าง — ตัวหนังสือใหญ่ เด่น
        if distance_m is not None and distance_m > 0:
            dist_text = f"~{distance_m:.0f} m"
            scale_dist = 1.3
            th_dist = 3
            dy_bottom = min(h - 8, y + h_box + 28)
            cv2.putText(frame, dist_text, (x, dy_bottom), font, scale_dist, (0, 0, 0), th_dist + 2)
            cv2.putText(frame, dist_text, (x, dy_bottom), font, scale_dist, color_text_hud, th_dist)

    # 5. ข้อความ READY / THINGS (THINGS สีน้ำเงิน)
    label = "READY" if ready_to_fire else "THINGS"
    if ready_to_fire:
        color_text = (255, 255, 255) if (color_use == (0, 0, 0)) else color_use
    else:
        color_text = (255, 0, 0)  # BGR blue สำหรับ THINGS
    cv2.putText(frame, label, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_text, 3)

    # 6. ลูกศรแทนข้อความทิศทาง (วางใต้ reticle วงนอกเดิม ไม่ใช้วงใหญ่)
    arrow_r = max(4, int(min_side * RETICLE_RADIUS_RATIOS[-2]))  # วง 0.08 ไม่ใช้วงใหญ่ 0.20
    arrow_y_center = cy_frame + arrow_r + 35
    al = min(ARROW_LEN, w // 5)
    thickness = ARROW_THICKNESS
    tip_len = 0.25
    if guide_h == "RIGHT":
        cv2.arrowedLine(
            frame,
            (cx_frame - al, arrow_y_center),
            (cx_frame + al, arrow_y_center),
            color_use, thickness, tipLength=tip_len
        )
    elif guide_h == "LEFT":
        cv2.arrowedLine(
            frame,
            (cx_frame + al, arrow_y_center),
            (cx_frame - al, arrow_y_center),
            color_use, thickness, tipLength=tip_len
        )
    if guide_v == "DOWN":
        cv2.arrowedLine(
            frame,
            (cx_frame, arrow_y_center - al),
            (cx_frame, arrow_y_center + al),
            color_use, thickness, tipLength=tip_len
        )
    elif guide_v == "UP":
        cv2.arrowedLine(
            frame,
            (cx_frame, arrow_y_center + al),
            (cx_frame, arrow_y_center - al),
            color_use, thickness, tipLength=tip_len
        )
    return frame


def draw_hint_keys_normal(frame, w, h):
    """Draw bottom HUD: Press [C] Aim center  [S] Ballistics  [Q] Quit + settings key hint."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = 1.0
    scale2 = 0.85
    thickness = 2
    pad_box = 20
    line1 = f"Press [{KEY_ADJUST_CENTER}] Aim center  [{KEY_SETTINGS}] Ballistics  [{KEY_QUIT}] Quit"
    line2 = "Settings: C = aim center, S = ballistics, Enter = save & exit"
    (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thickness)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thickness)
    x = 20
    step_y = 36
    y1 = h - 85
    y2 = h - 44
    box_w = min(w - x - 10, max(tw1, tw2) + pad_box)
    cv2.rectangle(frame, (x, y1 - th1 - 6), (x + box_w, y2 + 6), (0, 0, 0), -1)
    cv2.putText(frame, line1, (x + 6, y1), font, scale1, (255, 255, 255), thickness)
    cv2.putText(frame, line2, (x + 6, y2), font, scale2, (200, 200, 200), thickness)


def draw_adjust_center_hud(frame, config, w, h):
    """Aim center setup: draw reticle at current center + HUD instructions + bottom key hint."""
    cx, cy = config.get_center(w, h)
    color = (0, 255, 0)
    min_side = min(h, w)
    for ratio in RETICLE_RADIUS_RATIOS:
        r = max(4, int(min_side * ratio))
        cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 2, color, -1)
    cv2.line(frame, (0, cy), (w, cy), color, 1)
    cv2.line(frame, (cx, 0), (cx, h), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thickness = 2
    lines = [
        "Aim center setup",
        "Arrow / W A S D: Move reticle",
        "R: Reset to screen center",
        "Enter: Save and exit",
    ]
    y0 = 68
    step = 36
    for i, line in enumerate(lines):
        y = y0 + i * step
        cv2.putText(frame, line, (20, y), font, scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, line, (20, y), font, scale, (255, 255, 255), thickness)
    bottom_hint = "Keys: Arrow/WASD move, R reset, Enter save & exit"
    (tw, th), _ = cv2.getTextSize(bottom_hint, font, 0.9, 2)
    by = h - 32
    bx = 20
    box_w = min(w - bx - 10, tw + 20)
    cv2.rectangle(frame, (bx, by - th - 6), (bx + box_w, by + 6), (0, 0, 0), -1)
    cv2.putText(frame, bottom_hint, (bx + 6, by), font, 0.9, (255, 255, 255), 2)


def draw_settings_overlay(frame, config, selected_field):
    """Ballistics + target size: show values and key hints; panel size from text so background covers all."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.05
    thickness = 2
    pad_x = 56
    pad_y = 50
    title = "Ballistics (marksman)"
    title_scale = 1.2
    fields = [
        ("Effective range (m)", config.effective_range_m),
        ("Muzzle velocity (m/s)", config.muzzle_velocity_ms),
        ("Bullet weight (g)", config.bullet_weight_g),
        ("Target size (m)", config.target_size_m),
    ]
    footer = "1/2/3/4: Select  Up/Down: +/-  Enter: Save"
    (tw_title, th_title), _ = cv2.getTextSize(title, font, title_scale, 2)
    max_fw = tw_title
    for label, value in fields:
        (fw, _), _ = cv2.getTextSize(f"  > {label}: {value:.1f}", font, scale, thickness)
        max_fw = max(max_fw, fw)
    (tw_foot, th_foot), _ = cv2.getTextSize(footer, font, 0.85, 2)
    max_fw = max(max_fw, tw_foot)
    line_h = 48
    panel_w = max_fw + pad_x
    panel_h = th_title + 24 + len(fields) * line_h + 20 + th_foot + 16
    panel_w = min(w - 40, max(420, panel_w))
    panel_h = min(h - 80, max(320, panel_h))
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, title, (x0 + 24, y0 + 40), font, title_scale, (255, 255, 255), 2)
    y = y0 + 82
    for i, (label, value) in enumerate(fields):
        marker = ">" if i == selected_field else " "
        text = f"  {marker} {label}: {value:.2f}" if i == 3 else f"  {marker} {label}: {value:.1f}"
        color = (0, 255, 255) if i == selected_field else (220, 220, 220)
        cv2.putText(frame, text, (x0 + 24, y), font, scale, color, thickness)
        y += line_h
    cv2.putText(frame, footer, (x0 + 24, y0 + panel_h - 32), font, 0.85, (180, 180, 180), 2)
    bottom_hint = "Keys: 1/2/3/4 select  Up/Down +/-  Enter save & exit"
    (tw, th), _ = cv2.getTextSize(bottom_hint, font, 0.95, 2)
    by = h - 32
    bx = 20
    box_w = min(w - bx - 10, tw + 20)
    cv2.rectangle(frame, (bx, by - th - 6), (bx + box_w, by + 6), (0, 0, 0), -1)
    cv2.putText(frame, bottom_hint, (bx + 6, by), font, 0.95, (255, 255, 255), 2)


def main():
    camera_name = CAMERA_NAME if CAMERA_NAME is not None else ACTIVE_CAMERA
    cam_config = get_camera_config(camera_name)
    fov_h = cam_config.get("fov_horizontal", 60.0)
    fov_v = cam_config.get("fov_vertical", 36.0)
    print("Gun Aim Assist: starting camera from config", camera_name)

    # ------------------------------------------------------------------
    # Optional: Initialize cam4 arm controller + tracker (homing + reference pose)
    # ------------------------------------------------------------------
    arm_controller = None
    arm_tracker = None
    arm_mode_manager = None
    joystick_reader = None
    joystick_mapper = None

    if Cam4ArmController is not None and Cam4ArmTracker is not None:
        cfg_mod = __import__("config")
        use_arm = getattr(cfg_mod, "CAM4_ARM_ENABLED", False)
        use_sim = getattr(cfg_mod, "CAM4_ARM_SIMULATION_MODE", False)
        # โหมดจริง: จำกัดให้ใช้กับ cam4 เท่านั้น, โหมดจำลอง: ใช้กับกล้องไหนก็ได้
        if use_arm and (camera_name == "cam4" or use_sim):
            try:
                if use_sim and SimCam4ArmController is not None:
                    print("Gun Aim Assist: initializing SimCam4ArmController (simulation mode)...")
                    arm_controller = SimCam4ArmController()
                else:
                    print("Gun Aim Assist: initializing Cam4ArmController...")
                    arm_controller = Cam4ArmController()
                if arm_controller.connect():
                    arm_tracker = Cam4ArmTracker(arm_controller, camera_name="cam4")
                else:
                    print("⚠️ Gun Aim Assist: Cam4ArmController connect() failed, running without arm.")
                    arm_controller = None
            except Exception as e:
                print(f"❌ Gun Aim Assist: failed to initialize Cam4 arm: {e}")
                arm_controller = None

    # ------------------------------------------------------------------
    # Optional: Initialize joystick + arm mode manager (AUTO / MANUAL / SAFE)
    # ------------------------------------------------------------------
    if ArmModeManager is not None:
        arm_mode_manager = ArmModeManager(has_arm=arm_controller is not None)
    if JoystickReader is not None and JoystickArmMapper is not None and arm_controller is not None:
        try:
            joystick_reader = JoystickReader()
            if getattr(joystick_reader, "enabled", False):
                cfg = __import__("config")
                max_pan = getattr(cfg, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0)
                max_tilt = getattr(cfg, "CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0)
                deadzone = getattr(cfg, "CAM4_ARM_JOYSTICK_DEADZONE", 0.05)
                stick_exp = getattr(cfg, "CAM4_ARM_JOYSTICK_STICK_EXPONENT", 2.0)
                scale_min = getattr(cfg, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MIN", 0.1)
                scale_max = getattr(cfg, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MAX", 10.0)
                speed_smooth = getattr(cfg, "CAM4_ARM_JOYSTICK_SPEED_SMOOTH_ALPHA", 0.82)
                sens_str = getattr(cfg, "CAM4_ARM_JOYSTICK_SENSITIVITY", "medium")
                initial_sens = _parse_sensitivity_mode(sens_str)
                joystick_mapper = JoystickArmMapper(
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
            print(f"Gun Aim Assist: joystick init failed: {e}")
            joystick_reader = None
            joystick_mapper = None

    cam = build_camera_from_config(camera_name)
    cam.start()

    print("Loading YOLO model...")
    yolo_model, yolo_center_imgsz = load_yolo_for_aim_assist()
    if yolo_model is None:
        print("YOLO model not available. Exiting.")
        cam.release()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size()
    if DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    config = ShooterConfig()
    config.load()

    if USE_SHORT_BEEP:
        beep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SOUND_FILE_SHORT)
        create_short_beep_wav(beep_path, duration_sec=BEEP_SHORT_DURATION_SEC)

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
    target_missing_count = 0  # จำนวนครั้งที่เป้าเดิมไม่โผล่ใน detection (ใช้กับ grace)
    smoothed_aim_cx = None
    smoothed_aim_cy = None
    prev_ready_to_fire = False
    last_ready_sound_time = 0.0
    last_approach_beep_time = 0.0

    # สถานะการยิง (ใช้สำหรับเสียงปืน + flash ศูนย์เล็ง)
    is_firing = False
    last_fire_time = 0.0
    last_fire_gcode_time = 0.0  # หน่วงส่ง G-code ยิง (cooldown ระหว่างนัด)

    # edge detection สำหรับปุ่ม 12 สลับโหมดความเร็วจอยสติ๊ก (ช้า/กลาง/สูง)
    prev_sensitivity_cycle_pressed = False

    # เวลาใช้สำหรับ dt ในการควบคุมแขน
    last_loop_time = time.time()

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
        radius_px = int(min_side * RETICLE_RADIUS_RATIOS[0])
        if CENTER_RADIUS_PX > 0:
            radius_px = CENTER_RADIUS_PX
        radius_px = max(10, radius_px)

        # dt สำหรับการคำนวณความเร็วหมุนของแขน (manual)
        now_loop = time.time()
        dt_loop = now_loop - last_loop_time if now_loop > last_loop_time else 0.0
        last_loop_time = now_loop

        if app_mode == "normal":
            frame_counter += 1
            if frame_counter % YOLO_INTERVAL == 0:
                center_crop, x0, y0, cw, ch = crop_center_and_resize(
                    frame, CENTER_CROP_RATIO, yolo_center_imgsz
                )
                detections_crop = detect_yolo_full_frame(
                    yolo_model, center_crop, YOLO_CONF_MIN, imgsz=yolo_center_imgsz
                )
                last_detections = map_detections_to_full_frame(
                    detections_crop, x0, y0, cw, ch, yolo_center_imgsz
                )
                # Sticky target + grace: ไม่สลับเป้าเมื่อ bbox หายไปแป๊บ
                sticky_radius_px = max(40, int(min_side * TARGET_STICKY_RADIUS_RATIO))
                if last_target_det is None:
                    last_target_det = pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                    target_missing_count = 0
                else:
                    same_target = find_detection_near_reference(
                        last_target_det, last_detections, sticky_radius_px, min_side
                    )
                    if same_target is not None:
                        last_target_det = same_target
                        target_missing_count = 0
                    else:
                        target_missing_count += 1
                        if target_missing_count >= TARGET_GRACE_UPDATES:
                            last_target_det = pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                            target_missing_count = 0
                        # else: ยังใช้ last_target_det เดิม (grace)

            target_det = last_target_det
            ready_to_fire = False
            if target_det is not None:
                x, y, w_d, h_d, _ = target_det
                cx_t = x + w_d // 2
                cy_t = y + h_d // 2
                # Smooth ศูนย์เป้าเพื่อลดกระตุกจาก bbox ที่เปลี่ยนขนาดทุกเฟรม
                alpha = AIM_CENTER_SMOOTH_ALPHA
                if smoothed_aim_cx is None:
                    smoothed_aim_cx, smoothed_aim_cy = float(cx_t), float(cy_t)
                else:
                    smoothed_aim_cx = alpha * smoothed_aim_cx + (1.0 - alpha) * cx_t
                    smoothed_aim_cy = alpha * smoothed_aim_cy + (1.0 - alpha) * cy_t
                d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                ready_to_fire = d <= radius_px

                # ส่งตำแหน่งเป้า (smoothed) + ศูนย์เล็ง (cx_frame, cy_frame) ให้ tracker ใช้เป็นจุดอ้างอิง
                if arm_tracker is not None and arm_mode_manager is not None:
                    if arm_mode_manager.mode == MODE_AUTO:
                        arm_tracker.update_from_detection(
                            smoothed_aim_cx, smoothed_aim_cy, w, h, cx_frame, cy_frame
                        )
                    elif arm_mode_manager.mode == MODE_LOCK:
                        arm_tracker.update_from_detection(
                            smoothed_aim_cx, smoothed_aim_cy, w, h, cx_frame, cy_frame
                        )

            now = time.time()
            if ready_to_fire and SOUND_ON_READY:
                if not prev_ready_to_fire:
                    _play_ready_sound()
                    last_ready_sound_time = now
                elif now - last_ready_sound_time >= SOUND_READY_INTERVAL:
                    _play_ready_sound()
                    last_ready_sound_time = now
            elif target_det is not None and SOUND_APPROACH_ON and smoothed_aim_cx is not None:
                # บีปเข้าใกล้: ใช้ระยะจาก smoothed ถึงศูนย์เล็ง
                d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                interval = compute_approach_beep_interval(d, radius_px, min_side)
                if interval is not None and (now - last_approach_beep_time) >= interval:
                    _play_ready_sound()
                    last_approach_beep_time = now
            prev_ready_to_fire = ready_to_fire

            guide_h, guide_v = compute_guide_direction(
                target_det, cx_frame, cy_frame, radius_px, ready_to_fire
            )
            distance_m = None
            if target_det is not None:
                x_t, y_t, w_d, h_d, _ = target_det
                distance_m = estimate_distance_m(w_d, h_d, w, h, fov_h, fov_v, config.target_size_m)
            draw_hud(frame, cx_frame, cy_frame, radius_px, target_det, ready_to_fire, guide_h, guide_v, is_day=is_daytime(), distance_m=distance_m, all_detections=last_detections)
            draw_hint_keys_normal(frame, w, h)
        elif app_mode == "adjust_center":
            draw_adjust_center_hud(frame, config, w, h)
        elif app_mode == "settings":
            draw_settings_overlay(frame, config, settings_selected_field)

        # ------------------------------------------------------------------
        # Manual joystick control (MODE_MANUAL)
        # ------------------------------------------------------------------
        if (
            arm_controller is not None
            and joystick_reader is not None
            and joystick_mapper is not None
            and arm_mode_manager is not None
        ):
            js_state = joystick_reader.read()
            # ปุ่ม 12: สลับโหมดความเร็วจอยสติ๊ก (ช้า → กลาง → สูง → ช้า)
            if js_state.sensitivity_cycle_pressed and not prev_sensitivity_cycle_pressed:
                joystick_mapper.cycle_sensitivity_mode()
            prev_sensitivity_cycle_pressed = js_state.sensitivity_cycle_pressed
            # ปุ่มโหมดจาก joystick (ปุ่ม 2 ซ้ำในโหมด LOCK = ปลด LOCK กลับไป MANUAL)
            if js_state.mode_switch is not None:
                if js_state.mode_switch == MODE_LOCK and arm_mode_manager.mode == MODE_LOCK:
                    arm_mode_manager.set_mode(MODE_MANUAL)
                else:
                    arm_mode_manager.set_mode(js_state.mode_switch)

            # ปุ่มยิงจาก joystick (ทุกโหมด ยกเว้น SAFE)
            if js_state.fire_pressed and arm_mode_manager.mode != MODE_SAFE:
                now_fire = time.time()
                if now_fire - last_fire_time >= FIRE_SOUND_INTERVAL:
                    _play_fire_sound()
                    is_firing = True
                    last_fire_time = now_fire
                    # ส่ง G-code ยิงเฉพาะเมื่อครบ cooldown (หน่วงระหว่างนัด)
                    fire_cooldown = getattr(__import__("config"), "CAM4_ARM_FIRE_COOLDOWN_SEC", 0.5)
                    if arm_controller is not None and hasattr(arm_controller, "fire"):
                        if now_fire - last_fire_gcode_time >= fire_cooldown:
                            arm_controller.fire()
                            last_fire_gcode_time = now_fire

            # ควบคุมแขนด้วย joystick เฉพาะเมื่ออยู่โหมด MANUAL
            if arm_mode_manager.mode == MODE_MANUAL:
                joystick_mapper.apply(js_state, dt_loop)

        # ถ้า controller รายงานว่า unhealthy ให้บังคับโหมด SAFE
        if arm_controller is not None and arm_mode_manager is not None:
            if hasattr(arm_controller, "is_healthy") and not arm_controller.is_healthy:
                arm_mode_manager.set_mode(MODE_SAFE)

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
        # ปรับสถานะ flash การยิง (หมดอายุเมื่อเกิน FIRE_FLASH_DURATION)
        if is_firing and (time.time() - last_fire_time) > FIRE_FLASH_DURATION:
            is_firing = False

        fps_text = f"FPS: {fps:.1f}"
        (fps_tw, fps_th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(frame, fps_text, (w - fps_tw - 28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # แสดงสถานะโหมดแขน (AUTO / MANUAL / SAFE / LOCK)
        if arm_mode_manager is not None:
            mode_label, mode_color = arm_mode_manager.label_and_color()
            cv2.putText(
                frame,
                mode_label,
                (20, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                mode_color,
                2,
            )
        # แสดงโหมดความเร็วจอยสติ๊ก (ช้า/กลาง/สูง) เมื่อมีจอยสติ๊ก
        if joystick_mapper is not None:
            joy_label = f"JOY: {joystick_mapper.get_sensitivity_label()}"
            cv2.putText(
                frame,
                joy_label,
                (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        # ถ้ากำลังยิง ให้ flash ศูนย์เล็งทับ HUD ปกติ
        if is_firing:
            fire_color = (255, 255, 0)  # เหลืองสว่าง
            cv2.circle(frame, (cx_frame, cy_frame), radius_px + 6, fire_color, 3)
            cv2.circle(frame, (cx_frame, cy_frame), 6, fire_color, -1)

        # แสดง overlay วงกลมสีขาวสำหรับตำแหน่งเล็งของแขน (โหมดจำลองเท่านั้น)
        if arm_controller is not None and arm_tracker is not None and arm_mode_manager is not None:
            # ถ้า controller เป็น simulation (SimCam4ArmController หรือ flag simulation)
            is_sim_arm = getattr(arm_controller, "is_simulation_mode", False)
            if is_sim_arm:
                try:
                    sim_px, sim_py = arm_tracker.arm_angles_to_pixel(
                        getattr(arm_controller, "pos_x", 0.0),
                        getattr(arm_controller, "pos_y", 0.0),
                        w,
                        h,
                    )
                    center = (int(round(sim_px)), int(round(sim_py)))
                    radius = max(20, int(min(w, h) * 0.025))
                    cv2.circle(frame, center, radius, (255, 255, 255), 3)
                    cv2.circle(frame, center, 3, (255, 255, 255), -1)
                except Exception:
                    # overlay ไม่ควรทำให้ loop ล้ม
                    pass

        if DISPLAY_FULLSCREEN:
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
            max_w = min(screen_w, DISPLAY_MAX_WIDTH)
            max_h = min(screen_h, DISPLAY_MAX_HEIGHT)
            scale = min(max_w / w, max_h / h, 1.0)
            display_w = int(w * scale)
            display_h = int(h * scale)
            if (display_w, display_h) != (w, h):
                display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame
        try:
            cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
        except cv2.error:
            pass
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1)
        if key == -1:
            pass
        elif key in (ord("q"), ord("Q")) and loop_frames >= GRACE_FRAMES:
            break
        elif app_mode == "normal":
            # เปลี่ยนโหมด UI
            if key in (ord("c"), ord("C")):
                app_mode = "adjust_center"
            elif key in (ord("s"), ord("S")):
                app_mode = "settings"

            # เปลี่ยนโหมดแขน (AUTO / MANUAL / SAFE / LOCK) ด้วยคีย์บอร์ด
            if arm_mode_manager is not None:
                if key in (ord("a"), ord("A")):
                    arm_mode_manager.set_mode(MODE_AUTO)
                elif key in (ord("m"), ord("M")):
                    arm_mode_manager.set_mode(MODE_MANUAL)
                elif key in (ord("f"), ord("F")):
                    arm_mode_manager.set_mode(MODE_SAFE)
                elif key in (ord("l"), ord("L")):
                    arm_mode_manager.set_mode(MODE_LOCK)

            # ยิงด้วยคีย์บอร์ด (spacebar) ยกเว้น SAFE
            if key in (32,) and arm_mode_manager is not None and arm_mode_manager.mode != MODE_SAFE:
                now_fire = time.time()
                if now_fire - last_fire_time >= FIRE_SOUND_INTERVAL:
                    _play_fire_sound()
                    is_firing = True
                    last_fire_time = now_fire
                    fire_cooldown = getattr(__import__("config"), "CAM4_ARM_FIRE_COOLDOWN_SEC", 0.5)
                    if arm_controller is not None and hasattr(arm_controller, "fire"):
                        if now_fire - last_fire_gcode_time >= fire_cooldown:
                            arm_controller.fire()
                            last_fire_gcode_time = now_fire
        elif app_mode == "adjust_center":
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
    print("Gun Aim Assist: stopped.")


if __name__ == "__main__":
    main()


