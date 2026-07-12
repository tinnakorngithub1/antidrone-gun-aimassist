"""
Cam4 Arm Mouse Grid Calibrator
------------------------------
- สอน mapping (x,y) บน output frame -> (pan_deg, tilt_deg) แบบ grid หยาบทั้งจอ + grid ละเอียดช่องกลาง
- โหมด Calibration: สอน coarse แล้ว fine; โหมด Test: โหลด JSON คลิกทดสอบ ปรับจอยยืนยันได้
- ไฟล์ใหม่เท่านั้น ไม่แก้ flow โปรเจกต์เดิม; บันทึกที่ calibration_data/cam4_mouse_grid_lookup.json
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import config
except ImportError:
    config = None

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
    from joystick_cam4_controller import JoystickReader, JoystickArmMapper, SENSITIVITY_LOW
except ImportError:
    JoystickReader = None
    JoystickArmMapper = None

try:
    from cam4_arm_grid_lookup import (
        MOUSE_GRID_LOOKUP_FILE,
        load as load_grid_json,
        get_crosshair,
        pixel_to_arm_degrees,
        arm_degrees_to_pixel,
    )
except ImportError:
    MOUSE_GRID_LOOKUP_FILE = Path(__file__).resolve().parent / "calibration_data" / "cam4_mouse_grid_lookup.json"
    load_grid_json = None
    get_crosshair = None
    pixel_to_arm_degrees = None
    arm_degrees_to_pixel = None

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
JSON_PATH = CALIBRATION_DIR / "cam4_mouse_grid_lookup.json"  # homography + grid only
PX_DEG_JSON_PATH = CALIBRATION_DIR / "cam4_pixel_per_degree.json"  # Px/deg only (แยกจาก homography)
# ค่า default เมื่อไม่มี JSON หรือไม่มีค่า — ปรับเองได้ (ใช้ทั้งโหมด L และ P ใกล้ศูนย์เล็ง)
# แกน Y เป็นลบเพื่อให้คลิกด้านบน = เงยขึ้น (ไม่หมุนลง)
PX_PER_DEG_X_DEFAULT = 50.0
PX_PER_DEG_Y_DEFAULT = -50.0
TEST_MOVE_LOG_PATH = CALIBRATION_DIR / "cam4_test_move_log_homography.jsonl"  # log สำหรับ recal homography
TEST_MOVE_LOG_PXDEG_PATH = CALIBRATION_DIR / "cam4_test_move_log_pxdeg.jsonl"  # log สำหรับ recal Px/deg
PATH_SIM_CLICK = CALIBRATION_DIR / "path_sim_click.jsonl"  # โหมด L คลิกเอง: ตำแหน่งคลิก + โดรน + timestamp
SIM_MOUSE_ENDPOINT_PATH = CALIBRATION_DIR / "sim_mouse_endpoint.json"  # โหมด L / gun_aim_assist กด N: จุดปลายเมาส์
SIM_MOUSE_ENDPOINT_TTL_SEC = 10.0

GRID_COLS = 3
GRID_ROWS = 3
FINE_ROWS = 3
FINE_COLS = 3

# ขนาดคำนวณและ ref ตายตัว = 4K (ให้ตรงกับ frame กล้อง 4K / gun_aim_assist_grid_calib)
REF_OUTPUT_W = 3840
REF_OUTPUT_H = 2160

# ตัวหนังสือและ HUD ให้เหมาะสมกับ resolution (4K = 2x จาก 1920)
HUD_TEXT_SCALE = max(1.0, REF_OUTPUT_W / 1920.0)
FONT_SMALL = 0.5 * HUD_TEXT_SCALE
FONT_MID = 0.55 * HUD_TEXT_SCALE
FONT_LARGE = 0.65 * HUD_TEXT_SCALE
FONT_TITLE = 1.2 * HUD_TEXT_SCALE
THICKNESS_THIN = max(1, int(1 * HUD_TEXT_SCALE))
THICKNESS_THICK = max(2, int(2 * HUD_TEXT_SCALE))
# โหมด Test: capture รอบจุดคลิก — ใหญ่ขึ้น 2.5x จากเดิม (150,100) → (375,250)
TEST_CAPTURE_HALF_W = 375
TEST_CAPTURE_HALF_H = 250

# Colors BGR
COLOR_GRID_PENDING = (160, 160, 160)
COLOR_GRID_DONE = (0, 200, 0)
COLOR_NEXT_HIGHLIGHT = (0, 255, 255)
COLOR_CROSSHAIR = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)

# Keys (keyboard)
KEY_MODE_CALIB = ord("c")
KEY_MODE_TEST = ord("t")
KEY_SAVE = ord("s")
KEY_GRID_TOGGLE = ord("g")  # โหมด Test: สลับใช้แค่ grid (ไม่ใช้ homography)
KEY_PX_DEG_TOGGLE = ord("d")  # โหมด Test: สลับใช้ pixel-per-degree (linear) แทน homography
KEY_HOME = ord("h")  # โหมด Test: สั่งกลับ home (0,0)
KEY_RECALIB = ord("r")  # โหมด Test: re-calibrate จาก log (click_pixel → actual_arm) แล้ว save
KEY_P = ord("p")  # โหมด Test: continuous test — คลิกไปเรื่อยๆ ไม่กลับ home, variable step
KEY_L = ord("l")  # โหมด Test: sim track — เป้าจากจำลองการบินโดรน (ไม่คลิก), variable step เหมือน P
KEY_NEXT_PATTERN = ord("]")  # โหมด L: สลับ pattern การบิน (ตรง, วงกลม, แปด, ซิกแซก, วงรี, สุ่ม)
KEY_SPEED_LOCK = ord("k")  # โหมด L: สลับ speed lock — Auto / Fast only / Slow only / Stop only
KEY_TELEMETRY_PLOT = ord("v")  # โหมด Test: แสดงกราฟ telemetry โหมด P/L (Error, Velocity, Position vs time)
KEY_J = ord("j")  # โหมด Test: Teaching — บันทึก transition สำหรับเทรนโมเดล
KEY_TEACHING_SOURCE = ord("a")  # ในโหมด J: สลับ Click / Auto
KEY_TEACHING_USE_MODEL = ord("m")  # ในโหมด J: สลับใช้ Model / ไม่ใช้
KEY_TEACHING_RETRAIN = ord("f")  # ในโหมด J: รีเทรนโมเดล (ค้างจนเสร็จแล้วแจ้งความแม่นยำ)
KEY_ESC = 27
KEY_ENTER = 13  # Enter เพื่อยืนยันตอน verify หลัง Save
KEY_ENTER_LF = 10
# Arrow keys: Linux 81/82/83/84, Mac extended 63234/63232/63235/63233,
# Jetson/OpenCV GTK extended 65361/65362/65363/65364 (same as 22 adjust_center)
KEY_ARROW_LEFT = 81
KEY_ARROW_UP = 82
KEY_ARROW_RIGHT = 83
KEY_ARROW_DOWN = 84
KEY_ARROW_LEFT_MAC = 63234
KEY_ARROW_UP_MAC = 63232
KEY_ARROW_RIGHT_MAC = 63235
KEY_ARROW_DOWN_MAC = 63233
KEY_ARROW_LEFT_JETSON = 65361
KEY_ARROW_UP_JETSON = 65362
KEY_ARROW_RIGHT_JETSON = 65363
KEY_ARROW_DOWN_JETSON = 65364
ARROW_KEYS_LEFT = (KEY_ARROW_LEFT, KEY_ARROW_LEFT_MAC, KEY_ARROW_LEFT_JETSON)
ARROW_KEYS_RIGHT = (KEY_ARROW_RIGHT, KEY_ARROW_RIGHT_MAC, KEY_ARROW_RIGHT_JETSON)
ARROW_KEYS_UP = (KEY_ARROW_UP, KEY_ARROW_UP_MAC, KEY_ARROW_UP_JETSON)
ARROW_KEYS_DOWN = (KEY_ARROW_DOWN, KEY_ARROW_DOWN_MAC, KEY_ARROW_DOWN_JETSON)
ARROW_STEP_DEG = 0.1
KEY_N = ord("n")  # โหมด Test: crop + CSRT — จุดส้ม=center bbox (แยกจาก L เมาส์)
# CSRT (โหมด N): จาก gun_aim_assist_crop_csrt
TRACK_SCALE = 0.25
CSRT_MIN_BBOX = 4
CSRT_SMOOTH_ALPHA = 0.28
N_MODE_R_MAX_PX = 10  # r จาก error_px: error สูง → r สูงสุด 10 px, ลดลงตาม error_px จน r=0 เมื่อ error_px≈0

# Joystick button index 0 = confirm (ปุ่ม 1)
JOY_CONFIRM_BUTTON_INDEX = 0

# Smooth move: ระยะเวลา interpolate จากจุดเริ่มถึงเป้าหมาย (วินาที)
SMOOTH_MOVE_DURATION_SEC = 0.6
# โหมด P (continuous test): step และความเร็วจากสูตรต่อเนื่องตามระยะ (ไม่แบ่ง zone ขั้น)
# Step น้อยสุดที่แขนขยับได้ = 0.1 mm; สูตร max_step_mm = clamp(STEP_MIN_MM, f(error), STEP_MAX_MM)
NEAR_CROSSHAIR_THRESHOLD_PX = 120  # ระยะจากศูนย์เล็ง (px ที่ 1920): ≤ ใช้ px/deg, > ใช้ homography
STEP_MIN_MM = 0.1  # ขั้นต่ำที่แขนขยับได้
STEP_MAX_MM = 10.0  # step สูงสุดเมื่อไกล (mm)
# สัมประสิทธิ์สำหรับ f(error) ต่อเนื่อง: มี px_per_deg ใช้ error_px/scale; ไม่มีใช้ error_mm*scale
# scale เล็กลง → max_step_mm ใหญ่ขึ้น → เข้าเป้าเร็วขึ้น (จูนได้ผ่าน config)
CONTINUOUS_STEP_SCALE_PX = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_SCALE_PX", 18.0) if config else 18.0  # max_step_mm = error_px / scale
CONTINUOUS_STEP_SCALE_MM = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_SCALE_MM", 0.4) if config else 0.4   # fallback: max_step_mm = error_mm * scale
STEP_VERY_NEAR_MM = getattr(config, "CAM4_ARM_AIM_STEP_VERY_NEAR_MM", 0.1) if config else 0.1
STEP_NEAR_MM = getattr(config, "CAM4_ARM_AIM_STEP_NEAR_MM", 1.0) if config else 1.0
STEP_FAR_MM = getattr(config, "CAM4_ARM_AIM_STEP_FAR_MM", 10.0) if config else 10.0
ERROR_THRESHOLD_VERY_NEAR_MM = getattr(config, "CAM4_ARM_AIM_ERROR_VERY_NEAR_MM", 0.2) if config else 0.2
ERROR_THRESHOLD_NEAR_MM = getattr(config, "CAM4_ARM_AIM_ERROR_NEAR_MM", 0.5) if config else 0.5
CONTINUOUS_THROTTLE_SEC = getattr(config, "CAM4_ARM_CONTINUOUS_THROTTLE_SEC", 0.01) if config else 0.01  # throttle (ต่ำลง = ส่งบ่อย = เร็ว)
CONTINUOUS_DEADZONE_DEG = 0.02  # error น้อยกว่านี้ไม่ส่ง move (ต่ำลง = ยังขยับเมื่อใกล้เป้ามาก)
# โหมด P แบบ velocity เหมือนจอยสติก: ไกล = โยกมาก = เร็ว, ใกล้ = โยกน้อย = ช้า (smooth)
# โหมด P ใช้ค่าของตัวเอง (CONTINUOUS_*) ไม่แชร์กับจอยสติกจริง (JOYSTICK_*)
CONTINUOUS_MAX_RATE_DEG_PER_SEC = (
    getattr(config, "CAM4_ARM_CONTINUOUS_MAX_RATE_DEG", None) or getattr(config, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0)
) if config else 80.0
CONTINUOUS_REF_SCALE_DEG = getattr(config, "CAM4_ARM_AIM_REF_SCALE_DEG", 8.0) if config else 8.0  # error เกินนี้ = ความเร็วเต็ม
CONTINUOUS_STICK_EXPONENT = (
    getattr(config, "CAM4_ARM_CONTINUOUS_STICK_EXPONENT", None) or getattr(config, "CAM4_ARM_JOYSTICK_STICK_EXPONENT", 2.0)
) if config else 2.0  # power curve โหมด P
# โหมด P แบบ EV: ความเร็วสูงสุดอ้างอิงจาก CAM4_ARM_FEED_RATE (mm/min) → mm/s = FEED_RATE/60
_feed_rate_mm_min = getattr(config, "CAM4_ARM_FEED_RATE", 10000) if config else 10000
_max_speed_from_feed_mm_s = _feed_rate_mm_min / 60.0
_configured_max_speed = getattr(config, "CAM4_ARM_CONTINUOUS_MAX_SPEED_MM_S", _max_speed_from_feed_mm_s) if config else _max_speed_from_feed_mm_s
CONTINUOUS_MAX_SPEED_MM_S = min(_configured_max_speed, _max_speed_from_feed_mm_s)  # ไม่เกินที่แขนรับได้
CONTINUOUS_ACCEL_MM_S2 = getattr(config, "CAM4_ARM_CONTINUOUS_ACCEL_MM_S2", 400.0) if config else 400.0
CONTINUOUS_DECEL_MM_S2 = getattr(config, "CAM4_ARM_CONTINUOUS_DECEL_MM_S2", 400.0) if config else 400.0
CONTINUOUS_DEADZONE_MM = getattr(config, "CAM4_ARM_CONTINUOUS_DEADZONE_MM", 0.01) if config else 0.01  # เล็กมากเพื่อ error → 0
# Velocity profile: smoothstep so small error → small step (no jerk when clicking near target)
CONTINUOUS_VELOCITY_PROFILE_REF_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_VELOCITY_PROFILE_REF_DEG", 5.0) if config else 5.0
# ใกล้เป้า: ความเร็วขั้นต่ำเพื่อตามวัตถุทัน (smooth ไม่กระตุก)
CONTINUOUS_MIN_NEAR_SPEED_MM_S = getattr(config, "CAM4_ARM_CONTINUOUS_MIN_NEAR_SPEED_MM_S", 20.0) if config else 20.0
CONTINUOUS_NEAR_THRESHOLD_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_NEAR_THRESHOLD_DEG", 3.0) if config else 3.0
# ไกล: เพิ่ม step ให้ปิดระยะเร็ว (แนวทาง 2)
CONTINUOUS_FAST_THRESHOLD_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_FAST_THRESHOLD_DEG", 5.0) if config else 5.0
CONTINUOUS_FAST_STEP_BOOST = getattr(config, "CAM4_ARM_CONTINUOUS_FAST_STEP_BOOST", 2.0) if config else 2.0
# Smoothing error สำหรับสูตรจอยสติก (ลดกระตุกเมื่อ error กระเด้ง)
CONTINUOUS_ERROR_SMOOTH_ALPHA = getattr(config, "CAM4_ARM_CONTINUOUS_ERROR_SMOOTH_ALPHA", 0.4) if config else 0.4
# Fallback เมื่อไม่มี px_per_deg: ใช้สูตรต่อเนื่องจาก error_mm เท่านั้น (ไม่ใช้ zone ขั้น)
CONTINUOUS_ZONE_NEAR_MM = 0.5
CONTINUOUS_ZONE_MID_MM = 5.0
CONTINUOUS_STEP_FAR_MM = 10.0  # ใช้ร่วมกับ STEP_MAX_MM
# โหมด P zone step (ตามโดรนทัน มั่นยำ ไม่กระตุก): อ่านจาก config
CONTINUOUS_P_ZONE_FAR_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_ZONE_FAR_DEG", 5.0) if config else 5.0
CONTINUOUS_P_VERY_FAR_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_VERY_FAR_DEG", 10.0) if config else 10.0
CONTINUOUS_P_ZONE_NEAR_DEG = getattr(config, "CAM4_ARM_CONTINUOUS_ZONE_NEAR_DEG", 0.5) if config else 0.5
CONTINUOUS_P_STEP_FAR_MM = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_FAR_MM", 3.0) if config else 3.0
CONTINUOUS_P_STEP_VERY_FAR_MM = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_VERY_FAR_MM", 8.0) if config else 8.0
CONTINUOUS_P_STEP_MID_MM = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_MID_MM", 1.0) if config else 1.0
CONTINUOUS_P_STEP_NEAR_MM = getattr(config, "CAM4_ARM_CONTINUOUS_STEP_NEAR_MM", 0.1) if config else 0.1
CONTINUOUS_P_THROTTLE_FAR_SEC = getattr(config, "CAM4_ARM_CONTINUOUS_THROTTLE_FAR_SEC", 0.02) if config else 0.02
CONTINUOUS_P_THROTTLE_MID_SEC = getattr(config, "CAM4_ARM_CONTINUOUS_THROTTLE_MID_SEC", 0.02) if config else 0.02
CONTINUOUS_P_THROTTLE_NEAR_SEC = getattr(config, "CAM4_ARM_CONTINUOUS_THROTTLE_NEAR_SEC", 0.02) if config else 0.02
CONTINUOUS_ARM_FASTER_THAN_TARGET_FACTOR = getattr(config, "CAM4_ARM_CONTINUOUS_ARM_FASTER_THAN_TARGET_FACTOR", 1.2) if config else 1.2
# โหมด J Teaching sim ใกล้ 0,0 (หน่วย degree)
TEACHING_SIM_AMPLITUDE_DEG = getattr(config, "CAM4_ARM_TEACHING_SIM_AMPLITUDE_DEG", 4.0) if config else 4.0
# โหมด L sim โดรน: เป้ารอบ 0,0 ไม่เกินกี่องศา (ความเร็วสมจริง 9 แพตเทิร์น)
DRONE_SIM_MAX_DEG = getattr(config, "CAM4_ARM_DRONE_SIM_MAX_DEG", 15.0) if config else 15.0
# โหมด L: Kalman ทำนายตำแหน่งล่วงหน้า (lookahead วินาที) — แขนหมุนไปรอ ลดการสั่นกระตุก
SIM_KALMAN_LOOKAHEAD_SEC = getattr(config, "CAM4_ARM_SIM_KALMAN_LOOKAHEAD_SEC", 0.15) if config else 0.15
# สลับ pan/tilt ตอนส่ง G0: ถ้าแขนได้แค่ขึ้น-ลง ไม่ได้ซ้าย-ขวา ให้ True (ส่ง tilt→X, pan→Y)
SWAP_PAN_TILT = getattr(config, "CAM4_ARM_SWAP_PAN_TILT", True) if config else True
# กล้องติดแขน: True = ไม่คำนวณ/ไม่แสดง/ไม่ log error (pixel) เพราะภาพขยับตามแขน
CAMERA_ON_ARM = getattr(config, "CAM4_ARM_CAMERA_ON_ARM", True) if config else True

# Reticle: ขนาด fix สำหรับ Full HD (ไม่ใหญ่เกิน)
RETICLE_RADIUS_PX = (12, 20, 32, 52)  # วงใน -> วงนอก (พิกเซล)
# Reference panel: แสดงขนาดเท่าช่อง grid / capture รอบคลิก; ถ้าใหญ่เกิน scale ลงให้ fit ไม่เกินนี้ (4K ใช้ค่ามากขึ้น)
REFERENCE_PANEL_MAX_W = int(480 * HUD_TEXT_SCALE)
REFERENCE_PANEL_MAX_H = int(360 * HUD_TEXT_SCALE)
COLOR_HUD_BLACK = (0, 0, 0)
COLOR_HUD_RED = (0, 0, 255)  # BGR red for left HUD (bottom-left)
# Left HUD: bottom-left corner, row spacing
HUD_LEFT_LINE_H = int(22 * HUD_TEXT_SCALE)
HUD_BOTTOM_MARGIN = int(28 * HUD_TEXT_SCALE)
FONT_HUD = 0.65 * HUD_TEXT_SCALE  # slightly larger for left HUD
# Legacy Y constants (kept for any refs; actual Y computed from frame height at draw time)
Y_HUD_MSG = int(26 * HUD_TEXT_SCALE)
Y_HUD_MODE = int(50 * HUD_TEXT_SCALE)
Y_HUD_KEYS = int(66 * HUD_TEXT_SCALE)
Y_HUD_ARM = int(74 * HUD_TEXT_SCALE)
Y_HUD_TEST_MAP = int(92 * HUD_TEXT_SCALE)
Y_HUD_P = Y_HUD_TEST_MAP + HUD_LEFT_LINE_H
Y_HUD_L1 = Y_HUD_P + HUD_LEFT_LINE_H
Y_HUD_L2 = Y_HUD_L1 + HUD_LEFT_LINE_H
Y_HUD_L3 = Y_HUD_L2 + HUD_LEFT_LINE_H
Y_HUD_J1 = Y_HUD_L3 + HUD_LEFT_LINE_H
COLOR_CLICK_MARKER = (0, 0, 255)   # BGR แดง = จุดที่คลิก
COLOR_CROSSHAIR_IN_REF = (0, 255, 0)  # BGR เขียว = ศูนย์เล็ง (เทียบกับจุดคลิก)
COLOR_PATH_YELLOW = (0, 255, 255)  # BGR เหลือง = เส้นทาง crosshair → จุดคลิก
COLOR_ERROR_VECTOR = (255, 0, 255)  # BGR ม่วง = เส้นจากจุดคลิก → จุดที่แขนชี้จริง (error ในพิกเซล)
COLOR_ACTUAL_POINT = (255, 255, 0)  # BGR ฟ้า = จุดที่แขนชี้จริงบนจอ


def _coarse_order(center_r: int, center_c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """ลำดับเซลล์ coarse: เริ่มจากกลาง แล้วเรียงตามระยะจากกลาง."""
    out = [(center_r, center_c)]
    seen = {(center_r, center_c)}
    for radius in range(1, max(rows, cols) + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = center_r + dr, center_c + dc
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in seen:
                    seen.add((r, c))
                    out.append((r, c))
    return out


def _fine_order(fine_rows: int, fine_cols: int) -> List[Tuple[int, int]]:
    """ลำดับจุด fine: จากศูนย์กลางออกไป (แถวเรียง)."""
    cr, cc = fine_rows // 2, fine_cols // 2
    out = [(cr, cc)]
    seen = {(cr, cc)}
    for radius in range(1, max(fine_rows, fine_cols) + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = cr + dr, cc + dc
                if 0 <= r < fine_rows and 0 <= c < fine_cols and (r, c) not in seen:
                    seen.add((r, c))
                    out.append((r, c))
    return out


def _initial_pan_tilt_from_fov(px: float, py: float, cx: float, cy: float, w: int, h: int) -> Tuple[float, float]:
    """มุมเริ่มต้นจาก FOV (ใช้เมื่อยังไม่มีเซลล์เพื่อนบ้าน)."""
    if config is None:
        return 0.0, 0.0
    try:
        cfg = config.get_camera_config("cam4")
        fov_h = cfg.get("fov_horizontal", 60.0)
        fov_v = cfg.get("fov_vertical", 36.0)
    except Exception:
        fov_h, fov_v = 60.0, 36.0
    dx = (px - cx) / w * fov_h
    dy = -(py - cy) / h * fov_v
    return dx, dy


def _initial_pan_tilt_from_cells(
    row: int,
    col: int,
    cells: Dict[str, List[float]],
    center_r: int,
    center_c: int,
    fallback_cx: float,
    fallback_cy: float,
    px: float,
    py: float,
    w: int,
    h: int,
) -> Tuple[float, float]:
    """มุมเริ่มต้นจากเซลล์ที่สอนแล้ว (เพื่อนบ้านหรือ center)."""
    key = f"{row}_{col}"
    if key in cells and cells[key]:
        return cells[key][0], cells[key][1]
    for nr, nc in [(center_r, center_c), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
        k = f"{nr}_{nc}"
        if k in cells and cells[k]:
            return cells[k][0], cells[k][1]
    return _initial_pan_tilt_from_fov(px, py, fallback_cx, fallback_cy, w, h)


# -----------------------------------------------------------------------------
# App state
# -----------------------------------------------------------------------------
class CalibratorState:
    def __init__(self) -> None:
        self.mode = "calibration"  # "calibration" | "test"
        self.output_w = 0
        self.output_h = 0
        self.crosshair_x = 0.0
        self.crosshair_y = 0.0
        self.cells: Dict[str, List[float]] = {}
        self.center_fine_grid: Dict[str, Any] = {"fine_rows": FINE_ROWS, "fine_cols": FINE_COLS, "fine_cells": {}}
        self.coarse_order: List[Tuple[int, int]] = []
        self.fine_order: List[Tuple[int, int]] = []
        self.coarse_index = 0
        self.fine_index = 0
        self.phase = "coarse"  # "coarse" | "fine"
        self.pending_click: Optional[Tuple[int, int, int, int]] = None  # (row, col, px_out, py_out) or (fi, fj, ...) for fine
        self.moved_for_pending = False  # ส่งแขนไปแล้วสำหรับ pending นี้ (ไม่ส่งซ้ำทุกเฟรม)
        self.display_w = 0
        self.display_h = 0
        # ภาพจริงกินพื้นที่แค่ส่วนหนึ่งของหน้าต่าง (fullscreen letterbox: แถบดำซ้าย-ขวา/บน-ล่าง)
        # ถ้าไม่หัก offset ออก พิกัดเมาส์จะเพี้ยน — ตอน cam4 (16:9 เท่าจอ) offset=0 เลยไม่เคยโผล่
        # แต่กล้อง 4:3 บนจอ 16:9 จะเพี้ยนถึง ~240px = คาลิเบรตออกมาผิดทั้งชุด
        self.content_x = 0      # ขอบซ้ายของภาพในหน้าต่าง (px)
        self.content_y = 0
        self.content_w = 0      # ขนาดภาพที่วาดจริง (0 = ไม่มีข้อมูล → ใช้ display_w/h ทั้งผืน)
        self.content_h = 0
        self.test_data: Optional[Dict[str, Any]] = None
        self.reference_capture: Optional[np.ndarray] = None  # crop เซลล์ที่คลิก แสดงเป็นภาพอ้างอิง
        self.save_message_until = 0.0  # แสดง "Saved!" จนถึงเวลานี้ (time.time())
        self.save_accuracy_text: Optional[str] = None  # ข้อความความแม่นยำหลัง Save (Mean/Max deg)
        # Smooth move: หมุนแบบ interpolate ไม่กระตุก
        self.smooth_target: Optional[Tuple[float, float]] = None  # (pan, tilt) เป้าหมาย
        self.smooth_start: Optional[Tuple[float, float]] = None  # (pan, tilt) จุดเริ่ม
        self.smooth_start_time = 0.0
        self.smooth_duration = 0.6  # วินาที
        self.limit_clipped_message_until = 0.0  # แสดง "Target clipped to limit" จนถึงเวลานี้
        # หลัง Save: รีเชคทีละจุด — หมุนไปมุมที่เก็บไว้ รอ ENTER ยืนยัน (หรือ ESC ข้าม)
        self.verify_points: List[Tuple[str, float, float]] = []  # (label, pan, tilt)
        self.verify_index: Optional[int] = None  # None = ไม่อยู่โหมด verify
        self.verify_last_moved_index = -1  # ส่ง move ไปแล้วสำหรับจุดนี้
        # โหมด Test: แสดง mapping ล่าสุด Pixel (Full HD) → Arm
        self.last_test_pixel: Optional[Tuple[float, float]] = None  # (px, py)
        self.last_test_arm: Optional[Tuple[float, float]] = None  # (X, Y)
        self.pending_test_capture: Optional[Tuple[float, float]] = None  # (px_out, py_out) รอ crop เฟรมถัดไป
        self.test_reference_capture: Optional[np.ndarray] = None  # crop รอบจุดคลิก แสดงใน panel
        self.test_use_grid_only: bool = False  # True = ใช้แค่ grid ไม่ใช้ homography (กด 'g' สลับ)
        self.use_pixel_per_degree: bool = False  # True = ใช้ linear px/deg แทน homography (กด 'd' สลับ)
        self.px_per_deg_x: Optional[float] = None  # pixel ต่อ 1 องศา แกน X (pan)
        self.px_per_deg_y: Optional[float] = None  # pixel ต่อ 1 องศา แกน Y (tilt)
        # Test preview: รอ ENTER ก่อน move; กลับ home ก่อนคลิกถัดไป; feedback Y/N + log
        self.pending_test_move: Optional[Tuple[float, float, float, float]] = None  # (cmd1, cmd2, px_out, py_out)
        self.test_awaiting_home: bool = False
        self.test_feedback_pending: bool = False  # รอ Y/N ว่าเป้าตรงหรือไม่
        self.test_feedback_choice_pending: bool = False
        self.test_home_required_message_until: float = 0.0  # แสดง "Go to home first" จนถึงเวลานี้
        self.last_test_actual_arm: Optional[Tuple[float, float]] = None  # ตำแหน่งแขนจริงหลัง move (จาก GRBL)
        self.last_test_actual_pixel: Optional[Tuple[float, float]] = None  # ตำแหน่งแขนจริงในพิกเซล (inverse mapping) สำหรับ error vector
        self.last_test_crosshair_pixel: Optional[Tuple[float, float]] = None  # x,y pixel ศูนย์เล็ง เก็บหลังแขนนิ่ง + เก็บอีกครั้งหลังขยับให้ตรง (กด ENTER ยืนยัน)
        self.test_awaiting_confirm: bool = False  # True หลัง move จบ — รอปรับลูกศรแล้วกด ENTER ยืนยัน ตอนนั้นเก็บ actual_arm + crosshair อีกครั้ง
        self.test_move_log_pending: bool = False  # True หลัง move จบ — รอกด H แล้วค่อย log
        self.recal_message_until: float = 0.0  # แสดง "Re-calibrated!" จนถึงเวลานี้
        self.recal_message_text: Optional[str] = None  # ข้อความเพิ่ม (เช่น accuracy)
        self.recal_accuracy_text: Optional[str] = None  # ค่าความแม่นยำหลัง recal แสดงค้างตลอด (จนกว่าจะ recal ใหม่)
        # โหมด P: continuous test — คลิกไปเรื่อยๆ ไม่กลับ home, variable step
        self.test_continuous: bool = False
        self.continuous_target_pan: Optional[float] = None
        self.continuous_target_tilt: Optional[float] = None
        self.last_continuous_arm_move_time: float = 0.0
        # เก็บเป้าเดิม + เวลา สำหรับคำนวณความเร็วเป้า (โดรน) — แขนต้องเร็วกว่าเป้า
        self.continuous_target_pan_prev: Optional[float] = None
        self.continuous_target_tilt_prev: Optional[float] = None
        self.continuous_target_time_prev: float = 0.0
        # โหมด P/L: ความเร็วปัจจุบัน (mm/s), telemetry, และ smoothing error สำหรับจอยสติก (ไม่กระตุก)
        self.continuous_velocity_mm_s: float = 0.0
        self.continuous_error_deg_smooth: Optional[float] = None  # สำหรับ ratio ในสูตรจอยสติก; reset เมื่อเป้าเปลี่ยน
        self.continuous_telemetry: deque = None  # set in __init__
        # โหมด L: sim track — เป้าจากจำลองการบินโดรน
        self.continuous_telemetry = deque(maxlen=3000)  # (t, pan, tilt, pan_tgt, tilt_tgt, error_mm, velocity_mm_s)
        self.test_track_sim: bool = False
        self.sim_t_start: float = 0.0
        self.sim_pattern_index: int = 0
        self.sim_random_px: float = 0.0
        self.sim_random_py: float = 0.0
        self.sim_display_px: Optional[float] = None  # สำหรับวาดจุดโดรนจำลองบนเฟรม
        self.sim_display_py: Optional[float] = None
        self.track_sim_arm_auto: bool = True  # โหมด L: True=Auto sim, False=Manual (ลากเมาส์บิน)
        self.sim_drag_active: bool = False  # โหมด L Manual: กำลังลากเมาส์ซ้ายค้าง
        self.sim_speed_state: str = "medium"  # โหมด L Auto: "stop"|"slow"|"medium"|"fast"
        self.sim_speed_switch_at: float = 0.0
        self.sim_speed_lock: Optional[str] = None  # None=สุ่มสลับ, "fast"|"slow"|"stop"=ใช้โหมดเดียว [K]
        self.sim_internal_time: float = 0.0  # เวลาเชิงจำลอง เดินตาม speed_scale
        self.last_sim_detection_time: float = 0.0
        self.sim_detection_interval_sec: float = 0.06  # ~15–20 Hz
        self.last_sim_update_time: float = 0.0  # สำหรับคำนวณ dt ให้ sim_internal_time
        self.sim_kalman_x: Optional[Any] = None  # Kalman state [px, py, vx, vy] — ทำนายตำแหน่งล่วงหน้า
        self.sim_kalman_P: Optional[Any] = None  # Kalman covariance 4x4
        self.sim_kalman_last_time: float = 0.0
        self.l_track_joystick_style: bool = False  # โหมด L: True=ไล่ตามแบบจอยสติก (smooth), False=variable step
        # โหมด J (Teaching): บันทึก transition สำหรับเทรนโมเดล
        self.test_teaching: bool = False
        self.teaching_source: str = "auto"  # "click" | "auto"
        self.teaching_use_model: bool = False
        self._teaching_model: Any = None
        self._teaching_predict_fn: Any = None
        self._teaching_normalize_fn: Any = None
        self._teaching_model_input_dim: int = 10
        self.teaching_csv_path: Optional[Path] = None
        self.teaching_csv_handle: Any = None
        self.teaching_pattern_index: int = 0
        self.teaching_pattern_switch_at: float = 0.0
        self.teaching_sim_start: float = 0.0
        self.teaching_last_state: Optional[Tuple[float, ...]] = None
        self.teaching_last_action: Optional[Tuple[float, float]] = None
        self.teaching_last_target: Optional[Tuple[float, float]] = None
        self.teaching_last_ts: Optional[float] = None
        self.teaching_last_err: Optional[Tuple[float, float, float]] = None
        self.teaching_last_use_model: Optional[bool] = None
        self.teaching_retrain_in_progress: bool = False
        self.teaching_retrain_result: Optional[str] = None
        self.teaching_last_sent_delta: Tuple[float, float] = (0.0, 0.0)
        self.teaching_t_prev_send: Optional[float] = None
        self.teaching_err_prev: Optional[Tuple[float, float]] = None
        # โหมด N (crop + CSRT): แยกจาก L (เมาส์)
        self.test_track_csrt: bool = False
        self.sim_csrt_tracker: Any = None
        self.sim_csrt_pending_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) output space
        self.sim_csrt_initialized: bool = False
        self.sim_csrt_bbox: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h) ล่าสุดสำหรับวาด
        self.sim_csrt_smooth_px: Optional[float] = None
        self.sim_csrt_smooth_py: Optional[float] = None
        self.sim_csrt_lost: bool = False
        self.sim_csrt_drag_start: Optional[Tuple[float, float]] = None
        self.sim_csrt_drag_current: Optional[Tuple[float, float]] = None
        self.sim_csrt_error_px: Optional[float] = None  # ระยะศูนย์เล็ง→center bbox (สำหรับ HUD)
        self.sim_csrt_r: Optional[float] = None  # ระยะ virtual จากศูนย์เล็ง (สำหรับ HUD)


_state = CalibratorState()

# ชื่อ pattern จำลองโดรนโหมด L (9 แบบ, หน่วย degree รอบ 0,0 ไม่เกิน DRONE_SIM_MAX_DEG, ความเร็วสมจริง)
SIM_PATTERN_NAMES = [
    "Hover", "Circle", "Figure-8", "Straight H", "Straight V", "Zigzag",
    "Ellipse", "Drift", "Back-forth",
]
# โหมด J Teaching: pattern ใกล้ 0,0 (หน่วย degree)
TEACHING_PATTERN_NAMES = [
    "Continuous", "Back-forth", "Up-down", "Hover", "Slow drift", "Random",
]


def _teaching_build_state_10(
    arm: Any,
    state: "CalibratorState",
    pan_target: float,
    tilt_target: float,
) -> Tuple[Tuple[float, ...], float, float, float]:
    """
    สร้าง state 10 มิติสำหรับ aim_controller_model: (err_pan, err_tilt, error_deg,
    last_delta_pan, last_delta_tilt, dt_send, d_pan, d_tilt, current_pan_deg, current_tilt_deg).
    คืน (state_tuple, err_pan, err_tilt, error_deg).
    """
    current_pan = getattr(arm, "pos_x", 0.0)
    current_tilt = getattr(arm, "pos_y", 0.0)
    err_pan = pan_target - current_pan
    err_tilt = tilt_target - current_tilt
    error_deg = math.hypot(err_pan, err_tilt)
    last_dp, last_dt = getattr(state, "teaching_last_sent_delta", (0.0, 0.0))
    t_prev = getattr(state, "teaching_t_prev_send", None)
    dt_send = (time.time() - t_prev) if t_prev is not None else 0.0
    err_prev = getattr(state, "teaching_err_prev", None)
    d_pan = (err_pan - err_prev[0]) / dt_send if err_prev is not None and dt_send > 1e-6 else 0.0
    d_tilt = (err_tilt - err_prev[1]) / dt_send if err_prev is not None and dt_send > 1e-6 else 0.0
    state_10 = (err_pan, err_tilt, error_deg, last_dp, last_dt, dt_send, d_pan, d_tilt, current_pan, current_tilt)
    return state_10, err_pan, err_tilt, error_deg


def _teaching_sim_target_near_origin(t_sec: float, state: "CalibratorState") -> Tuple[float, float]:
    """
    คืน (pan_target_deg, tilt_target_deg) อยู่แถว (0, 0) สำหรับโหมด J Auto.
    Amplitude เล็ก (TEACHING_SIM_AMPLITUDE_DEG), หลายแพตเทิร์น.
    """
    amp = TEACHING_SIM_AMPLITUDE_DEG
    n_patterns = len(TEACHING_PATTERN_NAMES)
    idx = getattr(state, "teaching_pattern_index", 0) % n_patterns
    if idx == 0:  # Continuous: วงกลมเล็ก
        period = 12.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        pan = amp * 0.8 * math.cos(angle)
        tilt = amp * 0.6 * math.sin(angle)
    elif idx == 1:  # Back-forth: pan แกว่ง
        period = 8.0
        phase = (t_sec % period) / period
        pan = amp * (1.0 if phase < 0.5 else -1.0) * (2 * (phase if phase < 0.5 else 1.0 - phase) * 2)
        tilt = amp * 0.2 * math.sin(t_sec * 0.5)
    elif idx == 2:  # Up-down: tilt แกว่ง
        period = 7.0
        phase = (t_sec % period) / period
        tilt = amp * (1.0 if phase < 0.5 else -1.0) * (2 * (phase if phase < 0.5 else 1.0 - phase) * 2)
        pan = amp * 0.2 * math.sin(t_sec * 0.4)
    elif idx == 3:  # Hover: แกว่งเล็กน้อย
        pan = amp * 0.3 * math.sin(t_sec * 0.6)
        tilt = amp * 0.3 * math.cos(t_sec * 0.8)
    elif idx == 4:  # Slow drift: random walk ช้า
        r = getattr(state, "_teaching_drift_pan", 0.0)
        s = getattr(state, "_teaching_drift_tilt", 0.0)
        state._teaching_drift_pan = r + 0.02 * (math.sin(t_sec * 1.1) + 0.3 * math.cos(t_sec * 2.3))
        state._teaching_drift_tilt = s + 0.02 * (math.cos(t_sec * 1.3) + 0.3 * math.sin(t_sec * 2.1))
        pan = max(-amp, min(amp, getattr(state, "_teaching_drift_pan", 0.0) * amp))
        tilt = max(-amp, min(amp, getattr(state, "_teaching_drift_tilt", 0.0) * amp))
    else:  # 5: Random
        dx = math.sin(t_sec * 5.1) * 0.5 + math.cos(t_sec * 7.3) * 0.3
        dy = math.cos(t_sec * 4.7) * 0.5 + math.sin(t_sec * 6.9) * 0.3
        rp = getattr(state, "_teaching_rand_pan", 0.0)
        rt = getattr(state, "_teaching_rand_tilt", 0.0)
        if t_sec < 0.05:
            rp, rt = 0.0, 0.0
        state._teaching_rand_pan = rp + dx * 0.15
        state._teaching_rand_tilt = rt + dy * 0.15
        pan = max(-amp, min(amp, getattr(state, "_teaching_rand_pan", 0.0)))
        tilt = max(-amp, min(amp, getattr(state, "_teaching_rand_tilt", 0.0)))
    return float(pan), float(tilt)


def _px_display_to_output(display_x: int, display_y: int) -> Tuple[float, float]:
    """แมปพิกัดเมาส์จากหน้าต่างแสดงผล → พิกัดใน output frame.

    หักพื้นที่ letterbox (แถบดำ) ออกก่อน: ตอน fullscreen ภาพถูก fit เข้าจอแล้ววางกลาง
    → พิกัดเมาส์นับจากขอบ 'หน้าต่าง' ไม่ใช่ขอบ 'ภาพ'. ถ้าไม่หัก คลิกจะเพี้ยนตามขนาดแถบดำ
    (cam4 16:9 บนจอ 16:9 → แถบ = 0 เลยไม่เคยเห็นบั๊ก; กล้อง 4:3 → เพี้ยน ~240px)
    """
    if _state.display_w <= 0 or _state.display_h <= 0:
        return float(display_x), float(display_y)
    cw = _state.content_w or _state.display_w
    ch = _state.content_h or _state.display_h
    x = (float(display_x) - _state.content_x) * _state.output_w / cw
    y = (float(display_y) - _state.content_y) * _state.output_h / ch
    return max(0.0, min(_state.output_w - 1.0, x)), max(0.0, min(_state.output_h - 1.0, y))


def _pan_tilt_from_click_near_far(
    px_out: float, py_out: float, state: "CalibratorState"
) -> Tuple[Optional[float], Optional[float]]:
    """คืน (pan_deg, tilt_deg) จากคลิก: ใกล้ศูนย์เล็ง → px/deg, ไกล → homography. ไม่ clip limit."""
    if state.output_w <= 0 or state.output_h <= 0:
        return None, None
    cx = state.crosshair_x
    cy = state.crosshair_y
    dist = math.sqrt((px_out - cx) ** 2 + (py_out - cy) ** 2)
    near_thresh = NEAR_CROSSHAIR_THRESHOLD_PX * (state.output_w / 1920.0)
    if dist <= near_thresh and state.px_per_deg_x is not None and state.px_per_deg_y is not None:
        pan = (px_out - cx) / state.px_per_deg_x
        tilt = (py_out - cy) / state.px_per_deg_y
        return pan, tilt
    if state.test_data and pixel_to_arm_degrees:
        res = pixel_to_arm_degrees(
            px_out, py_out, state.test_data, REF_OUTPUT_W, REF_OUTPUT_H, use_homography=True
        )
        if res is not None:
            return res[0], res[1]
    if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
        pan = (px_out - cx) / state.px_per_deg_x
        tilt = (py_out - cy) / state.px_per_deg_y
        return pan, tilt
    pan, tilt = _initial_pan_tilt_from_fov(
        px_out, py_out, cx, cy, REF_OUTPUT_W, REF_OUTPUT_H
    )
    return pan, tilt


def _l_manual_target_from_pixel(
    px_out: float, py_out: float, state: "CalibratorState", arm: Any
) -> Optional[Tuple[float, float]]:
    """โหมด L Manual: คืน (cmd1, cmd2) จาก pixel สำหรับ continuous target. คืน None ถ้าคำนวณไม่ได้."""
    if state.output_w <= 0 or state.output_h <= 0:
        return None
    if hasattr(arm, "sync_position_from_grbl"):
        try:
            arm.sync_position_from_grbl()
        except Exception:
            pass
    pan_cur = getattr(arm, "pos_x", 0.0)
    tilt_cur = getattr(arm, "pos_y", 0.0)
    cx, cy = state.crosshair_x, state.crosshair_y
    if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
        dx_px = px_out - cx
        dy_px = py_out - cy
        delta_pan = dx_px / state.px_per_deg_x
        delta_tilt = dy_px / state.px_per_deg_y
        if CAMERA_ON_ARM:
            target_pan = pan_cur + delta_pan
            target_tilt = tilt_cur + delta_tilt
        else:
            target_pan = delta_pan
            target_tilt = delta_tilt
    else:
        target_pan, target_tilt = _initial_pan_tilt_from_fov(
            px_out, py_out, cx, cy, REF_OUTPUT_W, REF_OUTPUT_H
        )
        if target_pan is None and target_tilt is None:
            return None
        target_pan = 0.0 if target_pan is None else target_pan
        target_tilt = 0.0 if target_tilt is None else target_tilt
    x_lim = getattr(arm, "_effective_x_limits", None)
    y_lim = getattr(arm, "_effective_y_limits", None)
    if x_lim is None or y_lim is None:
        margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
        xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
        yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
        x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
        y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
    x_lo, x_hi = x_lim[0], x_lim[1]
    y_lo, y_hi = y_lim[0], y_lim[1]
    if SWAP_PAN_TILT:
        cmd1 = float(np.clip(target_tilt, x_lo, x_hi))
        cmd2 = float(np.clip(target_pan, y_lo, y_hi))
    else:
        cmd1 = float(np.clip(target_pan, x_lo, x_hi))
        cmd2 = float(np.clip(target_tilt, y_lo, y_hi))
    return (cmd1, cmd2)


def _drone_sim_target_deg(
    t_sec: float,
    pattern_index: int,
    state: "CalibratorState",
) -> Tuple[float, float]:
    """
    คืน (pan_deg, tilt_deg) รอบ (0, 0) ในช่วง [-DRONE_SIM_MAX_DEG, +DRONE_SIM_MAX_DEG].
    โหมด L: 9 แพตเทิร์นรวม Hover ความเร็วสมจริง (period ยาว 12–22 s).
    """
    amp = DRONE_SIM_MAX_DEG
    n_patterns = 9
    idx = pattern_index % n_patterns

    if idx == 0:  # Hover: บิน hold แกว่งเล็กน้อยรอบศูนย์, period ~15 s
        period = 15.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        pan = amp * 0.25 * math.sin(angle * 0.8)
        tilt = amp * 0.25 * math.cos(angle * 0.7)
    elif idx == 1:  # Circle, period ~18 s
        period = 18.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        pan = amp * 0.9 * math.cos(angle)
        tilt = amp * 0.9 * math.sin(angle)
    elif idx == 2:  # Figure-8 (Lissajous), period ~20 s
        period = 20.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        pan = amp * 0.9 * math.sin(angle)
        tilt = amp * 0.55 * math.sin(2.0 * angle)
    elif idx == 3:  # Straight H: แนวนอน ซ้าย↔ขวา, period ~16 s
        period = 16.0
        phase = (t_sec % period) / period
        pan = amp * 0.9 * (2.0 * (phase if phase < 0.5 else 1.0 - phase) * 2.0 - 1.0)
        tilt = amp * 0.12 * math.sin(t_sec * 0.4)
    elif idx == 4:  # Straight V: แนวตั้ง ขึ้น↔ลง, period ~14 s
        period = 14.0
        phase = (t_sec % period) / period
        tilt = amp * 0.9 * (2.0 * (phase if phase < 0.5 else 1.0 - phase) * 2.0 - 1.0)
        pan = amp * 0.12 * math.sin(t_sec * 0.35)
    elif idx == 5:  # Zigzag, period ~18 s
        period = 18.0
        seg = (t_sec % period) / period
        n_seg = 6
        i = min(int(seg * n_seg), n_seg - 1)
        frac = (seg * n_seg) - i
        step = (2.0 * amp * 0.9) / n_seg
        if i % 2 == 0:
            tilt = -amp * 0.9 + (i + frac) * step
        else:
            tilt = -amp * 0.9 + (i + 1 - frac) * step
        pan = amp * 0.7 if i % 2 == 0 else -amp * 0.7
    elif idx == 6:  # Ellipse, period ~20 s
        period = 20.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        pan = amp * 0.85 * math.cos(angle)
        tilt = amp * 0.45 * math.sin(angle)
    elif idx == 7:  # Drift: เคลื่อนช้า step เล็กต่อ t
        r = getattr(state, "_drone_sim_drift_pan", 0.0)
        s = getattr(state, "_drone_sim_drift_tilt", 0.0)
        state._drone_sim_drift_pan = r + 0.008 * (math.sin(t_sec * 0.9) + 0.25 * math.cos(t_sec * 1.8))
        state._drone_sim_drift_tilt = s + 0.008 * (math.cos(t_sec * 0.85) + 0.25 * math.sin(t_sec * 1.7))
        pan = max(-amp, min(amp, state._drone_sim_drift_pan * amp))
        tilt = max(-amp, min(amp, state._drone_sim_drift_tilt * amp))
    else:  # 8: Back-forth, period ~14 s
        period = 14.0
        phase = (t_sec % period) / period
        pan = amp * 0.9 * (2.0 * (phase if phase < 0.5 else 1.0 - phase) * 2.0 - 1.0)
        tilt = amp * 0.15 * math.sin(t_sec * 0.5)

    pan = max(-amp, min(amp, pan))
    tilt = max(-amp, min(amp, tilt))
    return pan, tilt


def _kalman_update_and_predict(
    state: "CalibratorState",
    px_meas: float,
    py_meas: float,
    now: float,
    lookahead_sec: float,
    output_w: int,
    output_h: int,
) -> Tuple[float, float]:
    """
    Kalman filter (constant velocity): อัปเดตด้วยการวัด (px_meas, py_meas) แล้วคืนตำแหน่งทำนายล่วงหน้า lookahead_sec.
    State: [px, py, vx, vy]. ใช้ทำนายตำแหน่งโดรน — แขนหมุนไปรอ ลดการสั่นกระตุก.
    """
    if state.sim_kalman_x is None or state.sim_kalman_P is None:
        state.sim_kalman_x = np.array([px_meas, py_meas, 0.0, 0.0], dtype=float)
        state.sim_kalman_P = np.eye(4, dtype=float) * 100.0
        state.sim_kalman_last_time = now
        return (px_meas, py_meas)

    dt = now - state.sim_kalman_last_time
    state.sim_kalman_last_time = now
    if dt <= 0:
        dt = 0.02
    dt = min(dt, 0.5)

    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    x = state.sim_kalman_x.copy()
    P = state.sim_kalman_P.copy()
    # Predict
    x = F @ x
    P = F @ P @ F.T
    q = 30.0
    Q = np.eye(4, dtype=float) * q * dt
    Q[0, 0] = Q[1, 1] = q * (dt ** 4) / 4.0
    Q[2, 2] = Q[3, 3] = q * (dt ** 2)
    P = P + Q

    # Update
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    z = np.array([px_meas, py_meas], dtype=float)
    y = z - (H @ x)
    R = np.eye(2, dtype=float) * 4.0
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P

    state.sim_kalman_x = x
    state.sim_kalman_P = P

    px_pred = float(x[0] + x[2] * lookahead_sec)
    py_pred = float(x[1] + x[3] * lookahead_sec)
    px_pred = max(0.0, min(output_w - 1.0, px_pred))
    py_pred = max(0.0, min(output_h - 1.0, py_pred))
    return (px_pred, py_pred)


def _kalman_predict_only(
    state: "CalibratorState",
    now: float,
    lookahead_sec: float,
    output_w: int,
    output_h: int,
) -> Optional[Tuple[float, float]]:
    """
    รันเฉพาะ predict step (ไม่มี measurement update). ใช้เมื่อไม่มี detection เพื่อให้แขนเคลื่อนต่อตาม trajectory ทำนาย.
    คืน (px_pred, py_pred) หรือ None ถ้าไม่มี state.
    """
    if state.sim_kalman_x is None or state.sim_kalman_P is None:
        return None
    dt = now - state.sim_kalman_last_time
    state.sim_kalman_last_time = now
    if dt <= 0:
        dt = 0.02
    dt = min(dt, 0.5)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    x = state.sim_kalman_x.copy()
    P = state.sim_kalman_P.copy()
    x = F @ x
    P = F @ P @ F.T
    q = 30.0
    Q = np.eye(4, dtype=float) * q * dt
    Q[0, 0] = Q[1, 1] = q * (dt ** 4) / 4.0
    Q[2, 2] = Q[3, 3] = q * (dt ** 2)
    P = P + Q
    state.sim_kalman_x = x
    state.sim_kalman_P = P
    px_pred = float(x[0] + x[2] * lookahead_sec)
    py_pred = float(x[1] + x[3] * lookahead_sec)
    px_pred = max(0.0, min(output_w - 1.0, px_pred))
    py_pred = max(0.0, min(output_h - 1.0, py_pred))
    return (px_pred, py_pred)


def _drone_sim_position(
    t_sec: float,
    output_w: int,
    output_h: int,
    pattern_index: int,
    state: "CalibratorState",
) -> Tuple[float, float]:
    """
    คืน (px, py) ใน output space จากจำลองการบินโดรน ตาม pattern_index.
    โหมด L ใช้ _drone_sim_target_deg แทน; ฟังก์ชันนี้เก็บไว้สำหรับอ้างอิงอื่น.
    """
    if output_w <= 0 or output_h <= 0:
        return float(output_w) / 2.0, float(output_h) / 2.0
    cx = float(output_w) / 2.0
    cy = float(output_h) / 2.0
    margin = 0.15
    r_min = margin * min(output_w, output_h)
    r_max = (0.5 - margin) * min(output_w, output_h)
    n_patterns = len(SIM_PATTERN_NAMES)
    idx = pattern_index % n_patterns

    if idx == 0:  # Straight: ซ้าย→ขวา แล้วกลับ
        period = 12.0
        phase = (t_sec % period) / period
        if phase < 0.5:
            px = cx - r_max + (2 * phase) * (2 * r_max)
        else:
            px = cx + r_max - (2 * (phase - 0.5)) * (2 * r_max)
        py = cy
    elif idx == 1:  # Circle
        period = 10.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        radius = (r_min + r_max) / 2.0
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
    elif idx == 2:  # Figure-8 (Lissajous)
        period = 14.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        rx = (r_min + r_max) / 2.0
        ry = rx * 0.6
        px = cx + rx * math.sin(angle)
        py = cy + ry * math.sin(2.0 * angle)
    elif idx == 3:  # Zigzag: แนวตั้ง
        period = 8.0
        seg = (t_sec % period) / period
        n_seg = 6
        i = min(int(seg * n_seg), n_seg - 1)
        frac = (seg * n_seg) - i
        y0 = cy - r_max
        y1 = cy + r_max
        step = (y1 - y0) / n_seg
        if i % 2 == 0:
            py = y0 + (i + frac) * step
        else:
            py = y0 + (i + 1 - frac) * step
        px = cx + (r_max * 0.5 if i % 2 == 0 else -r_max * 0.5)
    elif idx == 4:  # Ellipse
        period = 11.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        rx = (r_min + r_max) * 0.8
        ry = (r_min + r_max) * 0.4
        px = cx + rx * math.cos(angle)
        py = cy + ry * math.sin(angle)
    elif idx == 5:  # Random walk (deterministic noise จากเวลา — ลื่นกว่า hash)
        speed = 120.0
        if t_sec < 0.05:
            state.sim_random_px = cx
            state.sim_random_py = cy
        dx = math.sin(t_sec * 5.1) * 0.7 + math.cos(t_sec * 7.3) * 0.3
        dy = math.cos(t_sec * 4.7) * 0.7 + math.sin(t_sec * 6.9) * 0.3
        state.sim_random_px = state.sim_random_px + dx * speed * 0.033
        state.sim_random_py = state.sim_random_py + dy * speed * 0.033
        state.sim_random_px = max(cx - r_max, min(cx + r_max, state.sim_random_px))
        state.sim_random_py = max(cy - r_max, min(cy + r_max, state.sim_random_py))
        px = state.sim_random_px
        py = state.sim_random_py
    elif idx == 6:  # Hover: นิ่งหรือแกว่งเล็กน้อย
        amp = 18.0
        px = cx + amp * math.sin(t_sec * 0.6)
        py = cy + amp * math.cos(t_sec * 0.8)
    elif idx == 7:  # Small wander: เคลื่อนเล็กน้อยวนไปมา
        r_small = 0.06 * min(output_w, output_h)
        period = 18.0
        angle = 2.0 * math.pi * (t_sec % period) / period
        px = cx + r_small * (math.sin(angle) + 0.3 * math.sin(angle * 3))
        py = cy + r_small * (math.cos(angle) + 0.3 * math.cos(angle * 2))
    elif idx == 8:  # Straight slow: บินช้า ซ้าย↔ขวา
        period = 25.0
        phase = (t_sec % period) / period
        if phase < 0.5:
            px = cx - r_max + (2 * phase) * (2 * r_max)
        else:
            px = cx + r_max - (2 * (phase - 0.5)) * (2 * r_max)
        py = cy
    elif idx == 9:  # Straight fast: บินเร็ว ซ้าย↔ขวา
        period = 4.0
        phase = (t_sec % period) / period
        if phase < 0.5:
            px = cx - r_max + (2 * phase) * (2 * r_max)
        else:
            px = cx + r_max - (2 * (phase - 0.5)) * (2 * r_max)
        py = cy
    elif idx == 10:  # Fly up: บินขึ้น (py ลดลง)
        period = 10.0
        phase = (t_sec % period) / period
        py = cy - r_max + phase * (2 * r_max)
        px = cx
    elif idx == 11:  # Fly down: บินลง (py เพิ่มขึ้น)
        period = 10.0
        phase = (t_sec % period) / period
        py = cy + r_max - phase * (2 * r_max)
        px = cx
    else:  # 12: Back-forth แนวตั้ง ขึ้น↔ลง
        period = 9.0
        phase = (t_sec % period) / period
        if phase < 0.5:
            py = cy - r_max + (2 * phase) * (2 * r_max)
        else:
            py = cy + r_max - (2 * (phase - 0.5)) * (2 * r_max)
        px = cx

    return max(0.0, min(output_w - 1.0, px)), max(0.0, min(output_h - 1.0, py))


def _apply_arm_move_relative(arm: Any, delta_pan: float, delta_tilt: float) -> None:
    """ส่ง move_relative ตาม SWAP_PAN_TILT.
    Simulation mode: อัปเดต pos_x/pos_y โดยไม่ส่ง G-code จริง (arm.move_relative handles this)."""
    if arm is None:
        return
    if SWAP_PAN_TILT:
        cmd1, cmd2 = delta_tilt, delta_pan
    else:
        cmd1, cmd2 = delta_pan, delta_tilt
    arm.move_relative(cmd1, cmd2, blocking=False)


def _variable_step_toward_target(
    arm: Any,
    pan_target: float,
    tilt_target: float,
    last_arm_move_time: float,
    throttle_sec: float,
    state: "CalibratorState",
    deadzone_deg: Optional[float] = None,
    step_scale: Optional[float] = None,
    profile_ref_deg: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    ขยับแขนไปทางเป้าโหมด P/L: มีระยะห่างแล้วพุ่งเข้าไปตาม vector เต็มที่ (min(ระยะเหลือ, feed×dt)).
    แขนต้องเร็วกว่าเป้า (step อย่างน้อย = ความเร็วเป้า×dt×ตัวคูณ). cap ด้วย error_mm และ feed rate.
    คืน (last_move_time, move_pan, move_tilt).
    Simulation mode: อัปเดต pos_x/pos_y โดยไม่ส่ง G-code จริง.
    """
    if arm is None:
        return last_arm_move_time, 0.0, 0.0
    now = time.time()
    if (now - last_arm_move_time) < throttle_sec:
        return last_arm_move_time, 0.0, 0.0
    dt = now - last_arm_move_time
    current_pan = getattr(arm, "pos_x", 0.0)
    current_tilt = getattr(arm, "pos_y", 0.0)
    delta_pan = pan_target - current_pan
    delta_tilt = tilt_target - current_tilt
    error_deg = math.hypot(delta_pan, delta_tilt)
    mm_per_deg_pan = getattr(arm, "mm_per_deg_pan", getattr(config, "CAM4_ARM_MM_PER_DEG_PAN", 1.0) if config else 1.0)
    mm_per_deg_tilt = getattr(arm, "mm_per_deg_tilt", getattr(config, "CAM4_ARM_MM_PER_DEG_TILT", 1.0) if config else 1.0)
    mm_per_deg = (mm_per_deg_pan + mm_per_deg_tilt) / 2.0
    if mm_per_deg <= 1e-9:
        return last_arm_move_time, 0.0, 0.0
    error_mm = error_deg * mm_per_deg

    # Deadzone (mm): หยุดส่ง move เมื่อ error_mm ≤ DEADZONE
    if error_mm <= CONTINUOUS_DEADZONE_MM:
        state.continuous_velocity_mm_s = 0.0
        return last_arm_move_time, 0.0, 0.0

    # ความเร็วเป้า (โดรน) จากการเปลี่ยนตำแหน่ง — แขนต้องอย่างน้อยเท่าความเร็วเป้า จึงจะตามทัน
    dt_target = now - getattr(state, "continuous_target_time_prev", now - 0.05)
    if dt_target >= 0.01:
        prev_pan = getattr(state, "continuous_target_pan_prev", None)
        prev_tilt = getattr(state, "continuous_target_tilt_prev", None)
        if prev_pan is None or prev_tilt is None:
            prev_pan, prev_tilt = pan_target, tilt_target
        target_velocity_deg_s = math.hypot(pan_target - prev_pan, tilt_target - prev_tilt) / dt_target
        target_speed_mm_s = target_velocity_deg_s * mm_per_deg
        min_step_to_match_target_mm = target_speed_mm_s * dt
    else:
        min_step_to_match_target_mm = 0.0
    state.continuous_target_pan_prev = pan_target
    state.continuous_target_tilt_prev = tilt_target
    state.continuous_target_time_prev = now

    # มีระยะห่าง = พุ่งเข้าไปตาม vector เต็มที่ (ไม่รอ): step = min(ระยะที่เหลือ, ความเร็วสูงสุด×dt)
    max_step_by_feed = dt * _max_speed_from_feed_mm_s if dt > 1e-6 else error_mm
    # แขนต้องเร็วกว่าเป้า: อย่างน้อย = ความเร็วเป้า × dt × ตัวคูณ (floor สำหรับตามเป้าเคลื่อนที่ให้ทัน)
    min_step_arm = min_step_to_match_target_mm * CONTINUOUS_ARM_FASTER_THAN_TARGET_FACTOR
    # 'closing step' (ปิดระยะที่เหลือ) — ใช้ velocity profile ให้นุ่มตอนเข้าใกล้เป้านิ่ง
    step_close = min(error_mm, max_step_by_feed)
    # ref เล็ก = ปิดระยะเต็มที่ (สำหรับ LOCK ตามเป้าเคลื่อนที่); ref ใหญ่ = นุ่ม (คลิกจุดนิ่ง)
    ref_deg = CONTINUOUS_VELOCITY_PROFILE_REF_DEG if profile_ref_deg is None else profile_ref_deg
    if ref_deg > 1e-9:
        t = min(1.0, error_deg / ref_deg)
        v_scale = t * t * (3.0 - 2.0 * t)
        v_scale = max(v_scale, 0.01)
        step_close = step_close * v_scale
    # ⚠️ keep-up floor ต้อง 'รอด' จาก velocity profile — ไม่งั้นแขนคลานตอนตามเป้าเคลื่อนที่ใกล้ ๆ
    # (capped ที่ error_mm อยู่แล้ว → ไม่ overshoot เลยเป้า)
    step_mm = max(step_close, min_step_arm, STEP_MIN_MM)
    step_mm = min(step_mm, error_mm, max_step_by_feed)
    if step_mm < 1e-6:
        return last_arm_move_time, 0.0, 0.0

    # Optional step_scale (e.g. from yellow vector cap in HUD CSRT/ORB): 0 < step_scale <= 1
    if step_scale is not None and 0.0 < step_scale <= 1.0:
        step_mm = step_mm * step_scale
        if step_mm < 1e-6:
            return last_arm_move_time, 0.0, 0.0

    # แปลง step_mm → move_pan, move_tilt (รักษาทิศทาง)
    scale_deg = (step_mm / mm_per_deg) / error_deg if error_deg > 1e-9 else 0.0
    move_pan = delta_pan * scale_deg
    move_tilt = delta_tilt * scale_deg
    _apply_arm_move_relative(arm, move_pan, move_tilt)

    state.continuous_velocity_mm_s = step_mm / dt if dt > 1e-9 else 0.0
    # Telemetry สำหรับกราฟจูน F1
    tel = getattr(state, "continuous_telemetry", None)
    if tel is not None:
        tel.append((now, current_pan, current_tilt, pan_target, tilt_target, error_mm, state.continuous_velocity_mm_s))

    return now, move_pan, move_tilt


def _l_joystick_style_toward_target(
    arm: Any,
    pan_target: float,
    tilt_target: float,
    last_arm_move_time: float,
    throttle_sec: float,
    state: "CalibratorState",
) -> Tuple[float, float, float]:
    """
    โหมด L แบบจอยสติกจำลอง: error → virtual stick magnitude → rate*dt → move_relative.
    คืน (last_move_time, move_pan, move_tilt). ส่ง move เฉพาะเมื่อผ่าน throttle แล้ว.
    Simulation mode: อัปเดต pos_x/pos_y โดยไม่ส่ง G-code จริง.
    """
    if arm is None:
        return last_arm_move_time, 0.0, 0.0
    now = time.time()
    if (now - last_arm_move_time) < throttle_sec:
        return last_arm_move_time, 0.0, 0.0
    dt = now - last_arm_move_time
    current_pan = getattr(arm, "pos_x", 0.0)
    current_tilt = getattr(arm, "pos_y", 0.0)
    delta_pan = pan_target - current_pan
    delta_tilt = tilt_target - current_tilt
    error_deg = math.hypot(delta_pan, delta_tilt)
    if error_deg <= CONTINUOUS_DEADZONE_DEG:
        state.continuous_velocity_mm_s = 0.0
        return last_arm_move_time, 0.0, 0.0
    ref_scale = CONTINUOUS_REF_SCALE_DEG
    exponent = CONTINUOUS_STICK_EXPONENT
    max_rate = CONTINUOUS_MAX_RATE_DEG_PER_SEC
    # virtual stick magnitude 0..1 จาก error (power curve: ใกล้เป้า = ช้า)
    mag = min(1.0, error_deg / ref_scale) if ref_scale > 1e-9 else 1.0
    mag = (1.0 if mag > 0 else 0.0) * (abs(mag) ** exponent)
    rate_deg_s = max_rate * mag
    step_deg = rate_deg_s * dt
    step_deg = min(step_deg, error_deg)
    if step_deg < 1e-6:
        return last_arm_move_time, 0.0, 0.0
    scale = (step_deg / error_deg) if error_deg > 1e-9 else 0.0
    move_pan = delta_pan * scale
    move_tilt = delta_tilt * scale
    _apply_arm_move_relative(arm, move_pan, move_tilt)
    mm_per_deg = (getattr(arm, "mm_per_deg_pan", 1.0) + getattr(arm, "mm_per_deg_tilt", 1.0)) / 2.0
    state.continuous_velocity_mm_s = (step_deg * mm_per_deg / dt) if dt > 1e-9 else 0.0
    tel = getattr(state, "continuous_telemetry", None)
    if tel is not None:
        tel.append((now, current_pan, current_tilt, pan_target, tilt_target, error_deg * mm_per_deg, state.continuous_velocity_mm_s))
    return now, move_pan, move_tilt


def _show_continuous_telemetry_plot(state: "CalibratorState", live: bool = False) -> None:
    """
    แสดงกราฟจูนแบบ F1 จาก buffer telemetry โหมด P/L: Error vs t, Velocity vs t, Pan/Tilt vs t.
    live: ถ้า True อัปเดต real-time (ไม่ implement ในนี้ แสดง snapshot); optional บันทึก CSV.
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tel = getattr(state, "continuous_telemetry", None)
    if tel is None or len(tel) == 0:
        return
    # ลำดับ: (t, pan, tilt, pan_target, tilt_target, error_mm, velocity_mm_s)
    t0 = tel[0][0]
    t = [row[0] - t0 for row in tel]
    pan = [row[1] for row in tel]
    tilt = [row[2] for row in tel]
    pan_tgt = [row[3] for row in tel]
    tilt_tgt = [row[4] for row in tel]
    error_mm = [row[5] for row in tel]
    velocity_mm_s = [row[6] for row in tel]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, error_mm, color="C0")
    axes[0].set_ylabel("Error (mm)")
    axes[0].set_title("Error vs Time")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, velocity_mm_s, color="C1")
    axes[1].set_ylabel("Velocity (mm/s)")
    axes[1].set_title("Velocity vs Time")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, pan, label="pan", color="C2")
    axes[2].plot(t, tilt, label="tilt", color="C3")
    axes[2].plot(t, pan_tgt, "--", label="pan_target", color="C2", alpha=0.7)
    axes[2].plot(t, tilt_tgt, "--", label="tilt_target", color="C3", alpha=0.7)
    axes[2].set_ylabel("deg")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Position vs Time")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    # บันทึก CSV (optional)
    try:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = CALIBRATION_DIR / f"continuous_tune_{ts}.csv"
        with open(csv_path, "w") as f:
            f.write("t,pan,tilt,pan_target,tilt_target,error_mm,velocity_mm_s\n")
            for row in tel:
                f.write(",".join(str(x) for x in row) + "\n")
    except Exception:
        pass
    plt.show()


def _draw_grids(
    frame: np.ndarray,
    state: CalibratorState,
    next_coarse: Optional[Tuple[int, int]],
    next_fine: Optional[Tuple[int, int]],
) -> None:
    """วาด coarse grid ทั้งจอ + fine grid ในช่องกลาง; สีตามสถานะ; ไฮไลต์เซลล์ถัดไป."""
    h, w = frame.shape[:2]
    if state.output_w <= 0 or state.output_h <= 0:
        return
    scale_x = w / state.output_w
    scale_y = h / state.output_h

    center_r = GRID_ROWS // 2
    center_c = GRID_COLS // 2
    cell_w = state.output_w / GRID_COLS
    cell_h = state.output_h / GRID_ROWS

    # Coarse grid
    for row in range(GRID_ROWS + 1):
        y0 = int(row * cell_h * scale_y)
        cv2.line(frame, (0, y0), (w, y0), COLOR_GRID_PENDING, 1)
    for col in range(GRID_COLS + 1):
        x0 = int(col * cell_w * scale_x)
        cv2.line(frame, (x0, 0), (x0, h), COLOR_GRID_PENDING, 1)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            key = f"{row}_{col}"
            done = key in state.cells and state.cells[key]
            is_next = next_coarse == (row, col)
            x0 = int(col * cell_w * scale_x)
            y0 = int(row * cell_h * scale_y)
            x1 = int((col + 1) * cell_w * scale_x)
            y1 = int((row + 1) * cell_h * scale_y)
            if is_next:
                cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_NEXT_HIGHLIGHT, 3)
            elif done:
                cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_GRID_DONE, 2)

    # Fine grid ในช่องกลาง
    left = center_c * cell_w
    top = center_r * cell_h
    cw = cell_w
    ch = cell_h
    fw = cw / FINE_COLS
    fh = ch / FINE_ROWS
    for fi in range(FINE_ROWS + 1):
        for fj in range(FINE_COLS + 1):
            fx = left + fj * fw
            fy = top + fi * fh
            ix = int(fx * scale_x)
            iy = int(fy * scale_y)
            if fi < FINE_ROWS + 1 and fj < FINE_COLS + 1:
                if fi < FINE_ROWS and fj < FINE_COLS:
                    key = f"{fi}_{fj}"
                    done = key in state.center_fine_grid.get("fine_cells", {})
                    is_next = next_fine == (fi, fj)
                    ix0 = int((left + fj * fw) * scale_x)
                    iy0 = int((top + fi * fh) * scale_y)
                    ix1 = int((left + (fj + 1) * fw) * scale_x)
                    iy1 = int((top + (fi + 1) * fh) * scale_y)
                    if is_next:
                        cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), COLOR_NEXT_HIGHLIGHT, 2)
                    elif done:
                        cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), COLOR_GRID_DONE, 1)
            if fi < FINE_ROWS + 1 and fj < FINE_COLS + 1:
                cv2.circle(frame, (ix, iy), 1, COLOR_GRID_PENDING, 1)

    # Reticle (ตำแหน่งจาก JSON; ขนาด fix ตาม REF Full HD)
    cx = int(state.crosshair_x)
    cy = int(state.crosshair_y)
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

    # จุดที่เมาส์คลิกตอน calibration — กากบาทสีแดงบนจอหลักให้เห็นชัด
    if state.mode == "calibration" and state.pending_click is not None and len(state.pending_click) >= 4:
        px_click = int(state.pending_click[2])
        py_click = int(state.pending_click[3])
        click_cross_len = int(30 * HUD_TEXT_SCALE)
        click_cross_thick = max(2, int(4 * HUD_TEXT_SCALE))
        cv2.line(
            frame,
            (max(0, px_click - click_cross_len), py_click),
            (min(w, px_click + click_cross_len), py_click),
            COLOR_CLICK_MARKER,
            click_cross_thick,
        )
        cv2.line(
            frame,
            (px_click, max(0, py_click - click_cross_len)),
            (px_click, min(h, py_click + click_cross_len)),
            COLOR_CLICK_MARKER,
            click_cross_thick,
        )
    # โหมด Test: ยกเลิกปักหมุด — ไม่วาดจุดคลิก/เส้น error (เก็บค่าไว้ใช้ใน log เท่านั้น)
    # โหมด Test: เส้นสีเหลืองจาก crosshair ไปจุดคลิก (เส้นทางที่จะหมุนไป)
    if state.mode == "test" and getattr(state, "pending_test_move", None) is not None:
        pt1 = (int(state.crosshair_x), int(state.crosshair_y))
        pt2 = (int(state.pending_test_move[2]), int(state.pending_test_move[3]))
        cv2.line(frame, pt1, pt2, COLOR_PATH_YELLOW, 3)
        cv2.circle(frame, pt2, 8, COLOR_PATH_YELLOW, 2)
    # โหมด L: จุดสีส้ม (ตำแหน่งเมาส์) — โหมด N ใช้จุดแดง + เส้นเวกเตอร์ แยกต่างหาก
    if state.mode == "test" and getattr(state, "test_track_sim", False):
        px = getattr(state, "sim_display_px", None)
        py = getattr(state, "sim_display_py", None)
        if px is not None and py is not None:
            ix = int(px * scale_x) if state.output_w > 0 else int(px)
            iy = int(py * scale_y) if state.output_h > 0 else int(py)
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(frame, (ix, iy), 12, (0, 165, 255), 3)  # BGR ส้ม
                cv2.circle(frame, (ix, iy), 4, (0, 200, 255), -1)
    # โหมด N หลัง CSRT init: จุดแดงที่ center bbox + เส้นเวกเตอร์จากศูนย์เล็งไปจุดแดง
    if state.mode == "test" and getattr(state, "test_track_csrt", False) and getattr(state, "sim_csrt_initialized", False) and not getattr(state, "sim_csrt_lost", False) and state.output_w > 0 and state.output_h > 0:
        bx = getattr(state, "sim_csrt_smooth_px", None)
        by = getattr(state, "sim_csrt_smooth_py", None)
        if bx is not None and by is not None:
            ix_red = int(bx * scale_x)
            iy_red = int(by * scale_y)
            ix_ch = int(state.crosshair_x * scale_x)
            iy_ch = int(state.crosshair_y * scale_y)
            ix_red = max(0, min(ix_red, w - 1))
            iy_red = max(0, min(iy_red, h - 1))
            ix_ch = max(0, min(ix_ch, w - 1))
            iy_ch = max(0, min(iy_ch, h - 1))
            cv2.line(frame, (ix_ch, iy_ch), (ix_red, iy_red), (0, 0, 255), 2)  # BGR red = vector
            cv2.circle(frame, (ix_red, iy_red), 10, (0, 0, 255), 3)  # BGR red = bbox center
            cv2.circle(frame, (ix_red, iy_red), 4, (0, 0, 255), -1)
            # โหมด N: จุดส้ม = ตำแหน่งนำทาง (ระยะ r จากศูนย์เล็ง) เหมือน L — บอกว่าเมาส์ต้องชี้ตรงไหน
            px = getattr(state, "sim_display_px", None)
            py = getattr(state, "sim_display_py", None)
            if px is not None and py is not None and state.output_w > 0 and state.output_h > 0:
                ix = int(px * scale_x)
                iy = int(py * scale_y)
                if 0 <= ix < w and 0 <= iy < h:
                    cv2.circle(frame, (ix, iy), 12, (0, 165, 255), 3)  # BGR orange = guidance point
                    cv2.circle(frame, (ix, iy), 4, (0, 200, 255), -1)
    # โหมด N ยังไม่ init: วาดสี่เหลี่ยมขณะลากเลือก bbox
    if state.mode == "test" and getattr(state, "test_track_csrt", False) and not getattr(state, "sim_csrt_initialized", False):
        ds = getattr(state, "sim_csrt_drag_start", None)
        dc = getattr(state, "sim_csrt_drag_current", None)
        if ds is not None and dc is not None and state.output_w > 0 and state.output_h > 0:
            x1, y1 = ds[0], ds[1]
            x2, y2 = dc[0], dc[1]
            ix1 = int(x1 * scale_x)
            iy1 = int(y1 * scale_y)
            ix2 = int(x2 * scale_x)
            iy2 = int(y2 * scale_y)
            ix1 = max(0, min(ix1, w - 1))
            iy1 = max(0, min(iy1, h - 1))
            ix2 = max(0, min(ix2, w - 1))
            iy2 = max(0, min(iy2, h - 1))
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
    # โหมด N หลัง CSRT init: วาด bbox วัตถุที่ติดตาม
    if state.mode == "test" and getattr(state, "test_track_csrt", False) and getattr(state, "sim_csrt_bbox", None) is not None and state.output_w > 0 and state.output_h > 0:
        xb, yb, wb, hb = state.sim_csrt_bbox
        ix0 = int(xb * scale_x)
        iy0 = int(yb * scale_y)
        ix1 = int((xb + wb) * scale_x)
        iy1 = int((yb + hb) * scale_y)
        ix0 = max(0, min(ix0, w - 1))
        iy0 = max(0, min(iy0, h - 1))
        ix1 = max(0, min(ix1, w))
        iy1 = max(0, min(iy1, h))
        if ix1 > ix0 and iy1 > iy0:
            cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (0, 255, 0), 2)

    # Reference panel: แสดงภาพ capture ขนาดเท่าช่อง grid + จุดคลิก(แดง) และศูนย์เล็ง(เขียว)
    if state.reference_capture is not None:
        ref = state.reference_capture
        ph, pw = ref.shape[0], ref.shape[1]
        # ถ้าใหญ่เกิน scale ลงให้ fit มุมจอ
        if pw > REFERENCE_PANEL_MAX_W or ph > REFERENCE_PANEL_MAX_H:
            scale_ref = min(REFERENCE_PANEL_MAX_W / pw, REFERENCE_PANEL_MAX_H / ph, 1.0)
            pw_d = int(pw * scale_ref)
            ph_d = int(ph * scale_ref)
            ref_draw = cv2.resize(ref, (pw_d, ph_d), interpolation=cv2.INTER_LINEAR)
        else:
            ref_draw = ref
            pw_d, ph_d = pw, ph
        margin = 10
        ox = w - pw_d - margin
        oy = h - ph_d - margin
        if ox >= 0 and oy >= 0:
            roi = frame[oy : oy + ph_d, ox : ox + pw_d]
            if roi.shape[:2] == ref_draw.shape[:2]:
                np.copyto(roi, ref_draw)
            else:
                ref_fit = cv2.resize(ref_draw, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                np.copyto(roi, ref_fit)
            cv2.rectangle(frame, (ox, oy), (ox + pw_d, oy + ph_d), COLOR_NEXT_HIGHLIGHT, 2)
            label = "Adjust joystick then Button 0"
            cv2.putText(frame, label, (ox, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, COLOR_TEXT, THICKNESS_THICK)
            cv2.putText(frame, label, (ox, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, COLOR_HUD_BLACK, THICKNESS_THIN)
    # โหมด Test: panel แสดง crop รอบจุดคลิก — เปรียบเทียบว่าแขนเล็งตรงจุดนี้หรือไม่
    if state.mode == "test" and state.test_reference_capture is not None:
        ref = state.test_reference_capture
        ph, pw = ref.shape[0], ref.shape[1]
        if pw > REFERENCE_PANEL_MAX_W or ph > REFERENCE_PANEL_MAX_H:
            scale_ref = min(REFERENCE_PANEL_MAX_W / pw, REFERENCE_PANEL_MAX_H / ph, 1.0)
            pw_d = int(pw * scale_ref)
            ph_d = int(ph * scale_ref)
            ref_draw = cv2.resize(ref, (pw_d, ph_d), interpolation=cv2.INTER_LINEAR)
        else:
            ref_draw = ref
            pw_d, ph_d = pw, ph
        margin = 10
        ox = w - pw_d - margin
        oy = h - ph_d - margin
        if ox >= 0 and oy >= 0:
            roi = frame[oy : oy + ph_d, ox : ox + pw_d]
            if roi.shape[:2] == ref_draw.shape[:2]:
                np.copyto(roi, ref_draw)
            else:
                ref_fit = cv2.resize(ref_draw, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                np.copyto(roi, ref_fit)
            cv2.rectangle(frame, (ox, oy), (ox + pw_d, oy + ph_d), COLOR_NEXT_HIGHLIGHT, 2)
            label = "Target (arm should point here)"
            cv2.putText(frame, label, (ox, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, COLOR_TEXT, THICKNESS_THICK)
            cv2.putText(frame, label, (ox, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, COLOR_HUD_BLACK, THICKNESS_THIN)


def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
    state = _state
    # โหมด N (crop + CSRT): LBUTTONUP — สร้าง pending_bbox จาก drag
    if event == cv2.EVENT_LBUTTONUP:
        if (
            state.mode == "test"
            and getattr(state, "test_track_csrt", False)
            and getattr(state, "sim_csrt_drag_start", None) is not None
            and getattr(state, "sim_csrt_drag_current", None) is not None
            and state.output_w > 0
            and state.output_h > 0
        ):
            x1, y1 = state.sim_csrt_drag_start
            x2, y2 = state.sim_csrt_drag_current
            x_min = max(0, min(x1, x2))
            x_max = min(state.output_w, max(x1, x2))
            y_min = max(0, min(y1, y2))
            y_max = min(state.output_h, max(y1, y2))
            bw = x_max - x_min
            bh = y_max - y_min
            if bw >= CSRT_MIN_BBOX and bh >= CSRT_MIN_BBOX:
                state.sim_csrt_pending_bbox = (int(x_min), int(y_min), int(bw), int(bh))
            state.sim_csrt_drag_start = None
            state.sim_csrt_drag_current = None
            return
        if getattr(state, "sim_drag_active", False):
            state.sim_drag_active = False
        return
    # โหมด L Manual: MOUSEMOVE — อัปเดตเป้าตามตำแหน่งลาก
    if event == cv2.EVENT_MOUSEMOVE:
        # โหมด N (crop + CSRT): กำลังลากเลือก bbox
        if (
            state.mode == "test"
            and getattr(state, "test_track_csrt", False)
            and getattr(state, "sim_csrt_drag_start", None) is not None
            and state.display_w > 0
            and state.display_h > 0
        ):
            px_out, py_out = _px_display_to_output(x, y)
            state.sim_csrt_drag_current = (px_out, py_out)
        # โหมด L: จุดสีส้ม = ตำแหน่งเมาส์ — เฉพาะเมื่อไม่เปิดโหมด N (ไม่ให้เมาส์เขียนทับค่าจาก CSRT)
        elif (
            state.mode == "test"
            and getattr(state, "test_track_sim", False)
            and not getattr(state, "test_track_csrt", False)
            and state.display_w > 0
            and state.display_h > 0
        ):
            px_out, py_out = _px_display_to_output(x, y)
            state.sim_display_px = px_out
            state.sim_display_py = py_out
        if (
            state.mode == "test"
            and getattr(state, "test_track_sim", False)
            and not getattr(state, "track_sim_arm_auto", True)
            and getattr(state, "sim_drag_active", False)
        ):
            if isinstance(param, dict) and "arm" in param and state.output_w > 0 and state.output_h > 0:
                arm = param["arm"]
                px_out, py_out = _px_display_to_output(x, y)
                res = _l_manual_target_from_pixel(px_out, py_out, state, arm)
                if res is not None:
                    state.continuous_target_pan, state.continuous_target_tilt = res
                    state.continuous_velocity_mm_s = 0.0
                    state.continuous_error_deg_smooth = None
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    px_out, py_out = _px_display_to_output(x, y)
    if state.mode == "test":
        if not isinstance(param, dict) or "arm" not in param:
            return
        arm = param["arm"]
        if state.output_w <= 0 or state.output_h <= 0:
            return
        # โหมด N (crop + CSRT): เริ่มลากเลือก bbox
        if getattr(state, "test_track_csrt", False) and not getattr(state, "sim_csrt_initialized", False):
            state.sim_csrt_drag_start = (px_out, py_out)
            state.sim_csrt_drag_current = None
            return
        # โหมด L Manual: เริ่มลาก — ตั้ง sim_drag_active และเป้าจาก (x,y)
        if getattr(state, "test_track_sim", False) and not getattr(state, "track_sim_arm_auto", True):
            state.sim_drag_active = True
            res = _l_manual_target_from_pixel(px_out, py_out, state, arm)
            if res is not None:
                state.continuous_target_pan, state.continuous_target_tilt = res
                state.continuous_velocity_mm_s = 0.0
                state.continuous_error_deg_smooth = None
            return
        # โหมด P: continuous test — คลิกได้เรื่อยๆ ไม่ต้องอยู่ home
        # โหมด P ใช้ px/deg เสมอเมื่อมีค่า (ทั้งใกล้และไกล ไม่เรียก homography)
        if getattr(state, "test_continuous", False):
            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            pan_cur = getattr(arm, "pos_x", 0.0)
            tilt_cur = getattr(arm, "pos_y", 0.0)
            cx = state.crosshair_x
            cy = state.crosshair_y
            if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                # โหมด P: ใช้ px/deg เสมอ (ไม่ใช้ homography)
                dx_px = px_out - cx
                dy_px = py_out - cy
                delta_pan = dx_px / state.px_per_deg_x
                delta_tilt = dy_px / state.px_per_deg_y
                if CAMERA_ON_ARM:
                    target_pan = pan_cur + delta_pan
                    target_tilt = tilt_cur + delta_tilt
                else:
                    target_pan = delta_pan
                    target_tilt = delta_tilt
            else:
                # ไม่มี px_per_deg: fallback ไม่ใช้ homography (ใช้ FOV)
                target_pan, target_tilt = _initial_pan_tilt_from_fov(
                    px_out, py_out, cx, cy, REF_OUTPUT_W, REF_OUTPUT_H
                )
                if target_pan is None and target_tilt is None:
                    return
                if target_pan is None:
                    target_pan = 0.0
                if target_tilt is None:
                    target_tilt = 0.0
            x_lim = getattr(arm, "_effective_x_limits", None)
            y_lim = getattr(arm, "_effective_y_limits", None)
            if x_lim is None or y_lim is None:
                margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
                xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
                yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
                x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
                y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
            x_lo, x_hi = x_lim[0], x_lim[1]
            y_lo, y_hi = y_lim[0], y_lim[1]
            if SWAP_PAN_TILT:
                cmd1 = float(np.clip(target_tilt, x_lo, x_hi))
                cmd2 = float(np.clip(target_pan, y_lo, y_hi))
            else:
                cmd1 = float(np.clip(target_pan, x_lo, x_hi))
                cmd2 = float(np.clip(target_tilt, y_lo, y_hi))
            # เก็บเป้าในลำดับเดียวกับ pos_x, pos_y (เมื่อ SWAP: pos_x=tilt, pos_y=pan)
            state.continuous_target_pan = cmd1
            state.continuous_target_tilt = cmd2
            state.continuous_velocity_mm_s = 0.0
            state.continuous_error_deg_smooth = None  # เป้าใหม่ → reset smoothing
            state.pending_test_capture = (px_out, py_out)
            return
        # โหมด J (Click): ตั้งเป้าจากคลิกเหมือน P
        if getattr(state, "test_teaching", False) and getattr(state, "teaching_source", "auto") == "click":
            if hasattr(arm, "sync_position_from_grbl"):
                try:
                    arm.sync_position_from_grbl()
                except Exception:
                    pass
            pan_cur = getattr(arm, "pos_x", 0.0)
            tilt_cur = getattr(arm, "pos_y", 0.0)
            cx, cy = state.crosshair_x, state.crosshair_y
            if CAMERA_ON_ARM and state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                dx_px = px_out - cx
                dy_px = py_out - cy
                delta_pan = dx_px / state.px_per_deg_x
                delta_tilt = dy_px / state.px_per_deg_y
                target_pan = pan_cur + delta_pan
                target_tilt = tilt_cur + delta_tilt
            else:
                target_pan, target_tilt = _pan_tilt_from_click_near_far(px_out, py_out, state)
                if target_pan is None:
                    target_pan = 0.0
                if target_tilt is None:
                    target_tilt = 0.0
            x_lim = getattr(arm, "_effective_x_limits", None)
            y_lim = getattr(arm, "_effective_y_limits", None)
            if x_lim is None or y_lim is None:
                margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
                xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
                yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
                x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
                y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
            x_lo, x_hi = x_lim[0], x_lim[1]
            y_lo, y_hi = y_lim[0], y_lim[1]
            if SWAP_PAN_TILT:
                cmd1 = float(np.clip(target_tilt, x_lo, x_hi))
                cmd2 = float(np.clip(target_pan, y_lo, y_hi))
            else:
                cmd1 = float(np.clip(target_pan, x_lo, x_hi))
                cmd2 = float(np.clip(target_tilt, y_lo, y_hi))
            state.continuous_target_pan = cmd1
            state.continuous_target_tilt = cmd2
            state.continuous_velocity_mm_s = 0.0
            state.continuous_error_deg_smooth = None
            return
        if getattr(state, "test_awaiting_home", False):
            return
        if getattr(state, "test_awaiting_confirm", False):
            return  # รอปรับลูกศรแล้วกด ENTER หรือกด H กลับ home ก่อน
        if hasattr(arm, "sync_position_from_grbl"):
            try:
                arm.sync_position_from_grbl()
            except Exception:
                pass
        pos_x = getattr(arm, "pos_x", 0.0)
        pos_y = getattr(arm, "pos_y", 0.0)
        if abs(pos_x) > 1.0 or abs(pos_y) > 1.0:
            state.test_home_required_message_until = time.time() + 3.0
            return
        target_pan, target_tilt = None, None
        if getattr(state, "use_pixel_per_degree", False) and state.px_per_deg_x is not None and state.px_per_deg_y is not None:
            cx, cy = state.crosshair_x, state.crosshair_y
            target_pan = (px_out - cx) / state.px_per_deg_x
            target_tilt = (py_out - cy) / state.px_per_deg_y
        elif state.test_data and pixel_to_arm_degrees:
            use_homography = not getattr(state, "test_use_grid_only", False)
            res = pixel_to_arm_degrees(
                px_out, py_out, state.test_data, REF_OUTPUT_W, REF_OUTPUT_H,
                use_homography=use_homography,
            )
            if res is not None:
                target_pan, target_tilt = res[0], res[1]
        if target_pan is None or target_tilt is None:
            target_pan, target_tilt = _initial_pan_tilt_from_fov(
                px_out, py_out,
                state.crosshair_x, state.crosshair_y,
                REF_OUTPUT_W, REF_OUTPUT_H,
            )
        x_lim = getattr(arm, "_effective_x_limits", None)
        y_lim = getattr(arm, "_effective_y_limits", None)
        if x_lim is None or y_lim is None:
            margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
            xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
            yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
            x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
            y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
        x_lo, x_hi = x_lim[0], x_lim[1]
        y_lo, y_hi = y_lim[0], y_lim[1]
        if SWAP_PAN_TILT:
            cmd1 = float(np.clip(target_tilt, x_lo, x_hi))
            cmd2 = float(np.clip(target_pan, y_lo, y_hi))
            clipped = (target_tilt != cmd1) or (target_pan != cmd2)
        else:
            cmd1 = float(np.clip(target_pan, x_lo, x_hi))
            cmd2 = float(np.clip(target_tilt, y_lo, y_hi))
            clipped = (target_pan != cmd1) or (target_tilt != cmd2)
        if clipped:
            state.limit_clipped_message_until = time.time() + 2.0
        state.pending_test_move = (cmd1, cmd2, px_out, py_out)
        state.last_test_pixel = (px_out, py_out)
        state.last_test_arm = (cmd1, cmd2)
        state.last_test_actual_arm = None
        state.last_test_actual_pixel = None
        state.last_test_crosshair_pixel = None
        state.test_awaiting_confirm = False
        state.test_move_log_pending = False  # คลิกจุดใหม่ = ไม่ log จุดเก่าตอนกด H
        state.pending_test_capture = (px_out, py_out)
        return
    # Calibration mode: กำหนดเซลล์ที่คลิก
    cell_w = state.output_w / GRID_COLS
    cell_h = state.output_h / GRID_ROWS
    col = int(px_out / cell_w)
    row = int(py_out / cell_h)
    col = max(0, min(GRID_COLS - 1, col))
    row = max(0, min(GRID_ROWS - 1, row))
    center_r, center_c = GRID_ROWS // 2, GRID_COLS // 2
    if state.phase == "coarse":
        # คลิกที่ไหนก็แสดง reference; ยืนยันได้เฉพาะเมื่อคลิกเซลล์ถัดไป
        state.pending_click = (row, col, int(px_out), int(py_out))
        state.reference_capture = None
    else:
        if (row, col) != (center_r, center_c):
            return
        local_x = px_out - center_c * cell_w
        local_y = py_out - center_r * cell_h
        fj = int(local_x / (cell_w / FINE_COLS))
        fi = int(local_y / (cell_h / FINE_ROWS))
        fj = max(0, min(FINE_COLS - 1, fj))
        fi = max(0, min(FINE_ROWS - 1, fi))
        state.pending_click = (fi, fj, int(px_out), int(py_out))
        state.reference_capture = None


def _get_calibration_points_from_data(
    data: Dict[str, Any], ow: int, oh: int
) -> Tuple[List[List[float]], List[List[float]]]:
    """คืน (src_pts, dst_pts) จาก data (cells + fine_cells ใน JSON). src = [px, py], dst = [pan, tilt]."""
    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    cells = data.get("cells") or {}
    if not isinstance(cells, dict):
        return src_pts, dst_pts
    grid_cols = int(data.get("grid_cols", GRID_COLS))
    grid_rows = int(data.get("grid_rows", GRID_ROWS))
    cell_w = ow / max(1, grid_cols)
    cell_h = oh / max(1, grid_rows)
    center_r, center_c = grid_rows // 2, grid_cols // 2
    for key, val in cells.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0]), int(parts[1])
            px = (c + 0.5) * cell_w
            py = (r + 0.5) * cell_h
            src_pts.append([px, py])
            dst_pts.append([float(val[0]), float(val[1])])
        except (ValueError, IndexError):
            continue
    center_fine = data.get("center_fine_grid")
    if isinstance(center_fine, dict):
        fine_cells = center_fine.get("fine_cells") or {}
        fine_rows = int(center_fine.get("fine_rows", FINE_ROWS))
        fine_cols = int(center_fine.get("fine_cols", FINE_COLS))
        left = center_c * cell_w
        top = center_r * cell_h
        fw = cell_w / max(1, fine_cols)
        fh = cell_h / max(1, fine_rows)
        for key, val in fine_cells.items():
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                continue
            parts = key.split("_")
            if len(parts) != 2:
                continue
            try:
                fi, fj = int(parts[0]), int(parts[1])
                px = left + (fj + 0.5) * fw
                py = top + (fi + 0.5) * fh
                src_pts.append([px, py])
                dst_pts.append([float(val[0]), float(val[1])])
            except (ValueError, IndexError):
                continue
    return src_pts, dst_pts


def _get_calibration_points(
    state: CalibratorState, ow: int, oh: int
) -> Tuple[List[List[float]], List[List[float]]]:
    """คืน (src_pts, dst_pts) จาก cells + fine_cells. src = [px, py], dst = [pan, tilt]."""
    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    cell_w = ow / GRID_COLS
    cell_h = oh / GRID_ROWS
    center_r, center_c = GRID_ROWS // 2, GRID_COLS // 2
    for key, val in state.cells.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0]), int(parts[1])
            px = (c + 0.5) * cell_w
            py = (r + 0.5) * cell_h
            src_pts.append([px, py])
            dst_pts.append([float(val[0]), float(val[1])])
        except (ValueError, IndexError):
            continue
    fine_cells = state.center_fine_grid.get("fine_cells") or {}
    left = center_c * cell_w
    top = center_r * cell_h
    fw = cell_w / FINE_COLS
    fh = cell_h / FINE_ROWS
    for key, val in fine_cells.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            fi, fj = int(parts[0]), int(parts[1])
            px = left + (fj + 0.5) * fw
            py = top + (fi + 0.5) * fh
            src_pts.append([px, py])
            dst_pts.append([float(val[0]), float(val[1])])
        except (ValueError, IndexError):
            continue
    return src_pts, dst_pts


def _load_pixel_per_degree_json() -> Optional[Dict[str, Any]]:
    """โหลดไฟล์ cam4_pixel_per_degree.json คืน dict หรือ None."""
    if not PX_DEG_JSON_PATH.is_file():
        return None
    try:
        with open(PX_DEG_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _compute_pixel_per_degree_from_points(
    src_pts: List[List[float]],
    dst_pts: List[List[float]],
    cx: float,
    cy: float,
) -> Tuple[Optional[float], Optional[float]]:
    """คำนวณ pixel ต่อ 1 องศา (เฉลี่ย) จากจุด calibration. คืน (px_per_deg_x, px_per_deg_y) หรือ (None, None)."""
    if not src_pts or len(src_pts) != len(dst_pts):
        return None, None
    ratios_x: List[float] = []
    ratios_y: List[float] = []
    for (px, py), (pan, tilt) in zip(src_pts, dst_pts):
        if abs(pan) > 1e-6:
            ratios_x.append((float(px) - cx) / pan)
        if abs(tilt) > 1e-6:
            ratios_y.append((float(py) - cy) / tilt)
    px_per_deg_x = float(np.mean(ratios_x)) if ratios_x else None
    px_per_deg_y = float(np.mean(ratios_y)) if ratios_y else None
    return px_per_deg_x, px_per_deg_y


def _compute_calibration_accuracy(
    H: List[List[float]], src_pts: List[List[float]], dst_pts: List[List[float]]
) -> Optional[Tuple[float, float]]:
    """คำนวณ reprojection error จาก homography. คืน (mean_deg, max_deg) หรือ None."""
    if not H or len(H) != 3 or len(src_pts) != len(dst_pts) or len(src_pts) == 0:
        return None
    try:
        Hnp = np.array(H, dtype=np.float64)
        errors: List[float] = []
        for (px, py), (pan_true, tilt_true) in zip(src_pts, dst_pts):
            p = Hnp @ np.array([px, py, 1.0])
            if abs(p[2]) < 1e-9:
                continue
            pan_pred = p[0] / p[2]
            tilt_pred = p[1] / p[2]
            err = math.sqrt((pan_pred - pan_true) ** 2 + (tilt_pred - tilt_true) ** 2)
            errors.append(err)
        if not errors:
            return None
        return (float(np.mean(errors)), float(np.max(errors)))
    except Exception:
        return None


def _compute_homography(state: CalibratorState, ow: int, oh: int) -> Optional[List[List[float]]]:
    """สร้าง point correspondences จาก cells + fine_cells แล้ว fit homography (px,py)->(pan,tilt). คืน 3x3 list หรือ None."""
    src_pts, dst_pts = _get_calibration_points(state, ow, oh)
    if len(src_pts) < 4:
        return None
    try:
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            return None
        return H.tolist()
    except Exception:
        return None


def _build_verify_points(state: CalibratorState) -> List[Tuple[str, float, float]]:
    """สร้างรายการจุด (label, pan, tilt) ตาม coarse_order แล้ว fine_order — ใช้ตอนรีเชคหลัง Save."""
    out: List[Tuple[str, float, float]] = []
    for (r, c) in state.coarse_order:
        key = f"{r}_{c}"
        if key in state.cells and state.cells[key]:
            pan, tilt = state.cells[key][0], state.cells[key][1]
            out.append((f"coarse ({r},{c})", pan, tilt))
    fine_cells = state.center_fine_grid.get("fine_cells") or {}
    for (fi, fj) in state.fine_order:
        key = f"{fi}_{fj}"
        if key in fine_cells and fine_cells[key]:
            pan, tilt = fine_cells[key][0], fine_cells[key][1]
            out.append((f"fine ({fi},{fj})", pan, tilt))
    return out


def _append_test_move_log(
    click_pixel: Tuple[float, float],
    target_arm: Tuple[float, float],
    actual_arm: Tuple[float, float],
    log_path: Path,
    output_w: int = REF_OUTPUT_W,
    output_h: int = REF_OUTPUT_H,
    actual_pixel: Optional[Tuple[float, float]] = None,
    crosshair_pixel: Optional[Tuple[float, float]] = None,
    error_pixel_x: Optional[float] = None,
    error_pixel_y: Optional[float] = None,
    error_pixel_dist: Optional[float] = None,
) -> None:
    """บันทึก click, target, actual ลง log_path (แยก homography / pxdeg). รูปแบบรองรับ re-calibration."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tx, ty = float(target_arm[0]), float(target_arm[1])
        ax, ay = float(actual_arm[0]), float(actual_arm[1])
        px, py = float(click_pixel[0]), float(click_pixel[1])
        err_x = tx - ax
        err_y = ty - ay
        err_deg = math.sqrt(err_x * err_x + err_y * err_y)
        record = {
            "schema": "cam4_test_move_v1",
            "timestamp": time.time(),
            "output_width": output_w,
            "output_height": output_h,
            "click_pixel": [px, py],
            "target_arm": [tx, ty],
            "actual_arm": [ax, ay],
            "error_x": err_x,
            "error_y": err_y,
            "error_deg": err_deg,
            "input_pixel_norm": [px / max(1, output_w), py / max(1, output_h)],
            "actual_arm_deg": [ax, ay],
        }
        if actual_pixel is not None:
            record["actual_pixel"] = [float(actual_pixel[0]), float(actual_pixel[1])]
        if crosshair_pixel is not None:
            record["crosshair_pixel"] = [float(crosshair_pixel[0]), float(crosshair_pixel[1])]
        if error_pixel_x is not None:
            record["error_pixel_x"] = error_pixel_x
        if error_pixel_y is not None:
            record["error_pixel_y"] = error_pixel_y
        if error_pixel_dist is not None:
            record["error_pixel_dist"] = error_pixel_dist
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        msg = f"  Logged to {log_path.name}: target=({tx:.2f},{ty:.2f}) actual=({ax:.2f},{ay:.2f}) error_deg={err_deg:.3f}"
        if error_pixel_dist is not None:
            msg += f" error_px={error_pixel_dist:.1f}px"
        print(msg)
    except Exception as e:
        print(f"  Failed to write move log: {e}")


def _recalibrate_from_log(state: CalibratorState) -> bool:
    """
    อ่าน cam4_test_move_log.jsonl รวมจุด (click_pixel → actual_arm)
    - โหมด homography: refit homography แล้ว save
    - โหมด pixel_per_degree: fit scale จาก log แล้ว save pixel_per_degree_x/y
    """
    try:
        if not JSON_PATH.is_file():
            print("  Re-cal: No calibration JSON found.")
            return False
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        ow = int(data.get("output_width", REF_OUTPUT_W))
        oh = int(data.get("output_height", REF_OUTPUT_H))

        if getattr(state, "use_pixel_per_degree", False):
            # ใช้ center of frame เป็น origin (ไม่ใช้ crosshair จาก JSON)
            cx = float(ow) / 2.0
            cy = float(oh) / 2.0
            # โหมด Px/deg: fit scale จาก log แล้วบันทึกเฉพาะไฟล์ cam4_pixel_per_degree.json
            log_px: List[List[float]] = []
            log_arm: List[List[float]] = []
            if TEST_MOVE_LOG_PXDEG_PATH.is_file():
                with open(TEST_MOVE_LOG_PXDEG_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("schema") != "cam4_test_move_v1":
                                continue
                            cp = rec.get("click_pixel")
                            aa = rec.get("actual_arm")
                            if not cp or not aa or len(cp) < 2 or len(aa) < 2:
                                continue
                            px, py = float(cp[0]), float(cp[1])
                            ax, ay = float(aa[0]), float(aa[1])
                            log_w = int(rec.get("output_width", ow))
                            log_h = int(rec.get("output_height", oh))
                            if log_w != ow or log_h != oh:
                                px = px * ow / max(1, log_w)
                                py = py * oh / max(1, log_h)
                            log_px.append([px, py])
                            log_arm.append([ax, ay])
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
            if len(log_px) < 2:
                print(f"  Re-cal (Px/deg): Need at least 2 points in {TEST_MOVE_LOG_PXDEG_PATH.name}, got {len(log_px)}.")
                return False
            ratios_x = [(log_px[i][0] - cx) / ax for i, (_, (ax, ay)) in enumerate(zip(log_px, log_arm)) if abs(ax) > 1e-6]
            ratios_y = [(log_px[i][1] - cy) / ay for i, (_, (ax, ay)) in enumerate(zip(log_px, log_arm)) if abs(ay) > 1e-6]
            if not ratios_x or not ratios_y:
                print("  Re-cal (Px/deg): Not enough variation in pan/tilt.")
                return False
            ppdx = float(np.mean(ratios_x))
            ppdy = float(np.mean(ratios_y))
            state.px_per_deg_x = ppdx
            state.px_per_deg_y = ppdy
            # บันทึกเฉพาะไฟล์ Px/deg (ไม่แตะ homography file)
            px_deg_data = {
                "output_width": ow,
                "output_height": oh,
                "crosshair": {"x": cx, "y": cy},
                "pixel_per_degree_x": ppdx,
                "pixel_per_degree_y": ppdy,
            }
            PX_DEG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PX_DEG_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(px_deg_data, f, indent=2)
            # คำนวณ accuracy: error องศาจาก linear model
            errors_deg = []
            for (px, py), (ax, ay) in zip(log_px, log_arm):
                pred_pan = (px - cx) / ppdx
                pred_tilt = (py - cy) / ppdy
                err = math.sqrt((pred_pan - ax) ** 2 + (pred_tilt - ay) ** 2)
                errors_deg.append(err)
            mean_deg = float(np.mean(errors_deg))
            max_deg = float(np.max(errors_deg))
            state.recal_message_text = f"Mean err: {mean_deg:.3f} deg  Max: {max_deg:.3f} deg"
            state.recal_accuracy_text = f"Recal: Mean err {mean_deg:.3f} deg  Max {max_deg:.3f} deg"
            print(f"  Re-cal (Px/deg): saved to {PX_DEG_JSON_PATH.name}  px_per_deg=({ppdx:.2f}, {ppdy:.2f})  {state.recal_message_text}")
            state.recal_message_until = time.time() + 2.5
            return True

        # โหมด homography
        src_pts, dst_pts = _get_calibration_points_from_data(data, ow, oh)
        n_orig = len(src_pts)

        if TEST_MOVE_LOG_PATH.is_file():
            with open(TEST_MOVE_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("schema") != "cam4_test_move_v1":
                            continue
                        cp = rec.get("click_pixel")
                        aa = rec.get("actual_arm")
                        if not cp or not aa or len(cp) < 2 or len(aa) < 2:
                            continue
                        px, py = float(cp[0]), float(cp[1])
                        ax, ay = float(aa[0]), float(aa[1])
                        log_w = int(rec.get("output_width", ow))
                        log_h = int(rec.get("output_height", oh))
                        if log_w != ow or log_h != oh:
                            px = px * ow / max(1, log_w)
                            py = py * oh / max(1, log_h)
                        src_pts.append([px, py])
                        dst_pts.append([ax, ay])
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        n_total = len(src_pts)
        if n_total < 4:
            print(f"  Re-cal: Need at least 4 points, got {n_total} (orig {n_orig} + log {n_total - n_orig}).")
            return False

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            print("  Re-cal: findHomography failed.")
            return False

        data["homography"] = H.tolist()
        # ไม่เก็บ pixel_per_degree ในไฟล์ homography
        data.pop("pixel_per_degree_x", None)
        data.pop("pixel_per_degree_y", None)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Re-calibrated: {n_orig} orig + {n_total - n_orig} from log → homography updated, saved to {JSON_PATH.name}")

        acc = _compute_calibration_accuracy(H.tolist(), src_pts, dst_pts)
        if acc is not None:
            mean_deg, max_deg = acc
            state.recal_message_text = f"Mean err: {mean_deg:.3f} deg  Max: {max_deg:.3f} deg"
            state.recal_accuracy_text = f"Recal: Mean err {mean_deg:.3f} deg  Max {max_deg:.3f} deg"
            print(f"  {state.recal_message_text}")
        else:
            state.recal_message_text = None
            state.recal_accuracy_text = None
        state.recal_message_until = time.time() + 2.5

        if load_grid_json:
            state.test_data = load_grid_json(JSON_PATH)
        return True
    except Exception as e:
        print(f"  Re-cal failed: {e}")
        return False


def _save_json(state: CalibratorState, path: Optional[Path] = None) -> bool:
    p = path or JSON_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ow = state.output_w or REF_OUTPUT_W
        oh = state.output_h or REF_OUTPUT_H
        homography = _compute_homography(state, ow, oh)
        # บันทึก crosshair = กลางเฟรมเสมอ เพื่อให้ calibration และ test ใช้จุดอ้างอิง (0,0) pixel เดียวกัน
        data = {
            "output_width": ow,
            "output_height": oh,
            "crosshair": {"x": float(ow) / 2.0, "y": float(oh) / 2.0},
            "grid_cols": GRID_COLS,
            "grid_rows": GRID_ROWS,
            "cells": state.cells,
            "center_fine_grid": {
                "fine_rows": FINE_ROWS,
                "fine_cols": FINE_COLS,
                "fine_cells": state.center_fine_grid.get("fine_cells", {}),
            },
        }
        if homography is not None:
            data["homography"] = homography
        # ไม่เก็บ pixel_per_degree ในไฟล์ homography — ใช้ไฟล์แยก cam4_pixel_per_degree.json
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {p}")
        state.save_message_until = time.time() + 2.5  # แสดง "Saved!" บนจอ 2.5 วินาที
        state.save_accuracy_text = None
        if homography is not None:
            src_pts, dst_pts = _get_calibration_points(state, ow, oh)
            acc = _compute_calibration_accuracy(homography, src_pts, dst_pts)
            if acc is not None:
                mean_deg, max_deg = acc
                state.save_accuracy_text = f"Mean err: {mean_deg:.3f} deg  Max: {max_deg:.3f} deg"
                print(f"  Calibration accuracy: {state.save_accuracy_text}")
        # เริ่มโหมดรีเชค: หมุนไปแต่ละจุดที่เก็บไว้ รอ ENTER ยืนยัน
        state.verify_points = _build_verify_points(state)
        state.verify_index = 0 if state.verify_points else None
        state.verify_last_moved_index = -1
        if state.verify_points:
            print(f"  Verify: {len(state.verify_points)} points — ENTER=next, ESC=skip")
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False


def main() -> None:
    import sys
    camera_name = "cam4"
    if "--camera" in sys.argv:
        idx = sys.argv.index("--camera")
        if idx + 1 < len(sys.argv):
            camera_name = sys.argv[idx + 1].strip()
    # ใช้ path ตามชื่อกล้อง เพื่อบันทึก/โหลด calibration ต่อกล้อง
    global PX_DEG_JSON_PATH, JSON_PATH, TEST_MOVE_LOG_PATH, TEST_MOVE_LOG_PXDEG_PATH
    PX_DEG_JSON_PATH = CALIBRATION_DIR / f"{camera_name}_pixel_per_degree.json"
    JSON_PATH = CALIBRATION_DIR / f"{camera_name}_mouse_grid_lookup.json"
    TEST_MOVE_LOG_PATH = CALIBRATION_DIR / f"{camera_name}_test_move_log_homography.jsonl"
    TEST_MOVE_LOG_PXDEG_PATH = CALIBRATION_DIR / f"{camera_name}_test_move_log_pxdeg.jsonl"
    print("Calibrator camera:", camera_name, "->", PX_DEG_JSON_PATH.name)

    if build_camera_from_config is None or ShooterConfig is None:
        print("gun_aim_assist not available (build_camera_from_config, ShooterConfig).")
        return
    if Cam4ArmController is None:
        print("cam4_arm_controller not available.")
        return

    cam = build_camera_from_config(camera_name)
    cam.start()
    time.sleep(0.3)

    shooter_cfg = ShooterConfig()
    shooter_cfg.load()

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

    state = _state
    state.output_w = 0
    state.output_h = 0
    state.cells = {}
    state.center_fine_grid = {"fine_rows": FINE_ROWS, "fine_cols": FINE_COLS, "fine_cells": {}}
    state.coarse_order = _coarse_order(GRID_ROWS // 2, GRID_COLS // 2, GRID_ROWS, GRID_COLS)
    state.fine_order = _fine_order(FINE_ROWS, FINE_COLS)
    state.coarse_index = 0
    state.fine_index = 0
    state.phase = "coarse"
    state.pending_click = None
    state.moved_for_pending = False
    state.reference_capture = None
    state.mode = "calibration"
    state.smooth_target = None
    state.smooth_start = None

    win_name = "Cam4 Mouse Grid Calibrator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, _on_mouse, param={"arm": arm})

    try:
        from gun_aim_assist import get_screen_size
        screen_w, screen_h = get_screen_size()
    except Exception:
        screen_w, screen_h = 1920, 1080

    last_loop_time = time.time()
    last_confirm_time = 0.0
    last_sync_time = 0.0
    SYNC_INTERVAL_SEC = 0.25  # sync ตำแหน่งจาก GRBL ทุกช่วงนี้

    try:
        while True:
            active, frame, _ = cam.read()
            if not active or frame is None:
                time.sleep(0.02)
                continue
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # บังคับขนาดคำนวณเป็น Full HD — ให้พิกัดคลิกและ lookup ตรงกัน
            if frame.shape[1] != REF_OUTPUT_W or frame.shape[0] != REF_OUTPUT_H:
                frame = cv2.resize(frame, (REF_OUTPUT_W, REF_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
            if state.output_w <= 0:
                state.output_w = REF_OUTPUT_W
                state.output_h = REF_OUTPUT_H
                # Calibration และ Test: ใช้ center of frame เป็น (0,0) pixel — ไม่ใช้ crosshair จาก JSON
                state.crosshair_x = state.output_w / 2.0
                state.crosshair_y = state.output_h / 2.0
                # โหลด pixel_per_degree จากไฟล์แยก; ไม่มีหรือไม่มีค่า → ใช้ PX_PER_DEG_*_DEFAULT
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

            # โหมด Test: บังคับ crosshair = กลางเฟรม (0,0) pixel ทุกเฟรม (ทั้ง Px/deg และ homography)
            if (
                state.mode == "test"
                and state.output_w > 0
                and state.output_h > 0
            ):
                state.crosshair_x = state.output_w / 2.0
                state.crosshair_y = state.output_h / 2.0

            # โหมด N (crop + CSRT): Init tracker จาก pending_bbox
            if (
                state.mode == "test"
                and getattr(state, "test_track_csrt", False)
                and getattr(state, "sim_csrt_pending_bbox", None) is not None
                and state.output_w > 0
                and state.output_h > 0
            ):
                xb, yb, wb, hb = state.sim_csrt_pending_bbox
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
                            state.sim_csrt_tracker = tracker
                            state.sim_csrt_initialized = True
                            state.sim_csrt_bbox = (xb, yb, wb, hb)
                            cx = xb + wb * 0.5
                            cy = yb + hb * 0.5
                            state.sim_csrt_smooth_px = cx
                            state.sim_csrt_smooth_py = cy
                            state.sim_display_px = cx
                            state.sim_display_py = cy
                            state.sim_csrt_lost = False
                    except Exception:
                        pass
                state.sim_csrt_pending_bbox = None

            # Capture พื้นที่เซลล์ที่คลิก เป็นภาพอ้างอิง (โหมด Calibration)
            if (
                state.mode == "calibration"
                and state.pending_click is not None
                and state.reference_capture is None
                and state.output_w > 0
                and state.output_h > 0
            ):
                cell_w = state.output_w / GRID_COLS
                cell_h = state.output_h / GRID_ROWS
                center_r, center_c = GRID_ROWS // 2, GRID_COLS // 2
                pl = state.pending_click
                if state.phase == "coarse" and len(pl) >= 4:
                    row, col = int(pl[0]), int(pl[1])
                    x0 = int(col * cell_w)
                    y0 = int(row * cell_h)
                    x1 = int((col + 1) * cell_w)
                    y1 = int((row + 1) * cell_h)
                else:
                    left = center_c * cell_w
                    top = center_r * cell_h
                    fw = cell_w / FINE_COLS
                    fh = cell_h / FINE_ROWS
                    fi, fj = int(pl[0]), int(pl[1])
                    x0 = int(left + fj * fw)
                    y0 = int(top + fi * fh)
                    x1 = int(left + (fj + 1) * fw)
                    y1 = int(top + (fi + 1) * fh)
                x0 = max(0, min(w, x0))
                y0 = max(0, min(h, y0))
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                if x1 > x0 and y1 > y0:
                    roi = frame[y0:y1, x0:x1].copy()
                    # จุดที่คลิก (output) ในพิกัด crop
                    cx_click = int(pl[2] - x0)
                    cy_click = int(pl[3] - y0)
                    ch_x = int(state.crosshair_x - x0)
                    ch_y = int(state.crosshair_y - y0)
                    roi_h, roi_w = roi.shape[0], roi.shape[1]
                    # กากบาทสีแดงที่ตำแหน่งคลิก — ยาวและหนาให้เห็นชัด (scale ตาม resolution)
                    cross_len = int(30 * HUD_TEXT_SCALE)
                    cross_thick = max(2, int(4 * HUD_TEXT_SCALE))
                    cv2.line(
                        roi,
                        (max(0, cx_click - cross_len), cy_click),
                        (min(roi_w, cx_click + cross_len), cy_click),
                        COLOR_CLICK_MARKER,
                        cross_thick,
                    )
                    cv2.line(
                        roi,
                        (cx_click, max(0, cy_click - cross_len)),
                        (cx_click, min(roi_h, cy_click + cross_len)),
                        COLOR_CLICK_MARKER,
                        cross_thick,
                    )
                    # วาดศูนย์เล็งใน crop ถ้าอยู่ในขอบ (เขียว) เพื่อเทียบกับจุดคลิก
                    if 0 <= ch_x < roi_w and 0 <= ch_y < roi_h:
                        r_ch = max(2, int(8 * HUD_TEXT_SCALE))
                        cv2.circle(roi, (ch_x, ch_y), r_ch, COLOR_CROSSHAIR_IN_REF, THICKNESS_THICK)
                        cv2.line(roi, (ch_x - r_ch - 2, ch_y), (ch_x + r_ch + 2, ch_y), COLOR_CROSSHAIR_IN_REF, THICKNESS_THICK)
                        cv2.line(roi, (ch_x, ch_y - r_ch - 2), (ch_x, ch_y + r_ch + 2), COLOR_CROSSHAIR_IN_REF, THICKNESS_THICK)
                    state.reference_capture = roi

            # โหมด Test: capture crop รอบจุดคลิก แสดงใน panel (เปรียบเทียบว่าแขนเล็งตรงหรือไม่)
            if (
                state.mode == "test"
                and state.pending_test_capture is not None
                and state.output_w > 0
                and state.output_h > 0
            ):
                px, py = state.pending_test_capture[0], state.pending_test_capture[1]
                half_w, half_h = TEST_CAPTURE_HALF_W, TEST_CAPTURE_HALF_H
                x0 = max(0, int(px) - half_w)
                y0 = max(0, int(py) - half_h)
                x1 = min(w, int(px) + half_w)
                y1 = min(h, int(py) + half_h)
                if x1 > x0 and y1 > y0:
                    roi = frame[y0:y1, x0:x1].copy()
                    roi_h, roi_w = roi.shape[0], roi.shape[1]
                    cx = int(px - x0)
                    cy = int(py - y0)
                    cross_len = int(60 * HUD_TEXT_SCALE)  # ให้เห็นชัดใน capture ขนาดใหญ่
                    cross_thick = max(2, int(8 * HUD_TEXT_SCALE))
                    cv2.line(
                        roi,
                        (max(0, cx - cross_len), cy),
                        (min(roi_w, cx + cross_len), cy),
                        COLOR_CLICK_MARKER,
                        cross_thick,
                    )
                    cv2.line(
                        roi,
                        (cx, max(0, cy - cross_len)),
                        (cx, min(roi_h, cy + cross_len)),
                        COLOR_CLICK_MARKER,
                        cross_thick,
                    )
                    state.test_reference_capture = roi
                state.pending_test_capture = None

            now = time.time()
            dt = now - last_loop_time if last_loop_time else 0.02
            last_loop_time = now

            # Sync ตำแหน่งจาก GRBL เป็นระยะ เพื่อให้ pos_x/pos_y เป็นค่าล่าสุด
            if (now - last_sync_time) >= SYNC_INTERVAL_SEC:
                last_sync_time = now
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
            if getattr(state, "test_awaiting_home", False):
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
                px_arm = getattr(arm, "pos_x", 0.0)
                py_arm = getattr(arm, "pos_y", 0.0)
                if abs(px_arm) < 1.0 and abs(py_arm) < 1.0:
                    state.test_awaiting_home = False

            # โหมด J: เขียนแถว CSV เมื่อมี next_state (ตำแหน่งแขนล่าสุดหลังส่ง move)
            if (
                getattr(state, "test_teaching", False)
                and state.teaching_last_state is not None
                and state.teaching_last_action is not None
                and state.teaching_last_target is not None
                and state.teaching_last_ts is not None
                and state.teaching_csv_handle is not None
                and arm is not None
            ):
                try:
                    pan_cur = getattr(arm, "pos_x", 0.0)
                    tilt_cur = getattr(arm, "pos_y", 0.0)
                    pt, tt = state.teaching_last_target[0], state.teaching_last_target[1]
                    next_err_pan = pt - pan_cur
                    next_err_tilt = tt - tilt_cur
                    next_error_deg = math.hypot(next_err_pan, next_err_tilt)
                    from datetime import datetime
                    ts_utc = datetime.utcfromtimestamp(state.teaching_last_ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    use_model_val = 1 if getattr(state, "teaching_last_use_model", False) else 0
                    row = [ts_utc] + list(state.teaching_last_state) + list(state.teaching_last_action) + [next_err_pan, next_err_tilt, next_error_deg, use_model_val]
                    state.teaching_csv_handle.write(",".join(str(x) for x in row) + "\n")
                    state.teaching_csv_handle.flush()
                except Exception:
                    pass
                state.teaching_last_state = None
                state.teaching_last_action = None
                state.teaching_last_target = None
                state.teaching_last_ts = None
                state.teaching_last_err = None
                state.teaching_last_use_model = None

            # โหมด J (Auto): เป้าจากจำลองใกล้ 0,0 — อัปเดตทุกเฟรม (เมื่อไม่กำลังรีเทรน)
            if (
                state.mode == "test"
                and getattr(state, "test_teaching", False)
                and getattr(state, "teaching_source", "auto") == "auto"
                and not getattr(state, "teaching_retrain_in_progress", False)
                and arm is not None
            ):
                if now >= getattr(state, "teaching_pattern_switch_at", 0.0):
                    import random
                    state.teaching_pattern_index = random.randint(0, max(0, len(TEACHING_PATTERN_NAMES) - 1))
                    state.teaching_pattern_switch_at = now + random.uniform(5.0, 18.0)
                t_sec = now - getattr(state, "teaching_sim_start", now)
                pan_tgt, tilt_tgt = _teaching_sim_target_near_origin(t_sec, state)
                x_lim = getattr(arm, "_effective_x_limits", None)
                y_lim = getattr(arm, "_effective_y_limits", None)
                if x_lim is None or y_lim is None:
                    margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
                    xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
                    yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
                    x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
                    y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
                x_lo, x_hi = x_lim[0], x_lim[1]
                y_lo, y_hi = y_lim[0], y_lim[1]
                if SWAP_PAN_TILT:
                    cmd1 = float(np.clip(tilt_tgt, x_lo, x_hi))
                    cmd2 = float(np.clip(pan_tgt, y_lo, y_hi))
                else:
                    cmd1 = float(np.clip(pan_tgt, x_lo, x_hi))
                    cmd2 = float(np.clip(tilt_tgt, y_lo, y_hi))
                state.continuous_target_pan = cmd1
                state.continuous_target_tilt = cmd2

            # โหมด L Auto เท่านั้น: เป้าแขนจาก sim_display_px/py (เมาส์). โหมด N ทำเหมือน L แต่แค่คำนวณเวกเตอร์+จุดส้มนำทาง ยังไม่ส่งหมุนแขน
            if (
                state.mode == "test"
                and getattr(state, "test_track_sim", False)
                and getattr(state, "track_sim_arm_auto", True)
                and not getattr(state, "sim_drag_active", False)
            ):
                x_lim = getattr(arm, "_effective_x_limits", None)
                y_lim = getattr(arm, "_effective_y_limits", None)
                if x_lim is None or y_lim is None:
                    margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if config else 2.0
                    xr = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if config else (-65.0, 65.0)
                    yr = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if config else (-35.0, 35.0)
                    x_lim = (xr[0] + margin, xr[1] - margin) if x_lim is None else x_lim
                    y_lim = (yr[0] + margin, yr[1] - margin) if y_lim is None else y_lim
                x_lo, x_hi = x_lim[0], x_lim[1]
                y_lo, y_hi = y_lim[0], y_lim[1]
                # โหมด N และ tracker lost: ตั้งเป้าแขน = ตำแหน่งปัจจุบัน (อยู่ที่เดิม รอ crop ใหม่)
                if getattr(state, "test_track_csrt", False) and getattr(state, "sim_csrt_lost", False):
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    pan_cur = getattr(arm, "pos_x", 0.0)
                    tilt_cur = getattr(arm, "pos_y", 0.0)
                    if SWAP_PAN_TILT:
                        state.continuous_target_pan = float(np.clip(pan_cur, y_lo, y_hi))
                        state.continuous_target_tilt = float(np.clip(tilt_cur, x_lo, x_hi))
                    else:
                        state.continuous_target_pan = float(np.clip(pan_cur, x_lo, x_hi))
                        state.continuous_target_tilt = float(np.clip(tilt_cur, y_lo, y_hi))
                # เป้าแขนจากตำแหน่งเมาส์ (sim_display_px/py มาจาก EVENT_MOUSEMOVE หรือ N virtual)
                elif (
                    state.sim_display_px is not None
                    and state.sim_display_py is not None
                    and state.output_w > 0
                    and state.output_h > 0
                ):
                    if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                        px_target, py_target = _kalman_update_and_predict(
                            state,
                            state.sim_display_px,
                            state.sim_display_py,
                            now,
                            SIM_KALMAN_LOOKAHEAD_SEC,
                            state.output_w,
                            state.output_h,
                        )
                        if hasattr(arm, "sync_position_from_grbl"):
                            try:
                                arm.sync_position_from_grbl()
                            except Exception:
                                pass
                        pan_cur = getattr(arm, "pos_x", 0.0)
                        tilt_cur = getattr(arm, "pos_y", 0.0)
                        cx = getattr(state, "crosshair_x", state.output_w / 2.0)
                        cy = getattr(state, "crosshair_y", state.output_h / 2.0)
                        dx_px = px_target - cx
                        dy_px = py_target - cy
                        delta_pan = dx_px / state.px_per_deg_x
                        delta_tilt = dy_px / state.px_per_deg_y
                        if CAMERA_ON_ARM:
                            target_pan = pan_cur + delta_pan
                            target_tilt = tilt_cur + delta_tilt
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
                        # ไม่มี px_per_deg: ใช้ _l_manual_target_from_pixel ทุก detection_interval
                        if now - getattr(state, "last_sim_detection_time", 0.0) >= getattr(state, "sim_detection_interval_sec", 0.06):
                            state.last_sim_detection_time = now
                            res = _l_manual_target_from_pixel(
                                state.sim_display_px, state.sim_display_py, state, arm
                            )
                            if res is not None:
                                state.continuous_target_pan, state.continuous_target_tilt = res
                                state.continuous_velocity_mm_s = 0.0
                                state.continuous_error_deg_smooth = None

            # โหมด N (crop + CSRT): Update tracker ทุกเฟรม → center bbox (จุดแดง), เป้าแขน = virtual ที่ระยะ r (0–10 px)
            if (
                state.mode == "test"
                and getattr(state, "test_track_csrt", False)
                and getattr(state, "sim_csrt_initialized", False)
                and getattr(state, "sim_csrt_tracker", None) is not None
                and state.output_w > 0
                and state.output_h > 0
            ):
                tr = state.sim_csrt_tracker
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
                        a = CSRT_SMOOTH_ALPHA
                        if state.sim_csrt_smooth_px is None or state.sim_csrt_smooth_py is None:
                            state.sim_csrt_smooth_px = cx_bbox
                            state.sim_csrt_smooth_py = cy_bbox
                        else:
                            state.sim_csrt_smooth_px = a * cx_bbox + (1.0 - a) * state.sim_csrt_smooth_px
                            state.sim_csrt_smooth_py = a * cy_bbox + (1.0 - a) * state.sim_csrt_smooth_py
                        state.sim_csrt_bbox = (xb, yb, wb, hb)
                        state.sim_csrt_lost = False
                        # r คำนวณจาก error_px: error สูง → r สูงสุด 10, ลดลงตาม error_px จน r=0 เมื่อ error_px≈0
                        ch_x = getattr(state, "crosshair_x", state.output_w / 2.0)
                        ch_y = getattr(state, "crosshair_y", state.output_h / 2.0)
                        bx = state.sim_csrt_smooth_px
                        by = state.sim_csrt_smooth_py
                        error_px = math.sqrt((bx - ch_x) ** 2 + (by - ch_y) ** 2)
                        if error_px < 1e-6:
                            state.sim_display_px = ch_x
                            state.sim_display_py = ch_y
                            state.sim_csrt_error_px = 0.0
                            state.sim_csrt_r = 0.0
                        else:
                            ux = (bx - ch_x) / error_px
                            uy = (by - ch_y) / error_px
                            r = min(N_MODE_R_MAX_PX, error_px)
                            state.sim_display_px = ch_x + ux * r
                            state.sim_display_py = ch_y + uy * r
                            state.sim_csrt_error_px = error_px
                            state.sim_csrt_r = r
                    else:
                        state.sim_csrt_bbox = None
                        state.sim_csrt_lost = True
                        state.sim_csrt_error_px = None
                        state.sim_csrt_r = None
                except Exception:
                    state.sim_csrt_bbox = None
                    state.sim_csrt_lost = True
                    state.sim_csrt_error_px = None
                    state.sim_csrt_r = None

            # โหมด P, L หรือ J: variable step ไป continuous target (ยกเว้น J กำลังรีเทรน). โหมด L Manual ส่ง move เมื่อลากเมาส์ตั้งเป้าแล้ว
            if (
                state.mode == "test"
                and (getattr(state, "test_continuous", False) or getattr(state, "test_track_sim", False) or getattr(state, "test_teaching", False) or getattr(state, "test_track_csrt", False))
                and state.continuous_target_pan is not None
                and state.continuous_target_tilt is not None
                and not getattr(state, "teaching_retrain_in_progress", False)
            ):
                pan_tgt = state.continuous_target_pan
                tilt_tgt = state.continuous_target_tilt
                # Throttle ตามโซน: โหมด L (sim) ห่างแล้วพุ่งเข้าไปเลยเหมือน P — ใช้ throttle เล็ก; โหมด P ใช้ 0.02 s
                current_pan = getattr(arm, "pos_x", 0.0)
                current_tilt = getattr(arm, "pos_y", 0.0)
                error_deg = math.hypot(pan_tgt - current_pan, tilt_tgt - current_tilt)
                is_track_sim = getattr(state, "test_track_sim", False)
                if is_track_sim and error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
                    # โหมด L: ห่าง = ส่งบ่อย พุ่งเข้าเป้าให้ทันโดรน
                    throttle_sec = CONTINUOUS_THROTTLE_SEC
                elif error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
                    throttle_sec = CONTINUOUS_P_THROTTLE_FAR_SEC
                elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
                    throttle_sec = CONTINUOUS_P_THROTTLE_MID_SEC
                else:
                    throttle_sec = CONTINUOUS_P_THROTTLE_NEAR_SEC
                min_interval = getattr(arm, "_min_move_interval_sec", 0.02)
                if is_track_sim and error_deg > CONTINUOUS_P_VERY_FAR_DEG:
                    throttle_sec = max(throttle_sec, min(min_interval, 0.01))
                else:
                    throttle_sec = max(throttle_sec, min_interval)
                is_teaching = getattr(state, "test_teaching", False)
                use_model = getattr(state, "teaching_use_model", False) and is_teaching
                model_loaded = use_model and getattr(state, "_teaching_model", None) is not None and getattr(state, "_teaching_predict_fn", None) is not None

                if is_teaching:
                    state_10, err_pan, err_tilt, error_deg = _teaching_build_state_10(arm, state, pan_tgt, tilt_tgt)
                    state.teaching_last_state = state_10
                    state.teaching_last_target = (pan_tgt, tilt_tgt)
                    state.teaching_last_ts = time.time()
                    state.teaching_last_err = (err_pan, err_tilt, error_deg)
                    state.teaching_last_use_model = use_model

                if use_model and model_loaded and (time.time() - state.last_continuous_arm_move_time) >= throttle_sec:
                    norm_fn = getattr(state, "_teaching_normalize_fn", None)
                    pred_fn = getattr(state, "_teaching_predict_fn", None)
                    if norm_fn is not None and pred_fn is not None:
                        state_vec = norm_fn(*state_10)
                        dim = getattr(state, "_teaching_model_input_dim", 10)
                        state_vec = state_vec[:dim]
                        mm_per_deg = (getattr(arm, "mm_per_deg_pan", 1.0) + getattr(arm, "mm_per_deg_tilt", 1.0)) / 2.0
                        max_step_deg = (CONTINUOUS_STEP_FAR_MM / mm_per_deg) if mm_per_deg > 1e-9 else 10.0
                        delta_pan, delta_tilt = pred_fn(state._teaching_model, state_vec, max_step_deg)
                        _apply_arm_move_relative(arm, delta_pan, delta_tilt)
                        state.last_continuous_arm_move_time = time.time()
                        state.teaching_last_action = (delta_pan, delta_tilt)
                        state.teaching_last_sent_delta = (delta_pan, delta_tilt)
                        state.teaching_t_prev_send = state.last_continuous_arm_move_time
                        state.teaching_err_prev = (err_pan, err_tilt)
                elif is_track_sim and getattr(state, "l_track_joystick_style", False):
                    joy_throttle = max(throttle_sec, getattr(arm, "_min_move_interval_sec", 0.02))
                    new_time, move_pan, move_tilt = _l_joystick_style_toward_target(
                        arm,
                        pan_tgt,
                        tilt_tgt,
                        state.last_continuous_arm_move_time,
                        joy_throttle,
                        state,
                    )
                    state.last_continuous_arm_move_time = new_time
                else:
                    new_time, move_pan, move_tilt = _variable_step_toward_target(
                        arm,
                        pan_tgt,
                        tilt_tgt,
                        state.last_continuous_arm_move_time,
                        throttle_sec,
                        state,
                    )
                    state.last_continuous_arm_move_time = new_time
                    if is_teaching:
                        state.teaching_last_action = (move_pan, move_tilt)
                        state.teaching_last_sent_delta = (move_pan, move_tilt)
                        state.teaching_t_prev_send = new_time
                        state.teaching_err_prev = state.teaching_last_err

            # Smooth move: interpolate จาก smooth_start ไป smooth_target (ease-in-out)
            if state.smooth_target is not None and state.smooth_start is not None:
                elapsed = now - state.smooth_start_time
                t = min(1.0, elapsed / max(1e-6, state.smooth_duration))
                # smoothstep: t_ease = 3*t^2 - 2*t^3
                t_ease = t * t * (3.0 - 2.0 * t)
                pan = state.smooth_start[0] + t_ease * (state.smooth_target[0] - state.smooth_start[0])
                tilt = state.smooth_start[1] + t_ease * (state.smooth_target[1] - state.smooth_start[1])
                arm.move_absolute(pan, tilt, blocking=False)
                if t >= 1.0:
                    target_cmd = (state.smooth_target[0], state.smooth_target[1])
                    try:
                        arm.move_absolute(state.smooth_target[0], state.smooth_target[1], blocking=False)
                    except Exception:
                        pass
                    state.smooth_target = None
                    state.smooth_start = None
                    if state.mode == "test":
                        # เก็บเฉพาะ x,y pixel ศูนย์เล็ง ตอนแขนนิ่ง — actual_arm จะเก็บตอนผู้ใช้ปรับลูกศรแล้วกด ENTER ยืนยัน
                        state.last_test_crosshair_pixel = (state.crosshair_x, state.crosshair_y)
                        state.last_test_actual_arm = None
                        state.last_test_actual_pixel = None
                        state.test_awaiting_confirm = True  # รอปรับลูกศรแล้วกด ENTER → จะ sync + เก็บ actual + crosshair อีกครั้ง
                        state.test_move_log_pending = True

            # Joystick (โหมด Test ไม่ส่งจอยไปแขน — ขยับแขนเฉพาะจากการคลิก)
            if joystick_reader and joystick_mapper and getattr(joystick_reader, "enabled", False):
                js = joystick_reader.read()
                if state.mode != "test":
                    joystick_mapper.apply(js, dt)
                confirm = _joystick_confirm_pressed(joystick_reader)
                if confirm:
                    if now - last_confirm_time < 0.3:
                        continue
                    last_confirm_time = now
                    if state.mode == "calibration" and state.pending_click is not None:
                        pl = state.pending_click
                        if state.phase == "coarse" and len(pl) >= 4:
                            row, col = pl[0], pl[1]
                            next_c = state.coarse_order[state.coarse_index] if state.coarse_index < len(state.coarse_order) else None
                            if next_c == (row, col):
                                if hasattr(arm, "sync_position_from_grbl"):
                                    try:
                                        arm.sync_position_from_grbl()
                                    except Exception:
                                        pass
                                state.cells[f"{row}_{col}"] = [arm.pos_x, arm.pos_y]
                                state.pending_click = None
                                state.moved_for_pending = False
                                state.reference_capture = None
                                state.coarse_index += 1
                                if state.coarse_index >= len(state.coarse_order):
                                    state.phase = "fine"
                                    state.coarse_index = len(state.coarse_order)
                                # กลับ home (0,0) หลังทุกเซลล์ — จุดเริ่มต้นเท่ากัน ความแม่นยำสูง
                                try:
                                    arm.go_home(blocking=True)
                                except Exception:
                                    pass
                        elif state.phase == "fine" and len(pl) >= 4:
                            fi, fj = pl[0], pl[1]
                            next_f = state.fine_order[state.fine_index] if state.fine_index < len(state.fine_order) else None
                            if next_f == (fi, fj):
                                if hasattr(arm, "sync_position_from_grbl"):
                                    try:
                                        arm.sync_position_from_grbl()
                                    except Exception:
                                        pass
                                state.center_fine_grid.setdefault("fine_cells", {})[f"{fi}_{fj}"] = [arm.pos_x, arm.pos_y]
                                state.pending_click = None
                                state.moved_for_pending = False
                                state.reference_capture = None
                                state.fine_index += 1
                                # กลับ home (0,0) หลังทุกเซลล์ fine
                                try:
                                    arm.go_home(blocking=True)
                                except Exception:
                                    pass
                    elif state.mode == "test":
                        # ในโหมด Test การกดยืนยันสามารถเก็บค่าปรับกลับได้ (optional: อัปเดต cells ใน memory)
                        pass

            # โหมดรีเชคหลัง Save: หมุนไปมุมที่เก็บไว้ทีละจุด รอ ENTER ยืนยัน
            if state.verify_index is not None and state.verify_points:
                idx = state.verify_index
                if idx < len(state.verify_points) and idx != state.verify_last_moved_index:
                    label, pan, tilt = state.verify_points[idx]
                    try:
                        if SWAP_PAN_TILT:
                            arm.move_absolute(tilt, pan, blocking=False)
                        else:
                            arm.move_absolute(pan, tilt, blocking=False)
                    except Exception:
                        pass
                    state.verify_last_moved_index = idx

            next_coarse = None
            next_fine = None
            if state.mode == "calibration":
                if state.phase == "coarse" and state.coarse_index < len(state.coarse_order):
                    next_coarse = state.coarse_order[state.coarse_index]
                elif state.phase == "fine" and state.fine_index < len(state.fine_order):
                    next_fine = state.fine_order[state.fine_index]

            _draw_grids(frame, state, next_coarse, next_fine)

            # Left HUD: bottom-left corner (Y from frame bottom), red, larger font
            _h = frame.shape[0]
            _lh = HUD_LEFT_LINE_H
            _y0 = _h - HUD_BOTTOM_MARGIN
            y_msg = _y0
            y_mode = _y0 - _lh
            y_keys = _y0 - 2 * _lh
            y_arm = _y0 - 3 * _lh
            y_test_map = _y0 - 4 * _lh
            y_p = _y0 - 5 * _lh
            y_l1 = _y0 - 6 * _lh
            y_l2 = _y0 - 7 * _lh
            y_l3 = _y0 - 8 * _lh
            y_j1 = _y0 - 9 * _lh

            # Guide text (หรือข้อความโหมด verify)
            if state.verify_index is not None and state.verify_points and state.verify_index < len(state.verify_points):
                label = state.verify_points[state.verify_index][0]
                pan, tilt = state.verify_points[state.verify_index][1], state.verify_points[state.verify_index][2]
                n, total = state.verify_index + 1, len(state.verify_points)
                msg = f"Verify {n}/{total}: {label} — ENTER=next ESC=skip"
                cv2.putText(frame, msg, (10, y_msg), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                if SWAP_PAN_TILT:
                    map_verify = f"Stored (Full HD→Arm): X={tilt:.2f}  Y={pan:.2f}"
                else:
                    map_verify = f"Stored (Full HD→Arm): X={pan:.2f}  Y={tilt:.2f}"
                cv2.putText(frame, map_verify, (10, y_mode), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            else:
                if state.mode == "calibration":
                    if state.phase == "coarse" and next_coarse is not None:
                        msg = f"Calib: Click coarse grid {next_coarse} then adjust joystick, Button 0 confirm"
                    elif state.phase == "fine" and next_fine is not None:
                        msg = f"Calib: Click fine {next_fine} in center, adjust 0.1mm, Button 0 confirm"
                    else:
                        msg = "Calib: S=Save, T=Test, ESC=Quit"
                elif state.mode == "test" and getattr(state, "pending_test_move", None) is not None:
                    c1, c2 = state.pending_test_move[0], state.pending_test_move[1]
                    msg = f"Target from (0,0): X={c1:.2f} Y={c2:.2f} — ENTER=move ESC=cancel"
                elif state.mode == "test" and getattr(state, "test_awaiting_confirm", False):
                    msg = "Adjust with Arrows (0.1°), then ENTER to confirm — then [H] to go home"
                elif state.mode == "test" and getattr(state, "test_awaiting_home", False):
                    msg = "Returning to 0,0 — wait, then click next test point"
                elif state.mode == "test" and getattr(state, "test_home_required_message_until", 0) > 0 and time.time() < state.test_home_required_message_until:
                    msg = "Go to home (0,0) first — press H to go home"
                elif state.mode == "test":
                    ax = getattr(arm, "pos_x", 0.0)
                    ay = getattr(arm, "pos_y", 0.0)
                    if abs(ax) > 1.0 or abs(ay) > 1.0:
                        msg = "Press [H] to go home, then click next test point"
                    else:
                        msg = "Test: At 0,0 — click, ENTER=move. [H]=home [R]=recal S=Save C=Calib ESC=Quit"
                else:
                    msg = "Test: At 0,0 — click, ENTER=move. [H]=home [R]=recal S=Save C=Calib ESC=Quit"
                cv2.putText(frame, msg, (10, y_msg), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            mode_txt = f"Mode: {state.mode}  [C]alib [T]est [S]ave"
            cv2.putText(frame, mode_txt, (10, y_mode), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            if state.mode == "calibration":
                keys_txt = "Keys: [S]=Save [T]=Test [ESC]=Quit  |  Joystick + Button0 = confirm cell"
            else:
                keys_txt = "Keys: [J]=teach [A]=Click/Auto [M]=Model [F]=Retrain [P]=cont [L]=sim [N]=crop+CSRT [V]=plot [H]=home [R]=recal [S]=Save [C]=Calib [ESC]=Quit"
            cv2.putText(frame, keys_txt, (10, y_keys), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            ax = getattr(arm, "pos_x", 0.0)
            ay = getattr(arm, "pos_y", 0.0)
            arm_txt = f"From home (0,0): X={ax:.1f}  Y={ay:.1f}"
            cv2.putText(frame, arm_txt, (10, y_arm), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            # Test mode: Test map (row 4), then P (row 5), L (rows 6-8), J (row 9+) — no overlap
            if state.mode == "test":
                if getattr(state, "use_pixel_per_degree", False):
                    if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                        map_mode_txt = f"Test map: Px/deg {state.px_per_deg_x:.1f}, {state.px_per_deg_y:.1f}  [D]"
                    else:
                        map_mode_txt = "Test map: Px/deg (need cal)  [D]"
                else:
                    map_mode_txt = "Test map: Homography" + (" (grid)" if getattr(state, "test_use_grid_only", False) else "") + "  [D]"
                cv2.putText(frame, map_mode_txt, (10, y_test_map), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            if state.mode == "test" and getattr(state, "test_continuous", False):
                cv2.putText(
                    frame, "P: Continuous — click to move [P] back",
                    (10, y_p), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                )
            if state.mode == "test" and getattr(state, "test_track_sim", False):
                auto_manual = "Auto" if getattr(state, "track_sim_arm_auto", True) else "Manual"
                track_style = "Same joystick" if getattr(state, "l_track_joystick_style", False) else "Variable step"
                cv2.putText(
                    frame, f"L: Sim — orange=mouse (x,y px)  {auto_manual} [A]  track: {track_style} [J]  [L]=off",
                    (10, y_l1), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                )
                if not getattr(state, "track_sim_arm_auto", True):
                    cv2.putText(
                        frame, "Drag = set target, arm follows",
                        (10, y_l2), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )
                elif getattr(state, "l_track_joystick_style", False):
                    cv2.putText(
                        frame, "Same joystick",
                        (10, y_l2), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )
                else:
                    cv2.putText(
                        frame, "Orange dot = mouse position, arm follows",
                        (10, y_l2), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )
                if getattr(state, "track_sim_arm_auto", True):
                    sx = getattr(state, "sim_display_px", None)
                    sy = getattr(state, "sim_display_py", None)
                else:
                    sx, sy = None, None
                if sx is not None and sy is not None and not getattr(state, "test_track_csrt", False):
                    dx = sx - state.crosshair_x
                    dy = sy - state.crosshair_y
                    err_dist_px = math.sqrt(dx * dx + dy * dy)
                    cv2.putText(
                        frame, f"Sim err: dX={dx:.1f} dY={dy:.1f} dist={err_dist_px:.1f} px",
                        (10, y_l3), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )
            if state.mode == "test" and getattr(state, "test_track_csrt", False):
                if getattr(state, "sim_csrt_initialized", False):
                    err_px = getattr(state, "sim_csrt_error_px", None)
                    r_val = getattr(state, "sim_csrt_r", None)
                    if err_px is not None and r_val is not None:
                        n_txt = f"N: error_px={err_px:.1f} r={r_val:.1f} — orange=guidance. Rotate arm? (no) [N]=off"
                    else:
                        n_txt = "N: Red=bbox center, orange=guidance. Rotate arm? (no) [N]=off"
                else:
                    n_txt = "N: Drag to select object  [N]=off"
                cv2.putText(
                    frame, n_txt,
                    (10, y_l3), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                )
            if state.mode == "test" and getattr(state, "test_teaching", False):
                src = getattr(state, "teaching_source", "auto")
                use_m = "On" if getattr(state, "teaching_use_model", False) else "Off"
                pidx = getattr(state, "teaching_pattern_index", 0) % len(TEACHING_PATTERN_NAMES)
                pname = TEACHING_PATTERN_NAMES[pidx] if src == "auto" else "-"
                j_line1 = f"J: Teaching  [A]=Click/Auto  [M]=Model  [F]=Retrain  source={src}  model={use_m}  pattern={pname}"
                cv2.putText(frame, j_line1, (10, y_j1), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                if getattr(state, "teaching_csv_path", None) is not None:
                    csv_name = state.teaching_csv_path.name
                    cv2.putText(frame, f"Recording: {csv_name}", (10, y_j1 + _lh), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                if getattr(state, "teaching_retrain_in_progress", False):
                    cv2.putText(frame, "Retraining... (wait)", (10, y_j1 + _lh * 2), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                if getattr(state, "teaching_retrain_result", None):
                    res_y = y_j1 + (_lh * 3 if getattr(state, "teaching_retrain_in_progress", False) else _lh * 2)
                    cv2.putText(frame, f"Retrain result: {state.teaching_retrain_result}", (10, res_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            # มุมขวาล่าง: โหมดแมป (Homography / Px/deg) + ค่าความแม่นยำ recal
            HUD_BR_Y1 = h - int(40 * HUD_TEXT_SCALE)
            HUD_BR_Y2 = h - int(18 * HUD_TEXT_SCALE)
            if state.mode == "test":
                if getattr(state, "use_pixel_per_degree", False):
                    if state.px_per_deg_x is not None and state.px_per_deg_y is not None:
                        mode_br = f"Map: Px/deg {state.px_per_deg_x:.1f}, {state.px_per_deg_y:.1f}"
                    else:
                        mode_br = "Map: Px/deg (need cal)"
                else:
                    mode_br = "Map: Homography" + (" (grid only)" if getattr(state, "test_use_grid_only", False) else "")
                (tw, _), _ = cv2.getTextSize(mode_br, cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, THICKNESS_THIN)
                cv2.putText(frame, mode_br, (w - tw - 10, HUD_BR_Y1), cv2.FONT_HERSHEY_SIMPLEX, FONT_SMALL, (0, 0, 255), THICKNESS_THIN)
            if getattr(state, "recal_accuracy_text", None):
                (tw2, _), _ = cv2.getTextSize(state.recal_accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_MID, THICKNESS_THICK)
                cv2.putText(frame, state.recal_accuracy_text, (w - tw2 - 10, HUD_BR_Y2), cv2.FONT_HERSHEY_SIMPLEX, FONT_MID, (0, 0, 255), THICKNESS_THICK)
            # Test: click/actual/error block above main HUD (bottom-left)
            TEST_HUD_TOP = y_j1 - 6 * _lh
            if state.mode == "test":
                LINE_H = _lh
            if state.mode == "test" and state.last_test_pixel is not None:
                px, py = state.last_test_pixel[0], state.last_test_pixel[1]
                click_px_txt = f"Click pixel (before move): ({px:.1f}, {py:.1f})"
                cv2.putText(frame, click_px_txt, (10, TEST_HUD_TOP), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                if state.last_test_arm is not None:
                    tx, ty = state.last_test_arm[0], state.last_test_arm[1]
                    map_txt = f"Target (will move to): X={tx:.2f} Y={ty:.2f}"
                    cv2.putText(frame, map_txt, (10, TEST_HUD_TOP + LINE_H), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            # โหมด Test: หลังกด ENTER แสดงตำแหน่งจริง (ถ้ามีจาก sync) + ความคลาดเคลื่อน — เริ่มบรรทัดหลัง block ด้านบน
            if state.mode == "test" and state.last_test_arm is not None:
                y_row = TEST_HUD_TOP + LINE_H * 2 if state.last_test_pixel is not None else TEST_HUD_TOP + LINE_H
                if getattr(state, "last_test_actual_arm", None) is not None:
                    ax, ay = state.last_test_actual_arm[0], state.last_test_actual_arm[1]
                    tx, ty = state.last_test_arm[0], state.last_test_arm[1]
                    err_x, err_y = tx - ax, ty - ay
                    err_deg = math.sqrt(err_x * err_x + err_y * err_y)
                    actual_txt = f"Actual (after move): X={ax:.2f} Y={ay:.2f}"
                    cv2.putText(frame, actual_txt, (10, y_row), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                    err_txt = f"Error (deg): dX={err_x:.3f} dY={err_y:.3f}  dist={err_deg:.3f} deg"
                    cv2.putText(frame, err_txt, (10, y_row + LINE_H), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                    y_row += LINE_H * 2
                # x,y pixel ศูนย์เล็ง + Error (pixel) — แสดงเฉพาะเมื่อกล้องไม่ติดแขน (CAMERA_ON_ARM=False)
                if not CAMERA_ON_ARM and getattr(state, "last_test_crosshair_pixel", None) is not None:
                    ch = state.last_test_crosshair_pixel
                    crosshair_txt = f"Crosshair pixel (after adjust): ({ch[0]:.1f}, {ch[1]:.1f})"
                    cv2.putText(frame, crosshair_txt, (10, y_row), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
                    y_row += LINE_H
                    if state.last_test_pixel is not None:
                        epx = state.last_test_pixel[0] - ch[0]
                        epy = state.last_test_pixel[1] - ch[1]
                        err_px_dist = math.sqrt(epx * epx + epy * epy)
                        err_px_txt = f"Error (pixel, click vs crosshair): dX={epx:.1f} dY={epy:.1f}  dist={err_px_dist:.1f} px"
                        cv2.putText(frame, err_px_txt, (10, y_row), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            if state.mode == "test" and getattr(state, "test_use_grid_only", False):
                has_actual = getattr(state, "last_test_actual_arm", None) is not None
                has_crosshair = getattr(state, "last_test_crosshair_pixel", None) is not None
                has_crosshair_visible = has_crosshair and not CAMERA_ON_ARM
                y_base = TEST_HUD_TOP + LINE_H * 2 if state.last_test_pixel is not None else TEST_HUD_TOP + LINE_H
                y_g = y_base + LINE_H * 2 if (has_actual and has_crosshair_visible) else (y_base + LINE_H if has_crosshair_visible else (y_base + LINE_H * 2 if has_actual else y_base))
                cv2.putText(frame, "Grid only (no homography) [g]", (10, y_g), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            # แสดง "Target clipped to limit" เมื่อเป้าหมายอยู่นอก limit (bottom-left, above HUD)
            if getattr(state, "limit_clipped_message_until", 0) > 0 and time.time() < state.limit_clipped_message_until:
                clip_msg = "Target clipped to limit"
                cv2.putText(frame, clip_msg, (10, _y0 - 10 * _lh), cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN)
            # แสดง "Saved!" และความแม่นยำ (reprojection error) หลังบันทึกสำเร็จ
            if getattr(state, "save_message_until", 0) > 0 and time.time() < state.save_message_until:
                save_msg = "Saved!"
                (tw, th), _ = cv2.getTextSize(save_msg, cv2.FONT_HERSHEY_SIMPLEX, FONT_TITLE, THICKNESS_THICK)
                cx, cy = w // 2 - tw // 2, int(80 * HUD_TEXT_SCALE)
                cv2.putText(frame, save_msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, FONT_TITLE, (0, 255, 0), THICKNESS_THICK)
                if getattr(state, "save_accuracy_text", None):
                    cv2.putText(
                        frame, state.save_accuracy_text, (10, _y0 - 10 * _lh),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )
            # แสดง "Re-calibrated!" หลังกด [R]
            if getattr(state, "recal_message_until", 0) > 0 and time.time() < state.recal_message_until:
                rec_msg = "Re-calibrated!"
                (tw, th), _ = cv2.getTextSize(rec_msg, cv2.FONT_HERSHEY_SIMPLEX, FONT_TITLE, THICKNESS_THICK)
                cx, cy = w // 2 - tw // 2, int(80 * HUD_TEXT_SCALE)
                cv2.putText(frame, rec_msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, FONT_TITLE, (0, 255, 255), THICKNESS_THICK)
                if getattr(state, "recal_message_text", None):
                    cv2.putText(
                        frame, state.recal_message_text, (10, _y0 - 10 * _lh),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_HUD, COLOR_HUD_RED, THICKNESS_THIN
                    )

            # โหมด Calibration: ไม่ขยับแขนตอนคลิก — ผู้ใช้หมุนจอยเองแล้วกดปุ่มยืนยัน
            # โหมด Test: คลิกแล้วแขนไปมุมเป้าหมาย (Full HD + ไฟล์ save ล่าสุด)

            scale = min(screen_w / w, screen_h / h, 1.0)
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            state.display_w = scaled_w
            state.display_h = scaled_h
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
            # โหมดรีเชค: ENTER = จุดถัดไป, ESC = ออกจาก verify
            if state.verify_index is not None:
                if key8 in (KEY_ENTER, KEY_ENTER_LF):
                    state.verify_index += 1
                    if state.verify_index >= len(state.verify_points):
                        state.verify_index = None
                        state.verify_points = []
                        state.verify_last_moved_index = -1
                elif key8 == KEY_ESC:
                    state.verify_index = None
                    state.verify_points = []
                    state.verify_last_moved_index = -1
            elif getattr(state, "pending_test_move", None) is not None:
                if key8 in (KEY_ENTER, KEY_ENTER_LF):
                    cmd1, cmd2, px_out, py_out = state.pending_test_move
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    pos_x = getattr(arm, "pos_x", 0.0)
                    pos_y = getattr(arm, "pos_y", 0.0)
                    state.smooth_start = (pos_x, pos_y)
                    state.smooth_target = (cmd1, cmd2)
                    state.smooth_start_time = time.time()
                    state.smooth_duration = SMOOTH_MOVE_DURATION_SEC
                    state.pending_test_move = None
                elif key8 == KEY_ESC:
                    state.pending_test_move = None
            elif getattr(state, "test_awaiting_confirm", False) and state.mode == "test" and key8 in (KEY_ENTER, KEY_ENTER_LF):
                # ปรับแขนด้วยลูกศรให้ตรงแล้วกด ENTER ยืนยัน — เก็บ actual_arm + ศูนย์เล็งหลังขยับให้ตรง
                if hasattr(arm, "sync_position_from_grbl"):
                    try:
                        arm.sync_position_from_grbl()
                    except Exception:
                        pass
                actual_x = getattr(arm, "pos_x", 0.0)
                actual_y = getattr(arm, "pos_y", 0.0)
                state.last_test_actual_arm = (actual_x, actual_y)
                state.last_test_crosshair_pixel = (state.crosshair_x, state.crosshair_y)  # ศูนย์เล็งหลังจากขยับให้ตรง
                state.last_test_actual_pixel = None
                if load_grid_json and arm_degrees_to_pixel and state.last_test_pixel is not None:
                    data = load_grid_json(JSON_PATH)
                    if data is not None:
                        apx = arm_degrees_to_pixel(actual_x, actual_y, data)
                        if apx is not None:
                            state.last_test_actual_pixel = apx
                state.test_awaiting_confirm = False
            else:
                if key8 == KEY_ESC:
                    if state.cells or state.center_fine_grid.get("fine_cells"):
                        _save_json(state)
                    break
                if key8 == KEY_SAVE:
                    _save_json(state)
            if key8 == KEY_MODE_TEST:
                if JSON_PATH.is_file() and load_grid_json:
                    state.test_data = load_grid_json(JSON_PATH)
                    state.mode = "test"
                    # เข้า Test โหมด homography: ตั้ง crosshair จากไฟล์ calibration เพื่อให้ศูนย์เล็งตรงกับที่บันทึก
                    if state.mode == "test" and not getattr(state, "use_pixel_per_degree", False) and state.test_data is not None and get_crosshair:
                        data = state.test_data
                        cx, cy = get_crosshair(data)
                        ow = int(data.get("output_width", REF_OUTPUT_W))
                        oh = int(data.get("output_height", REF_OUTPUT_H))
                        out_w = state.output_w or REF_OUTPUT_W
                        out_h = state.output_h or REF_OUTPUT_H
                        if ow > 0 and oh > 0 and (ow != out_w or oh != out_h):
                            cx = cx * out_w / ow
                            cy = cy * out_h / oh
                        state.crosshair_x = cx
                        state.crosshair_y = cy
                state.smooth_target = None
                state.smooth_start = None
                state.limit_clipped_message_until = 0.0
                state.pending_test_move = None
                state.test_feedback_pending = False
                state.test_feedback_choice_pending = False
                state.test_home_required_message_until = 0.0
                if state.mode == "test":
                    if hasattr(arm, "sync_position_from_grbl"):
                        try:
                            arm.sync_position_from_grbl()
                        except Exception:
                            pass
                    px_arm = getattr(arm, "pos_x", 0.0)
                    py_arm = getattr(arm, "pos_y", 0.0)
                    if abs(px_arm) > 1.0 or abs(py_arm) > 1.0:
                        try:
                            arm.go_home(blocking=False)
                        except Exception:
                            pass
                        state.test_awaiting_home = True
                    else:
                        state.test_awaiting_home = False
            if key8 == KEY_GRID_TOGGLE and state.mode == "test":
                state.test_use_grid_only = not state.test_use_grid_only
            if key8 == KEY_PX_DEG_TOGGLE and state.mode == "test":
                state.use_pixel_per_degree = not state.use_pixel_per_degree
                # กด D ปิด Px/deg: คืน crosshair จากไฟล์ calibration หลัก (homography)
                if not state.use_pixel_per_degree and load_grid_json and get_crosshair:
                    data = load_grid_json(JSON_PATH)
                    if data is not None:
                        cx, cy = get_crosshair(data)
                        ow = int(data.get("output_width", REF_OUTPUT_W))
                        oh = int(data.get("output_height", REF_OUTPUT_H))
                        out_w = state.output_w or REF_OUTPUT_W
                        out_h = state.output_h or REF_OUTPUT_H
                        if ow > 0 and oh > 0 and (ow != out_w or oh != out_h):
                            cx = cx * out_w / ow
                            cy = cy * out_h / oh
                        state.crosshair_x = cx
                        state.crosshair_y = cy
            if key8 == KEY_P and state.mode == "test":
                state.test_continuous = not getattr(state, "test_continuous", False)
                if state.test_continuous:
                    state.pending_test_move = None
                    state.smooth_target = None
                    state.smooth_start = None
                    state.continuous_velocity_mm_s = 0.0
                else:
                    state.continuous_target_pan = None
                    state.continuous_target_tilt = None
                    state.continuous_velocity_mm_s = 0.0
                    state.continuous_error_deg_smooth = None
            if key8 == KEY_L and state.mode == "test":
                state.test_track_sim = not getattr(state, "test_track_sim", False)
                if state.test_track_sim:
                    state.sim_t_start = time.time()
                    state.track_sim_arm_auto = True
                    state.sim_drag_active = False
                    state.sim_display_px = None
                    state.sim_display_py = None
                    state.sim_kalman_x = None
                    state.sim_kalman_P = None
                    state.sim_internal_time = 0.0
                    state.last_sim_update_time = 0.0
                    state.last_sim_detection_time = 0.0
                    import random
                    state.sim_speed_switch_at = time.time() + random.uniform(2.0, 6.0)
                    state.sim_speed_state = "medium"
                    state.pending_test_move = None
                    state.smooth_target = None
                    state.smooth_start = None
                    state.continuous_velocity_mm_s = 0.0
                else:
                    state.sim_drag_active = False
                    state.continuous_target_pan = None
                    state.continuous_target_tilt = None
                    state.continuous_velocity_mm_s = 0.0
                    state.continuous_error_deg_smooth = None
                    state.sim_display_px = None
                    state.sim_display_py = None
                    state.sim_kalman_x = None
                    state.sim_kalman_P = None
            if key8 == KEY_N and state.mode == "test":
                state.test_track_csrt = not getattr(state, "test_track_csrt", False)
                state.sim_csrt_initialized = False
                state.sim_csrt_tracker = None
                state.sim_csrt_pending_bbox = None
                state.sim_csrt_bbox = None
                state.sim_csrt_drag_start = None
                state.sim_csrt_drag_current = None
                state.sim_csrt_smooth_px = None
                state.sim_csrt_smooth_py = None
                state.sim_csrt_error_px = None
                state.sim_csrt_r = None
                state.sim_csrt_lost = False
                if state.test_track_csrt:
                    state.sim_display_px = None
                    state.sim_display_py = None
                else:
                    state.continuous_target_pan = None
                    state.continuous_target_tilt = None
                    state.continuous_velocity_mm_s = 0.0
                    state.continuous_error_deg_smooth = None
            if key8 == KEY_J and state.mode == "test":
                if getattr(state, "test_track_sim", False):
                    state.l_track_joystick_style = not getattr(state, "l_track_joystick_style", False)
                else:
                    state.test_teaching = not getattr(state, "test_teaching", False)
                if state.test_teaching:
                    state.teaching_sim_start = time.time()
                    import random
                    state.teaching_pattern_switch_at = time.time() + random.uniform(5.0, 18.0)
                    state.pending_test_move = None
                    state.smooth_target = None
                    state.smooth_start = None
                    state.continuous_velocity_mm_s = 0.0
                    try:
                        from datetime import datetime
                        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        state.teaching_csv_path = CALIBRATION_DIR / f"teaching_{ts}.csv"
                        state.teaching_csv_handle = open(state.teaching_csv_path, "w")
                        header = "timestamp_utc,err_pan,err_tilt,error_deg,last_delta_pan,last_delta_tilt,dt_send,d_pan,d_tilt,current_pan_deg,current_tilt_deg,action_pan,action_tilt,next_err_pan,next_err_tilt,next_error_deg,use_model"
                        state.teaching_csv_handle.write(header + "\n")
                        state.teaching_csv_handle.flush()
                    except Exception:
                        state.teaching_csv_handle = None
                        state.teaching_csv_path = None
                else:
                    if getattr(state, "teaching_csv_handle", None) is not None:
                        try:
                            state.teaching_csv_handle.close()
                        except Exception:
                            pass
                        state.teaching_csv_handle = None
                    state.teaching_csv_path = None
                    state.continuous_target_pan = None
                    state.continuous_target_tilt = None
                    state.continuous_velocity_mm_s = 0.0
                    state.teaching_last_state = None
                    state.teaching_last_action = None
                    state.teaching_last_target = None
                    state.teaching_last_ts = None
                    state.teaching_last_err = None
                    state.teaching_last_use_model = None
                    state.teaching_retrain_result = None
            if key8 == KEY_TEACHING_SOURCE and state.mode == "test":
                if getattr(state, "test_teaching", False):
                    state.teaching_source = "auto" if getattr(state, "teaching_source", "auto") == "click" else "click"
                    if state.teaching_source == "auto":
                        state.teaching_sim_start = time.time()
                        import random
                        state.teaching_pattern_switch_at = time.time() + random.uniform(5.0, 18.0)
                elif getattr(state, "test_track_sim", False):
                    state.track_sim_arm_auto = not getattr(state, "track_sim_arm_auto", True)
            if key8 == KEY_TEACHING_USE_MODEL and state.mode == "test" and getattr(state, "test_teaching", False):
                state.teaching_use_model = not getattr(state, "teaching_use_model", False)
                if state.teaching_use_model and getattr(state, "_teaching_model", None) is None:
                    try:
                        _root = Path(__file__).resolve().parent
                        _rel = getattr(config, "CAM4_ARM_LEARNED_AIM_MODEL_PATH", "aim_controller_model/aim_model.pt") if config else "aim_controller_model/aim_model.pt"
                        model_path = _root / _rel
                        onnx_path = model_path.with_suffix(".onnx") if model_path.suffix.lower() == ".pt" else model_path
                        if onnx_path.suffix.lower() == ".onnx" and onnx_path.exists():
                            from aim_controller_model.model import load_onnx, predict_delta_onnx, normalize_state
                            state._teaching_model = load_onnx(onnx_path)
                            if state._teaching_model is not None:
                                sh = state._teaching_model.get_inputs()[0].shape
                                state._teaching_model_input_dim = int(sh[1]) if len(sh) > 1 else 10
                                state._teaching_predict_fn = lambda m, sv, ms: predict_delta_onnx(m, sv, ms)
                                state._teaching_normalize_fn = normalize_state
                        else:
                            from aim_controller_model.model import load_model, predict_delta, normalize_state
                            state._teaching_model = load_model(model_path)
                            if state._teaching_model is not None:
                                state._teaching_model_input_dim = getattr(state._teaching_model, "input_dim", 10)
                                state._teaching_predict_fn = predict_delta
                                state._teaching_normalize_fn = normalize_state
                    except Exception:
                        state._teaching_model = None
                        state._teaching_use_model = False
            if key8 == KEY_TEACHING_RETRAIN and state.mode == "test" and getattr(state, "test_teaching", False) and not getattr(state, "teaching_retrain_in_progress", False):
                state.teaching_retrain_in_progress = True
                try:
                    if getattr(state, "teaching_csv_handle", None) is not None:
                        try:
                            state.teaching_csv_handle.flush()
                        except Exception:
                            pass
                    npz_dir = Path(__file__).resolve().parent / "aim_controller_model"
                    npz_path = npz_dir / "aim_buffer.npz"
                    csv_paths = list(CALIBRATION_DIR.glob("teaching_*.csv")) if CALIBRATION_DIR.exists() else []
                    if state.teaching_csv_path is not None and state.teaching_csv_path.exists() and state.teaching_csv_path not in csv_paths:
                        csv_paths.append(state.teaching_csv_path)
                    states_list, actions_list, next_states_list = [], [], []
                    for csv_path in csv_paths:
                        try:
                            with open(csv_path, "r") as f:
                                lines = f.readlines()
                            if len(lines) < 2:
                                continue
                            for line in lines[1:]:
                                parts = line.strip().split(",")
                                if len(parts) >= 16:
                                    st = tuple(float(parts[i]) for i in range(1, 11))
                                    ac = (float(parts[11]), float(parts[12]))
                                    ns = (float(parts[13]), float(parts[14]), float(parts[15]))
                                    states_list.append(st)
                                    actions_list.append(ac)
                                    next_states_list.append(ns)
                        except Exception:
                            continue
                    if len(states_list) >= 20:
                        npz_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(npz_path, states=np.array(states_list, dtype=np.float32), actions=np.array(actions_list, dtype=np.float32), next_states=np.array(next_states_list, dtype=np.float32))
                        from aim_controller_model.train import run_retrain
                        model_path = Path(__file__).resolve().parent / getattr(config, "CAM4_ARM_LEARNED_AIM_MODEL_PATH", "aim_controller_model/aim_model.pt") if config else npz_dir / "aim_model.pt"
                        th_red = getattr(config, "CAM4_ARM_LEARNED_AIM_THRESHOLD_RED_DEG", 0.35) if config else 0.35
                        th_orange = getattr(config, "CAM4_ARM_LEARNED_AIM_THRESHOLD_ORANGE_DEG", 0.7) if config else 0.7
                        run_retrain(npz_path, model_path, threshold_red_deg=th_red, threshold_orange_deg=th_orange, min_transitions=20)
                        state.teaching_retrain_result = f"Retrain done: {len(states_list)} transitions"
                    else:
                        state.teaching_retrain_result = f"Need >= 20 transitions (got {len(states_list)})"
                except Exception as e:
                    state.teaching_retrain_result = f"Retrain error: {e}"
                state.teaching_retrain_in_progress = False
            if key8 == KEY_NEXT_PATTERN and state.mode == "test" and getattr(state, "test_track_sim", False):
                state.sim_pattern_index = (getattr(state, "sim_pattern_index", 0) + 1) % len(SIM_PATTERN_NAMES)
            # โหมด L: [K] สลับ speed lock — Auto → Fast only → Slow only → Stop only → Auto
            if key8 == KEY_SPEED_LOCK and state.mode == "test" and getattr(state, "test_track_sim", False):
                lock = getattr(state, "sim_speed_lock", None)
                state.sim_speed_lock = ("fast" if lock is None else "slow" if lock == "fast" else "stop" if lock == "slow" else None)
            if key8 == KEY_TELEMETRY_PLOT and state.mode == "test":
                _show_continuous_telemetry_plot(state, live=False)
            # โหมด Test: กด 0-9 เลือก pattern การบินจำลองโดยตรง (ใช้ได้แม้ยังไม่เปิด L)
            if state.mode == "test" and ord("0") <= key8 <= ord("9"):
                state.sim_pattern_index = (key8 - ord("0")) % len(SIM_PATTERN_NAMES)
            if key8 == KEY_RECALIB and state.mode == "test" and state.verify_index is None:
                _recalibrate_from_log(state)
            if key8 == KEY_HOME and state.mode == "test" and state.verify_index is None:
                # กด H: เก็บค่าทุกอย่างลง log — click_pixel, target_arm, crosshair_pixel, error (click vs crosshair)
                if getattr(state, "test_move_log_pending", False) and state.last_test_pixel is not None and state.last_test_arm is not None:
                    actual_arm = getattr(state, "last_test_actual_arm", None) or state.last_test_arm
                    actual_px = getattr(state, "last_test_actual_pixel", None)
                    if CAMERA_ON_ARM:
                        crosshair_px, err_px, err_py, err_dist = None, None, None, None
                    else:
                        crosshair_px = getattr(state, "last_test_crosshair_pixel", None)
                        err_px, err_py, err_dist = None, None, None
                        if crosshair_px is not None and state.last_test_pixel is not None:
                            px_click, py_click = state.last_test_pixel[0], state.last_test_pixel[1]
                            err_px = px_click - crosshair_px[0]
                            err_py = py_click - crosshair_px[1]
                            err_dist = math.sqrt(err_px * err_px + err_py * err_py)
                    log_path = TEST_MOVE_LOG_PXDEG_PATH if getattr(state, "use_pixel_per_degree", False) else TEST_MOVE_LOG_PATH
                    _append_test_move_log(
                        state.last_test_pixel,
                        state.last_test_arm,
                        actual_arm,
                        log_path,
                        output_w=state.output_w or REF_OUTPUT_W,
                        output_h=state.output_h or REF_OUTPUT_H,
                        actual_pixel=actual_px,
                        crosshair_pixel=crosshair_px,
                        error_pixel_x=err_px,
                        error_pixel_y=err_py,
                        error_pixel_dist=err_dist,
                    )
                    state.test_move_log_pending = False
                try:
                    arm.go_home(blocking=False)
                except Exception:
                    pass
                state.test_awaiting_home = True
                state.pending_test_move = None
            if key8 == KEY_MODE_CALIB:
                state.mode = "calibration"
                state.test_data = None
                state.test_reference_capture = None
                state.pending_test_move = None
                state.test_awaiting_home = False
                state.test_feedback_pending = False
                state.test_feedback_choice_pending = False
                state.test_awaiting_confirm = False
                state.test_move_log_pending = False
                state.test_home_required_message_until = 0.0
                state.smooth_target = None
                state.smooth_start = None
                state.limit_clipped_message_until = 0.0
                state.test_continuous = False
                state.test_track_sim = False
                state.test_teaching = False
                state.sim_kalman_x = None
                state.sim_kalman_P = None
                if getattr(state, "teaching_csv_handle", None) is not None:
                    try:
                        state.teaching_csv_handle.close()
                    except Exception:
                        pass
                    state.teaching_csv_handle = None
                state.teaching_csv_path = None
                state.continuous_target_pan = None
                state.continuous_target_tilt = None
                state.continuous_velocity_mm_s = 0.0
                state.sim_display_px = None
                state.sim_display_py = None
                state.teaching_last_state = None
                state.teaching_last_action = None
                state.teaching_last_target = None
                state.teaching_last_ts = None
                state.teaching_last_err = None
                state.teaching_last_use_model = None
                state.teaching_retrain_result = None
            # ลูกศรคีย์บอร์ด: ขยับแขน step 0.1° สำหรับความแม่นยำสูง
            if key in ARROW_KEYS_LEFT:
                try:
                    arm.move_relative(-ARROW_STEP_DEG, 0.0, blocking=False)
                except Exception:
                    pass
            elif key in ARROW_KEYS_RIGHT:
                try:
                    arm.move_relative(ARROW_STEP_DEG, 0.0, blocking=False)
                except Exception:
                    pass
            elif key in ARROW_KEYS_UP:
                try:
                    arm.move_relative(0.0, ARROW_STEP_DEG, blocking=False)
                except Exception:
                    pass
            elif key in ARROW_KEYS_DOWN:
                try:
                    arm.move_relative(0.0, -ARROW_STEP_DEG, blocking=False)
                except Exception:
                    pass
    finally:
        cv2.destroyAllWindows()
        if getattr(config, "CAM4_ARM_RETURN_TO_REF_ON_DISCONNECT", True):
            try:
                arm.go_home(blocking=True)
            except Exception:
                pass
        arm.disconnect() if hasattr(arm, "disconnect") else None
        cam.release()


def _joystick_confirm_pressed(joystick_reader) -> bool:
    """ตรวจว่าปุ่มยืนยัน (ปุ่ม 1 = index 0) ถูกกดหรือไม่."""
    if joystick_reader is None or not getattr(joystick_reader, "enabled", False):
        return False
    js = getattr(joystick_reader, "joystick", None)
    if js is None:
        return False
    try:
        n = js.get_numbuttons()
        if JOY_CONFIRM_BUTTON_INDEX < n:
            return bool(js.get_button(JOY_CONFIRM_BUTTON_INDEX))
    except Exception:
        pass
    return False


if __name__ == "__main__":
    main()

