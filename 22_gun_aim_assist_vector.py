"""
gun_aim_assist_vector.py
Gun Aim Assist Vector: ระบบช่วยเล็งกล้องติดปืน (drive แขนแบบ vector)

--- Keyboard Controls ---
  Q           : ออกจากโปรแกรม
  A           : สลับโหมด AUTO   (cam8 cue ขับแขน)
  M           : เข้า/ออกโหมด Mapping cam8->arm (เข้าแล้วแขนไป REF อัตโนมัติ; คลิก cell, C/Enter บันทึก cell, H=go REF, S=บันทึก JSON)
  MANUAL      : ปุ่ม 9 บนจอย หรือปุ่ม 3 ปลด LOCK -> MANUAL
  F           : สลับโหมด SAFE   (แขนหยุด)
  L           : สลับโหมด LOCK   (lock ตาม YOLO detection)
  I           : เปิด/ปิดโดรนจำลอง (SIMULATION) — default ปิด (โหมดจริง); เปิดแล้วซ้อม LOCK+ยิงได้ (←→↑↓ บังคับ, V pattern, N สลับระยะ 50-200m)
  SPACE       : ยิง (ยกเว้น SAFE mode)
  C           : เข้าโหมดปรับ center ศูนย์เล็ง (WASD/arrow เลื่อน, Enter บันทึก, R reset)
  S           : เข้าโหมด settings (1-4 เลือก field, arrow ปรับค่า, Enter บันทึก)
  R           : แสดง/ซ่อน calibration status overlay (pixel_per_degree ของกล้องที่เลือก)
  P           : reconnect แขน (homing+home ใหม่ หลัง power-cycle โดยไม่ต้องปิดโปรแกรม)
  H           : สั่งแขนกลับ home ทันที (go_home = G0 ไป (0,0) เบา ๆ ไม่ re-home; บังคับ SAFE ระหว่างวิ่ง)
  J           : เข้า/ออกโหมด JOG — หมุนแขนเองด้วยคีย์บอร์ด ทำงานแม้แขนค้าง/STALL/SAFE
                (เข้าแล้ว: ปลดล็อก $X อัตโนมัติ; ลูกศร/WASD ขยับ, [ ] ปรับ step, X ปลดล็อกซ้ำ,
                 Z ตั้ง home ตรงนี้ (re-zero, สำหรับแขนไม่มี limit switch), H กลับ home, J/Esc ออก)
  B           : เปิด/ปิด online residual bias learning — AUTO mode only
                (HUD แสดง BIAS:ON/OFF(B) + จำนวน cell ที่มีข้อมูล)
  Y           : สอน aim trim จากผลยิง LOCK/sim ที่ผ่านมา (least-squares หา offset, cap step,
                persist calibration_data/<cam>_fire_tune.json). sim = ครู ground-truth ที่แม่นสุด

--- Joystick Controls ---
  Joystick axis  : ควบคุมแขน (MANUAL/LOCK)
  ปุ่ม 5 (idx 4) : เข้า LOCK + หมุนแขนไป center YOLO ที่ใกล้ศูนย์เล็งที่สุด
  ปุ่ม 3 (idx 2) : ปลด LOCK -> MANUAL
  ปุ่ม sensitivity: สลับความเร็ว joystick (ช้า/กลาง/สูง)
  ปุ่ม class cycle (ค่าเริ่มต้น 8): กดค้างเลื่อน YOLO target class (เฉพาะโหมด multiclass ใน config)
  ปุ่ม detection toggle (ค่าเริ่มต้น 10): multiclass = RGB/Thermal | drone_only = 640/1280
  T           : สลับ detection engine (ตาม CAM4_ARM_YOLO_DETECTION_MODE ใน config)

--- Calibration ---
  cam8 -> arm  : กด M ในแอป (mapping mode) หรือ python3 cam8_arm_grid_calibrator.py
  camera px/deg: กด G ในแอป (embedded calibrator) หรือ python3 cam4_arm_mouse_grid_calibrator.py --camera <camera_name>
  bias reset   : ลบ calibration_data/cam8_auto_residual_bias.json แล้วรันใหม่
  AUTO+cam8    : ถ้า CAM8_CROSSHAIR_TRIM_ENABLED บวกชดเชยมุมจากศูนย์เล็ง cam4 เทียบกลางเฟรม (px/deg)
"""
import cv2
import datetime
import json
import math
import numpy as np
import os
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from collections import deque

try:
    from fast_motion_sky import CameraStream
    from config import get_camera_config, ACTIVE_CAMERA
    from camera_startup import (
        get_runtime_cleanup,
        open_camera_with_retries,
        wait_for_camera_first_frame,
        cam_hud_label_and_color,
        get_stream_health,
        network_loss_placeholder,
    )
except ImportError as e:
    print(f"Error: {e}")
    raise

# Optional: โดรนเสมือนสำหรับทดสอบ LOCK กับแขนจริง (config.LOCK_SIM_TARGET_ENABLED)
try:
    from lock_sim_target import VirtualDroneTarget
except ImportError:
    VirtualDroneTarget = None

# Optional: Arm controller + tracker for cam4 (จะใช้เฉพาะเมื่อ config.CAM4_ARM_ENABLED = True)
try:
    from cam4_arm_controller import Cam4ArmController, SimCam4ArmController
    from cam4_arm_tracker import Cam4ArmTracker
except ImportError as _arm_import_err:
    Cam4ArmController = None
    SimCam4ArmController = None
    Cam4ArmTracker = None
    print(f"⚠️ cam4_arm_controller/tracker import failed — arm disabled: {_arm_import_err}")

# Optional: Arm cue receiver (รับ cam8 confirmed target จาก 11 ผ่าน UDP)
try:
    from arm_cue_receiver import ArmCueReceiver
except ImportError:
    ArmCueReceiver = None
    print("⚠️ arm_cue_receiver not found — cam8 AUTO cue disabled")

try:
    from effector_ws_exporter import EffectorWsExporter, build_effector_payload
except ImportError:
    EffectorWsExporter = None
    build_effector_payload = None
    print("⚠️ effector_ws_exporter not found — effector WS disabled")

try:
    from rtmp_display_streamer import RtmpDisplayStreamer as _RtmpDisplayStreamer
except ImportError:
    _RtmpDisplayStreamer = None
    print("⚠️ rtmp_display_streamer not found — RTMP streaming disabled")

# Cam8 → arm calibration lookup (ใช้ pixel_to_arm_degrees กับ cam8_mouse_grid_lookup.json)
try:
    from cam4_arm_grid_lookup import pixel_to_arm_degrees as _cam8_pixel_to_arm_degrees, load as _cam8_calib_load
    _CAM8_CALIB_FILE = Path(__file__).resolve().parent / "calibration_data" / "cam8_mouse_grid_lookup.json"
except ImportError:
    _cam8_pixel_to_arm_degrees = None
    _cam8_calib_load = None
    _CAM8_CALIB_FILE = None

# Cam8 JSON still stores homography; False = lookup uses coarse/fine grid only (see cam4_arm_grid_lookup).
CAM8_PIXEL_TO_ARM_USE_HOMOGRAPHY = False

# After cam8 lookup, add pan/tilt trim so LOS matches cam4 reticle vs geometric frame center (px/deg).
CAM8_CROSSHAIR_TRIM_ENABLED = True

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
    import cam8_mapping_embed as cam8_mapping_embed
except ImportError:
    cam8_mapping_embed = None

try:
    import cam4_pxdeg_calibrator_embed as cam4_pxdeg_calibrator_embed
except ImportError:
    cam4_pxdeg_calibrator_embed = None

# ค่าที่ตั้งได้ในแอป (หน้า Settings) + ตัวช่วยคาลิเบรตอัตโนมัติตอนเปลี่ยนกล้อง
try:
    import runtime_config
except ImportError:
    runtime_config = None

try:
    import settings_screen
except ImportError as _e:
    settings_screen = None
    print(f"⚠️ settings_screen ใช้ไม่ได้ ({_e}) — ปุ่ม S จะเป็นหน้า ballistics แบบเดิม")

try:
    import calib_wizard
except ImportError as _e:
    calib_wizard = None
    print(f"⚠️ calib_wizard ใช้ไม่ได้ ({_e}) — ปุ่ม W (คาลิเบรตอัตโนมัติ) ถูกปิด")

# Vector drive (px_per_degree + _variable_step_toward_target) from calibrator / HUD
try:
    from cam4_arm_mouse_grid_calibrator import (
        _variable_step_toward_target,
        _apply_arm_move_relative,
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
    _apply_arm_move_relative = None
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

# =============================================================================
# Constants
# =============================================================================
# --- CSRT / LOCK (track scale, bbox) ---
TRACK_SCALE = 0.25
CSRT_MIN_BBOX = 4
CSRT_UPDATE_EVERY_N_FRAMES = 2
VIRTUAL_CLICK_CAP_PX = 200.0
LOCK_CSRT_JOYSTICK_BBOX_HALF_PX = 120  # 80 * 1.5 = bbox ใหญ่ขึ้น 50% (240×240 px)

# --- Yellow vector (retrained params, same as HUD) ---
YELLOW_MAX_PX = 250
YELLOW_GRADIENT_SCALE_PX = 120.0
YELLOW_SMOOTH_ALPHA = 0.90
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
_VECTOR_CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
YELLOW_PARAMS_JSON = _VECTOR_CALIB_DIR / "cam4_yellow_vector_params.json"
YELLOW_MAX_CURVE_JSON = _VECTOR_CALIB_DIR / "cam4_yellow_max_curve.json"
YELLOW_MAX_FAR_DEG = 3.0
YELLOW_MAX_NEAR_DEG = 0.5
YELLOW_NEAR_FAR_PX = 200.0

try:
    import config as _config_mod
except ImportError:
    _config_mod = None

_CALIBRATION_DIR = None
_PX_DEG_JSON_PATH = None
# กล้องที่ใช้อยู่ (ตั้งใน main); path pixel_per_degree ขึ้นกับชื่อกล้อง
_ACTIVE_CAMERA_NAME = None

# --- Arm cue receiver (cam8 → arm AUTO) ---
ARM_CUE_RECEIVER_PORT = 5765
ARM_CUE_TTL_MS = 500  # ต้องไม่เกิน cue_ttl_ms ที่ฝั่ง 11 ส่งมา

# --- Cam8 cue bbox size matching (ช่วยเลือก detection บน cam4 ใน LOCK/sticky) ---
# เปิด/ปิด feature นี้; ปิดไว้ก่อนเป็น default จนกว่าจะทดสอบ
CAM8_CUE_SIZE_MATCH_ENABLED: bool = False
# น้ำหนักของ size penalty ในสมการ: score = dist_crosshair + weight * size_penalty_px
CAM8_CUE_SIZE_MATCH_WEIGHT: float = 0.5
# ขนาด size_penalty สูงสุดที่อนุญาต (norm L1) ก่อนกรองออก; None = ไม่กรอง
CAM8_CUE_SIZE_MAX_TOL: float = 0.30  # 30% norm ผิดพลาดถือว่าน่าจะไม่ใช่ตัวเดียวกัน


# =============================================================================
# Calibration
# =============================================================================
def _set_active_camera_for_calibration(camera_name: str) -> None:
    """ตั้งชื่อกล้องที่ใช้อยู่ เพื่อให้ _get_px_deg_path() ชี้ไปไฟล์ที่ถูกกล้อง."""
    global _ACTIVE_CAMERA_NAME, _CALIBRATION_DIR, _PX_DEG_JSON_PATH
    _ACTIVE_CAMERA_NAME = camera_name or "cam4"
    _CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
    _PX_DEG_JSON_PATH = _CALIBRATION_DIR / f"{_ACTIVE_CAMERA_NAME}_pixel_per_degree.json"


def _get_px_deg_path():
    global _CALIBRATION_DIR, _PX_DEG_JSON_PATH, _ACTIVE_CAMERA_NAME
    if _CALIBRATION_DIR is None or _PX_DEG_JSON_PATH is None:
        _CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
        name = _ACTIVE_CAMERA_NAME if _ACTIVE_CAMERA_NAME else "cam4"
        _PX_DEG_JSON_PATH = _CALIBRATION_DIR / f"{name}_pixel_per_degree.json"
    return _PX_DEG_JSON_PATH


def get_calibration_status(camera_name: str) -> dict:
    """
    เช็คสถานะ calibration pixel_to_degree ของกล้องที่กำหนด.
    คืน dict: camera_name, path, exists, px_per_deg_x, px_per_deg_y, message.
    """
    calib_dir = Path(__file__).resolve().parent / "calibration_data"
    path = calib_dir / f"{camera_name}_pixel_per_degree.json"
    exists = path.is_file()
    px_x = PX_PER_DEG_X_DEFAULT
    px_y = PX_PER_DEG_Y_DEFAULT
    if exists:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _x = data.get("pixel_per_degree_x")
                _y = data.get("pixel_per_degree_y")
                if _x is not None:
                    px_x = float(_x)
                if _y is not None:
                    px_y = float(_y)
        except Exception:
            pass
    message = "OK" if exists else "NO FILE (default)"
    return {
        "camera_name": camera_name,
        "path": path,
        "path_short": path.name,
        "exists": exists,
        "px_per_deg_x": px_x,
        "px_per_deg_y": px_y,
        "message": message,
    }


def check_camera_geometry(camera_name: str, cam_config: dict, frame_w=None, frame_h=None) -> dict:
    """
    ตรวจว่า 'เรขาคณิตกล้อง' ที่จะใช้เล็ง/ยิง เชื่อถือได้ไหม — เรียกตอน startup.

    เหตุผล: เปลี่ยนกล้องแล้วลืมคาลิเบรต ระบบจะเงียบ ๆ ตกไปใช้ default 50/-50 px/deg
    → ทุก bearing ผิด → ยิงพลาดทุกนัดโดยไม่มีอะไรเตือน. ตัวนี้ทำให้ 'ผิดแล้วดัง'.

    คืน dict: ok, fatal, ppd_x, ppd_y, fov_h, fov_v, problems[], warnings[]
      fatal = True → ห้ามขยับแขน/ยิง (บังคับ SAFE)
    """
    calib_dir = Path(__file__).resolve().parent / "calibration_data"
    path = calib_dir / f"{camera_name}_pixel_per_degree.json"
    problems: List[str] = []
    warnings: List[str] = []
    ppd_x = ppd_y = None
    out_w = out_h = None

    if not path.is_file():
        problems.append(f"CALIB:MISSING — ไม่มี {path.name} (ต้องคาลิเบรต px/deg ก่อน: กด G ในแอป)")
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ppd_x = data.get("pixel_per_degree_x")
            ppd_y = data.get("pixel_per_degree_y")
            out_w = data.get("output_width")
            out_h = data.get("output_height")
        except Exception as e:
            problems.append(f"CALIB:UNREADABLE — อ่าน {path.name} ไม่ได้: {e}")

        if ppd_x is None or ppd_y is None:
            problems.append(f"CALIB:INCOMPLETE — {path.name} ไม่มี pixel_per_degree_x/y")
        else:
            ppd_x = float(ppd_x)
            ppd_y = float(ppd_y)
            if abs(ppd_x) < 1e-6 or abs(ppd_y) < 1e-6:
                problems.append(f"CALIB:ZERO-PPD — ppd={ppd_x:.2f}/{ppd_y:.2f} ใช้ไม่ได้")
            # ทั้งระบบ tilt (LOCK_TILT_SIGN, LOCK_FIRE_TILT_UP_SIGN) ตั้งอยู่บนสมมติฐาน ppd_y < 0
            # ถ้ากล้องใหม่คาลิเบรตออกมาเป็นบวก แขนจะวิ่ง 'หนี' เป้าในแกน tilt + gravity drop กลับทิศ
            elif ppd_y > 0:
                problems.append(
                    f"CALIB:TILT-SIGN — pixel_per_degree_y = {ppd_y:+.2f} (บวก) แต่ระบบสมมติว่าติดลบ "
                    "→ แขนจะวิ่งหนีเป้าในแกน tilt. คาลิเบรตใหม่ หรือกลับ LOCK_TILT_SIGN/LOCK_FIRE_TILT_UP_SIGN"
                )

        if out_w and out_h and frame_w and frame_h:
            if int(out_w) != int(frame_w) or int(out_h) != int(frame_h):
                problems.append(
                    f"CALIB:SIZE-MISMATCH — คาลิเบรตไว้ที่ {out_w}×{out_h} แต่เฟรมจริง {frame_w}×{frame_h}. "
                    "ppd ผูกกับความละเอียดโดยตรง (ไม่ rescale ให้) → ต้องคาลิเบรตใหม่ที่ความละเอียดนี้"
                )

    # FOV ที่ 'จริง' = ขนาดภาพที่คาลิเบรตไว้ ÷ ppd — ไม่ใช่ค่าที่พิมพ์มือใน config
    fov_h = fov_v = None
    ref_w = frame_w or out_w
    ref_h = frame_h or out_h
    if ppd_x and ppd_y and ref_w and ref_h:
        fov_h, fov_v = fov_from_ppd(int(ref_w), int(ref_h), ppd_x, ppd_y)
        cfg_h = cam_config.get("fov_horizontal")
        cfg_v = cam_config.get("fov_vertical")
        for label, derived, cfg in (("H", fov_h, cfg_h), ("V", fov_v, cfg_v)):
            if derived and cfg and abs(derived - float(cfg)) / derived > 0.10:
                warnings.append(
                    f"FOV:{label} config={float(cfg):.1f}° แต่ ppd บอกว่า {derived:.1f}° (ต่าง >10%) "
                    "→ ใช้ค่าจาก ppd (แม่นกว่า). แก้ config ให้ตรงด้วย"
                )

    return {
        "camera_name": camera_name,
        "ok": not problems,
        "fatal": bool(problems),
        "ppd_x": ppd_x,
        "ppd_y": ppd_y,
        "fov_h": fov_h,
        "fov_v": fov_v,
        "problems": problems,
        "warnings": warnings,
    }


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
    """Return allowed yellow max (px) at this error_deg. Far -> small, close -> YELLOW_MAX_PX."""
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
    if error_deg <= YELLOW_MAX_NEAR_DEG:
        return YELLOW_MAX_PX
    if error_deg >= YELLOW_MAX_FAR_DEG:
        return YELLOW_NEAR_FAR_PX
    t = (error_deg - YELLOW_MAX_NEAR_DEG) / (YELLOW_MAX_FAR_DEG - YELLOW_MAX_NEAR_DEG)
    return YELLOW_NEAR_FAR_PX + t * (YELLOW_MAX_PX - YELLOW_NEAR_FAR_PX)



def _load_px_per_deg():
    """Load pixel_per_degree from calibrator or JSON. Returns (px_per_deg_x, px_per_deg_y)."""
    if _load_pixel_per_degree_json is not None:
        data = _load_pixel_per_degree_json()
    else:
        data = None
        path = _get_px_deg_path()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
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


# =============================================================================
# Arm control
# =============================================================================
class _ArmDriveState:
    """State for _variable_step_toward_target (continuous target prev, velocity)."""
    continuous_target_pan_prev = None
    continuous_target_tilt_prev = None
    continuous_target_time_prev = None
    continuous_velocity_mm_s = 0.0


class _ArmPoseHistory:
    """ประวัติท่าแขน (t, pan, tilt) สำหรับ ego-motion compensation ใน LOCK:
    ผล YOLO มาช้ากว่าภาพ (pipeline latency) — bearing ของเป้าต้องคิดจากท่าแขน
    'ณ เวลาเก็บภาพ' ไม่ใช่ท่าปัจจุบัน มิฉะนั้นการขยับแขนระหว่างรอผลจะถูกนับซ้ำ
    (convention เดียวกับ apan_delayed ใน 23_arm_chase_sim.py)."""

    def __init__(self, maxlen: int = 240) -> None:
        self._buf: deque = deque(maxlen=maxlen)  # ~4-8 วินาทีที่ 30-60 loop/s

    def append(self, t: float, pan: float, tilt: float) -> None:
        self._buf.append((t, pan, tilt))

    def pose_at(self, t: float) -> Optional[Tuple[float, float]]:
        """คืน (pan, tilt) interpolate เชิงเส้น ณ เวลา t; None ถ้ายังไม่มีข้อมูล."""
        buf = self._buf
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][1], buf[0][2]
        if t >= buf[-1][0]:
            return buf[-1][1], buf[-1][2]
        # scan จากท้าย: t อยู่ใกล้ปลาย buffer เสมอ (latency สั้นกว่าความยาว buffer มาก)
        for i in range(len(buf) - 1, 0, -1):
            t1, p1, q1 = buf[i]
            t0, p0, q0 = buf[i - 1]
            if t0 <= t <= t1:
                if (t1 - t0) < 1e-6:
                    return p1, q1
                a = (t - t0) / (t1 - t0)
                return p0 + a * (p1 - p0), q0 + a * (q1 - q0)
        return buf[-1][1], buf[-1][2]

    def clear(self) -> None:
        self._buf.clear()


# --- Kalman Filter (ปรับได้) ---
KALMAN_PROCESS_NOISE_Q = 0.1        # ค่าสูง = ตาม turn โดรนได้เร็ว (0.01=นิ่ง, 1.0=หักเลี้ยวบ่อย)
KALMAN_MEASUREMENT_NOISE_R = 25.0   # measurement noise baseline
KALMAN_COAST_MAX_SEC = 0.5          # coast ได้นานสูงสุดเมื่อ CSRT/YOLO พลาด (7-8 เฟรมที่ 15fps)
AUTO_KALMAN_COAST_MAX_SEC = 0.5     # coast AUTO mode เมื่อ YOLO พลาด
# bounded-coast: ยุบ velocity ตอน coast แบบ exp(-t/tau) → displacement = v·tau·(1-exp(-t/tau))
# แทน v·t เชิงเส้น กัน overshoot ตอน coast นานที่เป้าเลี้ยว (จูน+validate ใน sim: fix danger zone
# ตอน det-fps ต่ำ/miss โดยไม่เสีย sweet spot; form นี้ self-adaptive ต่อ fps). 0 = ปิด (linear เดิม)
KALMAN_COAST_DECAY_TAU = 0.30       # วินาที
LEAD_TIME_SEC = 0.20                # เผื่อล่วงหน้า (ชดเชย pipeline lag ~300-400ms บน 15fps RTSP)

# PD controller gains (ปรับหลังทดสอบแขนจริง)
PD_KP_PAN  = 1.0    # Proportional gain แกน pan
PD_KD_PAN  = 0.05   # Derivative gain แกน pan (เบรกก่อนถึงเป้า)
PD_KP_TILT = 1.0    # Proportional gain แกน tilt
PD_KD_TILT = 0.05   # Derivative gain แกน tilt

# Adaptive yellow vector smoothing
ADAPTIVE_ALPHA_SLOW = 0.94          # alpha เมื่อเป้านิ่ง
ADAPTIVE_ALPHA_FAST = 0.75          # alpha เมื่อเป้าเคลื่อนเร็ว (responsive มากขึ้น)
ADAPTIVE_ALPHA_SPEED_THRESHOLD_PX = 80.0  # px/s ที่ใช้แยก slow/fast

# LOCK mode deadzone — หยุดส่ง command เมื่อแขนเข้าใกล้เป้าพอแล้ว (ป้องกันวิ่งเลยและสั่น)
LOCK_DEADZONE_DEG = 0.5   # องศา — ถ้า error < ค่านี้ หยุดทันที (ปรับได้: 0.3=ถี่, 1.0=กว้าง)
# Pixel deadzone + step scale: ลดเด้งจาก CSRT jitter
LOCK_DEADZONE_PX = 8      # px — (เลิกใช้ใน _tick_lock แล้ว: bearing space ใช้ LOCK_DEADZONE_DEG แทน)
LOCK_STEP_SCALE_REF_PX = 50.0  # ระยะ px ที่ step_scale = 1.0; ใกล้กว่านี้ลด step (ลด overshoot)
# Phase  approach: ตอนเริ่มกด lock หมุนเร็วเข้า bbox (วินาที) — ในช่วงนี้ใช้ step_scale=1.0, throttle สั้น
LOCK_APPROACH_PHASE_SEC = 0.8
# ปิด tracking หลังหมุนไป center — แค่ครั้งเดียว ไม่ส่ง move ตาม bbox ต่อ (one-shot)
# ตั้ง False เพื่อให้ LOCK auto-track ต่อเนื่อง (จำเป็นต้อง sync IoU smooth center เข้า state ทุกเฟรม)
LOCK_DISABLE_TRACKING = False

# --- LOCK auto-track (ตัวขับ _tick_lock: แขนตามเป้าที่ล็อกต่อเนื่อง) ---
LOCK_AUTOTRACK_ENABLED = True     # ให้แขน auto-track เป้าที่ล็อก (ปิด = LOCK ไม่ขับแขน เหมือน stub เดิม)
LOCK_PAN_SIGN = 1.0               # ⚠️ ถ้าทดสอบแล้วแขน 'วิ่งหนี' เป้าแกน pan → สลับเป็น -1.0
LOCK_TILT_SIGN = 1.0              # ⚠️ ถ้าแกน tilt วิ่งหนี → -1.0
LOCK_KALMAN_COAST_MAX_SEC = 0.8   # coast ต่อได้นานสูงสุดตอน YOLO หลุด (ไม่ต้อง re-lock)
                                  # ↑ 0.5→0.8: motion ไม่ขับแขนแล้ว (LOCK_MOTION_DRIVES_ARM=False) → coast
                                  # จาก YOLO เป็นสะพานเดียวข้าม gap. โดรนจริง det~0.25@15fps → gap ยาวได้ ~0.5s+
                                  # ค่าเดิม 0.5 = แขน 'ค้างแล้วกระโดด' ตอน gap ยาว. bounded-coast (decay_tau)
                                  # ยัง cap ไม่ให้ไหลเลยตอนเป้าเลี้ยว จึงยืดได้ปลอดภัย
LOCK_TRACK_STEP_SCALE = 1.0       # gain (bearing space + lead ไม่ double-count แล้ว → ตั้ง 1.0 ได้)
LOCK_TRACK_PROFILE_REF_DEG = 1.0  # ปิดระยะเต็มที่จนเหลือ 1° (ตามเป้าเคลื่อนที่แน่น); ค่าน้อย=แน่น/เสี่ยงสั่น
# --- LOCK lead / predictive aiming (แก้อาการ 'วิ่งตาม' — เล็งดักหน้าเป้า) ---
# เป้าหมาย: เล็งไปที่ตำแหน่งที่โดรน 'จะไป' = ตำแหน่งปัจจุบัน + velocity × lead_time
# lead_time = age(เก่าตั้งแต่ deliver) + pipeline_latency(capture→deliver, measurement เก่าเท่านี้)
#            + LOCK_LEAD_EXTRA_SEC(เผื่อเวลาแขนเคลื่อน/สั่งงาน)
# ใช้ predict เชิงเส้น (v·t) สำหรับ lead ปกติ; cap ระยะกัน velocity noise เหวี่ยง;
# ตอนเป้าหลุดนาน → กลับไป bounded coast (decay) กัน overshoot ตอนเป้าเลี้ยว
LOCK_LEAD_ENABLED = True
LOCK_LEAD_EXTRA_SEC = 0.05        # เผื่อ actuation lag (จูนใน sim: 0.05 ดีสุดที่ ω≤10; สูงไป=overshoot)
LOCK_LEAD_MAX_DEG = 6.0           # cap ระยะ lead สูงสุด (deg) กัน velocity noise เหวี่ยงแขน
# Velocity deadband สำหรับ lead: เป้าช้ากว่า _LO (deg/s) → lead=0 (นิ่ง lead จริงจิ๊บจ๊อย ที่เห็นคือ noise);
# เร็วกว่า _HI → lead เต็ม (ตามทัน); ระหว่างนั้น ramp เชิงเส้น. แก้ tilt แกว่งบนโดรน near-hover (test 14:42)
LOCK_LEAD_VEL_FLOOR_LO = 1.5      # deg/s: ต่ำกว่านี้ = ถือว่านิ่ง → ไม่ lead (กัน accel/vel noise เหวี่ยง tilt)
LOCK_LEAD_VEL_FLOOR_HI = 4.0      # deg/s: สูงกว่านี้ = เคลื่อนจริง → lead เต็ม 100%
# --- Close-range lead attenuation (whip guard, test 16:29) ---
# อาการ: เป้า 'ใกล้กว่าจุดเล็งมากๆ' → ไม่แม่น + แขนเหวี่ยงแรง. สาเหตุ: ระยะใกล้ ω (เชิงมุม) พุ่งสูง
# 'และไม่คงที่' (dω/dt สูงตอนเฉียดใกล้) → lead แบบ v·t (สมมติ ω คงที่) ทำนายเกินหน้าเป้า → over-lead
# เหวี่ยงแขน (log: lead offset เฉลี่ย 2.9° สูงสุด 16° = 43% ของระยะแขนเหวี่ยงต่อเฟรม; โตตาม speed).
# ระยะใกล้ ToF กระสุนจิ๊บจ๊อย → firing-lead≈0 อยู่แล้ว จึงแทบไม่ต้องการ tracking-lead. หรี่ lead ตามระยะ:
# ไกล (ω ต่ำ+คงที่ ทำนายแม่น) = lead เต็ม; ใกล้ = เหลือ MIN_GAIN. ไม่กระทบการ 'ตาม' เป้าเร็วจริง (ตัด
# เฉพาะส่วน lead ที่ล้ำหน้า ไม่ตัดการวิ่งเข้าหา base). range หายไป (None) → gain=1 (ไม่เปลี่ยนพฤติกรรมเดิม).
LOCK_LEAD_RANGE_ATTEN_ENABLED = True
LOCK_LEAD_RANGE_NEAR_M = 20.0     # ≤ ระยะนี้ = หรี่ tracking lead เหลือ MIN_GAIN (กันเหวี่ยงตอนเป้าใกล้)
LOCK_LEAD_RANGE_FAR_M = 45.0      # ≥ ระยะนี้ = lead เต็ม (เป้าไกล ω ต่ำ+คงที่ → ทำนายแม่น)
LOCK_LEAD_RANGE_MIN_GAIN = 0.2    # สัดส่วน lead ที่ยังเหลือตอนใกล้สุด (ไม่ 0 เพื่อยัง compensate actuation lag บ้าง)
# --- Close-range slew guard (sim-verified: หลัง fix lead แล้ว whip ที่เหลือ = reacquire snap) ---
# หลังหรี่ lead แล้ว whip ที่เหลือตอนระยะใกล้ = reacquire หลัง detection gap → base(KF) snap ไปตำแหน่งใหม่
# → target กระโดด → แขนวิ่งชนเพดานไล่. จำกัด 'อัตราการเปลี่ยนคำสั่ง track' (deg/s) ให้แน่นขึ้นตอนใกล้:
# reacquire snap ถูกไถเข้าแบบ ramp หลาย tick แทนกระโดดทีเดียว. cap ยังกว้างพอให้ตามการเคลื่อนจริงที่
# ระยะนั้น (เป้าตัดหน้าใกล้สุด ~28°/s → NEAR cap 45°/s ยังตามทัน) แต่กัน reacquire teleport (100°/s+).
# ระยะไกล = cap หลวม (เป้าไกล ω ต่ำ ไม่กระทบ). ใช้เฉพาะ tracking drive — 'ไม่' แตะ snap-and-shoot
# (move_absolute) และ 'ไม่' แตะ fire readiness (error_deg ใช้จุดยิงจริงที่ไม่ถูก slew-limit → ยิงยังแม่น).
LOCK_SLEW_GUARD_ENABLED = True
LOCK_SLEW_GUARD_NEAR_M = 20.0        # ≤ ระยะนี้ = ใช้ NEAR cap (แน่นสุด)
LOCK_SLEW_GUARD_FAR_M = 45.0         # ≥ ระยะนี้ = ใช้ FAR cap (หลวม)
LOCK_SLEW_GUARD_NEAR_DEG_S = 45.0    # deg/s: เพดานอัตราคำสั่งตอนใกล้ (ตามเป้าตัดหน้าใกล้ได้ แต่กัน reacquire teleport)
LOCK_SLEW_GUARD_FAR_DEG_S = 200.0    # deg/s: เกือบไม่จำกัดตอนไกล (คง responsiveness เป้าไกล)
LOCK_LEAD_COAST_SWITCH_SEC = 0.32 # age เกินค่านี้ = ถือว่าเป้าหลุด → ใช้ bounded coast แทน linear lead
                                  # ↑ 0.2→0.32: ให้แขน 'เล็งดักหน้า (linear lead)' ต่อเนื่องตลอด gap ปกติ
                                  # ของ det~0.25@15fps (~0.27s) แทนที่จะสลับไป decay-coast (ที่ยุบ velocity →
                                  # ตามเป้าเคลื่อนไม่ทัน) เร็วเกินไป. เกิน 0.32s = gap ยาวผิดปกติ/เป้าอาจหักเลี้ยว
                                  # แล้ว → bounded coast ปลอดภัยกว่า. lead ยัง cap ที่ LOCK_LEAD_MAX_DEG กัน noise
LOCK_PIPELINE_LATENCY_FALLBACK = 0.15  # ใช้เมื่อยังไม่มีค่าวัด avg_pipeline_latency
# Ego-comp capture latency: bearing = pose(ณ เวลาเก็บภาพ) + pixel_offset/ppd. pose_hist บันทึกท่าแขน
# 'ตอน grab เฟรม' แต่ 'เนื้อภาพ' เก่ากว่านั้น = camera latency (sensor→network→decode). ตอนแขน slew
# เร็ว → pose ที่ใช้ 'ล้ำหน้า' เนื้อภาพไป slew_rate×latency → bearing เพี้ยนไปทางที่แขนหมุน → แขนไล่
# เลยตำแหน่งโดรน (อาการ 'โยกแขนยาวเกินเป้าตอนกระโดดเข้า'). แก้: ใช้ pose ที่ (t_capture − latency นี้).
# จูน: overshoot ตอน slew → เพิ่ม; แขนตามช้า/ลากหลัง → ลด. 0 = ปิด (พฤติกรรมเดิม).
LOCK_EGO_COMP_LATENCY_SEC = 0.06
# --- Firing solution (intercept): เล็งดักหน้าเผื่อเวลากระสุนบิน → กดยิงแล้วโดนโดรนที่กำลังบิน ---
# lead_ยิง = velocity_เป้า × t_flight ; t_flight = ระยะ / muzzle_velocity (ประเมินระยะจากขนาด bbox)
# + ชดเชย gravity drop (เล็กที่ระยะ/ความเร็วนี้ แต่ใส่ให้ครบเพื่องานวิจัย)
# เมื่อเปิด: แขนชี้จุด intercept ต่อเนื่อง (โดรนจะเยื้องจากศูนย์เท่า lead) กดยิงเมื่อไรก็โดน
LOCK_FIRE_SOLUTION_ENABLED = True
LOCK_FIRE_DROP_COMP = True         # ชดเชย gravity drop บนแกน tilt
LOCK_FIRE_TILT_UP_SIGN = -1.0      # ทิศ 'เงยขึ้น' ของแกน tilt (ppd_y<0 → เงยขึ้น = tilt ลด) ปรับถ้าตรงข้าม
LOCK_FIRE_MAX_LEAD_DEG = 8.0       # cap lead ยิง กัน velocity noise
GRAVITY_MS2 = 9.80665
# --- Fire readiness (human-in-the-loop): ขึ้น "SHOOT" เมื่อคำนวณแล้วว่ายิงโดนแน่ ---
# โดนแน่ = predicted miss (ลำกล้อง→จุด intercept) ≤ effective hit radius ต่อเนื่องหลายเฟรม
LOCK_FIRE_READY_FRAMES = 2         # ต้องเข้าเกณฑ์ต่อเนื่องกี่เฟรมถึงขึ้น SHOOT (ต่ำ=ยิงเร็ว)
# อายุ measurement สูงสุด (วินาที) ที่ยัง 'พร้อมยิง' ได้ — แยกจาก LOCK_LEAD_COAST_SWITCH_SEC
# (ที่ใช้เรื่อง 'ขับแขนตามให้ทัน') โดยเจตนา: การยืดหน้าต่าง lead ให้แขนตามเป้าทันบนโดรนจริง ต้อง
# 'ไม่' ทำให้เกณฑ์ยิงหลวมตามไปด้วย — ยิงยังต้องอยู่บน measurement สด (≤0.2s) เท่าเดิม (ล็อกไว้)
LOCK_FIRE_STABLE_MAX_AGE_SEC = 0.2
# Effective hit radius (เมตร ที่ระยะเป้า) = ขนาดโดรน + การกระจายกระสุน/burst — ยิงได้จริงในทางปฏิบัติ
# แปลงเป็นองศาตามระยะ: hit_radius_deg = atan2(HIT_RADIUS_M, range)
LOCK_FIRE_HIT_RADIUS_M = 0.35      # รัศมีปะทะ (m) — 0.35m ≈ เข้าตัวโดรน 0.3m จริง (เดิม 1.2m หลวมเกิน
                                   #   = ทรงกลมกว้างกว่าตัวโดรน 4×; slug ต้องเข้าตัว ไม่ใช่เฉียด 1.2m)
LOCK_FIRE_READY_MIN_RADIUS_DEG = 0.15  # ขั้นต่ำ (เป้าใกล้มากไม่ให้ผ่อนเกิน)
# Fire-safe zone (ความปลอดภัย): ยิงได้เฉพาะเมื่อ 'ทิศเล็ง' อยู่ในโคนกลาง — กันยิงทิศอันตราย
# ⚠️ ปิดตามคำสั่งผู้ใช้ (ทดสอบยิงโดรนจริง) — ยิงได้ทุกมุม, ลิมิตแขน ±63°/±33° ยังคุมมุมเล็งอยู่
LOCK_FIRE_SAFE_ENABLED = False
LOCK_FIRE_SAFE_PAN_DEG = 45.0      # |target_pan| ต้อง ≤ ค่านี้ถึงจะยิงได้ (ใช้เมื่อ ENABLED=True)
LOCK_FIRE_SAFE_TILT_DEG = 25.0     # |target_tilt| ต้อง ≤ ค่านี้ (ใช้เมื่อ ENABLED=True)
LOCK_FIRE_PRED_UNCERT_K = 1.5      # ตัวคูณความไม่แน่นอน = prediction_residual_rate·horizon·K
                                   # จูนใน sim: 1.5 = ยิงถี่ที่ ω≤8 (hit 88-100%), ปฏิเสธที่ ω≥15
LOCK_FIRE_NOISE_FLOOR_DEG = 0.6    # พื้น uncertainty ของจุด intercept จาก 'sensor noise × latency' (deg).
                                   # เดิม 0.05 = ค่า sim-quality (detection เป๊ะ) → บนโดรนจริง uncert ประเมิน
                                   # ต่ำเกิน → 'มั่นใจ' ผิดที่ระยะไกล → ยิงแล้วพลาด (bbox jitter ~0.6° proj ผ่าน
                                   # pipe_lat 0.15s = intercept error ~0.6-1.8°). ตั้ง 0.6 → uncert สะท้อน noise
                                   # จริง → gate ยอมยิงเฉพาะตอน hit_r ใหญ่พอ (ระยะใกล้) ที่โดนได้จริง; ระยะไกล
                                   # (hit_r เล็ก) uncert>hit_r×K → ไม่ยิง (แก้ 'ยิงไม่แม่น' = งดยิงนัดที่พลาดแน่).
                                   # ground-truth sim: 0.6 → ยิง+โดน ≤10m (60-100%), งดยิง ≥12m (ที่ miss แน่).
# --- Predictive-intercept firing (ดักยิงล่วงหน้า) ---
# doctrine: กดยิง = 'ตั้งใจยิง' (authorize) ไม่ยิงทันที → lock ตามเป้าต่อ + คำนวณจุด intercept realtime
#   จนกว่าจะ 'มั่นใจว่าทำนายแม่นพอจะโดน' (confidence gate ด้านล่าง) → กระโดดไปจุดนั้น (snap) แล้วยิง = แม่น
#   เป้า juke/คาดเดายาก → ไม่มั่นใจ → ตามต่อ ไม่ยิง (กันยิงพลาด). impact_log/HUD รายงาน miss จริงให้จูน
LOCK_FIRE_MAX_RESID_DEG_S = 3.0    # เป้า juke เกินนี้ (deg/s prediction residual) = คาดเดายาก → ไม่ยิง รอจังหวะนิ่ง
# resid = ความเบี่ยงจากทำนาย const-velocity ต่อเฟรม. ปัญหา: ระยะใกล้ เป้า subtend มุมกว้าง → bbox
# center wander หลายองศา/เฟรม → resid พุ่งสูงจาก 'sensor noise' ไม่ใช่ 'เป้าหลบจริง' → confidence gate
# ไม่เปิดเลย = 'ไล่ตามไม่ยิงซักที' (โดรนจริงเคลื่อนตลอด resid ไม่เคยลงต่ำ 3°/s). แก้: หัก noise floor (องศา)
# ออกจาก resid ก่อนคิด rate → เหลือเฉพาะส่วน 'เกินคาด' = การหลบจริง. real juke (เบี่ยงหลายองศา/เฟรม)
# ยังทะลุ floor → ถูกจับได้อยู่; แต่ jitter ระดับ noise ถูกกรองออก → ยิงเป้าที่บินเรียบ(แต่เร็ว)ได้.
LOCK_FIRE_RESID_NOISE_FLOOR_DEG = 0.4  # bbox-center jitter ที่คาดได้/เฟรม — หักก่อนคิด resid rate (0=ปิด, พฤติกรรมเดิม)
# Snap-and-shoot: เมื่อ authorized + มั่นใจ → 'กระโดดตรงไป intercept → ยิง' (ไม่รอ throttled approach)
# ใช้เมื่อ barrel อยู่ห่างจุดดักไม่เกิน SNAP_MAX (จุด intercept อยู่ใกล้ตำแหน่งที่ track อยู่แล้ว)
LOCK_FIRE_SNAP_AND_SHOOT = True    # authorized + มั่นใจ + ใกล้พอ → move_absolute กระโดดไป intercept แล้วยิง
LOCK_FIRE_SNAP_MAX_DEG = 5.0       # barrel ห่างจุด intercept ≤ ค่านี้ = commit กระโดด (ไกลกว่านี้ = ค่อย ๆ เข้าก่อน)
# Confidence gate: กดยิง = 'ตั้งใจยิง' (authorize) ไม่ยิงทันที — track ต่อจนกว่าจะ 'มั่นใจว่าทำนายแม่นพอจะโดน'
#   มั่นใจ = ความไม่แน่นอนของจุด intercept ที่ทำนาย (uncert จาก resid/accel) ≤ hit_radius × K ต่อเนื่องหลายเฟรม
#   เป้า juke/คาดเดายาก → uncert สูง → ไม่มั่นใจ → lock ตามต่อ ไม่ยิง (รอจังหวะที่ทำนายได้แม่น)
LOCK_FIRE_CONFIDENCE_K = 0.7       # ยอมให้ uncert ของจุดทำนาย ≤ hit_radius × K ถึงจะยิง — <1 = predicted miss
                                   #   ต้องอยู่ 'ในรัศมีโดนอย่างมีมาร์จิน' (เดิม 1.5 = ยอมพลาดตั้งแต่ต้น → miss บ่อย)
LOCK_FIRE_CONFIDENT_FRAMES = 4     # ต้องมั่นใจต่อเนื่องกี่เฟรม (มากขึ้น = ต้องนิ่งจริง ไม่ใช่ noise แวบเดียว)
LOCK_FIRE_CURVATURE_K = 0.5        # โทษเป้าที่กำลังเลี้ยว: uncert += K·accel·horizon² (เดิม 0.15 = ประเมินต่ำ)
_lock_fire_ready_count = [0]       # นับเฟรมที่เข้าเกณฑ์ยิงโดนต่อเนื่อง
_lock_confident_count = [0]        # นับเฟรมที่ 'มั่นใจว่าทำนายแม่นพอจะโดน' ต่อเนื่อง
_lock_vel_hist = [0.0, 0.0, 0.0]   # [vpan, vtilt, t] รอบก่อน — ประเมินความเร่งเป้า
_lock_accel_ema = [0.0, 0.0]       # ความเร่งเป้า (deg/s²) smoothed สำหรับ constant-accel lead
_lock_meas_prev = [None]           # [b_pan,b_tilt,t,vpan,vtilt] measurement ก่อน — วัด prediction residual
_lock_pred_resid_rate = [0.0]      # deg/s: อัตราความคลาดเคลื่อนการทำนาย (สูง=เป้าหลบ คาดเดายาก)
# --- Outlier gate ก่อนป้อน lock_kalman (กันแขนสบัดออกจากเป้า) ---
# โดรนจริงเคลื่อนเรียบ: ที่ 15°/s ข้าม 1 เฟรม (67ms) ขยับ ~1°. measurement ที่กระโดด >MAX_STEP จาก
# ที่ kalman ทำนาย = การ associate ผิด (คว้า FP/object อื่น) → ถ้าป้อนดิบ ๆ แขนจะสบัดไปตาม. ทิ้ง
# outlier เฟรมเดียว (แขน coast ต่อ ไม่สบัด); แต่ถ้าเป้า 'อยู่ที่ใหม่จริง' (re-lock/หักเลี้ยวแรง)
# measurement จะเกาะกลุ่มที่ใหม่ต่อเนื่อง → พอครบ CONFIRM เฟรม ยอมรับ (snap ไป lock ใหม่)
LOCK_MEAS_OUTLIER_MAX_STEP_DEG = 8.0  # measurement ห่างจากที่ทำนาย > นี้ = outlier (โดรนจริงไม่กระโดดขนาดนี้/เฟรม)
LOCK_MEAS_OUTLIER_CONFIRM = 3         # outlier ที่เกาะกลุ่มกันครบกี่เฟรม = ของจริง (re-lock) → ยอมรับ
LOCK_MEAS_OUTLIER_CLUSTER_DEG = 4.0   # outlier 2 เฟรมถือว่า 'ที่เดียวกัน' ถ้าห่าง ≤ นี้
_lock_outlier_run = [0, 0.0, 0.0]     # [count, b_pan, b_tilt] ของ outlier ที่กำลังสะสม
# per-frame diag (เปิดด้วย env LOCK_FRAME_LOG=1): วิเคราะห์ intra-second ว่า pan/tilt jump มาจากอะไร
_lock_frame_log = [None]              # file handle (เปิดใน main)
_lock_meas_dbg = [0.0, 0.0, "none"]   # [b_pan, b_tilt, action] จาก measurement ล่าสุด (feed → tick อ่านไปเขียน log)
_lock_slew_prev = [None, None, 0.0]   # [cmd_pan, cmd_tilt, t] คำสั่ง track รอบก่อน — close-range slew guard
# --- Constant-acceleration lead: ทำนายโค้ง/เลี้ยว (pos + v·t + ½·a·t²) ให้ tracking แม่นขึ้น ---
LOCK_LEAD_ACCEL_ENABLED = True     # ใช้พจน์ความเร่งในการทำนาย (เป้าโค้ง/เลี้ยวแม่นขึ้น)
LOCK_LEAD_ACCEL_EMA_ALPHA = 0.35   # smoothing ความเร่ง (สูง=ตอบไว/สั่น, ต่ำ=นิ่ง/ช้า)
LOCK_LEAD_ACCEL_MAX_DEG = 3.0      # cap พจน์ ½·a·t² กัน accel noise เหวี่ยง (deg)
# --- LOCK bearing-space tracking (ego-motion compensation, แนวทางเดียวกับ 23_arm_chase_sim) ---
# lock_kalman ติดตาม "absolute bearing" (มุม pan/tilt ของเป้า) แทน pixel:
#   bearing = arm_pose(เวลาเก็บภาพ) + pixel_offset/ppd
# → การขยับแขนไม่ปนเข้า velocity ของเป้า (แก้ปัญหา cam4 ติดบนแขนโดยตรง)
#
# Q กับ R มีหน่วยต่างกันโดยธรรมชาติ — ต้องแยกกันคิด (เดิมหารด้วย 87² ทั้งคู่ = ผิดครึ่งหนึ่ง):
#   Q = process noise = "โดรนหักเลี้ยวแรงแค่ไหน" หน่วย deg²  → เป็น 'ฟิสิกส์ของเป้า'
#       ไม่เกี่ยวกับกล้อง → ต้องคงที่ ไม่ scale ตาม ppd
#   R = measurement noise = "bbox center สั่นกี่พิกเซล" หน่วย px² → เป็น 'คุณสมบัติของกล้อง'
#       → แปลงเป็น deg² ด้วย (σ_px/ppd)² ตาม ppd ของกล้องที่ใช้จริง (คำนวณตอน runtime)
# ค่าที่ cam4 (ppd 87.138): Q = 0.1/87² = 1.32e-5, R = (5/87.138)² = 3.29e-3 → เท่าของเดิมเป๊ะ
LOCK_BEARING_Q_DEG2 = 1.32e-5      # deg² — ความคล่องตัวของโดรน (camera-independent)
LOCK_MEAS_SIGMA_PX = 5.0           # 1σ ของ bbox-center jitter (พิกเซล) — ต้องวัดใหม่ต่อกล้อง
PPD_X_FALLBACK = 87.138            # ppd cam4 — ใช้เมื่อยังไม่ได้โหลดคาลิเบรต (คงพฤติกรรมเดิม)


def lock_bearing_kalman_qr(ppd_x, sigma_px: float = LOCK_MEAS_SIGMA_PX):
    """
    คืน (q, r) หน่วย deg² สำหรับ lock_kalman ตาม ppd ของกล้องที่ใช้จริง.
    Q คงที่ (ฟิสิกส์ของเป้า); R = (σ_px / ppd)² (jitter ของกล้องแปลงเป็นองศา).
    """
    ppd = abs(float(ppd_x)) if ppd_x else 0.0
    if ppd < 1e-6:
        ppd = PPD_X_FALLBACK
    return LOCK_BEARING_Q_DEG2, (float(sigma_px) / ppd) ** 2
LOCK_TRACK_DEBUG = True            # print telemetry _tick_lock (throttled) เพื่อวินิจฉัยอาการจริง (ปิดได้)
_lock_dbg_last_t = [0.0]          # throttle timer สำหรับ debug print
_lock_dbg2_last_t = [0.0]         # throttle timer สำหรับ guard-state debug


class _TargetKalman:
    """
    Kalman Filter 4-state (x, y, vx, vy) สำหรับ smooth + predict ตำแหน่งเป้า
    - update(cx, cy, conf): รับ YOLO confidence ปรับ measurement noise อัตโนมัติ
      conf สูง (ภาพชัด) → เชื่อ measurement; conf ต่ำ (เบลอ) → เชื่อ predict
    - predict_ahead(dt): ทำนายตำแหน่งล่วงหน้า dt วินาที
    - get_velocity_deg_s(ppd_x, ppd_y): ดึง velocity (deg/s) สำหรับ feedforward
    - get_speed_px_s(): ความเร็ว px/s สำหรับ adaptive alpha
    """

    def __init__(self, q: Optional[float] = None, r: Optional[float] = None) -> None:
        # q/r override ต่อ instance: หน่วยของ noise ต้องตรงกับหน่วยของ measurement
        # (default = pixel space; LOCK bearing-space ใช้หน่วยองศา → ส่งค่า scale แล้วเข้ามา)
        self._q = KALMAN_PROCESS_NOISE_Q if q is None else float(q)
        self._r = KALMAN_MEASUREMENT_NOISE_R if r is None else float(r)
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / 15.0  # สมมติ 15fps เป็น default dt
        # Transition matrix F: x' = x + vx*dt, y' = y + vy*dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        # Measurement matrix H: observe x, y only
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise Q
        q = self._q
        self.kf.processNoiseCov = q * np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov[2, 2] *= 10.0  # velocity noise สูงกว่า position
        self.kf.processNoiseCov[3, 3] *= 10.0
        # Measurement noise R (baseline)
        self.kf.measurementNoiseCov = self._r * np.eye(2, dtype=np.float32)
        # Error covariance init
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False
        self._last_update_time: float = 0.0

    def reset(self) -> None:
        self._initialized = False
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def set_noise(self, q: Optional[float] = None, r: Optional[float] = None) -> None:
        """ตั้ง Q/R ใหม่ตอน runtime — ใช้เมื่อคาลิเบรต ppd ใหม่ (R ผูกกับ ppd)."""
        if q is not None:
            self._q = float(q)
            self.kf.processNoiseCov = self._q * np.eye(4, dtype=np.float32)
            self.kf.processNoiseCov[2, 2] *= 10.0
            self.kf.processNoiseCov[3, 3] *= 10.0
        if r is not None:
            self._r = float(r)
            self.kf.measurementNoiseCov = self._r * np.eye(2, dtype=np.float32)
        self.reset()  # bearing เก่าคิดจาก ppd เดิม → ใช้ต่อไม่ได้

    def _update_dt(self) -> None:
        """อัปเดต dt ใน transition matrix จากเวลาจริง."""
        now = time.time()
        if self._last_update_time > 0:
            dt = max(0.005, min(now - self._last_update_time, 0.5))
        else:
            dt = 1.0 / 15.0
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        self._last_update_time = now

    def update(self, cx: float, cy: float, conf: float = 1.0) -> Tuple[float, float]:
        """
        อัปเดต Kalman ด้วย measurement (cx, cy) และ confidence.
        conf ต่ำ → R noise สูง → filter เชื่อ prediction มากกว่า measurement.
        คืน (smoothed_cx, smoothed_cy).
        """
        self._update_dt()
        conf_c = max(0.05, min(1.0, conf))
        R_scale = self._r / (conf_c ** 2)
        self.kf.measurementNoiseCov = R_scale * np.eye(2, dtype=np.float32)
        meas = np.array([[cx], [cy]], dtype=np.float32)
        if not self._initialized:
            self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self._initialized = True
        self.kf.predict()
        corrected = self.kf.correct(meas)
        return float(corrected[0]), float(corrected[1])

    def reseed_position(self, cx: float, cy: float) -> None:
        """ย้าย state position ไป (cx,cy) ทันที โดย 'คงความเร็ว (vx,vy) เดิมไว้' — ใช้ตอน outlier gate
        ยืนยัน re-lock: เป้าอยู่ที่ใหม่จริง แต่ 'ยังเคลื่อนที่' → ถ้า zero velocity (reset) แขนจะเลิก lead
        → ตามเป้าที่บินอยู่ไม่ทัน → re-outlier → re-seed วน = แขน hunting (เจอบนแกน pan ตอนเป้าบินขวาง).
        inflate ความไม่แน่นอนของ position นิดหน่อยให้ measurement ถัดไปดึงได้เร็ว."""
        if not self._initialized:
            self.update(cx, cy, 1.0)
            return
        self.kf.statePost[0, 0] = float(cx)
        self.kf.statePost[1, 0] = float(cy)
        # คง statePost[2],[3] (velocity) ไว้ — ไม่แตะ
        self.kf.errorCovPost[0, 0] = max(float(self.kf.errorCovPost[0, 0]), 1.0)
        self.kf.errorCovPost[1, 1] = max(float(self.kf.errorCovPost[1, 1]), 1.0)
        self._last_update_time = time.time()

    def predict_ahead(self, dt_sec: float, decay_tau: Optional[float] = None) -> Tuple[float, float]:
        """ทำนายตำแหน่งล่วงหน้า dt_sec วินาทีโดยไม่แก้ไข state.
        decay_tau ไม่ None/0 → bounded coast: สมมติ velocity ยุบ exp(-t/tau) ระหว่าง coast
        → displacement = v·tau·(1-exp(-dt/tau)) (อิ่มตัวที่ v·tau) แทน v·dt เชิงเส้น
        ป้องกัน overshoot ตอน coast นาน (fps ต่ำ/miss) ที่เป้ากำลังเลี้ยว. dt เล็ก → ≈ v·dt เดิม."""
        if not self._initialized:
            return 0.0, 0.0
        state = self.kf.statePost
        if decay_tau is not None and decay_tau > 1e-6:
            eff_dt = decay_tau * (1.0 - math.exp(-dt_sec / decay_tau))
        else:
            eff_dt = dt_sec
        px = float(state[0]) + float(state[2]) * eff_dt
        py = float(state[1]) + float(state[3]) * eff_dt
        return px, py

    def translate(self, dx: float, dy: float) -> None:
        """เลื่อนตำแหน่ง (x,y) ของ state ตามการเคลื่อนของกล้อง โดยคงความเร็ว (vx,vy) ไว้.
        ใช้ทำ ego-motion compensation: กล้องติดบนแขน พอแขนหมุน โลกเลื่อนในเฟรม."""
        if self._initialized:
            self.kf.statePost[0, 0] += float(dx)
            self.kf.statePost[1, 0] += float(dy)

    def get_velocity_raw(self) -> Tuple[float, float]:
        """คืน velocity ดิบของ state (vx, vy) — หน่วยตรงกับ measurement.
        LOCK ใช้ bearing space → คืน (deg/s, deg/s) ตรง ๆ สำหรับ firing lead."""
        if not self._initialized:
            return 0.0, 0.0
        s = self.kf.statePost
        return float(s[2]), float(s[3])

    def get_velocity_deg_s(self, px_per_deg_x: float, px_per_deg_y: float) -> Tuple[float, float]:
        """คืน velocity (deg/s) สำหรับ feedforward command."""
        if not self._initialized or abs(px_per_deg_x) < 1e-6 or abs(px_per_deg_y) < 1e-6:
            return 0.0, 0.0
        state = self.kf.statePost
        vx_px_s = float(state[2])
        vy_px_s = float(state[3])
        return vx_px_s / px_per_deg_x, vy_px_s / px_per_deg_y

    def get_speed_px_s(self) -> float:
        """คืนความเร็ว (px/s) สำหรับ adaptive alpha."""
        if not self._initialized:
            return 0.0
        state = self.kf.statePost
        return float(math.hypot(float(state[2]), float(state[3])))


class _ArmPDController:
    """
    PD Controller สำหรับขับแขน: output = Kp*error + Kd*d_error/dt
    เบรกก่อนถึงเป้า ลด overshoot เมื่อเป้านิ่ง
    """

    def __init__(
        self,
        kp_pan: float = PD_KP_PAN,
        kd_pan: float = PD_KD_PAN,
        kp_tilt: float = PD_KP_TILT,
        kd_tilt: float = PD_KD_TILT,
    ) -> None:
        self.kp_pan = kp_pan
        self.kd_pan = kd_pan
        self.kp_tilt = kp_tilt
        self.kd_tilt = kd_tilt
        self._prev_error_pan = 0.0
        self._prev_error_tilt = 0.0
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        self._prev_error_pan = 0.0
        self._prev_error_tilt = 0.0
        self._prev_time = None

    def compute(self, error_pan: float, error_tilt: float) -> Tuple[float, float]:
        now = time.time()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.033
        dt = max(dt, 1e-4)
        d_pan  = (error_pan  - self._prev_error_pan)  / dt
        d_tilt = (error_tilt - self._prev_error_tilt) / dt
        cmd_pan  = self.kp_pan  * error_pan  + self.kd_pan  * d_pan
        cmd_tilt = self.kp_tilt * error_tilt + self.kd_tilt * d_tilt
        self._prev_error_pan  = error_pan
        self._prev_error_tilt = error_tilt
        self._prev_time = now
        return cmd_pan, cmd_tilt


def _iou_bbox(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
    """คำนวณ IoU ระหว่าง 2 bbox (x,y,w,h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# =============================================================================
# Tracking
# =============================================================================
# --- IoU tracker (IOT) ---
IOT_IOU_THRESHOLD = 0.15      # IoU ต่ำสุดถือว่า match
IOT_CENTER_DIST_MAX_RATIO = 0.4  # fallback: ระยะ center ไม่เกิน max_dim × ratio
# Filter ก่อนเลือก: กล่องต้องอยู่ในบริเวณ (ระยะจาก pred) และขนาดใกล้เคียงกับที่ lock
IOT_REGION_RADIUS_RATIO = 0.25   # ระยะ center ไม่เกิน max_dim × ratio
IOT_SIZE_RATIO_MIN = 0.4        # อัตราส่วน w,h ต่อ track_bbox อย่างน้อย (ผ่อนให้รับขนาดเปลี่ยนตอนระยะเปลี่ยน)
IOT_SIZE_RATIO_MAX = 2.6
IOT_MISS_FRAMES_TO_LOST = 10    # hysteresis: พลาดกี่ครั้งติดกัน (ผล YOLO ที่ไม่ match) ถึงประกาศ lost
                                # ↑ จาก 3 → 10 = coast นานขึ้น (~0.4-0.7s) กัน LOCK หลุดตอน YOLO เจอ ๆ หาย ๆ
IOT_GATE_MISS_GROWTH = 0.6      # ทุกเฟรมที่พลาด ขยาย association gate += ค่านี้ × bbox_diag (re-catch โดรนที่โผล่กลับ)
# Slew-aware: ตอนแขนหมุน (> IOT_SLEW_BLIND_DEG/เฟรม) YOLO บอดเพราะภาพเบลอ = เรื่องปกติ
# → นับ miss แค่ IOT_SLEW_MISS_WEIGHT (ไม่ประกาศ lost ระหว่าง slew, coast เล็งตาม bearing ต่อ)
# หน่วยองศา ไม่ใช่พิกเซล: "ภาพเบลอแค่ไหน" ขึ้นกับมุมที่กล้องกวาด ไม่ใช่จำนวนพิกเซล
# (เดิม 22px @ ppd 87 ซึ่งคอมเมนต์เองก็เขียนว่า ≈0.25° อยู่แล้ว)
IOT_SLEW_BLIND_DEG = 0.25       # แขนหมุนเกินนี้/เฟรม = กำลัง slew → ถือว่า YOLO บอดคาดเดาได้
IOT_SLEW_MISS_WEIGHT = 0.2      # miss ตอน slew นับ 0.2 (settled=1.0) → lost ช้าลง ~5 เท่าตอนแขนหมุน

# --- YOLO re-acquire (แก้อาการ motion-assist ยึด track ผิดจน YOLO เจอโดรนแต่ match ไม่ได้ตลอด) ---
# บนโดรนจริง YOLO เจอแค่ ~20-50% ของเฟรม → motion-assist ทำงานเกือบทุกเฟรม และไป 'ลบ' miss_accum
# (apply_motion_measurement) ทำให้ miss_count ค้างที่ ~0 → association gate ไม่เคยโต → detection จริง
# ที่ตอนนี้เยื้องจากตำแหน่ง motion ที่ไหลไป ตกนอก gate ตลอด = re-match ไม่ได้ (พบใน log: det=1.0 match=0.0
# ต่อเนื่อง 8 วิ). แก้: นับ _yolo_miss_run แยก — เฉพาะเฟรมที่ YOLO ไม่ match — motion-assist 'แตะไม่ได้'
# → ใช้ขยาย gate ให้ YOLO ดึง track กลับ; ถ้า run ยาว (track เพี้ยนแน่) ให้เชื่อ YOLO เลือกตัวมั่นใจสุด
IOT_REACQUIRE_FRAMES = 5        # YOLO เห็นโดรนแต่ match ไม่ได้ติดกันกี่เฟรม = track เพี้ยน → เข้าโหมด re-acquire
IOT_REACQUIRE_GATE_CAP = 12     # cap การขยาย gate จาก _yolo_miss_run (กัน gate โตไม่จำกัดไปคว้า FP)
IOT_REACQUIRE_MIN_CONF = 0.25   # ตอน re-acquire (gate โต) ยกพื้น conf ขั้นต่ำของ candidate = ค่านี้ กันคว้า FP/noise ในหน้าต่างกว้าง (เลือกยัง nearest)

# --- Motion-assist tracker (LOCK ห้ามหลุด): ทำงานทุกเฟรมโดยไม่พึ่ง YOLO ---
# โดรนขยับตลอด → หักล้างการหมุนกล้อง (จาก encoder แขน) แล้ว frame-diff, blob ที่ 'ขยับ' = โดรน
# เหมาะกับโดรนเล็กบนฟ้าเรียบ (ไม่ใช้ texture/YOLO) + ทน blur (motion คือสัญญาณ)
LOCK_MOTION_ASSIST_ENABLED = True
# สำคัญ (โดรนจริง): motion-assist ให้ 'ประคอง lock ไม่ให้หลุด' เท่านั้น — ไม่เอาไปขับแขน.
# บนภาพจริง ego-motion comp ไม่เพอร์เฟกต์ (encoder lag/สั่น/ppd คลาด) → frame-diff เหลือ residual
# ของฉากหลัง → motion-assist เกาะ residual แล้วป้อน bearing ผิดเข้า lock_kalman → แขนกระตุก/เล็งผิด
# (อาการที่เจอจริง: "แขนเล็งผิดที่/สั่น"). ใน sim ไม่เจอเพราะ ego-comp เป๊ะ.
# False = ระหว่าง YOLO บอด แขน coast นิ่ง ๆ ตาม velocity YOLO ล่าสุด (นุ่มกว่า, ไม่ไล่ noise);
#         motion-assist ยังกันประกาศ lost + ประคอง track_bbox/gate ไว้ให้ YOLO ดึงกลับ.
# True  = พฤติกรรมเดิม (motion ขับแขนด้วย) — เหมาะเฉพาะตอน ego-comp เชื่อถือได้ (เช่น sim).
LOCK_MOTION_DRIVES_ARM = False
# ประคอง track ตอน RGB หลุด: motion-assist 'ขับแขนได้เฉพาะตอน YOLO บอดจริง' (gap) — ไม่ใช่ตอน track ปกติ
# เหตุผล: ตอน YOLO ยัง match อยู่ ห้าม motion แทรก (กันไล่ residual = จิตเตอร์ ที่เป็นปัญหาเดิม); แต่ตอน
# YOLO หายยาว (det=0 หลายวินาที — เจอ 48% ของเวลาบนโดรนจริง) แขนไม่ควร 'ค้าง' ให้เป้าลอยหนี แล้ว
# สบัดไล่ทีหลัง → ให้ motion บริดจ์แขนตามเป้าไประหว่าง gap. ป้องกัน motion เกาะ background ด้วย
# outlier gate (ทิ้ง measurement ที่กระโดด) + velocity deadband (ไม่ขยาย noise) ที่มีอยู่แล้ว.
LOCK_MOTION_BRIDGE_ENABLED = True
LOCK_MOTION_BRIDGE_MIN_MISS = 3   # YOLO ไม่ match ติดกันกี่เฟรม = gap จริง → เริ่มให้ motion บริดจ์แขน
# รัศมี search window รอบตำแหน่งที่ทำนาย — เชิงมุม (เป้าหลุดไปได้กี่องศาใน 1 เฟรม ไม่ใช่กี่พิกเซล)
LOCK_MOTION_SEARCH_DEG = 1.262  # = 110px @ cam4 (ppd 87.1)
LOCK_MOTION_DIFF_THRESH = 16    # threshold ความต่างความสว่าง (ยิ่งต่ำยิ่งไว แต่ noise เยอะ)
LOCK_MOTION_MIN_AREA = 2        # พื้นที่ blob ขั้นต่ำ (px²) กัน noise จุดเดียว
LOCK_MOTION_MAX_AREA_FRAC = 0.25  # blob ใหญ่เกิน (เมฆ/เงา) ตัดทิ้ง
LOCK_MOTION_CONF = 0.30         # ความเชื่อของ measurement จาก motion (ต่ำกว่า YOLO — YOLO มา override ได้)
# gate กันเกาะ 'ฟีเจอร์นิ่ง': โดรนที่ขยับ = blob กะทัดรัด; ขอบวัตถุนิ่งที่ ego-comp เหลือ residual
# = เส้นบาง/สลิเวอร์ → ตัดด้วย extent (พื้นที่/กรอบ) ต่ำ หรือ aspect (ยาว/กว้าง) สูง
LOCK_MOTION_MIN_EXTENT = 0.32   # a/(w·h) ต่ำกว่านี้ = โปร่ง/เป็นเส้น → ตัด
LOCK_MOTION_MAX_ASPECT = 4.0    # ด้านยาว/ด้านสั้น สูงกว่านี้ = เส้นขอบ → ตัด


def _lock_motion_track(gray_cur, gray_prev, pred_cx, pred_cy, sx, sy,
                       search_r=None, diff_thresh=LOCK_MOTION_DIFF_THRESH,
                       min_area=LOCK_MOTION_MIN_AREA, max_area_frac=LOCK_MOTION_MAX_AREA_FRAC,
                       ppd=None):
    """หาโดรนจาก 'การขยับ' หลังหักล้างการหมุนกล้อง (ego-motion).
    gray_cur/gray_prev = เฟรมเทา ณ ตอนนี้/ก่อนหน้า; (sx,sy) = กล้องเลื่อนกี่ px (ชดเชยด้วย np.roll).
    คืน (cx,cy) ตำแหน่งโดรนที่ 'ขยับ' ใกล้ prediction สุด หรือ None."""
    if gray_cur is None or gray_prev is None or gray_cur.shape != gray_prev.shape:
        return None
    if search_r is None:
        search_r = LOCK_MOTION_SEARCH_DEG * (abs(float(ppd)) if ppd else PPD_X_FALLBACK)
    h, w = gray_cur.shape
    prev_shift = np.roll(gray_prev, (int(round(sy)), int(round(sx))), axis=(0, 1))   # ego-comp (เร็ว)
    x0 = max(0, int(pred_cx - search_r)); x1 = min(w, int(pred_cx + search_r))
    y0 = max(0, int(pred_cy - search_r)); y1 = min(h, int(pred_cy + search_r))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    diff = cv2.absdiff(gray_cur[y0:y1, x0:x1], prev_shift[y0:y1, x0:x1])
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max_area_frac * (x1 - x0) * (y1 - y0)
    best, bestd = None, 1e18
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        # gate กันเกาะฟีเจอร์นิ่ง: รับเฉพาะ blob กะทัดรัด (โดรนขยับทั้งก้อน) ตัดเส้นบาง/ขอบวัตถุนิ่ง
        _bx0, _by0, _bw, _bh = cv2.boundingRect(c)
        _extent = a / max(1.0, float(_bw * _bh))
        _aspect = max(_bw, _bh) / max(1.0, float(min(_bw, _bh)))
        if _extent < LOCK_MOTION_MIN_EXTENT or _aspect > LOCK_MOTION_MAX_ASPECT:
            continue
        mm = cv2.moments(c)
        if mm["m00"] == 0:
            continue
        bx = x0 + mm["m10"] / mm["m00"]
        by = y0 + mm["m01"] / mm["m00"]
        d = math.hypot(bx - pred_cx, by - pred_cy)
        if d < bestd:
            bestd, best = d, (bx, by)
    return best
# --- Data association (กัน lock stealing: โดรนตัวอื่น/FP แย่การล็อก) ---
# gate แคบ 'ตามขนาดเป้า+การเคลื่อนที่' (ไม่ใช่ตามขนาดเฟรม) → ตัวที่แทรกไกลถูกตัด
IOT_ASSOC_GATE_BBOX_MULT = 3.0  # รัศมี association = bbox_diag × ค่านี้ + การเคลื่อนที่ที่คาด
IOT_ASSOC_MIN_DEG = 0.459       # รัศมีขั้นต่ำเชิงมุม กันเป้าเล็กจน gate แคบเกิน (= 40px @ cam4 ppd 87.1)
IOT_ASSOC_MOTION_MULT = 2.5     # เผื่อระยะที่เป้าเคลื่อนต่อเฟรม × ค่านี้
IOT_MIN_CONF = 0.08             # ตัด FP conf ต่ำมาก (ต่ำกว่านี้ไม่รับเป็น candidate)


class _SimpleIoUTracker:
    """
    IoU-based tracker สำหรับ LOCK mode แทน CSRT
    - ไม่ขึ้นกับ texture/สีพื้นหลัง เหมาะกับโดรนบนท้องฟ้าสีเรียบ
    - ทำงานโดยใช้ YOLO detections + Kalman predicted position
    - ไม่ต้อง resize frame เป็น 25% → เร็วกว่า CSRT
    - เมื่อ YOLO ไม่มี detection ใหม่ → ใช้ Kalman predict (coasting)
    """

    def __init__(self) -> None:
        self.initialized: bool = False
        self.lost: bool = True
        self.track_bbox: Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h)
        self.smooth_cx: Optional[float] = None
        self.smooth_cy: Optional[float] = None
        self.last_conf: Optional[float] = None  # conf ของ detection ที่ match ล่าสุด
        self.miss_count: int = 0                # นับผล YOLO ที่ไม่ match ติดกัน (hysteresis)
        self._miss_accum: float = 0.0           # ตัวสะสม miss แบบถ่วงน้ำหนัก (slew นับน้อย)
        self._yolo_miss_run: int = 0            # เฟรม YOLO ที่ไม่ match ติดกัน — motion-assist 'แตะไม่ได้' (ขยาย gate/re-acquire)
        self.last_match_was_reacquire: bool = False  # diag: match ล่าสุดเกิดจากโหมด re-acquire (track เพี้ยนแล้วดึงกลับ)
        self._arm_moved_deg: float = 0.0         # แขนหมุนกี่องศาในเฟรมล่าสุด (จาก encoder) → ใช้ตัดสิน slew
        self._kalman: Optional[_TargetKalman] = None
        self._last_pan: Optional[float] = None   # มุมแขน (pan) ตอน update ล่าสุด — ego-motion comp
        self._last_tilt: Optional[float] = None

    def init_from_bbox(self, xb: int, yb: int, wb: int, hb: int,
                       cur_pan: Optional[float] = None, cur_tilt: Optional[float] = None) -> None:
        """เริ่ม track จาก bbox ใหม่ (จาก YOLO หรือ joystick button 4).
        cur_pan/cur_tilt = มุมแขน ณ ตอน acquire — ใช้เป็น baseline ego-motion comp
        (เฟรมถัดไปหลังแขน slew ไป center เป้า จะชดเชยการกระโดดของ pixel ได้ถูก)."""
        self.track_bbox = (xb, yb, wb, hb)
        self.smooth_cx = float(xb + wb * 0.5)
        self.smooth_cy = float(yb + hb * 0.5)
        self.initialized = True
        self.lost = False
        self.last_conf = None
        self.miss_count = 0
        self._miss_accum = 0.0
        self._yolo_miss_run = 0
        self._arm_moved_deg = 0.0
        self._last_pan = cur_pan
        self._last_tilt = cur_tilt
        if self._kalman is None:
            self._kalman = _TargetKalman()
        self._kalman.reset()
        self._kalman.update(self.smooth_cx, self.smooth_cy, 1.0)

    def _register_miss(self) -> None:
        """ไม่ match ในผล YOLO รอบนี้ — ประกาศ lost เมื่อสะสม miss ครบ IOT_MISS_FRAMES_TO_LOST.
        ตอนแขน slew (ภาพเบลอ YOLO บอดตามคาด) นับ miss แค่ IOT_SLEW_MISS_WEIGHT → ไม่ประกาศ lost
        ระหว่างหมุน (ยัง coast เล็งตาม bearing ต่อ); พอแขนนิ่งแล้วยังหาไม่เจอถึงนับเต็มแล้ว lost."""
        _slewing = self._arm_moved_deg > IOT_SLEW_BLIND_DEG
        self._miss_accum += IOT_SLEW_MISS_WEIGHT if _slewing else 1.0
        self.miss_count = int(self._miss_accum)
        # นับ run แยกจาก _miss_accum: motion-assist ลด _miss_accum ได้ แต่ 'แตะ' ตัวนี้ไม่ได้
        # → gate โตต่อเนื่องตราบใดที่ YOLO ยัง match ไม่ได้ (แม้ motion-assist ยึด track ไว้)
        self._yolo_miss_run += 1
        if self._miss_accum >= IOT_MISS_FRAMES_TO_LOST:
            self.lost = True

    def update(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
        cur_pan: Optional[float] = None,
        cur_tilt: Optional[float] = None,
        ppd_x: Optional[float] = None,
        ppd_y: Optional[float] = None,
    ) -> Tuple[bool, Optional[Tuple[int,int,int,int]]]:
        """
        อัปเดต tracker ด้วย YOLO detections เฟรมนี้
        - Ego-motion comp: กล้องติดบนแขน → เลื่อน state ตามมุมแขนที่เปลี่ยนก่อน match
        - ใช้ IoU matching กับ Kalman predicted bbox
        - ถ้าไม่ match → lost=True (จะ coast ด้วย Kalman ใน main loop)
        คืน (success, bbox) โดย bbox = (x,y,w,h)
        """
        if not self.initialized or self._kalman is None:
            return False, None

        # --- Ego-motion compensation ---
        # กล้องอยู่บนแขน: แขนหมุน +Δpan → จุดคงที่ในโลกเลื่อน -Δpan·ppd_x ในเฟรม (แนวนอน)
        # เลื่อน state ที่เก็บไว้เข้าเฟรมกล้อง 'ปัจจุบัน' ก่อน แล้วค่อย predict การเคลื่อนของโดรนเอง
        # → detection จริงจะตกใน gate ไม่หลุดตอนแขน slew (ต้นเหตุ track หลุดใน LOCK)
        self._arm_moved_deg = 0.0
        if (cur_pan is not None and cur_tilt is not None and ppd_x and ppd_y
                and self._last_pan is not None and self._last_tilt is not None):
            d_pan = float(cur_pan) - self._last_pan
            d_tilt = float(cur_tilt) - self._last_tilt
            shift_x = -d_pan * float(ppd_x)
            shift_y = -d_tilt * float(ppd_y)
            # "แขนกำลัง slew ไหม" เป็นคำถามเชิงมุม — encoder บอกมุมมาตรง ๆ อยู่แล้ว
            # ไม่ต้องแปลงเป็นพิกเซลก่อนแล้วค่อยเทียบ threshold พิกเซล (ซึ่งจะผูกกับ ppd โดยไม่จำเป็น)
            self._arm_moved_deg = math.hypot(d_pan, d_tilt)
            if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
                if self.smooth_cx is not None:
                    self.smooth_cx += shift_x
                if self.smooth_cy is not None:
                    self.smooth_cy += shift_y
                if self.track_bbox is not None:
                    self.track_bbox = (int(self.track_bbox[0] + shift_x),
                                       int(self.track_bbox[1] + shift_y),
                                       self.track_bbox[2], self.track_bbox[3])
                self._kalman.translate(shift_x, shift_y)
        if cur_pan is not None and cur_tilt is not None:
            self._last_pan = float(cur_pan)
            self._last_tilt = float(cur_tilt)

        # Predict ตำแหน่งถัดไป (1/15 วินาที ≈ 67ms)
        pred_cx, pred_cy = self._kalman.predict_ahead(1.0 / 15.0)
        pw = self.track_bbox[2] if self.track_bbox else 80
        ph = self.track_bbox[3] if self.track_bbox else 80
        pred_bbox = (
            int(pred_cx - pw * 0.5),
            int(pred_cy - ph * 0.5),
            pw, ph,
        )

        if not detections:
            self._register_miss()
            return False, None

        tw = self.track_bbox[2] if self.track_bbox else 80
        th = self.track_bbox[3] if self.track_bbox else 80

        # association gate 'ตามขนาดเป้า + การเคลื่อนที่ที่คาด' (ไม่ใช่ตามเฟรม) — กันตัวแทรกไกล
        bbox_diag = math.hypot(tw, th)
        motion_px = math.hypot(pred_cx - (self.smooth_cx or pred_cx),
                               pred_cy - (self.smooth_cy or pred_cy))
        # พื้นของ gate เป็นปริมาณเชิงมุม (กันเป้าเล็กจน gate แคบเกิน) → แปลงตาม ppd ของกล้องที่ใช้
        _assoc_min_px = (
            IOT_ASSOC_MIN_DEG * abs(float(ppd_x)) if ppd_x
            else IOT_ASSOC_MIN_DEG * PPD_X_FALLBACK
        )
        assoc_gate = max(_assoc_min_px,
                         bbox_diag * IOT_ASSOC_GATE_BBOX_MULT + motion_px * IOT_ASSOC_MOTION_MULT)
        # ยิ่งพลาดติดกันหลายเฟรม ตำแหน่งยิ่งไม่แน่นอน → ขยาย gate เพื่อ re-catch โดรนที่โผล่กลับ
        # cap ไว้ที่ระดับ lost threshold กันไม่ให้ gate โตไม่จำกัดจนไปคว้า FP
        _mc = min(self.miss_count, IOT_MISS_FRAMES_TO_LOST)
        assoc_gate += _mc * bbox_diag * IOT_GATE_MISS_GROWTH
        # สำคัญ: ขยาย gate ตาม _yolo_miss_run ด้วย — ตัวนี้ motion-assist ลดไม่ได้ จึงโตจริงตอนโดน
        # 'ยึด track ผิด' (YOLO เจอแต่ match ไม่ได้ตลอด) → detection จริงกลับเข้ามาใน gate ได้
        _ymr = min(self._yolo_miss_run, IOT_REACQUIRE_GATE_CAP)
        assoc_gate += _ymr * bbox_diag * IOT_GATE_MISS_GROWTH

        # Filter: อยู่ใน gate + ขนาดใกล้เคียง + conf ไม่ต่ำเกิน (ตัด FP)
        candidates = []
        for det in detections:
            dx, dy, dw, dh, conf = _det_xywhc(det)
            if conf < IOT_MIN_CONF:
                continue
            cx_d = dx + dw * 0.5
            cy_d = dy + dh * 0.5
            dist = math.hypot(cx_d - pred_cx, cy_d - pred_cy)
            if dist > assoc_gate:
                continue
            if tw >= 1 and th >= 1:
                rw = (float(dw) + 1e-9) / tw
                rh = (float(dh) + 1e-9) / th
                if rw < IOT_SIZE_RATIO_MIN or rw > IOT_SIZE_RATIO_MAX or rh < IOT_SIZE_RATIO_MIN or rh > IOT_SIZE_RATIO_MAX:
                    continue
            det_bbox = (int(dx), int(dy), max(1, int(dw)), max(1, int(dh)))
            candidates.append((det, dist, _iou_bbox(pred_bbox, det_bbox)))

        # เลือก 'ตัวที่ใกล้จุดทำนายที่สุด' (nearest-neighbor) — เป้าที่ล็อกอยู่ใกล้ prediction เสมอ;
        # ตัวแทรก/FP conf สูงจะอยู่ไกลกว่า จึงไม่ถูกเลือก (แก้ lock stealing). iou เป็น tiebreak
        # ⚠️ บทเรียน test 14:16/14:26 (แขนสบัด 30-37°): 'เลือกตัวมั่นใจสุด' ตอน re-acquire = คว้า
        # object อื่น/FP ที่อยู่ไกล → track กระโดด → bearing กระโดด → แขนสบัดออกจากเป้า. คืน nearest-
        # neighbor เสมอ (gate ที่โตจาก _yolo_miss_run พาโดรนจริงกลับเข้ามาใกล้ prediction ให้ถูกเลือกเอง);
        # ตอน gate โต (re-acquire) แค่ยกพื้น conf ขั้นต่ำกัน FP ในหน้าต่างกว้าง — 'ไม่' เปลี่ยนเกณฑ์เป็น conf
        _reacquire_mode = self._yolo_miss_run >= IOT_REACQUIRE_FRAMES
        _conf_floor = IOT_REACQUIRE_MIN_CONF if _reacquire_mode else 0.0
        _pool = [c for c in candidates if _det_xywhc(c[0])[4] >= _conf_floor] or candidates
        _pool.sort(key=lambda x: (x[1], -x[2]))  # ใกล้สุดก่อน, iou มากเป็น tiebreak (ทั้งปกติ/re-acquire)
        best_det = _pool[0][0] if _pool else None

        if best_det is not None:
            dx, dy, dw, dh, conf = _det_xywhc(best_det)
            xb = max(0, int(dx))
            yb = max(0, int(dy))
            wb = max(CSRT_MIN_BBOX, min(int(dw), frame_w - xb))
            hb = max(CSRT_MIN_BBOX, min(int(dh), frame_h - yb))
            self.track_bbox = (xb, yb, wb, hb)
            self.smooth_cx = float(xb + wb * 0.5)
            self.smooth_cy = float(yb + hb * 0.5)
            self.lost = False
            self.last_conf = float(conf)
            self.miss_count = 0
            self._miss_accum = 0.0
            self.last_match_was_reacquire = _reacquire_mode  # diag: match นี้เกิดจากโหมด re-acquire ไหม
            self._yolo_miss_run = 0   # YOLO ดึง track กลับได้แล้ว — รีเซ็ต run
            self._kalman.update(self.smooth_cx, self.smooth_cy, float(conf))
            return True, self.track_bbox
        else:
            self._register_miss()
            return False, None

    def apply_motion_measurement(self, cx: float, cy: float, conf: float = LOCK_MOTION_CONF) -> None:
        """อัปเดตตำแหน่งจาก motion-assist (ไม่ใช่ YOLO) — ประคอง track ทุกเฟรมระหว่าง YOLO บอด/เบลอ.
        conf ต่ำกว่า YOLO → พอ YOLO เจอจริงจะ override; แต่ระหว่างไม่มี YOLO ก็ไม่หลุด."""
        if not self.initialized or self._kalman is None:
            return
        self.smooth_cx = float(cx)
        self.smooth_cy = float(cy)
        if self.track_bbox is not None:
            _w, _h = self.track_bbox[2], self.track_bbox[3]
            self.track_bbox = (int(cx - _w * 0.5), int(cy - _h * 0.5), _w, _h)
        self._kalman.update(float(cx), float(cy), float(conf))
        self.lost = False
        self.last_conf = float(conf)
        self._miss_accum = max(0.0, self._miss_accum - 2.0)  # ลด miss สะสม (motion ประคองอยู่)
        self.miss_count = int(self._miss_accum)

    def reset(self) -> None:
        self.initialized = False
        self.lost = True
        self.track_bbox = None
        self.smooth_cx = None
        self.smooth_cy = None
        self.last_conf = None
        self.miss_count = 0
        self._miss_accum = 0.0
        self._yolo_miss_run = 0
        self._arm_moved_deg = 0.0
        self._last_pan = None
        self._last_tilt = None
        if self._kalman is not None:
            self._kalman.reset()


# =============================================================================
# Sound
# =============================================================================
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
LOCK_FIRE_EFFECT_DURATION = 0.45  # ระยะเวลา effect ตอนโปรแกรมลั่นไกเองใน LOCK (muzzle flash + FIRE)


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

# --- Parameters (center, YOLO, display, target, HUD, keys) ---
# รัศมี "ตรงกลาง" เป็นอัตราส่วนของ min(W,H) — ภายในวงนี้ถือว่ายิงได้ (เช่น 0.05 = 5%)
CENTER_RADIUS_RATIO = 0.05
# ถ้ากำหนด CENTER_RADIUS_PX > 0 จะใช้ค่าพิกเซลแทน ratio
CENTER_RADIUS_PX = 0

# Confidence: ตอน detection (AUTO/ยังไม่ lock) ใช้ 0.2; ตอน tracking ใน LOCK ใช้ 0.05
YOLO_CONF_DETECT = 0.25
YOLO_CONF_LOCK = 0.15
YOLO_CONF_MIN = 0.1   # ใช้กับ detection (backward compat)
YOLO_CONF_MAX = 0.95
YOLO_CONF_STEP = 0.05


def _adjust_yolo_conf(value: float, delta: float) -> float:
    """Snap YOLO detect conf to YOLO_CONF_STEP increments within [MIN, MAX]."""
    v = round(value + delta, 2)
    v = round(v / YOLO_CONF_STEP) * YOLO_CONF_STEP
    return max(YOLO_CONF_MIN, min(YOLO_CONF_MAX, v))

# YOLO รันเฉพาะบริเวณกลาง: อัตราส่วนของความกว้าง/สูงที่ใช้เป็น crop กลาง (เช่น 0.5 = 50%)
CENTER_CROP_RATIO = 0.5

# --- LOCK ROI crop (ปิดช่องว่าง sim-vs-real ด้าน detection) ---
# ปัญหาโดรนจริง: cam4 = 4K (3840×2160) → center-crop 50% (1920×1080) → บีบเป็น 640×640
# (บีบแนวนอน 3.0×, แนวตั้ง 1.7× = ภาพบิดสัดส่วน) → โดรน 40px ใน 4K เหลือ ~13×24px บิดเบี้ยว
# เข้า YOLO → det_rate จริงแค่ ~0.2-0.5, conf ~0.35. ซ้ำร้าย lead-aim ตั้งใจชี้ลำกล้อง 'นำหน้า'
# โดรน → โดรนอยู่เยื้องกลางเฟรม → หลุดออกนอก center crop ได้ทั้งตัว = det_rate 0 เป็นช่วง ๆ
# (ตรง log 12:58). sim ฉีด detection ข้าม pipeline นี้ทั้งหมด เลยไม่เคยเห็นปัญหา.
# แก้: ตอน LOCK ครอบหน้าต่าง 640×640 'พิกเซลจริง' (native, ไม่ resize ไม่บิด) ตามตำแหน่ง track
# → โดรนคงขนาด native เต็ม ๆ + อยู่ในหน้าต่างเสมอไม่ว่าเยื้องไปไหน (ppd=64px/° → หน้าต่าง 10°,
# ขยับตามทุกเฟรม; เป้า 15°/s = 64px/เฟรม → margin ±320px เหลือเฟือ) — และเร็วขึ้นด้วย (ไม่ต้อง
# resize 1920×1080). เป้าใหญ่/ใกล้: ขยาย ROI ตาม bbox แล้วค่อยย่อ (ยังดีกว่า center crop เดิมมาก)
LOCK_ROI_ENABLED = True
LOCK_ROI_BBOX_MULT = 4.0   # ROI ต้องกว้างอย่างน้อย bbox_diag × ค่านี้ (เป้าใหญ่ → ROI ขยาย → resize ลง)
# YOLO ไม่ match → 'ขยาย' ROI แทนที่จะถอยไป center crop มุมกว้าง (บทเรียนจาก test 14:14: ถอย center
# crop = ตาบอด เพราะ crop นั้นแหละที่มองโดรนไม่เห็นตั้งแต่แรก → หลุดยาว 8 วิ). ขยายหน้าต่าง native
# รอบตำแหน่ง track ตาม _yolo_miss_run → ค้นกว้างขึ้นแต่ยังคมกว่า center crop มาก → เจอแล้ว match →
# run รีเซ็ต → ROI หดกลับคม. ยิ่งไม่เจอยิ่งกว้าง (แต่ cap) — คุม resolution↔พื้นที่ค้นให้สมดุล
# ROI ต้องครอบ 'มุมคงที่' ไม่ใช่ 'พิกเซลคงที่' — นี่คือสิ่งที่ทำให้ det_rate รอดตอนเปลี่ยนกล้อง:
# ขนาดโดรนใน input ของ YOLO = ขนาดเชิงมุม × ppd. ถ้า ROI ยึดพิกเซล (640 native) แล้วเปลี่ยนไป
# กล้อง ppd ต่ำกว่า โดรนจะเล็กลงในภาพที่ป้อน YOLO → หลุด distribution ที่เทรนมา.
# ตัวอย่าง: โดรน 0.3m ที่ 100m subtend 0.172° → cam4 (ppd 87) = 15px ใน YOLO input;
# ถ้ายึดพิกเซลแล้วย้ายไปกล้อง ppd 40 จะเหลือ 6.9px (ต่ำกว่า TARGET_MIN_BOX_PX) = ตรวจไม่เจอ.
# ยึดมุมแทน → ครอบพื้นที่เท่าเดิมเชิงมุม แล้ว resize เข้า 640 → โดรนคง 15px ทุกกล้อง.
# ค่าองศาด้านล่างเลือกให้ที่ cam4 (ppd 87.138) ออกมาเป็น 640/1920/130 px เท่าเดิมเป๊ะ
LOCK_ROI_SPAN_DEG = 7.3447        # มุมที่ ROI ครอบ (= 640px @ cam4)
LOCK_ROI_MISS_GROW_DEG = 1.4919   # ขยาย ROI ต่อ 1 เฟรมที่ YOLO ไม่ match (= 130px @ cam4)
LOCK_ROI_MAX_DEG = 22.034         # cap มุมของ ROI (= 1920px @ cam4) — ที่ max ยังเป็นจัตุรัส aspect ถูก
                                  # ต่างจาก center crop เดิมที่บิด 3×
# ขนาดที่ส่งเข้า YOLO สำหรับ crop กลาง (1280 = แม่น, 640 = เร็ว) — ถูก override โดย load
YOLO_CENTER_IMGSZ = 1280

# ความเร็ว: รัน YOLO แค่ทุก N เฟรม; เฟรมอื่นใช้ผลล่าสุด
YOLO_INTERVAL = 1
# True = โหลด engine 640 (เร็ว, ใช้ตอน tracking ด้วย)
USE_FAST_YOLO = True
# Engine 640 สำหรับ detection/tracking ความเร็ว (ใช้เมื่อ USE_FAST_YOLO)
YOLO_ENGINE_640_PATH = "rgb_multiclass_imgsz640.engine"
# Resize เฟรมก่อนแสดงผลถ้าใหญ่กว่านี้ (ลดเวลา imshow)
DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080
# เต็มจอไร้ขอบ (ไม่มีแถบชื่อ/ปุ่มปิด) — ภาพ fit ในจอ อาจมี letterbox
DISPLAY_FULLSCREEN = True
# cam8 mapping: H265 5120x1440 on Jetson Orin needs longer warm-up than cam4
CAM8_MAP_WARMUP_SEC = 12.0
CAM8_MAP_START_RETRIES = 2
CAM8_MAP_STALE_SEC = 3.0
CAM8_MAP_RECONNECT_INTERVAL_SEC = 5.0

# วงเล็งหลายชั้น: อัตราส่วนรัศมี [วงใน=ยิงได้, วงกลาง, วงนอก, วงใหญ่ 2.5× วงนอก] ของ min(W,H)
RETICLE_RADIUS_RATIOS = (0.03, 0.05, 0.08, 0.13)  # 0.20 = 0.08 × 2.5
# Sticky target: เมื่อมีเป้าแล้ว ถ้า bbox หายไปแป๊บไม่สลับไปตัวอื่น — รัศมีถือว่า "ตัวเดิม" (อัตราส่วน min(W,H))
TARGET_STICKY_RADIUS_RATIO = 0.10
# bbox เล็กสุดที่ถือว่าเป็นเป้าได้ (ไม่ตามจุดเล็ก/noise): min(width, height) ต้องไม่ต่ำกว่านี้
TARGET_MIN_BOX_PX = 10  # พิกเซล — ถ้า min(w,h) < ค่านี้ไม่เลือกเป็นเป้า
TARGET_MIN_BOX_RATIO = 0.01  # อัตราส่วนของ min(frame_w, frame_h) — ใช้ค่าที่ใหญ่กว่าระหว่าง MIN_PX กับ min_side * RATIO
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
DRONE_SIZE_M = 0.25  # 35 cm

# สีเส้นเล็งตามเวลา: กลางวัน = แดง/cyan เห็นชัด, กลางคืน = ขาว/เหลือง
DAY_HOUR_START = 6   # 6:00 = เริ่มกลางวัน
DAY_HOUR_END = 18    # 18:00 = เริ่มกลางคืน

# ใช้กล้องจาก config ตัวไหน (None = ใช้ ACTIVE_CAMERA)
CAMERA_NAME = None

WINDOW_NAME = "Gun Aim Assist"

# classes ของโมเดล rgb_multiclass_imgsz640.engine — ตรงกับ CLASS_NAMES ในโน้ตบุ๊กเทรน (ลำดับ id 0..7)
YOLO_CLASS_NAMES_MULTICLASS = (
    "drone",
    "person",
    "dog",
    "bird",
    "airplane",
    "car",
    "motorcycle",
    "boat",
)
YOLO_CLASS_NAMES_DRONE_ONLY = ("drone",)
# ค่าเริ่มต้นก่อนอ่าน config (main() จะตั้งใหม่ตามโหมด)
YOLO_CLASS_NAMES = YOLO_CLASS_NAMES_MULTICLASS


def _canonical_class_label(cls_id) -> str:
    """ชื่อ class สำหรับ HUD/label — ใช้เฉพาะรายการเทรน (ไม่ใช้ชื่อจาก engine เมื่อไม่มี metadata)."""
    try:
        i = int(cls_id)
    except (TypeError, ValueError):
        return "unknown"
    if 0 <= i < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[i]
    return "unknown"


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
KEY_SETTINGS = "S"        # หน้าตั้งค่า (settings_screen) — ทุกค่า ไม่ใช่แค่ ballistics
KEY_WIZARD = "W"          # calibration wizard — คาลิเบรตกล้องใหม่ให้จบในแอป
KEY_QUIT = "Q"
KEY_PXDEG_CALIB = "G"
KEY_CLASS_PREV = "["
KEY_CLASS_NEXT = "]"

# ขั้นตอนเลื่อนศูนย์เล็ง (พิกเซล)
AIM_CENTER_STEP_PX = 10


# =============================================================================
# Config / Camera
# =============================================================================
class ShooterConfig:
    """
    เก็บค่าศูนย์เล็ง (boresight) + ค่ากระสุน (ระยะหวังผล, ความเร็ว, น้ำหนัก).

    แยกสองอย่างนี้คนละไฟล์เพราะเป็นสมบัติของคนละสิ่ง:
      - ballistics (muzzle velocity / น้ำหนักกระสุน / ขนาดเป้า) = สมบัติของ 'ปืนกับกระสุน'
        → gun_aim_assist_config.json (ไฟล์เดียว ใช้ร่วมทุกกล้อง)
      - boresight (กล้องเล็งเบี้ยวจากลำกล้องเท่าไร) = สมบัติของ 'การติดตั้งกล้องตัวนั้น'
        → calibration_data/{cam}_boresight.json (แยกต่อกล้อง)

    boresight เก็บเป็น 'องศา' ไม่ใช่พิกเซล: ของเดิมเก็บ offset_x/offset_y เป็นพิกเซล 4K ดิบ
    (-450, +400 = -5.16°/-4.46°) ในไฟล์ global → เปลี่ยนกล้องแล้วค่าเดิมยังอยู่ พิกเซลชุดเดิม
    บนกล้องความละเอียดอื่นจะกลายเป็นมุมคนละค่า → ปืนเล็งเบี้ยว 5-10° = พลาดทุกนัด
    """
    FILENAME = "gun_aim_assist_config.json"

    def __init__(self, camera_name: str = "cam4"):
        self.offset_pan_deg = 0.0
        self.offset_tilt_deg = 0.0
        self.effective_range_m = 100
        self.muzzle_velocity_ms = 900
        self.bullet_weight_g = 9
        self.target_size_m = 0.30
        self._camera_name = camera_name
        self._ppd_x = PPD_X_FALLBACK
        self._ppd_y = -PPD_X_FALLBACK
        _here = os.path.dirname(os.path.abspath(__file__))
        self._path = os.path.join(_here, self.FILENAME)
        self._boresight_path = os.path.join(
            _here, "calibration_data", f"{camera_name}_boresight.json"
        )

    def set_ppd(self, ppd_x, ppd_y):
        """ต้องเรียกหลังโหลดคาลิเบรต — boresight เก็บเป็นองศา ต้องใช้ ppd แปลงกลับเป็นพิกเซล."""
        if ppd_x and abs(float(ppd_x)) > 1e-6:
            self._ppd_x = float(ppd_x)
        if ppd_y and abs(float(ppd_y)) > 1e-6:
            self._ppd_y = float(ppd_y)

    def load(self):
        # 1) ballistics (global)
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.effective_range_m = float(d.get("effective_range_m", 100))
                self.muzzle_velocity_ms = float(d.get("muzzle_velocity_ms", 900))
                self.bullet_weight_g = float(d.get("bullet_weight_g", 9))
                self.target_size_m = float(d.get("target_size_m", 0.30))
            except Exception:
                pass
        # 2) boresight (ต่อกล้อง)
        if os.path.isfile(self._boresight_path):
            try:
                with open(self._boresight_path, "r", encoding="utf-8") as f:
                    b = json.load(f)
                self.offset_pan_deg = float(b.get("offset_pan_deg", 0.0))
                self.offset_tilt_deg = float(b.get("offset_tilt_deg", 0.0))
                return
            except Exception:
                pass
        # 3) migration: ยังไม่มีไฟล์ boresight ต่อกล้อง แต่ไฟล์ global เก่ามี offset พิกเซลอยู่
        #    → แปลงเป็นองศาด้วย ppd ปัจจุบัน แล้วเขียนไฟล์ใหม่ (ทำครั้งเดียว)
        self._migrate_pixel_offset()

    def _migrate_pixel_offset(self):
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        if "offset_x" not in d and "offset_y" not in d:
            return
        ox = float(d.get("offset_x", 0))
        oy = float(d.get("offset_y", 0))
        self.offset_pan_deg = ox / self._ppd_x
        self.offset_tilt_deg = oy / self._ppd_y
        print(
            f"⚠️ Gun Aim Assist: ย้าย boresight เดิม (พิกเซล {ox:+.0f},{oy:+.0f} @ ppd "
            f"{self._ppd_x:.1f}/{self._ppd_y:.1f}) → องศา "
            f"({self.offset_pan_deg:+.2f}°, {self.offset_tilt_deg:+.2f}°) "
            f"เก็บที่ {os.path.basename(self._boresight_path)}. "
            "ถ้าเพิ่งเปลี่ยนกล้อง ค่านี้เป็นของกล้องเก่า — ต้อง zero ใหม่",
            flush=True,
        )
        self.save_boresight()

    def save_boresight(self):
        try:
            os.makedirs(os.path.dirname(self._boresight_path), exist_ok=True)
            with open(self._boresight_path, "w", encoding="utf-8") as f:
                json.dump({
                    "camera": self._camera_name,
                    "offset_pan_deg": self.offset_pan_deg,
                    "offset_tilt_deg": self.offset_tilt_deg,
                }, f, indent=2)
        except Exception:
            pass

    def save(self):
        self.save_boresight()
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({
                    "effective_range_m": self.effective_range_m,
                    "muzzle_velocity_ms": self.muzzle_velocity_ms,
                    "bullet_weight_g": self.bullet_weight_g,
                    "target_size_m": self.target_size_m,
                }, f, indent=2)
        except Exception:
            pass

    @property
    def offset_x(self):
        """boresight เป็นพิกเซลของกล้องที่ใช้อยู่ (derive จากองศา × ppd)."""
        return int(round(self.offset_pan_deg * self._ppd_x))

    @property
    def offset_y(self):
        return int(round(self.offset_tilt_deg * self._ppd_y))

    def get_center(self, frame_w, frame_h):
        cx = frame_w // 2 + self.offset_x
        cy = frame_h // 2 + self.offset_y
        cx = max(0, min(frame_w - 1, cx))
        cy = max(0, min(frame_h - 1, cy))
        return cx, cy

    def move(self, dx, dy):
        """เลื่อนศูนย์เล็งด้วยพิกเซล (จากคีย์บอร์ด) — เก็บเป็นองศา."""
        self.offset_pan_deg += float(dx) / self._ppd_x
        self.offset_tilt_deg += float(dy) / self._ppd_y

    def reset_aim_center(self):
        self.offset_pan_deg = 0.0
        self.offset_tilt_deg = 0.0


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


def estimate_distance_m_ppd(w_px, h_px, ppd_x, ppd_y, size_m=0.35):
    """
    คำนวณระยะถึงเป้า (เมตร) จากขนาด bbox + pixel_per_degree ที่ 'วัดจริง'.

    ต่างจาก estimate_distance_m() ที่ใช้ FOV ที่พิมพ์มือใน config: ppd คือ angular scale
    ที่คาลิเบรตจริงตรงกลางเฟรม — ซึ่งเป็นตรงที่เป้าอยู่ตอน LOCK พอดี → มุมที่เป้า subtend
    คือ w_px/|ppd_x| องศา ตรง ๆ ไม่ต้องผ่าน FOV เลย.

    เหตุผลที่ต้องมี: FOV ใน config เป็นค่าประมาณจากสเปคเลนส์และคลาดจากของจริงได้มาก
    (cam4 ตั้ง 60°×36° แต่ ppd บอกว่าจริง ๆ 44°×24° = ต่าง 36%) — range ผิด → t_flight,
    ballistic lead, hit_radius_deg, lead attenuation และ slew guard ผิดพร้อมกันหมด.
    """
    if w_px < 3 or h_px < 3:
        return None
    ppd_x = abs(float(ppd_x or 0.0))
    ppd_y = abs(float(ppd_y or 0.0))
    if ppd_x < 1e-6 and ppd_y < 1e-6:
        return None
    deg2rad = math.pi / 180.0
    theta_h = (w_px / ppd_x) * deg2rad if ppd_x >= 1e-6 else 0.0
    theta_v = (h_px / ppd_y) * deg2rad if ppd_y >= 1e-6 else 0.0
    if theta_h < 1e-6 and theta_v < 1e-6:
        return None
    dist_h = size_m / (2.0 * math.tan(theta_h / 2.0)) if theta_h >= 1e-6 else None
    dist_v = size_m / (2.0 * math.tan(theta_v / 2.0)) if theta_v >= 1e-6 else None
    if dist_h is not None and dist_v is not None:
        return (dist_h + dist_v) / 2.0
    return dist_h if dist_h is not None else dist_v


def fov_from_ppd(frame_w, frame_h, ppd_x, ppd_y):
    """FOV (องศา) ที่ 'สอดคล้องกับคาลิเบรตจริง' — ใช้แทนค่าที่พิมพ์มือใน config."""
    ppd_x = abs(float(ppd_x or 0.0))
    ppd_y = abs(float(ppd_y or 0.0))
    if ppd_x < 1e-6 or ppd_y < 1e-6 or frame_w <= 0 or frame_h <= 0:
        return None, None
    return frame_w / ppd_x, frame_h / ppd_y

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


def _yolo_detection_mode_from_config() -> str:
    if _config_mod is not None:
        return str(getattr(_config_mod, "CAM4_ARM_YOLO_DETECTION_MODE", "multiclass")).strip().lower()
    return "multiclass"


def _yolo_use_fast_from_config() -> bool:
    if _config_mod is not None and hasattr(_config_mod, "CAM4_ARM_YOLO_USE_FAST"):
        return bool(_config_mod.CAM4_ARM_YOLO_USE_FAST)
    return USE_FAST_YOLO


def _yolo_load_alt_on_start_from_config() -> bool:
    if _config_mod is not None and hasattr(_config_mod, "CAM4_ARM_YOLO_LOAD_ALT_ON_START"):
        return bool(_config_mod.CAM4_ARM_YOLO_LOAD_ALT_ON_START)
    return False


def _infer_imgsz_from_engine_path(path: str, default: int = 640) -> int:
    p = str(path).lower()
    if "imgsz1280" in p:
        return 1280
    if "imgsz640" in p:
        return 640
    if "imgsz1920" in p:
        return 1920
    return default


def _resolve_engine_path(engine_path_relative: str) -> str:
    if os.path.isabs(engine_path_relative):
        return engine_path_relative
    return os.path.join(os.path.dirname(__file__), engine_path_relative)


def load_tensorrt_engine(engine_path_relative: str, imgsz: int = None):
    """
    โหลด TensorRT/YOLO engine ไฟล์เดียว.
    Returns: (model, imgsz) หรือ (None, None).
    """
    if not (YOLO_AVAILABLE and UltralyticsYOLO is not None):
        return None, None
    path = _resolve_engine_path(engine_path_relative)
    if not os.path.exists(path):
        return None, None
    if imgsz is None:
        imgsz = _infer_imgsz_from_engine_path(path)
    try:
        m = UltralyticsYOLO(path, task="detect")
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, device=0, imgsz=imgsz)
        print(f"Gun Aim Assist: TensorRT loaded: {os.path.basename(path)} (imgsz={imgsz})")
        return m, imgsz
    except Exception as e:
        print(f"Gun Aim Assist: TensorRT load failed ({path}): {e}")
        return None, None


def load_fast_tensorrt_engine(engine_path_relative: str):
    """Backward-compatible wrapper — ตรวจ imgsz จากชื่อไฟล์."""
    return load_tensorrt_engine(engine_path_relative)


def _yolo_aim_engine_plan():
    """
    อ่าน config แล้วคืนแผนโหลด engine สำหรับ 22_gun_aim_assist_vector.
    Returns dict: mode, primary_path, alt_path, use_fast, enable_class_cycle, class_names
    """
    mode = _yolo_detection_mode_from_config()
    use_fast = _yolo_use_fast_from_config()
    if mode == "drone_only":
        p640 = "yolo_11n_day_night_200_2_imgsz640.engine"
        p1280 = "yolo_11n_day_night_200_2_imgsz1280.engine"
        if _config_mod is not None:
            p640 = getattr(_config_mod, "CAM4_ARM_YOLO_ENGINE_DRONE_640", p640)
            p1280 = getattr(_config_mod, "CAM4_ARM_YOLO_ENGINE_DRONE_1280", p1280)
        primary_path = p640 if use_fast else p1280
        alt_path = p1280 if use_fast else p640
        return {
            "mode": mode,
            "primary_path": primary_path,
            "alt_path": alt_path,
            "use_fast": use_fast,
            "enable_class_cycle": False,
            "class_names": list(YOLO_CLASS_NAMES_DRONE_ONLY),
        }
    p640 = YOLO_ENGINE_640_PATH
    p_thermal = "thermal_multiclass_imgsz640.engine"
    enable_class_cycle = True
    if _config_mod is not None:
        p640 = getattr(_config_mod, "CAM4_ARM_YOLO_ENGINE_RGB_640", p640)
        p_thermal = getattr(_config_mod, "CAM4_ARM_YOLO_ENGINE_THERMAL_640", p_thermal)
        enable_class_cycle = bool(getattr(_config_mod, "CAM4_ARM_YOLO_ENABLE_CLASS_CYCLE", True))
    return {
        "mode": mode,
        "primary_path": p640,
        "alt_path": p_thermal,
        "use_fast": use_fast,
        "enable_class_cycle": enable_class_cycle,
        "class_names": list(YOLO_CLASS_NAMES_MULTICLASS),
    }


def load_yolo_for_aim_assist():
    """
    โหลด YOLO ตาม CAM4_ARM_YOLO_DETECTION_MODE ใน config.
    Returns: (yolo_model, imgsz) หรือ (None, None).
    """
    plan = _yolo_aim_engine_plan()
    model, imgsz = load_tensorrt_engine(plan["primary_path"])
    if model is not None:
        print(
            f"Gun Aim Assist: YOLO primary loaded ({plan['mode']}, imgsz={imgsz})"
        )
        return model, imgsz
    if plan["mode"] == "drone_only":
        print("Gun Aim Assist: drone_only primary engine not available.")
        return None, None
    if USE_FAST_YOLO:
        print("Gun Aim Assist: multiclass 640 load failed, falling back to smart_detection_yolo_only")
    model, imgsz = load_yolo_model()
    if imgsz is None:
        imgsz = YOLO_CENTER_IMGSZ
    return model, imgsz


# =============================================================================
# Detection
# =============================================================================
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


def crop_roi_native(frame, cx, cy, target_size, bbox_diag=None, miss_run=0, ppd=None):
    """LOCK ROI crop: หน้าต่างสี่เหลี่ยมจัตุรัสรอบตำแหน่ง track (cx,cy) สำหรับส่งเข้า YOLO.
    - ROI ครอบ 'มุมคงที่' (LOCK_ROI_SPAN_DEG) → โดรนกินพิกเซลใน YOLO input เท่ากันทุกกล้อง
      ที่ cam4 (ppd 87.1) มุมนี้ = 640px = target_size พอดี → ไม่ resize (native, คมสุด)
    - ppd=None → ถอยไปใช้ target_size พิกเซลตรง ๆ (พฤติกรรมเดิม ตอนยังไม่มีคาลิเบรต)
    - เป้าใหญ่ (bbox_diag × LOCK_ROI_BBOX_MULT > roi) → ขยาย ROI ให้ครอบเป้าแล้ว resize ลง
    - miss_run > 0 (YOLO ไม่ match ติดกัน) → ขยาย ROI ค้นกว้างขึ้น (แทนถอย center crop ที่ตาบอด)
    - clamp ขอบเฟรม: หน้าต่างคงขนาดเดิม เลื่อนชิดขอบ (เป้าแค่เยื้องจากกลาง crop — mapping ถูกต้อง)
    Returns: (crop_resized, x0, y0, cw, ch) contract เดียวกับ crop_center_and_resize
    → ใช้ map_detections_to_full_frame ตัวเดิมได้เลย."""
    h, w = frame.shape[:2]
    _ppd = abs(float(ppd)) if ppd else 0.0
    if _ppd >= 1e-6:
        roi = int(round(LOCK_ROI_SPAN_DEG * _ppd))
        grow_px = LOCK_ROI_MISS_GROW_DEG * _ppd
        max_px = int(round(LOCK_ROI_MAX_DEG * _ppd))
    else:
        roi = int(target_size)
        grow_px = LOCK_ROI_MISS_GROW_DEG * PPD_X_FALLBACK
        max_px = int(round(LOCK_ROI_MAX_DEG * PPD_X_FALLBACK))
    if bbox_diag is not None and bbox_diag > 0:
        roi = max(roi, int(bbox_diag * LOCK_ROI_BBOX_MULT))
    if miss_run > 0:
        roi = int(roi + miss_run * grow_px)
    roi = min(roi, max_px, w, h)  # cap + กันเฟรมเล็กกว่า ROI
    x0 = int(round(cx - roi * 0.5))
    y0 = int(round(cy - roi * 0.5))
    x0 = max(0, min(x0, w - roi))
    y0 = max(0, min(y0, h - roi))
    crop = frame[y0 : y0 + roi, x0 : x0 + roi]
    if roi == target_size:
        crop = np.ascontiguousarray(crop)  # native — ไม่ resize (slice เป็น view ต้องทำ contiguous)
    else:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop, x0, y0, roi, roi


def map_detections_to_full_frame(detections, x0, y0, cw, ch, crop_size):
    """
    แปลง bbox จากพิกัด crop (crop_size x crop_size) กลับเป็นพิกัดเฟรมเต็ม.
    detections: [(x, y, w, h, conf, ...), ...] ใน crop space.
    """
    if not detections:
        return []
    scale_x = cw / crop_size
    scale_y = ch / crop_size
    out = []
    for det in detections:
        x, y, bw, bh, conf = det[:5]
        cls_id = int(det[5]) if len(det) > 5 else -1
        cls_name = _canonical_class_label(cls_id)
        x_full = x0 + int(x * scale_x)
        y_full = y0 + int(y * scale_y)
        w_full = max(1, int(bw * scale_x))
        h_full = max(1, int(bh * scale_y))
        out.append((x_full, y_full, w_full, h_full, conf, cls_id, cls_name))
    return out


def _det_xywhc(det):
    """คืน (x,y,w,h,conf) รองรับ detection แบบเดิมและแบบ multiclass."""
    x, y, w, h, conf = det[:5]
    return int(x), int(y), int(w), int(h), float(conf)


def _det_cls(det):
    """คืน (cls_id, cls_name) โดยชื่อมาจาก YOLO_CLASS_NAMES เท่านั้น (ตรงโน้ตบุ๊กเทรน)."""
    cls_id = int(det[5]) if len(det) > 5 else -1
    return cls_id, _canonical_class_label(cls_id)


def _hud_class_display_name(cls_name: str) -> str:
    """ข้อความบน HUD — ใช้ชื่อจากรายการเทรนเท่านั้น."""
    s = str(cls_name or "").strip().lower()
    if s == "unknown" or not s:
        return "unknown"
    if s in YOLO_CLASS_NAMES:
        return s
    return s.replace("_", " ")


def filter_detections_by_active_class(detections, active_class_idx):
    """กรอง detection ตาม class ที่เลือก (single-class)."""
    if not detections:
        return []
    return [det for det in detections if _det_cls(det)[0] == int(active_class_idx)]


def _distance_to_crosshair(det, cx_frame, cy_frame):
    """ระยะจากศูนย์ bbox ถึงศูนย์เล็ง (พิกเซล)."""
    x, y, w, h, _ = _det_xywhc(det)
    cx_t = x + w // 2
    cy_t = y + h // 2
    return math.sqrt((cx_t - cx_frame) ** 2 + (cy_t - cy_frame) ** 2)


def _distance_to_point(det, ref_cx, ref_cy):
    """ระยะจากศูนย์ bbox ถึงจุด (ref_cx, ref_cy) พิกเซล."""
    x, y, w, h, _ = _det_xywhc(det)
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
    x, y, w, h, _ = _det_xywhc(last_target_det)
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


def _cam8_cue_size_norm(cue_dict):
    """
    ดึง normalized bbox size จาก cam8 cue dict.
    Returns: (w_norm, h_norm) หรือ None ถ้าไม่มีข้อมูล.
    """
    if cue_dict is None:
        return None
    wn = cue_dict.get("bbox_w_norm")
    hn = cue_dict.get("bbox_h_norm")
    if wn is None or hn is None:
        return None
    return float(wn), float(hn)


def _size_mismatch_penalty(det, cam4_w, cam4_h, cue_w_norm, cue_h_norm):
    """
    คำนวณ size mismatch penalty ระหว่าง detection bbox บน cam4 กับ cue norm size.
    ใช้ L1 norm difference บน normalized size; scale เป็นพิกเซลสำหรับรวมกับ dist.
    Returns: float penalty (0 = ตรงกัน, สูง = ผิดพลาดมาก).
    """
    if det is None or cam4_w <= 0 or cam4_h <= 0:
        return 0.0
    dx, dy, dw, dh = det[0], det[1], det[2], det[3]
    det_wn = float(dw) / cam4_w
    det_hn = float(dh) / cam4_h
    penalty_norm = abs(det_wn - cue_w_norm) + abs(det_hn - cue_h_norm)
    # scale ให้อยู่ในหน่วยเดียวกับ dist_to_crosshair (พิกเซล)
    return penalty_norm * float(min(cam4_w, cam4_h))


def pick_best_target_with_cue(detections, cx_frame, cy_frame, min_side=None,
                               optional_cue=None, frame_w=None, frame_h=None):
    """
    เลือก detection ที่เหมาะที่สุด โดยถ้ามี cam8 cue ที่มีข้อมูล bbox_norm จะนำ size penalty
    มาช่วย scoring: score = dist_crosshair + weight * size_penalty_px.
    ถ้า CAM8_CUE_SIZE_MATCH_ENABLED=False หรือไม่มี cue/norm → พฤติกรรมเหมือน pick_best_target.
    """
    if not detections:
        return None
    detections = _filter_detections_by_min_size(detections, min_side)
    if not detections:
        return None

    cue_size = None
    if (CAM8_CUE_SIZE_MATCH_ENABLED and optional_cue is not None
            and frame_w is not None and frame_h is not None):
        cue_size = _cam8_cue_size_norm(optional_cue)

    if cue_size is None:
        return min(detections, key=lambda det: _distance_to_crosshair(det, cx_frame, cy_frame))

    cue_w_norm, cue_h_norm = cue_size
    fw, fh = float(frame_w), float(frame_h)

    scored = []
    for det in detections:
        dist = _distance_to_crosshair(det, cx_frame, cy_frame)
        penalty = _size_mismatch_penalty(det, fw, fh, cue_w_norm, cue_h_norm)
        if CAM8_CUE_SIZE_MAX_TOL is not None:
            penalty_norm = abs(float(det[2]) / fw - cue_w_norm) + abs(float(det[3]) / fh - cue_h_norm)
            if penalty_norm > CAM8_CUE_SIZE_MAX_TOL:
                continue
        scored.append((dist + CAM8_CUE_SIZE_MATCH_WEIGHT * penalty, det))

    if not scored:
        # กรองออกหมด — fallback ไม่กรอง
        return min(detections, key=lambda det: _distance_to_crosshair(det, cx_frame, cy_frame))

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def compute_guide_direction(target_det, cx_frame, cy_frame, radius_px, ready_to_fire):
    """
    คำนวณทิศทางแนะนำ: ขยับซ้าย/ขวา/ขึ้น/ลง เมื่อเป้าอยู่นอกวงยิงได้.
    Returns: (guide_h, guide_v) โดย guide_h เป็น "LEFT" | "RIGHT" | None, guide_v เป็น "UP" | "DOWN" | None.
    """
    if target_det is None or ready_to_fire:
        return None, None
    x, y, w_d, h_d, _ = _det_xywhc(target_det)
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
    ax, ay, aw, ah, _ = _det_xywhc(a)
    bx, by, bw, bh, _ = _det_xywhc(b)
    return (ax, ay, aw, ah) == (bx, by, bw, bh)


# =============================================================================
# HUD
# =============================================================================
def _draw_hud_segments_row(frame, x_start, y_pos, font, font_scale, thickness, parts, max_x, sep=" | "):
    """Draw colored HUD segments on one horizontal row; stop when exceeding max_x."""
    x = int(x_start)
    sep_w = cv2.getTextSize(sep, font, font_scale, thickness)[0][0] if sep else 0
    for i, (text, color) in enumerate(parts):
        if not text:
            continue
        if i > 0 and sep:
            if x + sep_w > max_x:
                break
            cv2.putText(frame, sep, (x, y_pos), font, font_scale, (180, 180, 180), thickness)
            x += sep_w
        chunk = str(text)
        tw = cv2.getTextSize(chunk, font, font_scale, thickness)[0][0]
        if x + tw > max_x:
            while chunk and x + cv2.getTextSize(chunk + "…", font, font_scale, thickness)[0][0] > max_x:
                chunk = chunk[:-1]
            chunk = (chunk + "…") if chunk else ""
            tw = cv2.getTextSize(chunk, font, font_scale, thickness)[0][0]
        if not chunk:
            break
        cv2.putText(frame, chunk, (x, y_pos), font, font_scale, color, thickness)
        x += tw


def _gun_detection_line_and_color(ctx):
    """Return (det_line, det_color) for bottom HUD metrics row."""
    yolo_thermal_model = ctx.get("yolo_thermal_model")
    yolo_detection_mode = ctx.get("yolo_detection_mode", "multiclass")
    detection_use_thermal = ctx.get("detection_use_thermal", False)
    yolo_alt_imgsz = ctx.get("yolo_alt_imgsz", 1280)
    yolo_primary_imgsz = ctx.get("yolo_primary_imgsz", 640)
    if yolo_thermal_model is not None:
        if yolo_detection_mode == "drone_only":
            cur_imgsz = yolo_alt_imgsz if detection_use_thermal else yolo_primary_imgsz
            det_line = f"DET:imgsz {cur_imgsz}"
            det_color = (0, 200, 255) if detection_use_thermal else (0, 255, 220)
        else:
            det_line = "DET:THERMAL" if detection_use_thermal else "DET:RGB"
            det_color = (0, 140, 255) if detection_use_thermal else (0, 255, 220)
    elif yolo_detection_mode == "drone_only":
        det_line = f"DET:imgsz {yolo_primary_imgsz}"
        if yolo_thermal_model is None:
            det_line += " lazy"
        det_color = (0, 255, 220)
    else:
        det_line = "DET:RGB"
        det_color = (0, 255, 220)
    return det_line, det_color


def _gun_key_hint_parts():
    """Compact key-hint segments for status HUD row."""
    _white = (255, 255, 255)
    _gray = (180, 180, 180)
    parts = [
        (f"[{KEY_WIZARD}]คาลิเบรต", (0, 220, 255)),
        (f"[{KEY_SETTINGS}]ตั้งค่า", (0, 220, 255)),
        ("[M]map", _white),
        (f"[{KEY_PXDEG_CALIB}]px/deg", _white),
        ("[P]rearm", _white),
        ("[H]home", _white),
        ("[J]jog", _white),
        ("[I]sim", _white),
        (f"[{KEY_ADJUST_CENTER}]center", _white),
        (f"[{KEY_QUIT}]quit", _white),
    ]
    try:
        import config as _cfg_hint
        _btn_cc = int(getattr(_cfg_hint, "CAM4_ARM_JOYSTICK_BUTTON_CLASS_CYCLE", 8))
        _btn_det = int(getattr(_cfg_hint, "CAM4_ARM_JOYSTICK_BUTTON_DETECTION_TOGGLE", 10))
        _det_mode = str(getattr(_cfg_hint, "CAM4_ARM_YOLO_DETECTION_MODE", "multiclass")).strip().lower()
        _class_cycle = bool(getattr(_cfg_hint, "CAM4_ARM_YOLO_ENABLE_CLASS_CYCLE", True))
    except Exception:
        _btn_cc = 8
        _btn_det = 10
        _det_mode = "multiclass"
        _class_cycle = True
    if _det_mode != "drone_only" and _class_cycle:
        parts.append((f"[{KEY_CLASS_PREV}/{KEY_CLASS_NEXT}]class", _gray))
        parts.append((f"Btn{_btn_cc} class", _gray))
    if _det_mode == "drone_only":
        parts.append((f"Btn{_btn_det}/[T] 640/1280", _gray))
    else:
        parts.append((f"Btn{_btn_det}/[T] RGB-Therm", _gray))
    return parts


def _arm_hud_label_and_color(arm_controller):
    """Return (label, BGR) for arm GRBL link + motion responsiveness."""
    if arm_controller is None:
        return "ARM:off", (120, 120, 120)
    if getattr(arm_controller, "is_simulation_mode", False):
        return "ARM:SIM", (0, 200, 255)
    if getattr(arm_controller, "_reconnect_in_progress", False):
        return "ARM:BUSY", (0, 200, 255)
    healthy = bool(getattr(arm_controller, "is_healthy", True))
    motion_ok = bool(getattr(arm_controller, "motion_verified", True))
    if hasattr(arm_controller, "_serial_is_open"):
        serial_open = bool(arm_controller._serial_is_open())
    else:
        ser = getattr(arm_controller, "ser", None)
        serial_open = ser is not None and getattr(ser, "is_open", False)
    probe_streak = int(getattr(arm_controller, "_idle_probe_fail_streak", 0))
    homing_fault = getattr(arm_controller, "_homing_fault", None)
    _cfg = _config_mod
    _max_recon = int(getattr(_cfg, "CAM4_ARM_RECONNECT_MAX_ATTEMPTS", 5)) if _cfg else 5
    _fail_n = int(getattr(arm_controller, "_reconnect_fail_count", 0))
    if (
        not getattr(arm_controller, "is_simulation_mode", False)
        and _fail_n >= _max_recon
        and not (healthy and serial_open and motion_ok)
    ):
        return "ARM:HOLD", (0, 140, 255)
    if homing_fault == "no_motion" and serial_open:
        return "ARM:NOPWR", (0, 80, 255)
    if homing_fault == "homing_fail":
        return "ARM:HOMEFAIL", (0, 60, 255)
    if healthy and serial_open and motion_ok and probe_streak >= 1:
        return "ARM:WARN", (0, 200, 120)
    if healthy and serial_open and motion_ok:
        return "ARM:OK", (0, 220, 0)
    if healthy and serial_open and not motion_ok:
        return "ARM:STALL", (0, 80, 255)
    if not serial_open:
        return "ARM:DISC", (0, 100, 255)
    return "ARM:ERR", (0, 80, 255)


def _manual_arm_reconnect(arm_controller) -> None:
    """Force full arm homing reconnect (reset auto-retry counter). Non-blocking."""
    if arm_controller is None:
        return
    if getattr(arm_controller, "is_simulation_mode", False):
        print("[ARM] reconnect skipped (simulation mode)")
        return
    arm_controller._reconnect_fail_count = 0
    if hasattr(arm_controller, "_reset_stall_tracking"):
        arm_controller._reset_stall_tracking()
    if hasattr(arm_controller, "clear_homing_fault"):
        arm_controller.clear_homing_fault()
    elif hasattr(arm_controller, "_homing_fault"):
        arm_controller._homing_fault = None
    if hasattr(arm_controller, "request_reconnect"):
        if arm_controller.request_reconnect("manual-P"):
            print("[ARM] reconnect started in background (ARM:BUSY until homed)")
        else:
            print("[ARM] reconnect already in progress")
    elif hasattr(arm_controller, "try_reconnect"):
        arm_controller.try_reconnect()


def _manual_arm_go_home(arm_controller, arm_mode_manager=None) -> None:
    """สั่งแขนกลับตำแหน่ง home (0,0) ทันที — G0 ธรรมดา ไม่ re-home (เบา/เร็วกว่าปุ่ม P).

    บังคับ MODE:SAFE ก่อน เพื่อไม่ให้ AUTO/LOCK/จอยแย่งขับระหว่างวิ่งกลับ home.
    รันใน background thread (blocking=True) เพื่อให้ move เสร็จจริงโดยไม่ค้าง UI loop.
    """
    if arm_controller is None:
        return
    if arm_mode_manager is not None:
        try:
            arm_mode_manager.set_mode(MODE_SAFE)
        except Exception:
            pass

    def _worker():
        try:
            arm_controller.go_home(blocking=True)
            print("[ARM] go home (H) — แขนกลับตำแหน่ง home แล้ว")
        except Exception as e:
            print(f"[ARM] go home failed: {e}")

    threading.Thread(target=_worker, daemon=True, name="arm-go-home").start()


def _arm_jog_unlock(arm_controller) -> None:
    """ปลด GRBL alarm/lock ($X) + ตั้ง absolute mode เพื่อให้ jog ได้แม้แขนค้าง/STALL/SAFE.

    เลียนแบบ _fallback_manual_connect: $X ปลด alarm แล้วขยับด้วย G0 ได้ทันที.
    เคลียร์ homing fault + stall tracking กันไม่ให้ fault monitor วน re-home ระหว่าง jog.

    ⚠️ ใช้ wait_ok=False (ไม่รอ "ok") ด้วย 2 เหตุผล:
       1) ไม่บล็อก GUI loop (wait_ok=True บล็อกได้ถึง 3 วิ/คำสั่ง → จอค้าง → โดน watchdog kill)
       2) ไม่ถือ serial lock ค้าง — ถ้าถือค้าง jog move (move_relative blocking=False) จะ
          acquire lock ไม่ได้แล้ว drop เงียบ ๆ ทำให้ "แขนไม่ขยับ". คำสั่งถูกส่งตามลำดับ FIFO
          ทาง serial อยู่แล้ว GRBL จึงประมวลผล $X ก่อน G0 ของ jog เสมอ.
    """
    if arm_controller is None or getattr(arm_controller, "is_simulation_mode", False):
        return
    if not hasattr(arm_controller, "_send_gcode"):
        return
    try:
        arm_controller._send_gcode("$X", wait_ok=False)     # ปลด alarm lock (ไม่รอ ok)
        arm_controller._send_gcode("G90", wait_ok=False)    # absolute positioning
        if getattr(arm_controller, "grbl_units_mm", False):
            arm_controller._send_gcode("G21", wait_ok=False)
        if hasattr(arm_controller, "clear_homing_fault"):
            arm_controller.clear_homing_fault()
        if hasattr(arm_controller, "_reset_stall_tracking"):
            arm_controller._reset_stall_tracking()
        arm_controller.is_healthy = True
        print("[JOG] unlock ($X) — พร้อม jog ด้วยลูกศร/WASD")
    except Exception as e:
        print(f"[JOG] unlock failed: {e}")


def _arm_jog_step(arm_controller, dpan_deg: float, dtilt_deg: float) -> None:
    """ขยับแขนทีละ step (relative) จากปุ่มคีย์บอร์ด — ส่ง G0 ตรง ใช้ได้ทุกโหมด."""
    if arm_controller is None:
        return
    try:
        if hasattr(arm_controller, "touch_activity"):
            arm_controller.touch_activity()
        arm_controller.move_relative(float(dpan_deg), float(dtilt_deg), blocking=False)
    except Exception as e:
        print(f"[JOG] move failed: {e}")


def _arm_jog_set_home(arm_controller) -> None:
    """ตั้งตำแหน่งแขนปัจจุบันเป็น home ใหม่ (G92 X0 Y0) — 'รีเซต home' แบบ manual.

    สำหรับแขนที่ไม่มี limit switch (homing จริงไม่ได้): jog แขนไปตำแหน่งที่อยากให้เป็น home
    แล้วกดปุ่มนี้ → ตำแหน่งนั้นกลายเป็น (0,0). หลังจากนี้ปุ่ม H (go home) จะกลับมาที่นี่.
    """
    if arm_controller is None:
        return
    if not getattr(arm_controller, "is_simulation_mode", False) and hasattr(arm_controller, "_send_gcode"):
        try:
            arm_controller._send_gcode("G92 X0 Y0", wait_ok=False)
        except Exception as e:
            print(f"[JOG] set home failed: {e}")
            return
    arm_controller.pos_x = 0.0
    arm_controller.pos_y = 0.0
    arm_controller.target_x = 0.0
    arm_controller.target_y = 0.0
    if hasattr(arm_controller, "_manual_no_home"):
        arm_controller._manual_no_home = False   # ตั้ง home แล้ว = มี reference
    print("[JOG] set home here (G92 X0 Y0) — ตำแหน่งนี้เป็น home ใหม่แล้ว")


def _arm_mode_name(mode: int) -> str:
    return {MODE_AUTO: "AUTO", MODE_MANUAL: "MANUAL", MODE_LOCK: "LOCK", MODE_SAFE: "SAFE"}.get(
        mode, str(mode)
    )


def _camera_operator_lock_active(cam, network_loss_mode: bool) -> bool:
    """True when camera offline — operator must stay in MODE:SAFE."""
    _cfg = _config_mod
    if _cfg is not None and not getattr(_cfg, "CAM_FORCE_SAFE_ON_CAMERA_LOSS", True):
        return False
    if network_loss_mode:
        return True
    if cam is not None:
        try:
            if get_stream_health(cam) == "failed":
                return True
        except Exception:
            pass
    return False


def _tick_camera_loss_mode_safe(
    cam,
    network_loss_mode: bool,
    arm_mode_manager,
    mode_before_camera_loss,
    arm_controller=None,
):
    """
    Camera DISC/WAIT/FAIL → force MODE:SAFE until CAM:OK.
    Blocks mode restore while camera or arm still requires safe.
    """
    if arm_mode_manager is None:
        return mode_before_camera_loss
    if not _camera_operator_lock_active(cam, network_loss_mode):
        arm_needs_safe = False
        if arm_controller is not None and hasattr(arm_controller, "arm_requires_safe_mode"):
            arm_needs_safe = bool(arm_controller.arm_requires_safe_mode())
        if (
            mode_before_camera_loss is not None
            and not arm_needs_safe
            and not getattr(arm_controller, "_reconnect_in_progress", False)
        ):
            restore = mode_before_camera_loss
            mode_before_camera_loss = None
            arm_mode_manager.set_mode(restore)
            print(f"[CAM] stream OK — restored operator mode → {_arm_mode_name(restore)}")
        return mode_before_camera_loss

    if arm_mode_manager.mode != MODE_SAFE and mode_before_camera_loss is None:
        mode_before_camera_loss = arm_mode_manager.mode
        print(
            f"[CAM] camera offline — MODE:SAFE "
            f"(saved {_arm_mode_name(mode_before_camera_loss)})"
        )
    arm_mode_manager.set_mode(MODE_SAFE)
    return mode_before_camera_loss


def _tick_arm_mode_fault_and_restore(
    arm_controller,
    arm_mode_manager,
    mode_before_arm_fault,
    *,
    camera_operator_lock: bool = False,
):
    """
    Save operator mode before arm fault (incl. ARM:STALL) → force SAFE.
    Restore saved mode only when arm_link_ok() (full ARM:OK after home/verify).
    """
    if arm_controller is None or arm_mode_manager is None:
        return mode_before_arm_fault
    if getattr(arm_controller, "is_simulation_mode", False):
        return mode_before_arm_fault

    needs_safe = (
        arm_controller.arm_requires_safe_mode()
        if hasattr(arm_controller, "arm_requires_safe_mode")
        else not (
            hasattr(arm_controller, "arm_link_ok") and arm_controller.arm_link_ok()
        )
    )

    if needs_safe:
        if arm_mode_manager.mode != MODE_SAFE and mode_before_arm_fault is None:
            mode_before_arm_fault = arm_mode_manager.mode
        arm_mode_manager.set_mode(MODE_SAFE)
        return mode_before_arm_fault

    link_ok = (
        arm_controller.arm_link_ok()
        if hasattr(arm_controller, "arm_link_ok")
        else False
    )
    if (
        mode_before_arm_fault is not None
        and link_ok
        and not getattr(arm_controller, "_reconnect_in_progress", False)
        and not camera_operator_lock
    ):
        restore = mode_before_arm_fault
        mode_before_arm_fault = None
        arm_mode_manager.set_mode(restore)
        print(f"[ARM] restored operator mode → {_arm_mode_name(restore)}")

    return mode_before_arm_fault


def _joy_hud_label_and_color(joystick_reader):
    """Return (label, BGR) for joystick link status."""
    if joystick_reader is None:
        return "JOY:off", (120, 120, 120)
    if getattr(joystick_reader, "_reconnect_in_progress", False) or getattr(
        joystick_reader, "_pending_probe", False
    ):
        return "JOY:WAIT", (0, 200, 255)
    if getattr(joystick_reader, "enabled", False):
        return "JOY:OK", (0, 220, 0)
    if getattr(joystick_reader, "pygame_available", True):
        return "JOY:DISC", (0, 100, 255)
    return "JOY:off", (120, 120, 120)


def _gun_status_hud_parts(ctx):
    """Build status row segments (mode, AUTO diag, WS/EFF, key hints)."""
    parts = []
    mode_label = ctx.get("mode_label")
    if mode_label:
        parts.append((mode_label, ctx.get("mode_color", (255, 255, 255))))
    if ctx.get("arm_label"):
        parts.append((ctx["arm_label"], ctx.get("arm_color", (255, 255, 255))))
    if ctx.get("cam_label"):
        parts.append((ctx["cam_label"], ctx.get("cam_color", (255, 255, 255))))
    if ctx.get("show_auto_diag"):
        parts.append((ctx["src_label"], ctx["src_color"]))
        parts.append((ctx["cue_label"], ctx["cue_color"]))
        parts.append((ctx["cal_label"], ctx["cal_color"]))
        parts.append((ctx["bias_label"], ctx["bias_color"]))
    if ctx.get("ws_label"):
        parts.append((ctx["ws_label"], ctx["ws_color"]))
    if ctx.get("eff_part"):
        parts.append((ctx["eff_part"], ctx["eff_color"]))
    parts.extend(_gun_key_hint_parts())
    return parts


def _mapping_key_hint_parts():
    """Key hints for cam8 mapping mode status row."""
    _white = (255, 255, 255)
    _gray = (180, 180, 180)
    return [
        ("[H]ref", _white),
        ("[P]rearm", _white),
        ("[C]cell", _white),
        ("[S]JSON", _white),
        ("[M]exit", _gray),
    ]


def _pxdeg_key_hint_parts():
    """Key hints for embedded px/deg calibrator mode status row."""
    _white = (255, 255, 255)
    _gray = (180, 180, 180)
    return [
        ("click", _gray),
        ("[Enter]move", _white),
        ("[Joy]fine", _white),
        ("[Btn0]ok", _white),
        ("[H]log+home", _white),
        ("[R]recal", _white),
        ("[S]save", _white),
        ("[P]rearm", _white),
        (f"[{KEY_PXDEG_CALIB}]exit", _gray),
    ]


def _mapping_status_hud_parts(ctx):
    """Build status row for cam8 mapping mode."""
    parts = [
        ("MAPPING", (0, 220, 255)),
    ]
    if ctx.get("arm_label"):
        parts.append((ctx["arm_label"], ctx.get("arm_color", (255, 255, 255))))
    if ctx.get("cam_label"):
        parts.append((ctx["cam_label"], ctx.get("cam_color", (255, 255, 255))))
    if ctx.get("map_label"):
        parts.append((ctx["map_label"], ctx.get("map_color", (0, 200, 255))))
    if ctx.get("ref_label"):
        parts.append((ctx["ref_label"], ctx.get("ref_color", (200, 200, 200))))
    if ctx.get("cam8_label"):
        parts.append((ctx["cam8_label"], ctx.get("cam8_color", (200, 200, 200))))
    parts.extend(_mapping_key_hint_parts())
    return parts


def _pxdeg_status_hud_parts(ctx):
    """Build status row for px/deg calibrator mode."""
    parts = [
        ("PXDEG", (0, 220, 255)),
    ]
    if ctx.get("arm_label"):
        parts.append((ctx["arm_label"], ctx.get("arm_color", (255, 255, 255))))
    if ctx.get("cam_label"):
        parts.append((ctx["cam_label"], ctx.get("cam_color", (255, 255, 255))))
    if ctx.get("px_label"):
        parts.append((ctx["px_label"], ctx.get("px_color", (0, 200, 255))))
    step = ctx.get("step_label")
    if step:
        parts.append((step, ctx.get("step_color", (200, 200, 200))))
    parts.extend(_pxdeg_key_hint_parts())
    return parts


def _cam8_stream_hud_label(has_signal, status_msg=""):
    """Return (label, BGR) for cam8 live stream in mapping mode."""
    if has_signal:
        return "CAM8:OK", (0, 220, 0)
    msg = (status_msg or "").lower()
    if "connect" in msg:
        return "CAM8:WAIT", (0, 200, 255)
    if "retry" in msg or "disconnect" in msg or "off" in msg:
        return "CAM8:DISC", (0, 100, 255)
    return "CAM8:ERR", (0, 80, 255)


def _draw_two_row_bottom_hud(frame, metrics, status_parts, status_sample):
    """Shared two-row bottom HUD: metrics row (bottom) + colored status segments (above)."""
    h, w = frame.shape[:2]
    ui_scale = max(0.3, min(1.5, min(h, w) / 1080.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    hud_bar_height = max(40, int(110 * ui_scale))
    y_pos = h - max(6, int(20 * ui_scale))
    x_start = max(4, int(20 * ui_scale))
    margin_right = max(20, int(40 * ui_scale))
    available_width = w - x_start - margin_right
    spacing = max(3, int(28 * ui_scale))
    font_scale = max(0.22, round(1.0 * ui_scale, 2))
    thickness = max(1, int(round(2 * ui_scale)))
    min_font_scale = 0.18

    while font_scale >= min_font_scale:
        total_w = 0
        n_slots = 0
        for text, _col in metrics:
            if not text:
                continue
            total_w += cv2.getTextSize(str(text), font, font_scale, thickness)[0][0]
            n_slots += 1
        if n_slots > 1:
            total_w += (n_slots - 1) * spacing
        w_status = cv2.getTextSize(status_sample, font, font_scale, thickness)[0][0]
        if total_w <= available_width and w_status <= available_width:
            break
        font_scale = round(font_scale - 0.02, 2)
        if font_scale < min_font_scale:
            font_scale = min_font_scale
            break
    thickness = max(1, int(round(2 * font_scale)))

    cv2.rectangle(frame, (0, h - hud_bar_height), (w, h), (0, 0, 0), -1)

    (_, _lh_tmp), _lb_tmp = cv2.getTextSize("Ay", font, font_scale, thickness)
    y_status = y_pos - (_lh_tmp + _lb_tmp) - max(4, int(8 * ui_scale))

    x = x_start
    first = True
    for text, color in metrics:
        if not text:
            continue
        chunk = str(text)
        if not first:
            x += spacing
        first = False
        cv2.putText(frame, chunk, (x, y_pos), font, font_scale, color, thickness)
        x += cv2.getTextSize(chunk, font, font_scale, thickness)[0][0]

    _draw_hud_segments_row(
        frame,
        x_start,
        y_status,
        font,
        font_scale,
        thickness,
        status_parts,
        x_start + available_width,
    )


def draw_gun_bottom_hud(frame, ctx):
    """Two-row bottom HUD bar (cam8 style): status row + metrics row."""
    fps_text = f"FPS:{ctx['fps']:.1f}"
    lat_text = f"LAT:{ctx['lat_ms']:.0f}ms" if ctx.get("has_lat") else "LAT:---"
    conf_text = f"CONF:{float(ctx.get('conf', 0.0)):.2f}(+/-)"
    joy_text = ctx.get("joy_label", "JOY:off")
    joy_color = ctx.get("joy_color", (120, 120, 120))
    spd_text = ctx.get("spd_label", "SPD:---")
    spd_color = ctx.get("spd_color", (120, 120, 120))
    cls_text = f"CLASS:{ctx['class_name']}" if ctx.get("class_name") else ""
    det_line, det_color = _gun_detection_line_and_color(ctx)

    metrics = [
        (fps_text, (255, 255, 255)),
        (lat_text, (200, 200, 200)),
        (conf_text, (200, 200, 200)),
        (joy_text, joy_color),
        (spd_text, spd_color),
    ]
    if cls_text:
        metrics.append((cls_text, (0, 255, 220)))
    metrics.append((det_line, det_color))

    status_sample = (
        "AUTO | ARM:DISC | SRC:cam8 | CUE:999ms | CAL8:OK | BIAS:ON(B) 99c | WS:ERR | EFF:acquiring | "
        "[C]center [S]ballistics [Q]quit"
    )
    _draw_two_row_bottom_hud(frame, metrics, _gun_status_hud_parts(ctx), status_sample)


def draw_mapping_bottom_hud(frame, ctx):
    """Two-row bottom HUD for cam8 mapping mode (same layout as main screen)."""
    pan = float(ctx.get("pan", 0.0))
    tilt = float(ctx.get("tilt", 0.0))
    metrics = [
        (f"FPS:{ctx['fps']:.1f}", (255, 255, 255)),
        (f"PAN:{pan:+.1f}", (200, 200, 200)),
        (f"TILT:{tilt:+.1f}", (200, 200, 200)),
        (ctx.get("joy_label", "JOY:off"), ctx.get("joy_color", (120, 120, 120))),
        (ctx.get("spd_label", "SPD:---"), ctx.get("spd_color", (120, 120, 120))),
    ]
    cell_label = ctx.get("cell_label")
    if cell_label:
        metrics.append((cell_label, ctx.get("cell_color", (0, 255, 180))))

    status_sample = (
        "MAPPING | ARM:OK | MAP:999/999 | REF:OK | CAM8:OK | "
        "[H]ref [P]rearm [C]cell [S]JSON [M]exit"
    )
    _draw_two_row_bottom_hud(frame, metrics, _mapping_status_hud_parts(ctx), status_sample)


def draw_pxdeg_bottom_hud(frame, ctx):
    """Two-row bottom HUD for embedded px/deg calibrator (same layout as mapping)."""
    pan = float(ctx.get("pan", 0.0))
    tilt = float(ctx.get("tilt", 0.0))
    metrics = [
        (f"FPS:{ctx['fps']:.1f}", (255, 255, 255)),
        (f"PAN:{pan:+.1f}", (200, 200, 200)),
        (f"TILT:{tilt:+.1f}", (200, 200, 200)),
        (ctx.get("joy_label", "JOY:off"), ctx.get("joy_color", (120, 120, 120))),
        (ctx.get("spd_label", "SPD:---"), ctx.get("spd_color", (120, 120, 120))),
    ]
    px_short = ctx.get("px_short")
    if px_short:
        metrics.append((px_short, ctx.get("px_short_color", (0, 220, 255))))

    status_sample = (
        "PXDEG | ARM:OK | CAM:OK | PX:85,-89 | click | "
        "[Enter]move [H]log+home [R]recal [S]save [P]rearm [G]exit"
    )
    _draw_two_row_bottom_hud(frame, metrics, _pxdeg_status_hud_parts(ctx), status_sample)


def draw_hud(
    frame,
    cx_frame,
    cy_frame,
    radius_px,
    target_det,
    ready_to_fire,
    guide_h=None,
    guide_v=None,
    is_day=None,
    distance_m=None,
    all_detections=None,
    active_class_name=None,
    active_class_idx=None,
):
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
        x_t, y_t, w_d, h_d, _ = _det_xywhc(target_det)
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

    # 3. bbox อื่น (ถ้ามี) — เส้นบาง สีเทา + ป้ายชื่อ class
    color_other = (128, 128, 128)
    font_small = cv2.FONT_HERSHEY_SIMPLEX
    if all_detections:
        for det in all_detections:
            if target_det is not None and _det_eq(det, target_det):
                continue
            xo, yo, wo, ho, oconf = _det_xywhc(det)
            oid, oname = _det_cls(det)
            cv2.rectangle(frame, (xo, yo), (xo + wo, yo + ho), color_other, 1)
            cls_disp = _hud_class_display_name(oname)
            opct = min(99, max(0, int(round(oconf * 100))))
            tag = f"{cls_disp} {opct}%"
            scale_s = 0.5
            thk = 1
            (tw_s, th_txt), _ = cv2.getTextSize(tag, font_small, scale_s, thk)
            ty_s = max(th_txt + 2, yo - 4)
            tx_s = max(0, min(w - tw_s - 2, xo))
            cv2.rectangle(frame, (tx_s, ty_s - th_txt - 2), (tx_s + tw_s + 4, ty_s + 2), (0, 0, 0), -1)
            lbl_color = (200, 255, 200) if (active_class_idx is not None and oid == int(active_class_idx)) else (220, 220, 220)
            cv2.putText(frame, tag, (tx_s + 2, ty_s), font_small, scale_s, lbl_color, thk)

    # 4. เป้าหมายหลัก (conf สูงสุด) — bbox หนา ป้ายใหญ่ ตัวหนังสือเด่น
    if target_det is not None:
        x, y, w_box, h_box, conf = _det_xywhc(target_det)
        _cls_id, _cls_name = _det_cls(target_det)
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

        # ป้ายเหนือ bbox: ชื่อ class + confidence
        conf_pct = min(99, max(0, int(round(conf * 100))))
        label_drone = f"{_hud_class_display_name(_cls_name)} {conf_pct}%"
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
            #cv2.putText(frame, dist_text, (x, dy_bottom), font, scale_dist, (0, 0, 0), th_dist + 2)
            #cv2.putText(frame, dist_text, (x, dy_bottom), font, scale_dist, color_text_hud, th_dist)

    # 5. READY / THINGS — มุมขวาบน (สไตล์ cam8)
    s = max(0.3, min(1.5, min_side / 1080.0))
    label = "READY" if ready_to_fire else "THINGS"
    font_things = cv2.FONT_HERSHEY_SIMPLEX
    scale_label = 2.0 * s
    th_label = max(1, int(3 * s))
    if ready_to_fire:
        color_text = (255, 255, 255) if (color_use == (0, 0, 0)) else color_use
    else:
        color_text = (255, 128, 0)  # BGR bluish
    (tw_label, _), _ = cv2.getTextSize(label, font_things, scale_label, th_label)
    x_label = w - tw_label - int(20 * s)
    y_label = int(50 * s)
    cv2.putText(frame, label, (x_label, y_label), font_things, scale_label, color_text, th_label)

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
    """Legacy stub — key hints now live in draw_gun_bottom_hud status row."""
    del frame, w, h


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


# =============================================================================
# Mode tick functions (AUTO / LOCK / MANUAL / SAFE)
# =============================================================================
def _tick_safe(ctx):
    """SAFE mode: ไม่ส่งคำสั่งแขน."""
    pass


def _tick_manual(ctx):
    """MANUAL mode: ควบคุมแขนด้วยจอยสติ๊กเท่านั้น."""
    joystick_mapper = ctx.get("joystick_mapper")
    js_state = ctx.get("js_state")
    dt_loop = ctx.get("dt_loop", 0.033)
    if joystick_mapper is not None and js_state is not None:
        joystick_mapper.apply(js_state, dt_loop)


# =============================================================================
# Online Residual Bias Learner (AUTO-only)
# =============================================================================
# Config — prefix CAM8_AUTO_ONLINE_BIAS_
CAM8_AUTO_ONLINE_BIAS_ENABLED: bool = False      # ปิดไว้จน test ครบ; เปิดใน config หรือแก้ที่นี่
_BIAS_FILE = Path(__file__).resolve().parent / "calibration_data" / "cam8_auto_residual_bias.json"
_BIAS_VERSION = 1
_BIAS_ALPHA_DEFAULT = 0.03           # EMA learning rate per gated sample
_BIAS_B_PAN_MAX_DEG = 2.0            # clamp per cell
_BIAS_B_TILT_MAX_DEG = 2.0
_BIAS_R_MAX_FRAC = 0.08              # gate: ≤8% of min(frame_w, frame_h) from crosshair
_BIAS_SETTLE_FRAMES = 8              # รอ N เฟรมก่อนวัด residual
_BIAS_SETTLE_THRESH_PX = 2.0         # residual เปลี่ยน <2px ต่อเนื่อง 5 เฟรม = นิ่ง
_BIAS_SETTLE_STABLE_FRAMES = 5
_BIAS_SAVE_INTERVAL_S = 90.0         # save ทุก 90 วิ (ถ้ามีการเปลี่ยนแปลง)
_BIAS_RATE_LIMIT_S = 5.0             # ต่ำสุด 5 วิ/ครั้ง ต่อ cell
_BIAS_EMERGENCY_THRESH_DEG = 3.0     # หยุด update ถ้า residual หลัง compensate >3°
_BIAS_GLOBAL_CAP_DEG = 2.0           # cap bias sum ก่อนใส่คำสั่ง


class ResidualBiasLearner:
    """
    AUTO-only online learning of per-cell residual bias.
    เรียน bias ต่อ cell (แมปจาก cam8 cue pixel → cell ตาม coarse grid ของ calibration).
    เก็บ EMA(bias_pan, bias_tilt) ต่อ cell และ persist ลงไฟล์ JSON.

    State machine:
      IDLE → COARSE (fresh cue + AUTO) → SETTLE (move issued) → MEASURE → GATE/UPDATE → IDLE
    """
    _STATE_IDLE = "idle"
    _STATE_COARSE = "coarse"
    _STATE_SETTLE = "settle"
    _STATE_MEASURE = "measure"

    def __init__(self, calib_json_path=None):
        self._enabled = CAM8_AUTO_ONLINE_BIAS_ENABLED
        self._state = self._STATE_IDLE
        self._cells = {}          # key "i_j" -> {"bias_pan":f, "bias_tilt":f, "n":int, "mse_pan":f, "last_update_t":f}
        self._grid_rows = 5
        self._grid_cols = 9
        self._alpha = _BIAS_ALPHA_DEFAULT
        self._dirty = False
        self._last_save_t = time.time()
        self._settle_count = 0
        self._settle_stable_count = 0
        self._prev_residual_px = None
        self._locked_cue_cx = None
        self._locked_cue_cy = None
        self._locked_cue_fw = None
        self._locked_cue_fh = None
        self._last_cell_update_t = {}  # cell_key -> timestamp

        # load grid dims from calib file
        if calib_json_path is not None:
            try:
                with open(calib_json_path, "r") as f:
                    _d = json.load(f)
                self._grid_rows = int(_d.get("grid_rows", self._grid_rows))
                self._grid_cols = int(_d.get("grid_cols", self._grid_cols))
            except Exception:
                pass

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        bak = str(_BIAS_FILE) + ".bak"
        for path in [str(_BIAS_FILE), bak]:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if data.get("version") != _BIAS_VERSION:
                    break
                self._grid_rows = int(data.get("grid_rows", self._grid_rows))
                self._grid_cols = int(data.get("grid_cols", self._grid_cols))
                self._alpha = float(data.get("meta", {}).get("alpha", _BIAS_ALPHA_DEFAULT))
                raw = data.get("cells", {})
                self._cells = {}
                for k, v in raw.items():
                    self._cells[k] = {
                        "bias_pan": float(v.get("bias_pan", 0.0)),
                        "bias_tilt": float(v.get("bias_tilt", 0.0)),
                        "n": int(v.get("n", 0)),
                        "mse_pan": float(v.get("mse_pan", 0.0)),
                        "last_update_t": float(v.get("last_update_t", 0.0)),
                    }
                print(f"[BiasLearner] Loaded {len(self._cells)} cells from {path}")
                return
            except Exception:
                continue
        self._cells = {}

    def save(self, force=False):
        if not self._dirty and not force:
            return
        now = time.time()
        if not force and (now - self._last_save_t) < _BIAS_SAVE_INTERVAL_S:
            return
        data = {
            "version": _BIAS_VERSION,
            "grid_rows": self._grid_rows,
            "grid_cols": self._grid_cols,
            "reference": "cam8_cue_px_cell",
            "cells": self._cells,
            "meta": {"alpha": self._alpha, "updated_at": now},
        }
        tmp = str(_BIAS_FILE) + ".tmp"
        bak = str(_BIAS_FILE) + ".bak"
        try:
            _BIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            if _BIAS_FILE.exists():
                import shutil as _shutil
                _shutil.copy2(str(_BIAS_FILE), bak)
            os.replace(tmp, str(_BIAS_FILE))
            self._dirty = False
            self._last_save_t = now
        except Exception as e:
            print(f"[BiasLearner] save error: {e}")

    # ------------------------------------------------------------------
    # Cell helpers
    # ------------------------------------------------------------------
    def _cell_key(self, cx, cy, fw, fh):
        j = int(max(0, min(self._grid_cols - 1, int((cx / fw) * self._grid_cols))))
        i = int(max(0, min(self._grid_rows - 1, int((cy / fh) * self._grid_rows))))
        return f"{i}_{j}"

    def _get_bias(self, cell_key):
        c = self._cells.get(cell_key)
        if c is None:
            return 0.0, 0.0
        return float(c["bias_pan"]), float(c["bias_tilt"])

    def _update_cell(self, cell_key, d_pan_deg, d_tilt_deg):
        now = time.time()
        last_t = self._last_cell_update_t.get(cell_key, 0.0)
        if now - last_t < _BIAS_RATE_LIMIT_S:
            return
        if cell_key not in self._cells:
            self._cells[cell_key] = {"bias_pan": 0.0, "bias_tilt": 0.0, "n": 0, "mse_pan": 0.0, "last_update_t": 0.0}
        c = self._cells[cell_key]
        alpha = self._alpha
        new_pan = (1.0 - alpha) * c["bias_pan"] + alpha * d_pan_deg
        new_tilt = (1.0 - alpha) * c["bias_tilt"] + alpha * d_tilt_deg
        # clamp per cell
        new_pan = max(-_BIAS_B_PAN_MAX_DEG, min(_BIAS_B_PAN_MAX_DEG, new_pan))
        new_tilt = max(-_BIAS_B_TILT_MAX_DEG, min(_BIAS_B_TILT_MAX_DEG, new_tilt))
        c["bias_pan"] = new_pan
        c["bias_tilt"] = new_tilt
        c["n"] = c.get("n", 0) + 1
        err_sq = d_pan_deg ** 2
        c["mse_pan"] = (1.0 - alpha) * c.get("mse_pan", 0.0) + alpha * err_sq
        c["last_update_t"] = now
        self._last_cell_update_t[cell_key] = now
        self._dirty = True

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def get_bias(self, cue_dict):
        """คืน (bias_pan_deg, bias_tilt_deg) ที่ควร offset เข้าไปใน target ก่อนสั่งแขน."""
        if not self._enabled or cue_dict is None:
            return 0.0, 0.0
        cx = cue_dict.get("cx")
        cy = cue_dict.get("cy")
        fw = cue_dict.get("frame_w")
        fh = cue_dict.get("frame_h")
        if None in (cx, cy, fw, fh) or fw <= 0 or fh <= 0:
            return 0.0, 0.0
        key = self._cell_key(float(cx), float(cy), float(fw), float(fh))
        bp, bt = self._get_bias(key)
        # global cap
        total = math.hypot(bp, bt)
        if total > _BIAS_GLOBAL_CAP_DEG:
            scale = _BIAS_GLOBAL_CAP_DEG / total
            bp *= scale
            bt *= scale
        return bp, bt

    def tick(self, arm_mode, cue_dict, target_det, cam4_w, cam4_h,
             cx_frame, cy_frame, px_per_deg_x, px_per_deg_y,
             move_issued_this_frame):
        """
        เรียกทุกเฟรมใน AUTO mode เพื่ออัปเดต state machine และ bias.
        move_issued_this_frame: True เมื่อแขนถูกสั่งเคลื่อนที่เฟรมนี้.
        """
        if not self._enabled:
            self.save()
            return

        from joystick_cam4_controller import MODE_AUTO
        is_auto = (arm_mode == MODE_AUTO)

        if not is_auto or cue_dict is None:
            self._state = self._STATE_IDLE
            self._settle_count = 0
            self._settle_stable_count = 0
            self._prev_residual_px = None
            self.save()
            return

        # IDLE → COARSE when AUTO + fresh cue
        if self._state == self._STATE_IDLE:
            self._state = self._STATE_COARSE
            self._locked_cue_cx = cue_dict.get("cx")
            self._locked_cue_cy = cue_dict.get("cy")
            self._locked_cue_fw = cue_dict.get("frame_w")
            self._locked_cue_fh = cue_dict.get("frame_h")
            self._settle_count = 0
            self._settle_stable_count = 0
            self._prev_residual_px = None

        # COARSE → SETTLE when move issued
        if self._state == self._STATE_COARSE and move_issued_this_frame:
            self._state = self._STATE_SETTLE
            self._settle_count = 0
            self._settle_stable_count = 0
            self._prev_residual_px = None

        # SETTLE: wait for arm to stop moving + residual stable
        if self._state == self._STATE_SETTLE:
            self._settle_count += 1
            if target_det is not None and cam4_w > 0 and cam4_h > 0:
                dx, dy = float(target_det[0] + target_det[2] / 2) - cx_frame, float(target_det[1] + target_det[3] / 2) - cy_frame
                res_px = math.hypot(dx, dy)
                if self._prev_residual_px is not None and abs(res_px - self._prev_residual_px) < _BIAS_SETTLE_THRESH_PX:
                    self._settle_stable_count += 1
                else:
                    self._settle_stable_count = 0
                self._prev_residual_px = res_px
            if self._settle_count >= _BIAS_SETTLE_FRAMES and self._settle_stable_count >= _BIAS_SETTLE_STABLE_FRAMES:
                self._state = self._STATE_MEASURE

        # MEASURE → gate → update
        if self._state == self._STATE_MEASURE:
            if target_det is None or cam4_w <= 0 or cam4_h <= 0:
                self._state = self._STATE_IDLE
                self.save()
                return

            det_cx = float(target_det[0] + target_det[2] / 2)
            det_cy = float(target_det[1] + target_det[3] / 2)
            dx_px = det_cx - cx_frame
            dy_px = det_cy - cy_frame
            dist_px = math.hypot(dx_px, dy_px)
            r_max = _BIAS_R_MAX_FRAC * float(min(cam4_w, cam4_h))

            # gate: target must be within R_MAX of crosshair
            if dist_px > r_max:
                self._state = self._STATE_IDLE
                self.save()
                return

            # convert px → degrees — ppd_y เป็นค่าลบตาม convention (แกน y ภาพกลับทิศกับ tilt)
            # ห้ามเช็ก > 0: ค่าจาก calibration เช่น cam4 = (87.1, -89.7)
            if (px_per_deg_x is not None and abs(px_per_deg_x) > 1e-6
                    and px_per_deg_y is not None and abs(px_per_deg_y) > 1e-6):
                d_pan_deg = dx_px / px_per_deg_x
                d_tilt_deg = dy_px / px_per_deg_y
            else:
                self._state = self._STATE_IDLE
                return

            # gate: bbox size matching (if enabled and cue has norm size)
            if CAM8_CUE_SIZE_MATCH_ENABLED:
                cue_size = _cam8_cue_size_norm(cue_dict)
                if cue_size is not None and cam4_w > 0 and cam4_h > 0:
                    penalty_norm = (abs(float(target_det[2]) / cam4_w - cue_size[0])
                                    + abs(float(target_det[3]) / cam4_h - cue_size[1]))
                    if penalty_norm > CAM8_CUE_SIZE_MAX_TOL:
                        self._state = self._STATE_IDLE
                        self.save()
                        return

            # emergency: residual after compensation still too large
            if math.hypot(d_pan_deg, d_tilt_deg) > _BIAS_EMERGENCY_THRESH_DEG:
                self._state = self._STATE_IDLE
                self.save()
                return

            # update cell
            fw = self._locked_cue_fw or cue_dict.get("frame_w")
            fh = self._locked_cue_fh or cue_dict.get("frame_h")
            cx8 = self._locked_cue_cx or cue_dict.get("cx")
            cy8 = self._locked_cue_cy or cue_dict.get("cy")
            if None not in (fw, fh, cx8, cy8) and fw > 0 and fh > 0:
                key = self._cell_key(float(cx8), float(cy8), float(fw), float(fh))
                self._update_cell(key, d_pan_deg, d_tilt_deg)

            self._state = self._STATE_IDLE
            self.save()


# Module-level singleton; initialized when arm calib loads
_residual_bias_learner: Optional["ResidualBiasLearner"] = None


def _get_bias_learner() -> Optional["ResidualBiasLearner"]:
    return _residual_bias_learner


# =============================================================================
# FireTuneLearner — สอน 'aim trim' (pan/tilt offset) จากผลยิงจริง/sim ที่ผ่านมา
# แนวคิด: ทุกนัดที่ยิง เก็บ error เวกเตอร์ (ลำกล้อง − เป้า ณ กระสุนถึง) เทียบทิศเป้าวิ่ง
#   - นัดเป้านิ่ง (v เล็ก): error ทั้งเวกเตอร์ = static offset ตรง ๆ → 2 สมการ (แกน pan, แกน tilt)
#   - นัดเป้าวิ่ง: ใช้เฉพาะองค์ประกอบ cross-track (ตั้งฉากทิศวิ่ง) = static offset ที่ 'ไม่ปน lead error'
# กด 'y' (teach) → แก้สมการ least-squares หา offset ที่อธิบายทุกนัด แล้วปรับ trim (cap step กันเพี้ยน)
# trim ถูก 'บวกเข้า aim' ใน _tick_lock และ persist ที่ calibration_data/<cam>_fire_tune.json
# ปลอดภัย: ไม่แตะ muzzle/lead (ตาม scope ที่เลือก) — แก้แค่ offset เชิงเรขาคณิต (เหมือน crosshair trim)
# =============================================================================
FIRE_TUNE_MIN_SAMPLES = 5          # ต้องมีอย่างน้อยกี่นัดถึงจะยอมสอน
FIRE_TUNE_MOVING_VMAG_DEG_S = 2.0  # v เกินนี้ = เป้าวิ่ง (ใช้ cross เท่านั้น); ต่ำกว่า = ถือว่านิ่ง
FIRE_TUNE_MAX_STEP_DEG = 1.0       # ปรับ trim ได้สูงสุดกี่องศาต่อการกดสอน 1 ครั้ง (กันกระโดด)
FIRE_TUNE_MAX_TRIM_DEG = 5.0       # เพดาน trim สะสมรวม (safety clamp — ห้ามเลื่อนศูนย์เกินนี้)


class FireTuneLearner:
    def __init__(self, camera_name: str):
        cam = (camera_name or "cam4").strip()
        self._path = Path(__file__).resolve().parent / "calibration_data" / f"{cam}_fire_tune.json"
        self.trim_pan = 0.0
        self.trim_tilt = 0.0
        self.n_updates = 0
        self._samples = []   # list of (ax, ay, val): สมการ axis·offset = val
        self.load()

    def load(self):
        try:
            if self._path.is_file():
                with open(self._path) as f:
                    d = json.load(f)
                self.trim_pan = float(d.get("trim_pan", 0.0))
                self.trim_tilt = float(d.get("trim_tilt", 0.0))
                self.n_updates = int(d.get("n_updates", 0))
                print(f"[FireTune] Loaded trim=({self.trim_pan:+.2f},{self.trim_tilt:+.2f})deg "
                      f"from {self._path.name} (updates={self.n_updates})", flush=True)
        except Exception as e:
            print(f"[FireTune] load error: {e}")

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({
                    "trim_pan": self.trim_pan,
                    "trim_tilt": self.trim_tilt,
                    "n_updates": self.n_updates,
                }, f, indent=2)
        except Exception as e:
            print(f"[FireTune] save error: {e}")

    def get_trim(self) -> Tuple[float, float]:
        return self.trim_pan, self.trim_tilt

    def add_sample(self, err_pan: float, err_tilt: float,
                   v_pan: float, v_tilt: float, stable: bool = True) -> None:
        """เก็บ error 1 นัด. err = ลำกล้อง(ตอนยิง) − เป้า(ณ กระสุนถึง). stable=นัดที่ track นิ่งพอ."""
        if not stable:
            return
        vmag = math.hypot(v_pan, v_tilt)
        if vmag < FIRE_TUNE_MOVING_VMAG_DEG_S:
            # เป้านิ่ง → error ทั้งเวกเตอร์คือ static offset: 2 สมการ แกนตั้งฉากกัน
            self._samples.append((1.0, 0.0, err_pan))
            self._samples.append((0.0, 1.0, err_tilt))
        else:
            # เป้าวิ่ง → ใช้เฉพาะ cross-track (ตั้งฉากทิศวิ่ง) กันปน lead/TOF error
            ux, uy = v_pan / vmag, v_tilt / vmag
            ax, ay = -uy, ux                       # cross-axis (ซ้ายมือของทิศวิ่ง)
            val = err_pan * ax + err_tilt * ay
            self._samples.append((ax, ay, val))
        if len(self._samples) > 400:
            self._samples = self._samples[-400:]

    def sample_count(self) -> int:
        return len(self._samples)

    def teach(self) -> Optional[dict]:
        """แก้ least-squares: หา offset (ox,oy) ที่อธิบายทุกสมการ axis·offset=val ดีสุด
        แล้วปรับ trim -= offset (cap step). คืน dict สรุป หรือ None ถ้าข้อมูลไม่พอ/ไม่ observable."""
        n = len(self._samples)
        if n < FIRE_TUNE_MIN_SAMPLES:
            print(f"[FireTune] ยังสอนไม่ได้ — มี {n} สมการ ต้องการ ≥{FIRE_TUNE_MIN_SAMPLES} "
                  f"(ยิงในโหมด LOCK/sim เพิ่ม)", flush=True)
            return None
        A = np.zeros((2, 2)); b = np.zeros(2)
        for ax, ay, val in self._samples:
            v = np.array([ax, ay])
            A += np.outer(v, v)
            b += v * val
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-6:
            print("[FireTune] สอนไม่ได้ — ทิศเป้าไม่หลากพอ (offset ไม่ observable ทั้ง 2 แกน). "
                  "ลองยิงเป้าที่วิ่งหลายทิศ หรือมีนัดเป้านิ่งบ้าง", flush=True)
            return None
        ox = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
        oy = (A[0, 0] * b[1] - A[0, 1] * b[0]) / det
        # residual RMS ก่อนแก้ (ประเมินความสอดคล้องของข้อมูล)
        res = [ (ax * ox + ay * oy - val) for ax, ay, val in self._samples ]
        rms = float(np.sqrt(np.mean(np.square(res)))) if res else 0.0
        # แก้ trim ลบ residual offset ออก + cap ระยะก้าวต่อครั้ง
        d_pan, d_tilt = -ox, -oy
        dmag = math.hypot(d_pan, d_tilt)
        capped = False
        if dmag > FIRE_TUNE_MAX_STEP_DEG and dmag > 1e-9:
            s = FIRE_TUNE_MAX_STEP_DEG / dmag
            d_pan *= s; d_tilt *= s; capped = True
        new_pan = float(np.clip(self.trim_pan + d_pan, -FIRE_TUNE_MAX_TRIM_DEG, FIRE_TUNE_MAX_TRIM_DEG))
        new_tilt = float(np.clip(self.trim_tilt + d_tilt, -FIRE_TUNE_MAX_TRIM_DEG, FIRE_TUNE_MAX_TRIM_DEG))
        old = (self.trim_pan, self.trim_tilt)
        self.trim_pan, self.trim_tilt = new_pan, new_tilt
        self.n_updates += 1
        self.save()
        self._samples = []   # ใช้ข้อมูลชุดนี้แล้ว เคลียร์กันนับซ้ำ
        summary = {
            "n": n, "offset": (ox, oy), "delta": (d_pan, d_tilt), "capped": capped,
            "trim_old": old, "trim_new": (new_pan, new_tilt), "rms_deg": rms,
        }
        print(f"[FireTune] ✅ สอนแล้ว (n={n}): offset=({ox:+.2f},{oy:+.2f}) rms={rms:.2f}° "
              f"→ trim ({old[0]:+.2f},{old[1]:+.2f}) → ({new_pan:+.2f},{new_tilt:+.2f})°"
              f"{' [capped]' if capped else ''}  saved.", flush=True)
        return summary


_fire_tune_learner: Optional["FireTuneLearner"] = None


def _cam4_crosshair_trim_deg(ctx: Dict[str, Any]) -> Tuple[float, float]:
    """Pan/tilt deg to align cam8 absolute target with reticle vs frame geometric center (px/deg like bias)."""
    if not CAM8_CROSSHAIR_TRIM_ENABLED:
        return (0.0, 0.0)
    w = ctx.get("w")
    h = ctx.get("h")
    cx_frame = ctx.get("cx_frame")
    cy_frame = ctx.get("cy_frame")
    ppx = ctx.get("px_per_deg_x")
    ppy = ctx.get("px_per_deg_y")
    if w is None or h is None or cx_frame is None or cy_frame is None:
        return (0.0, 0.0)
    # ppd_y ติดลบเสมอตาม convention (แกน y ภาพกลับทิศกับ tilt) — เช็ค <= 0 จะ return (0,0) ทุกครั้ง
    # = ปิด CAM8_CROSSHAIR_TRIM ทิ้งแบบเงียบ ๆ. ที่ต้องกันคือ 'ศูนย์' ไม่ใช่ 'ติดลบ'
    if ppx is None or ppy is None or abs(float(ppx)) < 1e-6 or abs(float(ppy)) < 1e-6:
        return (0.0, 0.0)
    geom_cx = float(w) * 0.5
    geom_cy = float(h) * 0.5
    trim_pan = (geom_cx - float(cx_frame)) / float(ppx)
    trim_tilt = (geom_cy - float(cy_frame)) / float(ppy)
    return (trim_pan, trim_tilt)


def _tick_auto(ctx):
    """AUTO mode: drive arm from cam8 cue only (no cam4 fallback)."""
    arm_controller = ctx.get("arm_controller")
    arm_drive_state = ctx.get("arm_drive_state")
    x_lo = ctx.get("x_lo")
    y_lo = ctx.get("y_lo")
    x_hi = ctx.get("x_hi")
    y_hi = ctx.get("y_hi")
    if (
        arm_controller is None
        or arm_drive_state is None
        or x_lo is None
    ):
        return
    if _variable_step_toward_target is None:
        return

    # ------------------------------------------------------------------
    # เส้นทาง cam8 cue: ใช้ calibration cam8→arm ขับแขนตรงๆ (ข้าม Kalman)
    # ------------------------------------------------------------------
    _cam8_cue = ctx.get("latest_cam8_cue")
    _cam8_calib = ctx.get("cam8_calib_data")
    _cam8_target = None
    _cam8_source_ok = (
        _cam8_cue is not None
        and str(_cam8_cue.get("source_camera", "")).lower() == "cam8"
    )
    if _cam8_source_ok and _cam8_calib is not None and _cam8_pixel_to_arm_degrees is not None:
        _cx8 = _cam8_cue.get("cx")
        _cy8 = _cam8_cue.get("cy")
        _fw8 = _cam8_cue.get("frame_w")
        _fh8 = _cam8_cue.get("frame_h")
        if _cx8 is not None and _cy8 is not None:
            _cam8_target = _cam8_pixel_to_arm_degrees(
                float(_cx8),
                float(_cy8),
                _cam8_calib,
                _fw8,
                _fh8,
                use_homography=CAM8_PIXEL_TO_ARM_USE_HOMOGRAPHY,
            )

    if _cam8_target is not None:
        # cam8 cue path: ขับแขนไปยัง absolute target ที่ได้จาก calibration cam8→arm
        cam8_pan_deg, cam8_tilt_deg = _cam8_target

        _trim_pan, _trim_tilt = _cam4_crosshair_trim_deg(ctx)
        cam8_pan_deg += _trim_pan
        cam8_tilt_deg += _trim_tilt

        # apply residual bias compensation (if learner active)
        _bias_learner = _get_bias_learner()
        _bias_pan, _bias_tilt = 0.0, 0.0
        if _bias_learner is not None and CAM8_AUTO_ONLINE_BIAS_ENABLED:
            _bias_pan, _bias_tilt = _bias_learner.get_bias(_cam8_cue)
        cam8_pan_deg += _bias_pan
        cam8_tilt_deg += _bias_tilt

        pan_cur = getattr(arm_controller, "pos_x", 0.0)
        tilt_cur = getattr(arm_controller, "pos_y", 0.0)
        last_continuous_arm_move_time = ctx.get("last_continuous_arm_move_time", 0.0)
        if SWAP_PAN_TILT:
            target_pan = float(np.clip(cam8_pan_deg, y_lo, y_hi))
            target_tilt = float(np.clip(cam8_tilt_deg, x_lo, x_hi))
        else:
            target_pan = float(np.clip(cam8_pan_deg, x_lo, x_hi))
            target_tilt = float(np.clip(cam8_tilt_deg, y_lo, y_hi))
        error_deg = math.hypot(target_pan - pan_cur, target_tilt - tilt_cur)
        if error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
            throttle_sec = CONTINUOUS_THROTTLE_SEC
        elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
            throttle_sec = CONTINUOUS_P_THROTTLE_MID_SEC
        else:
            throttle_sec = CONTINUOUS_P_THROTTLE_NEAR_SEC
        last_continuous_arm_move_time, move_pan, move_tilt = _variable_step_toward_target(
            arm_controller, target_pan, target_tilt, last_continuous_arm_move_time, throttle_sec, arm_drive_state, step_scale=1.0
        )
        _move_issued = (move_pan != 0.0 or move_tilt != 0.0)
        ctx["last_continuous_arm_move_time"] = last_continuous_arm_move_time
        ctx["auto_cue_source"] = "cam8"

        # tick bias learner state machine
        if _bias_learner is not None and CAM8_AUTO_ONLINE_BIAS_ENABLED:
            _target_det = ctx.get("target_det")
            _w = ctx.get("w", 0)
            _h = ctx.get("h", 0)
            _cx_frame = ctx.get("cx_frame", _w / 2)
            _cy_frame = ctx.get("cy_frame", _h / 2)
            _px_per_deg_x = ctx.get("px_per_deg_x")
            _px_per_deg_y = ctx.get("px_per_deg_y")
            _arm_mode = ctx.get("arm_mode", None)
            _bias_learner.tick(
                arm_mode=_arm_mode,
                cue_dict=_cam8_cue,
                target_det=_target_det,
                cam4_w=_w,
                cam4_h=_h,
                cx_frame=_cx_frame,
                cy_frame=_cy_frame,
                px_per_deg_x=_px_per_deg_x,
                px_per_deg_y=_px_per_deg_y,
                move_issued_this_frame=_move_issued,
            )
        ctx["last_target_pan"] = target_pan
        ctx["last_target_tilt"] = target_tilt
        ctx["last_arm_error_deg"] = error_deg
        return

    # No valid cam8 cue (or no calibration): hold position in AUTO (strict cam8-only mode).
    if _cam8_cue is not None and not _cam8_source_ok:
        ctx["auto_cue_source"] = "cam8_wait(non-cam8)"
    else:
        ctx["auto_cue_source"] = "cam8_wait"
    return


def _lock_feed_bearing_measurement(
    lock_iou_tracker,
    lock_kalman,
    pose_hist,
    arm_controller,
    t_capture,
    cx_frame,
    cy_frame,
    px_per_deg_x,
    px_per_deg_y,
    ego_comp_latency_sec: float = LOCK_EGO_COMP_LATENCY_SEC,
):
    """Ego-motion compensation: แปลงเป้าที่ IoU tracker match ล่าสุดเป็น absolute bearing
    (ท่าแขน ณ เวลาเก็บภาพ t_capture + pixel_offset/ppd) แล้วป้อนเข้า lock_kalman (bearing space).
    bearing ไม่เปลี่ยนตามการขยับแขน → ไม่ double-count (convention เดียวกับ 23_arm_chase_sim).
    ต้องเรียกเฉพาะตอน tracker match จริง → ตอนเป้าหลุด filter จะ coast ตามเวลา
    ไม่โดนป้อนค่าค้างซ้ำ (แก้บั๊กแขนไล่ตำแหน่งผี). คืน True เมื่อป้อน measurement สำเร็จ."""
    if (
        arm_controller is None or lock_kalman is None or lock_iou_tracker is None
        or px_per_deg_x is None or abs(px_per_deg_x) <= 1e-6
        or px_per_deg_y is None or abs(px_per_deg_y) <= 1e-6
        or lock_iou_tracker.smooth_cx is None or lock_iou_tracker.smooth_cy is None
        or cx_frame is None or cy_frame is None
    ):
        return False
    # ท่าแขน ณ 'เวลาเก็บภาพจริง' = grab time − camera latency (เนื้อภาพเก่ากว่า pose ที่บันทึกตอน grab)
    # → กัน arm-motion leak เข้า bearing ตอน slew (แขนไล่เลยเป้า). latency=0 = ปิด
    # latency เป็นคุณสมบัติ 'ต่อกล้อง' (sensor→network→decode) → มาจาก CAMERAS[cam]["ego_comp_latency_sec"]
    pose_cap = pose_hist.pose_at(t_capture - ego_comp_latency_sec) if pose_hist is not None else None
    if pose_cap is None:
        pose_cap = (
            float(getattr(arm_controller, "pos_x", 0.0)),
            float(getattr(arm_controller, "pos_y", 0.0)),
        )
    b_pan = pose_cap[0] + LOCK_PAN_SIGN * (
        (float(lock_iou_tracker.smooth_cx) - float(cx_frame)) / px_per_deg_x
    )
    b_tilt = pose_cap[1] + LOCK_TILT_SIGN * (
        (float(lock_iou_tracker.smooth_cy) - float(cy_frame)) / px_per_deg_y
    )
    conf = lock_iou_tracker.last_conf if lock_iou_tracker.last_conf is not None else 1.0
    # --- Outlier gate (กันแขนสบัด): measurement ที่กระโดดไกลจากที่ kalman ทำนาย = associate ผิด ---
    # ทิ้ง outlier เฟรมเดียว (แขน coast ต่อ ไม่สบัด); ถ้าเกาะกลุ่มที่ใหม่ครบ CONFIRM = re-lock จริง → รับ
    _lock_meas_dbg[0], _lock_meas_dbg[1], _lock_meas_dbg[2] = b_pan, b_tilt, "fed"  # default (gate ทับเป็น rej/reseed)
    _kal_init = getattr(lock_kalman, "_initialized", False)
    if _kal_init:
        _age = t_capture - getattr(lock_kalman, "_last_update_time", t_capture)
        if 0.0 <= _age < 0.5:   # gate เฉพาะตอนมี track สดพอจะทำนายได้ (gap ยาว = re-lock คาดได้ ไม่ gate)
            _pp, _pt = lock_kalman.predict_ahead(_age)
            _jump = math.hypot(b_pan - _pp, b_tilt - _pt)
            if _jump > LOCK_MEAS_OUTLIER_MAX_STEP_DEG:
                _run = _lock_outlier_run
                if _run[0] > 0 and math.hypot(b_pan - _run[1], b_tilt - _run[2]) <= LOCK_MEAS_OUTLIER_CLUSTER_DEG:
                    _run[0] += 1
                else:
                    _run[0] = 1
                _run[1], _run[2] = b_pan, b_tilt
                if _run[0] < LOCK_MEAS_OUTLIER_CONFIRM:
                    _lock_meas_dbg[0], _lock_meas_dbg[1], _lock_meas_dbg[2] = b_pan, b_tilt, "rej"
                    return False   # ทิ้งเฟรมนี้ — ไม่ป้อน kalman → แขนไม่สบัด (ยัง coast บน velocity เดิม)
                # ยืนยันแล้วว่าเป้าอยู่ที่ใหม่จริง (re-lock/หักเลี้ยวแรง) → snap position ไปที่ใหม่
                # 'โดยคงความเร็วเดิมไว้' (reseed_position) แทน reset ที่ zero velocity — เป้าที่บินอยู่
                # ถ้า zero velocity แขนจะเลิก lead ตามไม่ทัน → re-outlier → วน = hunting (เจอบน pan ตอน
                # เป้าบินขวาง). กัน re-lock คืบช้าด้วย (bearing KF smoothing หนัก ป้อน step ขยับทีละ 0.1°)
                lock_kalman.reseed_position(b_pan, b_tilt)
                _run[0] = 0        # เฟรมถัดไป in-range แล้ว → feed ปกติ
                _lock_meas_dbg[0], _lock_meas_dbg[1], _lock_meas_dbg[2] = b_pan, b_tilt, "reseed"
            else:
                _lock_outlier_run[0] = 0   # measurement อยู่ในพิสัย → ล้างตัวสะสม outlier
                _lock_meas_dbg[0], _lock_meas_dbg[1], _lock_meas_dbg[2] = b_pan, b_tilt, "fed"
    # --- prediction-residual: 'ที่ผ่านมาทำนายแม่นแค่ไหน' = ความคาดเดาได้ของเป้า ---
    # ทำนายตำแหน่ง measurement นี้จาก measurement ก่อน + velocity ก่อน (const-velocity)
    # ถ้าแม่น (เป้านิ่ง/บินเรียบ) resid→0; ถ้าพลาด (เป้าหักเลี้ยว) resid สูง → readiness งดยิง
    vprev_p, vprev_t = lock_kalman.get_velocity_raw()
    prev = _lock_meas_prev[0]
    if prev is not None:
        _dt_m = t_capture - prev[2]
        if 1e-3 < _dt_m < 0.5:
            _pred_p = prev[0] + prev[3] * _dt_m
            _pred_t = prev[1] + prev[4] * _dt_m
            _resid = math.hypot(b_pan - _pred_p, b_tilt - _pred_t)
            _resid = max(0.0, _resid - LOCK_FIRE_RESID_NOISE_FLOOR_DEG)  # หัก sensor jitter → เหลือการหลบจริง
            _rate = _resid / _dt_m   # deg ต่อวินาที ของความคลาดเคลื่อนการทำนาย
            _a = 0.3
            _lock_pred_resid_rate[0] = (1 - _a) * _lock_pred_resid_rate[0] + _a * _rate
    _lock_meas_prev[0] = [b_pan, b_tilt, t_capture, vprev_p, vprev_t]
    lock_kalman.update(b_pan, b_tilt, float(conf))
    return True


def compute_firing_lead(vpan_deg_s, vtilt_deg_s, range_m, muzzle_ms,
                        drop_comp=True, tilt_up_sign=-1.0, max_lead_deg=8.0):
    """Firing/intercept solution: จุดเล็งเผื่อเวลากระสุนบิน.
      t_flight = range / muzzle_velocity
      lead_เชิงมุม = velocity_เป้า(deg/s) × t_flight   (โดรนเคลื่อนไปเท่านี้ระหว่างกระสุนบิน)
      drop = 0.5·g·t² (m) → มุมเงยชดเชย = atan(drop/range)
    คืน (lead_pan_deg, lead_tilt_deg, t_flight_sec, drop_deg). lead ถูก cap ที่ max_lead_deg."""
    if range_m is None or range_m <= 0 or muzzle_ms <= 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    t_flight = range_m / muzzle_ms
    lead_pan = vpan_deg_s * t_flight
    lead_tilt = vtilt_deg_s * t_flight
    mag = math.hypot(lead_pan, lead_tilt)
    if mag > max_lead_deg and mag > 1e-6:
        s = max_lead_deg / mag
        lead_pan *= s
        lead_tilt *= s
    drop_deg = 0.0
    if drop_comp:
        drop_m = 0.5 * GRAVITY_MS2 * t_flight * t_flight
        drop_deg = math.degrees(math.atan2(drop_m, range_m))
    return lead_pan, lead_tilt + tilt_up_sign * drop_deg, t_flight, drop_deg


def _tick_lock(ctx):
    """LOCK mode: auto-track ใน bearing space (ego-motion compensated).
    main loop แปลงเป้าที่ track ได้เป็น absolute bearing (ท่าแขน ณ เวลาเก็บภาพ + pixel_offset/ppd)
    แล้วป้อนเข้า lock_kalman — ฟังก์ชันนี้แค่ขับแขนเข้าหา bearing ที่ filter ไว้
    (แนวทางเดียวกับ 23_arm_chase_sim ที่ validate แล้ว: การขยับแขนไม่ถูกนับซ้ำเข้า velocity เป้า).
    ตอน YOLO หลุด: ไม่มี measurement ใหม่ → predict_ahead ทำ bounded coast (τ=KALMAN_COAST_DECAY_TAU)
    จนครบ LOCK_KALMAN_COAST_MAX_SEC แล้ว hold ตำแหน่ง. ปิดได้ด้วย LOCK_AUTOTRACK_ENABLED=False."""
    if not LOCK_AUTOTRACK_ENABLED:
        return
    arm_controller = ctx.get("arm_controller")
    lock_arm_drive_state = ctx.get("lock_arm_drive_state")
    lock_kalman = ctx.get("lock_kalman")
    x_lo = ctx.get("x_lo"); y_lo = ctx.get("y_lo"); x_hi = ctx.get("x_hi"); y_hi = ctx.get("y_hi")
    lock_lost = ctx.get("lock_csrt_lost", False)
    lock_init = ctx.get("lock_csrt_initialized", False)
    lock_last_arm_move_time = ctx.get("lock_last_arm_move_time", 0.0)

    if LOCK_TRACK_DEBUG and (time.time() - _lock_dbg2_last_t[0]) > 1.0:
        _lock_dbg2_last_t[0] = time.time()
        print(f"[LOCK dbg] init={lock_init} lost={lock_lost} "
              f"kal_init={getattr(lock_kalman, '_initialized', False)} "
              f"ctrl={arm_controller is not None} ds={lock_arm_drive_state is not None}", flush=True)
    if (arm_controller is None or lock_arm_drive_state is None or _variable_step_toward_target is None
            or lock_kalman is None or x_lo is None):
        return
    if not lock_init or not lock_kalman._initialized:
        return  # ยังไม่ได้ล็อกเป้า / ยังไม่มี bearing measurement แรก → ไม่ขับแขน

    now = time.time()
    # measurement เข้า filter เฉพาะตอน tracker match จริง (main loop) → age = ความเก่าของเป้า
    coast_elapsed = now - lock_kalman._last_update_time
    if coast_elapsed > LOCK_KALMAN_COAST_MAX_SEC:
        return  # เป้าหายเกินหน้าต่าง coast → hold ตำแหน่ง (รอเป้ากลับมา/ผู้ใช้ปลดล็อก)

    # ประเมินความเร่งเป้า (vector) จากการเปลี่ยน velocity — ใช้ทั้ง lead(CA) และ readiness
    _vpan_now, _vtilt_now = lock_kalman.get_velocity_raw()
    _dt_v = now - _lock_vel_hist[2] if _lock_vel_hist[2] > 0 else 0.0
    if 1e-3 < _dt_v < 0.5:
        _ax = (_vpan_now - _lock_vel_hist[0]) / _dt_v
        _ay = (_vtilt_now - _lock_vel_hist[1]) / _dt_v
        a = LOCK_LEAD_ACCEL_EMA_ALPHA
        _lock_accel_ema[0] = (1 - a) * _lock_accel_ema[0] + a * _ax
        _lock_accel_ema[1] = (1 - a) * _lock_accel_ema[1] + a * _ay
    _lock_vel_hist[0], _lock_vel_hist[1], _lock_vel_hist[2] = _vpan_now, _vtilt_now, now
    _acc_pan, _acc_tilt = _lock_accel_ema[0], _lock_accel_ema[1]
    _acc_mag = math.hypot(_acc_pan, _acc_tilt)

    # เล็งดักหน้า (lead) แทนวิ่งตาม: aim = pos + v·t + ½·a·t² (constant-acceleration ทำนายโค้ง)
    #   lead_time = age + pipeline_latency (measurement เก่าเท่านี้) + actuation lead
    pipe_lat = ctx.get("avg_pipeline_latency") or LOCK_PIPELINE_LATENCY_FALLBACK
    _lead_active = LOCK_LEAD_ENABLED and (coast_elapsed <= LOCK_LEAD_COAST_SWITCH_SEC)
    _lead_gain = 0.0   # diag (per-frame log): สัดส่วน lead ที่ใช้จริงหลัง velocity deadband
    base_pan = base_tilt = 0.0
    if _lead_active:
        now_dt = coast_elapsed + float(pipe_lat)
        lead_sec = now_dt + LOCK_LEAD_EXTRA_SEC
        base_pan, base_tilt = lock_kalman.predict_ahead(now_dt, decay_tau=None)   # ตำแหน่งปัจจุบันจริง
        aim_pan, aim_tilt = lock_kalman.predict_ahead(lead_sec, decay_tau=None)   # จุดดักหน้า (v·t)
        if LOCK_LEAD_ACCEL_ENABLED:
            # เพิ่มพจน์ ½·a·t² (ทำนายโค้ง) — cap กัน accel noise
            _ca_p = 0.5 * _acc_pan * lead_sec * lead_sec
            _ca_t = 0.5 * _acc_tilt * lead_sec * lead_sec
            _ca_mag = math.hypot(_ca_p, _ca_t)
            if _ca_mag > LOCK_LEAD_ACCEL_MAX_DEG and _ca_mag > 1e-6:
                _cs = LOCK_LEAD_ACCEL_MAX_DEG / _ca_mag
                _ca_p *= _cs; _ca_t *= _cs
            aim_pan += _ca_p
            aim_tilt += _ca_t
            base_pan += 0.5 * _acc_pan * now_dt * now_dt
            base_tilt += 0.5 * _acc_tilt * now_dt * now_dt
        ld_p = aim_pan - base_pan
        ld_t = aim_tilt - base_tilt
        # velocity deadband: เป้าเกือบนิ่ง (speed ต่ำ) → lead ที่คำนวณ (v·t + ½a·t²) ส่วนใหญ่มาจาก noise
        # ของ velocity/accel estimate ไม่ใช่การเคลื่อนจริง → หรี่ lead ลงตาม speed. โดรนจริง test นี้
        # เคลื่อนเชิงมุม ~0.2-1.5°/s (เกือบนิ่ง) แต่ tilt แกว่ง ±7° = lead ขยาย noise (โดยเฉพาะพจน์ accel
        # ที่เป็น 2nd-derivative ของสัญญาณ noisy → แกว่งหนัก บนแกน tilt ที่ bbox บน/ล่างสั่นกว่าซ้าย/ขวา).
        # เป้าเร็วจริง (speed สูง) → gain=1 lead เต็ม (ยังตามทัน); หรี่เฉพาะตอนช้าที่ lead จริงจิ๊บจ๊อยอยู่แล้ว
        _spd = math.hypot(_vpan_now, _vtilt_now)
        _lead_gain = (_spd - LOCK_LEAD_VEL_FLOOR_LO) / max(1e-6, LOCK_LEAD_VEL_FLOOR_HI - LOCK_LEAD_VEL_FLOOR_LO)
        _lead_gain = max(0.0, min(1.0, _lead_gain))
        # close-range attenuation: หรี่ lead ตามระยะ (เป้าใกล้ = ω ไม่คงที่ → v·t เกิน → เหวี่ยง)
        if LOCK_LEAD_RANGE_ATTEN_ENABLED:
            _rng_now = ctx.get("lock_range_m")
            if _rng_now:
                _rg = (float(_rng_now) - LOCK_LEAD_RANGE_NEAR_M) / max(
                    1e-6, LOCK_LEAD_RANGE_FAR_M - LOCK_LEAD_RANGE_NEAR_M)
                _range_gain = LOCK_LEAD_RANGE_MIN_GAIN + (1.0 - LOCK_LEAD_RANGE_MIN_GAIN) * max(0.0, min(1.0, _rg))
                _lead_gain *= _range_gain
        ld_p *= _lead_gain
        ld_t *= _lead_gain
        # cap ระยะ lead รวม กัน noise เหวี่ยงแขน
        ld_mag = math.hypot(ld_p, ld_t)
        if ld_mag > LOCK_LEAD_MAX_DEG and ld_mag > 1e-6:
            _s = LOCK_LEAD_MAX_DEG / ld_mag
            ld_p *= _s
            ld_t *= _s
        aim_pan = base_pan + ld_p    # ประกอบ aim กลับจาก base + lead (หลัง gain/cap) เสมอ
        aim_tilt = base_tilt + ld_t
    else:
        # เป้าหลุด/นิ่ง → bounded coast (decay) กัน overshoot ตอนเป้าเลี้ยว (validate 31/32)
        aim_pan, aim_tilt = lock_kalman.predict_ahead(coast_elapsed, decay_tau=KALMAN_COAST_DECAY_TAU)

    # Firing solution: จุดยิงจริง = 'ตำแหน่งปัจจุบันจริง (base) + lead กระสุนบิน' — ไม่รวม actuation EXTRA
    #   (EXTRA เป็น lead ให้แขน 'ตามทัน' ตอน track เท่านั้น; ตอน snap วางลำกล้องตรงจุดแล้วยิงทันที ไม่มี lag
    #    → ถ้าเอา EXTRA มาใส่จุดยิงด้วย = over-lead v×EXTRA พลาดไปข้างหน้า โดยเฉพาะเป้าเร็ว)
    fire_t_flight = 0.0
    fire_lead_deg = 0.0
    fire_range_m = ctx.get("lock_range_m")
    _base_pan = base_pan if _lead_active else aim_pan     # ตำแหน่งจริงตอนนี้ (coast: ใช้ aim)
    _base_tilt = base_tilt if _lead_active else aim_tilt
    _flead_p = _flead_t = 0.0
    if LOCK_FIRE_SOLUTION_ENABLED and _lead_active:
        _vpan, _vtilt = lock_kalman.get_velocity_raw()
        _muzzle = float(ctx.get("muzzle_velocity_ms", 900.0) or 900.0)
        _flead_p, _flead_t, fire_t_flight, _drop = compute_firing_lead(
            _vpan, _vtilt, fire_range_m, _muzzle,
            drop_comp=LOCK_FIRE_DROP_COMP, tilt_up_sign=LOCK_FIRE_TILT_UP_SIGN,
            max_lead_deg=LOCK_FIRE_MAX_LEAD_DEG,
        )
        aim_pan += _flead_p      # tracking aim (มี EXTRA) — ใช้ขับแขนตามต่อเนื่อง
        aim_tilt += _flead_t
        fire_lead_deg = math.hypot(_flead_p, _flead_t)

    # learned fire trim (offset เชิงเรขาคณิต) — บวกเข้าทั้งจุด track และจุดยิง
    _trim_p = ctx.get("fire_trim_pan", 0.0)
    _trim_t = ctx.get("fire_trim_tilt", 0.0)
    aim_pan += _trim_p
    aim_tilt += _trim_t
    # จุดยิงจริง (snap/readiness/impact) = base + firing_lead + trim  (ไม่มี EXTRA)
    fire_aim_pan = _base_pan + _flead_p + _trim_p
    fire_aim_tilt = _base_tilt + _flead_t + _trim_t

    pan_cur = getattr(arm_controller, "pos_x", 0.0)
    tilt_cur = getattr(arm_controller, "pos_y", 0.0)
    if SWAP_PAN_TILT:  # config ปัจจุบัน CAM4_ARM_SWAP_PAN_TILT=False (ยืนยันแล้ว) — คง branch ให้ตรง acquire
        target_pan = float(np.clip(aim_pan, y_lo, y_hi))          # tracking drive
        target_tilt = float(np.clip(aim_tilt, x_lo, x_hi))
        fire_target_pan = float(np.clip(fire_aim_pan, y_lo, y_hi))  # จุดยิงจริง
        fire_target_tilt = float(np.clip(fire_aim_tilt, x_lo, x_hi))
    else:
        target_pan = float(np.clip(aim_pan, x_lo, x_hi))
        target_tilt = float(np.clip(aim_tilt, y_lo, y_hi))
        fire_target_pan = float(np.clip(fire_aim_pan, x_lo, x_hi))
        fire_target_tilt = float(np.clip(fire_aim_tilt, y_lo, y_hi))
    # --- Close-range slew guard: จำกัดอัตราการเปลี่ยนคำสั่ง 'track' (deg/s) ตามระยะ ---
    # reacquire หลัง gap → base(KF) snap → target กระโดด → แขนกระชาก. ไถเข้าแบบ ramp แทนกระโดด:
    # จำกัด step ต่อ tick = cap_rate × dt (cap แน่นตอนใกล้, หลวมตอนไกล). steady tracking ที่ระยะนั้น
    # (ω·dt) ต่ำกว่า cap อยู่แล้ว → ไม่โดนหรี่; โดนเฉพาะ reacquire teleport. ไม่แตะ fire_target/snap.
    if LOCK_SLEW_GUARD_ENABLED:
        _sp, _stl, _stt = _lock_slew_prev
        if _sp is not None and _stt > 0:
            _sdt = now - _stt
            if 1e-3 < _sdt < 0.5:   # dt สมเหตุผล (ข้ามเฟรมยาว = เพิ่งกลับมา track → ไม่ limit)
                _rng_g = ctx.get("lock_range_m")
                if _rng_g:
                    _fg = (float(_rng_g) - LOCK_SLEW_GUARD_NEAR_M) / max(
                        1e-6, LOCK_SLEW_GUARD_FAR_M - LOCK_SLEW_GUARD_NEAR_M)
                    _cap_rate = LOCK_SLEW_GUARD_NEAR_DEG_S + (
                        LOCK_SLEW_GUARD_FAR_DEG_S - LOCK_SLEW_GUARD_NEAR_DEG_S) * max(0.0, min(1.0, _fg))
                else:
                    _cap_rate = LOCK_SLEW_GUARD_FAR_DEG_S   # ไม่รู้ระยะ → หลวม (พฤติกรรมเดิม)
                _max_step = _cap_rate * _sdt
                _dp = target_pan - _sp
                _dtl = target_tilt - _stl
                _dmag = math.hypot(_dp, _dtl)
                if _dmag > _max_step and _dmag > 1e-9:
                    _sc = _max_step / _dmag
                    target_pan = _sp + _dp * _sc
                    target_tilt = _stl + _dtl * _sc
        _lock_slew_prev[0] = target_pan
        _lock_slew_prev[1] = target_tilt
        _lock_slew_prev[2] = now
    # error สำหรับ 'ความพร้อมยิง' = แขน → จุดยิงจริง (ไม่ใช่จุด track ที่ล้ำ EXTRA)
    error_deg = math.hypot(fire_target_pan - pan_cur, fire_target_tilt - tilt_cur)

    # (per-frame diag ย้ายไปเขียนหลังคำนวณ fire readiness ด้านล่าง — จะได้ log resid/uncert/confident/ready ด้วย)

    # --- Fire readiness (ซื่อสัตย์): 'ยิงแล้วโดนจริงไหม' ต้องรวมความไม่แน่นอนจาก latency ---
    # pred_miss = arm→intercept (error_deg) เป็นแค่ความคลาดเคลื่อนของการเล็ง
    # แต่จุด intercept ถูก 'ทำนาย' ล่วงหน้าข้าม pipeline latency → มี uncertainty
    #   uncertainty ≈ |velocity| × (pipe_lat + t_flight) × K  (โดรนยิ่งเร็ว/latency ยิ่งสูง ยิ่งพลาด)
    # expected_miss = error_deg + uncertainty ; READY เมื่อ expected_miss ≤ hit_radius
    hit_radius_deg = None
    fire_ready = False
    # fire-safe zone: ทิศเล็งต้องอยู่ในโคนกลางที่ปลอดภัย (กันยิงบริเวณอันตราย)
    fire_safe_zone = (not LOCK_FIRE_SAFE_ENABLED) or (
        abs(target_pan) <= LOCK_FIRE_SAFE_PAN_DEG and abs(target_tilt) <= LOCK_FIRE_SAFE_TILT_DEG
    )
    # ความไม่แน่นอนของ intercept = 'ความคาดเดาไม่ได้ของเป้า' วัดจาก prediction residual จริง
    # (แม่นกว่าใช้ accel ที่ smooth): resid_rate (deg/s) × horizon = คาดว่าจะพลาดเพิ่มเท่านี้
    # เป้าหลบหลีก → resid_rate สูง → uncert สูง → 'ไม่ยิง' (แก้อาการโดนบ้างเฉียดบ้าง)
    _horizon = float(pipe_lat) + fire_t_flight
    _uncert = (_lock_pred_resid_rate[0] * _horizon * LOCK_FIRE_PRED_UNCERT_K
               + LOCK_FIRE_CURVATURE_K * _acc_mag * _horizon * _horizon
               + LOCK_FIRE_NOISE_FLOOR_DEG)
    expected_miss = error_deg + _uncert   # honest predicted miss (arm→intercept + uncert) — โชว์ HUD
    predict_uncert_deg = _uncert          # ความไม่แน่นอน 'ของจุดที่ทำนาย' (ไม่รวม error แขน — snap แก้ให้)
    confident_ready = False
    if fire_range_m:
        # hit_radius = รัศมี 'โดนจริง' (scoring/HUD) — เข้มตามขนาดโดรน
        hit_radius_deg = max(
            LOCK_FIRE_READY_MIN_RADIUS_DEG,
            math.degrees(math.atan2(LOCK_FIRE_HIT_RADIUS_M, float(fire_range_m))),
        )
        _stable = (not lock_lost) and (coast_elapsed <= LOCK_FIRE_STABLE_MAX_AGE_SEC)
        # มั่นใจ = ทำนายจุด intercept แม่นพอจะโดน (uncert ≤ hit_radius × K) — ไม่เกี่ยว error แขน (snap แก้)
        _confident = (_stable and fire_safe_zone
                      and predict_uncert_deg <= hit_radius_deg * LOCK_FIRE_CONFIDENCE_K
                      and _lock_pred_resid_rate[0] <= LOCK_FIRE_MAX_RESID_DEG_S)
        if _confident:
            _lock_confident_count[0] += 1
        else:
            _lock_confident_count[0] = 0
        confident_ready = _lock_confident_count[0] >= LOCK_FIRE_CONFIDENT_FRAMES
        # fire_ready (HUD "SHOOT" + auto-fire): มั่นใจ 'และ' แขนอยู่บนเป้าจริงแล้ว
        # (โหมด snap: กระโดดให้ error≈0 ก่อน แล้วค่อยตั้ง flag นี้ → ไม่ยิงตอนแขนยังห่าง)
        aim_converged = error_deg <= hit_radius_deg
        if _confident and aim_converged:
            _lock_fire_ready_count[0] += 1
        else:
            _lock_fire_ready_count[0] = 0
        fire_ready = _lock_fire_ready_count[0] >= LOCK_FIRE_READY_FRAMES
    else:
        _lock_fire_ready_count[0] = 0
        _lock_confident_count[0] = 0
    ctx["lock_fire_ready"] = fire_ready
    ctx["lock_fire_safe_zone"] = fire_safe_zone
    ctx["lock_hit_radius_deg"] = hit_radius_deg
    ctx["lock_predict_uncert_deg"] = predict_uncert_deg
    ctx["lock_confident"] = confident_ready
    ctx["lock_pred_miss_deg"] = error_deg
    ctx["lock_expected_miss_deg"] = expected_miss
    ctx["lock_pipe_lat"] = float(pipe_lat)
    ctx["lock_fire_t_flight"] = fire_t_flight
    ctx["lock_fire_lead_deg"] = fire_lead_deg
    ctx["last_arm_error_deg"] = error_deg

    # --- per-frame diag (LOCK_FRAME_LOG=1): jump มาจากไหน + 'ทำไมไม่ยิง' (gate values) ---
    # base(KF) vs aim(cmd) vs meas แยกต้นเหตุ jump; resid/uncert/hit_r/errfire/confident/ready = fire gate
    # ทำไมไม่ยิง: ถ้า confident=0 ดู resid (>MAX_RESID?) หรือ uncert (>hit_r×K?); ถ้า confident=1 แต่ ready=0
    # = แขนยังไม่เข้าจุด (errfire>hit_r). block code: R=resid สูง U=uncert สูง A=age/coast S=safezone n=none
    if _lock_frame_log[0] is not None:
        try:
            _vp_dbg, _vt_dbg = lock_kalman.get_velocity_raw()
            _hr = hit_radius_deg if hit_radius_deg is not None else 0.0
            _blk = ""
            if not confident_ready:
                if _lock_pred_resid_rate[0] > LOCK_FIRE_MAX_RESID_DEG_S: _blk += "R"
                if hit_radius_deg is not None and predict_uncert_deg > hit_radius_deg * LOCK_FIRE_CONFIDENCE_K: _blk += "U"
                if coast_elapsed > LOCK_FIRE_STABLE_MAX_AGE_SEC: _blk += "A"
                if not fire_safe_zone: _blk += "S"
                if not _blk: _blk = "c"   # confident กำลังนับเฟรมอยู่ (ยังไม่ครบ CONFIDENT_FRAMES)
            elif not fire_ready:
                _blk = "e" if error_deg > _hr else "r"   # e=แขนยังไม่เข้าจุด, r=กำลังนับ READY_FRAMES
            else:
                _blk = "F"   # fire ready
            _lock_frame_log[0].write(
                f"{now:.3f},{coast_elapsed*1000:.0f},{pan_cur:.2f},{tilt_cur:.2f},"
                f"{base_pan:.2f},{base_tilt:.2f},{target_pan:.2f},{target_tilt:.2f},"
                f"{_vp_dbg:.2f},{_vt_dbg:.2f},{math.hypot(_vp_dbg,_vt_dbg):.2f},{_lead_gain:.2f},{fire_lead_deg:.2f},"
                f"{_lock_meas_dbg[0]:.2f},{_lock_meas_dbg[1]:.2f},{_lock_meas_dbg[2]},"
                f"{_lock_pred_resid_rate[0]:.2f},{predict_uncert_deg:.2f},{_hr:.2f},{error_deg:.2f},"
                f"{int(confident_ready)},{int(fire_ready)},{_blk}\n"
            )
            _lock_frame_log[0].flush()
        except Exception:
            pass

    # --- SNAP-AND-SHOOT: authorized + 'มั่นใจว่าทำนายแม่นพอจะโดน' → กระโดดไป intercept → ยิง ---
    # กดยิง = ตั้งใจยิง (ไม่ยิงทันที). ระหว่างที่ยังไม่มั่นใจ (confident_ready=False) → ตกไปข้างล่าง
    #   = lock ตามเป้าต่อไปเรื่อย ๆ จนกว่าจะทำนายจุดได้แม่น แล้วค่อยกระโดดยิง (แม่น)
    _fire_auth = ctx.get("fire_authorized", False)
    if (LOCK_FIRE_SNAP_AND_SHOOT and _fire_auth and _lead_active
            and confident_ready                                          # มั่นใจว่าทำนายแม่นพอจะโดนแล้ว
            and fire_safe_zone and not lock_lost
            and error_deg <= LOCK_FIRE_SNAP_MAX_DEG):                    # ใกล้พอจะ commit กระโดด
        try:
            arm_controller.move_absolute(fire_target_pan, fire_target_tilt, blocking=True)  # กระโดดไปจุดยิงจริง
        except Exception:
            pass
        ctx["lock_fire_ready"] = True     # แขนถึงจุดยิงแล้ว → main loop ลั่นไกทันที
        ctx["lock_fire_snap"] = True
        ctx["lock_last_arm_move_time"] = now
        ctx["last_target_pan"] = fire_target_pan
        ctx["last_target_tilt"] = fire_target_tilt
        ctx["last_arm_error_deg"] = 0.0
        if LOCK_TRACK_DEBUG:
            print(f"[LOCK] SNAP-AND-SHOOT → fire_pt=({fire_target_pan:+.2f},{fire_target_tilt:+.2f}) "
                  f"was_err={error_deg:.2f}deg uncert={predict_uncert_deg:.2f}deg "
                  f"resid={_lock_pred_resid_rate[0]:.1f}d/s lead={fire_lead_deg:.2f}deg", flush=True)
        return

    # deadzone ปรับตาม hit radius: เป้าเล็ก/ไกล ต้องเล็งแม่นกว่า deadzone ปกติถึงจะยิงโดน
    eff_deadzone = LOCK_DEADZONE_DEG
    if hit_radius_deg is not None:
        eff_deadzone = min(LOCK_DEADZONE_DEG, max(LOCK_FIRE_READY_MIN_RADIUS_DEG, hit_radius_deg * 0.6))
    if error_deg < eff_deadzone:
        return  # ใกล้เป้าพอแล้ว — กัน hunting/สั่น (ctx fire_ready ตั้งไว้แล้วด้านบน)
    # throttle ตามระยะ error (เหมือน _tick_auto)
    if error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
        throttle_sec = CONTINUOUS_THROTTLE_SEC
    elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
        throttle_sec = CONTINUOUS_P_THROTTLE_MID_SEC
    else:
        throttle_sec = CONTINUOUS_P_THROTTLE_NEAR_SEC
    lock_last_arm_move_time, _mp, _mt = _variable_step_toward_target(
        arm_controller, target_pan, target_tilt, lock_last_arm_move_time, throttle_sec,
        lock_arm_drive_state, step_scale=LOCK_TRACK_STEP_SCALE,
        profile_ref_deg=LOCK_TRACK_PROFILE_REF_DEG,
    )
    if LOCK_TRACK_DEBUG and (now - _lock_dbg_last_t[0]) > 0.3:
        _lock_dbg_last_t[0] = now
        _coasting = lock_lost or coast_elapsed > 0.15
        _spd = lock_kalman.get_speed_px_s()  # bearing space → deg/s
        _rng_s = f"{fire_range_m:.0f}m" if fire_range_m else "?"
        print(f"[LOCK] aim=({aim_pan:+6.2f},{aim_tilt:+6.2f}) cur=({pan_cur:+6.1f},{tilt_cur:+6.1f}) "
              f"-> tgt=({target_pan:+6.1f},{target_tilt:+6.1f}) move=({_mp:+.2f},{_mt:+.2f}) "
              f"err={error_deg:4.1f}deg vel={_spd:4.1f}d/s "
              f"{'LEAD' if _lead_active else 'coast'} age={coast_elapsed*1000:3.0f}ms "
              f"fire[rng={_rng_s} tof={fire_t_flight*1000:3.0f}ms lead={fire_lead_deg:.2f}deg "
              f"uncert={predict_uncert_deg:.2f} hit_r={hit_radius_deg if hit_radius_deg is None else round(hit_radius_deg,2)} "
              f"resid={_lock_pred_resid_rate[0]:.1f}d/s "
              f"{'CONFIDENT' if confident_ready else 'predicting'} "
              f"{'READY-SHOOT' if fire_ready else 'aiming'}]"
              f"{' COAST' if _coasting else ''}", flush=True)
    ctx["lock_last_arm_move_time"] = lock_last_arm_move_time
    ctx["last_target_pan"] = fire_target_pan   # จุดยิงจริง (ใช้เทียบใน impact log)
    ctx["last_target_tilt"] = fire_target_tilt
    ctx["last_arm_error_deg"] = error_deg
    ctx["lock_fire_t_flight"] = fire_t_flight
    ctx["lock_fire_lead_deg"] = fire_lead_deg


def main():
    global CAM8_AUTO_ONLINE_BIAS_ENABLED
    runtime_cleanup = get_runtime_cleanup()
    runtime_cleanup.install()

    # ค่าที่ operator แก้ในหน้า Settings ทับตัวแปร module-level ของไฟล์นี้ (fire gate, noise floor,
    # YOLO conf, LOCK_ROI_SPAN_DEG ฯลฯ ซึ่งไม่ได้อยู่ใน config.py) — ต้องทำก่อนค่าพวกนี้
    # ถูกก๊อปลง local ด้านล่าง. ฝั่ง config.py ทับไปแล้วตอน import
    if runtime_config is not None:
        _rc_mod = runtime_config.apply_to_module(globals())
        if _rc_mod:
            print(f"Gun Aim Assist: runtime override {len(_rc_mod)} ค่า -> " + ", ".join(_rc_mod))

    camera_name = CAMERA_NAME if CAMERA_NAME is not None else ACTIVE_CAMERA
    _set_active_camera_for_calibration(camera_name)
    cam_config = get_camera_config(camera_name)

    # --- Camera geometry: ppd ที่คาลิเบรตจริง คือแหล่งความจริงเดียว ---
    # FOV ที่พิมพ์มือใน config เป็นแค่ fallback ตอนยังไม่มีคาลิเบรต (และมักผิด: cam4 เคยตั้ง
    # 60°×36° ทั้งที่ ppd บอกว่า 44°×24°) — ระยะ/hit_radius/lead/slew guard พึ่ง FOV ทั้งหมด
    _geom = check_camera_geometry(
        camera_name, cam_config,
        cam_config.get("width"), cam_config.get("height"),
    )
    for _w in _geom["warnings"]:
        print(f"⚠️  Gun Aim Assist: {_w}", flush=True)
    for _p in _geom["problems"]:
        print(f"❌ Gun Aim Assist: {_p}", flush=True)
    calib_fatal = _geom["fatal"]
    if calib_fatal:
        print(
            "❌ Gun Aim Assist: เรขาคณิตกล้องเชื่อถือไม่ได้ → AUTO/LOCK ถูกบล็อก "
            "(MANUAL/SAFE ยังใช้ได้) จนกว่าจะคาลิเบรตให้ถูกกล้อง\n"
            f"   👉 กด [{KEY_WIZARD}] ในแอปเพื่อคาลิเบรตอัตโนมัติ "
            "(แขนหมุนเอง วัด ppd/latency/noise ให้ — หันกล้องไปทางที่มีต้นไม้หรืออาคารก่อน)",
            flush=True,
        )
    fov_h = _geom["fov_h"] or cam_config.get("fov_horizontal", 60.0)
    fov_v = _geom["fov_v"] or cam_config.get("fov_vertical", 36.0)
    # ego-comp latency ต่อกล้อง (sensor→network→decode ต่างรุ่นต่างกัน)
    ego_comp_latency = float(
        cam_config.get("ego_comp_latency_sec", LOCK_EGO_COMP_LATENCY_SEC)
    )
    print(
        f"Gun Aim Assist: {camera_name} geometry — ppd=({_geom['ppd_x']}, {_geom['ppd_y']}) "
        f"fov=({fov_h:.1f}°, {fov_v:.1f}°) ego_lat={ego_comp_latency*1000:.0f}ms",
        flush=True,
    )
    print("Gun Aim Assist: starting camera from config", camera_name)

    # ------------------------------------------------------------------
    # Optional: Initialize cam4 arm controller + tracker (homing + reference pose)
    # ------------------------------------------------------------------
    arm_controller = None
    arm_tracker = None
    arm_mode_manager = None
    joystick_reader = None
    joystick_mapper = None

    arm_calibration_ready = True
    cfg_mod = __import__("config") if _config_mod is None else _config_mod
    use_arm = bool(getattr(cfg_mod, "CAM4_ARM_ENABLED", False))
    use_sim = bool(getattr(cfg_mod, "CAM4_ARM_SIMULATION_MODE", False))
    _exit_on_arm_fail = bool(getattr(cfg_mod, "CAM4_ARM_EXIT_IF_HOMING_FAILS", True))

    if ArmModeManager is not None and use_arm:
        arm_mode_manager = ArmModeManager(has_arm=True)
        arm_mode_manager.set_mode(MODE_SAFE)
        print("Gun Aim Assist: MODE:SAFE during startup arm homing")

    if not use_arm:
        print("Gun Aim Assist: arm skipped — CAM4_ARM_ENABLED=False in config.py")
    elif Cam4ArmController is None or Cam4ArmTracker is None:
        print(
            "❌ Gun Aim Assist: CAM4_ARM_ENABLED=True but cam4_arm_controller/tracker "
            "could not be imported — install deps (pyserial, numpy) and check files."
        )
        if _exit_on_arm_fail:
            print("Gun Aim Assist: exiting (CAM4_ARM_EXIT_IF_HOMING_FAILS=True).")
            return
    else:
        print(
            f"Gun Aim Assist: arm enabled — homing before camera/YOLO "
            f"(simulation={use_sim}, port={getattr(cfg_mod, 'CAM4_ARM_SERIAL_PORT', '?')})"
        )
        try:
            if use_sim and SimCam4ArmController is not None:
                print("Gun Aim Assist: initializing SimCam4ArmController (simulation mode)...")
                arm_controller = SimCam4ArmController()
            else:
                print("Gun Aim Assist: initializing Cam4ArmController...")
                arm_controller = Cam4ArmController()
            if arm_controller.connect():
                arm_tracker = Cam4ArmTracker(arm_controller, camera_name=camera_name)
                print("Gun Aim Assist: arm homing/connect OK — proceeding to camera.")
                if arm_mode_manager is not None:
                    arm_mode_manager.set_mode(MODE_MANUAL)
                    print("Gun Aim Assist: startup homing done → MODE:MANUAL")
            else:
                print("❌ Gun Aim Assist: Cam4ArmController connect()/homing failed.")
                arm_controller = None
        except Exception as e:
            print(f"❌ Gun Aim Assist: failed to initialize Cam4 arm: {e}")
            arm_controller = None

        if arm_controller is None and _exit_on_arm_fail:
            print(
                "Gun Aim Assist: exiting — arm homing/connect is required "
                "(set CAM4_ARM_EXIT_IF_HOMING_FAILS=False to run without arm)."
            )
            return

    runtime_cleanup.set_resources(arm_controller=arm_controller)
    if arm_mode_manager is None and ArmModeManager is not None:
        arm_mode_manager = ArmModeManager(has_arm=arm_controller is not None)
    # JoystickReader แยกจากแขน — สลับ YOLO class / ความไวจอย ใช้ได้แม้ไม่มี arm (ขับแขนต้องมี mapper)
    if JoystickReader is not None:
        try:
            joystick_reader = JoystickReader()
            if getattr(joystick_reader, "enabled", False) and arm_controller is None:
                print(
                    "Gun Aim Assist: joystick enabled (class cycle / sensitivity); "
                    "arm not connected — no stick drive."
                )
        except Exception as e:
            print(f"Gun Aim Assist: joystick reader init failed: {e}")
            joystick_reader = None
    if joystick_reader is not None and JoystickArmMapper is not None and arm_controller is not None:
        try:
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
        except Exception as e:
            print(f"Gun Aim Assist: joystick mapper init failed: {e}")
            joystick_mapper = None

    # ------------------------------------------------------------------
    # Optional: Arm cue receiver (รับ confirmed target จาก 11/cam8 ผ่าน UDP)
    # ------------------------------------------------------------------
    arm_cue_receiver = None
    cam8_calib_data = None
    if ArmCueReceiver is not None and arm_controller is not None:
        _cue_port = ARM_CUE_RECEIVER_PORT
        if _config_mod is not None:
            _cue_port = getattr(_config_mod, "ARM_CUE_RECEIVER_PORT", _cue_port)
        arm_cue_receiver = ArmCueReceiver(port=_cue_port, default_ttl_ms=ARM_CUE_TTL_MS)
        arm_cue_receiver.start()

    # โหลด calibration cam8 → arm (แยกจาก cam4 calibration — ห้ามทับ)
    if _cam8_calib_load is not None and _CAM8_CALIB_FILE is not None:
        cam8_calib_data = _cam8_calib_load(_CAM8_CALIB_FILE)
        if cam8_calib_data is not None:
            print(f"[22] cam8 calibration loaded: {_CAM8_CALIB_FILE.name}")
        else:
            print(f"[22] cam8 calibration NOT found: {_CAM8_CALIB_FILE} — run cam8_arm_grid_calibrator.py first")

    # Effector WS exporter (telemetry → C2)
    effector_exporter = None
    effector_geo = None
    cam8_survey_geo = None
    effector_fov_deg = 30.0
    cam8_fov_for_cue = 180.0
    home_arm_pan = 0.0
    if cam8_calib_data is not None:
        _href = cam8_calib_data.get("home_reference") or cam8_calib_data.get("camera_reference") or {}
        home_arm_pan = float(_href.get("arm_pan", 0.0))
    if EffectorWsExporter is not None and build_effector_payload is not None and _config_mod is not None:
        effector_geo = _config_mod.get_effector_geo()
        cam8_survey_geo = _config_mod.get_cam8_survey_geo()
        effector_fov_deg = float(getattr(_config_mod, "EFFECTOR_FOV_DEG", 30.0))
        cam8_fov_for_cue = float(_config_mod.get_cam8_fov_for_cue())
        _eff_enabled = bool(getattr(_config_mod, "EFFECTOR_WS_ENABLED", False))
        _eff_log = bool(getattr(_config_mod, "EFFECTOR_WS_LOG_PAYLOAD", False))
        _eff_id = effector_geo.get("effector_id", "effector_id_1")
        _eff_url_tpl = getattr(
            _config_mod,
            "EFFECTOR_WS_URL_TEMPLATE",
            "wss://c2-api.thingsanalytic.com/ws/effector/{effector_id}/data/send",
        )
        _eff_url = _eff_url_tpl.format(effector_id=_eff_id)
        effector_exporter = EffectorWsExporter(
            ws_url=_eff_url,
            interval=float(getattr(_config_mod, "EFFECTOR_WS_INTERVAL_SEC", 0.4)),
            backoff_init=float(getattr(_config_mod, "EFFECTOR_WS_BACKOFF_INIT_SEC", 2.0)),
            backoff_max=float(getattr(_config_mod, "EFFECTOR_WS_BACKOFF_MAX_SEC", 30.0)),
            enabled=_eff_enabled,
            log_payload=_eff_log or (not _eff_enabled),
        )
        effector_exporter.start()

    # RTMP display streaming — pushes operator view to live/camnx2 (no-op when disabled)
    _rtmp_streamer = None
    _rtmp_last_ts = 0.0
    _rtmp_started = False
    _rtmp_min_interval = 0.1
    if _RtmpDisplayStreamer is not None and _config_mod is not None:
        _rtmp_enabled = bool(getattr(_config_mod, "RTMP_STREAM_ENABLED", False))
        _rtmp_url = str(getattr(_config_mod, "RTMP_STREAM_URL", "") or "").strip()
        if _rtmp_enabled and _rtmp_url:
            _rtmp_fps = float(getattr(_config_mod, "RTMP_STREAM_FPS", 10.0))
            _rtmp_min_interval = 1.0 / max(1.0, _rtmp_fps)
            _rtmp_streamer = _RtmpDisplayStreamer(
                url=_rtmp_url,
                fps=_rtmp_fps,
                bitrate=getattr(_config_mod, "RTMP_STREAM_BITRATE", "2M"),
                codec=getattr(_config_mod, "RTMP_STREAM_CODEC", "auto"),
                reconnect_max=int(getattr(_config_mod, "RTMP_STREAM_RECONNECT_MAX", 5)),
                reconnect_delay=float(getattr(_config_mod, "RTMP_STREAM_RECONNECT_DELAY", 5.0)),
                debug=bool(getattr(_config_mod, "RTMP_STREAM_DEBUG", False)),
            )

    # Init residual bias learner (ใช้ grid dims จาก cam8 calib ถ้ามี)
    global _residual_bias_learner
    _residual_bias_learner = ResidualBiasLearner(
        calib_json_path=_CAM8_CALIB_FILE if (_CAM8_CALIB_FILE is not None and _CAM8_CALIB_FILE.exists()) else None
    )

    # Init fire-tune learner (สอน aim trim จากผลยิง LOCK/sim — persist ข้าม session)
    global _fire_tune_learner
    _fire_tune_learner = FireTuneLearner(camera_name)

    cam = build_camera_from_config(camera_name)

    yolo_plan = _yolo_aim_engine_plan()
    yolo_detection_mode = yolo_plan["mode"]
    yolo_enable_class_cycle = yolo_plan["enable_class_cycle"]
    global YOLO_CLASS_NAMES
    YOLO_CLASS_NAMES = tuple(yolo_plan["class_names"])
    active_class_names = list(yolo_plan["class_names"])

    print(f"Loading YOLO model (mode={yolo_detection_mode})...")
    yolo_model, yolo_center_imgsz = load_yolo_for_aim_assist()
    if yolo_model is None:
        print("YOLO model not available. Exiting.")
        try:
            if cam is not None and hasattr(cam, "release"):
                cam.release()
        except Exception:
            pass
        return
    yolo_rgb_model = yolo_model
    yolo_primary_imgsz = yolo_center_imgsz
    yolo_infer_imgsz = yolo_center_imgsz
    yolo_alt_path = yolo_plan["alt_path"]
    yolo_thermal_model = None
    yolo_alt_imgsz = None
    if _yolo_load_alt_on_start_from_config():
        yolo_thermal_model, yolo_alt_imgsz = load_tensorrt_engine(yolo_alt_path)
        if yolo_thermal_model is None:
            if yolo_detection_mode == "drone_only":
                print(
                    "Gun Aim Assist: alternate drone engine not loaded — "
                    "imgsz toggle disabled (add 640+1280 engine files for Btn10 / [T])."
                )
            else:
                print(
                    "Gun Aim Assist: thermal engine not loaded — Detection: RGB only "
                    "(add engine file for [T] / Btn10 toggle)."
                )
    else:
        print(
            "Gun Aim Assist: alternate detection engine deferred — "
            "loads on first Btn10 / [T] press."
        )
    detection_use_thermal = False
    yolo_gen = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size()
    if DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # boresight เก็บเป็นองศา → ต้องรู้ ppd ก่อนถึงจะแปลงเป็นพิกเซลได้ → load() ถูกเลื่อนไปหลังโหลด ppd
    config = ShooterConfig(camera_name)
    _frame_geom_checked = [False]   # ตรวจขนาดเฟรมจริง vs คาลิเบรต ครั้งเดียวตอนเฟรมแรก

    if USE_SHORT_BEEP:
        beep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SOUND_FILE_SHORT)
        create_short_beep_wav(beep_path, duration_sec=BEEP_SHORT_DURATION_SEC)

    app_mode = "normal"
    jog_step_choices = [0.2, 0.5, 1.0, 2.0, 5.0]   # องศาต่อการกด 1 ครั้งในโหมด JOG
    jog_step_deg = 1.0
    # LOCK diagnostic log (เปิดด้วย env LOCK_LOG=1) — เก็บ det-rate/match-rate ต่อวินาที วิเคราะห์ real vs sim
    _lock_log = None
    if os.environ.get("LOCK_LOG") == "1":
        try:
            _lock_log = open(Path(__file__).resolve().parent / "lock_diag.log", "a")
            _lock_log.write("# t_iso,yolo_fps,det_rate,match_rate,motion_hz,lost_evt,best_conf,imgsz,arm_pan,arm_tilt,lat_ms,reacq,roi\n")
            _lock_log.flush()
            print("[LOCKLOG] เปิด — บันทึกลง lock_diag.log (เฉพาะตอนอยู่โหมด LOCK)")
        except Exception as _e:
            print(f"[LOCKLOG] open failed: {_e}")
    # per-frame aim diag (เปิดด้วย env LOCK_FRAME_LOG=1) — 15Hz วิเคราะห์ intra-second ว่า jump มาจากไหน
    if os.environ.get("LOCK_FRAME_LOG") == "1":
        try:
            _lock_frame_log[0] = open(Path(__file__).resolve().parent / "lock_frames.log", "a")
            _lock_frame_log[0].write("# t,coast_ms,arm_pan,arm_tilt,base_pan,base_tilt,aim_pan,aim_tilt,vpan,vtilt,speed,lead_gain,fire_lead,meas_pan,meas_tilt,action,resid,uncert,hit_r,errfire,confident,ready,block\n")
            _lock_frame_log[0].flush()
            print("[FRAMELOG] เปิด — บันทึก per-frame ลง lock_frames.log")
        except Exception as _e:
            print(f"[FRAMELOG] open failed: {_e}")
    _lld = {"yolo": 0, "real": 0, "match": 0, "motion": 0, "lost": 0, "bestconf": 0.0, "reacq": 0, "roi": 0, "t": time.time()}
    _lld_prev_lost = False
    cam8_stream = None
    mapping_prev_confirm = False
    pxdeg_prev_confirm = False
    mapping_canvas_w = 0
    mapping_top_h = 0
    mapping_bottom_h = 0
    last_cam8_ok_t = 0.0
    last_cam8_reconnect_t = 0.0
    mapping_has_cam8_frame = False
    mapping_cam8_status = ""
    show_calibration_status = False
    calibration_status_until = 0.0  # เวลาปิด overlay อัตโนมัติ (0 = ไม่ปิด)
    settings_selected_field = 0
    settings_step = (10, 50, 0.5, 0.05)
    settings_min_max = ((10, 500), (200, 1500), (1, 50), (0.1, 2.0))

    fps_times = deque(maxlen=30)
    frame_counter = 0
    loop_frames = 0
    GRACE_FRAMES = 60
    last_detections = []
    last_detections_all_classes = []  # ทุก class สำหรับวาด HUD (กล่องเทา + ชื่อ)
    last_target_det = None
    target_missing_count = 0  # จำนวนครั้งที่เป้าเดิมไม่โผล่ใน detection (ใช้กับ grace)
    smoothed_aim_cx = None
    smoothed_aim_cy = None
    prev_ready_to_fire = False
    last_ready_sound_time = 0.0
    last_approach_beep_time = 0.0
    last_target_pan = None
    last_target_tilt = None
    last_arm_error_deg = None
    active_class_idx = 0
    active_class_name = active_class_names[active_class_idx]
    joystick_class_cycle_initial_sec = 0.35
    joystick_class_cycle_interval_sec = 0.14
    if _config_mod is not None:
        joystick_class_cycle_initial_sec = float(
            getattr(_config_mod, "CAM4_ARM_JOYSTICK_CLASS_CYCLE_REPEAT_INITIAL_SEC", 0.35)
        )
        joystick_class_cycle_interval_sec = float(
            getattr(_config_mod, "CAM4_ARM_JOYSTICK_CLASS_CYCLE_REPEAT_INTERVAL_SEC", 0.14)
        )

    # สถานะการยิง (ใช้สำหรับเสียงปืน + flash ศูนย์เล็ง)
    is_firing = False
    last_fire_time = 0.0
    _lock_fire_flash_t = 0.0   # เวลาที่โปรแกรมลั่นไกเองล่าสุด (สำหรับ fire effect เห็นชัด)
    _lock_prev_gray = None      # เฟรมเทาก่อนหน้า (motion-assist tracker)
    _lock_prev_arm_pan = None
    _lock_prev_arm_tilt = None
    last_fire_gcode_time = 0.0  # หน่วงส่ง G-code ยิง (cooldown ระหว่างนัด)
    # Fire readiness (LOCK human-in-the-loop): SHOOT prompt + gated release
    lock_fire_ready = False
    lock_fire_safe_zone = True
    lock_hit_radius_deg = None
    lock_predict_uncert_deg = None
    lock_confident = False
    lock_pred_miss_deg = None
    lock_expected_miss_deg = None
    lock_pipe_lat = 0.0
    lock_fire_t_flight = 0.0
    lock_fire_lead_deg = 0.0
    fire_authorized = False     # ผู้ใช้กดยิงใน LOCK = อนุญาต (latch) → โปรแกรมลั่นไกเองตอน ready
    prev_fire_pressed = False   # ตรวจ edge การกดปุ่มยิง (กดครั้งเดียว = toggle)
    lock_shots_fired = 0
    lock_hits = 0
    lock_misses = 0
    pending_impacts = []        # [(impact_time, barrel_pan, barrel_tilt, t_flight, v_pan, v_tilt, lead_deg)] รอคำนวณปะทะ
    hit_flash = None            # (text, color, until_time) โชว์ผล HIT/MISS ตัวใหญ่
    _impact_log = None          # ไฟล์ log จุดกระสุนตกเทียบเป้า (โหมดจริง) — เปิด lazy ตอนยิงนัดแรก

    # edge detection สำหรับปุ่ม 12 สลับโหมดความเร็วจอยสติ๊ก (ช้า/กลาง/สูง)
    prev_sensitivity_cycle_pressed = False
    prev_class_cycle_pressed = False
    prev_detection_toggle_pressed = False
    class_cycle_next_repeat_t = 0.0
    mode_before_arm_fault = None  # restore AUTO/MANUAL/LOCK after arm reconnect
    mode_before_camera_loss = None  # restore after CAM:OK (while camera was offline)

    # เวลาใช้สำหรับ dt ในการควบคุมแขน
    last_loop_time = time.time()

    # Vector drive (AUTO/LOCK): px_per_degree, limits, arm drive state
    px_per_deg_x = None
    px_per_deg_y = None
    x_lo = y_lo = x_hi = y_hi = None
    arm_drive_state = None
    last_continuous_arm_move_time = 0.0
    if arm_controller is not None and _variable_step_toward_target is not None:
        _calib_path = _get_px_deg_path()
        # เรขาคณิตกล้องเชื่อถือไม่ได้ (ไม่มีไฟล์ / คนละความละเอียด / ppd_y เซ็นผิด / ppd=0)
        # → AUTO/LOCK ถูกบล็อก (MANUAL/SAFE ยังใช้ได้ = คนถือจอยยังเอาแขนกลับบ้านได้)
        # ที่ต้องมีนอกเหนือจาก is_file(): เปลี่ยนกล้องแล้วไฟล์ cam4 เก่ายังอยู่ = 'มีไฟล์' แต่ผิดกล้อง
        if calib_fatal:
            arm_calibration_ready = False
        if not _calib_path.is_file():
            arm_calibration_ready = False
            _force_safe_no_calib = False
            if _config_mod is not None:
                _force_safe_no_calib = bool(
                    getattr(_config_mod, "CAM4_ARM_FORCE_SAFE_WITHOUT_CALIB", False)
                )
            if _force_safe_no_calib:
                print(
                    f"⚠️ Gun Aim Assist: missing calibration for camera '{camera_name}' -> {_calib_path.name}. "
                    "Arm drive is kept in SAFE until calibration exists."
                )
            else:
                print(
                    f"⚠️ Gun Aim Assist: missing calibration for camera '{camera_name}' -> {_calib_path.name}. "
                    "Starting in MANUAL (joystick). AUTO/LOCK need calibration — run cam4_arm_mouse_grid_calibrator."
                )
        px_per_deg_x, px_per_deg_y = _load_px_per_deg()
        margin = getattr(_config_mod, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0) if _config_mod else 2.0
        xr = getattr(_config_mod, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if _config_mod else (-65.0, 65.0)
        yr = getattr(_config_mod, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if _config_mod else (-35.0, 35.0)
        x_lim = getattr(arm_controller, "_effective_x_limits", None) or (xr[0] + margin, xr[1] - margin)
        y_lim = getattr(arm_controller, "_effective_y_limits", None) or (yr[0] + margin, yr[1] - margin)
        x_lo, x_hi = x_lim[0], x_lim[1]
        y_lo, y_hi = y_lim[0], y_lim[1]
        arm_drive_state = _ArmDriveState()
        arm_drive_state.continuous_target_time_prev = time.time() - 0.05
        last_continuous_arm_move_time = time.time() - 0.05

    # ppd พร้อมแล้ว → boresight (องศา) แปลงเป็นพิกเซลได้ → โหลด config ตอนนี้
    # ถ้าไม่มีแขน ppd จะเป็น None → ShooterConfig ใช้ PPD_X_FALLBACK (แค่วาด HUD ไม่ได้ยิง)
    config.set_ppd(px_per_deg_x, px_per_deg_y)
    config.load()

    # LOCK mode: ใช้ bbox จาก YOLO เท่านั้น (IoU tracker + Kalman), ไม่ใช้ CSRT
    lock_iou_tracker = _SimpleIoUTracker()
    kb_lock_acquire = False   # keyboard 'K' one-shot: ล็อกเป้าใต้ crosshair โดยไม่ต้องใช้จอย
    lock_csrt_tracker = None
    lock_csrt_initialized = False
    lock_csrt_bbox: Optional[Tuple[int,int,int,int]] = None
    lock_csrt_smooth_px: Optional[float] = None
    lock_csrt_smooth_py: Optional[float] = None
    lock_csrt_lost: bool = False
    lock_track_frame_count: int = 0
    lock_last_arm_move_time: float = 0.0
    lock_arm_drive_state = None
    pending_lock_csrt_bbox = None
    if arm_controller is not None and _variable_step_toward_target is not None:
        lock_arm_drive_state = _ArmDriveState()
        lock_arm_drive_state.continuous_target_time_prev = time.time() - 0.05
        lock_last_arm_move_time = time.time() - 0.05
    if (
        arm_mode_manager is not None
        and arm_controller is not None
        and not arm_calibration_ready
        and _config_mod is not None
        and bool(getattr(_config_mod, "CAM4_ARM_FORCE_SAFE_WITHOUT_CALIB", False))
    ):
        arm_mode_manager.set_mode(MODE_SAFE)

    # โดรนเสมือน (virtual target) — ทดสอบ LOCK กับแขนจริง (สร้าง lazy ตอนได้ ppd/fov ในลูป)
    virtual_target = None
    _vt_enabled = bool(getattr(_config_mod, "LOCK_SIM_TARGET_ENABLED", False)) if _config_mod else False
    if _vt_enabled and VirtualDroneTarget is None:
        print("⚠️ LOCK_SIM_TARGET_ENABLED=True แต่ import lock_sim_target ไม่ได้ — ปิดโดรนเสมือน")
        _vt_enabled = False
    if _vt_enabled:
        print("🎯 LOCK_SIM_TARGET เปิดอยู่ — โดรนเสมือนจะถูกฉีดลงเฟรม (ปุ่ม I ปิด, ←→↑↓ บังคับ, V pattern, N สลับระยะ)")
    elif VirtualDroneTarget is not None:
        print("🛡️ โหมดจริง (ไม่มีโดรนจำลอง) — กดปุ่ม I เพื่อเปิดโดรนจำลองซ้อม LOCK+ยิง")

    # Cam8 cue state (อัพเดทแต่ละเฟรมจาก arm_cue_receiver; ห้ามแตะใน LOCK)
    auto_cue_source: str = "cam8_wait"   # "cam8" or "cam8_wait"

    # Yellow vector + virtual center (for drawing white/yellow arrows and cyan dot)
    loaded_yellow_scale, loaded_yellow_alpha = _load_yellow_params()
    yellow_smoothed_px = None
    yellow_scale_px = loaded_yellow_scale if loaded_yellow_scale is not None else YELLOW_GRADIENT_SCALE_PX
    yellow_smooth_alpha = loaded_yellow_alpha if loaded_yellow_alpha is not None else YELLOW_SMOOTH_ALPHA
    yellow_max_curve = _load_yellow_max_curve()
    yellow_capped = None
    vector_virtual_center_x = None
    vector_virtual_center_y = None
    vector_pan_cur = None
    vector_tilt_cur = None

    # ------------------------------------------------------------------
    # Kalman trackers (AUTO + LOCK), PD controller, state ใหม่
    # ------------------------------------------------------------------
    target_kalman = _TargetKalman()   # AUTO mode: แทน EMA
    # LOCK mode: Kalman ใน bearing space (องศา) — R ต้องมาจาก ppd ของกล้องที่ใช้จริง
    _lock_q, _lock_r = lock_bearing_kalman_qr(px_per_deg_x)
    lock_kalman = _TargetKalman(q=_lock_q, r=_lock_r)
    lock_arm_pose_hist = _ArmPoseHistory()  # ประวัติท่าแขน → หา pose ณ เวลาเก็บภาพ (ego-motion comp)
    pd_controller = _ArmPDController()

    # Kalman Coasting state
    auto_coast_start_time: float = 0.0   # เวลาที่ YOLO พลาดครั้งแรก (AUTO mode)
    lock_coast_start_time: float = 0.0   # เวลาที่ CSRT lost (LOCK mode)

    # Pipeline latency measurement (สำหรับ dynamic lead)
    import queue as _queue_mod
    import threading as _threading_mod
    pipeline_latency_history: deque = deque(maxlen=30)
    avg_pipeline_latency: float = LEAD_TIME_SEC  # initial estimate
    runtime_conf_detect: float = YOLO_CONF_DETECT  # ปรับด้วย +/- ; LOCK tracking ใช้ YOLO_CONF_LOCK คงที่
    current_yolo_conf: float = YOLO_CONF_DETECT  # conf ที่ส่งเข้า YOLO จริง (detect หรือ lock)

    # YOLO async thread queues
    _yolo_frame_q = _queue_mod.Queue(maxsize=1)
    _yolo_result_q = _queue_mod.Queue(maxsize=2)
    _yolo_t_frame: float = 0.0  # timestamp ที่ส่ง frame เข้า YOLO

    yolo_runtime = {
        "model": yolo_rgb_model,
        "imgsz": yolo_infer_imgsz,
        "lock": _threading_mod.Lock(),
    }

    def _yolo_worker(runtime, fq, rq):
        while True:
            item = fq.get()
            if item is None:
                break
            if len(item) >= 8:
                crop, x0, y0, cw, ch, t_sent, conf, gen = item[:8]
            elif len(item) >= 7:
                crop, x0, y0, cw, ch, t_sent, conf = item[:7]
                gen = 0
            else:
                crop, x0, y0, cw, ch, t_sent = item[:6]
                conf = YOLO_CONF_DETECT
                gen = 0
            with runtime["lock"]:
                m = runtime["model"]
                sz = runtime["imgsz"]
            if m is None:
                dets = []
            else:
                dets = detect_yolo_full_frame(m, crop, conf, imgsz=sz)
            try:
                rq.put_nowait((dets, x0, y0, cw, ch, t_sent, gen))
            except _queue_mod.Full:
                try:
                    rq.get_nowait()
                except _queue_mod.Empty:
                    pass
                rq.put_nowait((dets, x0, y0, cw, ch, t_sent, gen))

    _yolo_thread = _threading_mod.Thread(
        target=_yolo_worker,
        args=(yolo_runtime, _yolo_frame_q, _yolo_result_q),
        daemon=True,
    )
    _yolo_thread.start()

    # Sync GRBL throttle counter
    _sync_grbl_frame_counter: int = 0

    def _reset_target_tracking_state():
        nonlocal last_target_det, target_missing_count, smoothed_aim_cx, smoothed_aim_cy
        nonlocal lock_csrt_tracker, lock_csrt_initialized, lock_csrt_bbox, lock_csrt_smooth_px, lock_csrt_smooth_py
        nonlocal lock_csrt_lost, lock_track_frame_count, pending_lock_csrt_bbox
        last_target_det = None
        target_missing_count = 0
        smoothed_aim_cx = None
        smoothed_aim_cy = None
        lock_csrt_tracker = None
        lock_csrt_initialized = False
        lock_csrt_bbox = None
        lock_csrt_smooth_px = None
        lock_csrt_smooth_py = None
        lock_csrt_lost = False
        lock_track_frame_count = 0
        pending_lock_csrt_bbox = None
        lock_iou_tracker.reset()
        lock_kalman.reset()

    def _ensure_yolo_alt_engine() -> bool:
        nonlocal yolo_thermal_model, yolo_alt_imgsz
        if yolo_thermal_model is not None:
            return True
        print(f"Gun Aim Assist: loading alternate engine on demand ({os.path.basename(yolo_alt_path)})...")
        yolo_thermal_model, yolo_alt_imgsz = load_tensorrt_engine(yolo_alt_path)
        if yolo_thermal_model is None:
            if yolo_detection_mode == "drone_only":
                print("Gun Aim Assist: alternate imgsz engine load failed; toggle ignored.")
            else:
                print("Gun Aim Assist: thermal engine load failed; toggle ignored.")
            return False
        return True

    def _apply_detection_engine_toggle():
        nonlocal yolo_gen, detection_use_thermal, yolo_infer_imgsz
        if not _ensure_yolo_alt_engine():
            return
        detection_use_thermal = not detection_use_thermal
        next_imgsz = yolo_alt_imgsz if detection_use_thermal else yolo_primary_imgsz
        with yolo_runtime["lock"]:
            yolo_runtime["model"] = yolo_thermal_model if detection_use_thermal else yolo_rgb_model
            yolo_runtime["imgsz"] = next_imgsz
        yolo_infer_imgsz = next_imgsz
        yolo_gen += 1
        while True:
            try:
                _yolo_result_q.get_nowait()
            except _queue_mod.Empty:
                break
        _reset_target_tracking_state()
        if yolo_detection_mode == "drone_only":
            print(f"Gun Aim Assist: detection -> imgsz {next_imgsz}")
        else:
            print(f"Gun Aim Assist: detection -> {'THERMAL' if detection_use_thermal else 'RGB'}")

    def _mapping_layout_dims():
        if DISPLAY_FULLSCREEN:
            cw, ch = screen_w, screen_h
        else:
            cw = min(screen_w, DISPLAY_MAX_WIDTH)
            ch = min(screen_h, DISPLAY_MAX_HEIGHT)
        top_h = max(1, int(ch * 0.5))
        return cw, top_h, ch - top_h

    def _wait_cam8_first_frame(stream, timeout_sec=CAM8_MAP_WARMUP_SEC):
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            ok, f, _ = stream.read()
            if ok and f is not None and getattr(f, "size", 0) > 0:
                return True
            if not getattr(stream, "running", False):
                return False
            time.sleep(0.05)
        return False

    def _start_cam8_stream(force_restart=False):
        nonlocal cam8_stream, last_cam8_ok_t
        if cam8_stream is not None and not force_restart:
            ok, f, _ = cam8_stream.read()
            if ok and f is not None and getattr(cam8_stream, "running", False):
                return True
        if cam8_stream is not None:
            _stop_cam8_stream()
        for attempt in range(1, CAM8_MAP_START_RETRIES + 1):
            try:
                cam8_stream = build_camera_from_config("cam8")
                cam8_stream.start()
                if not getattr(cam8_stream, "running", False):
                    print(
                        f"[MAP] cam8 VideoCapture failed (attempt {attempt}/{CAM8_MAP_START_RETRIES}) "
                        f"— check RTSP / GStreamer / 192.168.144.112"
                    )
                    _stop_cam8_stream()
                    time.sleep(0.5)
                    continue
                print(
                    f"[MAP] cam8 pipeline OK — waiting first frame "
                    f"(up to {CAM8_MAP_WARMUP_SEC:.0f}s, H265 5120x1440)..."
                )
                if _wait_cam8_first_frame(cam8_stream):
                    last_cam8_ok_t = time.time()
                    print("[MAP] cam8 first frame received")
                    return True
                print(f"[MAP] cam8 timeout: no frame in {CAM8_MAP_WARMUP_SEC:.0f}s (attempt {attempt})")
                _stop_cam8_stream()
                time.sleep(0.5)
            except Exception as exc:
                print(f"[MAP] cam8 start failed: {exc}")
                cam8_stream = None
        return False

    def _stop_cam8_stream():
        nonlocal cam8_stream
        if cam8_stream is not None:
            try:
                cam8_stream.release()
            except Exception:
                pass
            cam8_stream = None
            print("[MAP] cam8 stream stopped")

    def _enter_cam8_mapping():
        nonlocal app_mode, mapping_prev_confirm
        nonlocal mapping_canvas_w, mapping_top_h, mapping_bottom_h
        if cam8_mapping_embed is None:
            print("[MAP] cam8_mapping_embed module not available")
            return
        if arm_controller is None:
            print("[MAP] need arm connected")
            return
        if _CAM8_CALIB_FILE is None:
            print("[MAP] calib file path unavailable")
            return
        if not _start_cam8_stream(force_restart=True):
            print("[MAP] cannot enter mapping — cam8 stream unavailable")
            return
        mapping_canvas_w, mapping_top_h, mapping_bottom_h = _mapping_layout_dims()
        cam8_mapping_embed.enter(
            WINDOW_NAME, _CAM8_CALIB_FILE, mapping_canvas_w, mapping_top_h, mapping_bottom_h
        )
        app_mode = "cam8_mapping"
        mapping_prev_confirm = False
        if arm_mode_manager is not None:
            arm_mode_manager.set_mode(MODE_MANUAL)
        cam8_mapping_embed.go_to_reference(arm_controller, _CAM8_CALIB_FILE)
        print("[MAP] Mapping ON — check REF on cam8 | click cell | C/Enter=save | H=ref | S=JSON | M/Esc=exit")

    def _exit_cam8_mapping(reload_calib=True):
        nonlocal app_mode, cam8_calib_data
        if cam8_mapping_embed is not None and cam8_mapping_embed.is_active():
            cam8_mapping_embed.leave(WINDOW_NAME)
        _stop_cam8_stream()
        app_mode = "normal"
        if reload_calib and _cam8_calib_load is not None and _CAM8_CALIB_FILE is not None:
            cam8_calib_data = _cam8_calib_load(_CAM8_CALIB_FILE)
            print(f"[MAP] calib reloaded: {'OK' if cam8_calib_data else 'MISSING'}")

    def _enter_pxdeg_calib():
        nonlocal app_mode, pxdeg_prev_confirm
        if cam4_pxdeg_calibrator_embed is None:
            print("[PXDEG] cam4_pxdeg_calibrator_embed module not available")
            return
        if arm_controller is None:
            print("[PXDEG] need arm connected")
            return
        if app_mode == "cam8_mapping":
            _exit_cam8_mapping(reload_calib=False)
        cam4_pxdeg_calibrator_embed.enter(WINDOW_NAME, arm_controller, camera_name)
        app_mode = "cam4_pxdeg_calib"
        pxdeg_prev_confirm = False
        if arm_mode_manager is not None:
            arm_mode_manager.set_mode(MODE_MANUAL)
        if joystick_mapper is not None:
            try:
                from joystick_cam4_controller import SENSITIVITY_LOW
                joystick_mapper.sensitivity_mode = SENSITIVITY_LOW
            except Exception:
                pass
        print(
            "[PXDEG] ON — click test point | ENTER/Btn0=move | joy/arrows=fine | "
            "Btn0/Enter=confirm | H=log+home | R=recal | S=save | G/Esc=exit"
        )

    def _reload_calibration(tag="CALIB"):
        """โหลด ppd ใหม่จากดิสก์ แล้ว 'ไล่คำนวณทุกอย่างที่ derive จากมัน' ใหม่

        ppd เป็นแหล่งความจริงเดียวของเรขาคณิตกล้อง → พอมันเปลี่ยน ต้องอัปเดตพร้อมกันทั้ง 3:
          FOV (→ ระยะ → hit_radius/lead/slew guard), Kalman R (= (σ/ppd)²), boresight (องศา→px)
        ถ้าอัปเดตไม่ครบ fire gate จะเพี้ยนแบบเงียบ ๆ
        """
        nonlocal arm_calibration_ready, px_per_deg_x, px_per_deg_y, fov_h, fov_v
        _calib_path = _get_px_deg_path()
        arm_calibration_ready = _calib_path.is_file()
        px_per_deg_x, px_per_deg_y = _load_px_per_deg()
        if px_per_deg_x is None or px_per_deg_y is None:
            print(f"[{tag}] calib reloaded: {'OK' if arm_calibration_ready else 'MISSING'}")
            return
        config.set_ppd(px_per_deg_x, px_per_deg_y)
        config.load()                      # boresight ของกล้องนี้ (องศา) → px ด้วย ppd ใหม่
        _fh, _fv = fov_from_ppd(w, h, px_per_deg_x, px_per_deg_y)
        if _fh and _fv:
            fov_h, fov_v = _fh, _fv
        _q, _r = lock_bearing_kalman_qr(px_per_deg_x)
        lock_kalman.set_noise(q=_q, r=_r)
        print(
            f"[{tag}] calib reloaded: {'OK' if arm_calibration_ready else 'MISSING'} "
            f"px=({px_per_deg_x:.1f},{px_per_deg_y:.1f}) "
            f"fov=({fov_h:.1f}°,{fov_v:.1f}°) kalman_r={_r:.2e}"
        )

    def _exit_pxdeg_calib(reload_calib=True):
        nonlocal app_mode
        if cam4_pxdeg_calibrator_embed is not None and cam4_pxdeg_calibrator_embed.is_active():
            cam4_pxdeg_calibrator_embed.leave(WINDOW_NAME)
        app_mode = "normal"
        if reload_calib:
            _reload_calibration("PXDEG")

    # ---------------- Calibration wizard (W) ----------------
    def _wizard_detection_center():
        """ป้อนศูนย์กลาง bbox ที่ track อยู่ให้ wizard ใช้วัด noise floor (σ ของ bbox jitter)"""
        if lock_iou_tracker is None:
            return None
        cx_t, cy_t = lock_iou_tracker.smooth_cx, lock_iou_tracker.smooth_cy
        if cx_t is None or cy_t is None or lock_iou_tracker.lost:
            return None
        return (float(cx_t), float(cy_t))

    def _enter_wizard():
        nonlocal app_mode
        if calib_wizard is None:
            print("[WIZARD] โมดูลไม่พร้อม")
            return
        if arm_controller is None:
            print("[WIZARD] ต้องมีแขนกลต่ออยู่")
            return
        # wizard ขับแขนเอง → กัน AUTO/LOCK แย่งสั่ง
        if arm_mode_manager is not None:
            arm_mode_manager.set_mode(MODE_MANUAL)
        calib_wizard.enter(WINDOW_NAME, camera_name, w, h, _wizard_detection_center)
        app_mode = "wizard"
        print("[WIZARD] ON — Space=รันขั้นตอน N/B=เปลี่ยนขั้น C=โหมดคลิก S=บันทึก W/Esc=ออก")

    def _exit_wizard(reload_calib=True):
        nonlocal app_mode
        if calib_wizard is not None and calib_wizard.is_active():
            calib_wizard.leave(WINDOW_NAME)
        app_mode = "normal"
        if reload_calib:
            _reload_calibration("WIZARD")

    # ---------------- Settings (S) ----------------
    def _enter_settings():
        nonlocal app_mode
        if settings_screen is None:
            app_mode = "settings"     # ถอยไปหน้า ballistics แบบเดิม
            return
        settings_screen.enter(WINDOW_NAME, camera_name, _config_mod, globals(), config)
        app_mode = "settings2"

    def _exit_settings():
        nonlocal app_mode
        if settings_screen is not None and settings_screen.is_active():
            settings_screen.leave(WINDOW_NAME)
        app_mode = "normal"

    print("Gun Aim Assist: starting camera stream (arm homing + YOLO ready)...")
    try:
        if hasattr(cam, "release"):
            cam.release()
    except Exception:
        pass
    cam, _first_cam_frame = open_camera_with_retries(build_camera_from_config, camera_name)
    if _first_cam_frame is None:
        print("❌ Camera/Video failed to open. Exiting.")
        if arm_controller is not None:
            try:
                arm_controller.disconnect()
            except Exception:
                pass
        return
    runtime_cleanup.set_resources(cam=cam, arm_controller=arm_controller)
    last_good_w, last_good_h = _first_cam_frame.shape[1], _first_cam_frame.shape[0]

    try:
        while True:
            t0 = time.time()
            active, frame, _ = cam.read()
            network_loss_mode = not active or frame is None
            if network_loss_mode:
                stream_health = "network loss"
                if hasattr(cam, "get_stream_health"):
                    try:
                        stream_health = cam.get_stream_health() or stream_health
                    except Exception:
                        pass
                ph_w, ph_h = last_good_w, last_good_h
                if hasattr(cam, "get_last_frame_size"):
                    try:
                        sz = cam.get_last_frame_size()
                        if sz and len(sz) == 2:
                            ph_w = int(sz[0]) or ph_w
                            ph_h = int(sz[1]) or ph_h
                    except Exception:
                        pass
                frame = network_loss_placeholder(ph_w, ph_h, camera_name, stream_health)
            else:
                last_good_w, last_good_h = frame.shape[1], frame.shape[0]

            loop_frames += 1
            js_state = None  # Filled after frame + get_center if joystick_reader.enabled.

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            h, w = frame.shape[:2]
            # เฟรมแรก: ตรวจว่าสตรีมที่ decode ได้จริง ตรงกับที่คาลิเบรตไว้ไหม
            # (config อาจพิมพ์ผิด/กล้องส่งมาคนละความละเอียด → ppd ทั้งชุดใช้ไม่ได้)
            if not _frame_geom_checked[0]:
                _frame_geom_checked[0] = True
                _g2 = check_camera_geometry(camera_name, cam_config, w, h)
                for _p in _g2["problems"]:
                    print(f"❌ Gun Aim Assist: [เฟรมจริง {w}×{h}] {_p}", flush=True)
                if _g2["fatal"] and arm_calibration_ready:
                    arm_calibration_ready = False
                    if arm_mode_manager is not None:
                        arm_mode_manager.set_mode(MODE_SAFE)
                    print(
                        "❌ Gun Aim Assist: เฟรมจริงไม่ตรงคาลิเบรต → บังคับ SAFE, บล็อก AUTO/LOCK",
                        flush=True,
                    )
            cx_frame, cy_frame = config.get_center(w, h)

            # บันทึกท่าแขนทุก loop — ใช้ interpolate หา pose ณ เวลาเก็บภาพ (LOCK bearing)
            _loop_grab_t = time.time()   # เวลา grab เฟรมนี้ — ใช้เป็น t_capture ของ motion-bridge (ตรงกับเนื้อภาพ)
            if arm_controller is not None:
                lock_arm_pose_hist.append(
                    _loop_grab_t,
                    float(getattr(arm_controller, "pos_x", 0.0)),
                    float(getattr(arm_controller, "pos_y", 0.0)),
                )

            # โดรนเสมือน: ฉีดลงเฟรมสด ก่อนส่งเข้า YOLO (ใช้มุมแขนจริง → ego-motion สมจริง)
            _vt_px = _vt_py = None; _vt_infov = False   # พิกัดโดรน sim บนจอ (ใช้วาด miss line)
            if _vt_enabled and not network_loss_mode:
                if virtual_target is None and px_per_deg_x is not None and abs(px_per_deg_x) > 1e-6:
                    virtual_target = VirtualDroneTarget(
                        ppd_x=px_per_deg_x, ppd_y=px_per_deg_y,
                        fov_h=fov_h, fov_v=fov_v,
                        pattern=getattr(_config_mod, "LOCK_SIM_TARGET_PATTERN", "realistic"),
                        omega_deg_s=float(getattr(_config_mod, "LOCK_SIM_TARGET_OMEGA_DEG_S", 8.0)),
                        amp_deg=float(getattr(_config_mod, "LOCK_SIM_TARGET_AMP_DEG", 15.0)),
                        tilt_amp_deg=float(getattr(_config_mod, "LOCK_SIM_TARGET_TILT_AMP_DEG", 5.0)),
                        box_deg=float(getattr(_config_mod, "LOCK_SIM_TARGET_BOX_DEG", 0.6)),
                        miss_rate=float(getattr(_config_mod, "LOCK_SIM_TARGET_MISS_RATE", 0.15)),
                        cls_id=int(getattr(_config_mod, "LOCK_SIM_TARGET_CLASS_ID", 0)),
                        range_m=float(getattr(_config_mod, "LOCK_SIM_TARGET_RANGE_M", 150.0)),
                        target_size_m=float(getattr(_config_mod, "LOCK_SIM_TARGET_SIZE_M", 0.30)),
                        max_speed_ms=float(getattr(_config_mod, "LOCK_SIM_TARGET_MAX_SPEED_MS", 18.0)),
                        max_accel_ms2=float(getattr(_config_mod, "LOCK_SIM_TARGET_MAX_ACCEL_MS2", 12.0)),
                    )
                    print(f"[SIMDRONE] range={virtual_target.range_m:.0f}m box={virtual_target.box_deg:.3f}deg "
                          f"max_ang_speed={virtual_target.max_ang_speed:.1f}deg/s pattern={virtual_target.pattern}", flush=True)
                if virtual_target is not None:
                    _vt_px, _vt_py, _vt_infov = virtual_target.on_frame(
                        frame,
                        float(getattr(arm_controller, "pos_x", 0.0)) if arm_controller is not None else 0.0,
                        float(getattr(arm_controller, "pos_y", 0.0)) if arm_controller is not None else 0.0,
                        cx_frame, cy_frame, time.time(),
                        draw=bool(getattr(_config_mod, "LOCK_SIM_TARGET_DRAW", True)),
                    )

            if joystick_reader is not None:
                joystick_reader.tick(t0)
                if getattr(joystick_reader, "enabled", False):
                    try:
                        js_state = joystick_reader.read()
                    except Exception:
                        js_state = None
            if arm_controller is not None and hasattr(arm_controller, "note_joystick_operator_active"):
                _joy_dz = float(
                    getattr(_config_mod, "CAM4_ARM_JOYSTICK_DEADZONE", 0.05)
                    if _config_mod
                    else 0.05
                )
                if js_state is not None:
                    _joy_active = (
                        abs(float(getattr(js_state, "axis_x", 0.0))) > _joy_dz
                        or abs(float(getattr(js_state, "axis_y", 0.0))) > _joy_dz
                    )
                else:
                    _joy_active = False
                arm_controller.note_joystick_operator_active(_joy_active)

            min_side = min(h, w)
            radius_px = int(min_side * RETICLE_RADIUS_RATIOS[0])
            if CENTER_RADIUS_PX > 0:
                radius_px = CENTER_RADIUS_PX
            radius_px = max(10, radius_px)

            # dt สำหรับการคำนวณความเร็วหมุนของแขน (manual)
            now_loop = time.time()
            dt_loop = now_loop - last_loop_time if now_loop > last_loop_time else 0.0
            last_loop_time = now_loop

            camera_operator_lock = _camera_operator_lock_active(cam, network_loss_mode)
            if arm_mode_manager is not None:
                mode_before_camera_loss = _tick_camera_loss_mode_safe(
                    cam,
                    network_loss_mode,
                    arm_mode_manager,
                    mode_before_camera_loss,
                    arm_controller,
                )
                camera_operator_lock = _camera_operator_lock_active(cam, network_loss_mode)

            if arm_controller is not None and hasattr(arm_controller, "set_camera_operator_lock"):
                arm_controller.set_camera_operator_lock(camera_operator_lock)

            if network_loss_mode:
                if arm_controller is not None and hasattr(arm_controller, "maybe_check_arm_liveness"):
                    arm_controller.maybe_check_arm_liveness()
                if arm_controller is not None and hasattr(arm_controller, "maybe_verify_drive_motion"):
                    arm_controller.maybe_verify_drive_motion()
                # ระหว่าง JOG: กันไม่ให้ fault monitor วน re-home แย่งขณะผู้ใช้หมุนแขนแก้อาการค้าง
                if app_mode == "jog":
                    pass
                elif arm_controller is not None and hasattr(arm_controller, "maybe_handle_arm_fault_recovery"):
                    arm_controller.maybe_handle_arm_fault_recovery(
                        camera_operator_lock=camera_operator_lock
                    )
                elif arm_controller is not None and hasattr(arm_controller, "maybe_handle_stall_recovery"):
                    arm_controller.maybe_handle_stall_recovery(
                        camera_operator_lock=camera_operator_lock
                    )
                if arm_mode_manager is not None:
                    mode_before_arm_fault = _tick_arm_mode_fault_and_restore(
                        arm_controller,
                        arm_mode_manager,
                        mode_before_arm_fault,
                        camera_operator_lock=camera_operator_lock,
                    )
                _cam_lbl, _cam_col = cam_hud_label_and_color(cam, False, None)
                _arm_lbl, _arm_col = _arm_hud_label_and_color(arm_controller)
                _joy_lbl, _joy_col = _joy_hud_label_and_color(joystick_reader)
                _hud_ctx = {
                    "fps": 0.0,
                    "lat_ms": 0.0,
                    "has_lat": False,
                    "conf": runtime_conf_detect,
                    "class_name": _hud_class_display_name(active_class_name) if active_class_name else None,
                    "yolo_thermal_model": yolo_thermal_model,
                    "yolo_detection_mode": yolo_detection_mode,
                    "detection_use_thermal": detection_use_thermal,
                    "yolo_alt_imgsz": yolo_alt_imgsz,
                    "yolo_primary_imgsz": yolo_primary_imgsz,
                    "cam_label": _cam_lbl,
                    "cam_color": _cam_col,
                    "arm_label": _arm_lbl,
                    "arm_color": _arm_col,
                    "joy_label": _joy_lbl,
                    "joy_color": _joy_col,
                    "spd_label": "SPD:---",
                    "spd_color": (120, 120, 120),
                }
                if arm_mode_manager is not None:
                    _mode_label, _mode_color = arm_mode_manager.label_and_color()
                    _hud_ctx["mode_label"] = _mode_label
                    _hud_ctx["mode_color"] = _mode_color
                draw_gun_bottom_hud(frame, _hud_ctx)
            elif app_mode == "normal":
                frame_counter += 1
                lock_csrt_active = (
                    arm_mode_manager is not None
                    and arm_mode_manager.mode == MODE_LOCK
                    and lock_csrt_initialized
                    and not lock_csrt_lost
                )
                # conf: LOCK tracking ใช้ YOLO_CONF_LOCK คงที่; detection ใช้ runtime_conf_detect (+/-)
                _yolo_conf = (
                    YOLO_CONF_LOCK
                    if (
                        arm_mode_manager is not None
                        and arm_mode_manager.mode == MODE_LOCK
                        and lock_iou_tracker.initialized
                        and not lock_iou_tracker.lost
                    )
                    else runtime_conf_detect
                )
                current_yolo_conf = _yolo_conf

                # --- YOLO async: ส่ง frame เข้า thread (รันทั้ง AUTO และ LOCK, ใช้ conf ตามโหมด) ---
                if frame_counter % YOLO_INTERVAL == 0:
                    # LOCK + track มีตำแหน่ง → ROI native crop ตามเป้า (โดรนคมชัดเต็ม native,
                    # อยู่ในหน้าต่างเสมอแม้ lead-aim เยื้องกลาง) — ปิดช่องว่าง sim-vs-real.
                    # YOLO ไม่ match → 'ขยาย ROI' ค้นกว้างขึ้น (ไม่ถอย center crop ที่ตาบอด);
                    # ไม่ล็อก/หลุด → center crop เดิม (AUTO + re-detect ไม่เปลี่ยนพฤติกรรม)
                    _lock_roi_used = False
                    if (LOCK_ROI_ENABLED and lock_csrt_active
                            and lock_iou_tracker.smooth_cx is not None
                            and lock_iou_tracker.smooth_cy is not None):
                        _bb = lock_iou_tracker.track_bbox
                        _bb_diag = math.hypot(_bb[2], _bb[3]) if _bb is not None else None
                        center_crop, x0_crop, y0_crop, cw_crop, ch_crop = crop_roi_native(
                            frame, lock_iou_tracker.smooth_cx, lock_iou_tracker.smooth_cy,
                            yolo_infer_imgsz, bbox_diag=_bb_diag,
                            miss_run=lock_iou_tracker._yolo_miss_run,
                            ppd=px_per_deg_x,
                        )
                        _lock_roi_used = True
                    else:
                        center_crop, x0_crop, y0_crop, cw_crop, ch_crop = crop_center_and_resize(
                            frame, CENTER_CROP_RATIO, yolo_infer_imgsz
                        )
                    if _lock_log is not None and _lock_roi_used:
                        _lld["roi"] += 1
                    _t_sent = time.time()
                    try:
                        _yolo_frame_q.put_nowait((
                            center_crop, x0_crop, y0_crop, cw_crop, ch_crop, _t_sent, _yolo_conf, yolo_gen
                        ))
                    except _queue_mod.Full:
                        pass  # YOLO ยังไม่ว่าง drop frame นี้

                # --- รับ YOLO result จาก thread (non-blocking) — ทิ้งผลที่ gen ไม่ตรงหลังสลับ engine ---
                try:
                    _picked = None
                    while True:
                        _yolo_res = _yolo_result_q.get_nowait()
                        if _yolo_res[-1] == yolo_gen:
                            _picked = _yolo_res
                except _queue_mod.Empty:
                    pass
                _yolo_matched_frame = False   # YOLO match เป้าใน LOCK เฟรมนี้ไหม (ใช้ตัดสินว่าจะรัน motion-assist)
                if _picked is not None:
                    _dets_crop, _x0, _y0, _cw, _ch, _t_yolo_sent, _res_gen = _picked
                    _all_detections = map_detections_to_full_frame(
                        _dets_crop, _x0, _y0, _cw, _ch, yolo_infer_imgsz
                    )
                    last_detections_all_classes = _all_detections
                    last_detections = filter_detections_by_active_class(_all_detections, active_class_idx)
                    # LOCK diag: นับ detection จริง (ก่อนฉีด sim) + conf สูงสุด
                    if _lock_log is not None:
                        _lld["yolo"] += 1
                        _rc = len(last_detections) if last_detections else 0
                        if _rc > 0:
                            _lld["real"] += 1
                            try:
                                _lld["bestconf"] = max(_lld["bestconf"],
                                                       max(float(_det_xywhc(_d)[4]) for _d in last_detections))
                            except Exception:
                                pass
                    # โดรนเสมือน: ฉีด detection ที่ตำแหน่ง ณ เฟรมที่ YOLO ประมวลผล (_t_yolo_sent)
                    # → ไหลผ่าน pipeline จริง (latency จริง) + bearing คิดจาก pose ณ เก็บภาพ (ไม่ double-count)
                    if (
                        virtual_target is not None
                        and bool(getattr(_config_mod, "LOCK_SIM_TARGET_INJECT_DETECTION", True))
                    ):
                        _vt_det = virtual_target.detection_at_capture(_t_yolo_sent)
                        if _vt_det is not None:
                            last_detections = list(last_detections) + [_vt_det]
                            last_detections_all_classes = list(last_detections_all_classes) + [_vt_det]
                    # วัด pipeline latency
                    _lat = time.time() - _t_yolo_sent
                    if 0.01 < _lat < 2.0:
                        pipeline_latency_history.append(_lat)
                        if pipeline_latency_history:
                            avg_pipeline_latency = sum(pipeline_latency_history) / len(pipeline_latency_history)
                    # Sticky target + grace
                    sticky_radius_px = max(40, int(min_side * TARGET_STICKY_RADIUS_RATIO))
                    _latest_cue_for_pick = arm_cue_receiver.get_latest_cue() if arm_cue_receiver is not None else None
                    if last_target_det is None:
                        last_target_det = pick_best_target_with_cue(
                            last_detections, cx_frame, cy_frame, min_side,
                            optional_cue=_latest_cue_for_pick, frame_w=w, frame_h=h,
                        )
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
                                last_target_det = pick_best_target_with_cue(
                                    last_detections, cx_frame, cy_frame, min_side,
                                    optional_cue=_latest_cue_for_pick, frame_w=w, frame_h=h,
                                )
                                target_missing_count = 0
                    # LOCK: อัปเดต IoU tracker จาก YOLO detections (ใช้ bbox จาก YOLO เท่านั้น)
                    # match แล้วแปลงเป็น absolute bearing ป้อน lock_kalman (ego-motion comp)
                    if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK and lock_iou_tracker.initialized and last_detections is not None:
                        _lock_matched, _ = lock_iou_tracker.update(
                            last_detections, w, h,
                            cur_pan=getattr(arm_controller, "pos_x", None) if arm_controller is not None else None,
                            cur_tilt=getattr(arm_controller, "pos_y", None) if arm_controller is not None else None,
                            ppd_x=px_per_deg_x, ppd_y=px_per_deg_y,
                        )
                        if _lock_matched:
                            _yolo_matched_frame = True
                            if _lock_log is not None:
                                _lld["match"] += 1
                                if lock_iou_tracker.last_match_was_reacquire:
                                    _lld["reacq"] += 1
                            _lock_feed_bearing_measurement(
                                lock_iou_tracker, lock_kalman, lock_arm_pose_hist, arm_controller,
                                _t_yolo_sent, cx_frame, cy_frame, px_per_deg_x, px_per_deg_y,
                                ego_comp_latency,
                            )

                # --- Motion-assist tracker (ทุกเฟรม, LOCK) — ประคอง track ระหว่าง YOLO บอด/เบลอ ---
                # โดรนขยับตลอด → หักล้างการหมุนกล้อง (encoder แขน) แล้ว frame-diff หา blob ที่ขยับ
                # ทำเฉพาะเฟรมที่ YOLO ไม่ได้ match (YOLO มา override ได้เมื่อภาพชัด) → LOCK ไม่หลุด
                if (LOCK_MOTION_ASSIST_ENABLED and arm_mode_manager is not None
                        and arm_mode_manager.mode == MODE_LOCK and lock_iou_tracker.initialized
                        and px_per_deg_x and px_per_deg_y):
                    _cp = float(getattr(arm_controller, "pos_x", 0.0)) if arm_controller is not None else 0.0
                    _ct = float(getattr(arm_controller, "pos_y", 0.0)) if arm_controller is not None else 0.0
                    _gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if (not _yolo_matched_frame and _lock_prev_gray is not None
                            and _lock_prev_arm_pan is not None
                            and lock_iou_tracker.smooth_cx is not None
                            and _lock_prev_gray.shape == _gray_cur.shape):
                        _msx = -(_cp - _lock_prev_arm_pan) * px_per_deg_x
                        _msy = -(_ct - _lock_prev_arm_tilt) * px_per_deg_y
                        _mres = _lock_motion_track(
                            _gray_cur, _lock_prev_gray,
                            lock_iou_tracker.smooth_cx + _msx, lock_iou_tracker.smooth_cy + _msy,
                            _msx, _msy,
                            ppd=px_per_deg_x,
                        )
                        if _mres is not None:
                            lock_iou_tracker.apply_motion_measurement(_mres[0], _mres[1])
                            # ป้อน bearing จาก motion เข้า lock_kalman (ขับแขน) เมื่อ:
                            #  - LOCK_MOTION_DRIVES_ARM (เดิม, ตลอดเวลา) — ปิดไว้กันจิตเตอร์ตอน track ปกติ, หรือ
                            #  - บริดจ์ gap: YOLO บอดยาว (_yolo_miss_run ≥ BRIDGE_MIN_MISS) → ให้แขนตามเป้าต่อ
                            #    ระหว่าง RGB หลุด (ไม่ค้างให้เป้าลอยหนี). outlier gate + deadband ใน feed กัน
                            #    motion เกาะ background พาแขนหลง (measurement กระโดด = ถูกทิ้ง)
                            _motion_bridge = (LOCK_MOTION_BRIDGE_ENABLED
                                              and lock_iou_tracker._yolo_miss_run >= LOCK_MOTION_BRIDGE_MIN_MISS)
                            if LOCK_MOTION_DRIVES_ARM or _motion_bridge:
                                _lock_feed_bearing_measurement(
                                    lock_iou_tracker, lock_kalman, lock_arm_pose_hist, arm_controller,
                                    _loop_grab_t, cx_frame, cy_frame, px_per_deg_x, px_per_deg_y,
                                    ego_comp_latency,
                                )
                            if _lock_log is not None:
                                _lld["motion"] += 1
                    _lock_prev_gray = _gray_cur
                    _lock_prev_arm_pan = _cp
                    _lock_prev_arm_tilt = _ct
                else:
                    _lock_prev_gray = None   # ออกจาก LOCK/ไม่ init → รีเซ็ต กัน diff ข้ามช่วง

                # Sync LOCK state จาก IoU tracker (ทุกเฟรม) — ข้ามเมื่อ LOCK_DISABLE_TRACKING เพื่อไม่ทับ lock_csrt_initialized/lost หลัง one-shot
                if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK and not LOCK_DISABLE_TRACKING:
                    lock_csrt_initialized = lock_iou_tracker.initialized
                    lock_csrt_lost = lock_iou_tracker.lost
                    lock_csrt_bbox = lock_iou_tracker.track_bbox
                    lock_csrt_smooth_px = lock_iou_tracker.smooth_cx
                    lock_csrt_smooth_py = lock_iou_tracker.smooth_cy

                # LOCK diag: นับ lost transition + flush สรุปทุก 1 วินาที (เฉพาะโหมด LOCK)
                if _lock_log is not None and arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK:
                    if lock_csrt_lost and not _lld_prev_lost:
                        _lld["lost"] += 1
                    _lld_prev_lost = lock_csrt_lost
                    _now_ll = time.time()
                    if _now_ll - _lld["t"] >= 1.0:
                        _yf = max(1, _lld["yolo"])
                        _lock_log.write(
                            f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                            f"{_lld['yolo'] / (_now_ll - _lld['t']):.1f},"
                            f"{_lld['real'] / _yf:.2f},{_lld['match'] / _yf:.2f},"
                            f"{_lld['motion'] / (_now_ll - _lld['t']):.1f},"
                            f"{_lld['lost']},{_lld['bestconf']:.2f},{yolo_infer_imgsz},"
                            f"{getattr(arm_controller, 'pos_x', 0.0):.1f},"
                            f"{getattr(arm_controller, 'pos_y', 0.0):.1f},"
                            f"{avg_pipeline_latency * 1000:.0f},"
                            f"{_lld['reacq']},{_lld['roi']}\n"
                        )
                        _lock_log.flush()
                        _lld = {"yolo": 0, "real": 0, "match": 0, "motion": 0, "lost": 0, "bestconf": 0.0, "reacq": 0, "roi": 0, "t": _now_ll}

                target_det = last_target_det
                if lock_csrt_active:
                    target_det = None
                ready_to_fire = False
                # เข้า block เมื่อมี target จาก YOLO, LOCK+CSRT กำลัง track หรือ LOCK มี pending (ปุ่ม 4) เพื่อให้ consume pending และ init CSRT
                lock_has_pending = (
                    arm_mode_manager is not None
                    and arm_mode_manager.mode == MODE_LOCK
                    and pending_lock_csrt_bbox is not None
                )

                # --- Kalman AUTO mode: แทน EMA ---
                # Update cam4-based tracking state when detections/lock state exist.
                # NOTE: AUTO arm drive dispatch is handled below every frame and is cam8-driven.
                if target_det is not None or lock_csrt_active or lock_has_pending:
                    if target_det is not None:
                        x, y, w_d, h_d, conf_det = _det_xywhc(target_det)
                        cx_t = x + w_d // 2
                        cy_t = y + h_d // 2
                        # Kalman update แทน EMA (confidence-weighted)
                        smoothed_aim_cx, smoothed_aim_cy = target_kalman.update(cx_t, cy_t, conf_det)
                        auto_coast_start_time = time.time()  # reset coast timer
                        d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                        ready_to_fire = d <= radius_px
                    elif (
                        arm_mode_manager is not None
                        and arm_mode_manager.mode == MODE_AUTO
                        and target_kalman._initialized
                    ):
                        # AUTO Kalman Coasting: YOLO พลาด ให้ predict ต่อ (bounded — ยุบ velocity กัน overshoot)
                        coast_elapsed = time.time() - auto_coast_start_time
                        if coast_elapsed < AUTO_KALMAN_COAST_MAX_SEC:
                            smoothed_aim_cx, smoothed_aim_cy = target_kalman.predict_ahead(
                                coast_elapsed, decay_tau=KALMAN_COAST_DECAY_TAU)
                            d = math.sqrt((smoothed_aim_cx - cx_frame) ** 2 + (smoothed_aim_cy - cy_frame) ** 2)
                            ready_to_fire = d <= radius_px
                    elif (
                        arm_mode_manager is not None
                        and arm_mode_manager.mode == MODE_LOCK
                        and lock_csrt_smooth_px is not None
                        and lock_csrt_smooth_py is not None
                    ):
                        # LOCK: เป้าเล็ง = ศูนย์ CSRT อยู่ภายใน radius_px จากศูนย์จอ
                        d_lock = math.sqrt((lock_csrt_smooth_px - cx_frame) ** 2 + (lock_csrt_smooth_py - cy_frame) ** 2)
                        ready_to_fire = d_lock <= radius_px

                # Vector drive: dispatch to mode tick (AUTO must run even when cam4 has no target_det).
                if arm_controller is not None and arm_mode_manager is not None and px_per_deg_x is not None and px_per_deg_y is not None and x_lo is not None:
                    _ctx = {
                        "arm_controller": arm_controller,
                        "arm_drive_state": arm_drive_state,
                        "target_kalman": target_kalman,
                        "pd_controller": pd_controller,
                        "smoothed_aim_cx": smoothed_aim_cx,
                        "smoothed_aim_cy": smoothed_aim_cy,
                        "cx_frame": cx_frame,
                        "cy_frame": cy_frame,
                        "px_per_deg_x": px_per_deg_x,
                        "px_per_deg_y": px_per_deg_y,
                        "x_lo": x_lo, "y_lo": y_lo, "x_hi": x_hi, "y_hi": y_hi,
                        "last_continuous_arm_move_time": last_continuous_arm_move_time,
                        "avg_pipeline_latency": avg_pipeline_latency,
                        "_sync_grbl_frame_counter": _sync_grbl_frame_counter,
                        "frame": frame, "w": w, "h": h,
                        # cam8 cue path (AUTO only; LOCK does not use these values)
                        "latest_cam8_cue": arm_cue_receiver.get_latest_cue() if arm_cue_receiver is not None else None,
                        "cam8_calib_data": cam8_calib_data,
                        "pending_lock_csrt_bbox": pending_lock_csrt_bbox,
                        "lock_csrt_initialized": lock_csrt_initialized,
                        "lock_csrt_tracker": lock_csrt_tracker,
                        "lock_csrt_smooth_px": lock_csrt_smooth_px,
                        "lock_csrt_smooth_py": lock_csrt_smooth_py,
                        "lock_csrt_bbox": lock_csrt_bbox,
                        "lock_csrt_lost": lock_csrt_lost,
                        "lock_track_frame_count": lock_track_frame_count,
                        "lock_last_arm_move_time": lock_last_arm_move_time,
                        "lock_arm_drive_state": lock_arm_drive_state,
                        "lock_kalman": lock_kalman,
                        "target_det": target_det,
                        "joystick_mapper": joystick_mapper,
                        "js_state": js_state,
                        "dt_loop": dt_loop,
                        "arm_mode": arm_mode_manager.mode,
                        # firing solution: ประเมินระยะจากขนาด bbox ที่ track (สำหรับ intercept lead)
                        # ใช้ ppd ที่คาลิเบรตจริง ไม่ผ่าน FOV ที่พิมพ์มือ — range คุม t_flight,
                        # ballistic lead, hit_radius_deg, lead attenuation และ slew guard พร้อมกันหมด
                        "lock_range_m": (
                            estimate_distance_m_ppd(
                                lock_csrt_bbox[2], lock_csrt_bbox[3],
                                px_per_deg_x, px_per_deg_y, config.target_size_m
                            ) if lock_csrt_bbox is not None else None
                        ) or config.effective_range_m,
                        "muzzle_velocity_ms": config.muzzle_velocity_ms,
                        "target_size_m": config.target_size_m,
                        # learned aim trim (สอนด้วยปุ่ม y จากผลยิงก่อนหน้า)
                        "fire_trim_pan": (_fire_tune_learner.trim_pan if _fire_tune_learner else 0.0),
                        "fire_trim_tilt": (_fire_tune_learner.trim_tilt if _fire_tune_learner else 0.0),
                        # ผู้ใช้กดยิงใน LOCK แล้ว → เปิด snap-and-shoot (กระโดดไป intercept แล้วยิง)
                        "fire_authorized": fire_authorized,
                    }
                    if app_mode == "jog":
                        pass  # JOG: ขับแขนด้วยคีย์บอร์ดเท่านั้น — ข้าม auto/lock/manual tick
                    elif arm_mode_manager.mode == MODE_AUTO:
                        _tick_auto(_ctx)
                    elif arm_mode_manager.mode == MODE_LOCK:
                        _tick_lock(_ctx)
                    elif arm_mode_manager.mode == MODE_MANUAL:
                        _tick_manual(_ctx)
                    elif arm_mode_manager.mode == MODE_SAFE:
                        _tick_safe(_ctx)
                    last_continuous_arm_move_time = _ctx.get("last_continuous_arm_move_time", last_continuous_arm_move_time)
                    _sync_grbl_frame_counter = _ctx.get("_sync_grbl_frame_counter", _sync_grbl_frame_counter)
                    auto_cue_source = _ctx.get("auto_cue_source", auto_cue_source)
                    # periodic bias save (respects internal interval)
                    if _residual_bias_learner is not None and CAM8_AUTO_ONLINE_BIAS_ENABLED:
                        _residual_bias_learner.save()
                    pending_lock_csrt_bbox = _ctx.get("pending_lock_csrt_bbox", pending_lock_csrt_bbox)
                    lock_csrt_initialized = _ctx.get("lock_csrt_initialized", lock_csrt_initialized)
                    lock_csrt_tracker = _ctx.get("lock_csrt_tracker", lock_csrt_tracker)
                    lock_csrt_smooth_px = _ctx.get("lock_csrt_smooth_px", lock_csrt_smooth_px)
                    lock_csrt_smooth_py = _ctx.get("lock_csrt_smooth_py", lock_csrt_smooth_py)
                    lock_csrt_bbox = _ctx.get("lock_csrt_bbox", lock_csrt_bbox)
                    lock_csrt_lost = _ctx.get("lock_csrt_lost", lock_csrt_lost)
                    lock_track_frame_count = _ctx.get("lock_track_frame_count", lock_track_frame_count)
                    lock_last_arm_move_time = _ctx.get("lock_last_arm_move_time", lock_last_arm_move_time)
                    last_target_pan = _ctx.get("last_target_pan", last_target_pan)
                    last_target_tilt = _ctx.get("last_target_tilt", last_target_tilt)
                    last_arm_error_deg = _ctx.get("last_arm_error_deg", last_arm_error_deg)
                    # firing readiness (LOCK): โปรแกรมคำนวณว่ายิงตอนนี้จะโดนไหม
                    lock_fire_ready = bool(_ctx.get("lock_fire_ready", False))
                    lock_fire_safe_zone = bool(_ctx.get("lock_fire_safe_zone", True))
                    lock_hit_radius_deg = _ctx.get("lock_hit_radius_deg")
                    lock_predict_uncert_deg = _ctx.get("lock_predict_uncert_deg")
                    lock_confident = bool(_ctx.get("lock_confident", False))
                    lock_pred_miss_deg = _ctx.get("lock_pred_miss_deg")
                    lock_expected_miss_deg = _ctx.get("lock_expected_miss_deg")
                    lock_pipe_lat = _ctx.get("lock_pipe_lat", 0.0)
                    lock_fire_t_flight = _ctx.get("lock_fire_t_flight", 0.0)
                    lock_fire_lead_deg = _ctx.get("lock_fire_lead_deg", 0.0)

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
                    x_t, y_t, w_d, h_d, _ = _det_xywhc(target_det)
                    distance_m = estimate_distance_m_ppd(
                        w_d, h_d, px_per_deg_x, px_per_deg_y, config.target_size_m
                    )
                if (
                    effector_exporter is not None
                    and build_effector_payload is not None
                    and effector_geo is not None
                    and cam8_survey_geo is not None
                    and arm_controller is not None
                    and arm_mode_manager is not None
                ):
                    _latest_cue = arm_cue_receiver.get_latest_cue() if arm_cue_receiver is not None else None
                    _eff_payload = build_effector_payload(
                        getattr(arm_controller, "pos_x", 0.0),
                        getattr(arm_controller, "pos_y", 0.0),
                        home_arm_pan,
                        last_target_pan,
                        last_target_tilt,
                        ready_to_fire,
                        arm_mode_manager.mode,
                        _latest_cue,
                        target_det,
                        distance_m,
                        lock_csrt_initialized,
                        lock_csrt_lost,
                        lock_csrt_smooth_px,
                        lock_csrt_smooth_py,
                        cx_frame,
                        cy_frame,
                        w,
                        h,
                        px_per_deg_x,
                        px_per_deg_y,
                        ARM_CUE_TTL_MS,
                        effector_geo,
                        cam8_survey_geo,
                        effector_fov_deg,
                        cam8_fov_for_cue,
                    )
                    effector_exporter.push(_eff_payload)
                draw_hud(
                    frame,
                    cx_frame,
                    cy_frame,
                    radius_px,
                    target_det,
                    ready_to_fire,
                    guide_h,
                    guide_v,
                    is_day=is_daytime(),
                    distance_m=distance_m,
                    all_detections=last_detections_all_classes,
                    active_class_name=active_class_name,
                    active_class_idx=active_class_idx,
                )

                # --- Fire-solution HUD (LOCK): SHOOT ใหญ่เมื่อคำนวณว่ายิงโดนแน่ ---
                if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK and LOCK_FIRE_SOLUTION_ENABLED:
                    _fh, _fw = frame.shape[:2]
                    _fs = max(1.2, _fw / 900.0)   # font scale ตามความละเอียด
                    _th = max(2, int(_fs * 2))
                    # --- HIT ZONE ring: รัศมี = hit_radius × ppd รอบศูนย์เล็ง (= แนวลำกล้อง).
                    # โดรนอยู่ในวง ⟺ angular miss ≤ hit_radius ⟺ นับ HIT. ทำให้เห็นว่าทำไมโดน/ไม่โดน
                    if lock_hit_radius_deg is not None and px_per_deg_x:
                        _hz_r = int(abs(lock_hit_radius_deg * px_per_deg_x))
                        if _hz_r > 2:
                            cv2.circle(frame, (int(cx_frame), int(cy_frame)), _hz_r,
                                       (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"HIT ZONE {lock_hit_radius_deg:.2f}deg",
                                        (int(cx_frame) - _hz_r, int(cy_frame) - _hz_r - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, _fs * 0.4, (0, 255, 255),
                                        max(1, _th // 2), cv2.LINE_AA)
                            # เส้น + ค่ามุม miss จริง จากศูนย์เล็งไปตัวโดรน sim (in/out ของวง = โดน/พลาด)
                            if _vt_px is not None and _vt_infov and px_per_deg_x and px_per_deg_y:
                                _mdx = (_vt_px - cx_frame) / px_per_deg_x
                                _mdy = (_vt_py - cy_frame) / px_per_deg_y
                                _mdeg = math.hypot(_mdx, _mdy)
                                _inzone = _mdeg <= lock_hit_radius_deg
                                _mcol = (0, 255, 0) if _inzone else (0, 0, 255)
                                cv2.line(frame, (int(cx_frame), int(cy_frame)),
                                         (int(_vt_px), int(_vt_py)), _mcol, 2, cv2.LINE_AA)
                                cv2.putText(frame, f"miss {_mdeg:.2f}deg {'IN' if _inzone else 'OUT'}",
                                            (int(_vt_px) + 8, int(_vt_py)),
                                            cv2.FONT_HERSHEY_SIMPLEX, _fs * 0.45, _mcol,
                                            max(1, _th // 2), cv2.LINE_AA)
                    if lock_fire_ready:
                        _flash = int(time.time() * 4) % 2 == 0   # กะพริบ ~2Hz
                        _col = (0, 255, 0) if _flash else (0, 180, 0)
                        _txt = ">>> SHOOT <<<" if fire_authorized else "SHOOT"
                        _sz, _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_DUPLEX, _fs * 2.0, _th + 1)
                        _tx = int((_fw - _sz[0]) / 2)
                        _ty = int(_fh * 0.16)
                        cv2.putText(frame, _txt, (_tx, _ty), cv2.FONT_HERSHEY_DUPLEX,
                                    _fs * 2.0, (0, 0, 0), _th + 4, cv2.LINE_AA)
                        cv2.putText(frame, _txt, (_tx, _ty), cv2.FONT_HERSHEY_DUPLEX,
                                    _fs * 2.0, _col, _th + 1, cv2.LINE_AA)
                        # กรอบเขียวรอบศูนย์เล็ง = solution ล็อกโดน
                        _bx = int(cx_frame); _by = int(cy_frame); _r = int(radius_px * 1.6)
                        cv2.rectangle(frame, (_bx - _r, _by - _r), (_bx + _r, _by + _r), _col, _th)
                    elif fire_authorized:
                        # armed แต่ยังคำนวณว่าจะยิงโดน — โปรแกรม LOCK+คำนวณต่อ, ยังไม่ยิง (ไม่มี effect)
                        # โชว์ banner ชัด ๆ กลางจอว่า 'จะยิงออโต้เมื่อคำนวณเสร็จ'
                        _pulse = int(time.time() * 3) % 2 == 0
                        _abcol = (0, 220, 255) if _pulse else (0, 150, 200)
                        _abtxt = "AUTO-FIRE ARMED - computing solution..."
                        _absz, _ = cv2.getTextSize(_abtxt, cv2.FONT_HERSHEY_DUPLEX, _fs * 0.95, _th)
                        _abx = int((_fw - _absz[0]) / 2); _aby = int(_fh * 0.16)
                        cv2.putText(frame, _abtxt, (_abx, _aby), cv2.FONT_HERSHEY_DUPLEX,
                                    _fs * 0.95, (0, 0, 0), _th + 3, cv2.LINE_AA)
                        cv2.putText(frame, _abtxt, (_abx, _aby), cv2.FONT_HERSHEY_DUPLEX,
                                    _fs * 0.95, _abcol, _th, cv2.LINE_AA)
                    # สถานะ authorize (มุมซ้ายบน) — ENGLISH เท่านั้น (OpenCV เขียนไทยไม่ได้ → ????)
                    _auth_txt = ("FIRE AUTHORIZED - auto-release when solution ready"
                                 if fire_authorized else "press FIRE = authorize 1 shot")
                    cv2.putText(frame, _auth_txt, (int(_fw * 0.02), int(_fh * 0.10)),
                                cv2.FONT_HERSHEY_DUPLEX, _fs * 0.8,
                                (0, 220, 255) if fire_authorized else (160, 160, 160),
                                _th, cv2.LINE_AA)
                    # เตือนเขตห้ามยิง (ความปลอดภัย) — แขนชี้นอกโคนปลอดภัย
                    if not lock_fire_safe_zone:
                        _nz = "NO-FIRE ZONE (unsafe sector)"
                        _nzsz, _ = cv2.getTextSize(_nz, cv2.FONT_HERSHEY_DUPLEX, _fs * 1.2, _th + 1)
                        cv2.putText(frame, _nz, (int((_fw - _nzsz[0]) / 2), int(_fh * 0.30)),
                                    cv2.FONT_HERSHEY_DUPLEX, _fs * 1.2, (0, 0, 255), _th + 1, cv2.LINE_AA)
                    # บรรทัดสถานะ solution + สถิติ HIT
                    if lock_hit_radius_deg is not None:
                        _un = f"{lock_predict_uncert_deg:.2f}" if lock_predict_uncert_deg is not None else "-"
                        _conf = "CONFIDENT" if lock_confident else "predicting"
                        _stat = (f"predict_uncert {_un} / hit_r {lock_hit_radius_deg:.2f}deg [{_conf}]  "
                                 f"err {lock_pred_miss_deg if lock_pred_miss_deg is None else round(lock_pred_miss_deg,2)}deg  "
                                 f"lat {lock_pipe_lat*1000:.0f}ms tof {lock_fire_t_flight*1000:.0f}ms  "
                                 f"HIT {lock_hits}/{lock_shots_fired}")
                        cv2.putText(frame, _stat, (int(_fw * 0.02), int(_fh * 0.22)),
                                    cv2.FONT_HERSHEY_SIMPLEX, _fs * 0.5,
                                    (0, 255, 0) if lock_fire_ready else (200, 200, 200),
                                    max(1, _th // 2), cv2.LINE_AA)
                    # ผล HIT/MISS ตัวใหญ่ (โชว์ชั่วคราวหลังกระสุนปะทะ)
                    if hit_flash is not None:
                        _htxt, _hcol, _huntil = hit_flash
                        if time.time() <= _huntil:
                            _hsz, _ = cv2.getTextSize(_htxt, cv2.FONT_HERSHEY_DUPLEX, _fs * 2.6, _th + 2)
                            _hx = int((_fw - _hsz[0]) / 2)
                            _hy = int(_fh * 0.5)
                            cv2.putText(frame, _htxt, (_hx, _hy), cv2.FONT_HERSHEY_DUPLEX,
                                        _fs * 2.6, (0, 0, 0), _th + 5, cv2.LINE_AA)
                            cv2.putText(frame, _htxt, (_hx, _hy), cv2.FONT_HERSHEY_DUPLEX,
                                        _fs * 2.6, _hcol, _th + 2, cv2.LINE_AA)
                        else:
                            hit_flash = None

                # Vector HUD: white/yellow arrows, cyan virtual center; LOCK: bbox + red dot
                if (
                    app_mode == "normal"
                    and arm_controller is not None
                    and arm_mode_manager is not None
                    and px_per_deg_x is not None
                    and px_per_deg_y is not None
                ):
                    has_auto_target = arm_mode_manager.mode == MODE_AUTO and smoothed_aim_cx is not None and smoothed_aim_cy is not None
                    has_lock_target = (
                        arm_mode_manager.mode == MODE_LOCK
                        and lock_csrt_initialized
                        and not lock_csrt_lost
                        and lock_csrt_smooth_px is not None
                        and lock_csrt_smooth_py is not None
                    )
                    if has_auto_target:
                        target_px = (float(smoothed_aim_cx), float(smoothed_aim_cy))
                    elif has_lock_target:
                        target_px = (float(lock_csrt_smooth_px), float(lock_csrt_smooth_py))
                    else:
                        target_px = None

                    if target_px is None:
                        vector_virtual_center_x = None
                        vector_virtual_center_y = None
                        yellow_smoothed_px = None
                    else:
                        wx = target_px[0] - cx_frame
                        wy = target_px[1] - cy_frame
                        white_len = math.hypot(wx, wy)
                        if white_len >= 1e-6:
                            ux = wx / white_len
                            uy = wy / white_len
                            cap_len = min(white_len, VIRTUAL_CLICK_CAP_PX)
                            vector_virtual_center_x = cx_frame + ux * cap_len
                            vector_virtual_center_y = cy_frame + uy * cap_len
                            pt_white = (int(round(target_px[0])), int(round(target_px[1])))

                            desired_yellow = YELLOW_MAX_PX * (1.0 - math.exp(-white_len / yellow_scale_px))
                            desired_yellow = max(0.0, min(desired_yellow, white_len, YELLOW_MAX_PX))
                            # Adaptive yellow alpha: ปรับตามความเร็วเป้า
                            # lock_kalman อยู่ใน bearing space (deg/s) → คูณ ppd กลับเป็น px/s ให้ threshold เดิมใช้ได้
                            if arm_mode_manager and arm_mode_manager.mode == MODE_AUTO:
                                _spd_for_alpha = target_kalman.get_speed_px_s()
                            else:
                                _ppd_for_spd = abs(px_per_deg_x) if (px_per_deg_x is not None and abs(px_per_deg_x) > 1e-6) else 1.0
                                _spd_for_alpha = lock_kalman.get_speed_px_s() * _ppd_for_spd
                            _yellow_alpha_use = ADAPTIVE_ALPHA_FAST if _spd_for_alpha > ADAPTIVE_ALPHA_SPEED_THRESHOLD_PX else ADAPTIVE_ALPHA_SLOW
                            if yellow_smoothed_px is None:
                                yellow_smoothed_px = desired_yellow
                            else:
                                yellow_smoothed_px = _yellow_alpha_use * yellow_smoothed_px + (1.0 - _yellow_alpha_use) * desired_yellow
                            error_deg = math.hypot(wx / px_per_deg_x, wy / px_per_deg_y)
                            yellow_cap_val = _yellow_max_at_error_deg(error_deg, yellow_max_curve)
                            yellow_capped = min(yellow_smoothed_px, yellow_cap_val)
                            yellow_draw_effective = min(yellow_capped, white_len)

                            # Draw: LOCK bbox + red dot, then cyan, white, yellow
                            if has_lock_target and lock_csrt_bbox is not None:
                                x0, y0, bw, bh = lock_csrt_bbox
                                cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0), 2)
                                rcx = int(round(lock_csrt_smooth_px))
                                rcy = int(round(lock_csrt_smooth_py))
                                cv2.circle(frame, (rcx, rcy), 10, (0, 0, 255), 3)
                                cv2.circle(frame, (rcx, rcy), 4, (0, 0, 255), -1)

                            if vector_virtual_center_x is not None and vector_virtual_center_y is not None:
                                ivx = int(round(vector_virtual_center_x))
                                ivy = int(round(vector_virtual_center_y))
                                cv2.circle(frame, (ivx, ivy), 8, (255, 255, 0), 2)
                                cv2.circle(frame, (ivx, ivy), 3, (255, 255, 0), -1)

                            cv2.arrowedLine(frame, (int(cx_frame), int(cy_frame)), pt_white, COLOR_WHITE, 2, tipLength=0.15)

                            if yellow_draw_effective >= 2:
                                pt_yellow = (
                                    int(round(cx_frame + ux * yellow_draw_effective)),
                                    int(round(cy_frame + uy * yellow_draw_effective)),
                                )
                                cv2.arrowedLine(frame, (int(cx_frame), int(cy_frame)), pt_yellow, COLOR_YELLOW, 2, tipLength=0.15)
                        else:
                            vector_virtual_center_x = None
                            vector_virtual_center_y = None
                            yellow_smoothed_px = None

            elif app_mode == "adjust_center":
                draw_adjust_center_hud(frame, config, w, h)
            elif app_mode == "settings":
                draw_settings_overlay(frame, config, settings_selected_field)
            elif app_mode == "settings2" and settings_screen is not None:
                frame = settings_screen.tick(frame)
            elif app_mode == "wizard" and calib_wizard is not None:
                frame = calib_wizard.tick(frame, arm_controller)
            elif app_mode == "cam8_mapping" and cam8_mapping_embed is not None:
                frame8 = None
                _cam8_status = "cam8 stream off"
                if cam8_stream is not None:
                    ok8, f8, _ = cam8_stream.read()
                    if ok8 and f8 is not None and f8.size > 0:
                        frame8 = f8
                        last_cam8_ok_t = t0
                    elif not getattr(cam8_stream, "running", False):
                        _cam8_status = "cam8 disconnected — retrying..."
                    elif t0 - last_cam8_ok_t < 2.0:
                        _cam8_status = "cam8 connecting..."
                    else:
                        _cam8_status = "cam8 no signal — check 192.168.144.112"
                    if (
                        frame8 is None
                        and t0 - last_cam8_ok_t > CAM8_MAP_STALE_SEC
                        and t0 - last_cam8_reconnect_t > CAM8_MAP_RECONNECT_INTERVAL_SEC
                    ):
                        last_cam8_reconnect_t = t0
                        print("[MAP] cam8 stale — reconnecting...")
                        _start_cam8_stream(force_restart=True)
                if arm_controller is not None:
                    cam8_mapping_embed.tick_arm_sync(arm_controller)
                    cam8_mapping_embed.tick_test_go(arm_controller)
                if js_state is not None and joystick_mapper is not None and arm_controller is not None:
                    joystick_mapper.apply(js_state, dt_loop)
                    if js_state.confirm_pressed and not mapping_prev_confirm:
                        cam8_mapping_embed.confirm_cell(arm_controller)
                    mapping_prev_confirm = bool(js_state.confirm_pressed)
                elif js_state is not None:
                    mapping_prev_confirm = bool(getattr(js_state, "confirm_pressed", False))
                frame = cam8_mapping_embed.render_composite(
                    frame8, frame, cam8_status=_cam8_status if frame8 is None else ""
                )
                mapping_has_cam8_frame = frame8 is not None
                mapping_cam8_status = _cam8_status
                h, w = frame.shape[:2]

            elif app_mode == "cam4_pxdeg_calib" and cam4_pxdeg_calibrator_embed is not None:
                frame = cam4_pxdeg_calibrator_embed.tick(frame, arm_controller)
                h, w = frame.shape[:2]
                if (
                    js_state is not None
                    and joystick_mapper is not None
                    and arm_controller is not None
                ):
                    cam4_pxdeg_calibrator_embed.apply_joystick_finetune(
                        js_state, dt_loop, arm_controller, joystick_mapper
                    )

            # ------------------------------------------------------------------
            # Joystick: สลับ class / ความไว (ใช้ได้แม้ไม่มีแขน) + ควบคุมแขนเมื่อมี mapper
            # ------------------------------------------------------------------
            if js_state is not None:
                if joystick_mapper is not None:
                    if js_state.sensitivity_cycle_pressed and not prev_sensitivity_cycle_pressed:
                        joystick_mapper.cycle_sensitivity_mode()
                prev_sensitivity_cycle_pressed = js_state.sensitivity_cycle_pressed
                # สลับ YOLO class: ปุ่ม CLASS_CYCLE (ค่าเริ่มต้น 8) — ไม่นับขณะยิง (กันเลขปุ่มชนกับยิงใน config)
                _cc_held = bool(getattr(js_state, "class_cycle_pressed", False))
                _fire_held = bool(getattr(js_state, "fire_pressed", False))
                if yolo_enable_class_cycle and _cc_held and not _fire_held and app_mode not in ("cam8_mapping", "cam4_pxdeg_calib"):
                    _now_cc = time.time()
                    if not prev_class_cycle_pressed:
                        active_class_idx = (active_class_idx + 1) % len(active_class_names)
                        active_class_name = active_class_names[active_class_idx]
                        _reset_target_tracking_state()
                        class_cycle_next_repeat_t = _now_cc + joystick_class_cycle_initial_sec
                    elif _now_cc >= class_cycle_next_repeat_t:
                        active_class_idx = (active_class_idx + 1) % len(active_class_names)
                        active_class_name = active_class_names[active_class_idx]
                        _reset_target_tracking_state()
                        class_cycle_next_repeat_t = _now_cc + joystick_class_cycle_interval_sec
                prev_class_cycle_pressed = _cc_held if yolo_enable_class_cycle else False
                # สลับ engine ตรวจจับ RGB/thermal — ไม่นับขณะยิง
                _det_toggle = bool(getattr(js_state, "detection_engine_toggle_pressed", False))
                if _det_toggle and not _fire_held:
                    if not prev_detection_toggle_pressed:
                        _apply_detection_engine_toggle()
                prev_detection_toggle_pressed = _det_toggle

            if (
                app_mode not in ("cam8_mapping", "cam4_pxdeg_calib")
                and arm_controller is not None
                and joystick_mapper is not None
                and arm_mode_manager is not None
                and js_state is not None
                and not camera_operator_lock
            ):
                # ปุ่ม 3 (index 2): ปลด LOCK → MANUAL (ไม่ว่าจะ lock มาจากปุ่ม 4 หรือ 5)
                if getattr(js_state, "unlock_pressed", False) and arm_mode_manager.mode == MODE_LOCK:
                    arm_mode_manager.set_mode(MODE_MANUAL)
                # ปุ่มโหมดจาก joystick — ปุ่ม 5 กดซ้ำใน LOCK = lock ใหม่จาก YOLO (reset state, เฟรมถัดไปหมุนไป center bbox)
                elif js_state.mode_switch is not None:
                    if (
                        not arm_calibration_ready
                        and js_state.mode_switch in (MODE_AUTO, MODE_LOCK)
                    ):
                        print(
                            "Gun Aim Assist: AUTO/LOCK need pixel_per_degree calibration — "
                            "run cam4_arm_mouse_grid_calibrator first."
                        )
                    elif js_state.mode_switch == MODE_LOCK and arm_mode_manager.mode == MODE_LOCK:
                        lock_csrt_tracker = None
                        lock_csrt_initialized = False
                        lock_use_yolo_only = True
                        lock_csrt_bbox = None
                        lock_csrt_smooth_px = None
                        lock_csrt_smooth_py = None
                        lock_csrt_lost = True
                        lock_track_frame_count = 0
                        pending_lock_csrt_bbox = None
                        lock_iou_tracker.reset()
                        lock_kalman.reset()
                    elif arm_calibration_ready or js_state.mode_switch in (MODE_MANUAL, MODE_SAFE):
                        arm_mode_manager.set_mode(js_state.mode_switch)
                    else:
                        arm_mode_manager.set_mode(MODE_MANUAL)
                    # เมื่อสลับออกจาก LOCK เคลียร์ CSRT tracker + lock_kalman
                    if arm_mode_manager.mode != MODE_LOCK:
                        lock_csrt_tracker = None
                        lock_csrt_initialized = False
                        lock_use_yolo_only = False
                        lock_csrt_bbox = None
                        lock_csrt_smooth_px = None
                        lock_csrt_smooth_py = None
                        lock_csrt_lost = False
                        lock_track_frame_count = 0
                        pending_lock_csrt_bbox = None
                        lock_iou_tracker.reset()
                        lock_kalman.reset()
                        pd_controller.reset()
                    # ปุ่ม 5: เข้า LOCK หรือกดซ้ำใน LOCK — หมุนแขนไป center bbox YOLO ที่ใกล้ศูนย์เล็งที่สุด (ครั้งเดียว)
                    if arm_mode_manager.mode == MODE_LOCK and (js_state.mode_switch == MODE_LOCK or kb_lock_acquire):
                        kb_lock_acquire = False   # consume keyboard one-shot acquire (ไม่มีจอยก็ล็อกได้ด้วยปุ่ม K)
                        # LOCK acquire: ผ่อน min-size filter — ล็อก detection ที่ใกล้ crosshair สุด
                        # ไม่ว่าจะเล็กแค่ไหน (pick_best_target กรอง TARGET_MIN_BOX_PX ทำให้เป้าเล็กล็อกไม่ได้)
                        target_det_yolo = (
                            min(last_detections, key=lambda _d: _distance_to_crosshair(_d, cx_frame, cy_frame))
                            if last_detections
                            else last_target_det
                        )
                        if LOCK_TRACK_DEBUG:
                            _nd = len(last_detections) if last_detections else 0
                            print(f"[LOCK] acquire: yolo_dets={_nd} target={target_det_yolo is not None} "
                                  f"ppd=({px_per_deg_x},{px_per_deg_y})", flush=True)
                        if (
                            target_det_yolo is not None
                            and px_per_deg_x is not None
                            and px_per_deg_y is not None
                            and x_lo is not None
                            and arm_controller is not None
                        ):
                            if hasattr(arm_controller, "sync_position_from_grbl"):
                                try:
                                    arm_controller.sync_position_from_grbl()
                                except Exception:
                                    pass
                            x, y, w_d, h_d, _ = _det_xywhc(target_det_yolo)
                            # init IoU tracker ด้วย bbox เป้าที่เลือก (conf 0.2) — เฟรมถัดไปจะใช้ conf 0.05 ติดตาม bbox นี้
                            lock_iou_tracker.init_from_bbox(
                                int(x), int(y), max(1, int(w_d)), max(1, int(h_d)),
                                cur_pan=getattr(arm_controller, "pos_x", None),
                                cur_tilt=getattr(arm_controller, "pos_y", None),
                            )
                            cx_t = x + w_d // 2
                            cy_t = y + h_d // 2
                            ch_x = float(cx_frame)
                            ch_y = float(cy_frame)
                            dx_px = cx_t - ch_x
                            dy_px = cy_t - ch_y
                            delta_pan = dx_px / px_per_deg_x
                            delta_tilt = dy_px / px_per_deg_y
                            pan_cur = getattr(arm_controller, "pos_x", 0.0)
                            tilt_cur = getattr(arm_controller, "pos_y", 0.0)
                            if CAMERA_ON_ARM:
                                tpan = pan_cur + delta_pan
                                ttilt = tilt_cur + delta_tilt
                            else:
                                tpan = delta_pan
                                ttilt = delta_tilt
                            if SWAP_PAN_TILT:
                                pan_deg = float(np.clip(tpan, y_lo, y_hi))
                                tilt_deg = float(np.clip(ttilt, x_lo, x_hi))
                            else:
                                pan_deg = float(np.clip(tpan, x_lo, x_hi))
                                tilt_deg = float(np.clip(ttilt, y_lo, y_hi))
                            try:
                                arm_controller.move_absolute(pan_deg, tilt_deg, blocking=True)
                            except Exception:
                                pass
                # ปุ่มยิง — LOCK = human-in-the-loop: 'กดครั้งเดียว = อนุญาต (latch)'
                # โปรแกรมรับหน้าที่คำนวณวิถี+จังหวะ แล้วลั่นไกเองเมื่อ solution การันตีโดน
                _in_lock = (arm_mode_manager.mode == MODE_LOCK) and LOCK_FIRE_SOLUTION_ENABLED
                _fire_edge = js_state.fire_pressed and not prev_fire_pressed  # ขอบขาขึ้น (กด)
                prev_fire_pressed = js_state.fire_pressed
                if _fire_edge and arm_mode_manager.mode != MODE_SAFE:
                    if _in_lock:
                        fire_authorized = not fire_authorized  # toggle: กด=อนุญาต, กดซ้ำ=ยกเลิก
                        print(f"[FIRE] {'AUTHORIZED — โปรแกรมจะลั่นไกเองเมื่อคำนวณว่าโดนแน่' if fire_authorized else 'CANCELLED'}", flush=True)
                    else:
                        # โหมด MANUAL/AUTO: ยิงทันที (manual override)
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
                if not _in_lock:
                    fire_authorized = False  # ออกจาก LOCK = ยกเลิกอนุญาต

                # โปรแกรมลั่นไกเอง 'ทีละนัด': authorized + ready → ยิง 1 นัด แล้วปลด authorize
                # (พลาดก็ track LOCK ต่อ รอผู้ใช้กดสั่งยิงใหม่)
                if fire_authorized and _in_lock and lock_fire_ready:
                    now_fire = time.time()
                    _play_fire_sound()
                    is_firing = True
                    last_fire_time = now_fire
                    _lock_fire_flash_t = now_fire   # trigger fire effect เห็นชัด (โปรแกรมยิงเอง)
                    last_fire_gcode_time = now_fire
                    lock_shots_fired += 1
                    fire_authorized = False   # ยิงทีละนัด — ปลด authorize ทันที
                    if arm_controller is not None and hasattr(arm_controller, "fire"):
                        arm_controller.fire()
                    # บันทึกวิถี: ลำกล้องชี้มุมแขน ณ ยิง; กระสุนถึงเป้าที่ now+t_flight
                    _bpan = float(getattr(arm_controller, "pos_x", 0.0)) if arm_controller else 0.0
                    _btilt = float(getattr(arm_controller, "pos_y", 0.0)) if arm_controller else 0.0
                    _tof = max(0.005, lock_fire_t_flight)
                    # เก็บ velocity เป้า(deg/s) + lead ที่ใส่ ณ ยิง → ใช้แยก along/cross-track ตอนปะทะ
                    _vp_fire, _vt_fire = (
                        lock_kalman.get_velocity_raw() if lock_kalman is not None else (0.0, 0.0)
                    )
                    pending_impacts.append(
                        (now_fire + _tof, _bpan, _btilt, _tof, _vp_fire, _vt_fire, lock_fire_lead_deg))
                    _hr = lock_hit_radius_deg if lock_hit_radius_deg is not None else 0.0
                    _em = lock_expected_miss_deg if lock_expected_miss_deg is not None else 0.0
                    print(f"[FIRE] SHOT #{lock_shots_fired} — expected_miss={_em:.2f}deg <= hit_r={_hr:.2f}deg "
                          f"tof={_tof*1000:.0f}ms lat={lock_pipe_lat*1000:.0f}ms -> computing impact...", flush=True)

                # คำนวณการปะทะจริง: เทียบทิศลำกล้อง(ตอนยิง) กับตำแหน่งจริงโดรน ณ เวลากระสุนถึง
                if pending_impacts:
                    _now_imp = time.time()
                    _still = []
                    for (imp_t, bpan, btilt, tof, vp_fire, vt_fire, lead_fire) in pending_impacts:
                        if _now_imp >= imp_t:
                            if virtual_target is not None:
                                _miss = math.hypot(bpan - virtual_target.pan, btilt - virtual_target.tilt)
                                _hr = lock_hit_radius_deg if lock_hit_radius_deg is not None else 0.5
                                # sim มี ground-truth → ครูที่แม่นสุด: ป้อน error เข้า FireTune ก่อน explode
                                if _fire_tune_learner is not None:
                                    _fire_tune_learner.add_sample(
                                        bpan - virtual_target.pan, btilt - virtual_target.tilt,
                                        vp_fire, vt_fire, stable=True)
                                if _miss <= _hr:
                                    lock_hits += 1
                                    hit_flash = ("HIT! LOCK RELEASED", (0, 255, 0), _now_imp + 1.5)
                                    virtual_target.explode(_now_imp)   # โดรนกระจุย + respawn
                                    # 🛑 SAFETY: โดรนถูกทำลาย → ปลด LOCK ทันที
                                    # ไม่ให้แขนตามซากที่ร่วง/หมุนลงพื้น (กันชี้ปืนลงพื้น = อันตราย)
                                    if arm_mode_manager is not None:
                                        arm_mode_manager.set_mode(MODE_MANUAL)
                                    lock_iou_tracker.reset()
                                    lock_kalman.reset()
                                    lock_csrt_tracker = None
                                    lock_csrt_initialized = False
                                    lock_csrt_bbox = None
                                    lock_csrt_smooth_px = None
                                    lock_csrt_smooth_py = None
                                    lock_csrt_lost = True
                                    lock_track_frame_count = 0
                                    pending_lock_csrt_bbox = None
                                    fire_authorized = False
                                    _lock_fire_ready_count[0] = 0
                                    _lock_meas_prev[0] = None
                                    _lock_pred_resid_rate[0] = 0.0
                                    print(f"[IMPACT] ✅ HIT  miss={_miss:.2f}° ≤ {_hr:.2f}° "
                                          f"({lock_hits}/{lock_shots_fired}) DRONE DESTROYED "
                                          f"→ 🛑 LOCK RELEASED (safety: no tracking of falling debris)", flush=True)
                                else:
                                    lock_misses += 1
                                    hit_flash = (f"MISS {_miss:.1f}°", (0, 0, 255), _now_imp + 1.2)
                                    print(f"[IMPACT] ❌ MISS miss={_miss:.2f}° > {_hr:.2f}° "
                                          f"({lock_hits}/{lock_shots_fired})", flush=True)
                            else:
                                # โหมดจริง: ไม่มี ground truth → เทียบทิศลำกล้อง(ตอนยิง) กับทิศเป้าที่ track ได้
                                # ณ เวลากระสุนถึง = วัดว่า 'เล็ง+นำเป้า' ลงตรงที่เป้าไปจริงไหม (จูน lead/ballistics/offset)
                                if last_target_pan is not None and last_target_tilt is not None:
                                    # miss vector = ลำกล้อง(ตอนยิง) − เป้า(ณ กระสุนถึง). ในโหมดจริง
                                    # last_target_* คือทิศเป้าล่าสุดที่ track ได้ (≈ ตำแหน่งเป้า ณ ปะทะ)
                                    _err_pan = bpan - last_target_pan
                                    _err_tilt = btilt - last_target_tilt
                                    _aim_err = math.hypot(_err_pan, _err_tilt)
                                    _hr = lock_hit_radius_deg if lock_hit_radius_deg is not None else 0.5
                                    _rng = tof * float(getattr(config, "muzzle_velocity_ms", 900.0))
                                    _tag = "on-target" if _aim_err <= _hr else "off"
                                    # แยก miss เป็น along-track (ตามทิศเป้าวิ่ง) และ cross-track (ตั้งฉาก):
                                    #   along-track มี bias คงที่ → lead/TOF ผิด (จูน muzzle_velocity_ms / LEAD_EXTRA)
                                    #     along>0 = เล็งนำหน้าไป (lead มากไป) ; along<0 = ตกหลังเป้า (lead น้อยไป)
                                    #   cross-track มี bias คงที่ → aim offset/px-deg/crosshair (จูน trim/offset)
                                    _vmag = math.hypot(vp_fire, vt_fire)
                                    if _vmag > 1e-6:
                                        _ux, _uy = vp_fire / _vmag, vt_fire / _vmag
                                        _along = _err_pan * _ux + _err_tilt * _uy
                                        _cross = _err_pan * (-_uy) + _err_tilt * _ux
                                    else:
                                        _along, _cross = 0.0, 0.0   # เป้านิ่ง → แยกทิศไม่ได้
                                    # ป้อนเข้า FireTune (โหมดจริง): teacher = ทิศเป้าที่ track ได้ ณ ปะทะ
                                    if _fire_tune_learner is not None:
                                        _fire_tune_learner.add_sample(_err_pan, _err_tilt, vp_fire, vt_fire,
                                                                      stable=(not lock_csrt_lost))
                                    print(f"[IMPACT] aim_err={_aim_err:.2f}deg ({_tag}, hit_r={_hr:.2f}) "
                                          f"along={_along:+.2f} cross={_cross:+.2f} "
                                          f"barrel=({bpan:+.1f},{btilt:+.1f}) "
                                          f"target@impact=({last_target_pan:+.1f},{last_target_tilt:+.1f}) "
                                          f"vel={_vmag:.1f}d/s lead={lead_fire:.2f} "
                                          f"tof={tof*1000:.0f}ms range~{_rng:.0f}m", flush=True)
                                    hit_flash = (f"aim {_aim_err:.1f}deg", (0, 200, 255), _now_imp + 1.2)
                                    if _impact_log is None:
                                        try:
                                            _impact_log = open(Path(__file__).resolve().parent / "impact_log.csv", "a")
                                            _impact_log.write("# t_iso,barrel_pan,barrel_tilt,tgt_pan,tgt_tilt,"
                                                              "err_pan,err_tilt,aim_err_deg,along_deg,cross_deg,"
                                                              "vel_deg_s,lead_deg,hit_r_deg,tof_ms,range_m\n")
                                        except Exception:
                                            _impact_log = None
                                    if _impact_log is not None:
                                        _impact_log.write(
                                            f"{datetime.datetime.now().isoformat(timespec='milliseconds')},"
                                            f"{bpan:.2f},{btilt:.2f},{last_target_pan:.2f},{last_target_tilt:.2f},"
                                            f"{_err_pan:.3f},{_err_tilt:.3f},{_aim_err:.3f},{_along:.3f},{_cross:.3f},"
                                            f"{_vmag:.2f},{lead_fire:.3f},{_hr:.3f},{tof * 1000:.0f},{_rng:.0f}\n")
                                        _impact_log.flush()
                        else:
                            _still.append((imp_t, bpan, btilt, tof, vp_fire, vt_fire, lead_fire))
                    pending_impacts = _still

                # ควบคุมแขนด้วย joystick เฉพาะเมื่ออยู่โหมด MANUAL (ยกเว้นระหว่าง JOG คีย์บอร์ด)
                if arm_mode_manager.mode == MODE_MANUAL and app_mode != "jog":
                    joystick_mapper.apply(js_state, dt_loop)

            # Skip drive-verify / idle-probe during px/deg calib (arrow nudge 0.1° triggers false stall)
            # และระหว่าง JOG — กันไม่ให้ fault monitor วน re-home แย่งขณะผู้ใช้หมุนแขนแก้อาการค้าง
            # โหมดที่ 'ขับแขนเอง' ต้องปิด fault monitor ไม่งั้นมันจะเห็นการขยับที่ไม่ได้สั่งจาก
            # ลูปหลักแล้วนึกว่าแขนเสีย (wizard สั่งแขนกวาด/สะบัดเป็นเรื่องปกติ)
            if app_mode not in ("cam4_pxdeg_calib", "jog", "wizard"):
                if arm_controller is not None and hasattr(arm_controller, "maybe_verify_drive_motion"):
                    arm_controller.maybe_verify_drive_motion()
                if arm_controller is not None and hasattr(arm_controller, "maybe_handle_arm_fault_recovery"):
                    arm_controller.maybe_handle_arm_fault_recovery(
                        camera_operator_lock=camera_operator_lock
                    )
                elif arm_controller is not None and hasattr(arm_controller, "maybe_handle_stall_recovery"):
                    arm_controller.maybe_handle_stall_recovery(
                        camera_operator_lock=camera_operator_lock
                    )
                if arm_controller is not None and hasattr(arm_controller, "maybe_check_arm_liveness"):
                    arm_controller.maybe_check_arm_liveness()
                if arm_controller is not None and hasattr(arm_controller, "maybe_run_idle_probe"):
                    arm_controller.maybe_run_idle_probe()
            if arm_mode_manager is not None:
                mode_before_camera_loss = _tick_camera_loss_mode_safe(
                    cam,
                    network_loss_mode,
                    arm_mode_manager,
                    mode_before_camera_loss,
                    arm_controller,
                )
                camera_operator_lock = _camera_operator_lock_active(cam, network_loss_mode)
                mode_before_arm_fault = _tick_arm_mode_fault_and_restore(
                    arm_controller,
                    arm_mode_manager,
                    mode_before_arm_fault,
                    camera_operator_lock=camera_operator_lock,
                )
            if arm_controller is not None and hasattr(arm_controller, "set_camera_operator_lock"):
                arm_controller.set_camera_operator_lock(camera_operator_lock)

            elapsed = time.time() - t0
            fps_times.append(elapsed)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
            # ปรับสถานะ flash การยิง (หมดอายุเมื่อเกิน FIRE_FLASH_DURATION)
            if is_firing and (time.time() - last_fire_time) > FIRE_FLASH_DURATION:
                is_firing = False
            # ปิด overlay calibration อัตโนมัติหลัง 10 วินาที
            if show_calibration_status and calibration_status_until > 0 and time.time() > calibration_status_until:
                show_calibration_status = False
                calibration_status_until = 0.0

            if app_mode == "normal":
                _hud_ctx = {
                    "fps": fps,
                    "lat_ms": avg_pipeline_latency * 1000.0,
                    "has_lat": bool(pipeline_latency_history),
                    "conf": runtime_conf_detect,
                    "class_name": _hud_class_display_name(active_class_name) if active_class_name else None,
                    "yolo_thermal_model": yolo_thermal_model,
                    "yolo_detection_mode": yolo_detection_mode,
                    "detection_use_thermal": detection_use_thermal,
                    "yolo_alt_imgsz": yolo_alt_imgsz,
                    "yolo_primary_imgsz": yolo_primary_imgsz,
                }
                _arm_lbl, _arm_col = _arm_hud_label_and_color(arm_controller)
                _hud_ctx["arm_label"] = _arm_lbl
                _hud_ctx["arm_color"] = _arm_col
                _cam_lbl, _cam_col = cam_hud_label_and_color(cam, True, frame)
                _hud_ctx["cam_label"] = _cam_lbl
                _hud_ctx["cam_color"] = _cam_col
                _joy_lbl, _joy_col = _joy_hud_label_and_color(joystick_reader)
                _hud_ctx["joy_label"] = _joy_lbl
                _hud_ctx["joy_color"] = _joy_col
                if joystick_mapper is not None:
                    _hud_ctx["spd_label"] = f"SPD:{joystick_mapper.get_sensitivity_label()}"
                    _hud_ctx["spd_color"] = (255, 255, 255)
                else:
                    _hud_ctx["spd_label"] = "SPD:---"
                    _hud_ctx["spd_color"] = (120, 120, 120)
                if arm_mode_manager is not None:
                    _mode_label, _mode_color = arm_mode_manager.label_and_color()
                    _hud_ctx["mode_label"] = _mode_label
                    _hud_ctx["mode_color"] = _mode_color
                    if arm_mode_manager.mode == MODE_AUTO and arm_cue_receiver is not None:
                        _cue_age = arm_cue_receiver.get_cue_age_ms()
                        _hud_ctx["show_auto_diag"] = True
                        _hud_ctx["src_label"] = f"SRC:{auto_cue_source}"
                        _hud_ctx["src_color"] = (0, 255, 100) if auto_cue_source == "cam8" else (0, 200, 255)
                        _hud_ctx["cue_label"] = (
                            f"CUE:{_cue_age:.0f}ms" if _cue_age is not None else "CUE:none"
                        )
                        _hud_ctx["cue_color"] = (
                            (0, 255, 0)
                            if (_cue_age is not None and _cue_age < ARM_CUE_TTL_MS)
                            else (0, 80, 255)
                        )
                        _hud_ctx["cal_label"] = "CAL8:OK" if cam8_calib_data is not None else "CAL8:MISSING"
                        _hud_ctx["cal_color"] = (0, 255, 0) if cam8_calib_data is not None else (0, 0, 255)
                        _n_cells = len(_residual_bias_learner._cells) if _residual_bias_learner else 0
                        _hud_ctx["bias_label"] = (
                            f"BIAS:{'ON(B)' if CAM8_AUTO_ONLINE_BIAS_ENABLED else 'OFF(B)'} {_n_cells}c"
                        )
                        _hud_ctx["bias_color"] = (
                            (0, 255, 128) if CAM8_AUTO_ONLINE_BIAS_ENABLED else (120, 120, 120)
                        )
                if effector_exporter is not None:
                    if _config_mod is not None and getattr(_config_mod, "EFFECTOR_WS_ENABLED", False):
                        _hud_ctx["ws_label"] = "WS:OK" if effector_exporter.is_connected else "WS:ERR"
                        _hud_ctx["ws_color"] = (
                            (0, 220, 0) if effector_exporter.is_connected else (0, 100, 255)
                        )
                    else:
                        _hud_ctx["ws_label"] = "WS:off"
                        _hud_ctx["ws_color"] = (120, 120, 120)
                    _eff_state = effector_exporter.last_lock_state
                    _eff_colors = {
                        "locked": (0, 0, 255),
                        "acquiring": (0, 165, 255),
                        "searching": (255, 200, 80),
                    }
                    _hud_ctx["eff_part"] = f"EFF:{_eff_state}"
                    _hud_ctx["eff_color"] = _eff_colors.get(_eff_state, (200, 200, 200))
                draw_gun_bottom_hud(frame, _hud_ctx)

            elif app_mode == "cam4_pxdeg_calib" and cam4_pxdeg_calibrator_embed is not None:
                _px_lbl, _px_col = cam4_pxdeg_calibrator_embed.hud_label()
                _step_lbl, _step_col = cam4_pxdeg_calibrator_embed.step_label()
                _px_hud_ctx = {
                    "fps": fps,
                    "pan": getattr(arm_controller, "pos_x", 0.0) if arm_controller else 0.0,
                    "tilt": getattr(arm_controller, "pos_y", 0.0) if arm_controller else 0.0,
                    "px_label": _px_lbl,
                    "px_color": _px_col,
                    "px_short": cam4_pxdeg_calibrator_embed.px_short_label(),
                    "px_short_color": _px_col,
                    "step_label": _step_lbl,
                    "step_color": _step_col,
                }
                _arm_lbl, _arm_col = _arm_hud_label_and_color(arm_controller)
                _px_hud_ctx["arm_label"] = _arm_lbl
                _px_hud_ctx["arm_color"] = _arm_col
                _cam_lbl, _cam_col = cam_hud_label_and_color(cam, True, frame)
                _px_hud_ctx["cam_label"] = _cam_lbl
                _px_hud_ctx["cam_color"] = _cam_col
                _joy_lbl, _joy_col = _joy_hud_label_and_color(joystick_reader)
                _px_hud_ctx["joy_label"] = _joy_lbl
                _px_hud_ctx["joy_color"] = _joy_col
                if joystick_mapper is not None:
                    _px_hud_ctx["spd_label"] = f"SPD:{joystick_mapper.get_sensitivity_label()}"
                    _px_hud_ctx["spd_color"] = (255, 255, 255)
                else:
                    _px_hud_ctx["spd_label"] = "SPD:---"
                    _px_hud_ctx["spd_color"] = (120, 120, 120)
                draw_pxdeg_bottom_hud(frame, _px_hud_ctx)

            elif app_mode == "cam8_mapping" and cam8_mapping_embed is not None:
                _mapped, _total = cam8_mapping_embed.mapped_cell_count()
                _cell_lbl, _cell_col = cam8_mapping_embed.selected_cell_label()
                _ref_lbl, _ref_col = cam8_mapping_embed.ref_hud_label_and_color()
                _cam8_lbl, _cam8_col = _cam8_stream_hud_label(
                    mapping_has_cam8_frame, mapping_cam8_status
                )
                _map_hud_ctx = {
                    "fps": fps,
                    "pan": getattr(arm_controller, "pos_x", 0.0) if arm_controller else 0.0,
                    "tilt": getattr(arm_controller, "pos_y", 0.0) if arm_controller else 0.0,
                    "map_label": f"MAP:{_mapped}/{_total}",
                    "map_color": (0, 200, 255),
                    "ref_label": _ref_lbl,
                    "ref_color": _ref_col,
                    "cam8_label": _cam8_lbl,
                    "cam8_color": _cam8_col,
                    "cell_label": _cell_lbl,
                    "cell_color": _cell_col,
                }
                _arm_lbl, _arm_col = _arm_hud_label_and_color(arm_controller)
                _map_hud_ctx["arm_label"] = _arm_lbl
                _map_hud_ctx["arm_color"] = _arm_col
                _cam_lbl, _cam_col = cam_hud_label_and_color(cam, True, frame)
                _map_hud_ctx["cam_label"] = _cam_lbl
                _map_hud_ctx["cam_color"] = _cam_col
                _joy_lbl, _joy_col = _joy_hud_label_and_color(joystick_reader)
                _map_hud_ctx["joy_label"] = _joy_lbl
                _map_hud_ctx["joy_color"] = _joy_col
                if joystick_mapper is not None:
                    _map_hud_ctx["spd_label"] = f"SPD:{joystick_mapper.get_sensitivity_label()}"
                    _map_hud_ctx["spd_color"] = (255, 255, 255)
                else:
                    _map_hud_ctx["spd_label"] = "SPD:---"
                    _map_hud_ctx["spd_color"] = (120, 120, 120)
                draw_mapping_bottom_hud(frame, _map_hud_ctx)

            # Overlay สถานะ calibration (pixel_to_degree) เมื่อกด R
            if show_calibration_status and app_mode == "normal":
                st = get_calibration_status(camera_name)
                bx, by = 20, 170
                pad = 8
                lines = [
                    f"Px/Deg: {st['camera_name']}",
                    st["path_short"],
                    f"{st['message']}  px_x={st['px_per_deg_x']:.1f} px_y={st['px_per_deg_y']:.1f}",
                    "Re-cal: run cam4_arm_mouse_grid_calibrator --camera " + st["camera_name"],
                    "R: close",
                ]
                max_w = 0
                line_heights = []
                for s in lines:
                    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    max_w = max(max_w, tw)
                    line_heights.append(th)
                box_h = sum(line_heights) + pad * 2 + (len(lines) - 1) * 4
                box_w = max_w + pad * 2
                cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
                cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (255, 255, 255), 1)
                color_ok = (200, 255, 200)
                color_missing = (200, 200, 255)
                text_color = color_ok if st["exists"] else color_missing
                y_off = by + pad + line_heights[0]
                for i, s in enumerate(lines):
                    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    line_color = text_color if i <= 2 else (220, 220, 220)
                    cv2.putText(frame, s, (bx + pad, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 1)
                    y_off += th + 4

            # Overlay โหมด JOG: หมุนแขนเองด้วยคีย์บอร์ด (แม้แขนค้าง/SAFE)
            if app_mode == "jog":
                _arm_lbl, _arm_col = _arm_hud_label_and_color(arm_controller)
                _jog_lines = [
                    ("== JOG MODE (manual arm) ==", (0, 255, 255)),
                    (f"step: {jog_step_deg:.2f} deg    arm: {_arm_lbl}", _arm_col),
                    ("arrows / WASD : rotate arm", (230, 230, 230)),
                    ("[ ]  : step down/up    X : unlock ($X)", (230, 230, 230)),
                    ("Z : set home here    H : go home", (180, 255, 180)),
                    ("J / Esc : exit", (230, 230, 230)),
                ]
                _jbx, _jby, _jpad = 20, 90, 10
                _jmax_w, _jlh = 0, []
                for _s, _ in _jog_lines:
                    (_tw, _th), _ = cv2.getTextSize(_s, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
                    _jmax_w = max(_jmax_w, _tw)
                    _jlh.append(_th)
                _jbox_h = sum(_jlh) + _jpad * 2 + (len(_jog_lines) - 1) * 6
                _jbox_w = _jmax_w + _jpad * 2
                cv2.rectangle(frame, (_jbx, _jby), (_jbx + _jbox_w, _jby + _jbox_h), (0, 0, 0), -1)
                cv2.rectangle(frame, (_jbx, _jby), (_jbx + _jbox_w, _jby + _jbox_h), (0, 255, 255), 2)
                _jy = _jby + _jpad + _jlh[0]
                for _i, (_s, _c) in enumerate(_jog_lines):
                    cv2.putText(frame, _s, (_jbx + _jpad, _jy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _c, 1)
                    _jy += _jlh[_i] + 6

            # Badge: SIM เปิดอยู่ (โดรนจำลอง) — เตือนว่าไม่ใช่โหมดจริง
            if _vt_enabled and app_mode == "normal":
                _sim_txt = "SIM ON [I]"
                _sim_fs, _sim_th = 1.6, 3
                (_stw, _sth), _ = cv2.getTextSize(_sim_txt, cv2.FONT_HERSHEY_SIMPLEX, _sim_fs, _sim_th)
                _sbx = w - _stw - 28
                cv2.rectangle(frame, (_sbx - 14, 16), (_sbx + _stw + 14, 16 + _sth + 20), (0, 0, 0), -1)
                cv2.rectangle(frame, (_sbx - 14, 16), (_sbx + _stw + 14, 16 + _sth + 20), (0, 220, 255), 3)
                cv2.putText(frame, _sim_txt, (_sbx, 16 + _sth + 8), cv2.FONT_HERSHEY_SIMPLEX, _sim_fs, (0, 220, 255), _sim_th)

            # ถ้ากำลังยิง ให้ flash ศูนย์เล็งทับ HUD ปกติ
            if is_firing:
                fire_color = (255, 255, 0)  # เหลืองสว่าง
                cv2.circle(frame, (cx_frame, cy_frame), radius_px + 6, fire_color, 3)
                cv2.circle(frame, (cx_frame, cy_frame), 6, fire_color, -1)

            # Fire effect ตอนโปรแกรมลั่นไกเอง (LOCK auto-release) — muzzle flash วงขยาย + "FIRE"
            _fe = time.time() - _lock_fire_flash_t
            if 0.0 <= _fe < LOCK_FIRE_EFFECT_DURATION:
                _p = _fe / LOCK_FIRE_EFFECT_DURATION            # 0→1
                _fade = max(0, int(255 * (1.0 - _p)))
                # วงระเบิดปากกระบอกขยายออก 2 วง (ส้ม/เหลือง)
                for _k, _c in ((0.0, (0, 165, 255)), (0.35, (0, 255, 255))):
                    _kp = max(0.0, _p - _k)
                    _rr = int(radius_px * (1.2 + 7.0 * _kp))
                    _th = max(1, int(7 * (1.0 - _p)))
                    if _th > 0:
                        cv2.circle(frame, (cx_frame, cy_frame), _rr, _c, _th, cv2.LINE_AA)
                # แกนกลางสว่างวูบ
                cv2.circle(frame, (cx_frame, cy_frame), max(3, int(10 * (1.0 - _p))),
                           (255, 255, 255), -1, cv2.LINE_AA)
                # ข้อความ "FIRE" ตัวใหญ่กลาง-บน
                _ft = "FIRE!"
                (_ftw, _fth), _ = cv2.getTextSize(_ft, cv2.FONT_HERSHEY_DUPLEX, 2.2, 4)
                cv2.putText(frame, _ft, (int((w - _ftw) / 2), int(h * 0.16)),
                            cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, _fade, 255), 4, cv2.LINE_AA)

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
                # ภาพจริงอยู่แค่ในกรอบนี้ — ที่เหลือคือแถบดำ. ทุกอย่างที่รับคลิกต้องหักออก
                # ไม่งั้นพิกัดเพี้ยน (cam4 16:9 บนจอ 16:9 → offset=0 บังเอิญรอด; กล้อง 4:3 → เพี้ยน)
                display_content_rect = (x_off, y_off, scaled_w, scaled_h)
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
                display_content_rect = (0, 0, display_w, display_h)
            try:
                cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
            except cv2.error:
                pass

            if app_mode == "cam4_pxdeg_calib" and cam4_pxdeg_calibrator_embed is not None:
                cam4_pxdeg_calibrator_embed.set_display_size(
                    display_w, display_h, display_content_rect
                )
            elif app_mode == "settings2" and settings_screen is not None:
                settings_screen.set_display_size(display_w, display_h, display_content_rect)
            elif app_mode == "wizard" and calib_wizard is not None:
                calib_wizard.set_display_size(display_w, display_h, display_content_rect)

            # RTMP: rate-limited submit (operator view with HUD overlays)
            if _rtmp_streamer is not None:
                if not _rtmp_started:
                    _rtmp_streamer.start(display_frame.shape[1], display_frame.shape[0])
                    _rtmp_started = True
                _now_rtmp = time.monotonic()
                if (_now_rtmp - _rtmp_last_ts) >= _rtmp_min_interval:
                    _rtmp_streamer.submit_frame(display_frame)
                    _rtmp_last_ts = _now_rtmp

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
                    _enter_settings()
                elif key in (ord("w"), ord("W")):
                    _enter_wizard()
                # ปุ่ม R: แสดง/ซ่อนสถานะ calibration (pixel_to_degree) ของกล้องปัจจุบัน
                elif key in (ord("r"), ord("R")):
                    show_calibration_status = not show_calibration_status
                    calibration_status_until = time.time() + 10.0 if show_calibration_status else 0.0
                elif key in (ord("p"), ord("P")):
                    _manual_arm_reconnect(arm_controller)
                # ปุ่ม H: สั่งแขนกลับ home ทันที (go_home เบา ๆ ไม่ re-home)
                elif key in (ord("h"), ord("H")):
                    _manual_arm_go_home(arm_controller, arm_mode_manager)
                # ปุ่ม J: เข้าโหมด JOG — หมุนแขนเองด้วยคีย์บอร์ด แม้แขนค้าง/STALL/SAFE
                elif key in (ord("j"), ord("J")):
                    if arm_controller is None:
                        print("[JOG] ไม่มี arm controller")
                    else:
                        if arm_mode_manager is not None:
                            try:
                                arm_mode_manager.set_mode(MODE_SAFE)
                            except Exception:
                                pass
                        if hasattr(arm_controller, "note_joystick_operator_active"):
                            arm_controller.note_joystick_operator_active(True)
                        _arm_jog_unlock(arm_controller)
                        app_mode = "jog"
                        print(
                            "[JOG] เข้าโหมด JOG — ลูกศร/WASD ขยับแขน | [ ] ปรับ step | "
                            "X ปลดล็อกซ้ำ | H กลับ home | J/Esc ออก"
                        )
                # ปุ่ม B: toggle online residual bias learning (AUTO mode)
                elif key in (ord("b"), ord("B")):
                    CAM8_AUTO_ONLINE_BIAS_ENABLED = not CAM8_AUTO_ONLINE_BIAS_ENABLED
                    print(f"[BiasLearner] ENABLED = {CAM8_AUTO_ONLINE_BIAS_ENABLED}")
                elif key in (ord("g"), ord("G")):
                    if cam4_pxdeg_calibrator_embed is not None:
                        _enter_pxdeg_calib()
                    else:
                        print("[PXDEG] cam4_pxdeg_calibrator_embed not available")
                # ปุ่มเลือก class (เฉพาะโหมด multiclass)
                elif yolo_enable_class_cycle and key == ord(KEY_CLASS_NEXT):
                    active_class_idx = (active_class_idx + 1) % len(active_class_names)
                    active_class_name = active_class_names[active_class_idx]
                    _reset_target_tracking_state()
                elif yolo_enable_class_cycle and key == ord(KEY_CLASS_PREV):
                    active_class_idx = (active_class_idx - 1) % len(active_class_names)
                    active_class_name = active_class_names[active_class_idx]
                    _reset_target_tracking_state()
                elif yolo_enable_class_cycle and key in (
                    ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8")
                ):
                    active_class_idx = int(chr(key)) - 1
                    active_class_idx = max(0, min(active_class_idx, len(active_class_names) - 1))
                    active_class_name = active_class_names[active_class_idx]
                    _reset_target_tracking_state()
                elif key in (ord("t"), ord("T")):
                    _apply_detection_engine_toggle()
                # ปุ่ม Y: สอน aim trim จากผลยิง LOCK/sim ที่ผ่านมา (persist ข้าม session)
                elif key in (ord("y"), ord("Y")):
                    if _fire_tune_learner is None:
                        print("[FireTune] ยังไม่พร้อม (learner ยังไม่ init)")
                    else:
                        print(f"[FireTune] teach จาก {_fire_tune_learner.sample_count()} สมการยิงล่าสุด...")
                        _fire_tune_learner.teach()
                # ปุ่ม I: เปิด/ปิดโดรนจำลอง (SIMULATION) — default ปิด (โหมดจริง)
                elif key in (ord("i"), ord("I")):
                    if VirtualDroneTarget is None:
                        print("[SIMDRONE] ใช้ไม่ได้ — import lock_sim_target ไม่สำเร็จ")
                    else:
                        _vt_enabled = not _vt_enabled
                        if not _vt_enabled:
                            virtual_target = None   # หยุดวาด + หยุด inject เข้า LOCK
                            print("[SIMDRONE] OFF — โหมดจริง (ไม่มีโดรนจำลอง)")
                        else:
                            print("[SIMDRONE] ON — โดรนจำลองจะขึ้นในเฟรม (LOCK+ยิงซ้อมได้), ←→↑↓ บังคับ, V pattern, N สลับระยะ 50-200m")
                elif key in (ord("+"), ord("=")):
                    runtime_conf_detect = _adjust_yolo_conf(runtime_conf_detect, YOLO_CONF_STEP)
                    print(f"YOLO detect conf -> {runtime_conf_detect:.2f} (lock fixed {YOLO_CONF_LOCK:.2f})")
                elif key == ord("-"):
                    runtime_conf_detect = _adjust_yolo_conf(runtime_conf_detect, -YOLO_CONF_STEP)
                    print(f"YOLO detect conf -> {runtime_conf_detect:.2f} (lock fixed {YOLO_CONF_LOCK:.2f})")
                # โดรนเสมือน: ปุ่มลูกศรบังคับ (manual), 'v' สลับ pattern — ไม่ชนปุ่ม k/l (acquire/LOCK)
                elif virtual_target is not None and key in (81, 65361):   # ←
                    virtual_target.press_direction("left")
                elif virtual_target is not None and key in (83, 65363):   # →
                    virtual_target.press_direction("right")
                elif virtual_target is not None and key in (82, 65362):   # ↑
                    virtual_target.press_direction("up")
                elif virtual_target is not None and key in (84, 65364):   # ↓
                    virtual_target.press_direction("down")
                elif virtual_target is not None and key in (ord("v"), ord("V")):
                    print(f"[SIMDRONE] pattern -> {virtual_target.cycle_pattern()}")
                elif virtual_target is not None and key in (ord("n"), ord("N")):
                    # สลับระยะโดรน 50→100→150→200m (ระยะกำหนดขนาด+conf+ความเร็วเชิงมุม)
                    _sim_ranges = [50.0, 100.0, 150.0, 200.0]
                    _cur_r = min(_sim_ranges, key=lambda r: abs(r - virtual_target.range_m))
                    _nxt_r = _sim_ranges[(_sim_ranges.index(_cur_r) + 1) % len(_sim_ranges)]
                    virtual_target.set_range(_nxt_r)
                    _px = virtual_target.box_deg * px_per_deg_x if px_per_deg_x else 0.0
                    print(f"[SIMDRONE] range -> {_nxt_r:.0f}m  ~{_px:.0f}px  conf={virtual_target.det_conf:.2f}  "
                          f"ang_speed={virtual_target.max_ang_speed:.1f}deg/s", flush=True)

                # เปลี่ยนโหมดแขน (AUTO / MANUAL / SAFE / LOCK) ด้วยคีย์บอร์ด
                if arm_mode_manager is not None and not camera_operator_lock:
                    if key in (ord("a"), ord("A")):
                        if arm_calibration_ready:
                            arm_mode_manager.set_mode(MODE_AUTO)
                        else:
                            print(
                                "Gun Aim Assist: AUTO needs pixel_per_degree calibration — "
                                "run cam4_arm_mouse_grid_calibrator first."
                            )
                    elif key in (ord("m"), ord("M")):
                        if cam8_mapping_embed is not None:
                            _enter_cam8_mapping()
                        else:
                            arm_mode_manager.set_mode(MODE_MANUAL)
                    elif key in (ord("f"), ord("F")):
                        arm_mode_manager.set_mode(MODE_SAFE)
                    elif key in (ord("l"), ord("L")):
                        if arm_calibration_ready:
                            arm_mode_manager.set_mode(MODE_LOCK)
                        else:
                            print(
                                "Gun Aim Assist: LOCK needs pixel_per_degree calibration — "
                                "run cam4_arm_mouse_grid_calibrator first."
                            )
                    elif key in (ord("k"), ord("K")):
                        # keyboard acquire: ล็อกเป้าที่ดีสุดใต้ crosshair (ไม่ต้องใช้จอย) — ต้องอยู่โหมด LOCK
                        if arm_mode_manager.mode == MODE_LOCK:
                            kb_lock_acquire = True
                            print("[LOCK] keyboard acquire (K) — ล็อกเป้าใต้ crosshair")
                        else:
                            print("[LOCK] กด L เข้าโหมด LOCK ก่อน แล้วค่อยกด K ล็อกเป้า")
                    # เมื่อสลับออกจาก LOCK ด้วยคีย์บอร์ด เคลียร์ CSRT tracker + lock_kalman
                    if arm_mode_manager.mode != MODE_LOCK:
                        lock_csrt_tracker = None
                        lock_csrt_initialized = False
                        lock_csrt_bbox = None
                        lock_csrt_smooth_px = None
                        lock_csrt_smooth_py = None
                        lock_csrt_lost = False
                        lock_track_frame_count = 0
                        pending_lock_csrt_bbox = None
                        lock_kalman.reset()
                        pd_controller.reset()

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
            elif app_mode == "settings2" and settings_screen is not None:
                _r = settings_screen.handle_key(key)
                if _r == "exit":
                    if settings_screen.is_dirty():
                        print("[SETTINGS] ออกโดยไม่บันทึก — ค่าที่แก้ถูกทิ้ง (Ctrl+S เพื่อบันทึก)")
                    _exit_settings()
                elif _r in ("saved", "reset"):
                    # ค่า live ถูก apply ไปแล้วตอนแก้; ที่ต้องทำคือ derive ค่าที่ผูกกับ ppd ใหม่
                    # (เช่น แก้ LOCK_MEAS_SIGMA_PX → Kalman R ต้องคำนวณใหม่)
                    if px_per_deg_x:
                        _q2, _r2 = lock_bearing_kalman_qr(px_per_deg_x)
                        lock_kalman.set_noise(q=_q2, r=_r2)
            elif app_mode == "wizard" and calib_wizard is not None:
                _r = calib_wizard.handle_key(key, arm_controller)
                if _r == "exit":
                    _exit_wizard(reload_calib=True)
                elif _r == "saved":
                    _reload_calibration("WIZARD")
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
            elif app_mode == "cam4_pxdeg_calib":
                if key in (27,) or key in (ord("g"), ord("G")):
                    _exit_pxdeg_calib()
                elif cam4_pxdeg_calibrator_embed is not None and arm_controller is not None:
                    _px_key_result = cam4_pxdeg_calibrator_embed.handle_key(key, arm_controller)
                    if _px_key_result == "recalibrated":
                        _exit_pxdeg_calib(reload_calib=True)
                        _enter_pxdeg_calib()
                    elif _px_key_result == "saved":
                        _exit_pxdeg_calib(reload_calib=True)
                if js_state is not None:
                    if (
                        getattr(js_state, "confirm_pressed", False)
                        and not pxdeg_prev_confirm
                        and cam4_pxdeg_calibrator_embed is not None
                        and arm_controller is not None
                    ):
                        _px_confirm = cam4_pxdeg_calibrator_embed.confirm_via_joystick(arm_controller)
                        if _px_confirm == "recalibrated":
                            _exit_pxdeg_calib(reload_calib=True)
                            _enter_pxdeg_calib()
                        elif _px_confirm == "saved":
                            _exit_pxdeg_calib(reload_calib=True)
                    pxdeg_prev_confirm = bool(getattr(js_state, "confirm_pressed", False))
            elif app_mode == "cam8_mapping":
                if key in (27,) or key in (ord("m"), ord("M")):
                    _exit_cam8_mapping()
                elif key in (ord("p"), ord("P")):
                    _manual_arm_reconnect(arm_controller)
                elif key in (ord("h"), ord("H")):
                    if arm_controller is not None and cam8_mapping_embed is not None and _CAM8_CALIB_FILE is not None:
                        cam8_mapping_embed.go_to_reference(arm_controller, _CAM8_CALIB_FILE)
                elif key in (ord("c"), ord("C"), 13, 10):
                    if arm_controller is not None and cam8_mapping_embed is not None:
                        cam8_mapping_embed.confirm_cell(arm_controller)
                elif key in (ord("s"), ord("S")):
                    if (
                        arm_controller is not None
                        and cam8_mapping_embed is not None
                        and _CAM8_CALIB_FILE is not None
                    ):
                        if cam8_mapping_embed.save_json(arm_controller, _CAM8_CALIB_FILE):
                            if _cam8_calib_load is not None:
                                cam8_calib_data = _cam8_calib_load(_CAM8_CALIB_FILE)
                                print(f"[MAP] calib reloaded: {'OK' if cam8_calib_data else 'MISSING'}")
            elif app_mode == "jog":
                # JOG: หมุนแขนเองด้วยคีย์บอร์ด — ส่ง G0 ตรง ไม่ผ่าน mode tick → ทำงานแม้ SAFE/fault
                if key in (27,) or key in (ord("j"), ord("J")):
                    if arm_controller is not None and hasattr(arm_controller, "note_joystick_operator_active"):
                        arm_controller.note_joystick_operator_active(False)
                    app_mode = "normal"
                    print("[JOG] ออกจากโหมด JOG")
                # หมายเหตุ: ไม่ใช้รหัสลูกศรเก่า 81-84 เพราะชนกับตัวอักษร
                # (83=ord("S"), 81=ord("Q")) → ใช้รหัส GTK 65361-65364 + WASD แทน
                elif key in (65363, ord("d"), ord("D")):        # → / D : pan +
                    _arm_jog_step(arm_controller, +jog_step_deg, 0.0)
                elif key in (65361, ord("a"), ord("A")):        # ← / A : pan -
                    _arm_jog_step(arm_controller, -jog_step_deg, 0.0)
                elif key in (65362, ord("w"), ord("W")):        # ↑ / W : tilt +
                    _arm_jog_step(arm_controller, 0.0, +jog_step_deg)
                elif key in (65364, ord("s"), ord("S")):        # ↓ / S : tilt -
                    _arm_jog_step(arm_controller, 0.0, -jog_step_deg)
                elif key in (ord("]"), ord("+"), ord("=")):
                    _ji = jog_step_choices.index(jog_step_deg) if jog_step_deg in jog_step_choices else 2
                    jog_step_deg = jog_step_choices[min(len(jog_step_choices) - 1, _ji + 1)]
                    print(f"[JOG] step -> {jog_step_deg:.2f} deg")
                elif key in (ord("["), ord("-"), ord("_")):
                    _ji = jog_step_choices.index(jog_step_deg) if jog_step_deg in jog_step_choices else 2
                    jog_step_deg = jog_step_choices[max(0, _ji - 1)]
                    print(f"[JOG] step -> {jog_step_deg:.2f} deg")
                elif key in (ord("x"), ord("X")):
                    _arm_jog_unlock(arm_controller)
                elif key in (ord("z"), ord("Z")):
                    _arm_jog_set_home(arm_controller)   # ตั้งตำแหน่งนี้เป็น home ใหม่ (re-zero)
                elif key in (ord("h"), ord("H")):
                    _manual_arm_go_home(arm_controller, arm_mode_manager)

    except KeyboardInterrupt:
        pass
    finally:
        # --- save residual bias on shutdown ---
        try:
            if _residual_bias_learner is not None:
                _residual_bias_learner.save(force=True)
        except Exception as _e:
            print(f"[BiasLearner] shutdown save error: {_e}")
        try:
            if _fire_tune_learner is not None:
                _fire_tune_learner.save()
        except Exception as _e:
            print(f"[FireTune] shutdown save error: {_e}")
        # --- cleanup: disconnect arm, stop YOLO thread, stop cue receiver, release camera ---
        if arm_controller is not None:
            try:
                arm_controller.disconnect()
            except Exception:
                pass
        if arm_cue_receiver is not None:
            try:
                arm_cue_receiver.stop()
            except Exception:
                pass
        if effector_exporter is not None:
            try:
                effector_exporter.stop()
            except Exception:
                pass
        if _rtmp_streamer is not None:
            try:
                _rtmp_streamer.stop()
            except Exception:
                pass
        if _yolo_frame_q is not None:
            try:
                _yolo_frame_q.put_nowait(None)
            except Exception:
                pass
        if cam8_stream is not None:
            try:
                cam8_stream.release()
            except Exception:
                pass
        if cam8_mapping_embed is not None and cam8_mapping_embed.is_active():
            try:
                cam8_mapping_embed.leave(WINDOW_NAME)
            except Exception:
                pass
        if cam4_pxdeg_calibrator_embed is not None and cam4_pxdeg_calibrator_embed.is_active():
            try:
                cam4_pxdeg_calibrator_embed.leave(WINDOW_NAME)
            except Exception:
                pass
        if settings_screen is not None and settings_screen.is_active():
            try:
                settings_screen.leave(WINDOW_NAME)
            except Exception:
                pass
        if calib_wizard is not None and calib_wizard.is_active():
            try:
                calib_wizard.leave(WINDOW_NAME)
            except Exception:
                pass
        cam.release()
        cv2.destroyAllWindows()
        print("Gun Aim Assist: stopped.")


if __name__ == "__main__":
    main()



