"""
gun_aim_assist_vector.py
Gun Aim Assist Vector: ระบบช่วยเล็งกล้องติดปืน (drive แขนแบบ vector)
- เหมือน Gun Aim Assist แต่การขยับแขนจาก bbox YOLO ใช้วิธีของ cam4_mouse_click_vector_hud:
  - โหมด AUTO: ใช้ศูนย์กลาง bbox YOLO เป็นจุดเป้า แล้ว drive ด้วย _variable_step_toward_target + px_per_degree
  - โหมด LOCK: ใช้ bbox YOLO; ปุ่ม 5 เข้า LOCK หรือกดซ้ำ = หมุนแขนไป center bbox YOLO ที่ใกล้ศูนย์เล็งที่สุด
- ปุ่ม LOCK = ปุ่ม 5 (index 4); จาก MANUAL/SAFE/AUTO = สลับเป็น LOCK + หมุนแขนไป center YOLO; กดซ้ำใน LOCK = หมุนแขนไป center YOLO
- ปุ่ม 3 (index 2): ปลด LOCK → MANUAL
- ปุ่ม 4 (index 3) และปุ่ม 6 (index 5): ลบวิธีการออกชั่วคราว — จะสร้างวิธีการใหม่ภายหลัง
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
from pathlib import Path
from typing import Any, Optional, Tuple
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

# Optional: grid/homography lookup (เหมือน cam4_arm_mouse_grid_calibrator — แม่นกว่าสูตร linear เดียว)
try:
    from cam4_arm_grid_lookup import load as load_grid_json, pixel_to_arm_degrees as pixel_to_arm_degrees_grid
except ImportError:
    load_grid_json = None
    pixel_to_arm_degrees_grid = None

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
        NEAR_CROSSHAIR_THRESHOLD_PX,
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
    NEAR_CROSSHAIR_THRESHOLD_PX = 120

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


def _get_test_px_deg_log_path(camera_name: str) -> Path:
    """Path ไฟล์ log สำหรับโหมดเทส pixel_to_degree (คลิก → แขน → บันทึกจุด)."""
    calib_dir = Path(__file__).resolve().parent / "calibration_data"
    return calib_dir / f"{camera_name}_test_move_log_pxdeg.jsonl"


def _append_test_px_deg_log(
    click_pixel: Tuple[float, float],
    target_arm: Tuple[float, float],
    actual_arm: Tuple[float, float],
    camera_name: str,
    output_w: int,
    output_h: int,
) -> None:
    """บันทึกหนึ่งจุด (click_pixel, target_arm, actual_arm) ลง log รูปแบบเดียวกับ calibrator."""
    log_path = _get_test_px_deg_log_path(camera_name)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tx, ty = float(target_arm[0]), float(target_arm[1])
        ax, ay = float(actual_arm[0]), float(actual_arm[1])
        px, py = float(click_pixel[0]), float(click_pixel[1])
        record = {
            "schema": "cam4_test_move_v1",
            "timestamp": time.time(),
            "output_width": output_w,
            "output_height": output_h,
            "click_pixel": [px, py],
            "target_arm": [tx, ty],
            "actual_arm": [ax, ay],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Test Px/Deg: logged to {log_path.name}  click=({px:.0f},{py:.0f}) actual=({ax:.2f},{ay:.2f})")
    except Exception as e:
        print(f"  Failed to write test px/deg log: {e}")


def _recalibrate_px_deg_from_log(camera_name: str, output_w: int, output_h: int) -> bool:
    """อ่าน log โหมดเทส pixel_to_degree แล้ว fit ppdx/ppdy บันทึกลง {camera_name}_pixel_per_degree.json.
    ใช้ตำแหน่งศูนย์เล็งจาก config (gun_aim_assist_config_{camera_name}.json) เป็น origin ตอน fit.
    """
    calib_dir = Path(__file__).resolve().parent / "calibration_data"
    log_path = calib_dir / f"{camera_name}_test_move_log_pxdeg.jsonl"
    out_path = calib_dir / f"{camera_name}_pixel_per_degree.json"
    if not log_path.is_file():
        print("  Re-cal (Px/deg): No log file.", log_path.name)
        return False
    # ใช้ศูนย์เล็งจาก config ตอนนั้น — ถ้าไม่มีไฟล์ใช้กลางจอ
    config_recal = ShooterConfig(camera_name)
    config_recal.load()
    cx = (float(output_w) / 2.0) + config_recal.offset_x
    cy = (float(output_h) / 2.0) + config_recal.offset_y
    log_px: list = []
    log_arm: list = []
    with open(log_path, "r", encoding="utf-8") as f:
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
                log_w = int(rec.get("output_width", output_w))
                log_h = int(rec.get("output_height", output_h))
                if log_w != output_w or log_h != output_h:
                    px = px * output_w / max(1, log_w)
                    py = py * output_h / max(1, log_h)
                log_px.append([px, py])
                log_arm.append([ax, ay])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if len(log_px) < 2:
        print(f"  Re-cal (Px/deg): Need at least 2 points, got {len(log_px)}.")
        return False
    ratios_x = [
        (log_px[i][0] - cx) / ax
        for i, (_, (ax, ay)) in enumerate(zip(log_px, log_arm))
        if abs(ax) > 1e-6
    ]
    ratios_y = [
        (log_px[i][1] - cy) / ay
        for i, (_, (ax, ay)) in enumerate(zip(log_px, log_arm))
        if abs(ay) > 1e-6
    ]
    if not ratios_x or not ratios_y:
        print("  Re-cal (Px/deg): Not enough variation in pan/tilt.")
        return False
    ppdx = float(np.mean(ratios_x))
    ppdy = float(np.mean(ratios_y))
    px_deg_data = {
        "output_width": output_w,
        "output_height": output_h,
        "crosshair": {"x": cx, "y": cy},
        "pixel_per_degree_x": ppdx,
        "pixel_per_degree_y": ppdy,
    }
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(px_deg_data, f, indent=2)
        mean_err = float(
            np.mean(
                [
                    math.sqrt(((log_px[i][0] - cx) / ppdx - log_arm[i][0]) ** 2 + ((log_px[i][1] - cy) / ppdy - log_arm[i][1]) ** 2)
                    for i in range(len(log_px))
                ]
            )
        )
        print(f"  Re-cal (Px/deg): saved {out_path.name}  px_per_deg=({ppdx:.2f}, {ppdy:.2f})  mean_err_deg={mean_err:.3f}")
        if len(log_px) >= 4:
            _save_homography_from_log(camera_name, output_w, output_h, log_px, log_arm, calib_dir)
        return True
    except Exception as e:
        print(f"  Re-cal (Px/deg): save failed: {e}")
        return False


def _save_homography_from_log(
    camera_name: str,
    output_w: int,
    output_h: int,
    log_px: list,
    log_arm: list,
    calib_dir: Optional[Path] = None,
) -> None:
    """Fit homography จาก (log_px -> log_arm) แล้วบันทึก {camera_name}_mouse_grid_lookup.json."""
    if calib_dir is None:
        calib_dir = Path(__file__).resolve().parent / "calibration_data"
    out_path = calib_dir / f"{camera_name}_mouse_grid_lookup.json"
    if len(log_px) < 4 or len(log_arm) < 4:
        return
    try:
        src = np.array(log_px, dtype=np.float32)
        dst = np.array(log_arm, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is None:
            return
        n = min(4, len(log_arm))
        cells = {f"{i // 2}_{i % 2}": [float(log_arm[i][0]), float(log_arm[i][1])] for i in range(n)}
        grid_data = {
            "output_width": output_w,
            "output_height": output_h,
            "homography": H.tolist(),
            "grid_rows": 2,
            "grid_cols": 2,
            "cells": cells,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grid_data, f, indent=2)
        print(f"  Re-cal (Homography): saved {out_path.name}  ({len(log_px)} points)")
    except Exception as e:
        print(f"  Re-cal (Homography): save failed: {e}")


def _on_mouse_test_px_deg(event, x, y, _flags, param):
    """Mouse callback: ในโหมด test_px_deg เท่านั้น เก็บคลิกเป็น frame coordinates (รองรับ fullscreen letterbox)."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if param.get("app_mode") != "test_px_deg":
        return
    fw = param.get("frame_w") or 1
    fh = param.get("frame_h") or 1
    x_off = param.get("x_off") or 0
    y_off = param.get("y_off") or 0
    sw = max(1, param.get("scaled_w") or 1)
    sh = max(1, param.get("scaled_h") or 1)
    # แปลงพิกัดจอ → พิกัด frame (รองรับ fullscreen ที่ภาพอยู่กลางจอ)
    fx = (x - x_off) * fw / sw
    fy = (y - y_off) * fh / sh
    fx = max(0.0, min(fw - 1.0, fx))
    fy = max(0.0, min(fh - 1.0, fy))
    param["pending_click"] = (fx, fy)
    param["last_click"] = (fx, fy)  # ให้ HUD วาดตำแหน่งคลิกล่าสุด


def _count_test_px_deg_log_points(camera_name: str) -> int:
    """นับจำนวนบรรทัด (จุด) ใน log โหมดเทส pixel_to_degree."""
    log_path = _get_test_px_deg_log_path(camera_name)
    if not log_path.is_file():
        return 0
    n = 0
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("schema") == "cam4_test_move_v1" and rec.get("click_pixel") and rec.get("actual_arm"):
                        n += 1
                except Exception:
                    pass
    except Exception:
        pass
    return n


# โหมดเทส Px/Deg: ขนาด crop รอบจุดคลิก + ขนาดสูงสุด panel มุมขวาล่าง
TEST_PX_DEG_CAPTURE_HALF_W = 500
TEST_PX_DEG_CAPTURE_HALF_H = 360
# ขนาด panel มุมขวาล่าง: สัดส่วนของความกว้าง/สูงของเฟรม (สัมพันธ์กับ resolution กล้อง), ไม่เกินค่าสูงสุด
TEST_PX_DEG_PANEL_RATIO = 0.32  # ประมาณ 32% ของความกว้าง/สูงเฟรม
TEST_PX_DEG_PANEL_MAX_W = 800
TEST_PX_DEG_PANEL_MAX_H = 600
TEST_PX_DEG_ARROW_STEP_DEG = 0.2  # ลูกศร/WASD ปรับแขนทีละกี่องศา
TEST_PX_DEG_HOME_THRESHOLD_DEG = 2.5  # ต้องอยู่ที่ home (|pan|,|tilt| ≤ นี้) ก่อนรับคลิก; กด [G]=ยืนยันอยู่ home รับคลิกได้
COLOR_CLICK_MARKER = (0, 0, 255)  # BGR แดง = จุดที่คลิก (ศูนย์เล็ง/คลิก)
COLOR_PREDICTION_CROSSHAIR = (0, 255, 0)  # BGR เขียวสด = จุดทำนาย 0.3 s ให้เห็นชัด (ไม่ใช่ศูนย์เล็ง)


def draw_test_px_deg_hud(frame, w, h, camera_name: str, shared: Optional[dict] = None):
    """โหมดเทส Px/Deg: วาด crosshair กลางจอ + ข้อความมุมขวาบน + ตำแหน่งคลิกล่าสุด + panel มุมขวาล่าง."""
    cx, cy = w / 2.0, h / 2.0
    color = (0, 255, 255)
    cv2.line(frame, (0, int(cy)), (w, int(cy)), color, 2)
    cv2.line(frame, (int(cx), 0), (int(cx), h), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 15, color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
    # แสดงตำแหน่งคลิกล่าสุด (วงกลม + กากบาท)
    if shared and shared.get("last_click") is not None:
        lx, ly = shared["last_click"]
        ix, iy = int(round(lx)), int(round(ly))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(frame, (ix, iy), 12, (0, 255, 0), 2)
            cv2.line(frame, (ix - 8, iy), (ix + 8, iy), (0, 255, 0), 1)
            cv2.line(frame, (ix, iy - 8), (ix, iy + 8), (0, 255, 0), 1)
    n_pts = _count_test_px_deg_log_points(camera_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pending = shared.get("pending_confirm_click_pixel") is not None if shared else False
    lines = [
        "Test Px/Deg: at home → click → Arrows/WASD → Enter  [H]=home  [G]=accept at home",
        f"Points: {n_pts}  [R]=recal  [T]=exit" + ("  (adjust then Enter)" if pending else ""),
    ]
    # ข้อความมุมขวาบน ไม่บังภาพ
    y0 = 52
    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, 0.75, 1)
        max_tw = max(max_tw, tw)
    x_right = w - max_tw - 24
    x_right = max(20, x_right)
    for i, line in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(frame, line, (x_right, y), font, 0.75, (0, 0, 0), 3)
        cv2.putText(frame, line, (x_right, y), font, 0.75, (255, 255, 255), 1)

    # ข้อความหลังกด R (สำเร็จ หรือ ต้องมี ≥2 จุด)
    if shared and shared.get("recal_message") and time.time() < shared.get("recal_message_until", 0):
        msg = shared["recal_message"]
        (mw, mh), _ = cv2.getTextSize(msg, font, 0.7, 1)
        mx = (w - mw) // 2
        my = h - 80
        cv2.putText(frame, msg, (mx, my), font, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, msg, (mx, my), font, 0.7, (0, 255, 255), 1)
    # ต้องอยู่ที่ home (0,0) ก่อนคลิก — เหมือน grid calibrator
    if shared and shared.get("home_required_message_until", 0) > 0 and time.time() < shared["home_required_message_until"]:
        msg = "Go to home (0,0) first — then click"
        (mw, mh), _ = cv2.getTextSize(msg, font, 0.7, 1)
        mx = (w - mw) // 2
        my = h - 120
        cv2.putText(frame, msg, (mx, my), font, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, msg, (mx, my), font, 0.7, (0, 165, 255), 1)

    # Window เล็กๆ มุมขวาล่าง: crop รอบจุดคลิก + กากบาทแดง (เหมือน calibrator) — ขนาดสัมพันธ์กับ resolution เฟรม
    if shared and shared.get("reference_capture") is not None:
        ref = shared["reference_capture"]
        ph, pw = ref.shape[0], ref.shape[1]
        panel_max_w = min(TEST_PX_DEG_PANEL_MAX_W, max(200, int(w * TEST_PX_DEG_PANEL_RATIO)))
        panel_max_h = min(TEST_PX_DEG_PANEL_MAX_H, max(150, int(h * TEST_PX_DEG_PANEL_RATIO)))
        if pw > panel_max_w or ph > panel_max_h:
            scale_ref = min(panel_max_w / pw, panel_max_h / ph, 1.0)
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
            cv2.rectangle(frame, (ox, oy), (ox + pw_d, oy + ph_d), (0, 255, 255), 2)
            label = "Target (arm should point here)"
            cv2.putText(frame, label, (ox, oy - 4), font, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, label, (ox, oy - 4), font, 0.5, (255, 255, 255), 1)


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


# --- Kalman Filter (ปรับได้) ---
KALMAN_PROCESS_NOISE_Q = 0.1        # ค่าสูง = ตาม turn โดรนได้เร็ว (0.01=นิ่ง, 1.0=หักเลี้ยวบ่อย)
KALMAN_MEASUREMENT_NOISE_R = 25.0   # measurement noise baseline
KALMAN_COAST_MAX_SEC = 0.5          # coast ได้นานสูงสุดเมื่อ CSRT/YOLO พลาด (7-8 เฟรมที่ 15fps)
AUTO_KALMAN_COAST_MAX_SEC = 0.5     # coast AUTO mode เมื่อ YOLO พลาด
LEAD_TIME_SEC = 0.20                # เผื่อล่วงหน้า (ชดเชย pipeline lag ~300-400ms บน 15fps RTSP)
SYNC_GRBL_EVERY_N_FRAMES = 10      # sync GRBL position ทุก N เฟรม (แทนทุกเฟรม)

# PD controller gains (ปรับหลังทดสอบแขนจริง)
PD_KP_PAN  = 1.0    # Proportional gain แกน pan
PD_KD_PAN  = 0.05   # Derivative gain แกน pan (เบรกก่อนถึงเป้า)
PD_KP_TILT = 1.0    # Proportional gain แกน tilt
PD_KD_TILT = 0.05   # Derivative gain แกน tilt

# Feedforward velocity gain
KFF_PAN  = 0.5      # 0=ปิด feedforward, 1=เต็ม velocity แกน pan
KFF_TILT = 0.5      # feedforward แกน tilt

# Adaptive yellow vector smoothing
ADAPTIVE_ALPHA_SLOW = 0.94          # alpha เมื่อเป้านิ่ง
ADAPTIVE_ALPHA_FAST = 0.75          # alpha เมื่อเป้าเคลื่อนเร็ว (responsive มากขึ้น)
ADAPTIVE_ALPHA_SPEED_THRESHOLD_PX = 80.0  # px/s ที่ใช้แยก slow/fast

# LOCK mode deadzone — หยุดส่ง command เมื่อแขนเข้าใกล้เป้าพอแล้ว (ป้องกันวิ่งเลยและสั่น)
LOCK_DEADZONE_DEG = 0.5   # องศา — ถ้า error < ค่านี้ หยุดทันที (ปรับได้: 0.3=ถี่, 1.0=กว้าง)
# Pixel deadzone + step scale: ลดเด้งจาก CSRT jitter
LOCK_DEADZONE_PX = 8      # px — ถ้าระยะจากศูนย์เล็งถึงเป้า < ค่านี้ ไม่ส่ง move
LOCK_STEP_SCALE_REF_PX = 50.0  # ระยะ px ที่ step_scale = 1.0; ใกล้กว่านี้ลด step (ลด overshoot)
# Phase  approach: ตอนเริ่มกด lock หมุนเร็วเข้า bbox (วินาที) — ในช่วงนี้ใช้ step_scale=1.0, throttle สั้น
LOCK_APPROACH_PHASE_SEC = 0.8
# ปิด tracking หลังหมุนไป center — แค่ครั้งเดียว ไม่ส่ง move ตาม bbox ต่อ (one-shot)
LOCK_DISABLE_TRACKING = False

# Dynamic lead time limits
DYNAMIC_LEAD_MIN_SEC = 0.05
DYNAMIC_LEAD_MAX_SEC = 0.40


def _build_aim_state_10(
    arm: Any,
    target_pan: float,
    target_tilt: float,
    last_sent_delta: Tuple[float, float],
    t_prev_send: Optional[float],
    err_prev: Optional[Tuple[float, float]],
) -> Tuple[Tuple[float, ...], float, float, float]:
    """
    สร้าง state 10 มิติสำหรับ aim_controller_model: (err_pan, err_tilt, error_deg,
    last_delta_pan, last_delta_tilt, dt_send, d_pan, d_tilt, current_pan_deg, current_tilt_deg).
    คืน (state_tuple, err_pan, err_tilt, error_deg).
    """
    current_pan = getattr(arm, "pos_x", 0.0)
    current_tilt = getattr(arm, "pos_y", 0.0)
    err_pan = target_pan - current_pan
    err_tilt = target_tilt - current_tilt
    error_deg = math.hypot(err_pan, err_tilt)
    last_dp, last_dt = last_sent_delta
    dt_send = (time.time() - t_prev_send) if t_prev_send is not None else 0.0
    d_pan = (err_pan - err_prev[0]) / dt_send if err_prev is not None and dt_send > 1e-6 else 0.0
    d_tilt = (err_tilt - err_prev[1]) / dt_send if err_prev is not None and dt_send > 1e-6 else 0.0
    state_10 = (err_pan, err_tilt, error_deg, last_dp, last_dt, dt_send, d_pan, d_tilt, current_pan, current_tilt)
    return state_10, err_pan, err_tilt, error_deg


class _FakeArm:
    """Wrapper arm ที่ไม่ส่ง G-code แค่เก็บ pos_x/pos_y สำหรับให้ _variable_step_toward_target คืน rule delta โดยไม่ขยับแขนจริง."""
    def __init__(self, source_arm: Any) -> None:
        self.pos_x = getattr(source_arm, "pos_x", 0.0)
        self.pos_y = getattr(source_arm, "pos_y", 0.0)
        self.mm_per_deg_pan = getattr(source_arm, "mm_per_deg_pan", getattr(_config_mod, "CAM4_ARM_MM_PER_DEG_PAN", 1.0) if _config_mod else 1.0)
        self.mm_per_deg_tilt = getattr(source_arm, "mm_per_deg_tilt", getattr(_config_mod, "CAM4_ARM_MM_PER_DEG_TILT", 1.0) if _config_mod else 1.0)
        self._source = source_arm

    def move_relative(self, delta_pan: float, delta_tilt: float, blocking: bool = False) -> None:
        """ไม่ส่ง G-code; อัปเดต pos จำลองเพื่อให้ state ต่อเนื่อง."""
        self.pos_x += delta_pan
        self.pos_y += delta_tilt


class _TargetKalman:
    """
    Kalman Filter 4-state (x, y, vx, vy) สำหรับ smooth + predict ตำแหน่งเป้า
    - update(cx, cy, conf): รับ YOLO confidence ปรับ measurement noise อัตโนมัติ
      conf สูง (ภาพชัด) → เชื่อ measurement; conf ต่ำ (เบลอ) → เชื่อ predict
    - predict_ahead(dt): ทำนายตำแหน่งล่วงหน้า dt วินาที
    - get_velocity_deg_s(ppd_x, ppd_y): ดึง velocity (deg/s) สำหรับ feedforward
    - get_speed_px_s(): ความเร็ว px/s สำหรับ adaptive alpha
    """

    def __init__(self) -> None:
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
        q = KALMAN_PROCESS_NOISE_Q
        self.kf.processNoiseCov = q * np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov[2, 2] *= 10.0  # velocity noise สูงกว่า position
        self.kf.processNoiseCov[3, 3] *= 10.0
        # Measurement noise R (baseline)
        self.kf.measurementNoiseCov = KALMAN_MEASUREMENT_NOISE_R * np.eye(2, dtype=np.float32)
        # Error covariance init
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False
        self._last_update_time: float = 0.0

    def reset(self) -> None:
        self._initialized = False
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

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
        R_scale = KALMAN_MEASUREMENT_NOISE_R / (conf_c ** 2)
        self.kf.measurementNoiseCov = R_scale * np.eye(2, dtype=np.float32)
        meas = np.array([[cx], [cy]], dtype=np.float32)
        if not self._initialized:
            self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self._initialized = True
        self.kf.predict()
        corrected = self.kf.correct(meas)
        return float(corrected[0]), float(corrected[1])

    def predict_ahead(self, dt_sec: float) -> Tuple[float, float]:
        """ทำนายตำแหน่งล่วงหน้า dt_sec วินาทีโดยไม่แก้ไข state."""
        if not self._initialized:
            return 0.0, 0.0
        state = self.kf.statePost
        px = float(state[0]) + float(state[2]) * dt_sec
        py = float(state[1]) + float(state[3]) * dt_sec
        return px, py

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
IOT_SIZE_RATIO_MIN = 0.5        # อัตราส่วน w,h ต่อ track_bbox อย่างน้อย (ใกล้เคียงขนาดที่ lock)
IOT_SIZE_RATIO_MAX = 2.0


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
        self._kalman: Optional[_TargetKalman] = None

    def init_from_bbox(self, xb: int, yb: int, wb: int, hb: int) -> None:
        """เริ่ม track จาก bbox ใหม่ (จาก YOLO หรือ joystick button 4)."""
        self.track_bbox = (xb, yb, wb, hb)
        self.smooth_cx = float(xb + wb * 0.5)
        self.smooth_cy = float(yb + hb * 0.5)
        self.initialized = True
        self.lost = False
        if self._kalman is None:
            self._kalman = _TargetKalman()
        self._kalman.reset()
        self._kalman.update(self.smooth_cx, self.smooth_cy, 1.0)

    def update(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[bool, Optional[Tuple[int,int,int,int]]]:
        """
        อัปเดต tracker ด้วย YOLO detections เฟรมนี้
        - ใช้ IoU matching กับ Kalman predicted bbox
        - ถ้าไม่ match → lost=True (จะ coast ด้วย Kalman ใน main loop)
        คืน (success, bbox) โดย bbox = (x,y,w,h)
        """
        if not self.initialized or self._kalman is None:
            return False, None

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
            self.lost = True
            return False, None

        max_dim = max(frame_w, frame_h)
        region_radius = max_dim * IOT_REGION_RADIUS_RATIO
        tw = self.track_bbox[2] if self.track_bbox else 80
        th = self.track_bbox[3] if self.track_bbox else 80

        # Filter: อยู่ในบริเวณ (ระยะจาก pred) และขนาดใกล้เคียงกับที่ lock
        candidates = []
        for det in detections:
            dx, dy, dw, dh, conf = det
            cx_d = dx + dw * 0.5
            cy_d = dy + dh * 0.5
            dist = math.hypot(cx_d - pred_cx, cy_d - pred_cy)
            if dist > region_radius:
                continue
            if tw >= 1 and th >= 1:
                rw = (float(dw) + 1e-9) / tw
                rh = (float(dh) + 1e-9) / th
                if rw < IOT_SIZE_RATIO_MIN or rw > IOT_SIZE_RATIO_MAX or rh < IOT_SIZE_RATIO_MIN or rh > IOT_SIZE_RATIO_MAX:
                    continue
            det_bbox = (int(dx), int(dy), max(1, int(dw)), max(1, int(dh)))
            candidates.append((det, dist, _iou_bbox(pred_bbox, det_bbox)))

        # เรียงตาม conf สูงสุด แล้วเลือกตัวที่ IoU ผ่านเกณฑ์ (หรือ conf สูงสุดในกลุ่มที่ผ่าน)
        candidates.sort(key=lambda x: (x[0][4], x[2]), reverse=True)  # (det, dist, iou) -> sort by conf then iou
        best_det = None
        for det, _d, iou in candidates:
            if iou >= IOT_IOU_THRESHOLD:
                best_det = det
                break
        if best_det is None and candidates:
            # fallback: ใช้ระยะ center — เอา conf สูงสุดที่อยู่ในบริเวณ
            center_dist_max = max_dim * IOT_CENTER_DIST_MAX_RATIO
            for det, dist, iou in candidates:
                if dist < center_dist_max:
                    best_det = det
                    break
        if best_det is None and candidates:
            best_det = candidates[0][0]

        if best_det is not None:
            dx, dy, dw, dh, conf = best_det
            xb = max(0, int(dx))
            yb = max(0, int(dy))
            wb = max(CSRT_MIN_BBOX, min(int(dw), frame_w - xb))
            hb = max(CSRT_MIN_BBOX, min(int(dh), frame_h - yb))
            self.track_bbox = (xb, yb, wb, hb)
            self.smooth_cx = float(xb + wb * 0.5)
            self.smooth_cy = float(yb + hb * 0.5)
            self.lost = False
            self._kalman.update(self.smooth_cx, self.smooth_cy, float(conf))
            return True, self.track_bbox
        else:
            self.lost = True
            return False, None

    def reset(self) -> None:
        self.initialized = False
        self.lost = True
        self.track_bbox = None
        self.smooth_cx = None
        self.smooth_cy = None
        if self._kalman is not None:
            self._kalman.reset()


def _compute_dynamic_lead(
    avg_pipeline_latency: float,
    speed_px_s: float,
    px_per_deg_x: float = 50.0,
    arm_max_speed_deg_s: float = 240.0,
) -> float:
    """
    คำนวณ lead time แบบ dynamic:
    base = pipeline latency จริง × 1.1 (buffer 10%)
    + speed_factor เมื่อโดรนเร็ว
    clamp ไว้ใน [DYNAMIC_LEAD_MIN_SEC, DYNAMIC_LEAD_MAX_SEC]
    """
    base_lead = avg_pipeline_latency * 1.1
    speed_deg_s = speed_px_s / max(abs(px_per_deg_x), 1.0)
    speed_factor = min(speed_deg_s / max(arm_max_speed_deg_s, 1.0), 1.0) * 0.10
    return max(DYNAMIC_LEAD_MIN_SEC, min(base_lead + speed_factor, DYNAMIC_LEAD_MAX_SEC))


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
FIRE_ADJUST_MAX_PX = 250  # ตอนกดยิง: ปรับแขนได้เฉพาะเมื่อระยะจากศูนย์เล็งถึงเป้าไม่เกิน px นี้
FIRE_PREDICT_AHEAD_SEC = 0.6  # เฉพาะตอนกดยิง ใช้ Kalman predict_ahead(ค่านี้)


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
YOLO_CONF_DETECT = 0.1
YOLO_CONF_LOCK = 0.1
YOLO_CONF_MIN = 0.1   # ใช้กับ detection (backward compat)

# YOLO รันเฉพาะบริเวณกลาง: อัตราส่วนของความกว้าง/สูงที่ใช้เป็น crop กลาง (เช่น 0.5 = 50%)
CENTER_CROP_RATIO = 0.5
# ขนาดที่ส่งเข้า YOLO สำหรับ crop กลาง (1280 = แม่น, 640 = เร็ว) — ถูก override โดย load
YOLO_CENTER_IMGSZ = 1280

# ความเร็ว: รัน YOLO แค่ทุก N เฟรม; เฟรมอื่นใช้ผลล่าสุด
YOLO_INTERVAL = 1
# True = โหลด engine 640 (เร็ว, ใช้ตอน tracking ด้วย)
USE_FAST_YOLO = True
# Engine 640 สำหรับ detection/tracking ความเร็ว (ใช้เมื่อ USE_FAST_YOLO)
YOLO_ENGINE_640_PATH = "last_imgsz640.engine"
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


# =============================================================================
# Config / Camera
# =============================================================================
class ShooterConfig:
    """
    เก็บค่าศูนย์เล็งและค่ากระสุน (ระยะหวังผล, ความเร็ว, น้ำหนัก).
    บันทึก/โหลดจาก calibration_data/ แยกไฟล์ต่อกล้อง: gun_aim_assist_config_{camera_name}.json
    ถ้าไม่ระบุ camera_name ใช้ gun_aim_assist_config.json ใน calibration_data
    """
    FILENAME_DEFAULT = "gun_aim_assist_config.json"

    def __init__(self, camera_name: Optional[str] = None):
        self.offset_x = 0
        self.offset_y = 0
        self.effective_range_m = 100
        self.muzzle_velocity_ms = 900
        self.bullet_weight_g = 9
        self.target_size_m = 0.30
        self.camera_name = camera_name
        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data")
        if camera_name:
            self._path = os.path.join(_dir, f"gun_aim_assist_config_{camera_name}.json")
        else:
            self._path = os.path.join(_dir, self.FILENAME_DEFAULT)

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
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
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
    โหลด YOLO ตาม USE_FAST_YOLO: ถ้า True โหลด engine 640 (เร็ว, ใช้ตอน tracking ด้วย).
    Returns: (yolo_model, imgsz) หรือ (None, None).
    """
    if USE_FAST_YOLO and YOLO_AVAILABLE:
        base_dir = os.path.dirname(__file__)
        path_640 = YOLO_ENGINE_640_PATH if os.path.isabs(YOLO_ENGINE_640_PATH) else os.path.join(base_dir, YOLO_ENGINE_640_PATH)
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


def _move_arm_to_target_center(
    arm_controller,
    target_det,
    cx_frame,
    cy_frame,
    w,
    h,
    px_per_deg_x,
    px_per_deg_y,
    x_lo,
    x_hi,
    y_lo,
    y_hi,
    grid_data_for_drive,
    pixel_to_arm_degrees_grid,
    max_dist_px=None,
    target_px=None,
):
    """
    ย้ายแขนไปยังจุดเป้า (pixel). ถ้า target_px ให้ใช้ (cx_t, cy_t) = target_px;
    ไม่เช่นนั้นใช้ center จาก target_det. ถ้า max_dist_px ไม่ใช่ None จะย้ายเฉพาะเมื่อ dist_px <= max_dist_px.
    Returns True ถ้าย้ายได้, False ถ้าไม่.
    """
    if target_px is not None:
        cx_t = float(target_px[0])
        cy_t = float(target_px[1])
    else:
        if target_det is None:
            return False
        x, y, w_d, h_d, _ = target_det
        cx_t = x + w_d // 2
        cy_t = y + h_d // 2
    if (
        arm_controller is None
        or px_per_deg_x is None
        or px_per_deg_y is None
        or x_lo is None
    ):
        return False
    if hasattr(arm_controller, "sync_position_from_grbl"):
        try:
            arm_controller.sync_position_from_grbl()
        except Exception:
            pass
    ch_x = float(cx_frame)
    ch_y = float(cy_frame)
    dx_px = cx_t - ch_x
    dy_px = cy_t - ch_y
    dist_px = math.hypot(dx_px, dy_px)
    if max_dist_px is not None and dist_px > max_dist_px:
        return False
    near_thresh_lock = NEAR_CROSSHAIR_THRESHOLD_PX * (w / 1920.0)
    use_homography_lock = (
        dist_px > near_thresh_lock
        and grid_data_for_drive is not None
        and pixel_to_arm_degrees_grid is not None
    )
    did_move = False
    if use_homography_lock:
        pan_cur_lock = getattr(arm_controller, "pos_x", 0.0)
        tilt_cur_lock = getattr(arm_controller, "pos_y", 0.0)
        res_center_lock = pixel_to_arm_degrees_grid(
            ch_x, ch_y, grid_data_for_drive, w, h, use_homography=True
        )
        res_lock = pixel_to_arm_degrees_grid(
            cx_t, cy_t, grid_data_for_drive, w, h, use_homography=True
        )
        if res_center_lock is not None and res_lock is not None:
            tpan_lock = pan_cur_lock + (res_lock[0] - res_center_lock[0])
            ttilt_lock = tilt_cur_lock + (res_lock[1] - res_center_lock[1])
            if SWAP_PAN_TILT:
                pan_deg = float(np.clip(tpan_lock, y_lo, y_hi))
                tilt_deg = float(np.clip(ttilt_lock, x_lo, x_hi))
            else:
                pan_deg = float(np.clip(tpan_lock, x_lo, x_hi))
                tilt_deg = float(np.clip(ttilt_lock, y_lo, y_hi))
            try:
                arm_controller.move_absolute(pan_deg, tilt_deg, blocking=True)
                did_move = True
            except Exception:
                pass
    if not did_move:
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
            did_move = True
        except Exception:
            pass
    return did_move


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


# =============================================================================
# HUD
# =============================================================================
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
    """Draw a compact single-line bottom HUD that scales with frame size."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"[{KEY_ADJUST_CENTER}] Aim center   [{KEY_SETTINGS}] Ballistics   [{KEY_QUIT}] Quit"
    margin_x = max(8, int(w * 0.02))
    margin_bottom = max(6, int(h * 0.015))
    pad_x = max(6, int(w * 0.01))
    pad_y = max(3, int(h * 0.006))
    scale = max(0.45, min(0.85, w / 1100.0))
    thickness = 1 if w <= 800 else 2
    max_text_w = max(80, w - (margin_x * 2) - (pad_x * 2))

    while scale > 0.45:
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_text_w:
            break
        scale -= 0.03

    box_x1 = margin_x
    box_y2 = h - margin_bottom
    box_y1 = box_y2 - th - baseline - (pad_y * 2)
    box_x2 = min(w - margin_x, box_x1 + tw + (pad_x * 2))
    text_y = box_y2 - baseline - pad_y

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (box_x1 + pad_x, text_y), font, scale, (255, 255, 255), thickness)


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


def _tick_auto(ctx):
    """AUTO mode: drive แขนจาก Kalman + PD + feedforward + _variable_step_toward_target (และ learned aim ถ้ามี)."""
    arm_controller = ctx.get("arm_controller")
    arm_drive_state = ctx.get("arm_drive_state")
    target_kalman = ctx.get("target_kalman")
    pd_controller = ctx.get("pd_controller")
    smoothed_aim_cx = ctx.get("smoothed_aim_cx")
    smoothed_aim_cy = ctx.get("smoothed_aim_cy")
    cx_frame = ctx.get("cx_frame")
    cy_frame = ctx.get("cy_frame")
    px_per_deg_x = ctx.get("px_per_deg_x")
    px_per_deg_y = ctx.get("px_per_deg_y")
    x_lo = ctx.get("x_lo")
    y_lo = ctx.get("y_lo")
    x_hi = ctx.get("x_hi")
    y_hi = ctx.get("y_hi")
    if (
        arm_controller is None
        or arm_drive_state is None
        or target_kalman is None
        or pd_controller is None
        or smoothed_aim_cx is None
        or smoothed_aim_cy is None
        or px_per_deg_x is None
        or px_per_deg_y is None
        or x_lo is None
    ):
        return
    if _variable_step_toward_target is None:
        return

    _sync_grbl_frame_counter = ctx.get("_sync_grbl_frame_counter", 0)
    _sync_grbl_frame_counter += 1
    if _sync_grbl_frame_counter >= SYNC_GRBL_EVERY_N_FRAMES:
        _sync_grbl_frame_counter = 0
        if hasattr(arm_controller, "sync_position_from_grbl"):
            try:
                arm_controller.sync_position_from_grbl()
            except Exception:
                pass
    ctx["_sync_grbl_frame_counter"] = _sync_grbl_frame_counter

    pan_cur = getattr(arm_controller, "pos_x", 0.0)
    tilt_cur = getattr(arm_controller, "pos_y", 0.0)
    ch_x, ch_y = float(cx_frame), float(cy_frame)
    last_continuous_arm_move_time = ctx.get("last_continuous_arm_move_time", 0.0)
    avg_pipeline_latency = ctx.get("avg_pipeline_latency", LEAD_TIME_SEC)
    aim_last_sent_delta = ctx.get("aim_last_sent_delta", (0.0, 0.0))
    aim_t_prev_send = ctx.get("aim_t_prev_send")
    aim_err_prev = ctx.get("aim_err_prev")
    aim_buffer = ctx.get("aim_buffer")
    pending_aim_transition = ctx.get("pending_aim_transition")
    aim_model = ctx.get("aim_model")
    aim_predict_fn = ctx.get("aim_predict_fn")
    aim_normalize_fn = ctx.get("aim_normalize_fn")
    aim_fake_arm = ctx.get("aim_fake_arm")
    aim_blend_pd = ctx.get("aim_blend_pd", 0.5)
    aim_collect_data = ctx.get("aim_collect_data", False)
    aim_collect_modes = ctx.get("aim_collect_modes", set())
    aim_model_input_dim = ctx.get("aim_model_input_dim", 10)

    _speed_px = target_kalman.get_speed_px_s()
    _lead = _compute_dynamic_lead(avg_pipeline_latency, _speed_px, px_per_deg_x)
    aim_cx, aim_cy = target_kalman.predict_ahead(_lead)
    dx_px = aim_cx - ch_x
    dy_px = aim_cy - ch_y
    dist_px = math.hypot(dx_px, dy_px)
    w_ctx = ctx.get("w")
    h_ctx = ctx.get("h")
    near_thresh = NEAR_CROSSHAIR_THRESHOLD_PX * (float(w_ctx or 1920) / 1920.0)
    grid_data = ctx.get("grid_data")
    use_homography_far = (
        dist_px > near_thresh
        and grid_data is not None
        and pixel_to_arm_degrees_grid is not None
        and w_ctx is not None
        and h_ctx is not None
    )
    if use_homography_far:
        # ใช้จุดศูนย์เล็ง (crosshair) เป็น origin — แปลง relative to crosshair แล้วบวกตำแหน่งแขนปัจจุบัน
        res_center = pixel_to_arm_degrees_grid(ch_x, ch_y, grid_data, w_ctx, h_ctx, use_homography=True)
        res = pixel_to_arm_degrees_grid(aim_cx, aim_cy, grid_data, w_ctx, h_ctx, use_homography=True)
        if res_center is not None and res is not None:
            tpan = pan_cur + (res[0] - res_center[0])
            ttilt = tilt_cur + (res[1] - res_center[1])
            if SWAP_PAN_TILT:
                target_pan = float(np.clip(tpan, y_lo, y_hi))
                target_tilt = float(np.clip(ttilt, x_lo, x_hi))
            else:
                target_pan = float(np.clip(tpan, x_lo, x_hi))
                target_tilt = float(np.clip(ttilt, y_lo, y_hi))
        else:
            use_homography_far = False
    if not use_homography_far:
        raw_pan = dx_px / px_per_deg_x
        raw_tilt = dy_px / px_per_deg_y
        delta_pan, delta_tilt = pd_controller.compute(raw_pan, raw_tilt)
        vel_pan_deg_s, vel_tilt_deg_s = target_kalman.get_velocity_deg_s(px_per_deg_x, px_per_deg_y)
        _dt_ff = time.time() - last_continuous_arm_move_time
        delta_pan += KFF_PAN * vel_pan_deg_s * _dt_ff
        delta_tilt += KFF_TILT * vel_tilt_deg_s * _dt_ff
        if CAMERA_ON_ARM:
            tpan = pan_cur + delta_pan
            ttilt = tilt_cur + delta_tilt
        else:
            tpan = delta_pan
            ttilt = delta_tilt
        if SWAP_PAN_TILT:
            target_pan = float(np.clip(tpan, y_lo, y_hi))
            target_tilt = float(np.clip(ttilt, x_lo, x_hi))
        else:
            target_pan = float(np.clip(tpan, x_lo, x_hi))
            target_tilt = float(np.clip(ttilt, y_lo, y_hi))
    error_deg = math.hypot(target_pan - pan_cur, target_tilt - tilt_cur)
    if error_deg > CONTINUOUS_P_ZONE_FAR_DEG:
        throttle_sec = CONTINUOUS_THROTTLE_SEC
    elif error_deg > CONTINUOUS_P_ZONE_NEAR_DEG:
        throttle_sec = CONTINUOUS_P_THROTTLE_MID_SEC
    else:
        throttle_sec = CONTINUOUS_P_THROTTLE_NEAR_SEC
    min_interval = getattr(arm_controller, "_min_move_interval_sec", 0.02)
    if error_deg > CONTINUOUS_P_VERY_FAR_DEG:
        throttle_sec = max(throttle_sec, min(min_interval, 0.005))
    else:
        throttle_sec = max(throttle_sec, min_interval)

    state_10_auto, err_pan_auto, err_tilt_auto, _ = _build_aim_state_10(
        arm_controller, target_pan, target_tilt, aim_last_sent_delta, aim_t_prev_send, aim_err_prev
    )
    if pending_aim_transition is not None and aim_buffer is not None:
        _st10, _act, _tpan, _ttilt = pending_aim_transition
        _nep = _tpan - pan_cur
        _net = _ttilt - tilt_cur
        _ned = math.hypot(_nep, _net)
        aim_buffer.append({"state": _st10, "action": _act, "next_state": (_nep, _net, _ned), "next_error_deg": _ned, "error_deg": _st10[2]})
    pending_aim_transition = None

    now_auto = time.time()
    use_learned_auto = (
        aim_model is not None
        and aim_predict_fn is not None
        and aim_normalize_fn is not None
        and aim_fake_arm is not None
        and _apply_arm_move_relative is not None
        and (now_auto - last_continuous_arm_move_time) >= throttle_sec
    )
    if use_learned_auto:
        aim_fake_arm.pos_x = pan_cur
        aim_fake_arm.pos_y = tilt_cur
        _, rule_pan, rule_tilt = _variable_step_toward_target(
            aim_fake_arm, target_pan, target_tilt, last_continuous_arm_move_time, throttle_sec, arm_drive_state, step_scale=1.0
        )
        _mm = (getattr(arm_controller, "mm_per_deg_pan", 1.0) + getattr(arm_controller, "mm_per_deg_tilt", 1.0)) / 2.0
        _max_step = (3.0 / _mm) if _mm > 1e-9 else 10.0
        _sv = aim_normalize_fn(*state_10_auto)[:aim_model_input_dim]
        model_pan, model_tilt = aim_predict_fn(aim_model, _sv, _max_step)
        blend_pan = (1.0 - aim_blend_pd) * model_pan + aim_blend_pd * rule_pan
        blend_tilt = (1.0 - aim_blend_pd) * model_tilt + aim_blend_pd * rule_tilt
        _apply_arm_move_relative(arm_controller, blend_pan, blend_tilt)
        ctx["last_continuous_arm_move_time"] = now_auto
        ctx["aim_last_sent_delta"] = (blend_pan, blend_tilt)
        ctx["aim_t_prev_send"] = now_auto
        ctx["aim_err_prev"] = (err_pan_auto, err_tilt_auto)
        if aim_collect_data and "auto" in aim_collect_modes:
            ctx["pending_aim_transition"] = (state_10_auto, (blend_pan, blend_tilt), target_pan, target_tilt)
    else:
        last_continuous_arm_move_time, move_pan, move_tilt = _variable_step_toward_target(
            arm_controller, target_pan, target_tilt, last_continuous_arm_move_time, throttle_sec, arm_drive_state, step_scale=1.0
        )
        ctx["last_continuous_arm_move_time"] = last_continuous_arm_move_time
        ctx["aim_last_sent_delta"] = (move_pan, move_tilt)
        ctx["aim_t_prev_send"] = last_continuous_arm_move_time
        ctx["aim_err_prev"] = (err_pan_auto, err_tilt_auto)
        if aim_collect_data and "auto" in aim_collect_modes:
            ctx["pending_aim_transition"] = (state_10_auto, (move_pan, move_tilt), target_pan, target_tilt)


def _tick_lock(ctx):
    """LOCK mode: จัดการ pending bbox, init CSRT, รอแขนถึง, auto re-center, อัปเดต CSRT, ส่ง move ตาม lock_csrt_smooth."""
    arm_controller = ctx.get("arm_controller")
    frame = ctx.get("frame")
    w = ctx.get("w")
    h = ctx.get("h")
    cx_frame = ctx.get("cx_frame")
    cy_frame = ctx.get("cy_frame")
    px_per_deg_x = ctx.get("px_per_deg_x")
    px_per_deg_y = ctx.get("px_per_deg_y")
    x_lo = ctx.get("x_lo")
    y_lo = ctx.get("y_lo")
    x_hi = ctx.get("x_hi")
    y_hi = ctx.get("y_hi")
    ch_x = float(cx_frame) if cx_frame is not None else 0.0
    ch_y = float(cy_frame) if cy_frame is not None else 0.0

    if arm_controller is None or frame is None or w is None or h is None or px_per_deg_x is None or px_per_deg_y is None or x_lo is None:
        return
    if _variable_step_toward_target is None:
        return

    pending_lock_csrt_bbox = ctx.get("pending_lock_csrt_bbox")
    lock_csrt_initialized = ctx.get("lock_csrt_initialized", False)
    lock_csrt_tracker = ctx.get("lock_csrt_tracker")
    lock_csrt_smooth_px = ctx.get("lock_csrt_smooth_px")
    lock_csrt_smooth_py = ctx.get("lock_csrt_smooth_py")
    lock_csrt_bbox = ctx.get("lock_csrt_bbox")
    lock_csrt_lost = ctx.get("lock_csrt_lost", False)
    lock_track_frame_count = ctx.get("lock_track_frame_count", 0)
    lock_kalman = ctx.get("lock_kalman")

    # ปุ่ม 4 และ 6 ถูกลบออกชั่วคราว — จะสร้างวิธีการใหม่ภายหลัง

    # Write back all lock state to ctx
    ctx["pending_lock_csrt_bbox"] = pending_lock_csrt_bbox
    ctx["lock_csrt_initialized"] = lock_csrt_initialized
    ctx["lock_csrt_tracker"] = lock_csrt_tracker
    ctx["lock_csrt_smooth_px"] = lock_csrt_smooth_px
    ctx["lock_csrt_smooth_py"] = lock_csrt_smooth_py
    ctx["lock_csrt_bbox"] = lock_csrt_bbox
    ctx["lock_csrt_lost"] = lock_csrt_lost
    ctx["lock_track_frame_count"] = lock_track_frame_count


def main():
    camera_name = CAMERA_NAME if CAMERA_NAME is not None else ACTIVE_CAMERA
    _set_active_camera_for_calibration(camera_name)
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
        # โหมดจริง/จำลอง: ใช้แขนกับกล้องใดก็ได้ (cam3, cam4 หรือกล้องที่เพิ่มในอนาคต)
        if use_arm:
            try:
                if use_sim and SimCam4ArmController is not None:
                    print("Gun Aim Assist: initializing SimCam4ArmController (simulation mode)...")
                    arm_controller = SimCam4ArmController()
                else:
                    print("Gun Aim Assist: initializing Cam4ArmController...")
                    arm_controller = Cam4ArmController()
                if arm_controller.connect():
                    arm_tracker = Cam4ArmTracker(arm_controller, camera_name=camera_name)
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
    # Shared state สำหรับ mouse callback โหมดเทส Px/Deg (อ้างอิงจาก main loop)
    _test_px_deg_shared = {
        "app_mode": "normal",
        "pending_click": None,
        "last_click": None,
        "reference_capture": None,
        "pending_confirm_click_pixel": None,   # (px, py) รอ Enter บันทึก
        "pending_confirm_target_arm": None,   # (pan_deg, tilt_deg) ที่ส่งไป
        "recal_message": None,                 # ข้อความหลังกด R (สำเร็จ/ต้องมี ≥2 จุด)
        "recal_message_until": 0.0,
        "home_required_message_until": 0.0,    # แสดง "Go to home first" (เหมือน grid calibrator)
        "skip_home_check_once": False,         # กด G = ยืนยันอยู่ home รับคลิกครั้งถัดไปโดยไม่เช็ค pos_x/pos_y
        "frame_w": 1,
        "frame_h": 1,
        "display_w": 1,
        "display_h": 1,
        "x_off": 0,
        "y_off": 0,
        "scaled_w": 1,
        "scaled_h": 1,
    }
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse_test_px_deg, _test_px_deg_shared)

    config = ShooterConfig(camera_name)
    config.load()

    if USE_SHORT_BEEP:
        beep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SOUND_FILE_SHORT)
        create_short_beep_wav(beep_path, duration_sec=BEEP_SHORT_DURATION_SEC)

    app_mode = "normal"
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

    # Vector drive (AUTO/LOCK): px_per_degree, limits, arm drive state
    px_per_deg_x = None
    px_per_deg_y = None
    x_lo = y_lo = x_hi = y_hi = None
    arm_drive_state = None
    last_continuous_arm_move_time = 0.0
    grid_data_for_drive = None
    if arm_controller is not None and _variable_step_toward_target is not None:
        px_per_deg_x, px_per_deg_y = _load_px_per_deg()
        if load_grid_json is not None and pixel_to_arm_degrees_grid is not None:
            _grid_path = _VECTOR_CALIB_DIR / f"{camera_name}_mouse_grid_lookup.json"
            if _grid_path.is_file():
                grid_data_for_drive = load_grid_json(_grid_path)
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

    # LOCK mode: ใช้ bbox จาก YOLO เท่านั้น (IoU tracker + Kalman), ไม่ใช้ CSRT
    lock_iou_tracker = _SimpleIoUTracker()
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
    prev_csrt_center_bbox_pressed = False
    prev_lock_csrt_move_pressed = False
    if arm_controller is not None and _variable_step_toward_target is not None:
        lock_arm_drive_state = _ArmDriveState()
        lock_arm_drive_state.continuous_target_time_prev = time.time() - 0.05
        lock_last_arm_move_time = time.time() - 0.05

    # Learned aim model: state สำหรับ inference + เก็บข้อมูล
    aim_last_sent_delta: Tuple[float, float] = (0.0, 0.0)
    aim_t_prev_send: Optional[float] = None
    aim_err_prev: Optional[Tuple[float, float]] = None
    aim_buffer: list = []
    pending_aim_transition: Optional[Tuple[Tuple[float, ...], Tuple[float, float], float, float]] = None  # (state_10, action, target_pan, target_tilt)
    aim_model = None
    aim_predict_fn = None
    aim_normalize_fn = None
    aim_model_input_dim = 10
    aim_use_model = False
    aim_blend_pd = 0.5
    aim_collect_data = False
    aim_collect_modes: set = set()
    aim_fake_arm = None
    _aim_npz_cap = 50000  # จำกัดจำนวน transition ใน npz

    if _config_mod is not None and arm_controller is not None:
        aim_use_model = bool(getattr(_config_mod, "CAM4_ARM_USE_LEARNED_AIM_MODEL", False))
        aim_blend_pd = float(getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_BLEND_PD", 0.5))
        aim_blend_pd = max(0.0, min(1.0, aim_blend_pd))
        aim_collect_data = bool(getattr(_config_mod, "CAM4_ARM_AIM_COLLECT_DATA", False))
        modes_str = (getattr(_config_mod, "CAM4_ARM_AIM_COLLECT_DATA_MODES", "lock") or "").strip().lower()
        aim_collect_modes = {m.strip() for m in modes_str.split(",") if m.strip()}
        # ปิดเก็บข้อมูลเทรน model ไปก่อน — ไม่เก็บอะไรเลย
        aim_collect_data = False
        aim_collect_modes = set()
        if aim_use_model:
            _root = Path(__file__).resolve().parent
            _rel = getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_MODEL_PATH", "aim_controller_model/aim_model.pt")
            _model_path = _root / _rel if not Path(_rel).is_absolute() else Path(_rel)
            _onnx_path = _model_path.with_suffix(".onnx") if _model_path.suffix.lower() == ".pt" else _model_path
            try:
                if _onnx_path.suffix.lower() == ".onnx" and _onnx_path.exists():
                    from aim_controller_model.model import load_onnx, predict_delta_onnx, normalize_state
                    aim_model = load_onnx(_onnx_path)
                    if aim_model is not None:
                        sh = aim_model.get_inputs()[0].shape
                        aim_model_input_dim = int(sh[1]) if len(sh) > 1 else 10
                        aim_predict_fn = lambda m, sv, ms: predict_delta_onnx(m, sv, ms)
                        aim_normalize_fn = normalize_state
                        print(f"Gun Aim Assist: loaded learned aim ONNX from {_onnx_path}")
                elif _model_path.exists():
                    from aim_controller_model.model import load_model, predict_delta, normalize_state
                    aim_model = load_model(_model_path)
                    if aim_model is not None:
                        aim_model_input_dim = getattr(aim_model, "input_dim", 10)
                        aim_predict_fn = predict_delta
                        aim_normalize_fn = normalize_state
                        print(f"Gun Aim Assist: loaded learned aim model from {_model_path}")
            except Exception as e:
                print(f"Gun Aim Assist: could not load learned aim model: {e}")
                aim_model = None
                aim_predict_fn = None
                aim_normalize_fn = None
        if aim_model is not None and _variable_step_toward_target is not None:
            aim_fake_arm = _FakeArm(arm_controller)

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
    lock_kalman = _TargetKalman()     # LOCK mode: smooth CSRT center
    pd_controller = _ArmPDController()

    # Kalman Coasting state
    auto_coast_start_time: float = 0.0   # เวลาที่ YOLO พลาดครั้งแรก (AUTO mode)
    lock_coast_start_time: float = 0.0   # เวลาที่ CSRT lost (LOCK mode)

    # Pipeline latency measurement (สำหรับ dynamic lead)
    import queue as _queue_mod
    import threading as _threading_mod
    pipeline_latency_history: deque = deque(maxlen=30)
    avg_pipeline_latency: float = LEAD_TIME_SEC  # initial estimate
    current_yolo_conf: float = YOLO_CONF_DETECT  # ค่า conf ที่ใช้กับ YOLO ปัจจุบัน (สำหรับแสดงบน HUD)

    # YOLO async thread queues
    _yolo_frame_q = _queue_mod.Queue(maxsize=1)
    _yolo_result_q = _queue_mod.Queue(maxsize=2)
    _yolo_t_frame: float = 0.0  # timestamp ที่ส่ง frame เข้า YOLO

    def _yolo_worker(model, imgsz, fq, rq):
        while True:
            item = fq.get()
            if item is None:
                break
            if len(item) >= 7:
                crop, x0, y0, cw, ch, t_sent, conf = item
            else:
                crop, x0, y0, cw, ch, t_sent = item
                conf = YOLO_CONF_DETECT
            dets = detect_yolo_full_frame(model, crop, conf, imgsz=imgsz)
            try:
                rq.put_nowait((dets, x0, y0, cw, ch, t_sent))
            except _queue_mod.Full:
                try:
                    rq.get_nowait()
                except _queue_mod.Empty:
                    pass
                rq.put_nowait((dets, x0, y0, cw, ch, t_sent))

    _yolo_thread = _threading_mod.Thread(
        target=_yolo_worker,
        args=(yolo_model, yolo_center_imgsz, _yolo_frame_q, _yolo_result_q),
        daemon=True,
    )
    _yolo_thread.start()

    # Sync GRBL throttle counter
    _sync_grbl_frame_counter: int = 0

    try:
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

            # อ่านจอยสติ๊กครั้งเดียวต่อเฟรม (ใช้ทั้งตอนวาดกากบาทจุดทำนายและตอนจัดการปุ่ม)
            current_js_state = joystick_reader.read() if (
                joystick_reader is not None and getattr(joystick_reader, "enabled", False)
            ) else None

            if app_mode == "normal":
                frame_counter += 1
                lock_csrt_active = (
                    arm_mode_manager is not None
                    and arm_mode_manager.mode == MODE_LOCK
                    and lock_csrt_initialized
                    and not lock_csrt_lost
                )
                # conf: ตอน LOCK tracking ใช้ 0.05, ตอน detection ใช้ 0.2
                _yolo_conf = (
                    YOLO_CONF_LOCK
                    if (
                        arm_mode_manager is not None
                        and arm_mode_manager.mode == MODE_LOCK
                        and lock_iou_tracker.initialized
                        and not lock_iou_tracker.lost
                    )
                    else YOLO_CONF_DETECT
                )
                current_yolo_conf = _yolo_conf

                # --- YOLO async: ส่ง frame เข้า thread (รันทั้ง AUTO และ LOCK, ใช้ conf ตามโหมด) ---
                if frame_counter % YOLO_INTERVAL == 0:
                    center_crop, x0_crop, y0_crop, cw_crop, ch_crop = crop_center_and_resize(
                        frame, CENTER_CROP_RATIO, yolo_center_imgsz
                    )
                    _t_sent = time.time()
                    try:
                        _yolo_frame_q.put_nowait((
                            center_crop, x0_crop, y0_crop, cw_crop, ch_crop, _t_sent, _yolo_conf
                        ))
                    except _queue_mod.Full:
                        pass  # YOLO ยังไม่ว่าง drop frame นี้

                # --- รับ YOLO result จาก thread (non-blocking) ---
                try:
                    _yolo_res = _yolo_result_q.get_nowait()
                    _dets_crop, _x0, _y0, _cw, _ch, _t_yolo_sent = _yolo_res
                    last_detections = map_detections_to_full_frame(
                        _dets_crop, _x0, _y0, _cw, _ch, yolo_center_imgsz
                    )
                    # วัด pipeline latency
                    _lat = time.time() - _t_yolo_sent
                    if 0.01 < _lat < 2.0:
                        pipeline_latency_history.append(_lat)
                        if pipeline_latency_history:
                            avg_pipeline_latency = sum(pipeline_latency_history) / len(pipeline_latency_history)
                    # Sticky target + grace
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
                    # LOCK: อัปเดต IoU tracker จาก YOLO detections (ใช้ bbox จาก YOLO เท่านั้น)
                    if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK and lock_iou_tracker.initialized and last_detections is not None:
                        lock_iou_tracker.update(last_detections, w, h)
                except _queue_mod.Empty:
                    pass  # ยังไม่มี result ใหม่ ใช้ detections เดิม

                # Sync LOCK state จาก IoU tracker (ทุกเฟรม) — ข้ามเมื่อ LOCK_DISABLE_TRACKING เพื่อไม่ทับ lock_csrt_initialized/lost หลัง one-shot
                if arm_mode_manager is not None and arm_mode_manager.mode == MODE_LOCK and not LOCK_DISABLE_TRACKING:
                    lock_csrt_initialized = lock_iou_tracker.initialized
                    lock_csrt_lost = lock_iou_tracker.lost
                    lock_csrt_bbox = lock_iou_tracker.track_bbox
                    lock_csrt_smooth_px = lock_iou_tracker.smooth_cx
                    lock_csrt_smooth_py = lock_iou_tracker.smooth_cy
                    # อัปเดต lock_kalman เพื่อให้จุดทำนาย 0.3 s แสดงได้ (predict_ahead ต้อง _initialized)
                    if lock_csrt_initialized and not lock_csrt_lost and lock_csrt_smooth_px is not None and lock_csrt_smooth_py is not None:
                        lock_kalman.update(float(lock_csrt_smooth_px), float(lock_csrt_smooth_py), 1.0)

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
                if target_det is not None or lock_csrt_active or lock_has_pending:
                    if target_det is not None:
                        x, y, w_d, h_d, conf_det = target_det
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
                        # AUTO Kalman Coasting: YOLO พลาด ให้ predict ต่อ
                        coast_elapsed = time.time() - auto_coast_start_time
                        if coast_elapsed < AUTO_KALMAN_COAST_MAX_SEC:
                            smoothed_aim_cx, smoothed_aim_cy = target_kalman.predict_ahead(coast_elapsed)
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

                    # Vector drive: dispatch to mode tick
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
                            "grid_data": grid_data_for_drive,
                            "x_lo": x_lo, "y_lo": y_lo, "x_hi": x_hi, "y_hi": y_hi,
                            "last_continuous_arm_move_time": last_continuous_arm_move_time,
                            "avg_pipeline_latency": avg_pipeline_latency,
                            "_sync_grbl_frame_counter": _sync_grbl_frame_counter,
                            "aim_last_sent_delta": aim_last_sent_delta,
                            "aim_t_prev_send": aim_t_prev_send,
                            "aim_err_prev": aim_err_prev,
                            "aim_buffer": aim_buffer,
                            "pending_aim_transition": pending_aim_transition,
                            "aim_model": aim_model,
                            "aim_predict_fn": aim_predict_fn,
                            "aim_normalize_fn": aim_normalize_fn,
                            "aim_fake_arm": aim_fake_arm,
                            "aim_blend_pd": aim_blend_pd,
                            "aim_collect_data": aim_collect_data,
                            "aim_collect_modes": aim_collect_modes,
                            "aim_model_input_dim": aim_model_input_dim,
                            "frame": frame, "w": w, "h": h,
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
                        }
                        if arm_mode_manager.mode == MODE_AUTO:
                            _tick_auto(_ctx)
                        elif arm_mode_manager.mode == MODE_LOCK:
                            _tick_lock(_ctx)
                        elif arm_mode_manager.mode == MODE_MANUAL:
                            _tick_manual(_ctx)
                        elif arm_mode_manager.mode == MODE_SAFE:
                            _tick_safe(_ctx)
                        last_continuous_arm_move_time = _ctx.get("last_continuous_arm_move_time", last_continuous_arm_move_time)
                        _sync_grbl_frame_counter = _ctx.get("_sync_grbl_frame_counter", _sync_grbl_frame_counter)
                        aim_last_sent_delta = _ctx.get("aim_last_sent_delta", aim_last_sent_delta)
                        aim_t_prev_send = _ctx.get("aim_t_prev_send", aim_t_prev_send)
                        aim_err_prev = _ctx.get("aim_err_prev", aim_err_prev)
                        pending_aim_transition = _ctx.get("pending_aim_transition", pending_aim_transition)
                        pending_lock_csrt_bbox = _ctx.get("pending_lock_csrt_bbox", pending_lock_csrt_bbox)
                        lock_csrt_initialized = _ctx.get("lock_csrt_initialized", lock_csrt_initialized)
                        lock_csrt_tracker = _ctx.get("lock_csrt_tracker", lock_csrt_tracker)
                        lock_csrt_smooth_px = _ctx.get("lock_csrt_smooth_px", lock_csrt_smooth_px)
                        lock_csrt_smooth_py = _ctx.get("lock_csrt_smooth_py", lock_csrt_smooth_py)
                        lock_csrt_bbox = _ctx.get("lock_csrt_bbox", lock_csrt_bbox)
                        lock_csrt_lost = _ctx.get("lock_csrt_lost", lock_csrt_lost)
                        lock_track_frame_count = _ctx.get("lock_track_frame_count", lock_track_frame_count)
                        lock_last_arm_move_time = _ctx.get("lock_last_arm_move_time", lock_last_arm_move_time)

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
                            _spd_for_alpha = target_kalman.get_speed_px_s() if arm_mode_manager and arm_mode_manager.mode == MODE_AUTO else lock_kalman.get_speed_px_s()
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

                        # LOCK และ AUTO: จุดทำนาย 0.3 s — กากบาทใหญ่ + จุดวงกลมเขียวให้เห็นชัด
                        if target_px is not None:
                            _kalman = target_kalman if (arm_mode_manager is not None and arm_mode_manager.mode == MODE_AUTO) else lock_kalman
                            if getattr(_kalman, "_initialized", False):
                                pred_x, pred_y = _kalman.predict_ahead(0.3)
                                px_pred = int(round(pred_x))
                                py_pred = int(round(pred_y))
                                cross_len = 45
                                cross_thick = 4
                                x1 = max(0, px_pred - cross_len)
                                x2 = min(w, px_pred + cross_len)
                                y1 = max(0, py_pred - cross_len)
                                y2 = min(h, py_pred + cross_len)
                                # ขอบดำแล้ววาดทับด้วยสีเขียว ให้เห็นทุกพื้นหลัง
                                cv2.line(frame, (x1, py_pred), (x2, py_pred), (0, 0, 0), cross_thick + 2)
                                cv2.line(frame, (px_pred, y1), (px_pred, y2), (0, 0, 0), cross_thick + 2)
                                cv2.line(frame, (x1, py_pred), (x2, py_pred), COLOR_PREDICTION_CROSSHAIR, cross_thick)
                                cv2.line(frame, (px_pred, y1), (px_pred, y2), COLOR_PREDICTION_CROSSHAIR, cross_thick)
                                # จุดวงกลมทึบที่ตำแหน่งทำนาย
                                cv2.circle(frame, (px_pred, py_pred), 12, (0, 0, 0), 3)
                                cv2.circle(frame, (px_pred, py_pred), 10, COLOR_PREDICTION_CROSSHAIR, -1)

            elif app_mode == "adjust_center":
                draw_adjust_center_hud(frame, config, w, h)
            elif app_mode == "test_px_deg":
                draw_test_px_deg_hud(frame, w, h, camera_name, _test_px_deg_shared)
                # ประมวลผลคลิก: ต้องอยู่ที่ home ก่อน (เหมือน grid calibrator) แล้วส่งแขน — ใช้ grid/homography ถ้ามี
                if _test_px_deg_shared.get("pending_click") is not None and arm_controller is not None and x_lo is not None:
                    skip_check = _test_px_deg_shared.pop("skip_home_check_once", False)
                    if not skip_check:
                        if hasattr(arm_controller, "sync_position_from_grbl"):
                            try:
                                arm_controller.sync_position_from_grbl()
                            except Exception:
                                pass
                        pos_x = getattr(arm_controller, "pos_x", 0.0)
                        pos_y = getattr(arm_controller, "pos_y", 0.0)
                        if abs(pos_x) > TEST_PX_DEG_HOME_THRESHOLD_DEG or abs(pos_y) > TEST_PX_DEG_HOME_THRESHOLD_DEG:
                            _test_px_deg_shared["home_required_message_until"] = time.time() + 3.0
                        else:
                            skip_check = True
                    if skip_check:
                        px, py = _test_px_deg_shared.pop("pending_click")
                        _test_px_deg_shared["pending_click"] = None
                        cx_f, cy_f = config.get_center(w, h)
                        tpan, ttilt = None, None
                        grid_path = _VECTOR_CALIB_DIR / f"{camera_name}_mouse_grid_lookup.json"
                        if load_grid_json is not None and pixel_to_arm_degrees_grid is not None and grid_path.is_file():
                            grid_data = load_grid_json(grid_path)
                            if grid_data:
                                res = pixel_to_arm_degrees_grid(px, py, grid_data, w, h, use_homography=True)
                                if res is not None:
                                    tpan, ttilt = res[0], res[1]
                        if tpan is None or ttilt is None:
                            ppdx = px_per_deg_x if px_per_deg_x is not None else PX_PER_DEG_X_DEFAULT
                            ppdy = px_per_deg_y if px_per_deg_y is not None else PX_PER_DEG_Y_DEFAULT
                            dx = px - cx_f
                            dy = py - cy_f
                            tpan = dx / ppdx
                            ttilt = dy / ppdy
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
                        if hasattr(arm_controller, "sync_position_from_grbl"):
                            try:
                                arm_controller.sync_position_from_grbl()
                            except Exception:
                                pass
                        _test_px_deg_shared["pending_confirm_click_pixel"] = (px, py)
                        _test_px_deg_shared["pending_confirm_target_arm"] = (pan_deg, tilt_deg)
                        # Crop รอบจุดคลิก → วาดกากบาทแดง → เก็บไว้แสดงใน panel มุมขวาล่าง
                        half_w = TEST_PX_DEG_CAPTURE_HALF_W
                        half_h = TEST_PX_DEG_CAPTURE_HALF_H
                        x0 = max(0, int(px) - half_w)
                        y0 = max(0, int(py) - half_h)
                        x1 = min(w, int(px) + half_w)
                        y1 = min(h, int(py) + half_h)
                        if x1 > x0 and y1 > y0:
                            roi = frame[y0:y1, x0:x1].copy()
                            roi_h, roi_w = roi.shape[0], roi.shape[1]
                            cx_click = int(px - x0)
                            cy_click = int(py - y0)
                            cross_len = min(50, roi_w // 4, roi_h // 4)
                            cross_thick = max(2, min(4, cross_len // 15))
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
                            _test_px_deg_shared["reference_capture"] = roi
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
                js_state = current_js_state if current_js_state is not None else joystick_reader.read()
                # ปุ่ม 12: สลับโหมดความเร็วจอยสติ๊ก (ช้า → กลาง → สูง → ช้า)
                if js_state.sensitivity_cycle_pressed and not prev_sensitivity_cycle_pressed:
                    joystick_mapper.cycle_sensitivity_mode()
                prev_sensitivity_cycle_pressed = js_state.sensitivity_cycle_pressed
                # ปุ่ม 3 (index 2): ปลด LOCK → MANUAL (ไม่ว่าจะ lock มาจากปุ่ม 4 หรือ 5)
                if getattr(js_state, "unlock_pressed", False) and arm_mode_manager.mode == MODE_LOCK:
                    arm_mode_manager.set_mode(MODE_MANUAL)
                # ปุ่มโหมดจาก joystick — ปุ่ม 5 กดซ้ำใน LOCK = lock ใหม่จาก YOLO (reset state, เฟรมถัดไปหมุนไป center bbox)
                elif js_state.mode_switch is not None:
                    if js_state.mode_switch == MODE_LOCK and arm_mode_manager.mode == MODE_LOCK:
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
                    else:
                        arm_mode_manager.set_mode(js_state.mode_switch)
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
                    if arm_mode_manager.mode == MODE_LOCK and js_state.mode_switch == MODE_LOCK:
                        target_det_yolo = (
                            pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                            if last_detections
                            else last_target_det
                        )
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
                            x, y, w_d, h_d, _ = target_det_yolo
                            lock_iou_tracker.init_from_bbox(int(x), int(y), max(1, int(w_d)), max(1, int(h_d)))
                            _move_arm_to_target_center(
                                arm_controller,
                                target_det_yolo,
                                cx_frame,
                                cy_frame,
                                w,
                                h,
                                px_per_deg_x,
                                px_per_deg_y,
                                x_lo,
                                x_hi,
                                y_lo,
                                y_hi,
                                grid_data_for_drive,
                                pixel_to_arm_degrees_grid,
                                max_dist_px=None,
                            )
                # ปุ่มยิงจาก joystick (ทุกโหมด ยกเว้น SAFE): ย้ายแขนไปตำแหน่งทำนาย (0.6 s) ก่อน แล้วค่อยยิง + เล่นเสียง
                if js_state.fire_pressed and arm_mode_manager.mode != MODE_SAFE:
                    now_fire = time.time()
                    target_det_yolo = (
                        pick_best_target(last_detections, cx_frame, cy_frame, min_side)
                        if last_detections
                        else last_target_det
                    )
                    target_px = None
                    if arm_mode_manager.mode == MODE_AUTO and getattr(target_kalman, "_initialized", False):
                        target_px = target_kalman.predict_ahead(FIRE_PREDICT_AHEAD_SEC)
                    elif arm_mode_manager.mode == MODE_LOCK and getattr(lock_kalman, "_initialized", False):
                        target_px = lock_kalman.predict_ahead(FIRE_PREDICT_AHEAD_SEC)
                    if (
                        px_per_deg_x is not None
                        and px_per_deg_y is not None
                        and x_lo is not None
                        and (target_det_yolo is not None or target_px is not None)
                    ):
                        _move_arm_to_target_center(
                            arm_controller,
                            target_det_yolo,
                            cx_frame,
                            cy_frame,
                            w,
                            h,
                            px_per_deg_x,
                            px_per_deg_y,
                            x_lo,
                            x_hi,
                            y_lo,
                            y_hi,
                            grid_data_for_drive,
                            pixel_to_arm_degrees_grid,
                            max_dist_px=FIRE_ADJUST_MAX_PX,
                            target_px=target_px,
                        )
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
            # ปิด overlay calibration อัตโนมัติหลัง 10 วินาที
            if show_calibration_status and calibration_status_until > 0 and time.time() > calibration_status_until:
                show_calibration_status = False
                calibration_status_until = 0.0

            fps_text = f"FPS: {fps:.1f}"
            (fps_tw, fps_th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(frame, fps_text, (w - fps_tw - 28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            # แสดง pipeline latency (YOLO lag) บน HUD
            if pipeline_latency_history:
                lat_text = f"LAT:{avg_pipeline_latency*1000:.0f}ms"
                (lt_tw, _), _ = cv2.getTextSize(lat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.putText(frame, lat_text, (w - lt_tw - 28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            # แสดงค่า YOLO conf ที่ใช้อยู่ (0.2 = detection, 0.05 = LOCK tracking)
            conf_text = f"CONF:{current_yolo_conf:.2f}"
            (conf_tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(frame, conf_text, (w - conf_tw - 28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

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
                _test_px_deg_shared["x_off"] = x_off
                _test_px_deg_shared["y_off"] = y_off
                _test_px_deg_shared["scaled_w"] = scaled_w
                _test_px_deg_shared["scaled_h"] = scaled_h
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
                _test_px_deg_shared["x_off"] = 0
                _test_px_deg_shared["y_off"] = 0
                _test_px_deg_shared["scaled_w"] = display_w
                _test_px_deg_shared["scaled_h"] = display_h
            try:
                cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
            except cv2.error:
                pass
            _test_px_deg_shared["app_mode"] = app_mode
            _test_px_deg_shared["frame_w"] = w
            _test_px_deg_shared["frame_h"] = h
            _test_px_deg_shared["display_w"] = display_w
            _test_px_deg_shared["display_h"] = display_h
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
                # ปุ่ม T: เข้าโหมดเทส Px/Deg (คลิกเก็บจุด → กด R recal & save)
                elif key in (ord("t"), ord("T")):
                    app_mode = "test_px_deg"
                # ปุ่ม R: แสดง/ซ่อนสถานะ calibration (pixel_to_degree) ของกล้องปัจจุบัน
                elif key in (ord("r"), ord("R")):
                    show_calibration_status = not show_calibration_status
                    calibration_status_until = time.time() + 10.0 if show_calibration_status else 0.0

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
            elif app_mode == "test_px_deg":
                if key in (ord("t"), ord("T")) or key == 27:  # T หรือ ESC = ออก
                    app_mode = "normal"
                    _test_px_deg_shared["reference_capture"] = None
                    _test_px_deg_shared["pending_confirm_click_pixel"] = None
                    _test_px_deg_shared["pending_confirm_target_arm"] = None
                    _test_px_deg_shared["recal_message"] = None
                    _test_px_deg_shared["home_required_message_until"] = 0.0
                    _test_px_deg_shared["skip_home_check_once"] = False
                elif key in (ord("h"), ord("H")) and arm_controller is not None:
                    if hasattr(arm_controller, "go_home"):
                        try:
                            arm_controller.go_home(blocking=True)
                        except Exception:
                            pass
                    else:
                        try:
                            arm_controller.move_absolute(0.0, 0.0, blocking=True)
                        except Exception:
                            pass
                elif key in (ord("g"), ord("G")):
                    # ยืนยันอยู่ที่ home — รับคลิกครั้งถัดไปโดยไม่เช็ค pos_x/pos_y (เมื่อแขนอยู่ home จริงแต่ซอฟต์แวร์รายงานไม่ตรง)
                    _test_px_deg_shared["skip_home_check_once"] = True
                elif key in (ord("r"), ord("R")):
                    if _recalibrate_px_deg_from_log(camera_name, w, h):
                        px_per_deg_x, px_per_deg_y = _load_px_per_deg()
                        _test_px_deg_shared["recal_message"] = "Saved pixel_per_degree.json (R recal OK)"
                        _test_px_deg_shared["recal_message_until"] = time.time() + 5.0
                    else:
                        n_pts = _count_test_px_deg_log_points(camera_name)
                        log_path = _get_test_px_deg_log_path(camera_name)
                        if not log_path.is_file() or n_pts < 2:
                            _test_px_deg_shared["recal_message"] = f"R needs >= 2 points (have {n_pts}). Add: click -> Enter"
                            _test_px_deg_shared["recal_message_until"] = time.time() + 5.0
                # Enter: บันทึกจุดลง log แล้วหมุนกลับ home เพื่อทำจุดถัดไป
                elif key in (13, 10):
                    cp = _test_px_deg_shared.get("pending_confirm_click_pixel")
                    tp = _test_px_deg_shared.get("pending_confirm_target_arm")
                    if cp is not None and tp is not None and arm_controller is not None:
                        if hasattr(arm_controller, "sync_position_from_grbl"):
                            try:
                                arm_controller.sync_position_from_grbl()
                            except Exception:
                                pass
                        actual_arm = (getattr(arm_controller, "pos_x", 0.0), getattr(arm_controller, "pos_y", 0.0))
                        _append_test_px_deg_log(cp, tp, actual_arm, camera_name, w, h)
                        _test_px_deg_shared["pending_confirm_click_pixel"] = None
                        _test_px_deg_shared["pending_confirm_target_arm"] = None
                        _test_px_deg_shared["reference_capture"] = None
                        if hasattr(arm_controller, "go_home"):
                            try:
                                arm_controller.go_home(blocking=True)
                            except Exception:
                                pass
                        else:
                            try:
                                arm_controller.move_absolute(0.0, 0.0, blocking=True)
                            except Exception:
                                pass
                # ลูกศร/WASD: ปรับตำแหน่งแขนทีละ step องศา
                elif arm_controller is not None and hasattr(arm_controller, "move_relative"):
                    step_deg = TEST_PX_DEG_ARROW_STEP_DEG
                    if key in (83, 65363, ord("d")):
                        arm_controller.move_relative(step_deg, 0.0, blocking=False)
                    elif key in (81, 65361, ord("a")):
                        arm_controller.move_relative(-step_deg, 0.0, blocking=False)
                    elif key in (82, 65362, ord("w")):
                        arm_controller.move_relative(0.0, step_deg, blocking=False)
                    elif key in (84, 65364, ord("s")):
                        arm_controller.move_relative(0.0, -step_deg, blocking=False)
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

    except KeyboardInterrupt:
        pass
    finally:
        # --- Learned aim: merge buffer + retrain ---
        try:
            min_tr = int(getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_MIN_TRANSITIONS", 20)) if _config_mod else 20
            if len(aim_buffer) >= min_tr:
                npz_dir = Path(__file__).resolve().parent / "aim_controller_model"
                npz_path = npz_dir / "aim_buffer.npz"
                states_list = [list(t["state"]) for t in aim_buffer]
                actions_list = [list(t["action"]) for t in aim_buffer]
                next_states_list = [list(t["next_state"]) for t in aim_buffer]
                if npz_path.exists():
                    old = np.load(npz_path, allow_pickle=True)
                    old_s = old.get("states")
                    old_a = old.get("actions")
                    old_n = old.get("next_states")
                    if old_s is not None and old_a is not None and old_n is not None:
                        states_list = [old_s[i].tolist() if hasattr(old_s[i], "tolist") else list(old_s[i]) for i in range(len(old_s))] + states_list
                        actions_list = [old_a[i].tolist() if hasattr(old_a[i], "tolist") else list(old_a[i]) for i in range(len(old_a))] + actions_list
                        next_states_list = [old_n[i].tolist() if hasattr(old_n[i], "tolist") else list(old_n[i]) for i in range(len(old_n))] + next_states_list
                if len(states_list) > _aim_npz_cap:
                    states_list = states_list[-_aim_npz_cap:]
                    actions_list = actions_list[-_aim_npz_cap:]
                    next_states_list = next_states_list[-_aim_npz_cap:]
                npz_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    states=np.array(states_list, dtype=np.float32),
                    actions=np.array(actions_list, dtype=np.float32),
                    next_states=np.array(next_states_list, dtype=np.float32),
                )
                model_path = Path(__file__).resolve().parent / (getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_MODEL_PATH", "aim_controller_model/aim_model.pt") if _config_mod else "aim_controller_model/aim_model.pt")
                th_red = getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_THRESHOLD_RED_DEG", 0.35) if _config_mod else 0.35
                th_orange = getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_THRESHOLD_ORANGE_DEG", 0.7) if _config_mod else 0.7
                w_red = getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_WEIGHT_RED", 1.5) if _config_mod else 1.5
                w_orange = getattr(_config_mod, "CAM4_ARM_LEARNED_AIM_WEIGHT_ORANGE", 1.0) if _config_mod else 1.0
                from aim_controller_model.train import run_retrain
                run_retrain(npz_path, model_path, th_red, th_orange, min_tr, weight_red=w_red, weight_orange=w_orange)
                print(f"Gun Aim Assist: aim buffer merged and retrained ({len(states_list)} transitions).")
        except Exception as e:
            print(f"Gun Aim Assist: aim retrain on exit failed: {e}")
        # --- cleanup: disconnect arm, stop YOLO thread, release camera ---
        if arm_controller is not None:
            try:
                arm_controller.disconnect()
            except Exception:
                pass
        if _yolo_frame_q is not None:
            try:
                _yolo_frame_q.put_nowait(None)
            except Exception:
                pass
        cam.release()
        cv2.destroyAllWindows()
        print("Gun Aim Assist: stopped.")


if __name__ == "__main__":
    main()





