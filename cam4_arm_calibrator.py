"""
Cam4 Arm Calibrator
-------------------
- ใช้ใน lab เพื่อทำ hand-eye calibration ระหว่างกล้อง cam4 และแขนกล
- ผู้ใช้ขยับ ArUco board ไปทั่ว field ของ cam4
- แขน (ผ่าน Cam4ArmController) จะพยายามตามให้ ArUco เข้าใกล้ศูนย์เล็ง
- เมื่อภาพนิ่งพอ จะเก็บ sample (pixel_x, pixel_y, arm_x_deg, arm_y_deg)
- เมื่อได้ sample มากพอ จะคำนวณ mapping_matrix / inverse_mapping_matrix และบันทึกเป็น JSON

การใช้งาน (ใน lab):
    python cam4_arm_calibrator.py

ข้อกำหนด:
- กล้อง cam4 ต้องถูกกำหนดใน config.py (ACTIVE_CAMERA = "cam4" หรือระบุ camera_name เอง)
- Zoom ต้องคงที่ (ค่าที่จะใช้ยิงจริง)
- แขนกลต้องมี homing ผ่าน limit switch และ wired ถูกต้อง (Cam4ArmController จะเรียก $H)
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config
from cam4_arm_controller import Cam4ArmController
from fast_motion_sky import CameraStream
from gun_aim_assist import ShooterConfig


CALIBRATION_DIR = Path(__file__).parent / "calibration_data"
CALIBRATION_FILE = CALIBRATION_DIR / "cam4_hand_eye_pixel_to_degree.json"

MIN_SAMPLES = 40
MAX_SAMPLES = 200

ARUCO_DICT_NAME = getattr(config, "CAM4_CALIB_ARUCO_DICT", "DICT_6X6_250")
ARUCO_MARKER_LENGTH_M = getattr(config, "CAM4_CALIB_ARUCO_MARKER_LENGTH_M", 0.04)

SAMPLE_MAX_PIXEL_ERROR = getattr(config, "CAM4_CALIB_MAX_PIXEL_ERROR", 40.0)
SAMPLE_MAX_ANGULAR_VEL = getattr(config, "CAM4_CALIB_MAX_ANGULAR_VEL", 10.0)  # deg/s


@dataclass
class CalibSample:
    pixel_x: float
    pixel_y: float
    arm_x_deg: float
    arm_y_deg: float


def get_aruco_detector():
    """เตรียม dictionary + parameters สำหรับตรวจ ArUco."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV is built without aruco module. Install opencv-contrib-python.")

    aruco = cv2.aruco
    if not hasattr(aruco, ARUCO_DICT_NAME):
        raise RuntimeError(f"cv2.aruco has no dictionary '{ARUCO_DICT_NAME}'")

    dictionary = getattr(aruco, ARUCO_DICT_NAME)
    aruco_dict = aruco.Dictionary_get(dictionary)
    parameters = aruco.DetectorParameters_create()
    return aruco, aruco_dict, parameters


def detect_aruco_center(frame) -> Tuple[bool, float, float]:
    """คืน (found, cx, cy) ของจุด center ของ ArUco marker/board ในภาพ."""
    aruco, aruco_dict, parameters = get_aruco_detector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is None or len(corners) == 0:
        return False, 0.0, 0.0

    # ใช้ marker แรก (หรือจะเฉลี่ยทั้งหมดก็ได้)
    pts = corners[0][0]  # shape (4, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    return True, cx, cy


def compute_mapping(samples: List[CalibSample]):
    """
    คำนวณ mapping_matrix / inverse_mapping_matrix จาก sample list.

    เราต้องการเมทริกซ์ A 2x2 ที่:
        [d_pan; d_tilt] = A * [d_px; d_py]
    โดย d_* คือ offset จาก center ทั้งใน pixel และ degree
    """
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples for calibration.")

    # เลือก reference center:
    # ใช้ค่าเฉลี่ยขององศาเป็น center (หรือตำแหน่งที่ใกล้ 0,0 ก็ได้)
    mean_arm_x = float(np.mean([s.arm_x_deg for s in samples]))
    mean_arm_y = float(np.mean([s.arm_y_deg for s in samples]))

    # เลือก sample ที่ใกล้ center นี้ที่สุดเป็น reference
    ref_sample = min(
        samples,
        key=lambda s: (s.arm_x_deg - mean_arm_x) ** 2 + (s.arm_y_deg - mean_arm_y) ** 2,
    )

    pixel_center = np.array([ref_sample.pixel_x, ref_sample.pixel_y], dtype=np.float64)
    degree_center = np.array([ref_sample.arm_x_deg, ref_sample.arm_y_deg], dtype=np.float64)

    # เตรียม X (pixel offset) และ Y (degree offset)
    X = []
    Y = []
    for s in samples:
        d_px = s.pixel_x - pixel_center[0]
        d_py = s.pixel_y - pixel_center[1]
        d_pan = s.arm_x_deg - degree_center[0]
        d_tilt = s.arm_y_deg - degree_center[1]
        X.append([d_px, d_py])
        Y.append([d_pan, d_tilt])

    X = np.asarray(X, dtype=np.float64)  # (N, 2)
    Y = np.asarray(Y, dtype=np.float64)  # (N, 2)

    # แก้ least squares: X @ A^T ≈ Y  → A^T = (X^+ @ Y)
    # ใช้ lstsq แยกสำหรับแต่ละแกน
    A_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape (2, 2)
    mapping_matrix = A_T.T  # (2, 2)

    # inverse mapping
    try:
        inverse_mapping_matrix = np.linalg.inv(mapping_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Mapping matrix is singular; try collecting more diverse samples.")

    # ประเมิน error (forward & backward)
    pixel_errors = []
    for s in samples:
        d_px = s.pixel_x - pixel_center[0]
        d_py = s.pixel_y - pixel_center[1]
        d_pan, d_tilt = (mapping_matrix @ np.array([d_px, d_py])).tolist()

        # คำนวณพิกัดองศาที่ได้
        pan_est = degree_center[0] + d_pan
        tilt_est = degree_center[1] + d_tilt

        # แปลงกลับเป็น pixel (ใช้ inverse)
        d_pan_back = pan_est - degree_center[0]
        d_tilt_back = tilt_est - degree_center[1]
        d_px_est, d_py_est = (inverse_mapping_matrix @ np.array([d_pan_back, d_tilt_back])).tolist()
        px_est = pixel_center[0] + d_px_est
        py_est = pixel_center[1] + d_py_est

        err_px = math.hypot(px_est - s.pixel_x, py_est - s.pixel_y)
        pixel_errors.append(err_px)

    pixel_errors = np.asarray(pixel_errors, dtype=np.float64)
    rms_error = float(np.sqrt(np.mean(pixel_errors ** 2)))
    mean_error = float(np.mean(pixel_errors))
    max_error = float(np.max(pixel_errors))

    quality_metrics = {
        "rms_error_pixels": rms_error,
        "mean_error_pixels": mean_error,
        "max_error_pixels": max_error,
        "num_samples": len(samples),
    }

    reference_point = {
        "pixel_center": pixel_center.tolist(),
        "degree_center": degree_center.tolist(),
    }

    return mapping_matrix, inverse_mapping_matrix, reference_point, quality_metrics


def save_calibration(
    mapping_matrix: np.ndarray,
    inverse_mapping_matrix: np.ndarray,
    reference_point,
    quality_metrics,
    samples: List[CalibSample],
) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "camera_id": 4,
        "calibration_type": "hand_eye_pixel_to_degree",
        "output_resolution": {
            "width": config.get_camera_config("cam4")["width"],
            "height": config.get_camera_config("cam4")["height"],
        },
        "mapping_matrix": mapping_matrix.tolist(),
        "inverse_mapping_matrix": inverse_mapping_matrix.tolist(),
        "reference_point": reference_point,
        "quality_metrics": quality_metrics,
        "samples": [asdict(s) for s in samples],
    }

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved cam4 calibration to: {CALIBRATION_FILE}")
    print(
        f"   RMS error: {quality_metrics['rms_error_pixels']:.2f} px "
        f"(mean={quality_metrics['mean_error_pixels']:.2f}, max={quality_metrics['max_error_pixels']:.2f})"
    )


def main():
    print("=== Cam4 Arm Calibrator (hand-eye, zoom fixed) ===")
    print(f"Calibration file target: {CALIBRATION_FILE}")

    shooter_cfg = ShooterConfig()
    shooter_cfg.load()

    # เปิดแขนกลด้วย context manager (จะเรียก connect()/disconnect() ให้อัตโนมัติ)
    with Cam4ArmController() as arm:
        cam_name = "cam4"
        cam_cfg = config.get_camera_config(cam_name)

        print(f"Opening CameraStream for {cam_name} ({cam_cfg['width']}x{cam_cfg['height']})...")
        cam_stream = CameraStream(cam_name)
        cam_stream.start()

        try:
            samples: List[CalibSample] = []

            last_arm_x = arm.pos_x
            last_arm_y = arm.pos_y
            last_arm_time = time.time()

            print("เริ่ม calibration: ขยับ ArUco board ให้ครอบคลุม field ของกล้อง cam4.")
            print("ระบบจะเก็บ sample อัตโนมัติเมื่อภาพนิ่งและ error ใกล้ศูนย์เล็ง.")

            while True:
                frame = cam_stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                aim_cx, aim_cy = shooter_cfg.get_center(w, h)

                found, cx, cy = detect_aruco_center(frame)
                if not found:
                    cv2.circle(frame, (aim_cx, aim_cy), 8, (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Show ArUco board...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("cam4_calib", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # error จากศูนย์เล็ง (px)
                err_x = cx - aim_cx
                err_y = cy - aim_cy
                err_norm = math.hypot(err_x, err_y)

                # ใช้ mapping FOV คร่าว ๆ ให้แขนพยายามตาม
                fov_h = cam_cfg["fov_horizontal"]
                fov_v = cam_cfg["fov_vertical"]
                d_pan = (err_x / w) * fov_h
                d_tilt = -(err_y / h) * fov_v

                # PD อย่างง่ายมาก ๆ (P-only) สำหรับเฟสคาลิเบรต
                kp = getattr(config, "CAM4_CALIB_PD_KP", 0.4)
                arm_delta_x = kp * d_pan
                arm_delta_y = kp * d_tilt
                arm.move_relative(arm_delta_x, arm_delta_y, blocking=False)

                # คำนวณความเร็วเชิงมุมของแขน
                now = time.time()
                dt = now - last_arm_time if now > last_arm_time else 1e-3
                vel_x = (arm.pos_x - last_arm_x) / dt
                vel_y = (arm.pos_y - last_arm_y) / dt
                vel_norm = math.hypot(vel_x, vel_y)
                last_arm_x, last_arm_y, last_arm_time = arm.pos_x, arm.pos_y, now

                # เงื่อนไขเก็บ sample: แขนนิ่งพอ + pixel error ไม่เกิน threshold
                if (
                    err_norm <= SAMPLE_MAX_PIXEL_ERROR
                    and vel_norm <= SAMPLE_MAX_ANGULAR_VEL
                    and len(samples) < MAX_SAMPLES
                ):
                    sample = CalibSample(
                        pixel_x=cx,
                        pixel_y=cy,
                        arm_x_deg=arm.pos_x,
                        arm_y_deg=arm.pos_y,
                    )
                    samples.append(sample)
                    print(
                        f"  [+] sample #{len(samples)}: "
                        f"px=({cx:.1f},{cy:.1f}), arm=({arm.pos_x:.2f},{arm.pos_y:.2f})"
                    )

                # แสดง HUD เล็กน้อย
                cv2.circle(frame, (aim_cx, aim_cy), 8, (0, 0, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"samples: {len(samples)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"err_px: {err_norm:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("cam4_calib", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC = ยกเลิก
                    break
                if key == ord("s") and len(samples) >= MIN_SAMPLES:
                    # ผู้ใช้กด 's' เพื่อบังคับคำนวณแม็ปก่อนถึง MAX_SAMPLES
                    break

                if len(samples) >= MAX_SAMPLES:
                    break

            cv2.destroyWindow("cam4_calib")

            if len(samples) < MIN_SAMPLES:
                print(f"❌ Not enough samples collected ({len(samples)}/{MIN_SAMPLES}). Calibration aborted.")
                return

            print(f"📊 Computing mapping from {len(samples)} samples...")
            mapping_matrix, inverse_mapping_matrix, ref_point, quality = compute_mapping(samples)
            save_calibration(mapping_matrix, inverse_mapping_matrix, ref_point, quality, samples)

        finally:
            cam_stream.stop()


if __name__ == "__main__":
    main()

