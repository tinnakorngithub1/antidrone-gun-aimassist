"""
Cam4 Arm Tracker
----------------
- ใช้ในโหมดยิงจริง: รับตำแหน่งเป้าหมาย (pixel) บนภาพ cam4
- แปลงเป็นองศาแขนด้วยไฟล์ hand-eye calibration (ถ้ามี) หรือ FOV mapping (fallback)
- ส่งคำสั่งให้ Cam4ArmController หมุนแขนให้เป้าเข้าใกล้ศูนย์เล็ง

หมายเหตุ:
- เน้นโครงสร้างที่เรียบง่ายก่อน (absolute move ตามมุมคำนวณได้)
- สามารถต่อยอดเพิ่ม PD/alpha-beta filter ให้ลื่นขึ้นได้ภายหลัง
"""

from __future__ import annotations

import atexit
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from cam4_arm_controller import Cam4ArmController


CALIBRATION_DIR = Path(__file__).parent / "calibration_data"
CALIBRATION_FILE = CALIBRATION_DIR / "cam4_hand_eye_pixel_to_degree.json"


@dataclass
class Cam4HandEyeCalibration:
    inverse_mapping_matrix: np.ndarray
    mapping_matrix: np.ndarray
    pixel_center: np.ndarray
    degree_center: np.ndarray


class Cam4ArmTracker:
    """
    ตัวกลางระหว่าง YOLO/เป้าหมายบนภาพ cam4 กับ Cam4ArmController.

    ใช้ได้ 2 โหมด:
    - ถ้ามีไฟล์ hand-eye calibration: ใช้เมทริกซ์ inverse_mapping_matrix แปลง pixel → absolute degrees
    - ถ้าไม่มี: ใช้ FOV จาก config (pixel_to_angle) เป็น fallback (แม่นน้อยกว่าแต่ใช้งานได้)
    """

    def __init__(self, arm_controller: Cam4ArmController, camera_name: str = "cam4") -> None:
        self.arm = arm_controller
        self.camera_name = camera_name
        self.calib: Optional[Cam4HandEyeCalibration] = self._load_calibration()

        # fallback mapping จาก FOV
        self.pixel_to_degree_x, self.pixel_to_degree_y = config.get_pixel_to_degree(camera_name)

        # PD controller พื้นฐานบน space ขององศาแขน
        self.kp = getattr(config, "CAM4_ARM_PD_KP", 0.6)
        self.kd = getattr(config, "CAM4_ARM_PD_KD", 0.08)
        self.deadzone_deg = getattr(config, "CAM4_ARM_DEADZONE_DEG", 0.25)
        self.deadzone_near_deg = getattr(config, "CAM4_ARM_DEADZONE_NEAR_DEG", 0.5)
        self.deadzone_hold_deg = getattr(config, "CAM4_ARM_DEADZONE_HOLD_DEG", 0.2)
        self.deadzone_track_deg = getattr(config, "CAM4_ARM_DEADZONE_TRACK_DEG", 0.5)
        self.max_delta_deg = getattr(config, "CAM4_ARM_MAX_DELTA_DEG", 4.0)
        self.filter_alpha = getattr(config, "CAM4_ARM_FILTER_ALPHA", 0.7)

        # Step ตามระยะ (ใกล้มาก 0.05 mm, ใกล้ 0.1 mm, ห่าง 1 mm) + exponential smooth บน delta
        self.step_very_near_mm = getattr(config, "CAM4_ARM_AIM_STEP_VERY_NEAR_MM", 0.05)
        self.step_near_mm = getattr(config, "CAM4_ARM_AIM_STEP_NEAR_MM", 0.1)
        self.step_far_mm = getattr(config, "CAM4_ARM_AIM_STEP_FAR_MM", 1.0)
        self.error_threshold_very_near_mm = getattr(config, "CAM4_ARM_AIM_ERROR_VERY_NEAR_MM", 0.2)
        self.error_threshold_near_mm = getattr(config, "CAM4_ARM_AIM_ERROR_NEAR_MM", 0.5)
        self.smooth_alpha_delta = getattr(config, "CAM4_ARM_AIM_SMOOTH_ALPHA", 0.75)

        self.filtered_error_pan_deg = 0.0
        self.filtered_error_tilt_deg = 0.0
        self.last_error_pan_deg = 0.0
        self.last_error_tilt_deg = 0.0
        self.last_update_time: Optional[float] = None
        self.smoothed_delta_pan = 0.0
        self.smoothed_delta_tilt = 0.0
        self._in_hold_zone = True  # hysteresis: เริ่มในโซนนิ่ง จนกว่า error > TRACK

        # Learned aim: โหลด model (ถ้าเปิดใช้), buffer สำหรับเก็บ transition, t_prev_send สำหรับ dt
        self._use_learned_aim = getattr(config, "CAM4_ARM_USE_LEARNED_AIM_MODEL", False)
        self._aim_collect_data = getattr(config, "CAM4_ARM_AIM_COLLECT_DATA", False)
        # โหมดที่อนุญาตให้เก็บข้อมูล: "lock" -> LOCK only, "auto" -> AUTO only, "auto,lock" -> ทั้งคู่ (MODE_AUTO=0, MODE_LOCK=3)
        _modes_str = getattr(config, "CAM4_ARM_AIM_COLLECT_DATA_MODES", "auto,lock").strip().lower()
        _mode_map = {"auto": 0, "lock": 3}
        self._aim_collect_modes: set = set()
        for part in _modes_str.replace(" ", "").split(","):
            if part in _mode_map:
                self._aim_collect_modes.add(_mode_map[part])
        if self._aim_collect_data and not self._aim_collect_modes:
            self._aim_collect_modes = {0, 3}
        self._learned_model_path = Path(getattr(config, "CAM4_ARM_LEARNED_AIM_MODEL_PATH", "aim_controller_model/aim_model.pt"))
        self._min_transitions = getattr(config, "CAM4_ARM_LEARNED_AIM_MIN_TRANSITIONS", 500)
        self._threshold_red_deg = getattr(config, "CAM4_ARM_LEARNED_AIM_THRESHOLD_RED_DEG", 0.35)
        self._threshold_orange_deg = getattr(config, "CAM4_ARM_LEARNED_AIM_THRESHOLD_ORANGE_DEG", 0.7)
        self._weight_red = getattr(config, "CAM4_ARM_LEARNED_AIM_WEIGHT_RED", 1.5)
        self._weight_orange = getattr(config, "CAM4_ARM_LEARNED_AIM_WEIGHT_ORANGE", 1.0)
        self._learned_aim_blend_pd = getattr(config, "CAM4_ARM_LEARNED_AIM_BLEND_PD", 0.5)
        self._learned_model: Optional[Any] = None
        self._aim_buffer: List[Dict[str, Any]] = []
        self._t_prev_send: Optional[float] = None

        if self._use_learned_aim and self._learned_model_path.is_absolute() is False:
            self._learned_model_path = Path(__file__).parent / self._learned_model_path
        if self._use_learned_aim and self._learned_model_path.exists():
            try:
                from aim_controller_model.model import (
                    load_model,
                    load_onnx,
                    predict_delta,
                    predict_delta_onnx,
                    normalize_state,
                )
                self._normalize_state_fn = normalize_state
                if self._learned_model_path.suffix.lower() == ".onnx":
                    self._learned_model = load_onnx(self._learned_model_path)
                    if self._learned_model is not None:
                        self._predict_delta_fn = lambda m, sv, ms: predict_delta_onnx(m, sv, ms)
                        sh = self._learned_model.get_inputs()[0].shape
                        self._learned_model_input_dim = int(sh[1]) if len(sh) > 1 else 10
                        print(f"✅ Cam4ArmTracker: loaded learned aim ONNX from {self._learned_model_path}")
                else:
                    self._learned_model = load_model(self._learned_model_path)
                    if self._learned_model is not None:
                        self._predict_delta_fn = predict_delta
                        self._learned_model_input_dim = getattr(self._learned_model, "input_dim", 10)
                        print(f"✅ Cam4ArmTracker: loaded learned aim model from {self._learned_model_path}")
            except Exception as e:
                print(f"⚠️ Cam4ArmTracker: could not load learned model: {e}")
                self._learned_model = None
        if self._learned_model is None and self._use_learned_aim:
            self._use_learned_aim = False

        atexit.register(self._save_buffer_and_retrain)

        if self.calib is not None:
            print(
                f"✅ Cam4ArmTracker: loaded hand-eye calibration "
                f"from {CALIBRATION_FILE.name} "
                f"(pixel_center={self.calib.pixel_center}, degree_center={self.calib.degree_center})"
            )
        else:
            print(
                f"⚠️ Cam4ArmTracker: calibration file '{CALIBRATION_FILE.name}' not found or invalid. "
                f"Falling back to simple FOV mapping."
            )

    def _save_buffer_and_retrain(self) -> None:
        """atexit: เขียน buffer ลง .npz ถ้ามี transition ครบ แล้วเรียกรีเทรน (ถ้ามี PyTorch)."""
        if not self._aim_buffer:
            return
        complete = [t for t in self._aim_buffer if t.get("next_state") is not None]
        if len(complete) < self._min_transitions:
            return
        npz_path = Path(__file__).parent / "aim_controller_model" / "aim_buffer.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        states = np.array([t["state"] for t in complete], dtype=np.float32)
        actions = np.array([t["action"] for t in complete], dtype=np.float32)
        next_states = np.array([t["next_state"] for t in complete], dtype=np.float32)
        np.savez_compressed(npz_path, states=states, actions=actions, next_states=next_states)
        try:
            import sys
            _root = Path(__file__).resolve().parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from aim_controller_model.train import run_retrain
            model_path = self._learned_model_path if self._learned_model_path.is_absolute() else Path(__file__).parent / self._learned_model_path
            # บันทึกเป็น .pt เสมอ (ONNX export ทำแยกด้วย export_onnx.py)
            save_pt = model_path.with_suffix(".pt") if model_path.suffix.lower() == ".onnx" else model_path
            run_retrain(
                npz_path, save_pt,
                self._threshold_red_deg, self._threshold_orange_deg, self._min_transitions,
                weight_red=self._weight_red, weight_orange=self._weight_orange,
            )
        except Exception as e:
            print(f"aim_controller retrain: {e}")

    # ------------------------------------------------------------------
    # Calibration loading
    # ------------------------------------------------------------------
    def _load_calibration(self) -> Optional[Cam4HandEyeCalibration]:
        if not CALIBRATION_FILE.exists():
            return None
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            inv = np.asarray(data.get("inverse_mapping_matrix"), dtype=np.float64)
            mat = np.asarray(data.get("mapping_matrix"), dtype=np.float64)
            ref = data.get("reference_point", {})
            px_center = np.asarray(ref.get("pixel_center", [0.0, 0.0]), dtype=np.float64)
            deg_center = np.asarray(ref.get("degree_center", [0.0, 0.0]), dtype=np.float64)

            if inv.shape != (2, 2) or mat.shape != (2, 2):
                raise ValueError("mapping matrices must be 2x2")

            return Cam4HandEyeCalibration(
                inverse_mapping_matrix=inv,
                mapping_matrix=mat,
                pixel_center=px_center,
                degree_center=deg_center,
            )
        except Exception as e:
            print(f"❌ Cam4ArmTracker: failed to load calibration '{CALIBRATION_FILE}': {e}")
            return None

    # ------------------------------------------------------------------
    # Pixel → Arm degrees
    # ------------------------------------------------------------------
    def pixel_to_arm_degrees(
        self,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
        cx_center: Optional[float] = None,
        cy_center: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        แปลงพิกัด pixel (บนภาพ cam4) เป็นองศาแขน (pan, tilt).

        ถ้าให้ cx_center, cy_center จะใช้เป็นจุดอ้างอิง (ศูนย์เล็งจริง จาก gun_aim_assist)
        ไม่ให้จะใช้ calib.pixel_center หรือ (frame_w/2, frame_h/2).
        """
        if self.calib is not None:
            ref_x = cx_center if cx_center is not None else self.calib.pixel_center[0]
            ref_y = cy_center if cy_center is not None else self.calib.pixel_center[1]
            d_px = px - ref_x
            d_py = py - ref_y
            d_deg = self.calib.inverse_mapping_matrix @ np.array([d_px, d_py], dtype=np.float64)
            pan_deg = float(self.calib.degree_center[0] + d_deg[0])
            tilt_deg = float(self.calib.degree_center[1] + d_deg[1])
            return pan_deg, tilt_deg

        # Fallback: ใช้ FOV mapping จาก config (relative to frame center)
        cam_cfg = config.get_camera_config(self.camera_name)
        center_x = cam_cfg["width"] / 2.0
        center_y = cam_cfg["height"] / 2.0
        if frame_w != cam_cfg["width"] or frame_h != cam_cfg["height"]:
            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
        if cx_center is not None:
            center_x = cx_center
        if cy_center is not None:
            center_y = cy_center

        d_px = px - center_x
        d_py = py - center_y
        pan_deg = d_px * self.pixel_to_degree_x
        tilt_deg = -d_py * self.pixel_to_degree_y
        return pan_deg, tilt_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def arm_angles_to_pixel(self, pan_deg: float, tilt_deg: float, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """
        แปลงองศาแขน (pan, tilt) -> พิกัด pixel บนภาพ cam4
        ใช้สำหรับ overlay วงกลมตำแหน่งเล็ง (โหมดจำลอง).
        """
        # กรณีมี calibration matrix: ใช้ mapping_matrix (degree_offset -> pixel_offset)
        if self.calib is not None:
            d_pan = pan_deg - self.calib.degree_center[0]
            d_tilt = tilt_deg - self.calib.degree_center[1]
            d_px, d_py = (self.calib.mapping_matrix @ np.array([d_pan, d_tilt], dtype=np.float64)).tolist()
            px = float(self.calib.pixel_center[0] + d_px)
            py = float(self.calib.pixel_center[1] + d_py)
        else:
            # Fallback: ใช้ FOV mapping แบบ linear
            cam_cfg = config.get_camera_config(self.camera_name)
            width = frame_w
            height = frame_h
            fov_h = cam_cfg["fov_horizontal"]
            fov_v = cam_cfg["fov_vertical"]
            center_x = width / 2.0
            center_y = height / 2.0
            # angle = (offset / frame_size) * FOV  -> offset = angle/FOV * frame_size
            offset_x = (pan_deg / fov_h) * width
            offset_y = -(tilt_deg / fov_v) * height  # กลับเครื่องหมายให้สอดคล้องกับ pixel_to_arm_degrees
            px = center_x + offset_x
            py = center_y + offset_y

        # clamp ให้อยู่ในเฟรม
        px = max(0.0, min(frame_w - 1.0, px))
        py = max(0.0, min(frame_h - 1.0, py))
        return px, py

    def update_from_detection(
        self,
        cx: float,
        cy: float,
        frame_w: int,
        frame_h: int,
        cx_crosshair: Optional[float] = None,
        cy_crosshair: Optional[float] = None,
        aim_mode: Optional[int] = None,
    ) -> None:
        """
        เรียกจาก loop หลักของ gun_aim_assist เมื่อได้ center ของเป้าหมายบนภาพ cam4.
        cx_crosshair, cy_crosshair = ตำแหน่งศูนย์เล็งจริง (จาก config.get_center) ให้ตรงกับ bbox.
        """
        if self.arm is None or self.arm.is_simulation_mode:
            return

        now = time.perf_counter()
        if self.last_update_time is None:
            dt = 0.0
        else:
            dt = now - self.last_update_time
        self.last_update_time = now

        # เป้าในองศา (absolute) — ใช้ศูนย์เล็งเป็นจุดอ้างอิงถ้าส่งมา
        target_pan_deg, target_tilt_deg = self.pixel_to_arm_degrees(
            cx, cy, frame_w, frame_h, cx_crosshair, cy_crosshair
        )
        # มุมปัจจุบันของแขน (มาจาก state ภายใน controller)
        current_pan_deg = getattr(self.arm, "pos_x", 0.0)
        current_tilt_deg = getattr(self.arm, "pos_y", 0.0)

        raw_error_pan = target_pan_deg - current_pan_deg
        raw_error_tilt = target_tilt_deg - current_tilt_deg

        # Low-pass filter บน error
        a = self.filter_alpha
        self.filtered_error_pan_deg = a * self.filtered_error_pan_deg + (1.0 - a) * raw_error_pan
        self.filtered_error_tilt_deg = a * self.filtered_error_tilt_deg + (1.0 - a) * raw_error_tilt

        err_pan = self.filtered_error_pan_deg
        err_tilt = self.filtered_error_tilt_deg

        # mm_per_deg และ error ไว้ใช้ hysteresis + deadzone ขยายเมื่อใกล้
        mm_per_deg_pan = getattr(self.arm, "mm_per_deg_pan", getattr(config, "CAM4_ARM_MM_PER_DEG_PAN", 1.0))
        mm_per_deg_tilt = getattr(self.arm, "mm_per_deg_tilt", getattr(config, "CAM4_ARM_MM_PER_DEG_TILT", 1.0))
        mm_per_deg = (mm_per_deg_pan + mm_per_deg_tilt) / 2.0
        error_deg = math.hypot(err_pan, err_tilt)
        error_mm = error_deg * mm_per_deg if mm_per_deg > 1e-9 else 0.0

        # เติม next_state ให้ transition ล่าสุดที่ยังไม่มี (เมื่อโหมด AUTO/LOCK เรียกเราต่อเนื่อง)
        if self._aim_buffer and self._aim_buffer[-1].get("next_state") is None:
            self._aim_buffer[-1]["next_state"] = (err_pan, err_tilt, error_deg)

        # (2) Hysteresis: hold = ไม่ขยับ จนกว่า error > TRACK; ถ้า error <= HOLD เข้า hold
        if self._in_hold_zone:
            if error_deg > self.deadzone_track_deg:
                self._in_hold_zone = False
            else:
                self.last_error_pan_deg = err_pan
                self.last_error_tilt_deg = err_tilt
                return
        else:
            if error_deg <= self.deadzone_hold_deg:
                self._in_hold_zone = True
                self.smoothed_delta_pan = 0.0
                self.smoothed_delta_tilt = 0.0
                self.last_error_pan_deg = err_pan
                self.last_error_tilt_deg = err_tilt
                return

        # (1) Deadzone ขยายเมื่อใกล้ (โซนสีส้ม): ใช้ deadzone ใหญ่กว่าเพื่อไม่สั่น
        effective_deadzone = self.deadzone_near_deg if error_mm < self.error_threshold_near_mm else self.deadzone_deg
        if error_deg < effective_deadzone:
            self.last_error_pan_deg = err_pan
            self.last_error_tilt_deg = err_tilt
            return

        # Derivative term (ระวัง dt เล็ก)
        if dt <= 1e-4:
            d_pan = 0.0
            d_tilt = 0.0
        else:
            d_pan = (err_pan - self.last_error_pan_deg) / dt
            d_tilt = (err_tilt - self.last_error_tilt_deg) / dt

        self.last_error_pan_deg = err_pan
        self.last_error_tilt_deg = err_tilt

        # dt สำหรับ state = เวลาตั้งแต่ครั้งล่าสุดที่ส่งคำสั่ง
        dt_send = (now - self._t_prev_send) if self._t_prev_send is not None else 0.0
        old_smoothed_pan = self.smoothed_delta_pan
        old_smoothed_tilt = self.smoothed_delta_tilt

        # ระยะ error มีแล้วจากด้านบน; เลือก step size
        if error_mm < self.error_threshold_very_near_mm:
            max_step_mm = self.step_very_near_mm
        elif error_mm < self.error_threshold_near_mm:
            max_step_mm = self.step_near_mm
        else:
            max_step_mm = self.step_far_mm
        max_step_deg = max_step_mm / mm_per_deg if mm_per_deg > 1e-9 else self.max_delta_deg

        # PD output (ใช้เสมอ — เป็น baseline หรือสำหรับ blend)
        pd_pan = self.kp * err_pan + self.kd * d_pan
        pd_tilt = self.kp * err_tilt + self.kd * d_tilt
        pd_mag = math.hypot(pd_pan, pd_tilt)
        if pd_mag > max_step_deg and pd_mag > 1e-9:
            scale = max_step_deg / pd_mag
            pd_pan = float(pd_pan * scale)
            pd_tilt = float(pd_tilt * scale)

        if self._use_learned_aim and self._learned_model is not None:
            state_vec = self._normalize_state_fn(
                err_pan, err_tilt, error_deg,
                old_smoothed_pan, old_smoothed_tilt,
                dt_send, d_pan, d_tilt,
                current_pan_deg, current_tilt_deg,
            )
            dim = getattr(self, "_learned_model_input_dim", 10)
            state_vec = state_vec[:dim]
            model_pan, model_tilt = self._predict_delta_fn(
                self._learned_model, state_vec, max_step_deg
            )
            blend = max(0.0, min(1.0, getattr(self, "_learned_aim_blend_pd", 0.5)))
            delta_pan = (1.0 - blend) * model_pan + blend * pd_pan
            delta_tilt = (1.0 - blend) * model_tilt + blend * pd_tilt
            magnitude = math.hypot(delta_pan, delta_tilt)
            if magnitude > max_step_deg and magnitude > 1e-9:
                scale = max_step_deg / magnitude
                delta_pan = float(delta_pan * scale)
                delta_tilt = float(delta_tilt * scale)
        else:
            delta_pan, delta_tilt = pd_pan, pd_tilt

        # Exponential smoothing บน delta
        sa = self.smooth_alpha_delta
        self.smoothed_delta_pan = sa * self.smoothed_delta_pan + (1.0 - sa) * delta_pan
        self.smoothed_delta_tilt = sa * self.smoothed_delta_tilt + (1.0 - sa) * delta_tilt

        if self._aim_collect_data and (aim_mode is None or aim_mode in self._aim_collect_modes):
            self._aim_buffer.append({
                "state": [
                    err_pan, err_tilt, error_deg, old_smoothed_pan, old_smoothed_tilt,
                    dt_send, d_pan, d_tilt, current_pan_deg, current_tilt_deg,
                ],
                "action": [self.smoothed_delta_pan, self.smoothed_delta_tilt],
                "next_state": None,
            })

        self.arm.move_relative(self.smoothed_delta_pan, self.smoothed_delta_tilt, blocking=False)
        self._t_prev_send = now



