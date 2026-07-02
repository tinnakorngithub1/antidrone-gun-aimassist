#!/usr/bin/env python3
"""33_lock_bearing_sim.py — Closed-loop simulation ของ LOCK mode หลังแก้ P0 (bearing space)

ต่างจาก 23_arm_chase_sim ตรงที่ harness นี้รัน "โค้ด production จริง" จาก
22_gun_aim_assist_vector.py ทั้งเส้น: _SimpleIoUTracker → _lock_feed_bearing_measurement
→ lock_kalman (bearing space) → _tick_lock → _variable_step_toward_target
โดยจำลองเฉพาะโลกภายนอก: โดรน (bearing เชิงมุม), กล้องติดบนแขน (pixel = offset×ppd
ด้วยค่า calibration จริง รวมเครื่องหมายลบของ ppd_y), YOLO pipeline (latency, det-fps,
noise, miss) และฟิสิกส์แขน (rate+accel limit ตาม twin ที่ validate ใน 27)

รัน:  python3 33_lock_bearing_sim.py            # ทุก scenario, 3 seeds
ผลลัพธ์: ตาราง console + lock_bearing_sim_results.csv + lock_bearing_sim.png
"""
import importlib.util
import math
import os
import sys
import time as _time_mod

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)
sys.path.insert(0, PROJ)

_spec = importlib.util.spec_from_file_location("gaa", os.path.join(PROJ, "22_gun_aim_assist_vector.py"))
gaa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaa)
gaa.LOCK_TRACK_DEBUG = False

# ---------------------------------------------------------------- ค่าคงที่โลกจำลอง
FRAME_W, FRAME_H = 3840, 2160
CX, CY = 1920.0, 1080.0
PPD_X, PPD_Y = 87.13837567036975, -89.73408577160026  # cam4_pixel_per_degree.json (ค่าจริง รวมเครื่องหมาย)
BBOX_PX = 120                      # ขนาด bbox โดรนบนภาพ
X_LIM = (-63.0, 63.0)              # limits หลังหัก margin 2°
Y_LIM = (-33.0, 33.0)
LOOP_HZ = 30                       # main loop (display)
ARM_RATE = (80.0, 60.0)            # deg/s (CAM4_ARM_JOYSTICK_MAX_*_RATE)
ARM_ACCEL = 100.0                  # deg/s² (twin fitted, validate ใน 27 กับจุดวัดจริง)
SUB_DT = 0.001                     # ฟิสิกส์ sub-step

# ---------------------------------------------------------------- sim clock
_real_time = _time_mod.time
_sim_t = [0.0]


def _fake_time():
    return _sim_t[0]


class SimArm:
    """แขนจำลอง: pos_x/pos_y = ตำแหน่ง 'สั่ง' (อัปเดตทันทีเหมือน Cam4ArmController)
    ส่วน phys_* = ตำแหน่งจริงที่ไล่ตามด้วย rate+accel limit (กล้องเห็น phys)."""

    def __init__(self):
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.phys_pan = 0.0
        self.phys_tilt = 0.0
        self.vel_pan = 0.0
        self.vel_tilt = 0.0

    def move_relative(self, dx, dy, blocking=False):
        self.pos_x = float(np.clip(self.pos_x + dx, *X_LIM))
        self.pos_y = float(np.clip(self.pos_y + dy, *Y_LIM))

    def move_absolute(self, x, y, blocking=False):
        self.pos_x = float(np.clip(x, *X_LIM))
        self.pos_y = float(np.clip(y, *Y_LIM))
        if blocking:  # main loop แช่จนแขนถึง (เหมือน production ตอน acquire)
            t_end = _sim_t[0] + 3.0
            while _sim_t[0] < t_end:
                self.step_physics(SUB_DT)
                _sim_t[0] += SUB_DT
                if (abs(self.phys_pan - self.pos_x) < 0.05
                        and abs(self.phys_tilt - self.pos_y) < 0.05):
                    break
        return True

    @staticmethod
    def _axis_step(pos, vel, target, rate_max, accel, dt):
        err = target - pos
        if abs(err) > 1e-9:
            v_des = math.copysign(min(rate_max, math.sqrt(2.0 * accel * abs(err))), err)
        else:
            v_des = 0.0
        dv = max(-accel * dt, min(accel * dt, v_des - vel))
        vel += dv
        pos += vel * dt
        return pos, vel

    def step_physics(self, dt):
        self.phys_pan, self.vel_pan = self._axis_step(
            self.phys_pan, self.vel_pan, self.pos_x, ARM_RATE[0], ARM_ACCEL, dt)
        self.phys_tilt, self.vel_tilt = self._axis_step(
            self.phys_tilt, self.vel_tilt, self.pos_y, ARM_RATE[1], ARM_ACCEL, dt)


class Drone:
    """เป้าเชิงมุม: sine pan (peak rate = omega) + tilt เบา ๆ; รองรับ dropout เป็นช่วง"""

    def __init__(self, omega, pan0=8.0, tilt0=2.0, amp_pan=15.0, amp_tilt=5.0,
                 dropout_every=0.0, dropout_len=0.0):
        self.omega = float(omega)
        self.pan0, self.tilt0 = pan0, tilt0
        self.amp_pan, self.amp_tilt = amp_pan, amp_tilt
        self.dropout_every, self.dropout_len = dropout_every, dropout_len

    def bearing(self, t):
        if self.omega <= 1e-9:
            return self.pan0, self.tilt0
        w = self.omega / self.amp_pan  # rad/s ให้ peak rate = omega deg/s
        pan = self.pan0 + self.amp_pan * math.sin(w * t)
        tilt = self.tilt0 + self.amp_tilt * math.sin(w * t * 0.7 + 1.0)
        return pan, tilt

    def visible(self, t):
        if self.dropout_every <= 0:
            return True
        return (t % self.dropout_every) > self.dropout_len


def run_scenario(name, drone, duration=30.0, latency=0.150, det_fps=15.0,
                 noise_px=8.0, miss_rate=0.20, seed=1):
    """รัน 1 scenario คืน dict metrics + error timeline"""
    rng = np.random.default_rng(seed)
    _sim_t[0] = 1000.0  # ให้ timestamp > 0 เสมอ
    arm = SimArm()
    tracker = gaa._SimpleIoUTracker()
    kal = gaa._TargetKalman(q=gaa.LOCK_BEARING_KALMAN_Q, r=gaa.LOCK_BEARING_KALMAN_R)
    pose_hist = gaa._ArmPoseHistory()
    ds = gaa._ArmDriveState()
    ds.continuous_target_time_prev = _sim_t[0] - 0.05
    ctx = {
        "arm_controller": arm, "lock_arm_drive_state": ds, "lock_kalman": kal,
        "x_lo": X_LIM[0], "x_hi": X_LIM[1], "y_lo": Y_LIM[0], "y_hi": Y_LIM[1],
        "lock_csrt_lost": False, "lock_csrt_initialized": True,
        "lock_last_arm_move_time": _sim_t[0] - 0.05,
    }

    def target_pixel(t):
        """pixel ของโดรนบนภาพ cam4 (กล้องเล็งตาม phys pose)"""
        d_