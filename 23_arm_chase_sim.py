"""
23_arm_chase_sim.py
ระบบจำลอง (simulation) การหมุนแขนกล้องไล่ตามโดรนที่บิน — สำหรับงานวิจัย

ตอบคำถาม: "แขนจะหมุนไล่ให้ทันเป้าหมายโดรนที่บินได้หรือไม่ และต้องไล่อย่างไร"

แนวคิด
------
- กล้องติดบนแขน (camera-on-arm): จุดกลางเฟรม = ทิศที่แขนเล็งอยู่
- ดึง resolution + FOV ของกล้องที่เลือกใน config.py มาคำนวณ pixel_per_degree จริง
    px_per_deg_x = width  / fov_horizontal
    px_per_deg_y = height / fov_vertical
- จำลองโดรนเป็น "ตำแหน่งเชิงมุม" (pan/tilt องศา) ที่เคลื่อนด้วยความเร็วเชิงมุม ω (deg/s)
    ω ได้จากความเร็วเชิงเส้น v (m/s) และระยะ R (m):  ω = (v / R) · (180/π)
- จำลอง "ท่อ detection" ที่มีดีเลย์ (latency) + อัปเดตเป็นจังหวะ (det_fps) เหมือน YOLO จริง
- จำลอง controller แบบเดียวกับระบบจริง: P-control + feedforward lead, ส่งคำสั่งแบบ throttle,
  แขนหมุนได้ไม่เกินความเร็วสูงสุด (rate limit)
- รายงานผล: error การเล็ง (องศา), อยู่ในวง reticle (~1°) ไหม, และ "ไล่ทัน/ไม่ทัน"

วิธีรัน
------
  python3 23_arm_chase_sim.py                          # ใช้กล้องจาก config + ค่า default มีหน้าต่างแสดงผล
  python3 23_arm_chase_sim.py --speed-ms 40 --range-m 80
  python3 23_arm_chase_sim.py --pattern sine --omega-deg 60
  python3 23_arm_chase_sim.py --no-display             # รันเร็ว ไม่เปิดจอ พิมพ์สรุปอย่างเดียว
  python3 23_arm_chase_sim.py --sweep                  # หาความเร็วโดรนสูงสุดที่ยังไล่ทัน (feasibility)

ปุ่ม (โหมดแสดงผล): Q/ESC ออก, SPACE หยุด/เล่น
"""
import argparse
import csv
import math
import sys
from collections import deque

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — เซฟ PNG โดยไม่ต้องมีจอ
    import matplotlib.pyplot as _plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    import config as _config_mod
except ImportError:
    _config_mod = None


# =============================================================================
# โหลดพารามิเตอร์กล้อง + แขน จาก config
# =============================================================================
def load_camera_params(camera_name=None):
    """คืน dict: name, width, height, fov_h, fov_v, ppd_x, ppd_y"""
    w, h, fh, fv, name = 3840, 2160, 60.0, 36.0, "default"
    if _config_mod is not None:
        try:
            name = camera_name or getattr(_config_mod, "ACTIVE_CAMERA", "cam4")
            c = _config_mod.get_camera_config(name)
            w = int(c.get("width", w))
            h = int(c.get("height", h))
            fh = float(c.get("fov_horizontal", fh))
            fv = float(c.get("fov_vertical", fv))
        except Exception as e:
            print(f"[sim] อ่าน config กล้องไม่ได้ ({e}) — ใช้ค่า default")
    return {
        "name": name, "width": w, "height": h,
        "fov_h": fh, "fov_v": fv,
        "ppd_x": w / fh, "ppd_y": h / fv,
    }


def load_arm_params():
    """คืน dict ความสามารถของแขนจาก config (มี fallback)"""
    g = lambda k, d: float(getattr(_config_mod, k, d)) if _config_mod is not None else d
    xr = getattr(_config_mod, "CAM4_ARM_X_LIMITS", (-65.0, 65.0)) if _config_mod else (-65.0, 65.0)
    yr = getattr(_config_mod, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0)) if _config_mod else (-35.0, 35.0)
    return {
        "max_pan_rate": g("CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0),   # deg/s
        "max_tilt_rate": g("CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0),  # deg/s
        # ความเร่งจำกัด (deg/s^2) — เลียนแบบ GRBL planner $120/$121 (mm/s^2, mm_per_deg≈1)
        # ทำให้แขนเร่ง/เบรกไม่ทันทีตอนกลับทิศ → เพิ่ม lag จริงที่ rate-limit ล้วนจับไม่ได้
        "max_pan_accel": g("CAM4_ARM_MAX_PAN_ACCEL_DEG", 100.0),   # deg/s^2
        "max_tilt_accel": g("CAM4_ARM_MAX_TILT_ACCEL_DEG", 100.0),  # deg/s^2
        "x_lim": (float(xr[0]), float(xr[1])),
        "y_lim": (float(yr[0]), float(yr[1])),
        "cmd_interval": g("CAM4_ARM_MIN_MOVE_INTERVAL_MS", 60.0) / 1000.0,  # วินาที (throttle ส่งคำสั่ง)
    }


def accel_limited_step(cur, vel, target, max_rate, accel, dt, lim):
    """อัปเดต (pos, vel) เข้าหา target ด้วยโปรไฟล์ trapezoidal:
      - ความเร็ว ≤ max_rate
      - อัตราเปลี่ยนความเร็ว |Δvel| ≤ accel·dt
      - ชะลอให้ "หยุดพอดี" ที่ target (v_cap = sqrt(2·accel·|err|)) กันเลยเป้า
    accel ≤ 0 → ตกไปเป็น rate-limit ล้วน (เร่งทันที) เพื่อ backward-compat.
    คืน (pos ใหม่, vel ใหม่)."""
    err = target - cur
    if accel and accel > 0:
        v_cap = math.sqrt(2.0 * accel * abs(err))            # เร็วสุดที่ยังเบรกทันหยุดที่เป้า
        v_des = math.copysign(min(max_rate, v_cap), err)
        dv = max(-accel * dt, min(accel * dt, v_des - vel))  # จำกัดความเร่ง
        vel += dv
        cur += vel * dt
        if (target - cur) * err < 0:                         # ข้ามเป้า (เชิงตัวเลข) → หยุด
            cur, vel = target, 0.0
    else:
        max_step = max_rate * dt
        if abs(err) <= max_step:
            cur, vel = target, 0.0
        else:
            cur += math.copysign(max_step, err)
    cur = float(np.clip(cur, lim[0], lim[1]))
    return cur, vel


# =============================================================================
# โมเดลโดรน (ตำแหน่งเชิงมุม pan/tilt เทียบฐานแขน)
# =============================================================================
class DroneModel:
    def __init__(self, pattern, omega_deg, fov_h, fov_v, tilt_amp_deg=5.0):
        self.pattern = pattern
        self.omega = omega_deg          # deg/s (อัตราเชิงมุมหลัก แกน pan)
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.tilt_amp = tilt_amp_deg
        # เริ่มนอกขอบซ้ายเล็กน้อยสำหรับ cross; sine เริ่มที่ 0
        self._span = fov_h * 0.6        # กวาด ±span รอบศูนย์
        if pattern == "sine":
            # ตั้งความถี่ให้ peak rate = omega:  d=A sin(2πf t) → peak rate = A·2πf
            self.freq = omega_deg / (2.0 * math.pi * max(self._span, 1e-6))
        # tilt แกว่งช้า ๆ ให้มี movement สองแกน
        self.tilt_freq = 0.15

    def angle_at(self, t):
        """คืน (pan_deg, tilt_deg) ของโดรนจริง ณ เวลา t"""
        if self.pattern == "sine":
            pan = self._span * math.sin(2.0 * math.pi * self.freq * t)
        else:  # "cross": วิ่งทางเดียวด้วย omega คงที่ แล้ววนกลับเมื่อเลยขอบ
            period = (2.0 * self._span) / max(self.omega, 1e-6)  # เวลาเดินทางข้ามฟิลด์
            phase = (t % (2.0 * period)) / period                # 0..2
            pan = -self._span + self.omega * (phase * period) if phase <= 1.0 \
                else self._span - self.omega * ((phase - 1.0) * period)
        tilt = self.tilt_amp * math.sin(2.0 * math.pi * self.tilt_freq * t)
        return pan, tilt


# =============================================================================
# โมเดลแขน (rate-limited ต่อแกน + ขอบเขตมุม)
# =============================================================================
class ArmModel:
    def __init__(self, arm, swap=False):
        self.max_pan = arm["max_pan_rate"]
        self.max_tilt = arm["max_tilt_rate"]
        # accel จำกัด (deg/s^2); ค่า None/<=0 → rate-limit ล้วน
        self.accel_pan = float(arm.get("max_pan_accel", 0.0) or 0.0)
        self.accel_tilt = float(arm.get("max_tilt_accel", 0.0) or 0.0)
        self.x_lim = arm["x_lim"]
        self.y_lim = arm["y_lim"]
        self.pan = 0.0
        self.tilt = 0.0
        self.vel_pan = 0.0   # ความเร็วเชิงมุมปัจจุบัน (deg/s) — สำหรับ accel limit
        self.vel_tilt = 0.0
        self.target_pan = 0.0
        self.target_tilt = 0.0

    def set_target(self, tpan, ttilt):
        self.target_pan = float(np.clip(tpan, self.x_lim[0], self.x_lim[1]))
        self.target_tilt = float(np.clip(ttilt, self.y_lim[0], self.y_lim[1]))

    SUB_DT = 0.001   # sub-step ภายใน 1ms → ผล accel-limit ไม่ขึ้นกับ sim_dt ของผู้เรียก (เช่น 5ms)

    def step(self, dt):
        """หมุนเข้าหา target ด้วยความเร็ว ≤ max rate และความเร่ง ≤ accel (trapezoidal).
        แตก dt เป็น sub-step ≤1ms เพราะ accel-limited lag ไวต่อ dt มาก
        (ที่ dt=5ms under-predict ~8%, dt=50ms under-predict ~2.3×) — sub-step ทำให้แม่นเสมอ."""
        n = max(1, int(math.ceil(dt / self.SUB_DT)))
        h = dt / n
        for _ in range(n):
            self.pan, self.vel_pan = accel_limited_step(
                self.pan, self.vel_pan, self.target_pan, self.max_pan, self.accel_pan, h, self.x_lim)
            self.tilt, self.vel_tilt = accel_limited_step(
                self.tilt, self.vel_tilt, self.target_tilt, self.max_tilt, self.accel_tilt, h, self.y_lim)


# =============================================================================
# Controller — เลียนแบบระบบจริง: P + feedforward lead, มี latency + throttle
# =============================================================================
class ChaseController:
    """
    mode:
      "p"           = P เปล่า (ไม่มี feedforward lead) — นิ่งต่อ noise แต่ lag สูง
      "plead"       = P + feedforward lead จาก velocity แบบ finite-diff (raw) — lag ต่ำแต่ขยาย noise
      "kalman"      = P + lead จาก alpha-beta (steady-state Kalman) ที่กรอง pos+vel — สมดุล lag/noise
      "kalman_coast"= เหมือน kalman แต่ตอน YOLO miss จะ predict ต่อด้วย velocity (coasting)
      "kalman_coast_bounded"= coast แต่ decay velocity แบบ exp(-dt/coast_tau) ทุก step ตอน miss
                     → ที่ det-fps ต่ำ (miss-gap ยาว) velocity ยุบเร็ว = ไม่ overshoot โดรนที่เลี้ยว
                     (แก้จุดอ่อนของ kalman_coast ที่ backfire ตอน fps ต่ำ+ω สูง)
    """
    def __init__(self, ppd_x, ppd_y, kp, lead_sec, cmd_interval, latency_sec, det_dt,
                 noise_px=0.0, miss_rate=0.0, seed=0, mode="plead", alpha=0.5, beta=0.08,
                 coast_tau=0.30):
        self.ppd_x = ppd_x
        self.ppd_y = ppd_y
        self.kp = kp
        self.lead = lead_sec
        self.cmd_interval = cmd_interval
        self.latency = latency_sec
        self.det_dt = det_dt
        self.mode = mode
        self.alpha = alpha            # alpha-beta: น้ำหนักแก้ตำแหน่ง
        self.beta = beta              # alpha-beta: น้ำหนักแก้ความเร็ว
        self.coast_tau = coast_tau    # "kalman_coast_bounded": time-const decay ของ velocity ตอน coast (s)
        self._ab = None               # state ของ alpha-beta filter ต่อแกน
        # YOLO measurement noise: bbox center jitter (Gaussian, px) แปลงเป็นองศาต่อแกน
        self.noise_deg_x = noise_px / ppd_x if ppd_x else 0.0
        self.noise_deg_y = noise_px / ppd_y if ppd_y else 0.0
        self.miss_rate = miss_rate           # โอกาส detection หลุด/ไม่อัปเดต (zero-order hold)
        self.rng = np.random.RandomState(seed)
        self._last_cmd_t = -1e9
        self._last_det_t = -1e9
        self._meas = None          # (abs_pan_deg, abs_tilt_deg) — bearing สัมบูรณ์ที่ใช้เล็ง (อาจกรองแล้ว)
        self._prev_meas_drone = None  # (t, dpan, dtilt) สำหรับประเมินความเร็วแบบ finite-diff
        self._vel = (0.0, 0.0)     # ความเร็วเชิงมุมโดรนที่ประเมินได้ (deg/s)

    def _ab_update(self, ts, mp, mt):
        """alpha-beta filter ต่อแกน — คืน ((pos_pan, pos_tilt), (vel_pan, vel_tilt)) ที่กรองแล้ว"""
        if self._ab is None:
            self._ab = {"t": ts, "p": [mp, mt], "v": [0.0, 0.0]}
            return (mp, mt), (0.0, 0.0)
        dt = ts - self._ab["t"]
        self._ab["t"] = ts
        if dt <= 1e-6:
            dt = self.det_dt
        op, ov = [0.0, 0.0], [0.0, 0.0]
        for i, m in enumerate((mp, mt)):
            p = self._ab["p"][i] + self._ab["v"][i] * dt   # predict ตำแหน่ง
            v = self._ab["v"][i]
            r = m - p                                       # residual จาก measurement
            p += self.alpha * r
            v += (self.beta / dt) * r
            self._ab["p"][i], self._ab["v"][i] = p, v
            op[i], ov[i] = p, v
        return (op[0], op[1]), (ov[0], ov[1])

    def _ab_predict(self, ts, decay_tau=None):
        """coasting: เลื่อนตำแหน่งไปข้างหน้าด้วย velocity (ไม่มี measurement correction).
        decay_tau ไม่ None → คูณ velocity ด้วย exp(-dt/tau) ก่อนเลื่อน (bounded coast):
        dt ใหญ่ (fps ต่ำ) → decay แรง → velocity ยุบเร็ว ไม่ extrapolate เชิงเส้นข้ามช่วงยาว."""
        if self._ab is None:
            return self._meas, self._vel
        dt = ts - self._ab["t"]
        self._ab["t"] = ts
        if dt <= 1e-6:
            dt = self.det_dt
        decay = math.exp(-dt / decay_tau) if (decay_tau and decay_tau > 1e-6) else 1.0
        for i in range(2):
            self._ab["v"][i] *= decay                   # bounded: ยุบ velocity ตามเวลาที่ coast
            self._ab["p"][i] += self._ab["v"][i] * dt   # predict ตำแหน่ง
        return (self._ab["p"][0], self._ab["p"][1]), (self._ab["v"][0], self._ab["v"][1])

    def update(self, t, arm, hist):
        """
        hist: deque ของ (t, drone_pan, drone_tilt, arm_pan, arm_tilt)
        เลียนแบบ: detection สะท้อนเฟรมที่จับเมื่อ t-latency, อัปเดตทุก det_dt

        สำคัญ: แปลง pixel error เป็น "bearing สัมบูรณ์" ด้วยตำแหน่งแขน ณ เวลาที่จับเฟรม
        (apan_delayed) — ไม่ใช่ตำแหน่งแขนปัจจุบัน — เพื่อกัน double-count ของการขยับ
        แขนระหว่างดีเลย์ (นี่คือสิ่งที่ cam8-cue AUTO ทำผ่าน calibration)
            bearing_abs = apan_delayed + (pixel_err / ppd) = dpan_delayed
        """
        # 1) detection อัปเดตเป็นจังหวะ (zero-order hold ระหว่างรอบ)
        if t - self._last_det_t >= self.det_dt:
            self._last_det_t = t
            # YOLO อาจ miss (เฟรมนี้ไม่เจอ/conf ต่ำ) → ไม่อัปเดต ถือค่าเก่า
            missed = self.miss_rate > 0.0 and self.rng.random_sample() < self.miss_rate
            samp = _interp_hist(hist, t - self.latency)
            use_kalman = self.mode in ("kalman", "kalman_coast", "kalman_coast_bounded")
            if samp is not None and not missed:
                ts, dpan, dtilt, apan, atilt = samp
                # bearing สัมบูรณ์ของโดรน ณ เวลาที่จับเฟรม + bbox center jitter ของ YOLO (Gaussian)
                meas_pan = dpan + self.rng.normal(0.0, self.noise_deg_x)
                meas_tilt = dtilt + self.rng.normal(0.0, self.noise_deg_y)
                if use_kalman:
                    # กรอง pos+vel ด้วย alpha-beta → noise ไม่ถูกขยายตอนทำ feedforward
                    self._meas, self._vel = self._ab_update(ts, meas_pan, meas_tilt)
                else:
                    self._meas = (meas_pan, meas_tilt)
                    # velocity แบบ finite-diff (noise ถูกขยาย — ใช้ใน "plead")
                    if self._prev_meas_drone is not None:
                        pt, ppan, ptilt = self._prev_meas_drone
                        ddt = ts - pt
                        if ddt > 1e-6:
                            self._vel = ((meas_pan - ppan) / ddt, (meas_tilt - ptilt) / ddt)
                    self._prev_meas_drone = (ts, meas_pan, meas_tilt)
            elif missed and self.mode in ("kalman_coast", "kalman_coast_bounded") and self._ab is not None:
                # YOLO miss → coast: predict ต่อด้วย velocity (แทนการถือค่าเก่านิ่ง ๆ)
                tau = self.coast_tau if self.mode == "kalman_coast_bounded" else None
                self._meas, self._vel = self._ab_predict(t - self.latency, decay_tau=tau)

        if self._meas is None:
            return

        # 2) ส่งคำสั่งแบบ throttle (เหมือน G0 throttle ของ GRBL)
        if t - self._last_cmd_t < self.cmd_interval:
            return
        self._last_cmd_t = t

        abs_pan, abs_tilt = self._meas
        vel_pan, vel_tilt = self._vel
        eff_lead = 0.0 if self.mode == "p" else self.lead  # "p" = ไม่มี feedforward
        # P-control บน error สัมบูรณ์ (อ้างตำแหน่งแขนปัจจุบันทั้งคู่) + feedforward lead
        #   kp=1 → เล็งไปที่ bearing พอดี; kp<1 = หน่วง; kp>1 = พุ่งเกิน
        tgt_pan = arm.pan + self.kp * (abs_pan - arm.pan) + vel_pan * eff_lead
        tgt_tilt = arm.tilt + self.kp * (abs_tilt - arm.tilt) + vel_tilt * eff_lead
        arm.set_target(tgt_pan, tgt_tilt)


def _interp_hist(hist, t_query):
    """interpolate ค่าใน history ที่เวลา t_query (linear). คืน tuple เดียวกับ element."""
    if not hist:
        return None
    if t_query <= hist[0][0]:
        return hist[0]
    if t_query >= hist[-1][0]:
        return hist[-1]
    lo, hi = 0, len(hist) - 1
    # hist เรียงตามเวลา → ค้นเชิงเส้นจากท้าย (ส่วนใหญ่ query ใกล้ปลาย)
    for i in range(len(hist) - 1, 0, -1):
        t0 = hist[i - 1][0]
        t1 = hist[i][0]
        if t0 <= t_query <= t1:
            a = (t_query - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(hist[i - 1][j] + a * (hist[i][j] - hist[i - 1][j]) for j in range(len(hist[i])))
    return hist[-1]


# =============================================================================
# คำนวณ feasibility เชิงทฤษฎี
# =============================================================================
def feasibility_note(omega, arm_max, latency, cmd_interval, lead, reticle_deg, det_dt=0.0):
    """คืน (verdict, ขนาด lag เชิงทฤษฎี deg, ข้อความอธิบาย)"""
    # เงื่อนไขจำเป็น: แขนต้องเร็วกว่าโดรน
    if omega > arm_max:
        return ("CANNOT-KEEP-UP", float("inf"),
                f"ω({omega:.1f}) > arm_max({arm_max:.1f}) deg/s — แขนหมุนช้ากว่าโดรน")
    # lag ค้างสภาวะคงตัว ≈ ω·(ดีเลย์รวม − lead)  โดยดีเลย์รวม = latency + ครึ่งคาบ detect ZOH + ครึ่งคาบ throttle
    eff_delay = latency + 0.5 * det_dt + 0.5 * cmd_interval - lead
    lag = abs(omega * eff_delay)
    if lag <= reticle_deg:
        return ("ON-TARGET", lag,
                f"lag≈{lag:.2f}° ≤ reticle {reticle_deg:.2f}° — เล็งตรงเป้า")
    return ("LAGS-BEHIND", lag,
            f"lag≈{lag:.2f}° > reticle {reticle_deg:.2f}° — ตามทันแต่ยังเล็งไม่ตรงกลาง")


# =============================================================================
# รันจำลอง 1 รอบ
# =============================================================================
def run_sim(cam, arm, args, display=True):
    ppd_x, ppd_y = cam["ppd_x"], cam["ppd_y"]
    w, h = cam["width"], cam["height"]
    reticle_px = max(10, int(min(w, h) * 0.03))
    reticle_deg = reticle_px / ppd_x

    # ω จากความเร็ว/ระยะ ถ้าระบุ --omega-deg จะ override
    if args.omega_deg is not None:
        omega = args.omega_deg
        src = f"omega={omega:.1f} deg/s (กำหนดตรง)"
    else:
        omega = (args.speed_ms / max(args.range_m, 1e-6)) * (180.0 / math.pi)
        src = f"v={args.speed_ms} m/s, R={args.range_m} m → ω={omega:.1f} deg/s"

    drone = DroneModel(args.pattern, omega, cam["fov_h"], cam["fov_v"])
    armM = ArmModel(arm)
    ctrl = ChaseController(
        ppd_x, ppd_y, args.kp, args.lead_ms / 1000.0,
        arm["cmd_interval"], args.latency_ms / 1000.0, 1.0 / args.det_fps,
        noise_px=args.noise_px, miss_rate=args.miss_rate, seed=args.seed,
        mode=getattr(args, "controller", "plead"),
        alpha=getattr(args, "ab_alpha", 0.5), beta=getattr(args, "ab_beta", 0.08),
        coast_tau=getattr(args, "coast_tau", 0.12),
    )

    sim_dt = 1.0 / 200.0
    hist = deque(maxlen=int(2.0 / sim_dt))  # เก็บ 2 วินาทีล่าสุด
    t = 0.0
    err_max = 0.0
    errs = []            # error (deg) เฉพาะที่อยู่ใน FOV และหลัง warmup (สำหรับ median/avg)
    on_target_frames = 0
    lost_frames = 0      # โดรนหลุด FOV (หลัง warmup)

    verdict, theo_lag, note = feasibility_note(
        omega, min(arm["max_pan_rate"], arm["max_tilt_rate"]),
        args.latency_ms / 1000.0, arm["cmd_interval"], args.lead_ms / 1000.0, reticle_deg,
        det_dt=1.0 / args.det_fps,
    )

    # เตรียมหน้าต่างแสดงผล
    disp_w, disp_h = 1280, int(1280 * h / w)
    scale = disp_w / w
    win = "Arm Chase Sim"
    paused = False
    trail = deque(maxlen=60)
    err_hist = deque(maxlen=int(6.0 / sim_dt))  # error 6 วินาทีล่าสุด (สำหรับกราฟ oscilloscope)
    if display and _HAS_CV2:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, disp_h)

    # เตรียมบันทึกวิดีโอ (headless ได้) — sample เฟรมให้ได้ video-fps แบบ real-time
    record_path = getattr(args, "record", None)
    writer = None
    frame_every = 1
    if record_path and _HAS_CV2:
        frame_every = max(1, int(round((1.0 / max(args.video_fps, 1.0)) / sim_dt)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record_path, fourcc, args.video_fps, (disp_w, disp_h))
        if not writer.isOpened():
            print(f"[sim] เปิด VideoWriter ไม่ได้ ({record_path}) — ข้ามการบันทึก")
            writer = None
    elif record_path and not _HAS_CV2:
        print("[sim] ไม่มี cv2 — บันทึกวิดีโอไม่ได้")

    step_idx = 0
    err_deg = 0.0
    while t < args.duration_s:
        if not paused:
            dpan, dtilt = drone.angle_at(t)
            hist.append((t, dpan, dtilt, armM.pan, armM.tilt))
            ctrl.update(t, armM, hist)
            armM.step(sim_dt)

            # error การเล็งจริง (ความจริง ณ ปัจจุบัน ไม่ใช่ที่ controller เห็น)
            epan = dpan - armM.pan
            etilt = dtilt - armM.tilt
            err_deg = math.hypot(epan, etilt)
            in_fov = abs(epan) <= cam["fov_h"] / 2 and abs(etilt) <= cam["fov_v"] / 2
            err_hist.append((t, err_deg))
            if t >= args.warmup_s:  # ตัดช่วง warmup (แขนวิ่งเข้าเป้าครั้งแรก)
                if in_fov:
                    errs.append(err_deg)
                    err_max = max(err_max, err_deg)
                    if err_deg <= reticle_deg:
                        on_target_frames += 1
                else:
                    lost_frames += 1
            t += sim_dt
            step_idx += 1

        want_frame = (display and _HAS_CV2) or (writer is not None)
        if want_frame and _HAS_CV2 and (step_idx % frame_every == 0):
            frame = _render(cam, armM, dpan, dtilt, ppd_x, ppd_y, reticle_px,
                            scale, disp_w, disp_h, trail, t, err_deg, reticle_deg,
                            omega, arm, verdict, note, src, args, meas=ctrl._meas,
                            err_hist=err_hist)
            if writer is not None:
                writer.write(frame)
            if display and _HAS_CV2:
                cv2.imshow(win, frame)
                k = cv2.waitKey(max(1, int(sim_dt * 1000))) & 0xFF
                if k in (ord("q"), ord("Q"), 27):
                    break
                elif k == 32:
                    paused = not paused

    if writer is not None:
        writer.release()
        print(f"[sim] บันทึกวิดีโอ: {record_path}  ({args.video_fps:.0f} fps, ~{args.duration_s:.0f}s)")
    if display and _HAS_CV2:
        cv2.destroyAllWindows()

    total = len(errs) + lost_frames
    avg_err = float(np.mean(errs)) if errs else float("nan")
    med_err = float(np.median(errs)) if errs else float("nan")
    on_target_pct = 100.0 * on_target_frames / total if total else 0.0
    lost_pct = 100.0 * lost_frames / total if total else 0.0
    return {
        "omega": omega, "src": src, "verdict": verdict, "theo_lag": theo_lag,
        "note": note, "avg_err_deg": avg_err, "med_err_deg": med_err, "max_err_deg": err_max,
        "on_target_pct": on_target_pct, "lost_pct": lost_pct, "reticle_deg": reticle_deg,
    }


def _render(cam, armM, dpan, dtilt, ppd_x, ppd_y, reticle_px, scale, dw, dh,
            trail, t, err_deg, reticle_deg, omega, arm, verdict, note, src, args,
            meas=None, err_hist=None):
    img = np.full((dh, dw, 3), 28, np.uint8)
    cx, cy = dw // 2, dh // 2
    # crosshair = ทิศแขนเล็ง (กลางเฟรม)
    rr = int(reticle_px * scale)
    on = err_deg <= reticle_deg
    col_reticle = (0, 230, 0) if on else (0, 200, 255)
    cv2.circle(img, (cx, cy), rr, col_reticle, 2)
    cv2.line(img, (cx - rr - 10, cy), (cx + rr + 10, cy), col_reticle, 1)
    cv2.line(img, (cx, cy - rr - 10), (cx, cy + rr + 10), col_reticle, 1)

    # ตำแหน่งโดรนในเฟรม (relative ต่อทิศแขน)
    px = cx + (dpan - armM.pan) * ppd_x * scale
    py = cy + (dtilt - armM.tilt) * ppd_y * scale
    ipx, ipy = int(px), int(py)
    trail.append((ipx, ipy))
    for i in range(1, len(trail)):
        cv2.line(img, trail[i - 1], trail[i], (90, 90, 90), 1)
    in_view = 0 <= ipx < dw and 0 <= ipy < dh
    bw = max(14, int(0.6 * ppd_x * scale))  # bbox โดรน ~0.6° กว้าง
    if in_view:
        cv2.rectangle(img, (ipx - bw // 2, ipy - bw // 2), (ipx + bw // 2, ipy + bw // 2),
                      (60, 60, 255), 2)
        cv2.line(img, (cx, cy), (ipx, ipy), (255, 255, 0), 1)
    else:
        # ลูกศรชี้ทิศโดรนที่หลุดเฟรม
        ang = math.atan2(py - cy, px - cx)
        ex = int(cx + math.cos(ang) * (dw * 0.45))
        ey = int(cy + math.sin(ang) * (dh * 0.45))
        cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

    # จุดที่ controller "เห็น" โดรน (bearing ที่วัดได้ + YOLO jitter) — แสดงเมื่อเปิด noise/miss
    if meas is not None and (args.noise_px > 0 or args.miss_rate > 0):
        mpx = int(round(cx + (meas[0] - armM.pan) * ppd_x * scale))
        mpy = int(round(cy + (meas[1] - armM.tilt) * ppd_y * scale))
        if 0 <= mpx < dw and 0 <= mpy < dh:
            cv2.drawMarker(img, (mpx, mpy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

    # HUD (ASCII เท่านั้น — OpenCV Hershey ไม่รองรับฟอนต์ไทย)
    vcol = {"ON-TARGET": (0, 230, 0), "LAGS-BEHIND": (0, 200, 255),
            "CANNOT-KEEP-UP": (0, 0, 255)}.get(verdict, (200, 200, 200))
    hint = {"ON-TARGET": "tracking centered within reticle",
            "LAGS-BEHIND": "in view but not centered: reduce latency / add lead",
            "CANNOT-KEEP-UP": "arm slower than target: needs faster arm"}.get(verdict, "")
    lines = [
        (f"[{cam['name']}] {cam['width']}x{cam['height']} FOV {cam['fov_h']:.0f}x{cam['fov_v']:.0f}  "
         f"px/deg {ppd_x:.1f}/{ppd_y:.1f}", (200, 200, 200)),
        (f"target rate {omega:.1f} deg/s", (180, 220, 255)),
        (f"arm max {arm['max_pan_rate']:.0f}/{arm['max_tilt_rate']:.0f} deg/s  "
         f"lat {args.latency_ms:.0f}ms  lead {args.lead_ms:.0f}ms  det {args.det_fps:.0f}fps  "
         f"jitter {args.noise_px:.0f}px  miss {args.miss_rate*100:.0f}%", (180, 180, 180)),
        (f"err {err_deg:5.2f} deg  (reticle {reticle_deg:.2f} deg)  t={t:5.1f}s", col_reticle),
        (f"VERDICT: {verdict} - {hint}", vcol),
        ("Q/ESC=quit  SPACE=pause", (140, 140, 140)),
    ]
    y = 24
    for txt, c in lines:
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
        y += 26

    if err_hist is not None:
        _draw_err_trace(img, err_hist, reticle_deg, dw, dh)
    return img


def _draw_err_trace(img, err_hist, reticle_deg, dw, dh):
    """กราฟ oscilloscope: aim error (deg) เทียบเวลา ที่แถบล่างของเฟรม"""
    if not err_hist or len(err_hist) < 2:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    H = 130
    y0 = dh - H
    sub = img[y0:dh, 0:dw]
    img[y0:dh, 0:dw] = cv2.addWeighted(sub, 0.25, np.full_like(sub, 18), 0.75, 0)
    cv2.line(img, (0, y0), (dw, y0), (80, 80, 80), 1)
    pad = 10
    plot_top, plot_bot = y0 + 24, dh - pad
    ph = max(1, plot_bot - plot_top)
    ts = [e[0] for e in err_hist]
    es = [e[1] for e in err_hist]
    t_lo, t_hi = ts[0], ts[-1]
    tw = max(t_hi - t_lo, 1e-3)
    ymax = max(reticle_deg * 3.0, max(es), 2.0)

    def X(tt):
        return int(pad + (tt - t_lo) / tw * (dw - 2 * pad))

    def Y(ee):
        return int(plot_bot - min(ee, ymax) / ymax * ph)

    ry = Y(reticle_deg)
    cv2.line(img, (pad, ry), (dw - pad, ry), (0, 180, 0), 1)
    cv2.putText(img, f"reticle {reticle_deg:.2f}", (dw - 160, ry - 4), font, 0.42, (0, 210, 0), 1)
    col = (0, 230, 0) if es[-1] <= reticle_deg else (0, 170, 255)
    pts = [(X(ts[i]), Y(es[i])) for i in range(len(ts))]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    cv2.putText(img, f"aim error (deg) vs time   now {es[-1]:.2f} deg   (y-max {ymax:.1f})",
                (pad, y0 + 17), font, 0.45, (220, 220, 220), 1)


# =============================================================================
# Sweep: หาความเร็วโดรนสูงสุดที่ยังไล่ทัน (on-target ≥ 80% เวลา)
# =============================================================================
def run_sweep(cam, arm, args):
    print("\n=== FEASIBILITY SWEEP (pattern=%s, R=%.0f m, lat=%.0fms, lead=%.0fms) ===" %
          (args.pattern, args.range_m, args.latency_ms, args.lead_ms))
    print(" v(m/s)  omega(deg/s)  med_err  max_err  on_tgt%  lost%  verdict")
    last_ok = None
    for v in range(5, 121, 5):
        a = argparse.Namespace(**vars(args))
        a.speed_ms = float(v)
        a.omega_deg = None
        r = run_sim(cam, arm, a, display=False)
        ok = r["med_err_deg"] <= r["reticle_deg"] and r["lost_pct"] < 5.0
        print(" %5d   %7.1f      %6.2f   %6.2f   %5.1f  %5.1f   %s%s" %
              (v, r["omega"], r["med_err_deg"], r["max_err_deg"], r["on_target_pct"],
               r["lost_pct"], r["verdict"], "  <= ตรงเป้า" if ok else ""))
        if ok:
            last_ok = (v, r["omega"])
    print("-" * 64)
    if last_ok:
        print("✅ ความเร็วโดรนสูงสุดที่ median error ≤ reticle ที่ R=%.0f m: ~%d m/s (ω≈%.1f deg/s)" %
              (args.range_m, last_ok[0], last_ok[1]))
        # แปลงเป็นระยะปลอดภัยขั้นต่ำสำหรับโดรนเร็ว
        vref = 30.0
        omega_arm = min(arm["max_pan_rate"], arm["max_tilt_rate"])
        r_min = vref / (math.radians(omega_arm)) if omega_arm > 0 else float("inf")
        print("ℹ️  ถ้าโดรนเร็ว %.0f m/s, แขน %.0f deg/s ตามทันเชิงมุมได้เมื่อระยะ R ≥ ~%.0f m (เงื่อนไขความเร็วล้วน)" %
              (vref, omega_arm, r_min))
    else:
        print("❌ ไม่มีความเร็วใดในช่วงทดสอบที่เล็งตรงเป้า ≥80%% — ลดดีเลย์/เพิ่ม lead/เพิ่มความเร็วแขน")


# =============================================================================
# Sweep ตามระยะ R: ความเร็วโดรนคงที่ → หาระยะ "ใกล้สุด" ที่ยังไล่ทัน
# (ยิ่งใกล้ ω ยิ่งสูง ยิ่งยาก; ยิ่งไกล ω ต่ำ ยิ่งง่าย)
# =============================================================================
def run_sweep_range(cam, arm, args):
    ranges = [20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
    arm_max = min(arm["max_pan_rate"], arm["max_tilt_rate"])
    print("\n=== RANGE SWEEP (pattern=%s, v=%.0f m/s, lat=%.0fms, lead=%.0fms) ===" %
          (args.pattern, args.speed_ms, args.latency_ms, args.lead_ms))
    print(" R(m)   omega(deg/s)  med_err  max_err  on_tgt%  lost%  verdict")
    closest_ok = None
    for R in ranges:
        a = argparse.Namespace(**vars(args))
        a.range_m = float(R)
        a.omega_deg = None
        r = run_sim(cam, arm, a, display=False)
        ok = r["med_err_deg"] <= r["reticle_deg"] and r["lost_pct"] < 5.0
        print(" %4d   %7.1f      %6.2f   %6.2f   %5.1f  %5.1f   %s%s" %
              (R, r["omega"], r["med_err_deg"], r["max_err_deg"], r["on_target_pct"],
               r["lost_pct"], r["verdict"], "  <= ตรงเป้า" if ok else ""))
        if ok and closest_ok is None:  # R น้อยสุด (ใกล้สุด) ที่ยัง OK — ไกลกว่านี้ยิ่งง่าย
            closest_ok = (R, r["omega"])
    print("-" * 64)
    if closest_ok:
        print("✅ ที่ v=%.0f m/s: ระยะ 'ใกล้สุด' ที่ยังเล็งตรงเป้า (median ≤ reticle) คือ ~%d m (ω≈%.1f deg/s)" %
              (args.speed_ms, closest_ok[0], closest_ok[1]))
    else:
        print("❌ ไม่มีระยะใดในช่วงทดสอบที่เล็งตรงเป้า — ลด latency / เพิ่ม lead")
    # ระยะ "ตามทันเชิงมุม" ขั้นต่ำ (ไม่หลุด FOV): arm_max ≥ ω → R ≥ v/(arm_max·π/180)
    r_keepup = args.speed_ms / math.radians(arm_max) if arm_max > 0 else float("inf")
    print("ℹ️  เงื่อนไขความเร็วล้วน (arm_max=%.0f deg/s): ตามทันเชิงมุมเมื่อ R ≥ ~%.0f m "
          "(ใกล้กว่านี้แขนหมุนไม่ทัน โดรนหลุดเฟรม)" % (arm_max, r_keepup))


# =============================================================================
# Heatmap 2 มิติ v × R → CSV (long-format + matrix verdict + matrix median error)
# =============================================================================
_VERDICT_CODE = {"CANNOT-KEEP-UP": 0, "LAGS-BEHIND": 1, "ON-TARGET": 2}
_VERDICT_CHAR = {0: ".", 1: "+", 2: "#"}  # preview ASCII


def run_heatmap_csv(cam, arm, args):
    vs = [float(v) for v in _frange(args.v_min, args.v_max, args.v_step)]
    rs = [float(x) for x in str(args.ranges).split(",") if x.strip()]
    dur = min(args.duration_s, 15.0)  # จำกัดเวลาต่อเซลล์ให้รันไว (มีหลายร้อยเซลล์)
    base = args.heatmap_csv
    if base.lower().endswith(".csv"):
        base = base[:-4]

    print("\n=== HEATMAP v × R (pattern=%s, lat=%.0fms, lead=%.0fms, %d×%d=%d cells, %.0fs/cell) ===" %
          (args.pattern, args.latency_ms, args.lead_ms, len(vs), len(rs), len(vs) * len(rs), dur))

    grid = {}  # (v,R) -> result dict
    for v in vs:
        for R in rs:
            a = argparse.Namespace(**vars(args))
            a.speed_ms = v
            a.range_m = R
            a.omega_deg = None
            a.duration_s = dur
            grid[(v, R)] = run_sim(cam, arm, a, display=False)

    # 1) long-format CSV (แถวละ 1 เซลล์ — โหลดเข้า pandas/Excel แล้ว pivot ได้)
    long_path = base + "_long.csv"
    with open(long_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["v_ms", "range_m", "omega_deg_s", "med_err_deg", "max_err_deg",
                       "on_target_pct", "lost_pct", "verdict", "verdict_code", "ok"])
        for v in vs:
            for R in rs:
                r = grid[(v, R)]
                code = _VERDICT_CODE.get(r["verdict"], -1)
                ok = int(r["med_err_deg"] <= r["reticle_deg"] and r["lost_pct"] < 5.0)
                wcsv.writerow([f"{v:.0f}", f"{R:.0f}", f"{r['omega']:.2f}",
                               f"{r['med_err_deg']:.3f}", f"{r['max_err_deg']:.3f}",
                               f"{r['on_target_pct']:.1f}", f"{r['lost_pct']:.1f}",
                               r["verdict"], code, ok])

    # 2) matrix CSV (rows=v, cols=R) — เปิดเป็น heatmap ได้ทันที
    verdict_path = base + "_verdict.csv"
    mederr_path = base + "_mederr.csv"
    _write_matrix(verdict_path, vs, rs, lambda v, R: _VERDICT_CODE.get(grid[(v, R)]["verdict"], -1), "%d")
    _write_matrix(mederr_path, vs, rs, lambda v, R: grid[(v, R)]["med_err_deg"], "%.3f")

    # 3) preview ASCII ในเทอร์มินัล
    print("\nverdict heatmap  (#=ON-TARGET  +=LAGS-BEHIND  .=CANNOT-KEEP-UP)")
    hdr = "v\\R  " + " ".join("%4.0f" % R for R in rs)
    print(hdr)
    for v in vs:
        row = "%4.0f " % v + "    ".join(
            _VERDICT_CHAR.get(_VERDICT_CODE.get(grid[(v, R)]["verdict"], -1), "?") for R in rs
        )
        print(row)
    print("\nบันทึกแล้ว:\n  %s  (long-format)\n  %s  (matrix: 0/1/2)\n  %s  (matrix: median error °)" %
          (long_path, verdict_path, mederr_path))

    # 4) พล็อต PNG (median-error heatmap + เส้นแบ่งโซน)
    if args.no_png:
        return
    if not _HAS_PLT:
        print("  (ข้ามพล็อต PNG — ไม่มี matplotlib; pip install matplotlib)")
        return
    png_path = base + ".png"
    reticle_deg = grid[(vs[0], rs[0])]["reticle_deg"]
    _plot_heatmap_png(png_path, vs, rs, grid, reticle_deg, cam, arm, args)
    print("  %s  (PNG heatmap)" % png_path)


def _plot_heatmap_png(path, vs, rs, grid, reticle_deg, cam, arm, args):
    """median-error heatmap: แกน x=R, y=v, สี=median error; overlay โซน + frontier."""
    nrows, ncols = len(vs), len(rs)
    Z = np.array([[grid[(v, R)]["med_err_deg"] for R in rs] for v in vs])
    lost = np.array([[grid[(v, R)]["lost_pct"] for R in rs] for v in vs])
    vmax = max(reticle_deg * 4.0, 2.0)  # ตัดสเกลสีให้เห็น contrast ช่วงใกล้ on-target

    fig, ax = _plt.subplots(figsize=(1.1 * ncols + 2, 0.5 * nrows + 2))
    im = ax.imshow(np.clip(Z, 0, vmax), origin="lower", aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"median aim error (deg)  [clip {vmax:.1f}]")

    ax.set_xticks(range(ncols)); ax.set_xticklabels([f"{R:.0f}" for R in rs])
    ax.set_yticks(range(nrows)); ax.set_yticklabels([f"{v:.0f}" for v in vs])
    ax.set_xlabel("range R (m)"); ax.set_ylabel("drone speed v (m/s)")
    ax.set_title("Arm chase feasibility — %s %dx%d FOV %.0fx%.0f\n"
                 "lat %.0fms  lead %.0fms  det %.0ffps  arm %.0f/%.0f deg/s  (reticle %.2f deg)" %
                 (cam["name"], cam["width"], cam["height"], cam["fov_h"], cam["fov_v"],
                  args.latency_ms, args.lead_ms, args.det_fps,
                  arm["max_pan_rate"], arm["max_tilt_rate"], reticle_deg), fontsize=9)

    # มาร์ก CANNOT-KEEP-UP (โดรนหลุด FOV) ด้วย 'x'
    for i in range(nrows):
        for j in range(ncols):
            if lost[i, j] >= 5.0:
                ax.text(j, i, "x", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    # เส้น frontier "on-target" (median ≤ reticle): R น้อยสุดต่อ v ที่ยัง ok
    fx, fy = [], []
    for i, v in enumerate(vs):
        for j, R in enumerate(rs):
            if Z[i, j] <= reticle_deg and lost[i, j] < 5.0:
                fx.append(j); fy.append(i); break
    if fx:
        ax.plot(fx, fy, "w-", lw=2, label="on-target frontier (median<=reticle)")
        ax.plot(fx, fy, "ko", ms=3)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    _plt.close(fig)


def _frange(lo, hi, step):
    vals = []
    x = lo
    while x <= hi + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def _write_matrix(path, vs, rs, cell_fn, fmt):
    """เขียน CSV แบบ matrix: มุมซ้ายบน = 'v\\R', หัวคอลัมน์ = R, แถว = v"""
    with open(path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["v\\R"] + [f"{R:.0f}" for R in rs])
        for v in vs:
            wcsv.writerow([f"{v:.0f}"] + [fmt % cell_fn(v, R) for R in rs])


# =============================================================================
# เทียบ controller (P / P+lead / Kalman-smoothed lead) บนแกน noise-vs-error
# =============================================================================
def _ab_stable(alpha, beta):
    """เงื่อนไขเสถียรภาพ alpha-beta filter"""
    return alpha > 0 and beta > 0 and alpha < 2.0 and (4.0 - 2.0 * alpha - beta) > 0


_CTRL_SERIES = [
    ("p", "P (no lead)", "#1f77b4"),
    ("plead", "P + raw lead", "#ff7f0e"),
    ("kalman", "Kalman-smoothed lead", "#2ca02c"),
    ("kalman_coast", "Kalman + coast on miss", "#d62728"),
]


def _compare_core(cam, arm, args, xvals, x_set, x_col, x_axis_label, title):
    """
    แกนกลางสำหรับเทียบ controller: กวาดค่า x (xvals) ต่อ controller แต่ละตัว,
    เฉลี่ย median error ข้ามหลาย seed (reps), เซฟ CSV + PNG.
      x_set(a, xv): mutate namespace a ตั้งค่าพารามิเตอร์ที่กำลังกวาด
    """
    dur = min(args.duration_s, 15.0)
    print(" %9s | " % x_col + " | ".join("%-22s" % lbl for _, lbl, _ in _CTRL_SERIES))
    reticle = None
    means = {k: [] for k, _, _ in _CTRL_SERIES}
    stds = {k: [] for k, _, _ in _CTRL_SERIES}
    for xv in xvals:
        cells = []
        for key, _, _ in _CTRL_SERIES:
            errs = []
            for s in range(args.reps):
                a = argparse.Namespace(**vars(args))
                a.controller = key
                a.seed = args.seed + s
                a.omega_deg = None
                a.duration_s = dur
                x_set(a, xv)
                r = run_sim(cam, arm, a, display=False)
                errs.append(r["med_err_deg"])
                reticle = r["reticle_deg"]
            means[key].append(float(np.mean(errs)))
            stds[key].append(float(np.std(errs)))
            cells.append(np.mean(errs))
        print(" %9.0f | " % xv + " | ".join("%6.2f deg            " % c for c in cells))

    base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
    csv_path = base + ".csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        head = [x_col]
        for k, lbl, _ in _CTRL_SERIES:
            head += [f"{lbl}_mean_deg", f"{lbl}_std_deg"]
        wcsv.writerow(head)
        for i, xv in enumerate(xvals):
            row = [f"{xv:.0f}"]
            for k, _, _ in _CTRL_SERIES:
                row += [f"{means[k][i]:.3f}", f"{stds[k][i]:.3f}"]
            wcsv.writerow(row)
    print("\nบันทึก: %s  (mean/std median-error ต่อ controller)" % csv_path)

    if not _HAS_PLT:
        print("  (ข้ามพล็อต PNG — ไม่มี matplotlib)")
        return
    png_path = base + ".png"
    fig, ax = _plt.subplots(figsize=(8, 5))
    for key, lbl, col in _CTRL_SERIES:
        m = np.array(means[key]); sd = np.array(stds[key])
        ax.plot(xvals, m, "-o", color=col, label=lbl)
        ax.fill_between(xvals, m - sd, m + sd, color=col, alpha=0.15)
    if reticle is not None:
        ax.axhline(reticle, ls="--", color="gray", label=f"reticle {reticle:.2f} deg")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("median aim error (deg)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png_path, dpi=120); _plt.close(fig)
    print("        %s  (PNG)" % png_path)


def run_compare_noise(cam, arm, args):
    noises = [float(x) for x in str(args.noise_list).split(",") if x.strip()]
    print("\n=== CONTROLLER COMPARE vs NOISE (v=%.0f m/s, R=%.0f m, lat=%.0fms, lead=%.0fms, "
          "miss=%.0f%%, reps=%d) ===" %
          (args.speed_ms, args.range_m, args.latency_ms, args.lead_ms, args.miss_rate * 100, args.reps))
    if args.miss_rate <= 0.0:
        print("  หมายเหตุ: miss-rate=0 → เส้น 'Kalman' กับ 'Kalman+coast' จะทับกัน (ใส่ --miss-rate เช่น 0.3 เพื่อเห็นผล coast)")
    title = ("Controller robustness to YOLO noise — %s  v=%.0f m/s R=%.0f m\n"
             "lat %.0fms  lead %.0fms  det %.0ffps  miss %.0f%%  ab(a=%.2f,b=%.3f)  (reps=%d)" %
             (cam["name"], args.speed_ms, args.range_m, args.latency_ms,
              args.lead_ms, args.det_fps, args.miss_rate * 100, args.ab_alpha, args.ab_beta, args.reps))
    _compare_core(cam, arm, args, noises,
                  lambda a, xv: setattr(a, "noise_px", xv),
                  "jitter_px", "YOLO bbox jitter (px std)", title)


def run_compare_miss(cam, arm, args):
    # miss_list เป็นเปอร์เซ็นต์ (0,10,20,...) — setter หารด้วย 100
    misses = [float(x) for x in str(args.miss_list).split(",") if x.strip()]
    print("\n=== CONTROLLER COMPARE vs MISS-RATE (v=%.0f m/s, R=%.0f m, lat=%.0fms, lead=%.0fms, "
          "jitter=%.0fpx, reps=%d) ===" %
          (args.speed_ms, args.range_m, args.latency_ms, args.lead_ms, args.noise_px, args.reps))
    title = ("Controller robustness to detection miss-rate — %s  v=%.0f m/s R=%.0f m\n"
             "lat %.0fms  lead %.0fms  det %.0ffps  jitter %.0fpx  ab(a=%.2f,b=%.3f)  (reps=%d)" %
             (cam["name"], args.speed_ms, args.range_m, args.latency_ms,
              args.lead_ms, args.det_fps, args.noise_px, args.ab_alpha, args.ab_beta, args.reps))
    _compare_core(cam, arm, args, misses,
                  lambda a, xv: setattr(a, "miss_rate", xv / 100.0),
                  "miss_pct", "detection miss rate (%)", title)


# =============================================================================
# Auto-tune: กวาด grid alpha × beta ของ Kalman หา median error ต่ำสุด
# =============================================================================
def run_autotune(cam, arm, args):
    alphas = [float(x) for x in str(args.alpha_list).split(",") if x.strip()]
    betas = [float(x) for x in str(args.beta_list).split(",") if x.strip()]
    mode = args.controller if args.controller in ("kalman", "kalman_coast") else "kalman_coast"
    if mode != args.controller:
        print(f"  หมายเหตุ: --controller={args.controller} ไม่ใช้ alpha-beta — auto-tune ใช้ '{mode}' แทน")
    dur = min(args.duration_s, 12.0)
    print("\n=== AUTO-TUNE alpha×beta (ctrl=%s, v=%.0f m/s R=%.0f m, jitter=%.0fpx, miss=%.0f%%, "
          "%d×%d cells, reps=%d) ===" %
          (mode, args.speed_ms, args.range_m, args.noise_px, args.miss_rate * 100,
           len(alphas), len(betas), args.reps))

    Z = np.full((len(alphas), len(betas)), np.nan)
    reticle = None
    best = None  # (err, alpha, beta)
    for i, a_val in enumerate(alphas):
        for j, b_val in enumerate(betas):
            if not _ab_stable(a_val, b_val):
                continue  # ข้ามเซลล์ไม่เสถียร -> คง NaN
            errs = []
            for s in range(args.reps):
                a = argparse.Namespace(**vars(args))
                a.controller = mode
                a.ab_alpha = a_val
                a.ab_beta = b_val
                a.seed = args.seed + s
                a.omega_deg = None
                a.duration_s = dur
                r = run_sim(cam, arm, a, display=False)
                errs.append(r["med_err_deg"])
                reticle = r["reticle_deg"]
            e = float(np.mean(errs))
            Z[i, j] = e
            if best is None or e < best[0]:
                best = (e, a_val, b_val)

    # ตารางในเทอร์มินัล (แถว=alpha, คอลัมน์=beta; × = ไม่เสถียร)
    print("alpha\\beta " + " ".join("%6.2f" % b for b in betas))
    for i, a_val in enumerate(alphas):
        cells = " ".join((" %5s" % "×") if np.isnan(Z[i, j]) else ("%6.2f" % Z[i, j])
                         for j in range(len(betas)))
        print("  %6.2f   %s" % (a_val, cells))
    if best is None:
        print("❌ ไม่มีคู่ alpha/beta ที่เสถียรในช่วงที่กำหนด")
        return
    print("-" * 60)
    print("✅ ดีสุด: alpha=%.3f beta=%.3f → median error %.3f° (reticle %.2f°)" %
          (best[1], best[2], best[0], reticle if reticle else float("nan")))
    print("   ใช้กับ run จริง:  --ab-alpha %.3f --ab-beta %.3f" % (best[1], best[2]))

    # CSV
    base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
    csv_path = base + ".csv"
    _write_matrix(csv_path, alphas, betas,
                  lambda a_val, b_val: Z[alphas.index(a_val)][betas.index(b_val)], "%.4f")
    print("\nบันทึก: %s  (matrix median error °, แถว=alpha คอลัมน์=beta)" % csv_path)

    # PNG heatmap
    if not _HAS_PLT:
        print("  (ข้ามพล็อต PNG — ไม่มี matplotlib)")
        return
    png_path = base + ".png"
    vmax = float(np.nanpercentile(Z, 90)) if np.isfinite(Z).any() else 1.0
    cmap = matplotlib.cm.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgray")
    fig, ax = _plt.subplots(figsize=(1.0 * len(betas) + 2, 0.7 * len(alphas) + 2))
    im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", aspect="auto",
                   cmap=cmap, vmin=0, vmax=max(vmax, (reticle or 1.0)))
    fig.colorbar(im, ax=ax).set_label("median aim error (deg)")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:.2f}" for b in betas])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("beta (velocity gain)"); ax.set_ylabel("alpha (position gain)")
    ax.set_title("Kalman (alpha-beta) auto-tune — %s  ctrl=%s\n"
                 "v=%.0f m/s R=%.0f m  jitter %.0fpx  miss %.0f%%  (reps=%d)" %
                 (cam["name"], mode, args.speed_ms, args.range_m, args.noise_px,
                  args.miss_rate * 100, args.reps), fontsize=9)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            if np.isnan(Z[i, j]):
                ax.text(j, i, "×", ha="center", va="center", color="dimgray", fontsize=9)
            else:
                ax.text(j, i, "%.2f" % Z[i, j], ha="center", va="center", color="black", fontsize=7)
    bi, bj = alphas.index(best[1]), betas.index(best[2])
    ax.plot(bj, bi, "*", color="blue", ms=18, markeredgecolor="white",
            label="best a=%.2f b=%.2f" % (best[1], best[2]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout(); fig.savefig(png_path, dpi=120); _plt.close(fig)
    print("        %s  (PNG alpha-beta heatmap)" % png_path)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="จำลองแขนหมุนไล่ตามโดรน (อิง resolution กล้องใน config)")
    ap.add_argument("--camera", default=None, help="ชื่อกล้องใน config (default = ACTIVE_CAMERA)")
    ap.add_argument("--speed-ms", type=float, default=30.0, help="ความเร็วเชิงเส้นโดรน (m/s)")
    ap.add_argument("--range-m", type=float, default=150.0, help="ระยะโดรน (m)")
    ap.add_argument("--omega-deg", type=float, default=None, help="กำหนด ω เชิงมุมตรง ๆ (deg/s) override speed/range")
    ap.add_argument("--pattern", choices=["cross", "sine"], default="cross", help="รูปแบบการบิน")
    ap.add_argument("--latency-ms", type=float, default=300.0, help="ดีเลย์ท่อ detection (ms)")
    ap.add_argument("--det-fps", type=float, default=15.0, help="อัตราอัปเดต detection (fps)")
    ap.add_argument("--lead-ms", type=float, default=200.0, help="feedforward lead time (ms)")
    ap.add_argument("--kp", type=float, default=1.0, help="P gain ของ controller")
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="YOLO bbox center jitter (Gaussian std, px ต่อแกน; 0=ปิด)")
    ap.add_argument("--miss-rate", type=float, default=0.0,
                    help="โอกาส detection หลุดต่อเฟรม (0..1; ถือค่าเก่า)")
    ap.add_argument("--seed", type=int, default=0, help="seed สุ่มสำหรับ noise (ทำซ้ำได้)")
    ap.add_argument("--coast-tau", type=float, default=0.30,
                    help="kalman_coast_bounded: time-const decay ของ velocity ตอน coast (s); "
                         "จูนแล้ว 0.30 (fix danger zone 100%, keep sweet 72%)")
    ap.add_argument("--controller", choices=["p", "plead", "kalman", "kalman_coast", "kalman_coast_bounded"],
                    default="plead",
                    help="ตัวควบคุม: p=ไม่มี lead, plead=raw lead, kalman=smoothed lead, kalman_coast=+coast ตอน miss")
    ap.add_argument("--ab-alpha", type=float, default=0.5,
                    help="alpha-beta filter: น้ำหนักแก้ตำแหน่ง (สูง=ตอบไว/นิ่งน้อย, ต่ำ=นิ่ง/ช้า)")
    ap.add_argument("--ab-beta", type=float, default=0.08,
                    help="alpha-beta filter: น้ำหนักแก้ความเร็ว (สูง=ไล่ velocity ไว/สั่นง่าย)")
    ap.add_argument("--autotune", action="store_true",
                    help="กวาด grid alpha×beta ของ Kalman หา median error ต่ำสุด → CSV + PNG")
    ap.add_argument("--alpha-list", default="0.2,0.35,0.5,0.65,0.8,0.95",
                    help="autotune: รายการ alpha คั่นด้วย comma")
    ap.add_argument("--beta-list", default="0.02,0.05,0.1,0.2,0.3,0.4",
                    help="autotune: รายการ beta คั่นด้วย comma")
    ap.add_argument("--compare-noise", action="store_true",
                    help="เทียบ controllers บนกราฟ noise-vs-error → CSV + PNG")
    ap.add_argument("--compare-miss", action="store_true",
                    help="เทียบ controllers บนกราฟ miss-rate-vs-error → CSV + PNG (เห็นผล coasting ตรง ๆ)")
    ap.add_argument("--noise-list", default="0,2,4,6,8,12,16,20",
                    help="compare-noise: ระดับ jitter (px) คั่นด้วย comma")
    ap.add_argument("--miss-list", default="0,10,20,30,40,50,60",
                    help="compare-miss: ระดับ miss-rate (%) คั่นด้วย comma")
    ap.add_argument("--reps", type=int, default=3, help="compare-noise: จำนวน seed เฉลี่ยต่อจุด")
    ap.add_argument("--out", default="controller_compare.csv", help="compare-noise: path output (.csv/.png)")
    ap.add_argument("--duration-s", type=float, default=30.0, help="ระยะเวลาจำลอง (s)")
    ap.add_argument("--warmup-s", type=float, default=3.0, help="ตัดช่วงเริ่มต้นออกจากสถิติ (s)")
    ap.add_argument("--no-display", action="store_true", help="ไม่เปิดหน้าต่าง รันเร็ว พิมพ์สรุป")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="บันทึกวิดีโอผลทดสอบเป็น MP4 (ทำงานแบบ headless ได้)")
    ap.add_argument("--video-fps", type=float, default=30.0, help="record: เฟรมเรตวิดีโอ (real-time)")
    ap.add_argument("--sweep", action="store_true", help="กวาดความเร็วโดรน (R คงที่) หาความเร็วสูงสุดที่ไล่ทัน")
    ap.add_argument("--sweep-range", action="store_true", help="กวาดระยะ R (ความเร็วคงที่) หาระยะใกล้สุดที่ไล่ทัน")
    ap.add_argument("--heatmap-csv", default=None, metavar="PATH",
                    help="สร้าง heatmap 2 มิติ v×R แล้วเซฟ CSV (เช่น chase_heatmap.csv)")
    ap.add_argument("--v-min", type=float, default=5.0, help="heatmap: ความเร็วต่ำสุด (m/s)")
    ap.add_argument("--v-max", type=float, default=120.0, help="heatmap: ความเร็วสูงสุด (m/s)")
    ap.add_argument("--v-step", type=float, default=5.0, help="heatmap: step ความเร็ว (m/s)")
    ap.add_argument("--ranges", default="20,30,40,50,75,100,150,200,300,400,500",
                    help="heatmap: รายการระยะ R คั่นด้วย comma (m)")
    ap.add_argument("--no-png", action="store_true", help="heatmap: ไม่ต้องพล็อต PNG (เซฟ CSV อย่างเดียว)")
    ap.add_argument("--arm-accel", type=float, default=None,
                    help="ความเร่งแขนจำกัด deg/s^2 ทั้ง 2 แกน (default จาก config ~100; 0=ปิด rate-limit ล้วน)")
    args = ap.parse_args()

    cam = load_camera_params(args.camera)
    arm = load_arm_params()
    if args.arm_accel is not None:
        arm["max_pan_accel"] = arm["max_tilt_accel"] = args.arm_accel

    print("=" * 64)
    print(f"กล้อง: {cam['name']}  {cam['width']}x{cam['height']}  FOV {cam['fov_h']}x{cam['fov_v']}")
    print(f"px/deg: {cam['ppd_x']:.2f} (pan), {cam['ppd_y']:.2f} (tilt)")
    print(f"แขน: max {arm['max_pan_rate']:.0f}/{arm['max_tilt_rate']:.0f} deg/s, "
          f"accel {arm['max_pan_accel']:.0f}/{arm['max_tilt_accel']:.0f} deg/s², "
          f"limit pan{arm['x_lim']} tilt{arm['y_lim']}, throttle {arm['cmd_interval']*1000:.0f}ms")
    _margin = 4.0 - 2.0 * args.ab_alpha - args.ab_beta
    _stable = _ab_stable(args.ab_alpha, args.ab_beta)
    print(f"alpha-beta (Kalman): alpha={args.ab_alpha:.3f} beta={args.ab_beta:.3f} "
          f"-> {'STABLE' if _stable else 'UNSTABLE!'} (margin 4-2a-b={_margin:.3f})")
    if not _stable:
        print("  ⚠️ ค่านี้อาจไม่เสถียร — ควรให้ 0<alpha<2, beta>0, และ (4-2*alpha-beta)>0")
    print("=" * 64)

    if args.sweep:
        run_sweep(cam, arm, args)
        return
    if args.sweep_range:
        run_sweep_range(cam, arm, args)
        return
    if args.heatmap_csv:
        run_heatmap_csv(cam, arm, args)
        return
    if args.compare_noise:
        run_compare_noise(cam, arm, args)
        return
    if args.compare_miss:
        run_compare_miss(cam, arm, args)
        return
    if args.autotune:
        run_autotune(cam, arm, args)
        return

    display = (not args.no_display) and _HAS_CV2
    if not args.no_display and not _HAS_CV2:
        print("[sim] ไม่มี cv2 — รันโหมด headless")
    r = run_sim(cam, arm, args, display=display)

    print("\n--- ผลสรุป ---")
    print(f"{r['src']}")
    print(f"VERDICT : {r['verdict']}  ({r['note']})")
    print(f"error (หลัง warmup): median {r['med_err_deg']:.2f}° | avg {r['avg_err_deg']:.2f}° "
          f"| max {r['max_err_deg']:.2f}° | reticle {r['reticle_deg']:.2f}°")
    print(f"เล็งตรงเป้า: {r['on_target_pct']:.1f}% ของเวลา | โดรนหลุด FOV: {r['lost_pct']:.1f}%")
    print("\nวิธีไล่ให้ทัน: (1) แขน max rate ต้อง ≥ ω ของโดรน  "
          "(2) ลด latency ของท่อ detect  (3) ใส่ feedforward lead ≈ latency  "
          "(4) เพิ่ม det-fps/Kp พอประมาณ กันสั่น")


if __name__ == "__main__":
    main()
