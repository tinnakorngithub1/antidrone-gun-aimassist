#!/usr/bin/env python3
"""
33_lock_sim_closedloop.py — closed-loop simulation ของโหมด LOCK โดยรัน "โค้ดจริง" จากไฟล์ 22

ต่างจาก 23_arm_chase_sim.py: ไฟล์ 23 ใช้ controller จำลองของตัวเอง — แต่ไฟล์นี้เรียก
_SimpleIoUTracker, _TargetKalman, _lock_feed_bearing_measurement, _tick_lock และ
_variable_step_toward_target "ตัวจริง" จาก 22_gun_aim_assist_vector.py จึงทดสอบ P0 fix
(ego-motion compensation / bearing space / bounded coast) ได้ตรงกับ production 100%

สิ่งที่ไฟล์ 23/24 ขาดและไฟล์นี้เพิ่ม:
  1. กล้องติดบนแขน: ตำแหน่ง pixel ของโดรน = (bearing_โดรน − มุม_แขนจริง)·ppd + ศูนย์กลาง
     → พอแขนขยับ ภาพเลื่อน (ego-motion) ซึ่งคือโจทย์จริงของงานวิจัย
  2. commanded vs actual: โค้ดอ่าน pos_x (commanded, อัปเดตทันที) แต่กล้องเห็นมุมจริง
     ที่ lag ตาม rate/accel ของ GRBL → จับ residual mechanical lag ที่ ego-comp ลบไม่ได้
  3. pipeline latency + det-fps + bbox noise + miss-rate ตามที่วัดจริง

Usage:
  python3 33_lock_sim_closedloop.py --selftest         # assert P0 ทำงาน (CI)
  python3 33_lock_sim_closedloop.py --omega 8 --latency-ms 150 --det-fps 15
  python3 33_lock_sim_closedloop.py --sweep            # ตาราง omega × latency → CSV
"""
import sys
import os
import math
import time
import argparse
import importlib.util

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- deterministic RNG (ไม่พึ่ง global random state — สำคัญต่อ resume/CI) ---
class _LCG:
    """linear congruential generator — ทำซ้ำได้ตาม seed."""
    def __init__(self, seed=12345):
        self.s = seed & 0xFFFFFFFF
    def rand(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF
    def gauss(self, sigma):
        # Box-Muller
        u1 = max(1e-9, self.rand())
        u2 = self.rand()
        return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _load_production_module():
    """import 22_gun_aim_assist_vector.py โดยไม่รัน main() (มี __name__ guard อยู่แล้ว)."""
    spec = importlib.util.spec_from_file_location("gaa22", "22_gun_aim_assist_vector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PhysicsArm:
    """แขนจำลองแบบ rate + trapezoidal-accel limit (เลียนแบบ GRBL G0 planner).
    แยก commanded (pos_x/pos_y ที่โค้ดอ่าน) กับ actual (มุมจริงที่กล้องเห็น):
      - move_relative/move_absolute → อัปเดต commanded ทันที (optimistic เหมือน controller จริง)
        และตั้ง physical target
      - step(dt) → เลื่อน actual เข้าหา target ด้วย rate/accel จำกัด
      - ทุก sync_every_n_moves ครั้ง commanded snap → actual (เลียน sync_position_from_grbl)
    ถ้า ideal=True → actual = commanded ทันที (ไม่มี mechanical lag) สำหรับแยกตัวแปร."""

    def __init__(self, max_rate=200.0, accel=650.0, x_lim=(-65.0, 65.0), y_lim=(-35.0, 35.0),
                 sync_every_n_moves=10, ideal=False):
        self.pos_x = 0.0          # commanded pan (โค้ดอ่านค่านี้)
        self.pos_y = 0.0          # commanded tilt
        self._ax = 0.0            # actual physical pan (กล้องเห็นค่านี้)
        self._ay = 0.0
        self._vx = 0.0            # actual angular velocity
        self._vy = 0.0
        self._tx = 0.0            # physical target pan
        self._ty = 0.0
        self.max_rate = max_rate
        self.accel = accel
        self.mm_per_deg_pan = 1.0
        self.mm_per_deg_tilt = 1.0
        self._effective_x_limits = x_lim
        self._effective_y_limits = y_lim
        self.is_simulation_mode = True
        self.sync_every_n_moves = sync_every_n_moves
        self._move_count = 0
        self.ideal = ideal

    def _clip(self, x, lim):
        return max(lim[0], min(lim[1], x))

    def move_absolute(self, x, y, blocking=False):
        self.pos_x = self._clip(float(x), self._effective_x_limits)
        self.pos_y = self._clip(float(y), self._effective_y_limits)
        self._tx = self.pos_x
        self._ty = self.pos_y
        if self.ideal:
            self._ax, self._ay = self.pos_x, self.pos_y
            self._vx = self._vy = 0.0
        self._move_count += 1
        if (not self.ideal) and self.sync_every_n_moves > 0 and self._move_count >= self.sync_every_n_moves:
            self._move_count = 0
            # sync: commanded ← actual (เหมือน sync_position_from_grbl)
            self.pos_x = self._ax
            self.pos_y = self._ay
        return True

    def move_relative(self, dx, dy, blocking=False):
        self.move_absolute(self.pos_x + float(dx), self.pos_y + float(dy), blocking)

    def sync_position_from_grbl(self):
        self.pos_x = self._ax
        self.pos_y = self._ay

    def _axis_step(self, a, v, tgt, dt):
        """trapezoidal: เร่ง/เบรกจำกัด accel, ความเร็วจำกัด max_rate, เบรกทันเพื่อหยุดที่ tgt."""
        err = tgt - a
        # ความเร็วสูงสุดที่ยังเบรกทันก่อนถึงเป้า
        v_cap = math.copysign(min(self.max_rate, math.sqrt(2.0 * self.accel * abs(err))), err) if abs(err) > 1e-9 else 0.0
        # เร่งเข้าหา v_cap ด้วย accel จำกัด
        dv = v_cap - v
        max_dv = self.accel * dt
        dv = max(-max_dv, min(max_dv, dv))
        v_new = v + dv
        v_new = max(-self.max_rate, min(self.max_rate, v_new))
        a_new = a + v_new * dt
        # ไม่ให้เลยเป้า (กันแกว่งจาก dt หยาบ)
        if (tgt - a) * (tgt - a_new) < 0:
            a_new = tgt
            v_new = 0.0
        return a_new, v_new

    def step(self, dt):
        if self.ideal:
            return
        self._ax, self._vx = self._axis_step(self._ax, self._vx, self._tx, dt)
        self._ay, self._vy = self._axis_step(self._ay, self._vy, self._ty, dt)

    # มุมจริงที่กล้อง (บนแขน) เห็น
    @property
    def actual_pan(self):
        return self._ax

    @property
    def actual_tilt(self):
        return self._ay


class DroneBearing:
    """โดรนในพิกัดเชิงมุม (pan/tilt bearing). pattern sine: pan แกว่ง ±amp, peak rate = omega."""
    def __init__(self, omega_deg_s, amp_deg=20.0, tilt_amp_deg=5.0):
        self.omega = omega_deg_s
        self.amp = amp_deg
        self.tilt_amp = tilt_amp_deg
        # ω = amp · 2π/T → T = amp·2π/ω ; ให้ peak angular rate = ω
        self.w = (omega_deg_s / amp_deg) if amp_deg > 1e-9 else 0.0  # rad/s ของ argument

    def bearing_at(self, t):
        pan = self.amp * math.sin(self.w * t)
        tilt = self.tilt_amp * math.sin(0.5 * self.w * t)
        return pan, tilt


def run_lock_sim(
    gaa,
    omega=8.0,
    latency_ms=150.0,
    det_fps=15.0,
    noise_px=8.0,
    miss_rate=0.2,
    duration=6.0,
    loop_hz=60.0,
    max_rate=200.0,
    accel=650.0,
    ideal_arm=False,
    ppd_x=87.14,
    ppd_y=-89.73,
    frame_w=3840,
    frame_h=2160,
    # FOV ต้องสอดคล้องกับ ppd (fov = frame/ppd) — เดิมตั้ง 60/36 ตามสเปคเลนส์ ซึ่งขัดกับ ppd
    # ที่ตั้งไว้บรรทัดบน → sim คำนวณระยะคนละชุดกับที่ ppd บอก
    fov_h=3840 / 87.14,   # 44.07°
    fov_v=2160 / 89.73,   # 24.07°
    seed=12345,
    warmup=1.0,
    verbose=False,
    box_deg=0.30,          # ขนาดเชิงมุมโดรน (เล็ก = ไกล) → กำหนดระยะ + hit radius
    target_size_m=0.30,
    muzzle_ms=900.0,
    realistic_conf=True,   # conf ผันผวนตามความเร็ว + detection หลุดตาม conf (สมจริง)
    fire_enabled=True,     # จำลองการยิง: วัด miss distance ตอนกระสุนถึงเป้า
    fire_interval=0.4,
):
    """รัน closed-loop LOCK ด้วยโค้ดจริงจาก gaa. คืน dict สรุปผล."""
    rng = _LCG(seed)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    drone = DroneBearing(omega)
    arm = PhysicsArm(max_rate=max_rate, accel=accel, ideal=ideal_arm)
    # ระยะจากขนาดเชิงมุม: box_deg = ขนาดจริง/ระยะ → range = size / tan(box_deg)
    range_m = target_size_m / max(1e-6, math.tan(math.radians(box_deg)))
    t_flight = range_m / muzzle_ms
    # hit radius = effective radius เดียวกับ production (LOCK_FIRE_HIT_RADIUS_M) เพื่อให้ gate/HIT สอดคล้อง
    _hit_m = getattr(gaa, "LOCK_FIRE_HIT_RADIUS_M", 1.2)
    _min_r = getattr(gaa, "LOCK_FIRE_READY_MIN_RADIUS_DEG", 0.15)
    hit_radius_deg = max(_min_r, math.degrees(math.atan2(_hit_m, range_m)))

    # --- production objects ตัวจริง ---
    tracker = gaa._SimpleIoUTracker()
    _q, _r = gaa.lock_bearing_kalman_qr(ppd_x)
    lock_kalman = gaa._TargetKalman(q=_q, r=_r)
    pose_hist = gaa._ArmPoseHistory()
    drive_state = gaa._ArmDriveState()
    drive_state.continuous_target_time_prev = -0.05  # init (main() ทำแบบนี้เช่นกัน)

    latency = latency_ms / 1000.0
    det_period = 1.0 / det_fps if det_fps > 0 else 1e9
    dt_loop = 1.0 / loop_hz
    x_lim = arm._effective_x_limits
    y_lim = arm._effective_y_limits

    # bbox px จากขนาดเชิงมุม box_deg (โดรนเล็ก/ไกล = box_deg เล็ก)
    box_px = max(gaa.CSRT_MIN_BBOX + 2, int(box_deg * abs(ppd_x)))

    def project(t):
        """คืน (px, py) ของโดรนในภาพ ณ เวลา t (กล้องเห็นมุมแขนจริง) หรือ None ถ้านอก FOV."""
        d_pan, d_tilt = drone.bearing_at(t)
        rel_pan = d_pan - arm.actual_pan
        rel_tilt = d_tilt - arm.actual_tilt
        px = cx + rel_pan * ppd_x
        py = cy + rel_tilt * ppd_y  # ppd_y ลบ → เครื่องหมายถูกต้อง
        if abs(rel_pan) > fov_h / 2.0 or abs(rel_tilt) > fov_v / 2.0:
            return None, None
        return px, py

    # patch global clock ก่อน seed (lock_kalman._last_update_time ต้องอิง sim clock)
    real_time = time.time
    fake_clock = [0.0]
    time.time = lambda: fake_clock[0]  # gaa + calibrator ใช้ time module ร่วมกัน

    # --- acquire: ให้เป้าเริ่มใกล้ศูนย์ (แขนเล็งเข้าเป้าเริ่มต้นแล้ว) ---
    d_pan0, d_tilt0 = drone.bearing_at(0.0)
    arm.pos_x = arm._ax = d_pan0
    arm.pos_y = arm._ay = d_tilt0
    px0, py0 = project(0.0)
    tracker.init_from_bbox(int(px0 - box_px / 2), int(py0 - box_px / 2), box_px, box_px)
    # seed bearing kalman ด้วย measurement แรก (bearing = pose + offset)
    gaa._lock_feed_bearing_measurement(
        tracker, lock_kalman, pose_hist, arm, 0.0, cx, cy, ppd_x, ppd_y
    )

    ctx = {
        "arm_controller": arm, "lock_arm_drive_state": drive_state, "lock_kalman": lock_kalman,
        "x_lo": x_lim[0] + 2.0, "x_hi": x_lim[1] - 2.0, "y_lo": y_lim[0] + 2.0, "y_hi": y_lim[1] - 2.0,
        "cx_frame": cx, "cy_frame": cy, "px_per_deg_x": ppd_x, "px_per_deg_y": ppd_y,
        "lock_csrt_initialized": True, "lock_csrt_lost": False,
        "lock_last_arm_move_time": 0.0,
        # production ส่ง latency ที่วัดจริง → lead ตรง (harness ต้องส่งเท่ากันไม่งั้น over/under-lead)
        "avg_pipeline_latency": latency,
        "lock_range_m": range_m,
        "muzzle_velocity_ms": muzzle_ms,
    }

    pending = []          # (t_ready, t_capture, dets)
    errors = []           # angular error (deg) หลัง warmup
    px_errors = []        # pixel error จากศูนย์เล็ง (สิ่งที่กำหนดว่า "on target")
    on_target = 0
    n_samples = 0
    reticle_deg = 1.0     # เกณฑ์ on-target (~ reticle radius / ppd)
    fire_miss = []        # miss distance (deg) ตอนกระสุนถึงเป้า (ยิงทุก fire_interval)
    last_fire_t = warmup

    def _drone_speed(tt):
        a = drone.bearing_at(tt); b = drone.bearing_at(tt + 1e-3)
        return math.hypot(b[0] - a[0], b[1] - a[1]) / 1e-3

    def _angdist(p1, t1, p2, t2):
        return math.hypot(p1 - p2, t1 - t2)

    n_steps = int(duration / dt_loop)
    last_det_t = -1e9
    try:
        for i in range(n_steps):
            t = i * dt_loop
            fake_clock[0] = t
            # 1) physics แขนเดินหน้า
            arm.step(dt_loop)
            # 2) บันทึกท่าแขน (commanded) — เหมือน production ทำทุก loop
            pose_hist.append(t, arm.pos_x, arm.pos_y)
            # 3) capture detection ตาม det-fps (สมจริง: conf ผันผวนตามความเร็ว + หลุดตาม conf)
            if t - last_det_t >= det_period - 1e-9:
                last_det_t = t
                px, py = project(t)
                if realistic_conf:
                    spd = _drone_speed(t)
                    # โดรนเร็ว/เล็ก → conf ต่ำลง + สั่น; โดนบังบางเฟรม
                    conf = 0.9 - 0.012 * spd - 0.5 * max(0.0, (0.3 - box_deg)) + rng.gauss(0.07)
                    conf = max(0.05, min(0.95, conf))
                    miss_p = miss_rate + 0.35 * (1.0 - conf)  # conf ต่ำ = หลุดบ่อย
                else:
                    conf = 0.85
                    miss_p = miss_rate
                dropped = rng.rand() < miss_p
                if px is not None and not dropped:
                    npx = px + rng.gauss(noise_px)
                    npy = py + rng.gauss(noise_px)
                    det = (int(npx - box_px / 2), int(npy - box_px / 2), box_px, box_px, conf, 0, "drone")
                    pending.append((t + latency, t, [det]))
                else:
                    pending.append((t + latency, t, []))
            # 4) deliver detections ที่ถึงเวลา (t_ready <= now) — เรียงตามเวลา
            ready = [p for p in pending if p[0] <= t + 1e-9]
            pending = [p for p in pending if p[0] > t + 1e-9]
            ready.sort(key=lambda p: p[0])
            for t_ready, t_cap, dets in ready:
                matched, _ = tracker.update(dets, frame_w, frame_h)
                if matched:
                    gaa._lock_feed_bearing_measurement(
                        tracker, lock_kalman, pose_hist, arm, t_cap, cx, cy, ppd_x, ppd_y
                    )
            ctx["lock_csrt_lost"] = tracker.lost
            # 5) ขับแขนด้วย _tick_lock ตัวจริง
            gaa._tick_lock(ctx)
            # 6) วัด error เชิงมุม + pixel (กล้องเห็นมุมจริง)
            if t >= warmup:
                d_pan, d_tilt = drone.bearing_at(t)
                e_deg = math.hypot(d_pan - arm.actual_pan, d_tilt - arm.actual_tilt)
                errors.append(e_deg)
                ppx, ppy = project(t)
                if ppx is not None:
                    e_px = math.hypot(ppx - cx, ppy - cy)
                    px_errors.append(e_px)
                n_samples += 1
                if e_deg <= reticle_deg:
                    on_target += 1
                # 7) จำลองการยิง 'ผ่าน readiness gate' (เหมือน production: ยิงเฉพาะตอนโปรแกรมมั่นใจ)
                #    miss = ระยะเชิงมุมระหว่างทิศลำกล้อง กับ 'ตำแหน่งจริงของโดรน ณ เวลากระสุนถึง'
                _ready = ctx.get("lock_fire_ready", False)
                if fire_enabled and _ready and (t - last_fire_t) >= fire_interval:
                    last_fire_t = t
                    imp = drone.bearing_at(t + t_flight)  # โดรนอยู่ไหนตอนกระสุนถึง
                    miss = _angdist(arm.actual_pan, arm.actual_tilt, imp[0], imp[1])
                    fire_miss.append(miss)
    finally:
        time.time = real_time

    errors.sort()
    def pct(a, p):
        if not a:
            return float("nan")
        k = min(len(a) - 1, int(p / 100.0 * len(a)))
        return a[k]
    fire_miss.sort()
    fm_med = fire_miss[len(fire_miss) // 2] if fire_miss else float("nan")
    hits = sum(1 for m in fire_miss if m <= hit_radius_deg)
    return {
        "omega": omega, "latency_ms": latency_ms, "det_fps": det_fps,
        "noise_px": noise_px, "miss_rate": miss_rate, "ideal_arm": ideal_arm,
        "box_deg": box_deg, "range_m": range_m, "t_flight_ms": t_flight * 1000.0,
        "median_deg": pct(errors, 50), "p90_deg": pct(errors, 90),
        "max_deg": max(errors) if errors else float("nan"),
        "on_target_pct": 100.0 * on_target / n_samples if n_samples else float("nan"),
        "mean_px": (sum(px_errors) / len(px_errors)) if px_errors else float("nan"),
        "fire_miss_med_deg": fm_med, "hit_radius_deg": hit_radius_deg,
        "hit_pct": 100.0 * hits / len(fire_miss) if fire_miss else float("nan"),
        "n_fires": len(fire_miss), "n": n_samples,
    }


# =========================================================================
# self-test: ยืนยัน P0 fix เชิงพฤติกรรม (สำหรับ CI / รันเร็ว)
# =========================================================================
def selftest(gaa):
    fails = []
    def check(name, cond, detail=""):
        print(f"{'PASS' if cond else 'FAIL'}: {name} {detail}")
        if not cond:
            fails.append(name)

    # 1) เป้าช้า (ω=5) + latency ปานกลาง + แขน ideal → ต้องเกาะเป้าได้ดี
    r = run_lock_sim(gaa, omega=5.0, latency_ms=100.0, det_fps=15.0, miss_rate=0.1,
                     ideal_arm=True, duration=6.0)
    check("slow target tracked (median <= 1.0 deg)", r["median_deg"] <= 1.0,
          f"median={r['median_deg']:.3f} on={r['on_target_pct']:.0f}%")

    # 2) ego-comp: เป้าปานกลางต้อง "เกาะแบบ bounded" ไม่ diverge (แยกจากเป้าเร็วที่ fail ชัด)
    #    median ~2° ที่ ω10/100ms เป็นพฤติกรรมถูกต้อง — ยังไม่มี lead compensation (P1)
    #    ที่จะลด staleness = latency (measurement timestamp ณ deliver ไม่ใช่ t_capture)
    _ego_ms = [run_lock_sim(gaa, omega=10.0, latency_ms=100.0, det_fps=15.0, miss_rate=0.15,
                            ideal_arm=True, duration=8.0, seed=s, fire_enabled=False)["median_deg"]
               for s in (111, 222, 333)]
    _ego_med = sum(_ego_ms) / len(_ego_ms)
    check("ego-comp holds moderate target bounded (3-seed median < 2.5 deg)", _ego_med < 2.5,
          f"median={_ego_med:.3f}")

    # 3) เป้าเร็วมาก (ω=25) ต้องแย่ลงชัด — ยืนยันว่า sim ไม่ได้ 'ง่ายเกินจริง'
    r_fast = run_lock_sim(gaa, omega=25.0, latency_ms=150.0, det_fps=15.0, miss_rate=0.2,
                          ideal_arm=False, duration=6.0)
    check("fast target degrades (median > slow target)", r_fast["median_deg"] > r["median_deg"],
          f"fast={r_fast['median_deg']:.3f} vs slow={r['median_deg']:.3f}")

    # 4) det-fps สูงต้องดีกว่า det-fps ต่ำ (ยืนยันคอขวด detection ตามผลทดลอง 30)
    r_lo = run_lock_sim(gaa, omega=12.0, latency_ms=100.0, det_fps=5.0, miss_rate=0.2,
                        ideal_arm=True, duration=6.0)
    r_hi = run_lock_sim(gaa, omega=12.0, latency_ms=100.0, det_fps=30.0, miss_rate=0.2,
                        ideal_arm=True, duration=6.0)
    check("higher det-fps improves tracking", r_hi["median_deg"] < r_lo["median_deg"],
          f"fps30={r_hi['median_deg']:.3f} < fps5={r_lo['median_deg']:.3f}")

    # 5) mechanical lag: แขนจริง (มี accel/rate limit) แย่กว่า ideal เล็กน้อย
    r_ideal = run_lock_sim(gaa, omega=10.0, latency_ms=120.0, det_fps=15.0, miss_rate=0.15,
                           ideal_arm=True, duration=6.0)
    r_real = run_lock_sim(gaa, omega=10.0, latency_ms=120.0, det_fps=15.0, miss_rate=0.15,
                          ideal_arm=False, duration=6.0)
    check("real arm >= ideal error (mechanical lag present)",
          r_real["median_deg"] >= r_ideal["median_deg"] - 1e-6,
          f"real={r_real['median_deg']:.3f} ideal={r_ideal['median_deg']:.3f}")

    # 6) lead tracking ช่วยลด lag เทียบปิด lead (เฉลี่ยหลาย seed — ผลเดี่ยวแกว่ง)
    def _avg_med(lead, om=8.0, seeds=(111, 222, 333)):
        gaa.LOCK_LEAD_ENABLED = lead
        ms = [run_lock_sim(gaa, omega=om, latency_ms=150.0, det_fps=15.0, miss_rate=0.15,
                           ideal_arm=False, duration=8.0, seed=s, fire_enabled=False)["median_deg"]
              for s in seeds]
        return sum(ms) / len(ms)
    _save = gaa.LOCK_LEAD_ENABLED
    m_nolead = _avg_med(False)
    m_lead = _avg_med(True)
    gaa.LOCK_LEAD_ENABLED = _save
    check("lead tracking reduces error vs no-lead (3-seed avg)", m_lead < m_nolead,
          f"lead={m_lead:.3f} < nolead={m_nolead:.3f}")

    # 7) ระบบยิงโดนสูงเมื่อเงื่อนไขเอื้อ (เป้าช้า/ใกล้/แสงดี) — พิสูจน์ว่า 'ยิงโดนได้จริง'
    #    (หมายเหตุวิจัย: กระสุน 900m/s → firing lead <0.2° แทบไม่มีผล การโดน = ความแม่นการ track)
    r_fav = run_lock_sim(gaa, omega=3.0, latency_ms=100.0, det_fps=25.0, miss_rate=0.05,
                         ideal_arm=False, box_deg=0.8, duration=10.0)
    check("hits reliably in favorable regime (slow/near, hit% >= 60)",
          r_fav["hit_pct"] >= 60,
          f"hit={r_fav['hit_pct']:.0f}% (fire_miss {r_fav['fire_miss_med_deg']:.2f}° "
          f"vs hit_r {r_fav['hit_radius_deg']:.2f}°, trk {r_fav['median_deg']:.2f}°)")

    print()
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


def run_firetest(gaa):
    """ตารางเทียบ hit-rate: firing solution ON vs OFF ที่ความเร็ว/ระยะต่าง ๆ (พิสูจน์ 'กดยิงแล้วร่วง')."""
    print("Firing-solution validation (โดรนเล็ก, det สมจริง conf ผันผวน+หลุด, แขนจริง)")
    print(f"{'omega':>6} {'box°':>5} {'range':>6} {'tof':>6} | {'OFF hit%':>8} {'OFF miss°':>9} | {'ON hit%':>8} {'ON miss°':>9}")
    for om, box in [(5, 0.6), (8, 0.5), (12, 0.4), (15, 0.3), (20, 0.3)]:
        gaa.LOCK_FIRE_SOLUTION_ENABLED = False
        off = run_lock_sim(gaa, omega=om, latency_ms=120.0, det_fps=20.0, miss_rate=0.1,
                           ideal_arm=False, box_deg=box, duration=8.0)
        gaa.LOCK_FIRE_SOLUTION_ENABLED = True
        on = run_lock_sim(gaa, omega=om, latency_ms=120.0, det_fps=20.0, miss_rate=0.1,
                          ideal_arm=False, box_deg=box, duration=8.0)
        print(f"{om:>6} {box:>5.1f} {on['range_m']:>5.0f}m {on['t_flight_ms']:>5.0f}ms | "
              f"{off['hit_pct']:>7.0f}% {off['fire_miss_med_deg']:>8.2f}° | "
              f"{on['hit_pct']:>7.0f}% {on['fire_miss_med_deg']:>8.2f}°")
    print("\nสรุป: ON ควรให้ hit% สูงกว่าและ miss° ต่ำกว่าชัด (เล็งดักหน้าเวลากระสุนบิน)")
    return 0


def run_campaign(gaa, out_csv="lock_campaign.csv"):
    """Campaign ครอบคลุม: ขนาดโดรน × ความเร็ว × สภาพแสง × latency → lock+ยิง เก็บผลทุกเงื่อนไข.
    สภาพแสง จำลองผ่านคุณภาพ detection: day=conf สูง/miss ต่ำ, dusk=กลาง, night=conf ต่ำ/miss สูง.
    ผล: tracking error + hit% + fire miss ต่อเงื่อนไข → ใช้หาขอบเขตที่ 'lock+ยิงโดน' ได้จริง."""
    # (ชื่อแสง, miss_rate, noise_px, det_fps) — night ใช้ thermal det ช้า/หลุดบ่อย
    lights = [("day", 0.06, 5.0, 20.0), ("dusk", 0.15, 9.0, 15.0), ("night", 0.35, 14.0, 10.0)]
    boxes = [("large@near", 1.2), ("med", 0.6), ("small", 0.3), ("tiny@far", 0.15)]
    omegas = [3, 6, 10, 15, 25]
    latency_ms = 120.0
    rows = []
    print(f"{'light':>6} {'size':>11} {'ω':>4} {'range':>6} | {'trk_med°':>8} {'on%':>5} | "
          f"{'hit%':>5} {'miss°':>6} {'tof':>6}")
    for lname, miss, noise, dfps in lights:
        for sname, box in boxes:
            for om in omegas:
                # เฉลี่ย 2 seed ต่อเงื่อนไข (ลด variance)
                accs = []
                for s in (101, 202):
                    r = run_lock_sim(gaa, omega=om, latency_ms=latency_ms, det_fps=dfps,
                                     noise_px=noise, miss_rate=miss, ideal_arm=False,
                                     box_deg=box, duration=8.0, seed=s)
                    accs.append(r)
                def avg(k):
                    vs = [a[k] for a in accs if a[k] == a[k]]
                    return sum(vs) / len(vs) if vs else float("nan")
                row = {
                    "light": lname, "size": sname, "box_deg": box, "omega": om,
                    "latency_ms": latency_ms, "det_fps": dfps, "miss_rate": miss, "noise_px": noise,
                    "range_m": accs[0]["range_m"], "t_flight_ms": accs[0]["t_flight_ms"],
                    "trk_median_deg": avg("median_deg"), "on_target_pct": avg("on_target_pct"),
                    "hit_pct": avg("hit_pct"), "fire_miss_med_deg": avg("fire_miss_med_deg"),
                    "hit_radius_deg": accs[0]["hit_radius_deg"],
                }
                rows.append(row)
                print(f"{lname:>6} {sname:>11} {om:>4} {row['range_m']:>5.0f}m | "
                      f"{row['trk_median_deg']:>8.2f} {row['on_target_pct']:>4.0f}% | "
                      f"{row['hit_pct']:>4.0f}% {row['fire_miss_med_deg']:>5.2f}° {row['t_flight_ms']:>4.0f}ms")
    cols = ["light", "size", "box_deg", "omega", "latency_ms", "det_fps", "miss_rate", "noise_px",
            "range_m", "t_flight_ms", "hit_radius_deg", "trk_median_deg", "on_target_pct",
            "hit_pct", "fire_miss_med_deg"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    # สรุป: เงื่อนไขที่ยิงโดน ≥70%
    good = [r for r in rows if r["hit_pct"] >= 70]
    print(f"\nเขียนผล → {out_csv} ({len(rows)} เงื่อนไข)")
    print(f"เงื่อนไขที่ 'ยิงโดน ≥70%': {len(good)}/{len(rows)}")
    if good:
        mo = max(r["omega"] for r in good)
        print(f"  ยิงโดนสูงสุดที่ ω ถึง {mo}°/s; ส่วนใหญ่เป็นเป้าใหญ่/ใกล้ + แสงดี")
    print("ข้อค้นพบ: hit% ถูกจำกัดด้วย tracking error เทียบขนาดเชิงมุมเป้า (กระสุนเร็ว lead เล็ก)")
    return 0


def run_sweep(gaa, out_csv="lock_sim_sweep.csv"):
    omegas = [5, 8, 10, 12, 15, 20, 25, 30]
    latencies = [50, 100, 150, 200, 300]
    rows = []
    print(f"{'omega':>6} {'lat_ms':>7} {'median':>8} {'p90':>7} {'on%':>6} {'mean_px':>8}")
    for om in omegas:
        for lat in latencies:
            r = run_lock_sim(gaa, omega=om, latency_ms=lat, det_fps=15.0, noise_px=8.0,
                             miss_rate=0.2, ideal_arm=False, duration=6.0)
            rows.append(r)
            print(f"{om:>6} {lat:>7} {r['median_deg']:>8.3f} {r['p90_deg']:>7.3f} "
                  f"{r['on_target_pct']:>6.1f} {r['mean_px']:>8.1f}")
    with open(out_csv, "w") as f:
        f.write("omega,latency_ms,det_fps,noise_px,miss_rate,median_deg,p90_deg,max_deg,on_target_pct,mean_px\n")
        for r in rows:
            f.write(f"{r['omega']},{r['latency_ms']},{r['det_fps']},{r['noise_px']},{r['miss_rate']},"
                    f"{r['median_deg']:.4f},{r['p90_deg']:.4f},{r['max_deg']:.4f},"
                    f"{r['on_target_pct']:.2f},{r['mean_px']:.2f}\n")
    print(f"\nเขียนผล → {out_csv}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="รัน assertion P0 (CI)")
    ap.add_argument("--firetest", action="store_true", help="เทียบ hit-rate firing solution ON/OFF")
    ap.add_argument("--campaign", action="store_true", help="ครอบคลุม ขนาด×ความเร็ว×แสง×latency → CSV")
    ap.add_argument("--sweep", action="store_true", help="ตาราง omega × latency → CSV")
    ap.add_argument("--omega", type=float, default=8.0, help="peak angular rate ของโดรน (deg/s)")
    ap.add_argument("--latency-ms", type=float, default=150.0)
    ap.add_argument("--det-fps", type=float, default=15.0)
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--ideal-arm", action="store_true", help="ปิด mechanical lag (actual=commanded)")
    args = ap.parse_args()

    print("Loading production module 22_gun_aim_assist_vector.py ...", flush=True)
    gaa = _load_production_module()
    gaa.LOCK_TRACK_DEBUG = False  # ปิด debug print ของ production ระหว่าง sim

    if args.selftest:
        return selftest(gaa)
    if args.firetest:
        return run_firetest(gaa)
    if args.campaign:
        return run_campaign(gaa)
    if args.sweep:
        return run_sweep(gaa)

    r = run_lock_sim(
        gaa, omega=args.omega, latency_ms=args.latency_ms, det_fps=args.det_fps,
        noise_px=args.noise_px, miss_rate=args.miss_rate, duration=args.duration,
        ideal_arm=args.ideal_arm,
    )
    print("\n--- ผลลัพธ์ ---")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:>16}: {v:.3f}")
        else:
            print(f"  {k:>16}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
