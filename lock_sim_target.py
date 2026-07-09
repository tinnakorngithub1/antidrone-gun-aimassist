#!/usr/bin/env python3
"""
lock_sim_target.py — โดรนเสมือน (virtual target) สำหรับทดสอบโหมด LOCK กับแขนจริง+กล้องจริง
โดยยังไม่ต้องบินโดรนจริง

หลักการ (สำคัญ — กล้องติดบนแขน):
  โดรนมีตำแหน่งใน "โลกจริง" เป็นมุม bearing (pan/tilt). ตำแหน่ง pixel บนภาพคำนวณจาก
      px = cx + (bearing_pan  − มุมแขน_pan ) · ppd_x
      py = cy + (bearing_tilt − มุมแขน_tilt) · ppd_y
  เมื่อแขนหมุนตามเป้า (มุมแขน → bearing) โดรนจะเลื่อนเข้าศูนย์กลางเหมือนเป้าจริง
  และเพราะใช้ "มุมแขนจริง" ภาพพื้นหลังจริงก็เลื่อนไปทางเดียวกัน → ego-motion สมจริง

การใช้งาน: เปิดใน config.py (LOCK_SIM_TARGET_ENABLED=True) แล้วรัน 22_gun_aim_assist_vector.py
ตามปกติ → เข้าโหมด LOCK (ปุ่ม 5 / L) แขนจะติดตามโดรนเสมือนบนแขนจริง

ปุ่มควบคุมโดรนเสมือน (ในหน้าต่างโปรแกรม):
  i/j/k/l : เลื่อนโดรน ขึ้น/ซ้าย/ลง/ขวา (โหมด manual)
  0       : pattern hover (ลอยนิ่ง + ดริฟต์เบา)
  8       : pattern sine (กวาด pan)
  9       : pattern figure8
  (ปุ่มเหล่านี้ต่อจาก main loop ผ่าน handle_key)
"""
import math
from collections import deque

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


class VirtualDroneTarget:
    def __init__(
        self,
        ppd_x,
        ppd_y,
        fov_h=60.0,
        fov_v=36.0,
        pattern="realistic",
        omega_deg_s=8.0,
        amp_deg=15.0,
        tilt_amp_deg=5.0,
        box_deg=0.6,
        miss_rate=0.15,
        cls_id=0,
        det_conf=0.85,
        manual_rate_deg_s=25.0,
        seed=2027,
        range_m=150.0,          # ระยะโดรน (m) — กำหนดขนาดเชิงมุม + ความเร็วเชิงมุม
        target_size_m=0.30,     # ขนาดจริงโดรน (m)
        max_speed_ms=18.0,      # ความเร็วโดรนสูงสุด (m/s) — บินจริง
        max_accel_ms2=12.0,     # อัตราเร่งโดรน (m/s²)
    ):
        self.ppd_x = float(ppd_x)
        self.ppd_y = float(ppd_y)
        self.fov_h = float(fov_h)
        self.fov_v = float(fov_v)
        self.pattern = pattern
        self.omega = float(omega_deg_s)
        self.amp = float(amp_deg)
        self.tilt_amp = float(tilt_amp_deg)
        self.miss_rate = float(miss_rate)
        self.cls_id = int(cls_id)
        self.det_conf = float(det_conf)
        self.manual_rate = float(manual_rate_deg_s)
        # ระยะ → ขนาดเชิงมุม (box_deg) และความเร็วเชิงมุมสูงสุด (linear/range)
        self.range_m = float(range_m)
        self.target_size_m = float(target_size_m)
        self.max_speed_ms = float(max_speed_ms)
        self.max_accel_ms2 = float(max_accel_ms2)
        # ขนาดเชิงมุมจากระยะ (โดรนไกล = เล็ก); ถ้าไม่ให้ range ใช้ box_deg ที่ส่งมา
        self.box_deg = math.degrees(math.atan2(self.target_size_m, self.range_m)) if range_m else float(box_deg)
        # ความเร็ว/เร่ง เชิงมุม (deg/s, deg/s²) = linear / range
        self.max_ang_speed = math.degrees(self.max_speed_ms / max(1.0, self.range_m))
        self.max_ang_accel = math.degrees(self.max_accel_ms2 / max(1.0, self.range_m))
        # โดรนไกล = ภาพเล็ก = YOLO conf ต่ำ (สมจริง) — ระยะกำหนด conf ของ detection ที่ฉีด
        if range_m:
            self.det_conf = self._conf_from_box(self.box_deg)

        # bearing state (deg) ในโลกจริง
        self.pan = 0.0
        self.tilt = 0.0
        self._vpan = 0.0        # ความเร็วเชิงมุม (deg/s) สำหรับ realistic flight
        self._vtilt = 0.0
        self._wp = None         # waypoint (pan,tilt) เป้าหมายบินไป
        self._phase = "cruise"  # cruise | hover | dash
        self._phase_until = 0.0
        self._t0 = None
        self._t_prev = None
        # manual keyboard velocity target (เลื่อนแบบ smooth)
        self._kb_dir_pan = 0.0
        self._kb_dir_tilt = 0.0
        self._kb_hold_until = {}   # key -> release-timestamp (จำลองการกดค้างสั้น ๆ)

        # explosion state (โดรนกระจุยเมื่อโดน)
        self.exploding = False
        self._explode_t0 = 0.0
        self._debris = []       # [(px,py,vx,vy)] เศษซาก
        self.destroyed_count = 0

        # capture buffer: (t_capture, px, py, in_fov) — ใช้ lookup ตอน YOLO result กลับมา
        self._cap_buf = deque(maxlen=300)
        # deterministic RNG (การบินอิสระ — ตัวเล็ง/ยิงไม่เห็น state นี้ ห้ามโกง)
        self._rng_s = seed & 0x7FFFFFFF
        self._flight_rng = (seed * 2654435761) & 0x7FFFFFFF  # แยก RNG การบินออกจาก detection

    @staticmethod
    def _conf_from_box(box_deg):
        """conf ของ detection ตามขนาดเชิงมุม: ใกล้(ใหญ่)=สูง, ไกล(เล็ก)=ต่ำ (สมจริง).
        อ้างอิง: 50m≈0.34°→0.76, 100m≈0.17°→0.46, 150m≈0.11°→0.35, 200m≈0.086°→0.30."""
        return max(0.15, min(0.85, 0.15 + box_deg * 1.8))

    def set_range(self, range_m):
        """เปลี่ยนระยะโดรน runtime → คำนวณ ขนาด/ความเร็วเชิงมุม/conf ใหม่ (เทสต์หลายระยะ)."""
        self.range_m = float(max(1.0, range_m))
        self.box_deg = math.degrees(math.atan2(self.target_size_m, self.range_m))
        self.max_ang_speed = math.degrees(self.max_speed_ms / self.range_m)
        self.max_ang_accel = math.degrees(self.max_accel_ms2 / self.range_m)
        self.det_conf = self._conf_from_box(self.box_deg)

    # ---- RNG ----
    def _rand(self):
        self._rng_s = (1103515245 * self._rng_s + 12345) & 0x7FFFFFFF
        return self._rng_s / 0x7FFFFFFF

    def _frand(self):
        """RNG แยกสำหรับการบิน — อิสระจาก detection (ห้ามโกง)."""
        self._flight_rng = (1103515245 * self._flight_rng + 12345) & 0x7FFFFFFF
        return self._flight_rng / 0x7FFFFFFF

    def _new_waypoint(self, now):
        """สุ่ม waypoint + phase ใหม่ (บินอิสระ ตัวเล็งไม่รู้ล่วงหน้า)."""
        rp = self.fov_h * 0.38
        rt = self.fov_v * 0.34
        self._wp = ((self._frand() * 2 - 1) * rp, (self._frand() * 2 - 1) * rt)
        r = self._frand()
        if r < 0.30:
            self._phase = "hover"; self._phase_until = now + 0.6 + self._frand() * 1.4
        elif r < 0.55:
            self._phase = "dash"; self._phase_until = now + 0.4 + self._frand() * 0.6
        else:
            self._phase = "cruise"; self._phase_until = now + 1.0 + self._frand() * 1.8

    # ---- trajectory ----
    def update_bearing(self, now):
        if self._t0 is None:
            self._t0 = now
            self._t_prev = now
        t = now - self._t0
        dt = max(0.0, min(0.1, now - (self._t_prev or now)))
        self._t_prev = now

        if self.pattern == "realistic":
            # บินอิสระ: ไล่ waypoint ด้วยความเร่ง/ความเร็วจำกัด (linear/range) + hover/dash/turn
            if self._wp is None or now >= self._phase_until:
                self._new_waypoint(now)
            spd_cap = self.max_ang_speed * (1.6 if self._phase == "dash" else
                                            0.15 if self._phase == "hover" else 1.0)
            # เร่งเข้าหา waypoint (steering) ด้วย accel จำกัด
            ex = self._wp[0] - self.pan
            ey = self._wp[1] - self.tilt
            dist = math.hypot(ex, ey)
            if dist > 1e-6:
                dvx = (ex / dist) * self.max_ang_accel * dt
                dvy = (ey / dist) * self.max_ang_accel * dt
            else:
                dvx = dvy = 0.0
            # เพิ่ม jitter เล็กน้อย (ลม/การควบคุมโดรนจริง)
            dvx += (self._frand() * 2 - 1) * self.max_ang_accel * dt * 0.3
            dvy += (self._frand() * 2 - 1) * self.max_ang_accel * dt * 0.3
            self._vpan += dvx
            self._vtilt += dvy
            # cap ความเร็ว
            sp = math.hypot(self._vpan, self._vtilt)
            if sp > spd_cap and sp > 1e-6:
                self._vpan *= spd_cap / sp
                self._vtilt *= spd_cap / sp
            self.pan += self._vpan * dt
            self.tilt += self._vtilt * dt
            # ถึง waypoint แล้ว → เลือกใหม่ (หักเลี้ยว)
            if dist < 0.3 and self._phase != "hover":
                self._new_waypoint(now)
        elif self.pattern == "manual":
            # เลื่อนตามทิศที่กดค้าง (ปล่อยแล้วหยุด)
            self._expire_keys(now)
            self.pan += self._kb_dir_pan * self.manual_rate * dt
            self.tilt += self._kb_dir_tilt * self.manual_rate * dt
        elif self.pattern == "hover":
            # โฉบนิ่ง + ดริฟต์ช้ามาก (peak rate ~0.7°/s) — เคสง่ายสุดสำหรับ bring-up
            self.pan = 2.0 * math.sin(0.3 * t) + 0.6 * math.sin(0.9 * t)
            self.tilt = 1.0 * math.sin(0.25 * t)
        elif self.pattern == "figure8":
            w = (self.omega / max(1e-6, self.amp))
            self.pan = self.amp * math.sin(w * t)
            self.tilt = self.tilt_amp * math.sin(2.0 * w * t)
        else:  # sine (default): กวาด pan, tilt แกว่งช้า — peak rate ≈ omega
            w = (self.omega / max(1e-6, self.amp))
            self.pan = self.amp * math.sin(w * t)
            self.tilt = self.tilt_amp * math.sin(0.5 * w * t)

        # clip อยู่ในช่วงที่ยังอยู่ใน FOV รอบ ๆ ได้ (กันโดรนหลุดไปไกลถาวร)
        self.pan = max(-40.0, min(40.0, self.pan))
        self.tilt = max(-25.0, min(25.0, self.tilt))

    def _expire_keys(self, now):
        expired = [k for k, tt in self._kb_hold_until.items() if now >= tt]
        for k in expired:
            del self._kb_hold_until[k]
        # ทิศ = รวมปุ่มที่ยังกดค้าง
        self._kb_dir_pan = 0.0
        self._kb_dir_tilt = 0.0
        for k in self._kb_hold_until:
            if k == "l":
                self._kb_dir_pan += 1.0
            elif k == "j":
                self._kb_dir_pan -= 1.0
            elif k == "k":
                self._kb_dir_tilt += 1.0   # ลง (tilt+): py = cy + tilt·ppd_y, ppd_y<0 → ลง
            elif k == "i":
                self._kb_dir_tilt -= 1.0

    # ---- projection (ใช้มุมแขนจริง) ----
    def project(self, arm_pan, arm_tilt, cx, cy):
        rel_pan = self.pan - float(arm_pan)
        rel_tilt = self.tilt - float(arm_tilt)
        px = cx + rel_pan * self.ppd_x
        py = cy + rel_tilt * self.ppd_y
        in_fov = (abs(rel_pan) <= self.fov_h / 2.0) and (abs(rel_tilt) <= self.fov_v / 2.0)
        return px, py, in_fov

    # ---- drawing ----
    def draw(self, frame, px, py, in_fov, show_label=True):
        if cv2 is None or not in_fov:
            return
        h, w = frame.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            return
        # ขนาดตาม box_deg (เส้นผ่านศูนย์กลางโดรน)
        r = max(6, int(0.5 * self.box_deg * abs(self.ppd_x)))
        body = (35, 35, 40)      # ตัวโดรนสีเข้ม (BGR)
        arm_c = (25, 25, 30)
        rotor = (60, 60, 70)
        # แขน 4 ทิศ (X-config)
        d = int(r * 1.15)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            ex, ey = ix + sx * d, iy + sy * d
            cv2.line(frame, (ix, iy), (ex, ey), arm_c, max(2, r // 6), cv2.LINE_AA)
            # rotor disc (จาง ๆ จำลอง motion blur)
            cv2.circle(frame, (ex, ey), max(4, r // 2), rotor, -1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), max(4, r // 2), (90, 90, 100), 1, cv2.LINE_AA)
        # ตัวกลาง
        cv2.circle(frame, (ix, iy), max(4, int(r * 0.55)), body, -1, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy), max(4, int(r * 0.55)), (80, 80, 90), 1, cv2.LINE_AA)
        if show_label:
            cv2.putText(frame, "SIM", (ix - r, iy - d - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, r / 60.0), (0, 200, 255),
                        max(1, r // 20), cv2.LINE_AA)

    # ---- detection synthesis ----
    def make_detection(self, px, py, conf=None):
        bw = max(6, int(self.box_deg * abs(self.ppd_x)))
        bh = max(6, int(self.box_deg * abs(self.ppd_y)))
        x = int(round(px - bw / 2.0))
        y = int(round(py - bh / 2.0))
        c = self.det_conf if conf is None else float(conf)
        # 7-tuple ตรง format production: (x,y,w,h,conf,cls_id,cls_name)
        return (x, y, bw, bh, c, self.cls_id, "drone")

    # ---- explosion (โดรนกระจุยเมื่อโดน) ----
    EXPLODE_DUR = 0.9   # วินาที แสดง debris ก่อน respawn

    def explode(self, now):
        """เรียกเมื่อโดนจริง — โดรนระเบิด แสดง debris แล้ว respawn (บินใหม่)."""
        if self.exploding:
            return
        self.exploding = True
        self._explode_t0 = now
        self.destroyed_count += 1
        # เศษซากกระจาย (พิกัด pixel-relative + ความเร็ว)
        self._debris = []
        for _ in range(22):
            ang = self._rand() * 2 * math.pi
            spd = 120.0 + self._rand() * 480.0   # px/s
            self._debris.append([0.0, 0.0, math.cos(ang) * spd, math.sin(ang) * spd,
                                 3.0 + self._rand() * 5.0])

    def _respawn(self, now):
        self.exploding = False
        self._debris = []
        # เกิดใหม่ที่ bearing สุ่ม + flight ใหม่ (อิสระ)
        self.pan = (self._frand() * 2 - 1) * self.fov_h * 0.35
        self.tilt = (self._frand() * 2 - 1) * self.fov_v * 0.3
        self._vpan = self._vtilt = 0.0
        self._wp = None
        self._t0 = None

    def _draw_explosion(self, frame, px, py, now):
        if cv2 is None:
            return
        h, w = frame.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        el = now - self._explode_t0
        frac = min(1.0, el / self.EXPLODE_DUR)
        # แฟลชวาบตอนแรก
        if frac < 0.25:
            fr = int(max(8, self.box_deg * abs(self.ppd_x) * 3) * (1 + frac * 4))
            cv2.circle(frame, (ix, iy), fr, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), int(fr * 0.6), (255, 255, 255), -1, cv2.LINE_AA)
        # เศษซากกระเด็น (จางลงตามเวลา)
        for d in self._debris:
            dx = ix + int(d[2] * el)
            dy = iy + int(d[3] * el + 0.5 * 300.0 * el * el)  # เศษตกด้วย 'แรงโน้มถ่วง' จอ
            if 0 <= dx < w and 0 <= dy < h:
                _sz = max(1, int(d[4] * (1 - frac)))
                cv2.circle(frame, (dx, dy), _sz, (40, 40, 60), -1, cv2.LINE_AA)
        # ควันจาง
        if 0 <= ix < w and 0 <= iy < h:
            _sr = int(max(6, self.box_deg * abs(self.ppd_x) * 2) * (1 + frac * 3))
            cv2.circle(frame, (ix, iy), _sr, (60, 60, 60), 2, cv2.LINE_AA)
        # ป้าย DESTROYED
        cv2.putText(frame, "DESTROYED", (ix - 40, iy - _sr - 8),
                    cv2.FONT_HERSHEY_DUPLEX, max(0.5, abs(self.ppd_x) / 200.0),
                    (0, 0, 255), 2, cv2.LINE_AA)

    # ---- per-frame hook (เรียกตอนอ่านเฟรม ก่อนส่งเข้า YOLO) ----
    def on_frame(self, frame, arm_pan, arm_tilt, cx, cy, now, draw=True):
        """อัปเดต bearing, project ด้วยมุมแขนจริง, วาดสไปรต์/ระเบิด, เก็บ capture buffer.
        คืน (px, py, in_fov). ระหว่างระเบิด: ไม่ผลิต detection (โดรนหาย) แล้ว respawn."""
        if self.exploding:
            px, py, in_fov = self.project(arm_pan, arm_tilt, cx, cy)
            if draw:
                self._draw_explosion(frame, px, py, now)
            if now - self._explode_t0 >= self.EXPLODE_DUR:
                self._respawn(now)
            self._cap_buf.append((now, px, py, False))  # in_fov=False → ไม่มี detection
            return px, py, False
        self.update_bearing(now)
        px, py, in_fov = self.project(arm_pan, arm_tilt, cx, cy)
        if draw:
            self.draw(frame, px, py, in_fov)
        self._cap_buf.append((now, px, py, in_fov))
        return px, py, in_fov

    def detection_at_capture(self, t_capture):
        """คืน detection ที่ตำแหน่งโดรน ณ เฟรมที่ YOLO ประมวลผล (t_capture) หรือ None
        ถ้านอก FOV / โดน miss / กำลังระเบิด. lookup จาก capture buffer เพื่อให้ตำแหน่งตรงภาพจริง."""
        if self.exploding or not self._cap_buf:
            return None
        best = min(self._cap_buf, key=lambda e: abs(e[0] - t_capture))
        _t, px, py, in_fov = best
        if not in_fov:
            return None
        if self._rand() < self.miss_rate:
            return None  # จำลอง YOLO miss
        return self.make_detection(px, py)

    # ---- keyboard ----
    def handle_key(self, key):
        """รับ key จาก cv2.waitKey. คืน True ถ้า consume แล้ว."""
        if key < 0:
            return False
        try:
            ch = chr(key & 0xFF).lower()
        except Exception:
            return False
        import time as _t
        if ch in ("i", "j", "k", "l"):
            self.pattern = "manual"
            # กดค้าง ~0.25s ต่อการกด (กดรัว = เลื่อนต่อเนื่อง)
            self._kb_hold_until[ch] = _t.time() + 0.25
            return True
        if ch == "0":
            self.pattern = "hover"
            self._t0 = None
            return True
        if ch == "8":
            self.pattern = "sine"
            self._t0 = None
            return True
        if ch == "9":
            self.pattern = "figure8"
            self._t0 = None
            return True
        return False

    # ---- API สำหรับปุ่มลูกศร (เรียกจาก main loop โดยตรง) ----
    _PATTERNS = ("sine", "hover", "figure8", "manual")

    def press_direction(self, name):
        """name in {up,down,left,right} — เข้าโหมด manual แล้วเลื่อนตามทิศ (กดค้าง ~0.25s)."""
        import time as _t
        self.pattern = "manual"
        key = {"left": "j", "right": "l", "down": "k", "up": "i"}.get(name)
        if key:
            self._kb_hold_until[key] = _t.time() + 0.25

    def cycle_pattern(self):
        try:
            idx = self._PATTERNS.index(self.pattern)
        except ValueError:
            idx = -1
        self.pattern = self._PATTERNS[(idx + 1) % len(self._PATTERNS)]
        if self.pattern != "manual":
            self._t0 = None  # รีสตาร์ท auto-pattern
        return self.pattern

    def status_str(self):
        return (f"SIMDRONE {self.pattern} bearing=({self.pan:+.1f},{self.tilt:+.1f}) "
                f"omega={self.omega:.0f} miss={self.miss_rate:.0%}")
