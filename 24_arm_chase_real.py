"""
24_arm_chase_real.py
รันแขนกลจริง (GRBL ผ่าน cam4_arm_controller) ให้หมุนตาม "ภาพสถานการณ์จำลอง"
จาก 23_arm_chase_sim.py แล้ววัดว่าแขนจริงตามทัน/แม่นแค่ไหน (hardware-in-the-loop)

ทำอะไร
------
- สร้าง trajectory เป้า (โดรนจำลอง) เป็น bearing pan/tilt ที่เคลื่อนด้วย ω (deg/s)
  ด้วย DroneModel ตัวเดียวกับ sim
- สั่งแขนจริงด้วย move_absolute() ที่ control rate (เช่น 20 Hz)
- อ่านตำแหน่ง "จริง" ของแขนเป็นจังหวะ (sync_position_from_grbl) แล้ววัด error จริง
  (target ที่สั่ง − ตำแหน่งจริงของแขน) = lag เชิงกลของระบบจริง
- เซฟ CSV log + สรุป median/avg/max error + on-target% และพล็อต PNG (ถ้ามี matplotlib)

ความปลอดภัย
----------
- ไม่ยิงเด็ดขาด (ไม่เรียก fire())
- เคารพ effective limits ของแขน (clip ทุกคำสั่ง)
- มี countdown ก่อนเริ่มขยับ (กัน Enter พลาด) — ข้ามด้วย --yes
- --sim ใช้ PhysicsSimArmController (ไม่ต่อฮาร์ดแวร์ แต่ rate-limited จำลองพลศาสตร์แขนจริง
  → วัด keep-up/แม่นได้); --sim-ideal = snap-to-target (err≈0) เช็ค logic/CSV เท่านั้น
- คืน home เมื่อจบ/กด Ctrl-C (ปิดด้วย --no-home-on-exit)

ตัวอย่าง
-------
  python3 24_arm_chase_real.py --sim                       # จำลองพลศาสตร์แขน (rate-limited) ไม่มีฮาร์ดแวร์
  python3 24_arm_chase_real.py --sim --omega-deg 140 --amp-pan 30   # บีบให้เกิน arm_max → เห็น LAGS-BEHIND
  python3 24_arm_chase_real.py --amp-pan 20 --amp-tilt 8 --speed-ms 30 --range-m 150
  python3 24_arm_chase_real.py --pattern sine --omega-deg 25 --duration-s 20 --yes
  python3 24_arm_chase_real.py --amp-pan 15 --omega-deg 10 --out run1.csv
"""
import argparse
import importlib.util
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import serial  # pyserial (สำหรับ --diagnose อ่าน GRBL ตรง)
except ImportError:
    serial = None

try:
    import config as _config_mod
except ImportError:
    _config_mod = None

try:
    from cam4_arm_controller import Cam4ArmController, SimCam4ArmController
except ImportError as e:
    print(f"❌ import cam4_arm_controller ไม่ได้: {e}")
    sys.exit(1)

try:
    from cam4_arm_homing import parse_grbl_status
except ImportError:
    parse_grbl_status = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

# โหลด DroneModel + camera params จาก 23_arm_chase_sim.py (ชื่อโมดูลขึ้นต้นด้วยเลข)
_SIM_PATH = Path(__file__).resolve().parent / "23_arm_chase_sim.py"
_spec = importlib.util.spec_from_file_location("arm_chase_sim", _SIM_PATH)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)
DroneModel = _sim.DroneModel
load_camera_params = _sim.load_camera_params


# =============================================================================
# โหมด --diagnose: เปิด serial อ่านสถานะ GRBL ตรง ๆ (ไม่แตะ homing/controller)
# =============================================================================
def _grbl_status(ser):
    """ส่ง '?' (realtime) อ่านบรรทัด <...> สถานะ"""
    ser.reset_input_buffer()
    ser.write(b"?")
    t0 = time.time()
    while time.time() - t0 < 1.5:
        line = ser.readline().decode(errors="replace").strip()
        if line.startswith("<") and line.endswith(">"):
            return line
    return None


def _grbl_send(ser, cmd, wait=0.8):
    """ส่งคำสั่ง อ่านบรรทัดจนเจอ ok/error หรือหมดเวลา — คืน list บรรทัด (ไม่รวม ok)"""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    out = []
    t0 = time.time()
    while time.time() - t0 < wait + 1.5:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            if time.time() - t0 > wait:
                break
            continue
        if line == "ok":
            break
        if line.startswith("error"):
            out.append(line)
            break
        out.append(line)
    return out


def _print_lines(tag, lines):
    print(f"\n[{tag}]")
    if not lines:
        print("  (ไม่ตอบ)")
    for l in lines:
        print("  " + l)


def _interpret_grbl(status, settings):
    print("\n" + "=" * 64)
    print("สรุป / ข้อแนะนำ")
    print("=" * 64)
    state = None
    if status and status.startswith("<"):
        state = status[1:].split("|")[0].split(">")[0]
        print(f"• สถานะ GRBL: {state}")
        if state.lower().startswith("alarm"):
            print("  ⚠️ อยู่ในสถานะ ALARM → ส่ง $X ปลดล็อกก่อน (หรือรัน --skip-homing ซึ่งทำ $X ให้)")
    else:
        print("• อ่านสถานะ (?) ไม่ได้ — GRBL อาจไม่ตอบ/บอร์ดไม่ใช่ GRBL/baud ผิด")

    def g(n):
        return settings.get(n)

    labels = {
        20: "soft limits ($20)", 21: "hard limits ($21)", 22: "homing enable ($22)",
        23: "homing dir mask ($23)", 24: "homing feed ($24)", 25: "homing seek ($25)",
        26: "homing debounce ($26)", 27: "homing pull-off ($27)",
        110: "X max rate ($110)", 111: "Y max rate ($111)",
        120: "X accel ($120)", 121: "Y accel ($121)",
    }
    if settings:
        print("• ค่าที่เกี่ยวข้อง:")
        for n in (20, 21, 22, 23, 24, 25, 27, 110, 111, 120, 121):
            if n in settings:
                print(f"    {labels.get(n, '$'+str(n)):24s} = {settings[n]}")

    h22 = g(22)
    if h22 is not None:
        if h22 in ("0", "0.0"):
            print("• ⚠️ homing ปิดอยู่ ($22=0) → คำสั่ง $H จะใช้ไม่ได้ (error:9/ไม่ home)")
            print("    วิธีแก้ที่ง่ายสุด: รันด้วย  --skip-homing  (ใช้ G92 ตั้งศูนย์ที่ตำแหน่งปัจจุบัน)")
            print("    หรือถ้ามี limit switch จริง: เปิด homing ด้วย  $22=1  แล้วตั้ง $23/$24/$25/$27 ให้ถูก")
        else:
            print("• homing เปิด ($22=1) แต่ยัง home ไม่ได้ → ตรวจ:")
            print(f"    - limit switch ต่อ/ทำงานไหม (hard limits $21={g(21)})")
            print("    - ทิศ homing ($23) / pull-off ($27) / feed ($24,$25)")
            print("    - มอเตอร์มีไฟ, E-stop ไม่กด, stepper driver ทำงาน")
            print("    ถ้าไม่อยากแก้ ใช้  --skip-homing  ข้ามไปก่อนได้")
    else:
        print("• อ่าน $$ ไม่ได้ → ตรวจ baud/สาย; ถ้าจำเป็นรัน --skip-homing")
    print("=" * 64)


def run_diagnose(args):
    if serial is None:
        print("❌ ไม่มี pyserial — ติดตั้งด้วย: pip install pyserial")
        return
    port = getattr(_config_mod, "CAM4_ARM_SERIAL_PORT", "/dev/ttyUSB0") if _config_mod else "/dev/ttyUSB0"
    baud = int(getattr(_config_mod, "CAM4_ARM_BAUD_RATE", 115200)) if _config_mod else 115200
    print("=" * 64)
    print(f"GRBL DIAGNOSE — port={port} baud={baud}")
    print("=" * 64)
    if not os.path.exists(port):
        print(f"❌ ไม่พบพอร์ต {port}")
        print("   เช็ค:  ls -l /dev/ttyUSB* /dev/ttyACM*")
        print("   สิทธิ์: sudo usermod -aG dialout $USER (แล้ว logout/login)")
        print("   หรือแก้ CAM4_ARM_SERIAL_PORT ใน config.py ให้ตรงพอร์ตจริง")
        return
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except Exception as e:
        print(f"❌ เปิดพอร์ตไม่ได้: {e}")
        print("   อาจถูกใช้งานโดยโปรเซสอื่น — ปิด 22_gun_aim_assist_vector.py / โปรแกรมอื่นที่จับพอร์ตก่อน")
        return
    try:
        time.sleep(0.2)
        ser.write(b"\r\n\r\n")
        time.sleep(0.6)
        banner = []
        t0 = time.time()
        while time.time() - t0 < 0.6:
            l = ser.readline().decode(errors="replace").strip()
            if l:
                banner.append(l)
        _print_lines("banner", banner)
        status = _grbl_status(ser)
        print(f"\n[?] realtime status: {status or '(ไม่ตอบ)'}")
        _print_lines("$I (build/version)", _grbl_send(ser, "$I"))
        _print_lines("$G (parser state)", _grbl_send(ser, "$G"))
        _print_lines("$# (offsets/G92)", _grbl_send(ser, "$#"))
        sett_lines = _grbl_send(ser, "$$", wait=2.0)
        settings = {}
        for l in sett_lines:
            m = re.match(r"\$(\d+)=([\-0-9.]+)", l)
            if m:
                settings[int(m.group(1))] = m.group(2)
        print(f"\n[$$] อ่าน setting ได้ {len(settings)} ค่า")
        _interpret_grbl(status, settings)
    finally:
        try:
            ser.close()
        except Exception:
            pass


def _query_mpos(arm):
    """อ่าน MPos (mm) ตรงจาก GRBL ผ่าน serial ของ controller. คืน (mx,my) หรือ None.
    ใช้เมื่อ sync_position_from_grbl ใช้ไม่ได้ (เช่น GRBL $10=1 รายงาน MPos ไม่ใช่ WPos
    → parser ได้ wpos=None → sync คืน False ตลอด)."""
    if parse_grbl_status is None:
        return None
    try:
        with arm._serial_lock:
            raw = arm._query_grbl_status_unlocked(timeout_sec=0.3)
    except Exception:
        return None
    if not raw:
        return None
    info = parse_grbl_status(raw)
    mx, my = info.get("mpos_x"), info.get("mpos_y")
    if mx is None or my is None:
        return None
    return float(mx), float(my)


class _RealPosReader:
    """วัดตำแหน่งแขนจริงแบบ differential จาก MPos: real_deg = center + (MPos-MPos0)/mm_per_deg.
    ถูกต้องโดยไม่ต้องพึ่ง WPos/WCO (WCO คงที่ → หักออกในผลต่าง). ลอง sync_position_from_grbl
    ก่อน (ถ้าอัปเดต pos จริงได้) แล้วค่อย fallback มาอ่าน MPos ตรง."""
    def __init__(self, arm, center_pan, center_tilt):
        self.arm = arm
        self.center_pan, self.center_tilt = center_pan, center_tilt
        self.mmpp = float(getattr(arm, "mm_per_deg_pan", 1.0)) or 1.0
        self.mmpt = float(getattr(arm, "mm_per_deg_tilt", 1.0)) or 1.0
        self.mpos0 = _query_mpos(arm)   # อ่านจุดอ้างอิงตอนแขนยังนิ่ง (= center)
        self.ok_count = 0
        self.fail_count = 0
        self.mode = None  # 'sync' | 'mpos' | None

    def read(self, fallback_pan, fallback_tilt):
        """คืน (act_pan, act_tilt, is_real). is_real=False → วัดจริงไม่ได้ (อย่าเชื่อค่า)."""
        # 1) ลอง sync_position_from_grbl (อ่าน WPos) — ใช้ได้ถ้า GRBL รายงาน WPos
        if hasattr(self.arm, "sync_position_from_grbl"):
            try:
                if self.arm.sync_position_from_grbl():
                    self.ok_count += 1; self.mode = "sync"
                    return float(self.arm.pos_x), float(self.arm.pos_y), True
            except Exception:
                pass
        # 2) fallback: MPos differential
        if self.mpos0 is not None:
            mp = _query_mpos(self.arm)
            if mp is not None:
                ap = self.center_pan + (mp[0] - self.mpos0[0]) / self.mmpp
                at = self.center_tilt + (mp[1] - self.mpos0[1]) / self.mmpt
                self.ok_count += 1; self.mode = "mpos"
                return ap, at, True
        self.fail_count += 1
        return fallback_pan, fallback_tilt, False


class PhysicsSimArmController:
    """แขนจำลอง 'มีพลศาสตร์' — rate-limited ต่อแกน + command throttle โดยใช้เวลานาฬิกาจริง
    (wall-clock) ในการอินทิเกรตตำแหน่ง. ต่างจาก SimCam4ArmController ที่ snap ไปเป้าทันที
    (err=0 เสมอ ทดสอบ keep-up ไม่ได้) — ตัวนี้ทำให้ --sim วัดได้จริงว่า 'แขนหมุนทัน/แม่นไหม'
    โดยไม่ต้องต่อฮาร์ดแวร์. อิงค่าความสามารถแขนจาก load_arm_params() (config)."""

    SUB_DT = 0.001   # อินทิเกรตย่อย 1ms — ทำให้ผลไม่ขึ้นกับ loop period (50ms) และตรงแขนจริง

    def __init__(self, arm_params, throttle=True, latency_ms=0.0):
        self.is_simulation_mode = True
        self.x_limits = tuple(arm_params["x_lim"])
        self.y_limits = tuple(arm_params["y_lim"])
        margin = float(getattr(_config_mod, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0)) if _config_mod else 2.0
        self._effective_x_limits = (self.x_limits[0] + margin, self.x_limits[1] - margin)
        self._effective_y_limits = (self.y_limits[0] + margin, self.y_limits[1] - margin)
        self.max_pan = float(arm_params["max_pan_rate"])
        self.max_tilt = float(arm_params["max_tilt_rate"])
        self.accel_pan = float(arm_params.get("max_pan_accel", 0.0) or 0.0)
        self.accel_tilt = float(arm_params.get("max_tilt_accel", 0.0) or 0.0)
        self.cmd_interval = float(arm_params["cmd_interval"]) if throttle else 0.0
        self.latency = max(0.0, latency_ms) / 1000.0   # transport delay ของคำสั่ง (วินาที); fit จริง≈0
        self.ref_pan = float(getattr(_config_mod, "CAM4_ARM_REF_PAN_DEG", 0.0)) if _config_mod else 0.0
        self.ref_tilt = float(getattr(_config_mod, "CAM4_ARM_REF_TILT_DEG", 0.0)) if _config_mod else 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0   # ความเร็วเชิงมุมปัจจุบัน (สำหรับ accel limit)
        self.vel_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self._cmd_hist = deque()   # (t, tx, ty) — ใช้เมื่อ latency>0
        self._last_t = None
        self._last_cmd_t = -1e9

    def connect(self):
        self.pos_x = self.target_x = self.ref_pan
        self.pos_y = self.target_y = self.ref_tilt
        self.vel_x = self.vel_y = 0.0
        self._cmd_hist.clear()
        self._last_t = time.time()
        accel_txt = (f"accel {self.accel_pan:.0f}/{self.accel_tilt:.0f}°/s² "
                     if self.accel_pan > 0 or self.accel_tilt > 0 else "")
        lat_txt = f"latency {self.latency*1000:.0f}ms " if self.latency > 0 else ""
        print(f"[real] โหมด --sim (physics): rate-limited "
              f"{self.max_pan:.0f}°/s pan, {self.max_tilt:.0f}°/s tilt, {accel_txt}{lat_txt}"
              f"throttle {self.cmd_interval*1000:.0f}ms, substep {self.SUB_DT*1000:.0f}ms — ไม่ต่อฮาร์ดแวร์")
        return True

    def disconnect(self):
        pass

    def _active_target(self, query_t):
        """target ที่ถูกสั่งไว้และ active ณ เวลา query_t (ค้นจาก _cmd_hist) — ใช้เมื่อ latency>0.
        query_t เพิ่มขึ้นเรื่อย ๆ ภายใน 1 _integrate จึง prune คำสั่งเก่าทิ้งได้ปลอดภัย."""
        while len(self._cmd_hist) >= 2 and self._cmd_hist[1][0] <= query_t:
            self._cmd_hist.popleft()
        if self._cmd_hist and self._cmd_hist[0][0] <= query_t:
            return self._cmd_hist[0][1], self._cmd_hist[0][2]
        return self.target_x, self.target_y   # ก่อนมีคำสั่งใน hist

    def _integrate(self, now):
        """อินทิเกรตตำแหน่งแบบ sub-step (1ms) เข้าหา target — rate + accel จำกัด.
        sub-step ทำให้ผลไม่ขึ้นกับ loop period (เดิม integrate ก้อน 50ms → under-lag).
        latency>0 → ค้น target ที่ active ณ (เวลา sub-step − latency) ต่อ sub-step (เนียน ไม่ quantize)."""
        if self._last_t is None:
            self._last_t = now
            return
        t0 = self._last_t
        dt = now - t0
        self._last_t = now
        if dt <= 0:
            return
        n = max(1, int(math.ceil(dt / self.SUB_DT)))
        h = dt / n
        use_lat = self.latency > 0
        tx, ty = self.target_x, self.target_y   # latency=0: คำสั่งที่ active ช่วงนี้ (ZOH 1 step)
        for i in range(n):
            if use_lat:
                tx, ty = self._active_target(t0 + (i + 0.5) * h - self.latency)
            self.pos_x, self.vel_x = _sim.accel_limited_step(
                self.pos_x, self.vel_x, tx, self.max_pan, self.accel_pan, h,
                self._effective_x_limits)
            self.pos_y, self.vel_y = _sim.accel_limited_step(
                self.pos_y, self.vel_y, ty, self.max_tilt, self.accel_tilt, h,
                self._effective_y_limits)

    def move_absolute(self, x_deg, y_deg, blocking=False):
        now = time.time()
        # อินทิเกรตช่วง [last,now] 'ก่อน' อัปเดต target → ใช้คำสั่งที่ active ช่วงนั้น (ZOH ถูกต้อง:
        # ระหว่าง [t_k,t_{k+1}] แขนวิ่งเข้าหา target ที่สั่งไว้ ณ t_k เหมือนระบบจริง)
        self._integrate(now)
        if self.cmd_interval > 0 and (now - self._last_cmd_t) < self.cmd_interval:
            return False  # throttled (เลียนแบบ Cam4ArmController)
        self._last_cmd_t = now
        self.target_x = float(np.clip(x_deg, self._effective_x_limits[0], self._effective_x_limits[1]))
        self.target_y = float(np.clip(y_deg, self._effective_y_limits[0], self._effective_y_limits[1]))
        if self.latency > 0:
            self._cmd_hist.append((now, self.target_x, self.target_y))
        return True

    def sync_position_from_grbl(self, status_timeout_sec=0.6):
        self._integrate(time.time())
        return True

    def go_home(self, blocking=True):
        self.target_x, self.target_y = self.ref_pan, self.ref_tilt
        self.pos_x, self.pos_y = self.ref_pan, self.ref_tilt  # คืน home (cosmetic)
        self.vel_x = self.vel_y = 0.0
        self._cmd_hist.clear()
        self._last_t = time.time()


def build_controller(args, arm_params):
    if args.sim_ideal:
        print("[real] โหมด --sim-ideal: SimCam4ArmController (snap-to-target, ทดสอบ logic/CSV เท่านั้น)")
        return SimCam4ArmController()
    if args.sim:
        return PhysicsSimArmController(arm_params, throttle=not args.no_throttle,
                                       latency_ms=args.sim_latency_ms)
    return Cam4ArmController()


def countdown(sec):
    print(f"\n⚠️  กำลังจะ 'หมุนแขนจริง' ใน {sec} วินาที — กด Ctrl-C เพื่อยกเลิก")
    for i in range(sec, 0, -1):
        print(f"   ...{i}", flush=True)
        time.sleep(1.0)
    print("   เริ่ม!\n")


def main():
    ap = argparse.ArgumentParser(description="หมุนแขนจริงตามสถานการณ์จำลอง + วัด tracking error จริง")
    ap.add_argument("--camera", default=None, help="ชื่อกล้องใน config (สำหรับคำนวณ reticle)")
    ap.add_argument("--sim", action="store_true",
                    help="dry-run ไม่มีฮาร์ดแวร์ — ใช้ PhysicsSimArmController (rate-limited จำลองพลศาสตร์ "
                         "แขนจริง → วัด keep-up/แม่นได้)")
    ap.add_argument("--sim-ideal", action="store_true",
                    help="dry-run แบบ snap-to-target (SimCam4ArmController) — err≈0 เสมอ ใช้เช็ค logic/CSV เท่านั้น")
    ap.add_argument("--no-throttle", action="store_true",
                    help="(physics sim) ปิด command throttle ของแขนจำลอง")
    ap.add_argument("--arm-accel", type=float, default=None,
                    help="(physics sim) override ความเร่งแขน deg/s^2 ทั้ง 2 แกน (default จาก config ~100; 0=ปิด)")
    ap.add_argument("--sim-latency-ms", type=float, default=0.0,
                    help="(physics sim) transport delay ของคำสั่ง ms (fit แขนจริง≈0; ใส่ถ้าวัด latency จริงได้)")
    ap.add_argument("--diagnose", action="store_true",
                    help="เปิด serial อ่านสถานะ GRBL ($?/$I/$G/$$) + แปลผล homing/alarm แล้วออก")
    ap.add_argument("--skip-homing", action="store_true",
                    help="ข้าม homing cycle: ใช้ $X + G92 X0 Y0 (ตั้งตำแหน่งปัจจุบันเป็นศูนย์) "
                         "— ใช้เมื่อแขน homing ไม่ผ่าน/ไม่มี limit switch")
    ap.add_argument("--yes", action="store_true", help="ข้าม countdown เริ่มทันที")
    ap.add_argument("--countdown", type=int, default=5, help="วินาที countdown ก่อนขยับ")
    # trajectory
    ap.add_argument("--pattern", choices=["cross", "sine"], default="sine", help="รูปแบบการเคลื่อน")
    ap.add_argument("--speed-ms", type=float, default=30.0, help="ความเร็วเชิงเส้นโดรน (m/s)")
    ap.add_argument("--range-m", type=float, default=150.0, help="ระยะโดรน (m)")
    ap.add_argument("--omega-deg", type=float, default=None, help="กำหนด ω เชิงมุมตรง ๆ (deg/s) override speed/range")
    ap.add_argument("--amp-pan", type=float, default=20.0, help="แอมพลิจูดการกวาด pan (องศา รอบ center)")
    ap.add_argument("--amp-tilt", type=float, default=8.0, help="แอมพลิจูดการกวาด tilt (องศา)")
    ap.add_argument("--center-pan", type=float, default=None, help="center pan (default = ตำแหน่งแขนหลัง homing)")
    ap.add_argument("--center-tilt", type=float, default=None, help="center tilt")
    # loop
    ap.add_argument("--rate", type=float, default=20.0, help="control rate ส่งคำสั่ง (Hz)")
    ap.add_argument("--measure-interval", type=float, default=0.4, help="ช่วงเวลาอ่านตำแหน่งจริง (s)")
    ap.add_argument("--duration-s", type=float, default=20.0, help="ระยะเวลารัน (s)")
    ap.add_argument("--reticle-deg", type=float, default=None, help="เกณฑ์ on-target (deg); default คำนวณจากกล้อง")
    ap.add_argument("--no-home-on-exit", action="store_true", help="ไม่ต้องคืน home ตอนจบ")
    ap.add_argument("--out", default="arm_chase_real.csv", help="path CSV log (.csv) + PNG คู่กัน")
    args = ap.parse_args()

    if args.diagnose:
        run_diagnose(args)
        return

    if args.sim_ideal:        # ideal ก็คือ dry-run ไม่มีฮาร์ดแวร์เช่นกัน
        args.sim = True

    cam = load_camera_params(args.camera)
    reticle_px = max(10, min(cam["width"], cam["height"]) * 0.03)
    reticle_deg = args.reticle_deg if args.reticle_deg is not None else reticle_px / cam["ppd_x"]

    if args.omega_deg is not None:
        omega = args.omega_deg
        src = f"omega={omega:.1f} deg/s (direct)"
    else:
        omega = (args.speed_ms / max(args.range_m, 1e-6)) * (180.0 / math.pi)
        src = f"v={args.speed_ms} m/s, R={args.range_m} m -> omega={omega:.1f} deg/s"

    print("=" * 64)
    print("ARM CHASE (REAL) —", src)
    print(f"pattern={args.pattern}  amp_pan=±{args.amp_pan}  amp_tilt=±{args.amp_tilt}  "
          f"rate={args.rate}Hz  dur={args.duration_s}s  reticle={reticle_deg:.2f} deg")
    print("=" * 64)

    arm_params = _sim.load_arm_params()
    if args.arm_accel is not None:
        arm_params["max_pan_accel"] = arm_params["max_tilt_accel"] = args.arm_accel
    arm = build_controller(args, arm_params)

    # ข้าม homing: เซ็ต config flag ชั่วคราวใน-process (ไม่แก้ config.py ถาวร)
    # connect() จะอ่าน flag นี้แล้วใช้ $X + G92 X0 Y0 แทน homing cycle
    if args.skip_homing and not args.sim:
        if _config_mod is not None:
            _config_mod.CAM4_ARM_SKIP_HOMING_USE_G92_ONLY = True
        print("[real] --skip-homing: ข้าม homing → ใช้ตำแหน่งปัจจุบันเป็นศูนย์ (G92 X0 Y0)")
        print("       ⚠️  จัดแขนให้อยู่ตำแหน่ง 'กลาง/ปลอดภัย' ด้วยมือก่อน แล้วเริ่มด้วย amplitude เล็ก")

    print("[real] connect%s... (แขนอาจขยับ)" % ("" if args.skip_homing else " + homing"))
    if not arm.connect():
        print("❌ connect/homing ล้มเหลว — ออก")
        if not args.skip_homing:
            print("   ลองแก้: ")
            print("     1) python3 24_arm_chase_real.py --skip-homing ...  (ข้าม homing ใช้ G92)")
            print("     2) เช็ค GRBL Alarm — ส่ง $X ปลดล็อก, เช็ค E-stop / ไฟมอเตอร์ / สาย USB")
            print("     3) ถ้าไม่มี limit switch ต้องใช้ --skip-homing (homing cycle จะล้มเสมอ)")
        else:
            print("   --skip-homing ก็ยังล้ม → เช็คพอร์ต serial (CAM4_ARM_SERIAL_PORT) / สิทธิ์ /dev/tty* / ไฟ GRBL")
        return
    arm_max = min(
        float(getattr(_config_mod, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0)) if _config_mod else 80.0,
        float(getattr(_config_mod, "CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0)) if _config_mod else 60.0,
    )
    if omega > arm_max:
        print(f"⚠️  ω({omega:.1f}) > arm_max(~{arm_max:.0f} deg/s) — แขนอาจหมุนตามไม่ทัน (ผลทดสอบจะเห็น lag)")

    # center = ตำแหน่งแขนหลัง homing (หรือ override)
    try:
        if hasattr(arm, "sync_position_from_grbl"):
            arm.sync_position_from_grbl()
    except Exception:
        pass
    center_pan = args.center_pan if args.center_pan is not None else float(getattr(arm, "pos_x", 0.0))
    center_tilt = args.center_tilt if args.center_tilt is not None else float(getattr(arm, "pos_y", 0.0))
    x_lo, x_hi = arm._effective_x_limits
    y_lo, y_hi = arm._effective_y_limits
    # หด amplitude ให้ center±amp อยู่ในลิมิต
    amp_pan = max(0.0, min(args.amp_pan, center_pan - x_lo, x_hi - center_pan))
    amp_tilt = max(0.0, min(args.amp_tilt, center_tilt - y_lo, y_hi - center_tilt))
    if amp_pan < args.amp_pan or amp_tilt < args.amp_tilt:
        print(f"[real] หด amplitude ให้พอดีลิมิต: pan ±{amp_pan:.1f}, tilt ±{amp_tilt:.1f} "
              f"(center pan={center_pan:.1f} tilt={center_tilt:.1f})")
    if amp_pan < 1.0:
        print("❌ ระยะกวาด pan เล็กเกินไป (center ใกล้ลิมิต) — ปรับ --center-pan หรือ --amp-pan")
        arm.disconnect()
        return

    # DroneModel: span = fov_h*0.6 -> ตั้ง fov_h = amp/0.6 ให้ span = amp
    drone = DroneModel(args.pattern, omega, fov_h=amp_pan / 0.6, fov_v=amp_tilt / 0.6, tilt_amp_deg=amp_tilt)

    if not args.yes and not args.sim:
        countdown(max(1, args.countdown))
    elif args.sim:
        print("[real] (--sim) เริ่มทันที")

    # ตัวอ่านตำแหน่งจริง (เฉพาะฮาร์ดแวร์) — แขนยังนิ่งตอนนี้ จึงเก็บ MPos0 = center
    reader = None
    if not args.sim:
        reader = _RealPosReader(arm, center_pan, center_tilt)
        if reader.mpos0 is None and not hasattr(arm, "sync_position_from_grbl"):
            print("[real] ⚠️  อ่านตำแหน่งจริงไม่ได้ (ไม่มี MPos และไม่มี sync) — ผล err จะไม่น่าเชื่อถือ")
        else:
            print(f"[real] real-pos readback: MPos0={reader.mpos0}  "
                  f"mm/deg pan={reader.mmpp:.3f} tilt={reader.mmpt:.3f}")

    period = 1.0 / max(1.0, args.rate)
    log = []  # (t, tgt_pan, tgt_tilt, act_pan, act_tilt, err_pan, err_tilt, err_deg)
    t0 = time.time()
    next_measure = 0.0
    sent_count = 0
    loop_count = 0
    try:
        while True:
            loop_start = time.time()
            t = loop_start - t0
            if t >= args.duration_s:
                break
            # target bearing จากสถานการณ์จำลอง
            pan_off, tilt_off = drone.angle_at(t)
            tgt_pan = float(np.clip(center_pan + pan_off, x_lo, x_hi))
            tgt_tilt = float(np.clip(center_tilt + tilt_off, y_lo, y_hi))
            sent = arm.move_absolute(tgt_pan, tgt_tilt, blocking=False)
            if sent is not False:  # real คืน True/False (False=throttled); sim คืน None=ส่งเสมอ
                sent_count += 1
            loop_count += 1

            # วัดตำแหน่งจริงเป็นจังหวะ (sync ช้า — ไม่ทำทุกลูป)
            if t >= next_measure:
                next_measure = t + args.measure_interval
                if args.sim:
                    act_pan = float(getattr(arm, "pos_x", tgt_pan))
                    act_tilt = float(getattr(arm, "pos_y", tgt_tilt))
                    is_real = True
                else:
                    act_pan, act_tilt, is_real = reader.read(tgt_pan, tgt_tilt)
                ep, et = tgt_pan - act_pan, tgt_tilt - act_tilt
                err = math.hypot(ep, et)
                log.append((t, tgt_pan, tgt_tilt, act_pan, act_tilt, ep, et, err))
                flag = "ON " if err <= reticle_deg else "lag"
                rmark = "" if is_real else "  ⚠อ่านตำแหน่งจริงไม่ได้"
                print(f"  t={t:5.1f}s  tgt=({tgt_pan:6.1f},{tgt_tilt:5.1f})  "
                      f"act=({act_pan:6.1f},{act_tilt:5.1f})  err={err:5.2f}deg [{flag}]{rmark}")

            # รักษา loop rate
            dt_sleep = period - (time.time() - loop_start)
            if dt_sleep > 0:
                time.sleep(dt_sleep)
    except KeyboardInterrupt:
        print("\n[real] ยกเลิกด้วย Ctrl-C")
    finally:
        if not args.no_home_on_exit:
            try:
                print("[real] คืน home...")
                arm.go_home(blocking=True)
            except Exception as e:
                print(f"[real] go_home ล้มเหลว: {e}")
        try:
            arm.disconnect()
        except Exception:
            pass

    _report(log, reticle_deg, args, cam, src, sent_count, loop_count, reader)


def _report(log, reticle_deg, args, cam, src, sent_count, loop_count, reader=None):
    if not log:
        print("\n[real] ไม่มีข้อมูลวัด — จบ")
        return
    # ตัด warmup 2 วินาทีแรก (แขนวิ่งเข้าจุดเริ่ม)
    samp = [r for r in log if r[0] >= 2.0] or log
    errs = [r[7] for r in samp]
    med = float(np.median(errs)); avg = float(np.mean(errs)); mx = float(np.max(errs))
    on_pct = 100.0 * sum(1 for e in errs if e <= reticle_deg) / len(errs)

    # ตรวจความน่าเชื่อถือของการวัด (ฮาร์ดแวร์): อ่านตำแหน่งจริงไม่ได้ → err ปลอม
    invalid = None
    if reader is not None:
        if reader.ok_count == 0:
            invalid = ("อ่านตำแหน่งแขนจริงไม่สำเร็จเลย (sync/MPos ใช้ไม่ได้) — "
                       "act = ค่าที่สั่ง → err ที่เห็นเป็น 'ศูนย์ปลอม' ไม่ใช่ผลจริง")
        elif mx < 1e-6:
            invalid = ("error = 0 ทุกจุด ทั้งที่แขนหมุน — น่าจะ readback ไม่อัปเดต (artifact)")

    print("\n--- ผลทดสอบแขนจริง ---")
    print(src)
    print(f"คำสั่งที่ส่งจริง: {sent_count}/{loop_count} ลูป")
    if reader is not None:
        print(f"อ่านตำแหน่งจริง: สำเร็จ {reader.ok_count} / ล้มเหลว {reader.fail_count} ครั้ง "
              f"(โหมด={reader.mode})")
    print(f"tracking error (หลัง warmup): median {med:.2f} deg | avg {avg:.2f} deg | max {mx:.2f} deg "
          f"| reticle {reticle_deg:.2f} deg")
    print(f"เล็งตรงเป้า (err ≤ reticle): {on_pct:.1f}% ของการวัด")
    if invalid:
        print("\n" + "!" * 64)
        print(f"⚠️  ผลไม่ถูกต้อง: {invalid}")
        print("    แก้: ตั้ง GRBL ให้รายงาน WPos ($10=0) หรือเช็คว่า MPos อ่านได้")
        print("!" * 64)
        print("VERDICT: INVALID (วัดตำแหน่งจริงไม่ได้)")
    else:
        verdict = "ON-TARGET" if med <= reticle_deg else "LAGS-BEHIND"
        print(f"VERDICT: {verdict}")

    # CSV
    base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
    csv_path = base + ".csv"
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t_s", "tgt_pan", "tgt_tilt", "act_pan", "act_tilt", "err_pan", "err_tilt", "err_deg"])
        for r in log:
            w.writerow(["%.3f" % r[0]] + ["%.3f" % v for v in r[1:]])
    print(f"\nบันทึก: {csv_path}")

    # PNG: target vs actual (pan,tilt) + error เทียบเวลา
    if not _HAS_PLT:
        print("  (ข้ามพล็อต PNG — ไม่มี matplotlib)")
        return
    ts = [r[0] for r in log]
    fig, (ax1, ax2, ax3) = _plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax1.plot(ts, [r[1] for r in log], "b-", label="target pan")
    ax1.plot(ts, [r[3] for r in log], "r.-", label="actual pan")
    ax1.set_ylabel("pan (deg)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(ts, [r[2] for r in log], "b-", label="target tilt")
    ax2.plot(ts, [r[4] for r in log], "r.-", label="actual tilt")
    ax2.set_ylabel("tilt (deg)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax3.plot(ts, [r[7] for r in log], "m.-", label="error")
    ax3.axhline(reticle_deg, ls="--", color="g", label=f"reticle {reticle_deg:.2f}")
    ax3.set_ylabel("error (deg)"); ax3.set_xlabel("time (s)"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    ax1.set_title("Real arm chase — %s  %s  (median err %.2f deg)" % (cam["name"], src, med), fontsize=9)
    fig.tight_layout(); fig.savefig(base + ".png", dpi=120); _plt.close(fig)
    print(f"        {base}.png  (target vs actual + error)")


if __name__ == "__main__":
    main()
