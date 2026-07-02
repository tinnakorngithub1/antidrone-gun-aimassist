"""
26_accel_sweep_real.py
กวาดค่า GRBL accel ($120/$121) บนแขนจริง แล้ววัด tracking error ที่แต่ละค่า
เพื่อพิสูจน์ว่า "เพิ่มความเร่งแขน" ช่วยลด lag ที่ ω สูงได้จริงแค่ไหน (คันโยกที่โมเดลทำนาย)

ทำอะไร
------
- อ่านค่า $120/$121 เดิมเก็บไว้
- สำหรับแต่ละ accel ในลิสต์: ตั้ง $120/$121 → รัน 24_arm_chase_real.py (วัดจริง) → parse median/on-target
- รัน sim ทำนายคู่กัน (24_arm_chase_real.py --sim --arm-accel ...) ไว้เทียบ
- **คืนค่า $120/$121 เดิมเสมอ** (finally — ครอบ Ctrl-C/error)
- สรุปเป็นตาราง + เซฟ CSV

ความปลอดภัย
----------
- ไม่ยิง (24 ไม่เรียก fire)
- accel สูงเกินทอร์ก → มอเตอร์ stall/สูญ step: เริ่มจากต่ำ, ดูแขนไม่สะดุด/ไฟไม่ตก
- เตือน + countdown ก่อนเริ่ม; --sim-only ลองแบบไม่แตะฮาร์ดแวร์ได้
- คืนค่า accel เดิมก่อนจบทุกกรณี

ตัวอย่าง
-------
  python3 26_accel_sweep_real.py --omega-deg 30 --accels 100,150,200
  python3 26_accel_sweep_real.py --sim-only --accels 70,100,150,250   # ดูเฉพาะ sim
"""
import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None

try:
    import config as _config_mod
except ImportError:
    _config_mod = None

_HERE = Path(__file__).resolve().parent
_REAL = _HERE / "24_arm_chase_real.py"

_MED_RE = re.compile(r"median\s+([\d.]+)\s+deg")
_MAX_RE = re.compile(r"max\s+([\d.]+)\s+deg")
_ON_RE = re.compile(r"reticle\):\s*([\d.]+)%")


def _port_baud():
    port = getattr(_config_mod, "CAM4_ARM_SERIAL_PORT", "/dev/cam4_arm") if _config_mod else "/dev/cam4_arm"
    baud = int(getattr(_config_mod, "CAM4_ARM_BAUD_RATE", 115200)) if _config_mod else 115200
    return port, baud


def _grbl_cmd(ser, cmd, wait=0.6):
    """ส่งคำสั่ง อ่านจน ok/error/timeout — คืน list บรรทัด"""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    out, t0 = [], time.time()
    while time.time() - t0 < wait + 1.0:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            if time.time() - t0 > wait:
                break
            continue
        if line == "ok":
            break
        out.append(line)
        if line.startswith("error"):
            break
    return out


def _read_accel(ser):
    """อ่าน $120,$121 ปัจจุบัน (คืน dict {120:val,121:val} หรือ {})"""
    lines = _grbl_cmd(ser, "$$", wait=1.5)
    cur = {}
    for l in lines:
        m = re.match(r"\$(120|121)=([\-0-9.]+)", l)
        if m:
            cur[int(m.group(1))] = float(m.group(2))
    return cur


def _set_accel(ser, accel):
    """ตั้ง $120=$121=accel; คืน True ถ้าไม่เจอ error"""
    ok = True
    for n in (120, 121):
        resp = _grbl_cmd(ser, f"${n}={accel:g}")
        if any(r.startswith("error") for r in resp):
            print(f"    ❌ ตั้ง ${n}={accel} ล้มเหลว: {resp}")
            ok = False
    return ok


def _run_chase(accel, args, sim):
    """รัน 24_arm_chase_real.py 1 รอบ คืน (median, max, on_pct) หรือ (None,None,None)"""
    cmd = [
        sys.executable, str(_REAL),
        "--omega-deg", str(args.omega_deg),
        "--amp-pan", str(args.amp_pan), "--amp-tilt", str(args.amp_tilt),
        "--rate", str(args.rate), "--duration-s", str(args.duration_s),
        "--out", str(_HERE / "_accel_sweep_tmp.csv"),
    ]
    if sim:
        cmd += ["--sim", "--arm-accel", str(accel)]
    else:
        cmd += ["--skip-homing", "--yes"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.duration_s + 60)
    except subprocess.TimeoutExpired:
        print("    ⏱️ timeout")
        return None, None, None
    txt = r.stdout + r.stderr
    med = _MED_RE.search(txt)
    mx = _MAX_RE.search(txt)
    on = _ON_RE.search(txt)
    if med is None:
        print("    ⚠️ parse ผลไม่ได้ — output ท้าย:")
        print("    " + "\n    ".join(txt.strip().splitlines()[-6:]))
        return None, None, None
    return (float(med.group(1)),
            float(mx.group(1)) if mx else float("nan"),
            float(on.group(1)) if on else float("nan"))


def main():
    ap = argparse.ArgumentParser(description="กวาด GRBL accel ($120/$121) วัด tracking error จริง")
    ap.add_argument("--accels", default="100,150,200", help="ลิสต์ค่า accel deg/s^2 (คั่นด้วย ,)")
    ap.add_argument("--omega-deg", type=float, default=30.0)
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--sim-only", action="store_true", help="รันเฉพาะ sim ไม่แตะฮาร์ดแวร์")
    ap.add_argument("--max-accel", type=float, default=300.0, help="เพดานกันตั้งสูงเกิน (กัน stall)")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--out", default="accel_sweep.csv")
    args = ap.parse_args()

    accels = [float(x) for x in str(args.accels).split(",") if x.strip()]
    over = [a for a in accels if a > args.max_accel]
    if over:
        print(f"❌ accel {over} เกินเพดาน --max-accel={args.max_accel} (กัน stall) — ลดค่าหรือเพิ่มเพดาน")
        return

    print("=" * 64)
    print(f"ACCEL SWEEP — ω={args.omega_deg} deg/s  amp_pan=±{args.amp_pan}  rate={args.rate}Hz")
    print(f"accels = {accels} deg/s^2   ({'SIM-ONLY' if args.sim_only else 'แขนจริง + sim'})")
    print("=" * 64)

    # --- sim predictions (เร็ว ไม่ต้องฮาร์ดแวร์) ---
    sim_rows = {}
    print("\n[sim] กำลังรันทำนาย...")
    for a in accels:
        med, mx, on = _run_chase(a, args, sim=True)
        sim_rows[a] = (med, mx, on)
        print(f"  sim  accel={a:6.0f}  median={med}  on-target={on}%")

    real_rows = {}
    ser = None
    orig = {}
    try:
        if not args.sim_only:
            if serial is None:
                print("❌ ไม่มี pyserial — ใช้ --sim-only หรือ pip install pyserial")
                return
            port, baud = _port_baud()
            print(f"\n[real] เปิด {port}@{baud} เพื่ออ่าน/ตั้ง accel...")
            ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(0.3)
            orig = _read_accel(ser)
            if not orig:
                print("❌ อ่าน $120/$121 เดิมไม่ได้ — ยกเลิก (กันคืนค่าไม่ได้)")
                return
            print(f"[real] ค่าเดิม: $120={orig.get(120)} $121={orig.get(121)}  (จะคืนค่านี้เมื่อจบ)")

            print(f"\n⚠️  จะ 'หมุนแขนจริง' {len(accels)} รอบ — จัดแขนกลาง/ปลอดภัย, ดู stall/ไฟตก")
            for i in range(args.countdown, 0, -1):
                print(f"   ...{i}", flush=True); time.sleep(1.0)

            for a in accels:
                print(f"\n[real] === accel={a:g} deg/s^2 ===")
                if not _set_accel(ser, a):
                    real_rows[a] = (None, None, None); continue
                # ปิดพอร์ตชั่วคราวให้ 24 จับพอร์ตได้
                ser.close()
                med, mx, on = _run_chase(a, args, sim=False)
                real_rows[a] = (med, mx, on)
                print(f"  REAL accel={a:6.0f}  median={med}  max={mx}  on-target={on}%")
                ser = serial.Serial(port, baud, timeout=1.0); time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n[real] ยกเลิกด้วย Ctrl-C — กำลังคืนค่า accel เดิม...")
    finally:
        if ser is not None and orig:
            try:
                if not ser.is_open:
                    port, baud = _port_baud()
                    ser = serial.Serial(port, baud, timeout=1.0); time.sleep(0.3)
                _set_accel(ser, orig[120])  # 120/121 เดิมถือว่าเท่ากัน (ตั้งทีละแกนด้านล่างถ้าต่าง)
                if orig.get(121) is not None and orig[121] != orig[120]:
                    _grbl_cmd(ser, f"$121={orig[121]:g}")
                print(f"[real] ✅ คืนค่า accel เดิม: $120={orig[120]} $121={orig.get(121)}")
            except Exception as e:
                print(f"[real] ⚠️ คืนค่า accel ไม่สำเร็จ: {e} — ตั้งเองด้วย $120={orig.get(120)} $121={orig.get(121)}")
            try:
                ser.close()
            except Exception:
                pass

    # --- สรุป ---
    print("\n" + "=" * 64)
    print(f"สรุป SWEEP (ω={args.omega_deg} deg/s, reticle≈1.0 deg)")
    print("=" * 64)
    hdr = f"{'accel':>8} | {'sim med':>8} {'sim on%':>8} | {'real med':>9} {'real max':>9} {'real on%':>9}"
    print(hdr); print("-" * len(hdr))
    for a in accels:
        sm = sim_rows.get(a, (None,)*3)
        rm = real_rows.get(a, (None,)*3)
        def f(x, w, d=2):
            return (f"{x:>{w}.{d}f}" if isinstance(x, float) and not math.isnan(x) else f"{'—':>{w}}")
        print(f"{a:>8.0f} | {f(sm[0],8)} {f(sm[2],8,1)} | "
              f"{f(rm[0],9)} {f(rm[1],9)} {f(rm[2],9,1)}")

    base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as fp:
        w = _csv.writer(fp)
        w.writerow(["accel_deg_s2", "sim_median", "sim_on_pct", "real_median", "real_max", "real_on_pct"])
        for a in accels:
            sm = sim_rows.get(a, (None,)*3); rm = real_rows.get(a, (None,)*3)
            w.writerow([a, sm[0], sm[2], rm[0], rm[1], rm[2]])
    print(f"\nบันทึก: {base}.csv")


if __name__ == "__main__":
    main()
