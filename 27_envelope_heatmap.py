"""
27_envelope_heatmap.py
Heatmap "capability envelope" ของแขนกล: median tracking error (และ on-target%)
บนกริด ω (อัตราเชิงมุมเป้า) × accel (ความเร่งแขน GRBL $120/$121) — ใช้ digital twin
ที่ validate กับแขนจริงแล้ว (sub-step ArmModel + control-ZOH loop เดียวกับที่ fit จริง).

- กลไกล้วน (ไม่รวม detection latency) — ตอบว่า "แขนหมุนทัน/เล็งตรงไหม" ที่แต่ละ (ω,accel)
- เส้น contour ที่ median = reticle = ขอบ ON-TARGET (ใต้เส้น = เล็งตรงกลางได้)
- overlay จุดวัดจริง (✕ + ค่า) ให้เห็นว่า heatmap reproduce ของจริง
- แกน ω มี label ความเร็วโดรนที่ระยะ R (default 150 m) คู่กัน

ตัวอย่าง:
  python3 27_envelope_heatmap.py
  python3 27_envelope_heatmap.py --rate 50 --out envelope_rate50.png
  python3 27_envelope_heatmap.py --omegas 5,10,15,20,25,30,35,40,45,50,60 --accels 50,75,100,150,200,250,300,400
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

_SIM_PATH = Path(__file__).resolve().parent / "23_arm_chase_sim.py"
_spec = importlib.util.spec_from_file_location("arm_chase_sim", _SIM_PATH)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    plt = None

# จุดวัดจริง cam4 (amp_pan=20, amp_tilt=8, rate=20Hz) สำหรับ overlay validate
# (accel, omega, real_median_deg) — ω15 ใช้ amp8 จึงไม่รวม (amp ต่างทำให้ ω_temporal ต่าง)
REAL_PTS = [(100, 30, 3.32), (150, 30, 2.33), (200, 30, 1.93), (100, 45, 7.99)]


def chase_metrics(omega, accel, amp_pan, amp_tilt, rate, reticle,
                  max_pan, max_tilt, duration=12.0, warmup=2.0,
                  measure_interval=0.4, dt=0.0005):
    """รัน mechanical chase แบบ deterministic (validated model) คืน (median_err, on_target_pct).
    control ZOH ที่ rate Hz → แขน (ArmModel sub-step ภายใน) ไล่ตาม → วัด |target − pos|."""
    drone = _sim.DroneModel("sine", omega, fov_h=amp_pan / 0.6, fov_v=amp_tilt / 0.6,
                            tilt_amp_deg=amp_tilt)
    arm = _sim.ArmModel({
        "max_pan_rate": max_pan, "max_tilt_rate": max_tilt,
        "max_pan_accel": accel, "max_tilt_accel": accel,
        "x_lim": (-65.0, 65.0), "y_lim": (-35.0, 35.0),
    })
    period = 1.0 / rate
    intended = (0.0, 0.0)
    next_ctrl = 0.0
    next_meas = warmup
    errs = []
    t = 0.0
    for _ in range(int(duration / dt)):
        if t >= next_ctrl:
            intended = drone.angle_at(t)        # ZOH: ถือคำสั่งจนกว่าจะถึง tick ถัดไป
            next_ctrl += period
        arm.set_target(intended[0], intended[1])
        arm.step(dt)
        if t >= next_meas:
            next_meas += measure_interval
            tp, tt = drone.angle_at(t)
            errs.append(math.hypot(tp - arm.pan, tt - arm.tilt))
        t += dt
    if not errs:
        return float("nan"), float("nan")
    med = float(np.median(errs))
    on = 100.0 * sum(1 for e in errs if e <= reticle) / len(errs)
    return med, on


def main():
    ap = argparse.ArgumentParser(description="Heatmap envelope ω×accel (digital twin ที่ validate แล้ว)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--omegas", default="5,10,15,20,25,30,35,40,45,50,60",
                    help="ลิสต์ ω deg/s (แกน Y)")
    ap.add_argument("--accels", default="50,75,100,150,200,250,300,400",
                    help="ลิสต์ accel deg/s^2 (แกน X)")
    ap.add_argument("--rate", type=float, default=20.0, help="control rate Hz")
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0, help="ระยะโดรนสำหรับ label ความเร็ว")
    ap.add_argument("--reticle-deg", type=float, default=None)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--out", default="envelope_heatmap.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]

    omegas = [float(x) for x in str(args.omegas).split(",") if x.strip()]
    accels = [float(x) for x in str(args.accels).split(",") if x.strip()]

    print("=" * 64)
    print(f"ENVELOPE ω×accel — {cam['name']}  rate={args.rate}Hz  amp ±{args.amp_pan}/{args.amp_tilt}  "
          f"reticle={reticle:.2f}°")
    print(f"grid: {len(omegas)} ω × {len(accels)} accel = {len(omegas)*len(accels)} cells")
    print("=" * 64)

    med = np.full((len(omegas), len(accels)), np.nan)
    on = np.full((len(omegas), len(accels)), np.nan)
    for i, w in enumerate(omegas):
        row = []
        for j, a in enumerate(accels):
            m, o = chase_metrics(w, a, args.amp_pan, args.amp_tilt, args.rate, reticle,
                                 args.max_pan_rate, args.max_tilt_rate,
                                 duration=args.duration_s, dt=args.dt)
            med[i, j] = m
            on[i, j] = o
            row.append(f"{m:4.1f}")
        print(f"  ω={w:5.1f} (v≈{w*math.pi/180*args.range_m:5.0f}m/s): med[deg] " + " ".join(row))

    # CSV
    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega_deg_s", "drone_v_ms@%gm" % args.range_m] +
                     [f"med_acc{int(a)}" for a in accels] + [f"on_acc{int(a)}" for a in accels])
        for i, w in enumerate(omegas):
            v = w * math.pi / 180 * args.range_m
            wtr.writerow([w, "%.0f" % v] + ["%.2f" % med[i, j] for j in range(len(accels))]
                         + ["%.0f" % on[i, j] for j in range(len(accels))])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is None:
        print("  (ข้ามพล็อต — ไม่มี matplotlib)")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))
    xext = np.arange(len(accels) + 1) - 0.5
    yext = np.arange(len(omegas) + 1) - 0.5

    def _xidx(a):   # map accel value -> fractional x index (สำหรับวาดเส้นโค้ง)
        return float(np.interp(a, accels, range(len(accels)),
                               left=-0.5, right=len(accels) - 0.5))

    def _draw_saturation(ax):
        # เส้น accel = ω²/amp_pan : เหนือ/ขวา = ไม่อิ่มตัว (โมเดล validated),
        # ใต้/ซ้าย = accel ไม่พอ เร่งตามไม่ทันยอด (อิ่มตัว → โมเดล optimistic เทียบจริง)
        ys, xs = [], []
        for i, w in enumerate(omegas):
            a_dem = w * w / args.amp_pan
            ys.append(i)
            xs.append(_xidx(a_dem))
        ax.plot(xs, ys, "--", color="magenta", lw=1.8, zorder=6)
        ax.text(xs[-1], len(omegas) - 0.7, "accel saturation\n(ซ้ายเส้น: optimistic)",
                color="magenta", fontsize=7.5, ha="right", va="top")

    # ---- panel ซ้าย: median error (log color) + contour reticle ----
    vmax = min(np.nanmax(med), 12.0)
    im = axL.pcolormesh(xext, yext, np.clip(med, 0.05, None),
                        norm=LogNorm(vmin=max(0.2, reticle * 0.3), vmax=max(vmax, reticle * 2)),
                        cmap="RdYlGn_r", shading="flat")
    fig.colorbar(im, ax=axL, label="median error (deg, log)")
    # contour ON-TARGET boundary (median = reticle) บนพิกัด index
    XX, YY = np.meshgrid(np.arange(len(accels)), np.arange(len(omegas)))
    try:
        cs = axL.contour(XX, YY, med, levels=[reticle], colors="black", linewidths=2.2)
        axL.clabel(cs, fmt=f"reticle {reticle:.2f}°  (ON-TARGET ↓)", fontsize=9)
    except Exception:
        pass
    # annotate ค่าในแต่ละช่อง
    for i in range(len(omegas)):
        for j in range(len(accels)):
            axL.text(j, i, f"{med[i, j]:.1f}", ha="center", va="center", fontsize=7,
                     color="black" if med[i, j] <= reticle * 2.2 else "white")
    # overlay จุดวัดจริง
    for a, w, rv in REAL_PTS:
        if a in accels and w in omegas:
            jj, ii = accels.index(a), omegas.index(w)
            axL.scatter([jj], [ii], s=240, facecolors="none", edgecolors="cyan", linewidths=2.5, zorder=5)
            axL.text(jj, ii - 0.34, f"real {rv:.1f}", ha="center", va="bottom", fontsize=7,
                     color="cyan", fontweight="bold")
    _draw_saturation(axL)
    axL.set_title(f"Median tracking error (deg) — rate {args.rate:.0f}Hz\n"
                  f"◯cyan = วัดจริง | ใต้เส้นดำ = ON-TARGET | --- magenta = accel saturation", fontsize=10)

    # ---- panel ขวา: on-target% ----
    im2 = axR.pcolormesh(xext, yext, on, cmap="RdYlGn", vmin=0, vmax=100, shading="flat")
    fig.colorbar(im2, ax=axR, label="on-target %")
    for i in range(len(omegas)):
        for j in range(len(accels)):
            axR.text(j, i, f"{on[i, j]:.0f}", ha="center", va="center", fontsize=7,
                     color="black" if on[i, j] > 30 else "white")
    _draw_saturation(axR)
    axR.set_title("On-target % (err ≤ reticle)", fontsize=10)

    for ax in (axL, axR):
        ax.set_xticks(range(len(accels)))
        ax.set_xticklabels([f"{int(a)}" for a in accels])
        ax.set_yticks(range(len(omegas)))
        ax.set_yticklabels([f"{w:.0f}  (v≈{w*math.pi/180*args.range_m:.0f})" for w in omegas])
        ax.set_xlabel("arm accel  $120/$121 (deg/s²)")
        ax.set_ylabel(f"ω (deg/s)   [v = โดรน m/s @ {args.range_m:.0f}m]")
        ax.axhline(-0.5, color="k", lw=0.5)

    fig.suptitle(f"Arm tracking envelope — {cam['name']}  (mechanical only; +detection latency 100-300ms ในระบบจริง)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"บันทึก PNG: {args.out}")

    # สรุป: accel ต่ำสุดต่อ ω ที่ทำให้ on-target (median ≤ reticle)
    print("\naccel ต่ำสุดที่ median ≤ reticle (ต่อ ω):")
    for i, w in enumerate(omegas):
        ok = [accels[j] for j in range(len(accels)) if med[i, j] <= reticle]
        v = w * math.pi / 180 * args.range_m
        print(f"  ω={w:5.1f} (v≈{v:5.0f} m/s): " +
              (f"accel ≥ {min(ok):.0f}°/s²" if ok else "ไม่ถึงในกริดนี้ (ต้องลด ω / เพิ่ม rate / ปรับ controller)"))


if __name__ == "__main__":
    main()
