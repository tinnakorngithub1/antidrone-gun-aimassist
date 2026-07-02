"""
29_latency_sensitivity.py
Heatmap "latency budget": detection latency × ω (accel/rate คงที่) — ตอบว่า
"ต้องลด detection latency เหลือเท่าไหร่ ถึงจะเล็งตรงโดรนที่ความเร็ว X ได้".

- ใช้ pipeline เดียวกับ 28 (ChaseController: latency + det-fps ZOH + kalman_coast +
  lead + noise/miss) ผ่าน chase_metrics ของ 28 (one source of truth)
- lead = latency ต่อ cell (ชดเชยเต็ม = จุดทำงานจริง) → แสดง "ขีดจำกัด latency พื้นฐาน"
  แม้ feedforward สมบูรณ์ (เหลือ det-ZOH + velocity-extrapolation error + noise)
- contour ที่ median = reticle = ขอบ ON-TARGET; พิมพ์ "latency สูงสุดที่ on-target" ต่อ ω

ตัวอย่าง:
  python3 29_latency_sensitivity.py
  python3 29_latency_sensitivity.py --accel 200 --rate 50 --det-fps 15
  python3 29_latency_sensitivity.py --lead-frac 0.0   # ไม่ชดเชย (raw latency) ดูว่าแย่แค่ไหน
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("env_realistic", _HERE / "28_envelope_realistic.py")
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)
chase_metrics = _env.chase_metrics
_sim = _env._sim

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    plt = None


def main():
    ap = argparse.ArgumentParser(description="Latency-sensitivity heatmap: latency × ω (latency budget)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--omegas", default="5,10,15,20,25,30,35,40,45,50,60")
    ap.add_argument("--latencies", default="0,25,50,75,100,150,200,300", help="detection latency (ms)")
    ap.add_argument("--accel", type=float, default=200.0, help="arm accel คงที่ (deg/s²)")
    ap.add_argument("--rate", type=float, default=50.0, help="control rate Hz คงที่")
    ap.add_argument("--lead-frac", type=float, default=1.0,
                    help="lead = lead_frac × latency (1.0 = ชดเชยเต็ม, 0.0 = ไม่ชดเชย)")
    ap.add_argument("--det-fps", type=float, default=15.0)
    ap.add_argument("--controller", default="kalman_coast")
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.2)
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0)
    ap.add_argument("--reticle-deg", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--mark-latency", type=float, default=150.0, help="เส้น latency อ้างอิงระบบปัจจุบัน (ms)")
    ap.add_argument("--out", default="latency_sensitivity.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]
    omegas = [float(x) for x in str(args.omegas).split(",") if x.strip()]
    lats = [float(x) for x in str(args.latencies).split(",") if x.strip()]
    seeds = list(range(args.seeds))

    print("=" * 70)
    print(f"LATENCY SENSITIVITY — {cam['name']}  accel={args.accel:.0f}°/s²  rate={args.rate:.0f}Hz  "
          f"reticle={reticle:.2f}°")
    print(f"  pipeline: det {args.det_fps:.0f}fps, {args.controller}, noise {args.noise_px:.0f}px, "
          f"miss {args.miss_rate*100:.0f}%, lead={args.lead_frac:.1f}×latency, seeds {args.seeds}")
    print(f"  grid {len(omegas)} ω × {len(lats)} latency = {len(omegas)*len(lats)} cells")
    print("=" * 70)

    med = np.full((len(omegas), len(lats)), np.nan)
    on = np.full((len(omegas), len(lats)), np.nan)
    for i, w in enumerate(omegas):
        for j, lat in enumerate(lats):
            m, o = chase_metrics(w, args.accel, cam, args.max_pan_rate, args.max_tilt_rate,
                                 args.amp_pan, args.amp_tilt, reticle,
                                 lat, lat * args.lead_frac, args.det_fps, args.controller,
                                 args.noise_px, args.miss_rate, 1.0/args.rate, seeds,
                                 duration=args.duration_s)
            med[i, j], on[i, j] = m, o
        v = w*math.pi/180*args.range_m
        print(f"  ω={w:5.1f} (v≈{v:5.0f}): med " + " ".join(f"{med[i,j]:4.1f}" for j in range(len(lats))))

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega", "v_ms"] + [f"med_lat{int(l)}" for l in lats]
                     + [f"on_lat{int(l)}" for l in lats])
        for i, w in enumerate(omegas):
            v = w*math.pi/180*args.range_m
            wtr.writerow([w, "%.0f" % v] + ["%.2f" % med[i, j] for j in range(len(lats))]
                         + ["%.0f" % on[i, j] for j in range(len(lats))])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is not None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7))
        xext = np.arange(len(lats)+1)-0.5
        yext = np.arange(len(omegas)+1)-0.5
        XX, YY = np.meshgrid(np.arange(len(lats)), np.arange(len(omegas)))
        vmax = min(np.nanmax(med), 14.0)
        im = axL.pcolormesh(xext, yext, np.clip(med, 0.05, None),
                            norm=LogNorm(vmin=max(0.2, reticle*0.3), vmax=max(vmax, reticle*2)),
                            cmap="RdYlGn_r", shading="flat")
        fig.colorbar(im, ax=axL, label="median error (deg, log)")
        for i in range(len(omegas)):
            for j in range(len(lats)):
                axL.text(j, i, f"{med[i,j]:.1f}", ha="center", va="center", fontsize=7,
                         color="black" if med[i, j] <= reticle*2.2 else "white")
        try:
            cs = axL.contour(XX, YY, med, levels=[reticle], colors="black", linewidths=2.4)
            axL.clabel(cs, fmt=f"ON-TARGET (med≤{reticle:.2f}°)", fontsize=8)
        except Exception:
            pass
        # เส้น latency อ้างอิงระบบปัจจุบัน
        if args.mark_latency in lats:
            jx = lats.index(args.mark_latency)
            axL.axvline(jx, color="magenta", ls="--", lw=1.8)
            axL.text(jx, len(omegas)-0.6, f"current\n~{args.mark_latency:.0f}ms", color="magenta",
                     fontsize=7.5, ha="center", va="top")
        axL.set_title(f"Median error vs latency — accel {args.accel:.0f}, rate {args.rate:.0f}Hz, "
                      f"lead={args.lead_frac:.1f}×lat\n{args.controller}, det {args.det_fps:.0f}fps, "
                      f"noise {args.noise_px:.0f}px, miss {args.miss_rate*100:.0f}% | ซ้าย=latency ต่ำ=ดี", fontsize=9)

        im2 = axR.pcolormesh(xext, yext, on, cmap="RdYlGn", vmin=0, vmax=100, shading="flat")
        fig.colorbar(im2, ax=axR, label="on-target %")
        for i in range(len(omegas)):
            for j in range(len(lats)):
                axR.text(j, i, f"{on[i,j]:.0f}", ha="center", va="center", fontsize=7,
                         color="black" if on[i, j] > 30 else "white")
        axR.set_title("On-target % (err ≤ reticle)", fontsize=9)

        for ax in (axL, axR):
            ax.set_xticks(range(len(lats))); ax.set_xticklabels([f"{int(l)}" for l in lats])
            ax.set_yticks(range(len(omegas)))
            ax.set_yticklabels([f"{w:.0f} (v≈{w*math.pi/180*args.range_m:.0f})" for w in omegas])
            ax.set_xlabel("detection latency (ms)")
            ax.set_ylabel(f"ω (deg/s)  [v=โดรน m/s @ {args.range_m:.0f}m]")
        fig.suptitle(f"Latency budget — {cam['name']}  (accel {args.accel:.0f}, rate {args.rate:.0f}Hz fixed)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(args.out, dpi=130)
        print(f"บันทึก PNG: {args.out}")

    # latency สูงสุดที่ on-target ต่อ ω = "latency budget"
    print(f"\nlatency สูงสุดที่ median ≤ reticle ({reticle:.2f}°) ต่อ ω  [accel {args.accel:.0f}, rate {args.rate:.0f}Hz]:")
    for i, w in enumerate(omegas):
        v = w*math.pi/180*args.range_m
        ok = [lats[j] for j in range(len(lats)) if med[i, j] <= reticle]
        budget = f"≤ {max(ok):.0f} ms" if ok else "✗ (แม้ latency=0 ก็ไม่ถึง — ลด noise/ω หรือเพิ่ม accel/rate)"
        print(f"  ω={w:5.1f} (v≈{v:5.0f} m/s):  latency {budget}")


if __name__ == "__main__":
    main()
