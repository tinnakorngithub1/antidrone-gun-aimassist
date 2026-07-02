"""
30_noise_detfps_sensitivity.py
Heatmap "detection-precision wall": noise_px × det_fps (ต่อ ω 1 ค่า/panel) —
ตอบว่าโดรนเร็ว (ω สูง) ที่ติด noise/fps wall ต้อง "ลด bbox noise / เพิ่ม det-fps"
เท่าไหร่ถึงจะเล็งตรง.

- latency=0 (default) เพื่อ ISOLATE มิติ noise/fps ออกจาก latency (เพดานพื้นฐาน)
  → real latency จะทำให้แย่ลงอีก (ดู 29 สำหรับมิติ latency)
- 1 panel ต่อ ω (default 15,20,30) เห็นว่าโซน on-target หดเมื่อโดรนเร็วขึ้น
- contour ที่ median=reticle = ขอบ ON-TARGET (มุมล่างขวา = noise ต่ำ+fps สูง = ดี)

ตัวอย่าง:
  python3 30_noise_detfps_sensitivity.py
  python3 30_noise_detfps_sensitivity.py --omega-panels 15,20,25,30 --latency-ms 50
  python3 30_noise_detfps_sensitivity.py --accel 200 --rate 50 --controller kalman_coast
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
    ap = argparse.ArgumentParser(description="noise×det-fps sensitivity (detection-precision wall ต่อ ω)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--omega-panels", default="15,20,30", help="ω (deg/s) — 1 panel ต่อค่า")
    ap.add_argument("--noises", default="0,2,4,6,8,12,16", help="bbox center jitter (px)")
    ap.add_argument("--det-fps-list", default="5,10,15,20,30,45,60")
    ap.add_argument("--accel", type=float, default=200.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--latency-ms", type=float, default=0.0, help="คงที่ (default 0 = isolate noise/fps)")
    ap.add_argument("--lead-frac", type=float, default=1.0)
    ap.add_argument("--controller", default="kalman_coast")
    ap.add_argument("--miss-rate", type=float, default=0.2)
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0)
    ap.add_argument("--reticle-deg", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--out", default="noise_detfps_sensitivity.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]
    omegas = [float(x) for x in str(args.omega_panels).split(",") if x.strip()]
    noises = [float(x) for x in str(args.noises).split(",") if x.strip()]
    fpss = [float(x) for x in str(args.det_fps_list).split(",") if x.strip()]
    seeds = list(range(args.seeds))

    print("=" * 70)
    print(f"NOISE×DET-FPS — {cam['name']}  accel={args.accel:.0f}  rate={args.rate:.0f}Hz  "
          f"latency={args.latency_ms:.0f}ms  reticle={reticle:.2f}°")
    print(f"  {args.controller}, miss {args.miss_rate*100:.0f}%, seeds {args.seeds} | "
          f"panels ω={omegas}  ({len(noises)}×{len(fpss)} cells/panel)")
    print("=" * 70)

    grids = {}
    for w in omegas:
        med = np.full((len(noises), len(fpss)), np.nan)
        for i, npx in enumerate(noises):
            for j, fps in enumerate(fpss):
                m, _ = chase_metrics(w, args.accel, cam, args.max_pan_rate, args.max_tilt_rate,
                                     args.amp_pan, args.amp_tilt, reticle,
                                     args.latency_ms, args.latency_ms*args.lead_frac, fps,
                                     args.controller, npx, args.miss_rate, 1.0/args.rate, seeds,
                                     duration=args.duration_s)
                med[i, j] = m
        grids[w] = med
        v = w*math.pi/180*args.range_m
        print(f"\n  ω={w:.0f} (v≈{v:.0f} m/s)  median error [แถว=noise px, คอลัมน์=det fps]:")
        print("        fps:" + "".join(f"{int(f):>6}" for f in fpss))
        for i, npx in enumerate(noises):
            print(f"    {npx:4.0f}px:" + "".join(f"{med[i,j]:6.1f}" for j in range(len(fpss))))

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega", "noise_px"] + [f"med_fps{int(fp)}" for fp in fpss])
        for w in omegas:
            for i, npx in enumerate(noises):
                wtr.writerow([w, npx] + ["%.2f" % grids[w][i, j] for j in range(len(fpss))])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is not None:
        n = len(omegas)
        fig, axes = plt.subplots(1, n, figsize=(5.2*n, 6.2), squeeze=False)
        axes = axes[0]
        XX, YY = np.meshgrid(np.arange(len(fpss)), np.arange(len(noises)))
        for k, w in enumerate(omegas):
            ax = axes[k]
            med = grids[w]
            vmax = min(np.nanmax(med), 10.0)
            im = ax.pcolormesh(np.arange(len(fpss)+1)-0.5, np.arange(len(noises)+1)-0.5,
                               np.clip(med, 0.05, None),
                               norm=LogNorm(vmin=max(0.2, reticle*0.3), vmax=max(vmax, reticle*2)),
                               cmap="RdYlGn_r", shading="flat")
            for i in range(len(noises)):
                for j in range(len(fpss)):
                    ax.text(j, i, f"{med[i,j]:.1f}", ha="center", va="center", fontsize=7,
                            color="black" if med[i, j] <= reticle*2.2 else "white")
            try:
                cs = ax.contour(XX, YY, med, levels=[reticle], colors="black", linewidths=2.4)
                ax.clabel(cs, fmt=f"ON-TARGET ≤{reticle:.2f}°", fontsize=7.5)
            except Exception:
                pass
            v = w*math.pi/180*args.range_m
            ax.set_title(f"ω={w:.0f}  (v≈{v:.0f} m/s @ {args.range_m:.0f}m)", fontsize=10)
            ax.set_xticks(range(len(fpss))); ax.set_xticklabels([f"{int(f)}" for f in fpss])
            ax.set_yticks(range(len(noises))); ax.set_yticklabels([f"{int(npx)}" for npx in noises])
            ax.set_xlabel("det fps  →  เพิ่ม = ดี")
            if k == 0:
                ax.set_ylabel("bbox noise (px)  →  ลด = ดี")
        fig.colorbar(im, ax=axes[-1], label="median error (deg, log)")
        fig.suptitle(f"Detection-precision wall — {cam['name']}  (accel {args.accel:.0f}, rate {args.rate:.0f}Hz, "
                     f"latency {args.latency_ms:.0f}ms; มุมล่างขวา=noise ต่ำ+fps สูง=ดีสุด)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(args.out, dpi=130)
        print(f"บันทึก PNG: {args.out}")

    # สรุป: เงื่อนไข noise/fps ที่ on-target ต่อ ω
    print(f"\nเงื่อนไข on-target (median ≤ {reticle:.2f}°) ต่อ ω  [latency {args.latency_ms:.0f}ms]:")
    for w in omegas:
        med = grids[w]
        oks = [(noises[i], fpss[j]) for i in range(len(noises)) for j in range(len(fpss)) if med[i, j] <= reticle]
        v = w*math.pi/180*args.range_m
        if not oks:
            print(f"  ω={w:4.0f} (v≈{v:4.0f}): ✗ ไม่ถึงในกริดนี้แม้ latency=0 (โดรนเร็วเกินระบบ)")
        else:
            max_noise = max(npx for npx, _ in oks)
            min_fps = min(fp for _, fp in oks)
            print(f"  ω={w:4.0f} (v≈{v:4.0f}): ต้อง noise ≤ {max_noise:.0f}px และ det-fps ≥ {min_fps:.0f}fps "
                  f"(มี {len(oks)} combo ที่ผ่าน)")


if __name__ == "__main__":
    main()
