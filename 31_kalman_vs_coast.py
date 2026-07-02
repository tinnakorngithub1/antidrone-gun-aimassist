"""
31_kalman_vs_coast.py
เทียบ controller kalman vs kalman_coast บนกริด ω × det-fps — โดยเฉพาะตอน det-fps ต่ำ
+ มี YOLO miss: ช่วง miss ที่ det-fps ต่ำจะนาน → kalman_coast (predict ต่อด้วย velocity)
ควรชนะ kalman (hold ค่าเก่านิ่ง) ชัดขึ้นเรื่อย ๆ.

3 panel: median[kalman] | median[kalman_coast] | improvement = kalman − coast (เขียว=coast ดีกว่า)
contour ขอบ on-target (median=reticle) ของแต่ละ controller บน panel ตัวเอง.
reuse chase_metrics ของ 28 (pipeline เดียวกับ envelope จริง).

ตัวอย่าง:
  python3 31_kalman_vs_coast.py
  python3 31_kalman_vs_coast.py --miss-rate 0.4 --latency-ms 150 --det-fps-list 3,5,8,12,15,20,30
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
    from matplotlib.colors import LogNorm, TwoSlopeNorm
except ImportError:
    plt = None


def main():
    ap = argparse.ArgumentParser(description="kalman vs kalman_coast envelope (ω × det-fps, ตอน fps ต่ำ+miss)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--omegas", default="5,10,15,20,25,30")
    ap.add_argument("--det-fps-list", default="3,5,8,12,15,20,30")
    ap.add_argument("--accel", type=float, default=200.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--latency-ms", type=float, default=100.0)
    ap.add_argument("--lead-frac", type=float, default=1.0)
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.3, help="ต้อง >0 ถึงจะเห็นข้อดี coast")
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0)
    ap.add_argument("--reticle-deg", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--out", default="kalman_vs_coast.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]
    omegas = [float(x) for x in str(args.omegas).split(",") if x.strip()]
    fpss = [float(x) for x in str(args.det_fps_list).split(",") if x.strip()]
    seeds = list(range(args.seeds))

    print("=" * 70)
    print(f"KALMAN vs KALMAN_COAST — {cam['name']}  accel={args.accel:.0f} rate={args.rate:.0f}Hz "
          f"lat={args.latency_ms:.0f}ms")
    print(f"  noise {args.noise_px:.0f}px, miss {args.miss_rate*100:.0f}%, seeds {args.seeds}, "
          f"reticle {reticle:.2f}° | grid {len(omegas)}ω × {len(fpss)}fps × 2 controller")
    print("=" * 70)

    def run(ctrl):
        med = np.full((len(omegas), len(fpss)), np.nan)
        on = np.full((len(omegas), len(fpss)), np.nan)
        for i, w in enumerate(omegas):
            for j, fps in enumerate(fpss):
                m, o = chase_metrics(w, args.accel, cam, args.max_pan_rate, args.max_tilt_rate,
                                     args.amp_pan, args.amp_tilt, reticle,
                                     args.latency_ms, args.latency_ms*args.lead_frac, fps, ctrl,
                                     args.noise_px, args.miss_rate, 1.0/args.rate, seeds,
                                     duration=args.duration_s)
                med[i, j], on[i, j] = m, o
        return med, on

    print("[1/2] kalman...")
    mk, ok = run("kalman")
    print("[2/2] kalman_coast...")
    mc, oc = run("kalman_coast")
    impr = mk - mc  # >0 = coast ดีกว่า (error ต่ำกว่า)

    print(f"\n  improvement = median[kalman] − median[kalman_coast]  (>0 = coast ชนะ) [แถว ω, คอลัมน์ fps]:")
    print("        fps:" + "".join(f"{int(f):>6}" for f in fpss))
    for i, w in enumerate(omegas):
        print(f"   ω{w:4.0f}  :" + "".join(f"{impr[i,j]:6.2f}" for j in range(len(fpss))))

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega", "det_fps", "med_kalman", "med_coast", "improvement",
                      "on_kalman", "on_coast"])
        for i, w in enumerate(omegas):
            for j, fps in enumerate(fpss):
                wtr.writerow([w, fps, "%.3f" % mk[i, j], "%.3f" % mc[i, j], "%.3f" % impr[i, j],
                              "%.0f" % ok[i, j], "%.0f" % oc[i, j]])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
        XX, YY = np.meshgrid(np.arange(len(fpss)), np.arange(len(omegas)))
        xe = np.arange(len(fpss)+1)-0.5
        ye = np.arange(len(omegas)+1)-0.5
        vmax = min(max(np.nanmax(mk), np.nanmax(mc)), 12.0)
        norm = LogNorm(vmin=max(0.2, reticle*0.3), vmax=max(vmax, reticle*2))
        for ax, M, ttl in ((axes[0], mk, "kalman"), (axes[1], mc, "kalman_coast")):
            im = ax.pcolormesh(xe, ye, np.clip(M, 0.05, None), norm=norm, cmap="RdYlGn_r", shading="flat")
            for i in range(len(omegas)):
                for j in range(len(fpss)):
                    ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7,
                            color="black" if M[i, j] <= reticle*2.2 else "white")
            try:
                cs = ax.contour(XX, YY, M, levels=[reticle], colors="black", linewidths=2.2)
                ax.clabel(cs, fmt=f"on-target ≤{reticle:.2f}°", fontsize=7)
            except Exception:
                pass
            ax.set_title(f"median error — {ttl}", fontsize=10)
        fig.colorbar(im, ax=axes[1], label="median error (deg, log)")

        # panel 3: improvement (coast − kalman, เขียว = coast ดีกว่า)
        amax = max(np.nanmax(np.abs(impr)), 0.1)
        im3 = axes[2].pcolormesh(xe, ye, impr, cmap="RdYlGn",
                                 norm=TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax), shading="flat")
        for i in range(len(omegas)):
            for j in range(len(fpss)):
                axes[2].text(j, i, f"{impr[i,j]:+.1f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im3, ax=axes[2], label="median[kalman] − median[coast]  (เขียว=coast ดีกว่า)")
        axes[2].set_title("coast advantage (deg)", fontsize=10)

        for ax in axes:
            ax.set_xticks(range(len(fpss))); ax.set_xticklabels([f"{int(f)}" for f in fpss])
            ax.set_yticks(range(len(omegas)))
            ax.set_yticklabels([f"{w:.0f} (v≈{w*math.pi/180*args.range_m:.0f})" for w in omegas])
            ax.set_xlabel("det fps  (ซ้าย=ต่ำ → miss-gap ยาว)")
        axes[0].set_ylabel(f"ω (deg/s)  [v=โดรน m/s @ {args.range_m:.0f}m]")
        fig.suptitle(f"kalman vs kalman_coast — {cam['name']}  (lat {args.latency_ms:.0f}ms, miss "
                     f"{args.miss_rate*100:.0f}%, noise {args.noise_px:.0f}px)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(args.out, dpi=130)
        print(f"บันทึก PNG: {args.out}")

    print(f"\nสรุป: coast advantage เฉลี่ยต่อ det-fps (median ข้าม ω):")
    for j, fps in enumerate(fpss):
        print(f"  det-fps {fps:5.0f}:  coast ดีกว่าเฉลี่ย {np.nanmean(impr[:,j]):+.2f}°")


if __name__ == "__main__":
    main()
