"""
32_controllers_compare.py
เทียบ controller หลายตัว (default: kalman, kalman_coast, kalman_coast_bounded) บนกริด ω × det-fps
— 1 panel median error ต่อ controller + 1 panel "winner" (ตัวไหน error ต่ำสุดในแต่ละช่อง).
ใช้ตรวจว่า kalman_coast_bounded ปิด "red zone" ของ coast (fps ต่ำ+ω สูง) ได้โดยไม่เสีย sweet zone.

reuse chase_metrics ของ 28 (pipeline เดียวกับ envelope จริง).

ตัวอย่าง:
  python3 32_controllers_compare.py --coast-tau 0.2
  python3 32_controllers_compare.py --controllers kalman,kalman_coast,kalman_coast_bounded --coast-tau 0.2
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
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
except ImportError:
    plt = None

_SHORT = {"kalman": "kal", "kalman_coast": "coast", "kalman_coast_bounded": "bnd",
          "plead": "plead", "p": "p"}


def main():
    ap = argparse.ArgumentParser(description="เทียบหลาย controller (ω×det-fps) + winner map")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--controllers", default="kalman,kalman_coast,kalman_coast_bounded")
    ap.add_argument("--omegas", default="5,10,15,20,25,30")
    ap.add_argument("--det-fps-list", default="3,5,8,12,15,20,30")
    ap.add_argument("--coast-tau", type=float, default=0.30)
    ap.add_argument("--accel", type=float, default=200.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--latency-ms", type=float, default=100.0)
    ap.add_argument("--lead-frac", type=float, default=1.0)
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.3)
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0)
    ap.add_argument("--reticle-deg", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--out", default="controllers_compare.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]
    ctrls = [c.strip() for c in str(args.controllers).split(",") if c.strip()]
    omegas = [float(x) for x in str(args.omegas).split(",") if x.strip()]
    fpss = [float(x) for x in str(args.det_fps_list).split(",") if x.strip()]
    seeds = list(range(args.seeds))

    print("=" * 70)
    print(f"CONTROLLERS COMPARE — {cam['name']}  accel {args.accel:.0f} rate {args.rate:.0f}Hz "
          f"lat {args.latency_ms:.0f}ms miss {args.miss_rate*100:.0f}% | coast_tau {args.coast_tau:.2f}")
    print(f"  {ctrls}  | grid {len(omegas)}ω × {len(fpss)}fps × {len(ctrls)} ctrl, seeds {args.seeds}")
    print("=" * 70)

    grids = {}
    for c in ctrls:
        med = np.full((len(omegas), len(fpss)), np.nan)
        for i, w in enumerate(omegas):
            for j, fps in enumerate(fpss):
                med[i, j] = chase_metrics(w, args.accel, cam, args.max_pan_rate, args.max_tilt_rate,
                                          args.amp_pan, args.amp_tilt, reticle,
                                          args.latency_ms, args.latency_ms*args.lead_frac, fps, c,
                                          args.noise_px, args.miss_rate, 1.0/args.rate, seeds,
                                          duration=args.duration_s, coast_tau=args.coast_tau)[0]
        grids[c] = med
        print(f"  {c}: mean median {np.nanmean(med):.2f}°")

    stack = np.stack([grids[c] for c in ctrls], axis=0)  # (ctrl, ω, fps)
    winner = np.argmin(stack, axis=0)

    # เทียบ bounded vs coast/kalman ในโซนอันตราย
    if "kalman_coast_bounded" in ctrls and "kalman_coast" in ctrls and "kalman" in ctrls:
        mk, mc, mb = grids["kalman"], grids["kalman_coast"], grids["kalman_coast_bounded"]
        danger = mc > mk + 1e-6
        sweet = mc < mk - 0.05
        print(f"\n  danger cells (coast แย่กว่า kalman): {int(danger.sum())} | "
              f"bounded ≤ kalman ในนั้น: {int((mb <= mk + 0.02)[danger].sum())}/{int(danger.sum())}")
        print(f"  sweet cells (coast ดีกว่า kalman): {int(sweet.sum())} | "
              f"bounded เก็บ ≥ครึ่ง advantage: {int((mb <= (mk+mc)/2)[sweet].sum())}/{int(sweet.sum())}")
        print(f"  mean regret bounded vs best(kalman,coast): "
              f"{np.nanmean(mb - np.minimum(mk, mc)):+.3f}° | worst {np.nanmax(mb - np.minimum(mk, mc)):+.2f}°")

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega", "det_fps"] + [f"med_{c}" for c in ctrls] + ["winner"])
        for i, w in enumerate(omegas):
            for j, fps in enumerate(fpss):
                wtr.writerow([w, fps] + ["%.3f" % grids[c][i, j] for c in ctrls] + [ctrls[winner[i, j]]])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is None:
        return
    n = len(ctrls)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.6*(n+1), 6.2))
    XX, YY = np.meshgrid(np.arange(len(fpss)), np.arange(len(omegas)))
    xe = np.arange(len(fpss)+1)-0.5
    ye = np.arange(len(omegas)+1)-0.5
    vmax = min(np.nanmax(stack), 12.0)
    norm = LogNorm(vmin=max(0.2, reticle*0.3), vmax=max(vmax, reticle*2))
    for k, c in enumerate(ctrls):
        ax = axes[k]
        M = grids[c]
        im = ax.pcolormesh(xe, ye, np.clip(M, 0.05, None), norm=norm, cmap="RdYlGn_r", shading="flat")
        for i in range(len(omegas)):
            for j in range(len(fpss)):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7,
                        color="black" if M[i, j] <= reticle*2.2 else "white")
        try:
            cs = ax.contour(XX, YY, M, levels=[reticle], colors="black", linewidths=2.0)
            ax.clabel(cs, fmt=f"≤{reticle:.2f}°", fontsize=7)
        except Exception:
            pass
        ax.set_title(f"median — {c}", fontsize=9.5)
    fig.colorbar(im, ax=axes[n-1], label="median error (deg, log)")

    # winner panel
    cmap = ListedColormap(["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:n])
    axw = axes[n]
    axw.pcolormesh(xe, ye, winner, cmap=cmap, norm=BoundaryNorm(range(n+1), n), shading="flat")
    for i in range(len(omegas)):
        for j in range(len(fpss)):
            axw.text(j, i, _SHORT.get(ctrls[winner[i, j]], ctrls[winner[i, j]][:4]),
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    axw.set_title("winner (error ต่ำสุด)", fontsize=9.5)
    from matplotlib.patches import Patch
    axw.legend(handles=[Patch(color=cmap(k), label=ctrls[k]) for k in range(n)],
               fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)

    for ax in axes:
        ax.set_xticks(range(len(fpss))); ax.set_xticklabels([f"{int(f)}" for f in fpss])
        ax.set_yticks(range(len(omegas)))
        ax.set_yticklabels([f"{w:.0f} (v≈{w*math.pi/180*args.range_m:.0f})" for w in omegas])
        ax.set_xlabel("det fps  (ซ้าย=ต่ำ)")
    axes[0].set_ylabel(f"ω (deg/s)  [v m/s @ {args.range_m:.0f}m]")
    fig.suptitle(f"Controller comparison — {cam['name']}  (lat {args.latency_ms:.0f}ms, miss "
                 f"{args.miss_rate*100:.0f}%, coast_tau {args.coast_tau:.2f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=130)
    print(f"บันทึก PNG: {args.out}")


if __name__ == "__main__":
    main()
