"""
28_envelope_realistic.py
Envelope ω×accel "ใช้งานจริง" — รวม detection pipeline (latency + det-fps ZOH +
Kalman/lead/coast + YOLO noise/miss) ทับบนโมเดลกลไกที่ validate แล้ว.

เทียบ 2 ฉาก:
  - CEILING  : กลไกล้วน (latency 0, detection สมบูรณ์) = เพดานจาก 27
  - REALISTIC: latency จริง + det-fps + controller (kalman_coast) + lead + noise/miss

ใช้ `ChaseController/ArmModel/DroneModel` จาก 23_arm_chase_sim.py (loop เดียวกับ run_sim).
เฉลี่ยหลาย seed (noise/miss เป็น stochastic). ออก PNG (median + on-target% + ขอบ
on-target ของ ceiling vs realistic ซ้อนกัน) + CSV.

ตัวอย่าง:
  python3 28_envelope_realistic.py
  python3 28_envelope_realistic.py --latency-ms 250 --det-fps 15 --controller kalman_coast
  python3 28_envelope_realistic.py --lead-ms 0 --controller p   # naive (ไม่ชดเชย) ดูว่าแย่แค่ไหน
"""
import argparse
import importlib.util
import math
from collections import deque
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

REAL_PTS = [(100, 30, 3.32), (150, 30, 2.33), (200, 30, 1.93), (100, 45, 7.99)]  # กลไกล้วน (อ้างอิง)


def chase_metrics(omega, accel, cam, max_pan, max_tilt, amp_pan, amp_tilt, reticle,
                  latency_ms, lead_ms, det_fps, controller, noise_px, miss_rate, cmd_interval,
                  seeds, duration=12.0, warmup=3.0, sim_dt=1.0/200, alpha=0.5, beta=0.08,
                  coast_tau=0.30):
    """รัน full pipeline loop (เหมือน 23.run_sim) เฉลี่ยหลาย seed → (median_err, on_target_pct)."""
    ppd_x, ppd_y = cam["ppd_x"], cam["ppd_y"]
    fov_h, fov_v = cam["fov_h"], cam["fov_v"]
    all_err, on, tot = [], 0, 0
    for sd in seeds:
        drone = _sim.DroneModel("sine", omega, fov_h=amp_pan/0.6, fov_v=amp_tilt/0.6, tilt_amp_deg=amp_tilt)
        arm = _sim.ArmModel({"max_pan_rate": max_pan, "max_tilt_rate": max_tilt,
                             "max_pan_accel": accel, "max_tilt_accel": accel,
                             "x_lim": (-65.0, 65.0), "y_lim": (-35.0, 35.0)})
        ctrl = _sim.ChaseController(ppd_x, ppd_y, 1.0, lead_ms/1000.0, cmd_interval,
                                    latency_ms/1000.0, 1.0/max(det_fps, 1e-6),
                                    noise_px=noise_px, miss_rate=miss_rate, seed=sd,
                                    mode=controller, alpha=alpha, beta=beta, coast_tau=coast_tau)
        hist = deque(maxlen=int(2.0/sim_dt))
        t = 0.0
        while t < duration:
            dpan, dtilt = drone.angle_at(t)
            hist.append((t, dpan, dtilt, arm.pan, arm.tilt))
            ctrl.update(t, arm, hist)
            arm.step(sim_dt)
            if t >= warmup:
                epan, etilt = dpan - arm.pan, dtilt - arm.tilt
                err = math.hypot(epan, etilt)
                all_err.append(err)
                tot += 1
                if err <= reticle:
                    on += 1
            t += sim_dt
    if not all_err:
        return float("nan"), float("nan")
    return float(np.median(all_err)), 100.0*on/tot


def _grid(omegas, accels, cam, args, scenario, seeds):
    med = np.full((len(omegas), len(accels)), np.nan)
    on = np.full((len(omegas), len(accels)), np.nan)
    for i, w in enumerate(omegas):
        for j, a in enumerate(accels):
            m, o = chase_metrics(w, a, cam, args.max_pan_rate, args.max_tilt_rate,
                                 args.amp_pan, args.amp_tilt, args._reticle,
                                 scenario["latency_ms"], scenario["lead_ms"], scenario["det_fps"],
                                 scenario["controller"], scenario["noise_px"], scenario["miss_rate"],
                                 1.0/args.rate, seeds, duration=args.duration_s)
            med[i, j], on[i, j] = m, o
    return med, on


def main():
    ap = argparse.ArgumentParser(description="Envelope ω×accel ใช้งานจริง (รวม detection latency)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--omegas", default="5,10,15,20,25,30,35,40,45,50,60")
    ap.add_argument("--accels", default="50,75,100,150,200,250,300,400")
    ap.add_argument("--rate", type=float, default=50.0, help="control/command rate Hz (cmd throttle=1/rate)")
    ap.add_argument("--amp-pan", type=float, default=20.0)
    ap.add_argument("--amp-tilt", type=float, default=8.0)
    ap.add_argument("--max-pan-rate", type=float, default=80.0)
    ap.add_argument("--max-tilt-rate", type=float, default=60.0)
    ap.add_argument("--range-m", type=float, default=150.0)
    ap.add_argument("--reticle-deg", type=float, default=None)
    # realistic pipeline
    ap.add_argument("--latency-ms", type=float, default=150.0, help="detection latency จริง (ms)")
    ap.add_argument("--lead-ms", type=float, default=None, help="feedforward lead (default = latency = ชดเชยเต็ม)")
    ap.add_argument("--det-fps", type=float, default=15.0)
    ap.add_argument("--controller", default="kalman_coast", choices=["p", "plead", "kalman", "kalman_coast"])
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=3, help="จำนวน seed เฉลี่ย (noise/miss stochastic)")
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--out", default="envelope_realistic.png")
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    args._reticle = args.reticle_deg if args.reticle_deg is not None else \
        max(10, min(cam["width"], cam["height"]) * 0.03) / cam["ppd_x"]
    lead = args.latency_ms if args.lead_ms is None else args.lead_ms
    omegas = [float(x) for x in str(args.omegas).split(",") if x.strip()]
    accels = [float(x) for x in str(args.accels).split(",") if x.strip()]
    seeds = list(range(args.seeds))

    CEIL = dict(latency_ms=0.0, lead_ms=0.0, det_fps=500.0, controller="p", noise_px=0.0, miss_rate=0.0)
    REAL = dict(latency_ms=args.latency_ms, lead_ms=lead, det_fps=args.det_fps,
                controller=args.controller, noise_px=args.noise_px, miss_rate=args.miss_rate)

    print("=" * 70)
    print(f"REALISTIC ENVELOPE — {cam['name']}  rate={args.rate}Hz  amp±{args.amp_pan}/{args.amp_tilt}  "
          f"reticle={args._reticle:.2f}°")
    print(f"  REALISTIC: latency {args.latency_ms:.0f}ms, lead {lead:.0f}ms, det {args.det_fps:.0f}fps, "
          f"{args.controller}, noise {args.noise_px:.0f}px, miss {args.miss_rate*100:.0f}%, seeds {args.seeds}")
    print(f"  grid {len(omegas)}×{len(accels)} = {len(omegas)*len(accels)} cells × 2 ฉาก")
    print("=" * 70)

    print("[1/2] CEILING (กลไกล้วน)...")
    med_c, on_c = _grid(omegas, accels, cam, args, CEIL, seeds=[0])
    print("[2/2] REALISTIC (มี latency)...")
    med_r, on_r = _grid(omegas, accels, cam, args, REAL, seeds=seeds)

    for i, w in enumerate(omegas):
        v = w*math.pi/180*args.range_m
        print(f"  ω={w:5.1f} (v≈{v:5.0f}): ceil med " +
              " ".join(f"{med_c[i,j]:4.1f}" for j in range(len(accels))) +
              " | real med " + " ".join(f"{med_r[i,j]:4.1f}" for j in range(len(accels))))

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    import csv as _csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["omega", "v_ms"] + [f"ceil_med_a{int(a)}" for a in accels]
                     + [f"real_med_a{int(a)}" for a in accels] + [f"real_on_a{int(a)}" for a in accels])
        for i, w in enumerate(omegas):
            v = w*math.pi/180*args.range_m
            wtr.writerow([w, "%.0f" % v] + ["%.2f" % med_c[i, j] for j in range(len(accels))]
                         + ["%.2f" % med_r[i, j] for j in range(len(accels))]
                         + ["%.0f" % on_r[i, j] for j in range(len(accels))])
    print(f"\nบันทึก CSV: {base}.csv")

    if plt is not None:
        _plot(omegas, accels, med_r, on_r, med_c, cam, args, lead)
        print(f"บันทึก PNG: {args.out}")

    # สรุปเปรียบเทียบ accel ต่ำสุดที่ on-target
    print(f"\naccel ต่ำสุดที่ median ≤ reticle ({args._reticle:.2f}°)  [CEILING → REALISTIC]:")
    for i, w in enumerate(omegas):
        v = w*math.pi/180*args.range_m
        okc = [accels[j] for j in range(len(accels)) if med_c[i, j] <= args._reticle]
        okr = [accels[j] for j in range(len(accels)) if med_r[i, j] <= args._reticle]
        sc = f"≥{min(okc):.0f}" if okc else "✗"
        sr = f"≥{min(okr):.0f}" if okr else "✗"
        print(f"  ω={w:5.1f} (v≈{v:5.0f} m/s):  ceiling {sc:>5}   →   realistic {sr:>5}")


def _plot(omegas, accels, med_r, on_r, med_c, cam, args, lead):
    reticle = args._reticle
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7))
    xext = np.arange(len(accels)+1)-0.5
    yext = np.arange(len(omegas)+1)-0.5
    XX, YY = np.meshgrid(np.arange(len(accels)), np.arange(len(omegas)))

    vmax = min(np.nanmax(med_r), 14.0)
    im = axL.pcolormesh(xext, yext, np.clip(med_r, 0.05, None),
                        norm=LogNorm(vmin=max(0.2, reticle*0.3), vmax=max(vmax, reticle*2)),
                        cmap="RdYlGn_r", shading="flat")
    fig.colorbar(im, ax=axL, label="median error (deg, log)")
    for i in range(len(omegas)):
        for j in range(len(accels)):
            axL.text(j, i, f"{med_r[i,j]:.1f}", ha="center", va="center", fontsize=7,
                     color="black" if med_r[i, j] <= reticle*2.2 else "white")
    # ขอบ on-target: realistic (ดำทึบ) vs ceiling (ฟ้าประ) → เห็น envelope หด
    try:
        c1 = axL.contour(XX, YY, med_r, levels=[reticle], colors="black", linewidths=2.4)
        axL.clabel(c1, fmt="REALISTIC ON-TARGET", fontsize=8)
    except Exception:
        pass
    try:
        c2 = axL.contour(XX, YY, med_c, levels=[reticle], colors="cyan", linewidths=2.0, linestyles="--")
        axL.clabel(c2, fmt="ceiling (กลไกล้วน)", fontsize=8)
    except Exception:
        pass
    axL.set_title(f"REALISTIC median error — rate {args.rate:.0f}Hz, latency {args.latency_ms:.0f}ms, "
                  f"lead {lead:.0f}ms\n{args.controller}, det {args.det_fps:.0f}fps, noise {args.noise_px:.0f}px, "
                  f"miss {args.miss_rate*100:.0f}%  | ฟ้าประ=เพดานกลไก", fontsize=9.5)

    im2 = axR.pcolormesh(xext, yext, on_r, cmap="RdYlGn", vmin=0, vmax=100, shading="flat")
    fig.colorbar(im2, ax=axR, label="on-target %")
    for i in range(len(omegas)):
        for j in range(len(accels)):
            axR.text(j, i, f"{on_r[i,j]:.0f}", ha="center", va="center", fontsize=7,
                     color="black" if on_r[i, j] > 30 else "white")
    axR.set_title("REALISTIC on-target % (err ≤ reticle)", fontsize=9.5)

    for ax in (axL, axR):
        ax.set_xticks(range(len(accels))); ax.set_xticklabels([f"{int(a)}" for a in accels])
        ax.set_yticks(range(len(omegas)))
        ax.set_yticklabels([f"{w:.0f} (v≈{w*math.pi/180*args.range_m:.0f})" for w in omegas])
        ax.set_xlabel("arm accel $120/$121 (deg/s²)")
        ax.set_ylabel(f"ω (deg/s)  [v=โดรน m/s @ {args.range_m:.0f}m]")
    fig.suptitle(f"Real-usage tracking envelope — {cam['name']}  (กลไก + detection pipeline)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
