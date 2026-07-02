"""
25_sim_sidebyside.py
อัดวิดีโอเทียบ "LAGS-BEHIND vs ON-TARGET" แบบ side-by-side ในไฟล์เดียว

- โดรนเคลื่อนเหมือนกันเป๊ะทั้งสองฝั่ง (deterministic) ต่างกันแค่ pipeline:
    ซ้าย  = ท่อ detect ปกติ (latency สูง, P+raw lead)        -> LAGS-BEHIND
    ขวา   = ท่อ detect ดีขึ้น (latency ต่ำ, Kalman+coast)     -> ON-TARGET
- reuse DroneModel/ArmModel/ChaseController/_render จาก 23_arm_chase_sim.py
- รัน headless ได้ (เซฟ MP4 ไม่ต้องมีจอ)

ตัวอย่าง:
  python3 25_sim_sidebyside.py --out compare.mp4
  python3 25_sim_sidebyside.py --speed-ms 25 --range-m 180 --pattern sine --duration-s 18
"""
import argparse
import importlib.util
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# โหลดชิ้นส่วนจาก 23_arm_chase_sim.py (ชื่อโมดูลขึ้นต้นด้วยเลข)
_SIM_PATH = Path(__file__).resolve().parent / "23_arm_chase_sim.py"
_spec = importlib.util.spec_from_file_location("arm_chase_sim", _SIM_PATH)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

LANE_W, LANE_H = 960, 540   # ขนาดต่อฝั่ง (16:9 ของ cam4)
BANNER_H = 70               # 2 บรรทัด: ชื่อ lane + scoreboard สด
DIVIDER = 6


def _draw_scoreboard(banner, x0, lane):
    """วาด scoreboard สด (on-target% + median + err ปัจจุบัน) ลงครึ่ง banner ของ lane"""
    on, med, err, _ = lane.live_stats()
    # แถบ on-target% ยาวตามเปอร์เซ็นต์ — เขียวเมื่อสูง แดงเมื่อต่ำ
    bar_x, bar_y, bar_w, bar_h = x0 + 14, 46, 150, 14
    cv2.rectangle(banner, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    fill = int(bar_w * on / 100.0)
    bar_col = (60, 220, 60) if on >= 67 else (0, 200, 255) if on >= 34 else (60, 60, 255)
    if fill > 0:
        cv2.rectangle(banner, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_col, -1)
    txt = f"on-target {on:3.0f}%   med {med:.2f}deg   err {err:.2f}deg"
    cv2.putText(banner, txt, (bar_x + bar_w + 12, bar_y + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_col, 1, cv2.LINE_AA)


def _lane_args(base, **over):
    """สร้าง Namespace args สำหรับ 1 lane (ค่าที่ _render/feasibility ใช้)"""
    d = dict(
        camera=base.camera, latency_ms=300.0, lead_ms=200.0, det_fps=15.0,
        noise_px=base.noise_px, miss_rate=base.miss_rate, kp=1.0,
        ab_alpha=0.5, ab_beta=0.3, controller="plead",
        warmup_s=base.warmup_s, seed=base.seed,
        speed_ms=base.speed_ms, range_m=base.range_m, pattern=base.pattern,
        omega_deg=base.omega_deg, duration_s=base.duration_s,
    )
    d.update(over)
    return argparse.Namespace(**d)


class Lane:
    """หนึ่งฝั่งของการเทียบ — ห่อ drone+arm+controller+สถานะ render"""
    def __init__(self, cam, arm, args, label, label_color):
        self.cam, self.arm, self.args = cam, arm, args
        self.label, self.label_color = label, label_color
        ppd_x, ppd_y = cam["ppd_x"], cam["ppd_y"]
        w, h = cam["width"], cam["height"]
        self.ppd_x, self.ppd_y = ppd_x, ppd_y
        self.reticle_px = max(10, int(min(w, h) * 0.03))
        self.reticle_deg = self.reticle_px / ppd_x
        if args.omega_deg is not None:
            self.omega = args.omega_deg
        else:
            self.omega = (args.speed_ms / max(args.range_m, 1e-6)) * (180.0 / math.pi)
        self.src = f"v={args.speed_ms} m/s R={args.range_m} m -> w={self.omega:.1f}"
        self.drone = _sim.DroneModel(args.pattern, self.omega, cam["fov_h"], cam["fov_v"])
        self.armM = _sim.ArmModel(arm)
        self.ctrl = _sim.ChaseController(
            ppd_x, ppd_y, args.kp, args.lead_ms / 1000.0, arm["cmd_interval"],
            args.latency_ms / 1000.0, 1.0 / args.det_fps,
            noise_px=args.noise_px, miss_rate=args.miss_rate, seed=args.seed,
            mode=args.controller, alpha=args.ab_alpha, beta=args.ab_beta,
        )
        self.sim_dt = 1.0 / 200.0
        self.hist = deque(maxlen=int(2.0 / self.sim_dt))
        self.err_hist = deque(maxlen=int(6.0 / self.sim_dt))
        self.trail = deque(maxlen=60)
        self.verdict, _, self.note = _sim.feasibility_note(
            self.omega, min(arm["max_pan_rate"], arm["max_tilt_rate"]),
            args.latency_ms / 1000.0, arm["cmd_interval"], args.lead_ms / 1000.0,
            self.reticle_deg, det_dt=1.0 / args.det_fps,
        )
        self.scale = LANE_W / w
        self._dpan = self._dtilt = self._err = 0.0
        self.errs, self.on_target, self.lost = [], 0, 0

    def step(self, t):
        dpan, dtilt = self.drone.angle_at(t)
        self.hist.append((t, dpan, dtilt, self.armM.pan, self.armM.tilt))
        self.ctrl.update(t, self.armM, self.hist)
        self.armM.step(self.sim_dt)
        epan, etilt = dpan - self.armM.pan, dtilt - self.armM.tilt
        err = math.hypot(epan, etilt)
        self.err_hist.append((t, err))
        self._dpan, self._dtilt, self._err = dpan, dtilt, err
        if t >= self.args.warmup_s:
            in_fov = abs(epan) <= self.cam["fov_h"] / 2 and abs(etilt) <= self.cam["fov_v"] / 2
            if in_fov:
                self.errs.append(err)
                if err <= self.reticle_deg:
                    self.on_target += 1
            else:
                self.lost += 1

    def render(self, t):
        return _sim._render(
            self.cam, self.armM, self._dpan, self._dtilt, self.ppd_x, self.ppd_y,
            self.reticle_px, self.scale, LANE_W, LANE_H, self.trail, t, self._err,
            self.reticle_deg, self.omega, self.arm, self.verdict, self.note, self.src,
            self.args, meas=self.ctrl._meas, err_hist=self.err_hist,
        )

    def live_stats(self):
        """สถานะสะสม ณ ปัจจุบัน สำหรับ scoreboard บน banner (อัปเดตทุกเฟรม)"""
        n = len(self.errs) + self.lost
        on = 100.0 * self.on_target / n if n else 0.0
        med = float(np.median(self.errs)) if self.errs else 0.0
        in_fov = self._err <= self.cam["fov_h"] / 2  # หยาบ ๆ สำหรับสีไฟสถานะ
        return on, med, self._err, in_fov

    def summary(self):
        if not self.errs:
            return "no data"
        med = float(np.median(self.errs))
        on = 100.0 * self.on_target / (len(self.errs) + self.lost)
        return f"median {med:.2f} deg | on-target {on:.0f}%"


def main():
    ap = argparse.ArgumentParser(description="วิดีโอเทียบ LAGS vs ON-TARGET side-by-side")
    ap.add_argument("--out", default="sidebyside.mp4", help="ไฟล์วิดีโอออก (MP4)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--pattern", choices=["cross", "sine"], default="sine")
    ap.add_argument("--speed-ms", type=float, default=25.0)
    ap.add_argument("--range-m", type=float, default=180.0)
    ap.add_argument("--omega-deg", type=float, default=None)
    ap.add_argument("--noise-px", type=float, default=8.0)
    ap.add_argument("--miss-rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--duration-s", type=float, default=18.0)
    ap.add_argument("--video-fps", type=float, default=30.0)
    # per-lane pipeline (ปรับได้)
    ap.add_argument("--lags-latency-ms", type=float, default=350.0)
    ap.add_argument("--lags-lead-ms", type=float, default=150.0)
    ap.add_argument("--ontgt-latency-ms", type=float, default=100.0)
    ap.add_argument("--ontgt-lead-ms", type=float, default=120.0)
    args = ap.parse_args()

    cam = _sim.load_camera_params(args.camera)
    arm = _sim.load_arm_params()

    lags_args = _lane_args(args, latency_ms=args.lags_latency_ms, lead_ms=args.lags_lead_ms,
                           controller="plead")
    ontgt_args = _lane_args(args, latency_ms=args.ontgt_latency_ms, lead_ms=args.ontgt_lead_ms,
                            controller="kalman_coast")
    left = Lane(cam, arm, lags_args, "LAGS-BEHIND (high latency, P+lead)", (60, 60, 255))
    right = Lane(cam, arm, ontgt_args, "ON-TARGET (low latency, Kalman+coast)", (60, 220, 60))

    print("=" * 64)
    print(f"SIDE-BY-SIDE — {cam['name']} {cam['width']}x{cam['height']}  pattern={args.pattern}")
    print(f"  LEFT  (LAGS) : lat {args.lags_latency_ms:.0f}ms lead {args.lags_lead_ms:.0f}ms plead")
    print(f"  RIGHT (ONTGT): lat {args.ontgt_latency_ms:.0f}ms lead {args.ontgt_lead_ms:.0f}ms kalman_coast")
    print(f"  shared: v={args.speed_ms} R={args.range_m} noise {args.noise_px}px miss {args.miss_rate*100:.0f}%")
    print("=" * 64)

    out_w = LANE_W * 2 + DIVIDER
    out_h = BANNER_H + LANE_H
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"❌ เปิด VideoWriter ไม่ได้: {args.out}")
        return

    sim_dt = 1.0 / 200.0
    frame_every = max(1, int(round((1.0 / args.video_fps) / sim_dt)))
    divider = np.full((LANE_H, DIVIDER, 3), 70, np.uint8)
    t = 0.0
    step_idx = 0
    while t < args.duration_s:
        left.step(t)
        right.step(t)
        if step_idx % frame_every == 0:
            lf = left.render(t)
            rf = right.render(t)
            body = np.hstack([lf, divider, rf])
            banner = np.full((BANNER_H, out_w, 3), 22, np.uint8)
            cv2.putText(banner, left.label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        left.label_color, 2, cv2.LINE_AA)
            cv2.putText(banner, right.label, (LANE_W + DIVIDER + 14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, right.label_color, 2, cv2.LINE_AA)
            _draw_scoreboard(banner, 0, left)
            _draw_scoreboard(banner, LANE_W + DIVIDER, right)
            cv2.line(banner, (LANE_W + DIVIDER // 2, 0), (LANE_W + DIVIDER // 2, BANNER_H), (70, 70, 70), 1)
            writer.write(np.vstack([banner, body]))
        t += sim_dt
        step_idx += 1

    writer.release()
    print(f"\n✅ เซฟวิดีโอ: {args.out}  ({args.video_fps:.0f} fps, ~{args.duration_s:.0f}s, {out_w}x{out_h})")
    print(f"   LEFT  (LAGS) : {left.summary()}")
    print(f"   RIGHT (ONTGT): {right.summary()}")


if __name__ == "__main__":
    main()
