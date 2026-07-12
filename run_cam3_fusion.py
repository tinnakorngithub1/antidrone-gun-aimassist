"""
run_cam3_fusion.py
------------------
รัน "cam3" แบบ fusion แยกเดี่ยว — ไม่แตะ file 11 / cam4 ที่ทำงานอยู่เลย
เปิดสองสตรีมสดของกล้อง bi-spectrum DS-2TD2628T (.201):
  cam3_rgb (channels/101)  +  cam3_thermal (channels/201)
แล้วแสดงภาพ "ทับกัน" ด้วย homography จาก cam3_rgb_thermal_homography.json:
  ตัวหลัก = เข้ม (opaque) | ตัวช่วย = จางๆ ทับด้านหลัง ให้เห็นวัตถุชัดขึ้น

โหมด:
  --primary day    (ค่าเริ่ม)  RGB เป็นหลัก + thermal จางช่วย → เห็นของร้อนพรายๆ
  --primary night              thermal เป็นหลัก + RGB จางช่วย → กลางคืน RGB มืด

ปุ่ม (โหมด GUI):
  S = สลับหลัก/ช่วย (day↔night)   [ / ] = ลด/เพิ่มความจางตัวช่วย
  C = สลับสีตัวช่วย (colormap/เทา)  V = สลับมุมมอง (fused / เทียบข้างกัน / หลักล้วน)
  W = เซฟภาพ   Q/ESC = ออก

ใช้:  python3 run_cam3_fusion.py
      python3 run_cam3_fusion.py --primary night --aux-alpha 0.45
      python3 run_cam3_fusion.py --headless        # เซฟ fused.jpg ไม่เปิดหน้าต่าง (เช็คเร็ว)
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

try:
    import config as app_config
except Exception:
    app_config = None
from rgb_thermal_fusion import RgbThermalFusion


class ThreadedStream:
    """อ่าน rtsp ใน thread แยก เก็บเฟรมล่าสุด (non-blocking) — ไม่ให้สตรีมช้าไปหน่วงอีกตัว."""
    def __init__(self, url, name):
        self.url, self.name = url, name
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.frame = None
        self.ts = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = f
                self.ts = time.time()

    def read(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.ts

    def opened(self):
        return self.cap.isOpened()

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def _rtsp(cam):
    if app_config is not None:
        try:
            return app_config.get_camera_config(cam)["rtsp_url"]
        except Exception as e:
            print(f"[WARN] config {cam}: {e}")
    return None


def fit(img, max_w=1400):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


class Fuser:
    """สร้างภาพ fused: base (หลัก, เข้ม) + aux (ช่วย, จาง) ในตารางพิกัดของ base."""
    def __init__(self, fuse: RgbThermalFusion):
        self.f = fuse
        self._mask_rgb = None   # cache overlap mask ในพิกัด RGB

    def _overlap_mask_rgb(self, shape):
        if self._mask_rgb is None or self._mask_rgb.shape[:2] != shape[:2]:
            m = np.zeros(shape[:2], np.uint8)
            cv2.fillConvexPoly(m, self.f.overlap_quad.astype(np.int32), 255)
            self._mask_rgb = m
        return self._mask_rgb

    @staticmethod
    def _ghost(gray, colormap):
        g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if colormap:
            return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def day(self, rgb, thermal, aux_alpha, colormap):
        """หลัก=RGB (เข้ม) + ช่วย=thermal (จาง) เฉพาะในกรวยซ้อนทับ."""
        warp = self.f.warp_thermal_to_rgb(thermal)              # thermal → grid RGB
        ghost = self._ghost(warp, colormap)
        mask = self._overlap_mask_rgb(rgb.shape)
        blended = cv2.addWeighted(rgb, 1.0, ghost, aux_alpha, 0)
        out = rgb.copy()
        out[mask > 0] = blended[mask > 0]                       # นอกกรวย = RGB ล้วน
        return out

    def night(self, rgb, thermal, aux_alpha, colormap):
        """หลัก=thermal (เข้ม) + ช่วย=RGB (จาง) ในตารางพิกัด thermal."""
        tw, th = self.f.thermal_size
        warp_rgb = cv2.warpPerspective(rgb, self.f.H_inv, (tw, th))   # RGB → grid thermal
        base = thermal if thermal.ndim == 3 else cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
        if colormap:
            base = cv2.applyColorMap(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(base, 1.0, warp_rgb, aux_alpha, 0)


def draw_hud(img, lines):
    y = 26
    for t in lines:
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y += 26
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", choices=["day", "night"], default="day")
    ap.add_argument("--aux-alpha", type=float, default=0.40)
    ap.add_argument("--colormap", action="store_true", help="ตัวช่วยเป็นสี (INFERNO) แทนเทา")
    ap.add_argument("--headless", action="store_true", help="เซฟ fused.jpg แล้วออก ไม่เปิดหน้าต่าง")
    a = ap.parse_args()

    try:
        fuse = RgbThermalFusion.load()
    except Exception as e:
        print(f"[FATAL] {e}"); return 1
    fuser = Fuser(fuse)

    rgb_url, th_url = _rtsp("cam3_rgb"), _rtsp("cam3_thermal")
    if not rgb_url or not th_url:
        print("[FATAL] ไม่พบ rtsp_url ของ cam3_rgb/cam3_thermal ใน config"); return 1
    print(f"[..] เปิดสตรีม\n  RGB     {rgb_url.split('@')[-1]}\n  THERMAL {th_url.split('@')[-1]}")
    srgb = ThreadedStream(rgb_url, "rgb")
    sth = ThreadedStream(th_url, "thermal")

    # รอเฟรมแรก
    t0 = time.time()
    while time.time() - t0 < 15:
        r, _ = srgb.read(); t, _ = sth.read()
        if r is not None and t is not None:
            break
        time.sleep(0.1)
    if r is None or t is None:
        print("[FATAL] รับเฟรมไม่ครบใน 15s (ตรวจกล้อง/เครือข่าย)")
        srgb.close(); sth.close(); return 1
    print("[OK] ได้เฟรมทั้งสองสตรีมแล้ว")

    primary = a.primary
    aux_alpha = a.aux_alpha
    colormap = a.colormap
    view = "fused"   # fused | side | primary

    def build(rgb, thermal):
        if primary == "day":
            fused = fuser.day(rgb, thermal, aux_alpha, colormap)
        else:
            fused = fuser.night(rgb, thermal, aux_alpha, colormap)
        return fused

    if a.headless:
        fused = build(r, t)
        cv2.imwrite("fused.jpg", fit(fused))
        rc, tc = fuse.aligned_pair(r, t)
        h = 520; s = h / rc.shape[0]
        cv2.imwrite("aligned_side.jpg", np.hstack([cv2.resize(rc,(int(rc.shape[1]*s),h)),
                                                   cv2.resize(tc,(int(tc.shape[1]*s),h))]))
        print("[HEADLESS] saved fused.jpg + aligned_side.jpg"); srgb.close(); sth.close(); return 0

    win = "cam3 FUSION (S=swap  [ ]=alpha  C=color  V=view  W=save  Q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frames, tlast, fps = 0, time.time(), 0.0
    while True:
        r, _ = srgb.read(); t, _ = sth.read()
        if r is None or t is None:
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                break
            continue
        if view == "side":
            rc, tc = fuse.aligned_pair(r, t)
            hh = 600; s = hh / rc.shape[0]
            disp = np.hstack([cv2.resize(rc,(int(rc.shape[1]*s),hh)), cv2.resize(tc,(int(tc.shape[1]*s),hh))])
        elif view == "primary":
            disp = fit(r if primary == "day" else (t if t.ndim==3 else cv2.cvtColor(t,cv2.COLOR_GRAY2BGR)))
        else:
            disp = fit(build(r, t))

        frames += 1
        if frames % 10 == 0:
            now = time.time(); fps = 10.0 / (now - tlast); tlast = now
        draw_hud(disp, [
            f"primary={'RGB(day)' if primary=='day' else 'THERMAL(night)'}  aux_alpha={aux_alpha:.2f}  color={'on' if colormap else 'off'}",
            f"view={view}  disp_fps={fps:.0f}",
        ])
        cv2.imshow(win, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('s'):
            primary = "night" if primary == "day" else "day"
        elif k == ord('['):
            aux_alpha = max(0.0, aux_alpha - 0.05)
        elif k == ord(']'):
            aux_alpha = min(1.0, aux_alpha + 0.05)
        elif k == ord('c'):
            colormap = not colormap
        elif k == ord('v'):
            view = {"fused": "side", "side": "primary", "primary": "fused"}[view]
        elif k == ord('w'):
            cv2.imwrite(f"cam3_fusion_snap_{frames}.jpg", disp)
            print(f"saved cam3_fusion_snap_{frames}.jpg")

    srgb.close(); sth.close(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
