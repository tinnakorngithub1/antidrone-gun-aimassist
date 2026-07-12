"""
cam3_rgb_thermal_calibrator.py
------------------------------
คาลิเบรตให้ภาพ RGB (cam3_rgb, channels/101) กับ THERMAL (cam3_thermal, channels/201)
ของกล้อง Hikvision DS-2TD2628T ตัวเดียวกัน "ทับกันเป๊ะ" — ตำแหน่ง+ขนาดวัตถุตรงกัน

หลักการ: กล้องตัวเดียว เลนส์ติดแข็ง → ความสัมพันธ์ระหว่างสองภาพเป็น homography คงที่
         จับจุดตรงกัน ≥4 คู่ในสองภาพ → หา H (thermal → rgb) → เซฟไว้ใช้ตอน runtime

Output: calibration_data/cam3_rgb_thermal_homography.json
        (H 3x3, ขนาดภาพสองใบ, กรอบพื้นที่ซ้อนทับ overlap, ค่า hot_is_bright)

การใช้งาน:
  python3 cam3_rgb_thermal_calibrator.py                 # ดึงเฟรมสดจากกล้องทั้งสอง (ffmpeg)
  python3 cam3_rgb_thermal_calibrator.py --rgb a.jpg --thermal b.jpg   # ใช้ภาพที่มีอยู่

ขั้นตอน (มี 2 หน้าต่าง RGB / THERMAL):
  1) คลิกจุดเด่นในภาพ RGB   → แล้วคลิก "จุดเดียวกัน" ในภาพ THERMAL   (ทำสลับ RGB→THERMAL)
  2) เก็บให้ครบ ≥4 คู่ (มุมอาคาร/ขอบหน้าต่าง/วัตถุคมชัด ที่ไกลๆ จะแม่นกว่า)
  3) กด S = คำนวณ H + เซฟ + โชว์ overlay ตรวจว่าทับกันเป๊ะ
คีย์: U=ลบคู่ล่าสุด  S=เซฟ+พรีวิว  P=พรีวิว overlay  R=ดึงเฟรมใหม่  Q/ESC=ออก
เคล็ด: เลือกจุดที่ระยะไกล (parallax ต่ำ) กระจายให้ทั่วเฟรม จะ align แม่นสุด
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    import config as app_config
except Exception:
    app_config = None

CALIB_DIR = Path("calibration_data")
OUT_PATH = CALIB_DIR / "cam3_rgb_thermal_homography.json"
RGB_CAM = "cam3_rgb"
THERMAL_CAM = "cam3_thermal"
DISP_MAX = 900  # ขนาดสูงสุดที่แสดงบนจอ (ภาพจริงความละเอียดเต็มถูกเก็บแยก)

WIN_RGB = "RGB (cam3_rgb) - คลิกจุด แล้วไปคลิกจุดเดียวกันใน THERMAL"
WIN_TH = "THERMAL (cam3_thermal) - คลิกจุดที่ตรงกับ RGB"
WIN_PREVIEW = "OVERLAY preview - เขียว=thermal ทับ RGB (ควรตรงขอบวัตถุ)"


def _rtsp_url(cam_name, cli_override=None):
    if cli_override:
        return cli_override
    if app_config is not None:
        try:
            return app_config.get_camera_config(cam_name)["rtsp_url"]
        except Exception as e:
            print(f"[WARN] อ่าน rtsp ของ {cam_name} จาก config ไม่ได้: {e}")
    return None


def grab_frame(cam_name, url):
    """ดึง 1 เฟรมจาก RTSP ผ่าน ffmpeg (เชื่อถือได้กว่า cv2 บน h265/UDP)."""
    if url is None:
        print(f"[ERR] ไม่มี rtsp_url สำหรับ {cam_name}")
        return None
    tmp = Path(tempfile.gettempdir()) / f"_grab_{cam_name}.jpg"
    cmd = ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", str(url),
           "-frames:v", "1", "-q:v", "2", str(tmp)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except Exception as e:
        print(f"[ERR] ffmpeg ดึงเฟรม {cam_name} ล้มเหลว: {e}")
        return None
    if not tmp.exists():
        print(f"[ERR] ดึงเฟรม {cam_name} ไม่ได้ (ตรวจ URL/เครือข่าย): {url}")
        return None
    img = cv2.imread(str(tmp))
    return img


def _fit_scale(img):
    h, w = img.shape[:2]
    s = min(1.0, DISP_MAX / max(h, w))
    disp = cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img.copy()
    return disp, s


class Picker:
    def __init__(self, rgb, thermal):
        self.rgb = rgb
        self.thermal = thermal
        self.rgb_disp, self.rgb_s = _fit_scale(rgb)
        self.th_disp, self.th_s = _fit_scale(thermal)
        self.rgb_pts = []   # full-res coords
        self.th_pts = []
        self.expect = "rgb"  # ต้องคลิก rgb ก่อน แล้วค่อย thermal
        self.H = None

    def _on_rgb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.expect == "rgb":
            self.rgb_pts.append((x / self.rgb_s, y / self.rgb_s))
            self.expect = "thermal"
            print(f"[{len(self.rgb_pts)}] RGB ({x/self.rgb_s:.0f},{y/self.rgb_s:.0f}) → คลิกจุดเดียวกันใน THERMAL")

    def _on_th(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.expect == "thermal":
            self.th_pts.append((x / self.th_s, y / self.th_s))
            self.expect = "rgb"
            print(f"    ↳ THERMAL ({x/self.th_s:.0f},{y/self.th_s:.0f}) เก็บคู่ที่ {len(self.th_pts)} แล้ว")

    def _draw(self):
        r = self.rgb_disp.copy()
        t = self.th_disp.copy()
        for i, (px, py) in enumerate(self.rgb_pts):
            p = (int(px * self.rgb_s), int(py * self.rgb_s))
            cv2.circle(r, p, 5, (0, 255, 0), -1)
            cv2.putText(r, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (px, py) in enumerate(self.th_pts):
            p = (int(px * self.th_s), int(py * self.th_s))
            cv2.circle(t, p, 5, (0, 255, 0), -1)
            cv2.putText(t, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        hint = f"pairs={len(self.th_pts)}  next={self.expect.upper()}  (>=4 ครบแล้วกด S)"
        cv2.putText(r, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(WIN_RGB, r)
        cv2.imshow(WIN_TH, t)

    def undo(self):
        # ลบให้กลับเป็นคู่สมบูรณ์: ถ้าเพิ่งคลิก rgb (ค้าง) ลบ rgb; ไม่งั้นลบทั้งคู่ล่าสุด
        if self.expect == "thermal" and self.rgb_pts:
            self.rgb_pts.pop(); self.expect = "rgb"; print("[U] ลบจุด RGB ที่ค้าง")
        elif self.th_pts:
            self.rgb_pts.pop(); self.th_pts.pop(); print("[U] ลบคู่ล่าสุด")

    def compute(self):
        n = len(self.th_pts)
        if n < 4:
            print(f"[ERR] ต้องมี ≥4 คู่ (ตอนนี้ {n})"); return False
        src = np.array(self.th_pts, dtype=np.float64)   # thermal
        dst = np.array(self.rgb_pts, dtype=np.float64)  # rgb
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("[ERR] หา homography ไม่ได้ (จุดอาจซ้ำแนว/น้อยเกิน)"); return False
        inliers = int(mask.sum()) if mask is not None else n
        self.H = H
        print(f"[OK] H คำนวณแล้ว: {inliers}/{n} inliers")
        return True

    def overlap_quad_bbox(self):
        """มุมภาพ thermal warp ไป rgb → รูปสี่เหลี่ยม (พื้นที่ที่ thermal ครอบใน rgb)."""
        th_h, th_w = self.thermal.shape[:2]
        rgb_h, rgb_w = self.rgb.shape[:2]
        corners = np.array([[[0, 0]], [[th_w, 0]], [[th_w, th_h]], [[0, th_h]]], dtype=np.float64)
        quad = cv2.perspectiveTransform(corners, self.H).reshape(-1, 2)
        xs = np.clip(quad[:, 0], 0, rgb_w); ys = np.clip(quad[:, 1], 0, rgb_h)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # x0,y0,x1,y1
        return quad.tolist(), bbox

    def save(self):
        quad, bbox = self.overlap_quad_bbox()
        CALIB_DIR.mkdir(exist_ok=True)
        data = {
            "rgb_camera": RGB_CAM,
            "thermal_camera": THERMAL_CAM,
            "rgb_size": [self.rgb.shape[1], self.rgb.shape[0]],
            "thermal_size": [self.thermal.shape[1], self.thermal.shape[0]],
            "H_thermal_to_rgb": self.H.tolist(),
            "overlap_quad_rgb": quad,      # 4 มุมของกรวย thermal ในพิกัด rgb
            "overlap_bbox_rgb": bbox,      # กรอบสี่เหลี่ยม x0,y0,x1,y1 สำหรับ crop
            "hot_is_bright": True,         # white-hot (ร้อน=สว่าง); ตั้ง False ถ้าใช้ black-hot
            "n_pairs": len(self.th_pts),
        }
        with open(OUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVE] {OUT_PATH}")

    def preview(self):
        if self.H is None:
            print("[!] ยังไม่ได้คำนวณ H (กด S ก่อน)"); return
        rgb_h, rgb_w = self.rgb.shape[:2]
        warp = cv2.warpPerspective(self.thermal, self.H, (rgb_w, rgb_h))
        if warp.ndim == 3:
            warp_g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        else:
            warp_g = warp
        over = self.rgb.copy()
        green = np.zeros_like(over); green[:, :, 1] = warp_g
        blend = cv2.addWeighted(over, 0.6, green, 0.8, 0)
        _, bbox = self.overlap_quad_bbox()
        cv2.rectangle(blend, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        disp, _ = _fit_scale(blend)
        cv2.imshow(WIN_PREVIEW, disp)
        print("[P] overlay: เขียว=thermal ทับ RGB — ขอบวัตถุควรตรงกัน; กรอบแดง=พื้นที่ซ้อนทับ")


def main():
    args = sys.argv[1:]
    rgb_path = th_path = None
    if "--rgb" in args:
        rgb_path = args[args.index("--rgb") + 1]
    if "--thermal" in args:
        th_path = args[args.index("--thermal") + 1]

    if rgb_path and th_path:
        rgb = cv2.imread(rgb_path); thermal = cv2.imread(th_path)
    else:
        print("[..] ดึงเฟรมสดจากกล้องทั้งสอง (ffmpeg)")
        rgb = grab_frame(RGB_CAM, _rtsp_url(RGB_CAM))
        thermal = grab_frame(THERMAL_CAM, _rtsp_url(THERMAL_CAM))
    if rgb is None or thermal is None:
        print("[FATAL] โหลดภาพไม่ครบ — ตรวจกล้อง/พาธ แล้วลองใหม่"); return 1
    print(f"[..] RGB {rgb.shape[1]}x{rgb.shape[0]}  |  THERMAL {thermal.shape[1]}x{thermal.shape[0]}")

    pk = Picker(rgb, thermal)
    cv2.namedWindow(WIN_RGB); cv2.namedWindow(WIN_TH)
    cv2.setMouseCallback(WIN_RGB, pk._on_rgb)
    cv2.setMouseCallback(WIN_TH, pk._on_th)
    print("คลิก RGB → THERMAL สลับกัน ≥4 คู่ | U=undo S=save P=preview R=recapture Q=quit")

    while True:
        pk._draw()
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('u'):
            pk.undo()
        elif k == ord('s'):
            if pk.compute():
                pk.save(); pk.preview()
        elif k == ord('p'):
            pk.preview()
        elif k == ord('r'):
            nr = grab_frame(RGB_CAM, _rtsp_url(RGB_CAM))
            nt = grab_frame(THERMAL_CAM, _rtsp_url(THERMAL_CAM))
            if nr is not None and nt is not None:
                pk = Picker(nr, nt)
                cv2.setMouseCallback(WIN_RGB, pk._on_rgb)
                cv2.setMouseCallback(WIN_TH, pk._on_th)
                print("[R] ดึงเฟรมใหม่แล้ว เริ่มจับจุดใหม่")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
