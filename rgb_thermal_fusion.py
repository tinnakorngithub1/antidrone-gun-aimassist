"""
rgb_thermal_fusion.py
---------------------
โมดูล runtime สำหรับกล้อง bi-spectrum DS-2TD2628T (cam3_rgb + cam3_thermal):
ใช้ homography จาก cam3_rgb_thermal_calibrator.py เพื่อ

  1) warp+crop ภาพสองใบให้ "ทับกันเป๊ะ" (aligned_pair) — ตำแหน่ง+ขนาดวัตถุตรงกัน
  2) เช็คว่า detection จาก RGB มี "ก้อนความร้อน" ที่ตำแหน่งเดียวกันใน thermal ไหม
     → ใช้ยืนยันเป้า / ลด false positive (นก/แสงวาบไม่มีความร้อน)

ปรัชญาการใช้: อย่าใช้เป็น veto แข็ง — thermal native 256x192 (~10 ppd จริง) เป้าเล็ก/ไกล/
เย็นอาจไม่โผล่ก้อนร้อนชัด. ให้ใช้ has_heat เป็น "คะแนนเสริมความมั่นใจ" และเช็คได้เฉพาะเป้า
ที่อยู่ในกรวยซ้อนทับ (~24.4°) เท่านั้น — นอกกรวย in_overlap=False → ตัดสินด้วย RGB ล้วน

โหลด: fuse = RgbThermalFusion.load()            # อ่าน calibration_data/cam3_rgb_thermal_homography.json
Runtime (เบา ไม่ warp ทั้งภาพ):
    res = fuse.check_detection(thermal_frame, (x, y, w, h))
    # res = {"in_overlap": bool, "has_heat": bool|None, "score": float, "thermal_pt": (tx,ty)}
Display (ทับกันเป๊ะ ไว้โชว์/ดีบัก):
    rgb_crop, thermal_crop = fuse.aligned_pair(rgb_frame, thermal_frame)
"""
import json
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = Path("calibration_data") / "cam3_rgb_thermal_homography.json"

# เกณฑ์ hotspot: ROI ต้องเด่นกว่าพื้นหลัง (ท้องฟ้า) กี่เท่าของ std ถึงนับว่า "มีความร้อน"
HOTSPOT_Z_THRESH = 3.0
# รัศมี ROI ใน thermal (พิกเซล) เมื่อประเมินจาก bbox ไม่ได้
ROI_R_MIN, ROI_R_MAX = 3, 20


class RgbThermalFusion:
    def __init__(self, data):
        self.rgb_size = tuple(data["rgb_size"])          # (w, h)
        self.thermal_size = tuple(data["thermal_size"])  # (w, h)
        self.H = np.array(data["H_thermal_to_rgb"], dtype=np.float64)  # thermal -> rgb
        self.H_inv = np.linalg.inv(self.H)               # rgb -> thermal
        self.overlap_quad = np.array(data["overlap_quad_rgb"], dtype=np.float32)  # 4x2 (rgb)
        self.overlap_bbox = list(data["overlap_bbox_rgb"])  # x0,y0,x1,y1 (rgb)
        self.hot_is_bright = bool(data.get("hot_is_bright", True))

    # ---------- โหลด ----------
    @classmethod
    def load(cls, path=CALIB_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบ {path} — รัน cam3_rgb_thermal_calibrator.py ก่อนเพื่อสร้าง homography")
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def try_load(cls, path=CALIB_PATH):
        """คืน instance หรือ None ถ้ายังไม่ได้ calibrate (ให้ caller ทำงานต่อแบบ RGB-only ได้)."""
        try:
            return cls.load(path)
        except Exception as e:
            print(f"[fusion] ปิดใช้งาน (ยังไม่ calibrate): {e}")
            return None

    # ---------- แปลงพิกัด ----------
    def rgb_to_thermal(self, x, y):
        p = np.array([[[float(x), float(y)]]], dtype=np.float64)
        tx, ty = cv2.perspectiveTransform(p, self.H_inv).reshape(2)
        return float(tx), float(ty)

    def is_in_overlap(self, x, y):
        return cv2.pointPolygonTest(self.overlap_quad, (float(x), float(y)), False) >= 0

    # ---------- warp + crop ให้ทับกันเป๊ะ (สำหรับแสดงผล/ดีบัก) ----------
    def warp_thermal_to_rgb(self, thermal):
        """thermal → grid ของ RGB (ขนาดเท่า rgb_size). thermal ถูกขยายขึ้นหา RGB (คงความคม RGB)."""
        w, h = self.rgb_size
        return cv2.warpPerspective(thermal, self.H, (w, h))

    def aligned_pair(self, rgb, thermal):
        """คืน (rgb_crop, thermal_warp_crop) ขนาดเท่ากัน พิกเซล (x,y) = ทิศเดียวกันเป๊ะ.
        ครอบเฉพาะกรอบซ้อนทับ (overlap_bbox) — ส่วนที่เกินถูก crop ออก."""
        x0, y0, x1, y1 = self.overlap_bbox
        warp = self.warp_thermal_to_rgb(thermal)
        rgb_crop = rgb[y0:y1, x0:x1].copy()
        th_crop = warp[y0:y1, x0:x1].copy()
        return rgb_crop, th_crop

    # ---------- เช็ค hotspot (เบา: ไม่ warp ทั้งภาพ) ----------
    @staticmethod
    def _to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def _roi_radius_from_box(self, box):
        if box is None:
            return 6
        x, y, w, h = box
        # map มุมกล่อง rgb → thermal เพื่อประเมินขนาดกล่องในสเปซ thermal
        pts = np.array([[[x, y]], [[x + w, y + h]]], dtype=np.float64)
        t = cv2.perspectiveTransform(pts, self.H_inv).reshape(2, 2)
        diag = np.hypot(*(t[1] - t[0]))
        return int(np.clip(diag * 0.6, ROI_R_MIN, ROI_R_MAX))

    def check_detection(self, thermal, box):
        """box = (x, y, w, h) ในพิกัด RGB. คืน dict สรุปว่ามีความร้อนตรงตำแหน่งไหม.
        in_overlap=False → เป้าอยู่นอกกรวย thermal เช็คไม่ได้ (has_heat=None)."""
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        out = {"in_overlap": False, "has_heat": None, "score": 0.0, "thermal_pt": None}
        if not self.is_in_overlap(cx, cy):
            return out
        tx, ty = self.rgb_to_thermal(cx, cy)
        out["thermal_pt"] = (tx, ty)
        tw, th = self.thermal_size
        if not (0 <= tx < tw and 0 <= ty < th):
            return out
        out["in_overlap"] = True

        g = self._to_gray(thermal).astype(np.float32)
        r = self._roi_radius_from_box(box)
        itx, ity = int(round(tx)), int(round(ty))

        # ROI = จานกลม r, background = วงแหวนรอบนอก (r .. 3r)
        yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
        dist2 = (xx - itx) ** 2 + (yy - ity) ** 2
        roi_m = dist2 <= r * r
        bg_m = (dist2 > (1.6 * r) ** 2) & (dist2 <= (3.0 * r) ** 2)
        if roi_m.sum() < 1 or bg_m.sum() < 8:
            return out

        roi = g[roi_m]; bg = g[bg_m]
        bg_med = float(np.median(bg))
        bg_std = float(np.std(bg)) + 1e-3
        if self.hot_is_bright:
            peak = float(np.percentile(roi, 95))
            z = (peak - bg_med) / bg_std
        else:  # black-hot: ร้อน = มืด
            peak = float(np.percentile(roi, 5))
            z = (bg_med - peak) / bg_std
        out["score"] = z
        out["has_heat"] = bool(z >= HOTSPOT_Z_THRESH)
        return out


# ---------- smoke test ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        fuse = RgbThermalFusion.load()
        rgb = cv2.imread(sys.argv[1]); thermal = cv2.imread(sys.argv[2])
        rc, tc = fuse.aligned_pair(rgb, thermal)
        print("aligned_pair:", rc.shape, tc.shape, "(ต้องขนาดเท่ากัน)")
        # ลองเช็คกลางภาพ
        w, h = fuse.rgb_size
        print("center check:", fuse.check_detection(thermal, (w // 2 - 30, h // 2 - 30, 60, 60)))
    else:
        print("usage: python3 rgb_thermal_fusion.py <rgb.jpg> <thermal.jpg>  (ต้องมี homography json แล้ว)")
