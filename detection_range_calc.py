"""
detection_range_calc.py
-----------------------
คำนวณระยะตรวจจับโดรน แยก 2 โมเดลตามฟิสิกส์จริง (อย่าปนกัน!):

  [SHAPE]  ตรวจด้วย "รูปทรง" — RGB กลางวัน + การแยกชนิดบน thermal
           จำกัดที่จำนวนพิกเซลบนเป้า (YOLO ต้องเห็นรูปร่าง)
           ใช้ ppd จริงของเซนเซอร์ (native). background มี clutter → ต้องพึ่ง appearance

  [POINT]  ตรวจ "จุดร้อน" — thermal กลางคืน (มอเตอร์ร้อน vs ท้องฟ้าเย็น)
           จำกัดที่ SNR ของก้อนร้อน ไม่ใช่จำนวนพิกเซล — จุด <1px ก็เห็นได้ถ้า ΔT/SNR พอ
           (นี่คือเหตุผลที่เห็นโดรน 300m กลางคืนได้ ทั้งที่ shape-model บอกว่าไม่ได้)

ยังมี:
  - px_on_target()      จำนวนพิกเซล(รูปทรง)บนเป้าที่ระยะหนึ่ง
  - aiming_precision_m() ความแม่นเล็ง (centroid) เป็นเมตรที่ระยะหนึ่ง

ใช้:  python3 detection_range_calc.py --size 0.5 --range 150
      python3 detection_range_calc.py --size 0.5 --range 300 --dT 30 --hot-area 0.003
"""
import argparse
import math

DEG = 180.0 / math.pi

# ── presets จากค่าที่วัด/คำนวณจริงในโปรเจกต์ ──
CAMERAS = {
    # ppd = px/deg (ตัวเลขที่ระบบใช้จริง), native_px = พิกเซลจริงของเซนเซอร์แนวนอน
    "cam3_rgb":     {"ppd": 55.4, "fov_h": 48.6, "native_px": 2688, "band": "vis"},
    "cam4":         {"ppd": 87.7, "fov_h": 43.8, "native_px": 3840, "band": "vis"},
    # thermal: ppd_output ใช้เล็ง/centroid; native core ใช้เรื่องรูปทรง+IFOV ของ point-source
    "cam3_thermal": {"ppd": 52.5, "fov_h": 24.4, "native_px": 256,  "band": "ir"},
}

# เกณฑ์พิกเซล (shape)
SHAPE_PX_MARGINAL = 8
SHAPE_PX_RELIABLE = 16

# ── SHAPE model ──
def angular_size_deg(size_m, range_m):
    return (size_m / range_m) * DEG

def px_on_target(size_m, range_m, ppd):
    """จำนวนพิกเซล(แนวขวาง)ที่เป้ากินบนภาพ = θ° × ppd."""
    return angular_size_deg(size_m, range_m) * ppd

def shape_range_m(size_m, ppd, n_px):
    """ระยะไกลสุดที่เป้ายังกิน ≥ n_px (แก้จาก px = size·ppd·DEG/R)."""
    return size_m * ppd * DEG / n_px

# ── POINT-SOURCE model (thermal night) ──
# ก้อนร้อนพื้นที่ A_hot บนพื้นหลังเย็น: ต่อ 1 พิกเซล(native) ความร้อนถูก 'เจือจาง' ด้วย
# fill-factor = A_hot / (พื้นที่ที่พิกเซลครอบที่ระยะ R). apparent ΔT ∝ 1/R².
# ตรวจจับได้เมื่อ SNR = apparentΔT / NETD ≥ SNR_thresh.
def _ifov_rad(native_ppd_deg):
    return (1.0 / native_ppd_deg) / DEG      # rad ต่อ 1 พิกเซล native

def pointsource_apparent_dT(range_m, dT_source, hot_area_m2, native_ppd_deg):
    ifov = _ifov_rad(native_ppd_deg)
    pixel_footprint_m2 = (range_m * ifov) ** 2       # พื้นที่ที่ 1 พิกเซลครอบ ที่ระยะ R
    fill = min(1.0, hot_area_m2 / pixel_footprint_m2)
    return dT_source * fill

def pointsource_range_m(dT_source, hot_area_m2, native_ppd_deg, netd_k, snr_thresh):
    """ระยะไกลสุดที่ SNR ยัง ≥ snr_thresh (regime จุดร้อนเล็กกว่าพิกเซล)."""
    ifov = _ifov_rad(native_ppd_deg)
    # dT_source·A/(R·ifov)² = snr·NETD  →  R = sqrt(dT·A/(snr·NETD)) / ifov
    return math.sqrt(dT_source * hot_area_m2 / (snr_thresh * netd_k)) / ifov

# ── การเล็ง (ทั้งสองโมเดลใช้ ppd_output ของภาพที่หา centroid) ──
def aiming_precision_m(range_m, ppd_output, centroid_px=0.3):
    """ความคลาดเคลื่อนเล็งเป็นเมตร = centroid_px / ppd (°) → ระยะทางที่ R."""
    err_deg = centroid_px / ppd_output
    return range_m * math.tan(err_deg / DEG)


def report(size_m, range_m, dT, hot_area, netd, snr, core_px, hit_radius=0.35):
    print(f"\n=== เป้า {size_m*100:.0f}cm ที่ {range_m:.0f}m  (hit radius {hit_radius}m) ===")
    th_ang = angular_size_deg(size_m, range_m)
    print(f"ขนาดเชิงมุมเป้า: {th_ang:.3f}°\n")

    print("[SHAPE] ตรวจด้วยรูปทรง (RGB กลางวัน / แยกชนิด thermal)")
    print(f"  ต้องการ ppd: ก้ำกึ่ง(8px)≥{SHAPE_PX_MARGINAL/th_ang:.0f}  เสถียร(16px)≥{SHAPE_PX_RELIABLE/th_ang:.0f}")
    for name, c in CAMERAS.items():
        # RGB ใช้ ppd จริง; thermal 'รูปทรง' ใช้ native core ppd (หยาบกว่า output)
        ppd_shape = c["ppd"] if c["band"] == "vis" else c["native_px"] / c["fov_h"]
        px = px_on_target(size_m, range_m, ppd_shape)
        r16 = shape_range_m(size_m, ppd_shape, SHAPE_PX_RELIABLE)
        r8 = shape_range_m(size_m, ppd_shape, SHAPE_PX_MARGINAL)
        tag = "✓เสถียร" if px >= SHAPE_PX_RELIABLE else ("~ก้ำกึ่ง" if px >= SHAPE_PX_MARGINAL else "✗ไม่พอ")
        print(f"  {name:13s} ppd_shape={ppd_shape:5.1f}  {px:5.1f}px  {tag:9s}"
              f"  เสถียรถึง {r16:4.0f}m / ก้ำกึ่ง {r8:4.0f}m")

    print("\n[POINT] ตรวจจุดร้อน — thermal กลางคืน (มอเตอร์ vs ฟ้าเย็น)")
    th = CAMERAS["cam3_thermal"]
    native_ppd = core_px / th["fov_h"]
    print(f"  พารามิเตอร์: ΔT={dT}K  hot_area={hot_area*1e4:.0f}cm²  NETD={netd*1000:.0f}mK"
          f"  SNR_thr={snr}  core={core_px}px (ppd_native={native_ppd:.1f})")
    appd = pointsource_apparent_dT(range_m, dT, hot_area, native_ppd)
    rmax = pointsource_range_m(dT, hot_area, native_ppd, netd, snr)
    snr_at = appd / netd
    tag = "✓เห็น" if snr_at >= snr else "✗จาง"
    print(f"  ที่ {range_m:.0f}m: apparentΔT={appd*1000:.0f}mK  SNR={snr_at:.1f}  {tag}")
    print(f"  ระยะตรวจจับจุดร้อนสูงสุด (SNR={snr}): ~{rmax:.0f}m")

    print("\n[เล็ง] ความแม่น centroid (ppd_output)")
    for name in ("cam3_rgb", "cam4", "cam3_thermal"):
        ppd = CAMERAS[name]["ppd"]
        e = aiming_precision_m(range_m, ppd, 0.3)
        tag = "✓" if e <= hit_radius else "✗"
        print(f"  {name:13s} centroid 0.3px → ±{e*100:.1f}cm ที่ {range_m:.0f}m  {tag} (เทียบ {hit_radius*100:.0f}cm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=0.5, help="ขนาดโดรน (m)")
    ap.add_argument("--range", type=float, default=150.0, help="ระยะ (m)")
    ap.add_argument("--dT", type=float, default=30.0, help="ΔT มอเตอร์-ท้องฟ้า (K)")
    ap.add_argument("--hot-area", type=float, default=0.003, help="พื้นที่จุดร้อนรวม (m²) เช่น 4 มอเตอร์")
    ap.add_argument("--netd", type=float, default=0.04, help="NETD ของ thermal (K)")
    ap.add_argument("--snr", type=float, default=6.0, help="SNR threshold ตรวจจับ")
    ap.add_argument("--core-px", type=float, default=256.0, help="ความละเอียด native ของ thermal core (px กว้าง)")
    a = ap.parse_args()
    report(a.size, a.range, a.dT, a.hot_area, a.netd, a.snr, a.core_px)


if __name__ == "__main__":
    main()
