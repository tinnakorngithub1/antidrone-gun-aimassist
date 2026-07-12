"""
Cam4 Arm Grid Lookup
--------------------
- โหลด JSON จาก calibration mouse grid (calibration_data/cam4_mouse_grid_lookup.json)
- คืน (pan_deg, tilt_deg) จากตำแหน่ง pixel บน output frame
- ถ้ามี calibration_points (เช่น cam8_mouse_grid_lookup.json): snap + IDW จากจุดวัดก่อน
  แล้วค่อย homography / bilinear grid ตามเดิม
- ดึง crosshair จาก JSON (รองรับ override ตอนใช้จริง)
- การเชื่อมกับ gun_aim_assist/tracker: opt-in เท่านั้น — ใน config ใส่ CAM4_ARM_USE_MOUSE_GRID_CALIBRATION = True
  เมื่อปิดหรือไม่กำหนด ระบบใช้ flow เดิม (hand-eye/FOV) ไม่กระทบการใช้งานปัจจุบัน
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"

# Lookup จาก calibration_points โดยตรง (ไม่ต้อง cal ใหม่ถ้า JSON มีรายการจุด)
SCATTER_LOOKUP_ENABLED = True
SCATTER_SNAP_EPSILON_PX = 22.0
SCATTER_IDW_K = 12
SCATTER_IDW_POWER = 2.0
MOUSE_GRID_LOOKUP_FILE = CALIBRATION_DIR / "cam4_mouse_grid_lookup.json"
PX_DEG_LOOKUP_FILE = CALIBRATION_DIR / "cam4_pixel_per_degree.json"  # แยกจาก homography


def load(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    โหลด JSON calibration grid. คืน dict หรือ None ถ้าไม่มีไฟล์/ผิดรูปแบบ.
    """
    p = path if path is not None else MOUSE_GRID_LOOKUP_FILE
    if not p.is_file():
        return None
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "output_width" not in data or "output_height" not in data or "cells" not in data:
            return None
        return data
    except Exception:
        return None


def load_pixel_per_degree(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    โหลดไฟล์ cam4_pixel_per_degree.json (โหมด linear Px/deg แยกจาก homography).
    คืน dict มี crosshair, pixel_per_degree_x, pixel_per_degree_y หรือ None.
    """
    p = path if path is not None else PX_DEG_LOOKUP_FILE
    if not p.is_file():
        return None
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("pixel_per_degree_x") is None or data.get("pixel_per_degree_y") is None:
            return None
        return data
    except Exception:
        return None


def get_crosshair(data: Dict[str, Any]) -> Tuple[float, float]:
    """
    ดึงตำแหน่งศูนย์เล็งจาก data. รองรับ override ถ้าใน data มี "crosshair_override".
    """
    override = data.get("crosshair_override")
    if override is not None and isinstance(override, dict):
        x = override.get("x")
        y = override.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    ch = data.get("crosshair") or data.get("aim_center")
    if isinstance(ch, dict):
        x = ch.get("x", 0)
        y = ch.get("y", 0)
        return float(x), float(y)
    # ไม่มี crosshair ในไฟล์ → ใช้กลางภาพ. ห้าม 'เดา' ขนาดภาพ (เดิม default 1920×1080 ซึ่งไม่ตรง
    # กับไฟล์ cam4 จริงที่เป็น 3840×2160 อยู่แล้ว → ศูนย์เล็งเพี้ยนครึ่งเฟรมแบบเงียบ ๆ)
    ow = data.get("output_width")
    oh = data.get("output_height")
    if not ow or not oh:
        raise ValueError(
            "grid lookup: ไฟล์คาลิเบรตไม่มีทั้ง crosshair และ output_width/height "
            "— ระบุขนาดภาพที่คาลิเบรตไว้ไม่ได้ ต้องคาลิเบรตใหม่"
        )
    return float(ow) / 2.0, float(oh) / 2.0


def _cells_2d(data: Dict[str, Any]) -> Tuple[Optional[list], int, int]:
    """คืน (grid 2D ของ [pan, tilt], rows, cols) จาก cells dict key "row_col"."""
    cells = data.get("cells")
    if not isinstance(cells, dict):
        return None, 0, 0
    rows = int(data.get("grid_rows", 0))
    cols = int(data.get("grid_cols", 0))
    if rows <= 0 or cols <= 0:
        return None, 0, 0
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for key, val in cells.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0]), int(parts[1])
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = [float(val[0]), float(val[1])]
        except (ValueError, IndexError):
            continue
    return grid, rows, cols


def _fine_cells_2d(center_fine: Dict[str, Any]) -> Tuple[Optional[list], int, int]:
    """คืน (fine grid 2D, fine_rows, fine_cols) จาก center_fine_grid."""
    if not isinstance(center_fine, dict):
        return None, 0, 0
    rows = int(center_fine.get("fine_rows", 0))
    cols = int(center_fine.get("fine_cols", 0))
    if rows <= 0 or cols <= 0:
        return None, 0, 0
    cells = center_fine.get("fine_cells")
    if not isinstance(cells, dict):
        return None, rows, cols
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for key, val in cells.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0]), int(parts[1])
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = [float(val[0]), float(val[1])]
        except (ValueError, IndexError):
            continue
    return grid, rows, cols


def _bilinear_from_grid(
    grid: list,
    rows: int,
    cols: int,
    fx: float,
    fy: float,
) -> Optional[Tuple[float, float]]:
    """fx,fy ใน [0, cols), [0, rows). Bilinear จาก grid 2D. คืน (pan, tilt) หรือ None."""
    c0 = max(0, min(cols - 1, int(fx)))
    r0 = max(0, min(rows - 1, int(fy)))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)
    tx = max(0.0, min(1.0, fx - int(fx)))
    ty = max(0.0, min(1.0, fy - int(fy)))

    p00 = grid[r0][c0] if grid[r0][c0] is not None else None
    p10 = grid[r0][c1] if grid[r0][c1] is not None else None
    p01 = grid[r1][c0] if grid[r1][c0] is not None else None
    p11 = grid[r1][c1] if grid[r1][c1] is not None else None

    # ถ้ามีครบ 4 มุม ทำ bilinear
    if p00 and p10 and p01 and p11:
        pan = (1 - tx) * (1 - ty) * p00[0] + tx * (1 - ty) * p10[0] + (1 - tx) * ty * p01[0] + tx * ty * p11[0]
        tilt = (1 - tx) * (1 - ty) * p00[1] + tx * (1 - ty) * p10[1] + (1 - tx) * ty * p01[1] + tx * ty * p11[1]
        return (pan, tilt)
    # fallback: ใช้จุดที่มี
    for p in (p00, p10, p01, p11):
        if p is not None:
            return (p[0], p[1])
    return None


def _in_center_cell(
    px: float,
    py: float,
    output_w: int,
    output_h: int,
    grid_cols: int,
    grid_rows: int,
    center_row: int,
    center_col: int,
) -> bool:
    """ตรวจว่า (px, py) อยู่ในเซลล์กลางของ coarse grid หรือไม่."""
    fx = (px / output_w) * grid_cols
    fy = (py / output_h) * grid_rows
    return int(fx) == center_col and int(fy) == center_row


def _calibration_samples_from_data(data: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """คืน [(px, py, pan, tilt), ...] จาก calibration_points ถ้ามีและอ่านได้."""
    raw = data.get("calibration_points")
    if not isinstance(raw, list) or not raw:
        return []
    out: List[Tuple[float, float, float, float]] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        x = p.get("cam8_px", p.get("output_px"))
        y = p.get("cam8_py", p.get("output_py"))
        ap = p.get("arm_pan")
        at = p.get("arm_tilt")
        if x is None or y is None or ap is None or at is None:
            continue
        try:
            fx = float(x)
            fy = float(y)
            pan = float(ap)
            tilt = float(at)
        except (TypeError, ValueError):
            continue
        if all(map(math.isfinite, (fx, fy, pan, tilt))):
            out.append((fx, fy, pan, tilt))
    return out


def _pixel_to_arm_scatter_snap_idw(
    px: float,
    py: float,
    data: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """
    Snap: ถ้าใกล้จุด cal ภายใน epsilon → คืนมุมจุดนั้น
    ไม่งั้น IDW จาก k จุดใกล้สุด (inverse distance^power)
    """
    samples = _calibration_samples_from_data(data)
    if not samples:
        return None
    ow = int(data.get("output_width", 0))
    oh = int(data.get("output_height", 0))
    eps = SCATTER_SNAP_EPSILON_PX
    if ow > 0 and oh > 0:
        eps = max(eps, min(ow, oh) * 0.015)

    best_i = 0
    best_d2 = float("inf")
    for i, (sx, sy, _, _) in enumerate(samples):
        dx = px - sx
        dy = py - sy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    if best_d2 < eps * eps:
        _, _, pan0, tilt0 = samples[best_i]
        return (pan0, tilt0)

    dists: List[Tuple[float, Tuple[float, float, float, float]]] = []
    tiny = 1e-9
    for sx, sy, pan, tilt in samples:
        d = math.hypot(px - sx, py - sy)
        dists.append((d, (sx, sy, pan, tilt)))
    dists.sort(key=lambda t: t[0])
    kn = min(SCATTER_IDW_K, len(dists))
    w_sum = 0.0
    pan_sum = 0.0
    tilt_sum = 0.0
    pwr = SCATTER_IDW_POWER
    for i in range(kn):
        d, quad = dists[i]
        _, _, pan, tilt = quad
        if d < tiny:
            return (pan, tilt)
        w = 1.0 / (d**pwr)
        w_sum += w
        pan_sum += w * pan
        tilt_sum += w * tilt
    if w_sum < 1e-15:
        return None
    return (pan_sum / w_sum, tilt_sum / w_sum)


def _apply_homography(px: float, py: float, H: List[List[float]]) -> Optional[Tuple[float, float]]:
    """ใช้ matrix 3x3 H แมป (px, py) → (pan, tilt). H เป็น list of 3 lists of 3 floats."""
    if not H or len(H) != 3 or len(H[0]) != 3:
        return None
    try:
        Hnp = np.array(H, dtype=np.float64)
        p = Hnp @ np.array([px, py, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))
    except Exception:
        return None


def _apply_homography_inv(pan_deg: float, tilt_deg: float, H: List[List[float]]) -> Optional[Tuple[float, float]]:
    """ใช้ H^(-1) แมป (pan_deg, tilt_deg) → (px, py). สำหรับแสดงตำแหน่งที่แขนชี้จริงในพิกเซล."""
    if not H or len(H) != 3 or len(H[0]) != 3:
        return None
    try:
        Hnp = np.array(H, dtype=np.float64)
        H_inv = np.linalg.inv(Hnp)
        q = H_inv @ np.array([pan_deg, tilt_deg, 1.0])
        if abs(q[2]) < 1e-9:
            return None
        return (float(q[0] / q[2]), float(q[1] / q[2]))
    except Exception:
        return None


def arm_degrees_to_pixel(
    pan_deg: float,
    tilt_deg: float,
    data: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """
    แปลง (pan_deg, tilt_deg) เป็นตำแหน่ง pixel บน output frame ( inverse ของ pixel_to_arm_degrees ).
    ใช้ได้เมื่อมี homography เท่านั้น — สำหรับแสดง error ในพิกเซลและ re-calibration.
    คืน (px, py) หรือ None ถ้าไม่มี homography / คำนวณไม่ได้.
    """
    if not data:
        return None
    homography = data.get("homography")
    if not isinstance(homography, list) or len(homography) != 3:
        return None
    return _apply_homography_inv(pan_deg, tilt_deg, homography)


def pixel_to_arm_degrees(
    px: float,
    py: float,
    data: Dict[str, Any],
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    use_homography: bool = True,
) -> Optional[Tuple[float, float]]:
    """
    แปลงตำแหน่ง pixel บน output frame เป็น (pan_deg, tilt_deg).
    ถ้ามี calibration_points และ SCATTER_LOOKUP_ENABLED: snap + IDW จากจุดวัดก่อน
    จากนั้นถ้า use_homography และมี homography ใช้ matrix; ไม่มีหรือผิดพลาด fallback bilinear grid.
    ถ้า frame_w/frame_h ไม่ตรงกับ output_width/height จะ scale px,py ก่อน lookup.
    """
    if not data:
        return None
    ow = int(data.get("output_width", 0))
    oh = int(data.get("output_height", 0))
    if ow <= 0 or oh <= 0:
        return None
    if frame_w is not None and frame_h is not None and (frame_w != ow or frame_h != oh):
        px = px * ow / frame_w
        py = py * oh / frame_h
    if SCATTER_LOOKUP_ENABLED:
        sc = _pixel_to_arm_scatter_snap_idw(px, py, data)
        if sc is not None:
            return sc
    if use_homography:
        homography = data.get("homography")
        if isinstance(homography, list) and len(homography) == 3:
            res = _apply_homography(px, py, homography)
            if res is not None:
                return res
    grid, rows, cols = _cells_2d(data)
    if grid is None or rows == 0 or cols == 0:
        return None

    fx = (px / ow) * cols
    fy = (py / oh) * rows
    center_row = rows // 2
    center_col = cols // 2

    # ถ้ามี center_fine_grid และจุดอยู่ในช่องกลาง ลองใช้ fine ก่อน
    center_fine = data.get("center_fine_grid")
    if isinstance(center_fine, dict) and _in_center_cell(px, py, ow, oh, cols, rows, center_row, center_col):
        fine_grid, fine_rows, fine_cols = _fine_cells_2d(center_fine)
        if fine_grid is not None and fine_rows > 0 and fine_cols > 0:
            # map (px,py) ภายในเซลล์กลางไปเป็น fine grid coords
            cell_w = ow / cols
            cell_h = oh / rows
            left = center_col * cell_w
            top = center_row * cell_h
            local_x = px - left
            local_y = py - top
            ffx = (local_x / cell_w) * fine_cols
            ffy = (local_y / cell_h) * fine_rows
            res = _bilinear_from_grid(fine_grid, fine_rows, fine_cols, ffx, ffy)
            if res is not None:
                return res

    return _bilinear_from_grid(grid, rows, cols, fx, fy)