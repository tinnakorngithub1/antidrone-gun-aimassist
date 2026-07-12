"""
วาดข้อความ 'ภาษาไทย' บนเฟรม OpenCV

ปัญหา: cv2.putText รองรับแค่ ASCII — ตัวอักษรไทยทุกตัวออกมาเป็น '?' (และ cv2.getTextSize
ก็คำนวณความกว้างจาก '?' ด้วย → จัดวางเพี้ยนตามไปหมด). HUD เดิมของโปรเจกต์เป็นอังกฤษล้วน
เลยไม่เคยเจอ แต่หน้า Settings/Wizard ที่เพิ่มเข้ามาเป็นภาษาไทย

วิธีแก้: เรนเดอร์ข้อความด้วย PIL + ฟอนต์ไทย (tlwg มีติดมากับ Jetson อยู่แล้ว) เป็น 'แผ่นแปะ'
เล็ก ๆ แล้ว alpha-blend ลงเฟรม — พร้อม cache ตาม (ข้อความ, ขนาด, สี) เพราะถ้าแปลงทั้งเฟรม 4K
เป็น PIL ทุกครั้งที่วาดข้อความจะช้ามาก (หน้า Settings วาดข้อความ ~40 ชิ้นต่อเฟรม)

API เลียนแบบ cv2 ให้สลับได้ตรง ๆ: org = มุมล่างซ้าย (baseline) เหมือน cv2.putText
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

# เรียงตามความชอบ: อ่านง่าย เป็น sans-serif, มีน้ำหนักปกติ
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/tlwg/Waree.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/Umpush.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # ไม่มีไทย — ที่พึ่งสุดท้าย
]

_font_path: Optional[str] = None
_fonts: Dict[int, "ImageFont.FreeTypeFont"] = {}
_patches: Dict[Tuple[str, int, Tuple[int, int, int], int], Tuple[np.ndarray, int]] = {}
_MAX_CACHE = 600

# cv2 HERSHEY_SIMPLEX ที่ scale 1.0 สูงประมาณ 22px — คูณ 30 ได้ขนาด PIL ที่ดูใกล้เคียงกัน
_PX_PER_SCALE = 30.0


def _find_font() -> Optional[str]:
    global _font_path
    if _font_path is not None:
        return _font_path or None
    import os
    for p in _FONT_CANDIDATES:
        if os.path.isfile(p):
            _font_path = p
            return p
    _font_path = ""
    return None


def available() -> bool:
    return _PIL and _find_font() is not None


def _get_font(px: int):
    f = _fonts.get(px)
    if f is None:
        f = ImageFont.truetype(_find_font(), px)
        _fonts[px] = f
    return f


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# ฟอนต์ไทย (tlwg) มีอักขระไทย + ° ± × ² ครบ แต่ไม่มีสัญลักษณ์พวกนี้ → ออกมาเป็นกล่องสี่เหลี่ยม
# แทนด้วยตัวที่ฟอนต์วาดได้ ทำตรงนี้ที่เดียว ทุกโมดูลได้ประโยชน์
_GLYPH_FALLBACK = {
    "→": "->", "←": "<-", "↑": "^", "↓": "v",
    "✓": "[OK]", "✗": "[X]", "⚠": "[!]", "…": "...",
    "σ": "sigma", "ω": "w", "Δ": "d", "≈": "~", "≤": "<=", "≥": ">=",
    "•": "-", "—": "-", "–": "-",
}


def _sanitize(text: str) -> str:
    for a, b in _GLYPH_FALLBACK.items():
        if a in text:
            text = text.replace(a, b)
    return text


def _patch(text: str, px: int, color: Tuple[int, int, int], stroke: int):
    """แผ่นข้อความ (alpha, ascent) — cache ไว้ ไม่เรนเดอร์ซ้ำทุกเฟรม"""
    key = (text, px, color, stroke)
    hit = _patches.get(key)
    if hit is not None:
        return hit
    font = _get_font(px)
    ascent, descent = font.getmetrics()
    try:
        w = int(round(font.getlength(text)))
    except AttributeError:                       # Pillow เก่า
        w = font.getsize(text)[0]
    w = max(1, w + 2 * stroke + 2)
    h = max(1, ascent + descent + 2 * stroke + 2)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text(
        (stroke + 1, stroke + 1), text, font=font, fill=255,
        stroke_width=stroke, stroke_fill=255,
    )
    alpha = np.asarray(img, dtype=np.float32) / 255.0
    if len(_patches) > _MAX_CACHE:
        _patches.clear()
    _patches[key] = (alpha, ascent + stroke + 1)
    return _patches[key]


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], font_scale: float,
             color: Tuple[int, int, int], thickness: int = 1) -> None:
    """วาดข้อความ (รองรับไทย). org = มุมล่างซ้าย เหมือน cv2.putText"""
    if not text:
        return
    text = _sanitize(text)
    # ASCII ล้วน → ใช้ cv2 ตามเดิม (เร็วกว่า และหน้าตาเข้ากับ HUD เดิม)
    if _is_ascii(text) or not available():
        cv2.putText(img, text, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, max(1, int(thickness)), cv2.LINE_AA)
        return

    px = max(8, int(round(font_scale * _PX_PER_SCALE)))
    stroke = 1 if thickness >= 3 else 0
    alpha, ascent = _patch(text, px, tuple(int(c) for c in color), stroke)
    ph, pw = alpha.shape
    x0 = int(org[0])
    y0 = int(org[1]) - ascent            # baseline → ขอบบนของแผ่น

    # ตัดส่วนที่ล้นขอบเฟรมออก
    ih, iw = img.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(iw, x0 + pw), min(ih, y0 + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    a = alpha[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0), None]
    roi = img[dy0:dy1, dx0:dx1].astype(np.float32)
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    img[dy0:dy1, dx0:dx1] = (roi * (1.0 - a) + col * a).astype(np.uint8)


def text_size(text: str, font_scale: float, thickness: int = 1) -> Tuple[int, int]:
    """(กว้าง, สูง) — คู่กับ put_text (cv2.getTextSize คำนวณไทยผิดเพราะมองเป็น '?')"""
    text = _sanitize(text)
    if _is_ascii(text) or not available():
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    max(1, int(thickness)))
        return w, h
    px = max(8, int(round(font_scale * _PX_PER_SCALE)))
    font = _get_font(px)
    try:
        w = int(round(font.getlength(text)))
    except AttributeError:
        w = font.getsize(text)[0]
    ascent, _descent = font.getmetrics()
    return w, ascent
