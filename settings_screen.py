"""
หน้า Settings ในแอป (OpenCV overlay) — ตั้งค่าได้ทุกอย่างโดยไม่ต้องแก้โค้ด

ออกแบบตามแพทเทิร์น embed เดิม (cam4_pxdeg_calibrator_embed): enter/leave/is_active/
set_display_size/tick/handle_key — main loop ส่งเฟรมมาให้วาดทับแล้วรับคีย์กลับไป

UI สร้างจาก runtime_config.SPEC ทั้งหมด → เพิ่มค่าที่ตั้งได้ = เพิ่มบรรทัดใน SPEC ไฟล์เดียว
ไม่ต้องแตะไฟล์นี้

สิ่งที่ตั้งใจให้ต่างจากหน้า S เดิม:
  - เดิมมีแค่ 4 ฟิลด์ ballistics และ 'ไม่มีปุ่มยกเลิก' (Esc ไม่ทำอะไร ออกได้ทางเดียวคือ Enter
    ซึ่งบันทึกทันที) → ที่นี่ Esc = ทิ้งการแก้ทั้งหมด
  - เดิมขนาดตัวอักษร hardcode → บนเฟรม 4K ตัวเล็กจิ๋ว → ที่นี่สเกลตาม ui_scale เหมือน bottom HUD
  - เดิมคีย์ลูกศรใช้โค้ด 82/84 ซึ่งชนกับ ord('R')/ord('T') → ที่นี่ใช้เฉพาะโค้ด GTK 65361-65364
  - ค่าที่ต้อง restart ถึงมีผล ติดป้ายให้เห็น ไม่ใช่แก้แล้วเงียบ ๆ ไม่เกิดอะไรขึ้น
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import hud_text as ht
import numpy as np

import runtime_config as rc

_active = False
_section_idx = 0
_field_idx = 0
_data: Dict[str, Any] = {}
_dirty = False
_camera_name = "cam4"
_cfg_mod = None
_gaa_globals: Optional[Dict[str, Any]] = None
_shooter = None
_apply_hook = None          # callback ให้ main() ซิงก์ค่าที่ถูกก๊อปลง local
_editing_text = False      # โหมดพิมพ์ (ฟิลด์ str) — คีย์ทั้งหมดถูกดูดเข้า buffer
_text_buf = ""
_status_msg = ""
_status_color = (200, 200, 200)

# พิกัดที่คลิกได้ (สร้างใหม่ทุกเฟรมตอนวาด) → [(x0,y0,x1,y1, kind, payload)]
_hitboxes: List[Tuple[int, int, int, int, str, Any]] = []

_display_w = _display_h = 0
_content = (0, 0, 0, 0)

C_BG = (22, 22, 26)
C_PANEL = (38, 38, 46)
C_SEL = (0, 140, 255)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 155)
C_OK = (80, 220, 120)
C_WARN = (60, 200, 255)
C_EDIT = (255, 200, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
def is_active() -> bool:
    return _active


def enter(window_name: str, camera_name: str, cfg_mod, gaa_globals: Dict[str, Any], shooter,
          apply_hook=None) -> None:
    """gaa_globals = globals() ของ 22 (เพื่ออ่าน/เขียนค่า fire gate ที่อยู่ที่นั่น)
    shooter    = ShooterConfig instance (ballistics + boresight)
    apply_hook = fn(key, value) ที่ main() ให้มา — เรียกทุกครั้งที่แก้ค่า live เพื่อซิงก์
                 ตัวแปร local ที่ก๊อปค่าไปตั้งแต่ startup (ไม่งั้นแก้แล้วโปรแกรมไม่สนใจ)"""
    global _active, _data, _dirty, _camera_name, _cfg_mod, _gaa_globals, _shooter, _apply_hook
    global _section_idx, _field_idx, _status_msg, _editing_text
    _active = True
    _apply_hook = apply_hook
    _camera_name = camera_name
    _cfg_mod = cfg_mod
    _gaa_globals = gaa_globals
    _shooter = shooter
    _data = rc.load()
    _dirty = False
    _section_idx = 0
    _field_idx = 0
    _editing_text = False
    _status_msg = ""
    # เติมรายชื่อกล้องที่มีจริงเข้า choices ของ ACTIVE_CAMERA
    f = rc.field_by_key("ACTIVE_CAMERA")
    if f is not None and cfg_mod is not None:
        f.choices = sorted(getattr(cfg_mod, "CAMERAS", {}).keys())
    cv2.setMouseCallback(window_name, _on_mouse)


def leave(window_name: str) -> None:
    global _active
    _active = False
    try:
        cv2.setMouseCallback(window_name, lambda *a: None)
    except cv2.error:
        pass


def set_display_size(display_w: int, display_h: int,
                     content_rect: Optional[Tuple[int, int, int, int]] = None) -> None:
    global _display_w, _display_h, _content
    _display_w, _display_h = int(display_w), int(display_h)
    _content = tuple(int(v) for v in content_rect) if content_rect else (0, 0, _display_w, _display_h)


def hud_label() -> str:
    return f"SETTINGS [{rc.SECTIONS[_section_idx]}]" + ("  *unsaved" if _dirty else "")


# ---------------------------------------------------------------------------
# ค่าปัจจุบัน: override (ถ้ามี) > ค่าจริงที่โหลดอยู่
# ---------------------------------------------------------------------------
def _current_value(f: rc.Field) -> Any:
    ov = rc.get_override(_data, f, _camera_name)
    if ov is not None:
        return tuple(ov) if f.kind == "pair" and isinstance(ov, list) else ov
    if f.scope == "root":
        return getattr(_cfg_mod, f.key, None)
    if f.scope == "camera":
        cams = getattr(_cfg_mod, "CAMERAS", {})
        return cams.get(_camera_name, {}).get(f.key)
    if f.scope == "shooter":
        return getattr(_shooter, f.key, None)
    # global: อาจอยู่ใน config.py หรือใน 22
    if _gaa_globals is not None and f.key in _gaa_globals:
        return _gaa_globals[f.key]
    return getattr(_cfg_mod, f.key, None)


def _set_value(f: rc.Field, v: Any) -> None:
    """เขียนลง override + apply ทันทีถ้าเป็นค่า live (ไม่งั้นรอ restart)"""
    global _dirty, _status_msg, _status_color
    rc.set_value(_data, f, list(v) if isinstance(v, tuple) else v, _camera_name)
    _dirty = True
    if not f.live:
        _status_msg = f"{f.label}: takes effect after restart"
        _status_color = C_WARN
        return
    # live → ให้มีผลเดี๋ยวนี้
    if f.scope == "shooter" and _shooter is not None:
        setattr(_shooter, f.key, v)
    elif f.scope == "camera" and _cfg_mod is not None:
        getattr(_cfg_mod, "CAMERAS", {}).get(_camera_name, {})[f.key] = v
    elif _gaa_globals is not None and f.key in _gaa_globals:
        _gaa_globals[f.key] = v
    elif _cfg_mod is not None:
        setattr(_cfg_mod, f.key, v)
    # บางค่าถูกก๊อปลง 'ตัวแปร local' ของ main() ตั้งแต่ startup (เช่น YOLO_CONF_DETECT →
    # runtime_conf_detect, ego_comp_latency_sec → ego_comp_latency) หรือถูกใช้คำนวณค่าอื่นต่อ
    # (LOCK_MEAS_SIGMA_PX → Kalman R) → แก้ module global อย่างเดียวไม่พอ ต้องให้ main() ซิงก์ตาม
    if _apply_hook is not None:
        try:
            _apply_hook(f.key, v)
        except Exception as e:
            print(f"[SETTINGS] apply hook failed for {f.key}: {e}")
    _status_msg = f"{f.label} = {_fmt(f, v)}  (applied now)"
    _status_color = C_OK


def _fmt(f: rc.Field, v: Any) -> str:
    if v is None:
        return "-"
    if f.kind == "bool":
        return "ON" if v else "OFF"
    if f.kind == "pair":
        return f"{float(v[0]):.0f} .. {float(v[1]):.0f}"
    if f.kind == "float":
        return f"{float(v):g}{(' ' + f.unit) if f.unit else ''}"
    if f.kind == "int":
        return f"{int(v)}{(' ' + f.unit) if f.unit else ''}"
    s = str(v)
    return s if len(s) <= 46 else s[:43] + "..."


def _fields() -> List[rc.Field]:
    return rc.SPEC[rc.SECTIONS[_section_idx]]


def _nudge(f: rc.Field, direction: int) -> None:
    """+/- ค่า ตาม step (ใช้กับลูกศรซ้าย-ขวา)"""
    v = _current_value(f)
    if f.kind == "bool":
        _set_value(f, not bool(v))
    elif f.kind == "enum":
        if not f.choices:
            return
        try:
            i = f.choices.index(v)
        except ValueError:
            i = 0
        _set_value(f, f.choices[(i + direction) % len(f.choices)])
    elif f.kind in ("int", "float"):
        step = f.step or (1 if f.kind == "int" else 0.05)
        nv = (float(v) if v is not None else 0.0) + direction * step
        if f.lo is not None:
            nv = max(f.lo, nv)
        if f.hi is not None:
            nv = min(f.hi, nv)
        _set_value(f, int(round(nv)) if f.kind == "int" else round(nv, 4))
    elif f.kind == "pair":
        # ลูกศรปรับ 'ความกว้าง' ของช่วงแบบสมมาตร (ลิมิตแขนมักสมมาตร)
        lo, hi = (float(v[0]), float(v[1])) if v else (0.0, 0.0)
        step = f.step or 1
        hi = hi + direction * step
        lo = lo - direction * step
        if f.lo is not None:
            lo = max(f.lo, lo)
        if f.hi is not None:
            hi = min(f.hi, hi)
        _set_value(f, (lo, hi))


def _begin_text_edit(f: rc.Field) -> None:
    global _editing_text, _text_buf, _status_msg, _status_color
    _editing_text = True
    v = _current_value(f)
    _text_buf = "" if v is None else str(v)
    _status_msg = "Type a value, then Enter = OK / Esc = cancel"
    _status_color = C_EDIT


def _commit_text(f: rc.Field) -> None:
    global _editing_text, _status_msg, _status_color
    _editing_text = False
    raw = _text_buf.strip()
    try:
        if f.kind == "int":
            _set_value(f, int(raw))
        elif f.kind == "float":
            _set_value(f, float(raw))
        else:
            _set_value(f, raw)
    except ValueError:
        _status_msg = f"Invalid value: {raw!r}"
        _status_color = (80, 80, 255)


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------
def _on_mouse(event, x, y, flags, param) -> None:
    global _section_idx, _field_idx
    if event != cv2.EVENT_LBUTTONDOWN or not _active:
        return
    cx, cy, cw, ch = _content
    if cw <= 0 or ch <= 0:
        return
    # พิกัดเมาส์เป็นของ 'หน้าต่าง' → หัก letterbox แล้วสเกลกลับเป็นพิกัดเฟรม (ที่ hitbox ใช้)
    fx = (float(x) - cx) / cw
    fy = (float(y) - cy) / ch
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        return
    mx = fx * _hb_w
    my = fy * _hb_h

    # ปุ่ม -/+/type อยู่ 'ข้างใน' กรอบเลือกแถว (ซึ่งคลุมทั้งแถว) → ต้องเช็คปุ่มก่อนเสมอ
    # ไม่งั้นคลิกปุ่มจะไปโดนกรอบแถวก่อนแล้ว return = แค่เลือกแถว ไม่ได้ปรับค่า
    def _hit(kinds):
        for (x0, y0, x1, y1, kind, payload) in _hitboxes:
            if kind in kinds and x0 <= mx <= x1 and y0 <= my <= y1:
                return kind, payload
        return None, None

    kind, payload = _hit(("dec", "inc", "edit"))
    if kind is None:
        kind, payload = _hit(("section", "field"))
    if kind is None:
        return

    if kind == "section":
        _section_idx = payload
        _field_idx = 0
    elif kind == "field":
        _field_idx = payload
    elif kind == "dec":
        _field_idx = payload
        _nudge(_fields()[payload], -1)
    elif kind == "inc":
        _field_idx = payload
        _nudge(_fields()[payload], +1)
    elif kind == "edit":
        _field_idx = payload
        _begin_text_edit(_fields()[payload])


_hb_w = _hb_h = 1   # ขนาดเฟรมตอนวาด hitbox ล่าสุด


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
def tick(frame: np.ndarray) -> np.ndarray:
    global _hitboxes, _hb_w, _hb_h
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    _hb_w, _hb_h = w, h
    _hitboxes = []

    s = max(0.35, min(2.0, min(h, w) / 1080.0))   # เดียวกับ bottom HUD — สเกลตามความละเอียด
    fs = 0.62 * s
    fs_sm = 0.5 * s
    th = max(1, int(round(1.6 * s)))

    # กรอบหลัก
    pad = int(40 * s)
    x0, y0 = pad, pad
    x1, y1 = w - pad, h - pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), C_BG, -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), C_SEL, max(1, int(2 * s)))

    # หัวเรื่อง
    yy = y0 + int(46 * s)
    ht.put_text(frame, f"SETTINGS  —  {_camera_name}", (x0 + int(24 * s), yy),
                0.85 * s, C_TEXT, max(1, int(2 * s)))

    # แท็บ
    yy += int(34 * s)
    tx = x0 + int(24 * s)
    tab_h = int(34 * s)
    for i, sec in enumerate(rc.SECTIONS):
        tw, _th = ht.text_size(sec, fs, th)
        bw = tw + int(28 * s)
        sel = (i == _section_idx)
        cv2.rectangle(frame, (tx, yy), (tx + bw, yy + tab_h),
                      C_SEL if sel else C_PANEL, -1)
        ht.put_text(frame, sec, (tx + int(14 * s), yy + int(23 * s)),
                    fs, (20, 20, 20) if sel else C_DIM, th)
        _hitboxes.append((tx, yy, tx + bw, yy + tab_h, "section", i))
        tx += bw + int(8 * s)

    # ฟิลด์
    yy += tab_h + int(24 * s)
    row_h = int(38 * s)
    label_x = x0 + int(28 * s)
    val_x = x0 + int((x1 - x0) * 0.46)
    btn_w = int(52 * s)      # ใหญ่พอให้กดด้วยเมาส์บนจอจริงได้สบาย

    for i, f in enumerate(_fields()):
        sel = (i == _field_idx)
        ry = yy + i * row_h
        if ry + row_h > y1 - int(90 * s):
            break
        if sel:
            cv2.rectangle(frame, (x0 + int(10 * s), ry - int(4 * s)),
                          (x1 - int(10 * s), ry + row_h - int(8 * s)), C_PANEL, -1)
        _hitboxes.append((x0 + 10, ry - 4, x1 - 10, ry + row_h - 8, "field", i))

        col = C_TEXT if sel else C_DIM
        tag = "" if f.live else " [RESTART]"
        ht.put_text(frame, f"{i+1}. {f.label}{tag}", (label_x, ry + int(22 * s)), fs, col if f.live else C_WARN, th)

        # ค่า (โหมดพิมพ์ = โชว์ buffer + เคอร์เซอร์)
        if sel and _editing_text:
            vtxt = _text_buf + "_"
            vcol = C_EDIT
        else:
            vtxt = _fmt(f, _current_value(f))
            vcol = C_OK if rc.get_override(_data, f, _camera_name) is not None else col
        ht.put_text(frame, vtxt, (val_x + btn_w + int(14 * s), ry + int(22 * s)), fs, vcol, th)

        if sel and not _editing_text:
            # ปุ่ม -/+ (คลิกได้) สำหรับค่าที่ nudge ได้
            if f.kind in ("int", "float", "bool", "enum", "pair"):
                cv2.rectangle(frame, (val_x, ry), (val_x + btn_w, ry + int(30 * s)), C_SEL, -1)
                ht.put_text(frame, "-", (val_x + int(12 * s), ry + int(21 * s)),
                            fs, (20, 20, 20), th)
                _hitboxes.append((val_x, ry, val_x + btn_w, ry + int(30 * s), "dec", i))
                bx = x1 - int(28 * s) - btn_w
                cv2.rectangle(frame, (bx, ry), (bx + btn_w, ry + int(30 * s)), C_SEL, -1)
                ht.put_text(frame, "+", (bx + int(10 * s), ry + int(21 * s)),
                            fs, (20, 20, 20), th)
                _hitboxes.append((bx, ry, bx + btn_w, ry + int(30 * s), "inc", i))
            if f.kind in ("str", "int", "float"):
                ex = x1 - int(28 * s) - btn_w - int(90 * s)
                cv2.rectangle(frame, (ex, ry), (ex + int(80 * s), ry + int(28 * s)), C_PANEL, -1)
                ht.put_text(frame, "type", (ex + int(8 * s), ry + int(21 * s)), fs_sm, C_TEXT, th)
                _hitboxes.append((ex, ry, ex + int(80 * s), ry + int(28 * s), "edit", i))

    # help ของฟิลด์ที่เลือก
    cur = _fields()[_field_idx] if _fields() else None
    by = y1 - int(78 * s)
    if cur is not None and cur.help:
        ht.put_text(frame, cur.help, (label_x, by), fs_sm, C_DIM, th)

    # สถานะ + คีย์
    if _status_msg:
        ht.put_text(frame, _status_msg, (label_x, by + int(24 * s)), fs_sm, _status_color, th)
    hint = ("Up/Down = select   Left/Right = adjust   E or Enter = type value   "
            "Tab = next tab   F5 = reset defaults   Ctrl+S = save   Esc = discard & exit")
    ht.put_text(frame, hint, (label_x, y1 - int(20 * s)), fs_sm, C_DIM, th)
    return frame


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
# ใช้เฉพาะโค้ด GTK: 82/84 ชนกับ ord('R')/ord('T') — บั๊กที่หน้า S เดิมมีอยู่
K_LEFT, K_UP, K_RIGHT, K_DOWN = 65361, 65362, 65363, 65364
K_ENTER = (13, 10)
K_ESC = 27
K_TAB = 9
K_F5 = 65474
K_CTRL_S = 19   # Ctrl+S


def handle_key(key: int) -> str:
    """คืน 'none' | 'saved' | 'exit' | 'reset'"""
    global _section_idx, _field_idx, _editing_text, _text_buf, _status_msg, _status_color, _data, _dirty
    if key < 0:
        return "none"
    flds = _fields()
    cur = flds[_field_idx] if flds else None

    # โหมดพิมพ์ดูดคีย์ทั้งหมด (ไม่งั้นพิมพ์ 'q' แล้วโปรแกรมปิด)
    if _editing_text and cur is not None:
        if key in K_ENTER:
            _commit_text(cur)
        elif key == K_ESC:
            _editing_text = False
            _status_msg = "Edit cancelled"
            _status_color = C_DIM
        elif key in (8, 127):          # backspace
            _text_buf = _text_buf[:-1]
        elif 32 <= key < 127:
            _text_buf += chr(key)
        return "none"

    if key == K_ESC:
        _status_msg = ""
        return "exit"                  # ทิ้งการแก้ — override ที่ยังไม่ save ไม่ถูกเขียนลงไฟล์

    if key == K_CTRL_S:
        if rc.save(_data):
            _dirty = False
            _status_msg = f"Saved to {rc.JSON_PATH.name}"
            _status_color = C_OK
            return "saved"
        _status_msg = "Save failed"
        _status_color = (80, 80, 255)
        return "none"

    if key == K_F5:
        rc.clear_all()
        _data = {}
        _dirty = False
        _status_msg = "All overrides cleared - using config.py defaults (restart to fully apply)"
        _status_color = C_WARN
        return "reset"

    if key == K_TAB:
        _section_idx = (_section_idx + 1) % len(rc.SECTIONS)
        _field_idx = 0
    elif key == K_UP:
        _field_idx = (_field_idx - 1) % max(1, len(flds))
    elif key == K_DOWN:
        _field_idx = (_field_idx + 1) % max(1, len(flds))
    elif key == K_LEFT and cur is not None:
        _nudge(cur, -1)
    elif key == K_RIGHT and cur is not None:
        _nudge(cur, +1)
    elif cur is not None and (key in K_ENTER or key in (ord("e"), ord("E"))):
        if cur.kind in ("str", "int", "float"):
            _begin_text_edit(cur)
        else:
            _nudge(cur, +1)
    elif ord("1") <= key <= ord("9"):
        i = key - ord("1")
        if i < len(flds):
            _field_idx = i
    return "none"


def is_dirty() -> bool:
    return _dirty
