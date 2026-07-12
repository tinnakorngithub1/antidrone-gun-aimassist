"""
ชั้น config ที่แก้ได้ตอนรัน (in-app settings) — วางทับ config.py โดยไม่แตะไฟล์โค้ด

ทำไมต้องมี:
  config.py เป็น module-level constants ล้วน ๆ ไม่มีทางเขียนกลับ ถ้าจะให้ตั้งค่าในแอปได้
  (เปลี่ยน IP กล้อง, สลับกล้อง, ปรับ fire gate) มีสองทาง:
    1. เขียนทับ config.py — ห้าม: ชนกับ git ทุกครั้ง, พังแล้วกู้ยาก, แก้ผิดตัวเดียวรันไม่ขึ้น
    2. ชั้น override แยกไฟล์ (ทางนี้) — config.py ยังเป็น 'ค่าตั้งต้นในโค้ด' ที่อ่าน/รีวิว/
       git diff ได้ตามปกติ ส่วนสิ่งที่ operator แก้หน้างานอยู่ใน JSON ก้อนเดียว ลบทิ้งได้
       เพื่อกลับสู่ค่าตั้งต้น

โครงสร้าง calibration_data/runtime_config.json:
  {
    "ACTIVE_CAMERA": "cam4",
    "cameras": { "cam4": {"rtsp_url": "...", "width": 3840, ...} },   # merge เข้า CAMERAS[cam]
    "globals": { "CAM4_ARM_SERIAL_PORT": "...", "LOCK_FIRE_HIT_RADIUS_M": 0.35 }
  }
  "globals" ใช้กับทั้ง namespace ของ config.py และของ 22_gun_aim_assist_vector.py
  (ค่า fire gate / noise floor อยู่ใน 22 ไม่ได้อยู่ใน config.py)

SPEC ด้านล่างคือ 'สัญญา' ตัวเดียวที่หน้า Settings ใช้สร้าง UI — เพิ่มฟิลด์ = เพิ่มบรรทัดใน SPEC
ไม่ต้องแตะโค้ด UI
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_DIR = Path(__file__).resolve().parent
JSON_PATH = _DIR / "calibration_data" / "runtime_config.json"


# ---------------------------------------------------------------------------
# Field spec — ขับหน้า Settings ทั้งหมด
# ---------------------------------------------------------------------------
# kind:  "str" | "int" | "float" | "bool" | "enum" | "pair"  (pair = ทูเพิลสองค่า เช่น limits)
# scope: "camera" = อยู่ใน CAMERAS[<active>]  |  "global" = ตัวแปร module-level
# live:  True  = แก้แล้วมีผลทันที
#        False = ต้อง restart (ค่านี้ถูกอ่านครั้งเดียวตอนเปิดโปรแกรม เช่น เปิดสตรีม/ต่อ serial)
#        หน้า Settings จะติดป้าย RESTART ให้เห็นชัด ไม่ใช่แก้แล้วเงียบ ๆ ไม่มีผล
class Field:
    def __init__(self, key, label, kind, scope, live,
                 lo=None, hi=None, step=None, choices=None, unit="", help=""):
        self.key = key
        self.label = label
        self.kind = kind
        self.scope = scope
        self.live = live
        self.lo = lo
        self.hi = hi
        self.step = step
        self.choices = choices or []
        self.unit = unit
        self.help = help


SECTIONS: List[str] = ["กล้อง", "แขนกล", "Detection", "การยิง"]

SPEC: Dict[str, List[Field]] = {
    "กล้อง": [
        Field("ACTIVE_CAMERA", "กล้องที่ใช้", "enum", "root", False,
              choices=[], help="สลับกล้อง — ต้อง restart และกล้องใหม่ต้องคาลิเบรตก่อนใช้ AUTO/LOCK"),
        Field("rtsp_url", "RTSP URL", "str", "camera", False),
        Field("udp_ip", "UDP IP", "str", "camera", False),
        Field("udp_port", "UDP port", "int", "camera", False, lo=1, hi=65535, step=1),
        Field("use_udp_direct", "ใช้ UDP direct", "bool", "camera", False),
        Field("stream_format", "รูปแบบสตรีม", "enum", "camera", False, choices=["h264", "h265"]),
        Field("width", "ความกว้าง", "int", "camera", False, lo=320, hi=7680, step=160, unit="px",
              help="ต้องตรงกับที่กล้องส่งมาจริง — ไม่ใช่ตัวเลขจากสเปค"),
        Field("height", "ความสูง", "int", "camera", False, lo=240, hi=4320, step=90, unit="px"),
        Field("ego_comp_latency_sec", "Ego-comp latency", "float", "camera", True,
              lo=0.0, hi=0.30, step=0.005, unit="s",
              help="latency กล้อง (sensor→decode) — ให้ wizard วัดให้ อย่าเดา"),
        Field("use_video_file", "เล่นจากไฟล์วิดีโอ", "bool", "camera", False),
        Field("video_filename", "ไฟล์วิดีโอ", "str", "camera", False),
    ],
    "แขนกล": [
        Field("CAM4_ARM_ENABLED", "เปิดใช้แขนกล", "bool", "global", False),
        Field("CAM4_ARM_SIMULATION_MODE", "โหมดจำลอง (ไม่ต่อ serial)", "bool", "global", False),
        Field("CAM4_ARM_SERIAL_PORT", "Serial port", "str", "global", False),
        Field("CAM4_ARM_BAUD_RATE", "Baud rate", "int", "global", False,
              lo=9600, hi=921600, step=9600),
        Field("CAM4_ARM_X_LIMITS", "ลิมิต pan (องศา)", "pair", "global", False,
              lo=-180, hi=180, step=1),
        Field("CAM4_ARM_Y_LIMITS", "ลิมิต tilt (องศา)", "pair", "global", False,
              lo=-90, hi=90, step=1),
        Field("CAM4_ARM_FEED_RATE", "Feed rate", "int", "global", False,
              lo=1000, hi=30000, step=1000, unit="mm/min"),
        Field("CAM4_ARM_HOME_FEED_RATE", "Feed rate ตอน home", "int", "global", False,
              lo=500, hi=20000, step=500, unit="mm/min"),
        Field("CAM4_ARM_RUN_HOMING_ON_START", "Home ตอนเปิดโปรแกรม", "bool", "global", False),
    ],
    "Detection": [
        Field("YOLO_CONF_DETECT", "Conf ตอน detect (AUTO)", "float", "global", True,
              lo=0.05, hi=0.95, step=0.05),
        Field("YOLO_CONF_LOCK", "Conf ตอน LOCK", "float", "global", True,
              lo=0.05, hi=0.95, step=0.05),
        Field("CAM4_ARM_YOLO_DETECTION_MODE", "โหมดโมเดล", "enum", "global", False,
              choices=["drone_only", "multiclass"]),
        Field("CAM4_ARM_YOLO_ENGINE_RGB_640", "Engine RGB", "str", "global", False),
        Field("CAM4_ARM_YOLO_ENGINE_THERMAL_640", "Engine thermal", "str", "global", False),
        Field("LOCK_ROI_SPAN_DEG", "ROI ที่ป้อน YOLO", "float", "global", True,
              lo=2.0, hi=30.0, step=0.5, unit="°",
              help="ครอบมุมคงที่ → โดรนกินพิกเซลเท่ากันทุกกล้อง"),
    ],
    "การยิง": [
        Field("muzzle_velocity_ms", "ความเร็วปากลำกล้อง", "float", "shooter", True,
              lo=100, hi=1500, step=25, unit="m/s"),
        Field("target_size_m", "ขนาดเป้า (โดรน)", "float", "shooter", True,
              lo=0.05, hi=2.0, step=0.05, unit="m",
              help="ใช้ประเมินระยะจากขนาด bbox — ผิด = ระยะผิด = fire gate ผิด"),
        Field("effective_range_m", "ระยะหวังผล", "float", "shooter", True,
              lo=10, hi=500, step=10, unit="m"),
        Field("bullet_weight_g", "น้ำหนักกระสุน", "float", "shooter", True,
              lo=1, hi=50, step=0.5, unit="g"),
        Field("LOCK_FIRE_HIT_RADIUS_M", "รัศมีปะทะ", "float", "global", True,
              lo=0.05, hi=2.0, step=0.05, unit="m"),
        Field("LOCK_FIRE_CONFIDENCE_K", "ตัวคูณความมั่นใจ", "float", "global", True,
              lo=0.1, hi=2.0, step=0.05,
              help="ยอมยิงเมื่อ uncert ≤ hit_radius × ค่านี้ — ต่ำ = เข้มงวด"),
        Field("LOCK_FIRE_NOISE_FLOOR_DEG", "Noise floor (intercept)", "float", "global", True,
              lo=0.0, hi=3.0, step=0.05, unit="°",
              help="ให้ wizard วัดให้ — สูงไป=ไล่ตามไม่ยิง, ต่ำไป=ยิงแล้วพลาด"),
        Field("LOCK_FIRE_RESID_NOISE_FLOOR_DEG", "Noise floor (residual)", "float", "global", True,
              lo=0.0, hi=3.0, step=0.05, unit="°"),
        Field("LOCK_MEAS_SIGMA_PX", "σ ของ bbox jitter", "float", "global", True,
              lo=0.5, hi=60.0, step=0.5, unit="px",
              help="ป้อน Kalman R = (σ/ppd)² — ให้ wizard วัดให้"),
    ],
}


def all_fields() -> List[Field]:
    out: List[Field] = []
    for sec in SECTIONS:
        out.extend(SPEC[sec])
    return out


def field_by_key(key: str) -> Optional[Field]:
    for f in all_fields():
        if f.key == key:
            return f
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load() -> Dict[str, Any]:
    if not JSON_PATH.is_file():
        return {}
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️ runtime_config: อ่าน {JSON_PATH.name} ไม่ได้ ({e}) — ใช้ค่าตั้งต้นจาก config.py")
        return {}


def save(data: Dict[str, Any]) -> bool:
    try:
        JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = JSON_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(JSON_PATH)   # atomic — ไฟฟ้าดับกลางคันไม่ทิ้งไฟล์ครึ่งใบ
        return True
    except Exception as e:
        print(f"❌ runtime_config: บันทึกไม่ได้: {e}")
        return False


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def apply_to_config(cfg_globals: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    เอา override ทับ namespace ของ config.py — ต้องเรียก *ก่อน* config.py flatten ค่าออกมา
    (CAMERA_WIDTH/RTSP_URL/... ถูกคำนวณตอน import จาก CAMERAS[ACTIVE_CAMERA])
    คืนรายการข้อความสรุปสิ่งที่ทับ (ไว้ print)
    """
    if data is None:
        data = load()
    if not data:
        return []
    applied: List[str] = []

    act = data.get("ACTIVE_CAMERA")
    cams = cfg_globals.get("CAMERAS", {})
    if act and act in cams:
        if act != cfg_globals.get("ACTIVE_CAMERA"):
            applied.append(f"ACTIVE_CAMERA = {act}")
        cfg_globals["ACTIVE_CAMERA"] = act

    for cam_name, overrides in (data.get("cameras") or {}).items():
        if cam_name not in cams or not isinstance(overrides, dict):
            continue
        for k, v in overrides.items():
            if cams[cam_name].get(k) != v:
                applied.append(f"CAMERAS[{cam_name}].{k} = {v!r}")
            cams[cam_name][k] = v

    for k, v in (data.get("globals") or {}).items():
        if k not in cfg_globals:
            continue   # ไม่ใช่ของ config.py (อาจเป็นของ 22) — apply_to_module จะจัดการ
        if isinstance(cfg_globals[k], tuple) and isinstance(v, list):
            v = tuple(v)
        if cfg_globals[k] != v:
            applied.append(f"{k} = {v!r}")
        cfg_globals[k] = v
    return applied


def apply_to_module(mod_globals: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    ทับตัวแปร module-level ของ 22_gun_aim_assist_vector (fire gate, noise floor, YOLO conf ฯลฯ
    ที่ไม่ได้อยู่ใน config.py) — เรียกตอนต้น main() ก่อนค่าพวกนี้ถูกก๊อปลง local
    """
    if data is None:
        data = load()
    applied: List[str] = []
    for k, v in (data.get("globals") or {}).items():
        if k not in mod_globals:
            continue
        if isinstance(mod_globals[k], tuple) and isinstance(v, list):
            v = tuple(v)
        if mod_globals[k] != v:
            applied.append(f"{k} = {v!r}")
        mod_globals[k] = v
    return applied


def set_value(data: Dict[str, Any], field: Field, value: Any, camera_name: str) -> None:
    """เขียนค่าลง dict ตาม scope ของฟิลด์ (ไม่บันทึกลงดิสก์ — เรียก save() เอง)"""
    if field.scope == "root":
        data[field.key] = value
    elif field.scope == "camera":
        data.setdefault("cameras", {}).setdefault(camera_name, {})[field.key] = value
    else:  # global / shooter
        data.setdefault("globals", {})[field.key] = value


def get_override(data: Dict[str, Any], field: Field, camera_name: str) -> Any:
    """ค่าที่ถูก override ไว้ หรือ None ถ้ายังไม่เคยแก้"""
    if field.scope == "root":
        return data.get(field.key)
    if field.scope == "camera":
        return (data.get("cameras") or {}).get(camera_name, {}).get(field.key)
    return (data.get("globals") or {}).get(field.key)


def clear_all() -> bool:
    """ลบ override ทั้งหมด → กลับไปใช้ค่าตั้งต้นใน config.py"""
    try:
        if JSON_PATH.is_file():
            JSON_PATH.unlink()
        return True
    except Exception:
        return False
