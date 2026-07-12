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

# ค่าตั้งต้น 'ในโค้ด' — บันทึกไว้ตอน apply override (ก่อนทับ) เพื่อให้หน้า Settings โชว์เทียบได้
# ว่ากำลังปรับออกจากฐานอะไร และย้อนกลับทีละฟิลด์ได้ ถ้าไม่มี override เลย ค่าปัจจุบัน = ค่าตั้งต้น
# (settings_screen จะเก็บให้เองตอน enter)
DEFAULTS: Dict[str, Any] = {}


def default_key(field: "Field", camera_name: str) -> str:
    return f"{camera_name}.{field.key}" if field.scope == "camera" else field.key


def remember_default(key: str, value: Any) -> None:
    if key not in DEFAULTS:
        DEFAULTS[key] = tuple(value) if isinstance(value, list) else value


def get_default(field: "Field", camera_name: str) -> Any:
    return DEFAULTS.get(default_key(field, camera_name))


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


SECTIONS: List[str] = ["Camera", "Arm", "Detection", "Tracking", "Firing"]

SPEC: Dict[str, List[Field]] = {
    "Camera": [
        Field("ACTIVE_CAMERA", "Active camera", "enum", "root", False,
              choices=[], help="Switch camera. Needs restart; a new camera must be calibrated before AUTO/LOCK"),
        Field("rtsp_url", "RTSP URL", "str", "camera", False),
        Field("udp_ip", "UDP IP", "str", "camera", False),
        Field("udp_port", "UDP port", "int", "camera", False, lo=1, hi=65535, step=1),
        Field("use_udp_direct", "UDP direct", "bool", "camera", False),
        Field("stream_format", "Stream format", "enum", "camera", False, choices=["h264", "h265"]),
        Field("width", "Width", "int", "camera", False, lo=320, hi=7680, step=160, unit="px",
              help="Must match the actual decoded stream size, not the datasheet number"),
        Field("height", "Height", "int", "camera", False, lo=240, hi=4320, step=90, unit="px"),
        Field("ego_comp_latency_sec", "Ego-comp latency", "float", "camera", True,
              lo=0.0, hi=0.30, step=0.005, unit="s",
              help="Camera latency (sensor to decode). Let the wizard measure it, do not guess"),
        Field("use_video_file", "Play from video file", "bool", "camera", False),
        Field("video_filename", "Video file", "str", "camera", False),
    ],
    "Arm": [
        Field("CAM4_ARM_ENABLED", "Arm enabled", "bool", "global", False),
        Field("CAM4_ARM_SIMULATION_MODE", "Simulation mode (no serial)", "bool", "global", False),
        Field("CAM4_ARM_SERIAL_PORT", "Serial port", "str", "global", False),
        Field("CAM4_ARM_BAUD_RATE", "Baud rate", "int", "global", False,
              lo=9600, hi=921600, step=9600),
        Field("CAM4_ARM_X_LIMITS", "Pan limits (deg)", "pair", "global", False,
              lo=-180, hi=180, step=1),
        Field("CAM4_ARM_Y_LIMITS", "Tilt limits (deg)", "pair", "global", False,
              lo=-90, hi=90, step=1),
        Field("CAM4_ARM_FEED_RATE", "Feed rate", "int", "global", False,
              lo=1000, hi=30000, step=1000, unit="mm/min"),
        Field("CAM4_ARM_HOME_FEED_RATE", "Feed rate (homing)", "int", "global", False,
              lo=500, hi=20000, step=500, unit="mm/min"),
        Field("CAM4_ARM_RUN_HOMING_ON_START", "Home on startup", "bool", "global", False),
    ],
    "Detection": [
        Field("YOLO_CONF_DETECT", "Conf: detect (AUTO)", "float", "global", True,
              lo=0.05, hi=0.95, step=0.05),
        Field("YOLO_CONF_LOCK", "Conf: LOCK", "float", "global", True,
              lo=0.05, hi=0.95, step=0.05),
        Field("CAM4_ARM_YOLO_DETECTION_MODE", "Model mode", "enum", "global", False,
              choices=["drone_only", "multiclass"]),
        Field("CAM4_ARM_YOLO_ENGINE_RGB_640", "Engine RGB", "str", "global", False),
        Field("CAM4_ARM_YOLO_ENGINE_THERMAL_640", "Engine thermal", "str", "global", False),
        Field("LOCK_ROI_SPAN_DEG", "YOLO ROI span", "float", "global", True,
              lo=2.0, hi=30.0, step=0.5, unit="°",
              help="Fixed angular span, so a drone occupies the same pixels on any camera"),
    ],
    "Tracking": [
        # ทั้งหมดถูกอ่านตอนเรียกใช้จริง (module global ใน _tick_lock) → แก้แล้วมีผลทันที
        Field("LOCK_DEADZONE_DEG", "Deadzone", "float", "global", True,
              lo=0.05, hi=3.0, step=0.05, unit="deg",
              help="Stop the arm when the angular error is below this. Small = tight but can hunt"),
        Field("LOCK_TRACK_STEP_SCALE", "Track step scale", "float", "global", True,
              lo=0.2, hi=2.0, step=0.05,
              help="Arm step gain while tracking. High = snappy but may overshoot"),
        Field("LOCK_LEAD_MAX_DEG", "Tracking lead cap", "float", "global", True,
              lo=0.0, hi=20.0, step=0.5, unit="deg",
              help="Cap on the predictive lead, guards against velocity noise slinging the arm"),
        Field("LOCK_LEAD_EXTRA_SEC", "Actuation lead", "float", "global", True,
              lo=0.0, hi=0.3, step=0.01, unit="s"),
        Field("LOCK_KALMAN_COAST_MAX_SEC", "Max coast", "float", "global", True,
              lo=0.1, hi=3.0, step=0.1, unit="s",
              help="How long to keep predicting after the target is lost before giving up"),
        Field("LOCK_SLEW_GUARD_NEAR_DEG_S", "Slew cap (near)", "float", "global", True,
              lo=5.0, hi=300.0, step=5.0, unit="deg/s"),
        Field("LOCK_SLEW_GUARD_FAR_DEG_S", "Slew cap (far)", "float", "global", True,
              lo=5.0, hi=400.0, step=10.0, unit="deg/s"),
        Field("LOCK_PIPELINE_LATENCY_FALLBACK", "Pipeline latency fallback", "float", "global", True,
              lo=0.0, hi=0.5, step=0.01, unit="s",
              help="Only used until the runtime measures the real value"),
    ],
    "Firing": [
        Field("muzzle_velocity_ms", "Muzzle velocity", "float", "shooter", True,
              lo=100, hi=1500, step=25, unit="m/s"),
        Field("target_size_m", "Target size (drone)", "float", "shooter", True,
              lo=0.05, hi=2.0, step=0.05, unit="m",
              help="Used to estimate range from bbox size. Wrong here = wrong range = wrong fire gate"),
        Field("effective_range_m", "Effective range", "float", "shooter", True,
              lo=10, hi=500, step=10, unit="m"),
        Field("bullet_weight_g", "Bullet weight", "float", "shooter", True,
              lo=1, hi=50, step=0.5, unit="g"),
        Field("boresight_zero_range_m", "Boresight zero range", "float", "shooter", True,
              lo=0, hi=300, step=5, unit="m",
              help="Range you zeroed the crosshair at by live fire. 0 = pure mechanical boresight. "
                   "Gravity drop at this range is already baked into the boresight, so it is "
                   "subtracted from the drop compensation - otherwise it is counted twice"),
        Field("LOCK_FIRE_HIT_RADIUS_M", "Hit radius", "float", "global", True,
              lo=0.05, hi=2.0, step=0.05, unit="m"),
        Field("LOCK_FIRE_CONFIDENCE_K", "Confidence K", "float", "global", True,
              lo=0.1, hi=2.0, step=0.05,
              help="Fire only when uncertainty <= hit_radius * K. Lower = stricter"),
        Field("LOCK_FIRE_NOISE_FLOOR_DEG", "Noise floor (intercept)", "float", "global", True,
              lo=0.0, hi=3.0, step=0.05, unit="°",
              help="Let the wizard measure it. Too high = chases but never fires; too low = fires and misses"),
        Field("LOCK_FIRE_RESID_NOISE_FLOOR_DEG", "Noise floor (residual)", "float", "global", True,
              lo=0.0, hi=3.0, step=0.05, unit="°"),
        Field("LOCK_MEAS_SIGMA_PX", "Sigma of bbox jitter", "float", "global", True,
              lo=0.5, hi=60.0, step=0.5, unit="px",
              help="Feeds Kalman R = (sigma/ppd)^2. Let the wizard measure it"),
        Field("LOCK_FIRE_MAX_RESID_DEG_S", "Max juke residual", "float", "global", True,
              lo=0.5, hi=15.0, step=0.5, unit="deg/s",
              help="Target jinking harder than this is unpredictable - hold fire"),
        Field("LOCK_FIRE_CONFIDENT_FRAMES", "Confident frames", "int", "global", True,
              lo=1, hi=20, step=1,
              help="Consecutive confident frames required before the gate opens"),
        Field("LOCK_FIRE_READY_FRAMES", "Ready frames", "int", "global", True,
              lo=1, hi=20, step=1),
        Field("CAM4_ARM_FIRE_COOLDOWN_SEC", "Fire cooldown", "float", "global", True,
              lo=0.0, hi=5.0, step=0.1, unit="s"),
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
        print(f"WARN runtime_config: cannot read {JSON_PATH.name} ({e}) - using config.py defaults")
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
        print(f"ERROR runtime_config: save failed: {e}")
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
            remember_default(f"{cam_name}.{k}", cams[cam_name].get(k))
            if cams[cam_name].get(k) != v:
                applied.append(f"CAMERAS[{cam_name}].{k} = {v!r}")
            cams[cam_name][k] = v

    for k, v in (data.get("globals") or {}).items():
        if k not in cfg_globals:
            continue   # ไม่ใช่ของ config.py (อาจเป็นของ 22) — apply_to_module จะจัดการ
        remember_default(k, cfg_globals[k])
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
        remember_default(k, mod_globals[k])
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


def clear_value(data: Dict[str, Any], field: Field, camera_name: str) -> None:
    """ลบ override ของฟิลด์เดียว → กลับไปใช้ค่าตั้งต้นในโค้ด"""
    if field.scope == "root":
        data.pop(field.key, None)
    elif field.scope == "camera":
        (data.get("cameras") or {}).get(camera_name, {}).pop(field.key, None)
    else:
        (data.get("globals") or {}).pop(field.key, None)


def clear_all() -> bool:
    """ลบ override ทั้งหมด → กลับไปใช้ค่าตั้งต้นใน config.py"""
    try:
        if JSON_PATH.is_file():
            JSON_PATH.unlink()
        return True
    except Exception:
        return False
