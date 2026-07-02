"""
cam8_arm_grid_calibrator.py
---------------------------
Calibration tool: map cam8 pixel → arm pan/tilt for AUTO cue.
Saves to calibration_data/cam8_mouse_grid_lookup.json (separate from cam4 calib).
Run on arm Jetson (192.168.144.66) only.

Usage:
  python3 cam8_arm_grid_calibrator.py               # Full calibration (3 phases)
  python3 cam8_arm_grid_calibrator.py --recal       # Fast recalibration using reference point
  python3 cam8_arm_grid_calibrator.py --cam-top cam8 --cam-bottom cam4   # override file defaults
  python3 cam8_arm_grid_calibrator.py --cam8 cam8 --cam4 cam4            # legacy (same as above)
  python3 cam8_arm_grid_calibrator.py --use-config-py        # Cameras from config.py
  python3 cam8_arm_grid_calibrator.py --use-file-11-cameras  # file 11 LOCAL_CAMERAS (not embedded)
  python3 cam8_arm_grid_calibrator.py --guided-grid   # Phase 2: red dot at each grid center in reachable zone

Edit CALIB_TOP_CAMERA / CALIB_BOTTOM_CAMERA and CALIB_LOCAL_CAMERAS below for streams and URLs.
Default: embedded CALIB_LOCAL_CAMERAS + fast_motion_sky.CameraStream (same shape as file 11).

─── 3 Phases (Full Calibration) ───────────────────────────────
Phase 0 — REFERENCE SETUP:
  Camera reference is fixed at center-x, 10% from bottom of cam8.
  Place a physical target at the reference mark ⊕ in the field.
  Rotate arm until cam4 sees the target at center, then press C.
  → Records home_arm_pan, home_arm_tilt (datum for recalibration).

Phase 1 — LIMIT (outline by order):
  Click on CAM8 (top) where you want each corner of the zone, then press C to record.
  Vertices connect in click order (polygon on screen; not a convex box). ≥3 vertices;
  SPACE → Phase 2. D deletes last vertex; L clears all. (Moving the arm is optional.)

Phase 2 — CALIBRATION:
  Default: click inside reachable zone → aim cam4 → C (≥9 points) → S save.
  With --guided-grid: red dot at each grid cell center in zone → aim cam4 → C;
  SPACE skips a cell; D undoes skip/point. Optional clicks after all dots → S save.

─── Recalibration (--recal) ────────────────────────────────────
Loads existing calibration and adjusts using reference point only:
  Rotate arm back to aim at reference mark ⊕ → press C to confirm.
  System computes delta (pan/tilt), applies to all points, re-fits homography → S.

Keys (all phases):
  C / Joystick B0   - confirm current step (reference / limit vertex / calib point)
  SPACE / Enter     - skip or advance to next phase
  D                 - delete last point
  S                 - save JSON
  R                 - reset calibration points (keeps home/limits)
  H                 - move arm to saved reference position
  L                 - reset limit sweep entirely
  Q / ESC           - quit
"""

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from fast_motion_sky import CameraStream as _CalibCameraStream
except ImportError:
    _CalibCameraStream = None

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
OUTPUT_JSON = CALIBRATION_DIR / "cam8_mouse_grid_lookup.json"

# Match detector script: LOCAL_CAMERAS + build_camera() inside this file
ANTIDRONE_11_SCRIPT = Path(__file__).resolve().parent / (
    "11_AntidroneAlert_vibe_grid_yolo_trian_motion_ignore.py"
)
_ANTIDRONE_11_MODULE_KEY = "_cam8_calibrator_antidrone11"


def _load_antidrone11_module():
    """Load file 11 once (cached in sys.modules). Same CameraStream setup as the main app."""
    if _ANTIDRONE_11_MODULE_KEY in sys.modules:
        return sys.modules[_ANTIDRONE_11_MODULE_KEY]
    if not ANTIDRONE_11_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing: {ANTIDRONE_11_SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        _ANTIDRONE_11_MODULE_KEY, ANTIDRONE_11_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("importlib could not load file 11")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ANTIDRONE_11_MODULE_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


# True = use CALIB_LOCAL_CAMERAS in this file. False = skip embedded (file 11 / config only).
CALIB_USE_EMBEDDED_CAMERA_PARAMS = True

# Same keys/fields as file 11 LOCAL_CAMERAS — edit RTSP/UDP here for calibration without touching 11.
CALIB_LOCAL_CAMERAS: Dict[str, Dict[str, Any]] = {
    "cam1": {
        "name": "cam1",
        "width": 2560,
        "height": 1440,
        "video_filename": "55.mp4",
        "use_video_file": True,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.1.203:554/Streaming/channels/201",
        "fov_horizontal": 96.1,
        "fov_vertical": 52.1,
    },
    "cam2": {
        "name": "cam2",
        "width": 2560,
        "height": 1440,
        "video_filename": "DroneNighttime.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.1.203:554/Streaming/channels/101",
        "fov_horizontal": 55.0,
        "fov_vertical": 33.0,
        "fov_tele_horizontal": 2.4,
        "fov_tele_vertical": 1.4,
        "fov_tele_diagonal": 2.8,
        "zoom_max": 25.0,
    },
    "cam3": {
        "name": "cam3",
        "width": 1280,
        "height": 720,
        "video_filename": "66.mp4",
        "use_video_file": True,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.144.201:554/Streaming/channels/201",
        "udp_ip": "192.168.144.201",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "fov_horizontal": 66.0,
        "fov_vertical": 33.0,
        "fov_tele_horizontal": 2.4,
        "fov_tele_vertical": 1.4,
        "zoom_max": 25.0,
    },
    "cam4": {
        "name": "cam4",
        "width": 3840,
        "height": 2160,
        "video_filename": "55.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Things22@192.168.144.15/11",
        "udp_ip": "192.168.144.15",
        "udp_port": 6600,
        "use_udp_direct": True,
        "stream_format": "h264",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam5": {
        "name": "cam5",
        "width": 1280,
        "height": 720,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": 0,
        "use_udp_direct": False,
        "stream_format": "h264",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam6": {
        "name": "cam6",
        "width": 1280,
        "height": 720,
        "video_filename": "55.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://192.168.144.108:554/stream=1",
        "udp_ip": "192.168.144.108",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam7": {
        "name": "cam7",
        "width": 1280,
        "height": 720,
        "video_filename": "55.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://192.168.144.108:555/stream=2",
        "udp_ip": "192.168.144.108",
        "udp_port": 555,
        "use_udp_direct": True,
        "stream_format": "h265",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam8": {
        "name": "cam8",
        "width": 5120,
        "height": 1440,
        "video_filename": "55.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Things22@192.168.144.112:554/Streaming/channels/101",
        "udp_ip": "192.168.144.112",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "fov_horizontal": 180.0,
        "fov_vertical": 40.0,
    },
    "cam9": {
        "name": "cam9",
        "width": 5120,
        "height": 1440,
        "video_filename": "55.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Things22@192.168.144.113:554/Streaming/channels/101",
        "udp_ip": "192.168.144.113",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "fov_horizontal": 180.0,
        "fov_vertical": 40.0,
    },
}


def _get_calib_embedded_config(camera_name: str) -> Dict[str, Any]:
    if camera_name not in CALIB_LOCAL_CAMERAS:
        raise KeyError(
            f"Camera '{camera_name}' not in CALIB_LOCAL_CAMERAS. "
            "Add it or use --use-file-11-cameras / --use-config-py."
        )
    return CALIB_LOCAL_CAMERAS[camera_name]


def _build_camera_embedded(camera_name: Optional[str] = None) -> Any:
    """CameraStream from CALIB_LOCAL_CAMERAS (same contract as file 11 build_camera)."""
    if _CalibCameraStream is None:
        raise RuntimeError("fast_motion_sky.CameraStream not available")
    name = camera_name or CALIB_TOP_CAMERA
    cfg = _get_calib_embedded_config(name)
    use_vf = cfg.get("use_video_file", False)
    source = cfg["video_filename"] if use_vf else cfg["rtsp_url"]
    if use_vf and isinstance(source, str) and source and not os.path.isabs(source):
        source = os.path.join(os.path.dirname(__file__), source)
    return _CalibCameraStream(
        source=source,
        width=cfg["width"],
        height=cfg["height"],
        use_video_file=use_vf,
        camera_name=cfg.get("name", name),
        udp_ip=cfg.get("udp_ip"),
        udp_port=cfg.get("udp_port"),
        use_udp_direct=cfg.get("use_udp_direct", False),
        stream_format=cfg.get("stream_format", "h264"),
    )


def _try_file11_build_camera():
    """Return (callable, desc) or None if file 11 cannot provide build_camera."""
    try:
        mod = _load_antidrone11_module()
        fn = getattr(mod, "build_camera", None)
        if callable(fn):
            return (
                fn,
                f"file 11 build_camera + LOCAL_CAMERAS ({ANTIDRONE_11_SCRIPT.name})",
            )
    except Exception as e:
        print(f"[CAL] Warning: cannot use file 11 for cameras ({e})")
    return None


def _resolve_camera_builder(use_config_py: bool, use_file_11: bool):
    """
    Priority:
      1) use_config_py -> gun_aim_assist / config.py
      2) use_file_11 -> file 11 build_camera
      3) embedded (CALIB_USE_EMBEDDED_CAMERA_PARAMS + CameraStream) -> CALIB_LOCAL_CAMERAS
      4) file 11, then config.py fallback
    Returns (builder_callable, description).
    """
    if use_config_py:
        if build_camera_from_config is None:
            raise RuntimeError("gun_aim_assist.build_camera_from_config not available")
        return (
            build_camera_from_config,
            "gun_aim_assist.build_camera_from_config (config.py CAMERAS)",
        )

    if use_file_11:
        got = _try_file11_build_camera()
        if got is not None:
            return got
        if build_camera_from_config is None:
            raise RuntimeError(
                "file 11 unavailable and gun_aim_assist.build_camera_from_config missing"
            )
        return (
            build_camera_from_config,
            "gun_aim_assist.build_camera_from_config (config.py) [file 11 failed]",
        )

    if CALIB_USE_EMBEDDED_CAMERA_PARAMS and _CalibCameraStream is not None:
        return (
            _build_camera_embedded,
            "embedded CALIB_LOCAL_CAMERAS in cam8_arm_grid_calibrator.py",
        )

    got = _try_file11_build_camera()
    if got is not None:
        return got

    if build_camera_from_config is None:
        raise RuntimeError(
            "gun_aim_assist.build_camera_from_config not available "
            "(embedded disabled / CameraStream missing, and file 11 failed). "
            "Check imports / run from project root."
        )
    return (
        build_camera_from_config,
        "gun_aim_assist.build_camera_from_config (config.py CAMERAS) [fallback]",
    )

# Home reference: center-x, 10% from bottom of cam8
HOME_REFERENCE_Y_RATIO = 0.9   # 90% down from top = 10% from bottom

# Grid overlay on cam8 + hybrid table saved in JSON (2× prior density for finer arm mapping).
GRID_COLS = 20
GRID_ROWS = 10
HYBRID_COARSE_GRID_COLS = 18
HYBRID_COARSE_GRID_ROWS = 10
HYBRID_FINE_GRID_COLS = 10
HYBRID_FINE_GRID_ROWS = 10

# Phase 2: --guided-grid visits every cell center inside reachable polygon (no lens calib).
# Default off; use click-anywhere mode without the flag.

# Joystick
JOY_CONFIRM_BUTTON = 0
SYNC_INTERVAL_SEC = 0.3

# Camera names: keys in CALIB_LOCAL_CAMERAS (default), file 11 LOCAL_CAMERAS, or config.py CAMERAS.
CALIB_TOP_CAMERA = "cam8"      # top panel — click / grid calibration (wide view)
CALIB_BOTTOM_CAMERA = "cam4"   # bottom panel — arm camera (crosshair aim)

# Display: top / bottom panels — same layout math as 11 multi_camera_top_bottom single_window
TOP_PANEL_HEIGHT_RATIO = 0.5  # match 11_AntidroneAlert_vibe_grid_yolo_trian_motion_ignore.py
# If primary screen size cannot be read, use this canvas (then mouse top_h is derived from ratio)
DISPLAY_FALLBACK_W = 1280
DISPLAY_FALLBACK_H = 720
# Downscale stacked window if screen is tall (fewer pixels → faster letterbox + imshow). 0 = no cap.
CALIB_DISPLAY_MAX_HEIGHT = 720
# Same as 11 composite: cv2.setWindowProperty(..., WINDOW_FULLSCREEN)
CALIB_USE_FULLSCREEN = True
# Match 11 composite loop (waitKeyEx(1)); keeps UI responsive
CALIB_WAIT_KEY_MS = 1

# =================================================================
# Dependencies
# =================================================================

try:
    from gun_aim_assist import build_camera_from_config
except ImportError:
    build_camera_from_config = None

try:
    from cam4_arm_controller import Cam4ArmController
except ImportError:
    Cam4ArmController = None

try:
    from joystick_cam4_controller import JoystickReader, JoystickArmMapper, SENSITIVITY_LOW
except ImportError:
    JoystickReader = None
    JoystickArmMapper = None
    SENSITIVITY_LOW = 0

try:
    import config as _config_mod
except ImportError:
    _config_mod = None

try:
    from cam4_arm_grid_lookup import arm_degrees_to_pixel as _a2p, pixel_to_arm_degrees as _p2a
except ImportError:
    _a2p = None
    _p2a = None


# =================================================================
# State
# =================================================================

class _CalibState:
    # Phase: "home" -> "limits" -> "calibration"
    phase: str = "home"

    # Phase 0: Home reference
    home_confirmed: bool = False
    home_arm_pan: float = 0.0
    home_arm_tilt: float = 0.0
    # home cam8 position computed from frame size (center-x, 90% down)
    home_cam8_px: float = 0.0
    home_cam8_py: float = 0.0

    # Phase 1: Limit outline (click CAM8 + C; polygon = vertices in recorded order)
    limit_corners: List[Dict] = None
    reachable_poly_cam8: Optional[np.ndarray] = None

    # Phase 2: Calibration
    cam8_click: Optional[Tuple[float, float]] = None
    points: List[Dict] = None
    # Guided grid (--guided-grid): targets = cell centers in reachable polygon
    calib_use_guided_grid: bool = False
    guided_targets: List[Dict[str, Any]] = None
    guided_idx: int = 0
    guided_built_for_w: int = 0
    guided_built_for_h: int = 0

    # Frame info (updated every frame)
    cam8_w: int = 0
    cam8_h: int = 0

    # Letterbox display params (updated by _letterbox_cam8)
    cam8_disp_scale: float = 1.0
    cam8_disp_off_x: int = 0
    cam8_disp_off_y: int = 0
    cam8_disp_scaled_w: int = 0
    cam8_disp_scaled_h: int = 0


_state = _CalibState()
_state.limit_corners = []
_state.points = []
_state.guided_targets = []


def _update_home_cam8_position() -> None:
    """Compute home mark position in cam8 pixel space from current frame size."""
    if _state.cam8_w > 0 and _state.cam8_h > 0:
        _state.home_cam8_px = _state.cam8_w / 2.0
        _state.home_cam8_py = _state.cam8_h * HOME_REFERENCE_Y_RATIO


def _get_screen_size() -> Tuple[int, int]:
    """Primary monitor size in pixels (same approach as file 11)."""
    try:
        from tkinter import Tk
        r = Tk()
        r.withdraw()
        sw = r.winfo_screenwidth()
        sh = r.winfo_screenheight()
        r.destroy()
        return max(320, int(sw)), max(240, int(sh))
    except Exception:
        return DISPLAY_FALLBACK_W, DISPLAY_FALLBACK_H


def _build_top_bottom_layouts(
    screen_w: int, screen_h: int, top_ratio: float = TOP_PANEL_HEIGHT_RATIO
) -> Dict[str, Dict[str, int]]:
    top_h = max(1, int(screen_h * float(top_ratio)))
    bottom_h = max(1, screen_h - top_h)
    return {
        "top": {"x": 0, "y": 0, "width": screen_w, "height": top_h},
        "bottom": {"x": 0, "y": top_h, "width": screen_w, "height": bottom_h},
    }


def _calib_build_layouts() -> Dict[str, Dict[str, int]]:
    """Screen size optionally capped by CALIB_DISPLAY_MAX_HEIGHT (keeps aspect)."""
    sw, sh = _get_screen_size()
    if CALIB_DISPLAY_MAX_HEIGHT > 0 and sh > CALIB_DISPLAY_MAX_HEIGHT:
        scale = CALIB_DISPLAY_MAX_HEIGHT / float(sh)
        sw = max(320, int(round(sw * scale)))
        sh = max(240, int(round(sh * scale)))
    return _build_top_bottom_layouts(sw, sh, TOP_PANEL_HEIGHT_RATIO)


def _configure_calib_window(win_name: str, canvas_w: int, canvas_h: int) -> None:
    """Order matches 11 _run_multi_mode_single_window (fullscreen + resize)."""
    if CALIB_USE_FULLSCREEN:
        try:
            cv2.setWindowProperty(
                win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        except cv2.error:
            pass
    cv2.resizeWindow(win_name, canvas_w, canvas_h)


def _fit_frame_to_layout(
    frame: Optional[np.ndarray], layout_w: int, layout_h: int
) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """
    Letterbox/pillarbox frame into layout_w x layout_h (same as 11 _fit_frame_to_layout).
    Returns (canvas_bgr, mapping) for cam8 click mapping; mapping None if no valid frame.
    """
    canvas = np.zeros((layout_h, layout_w, 3), dtype=np.uint8)
    if frame is None:
        return canvas, None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return canvas, None
    scale = min(layout_w / src_w, layout_h / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    offset_x = max(0, (layout_w - target_w) // 2)
    offset_y = max(0, (layout_h - target_h) // 2)
    if target_w == src_w and target_h == src_h:
        resized = frame
    else:
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas[offset_y : offset_y + target_h, offset_x : offset_x + target_w] = resized
    mapping = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "display_w": target_w,
        "display_h": target_h,
        "src_w": src_w,
        "src_h": src_h,
    }
    return canvas, mapping


# =================================================================
# Letterbox helpers (cam8 → _state for mouse mapping)
# =================================================================

def _letterbox_cam8(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Fit cam8 into panel; updates _state.cam8_disp_* for _disp_to_cam8."""
    disp8, mapping = _fit_frame_to_layout(frame, target_w, target_h)
    if mapping is not None:
        _state.cam8_disp_scale = mapping["display_w"] / float(mapping["src_w"])
        _state.cam8_disp_off_x = mapping["offset_x"]
        _state.cam8_disp_off_y = mapping["offset_y"]
        _state.cam8_disp_scaled_w = mapping["display_w"]
        _state.cam8_disp_scaled_h = mapping["display_h"]
    else:
        _state.cam8_disp_scale = 1.0
        _state.cam8_disp_off_x = 0
        _state.cam8_disp_off_y = 0
        _state.cam8_disp_scaled_w = 0
        _state.cam8_disp_scaled_h = 0
    return disp8


def _cam8_to_disp(orig_x: float, orig_y: float) -> Tuple[int, int]:
    """Convert cam8 pixel coords to display (letterboxed) coords."""
    s = _state.cam8_disp_scale
    return (
        int(orig_x * s) + _state.cam8_disp_off_x,
        int(orig_y * s) + _state.cam8_disp_off_y,
    )


def _disp_to_cam8(disp_x: int, disp_y: int) -> Optional[Tuple[float, float]]:
    """Convert display (letterboxed) coords to cam8 pixel coords.
    Returns None if click is outside the actual image area."""
    off_x = _state.cam8_disp_off_x
    off_y = _state.cam8_disp_off_y
    nw = _state.cam8_disp_scaled_w
    nh = _state.cam8_disp_scaled_h
    if nw <= 0 or nh <= 0:
        return None
    if disp_x < off_x or disp_x >= off_x + nw or disp_y < off_y or disp_y >= off_y + nh:
        return None
    s = _state.cam8_disp_scale
    if s <= 0:
        return None
    return (disp_x - off_x) / s, (disp_y - off_y) / s


# =================================================================
# Mouse callback
# =================================================================

def _on_mouse(event, x, y, flags, param):
    """CAM8 top strip: limits + calibration — left-click sets pending pixel; C confirms."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    top_h = param.get("cam8_top_h")
    if top_h is None:
        top_h = _calib_build_layouts()["top"]["height"]
    if y >= top_h:
        return
    if _state.cam8_w <= 0 or _state.cam8_h <= 0:
        return
    if _state.phase == "home":
        return
    if _guided_grid_has_pending():
        return
    result = _disp_to_cam8(x, y)
    if result is None:
        return
    orig_x, orig_y = result
    _state.cam8_click = (orig_x, orig_y)
    print(f"[CAL] click cam8: ({orig_x:.1f}, {orig_y:.1f})")


# =================================================================
# Grid + Polygon helpers
# =================================================================

def _draw_grid_overlay(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    """Draw N×M grid with reachable/unreachable tint (one blend pass — was 50× copy/blend)."""
    off_x = _state.cam8_disp_off_x
    off_y = _state.cam8_disp_off_y
    nw = _state.cam8_disp_scaled_w
    nh = _state.cam8_disp_scaled_h
    if nw <= 0 or nh <= 0:
        return
    cell_w = nw / GRID_COLS
    cell_h = nh / GRID_ROWS
    cw8 = _state.cam8_w if _state.cam8_w > 0 else 1
    ch8 = _state.cam8_h if _state.cam8_h > 0 else 1
    overlay = disp8.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx_cam8 = (c + 0.5) * cw8 / GRID_COLS
            cy_cam8 = (r + 0.5) * ch8 / GRID_ROWS
            x0 = off_x + int(c * cell_w)
            y0 = off_y + int(r * cell_h)
            x1 = off_x + int((c + 1) * cell_w)
            y1 = off_y + int((r + 1) * cell_h)
            inside = _is_in_reachable(cx_cam8, cy_cam8)
            fill = (0, 50, 0) if inside else (0, 0, 55)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)
    cv2.addWeighted(overlay, 0.35, disp8, 0.65, 0, disp8)
    for r in range(1, GRID_ROWS):
        cv2.line(disp8, (off_x, off_y + int(r * cell_h)),
                 (off_x + nw, off_y + int(r * cell_h)), (65, 65, 65), 1)
    for c in range(1, GRID_COLS):
        cv2.line(disp8, (off_x + int(c * cell_w), off_y),
                 (off_x + int(c * cell_w), off_y + nh), (65, 65, 65), 1)


def _draw_home_mark(disp8: np.ndarray, half_w: int, disp_h: int,
                    confirmed: bool = False) -> None:
    """Draw reference mark (+) at center-x, 90% down."""
    if _state.cam8_w <= 0 or _state.cam8_h <= 0:
        return
    hx, hy = _cam8_to_disp(_state.home_cam8_px, _state.home_cam8_py)
    col = (0, 255, 180) if confirmed else (0, 200, 255)
    thick = 2
    r = 14
    cv2.circle(disp8, (hx, hy), r, col, thick)
    cv2.line(disp8, (hx - r - 6, hy), (hx + r + 6, hy), col, thick)
    cv2.line(disp8, (hx, hy - r - 6), (hx, hy + r + 6), col, thick)
    label = "REF OK" if confirmed else "REF +"
    cv2.putText(disp8, label, (hx + r + 4, hy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


def _draw_reachable_polygon(disp8: np.ndarray) -> None:
    if _state.reachable_poly_cam8 is None:
        return
    poly_cam8 = _state.reachable_poly_cam8.reshape(-1, 2)
    pts_disp = np.array([_cam8_to_disp(p[0], p[1]) for p in poly_cam8], dtype=np.int32)
    cv2.polylines(disp8, [pts_disp.reshape(-1, 1, 2)], True, (0, 255, 80), 2)
    for i, pt in enumerate(_state.limit_corners):
        if _state.cam8_w <= 0 or _state.cam8_h <= 0:
            continue
        dx, dy = _cam8_to_disp(pt["cam8_px"], pt["cam8_py"])
        cv2.circle(disp8, (dx, dy), 7, (0, 220, 0), -1)
        tag = str(pt.get("label", f"P{i+1}"))
        cv2.putText(disp8, tag, (dx + 8, dy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)


def _build_reachable_polygon() -> None:
    if len(_state.limit_corners) < 3:
        _state.reachable_poly_cam8 = None
        return
    cpts = [[c["cam8_px"], c["cam8_py"]] for c in _state.limit_corners]
    _state.reachable_poly_cam8 = np.array(cpts, dtype=np.float32).reshape(-1, 1, 2)


def _is_in_reachable(cam8_px: float, cam8_py: float) -> bool:
    if _state.reachable_poly_cam8 is None:
        return True
    return cv2.pointPolygonTest(
        _state.reachable_poly_cam8, (float(cam8_px), float(cam8_py)), False
    ) >= 0


def _build_guided_grid_targets(cam8_w: int, cam8_h: int) -> List[Dict[str, Any]]:
    """Cell centers (cam8 px) whose center lies inside reachable polygon; row-major order."""
    if cam8_w <= 0 or cam8_h <= 0:
        return []
    cell_w = cam8_w / max(GRID_COLS, 1)
    cell_h = cam8_h / max(GRID_ROWS, 1)
    out: List[Dict[str, Any]] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            if _is_in_reachable(cx, cy):
                out.append(
                    {
                        "cam8_px": float(cx),
                        "cam8_py": float(cy),
                        "grid_r": int(r),
                        "grid_c": int(c),
                    }
                )
    return out


def _ensure_guided_targets_for_resolution() -> None:
    """Rebuild guided cell list when cam8 resolution changes (Phase 2 + guided mode)."""
    if not _state.calib_use_guided_grid or _state.phase != "calibration":
        return
    w, h = _state.cam8_w, _state.cam8_h
    if w <= 0 or h <= 0:
        return
    if w == _state.guided_built_for_w and h == _state.guided_built_for_h and _state.guided_targets:
        return
    prev_w, prev_h = _state.guided_built_for_w, _state.guided_built_for_h
    _state.guided_targets = _build_guided_grid_targets(w, h)
    _state.guided_built_for_w = w
    _state.guided_built_for_h = h
    if prev_w <= 0:
        _state.guided_idx = 0
    elif prev_w != w or prev_h != h:
        print("[CAL] WARNING: CAM8 resolution changed — guided progress reset to 0")
        _state.guided_idx = 0
    n = len(_state.guided_targets)
    print(f"[CAL] Guided grid: {n} cell centers in reachable zone (GRID {GRID_COLS}x{GRID_ROWS})")


def _guided_grid_has_pending() -> bool:
    return (
        _state.calib_use_guided_grid
        and _state.phase == "calibration"
        and _state.guided_targets
        and _state.guided_idx < len(_state.guided_targets)
    )


def _reset_guided_grid_for_new_cal_phase() -> None:
    """Clear built targets so Phase 2 rebuilds list on first valid cam8 frame."""
    _state.guided_targets = []
    _state.guided_built_for_w = 0
    _state.guided_built_for_h = 0
    _state.guided_idx = 0


# =================================================================
# Back-projection for validation
# =================================================================

def _backproject_arm_limits(calib_data: Dict) -> Optional[List]:
    if _a2p is None or calib_data is None or _config_mod is None:
        return None
    xl = getattr(_config_mod, "CAM4_ARM_X_LIMITS", (-65.0, 65.0))
    yl = getattr(_config_mod, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0))
    pts = []
    for pan, tilt in [(xl[0], yl[0]), (xl[1], yl[0]), (xl[1], yl[1]), (xl[0], yl[1])]:
        px = _a2p(pan, tilt, calib_data)
        if px is not None:
            pts.append(list(px))
    return pts if len(pts) >= 3 else None


# =================================================================
# Homography + save
# =================================================================

def _fit_homography(points: List[Dict]) -> Optional[List[List[float]]]:
    if len(points) < 4:
        return None
    src = np.array([[p["cam8_px"], p["cam8_py"]] for p in points], dtype=np.float64)
    dst = np.array([[p["arm_pan"], p["arm_tilt"]] for p in points], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H.tolist() if H is not None else None


def _compute_errors(points: List[Dict], H: List[List[float]]) -> List[float]:
    Hnp = np.array(H, dtype=np.float64)
    errors = []
    for p in points:
        v = Hnp @ np.array([p["cam8_px"], p["cam8_py"], 1.0])
        if abs(v[2]) < 1e-9:
            errors.append(float("inf"))
            continue
        errors.append(math.hypot(v[0] / v[2] - p["arm_pan"],
                                  v[1] / v[2] - p["arm_tilt"]))
    return errors


def _idw_arm_at(points: List[Dict], px: float, py: float, k: int = 10, power: float = 2.0) -> Tuple[float, float]:
    """
    Inverse-distance weighted interpolation from nearby calibration points.
    Used to build coarse/fine grid cells for hybrid mapping.
    """
    if not points:
        return 0.0, 0.0
    d_items = []
    for p in points:
        dx = float(px) - float(p["cam8_px"])
        dy = float(py) - float(p["cam8_py"])
        d2 = dx * dx + dy * dy
        if d2 <= 1e-9:
            return float(p["arm_pan"]), float(p["arm_tilt"])
        d_items.append((d2, float(p["arm_pan"]), float(p["arm_tilt"])))
    d_items.sort(key=lambda t: t[0])
    near = d_items[: max(1, min(k, len(d_items)))]
    w_sum = 0.0
    pan_sum = 0.0
    tilt_sum = 0.0
    for d2, pan, tilt in near:
        d = math.sqrt(d2)
        w = 1.0 / max(d, 1e-6) ** power
        w_sum += w
        pan_sum += pan * w
        tilt_sum += tilt * w
    if w_sum <= 1e-9:
        return near[0][1], near[0][2]
    return pan_sum / w_sum, tilt_sum / w_sum


def _build_hybrid_grid_cells(
    points: List[Dict],
    cam8_w: int,
    cam8_h: int,
    rows: int,
    cols: int,
) -> Dict[str, List[float]]:
    """Build coarse grid cells via IDW interpolation at cell centers."""
    cells: Dict[str, List[float]] = {}
    cell_w = cam8_w / max(cols, 1)
    cell_h = cam8_h / max(rows, 1)
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            pan, tilt = _idw_arm_at(points, cx, cy, k=10, power=2.0)
            cells[f"{r}_{c}"] = [float(pan), float(tilt)]
    return cells


def _build_center_fine_grid(
    points: List[Dict],
    cam8_w: int,
    cam8_h: int,
    coarse_rows: int,
    coarse_cols: int,
    fine_rows: int,
    fine_cols: int,
) -> Dict[str, Any]:
    """Build fine grid for the center coarse cell to improve central aiming precision."""
    center_row = coarse_rows // 2
    center_col = coarse_cols // 2
    coarse_cell_w = cam8_w / max(coarse_cols, 1)
    coarse_cell_h = cam8_h / max(coarse_rows, 1)
    left = center_col * coarse_cell_w
    top = center_row * coarse_cell_h
    fine_cells: Dict[str, List[float]] = {}
    for r in range(fine_rows):
        for c in range(fine_cols):
            fx = left + (c + 0.5) * (coarse_cell_w / max(fine_cols, 1))
            fy = top + (r + 0.5) * (coarse_cell_h / max(fine_rows, 1))
            pan, tilt = _idw_arm_at(points, fx, fy, k=10, power=2.0)
            fine_cells[f"{r}_{c}"] = [float(pan), float(tilt)]
    return {
        "center_row": center_row,
        "center_col": center_col,
        "fine_rows": fine_rows,
        "fine_cols": fine_cols,
        "fine_cells": fine_cells,
    }


def _build_data_dict(points, cam8_w, cam8_h, limit_corners,
                     home_pan, home_tilt) -> Dict[str, Any]:
    H = _fit_homography(points)
    cx8, cy8 = cam8_w / 2.0, cam8_h / 2.0
    coarse_rows = HYBRID_COARSE_GRID_ROWS
    coarse_cols = HYBRID_COARSE_GRID_COLS
    coarse_cells = _build_hybrid_grid_cells(points, cam8_w, cam8_h, coarse_rows, coarse_cols)
    center_fine = _build_center_fine_grid(
        points,
        cam8_w,
        cam8_h,
        coarse_rows=coarse_rows,
        coarse_cols=coarse_cols,
        fine_rows=HYBRID_FINE_GRID_ROWS,
        fine_cols=HYBRID_FINE_GRID_COLS,
    )
    data: Dict[str, Any] = {
        "source_camera": "cam8",
        "output_width": cam8_w,
        "output_height": cam8_h,
        "grid_rows": coarse_rows,
        "grid_cols": coarse_cols,
        "cells": coarse_cells,
        "center_fine_grid": center_fine,
        "crosshair": {"x": cx8, "y": cy8},
        "calibration_points": points,
        "num_points": len(points),
        "arm_limit_corners": limit_corners,
        "home_reference": {
            "cam8_px": cam8_w / 2.0,
            "cam8_py": cam8_h * HOME_REFERENCE_Y_RATIO,
            "arm_pan": home_pan,
            "arm_tilt": home_tilt,
            "y_ratio": HOME_REFERENCE_Y_RATIO,
            "description": f"center-x, {int(HOME_REFERENCE_Y_RATIO*100)}% from top",
        },
        # Preferred naming: keep both keys for backward compatibility.
        "camera_reference": {
            "cam8_px": cam8_w / 2.0,
            "cam8_py": cam8_h * HOME_REFERENCE_Y_RATIO,
            "arm_pan": home_pan,
            "arm_tilt": home_tilt,
            "y_ratio": HOME_REFERENCE_Y_RATIO,
            "description": f"center-x, {int(HOME_REFERENCE_Y_RATIO*100)}% from top",
        },
    }
    if H is not None:
        data["homography"] = H
        errors = _compute_errors(points, H)
        data["mean_error_deg"] = float(np.mean(errors))
        data["max_error_deg"] = float(np.max(errors))
        data["rms_error_deg"] = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        print(f"[CAL] Homography fit: mean={data['mean_error_deg']:.3f}° "
              f"max={data['max_error_deg']:.3f}° rms={data['rms_error_deg']:.3f}° "
              f"({len(points)} pts)")
        bp = _backproject_arm_limits(data)
        if bp:
            data["arm_limit_backprojected_cam8"] = bp
    else:
        print(f"[CAL] WARNING: need ≥4 points for homography (have {len(points)}).")
    if _state.reachable_poly_cam8 is not None:
        data["reachable_polygon_cam8"] = \
            _state.reachable_poly_cam8.reshape(-1, 2).tolist()
    data["mapping_mode"] = "hybrid_homography_coarse_fine"
    return data


def save_calibration(
    points: List[Dict],
    cam8_w: int,
    cam8_h: int,
    limit_corners: List[Dict],
    home_pan: float,
    home_tilt: float,
    output_path: Path = OUTPUT_JSON,
) -> bool:
    if not points:
        print("[CAL] No calibration points to save.")
        return False
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    data = _build_data_dict(points, cam8_w, cam8_h, limit_corners, home_pan, home_tilt)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[CAL] Saved → {output_path}")
    return True


# =================================================================
# Confirm helpers
# =================================================================

def _confirm_home(arm) -> None:
    """Phase home: record current arm position as home datum."""
    _update_home_cam8_position()
    pan = getattr(arm, "pos_x", 0.0)
    tilt = getattr(arm, "pos_y", 0.0)
    _state.home_arm_pan = pan
    _state.home_arm_tilt = tilt
    _state.home_confirmed = True
    print(f"[CAL] REFERENCE confirmed: arm=({pan:.3f}, {tilt:.3f})  "
          f"cam8=({_state.home_cam8_px:.0f}, {_state.home_cam8_py:.0f})")
    print("[CAL] Press SPACE/Enter -> Phase 1: Limit Sweep")


def _confirm_limit(arm) -> None:
    """Phase limits: append one vertex (click CAM8 + C). Polygon follows vertex order."""
    if _state.cam8_click is None:
        print("[CAL] Click CAM8 first, then C to add limit vertex")
        return
    pan = getattr(arm, "pos_x", 0.0)
    tilt = getattr(arm, "pos_y", 0.0)
    n = len(_state.limit_corners) + 1
    label = f"p{n}"
    c = {
        "cam8_px": float(_state.cam8_click[0]),
        "cam8_py": float(_state.cam8_click[1]),
        "arm_pan": float(pan),
        "arm_tilt": float(tilt),
        "label": label,
    }
    _state.limit_corners.append(c)
    _state.cam8_click = None
    _build_reachable_polygon()
    print(
        f"[CAL] Limit {label}: cam8=({c['cam8_px']:.0f},{c['cam8_py']:.0f})  "
        f"arm=({c['arm_pan']:.2f},{c['arm_tilt']:.2f})  ({n} pts)  SPACE → Phase 2 when done"
    )


def _confirm_calibration(arm) -> None:
    """Phase calibration: record a cam8->arm pair (guided red-dot or free click)."""
    if _state.calib_use_guided_grid:
        _ensure_guided_targets_for_resolution()
        if not _state.guided_targets:
            print(
                "[CAL] Guided grid has 0 cells (check limits / CAM8). "
                "Use click mode or fix polygon."
            )
            return
        if _state.guided_idx < len(_state.guided_targets):
            tgt = _state.guided_targets[_state.guided_idx]
            cx = float(tgt["cam8_px"])
            cy = float(tgt["cam8_py"])
            pan = getattr(arm, "pos_x", 0.0)
            tilt = getattr(arm, "pos_y", 0.0)
            pt = {
                "cam8_px": cx,
                "cam8_py": cy,
                "arm_pan": float(pan),
                "arm_tilt": float(tilt),
                "grid_r": int(tgt["grid_r"]),
                "grid_c": int(tgt["grid_c"]),
                "guided_cell": True,
            }
            _state.points.append(pt)
            _state.guided_idx += 1
            n = len(_state.guided_targets)
            print(
                f"[CAL] Guided cell {_state.guided_idx}/{n} (r{pt['grid_r']} c{pt['grid_c']}): "
                f"cam8=({cx:.0f},{cy:.0f})  arm=({pan:.3f},{tilt:.3f})"
            )
            if _state.guided_idx >= n:
                print(
                    "[CAL] All guided cells done — optional: click cam8 for extra points, then S"
                )
            return

    if _state.cam8_click is None:
        print("[CAL] Click on cam8 first, then press C")
        return
    cx, cy = _state.cam8_click
    if not _is_in_reachable(cx, cy):
        print(f"[CAL] WARNING: ({cx:.0f},{cy:.0f}) is outside reachable zone")
    pan = getattr(arm, "pos_x", 0.0)
    tilt = getattr(arm, "pos_y", 0.0)
    pt = {"cam8_px": float(cx), "cam8_py": float(cy),
          "arm_pan": float(pan), "arm_tilt": float(tilt)}
    _state.cam8_click = None
    _state.points.append(pt)
    print(f"[CAL] Pt {len(_state.points)}: cam8=({cx:.0f},{cy:.0f})  "
          f"arm=({pan:.3f},{tilt:.3f})")


def _do_confirm(arm) -> None:
    if _state.phase == "home":
        _confirm_home(arm)
    elif _state.phase == "limits":
        _confirm_limit(arm)
    else:
        _confirm_calibration(arm)


def _get_ref_pan_tilt_from_file(json_path: Path = OUTPUT_JSON) -> Optional[Tuple[float, float]]:
    """Load reference pan/tilt from saved calibration JSON (new or legacy key)."""
    try:
        if not json_path.is_file():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ref = data.get("camera_reference") or data.get("home_reference")
        if not isinstance(ref, dict):
            return None
        return float(ref["arm_pan"]), float(ref["arm_tilt"])
    except Exception:
        return None


def _goto_reference(arm, json_path: Path = OUTPUT_JSON) -> bool:
    """Move arm to reference. Priority: current confirmed state > saved file."""
    ref_pan_tilt: Optional[Tuple[float, float]] = None
    if _state.home_confirmed:
        ref_pan_tilt = (_state.home_arm_pan, _state.home_arm_tilt)
    else:
        ref_pan_tilt = _get_ref_pan_tilt_from_file(json_path)
    if ref_pan_tilt is None:
        print("[CAL] No saved reference available yet. Confirm reference with C first.")
        return False
    pan_ref, tilt_ref = ref_pan_tilt
    try:
        arm.move_absolute(float(pan_ref), float(tilt_ref), blocking=False)
        _state.home_arm_pan = float(pan_ref)
        _state.home_arm_tilt = float(tilt_ref)
        _state.home_confirmed = True
        print(f"[CAL] Go to REF -> pan={pan_ref:.3f} tilt={tilt_ref:.3f}")
        return True
    except Exception as e:
        print(f"[CAL] Go to REF failed: {e}")
        return False


# =================================================================
# Draw helpers
# =================================================================

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

# All instructional HUD lives in the bottom (CAM4) panel, one size, bottom-aligned stack.
BOTTOM_HUD_SCALE = 0.52
BOTTOM_HUD_THICK = 2
BOTTOM_HUD_GAP = 5
BOTTOM_HUD_MARGIN_X = 12
BOTTOM_HUD_MARGIN_BOTTOM = 8
BOTTOM_HUD_TEXT_BGR = (220, 220, 220)
BOTTOM_HUD_WARN_BGR = (80, 100, 255)
BOTTOM_HUD_OK_BGR = (100, 230, 120)


def _draw_hud_dark_panel(
    img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.52
) -> None:
    """Semi-opaque bar behind text (BGR)."""
    if w <= 1 or h <= 1:
        return
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    patch = np.full_like(roi, (24, 24, 24))
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, roi)


def _blank(h: int, w: int, msg: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    return img


def _draw_confirmed_points(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    if _state.cam8_w <= 0:
        return
    for i, pt in enumerate(_state.points):
        dx, dy = _cam8_to_disp(pt["cam8_px"], pt["cam8_py"])
        col = (0, 255, 0) if _is_in_reachable(pt["cam8_px"], pt["cam8_py"]) else (0, 100, 255)
        cv2.circle(disp8, (dx, dy), 8, col, 2)
        cv2.putText(disp8, str(i + 1), (dx + 10, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def _draw_pending_click(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    if _state.cam8_click is None or _state.cam8_w <= 0:
        return
    px, py = _cam8_to_disp(_state.cam8_click[0], _state.cam8_click[1])
    if _state.phase == "limits":
        col = (0, 180, 255)
        cv2.circle(disp8, (px, py), 10, col, 2)
        cv2.drawMarker(disp8, (px, py), col, cv2.MARKER_CROSS, 20, 2)
        return
    in_r = _is_in_reachable(_state.cam8_click[0], _state.cam8_click[1])
    col = (0, 180, 255) if in_r else (0, 0, 255)
    cv2.circle(disp8, (px, py), 10, col, 2)
    cv2.drawMarker(disp8, (px, py), col, cv2.MARKER_CROSS, 20, 2)
    if not in_r:
        cv2.putText(disp8, "OUT OF RANGE", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)


def _draw_guided_grid_target(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    """Large red marker at current guided cell center (CAM8 panel)."""
    if not _guided_grid_has_pending():
        return
    tgt = _state.guided_targets[_state.guided_idx]
    dx, dy = _cam8_to_disp(tgt["cam8_px"], tgt["cam8_py"])
    cv2.circle(disp8, (dx, dy), 16, (0, 0, 255), 3)
    cv2.circle(disp8, (dx, dy), 5, (0, 0, 255), -1)
    cv2.drawMarker(disp8, (dx, dy), (60, 60, 255), cv2.MARKER_CROSS, 32, 2)


def _collect_full_calib_bottom_hud_lines(
    pan: float,
    tilt: float,
    saved: bool,
    now: float,
    warn_msg: Optional[str],
    warn_until: float,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """(text, BGR) lines top-to-bottom; keys last so they sit at bottom when drawn."""
    t = BOTTOM_HUD_TEXT_BGR
    out: List[Tuple[str, Tuple[int, int, int]]] = []
    if warn_msg and now < warn_until:
        out.append((warn_msg, BOTTOM_HUD_WARN_BGR))
    phase = _state.phase
    if phase == "home":
        out.append(("PHASE 0 — REFERENCE", t))
        if _state.home_confirmed:
            out.append(("Reference saved — press SPACE for Phase 1", t))
        else:
            out.append(("Place target on yellow REF mark (top / CAM8)", t))
            out.append(("Center CAM4 crosshair on target, press C", t))
    elif phase == "limits":
        out.append(("PHASE 1 — LIMIT OUTLINE (click CAM8 + C, order = polygon)", t))
        out.append(("Click each corner on CAM8, press C to record (≥3); SPACE → Phase 2", t))
        nl = len(_state.limit_corners)
        out.append((f"Vertices: {nl}   Arm  pan {pan:+.1f}   tilt {tilt:+.1f}", t))
    else:
        ug = _state.calib_use_guided_grid and bool(_state.guided_targets)
        n_tgt = len(_state.guided_targets) if ug else 0
        gi = _state.guided_idx
        if ug and gi < n_tgt:
            out.append(("PHASE 2 — GUIDED GRID (aim CAM4 at RED dot)", t))
            out.append((f"Progress {gi + 1}/{n_tgt}  cell r{_state.guided_targets[gi]['grid_r']} c{_state.guided_targets[gi]['grid_c']}", t))
            out.append(("C = confirm   SPACE = skip cell   (no click needed)", t))
        elif ug and n_tgt > 0 and gi >= n_tgt:
            out.append(("PHASE 2 — guided cells done", t))
            out.append(("Optional: click CAM8 for extra points, C to add, S save", t))
            out.append((f"Calibration points: {len(_state.points)}", t))
        else:
            out.append(("PHASE 2 — CALIBRATION POINTS", t))
            out.append(("Click CAM8 (top) in green zone, aim CAM4, press C", t))
            if _state.reachable_poly_cam8 is not None:
                out.append(("Green = reachable   Orange click = outside zone", t))
            else:
                out.append(("No limit polygon — full frame treated reachable", t))
            out.append((f"Calibration points: {len(_state.points)}", t))

    out.append(("", t))
    out.append(("— Lower panel (CAM4) — aim at yellow crosshair —", t))
    out.append((f"Arm  pan {pan:+.1f}   tilt {tilt:+.1f}", t))
    out.append(("", t))

    if phase == "home":
        out.append(("C = confirm reference   SPACE = next   Q = quit", t))
        out.append(("H = move arm to saved reference", t))
    elif phase == "limits":
        out.append(("C / B0 = add vertex   SPACE = Phase 2   Q = quit", t))
        out.append(("D = delete last vertex   L = clear all limit points", t))
    else:
        if _state.calib_use_guided_grid and _state.guided_targets and _state.guided_idx < len(_state.guided_targets):
            out.append(("C = confirm cell   S = save JSON   Q = quit", t))
            out.append(("SPACE = skip   D = undo skip/point   R = reset pts   H / L = ref / limits", t))
        else:
            out.append(("C = save point   S = save JSON   Q = quit", t))
            out.append(("D = delete point   R = reset pts   H / L = ref / limits", t))
    if saved:
        out.append(("[ CALIBRATION SAVED ]", BOTTOM_HUD_OK_BGR))
    return out


def _collect_recal_bottom_hud_lines(
    pan: float,
    tilt: float,
    saved: bool,
    recal_done: bool,
    delta_pan: float,
    delta_tilt: float,
    n_pts: int,
    n_refine: int,
    test_mode: bool,
    stored_home_pan: float,
    stored_home_tilt: float,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    t = BOTTOM_HUD_TEXT_BGR
    out: List[Tuple[str, Tuple[int, int, int]]] = [
        ("RECAL — REFERENCE UPDATE (CAM8 top, REF mark)", t),
    ]
    if recal_done:
        out.append((f"Delta  pan {delta_pan:+.2f}   tilt {delta_tilt:+.2f}", t))
        out.append((f"{n_pts} pts updated   Refine {n_refine}   S = save", t))
    else:
        out.append(("Click CAM8 to refine, or C with no click for REF shift", t))
        out.append(
            (
                f"Stored ref  pan {stored_home_pan:+.1f}   tilt {stored_home_tilt:+.1f}",
                t,
            )
        )
    out.append((f"Live arm  pan {pan:+.1f}   tilt {tilt:+.1f}", t))
    out.append((f"TEST MODE: {'ON' if test_mode else 'OFF'}  (T)", t))
    out.append(("", t))
    out.append(("— CAM4 — aim REF at crosshair, then C —", t))
    out.append(("", t))
    out.append(("T = test   Click CAM8   C = confirm   S = save   Q = quit", t))
    out.append(("H = home ref   D = delete refine   R = reset refine", t))
    if saved:
        out.append(("[ SAVED ]", BOTTOM_HUD_OK_BGR))
    return out


def _draw_bottom_panel_unified_hud(
    display: np.ndarray,
    disp_w: int,
    canvas_h: int,
    top_h: int,
    lines: List[Tuple[str, Tuple[int, int, int]]],
) -> None:
    """Single font scale; stack from bottom of canvas within lower panel only; no overlap."""
    if not lines:
        return
    panel_h = canvas_h - top_h
    if panel_h < 32:
        return

    scale = float(BOTTOM_HUD_SCALE)
    thick = int(BOTTOM_HUD_THICK)
    gap = int(BOTTOM_HUD_GAP)

    def measure_height(sc: float) -> int:
        h = 0
        for text, _ in lines:
            if not text.strip():
                h += max(2, gap // 2)
                continue
            (_, th), bl = cv2.getTextSize(text, HUD_FONT, sc, thick)
            h += th + bl + gap
        return h - gap if h > 0 else 0

    total_h = measure_height(scale)
    # Keep text in the lower ~half of the bottom panel so crosshair stays visible.
    max_strip = max(24, int(panel_h * 0.48))
    while total_h > max_strip and scale > 0.34:
        scale *= 0.9
        total_h = measure_height(scale)

    margin_b = BOTTOM_HUD_MARGIN_BOTTOM
    pad_top = 6
    strip_y = canvas_h - margin_b - total_h - pad_top
    min_y = top_h + 2
    if strip_y < min_y:
        strip_y = min_y
    _draw_hud_dark_panel(display, 0, strip_y, disp_w, canvas_h - strip_y, 0.48)

    y = canvas_h - margin_b
    mx = BOTTOM_HUD_MARGIN_X
    for text, bgr in reversed(lines):
        if not text.strip():
            y -= max(2, gap // 2)
            continue
        (_, th), bl = cv2.getTextSize(text, HUD_FONT, scale, thick)
        cv2.putText(
            display, text, (mx, y), HUD_FONT, scale, bgr, thick, cv2.LINE_AA
        )
        y -= th + bl + gap


def _draw_cam4_crosshair(
    disp4: np.ndarray,
    panel_w: int,
    panel_h: int,
    mapping: Optional[Dict[str, int]] = None,
) -> None:
    """Crosshair at optical center of letterboxed image (or panel center if no mapping)."""
    if mapping is not None:
        cx = mapping["offset_x"] + mapping["display_w"] // 2
        cy = mapping["offset_y"] + mapping["display_h"] // 2
    else:
        cx, cy = panel_w // 2, panel_h // 2
    cv2.line(disp4, (cx - 35, cy), (cx + 35, cy), (0, 255, 255), 2)
    cv2.line(disp4, (cx, cy - 35), (cx, cy + 35), (0, 255, 255), 2)
    cv2.circle(disp4, (cx, cy), 45, (0, 255, 255), 1)


# =================================================================
# Main loop (shared for normal + recal)
# =================================================================

def _run_loop(arm, cam8, cam4, joystick, joy_mapper, mode: str = "full") -> None:
    """
    mode: "full"  = 3-phase calibration
          "recal" = fast recalibration using reference point only
    """
    win_name = "Cam8 Arm Calibrator" if mode == "full" else "Cam8 Arm Recalibrator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    _layouts = _calib_build_layouts()
    canvas_w = _layouts["top"]["width"]
    top_h = _layouts["top"]["height"]
    bottom_h = _layouts["bottom"]["height"]
    canvas_h = top_h + bottom_h
    _configure_calib_window(win_name, canvas_w, canvas_h)
    cv2.setMouseCallback(
        win_name, _on_mouse, param={"cam8_top_h": top_h}
    )

    last_sync = 0.0
    last_loop = time.time()
    prev_confirm = False
    saved = False
    warn_msg: Optional[str] = None
    warn_until: float = 0.0

    while True:
        now = time.time()
        dt = now - last_loop
        last_loop = now

        ok8, frame8, _ = cam8.read()
        ok4, frame4, _ = cam4.read()

        if ok8 and frame8 is not None:
            _state.cam8_h, _state.cam8_w = frame8.shape[:2]
            _update_home_cam8_position()
        else:
            frame8 = _blank(top_h, canvas_w, "cam8 no signal")
        if not (ok4 and frame4 is not None):
            frame4 = _blank(bottom_h, canvas_w, "cam4 no signal")

        if now - last_sync >= SYNC_INTERVAL_SEC:
            try:
                arm.sync_position_from_grbl()
            except Exception:
                pass
            last_sync = now

        confirm_pressed = False
        if joystick is not None and joy_mapper is not None:
            js = joystick.read()
            confirm_pressed = (
                bool(js.buttons[JOY_CONFIRM_BUTTON])
                if hasattr(js, "buttons") and len(js.buttons) > JOY_CONFIRM_BUTTON
                else False
            )
            joy_mapper.apply(js, dt)

        pan = getattr(arm, "pos_x", 0.0)
        tilt = getattr(arm, "pos_y", 0.0)

        # --- Draw cam8 (letterboxed — same as 11 _fit_frame_to_layout) ---
        disp8 = _letterbox_cam8(frame8, canvas_w, top_h)
        if _state.phase == "calibration" and _state.calib_use_guided_grid:
            _ensure_guided_targets_for_resolution()
        _draw_grid_overlay(disp8, canvas_w, top_h)
        _draw_reachable_polygon(disp8)
        _draw_home_mark(disp8, canvas_w, top_h, _state.home_confirmed)
        _draw_confirmed_points(disp8, canvas_w, top_h)
        _draw_pending_click(disp8, canvas_w, top_h)
        _draw_guided_grid_target(disp8, canvas_w, top_h)

        # --- Draw cam4 (letterboxed — same as 11) ---
        f4 = frame4
        if f4.ndim == 2:
            f4 = cv2.cvtColor(f4, cv2.COLOR_GRAY2BGR)
        elif f4.shape[2] == 4:
            f4 = cv2.cvtColor(f4, cv2.COLOR_BGRA2BGR)
        disp4, cam4_map = _fit_frame_to_layout(f4, canvas_w, bottom_h)
        _draw_cam4_crosshair(disp4, canvas_w, bottom_h, cam4_map)

        display = np.vstack([disp8, disp4])
        _hud_lines = _collect_full_calib_bottom_hud_lines(
            pan, tilt, saved, now, warn_msg, warn_until
        )
        _draw_bottom_panel_unified_hud(
            display, canvas_w, canvas_h, top_h, _hud_lines
        )
        cv2.imshow(win_name, display)

        # joystick confirm (edge)
        if confirm_pressed and not prev_confirm:
            _do_confirm(arm)
            saved = False
        prev_confirm = confirm_pressed

        key = cv2.waitKey(CALIB_WAIT_KEY_MS) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key in (ord("c"), ord("C")):
            _do_confirm(arm)
            saved = False

        elif key in (ord("d"), ord("D")):
            if _state.phase == "home":
                if _state.home_confirmed:
                    _state.home_confirmed = False
                    print("[CAL] Reference cleared.")
                    saved = False
            elif _state.phase == "limits" and _state.limit_corners:
                removed = _state.limit_corners.pop()
                _build_reachable_polygon()
                print(f"[CAL] Deleted limit vertex: {removed.get('label', '?')}")
                saved = False
            elif _state.phase == "calibration":
                if _state.calib_use_guided_grid and _state.guided_targets:
                    if _state.guided_idx > len(_state.points):
                        _state.guided_idx -= 1
                        print(
                            f"[CAL] Undo skip — guided {_state.guided_idx}/"
                            f"{len(_state.guided_targets)}"
                        )
                    elif _state.points:
                        _state.points.pop()
                        _state.guided_idx = len(_state.points)
                        print(
                            f"[CAL] Deleted point (remaining: {len(_state.points)}); "
                            f"guided index -> {_state.guided_idx}"
                        )
                    saved = False
                elif _state.points:
                    _state.points.pop()
                    print(
                        f"[CAL] Deleted calibration point (remaining: {len(_state.points)})"
                    )
                    saved = False

        elif key in (ord("s"), ord("S")):
            if not _state.home_confirmed:
                warn_msg = "WARNING: confirm REFERENCE first (Phase 0 -> press C)"
                warn_until = now + 3.0
                print("[CAL] WARNING: confirm reference first (Phase 0)")
            else:
                ok_s = save_calibration(
                    _state.points, _state.cam8_w, _state.cam8_h,
                    _state.limit_corners,
                    _state.home_arm_pan, _state.home_arm_tilt,
                )
                saved = ok_s

        elif key in (ord("r"), ord("R")):
            _state.points.clear()
            _state.cam8_click = None
            if _state.calib_use_guided_grid:
                _state.guided_idx = 0
            saved = False
            print("[CAL] Calibration points reset.")

        elif key in (ord("h"), ord("H")):
            _goto_reference(arm)

        elif key in (ord("l"), ord("L")):
            _state.limit_corners.clear()
            _state.reachable_poly_cam8 = None
            _state.phase = "limits"
            if _state.calib_use_guided_grid:
                _reset_guided_grid_for_new_cal_phase()
            saved = False
            print("[CAL] Limit vertices cleared — Phase 1")

        elif key in (ord(" "), 13):  # SPACE or Enter
            if _state.phase == "home":
                if not _state.home_confirmed:
                    print("[CAL] WARNING: Skipping reference setup -- recalibration will not work")
                _state.phase = "limits"
                print("[CAL] -> Phase 1: LIMIT SWEEP")
            elif _state.phase == "limits":
                _state.phase = "calibration"
                if _state.calib_use_guided_grid:
                    _reset_guided_grid_for_new_cal_phase()
                nl = len(_state.limit_corners)
                if nl < 3:
                    print(
                        f"[CAL] -> Phase 2 ({nl} limit vertices): "
                        "no polygon — full frame treated reachable until you redo limits"
                    )
                else:
                    print(f"[CAL] -> Phase 2: CALIBRATION ({nl} limit vertices)")
            elif _state.phase == "calibration" and _state.calib_use_guided_grid:
                if (
                    _state.guided_targets
                    and _state.guided_idx < len(_state.guided_targets)
                ):
                    _state.guided_idx += 1
                    print(
                        f"[CAL] Skipped cell — now {_state.guided_idx}/"
                        f"{len(_state.guided_targets)}"
                    )


# =================================================================
# Recalibration mode
# =================================================================

def _load_existing_json(path: Path) -> Optional[Dict]:
    if not path.is_file():
        print(f"[RECAL] ERROR: file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "camera_reference" not in data and "home_reference" not in data:
            print("[RECAL] ERROR: JSON has no camera_reference/home_reference -- run full calibration first")
            return None
        return data
    except Exception as e:
        print(f"[RECAL] ERROR: failed to load JSON: {e}")
        return None


def run_recal(arm, cam8, cam4, joystick, joy_mapper,
              json_path: Path = OUTPUT_JSON) -> None:
    """
    Recalibration mode: load existing JSON, show home mark, let operator re-aim.
    Press C -> compute delta -> apply to all points -> re-fit homography -> S.
    """
    data = _load_existing_json(json_path)
    if data is None:
        return

    stored_home = data.get("camera_reference") or data.get("home_reference")
    stored_home_pan = float(stored_home["arm_pan"])
    stored_home_tilt = float(stored_home["arm_tilt"])
    old_points: List[Dict] = list(data.get("calibration_points", []))
    old_limits: List[Dict] = list(data.get("arm_limit_corners", []))
    cam8_w_stored = int(data.get("output_width", 0))
    cam8_h_stored = int(data.get("output_height", 0))

    print(f"\n[RECAL] Loaded {len(old_points)} points  ref_stored: pan={stored_home_pan:.3f} tilt={stored_home_tilt:.3f}")
    print("[RECAL] Rotate arm to aim at REF mark (+) -> press C -> S to save\n")

    # Restore limit polygon if present
    rp = data.get("reachable_polygon_cam8")
    if rp:
        _state.reachable_poly_cam8 = np.array(rp, dtype=np.float32).reshape(-1, 1, 2)

    # Recal UI accepts cam8 clicks for multi-point refine.
    _state.phase = "calibration"
    _state.cam8_click = None

    win_name = "Cam8 Arm Recalibrator [RECAL MODE]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    _layouts = _calib_build_layouts()
    canvas_w = _layouts["top"]["width"]
    top_h = _layouts["top"]["height"]
    bottom_h = _layouts["bottom"]["height"]
    canvas_h = top_h + bottom_h
    _configure_calib_window(win_name, canvas_w, canvas_h)
    cv2.setMouseCallback(
        win_name, _on_mouse, param={"cam8_top_h": top_h}
    )

    last_sync = 0.0
    last_loop = time.time()
    prev_confirm = False
    recal_done = False
    saved = False
    delta_pan = 0.0
    delta_tilt = 0.0
    new_points: List[Dict] = old_points  # updated after confirm
    refine_points: List[Dict] = []
    test_mode = False
    last_test_click: Optional[Tuple[float, float]] = None

    while True:
        now = time.time()
        dt = now - last_loop
        last_loop = now

        ok8, frame8, _ = cam8.read()
        ok4, frame4, _ = cam4.read()

        if ok8 and frame8 is not None:
            _state.cam8_h, _state.cam8_w = frame8.shape[:2]
            _update_home_cam8_position()
        else:
            frame8 = _blank(top_h, canvas_w, "cam8 no signal")
        if not (ok4 and frame4 is not None):
            frame4 = _blank(bottom_h, canvas_w, "cam4 no signal")

        if now - last_sync >= SYNC_INTERVAL_SEC:
            try:
                arm.sync_position_from_grbl()
            except Exception:
                pass
            last_sync = now

        confirm_pressed = False
        if joystick is not None and joy_mapper is not None:
            js = joystick.read()
            confirm_pressed = (
                bool(js.buttons[JOY_CONFIRM_BUTTON])
                if hasattr(js, "buttons") and len(js.buttons) > JOY_CONFIRM_BUTTON
                else False
            )
            joy_mapper.apply(js, dt)

        pan = getattr(arm, "pos_x", 0.0)
        tilt = getattr(arm, "pos_y", 0.0)

        # Test mode: click cam8 -> auto move arm to predicted position (keeps click pending for C confirm).
        if test_mode and _state.cam8_click is not None:
            cur_click = (float(_state.cam8_click[0]), float(_state.cam8_click[1]))
            if last_test_click != cur_click:
                if _p2a is None:
                    print("[RECAL][TEST] pixel_to_arm_degrees unavailable.")
                else:
                    predicted = _p2a(cur_click[0], cur_click[1], data, _state.cam8_w, _state.cam8_h)
                    if predicted is None:
                        print(f"[RECAL][TEST] No mapping for click ({cur_click[0]:.0f},{cur_click[1]:.0f}).")
                    else:
                        try:
                            arm.move_absolute(float(predicted[0]), float(predicted[1]), blocking=False)
                            print(f"[RECAL][TEST] Go predicted -> pan={predicted[0]:.3f} tilt={predicted[1]:.3f}")
                        except Exception as e:
                            print(f"[RECAL][TEST] Move failed: {e}")
                last_test_click = cur_click

        # Draw cam8 (letterboxed)
        disp8 = _letterbox_cam8(frame8, canvas_w, top_h)
        _draw_grid_overlay(disp8, canvas_w, top_h)
        _draw_reachable_polygon(disp8)
        _draw_home_mark(disp8, canvas_w, top_h, confirmed=recal_done)
        _draw_pending_click(disp8, canvas_w, top_h)
        # Show existing calibration points
        if _state.cam8_w > 0 and cam8_w_stored > 0:
            scale_ratio = _state.cam8_disp_scale * (_state.cam8_w / cam8_w_stored)
            off_xr = _state.cam8_disp_off_x
            off_yr = _state.cam8_disp_off_y
            for pt in old_points:
                dx = off_xr + int(pt["cam8_px"] * scale_ratio)
                dy = off_yr + int(pt["cam8_py"] * scale_ratio)
                cv2.circle(disp8, (dx, dy), 5, (80, 200, 80), 1)
        # Show refine points captured during this recal run.
        for i, rp in enumerate(refine_points):
            rx, ry = _cam8_to_disp(rp["cam8_px"], rp["cam8_py"])
            cv2.circle(disp8, (rx, ry), 7, (0, 255, 255), 2)
            cv2.putText(disp8, f"R{i+1}", (rx + 8, ry - 4),
                        HUD_FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw cam4
        f4r = frame4
        if f4r.ndim == 2:
            f4r = cv2.cvtColor(f4r, cv2.COLOR_GRAY2BGR)
        elif f4r.shape[2] == 4:
            f4r = cv2.cvtColor(f4r, cv2.COLOR_BGRA2BGR)
        disp4, cam4_map = _fit_frame_to_layout(f4r, canvas_w, bottom_h)
        _draw_cam4_crosshair(disp4, canvas_w, bottom_h, cam4_map)

        display = np.vstack([disp8, disp4])
        _rh = _collect_recal_bottom_hud_lines(
            pan,
            tilt,
            saved,
            recal_done,
            delta_pan,
            delta_tilt,
            len(new_points),
            len(refine_points),
            test_mode,
            stored_home_pan,
            stored_home_tilt,
        )
        _draw_bottom_panel_unified_hud(display, canvas_w, canvas_h, top_h, _rh)
        cv2.imshow(win_name, display)

        # joystick confirm
        if confirm_pressed and not prev_confirm:
            if _state.cam8_click is not None:
                cx, cy = _state.cam8_click
                rp = {
                    "cam8_px": float(cx),
                    "cam8_py": float(cy),
                    "arm_pan": float(getattr(arm, "pos_x", 0.0)),
                    "arm_tilt": float(getattr(arm, "pos_y", 0.0)),
                }
                refine_points.append(rp)
                _state.cam8_click = None
                saved = False
                print(f"[RECAL] refine pt {len(refine_points)}: cam8=({cx:.0f},{cy:.0f}) "
                      f"arm=({rp['arm_pan']:.3f},{rp['arm_tilt']:.3f})")
            elif not recal_done:
                new_pan = getattr(arm, "pos_x", 0.0)
                new_tilt = getattr(arm, "pos_y", 0.0)
                delta_pan = new_pan - stored_home_pan
                delta_tilt = new_tilt - stored_home_tilt
                new_points = [
                    {**pt,
                     "arm_pan": pt["arm_pan"] + delta_pan,
                     "arm_tilt": pt["arm_tilt"] + delta_tilt}
                    for pt in old_points
                ]
                recal_done = True
                print(f"[RECAL] delta: pan={delta_pan:+.3f}deg tilt={delta_tilt:+.3f}deg  "
                      f"({len(new_points)} pts adjusted)  press S to save")
        prev_confirm = confirm_pressed

        key = cv2.waitKey(CALIB_WAIT_KEY_MS) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("c"), ord("C")):
            if _state.cam8_click is not None:
                cx, cy = _state.cam8_click
                rp = {
                    "cam8_px": float(cx),
                    "cam8_py": float(cy),
                    "arm_pan": float(getattr(arm, "pos_x", 0.0)),
                    "arm_tilt": float(getattr(arm, "pos_y", 0.0)),
                }
                refine_points.append(rp)
                _state.cam8_click = None
                saved = False
                print(f"[RECAL] refine pt {len(refine_points)}: cam8=({cx:.0f},{cy:.0f}) "
                      f"arm=({rp['arm_pan']:.3f},{rp['arm_tilt']:.3f})")
            elif not recal_done:
                new_pan = getattr(arm, "pos_x", 0.0)
                new_tilt = getattr(arm, "pos_y", 0.0)
                delta_pan = new_pan - stored_home_pan
                delta_tilt = new_tilt - stored_home_tilt
                new_points = [
                    {**pt,
                     "arm_pan": pt["arm_pan"] + delta_pan,
                     "arm_tilt": pt["arm_tilt"] + delta_tilt}
                    for pt in old_points
                ]
                recal_done = True
                print(f"[RECAL] delta: pan={delta_pan:+.3f}deg tilt={delta_tilt:+.3f}deg  "
                      f"({len(new_points)} pts adjusted)")
        elif key in (ord("h"), ord("H")):
            _goto_reference(arm)
        elif key in (ord("t"), ord("T")):
            test_mode = not test_mode
            last_test_click = None
            print(f"[RECAL][TEST] {'ON' if test_mode else 'OFF'}")
        elif key in (ord("d"), ord("D")):
            if refine_points:
                refine_points.pop()
                saved = False
                print(f"[RECAL] refine point deleted (remaining: {len(refine_points)})")
        elif key in (ord("r"), ord("R")):
            refine_points.clear()
            _state.cam8_click = None
            saved = False
            print("[RECAL] refine points reset.")
        elif key in (ord("s"), ord("S")):
            w8 = _state.cam8_w if _state.cam8_w > 0 else cam8_w_stored
            h8 = _state.cam8_h if _state.cam8_h > 0 else cam8_h_stored
            new_home_pan = stored_home_pan + delta_pan if recal_done else stored_home_pan
            new_home_tilt = stored_home_tilt + delta_tilt if recal_done else stored_home_tilt
            # Restore limit corners + polygon after applying delta
            new_limits = [
                {**c, "arm_pan": c["arm_pan"] + delta_pan,
                       "arm_tilt": c["arm_tilt"] + delta_tilt}
                for c in old_limits
            ]
                # Hybrid recal strategy:
                # Keep base points for global shape and merge refine points for local density.
            points_to_save = list(new_points)
            if len(refine_points) >= 1:
                merged: List[Dict] = []
                min_keep_dist_px = max(24.0, min(w8, h8) * 0.025)
                for bp in points_to_save:
                    too_close = False
                    for rp in refine_points:
                        if math.hypot(
                            float(bp["cam8_px"]) - float(rp["cam8_px"]),
                            float(bp["cam8_py"]) - float(rp["cam8_py"]),
                        ) < min_keep_dist_px:
                            too_close = True
                            break
                    if not too_close:
                        merged.append(bp)
                merged.extend(refine_points)
                points_to_save = merged
            if not points_to_save:
                print("[RECAL] WARNING: no points available to save.")
                continue
            _state.points = points_to_save
            _state.limit_corners = new_limits
            _state.home_confirmed = True
            _state.home_arm_pan = new_home_pan
            _state.home_arm_tilt = new_home_tilt
            ok_s = save_calibration(
                points_to_save, w8, h8, new_limits,
                new_home_pan, new_home_tilt,
            )
            saved = ok_s
            if ok_s:
                mode = "refine-merge" if len(refine_points) >= 1 else "delta-shift"
                print(f"[RECAL] ✓ Saved ({mode}) — new reference: pan={new_home_pan:.3f} tilt={new_home_tilt:.3f}")


# =================================================================
# Main entry point
# =================================================================

def main():
    # Parse args (defaults from CALIB_TOP_CAMERA / CALIB_BOTTOM_CAMERA in this file)
    cam8_name = CALIB_TOP_CAMERA
    cam4_name = CALIB_BOTTOM_CAMERA
    recal_mode = "--recal" in sys.argv
    _state.calib_use_guided_grid = "--guided-grid" in sys.argv
    _reset_guided_grid_for_new_cal_phase()
    use_config_py_only = "--use-config-py" in sys.argv
    use_file_11_cameras = "--use-file-11-cameras" in sys.argv
    if "--cam-top" in sys.argv:
        idx = sys.argv.index("--cam-top")
        if idx + 1 < len(sys.argv):
            cam8_name = sys.argv[idx + 1]
    if "--cam-bottom" in sys.argv:
        idx = sys.argv.index("--cam-bottom")
        if idx + 1 < len(sys.argv):
            cam4_name = sys.argv[idx + 1]
    if "--cam8" in sys.argv:
        idx = sys.argv.index("--cam8")
        if idx + 1 < len(sys.argv):
            cam8_name = sys.argv[idx + 1]
    if "--cam4" in sys.argv:
        idx = sys.argv.index("--cam4")
        if idx + 1 < len(sys.argv):
            cam4_name = sys.argv[idx + 1]

    mode_str = "RECALIBRATION" if recal_mode else "FULL CALIBRATION"
    _ly = _calib_build_layouts()
    _cv_w = _ly["top"]["width"]
    _cv_h = _ly["top"]["height"] + _ly["bottom"]["height"]
    _cap = f"{CALIB_DISPLAY_MAX_HEIGHT}px" if CALIB_DISPLAY_MAX_HEIGHT > 0 else "off"
    print(f"\n[CAL] ─── Cam8 Arm Grid Calibrator ({mode_str}) ───")
    print(
        f"[CAL] top_panel={cam8_name}  bottom_panel={cam4_name}  "
        f"(defaults: {CALIB_TOP_CAMERA} / {CALIB_BOTTOM_CAMERA})  output={OUTPUT_JSON}"
    )
    print(
        f"[CAL] Canvas: {_cv_w}x{_cv_h}  (screen cap height={_cap})  "
        f"top/bottom 50/50  fullscreen={CALIB_USE_FULLSCREEN}  "
        f"waitKey={CALIB_WAIT_KEY_MS}ms  (layout like file 11)"
    )
    if _state.calib_use_guided_grid and not recal_mode:
        print(
            f"[CAL] Phase 2: GUIDED GRID — red dot at each reachable cell center "
            f"({GRID_COLS}x{GRID_ROWS} layout)\n"
        )
    else:
        print()

    try:
        _build_cam, _cam_src = _resolve_camera_builder(
            use_config_py=use_config_py_only,
            use_file_11=use_file_11_cameras,
        )
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    print(f"[CAL] Camera pipeline: {_cam_src}")
    if use_config_py_only:
        print("[CAL] (--use-config-py: using config.py CAMERAS only)\n")
    elif use_file_11_cameras:
        print("[CAL] (--use-file-11-cameras: using file 11 LOCAL_CAMERAS)\n")
    if Cam4ArmController is None:
        print("❌ Cam4ArmController not available.")
        return

    cam8 = _build_cam(cam8_name)
    cam8.start()
    cam4 = _build_cam(cam4_name)
    cam4.start()
    time.sleep(0.5)

    arm = Cam4ArmController()
    if not arm.connect():
        print("❌ Arm connect failed.")
        cam8.release()
        cam4.release()
        return

    joystick = None
    joy_mapper = None
    if JoystickReader and JoystickArmMapper:
        try:
            joystick = JoystickReader()
            if getattr(joystick, "enabled", False):
                joy_mapper = JoystickArmMapper(arm, initial_sensitivity_mode=SENSITIVITY_LOW)
            else:
                joystick = None
        except Exception as e:
            print(f"[CAL] Joystick: {e}")

    had_unsaved_changes = False
    saved_during_session = False
    points_before = len(_state.points)
    mtime_before = OUTPUT_JSON.stat().st_mtime if OUTPUT_JSON.is_file() else None
    try:
        if recal_mode:
            _print_recal_help(cam8_name)
            run_recal(arm, cam8, cam4, joystick, joy_mapper)
        else:
            _print_full_help()
            _run_loop(arm, cam8, cam4, joystick, joy_mapper, mode="full")
        points_after = len(_state.points)
        had_unsaved_changes = points_after > points_before
        mtime_after = OUTPUT_JSON.stat().st_mtime if OUTPUT_JSON.is_file() else None
        if mtime_after is not None:
            saved_during_session = (
                mtime_before is None or mtime_after > mtime_before
            )
    finally:
        arm.disconnect()
        cam8.release()
        cam4.release()
        cv2.destroyAllWindows()
        print(f"[CAL] Exit — home={'yes' if _state.home_confirmed else 'no'}  "
              f"limits={len(_state.limit_corners)}  pts={len(_state.points)}")
        # Show unsaved warning only when user exits with new points that were not persisted.
        if had_unsaved_changes and not saved_during_session:
            print("[CAL] WARNING: Points not saved -- run again and press S")


def _print_full_help() -> None:
    print("Phase 0 REFERENCE SETUP:")
    print("  Place a target at the REF + mark on cam8 (center-x, 10% from bottom)")
    print("  Rotate arm until cam4 sees target at centre -> C to confirm -> SPACE for next")
    print("Phase 1 LIMIT OUTLINE:")
    print("  Click CAM8 (top) at each corner, C to add vertex — polygon follows click order (≥3)")
    print("  SPACE/Enter -> Phase 2   D delete last   L clear limits")
    print("Phase 2 CALIBRATION (default: free click):")
    print("  Click cam8 (top) -> aim CAM4 -> C  (repeat >=9) -> S save")
    print("  Or run with --guided-grid:")
    print("    After limits, aim CAM4 at each RED dot (cell center in zone); C confirm; SPACE skip; D undo")
    print("    Optional extra points by click+C after all red dots done -> S save\n")


def _print_recal_help(cam8_name: str) -> None:
    print("RECALIBRATION MODE:")
    print(f"  Loading: {OUTPUT_JSON}")
    print(f"  Place a target at the REF + mark (center-x, 10% from bottom) on {cam8_name}")
    print("  C without click: apply global REF delta shift")
    print("  T test mode: click cam8 -> auto-go predicted -> adjust -> C (repeat)")
    print("  S save: merge base + refine points and rebuild hybrid map\n")


if __name__ == "__main__":
    main()
