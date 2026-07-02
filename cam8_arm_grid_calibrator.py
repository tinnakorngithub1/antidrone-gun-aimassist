"""
cam8_arm_grid_calibrator.py
---------------------------
Calibration tool: map cam8 pixel → arm pan/tilt for AUTO cue.
Saves to calibration_data/cam8_mouse_grid_lookup.json (separate from cam4 calib).
Run on arm Jetson (192.168.144.66) only.

Usage:
  python3 cam8_arm_grid_calibrator.py               # Full calibration (3 phases)
  python3 cam8_arm_grid_calibrator.py --cam-top cam8 --cam-bottom cam4   # override file defaults
  python3 cam8_arm_grid_calibrator.py --cam8 cam8 --cam4 cam4            # legacy (same as above)
  python3 cam8_arm_grid_calibrator.py --use-config-py        # Cameras from config.py
  python3 cam8_arm_grid_calibrator.py --use-file-11-cameras  # file 11 LOCAL_CAMERAS (not embedded)

Edit CALIB_TOP_CAMERA / CALIB_BOTTOM_CAMERA and CALIB_LOCAL_CAMERAS below for streams and URLs.
Default: embedded CALIB_LOCAL_CAMERAS + fast_motion_sky.CameraStream (same shape as file 11).

─── 3 Phases (Full Calibration) ───────────────────────────────
Phase 0 — REFERENCE SETUP:
  Camera reference is fixed at center-x, 10% from bottom of cam8.
  Place a physical target at the reference mark ⊕ in the field.
  Rotate arm until cam4 sees the target at center, then press C.
  → Records home_arm_pan, home_arm_tilt.

Phase 1 — LIMIT (outline by order):
  Click on CAM8 (top) where you want each corner of the zone, then press C to record.
  Vertices connect in click order (polygon on screen; not a convex box). ≥3 vertices;
  SPACE → Phase 2. D deletes last vertex; L clears all. (Moving the arm is optional.)

Phase 2 — DIRECT CELL MAPPING (20x10):
  Click a grid cell in CAM8 → aim cam4 center → C to map/update cell.
  D deletes selected cell mapping. S saves even if mapped cells are incomplete.

Keys (all phases):
  C / Joystick B0   - confirm current step (reference / limit vertex / selected cell)
  SPACE / Enter     - advance to next phase
  D                 - delete selected cell mapping (Phase 2)
  S                 - save JSON
  R                 - reset mapped cells (keeps home/limits)
  H                 - move arm to saved reference position
  L                 - reset limit sweep entirely
  Q / ESC           - quit
"""

import importlib.util
import json
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

# Grid overlay and mapping table (fixed by requirement).
GRID_COLS = 20
GRID_ROWS = 10

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
    show_reachable_overlay: bool = True
    test_only_mode: bool = False
    test_go_pending: bool = False

    # Phase 2: Cell mapping
    cam8_click: Optional[Tuple[float, float]] = None
    selected_cell: Optional[Tuple[int, int]] = None
    selected_source_xy: Optional[Tuple[float, float]] = None
    cell_to_pan_tilt: Dict[str, List[float]] = None
    dirty: bool = False

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
_state.cell_to_pan_tilt = {}


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
    result = _disp_to_cam8(x, y)
    if result is None:
        return
    orig_x, orig_y = result
    if _state.phase == "limits":
        _state.cam8_click = (orig_x, orig_y)
        print(f"[CAL] limit click cam8: ({orig_x:.1f}, {orig_y:.1f})")
        return
    cell = _cam8_pixel_to_cell(orig_x, orig_y)
    if cell is None:
        return
    _state.selected_cell = cell
    _state.selected_source_xy = _cell_center_cam8(*cell)
    _state.cam8_click = _state.selected_source_xy
    if _state.test_only_mode:
        # In TEST-ONLY mode: selecting a cell should trigger immediate go-to mapped position.
        _state.test_go_pending = True
    print(
        f"[CAL] select cell r{cell[0]} c{cell[1]} at "
        f"({_state.selected_source_xy[0]:.1f}, {_state.selected_source_xy[1]:.1f})"
    )


def _cell_key(row: int, col: int) -> str:
    return f"{int(row)}_{int(col)}"


def _cam8_pixel_to_cell(px: float, py: float) -> Optional[Tuple[int, int]]:
    if _state.cam8_w <= 0 or _state.cam8_h <= 0:
        return None
    col = min(max(int((float(px) / float(_state.cam8_w)) * GRID_COLS), 0), GRID_COLS - 1)
    row = min(max(int((float(py) / float(_state.cam8_h)) * GRID_ROWS), 0), GRID_ROWS - 1)
    return row, col


def _cell_center_cam8(row: int, col: int) -> Tuple[float, float]:
    cell_w = float(_state.cam8_w) / float(max(1, GRID_COLS))
    cell_h = float(_state.cam8_h) / float(max(1, GRID_ROWS))
    cx = (float(col) + 0.5) * cell_w
    cy = (float(row) + 0.5) * cell_h
    return cx, cy


# =================================================================
# Grid + Polygon helpers
# =================================================================

def _draw_grid_overlay(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    """Draw mapping grid: gray unmapped, green mapped, yellow selected."""
    off_x = _state.cam8_disp_off_x
    off_y = _state.cam8_disp_off_y
    nw = _state.cam8_disp_scaled_w
    nh = _state.cam8_disp_scaled_h
    if nw <= 0 or nh <= 0:
        return
    cell_w = nw / GRID_COLS
    cell_h = nh / GRID_ROWS
    overlay = disp8.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx_cam8, cy_cam8 = _cell_center_cam8(r, c)
            x0 = off_x + int(c * cell_w)
            y0 = off_y + int(r * cell_h)
            x1 = off_x + int((c + 1) * cell_w)
            y1 = off_y + int((r + 1) * cell_h)
            key = _cell_key(r, c)
            inside = _is_in_reachable(cx_cam8, cy_cam8)
            if key in _state.cell_to_pan_tilt:
                fill = (0, 120, 0)
            else:
                fill = (70, 70, 70) if inside else (35, 35, 55)
            if _state.selected_cell == (r, c):
                fill = (0, 220, 220)
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
    if (not _state.show_reachable_overlay) or _state.reachable_poly_cam8 is None:
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


def _build_data_dict(cells, cam8_w, cam8_h, limit_corners,
                     home_pan, home_tilt) -> Dict[str, Any]:
    cx8, cy8 = cam8_w / 2.0, cam8_h / 2.0
    data: Dict[str, Any] = {
        "source_camera": "cam8",
        "output_width": cam8_w,
        "output_height": cam8_h,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "cells": dict(cells),
        "crosshair": {"x": cx8, "y": cy8},
        "num_cells_mapped": len(cells),
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
    if _state.reachable_poly_cam8 is not None:
        data["reachable_polygon_cam8"] = \
            _state.reachable_poly_cam8.reshape(-1, 2).tolist()
    data["mapping_mode"] = "direct_cell_mapping"
    return data


def save_calibration(
    cells: Dict[str, List[float]],
    cam8_w: int,
    cam8_h: int,
    limit_corners: List[Dict],
    home_pan: float,
    home_tilt: float,
    output_path: Path = OUTPUT_JSON,
) -> bool:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    data = _build_data_dict(cells, cam8_w, cam8_h, limit_corners, home_pan, home_tilt)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[CAL] Saved {len(cells)}/{GRID_ROWS * GRID_COLS} cells → {output_path}")
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
    """Phase calibration: record selected grid cell -> current arm pan/tilt."""
    if _state.selected_cell is None:
        print("[CAL] Click a grid cell on CAM8 first, then press C")
        return
    row, col = _state.selected_cell
    cx, cy = _cell_center_cam8(row, col)
    if not _is_in_reachable(cx, cy):
        print(f"[CAL] WARNING: cell r{row} c{col} center is outside reachable zone")
    pan = getattr(arm, "pos_x", 0.0)
    tilt = getattr(arm, "pos_y", 0.0)
    key = _cell_key(row, col)
    _state.cell_to_pan_tilt[key] = [float(pan), float(tilt)]
    _state.cam8_click = (cx, cy)
    _state.dirty = True
    print(
        f"[CAL] Mapped cell r{row} c{col}: cam8=({cx:.0f},{cy:.0f}) "
        f"arm=({pan:.3f},{tilt:.3f})  total={len(_state.cell_to_pan_tilt)}/{GRID_ROWS * GRID_COLS}"
    )


def _test_selected_cell(arm) -> bool:
    """Move arm to mapped pan/tilt for currently selected cell."""
    if _state.selected_cell is None:
        print("[CAL][TEST] No selected cell.")
        return False
    row, col = _state.selected_cell
    key = _cell_key(row, col)
    val = _state.cell_to_pan_tilt.get(key)
    if val is None or not isinstance(val, (list, tuple)) or len(val) < 2:
        print(f"[CAL][TEST] Cell r{row} c{col} is not mapped yet.")
        return False
    try:
        pan = float(val[0])
        tilt = float(val[1])
        arm.move_absolute(pan, tilt, blocking=False)
        print(f"[CAL][TEST] Go r{row} c{col} -> pan={pan:.3f} tilt={tilt:.3f}")
        return True
    except Exception as e:
        print(f"[CAL][TEST] Move failed: {e}")
        return False


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


def _load_cells_from_file(json_path: Path = OUTPUT_JSON) -> int:
    """Preload mapped cells from previous calibration JSON. Returns loaded count."""
    try:
        if not json_path.is_file():
            return 0
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = data.get("cells", {}) if isinstance(data, dict) else {}
        if not isinstance(cells, dict):
            return 0
        loaded: Dict[str, List[float]] = {}
        for key, val in cells.items():
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                continue
            try:
                row_s, col_s = str(key).split("_", 1)
                row = int(row_s)
                col = int(col_s)
                if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
                    continue
                loaded[_cell_key(row, col)] = [float(val[0]), float(val[1])]
            except Exception:
                continue
        _state.cell_to_pan_tilt = loaded
        _state.dirty = False
        return len(loaded)
    except Exception:
        return 0


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
    # Mapped state is already visible by green grid-cell overlay (same style as file 11).
    # Keep center markers off to reduce clutter on dense 20x10 grid.
    return


def _draw_pending_click(disp8: np.ndarray, half_w: int, disp_h: int) -> None:
    if _state.cam8_w <= 0:
        return
    if _state.phase == "limits":
        if _state.cam8_click is None:
            return
        px, py = _cam8_to_disp(_state.cam8_click[0], _state.cam8_click[1])
        col = (0, 180, 255)
        cv2.circle(disp8, (px, py), 10, col, 2)
        cv2.drawMarker(disp8, (px, py), col, cv2.MARKER_CROSS, 20, 2)
        return
    if _state.selected_source_xy is None:
        return
    px, py = _cam8_to_disp(_state.selected_source_xy[0], _state.selected_source_xy[1])
    in_r = _is_in_reachable(_state.selected_source_xy[0], _state.selected_source_xy[1])
    col = (0, 220, 220) if in_r else (0, 0, 255)
    # Match file 11 mapper selection marker size (smaller, unobtrusive).
    cv2.drawMarker(disp8, (px, py), col, cv2.MARKER_CROSS, 14, 1)
    if not in_r:
        cv2.putText(disp8, "OUT OF RANGE", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)


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
    if _state.test_only_mode:
        out.append(("TEST-ONLY MODE", t))
        out.append(("Click mapped cell -> arm goes immediately", t))
        out.append(("Adjust aim, then press C to overwrite selected cell", t))
        out.append(("Press S to save JSON", t))
        out.append(("", t))
        out.append(("C = overwrite selected   S = save JSON   T = exit test mode", t))
        return out
    phase = _state.phase
    if phase == "home":
        out.append(("PHASE 0 — REFERENCE", t))
        if _state.home_confirmed:
            out.append(("Reference saved — press SPACE for Phase 1", t))
        else:
            out.append(("Place target on yellow REF mark (top / CAM8)", t))
            out.append(("Center CAM4 crosshair on target, press C", t))
        out.append(("TIP: click CAM8 grid cell, press T to TEST mapped cell anytime", t))
    elif phase == "limits":
        out.append(("PHASE 1 — LIMIT OUTLINE (click CAM8 + C, order = polygon)", t))
        out.append(("Click each corner on CAM8, press C to record (≥3); SPACE → Phase 2", t))
        nl = len(_state.limit_corners)
        out.append((f"Vertices: {nl}   Arm  pan {pan:+.1f}   tilt {tilt:+.1f}", t))
    else:
        mapped = len(_state.cell_to_pan_tilt)
        total = GRID_ROWS * GRID_COLS
        out.append((f"PHASE 2 — CELL MAPPING ({GRID_COLS}x{GRID_ROWS})", t))
        out.append(("Click CAM8 grid cell, aim CAM4, press C to map/update", t))
        if _state.reachable_poly_cam8 is not None:
            out.append(("Gray=unmapped Green=mapped Yellow=selected", t))
        else:
            out.append(("No limit polygon — full frame treated reachable", t))
        out.append((f"Progress mapped {mapped}/{total}", t))
    out.append((f"Reachable overlay: {'ON' if _state.show_reachable_overlay else 'OFF'}", t))

    out.append(("", t))
    out.append(("— Lower panel (CAM4) — aim at yellow crosshair —", t))
    out.append((f"Arm  pan {pan:+.1f}   tilt {tilt:+.1f}", t))
    out.append(("", t))

    if phase == "home":
        out.append(("C = confirm reference   SPACE = next   Q = quit", t))
        out.append(("T = test selected cell   O = toggle reachable overlay", t))
        out.append(("H = move arm to saved reference", t))
    elif phase == "limits":
        out.append(("C / B0 = add vertex   SPACE = Phase 2   Q = quit", t))
        out.append(("D = delete last vertex   L = clear all limit points   O = toggle overlay", t))
    else:
        out.append(("C = confirm cell   S = save JSON   Q = quit", t))
        out.append(("T = test selected cell   O = toggle reachable overlay", t))
        out.append(("D = delete selected cell   R = clear mapped cells", t))
        out.append(("H / L = go ref / reset limits", t))
    if saved:
        out.append(("[ CALIBRATION SAVED ]", BOTTOM_HUD_OK_BGR))
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
# Main loop
# =================================================================

def _run_loop(arm, cam8, cam4, joystick, joy_mapper) -> None:
    """Main calibration loop for home/limits/cell-mapping phases."""
    win_name = "Cam8 Arm Calibrator"
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

        if _state.test_only_mode and _state.test_go_pending:
            _state.test_go_pending = False
            _test_selected_cell(arm)

        # --- Draw cam8 (letterboxed — same as 11 _fit_frame_to_layout) ---
        disp8 = _letterbox_cam8(frame8, canvas_w, top_h)
        _draw_grid_overlay(disp8, canvas_w, top_h)
        _draw_reachable_polygon(disp8)
        if not _state.test_only_mode:
            _draw_home_mark(disp8, canvas_w, top_h, _state.home_confirmed)
        _draw_confirmed_points(disp8, canvas_w, top_h)
        _draw_pending_click(disp8, canvas_w, top_h)

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
            if _state.test_only_mode:
                _confirm_calibration(arm)
            else:
                _do_confirm(arm)
            saved = False
        prev_confirm = confirm_pressed

        key = cv2.waitKey(CALIB_WAIT_KEY_MS) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key in (ord("c"), ord("C")):
            if _state.test_only_mode:
                _confirm_calibration(arm)
            else:
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
                if _state.selected_cell is not None:
                    key = _cell_key(*_state.selected_cell)
                    if key in _state.cell_to_pan_tilt:
                        _state.cell_to_pan_tilt.pop(key, None)
                        print(f"[CAL] Deleted mapped cell r{_state.selected_cell[0]} c{_state.selected_cell[1]}")
                    else:
                        print(f"[CAL] Selected cell r{_state.selected_cell[0]} c{_state.selected_cell[1]} is not mapped")
                    saved = False

        elif key in (ord("s"), ord("S")):
            if (not _state.test_only_mode) and (not _state.home_confirmed):
                warn_msg = "WARNING: confirm REFERENCE first (Phase 0 -> press C)"
                warn_until = now + 3.0
                print("[CAL] WARNING: confirm reference first (Phase 0)")
            else:
                ok_s = save_calibration(
                    _state.cell_to_pan_tilt, _state.cam8_w, _state.cam8_h,
                    _state.limit_corners,
                    _state.home_arm_pan, _state.home_arm_tilt,
                )
                saved = ok_s
                if _state.test_only_mode and ok_s:
                    print("[CAL][TEST] JSON saved.")

        elif key in (ord("r"), ord("R")):
            _state.cell_to_pan_tilt.clear()
            saved = False
            print("[CAL] Mapped cells reset.")

        elif key in (ord("t"), ord("T")):
            _state.test_only_mode = not _state.test_only_mode
            if _state.test_only_mode:
                print("[CAL][TEST] TEST-ONLY mode ON")
                ok_t = _test_selected_cell(arm)
                if ok_t and _state.selected_cell is not None:
                    r, c = _state.selected_cell
                    warn_msg = f"TEST GO r{r} c{c}"
                    warn_until = now + 1.8
            else:
                print("[CAL][TEST] TEST-ONLY mode OFF")
                _state.test_go_pending = False

        elif key in (ord("h"), ord("H")):
            _goto_reference(arm)

        elif key in (ord("o"), ord("O")):
            _state.show_reachable_overlay = not _state.show_reachable_overlay
            mode = "ON" if _state.show_reachable_overlay else "OFF"
            print(f"[CAL] Reachable overlay: {mode}")

        elif key in (ord("l"), ord("L")):
            _state.limit_corners.clear()
            _state.reachable_poly_cam8 = None
            _state.phase = "limits"
            saved = False
            print("[CAL] Limit vertices cleared — Phase 1")

        elif key == ord(" "):  # SPACE
            if _state.test_only_mode:
                continue
            if _state.phase == "home":
                if not _state.home_confirmed:
                    print("[CAL] WARNING: Skipping reference setup")
                _state.phase = "limits"
                print("[CAL] -> Phase 1: LIMIT SWEEP")
            elif _state.phase == "limits":
                _state.phase = "calibration"
                nl = len(_state.limit_corners)
                if nl < 3:
                    print(
                        f"[CAL] -> Phase 2 ({nl} limit vertices): "
                        "no polygon — full frame treated reachable until you redo limits"
                    )
                else:
                    print(f"[CAL] -> Phase 2: CALIBRATION ({nl} limit vertices)")

        elif key == 13:  # Enter
            if _state.test_only_mode:
                _confirm_calibration(arm)
                saved = False
                continue
            if _state.phase == "home":
                if not _state.home_confirmed:
                    print("[CAL] WARNING: Skipping reference setup")
                _state.phase = "limits"
                print("[CAL] -> Phase 1: LIMIT SWEEP")
            elif _state.phase == "limits":
                _state.phase = "calibration"
                nl = len(_state.limit_corners)
                if nl < 3:
                    print(
                        f"[CAL] -> Phase 2 ({nl} limit vertices): "
                        "no polygon — full frame treated reachable until you redo limits"
                    )
                else:
                    print(f"[CAL] -> Phase 2: CALIBRATION ({nl} limit vertices)")
            else:
                _do_confirm(arm)
                saved = False


# =================================================================
# Main entry point
# =================================================================

def main():
    # Parse args (defaults from CALIB_TOP_CAMERA / CALIB_BOTTOM_CAMERA in this file)
    cam8_name = CALIB_TOP_CAMERA
    cam4_name = CALIB_BOTTOM_CAMERA
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

    _ly = _calib_build_layouts()
    _cv_w = _ly["top"]["width"]
    _cv_h = _ly["top"]["height"] + _ly["bottom"]["height"]
    _cap = f"{CALIB_DISPLAY_MAX_HEIGHT}px" if CALIB_DISPLAY_MAX_HEIGHT > 0 else "off"
    print("\n[CAL] ─── Cam8 Arm Grid Calibrator (DIRECT CELL MAPPING) ───")
    print(
        f"[CAL] top_panel={cam8_name}  bottom_panel={cam4_name}  "
        f"(defaults: {CALIB_TOP_CAMERA} / {CALIB_BOTTOM_CAMERA})  output={OUTPUT_JSON}"
    )
    print(
        f"[CAL] Canvas: {_cv_w}x{_cv_h}  (screen cap height={_cap})  "
        f"top/bottom 50/50  fullscreen={CALIB_USE_FULLSCREEN}  "
        f"waitKey={CALIB_WAIT_KEY_MS}ms  (layout like file 11)"
    )
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

    preloaded_cells = _load_cells_from_file(OUTPUT_JSON)
    print(f"[CAL] Preloaded mapped cells: {preloaded_cells}/{GRID_ROWS * GRID_COLS}")

    had_unsaved_changes = False
    saved_during_session = False
    points_before = len(_state.cell_to_pan_tilt)
    mtime_before = OUTPUT_JSON.stat().st_mtime if OUTPUT_JSON.is_file() else None
    try:
        _print_full_help()
        _run_loop(arm, cam8, cam4, joystick, joy_mapper)
        points_after = len(_state.cell_to_pan_tilt)
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
              f"limits={len(_state.limit_corners)}  mapped={len(_state.cell_to_pan_tilt)}/{GRID_ROWS * GRID_COLS}")
        # Show unsaved warning only when user exits with new points that were not persisted.
        if had_unsaved_changes and not saved_during_session:
            print("[CAL] WARNING: Mapped cells not saved -- run again and press S")


def _print_full_help() -> None:
    print("Phase 0 REFERENCE SETUP:")
    print("  Place a target at the REF + mark on cam8 (center-x, 10% from bottom)")
    print("  Rotate arm until cam4 sees target at centre -> C to confirm -> SPACE for next")
    print("  If JSON exists: click any CAM8 grid cell -> T enters TEST-ONLY mode (HUD hidden)")
    print("  In TEST-ONLY mode: click mapped cell -> arm goes immediately; adjust -> C overwrite -> S save")
    print("  O = toggle reachable boundary overlay on CAM8")
    print("Phase 1 LIMIT OUTLINE:")
    print("  Click CAM8 (top) at each corner, C to add vertex — polygon follows click order (≥3)")
    print("  SPACE/Enter -> Phase 2   D delete last   L clear limits")
    print(f"Phase 2 DIRECT CELL MAPPING ({GRID_COLS}x{GRID_ROWS}):")
    print("  Click grid cell on cam8 (top) -> T test-only -> adjust aim -> C to map/update that cell")
    print(f"  D delete selected cell   R clear all mapped cells   S save (can save before {GRID_COLS * GRID_ROWS}/{GRID_COLS * GRID_ROWS})")
    print("  Enter = confirm selected cell in Phase 2\n")


if __name__ == "__main__":
    main()
