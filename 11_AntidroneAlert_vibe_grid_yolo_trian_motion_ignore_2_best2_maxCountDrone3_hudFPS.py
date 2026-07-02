#python3 11_AntidroneAlert_vibe_grid_yolo_trian_motion_ignore.py

import cv2
import datetime
import numpy as np
import time
import gc
import os
import math
import sys
import json
import copy
import queue
import threading
import subprocess
import traceback
import multiprocessing as mp
from collections import deque

from bottom_stitch_source import run_bottom_stitch_worker as _run_bottom_stitch_worker_impl
from canonical_path import (
    CANONICAL_PATH_SCHEMA_VERSION,
    ensure_path_ml_bundle,
    ensure_storage_layout,
)
from antidrone_ws_exporter import AntidroneWsExporter, build_drone_record

try:
    from arm_cue_sender import ArmCueSender
except ImportError:
    ArmCueSender = None
    print("⚠️ arm_cue_sender not found — arm cue disabled")

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics library not found. YOLO detection will be disabled.")
    print("   Install with: pip install ultralytics")

# YOLO Configuration (YOLO11n TensorRT engine)
YOLO_ENGINE_SIZE = "640"  # auto|640|1280
YOLO_MODEL_FILE_640 = "yolo_11n_day_night_200_2_imgsz640.engine"
YOLO_MODEL_FILE_1280 = "yolo_11n_day_night_200_2_imgsz1280.engine"
YOLO_INPUT_SIZE = 640  # ต้องตรงกับขนาดของ engine ที่ใช้
LOCAL_YOLO_CONF_THRESHOLD = 0.01
LOCAL_YOLO_DRONE_CONF_THRESHOLD = 0.5
# --- Arm cue confidence tier (ดู ARM_CUE_PROTOCOL.md) ---
# possible tier: เป้าที่ "มีความเป็นไปได้" แต่ยังไม่ถึงเกณฑ์ confirmed —
# ส่งให้ Jetson แขนชี้รอ human ตัดสินใจกด LOCK (ไม่เข้า alert/websocket)
LOCAL_YOLO_DRONE_POSSIBLE_CONF_THRESHOLD = 0.30  # YOLO conf >= นี้ (แต่ < เกณฑ์หลัก) → possible
ARM_CUE_POSSIBLE_PATH_MIN_CONF = 0.30            # validated path ต้องเคยมี YOLO conf >= นี้
ARM_CUE_MAX_CANDIDATES = 8                       # จำกัดจำนวน cue ต่อ datagram
YOLO_BBOX_MERGE_IOU = 0.5
YOLO_INTERVAL = 5  # ทำ YOLO ทุก 5 เฟรม
WIDE_SPLIT_ENABLED = True
WIDE_SPLIT_MIN_WIDTH = 3841
WIDE_SPLIT_OVERLAP_PX = 128
YOLO_OPEN_GRID_RADIUS = 2
YOLO_OPEN_GRID_TTL_FRAMES = 40
MOTION_YOLO_IOU_THRESH = 0.15
MOTION_YOLO_MAX_CENTER_DIST = 80
# When motion bbox is much smaller than YOLO, IoU stays low; IoMin fixes duplicate tracks.
MOTION_YOLO_IOMIN_THRESH = 0.22
# Display / count: treat YOLO as same object as path if boxes overlap (not only YOLO center inside path).
YOLO_PATH_MATCH_IOU = 0.04
YOLO_PATH_MATCH_IOMIN = 0.18
# เมื่อเฟรมกระตุก bbox YOLO กับ tracker อาจไม่ซ้อน — ถ้าจุดกลาง YOLO ใกล้เส้น path ให้ถือเป็นตัวเดียวกัน
YOLO_PATH_POLYLINE_FRAC = 0.04  # สัดส่วนของ min(frame_w, frame_h)
YOLO_PATH_POLYLINE_MIN_PX = 48
YOLO_PATH_POLYLINE_MAX_PX = 200
MOTION_BBOX_MERGE_IOU = 0.5  # merge กล่อง motion ที่ซ้อนกันก่อนส่ง tracker
# จำกัดจำนวน detection ต่อเฟรมที่ส่งเข้า Kalman เมื่อมี blob เยอะ (เช่น นกเต็มฟ้า)
TRACKER_INPUT_TOPK = 3
TRACKER_TOPK_AREA_WEIGHT = 1.0
TRACKER_TOPK_CONF_WEIGHT = 400.0
TRACKER_TOPK_EXISTING_BONUS = 3500.0
TRACKER_TOPK_MATCH_DIST_MIN = 72
TRACKER_TOPK_MATCH_DIST_FRAC = 0.055
HANDOFF_GRID_RADIUS = 2
HANDOFF_FRAMES = 20
SKY_MIN_AREA = 2
DENSE_MOTION_THRESHOLD = 300
SMALL_OBJ_AREA = 40
SMALL_OBJ_MIN_DIST = 2
STAR_STATIC_AREA = 25
STAR_STATIC_FRAMES = 12
STAR_STATIC_MAX_DIST = 2

# Active Label Trainer (Human-in-the-loop)
try:
    from active_label_trainer import ActiveLabelTrainer
    ACTIVE_LABEL_AVAILABLE = True
except Exception:
    ACTIVE_LABEL_AVAILABLE = False
    ActiveLabelTrainer = None

ENABLE_KINEMATIC_RULES = True
# Set True to log validation / tail-revoke messages (hot path; default off for FPS)
LOG_KINEMATIC_PATH = False
# ลด false "โดรน" เร็วเกิน: ยก threshold + ยาวประวัติ YOLO + path ยาวขึ้นก่อน validate/kinematic
YOLO_AVG_DRONE_THRESHOLD = 0.72
YOLO_CONF_HISTORY_MAXLEN = 5
YOLO_RED_MIN_HISTORY = 5  # ต้องมี confidence ครบ N ช่องประวัติก่อนขึ้น red (เท่า maxlen หรือน้อยกว่า)
PATH_VALIDATE_MIN_POINTS = 5  # เดิม 5 จุด — ต้องมี path ยาวขึ้นก่อน validated
# Optional future: run _is_window_smooth on smoothed_points for consistency with display/features.
PATH_MIN_DIST_PX = 3  # จุดใหม่เมื่อห่างจากจุดล่าสุดมากกว่านี้ (px); เดิม hard-coded 10
PATH_MIN_DIST_SMALL_OBJ_PX = 2  # เมื่อ area < SMALL_OBJ_AREA ใช้ค่านี้แทน PATH_MIN_DIST_PX
PATH_MAX_NODES = 100  # เพิ่มหน้าต่างย้อนหลังเมื่อจุดถี่ขึ้น (เดิม 50)
PATH_SMOOTH_WINDOW = 7  # moving average บน path; เดิม magic 5
# วาดเฉพาะหางล่าสุด N จุด — ไม่กระทบ deque / kinematic / Active Label (แก้แค่ HUD)
PATH_DISPLAY_MAX_POINTS = 40
# Deferred: separate dense_points deque or second MA pass for display-only smoothing.
KIN_MIN_POINTS_BEFORE_DRONE_CONFIRM = 10  # smoothed path อย่างน้อยนี้ก่อนตั้ง kinematic_confirmed

# Kinematic rules thresholds (tune at top of file)
KIN_AREA_VARIANCE_MAX = 0.15  # rigid body check for all patterns
KIN_FRAME_GAP_MAX = 3  # gap > this = tracking lost, skip pattern
# Hovering
KIN_HOVER_AVG_VEL_MAX = 2.0
KIN_HOVER_MAX_VEL_MAX = 3.0
KIN_HOVER_POS_CHANGE_MAX = 10.0
# Pivot
KIN_PIVOT_FIRST_HALF_VEL_MAX = 3.0
KIN_PIVOT_SECOND_HALF_VEL_MIN = 1.0
KIN_PIVOT_DIR_DOT_MAX = 0.5
KIN_PIVOT_ACCEL_MIN = 0.5
KIN_PIVOT_ACCEL_MAX = 5.0
# Vertical oscillation
KIN_VERT_OSC_MIN_DY = 1.0  # min |dy| before counting direction change
KIN_VERT_OSC_DIR_CHANGES_MIN = 2
KIN_VERT_OSC_VEL_MIN = 1.0
KIN_VERT_OSC_VEL_MAX = 25.0
# Closed loop / return-to-start (figure-8 style)
KIN_LOOP_RATIO_MAX = 0.3
KIN_LOOP_TOTAL_DIST_MIN = 50.0
# Erratic
KIN_ERRATIC_ANGLE_DEG_MIN = 30.0
KIN_ERRATIC_SHARP_TURNS_MIN = 2
KIN_ERRATIC_VEL_MIN = 1.0
KIN_ERRATIC_VEL_MAX = 25.0
# Curved turn
KIN_CURVE_ANGLE_AVG_MIN = 15.0
KIN_CURVE_ANGLE_AVG_MAX = 60.0
KIN_CURVE_ANGLE_MAX_MAX = 90.0
KIN_CURVE_SMOOTHNESS_MAX = 0.85
KIN_CURVE_VEL_MIN = 1.0
KIN_CURVE_VEL_MAX = 25.0
KIN_CURVE_SHARP_ANGLE_MIN = 45.0
KIN_CURVE_VEL_DIFF_MIN = 2.0
# Tail revoke (straight path = bird/plane)
KIN_TAIL_SMOOTHNESS_REVOKE = 0.85
KIN_TAIL_CV_REVOKE = 0.15
KIN_TAIL_LENGTH = 20

# YOLO start threshold from real-world size
DRONE_SIZE_M = 0.35
YOLO_START_DIST_M = 100.0
SHOW_YOLO_START_PX = False
YOLO_START_PX = None
SOUND_CONFIRM_FRAMES = 3

# Distance estimation: default target size per class (meters)
TARGET_SIZE_DRONE_M = 1.10    ##สำหรับปรับขนาดโดรนว่าอยากได้กี่เมตร ถ้าทดสอบอยากให้เห็นในแผนที่ชัดก็อาจจะกำหนดไว้ใหญ่กว่าความเป็นจริงจะได้เห็นโดรนในแผนที่ชัดไม่ทับซ้อนกับกล้อง
TARGET_SIZE_PLANE_M = 30.0
TARGET_SIZE_BIRD_M = 0.4
TARGET_SIZE_OBJ_M = 0.4
DAY_HOUR_START = 6
DAY_HOUR_END = 18
NIGHT_SIZE_FACTOR = 0.25  # at night we often see only lights; use smaller effective size

# Smoothed distance (m): median over recent per-track samples; YOLO-only uses spatial EMA
DISTANCE_HISTORY_MAXLEN = 10
YOLO_WS_DIST_EMA_ALPHA = 0.25
YOLO_WS_DIST_CELL_PX = 64
YOLO_WS_EMA_PRUNE_FRAMES = 45

ACTIVE_LABELING_DEFAULT = False  # press 't' to enable prompt
_CANONICAL_STORAGE_LAYOUT = ensure_storage_layout(os.path.dirname(__file__), CANONICAL_PATH_SCHEMA_VERSION)
# Isolated bundle: train/test/models — ไม่ปนกับ datasets/path/canonical_v2_*
PATH_ML_BUNDLE_ID = "path_ml_v1"
_PATH_ML_BUNDLE = ensure_path_ml_bundle(os.path.dirname(__file__), PATH_ML_BUNDLE_ID)
ACTIVE_LABEL_LEGACY_DATA_DIR = os.path.join(os.path.dirname(__file__), "active_teach")
ACTIVE_LABEL_DATA_DIR = _PATH_ML_BUNDLE["train"]
ACTIVE_LABEL_MODEL_DIR = _PATH_ML_BUNDLE["models"]
ACTIVE_LABEL_MIGRATED_DATA_DIR = _CANONICAL_STORAGE_LAYOUT["datasets"]["legacy_migrated_candidate"]
ACTIVE_LABEL_MODEL_FILENAME = f"path_model_{PATH_ML_BUNDLE_ID}.joblib"
ACTIVE_LABEL_MIN_SAMPLES = 12
ACTIVE_LABEL_CONFIRM_CONF = 0.85
ACTIVE_LABEL_CONFIRM_FRAMES = 3

# Websocket export settings (ดู config.py CAMERA_GEO สำหรับ lat/lng/heading ต่อกล้อง)
WS_EXPORT_ENABLED = True
WS_EXPORT_URL_TEMPLATE = "wss://bbdc-api.thingsanalytic.com/ws/antidrone/{camera_id}/data/send"
WS_EXPORT_INTERVAL_SEC = 0.4
WS_EXPORT_BACKOFF_INIT_SEC = 2.0
WS_EXPORT_BACKOFF_MAX_SEC = 30.0

# Arm cue sender settings: ส่ง confirmed drone target ไปยัง Jetson แขน (22_gun_aim_assist_vector.py)
ARM_CUE_ENABLED = True
ARM_CUE_HOST = "192.168.144.66"   # IP ของ Jetson ที่รัน 22
ARM_CUE_PORT = 5765
ARM_CUE_SEND_HZ =15.0            # ส่งสูงสุด 10 ครั้ง/วินาที
ARM_CUE_TTL_MS = 500              # ฝั่ง 22 ทิ้ง cue เมื่ออายุเกิน 500 ms
ARM_CUE_SOURCE_CAMERA = "cam8"    # ส่ง cue ไปแขนเฉพาะ instance ที่เป็นกล้องนี้เท่านั้น
ARM_REACHABLE_OVERLAY_ENABLED = True
ARM_REACHABLE_OVERLAY_COLOR = (0, 0, 255)   # red (BGR)
ARM_REACHABLE_OVERLAY_ALPHA = 0.22
ARM_REACHABLE_OVERLAY_THICKNESS = 2
ARM_CALIB_SYNC_ENABLED = True
ARM_CALIB_SYNC_REMOTE_HOST = ARM_CUE_HOST
ARM_CALIB_SYNC_REMOTE_USER = os.environ.get("USER", "ta")
ARM_CALIB_SYNC_REMOTE_PATH = "/home/ta/Shoot_gun_joystick_fastTrack_SelectiveCamera12/calibration_data/cam8_mouse_grid_lookup.json"
ARM_CALIB_SYNC_MAX_RETRIES = 10
ARM_CALIB_SYNC_BACKOFF_SEC = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
# CAMERA_GEO ต่อกล้อง — แก้ site_lat / site_lng / heading_deg ให้ตรงกับตำแหน่งและทิศจริงของกล้อง
# site_lat / site_lng : พิกัด GPS ของจุดติดตั้งกล้อง
# heading_deg        : ทิศที่กึ่งกลางภาพชี้ไป (0=เหนือ, 90=ตะวันออก, 180=ใต้, 270=ตะวันตก)
# camera_id          : ชื่อที่ใส่ใน websocket URL เช่น CAM001 → .../antidrone/CAM001/data/send
CAMERA_GEO = {
    # *** ตัวอย่างค่า — แก้ให้ตรงกับพิกัดจริงก่อนใช้งาน ***
    "cam1": {"camera_id": "CAM001", "site_lat": 14.0000, "site_lng": 100.0000, "heading_deg": 0.0},
    "cam2": {"camera_id": "CAM002", "site_lat": 14.0000, "site_lng": 100.0000, "heading_deg": 0.0},
    # cam3 = กล้อง bottom (BOTTOM_CAMERA_NAME) — ใช้จริงตอน BOTTOM_SOURCE_MODE = "single"
    #"cam3": {"camera_id": "CAM001", "site_lat": 14.0000, "site_lng": 102.5870, "heading_deg": 100.0},
    "cam3": {"camera_id": "CAM001", "site_lat": 14.0000, "site_lng": 102.6070, "heading_deg": 100.0},
    # cam4 = กล้อง top (TOP_CAMERA_NAME)
    #"cam4": {"camera_id": "CAM004", "site_lat": 14.0000, "site_lng": 102.5871, "heading_deg": 160.0},
    "cam4": {"camera_id": "CAM004", "site_lat": 14.0000, "site_lng": 102.6071, "heading_deg": 160.0},
    "cam5": {"camera_id": "CAM005", "site_lat": 14.0000, "site_lng": 100.0000, "heading_deg": 0.0},
    "cam6": {"camera_id": "CAM006", "site_lat": 14.0000, "site_lng": 100.0000, "heading_deg": 0.0},
    # cam7 = กล้อง bottom right (BOTTOM_RIGHT_CAMERA) หรือ SINGLE_CAMERA_NAME
    #"cam7": {"camera_id": "CAM007", "site_lat": 14.0000, "site_lng": 102.5900, "heading_deg": 0.0},
    "cam8": {"camera_id": "CAM008", "site_lat": 14.0000, "site_lng": 102.5905, "heading_deg": 0.0},
    "cam9": {"camera_id": "CAM009", "site_lat": 14.0000, "site_lng": 102.5906, "heading_deg": 180.0},
}


def _get_camera_geo(camera_name: str) -> dict:
    default = {"camera_id": camera_name.upper(), "site_lat": 0.0, "site_lng": 0.0, "heading_deg": 0.0}
    return CAMERA_GEO.get(camera_name, default)


# Local camera selection for this file only.
LOCAL_ACTIVE_CAMERA = "cam7"
LOCAL_CAMERAS = {
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
        "use_video_file": True,
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

# Single-file launcher configuration.
RUN_MODE = "multi_camera_top_bottom"  # single_camera | multi_camera_top_bottom
MULTI_CAMERA_VIEW_MODE = "single_window"  # single_window | split_windows
COMPOSITE_WINDOW_NAME = "TopBottom Composite"
# Time (seconds) to wait for worker to exit gracefully (save/retrain path model) before terminate
GRACEFUL_SHUTDOWN_JOIN_TIMEOUT = 60

# Single-camera mode
SINGLE_CAMERA_NAME = "cam7"
SINGLE_CAMERA_VIEW_MODE = "single_window"  # single_window | direct
SINGLE_YOLO_CONF_THRESHOLD = 0.01
SINGLE_YOLO_DRONE_CONF_THRESHOLD = 0.5

# Multi-camera top/bottom mode
TOP_PANEL_HEIGHT_RATIO = 0.5
TOP_CAMERA_NAME = "cam8"  # เปลี่ยนเป็น cam4 ถ้าอยากทดสอบ vdo   ## เปลี่ยนเป็น cam8 ถ้าใช้กล้อง 180 จริง
TOP_YOLO_CONF_THRESHOLD = 0.01
TOP_YOLO_DRONE_CONF_THRESHOLD = 0.5
TOP_HORIZON_FILE = "horizon_poly_cam8_top.npy"
TOP_IGNORE_MOTION_FILE = "motion_ignore_mask_cam8_top.npy"

BOTTOM_SOURCE_MODE = "single"  # single | stitched
BOTTOM_CAMERA_NAME = "cam4"  # เปลี่ยนเป็น cam3 ถ้าอยากทดสอบ vdo  ## เปลี่ยนเป็น cam9 ถ้าใช้กล้อง 180 จริง
BOTTOM_LEFT_CAMERA = "cam3"
BOTTOM_RIGHT_CAMERA = "cam7"
BOTTOM_CLONE_RIGHT_FROM_LEFT = False
BOTTOM_STITCH_PRESET = "bottom_stitch.jetson"
BOTTOM_YOLO_CONF_THRESHOLD = 0.01
BOTTOM_YOLO_DRONE_CONF_THRESHOLD = 0.5
BOTTOM_HORIZON_FILE = "horizon_poly_cam9_bottom.npy"
BOTTOM_IGNORE_MOTION_FILE = "motion_ignore_mask_cam9_bottom.npy"


def get_camera_config(camera_name=None):
    name = camera_name or LOCAL_ACTIVE_CAMERA
    if name not in LOCAL_CAMERAS:
        raise ValueError(
            f"Camera '{name}' not found in LOCAL_CAMERAS. Available: {list(LOCAL_CAMERAS.keys())}"
        )
    return LOCAL_CAMERAS[name]


def get_fov_for_canvas_point(cx, cy, cam_config, resolver=None):
    """Return (fov_h_deg, fov_v_deg, src_frame_w, src_frame_h) for a canvas pixel.

    For stitched cameras the synthetic config carries a ``fov_by_source`` dict so
    we can look up the actual FOV of the physical camera that owns the pixel.
    For single cameras the global fov_horizontal / fov_vertical are used directly.
    """
    if resolver is not None:
        try:
            res = resolver(cx, cy)
            if res is not None:
                region = res.get("resolved_region", "left")
                fov_map = cam_config.get("fov_by_source", {})
                fov = fov_map.get(region)
                if fov:
                    return (
                        float(fov["h"]),
                        float(fov["v"]),
                        res.get("source_frame_w"),
                        res.get("source_frame_h"),
                    )
        except Exception:
            pass
    return (
        float(cam_config.get("fov_horizontal", 60.0)),
        float(cam_config.get("fov_vertical", 36.0)),
        None,
        None,
    )


def _get_screen_size():
    """ได้ (width, height) ของจอหลัก (พิกเซล) สำหรับ fullscreen fit"""
    try:
        from tkinter import Tk
        r = Tk()
        r.withdraw()
        sw = r.winfo_screenwidth()
        sh = r.winfo_screenheight()
        r.destroy()
        return int(sw), int(sh)
    except Exception:
        return 1920, 1080


# =============================================================================
# 1. GLOBAL VARIABLES & MOUSE CALLBACK
# =============================================================================
horizon_points = []
temp_draw_points = []
drawing_mode = False
drawing_target = None
current_frame_w = 1280
current_frame_h = 720
display_w = 1280
display_h = 720
label_trainer_global = None
last_raw_frame = None
prev_raw_frame = None
ignore_motion_preview_mask = None
ignore_motion_brush_thickness = 18
ignore_motion_last_point = None
ignore_motion_is_painting = False
ignore_motion_is_erasing = False
# สำหรับ compositor: session/current drone count ต่อโซน (worker อัปเดตทุกเฟรม)
DRONE_HUD_COUNTS = {"session": 0, "current": 0}

def on_mouse_global(event, x, y, flags, userdata):
    global temp_draw_points, drawing_mode, drawing_target, current_frame_w, current_frame_h, display_w, display_h
    global label_trainer_global
    global last_raw_frame, prev_raw_frame
    global ignore_motion_preview_mask, ignore_motion_brush_thickness
    global ignore_motion_last_point, ignore_motion_is_painting, ignore_motion_is_erasing

    scale_x = current_frame_w / display_w if display_w > 0 else 1
    scale_y = current_frame_h / display_h if display_h > 0 else 1
    real_x = int(x * scale_x)
    real_y = int(y * scale_y)

    if not drawing_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if label_trainer_global is not None and last_raw_frame is not None:
                label_trainer_global.handle_mouse_click(real_x, real_y, last_raw_frame, prev_raw_frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if label_trainer_global is not None:
                label_trainer_global.handle_mouse_skip()
        return

    if drawing_target == "ignore_motion":
        if ignore_motion_preview_mask is None:
            return

        h_mask, w_mask = ignore_motion_preview_mask.shape[:2]
        px = int(np.clip(real_x, 0, w_mask - 1))
        py = int(np.clip(real_y, 0, h_mask - 1))
        brush = max(1, int(ignore_motion_brush_thickness))

        if event == cv2.EVENT_LBUTTONDOWN:
            ignore_motion_is_painting = True
            ignore_motion_is_erasing = False
            ignore_motion_last_point = (px, py)
            cv2.circle(ignore_motion_preview_mask, (px, py), max(1, brush // 2), 255, -1, lineType=cv2.LINE_AA)
        elif event == cv2.EVENT_RBUTTONDOWN:
            ignore_motion_is_erasing = True
            ignore_motion_is_painting = False
            ignore_motion_last_point = (px, py)
            cv2.circle(ignore_motion_preview_mask, (px, py), max(1, brush // 2), 0, -1, lineType=cv2.LINE_AA)
        elif event == cv2.EVENT_MOUSEMOVE:
            if (flags & cv2.EVENT_FLAG_LBUTTON) and ignore_motion_is_painting:
                if ignore_motion_last_point is None:
                    ignore_motion_last_point = (px, py)
                cv2.line(ignore_motion_preview_mask, ignore_motion_last_point, (px, py), 255, brush, lineType=cv2.LINE_AA)
                cv2.circle(ignore_motion_preview_mask, (px, py), max(1, brush // 2), 255, -1, lineType=cv2.LINE_AA)
                ignore_motion_last_point = (px, py)
            elif (flags & cv2.EVENT_FLAG_RBUTTON) and ignore_motion_is_erasing:
                if ignore_motion_last_point is None:
                    ignore_motion_last_point = (px, py)
                cv2.line(ignore_motion_preview_mask, ignore_motion_last_point, (px, py), 0, brush, lineType=cv2.LINE_AA)
                cv2.circle(ignore_motion_preview_mask, (px, py), max(1, brush // 2), 0, -1, lineType=cv2.LINE_AA)
                ignore_motion_last_point = (px, py)
        elif event == cv2.EVENT_LBUTTONUP:
            ignore_motion_is_painting = False
            ignore_motion_last_point = None
        elif event == cv2.EVENT_RBUTTONUP:
            ignore_motion_is_erasing = False
            ignore_motion_last_point = None
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        temp_draw_points.append((real_x, real_y))
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if len(temp_draw_points) > 0:
            last = temp_draw_points[-1]
            if abs(real_x - last[0]) + abs(real_y - last[1]) > 5:
                temp_draw_points.append((real_x, real_y))
        else:
            temp_draw_points.append((real_x, real_y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(temp_draw_points) > 0: del temp_draw_points[-10:]

# =============================================================================
# 2. YOLO HELPER FUNCTIONS
# =============================================================================

def compute_yolo_start_px(frame_w, frame_h, cx=None, cy=None, resolver=None):
    """
    คำนวณขนาดพิกเซลของวัตถุจริง (DRONE_SIZE_M) ที่ระยะ YOLO_START_DIST_M
    เพื่อเป็นเกณฑ์เริ่มใช้ YOLO

    cx, cy and resolver are optional; when provided, the FOV is resolved
    per-pixel using get_fov_for_canvas_point so stitched cameras use the
    correct source-camera FOV instead of the left-camera FOV for all pixels.
    """
    try:
        cam_config = get_camera_config()
        if cx is not None and cy is not None:
            fov_h, fov_v, src_w, src_h = get_fov_for_canvas_point(cx, cy, cam_config, resolver=resolver)
            if src_w and src_h:
                frame_w, frame_h = int(src_w), int(src_h)
        else:
            fov_h = float(cam_config.get("fov_horizontal", 60.0))
            fov_v = float(cam_config.get("fov_vertical", 36.0))
    except Exception:
        fov_h = 60.0
        fov_v = 36.0

    if frame_w <= 0 or frame_h <= 0:
        return None

    fov_h_rad = math.radians(fov_h)
    fov_v_rad = math.radians(fov_v)
    if fov_h_rad <= 0 or fov_v_rad <= 0:
        return None

    angle = 2.0 * math.atan(DRONE_SIZE_M / (2.0 * YOLO_START_DIST_M))
    px_w = (angle / fov_h_rad) * frame_w
    px_h = (angle / fov_v_rad) * frame_h
    return max(px_w, px_h)

def is_daytime():
    """True if current hour is within DAY_HOUR_START..DAY_HOUR_END (day)."""
    hour = datetime.datetime.now().hour
    return DAY_HOUR_START <= hour < DAY_HOUR_END

def get_target_size_m(display_cls, is_night=False):
    """Return target size in meters for distance estimation by class. At night apply NIGHT_SIZE_FACTOR."""
    cls_lower = (display_cls or "obj").lower().strip()
    if cls_lower == "drone":
        size_m = TARGET_SIZE_DRONE_M
    elif cls_lower == "plane" or cls_lower == "aircraft":
        size_m = TARGET_SIZE_PLANE_M
    elif cls_lower == "bird":
        size_m = TARGET_SIZE_BIRD_M
    else:
        size_m = TARGET_SIZE_OBJ_M
    if is_night:
        size_m = size_m * NIGHT_SIZE_FACTOR
    return size_m

def estimate_distance_m(w_px, h_px, frame_w, frame_h, fov_h_deg, fov_v_deg, size_m):
    """
    คำนวณระยะถึงเป้า (เมตร) คร่าวๆ จากขนาด bbox + FOV + ขนาดจริงวัตถุ (เมตร).
    คืน None ถ้า bbox เล็กเกินไปหรือ FOV ไม่ถูกต้อง.
    """
    if w_px < 3 or h_px < 3 or frame_w <= 0 or frame_h <= 0:
        return None
    if fov_h_deg <= 0 or fov_v_deg <= 0:
        return None
    deg2rad = math.pi / 180.0
    fov_h_rad = fov_h_deg * deg2rad
    fov_v_rad = fov_v_deg * deg2rad
    theta_h = (w_px / frame_w) * fov_h_rad
    theta_v = (h_px / frame_h) * fov_v_rad
    if theta_h < 1e-6 and theta_v < 1e-6:
        return None
    dist_h = size_m / (2.0 * math.tan(theta_h / 2.0)) if theta_h >= 1e-6 else None
    dist_v = size_m / (2.0 * math.tan(theta_v / 2.0)) if theta_v >= 1e-6 else None
    if dist_h is not None and dist_v is not None:
        return (dist_h + dist_v) / 2.0
    return dist_h if dist_h is not None else dist_v


def update_path_distance_smoothed(path_data, instant_m):
    """เก็บ instant_m ใน distance_history แล้วตั้ง path_data['distance_m'] เป็น median ของช่วงล่าสุด."""
    if path_data is None:
        return None
    hist = path_data.get("distance_history")
    if hist is None:
        if instant_m is not None and instant_m > 0:
            path_data["distance_m"] = float(instant_m)
        return path_data.get("distance_m")
    if instant_m is not None and instant_m > 0:
        hist.append(float(instant_m))
    if not hist:
        return path_data.get("distance_m")
    vals = sorted(hist)
    n = len(vals)
    mid = n // 2
    med = float(vals[mid]) if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    path_data["distance_m"] = med
    return med


def get_grid_cell_from_point(x, y, grid_filter):
    """
    หา grid cell index จาก point (x, y)
    Returns: (row_idx, col_idx) หรือ None
    """
    c_idx = min(max(x // grid_filter.cell_w, 0), grid_filter.cols - 1)
    r_idx = min(max(y // grid_filter.cell_h, 0), grid_filter.rows - 1)
    return (r_idx, c_idx)

def group_nearby_grid_cells(grid_cells, max_distance=1):
    """
    Group grid cells ที่อยู่ใกล้กัน (distance <= max_distance)

    Args:
        grid_cells: Set of (row, col) grid cells
        max_distance: ระยะห่างสูงสุดที่ถือว่าใกล้กัน (default=1)

    Returns:
        List of groups, แต่ละ group เป็น set ของ (row, col)
    """
    if not grid_cells:
        return []

    groups = []
    used = set()

    for cell in grid_cells:
        if cell in used:
            continue

        # สร้าง group ใหม่
        group = {cell}
        used.add(cell)

        # หา cells ที่อยู่ใกล้กัน (BFS)
        queue = [cell]
        while queue:
            r, c = queue.pop(0)
            for dr in range(-max_distance, max_distance + 1):
                for dc in range(-max_distance, max_distance + 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in grid_cells and neighbor not in used:
                        group.add(neighbor)
                        used.add(neighbor)
                        queue.append(neighbor)

        groups.append(group)

    return groups

def calculate_roi_from_grid_cells(grid_cells_group, grid_filter):
    """
    คำนวณ ROI box จาก grid cells group (ใช้แค่ grid พื้นที่เท่านั้น ไม่ padding)

    Args:
        grid_cells_group: Set of (row, col) grid cells
        grid_filter: AdaptiveSensitivityGrid instance

    Returns:
        ROI box (x1, y1, x2, y2) หรือ None
    """
    if not grid_cells_group:
        return None

    # หา min/max grid indices
    rows = [r for r, c in grid_cells_group]
    cols = [c for r, c in grid_cells_group]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # แปลง grid indices เป็น pixel coordinates (ไม่ padding)
    roi_x1 = min_c * grid_filter.cell_w
    roi_y1 = min_r * grid_filter.cell_h
    roi_x2 = (max_c + 1) * grid_filter.cell_w
    roi_y2 = (max_r + 1) * grid_filter.cell_h

    # จำกัดให้อยู่ในขอบเขตเฟรม
    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)
    roi_x2 = min(grid_filter.w, roi_x2)
    roi_y2 = min(grid_filter.h, roi_y2)

    return (roi_x1, roi_y1, roi_x2, roi_y2)

def detect_yolo_in_roi(yolo_model, frame, roi_box, conf_threshold):
    """
    ทำ YOLO detection ใน ROI

    Args:
        yolo_model: YOLO model instance
        frame: Full frame image
        roi_box: (x1, y1, x2, y2) ROI box
        conf_threshold: Confidence threshold

    Returns:
        List of detections: [(x, y, w, h, conf), ...]
    """
    if yolo_model is None or roi_box is None:
        return []

    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    try:
        results = yolo_model(
            roi,
            conf=conf_threshold,
            imgsz=YOLO_INPUT_SIZE,
            verbose=False,
            device=0,
            max_det=10,
            half=True,
        )
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    x1_roi, y1_roi, x2_roi, y2_roi = xyxy[i]
                    c = float(conf_arr[i])
                    x = int(x1_roi) + x1
                    y = int(y1_roi) + y1
                    w = int(x2_roi - x1_roi)
                    h = int(y2_roi - y1_roi)
                    detections.append((x, y, w, h, c))
        return detections
    except Exception as e:
        return []

def select_yolo_model_path(base_dir):
    if YOLO_ENGINE_SIZE == "640":
        candidates = [YOLO_MODEL_FILE_640]
    elif YOLO_ENGINE_SIZE == "1280":
        candidates = [YOLO_MODEL_FILE_1280]
    else:
        candidates = [YOLO_MODEL_FILE_1280, YOLO_MODEL_FILE_640]

    for fname in candidates:
        if fname:
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                return path, fname
    return None, candidates[0] if candidates else None

def detect_yolo_full_frame(yolo_model, frame, conf_threshold):
    """
    ทำ YOLO detection บนทั้งเฟรม
    Returns:
        List of detections: [(x, y, w, h, conf), ...]
    """
    if yolo_model is None:
        return []
    try:
        results = yolo_model(
            frame,
            conf=conf_threshold,
            imgsz=YOLO_INPUT_SIZE,
            verbose=False,
            device=0,
            max_det=50,
            half=True,
        )
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]
                    c = float(conf_arr[i])
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    detections.append((x, y, w, h, c))
        return detections
    except Exception:
        return []


def should_use_wide_split(camera_name, frame_w):
    if not WIDE_SPLIT_ENABLED or frame_w <= 0:
        return False
    return frame_w >= WIDE_SPLIT_MIN_WIDTH


def detect_yolo_wide_split(yolo_model, frame, conf_threshold, overlap_px=WIDE_SPLIT_OVERLAP_PX):
    """
    ทำ YOLO detection แบบแบ่งครึ่งซ้าย/ขวาสำหรับเฟรมกว้างมาก
    และ remap bbox กลับมาอยู่ในพิกัดของ full frame
    """
    if yolo_model is None or frame is None:
        return []

    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 1 or frame_h <= 1:
        return []

    overlap_px = max(0, int(overlap_px))
    mid_x = frame_w // 2
    left_roi = (0, 0, min(frame_w, mid_x + overlap_px), frame_h)
    right_roi = (max(0, mid_x - overlap_px), 0, frame_w, frame_h)

    detections = []
    detections.extend(detect_yolo_in_roi(yolo_model, frame, left_roi, conf_threshold))
    detections.extend(detect_yolo_in_roi(yolo_model, frame, right_roi, conf_threshold))
    return detections

def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def _intersection_area_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _bbox_iomin_xyxy(a, b):
    """intersection / min(area_a, area_b) — high when smaller box lies inside larger."""
    inter = _intersection_area_xyxy(a, b)
    if inter <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    m = min(area_a, area_b)
    return float(inter) / float(m) if m > 0 else 0.0


def yolo_bbox_matches_tracked_rect(yolo_x, yolo_y, yolo_w, yolo_h, rect,
                                   iou_thresh=None, iomin_thresh=None):
    """
    True if a YOLO box and tracker/path rect (x,y,w,h) are the same object for UI/counting.
    Uses IoU, IoMin, and mutual center-in-box so overlapping boxes are not shown/counted twice.
    """
    if iou_thresh is None:
        iou_thresh = YOLO_PATH_MATCH_IOU
    if iomin_thresh is None:
        iomin_thresh = YOLO_PATH_MATCH_IOMIN
    rx, ry, rw, rh = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    if rw <= 0 or rh <= 0 or yolo_w <= 0 or yolo_h <= 0:
        return False
    y_xyxy = (yolo_x, yolo_y, yolo_x + yolo_w, yolo_y + yolo_h)
    r_xyxy = (rx, ry, rx + rw, ry + rh)
    if _bbox_iou_xyxy(y_xyxy, r_xyxy) >= iou_thresh:
        return True
    if _bbox_iomin_xyxy(y_xyxy, r_xyxy) >= iomin_thresh:
        return True
    y_cx = yolo_x + yolo_w // 2
    y_cy = yolo_y + yolo_h // 2
    if rx <= y_cx <= rx + rw and ry <= y_cy <= ry + rh:
        return True
    r_cx = rx + rw // 2
    r_cy = ry + rh // 2
    if yolo_x <= r_cx <= yolo_x + yolo_w and yolo_y <= r_cy <= yolo_y + yolo_h:
        return True
    return False


def _dist_point_to_segment_sq(px, py, ax, ay, bx, by):
    """Squared distance from point P to segment AB."""
    abx = bx - ax
    aby = by - ay
    if abx == 0.0 and aby == 0.0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / float(abx * abx + aby * aby)
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return dx * dx + dy * dy


def _min_dist_point_to_polyline(px, py, points):
    """ระยะขั้นต่ำจากจุดไปยัง polyline (ลำดับจุดบน path)."""
    if not points or len(points) < 2:
        return float("inf")
    best_sq = float("inf")
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        ax, ay = float(p0[0]), float(p0[1])
        bx, by = float(p1[0]), float(p1[1])
        d2 = _dist_point_to_segment_sq(px, py, ax, ay, bx, by)
        if d2 < best_sq:
            best_sq = d2
    return math.sqrt(best_sq)


def _yolo_path_polyline_max_dist(frame_min_side):
    ms = int(frame_min_side) if frame_min_side and frame_min_side > 0 else 1080
    d = int(YOLO_PATH_POLYLINE_FRAC * ms)
    return float(max(YOLO_PATH_POLYLINE_MIN_PX, min(YOLO_PATH_POLYLINE_MAX_PX, d)))


def yolo_matches_validated_path(yolo_x, yolo_y, yolo_w, yolo_h, rect, path_points=None, frame_min_side=None):
    """
    YOLO กับ validated path เป็นตัวเดียวกันถ้า bbox match ตามเดิม หรือจุดกลาง YOLO อยู่ใกล้เส้น path
    (กรณีเฟรมกระตุก / กล่องไม่ซ้อนแต่ยังบินตาม path เดิม).
    """
    if yolo_bbox_matches_tracked_rect(yolo_x, yolo_y, yolo_w, yolo_h, rect):
        return True
    if not path_points or len(path_points) < 2:
        return False
    y_cx = float(yolo_x) + float(yolo_w) * 0.5
    y_cy = float(yolo_y) + float(yolo_h) * 0.5
    dist = _min_dist_point_to_polyline(y_cx, y_cy, list(path_points))
    return dist <= _yolo_path_polyline_max_dist(frame_min_side)


def merge_motion_and_yolo_for_tracker(valid_dets, yolo_rects, iou_threshold=None, max_center_dist=None,
                                      iomin_threshold=None):
    """
    รวม motion กับ YOLO ที่เป็นวัตถุเดียวกันให้เหลือ 1 rect ก่อนส่งเข้า tracker.
    คู่ที่ match: ใช้เฉพาะ yolo_rect; motion ที่ match แล้วไม่ใส่. motion ที่ไม่ match ใส่ตามเดิม.
    """
    if iou_threshold is None:
        iou_threshold = MOTION_YOLO_IOU_THRESH
    if max_center_dist is None:
        max_center_dist = MOTION_YOLO_MAX_CENTER_DIST
    if iomin_threshold is None:
        iomin_threshold = MOTION_YOLO_IOMIN_THRESH
    if not yolo_rects:
        return list(valid_dets)
    used_motion = [False] * len(valid_dets)
    result = []
    for yolo_rect in yolo_rects:
        yx, yy, yw, yh, _ = yolo_rect
        y_xyxy = (yx, yy, yx + yw, yy + yh)
        y_cx = yx + yw // 2
        y_cy = yy + yh // 2
        best_i = -1
        best_score = -1.0
        for i, m in enumerate(valid_dets):
            if used_motion[i]:
                continue
            mx, my, mw, mh, _ = m
            m_xyxy = (mx, my, mx + mw, my + mh)
            iou = _bbox_iou_xyxy(m_xyxy, y_xyxy)
            iomin = _bbox_iomin_xyxy(m_xyxy, y_xyxy)
            m_cx = mx + mw // 2
            m_cy = my + mh // 2
            dist = math.sqrt((y_cx - m_cx)**2 + (y_cy - m_cy)**2)
            if iou >= iou_threshold:
                score = iou
            elif iomin >= iomin_threshold:
                score = iomin * 0.98
            elif dist <= max_center_dist:
                score = max(0.0, 1.0 - dist / max_center_dist)
            else:
                score = -1.0
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            used_motion[best_i] = True
        result.append(yolo_rect)
    for i, m in enumerate(valid_dets):
        if not used_motion[i]:
            result.append(m)
    return result


def _best_yolo_conf_for_det_rect(det, yolo_dets):
    """Max YOLO conf among boxes that overlap det (IoU) or contain det center."""
    if not yolo_dets:
        return 0.0
    x, y, bw, bh, _ = det
    r_xyxy = (float(x), float(y), float(x + bw), float(y + bh))
    cx, cy = int(x + bw / 2.0), int(y + bh / 2.0)
    best = 0.0
    for yx, yy, yw, yh, yc in yolo_dets:
        y_xyxy = (float(yx), float(yy), float(yx + yw), float(yy + yh))
        if _bbox_iou_xyxy(y_xyxy, r_xyxy) >= 0.08:
            best = max(best, float(yc))
        elif yx <= cx < yx + yw and yy <= cy < yy + yh:
            best = max(best, float(yc))
    return best


def select_topk_tracker_dets(tracker, all_dets, yolo_dets, frame_w, frame_h, topk=None):
    """
    เมื่อมี detection มากกว่า topk: เก็บเฉพาะ top-K ตามคะแนน
    score = AREA_WEIGHT*log1p(area) + CONF_WEIGHT*yolo_conf + EXISTING_BONUS ถ้าใกล้จุดกลาง track เดิม
    (ลดโหลดเมื่อฟ้าเต็มนก; โบนัส track เดิมลดโอกาสเตะเป้าที่กำลังติดตาม)
    """
    if topk is None:
        topk = TRACKER_INPUT_TOPK
    if not all_dets or len(all_dets) <= topk:
        return list(all_dets)
    min_side = min(frame_w, frame_h)
    match_dist = max(TRACKER_TOPK_MATCH_DIST_MIN, int(TRACKER_TOPK_MATCH_DIST_FRAC * min_side))
    existing_centers = []
    if tracker is not None and getattr(tracker, "boxes", None):
        for _oid, box in tracker.boxes.items():
            ox, oy, ow, oh, _ = box
            existing_centers.append((ox + ow // 2, oy + oh // 2))

    scored = []
    y_list = yolo_dets if yolo_dets else []
    for det in all_dets:
        x, y, bw, bh, _ = det
        cx = int(x + bw / 2.0)
        cy = int(y + bh / 2.0)
        area = max(1, bw * bh)
        yconf = _best_yolo_conf_for_det_rect(det, y_list)
        score = TRACKER_TOPK_AREA_WEIGHT * math.log1p(area) + TRACKER_TOPK_CONF_WEIGHT * yconf
        if existing_centers:
            dmin = min(math.hypot(cx - ex, cy - ey) for ex, ey in existing_centers)
            if dmin <= match_dist:
                score += TRACKER_TOPK_EXISTING_BONUS
        scored.append((score, det))

    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:topk]]


def merge_overlapping_bboxes(detections, iou_threshold=0.5):
    if not detections or len(detections) <= 1:
        return detections
    merged = list(detections)
    changed = True
    while changed:
        changed = False
        new = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x, y, w, h, conf = merged[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            max_conf = conf
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                xj, yj, wj, hj, confj = merged[j]
                bx1, by1, bx2, by2 = xj, yj, xj + wj, yj + hj
                if _bbox_iou_xyxy((x1, y1, x2, y2), (bx1, by1, bx2, by2)) >= iou_threshold:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    max_conf = max(max_conf, confj)
                    used[j] = True
                    changed = True
            new.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(max_conf)))
        merged = new
    return merged

def nms_bboxes(detections, iou_threshold=0.5):
    """
    Non-Maximum Suppression: เก็บเฉพาะกล่องที่ confidence สูงสุดในกลุ่มที่ซ้อนกัน
    ทำให้ไม่มีสองกล่องใดซ้อนกัน (IOU >= threshold) ในผลลัพธ์
    """
    if not detections or len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    for det in sorted_dets:
        x, y, w, h, conf = det
        x1, y1, x2, y2 = x, y, x + w, y + h
        overlap = False
        for kept in keep:
            kx, ky, kw, kh, _ = kept
            kx1, ky1, kx2, ky2 = kx, ky, kx + kw, ky + kh
            if _bbox_iou_xyxy((x1, y1, x2, y2), (kx1, ky1, kx2, ky2)) >= iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(det)
    return keep

def get_cells_in_radius(center_cell, radius, grid_filter):
    if center_cell is None or grid_filter is None:
        return set()
    r0, c0 = center_cell
    cells = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r = r0 + dr
            c = c0 + dc
            if 0 <= r < grid_filter.rows and 0 <= c < grid_filter.cols:
                cells.add((r, c))
    return cells

def get_grid_cells_for_bbox(bbox, grid_filter, radius=0):
    if grid_filter is None:
        return set()
    x, y, w, h, _ = bbox
    if w <= 0 or h <= 0:
        return set()
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(grid_filter.w - 1, x + w - 1)
    y2 = min(grid_filter.h - 1, y + h - 1)
    c1 = max(int(x1 // grid_filter.cell_w), 0)
    c2 = min(int(x2 // grid_filter.cell_w), grid_filter.cols - 1)
    r1 = max(int(y1 // grid_filter.cell_h), 0)
    r2 = min(int(y2 // grid_filter.cell_h), grid_filter.rows - 1)
    cells = set()
    for r in range(r1 - radius, r2 + radius + 1):
        for c in range(c1 - radius, c2 + radius + 1):
            if 0 <= r < grid_filter.rows and 0 <= c < grid_filter.cols:
                cells.add((r, c))
    return cells

# =============================================================================
# 3. SETUP & IMPORTS
# =============================================================================
try:
    from fast_motion_sky import CameraStream
    from memory_manager import init_memory_management
    try:
        from skydroid_viewer import SkydroidCameraStream
    except ImportError:
        SkydroidCameraStream = None
except ImportError:
    print("⚠️ Custom modules not found. Using standalone mock classes.")
    SkydroidCameraStream = None
    SHOW_PERSISTENCE_PATHS = False

    class CameraStream:
        def __init__(self, source=None, width=None, height=None, use_video_file=None, camera_name=None,
                     udp_ip=None, udp_port=None, use_udp_direct=None, stream_format=None):
            self.source = source
        def start(self):
            source = 0 if self.source is None else self.source
            self.cap = cv2.VideoCapture(source)
            return self
        def read(self):
            ret, frame = self.cap.read()
            return ret, frame, None
        def release(self): self.cap.release()

    def init_memory_management(): pass
else:
    SHOW_PERSISTENCE_PATHS = False


def build_camera(camera_name=None):
    cfg = get_camera_config(camera_name)
    source = cfg["video_filename"] if cfg.get("use_video_file", False) else cfg["rtsp_url"]
    if cfg.get("use_video_file", False) and isinstance(source, str) and source and not os.path.isabs(source):
        source = os.path.join(os.path.dirname(__file__), source)

    return CameraStream(
        source=source,
        width=cfg["width"],
        height=cfg["height"],
        use_video_file=cfg.get("use_video_file", False),
        camera_name=cfg.get("name", camera_name or LOCAL_ACTIVE_CAMERA),
        udp_ip=cfg.get("udp_ip"),
        udp_port=cfg.get("udp_port"),
        use_udp_direct=cfg.get("use_udp_direct", False),
        stream_format=cfg.get("stream_format", "h264"),
    )

# =============================================================================
# 2.5. SOUND ALERT SYSTEM (Parallel Thread)
# =============================================================================
# Sound library check
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

class SoundAlert:
    """ระบบเสียงเตือนแบบ parallel thread - ไม่กระทบความเร็วโปรแกรม"""
    def __init__(self, sound_file="alarm_loud.wav"):
        self.sound_file = sound_file
        self.is_playing = False
        self.should_play = False
        self.last_red_detected_time = 0.0
        self.thread = None
        self.lock = threading.Lock()
        self.running = True

        # Initialize sound library
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                sound_path = os.path.join(os.path.dirname(__file__), sound_file)
                if os.path.exists(sound_path):
                    self.sound = pygame.mixer.Sound(sound_path)
                    self.sound_method = 'pygame'
                    print(f"✅ Sound Alert: Using pygame (file: {sound_path})")
                else:
                    print(f"⚠️ Sound Alert: File not found: {sound_path}")
                    self.sound_method = None
            except Exception as e:
                print(f"⚠️ Sound Alert: pygame init failed: {e}")
                self.sound_method = None
        elif PLAYSOUND_AVAILABLE:
            sound_path = os.path.join(os.path.dirname(__file__), sound_file)
            if os.path.exists(sound_path):
                self.sound_method = 'playsound'
                self.sound_file = sound_path
                print(f"✅ Sound Alert: Using playsound (file: {sound_path})")
            else:
                print(f"⚠️ Sound Alert: File not found: {sound_path}")
                self.sound_method = None
        else:
            self.sound_method = None
            print("⚠️ Sound Alert: No sound library available (install pygame or playsound)")

    def update(self, has_red_drone):
        """อัปเดตสถานะ - มี path สีแดง (kinematic_confirmed) หรือไม่"""
        with self.lock:
            if has_red_drone:
                self.should_play = True
                self.last_red_detected_time = time.time()
            else:
                # หยุดเสียงหลังจาก path สีแดงหายไป 0.1 วินาที
                if time.time() - self.last_red_detected_time > 0.1:
                    self.should_play = False

    def _play_sound_loop(self):
        """เล่นเสียงวนลูป (ใน thread แยก)"""
        if self.sound_method == 'pygame':
            while self.running:
                with self.lock:
                    should_play = self.should_play

                if should_play and not self.is_playing:
                    try:
                        self.sound.play(loops=-1)  # เล่นวนลูป
                        self.is_playing = True
                    except Exception as e:
                        print(f"⚠️ Sound Alert: Play error: {e}")
                        self.is_playing = False
                elif not should_play and self.is_playing:
                    try:
                        self.sound.stop()
                        self.is_playing = False
                    except Exception as e:
                        print(f"⚠️ Sound Alert: Stop error: {e}")
                        self.is_playing = False

                time.sleep(0.1)  # ตรวจสอบทุก 0.1 วินาที

        elif self.sound_method == 'playsound':
            # playsound ไม่รองรับ loop - ต้องเล่นซ้ำเอง
            while self.running:
                with self.lock:
                    should_play = self.should_play

                if should_play:
                    try:
                        playsound(self.sound_file, block=False)
                    except Exception as e:
                        print(f"⚠️ Sound Alert: Play error: {e}")
                    time.sleep(1.0)  # เล่นทุก 1 วินาที
                else:
                    time.sleep(0.1)

    def start(self):
        """เริ่ม thread เล่นเสียง"""
        if self.sound_method is None:
            return

        self.running = True
        self.thread = threading.Thread(target=self._play_sound_loop, daemon=True)
        self.thread.start()
        print("🔊 Sound Alert Thread Started")

    def stop(self):
        """หยุด thread เล่นเสียง"""
        self.running = False
        if self.sound_method == 'pygame' and self.is_playing:
            try:
                self.sound.stop()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=1.0)
        print("🔇 Sound Alert Thread Stopped")

# =============================================================================
# 3. MODULE: Graph Trajectory Manager (Smoothness Validator)
# =============================================================================
class GraphTrajectoryManager:
    def __init__(self, max_nodes=30, min_dist=5, enable_kinematic_rules=True, smooth_window=5):
        # paths เก็บ dict: {'points': deque, 'validated': bool, 'sky_origin': bool}
        self.paths = {}
        self.max_nodes = max_nodes
        self.min_dist = min_dist
        self.smooth_window = int(smooth_window)
        self.persistence_paths = {}  # สำหรับเก็บ paths ที่หายไปแต่ยังคง persist
        self.persistence_kalman_filters = {}  # เก็บ Kalman filter สำหรับ persistence paths
        self.MAX_PATHS = 15  # 🔥 จำกัดจำนวน path ไม่เกิน 15 วัตถุ
        self.enable_kinematic_rules = enable_kinematic_rules

    def _is_window_smooth(self, points):
        """
        เช็คความ Smooth ของชุดจุด 5 จุด
        หลักการ: คำนวณมุม (Cosine Similarity) ของเวกเตอร์แต่ละช่วง
        """
        if len(points) < 5: return False

        vectors = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            mag = math.sqrt(vx**2 + vy**2)
            if mag == 0: return False # จุดซ้ำ ไม่นับ
            vectors.append((vx/mag, vy/mag)) # Normalize vector

        # เช็คการเปลี่ยนทิศทาง (Dot Product)
        # ถ้า Dot Product > 0.5 แปลว่ามุม < 60 องศา (ถือว่า Smooth)
        # ถ้า Dot Product < 0 (ติดลบ) แปลว่าหักศอกหรือย้อนกลับ (Zigzag)
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i+1]
            dot_prod = v1[0]*v2[0] + v1[1]*v2[1]

            if dot_prod < 0.55:  # เข้มขึ้นเล็กน้อย (เดิม 0.5) — ลด validate ง่ายเกินจาก path สั้น
                return False

        return True

    def _apply_temporal_smoothing(self, points, areas, window_size=5):
        """
        ใช้ Moving Average สำหรับ smoothing พิกัด (x, y) และพื้นที่ (Area)
        เพื่อลดปัญหาพิกเซลสั่นในระยะไกล (Quantization Noise)

        Args:
            points: deque ของ (x, y) tuples
            areas: deque ของ area values
            window_size: ขนาด window สำหรับ moving average (caller passes PATH_SMOOTH_WINDOW)

        Returns:
            (smoothed_points, smoothed_areas): tuple ของ smoothed data
        """
        if len(points) < 2:
            return list(points), list(areas)

        # แปลงเป็น numpy array เพื่อประสิทธิภาพ
        points_arr = np.array(points, dtype=np.float32)
        areas_arr = np.array(areas, dtype=np.float32)

        # ใช้ window_size ที่ไม่เกินจำนวนจุดที่มี
        actual_window = min(window_size, len(points))

        smoothed_points = []
        smoothed_areas = []

        for i in range(len(points)):
            # คำนวณ window bounds
            start_idx = max(0, i - actual_window + 1)
            end_idx = i + 1

            # Moving average สำหรับ x, y
            window_points = points_arr[start_idx:end_idx]
            avg_x = np.mean(window_points[:, 0])
            avg_y = np.mean(window_points[:, 1])
            smoothed_points.append((int(avg_x), int(avg_y)))

            # Moving average สำหรับ area
            window_areas = areas_arr[start_idx:end_idx]
            avg_area = np.mean(window_areas)
            smoothed_areas.append(float(avg_area))

        return smoothed_points, smoothed_areas

    def _calculate_kinematics(self, obj_id):
        """
        คำนวณ kinematic features: velocity, acceleration, area variance, smoothness index
        ใช้ smoothed_points และ smoothed_areas เพื่อลด noise

        Args:
            obj_id: object ID

        Returns:
            dict: {
                'velocity': (vx, vy) in pixels/frame,
                'acceleration': (ax, ay) in pixels/frame²,
                'area_variance': float (std/mean ratio),
                'smoothness_index': float (0-1, higher = smoother),
                'velocity_magnitude': float
            }
        """
        if obj_id not in self.paths:
            return None

        path_data = self.paths[obj_id]
        smoothed_pts = path_data.get('smoothed_points', path_data['points'])
        smoothed_areas = path_data.get('smoothed_areas', path_data['areas'])

        if len(smoothed_pts) < 2:
            return None

        # ตรวจสอบ cache
        if not path_data.get('_kinematics_dirty', True) and 'kinematic_history' in path_data:
            cached = path_data['kinematic_history']
            if cached and 'velocity' in cached:
                return cached

        # แปลงเป็น numpy array
        pts_arr = np.array(list(smoothed_pts), dtype=np.float32)
        areas_arr = np.array(list(smoothed_areas), dtype=np.float32)

        # 1. Velocity & Acceleration (ใช้ numpy diff เพื่อประสิทธิภาพ)
        if len(pts_arr) >= 2:
            # Velocity: ความแตกต่างของตำแหน่งระหว่างเฟรม
            velocities = np.diff(pts_arr, axis=0)  # shape: (n-1, 2)
            # ใช้ velocity ล่าสุด
            if len(velocities) > 0:
                velocity = velocities[-1]  # (vx, vy)
                velocity_magnitude = np.linalg.norm(velocity)
            else:
                velocity = np.array([0.0, 0.0])
                velocity_magnitude = 0.0

            # Acceleration: ความแตกต่างของ velocity
            if len(velocities) >= 2:
                accelerations = np.diff(velocities, axis=0)  # shape: (n-2, 2)
                if len(accelerations) > 0:
                    acceleration = accelerations[-1]  # (ax, ay)
                else:
                    acceleration = np.array([0.0, 0.0])
            else:
                acceleration = np.array([0.0, 0.0])
        else:
            velocity = np.array([0.0, 0.0])
            acceleration = np.array([0.0, 0.0])
            velocity_magnitude = 0.0

        # 2. Area Variance (Rigid Body Check)
        if len(areas_arr) >= 2:
            area_mean = np.mean(areas_arr)
            area_std = np.std(areas_arr)
            # คำนวณ coefficient of variation (std/mean)
            if area_mean > 0:
                area_variance_ratio = area_std / area_mean
            else:
                area_variance_ratio = 0.0
        else:
            area_variance_ratio = 0.0

        # 3. Smoothness Index (ใช้ dot product ของ direction vectors)
        smoothness_index = 0.0
        if len(pts_arr) >= 3:
            # คำนวณ direction vectors
            direction_vectors = []
            for i in range(len(pts_arr) - 1):
                vec = pts_arr[i+1] - pts_arr[i]
                mag = np.linalg.norm(vec)
                if mag > 0:
                    direction_vectors.append(vec / mag)  # Normalize
                else:
                    direction_vectors.append(np.array([0.0, 0.0]))

            # คำนวณ dot products ระหว่าง direction vectors ที่ติดกัน
            if len(direction_vectors) >= 2:
                dot_products = []
                for i in range(len(direction_vectors) - 1):
                    dot = np.dot(direction_vectors[i], direction_vectors[i+1])
                    dot_products.append(dot)

                # Smoothness index = average dot product (clamp to 0-1)
                if len(dot_products) > 0:
                    smoothness_index = max(0.0, min(1.0, np.mean(dot_products)))

        # เก็บผลลัพธ์
        result = {
            'velocity': tuple(velocity),
            'acceleration': tuple(acceleration),
            'area_variance': float(area_variance_ratio),
            'smoothness_index': float(smoothness_index),
            'velocity_magnitude': float(velocity_magnitude)
        }

        # Cache ผลลัพธ์
        path_data['kinematic_history'] = result
        path_data['_kinematics_dirty'] = False

        return result

    def _is_drone_behavior(self, obj_id, frame_center=None):
        """
        ตรวจสอบพฤติกรรมเฉพาะของโดรน (Multirotor) จาก kinematic signature

        Args:
            obj_id: object ID
            frame_center: (cx, cy) ของกึ่งกลางเฟรม (optional, สำหรับ threat assessment)

        Returns:
            tuple: (is_drone, status_string)
                is_drone: bool
                status_string: 'DRONE_CONFIRMED', 'STATIONARY_DRONE', หรือ None
        """
        if obj_id not in self.paths:
            return False, None

        path_data = self.paths[obj_id]
        kinematics = self._calculate_kinematics(obj_id)

        if kinematics is None:
            return False, None

        smoothed_pts = path_data.get('smoothed_points', path_data['points'])
        smoothed_areas = path_data.get('smoothed_areas', path_data['areas'])

        if len(smoothed_pts) < 3:
            return False, None

        velocity = kinematics['velocity']
        velocity_mag = kinematics['velocity_magnitude']
        acceleration = kinematics['acceleration']
        area_variance = kinematics['area_variance']
        smoothness_index = kinematics['smoothness_index']

        # ตรวจสอบพื้นที่ปัจจุบัน
        current_area = smoothed_areas[-1] if len(smoothed_areas) > 0 else 0

        # 1. Hovering Detect: ความเร็วเข้าใกล้ 0 ติดต่อกัน > 7 เฟรม แต่ตำแหน่งยังคงที่
        if len(smoothed_pts) >= 8:
            # 🔥 ตรวจสอบว่ามี gap ใน path หรือไม่ (หายไปแล้วกลับมาใหม่)
            has_tracking_gap = False
            if 'points_with_frame' in path_data:
                frames_with_data = list(path_data['points_with_frame'])
                if len(frames_with_data) >= 8:
                    # ตรวจสอบ frame gaps ใน 8 จุดล่าสุด
                    recent_frames = [f[2] for f in frames_with_data[-8:]]  # frame number
                    frame_gaps = []
                    for i in range(len(recent_frames) - 1):
                        gap = recent_frames[i+1] - recent_frames[i]
                        frame_gaps.append(gap)

                    # ถ้ามี gap มาก (> KIN_FRAME_GAP_MAX เฟรม) = หายไปแล้วกลับมาใหม่ → ไม่ใช่ hovering
                    if any(gap > KIN_FRAME_GAP_MAX for gap in frame_gaps):
                        has_tracking_gap = True

            # ถ้ามี tracking gap → ข้าม hovering detection (ไม่ใช่ hovering จริง)
            if not has_tracking_gap:
                # ตรวจสอบ velocity ใน 7 เฟรมล่าสุด
                recent_velocities = []
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)
                for i in range(max(0, len(pts_arr) - 7), len(pts_arr) - 1):
                    vel = pts_arr[i+1] - pts_arr[i]
                    recent_velocities.append(np.linalg.norm(vel))

                if len(recent_velocities) >= 7:
                    avg_recent_velocity = np.mean(recent_velocities)
                    max_recent_velocity = np.max(recent_velocities)

                    # ตรวจสอบว่าตำแหน่งยังคงที่ (ไม่เคลื่อนที่มาก)
                    first_pos = pts_arr[-7]
                    last_pos = pts_arr[-1]
                    position_change = np.linalg.norm(last_pos - first_pos)

                    # Hovering: velocity ต่ำมาก แต่ตำแหน่งไม่เปลี่ยนมาก
                    if avg_recent_velocity < KIN_HOVER_AVG_VEL_MAX and max_recent_velocity < KIN_HOVER_MAX_VEL_MAX and position_change < KIN_HOVER_POS_CHANGE_MAX:
                        if area_variance < KIN_AREA_VARIANCE_MAX:
                            return True, 'STATIONARY_DRONE'

        # 2. Controlled Pivot: Hover แล้วเปลี่ยนทิศทางทันทีด้วยความเร่งสม่ำเสมอ
        if len(smoothed_pts) >= 5:
            pts_arr = np.array(list(smoothed_pts), dtype=np.float32)

            # ตรวจสอบว่ามี hover phase แล้วตามด้วย turn
            # ดู 5 จุดล่าสุด
            recent_5 = pts_arr[-5:]
            velocities_recent = []
            for i in range(len(recent_5) - 1):
                vel = recent_5[i+1] - recent_5[i]
                velocities_recent.append(np.linalg.norm(vel))

            # ตรวจสอบ pattern: ช่วงแรกช้า (hover) แล้วตามด้วยการเปลี่ยนทิศทาง
            if len(velocities_recent) >= 4:
                first_half_vel = np.mean(velocities_recent[:2])
                second_half_vel = np.mean(velocities_recent[2:])

                # ตรวจสอบการเปลี่ยนทิศทาง
                dir1 = recent_5[1] - recent_5[0]
                dir2 = recent_5[-1] - recent_5[-2]
                if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                    dir1_norm = dir1 / np.linalg.norm(dir1)
                    dir2_norm = dir2 / np.linalg.norm(dir2)
                    direction_change = np.dot(dir1_norm, dir2_norm)

                    # Controlled pivot: hover (ช้า) แล้ว turn (เปลี่ยนทิศทางมาก) ด้วยความเร่งสม่ำเสมอ
                    if first_half_vel < KIN_PIVOT_FIRST_HALF_VEL_MAX and second_half_vel > KIN_PIVOT_SECOND_HALF_VEL_MIN and direction_change < KIN_PIVOT_DIR_DOT_MAX:
                        accel_mag = np.linalg.norm(acceleration)
                        if KIN_PIVOT_ACCEL_MIN < accel_mag < KIN_PIVOT_ACCEL_MAX:
                            if area_variance < KIN_AREA_VARIANCE_MAX:
                                return True, 'DRONE_CONFIRMED'

        # 2.5. 🔥 ตรวจสอบการบินขึ้นๆลงๆ (Vertical Oscillation)
        # โดรนมักจะบินขึ้นๆลงๆ (vertical movement) ซึ่งนกไม่ทำ
        if len(smoothed_pts) >= 8:
            # 🔥 ตรวจสอบว่ามี gap ใน path หรือไม่ (หายไปแล้วกลับมาใหม่)
            has_tracking_gap = False
            if 'points_with_frame' in path_data:
                frames_with_data = list(path_data['points_with_frame'])
                if len(frames_with_data) >= 8:
                    recent_frames = [f[2] for f in frames_with_data[-8:]]
                    frame_gaps = []
                    for i in range(len(recent_frames) - 1):
                        gap = recent_frames[i+1] - recent_frames[i]
                        frame_gaps.append(gap)
                    if any(gap > KIN_FRAME_GAP_MAX for gap in frame_gaps):
                        has_tracking_gap = True

            # ถ้ามี tracking gap → ข้าม detection (ไม่ใช่พฤติกรรมจริง)
            if not has_tracking_gap:
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)

                # ตรวจสอบการเปลี่ยนแปลง y coordinate (vertical movement)
                y_coords = pts_arr[:, 1]  # y coordinates

                # คำนวณ vertical velocity (dy)
                vertical_velocities = []
                for i in range(len(y_coords) - 1):
                    dy = y_coords[i+1] - y_coords[i]
                    vertical_velocities.append(dy)

                if len(vertical_velocities) >= 5:
                    # สร้างสัญญาณทิศทาง: +1 ขึ้น, -1 ลง, 0 กลาง (กรอง jitter ด้วย min |dy|)
                    directions = []
                    for dy in vertical_velocities:
                        if dy >= KIN_VERT_OSC_MIN_DY:
                            directions.append(1)
                        elif dy <= -KIN_VERT_OSC_MIN_DY:
                            directions.append(-1)
                        else:
                            directions.append(0)
                    direction_changes = 0
                    for i in range(len(directions) - 1):
                        if (directions[i] == 1 and directions[i+1] == -1) or (directions[i] == -1 and directions[i+1] == 1):
                            direction_changes += 1
                    if direction_changes >= KIN_VERT_OSC_DIR_CHANGES_MIN:
                        if area_variance < KIN_AREA_VARIANCE_MAX:
                            if KIN_VERT_OSC_VEL_MIN < velocity_mag < KIN_VERT_OSC_VEL_MAX:
                                return True, 'DRONE_CONFIRMED'

        # 2.6. Closed loop / return-to-start (จุดสุดท้ายใกล้จุดแรก + ระยะรวมพอ)
        # ไม่ได้ตรวจรูปเลข 8 โดยตรง แค่ตรวจว่าบินวนกลับมาใกล้จุดเริ่มต้น
        if len(smoothed_pts) >= 10:
            # 🔥 ตรวจสอบว่ามี gap ใน path หรือไม่ (หายไปแล้วกลับมาใหม่)
            has_tracking_gap = False
            if 'points_with_frame' in path_data:
                frames_with_data = list(path_data['points_with_frame'])
                if len(frames_with_data) >= 10:
                    recent_frames = [f[2] for f in frames_with_data[-10:]]
                    frame_gaps = []
                    for i in range(len(recent_frames) - 1):
                        gap = recent_frames[i+1] - recent_frames[i]
                        frame_gaps.append(gap)
                    if any(gap > KIN_FRAME_GAP_MAX for gap in frame_gaps):
                        has_tracking_gap = True

            # ถ้ามี tracking gap → ข้าม detection (ไม่ใช่พฤติกรรมจริง)
            if not has_tracking_gap:
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)

                # ตรวจสอบว่ามีการวนกลับมาที่จุดเดิมหรือไม่ (closed loop)
                # ใช้ 10 จุดล่าสุด
                recent_10 = pts_arr[-10:]

                # คำนวณระยะห่างระหว่างจุดแรกกับจุดสุดท้าย
                first_pos = recent_10[0]
                last_pos = recent_10[-1]
                loop_distance = np.linalg.norm(last_pos - first_pos)

                # คำนวณระยะทางรวมที่เดินทาง
                total_distance = 0.0
                for i in range(len(recent_10) - 1):
                    dist = np.linalg.norm(recent_10[i+1] - recent_10[i])
                    total_distance += dist

                if total_distance > 0:
                    loop_ratio = loop_distance / total_distance
                    if loop_ratio < KIN_LOOP_RATIO_MAX and total_distance > KIN_LOOP_TOTAL_DIST_MIN:
                        if area_variance < KIN_AREA_VARIANCE_MAX:
                            return True, 'DRONE_CONFIRMED'

        # 2.7. 🔥 ตรวจสอบการบินวกไปวนมา (Erratic Movement)
        # โดรนมักจะบินวกไปวนมา (เปลี่ยนทิศทางบ่อย) ซึ่งนกไม่ทำ
        if len(smoothed_pts) >= 7:
            # 🔥 ตรวจสอบว่ามี gap ใน path หรือไม่ (หายไปแล้วกลับมาใหม่)
            has_tracking_gap = False
            if 'points_with_frame' in path_data:
                frames_with_data = list(path_data['points_with_frame'])
                if len(frames_with_data) >= 7:
                    recent_frames = [f[2] for f in frames_with_data[-7:]]
                    frame_gaps = []
                    for i in range(len(recent_frames) - 1):
                        gap = recent_frames[i+1] - recent_frames[i]
                        frame_gaps.append(gap)
                    if any(gap > KIN_FRAME_GAP_MAX for gap in frame_gaps):
                        has_tracking_gap = True

            # ถ้ามี tracking gap → ข้าม detection (ไม่ใช่พฤติกรรมจริง)
            if not has_tracking_gap:
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)

                # คำนวณมุมหัก (angle change) ระหว่าง direction vectors
                angles = []
                for i in range(len(pts_arr) - 2):
                    v1 = pts_arr[i+1] - pts_arr[i]
                    v2 = pts_arr[i+2] - pts_arr[i+1]
                    mag1 = np.linalg.norm(v1)
                    mag2 = np.linalg.norm(v2)
                    if mag1 > 0 and mag2 > 0:
                        v1_norm = v1 / mag1
                        v2_norm = v2 / mag2
                        dot = np.dot(v1_norm, v2_norm)
                        dot = max(-1.0, min(1.0, dot))
                        angle = np.arccos(dot) * 180.0 / np.pi
                        angles.append(angle)

                if len(angles) >= 5:
                    sharp_turns = sum(1 for a in angles if a > KIN_ERRATIC_ANGLE_DEG_MIN)
                    if sharp_turns >= KIN_ERRATIC_SHARP_TURNS_MIN:
                        if area_variance < KIN_AREA_VARIANCE_MAX:
                            if KIN_ERRATIC_VEL_MIN < velocity_mag < KIN_ERRATIC_VEL_MAX:
                                return True, 'DRONE_CONFIRMED'

        # 2.8. 🔥 ตรวจสอบการตีโค้ง/เลี้ยวหักศอก (Curved Turn Detection)
        # โดรนมักจะตีโค้งหรือเลี้ยวหักศอก ไม่ใช่เส้นตรง
        if len(smoothed_pts) >= 5:
            # 🔥 ตรวจสอบว่ามี gap ใน path หรือไม่ (หายไปแล้วกลับมาใหม่)
            has_tracking_gap = False
            if 'points_with_frame' in path_data:
                frames_with_data = list(path_data['points_with_frame'])
                if len(frames_with_data) >= 5:
                    recent_frames = [f[2] for f in frames_with_data[-5:]]
                    frame_gaps = []
                    for i in range(len(recent_frames) - 1):
                        gap = recent_frames[i+1] - recent_frames[i]
                        frame_gaps.append(gap)
                    if any(gap > KIN_FRAME_GAP_MAX for gap in frame_gaps):
                        has_tracking_gap = True

            # ถ้ามี tracking gap → ข้าม detection (ไม่ใช่พฤติกรรมจริง)
            if not has_tracking_gap:
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)

                # คำนวณ curvature (ความโค้ง) จาก 5 จุดล่าสุด
                recent_5 = pts_arr[-5:]

                # คำนวณมุมหัก (angle change) ระหว่าง direction vectors
                angles = []
                for i in range(len(recent_5) - 2):
                    v1 = recent_5[i+1] - recent_5[i]
                    v2 = recent_5[i+2] - recent_5[i+1]
                    mag1 = np.linalg.norm(v1)
                    mag2 = np.linalg.norm(v2)
                    if mag1 > 0 and mag2 > 0:
                        v1_norm = v1 / mag1
                        v2_norm = v2 / mag2
                        dot = np.dot(v1_norm, v2_norm)
                        dot = max(-1.0, min(1.0, dot))
                        angle = np.arccos(dot) * 180.0 / np.pi
                        angles.append(angle)

                if len(angles) > 0:
                    avg_angle_change = np.mean(angles)
                    max_angle_change = np.max(angles)

                    if KIN_CURVE_ANGLE_AVG_MIN <= avg_angle_change <= KIN_CURVE_ANGLE_AVG_MAX and max_angle_change <= KIN_CURVE_ANGLE_MAX_MAX:
                        if smoothness_index < KIN_CURVE_SMOOTHNESS_MAX:
                            if area_variance < KIN_AREA_VARIANCE_MAX:
                                if KIN_CURVE_VEL_MIN < velocity_mag < KIN_CURVE_VEL_MAX:
                                    return True, 'DRONE_CONFIRMED'
                    if max_angle_change > KIN_CURVE_SHARP_ANGLE_MIN and max_angle_change <= KIN_CURVE_ANGLE_MAX_MAX:
                        if area_variance < KIN_AREA_VARIANCE_MAX:
                            pts_arr_full = np.array(list(smoothed_pts), dtype=np.float32)
                            velocities_check = []
                            for i in range(len(pts_arr_full) - 1):
                                vel = pts_arr_full[i+1] - pts_arr_full[i]
                                velocities_check.append(np.linalg.norm(vel))
                            if len(velocities_check) >= 3:
                                vel_before = velocities_check[-3] if len(velocities_check) >= 3 else velocities_check[0]
                                vel_after = velocities_check[-1]
                                if abs(vel_after - vel_before) > KIN_CURVE_VEL_DIFF_MIN:
                                    return True, 'DRONE_CONFIRMED'

        # 3. Long-range Small Object Handling: area < 20 pixels
        if current_area < 20:
            # ผ่อนปรน area_variance เป็น 20%
            if area_variance < 0.20:
                # เน้นดูความนิ่งของทิศทาง (smoothness)
                if smoothness_index > 0.6:  # ทิศทางค่อนข้างนิ่ง
                    # ตรวจสอบ velocity ที่เหมาะสม (ไม่เร็วเกินไป)
                    if 0.5 < velocity_mag < 15.0:
                        return True, 'DRONE_CONFIRMED'

        # 4. General Drone Characteristics
        # Rigid body (area variance ต่ำ) + smooth movement + velocity ที่เหมาะสม
        # ⚠️ ปรับปรุง: ต้องแยกโดรนออกจากนก/เครื่องบินที่บินตรงๆ
        if area_variance < 0.10:  # Rigid body check
            if smoothness_index > 0.5:  # Smooth movement
                if 1.0 < velocity_mag < 20.0:  # Velocity ที่เหมาะสมสำหรับโดรน
                    # 🔥 ตรวจสอบ velocity consistency
                    # เส้นตรงมากๆ + ความเร็วคงที่ = อาจเป็นนกหรือเครื่องบินระยะไกล (ไม่ใช่โดรน)
                    if len(smoothed_pts) >= 5:
                        pts_arr = np.array(list(smoothed_pts), dtype=np.float32)
                        # คำนวณ velocity ของแต่ละช่วง
                        velocities = []
                        for i in range(len(pts_arr) - 1):
                            vel = pts_arr[i+1] - pts_arr[i]
                            velocities.append(np.linalg.norm(vel))

                        if len(velocities) >= 3:
                            vel_std = np.std(velocities)
                            vel_mean = np.mean(velocities)

                            if vel_mean > 0:
                                cv = vel_std / vel_mean  # Coefficient of Variation

                                # ❌ กรณีที่ไม่ใช่โดรน: เส้นตรงมาก + ความเร็วคงที่ = นกหรือเครื่องบินระยะไกล
                                if smoothness_index > KIN_TAIL_SMOOTHNESS_REVOKE and cv < KIN_TAIL_CV_REVOKE:
                                    return False, None  # ไม่ใช่โดรน

                                # 2. เส้นตรงมาก (smoothness > 0.75) + ความเร็วคงที่มาก (CV < 0.10)
                                #    = นกหรือเครื่องบินที่บินตรงๆ
                                if smoothness_index > 0.75 and cv < 0.10:
                                    return False, None  # ไม่ใช่โดรน

                                # ✅ โดรนควรมีความแปรปรวนในความเร็ว (CV 0.15-0.5)
                                #    หรือมีการเปลี่ยนแปลงทิศทางบ้าง (smoothness ไม่สูงเกินไป)
                                if 0.15 <= cv <= 0.5:
                                    # มีความแปรปรวนในความเร็ว = น่าจะเป็นโดรน
                                    return True, 'DRONE_CONFIRMED'

                                # ถ้า CV ต่ำมาก (< 0.15) แต่ smoothness ไม่สูงเกินไป (< 0.8)
                                # อาจเป็นโดรนที่บินช้าๆ และสม่ำเสมอ
                                if cv < 0.15 and smoothness_index < 0.8:
                                    # ตรวจสอบ acceleration: โดรนมักมีความเร่ง/ความหน่วงบ้าง
                                    accel_mag = np.linalg.norm(acceleration)
                                    if accel_mag > 0.3:  # มีความเร่งบ้าง
                                        return True, 'DRONE_CONFIRMED'
                                    else:
                                        return False, None  # ไม่มีความเร่ง = อาจเป็นนก/เครื่องบิน

                                # ถ้า CV สูงเกินไป (> 0.5) = ความเร็วไม่สม่ำเสมอมาก = อาจเป็นนกที่บินไม่สม่ำเสมอ
                                if cv > 0.5:
                                    return False, None

                    # ถ้าไม่มีข้อมูลเพียงพอ ให้ใช้เงื่อนไขเดิมแต่เพิ่มข้อจำกัด
                    # เส้นตรงมากเกินไป (smoothness > 0.9) = ไม่น่าจะเป็นโดรน (น่าจะเป็นนก/เครื่องบิน)
                    if smoothness_index > 0.9:
                        return False, None

                    # เงื่อนไขเดิม: smoothness ไม่สูงเกินไป (< 0.8) = อาจเป็นโดรน
                    # แต่ต้องมี acceleration บ้าง (โดรนมักมีการเปลี่ยนแปลงความเร็ว)
                    accel_mag = np.linalg.norm(acceleration)
                    if smoothness_index <= 0.8 and accel_mag > 0.2:
                        return True, 'DRONE_CONFIRMED'

        return False, None

    def _assess_threat_level(self, obj_id, frame_center):
        """
        ประเมินระดับภัยคุกคาม: ตรวจสอบ area เพิ่มขึ้น + ทิศทางมุ่งสู่กึ่งกลางภาพ

        Args:
            obj_id: object ID
            frame_center: (cx, cy) ของกึ่งกลางเฟรม

        Returns:
            str: 'HIGH_PRIORITY_THREAT' หรือ None
        """
        if obj_id not in self.paths:
            return None

        path_data = self.paths[obj_id]
        smoothed_pts = path_data.get('smoothed_points', path_data['points'])
        smoothed_areas = path_data.get('smoothed_areas', path_data['areas'])

        if len(smoothed_pts) < 5 or len(smoothed_areas) < 5:
            return None

        # 1. ตรวจสอบ area เพิ่มขึ้นอย่างต่อเนื่อง
        areas_arr = np.array(list(smoothed_areas), dtype=np.float32)
        recent_areas = areas_arr[-5:]  # 5 จุดล่าสุด

        # คำนวณ trend ของ area (linear regression แบบง่าย)
        x = np.arange(len(recent_areas))
        area_trend = np.polyfit(x, recent_areas, 1)[0]  # slope

        # Area ต้องเพิ่มขึ้น (slope > 0) และเพิ่มขึ้นอย่างมีนัยสำคัญ
        if area_trend <= 0:
            return None

        # ตรวจสอบว่าการเพิ่มขึ้นมีนัยสำคัญ (เพิ่มขึ้น > 10% ใน 5 เฟรม)
        area_increase_ratio = (recent_areas[-1] - recent_areas[0]) / max(recent_areas[0], 1.0)
        if area_increase_ratio < 0.10:  # เพิ่มขึ้นน้อยกว่า 10%
            return None

        # 2. ตรวจสอบทิศทางมุ่งสู่กึ่งกลางภาพ
        pts_arr = np.array(list(smoothed_pts), dtype=np.float32)
        recent_pts = pts_arr[-5:]

        # คำนวณทิศทางเฉลี่ยของ 5 จุดล่าสุด
        directions = []
        for i in range(len(recent_pts) - 1):
            dir_vec = recent_pts[i+1] - recent_pts[i]
            mag = np.linalg.norm(dir_vec)
            if mag > 0:
                directions.append(dir_vec / mag)

        if len(directions) == 0:
            return None

        avg_direction = np.mean(directions, axis=0)
        avg_direction = avg_direction / np.linalg.norm(avg_direction)  # Normalize

        # คำนวณเวกเตอร์จากตำแหน่งปัจจุบันไปยังกึ่งกลางภาพ
        current_pos = recent_pts[-1]
        to_center = np.array([frame_center[0] - current_pos[0],
                             frame_center[1] - current_pos[1]])
        to_center_mag = np.linalg.norm(to_center)

        if to_center_mag > 0:
            to_center_norm = to_center / to_center_mag

            # คำนวณ dot product ระหว่างทิศทางการเคลื่อนที่กับทิศทางไปยังกึ่งกลาง
            alignment = np.dot(avg_direction, to_center_norm)

            # ถ้า alignment > 0.5 แปลว่ามุ่งสู่กึ่งกลาง
            if alignment > 0.5:
                return 'HIGH_PRIORITY_THREAT'

        return None

    def _linear_predict_position(self, last_point, velocity, frames_elapsed):
        """
        ทำนายตำแหน่งใหม่จากความเร็วล่าสุด (Linear Prediction)

        Args:
            last_point: (x, y) ตำแหน่งล่าสุด
            velocity: (vx, vy) ความเร็วล่าสุด
            frames_elapsed: จำนวนเฟรมที่ผ่านไป

        Returns:
            (x, y): ตำแหน่งที่ทำนาย
        """
        predicted_x = last_point[0] + velocity[0] * frames_elapsed
        predicted_y = last_point[1] + velocity[1] * frames_elapsed
        return (int(predicted_x), int(predicted_y))

    def _create_persistence_kalman(self, cx, cy, vx=0, vy=0):
        """
        สร้าง Kalman filter สำหรับ persistence path prediction

        Args:
            cx, cy: ตำแหน่งเริ่มต้น (center x, center y)
            vx, vy: ความเร็วเริ่มต้น (velocity x, velocity y)

        Returns:
            cv2.KalmanFilter: Kalman filter instance
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
        kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05  # เพิ่ม noise เล็กน้อย
        kf.statePre = np.array([[cx], [cy], [vx], [vy]], np.float32)
        kf.statePost = np.array([[cx], [cy], [vx], [vy]], np.float32)
        return kf

    def update(self, tracked_objects, congested_zones, sky_mask, frame_center=None, frame_number=0,
               frame_width=None, frame_height=None, valid_detections=None, yolo_origin_ids=None):
        current_ids = set(tracked_objects.keys())
        stored_ids = set(self.paths.keys())
        dead_ids = stored_ids - current_ids

        # 🔥 TRACK PERSISTENCE: ตรวจสอบวัตถุที่หายไปแต่ kinematic_confirmed == True
        persistence_to_remove = []
        for pid, pdata in list(self.persistence_paths.items()):
            # ตรวจสอบว่ามี Kalman filter หรือไม่
            if pid not in self.persistence_kalman_filters:
                persistence_to_remove.append(pid)
                continue

            kf = self.persistence_kalman_filters[pid]

            # ใช้ Kalman filter ทำนายตำแหน่งใหม่
            pred = kf.predict()
            predicted_x = int(pred[0])
            predicted_y = int(pred[1])
            predicted_pos = (predicted_x, predicted_y)

            # ตรวจสอบขอบเขตเฟรม
            if frame_width is not None and frame_height is not None:
                if predicted_x < 0 or predicted_x >= frame_width or predicted_y < 0 or predicted_y >= frame_height:
                    # อยู่นอกเฟรม → ลบทันที
                    persistence_to_remove.append(pid)
                    continue
                # ถ้าใกล้ขอบเฟรมมาก (< 50 pixels) → ลดอายุเร็วขึ้น
                margin = 50
                if (predicted_x < margin or predicted_x >= frame_width - margin or
                    predicted_y < margin or predicted_y >= frame_height - margin):
                    pdata['persistence_counter'] -= 2  # ลดอายุเร็วขึ้น 2 เท่า
                else:
                    pdata['persistence_counter'] -= 1
            else:
                pdata['persistence_counter'] -= 1

            # ตรวจสอบ motion detection ใกล้ตำแหน่งที่ทำนาย
            has_nearby_detection = False
            if valid_detections is not None:
                for det in valid_detections:
                    det_x, det_y = det[0] + det[2] // 2, det[1] + det[3] // 2
                    dist = math.sqrt((predicted_x - det_x)**2 + (predicted_y - det_y)**2)
                    if dist < 100:  # รัศมี 100 pixels
                        has_nearby_detection = True
                        # วัตถุกลับมาแล้ว → ลบ persistence path
                        persistence_to_remove.append(pid)
                        break

            if not has_nearby_detection:
                # ไม่มี detection ใกล้เคียง → เพิ่ม counter
                pdata['no_detection_count'] = pdata.get('no_detection_count', 0) + 1
                if pdata['no_detection_count'] >= 3:
                    # ไม่มี detection ต่อเนื่อง 3 เฟรม → ลบ persistence path
                    persistence_to_remove.append(pid)
                    continue
            else:
                pdata['no_detection_count'] = 0

            # ตรวจสอบ congestion zones
            if congested_zones is not None:
                for zone in congested_zones:
                    zx1, zy1, zx2, zy2 = zone
                    if zx1 <= predicted_x <= zx2 and zy1 <= predicted_y <= zy2:
                        # อยู่ใน congestion zone → ลบ persistence path
                        persistence_to_remove.append(pid)
                        break

            # ตรวจสอบ sky mask
            if sky_mask is not None and predicted_y < sky_mask.shape[0] and predicted_x < sky_mask.shape[1]:
                is_sky = sky_mask[predicted_y, predicted_x] > 0
                # ถ้าวัตถุเดิมอยู่บนฟ้า แต่ตำแหน่งที่ทำนายไม่ใช่ท้องฟ้า → ลดอายุเร็วขึ้น
                if pdata.get('was_sky', False) and not is_sky:
                    pdata['persistence_counter'] -= 1  # ลดอายุเพิ่ม

            # อัปเดตตำแหน่งล่าสุด
            pdata['last_point'] = predicted_pos

            # เพิ่ม velocity decay โดยการปรับ processNoiseCov ให้เพิ่มขึ้นเมื่อเวลาผ่านไป
            frames_elapsed = 8 - pdata['persistence_counter']
            if frames_elapsed > 0:
                # เพิ่ม noise เมื่อเวลาผ่านไป (ทำให้ความไม่แน่นอนเพิ่มขึ้น)
                decay_factor = 1.0 + (frames_elapsed * 0.02)  # เพิ่ม 2% ต่อเฟรม
                kf.processNoiseCov = np.eye(4, dtype=np.float32) * (0.05 * decay_factor)

            # ตรวจสอบอายุ
            if pdata['persistence_counter'] <= 0:
                persistence_to_remove.append(pid)

        # ลบ persistence paths ที่หมดอายุ (พร้อม cleanup)
        for pid in persistence_to_remove:
            if pid in self.persistence_paths:
                # Cleanup kinematic data
                pdata = self.persistence_paths[pid]
                if 'kinematic_history' in pdata:
                    del pdata['kinematic_history']
                del self.persistence_paths[pid]
            # Cleanup Kalman filter
            if pid in self.persistence_kalman_filters:
                del self.persistence_kalman_filters[pid]

        # ตรวจสอบ dead_ids ที่ kinematic_confirmed == True
        for did in list(dead_ids):
            if did in self.paths:
                path_data = self.paths[did]
                if path_data.get('kinematic_confirmed', False):
                    # เก็บไว้ใน persistence_paths
                    kinematics = self._calculate_kinematics(did)
                    if kinematics:
                        last_pt = path_data['smoothed_points'][-1] if len(path_data['smoothed_points']) > 0 else path_data['points'][-1]
                        velocity = kinematics['velocity']
                        # ตรวจสอบว่าวัตถุอยู่บนฟ้าหรือไม่
                        was_sky = False
                        if sky_mask is not None and last_pt[1] < sky_mask.shape[0] and last_pt[0] < sky_mask.shape[1]:
                            was_sky = sky_mask[last_pt[1], last_pt[0]] > 0
                        # สร้าง Kalman filter สำหรับ persistence path
                        kf = self._create_persistence_kalman(last_pt[0], last_pt[1], velocity[0], velocity[1])
                        self.persistence_kalman_filters[did] = kf
                        self.persistence_paths[did] = {
                            'last_point': last_pt,
                            'last_velocity': velocity,
                            'persistence_counter': 4,  # ลด runaway เมื่อโดรนหาย (เดิม 8)
                            'drone_status': path_data.get('drone_status'),
                            'kinematic_confirmed': True,
                            'no_detection_count': 0,  # นับจำนวนเฟรมที่ไม่มี detection ใกล้เคียง
                            'was_sky': was_sky  # เก็บสถานะว่าวัตถุอยู่บนฟ้าหรือไม่
                        }

        # Cleanup paths ที่ไม่ใช่ kinematic_confirmed
        for did in dead_ids:
            if did in self.paths:
                path_data = self.paths[did]
                # 🔥 CLEANUP: ลบ kinematic_history และข้อมูลที่เกี่ยวข้อง
                if 'kinematic_history' in path_data:
                    del path_data['kinematic_history']
                # ลบ smoothed data (ถ้ามี)
                if 'smoothed_points' in path_data:
                    path_data['smoothed_points'].clear()
                if 'smoothed_areas' in path_data:
                    path_data['smoothed_areas'].clear()
                del self.paths[did]

        # Cleanup persistence_paths ที่กลับมาแล้ว (อยู่ใน tracked_objects)
        for pid in list(self.persistence_paths.keys()):
            if pid in current_ids:
                # วัตถุกลับมาแล้ว ลบออกจาก persistence_paths
                if pid in self.persistence_paths:
                    pdata = self.persistence_paths[pid]
                    if 'kinematic_history' in pdata:
                        del pdata['kinematic_history']
                    del self.persistence_paths[pid]
                # Cleanup Kalman filter
                if pid in self.persistence_kalman_filters:
                    del self.persistence_kalman_filters[pid]

        # Update
        yolo_ids = yolo_origin_ids if yolo_origin_ids is not None else set()
        for obj_id, (rect, is_real) in tracked_objects.items():
            x, y, w, h, _ = rect
            center = (int(x + w/2), int(y + h/2))
            from_yolo = obj_id in yolo_ids

            # 1. Congestion Check (ถ้าเข้าดง ตัดทิ้งเลย) — ยกเว้น YOLO-origin ให้สร้าง path ได้ทุกที่
            in_congestion = False
            for zone in congested_zones:
                zx1, zy1, zx2, zy2 = zone
                if zx1 <= center[0] <= zx2 and zy1 <= center[1] <= zy2:
                    in_congestion = True
                    break

            if in_congestion and not from_yolo:
                if obj_id in self.paths: del self.paths[obj_id]
                continue

            if not is_real: continue

            # 2. Add Point Logic
            if obj_id not in self.paths:
                # Check Sky Origin (YOLO-origin: allow anywhere; else require sky_mask)
                is_origin_sky = False
                if from_yolo:
                    is_origin_sky = True
                elif sky_mask is not None:
                    check_y = min(max(center[1], 0), sky_mask.shape[0]-1)
                    check_x = min(max(center[0], 0), sky_mask.shape[1]-1)
                    if sky_mask[check_y, check_x] > 0: is_origin_sky = True
                else:
                    is_origin_sky = True

                if is_origin_sky:
                    # 🔥 ตรวจสอบจำนวน paths: ถ้าเกิน MAX_PATHS ให้ลบ path เก่าที่ไม่สำคัญออก (evict non-from_yolo ก่อน)
                    if len(self.paths) >= self.MAX_PATHS:
                        paths_to_remove = []

                        # ลบ path ที่ยังไม่ validated และไม่ใช่ kinematic_confirmed ก่อน (ไม่ใช่ from_yolo ก่อน)
                        for pid, pdata in self.paths.items():
                            if not pdata.get('validated', False) and not pdata.get('kinematic_confirmed', False) and not pdata.get('from_yolo', False):
                                paths_to_remove.append(pid)
                        if len(paths_to_remove) < (len(self.paths) - self.MAX_PATHS + 1):
                            for pid, pdata in self.paths.items():
                                if pid not in paths_to_remove and not pdata.get('validated', False) and not pdata.get('kinematic_confirmed', False):
                                    paths_to_remove.append(pid)

                        # ถ้ายังไม่พอ ลบ path ที่ validated แต่ไม่ใช่ kinematic_confirmed (ไม่ใช่ from_yolo ก่อน)
                        if len(paths_to_remove) < (len(self.paths) - self.MAX_PATHS + 1):
                            for pid, pdata in self.paths.items():
                                if pid not in paths_to_remove and pdata.get('validated', False) and not pdata.get('kinematic_confirmed', False) and not pdata.get('from_yolo', False):
                                    paths_to_remove.append(pid)
                        if len(paths_to_remove) < (len(self.paths) - self.MAX_PATHS + 1):
                            for pid, pdata in self.paths.items():
                                if pid not in paths_to_remove and pdata.get('validated', False) and not pdata.get('kinematic_confirmed', False):
                                    paths_to_remove.append(pid)

                        # ถ้ายังไม่พอ ลบ path ที่มี YOLO confidence ต่ำสุด
                        if len(paths_to_remove) < (len(self.paths) - self.MAX_PATHS + 1):
                            sorted_paths = sorted(
                                [(pid, pdata.get('max_yolo_conf', 0.0)) for pid, pdata in self.paths.items()
                                 if pid not in paths_to_remove],
                                key=lambda x: x[1]
                            )
                            remaining = (len(self.paths) - self.MAX_PATHS + 1) - len(paths_to_remove)
                            for pid, _ in sorted_paths[:remaining]:
                                paths_to_remove.append(pid)

                        for pid in paths_to_remove:
                            if pid in self.paths:
                                path_data = self.paths[pid]
                                if 'kinematic_history' in path_data:
                                    del path_data['kinematic_history']
                                if 'smoothed_points' in path_data:
                                    path_data['smoothed_points'].clear()
                                if 'smoothed_areas' in path_data:
                                    path_data['smoothed_areas'].clear()
                                del self.paths[pid]

                    # สร้าง path ใหม่
                    area = w * h
                    self.paths[obj_id] = {
                        'points': deque([center], maxlen=self.max_nodes),
                        'points_with_frame': deque([(center[0], center[1], frame_number)], maxlen=self.max_nodes),  # 🔥 เก็บ frame number
                        'areas': deque([area], maxlen=self.max_nodes),
                        'smoothed_points': deque([center], maxlen=self.max_nodes),
                        'smoothed_areas': deque([float(area)], maxlen=self.max_nodes),
                        'validated': False,  # รอเช็ค smooth ครบ PATH_VALIDATE_MIN_POINTS จุด
                        'sky_origin': True,
                        'from_yolo': from_yolo,  # YOLO-origin path (priority, create anywhere)
                        'max_yolo_conf': 0.0,  # YOLO confidence สูงสุดที่เคยเจอ
                        'yolo_conf_history': deque(maxlen=YOLO_CONF_HISTORY_MAXLEN),
                        'yolo_state': 'yellow',  # yellow/orange/red
                        'kinematic_confirmed': False,  # ถูกยืนยันโดย kinematic signature
                        'drone_status': None,  # None, 'DRONE_CONFIRMED', 'STATIONARY_DRONE', 'HIGH_PRIORITY_THREAT'
                        'kinematic_history': {},  # เก็บ velocity, acceleration, area_variance, smoothness_index
                        'persistence_counter': 0,  # สำหรับ track persistence
                        'last_velocity': (0.0, 0.0),  # (vx, vy) สำหรับ linear prediction
                        '_kinematics_dirty': True,  # flag สำหรับ cache invalidation
                        'distance_history': deque(maxlen=DISTANCE_HISTORY_MAXLEN),
                    }
            else:
                # Update existing path
                if self.paths[obj_id]['sky_origin']:
                    pts = self.paths[obj_id]['points']
                    areas = self.paths[obj_id]['areas']
                    last_pt = pts[-1]
                    area = w * h
                    min_dist_px = self.min_dist
                    if area < SMALL_OBJ_AREA:
                        min_dist_px = PATH_MIN_DIST_SMALL_OBJ_PX
                    dist_sq = (center[0]-last_pt[0])**2 + (center[1]-last_pt[1])**2

                    if dist_sq > (min_dist_px * min_dist_px):
                        pts.append(center)
                        areas.append(area)

                        # ตัดดาวกระพริบ (เล็กมาก + แทบไม่ขยับ)
                        if len(pts) >= STAR_STATIC_FRAMES and area < STAR_STATIC_AREA:
                            tail = list(pts)[-STAR_STATIC_FRAMES:]
                            dx = tail[-1][0] - tail[0][0]
                            dy = tail[-1][1] - tail[0][1]
                            disp = math.sqrt(dx*dx + dy*dy)
                            if disp < STAR_STATIC_MAX_DIST:
                                del self.paths[obj_id]
                                continue

                        # 🔥 เก็บ frame number ด้วย
                        if 'points_with_frame' not in self.paths[obj_id]:
                            self.paths[obj_id]['points_with_frame'] = deque(maxlen=self.max_nodes)
                        self.paths[obj_id]['points_with_frame'].append((center[0], center[1], frame_number))

                        # 🔥 TEMPORAL SMOOTHING (Moving Average)
                        if len(pts) >= 2:
                            smoothed_pts, smoothed_areas_list = self._apply_temporal_smoothing(
                                list(pts), list(areas), window_size=self.smooth_window
                            )
                            # อัปเดต smoothed data
                            self.paths[obj_id]['smoothed_points'] = deque(smoothed_pts, maxlen=self.max_nodes)
                            self.paths[obj_id]['smoothed_areas'] = deque(smoothed_areas_list, maxlen=self.max_nodes)
                            self.paths[obj_id]['_kinematics_dirty'] = True  # ต้องคำนวณ kinematics ใหม่

                        # 🔥 VALIDATION LOGIC (เช็คทุกครั้งที่มีจุดใหม่จนกว่าจะผ่าน)
                        if not self.paths[obj_id]['validated']:
                            if len(pts) >= PATH_VALIDATE_MIN_POINTS:
                                recent_n = list(pts)[-PATH_VALIDATE_MIN_POINTS:]
                                if self._is_window_smooth(recent_n):
                                    self.paths[obj_id]['validated'] = True
                                    if LOG_KINEMATIC_PATH:
                                        print(f"✅ Obj {obj_id} Validated (Smooth Path, {PATH_VALIDATE_MIN_POINTS} pts)")

                        # 🔥 KINEMATIC ANALYSIS: ตรวจสอบ drone behavior
                        if self.enable_kinematic_rules and len(pts) >= 3:  # ต้องมีข้อมูลเพียงพอ
                            # คำนวณ kinematics (ถ้ายังไม่ได้ cache)
                            kinematics = self._calculate_kinematics(obj_id)
                            if kinematics:
                                # อัปเดต last_velocity
                                self.paths[obj_id]['last_velocity'] = kinematics['velocity']

                                # 🔥 รีเช็คหาง path 20 จุดล่าสุด: ถ้าหาง path กลายเป็นเส้นตรง → ปลด kinematic_confirmed และ yolo_state
                                smoothness_index = kinematics.get('smoothness_index', 0.0)
                                velocity_mag = kinematics.get('velocity_magnitude', 0.0)

                                # 🔥 ใช้หาง path 20 จุดล่าสุด (ส่วนท้าย) แทน smoothed_pts ทั้งหมด
                                if len(pts) >= 5:
                                    # ใช้ 20 จุดล่าสุด (หาง path) เพื่อตรวจสอบว่าตอนนี้มันตรงหรือไม่
                                    tail_length = min(KIN_TAIL_LENGTH, len(smoothed_pts))
                                    tail_pts = list(smoothed_pts)[-tail_length:] if len(smoothed_pts) >= tail_length else list(smoothed_pts)

                                    if len(tail_pts) >= 5:
                                        pts_arr = np.array(tail_pts, dtype=np.float32)  # ใช้หาง path
                                        velocities = []
                                        for i in range(len(pts_arr) - 1):
                                            vel = pts_arr[i+1] - pts_arr[i]
                                            velocities.append(np.linalg.norm(vel))

                                        if len(velocities) >= 3:
                                            vel_std = np.std(velocities)
                                            vel_mean = np.mean(velocities)

                                            if vel_mean > 0:
                                                cv = vel_std / vel_mean  # Coefficient of Variation

                                                # 🔥 คำนวณ smoothness_index ของหาง path (20 จุดล่าสุด)
                                                tail_smoothness = 0.0
                                                if len(pts_arr) >= 3:
                                                    direction_vectors = []
                                                    for i in range(len(pts_arr) - 1):
                                                        vec = pts_arr[i+1] - pts_arr[i]
                                                        mag = np.linalg.norm(vec)
                                                        if mag > 0:
                                                            direction_vectors.append(vec / mag)
                                                        else:
                                                            direction_vectors.append(np.array([0.0, 0.0]))

                                                    if len(direction_vectors) >= 2:
                                                        dot_products = []
                                                        for i in range(len(direction_vectors) - 1):
                                                            dot = np.dot(direction_vectors[i], direction_vectors[i+1])
                                                            dot_products.append(dot)

                                                        if len(dot_products) > 0:
                                                            tail_smoothness = max(0.0, min(1.0, np.mean(dot_products)))

                                                # ❌ ถ้าหาง path กลายเป็นเส้นตรงมาก + ความเร็วคงที่ = นกหรือเครื่องบินระยะไกล
                                                if tail_smoothness > KIN_TAIL_SMOOTHNESS_REVOKE and cv < KIN_TAIL_CV_REVOKE:
                                                    # ปลด kinematic_confirmed
                                                    if self.paths[obj_id].get('kinematic_confirmed', False):
                                                        self.paths[obj_id]['kinematic_confirmed'] = False
                                                        self.paths[obj_id]['drone_status'] = None
                                                        if LOG_KINEMATIC_PATH:
                                                            print(f"⚠️ Obj {obj_id} Tail path (20 pts) is too straight - removed kinematic_confirmed")

                                                    # 🔥 เปลี่ยน yolo_state เป็น orange (ไม่ใช่ red) แม้ยังไม่มี YOLO detection ใหม่
                                                    current_yolo_state = self.paths[obj_id].get('yolo_state', 'yellow')
                                                    if current_yolo_state == 'red':
                                                        # ตรวจสอบค่าเฉลี่ย YOLO conf
                                                        conf_history = list(self.paths[obj_id].get('yolo_conf_history', []))
                                                        if len(conf_history) >= 3:
                                                            avg_conf = sum(conf_history) / len(conf_history)
                                                        elif len(conf_history) > 0:
                                                            avg_conf = sum(conf_history) / len(conf_history)
                                                        else:
                                                            avg_conf = 0.0

                                                        # ถ้าหาง path ตรงมาก → เปลี่ยนเป็น orange (ไม่ใช่ red)
                                                        # Path ยังคงอยู่ (ไม่หายไป)
                                                        if avg_conf >= YOLO_AVG_DRONE_THRESHOLD:
                                                            self.paths[obj_id]['yolo_state'] = 'orange'
                                                            if LOG_KINEMATIC_PATH:
                                                                print(f"⚠️ Obj {obj_id} Tail path (20 pts) is too straight - changed from RED to ORANGE")
                                                    elif current_yolo_state == 'orange':
                                                        # ถ้าเป็น orange อยู่แล้ว ให้คงไว้ (ไม่ต้องทำอะไร)
                                                        pass
                                                    else:
                                                        # ถ้าเป็น yellow หรืออื่นๆ และหาง path ตรงมาก → เปลี่ยนเป็น orange
                                                        # เพื่อให้เห็นว่า path ยังอยู่แต่ไม่ใช่โดรน
                                                        self.paths[obj_id]['yolo_state'] = 'orange'
                                                        if LOG_KINEMATIC_PATH:
                                                            print(f"⚠️ Obj {obj_id} Tail path (20 pts) is too straight - changed to ORANGE")

                                                    # 🔥 ข้ามการตรวจสอบ drone behavior ถ้าหาง path ตรง (เพื่อป้องกัน kinematic_confirmed กลับเป็น True)
                                                    # Path ยังคงอยู่และต่อได้ แค่เปลี่ยนสี
                                                else:
                                                    # ถ้าหาง path ไม่ตรง → ตรวจสอบ drone behavior ตามปกติ
                                                    sp_for_kin = list(
                                                        self.paths[obj_id].get(
                                                            "smoothed_points",
                                                            self.paths[obj_id]["points"],
                                                        )
                                                    )
                                                    if len(sp_for_kin) >= KIN_MIN_POINTS_BEFORE_DRONE_CONFIRM:
                                                        is_drone, drone_status = self._is_drone_behavior(
                                                            obj_id, frame_center
                                                        )
                                                        if is_drone:
                                                            self.paths[obj_id]['kinematic_confirmed'] = True
                                                            self.paths[obj_id]['drone_status'] = drone_status

                                                            if frame_center is not None:
                                                                threat_level = self._assess_threat_level(obj_id, frame_center)
                                                                if threat_level:
                                                                    self.paths[obj_id]['drone_status'] = threat_level

                                                            if obj_id in self.persistence_paths:
                                                                del self.persistence_paths[obj_id]

    def update_yolo_confidence(self, obj_id, yolo_conf):
        """
        อัปเดต YOLO confidence และ state สำหรับ object
        ใช้ค่าเฉลี่ยหลายเฟรมล่าสุด + ต้องครบประวัติก่อนขึ้น red
        🔥 ตรวจสอบว่า path เป็นเส้นตรงหรือไม่ - ถ้าเป็นเส้นตรงไม่ควรเป็นสีแดง
        """
        if obj_id not in self.paths:
            return

        # อัปเดต max confidence
        if yolo_conf > self.paths[obj_id]['max_yolo_conf']:
            self.paths[obj_id]['max_yolo_conf'] = yolo_conf

        # 🔥 เพิ่มค่า confidence ลงใน history
        if 'yolo_conf_history' not in self.paths[obj_id]:
            self.paths[obj_id]['yolo_conf_history'] = deque(maxlen=YOLO_CONF_HISTORY_MAXLEN)
        self.paths[obj_id]['yolo_conf_history'].append(yolo_conf)

        conf_history = list(self.paths[obj_id]['yolo_conf_history'])
        if len(conf_history) > 0:
            avg_conf = sum(conf_history) / len(conf_history)
        else:
            avg_conf = 0.0

        # 🔥 ตรวจสอบว่า path เป็นเส้นตรงหรือไม่ (ถ้าเป็นเส้นตรงไม่ควรเป็นสีแดง)
        kinematics = self._calculate_kinematics(obj_id)
        if kinematics:
            smoothness_index = kinematics.get('smoothness_index', 0.0)
            velocity_mag = kinematics.get('velocity_magnitude', 0.0)

            # ตรวจสอบ velocity consistency
            path_data = self.paths[obj_id]
            smoothed_pts = path_data.get('smoothed_points', path_data['points'])

            if len(smoothed_pts) >= 5:
                pts_arr = np.array(list(smoothed_pts), dtype=np.float32)
                velocities = []
                for i in range(len(pts_arr) - 1):
                    vel = pts_arr[i+1] - pts_arr[i]
                    velocities.append(np.linalg.norm(vel))

                if len(velocities) >= 3:
                    vel_std = np.std(velocities)
                    vel_mean = np.mean(velocities)

                    if vel_mean > 0:
                        cv = vel_std / vel_mean  # Coefficient of Variation

                        # ❌ ถ้าเส้นตรงมาก + ความเร็วคงที่ = นกหรือเครื่องบินระยะไกล → ไม่ควรเป็นสีแดง
                        if smoothness_index > KIN_TAIL_SMOOTHNESS_REVOKE and cv < KIN_TAIL_CV_REVOKE:
                            # จำกัด yolo_state ไม่ให้เป็น 'red' แม้ avg_conf >= YOLO_AVG_DRONE_THRESHOLD
                            if avg_conf >= YOLO_AVG_DRONE_THRESHOLD:
                                self.paths[obj_id]['yolo_state'] = 'orange'  # ใช้ orange แทน red
                            elif avg_conf >= 0.01:
                                self.paths[obj_id]['yolo_state'] = 'orange'
                            else:
                                self.paths[obj_id]['yolo_state'] = 'yellow'
                            return  # หยุดที่นี่

        # อัปเดต state — red เฉพาะเมื่อประวัติครบ (ลดกระโดดแดงเร็ว)
        if (
            len(conf_history) >= YOLO_RED_MIN_HISTORY
            and avg_conf >= YOLO_AVG_DRONE_THRESHOLD
        ):
            self.paths[obj_id]['yolo_state'] = 'red'
        elif avg_conf >= YOLO_AVG_DRONE_THRESHOLD:
            self.paths[obj_id]['yolo_state'] = 'orange'
        elif avg_conf >= 0.01:
            self.paths[obj_id]['yolo_state'] = 'orange'
        else:
            self.paths[obj_id]['yolo_state'] = 'yellow'

    def draw(self, frame, ui_scale=1.0, show_non_drone_paths=True):
        # show_non_drone_paths: False = วาดเฉพาะ path โดรน (แดง), True = วาดทุก path
        # วาด persistence paths (วัตถุที่หายไปแต่ยัง persist)
        if SHOW_PERSISTENCE_PATHS:
            for pid, pdata in self.persistence_paths.items():
                if pdata.get('kinematic_confirmed', False):
                    # วาด predicted position (scale ตาม resolution)
                    pred_pos = pdata['last_point']
                    _r = max(2, int(5 * ui_scale))
                    _t = max(1, int(2 * ui_scale))
                    cv2.circle(frame, pred_pos, _r, (0, 0, 255), _t)  # Red circle
                    # แสดง label พร้อม background rectangle
                    status = pdata.get('drone_status', 'DRONE_CONFIRMED')
                    label = f"[{pid}] {status}"
                    label_x = pred_pos[0] + 10
                    label_y = pred_pos[1] - 10
                    lbl_scale = max(0.25, 0.6 * ui_scale)
                    lbl_thick = max(1, int(2 * ui_scale))
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, lbl_thick
                    )
                    cv2.rectangle(frame,
                                 (label_x - 2, label_y - text_height - 2),
                                 (label_x + text_width + 2, label_y + baseline + 2),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, label, (label_x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, (0, 0, 255), lbl_thick)

        # วาด paths ปกติ
        for obj_id, data in self.paths.items():
            # กฎเหล็ก: ต้อง Validated แล้วเท่านั้น (ครบ 5 จุด + Smooth)
            if not data['validated']: continue
            # ปิด path ตัวที่ไม่ใช่โดรน (กด V = แสดงเฉพาะโดรน)
            if not show_non_drone_paths and not data.get('kinematic_confirmed', False):
                continue

            sp = data.get("smoothed_points")
            raw_pts = data["points"]
            if sp is not None and len(sp) >= 2:
                draw_pts = list(sp)
            else:
                draw_pts = list(raw_pts)
            if len(draw_pts) < 2:
                continue

            display_pts = draw_pts[-PATH_DISPLAY_MAX_POINTS:]

            # 🔥 ตรวจสอบ kinematic_confirmed ก่อน yolo_state
            if data.get('kinematic_confirmed', False):
                # Path เป็นโดรน → สีแดงเลย (ไม่ต้องรอ YOLO conf)
                path_color = (0, 0, 255)  # Red (BGR)
            else:
                # ใช้ YOLO state
                state = data.get('yolo_state', 'yellow')
                if state == 'red':
                    path_color = (0, 0, 255)  # Red (BGR) - จากค่าเฉลี่ย >= YOLO_AVG_DRONE_THRESHOLD
                elif state == 'orange':
                    path_color = (0, 165, 255)  # Orange (BGR)
                else:
                    path_color = (0, 255, 255)  # Yellow (BGR)

            pts = np.array(display_pts, np.int32).reshape((-1, 1, 2))
            path_thick = max(1, int(2 * ui_scale))
            cv2.polylines(frame, [pts], False, path_color, path_thick, cv2.LINE_AA)
            cv2.circle(frame, display_pts[0], max(1, int(2 * ui_scale)), (0, 200, 200), -1)
            cv2.circle(frame, display_pts[-1], max(1, int(3 * ui_scale)), path_color, -1)

            # ❌ ลบ label ออก (เพราะ main loop มี label อยู่แล้ว)
            # ไม่ต้องวาด label ที่นี่ เพื่อหลีกเลี่ยงการซ้อนกัน

# =============================================================================
# 4. MODULE: Fast Kalman Tracker
# =============================================================================
class FastKalmanTracker:
    def __init__(self, max_disappeared=30, max_distance=150):
        self.next_object_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.kalman_filters = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.matched_status = {}

    def _create_kalman(self, cx, cy):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        return kf

    def update(self, rects):
        self.matched_status = {oid: False for oid in self.objects}
        input_centroids = []
        input_rects = []
        for r in rects:
            x, y, w, h, _ = r
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids.append((cx, cy))
            input_rects.append(r)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_rects[i])
            return self._pack_output()

        object_ids = list(self.objects.keys())
        predictions = []
        for obj_id in object_ids:
            pred = self.kalman_filters[obj_id].predict()
            predictions.append((int(pred[0]), int(pred[1])))

        used_rows = set()
        used_cols = set()

        if len(input_centroids) > 0:
            D = np.zeros((len(object_ids), len(input_centroids)))
            for i in range(len(object_ids)):
                for j in range(len(input_centroids)):
                    dx = predictions[i][0] - input_centroids[j][0]
                    dy = predictions[i][1] - input_centroids[j][1]
                    D[i, j] = np.sqrt(dx*dx + dy*dy)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                obj_id = object_ids[row]
                curr_box = self.boxes[obj_id]
                area = curr_box[2] * curr_box[3]
                # วัตถุเล็ก: อย่าล็อกที่ 50px เมื่อ max_distance ใหญ่ (เฟรมกระตุก)
                if area < 400:
                    allowed_dist = max(55, min(int(self.max_distance * 0.82), self.max_distance))
                else:
                    allowed_dist = self.max_distance
                if D[row, col] > allowed_dist: continue

                self._update_existing(obj_id, input_centroids[col], input_rects[col])
                used_rows.add(row)
                used_cols.add(col)

        for i in range(len(input_centroids)):
            if i not in used_cols: self._register(input_centroids[i], input_rects[i])

        for i in range(len(object_ids)):
            if i not in used_rows:
                obj_id = object_ids[i]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
                else:
                    # ไม่เลื่อนกล่องตามค่าทาย — คงกล่องล่าสุดจาก measurement จริง (ไม่ให้กล่องนำหน้า blob)
                    # correct Kalman ด้วยจุดกลางกล่องเดิม เพื่อไม่ให้ state/velocity สะสมนำหน้าเมื่อไม่มี detection
                    ox, oy, ow, oh, _ = self.boxes[obj_id]
                    cx = np.float32(ox + ow / 2.0)
                    cy = np.float32(oy + oh / 2.0)
                    meas = np.array([[cx], [cy]], np.float32)
                    self.kalman_filters[obj_id].correct(meas)
                    self.objects[obj_id] = (int(cx), int(cy))

        return self._pack_output()

    def _pack_output(self):
        output = {}
        for oid, box in self.boxes.items():
            output[oid] = (box, self.matched_status.get(oid, False))
        return output

    def _register(self, centroid, rect):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.kalman_filters[self.next_object_id] = self._create_kalman(centroid[0], centroid[1])
        self.matched_status[self.next_object_id] = True
        self.next_object_id += 1

    def register(self, rect):
        """Register a single detection as a new object (e.g. from YOLO). rect = (x, y, w, h, _). Returns (obj_id, box)."""
        x, y, w, h, _ = rect
        cx = int(x + w / 2.0)
        cy = int(y + h / 2.0)
        obj_id = self.next_object_id
        self._register((cx, cy), rect)
        return (obj_id, self.boxes[obj_id])

    def _update_existing(self, obj_id, centroid, rect):
        self.disappeared[obj_id] = 0
        self.objects[obj_id] = centroid
        self.boxes[obj_id] = rect
        kf = self.kalman_filters[obj_id]
        meas = np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]])
        kf.correct(meas)
        self.matched_status[obj_id] = True

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.boxes[obj_id]
        del self.disappeared[obj_id]
        del self.kalman_filters[obj_id]
        if obj_id in self.matched_status: del self.matched_status[obj_id]

# =============================================================================
# 5. MODULE: Adaptive Grid Filter
# =============================================================================
class AdaptiveSensitivityGrid:
    def __init__(self, frame_width, frame_height, grid_rows=8, grid_cols=16):
        self.w = frame_width
        self.h = frame_height
        self.rows = grid_rows
        self.cols = grid_cols
        self.cell_w = self.w // self.cols
        self.cell_h = self.h // self.rows
        self.grid_area_size = self.cell_w * self.cell_h
        self.noise_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.heat_timer = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.DECAY_RATE = 2.0
        self.HEAT_INCREMENT = 10.0
        self.MAX_HEAT = 100.0
        self.NOISE_THRESHOLD = 30.0
        self.MAX_OBJECTS_PER_ZONE = 4
        self.MIN_AREA_GLOBAL = 20

    def update(self):
        self.noise_map -= self.DECAY_RATE
        self.noise_map = np.clip(self.noise_map, 0, self.MAX_HEAT)
        hot_zones = self.noise_map > 90.0
        self.heat_timer[hot_zones] += 1
        cool_zones = self.noise_map < 5.0
        self.heat_timer[cool_zones] = 0

    def filter_and_update(self, detections, min_area=None):
        """
        Args:
            detections: List of (x, y, w, h, area)
            min_area: Minimum area threshold (default: self.MIN_AREA_GLOBAL).
                req_area tiers below scale from this (sky min_area=2 → small blobs pass;
                ground min_area=20 → same behavior as legacy MIN_AREA_GLOBAL).
        """
        if min_area is None:
            min_area = self.MIN_AREA_GLOBAL
        base_area = max(int(min_area), 1)

        valid_detections = []
        congested_zones = []
        frame_activity = np.zeros_like(self.noise_map)
        grid_buckets = {}

        for det in detections:
            x, y, w_obj, h_obj, area = det
            if area < min_area: continue
            cx = x + w_obj // 2; cy = y + h_obj // 2
            c_idx = min(max(cx // self.cell_w, 0), self.cols - 1)
            r_idx = min(max(cy // self.cell_h, 0), self.rows - 1)
            if (r_idx, c_idx) not in grid_buckets: grid_buckets[(r_idx, c_idx)] = []
            grid_buckets[(r_idx, c_idx)].append(det)

        for (r, c), dets in grid_buckets.items():
            current_noise = self.noise_map[r, c]
            current_timer = self.heat_timer[r, c]

            # Level 1: > 60 frames (Strict Count)
            is_lvl1_strict = current_timer > 60
            # Level 2: > 120 frames (Large Size Only)
            is_lvl2_large = current_timer > 120
            # 🔥 Level 3: > 150 frames (Sleep Mode - เข้าไวขึ้น)
            is_lvl3_sleep = current_timer > 150

            # =========================================================
            # 🛑 LOGIC การกรองแบบเด็ดขาด
            # =========================================================

            # CASE 1: SLEEP MODE (ห้ามวาดอะไรเลย เว้นแต่จะเป็น Giant)
            if is_lvl3_sleep:
                found_giant = False
                for det in dets:
                    _, _, w_box, h_box, area = det

                    # กฎข้อที่ 1: ต้องใหญ่กว่า 50% ของช่อง Grid
                    is_giant_size = area > (self.grid_area_size * 0.5)

                    # 🔥 กฎข้อที่ 2: ความหนาแน่น (Density) ต้องสูง
                    # ป้องกัน "พุ่มไม้ฟูๆ" ที่ขนาดใหญ่แต่เนื้อกลวง
                    box_area = w_box * h_box
                    density = area / box_area if box_area > 0 else 0
                    is_solid = density > 0.45

                    if is_giant_size and is_solid:
                        valid_detections.append(det)
                        found_giant = True

                # ถ้าเจอ Giant ให้เลี้ยงความร้อนไว้ (จะได้ไม่หลุด Sleep)
                if found_giant: frame_activity[r, c] = 1.0

                # ⛔ จบ Loop ทันที ตัดวงจรทุกอย่างใน Grid นี้
                continue

            # CASE 2: High Activity (กล่องส้ม)
            # แต่ถ้าเริ่มเข้า Level 2 (ร้อนนานเกิน 4 วิ) จะเลิกวาดกล่องส้มแล้ว (เพราะมันรก)
            if len(dets) > self.MAX_OBJECTS_PER_ZONE:
                if not is_lvl2_large:
                    x1 = c * self.cell_w
                    y1 = r * self.cell_h
                    congested_zones.append((x1, y1, x1+self.cell_w, y1+self.cell_h))

                # ถึงไม่วาด ก็ต้องเพิ่ม Heat
                frame_activity[r, c] = 2.0

            # CASE 3: Normal / Strict Filtering
            else:
                if is_lvl1_strict and len(dets) != 1:
                    continue

                for det in dets:
                    _, _, w_box, h_box, area = det

                    # 1. Base Requirement (scale with caller min_area — not always MIN_AREA_GLOBAL)
                    if current_noise < self.NOISE_THRESHOLD:
                        req_area = base_area
                    else:
                        req_area = base_area * 2

                    # 2. Strict Size (Level 2)
                    if is_lvl2_large:
                        # บังคับขนาด 5 เท่า AND ความหนาแน่นต้องได้
                        req_area = max(req_area, base_area * 5)

                        box_area = w_box * h_box
                        density = area / box_area if box_area > 0 else 0
                        if density < 0.4:  # ถ้าใหญ่แต่กลวง (เช่น เงาไม้) ไม่เอา
                            continue

                    if area >= req_area:
                        valid_detections.append(det)
                        frame_activity[r, c] = 1.0

        self.noise_map += (frame_activity * self.HEAT_INCREMENT)
        self.noise_map = np.clip(self.noise_map, 0, self.MAX_HEAT)
        return valid_detections, congested_zones

    def draw_debug(self, frame):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.noise_map[r, c] > self.NOISE_THRESHOLD:
                    x1 = c * self.cell_w; y1 = r * self.cell_h
                    h_slice = min(self.cell_h, frame.shape[0] - y1)
                    w_slice = min(self.cell_w, frame.shape[1] - x1)
                    if h_slice > 0 and w_slice > 0:
                        overlay = frame[y1:y1+h_slice, x1:x1+w_slice]
                        white_rect = np.full(overlay.shape, (0, 0, 200), dtype=np.uint8)
                        cv2.addWeighted(overlay, 0.4, white_rect, 0.6, 0, overlay)
                        frame[y1:y1+h_slice, x1:x1+w_slice] = overlay

# =============================================================================
# Arm reachable overlay + calibration sync helpers
# =============================================================================
def _load_arm_reachable_overlay(calib_path):
    """Load reachable polygon + calibration resolution from cam8 calibration JSON."""
    try:
        if not os.path.isfile(calib_path):
            return None, None
        with open(calib_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        poly = cal_data.get("reachable_polygon_cam8")
        if (not poly) and isinstance(cal_data.get("arm_limit_corners"), list):
            poly = [[c.get("cam8_px"), c.get("cam8_py")] for c in cal_data.get("arm_limit_corners", [])]
        ow = int(cal_data.get("output_width", 0))
        oh = int(cal_data.get("output_height", 0))
        if isinstance(poly, list) and len(poly) >= 3 and ow > 0 and oh > 0:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            return pts, (ow, oh)
    except Exception as e:
        print(f"[ARM-OVERLAY] failed to parse calibration: {e}")
    return None, None


def _sync_arm_calibration_once(local_path):
    """Try one remote->local pull via rsync/scp. Returns True if file exists after sync."""
    remote = f"{ARM_CALIB_SYNC_REMOTE_USER}@{ARM_CALIB_SYNC_REMOTE_HOST}:{ARM_CALIB_SYNC_REMOTE_PATH}"
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except Exception:
        pass
    cmds = [
        ["rsync", "-az", "--timeout=3", remote, local_path],
        ["scp", "-q", remote, local_path],
    ]
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0, check=False)
            if r.returncode == 0 and os.path.isfile(local_path):
                return True
        except Exception:
            continue
    return False


# =============================================================================
# 6. MAIN LOOP
# =============================================================================
def main(camera_override=None, active_camera_name=None):
    global horizon_points, temp_draw_points, drawing_mode, drawing_target
    global current_frame_w, current_frame_h, display_w, display_h
    global label_trainer_global
    global last_raw_frame, prev_raw_frame
    global YOLO_START_PX
    global ignore_motion_preview_mask, ignore_motion_brush_thickness
    global ignore_motion_last_point, ignore_motion_is_painting, ignore_motion_is_erasing

    init_memory_management()
    camera_name = active_camera_name or LOCAL_ACTIVE_CAMERA
    cam = camera_override if camera_override is not None else build_camera(camera_name)
    cam.start()

    # ตรวจสอบว่าเปิดกล้องได้จริง ถ้าไม่ได้ให้ exit ก่อนโหลด YOLO (ป้องกัน CUDA OOM / segfault)
    for _ in range(90):
        active, frame, _ = cam.read()
        if active and frame is not None:
            break
        time.sleep(0.033)
    else:
        print("❌ Camera/Video failed to open. Exiting.")
        if hasattr(cam, "release"):
            cam.release()
        return

    grid_filter = None
    tracker = None
    graph_manager = None
    sound_alert = None  # Sound alert system
    label_trainer = None  # Active label trainer
    sound_confirm_counter = 0
    last_yolo_full_dets = []
    yolo_open_cells = set()
    yolo_open_cells_ttl = 0
    handoff_state = {}

    # Websocket exporter — สร้างต่อ camera instance และรันใน background thread
    _ws_cam_geo = _get_camera_geo(camera_name)
    _ws_cam_config = get_camera_config(camera_name)
    _ws_url = WS_EXPORT_URL_TEMPLATE.format(camera_id=_ws_cam_geo.get("camera_id", camera_name.upper()))
    ws_exporter = AntidroneWsExporter(
        camera_geo=_ws_cam_geo,
        cam_config=_ws_cam_config,
        ws_url=_ws_url,
        interval=WS_EXPORT_INTERVAL_SEC,
        backoff_init=WS_EXPORT_BACKOFF_INIT_SEC,
        backoff_max=WS_EXPORT_BACKOFF_MAX_SEC,
        enabled=WS_EXPORT_ENABLED,
    )
    ws_exporter.start()

    # Arm cue sender: ส่ง confirmed target ไปยัง Jetson แขน (22_gun_aim_assist_vector.py)
    cue_sender = None
    _arm_cue_enabled_for_this_worker = (
        ArmCueSender is not None
        and ARM_CUE_ENABLED
        and str(camera_name).lower() == str(ARM_CUE_SOURCE_CAMERA).lower()
    )
    if _arm_cue_enabled_for_this_worker:
        cue_sender = ArmCueSender(
            host=ARM_CUE_HOST,
            port=ARM_CUE_PORT,
            send_interval_hz=ARM_CUE_SEND_HZ,
            cue_ttl_ms=ARM_CUE_TTL_MS,
            source_camera=camera_name,
            enabled=True,
        )
        cue_sender.start()
    elif ArmCueSender is not None and ARM_CUE_ENABLED:
        print(
            f"[ArmCueSender] disabled for camera={camera_name} "
            f"(allowed source={ARM_CUE_SOURCE_CAMERA})"
        )

    # Optional: overlay reachable arm zone from cam8 calibration on cam8 feed.
    arm_reachable_poly_cam8 = None
    arm_reachable_calib_wh = None
    arm_overlay_lock = threading.Lock()
    arm_overlay_sync_done = False
    arm_overlay_sync_thread = None
    arm_overlay_stop_event = threading.Event()
    _calib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration_data",
        "cam8_mouse_grid_lookup.json",
    )
    if ARM_REACHABLE_OVERLAY_ENABLED and str(camera_name).lower() == "cam8":
        _pts, _wh = _load_arm_reachable_overlay(_calib_path)
        if _pts is not None and _wh is not None:
            with arm_overlay_lock:
                arm_reachable_poly_cam8 = _pts
                arm_reachable_calib_wh = _wh
            arm_overlay_sync_done = True
            print(f"[ARM-OVERLAY] loaded reachable polygon ({len(_pts)} pts) from local calibration")
        elif ARM_CALIB_SYNC_ENABLED:
            print(f"[ARM-OVERLAY] local calibration unavailable, auto-sync from {ARM_CALIB_SYNC_REMOTE_HOST}...")

            def _sync_worker():
                nonlocal arm_reachable_poly_cam8, arm_reachable_calib_wh, arm_overlay_sync_done
                retries = 0
                while (not arm_overlay_stop_event.is_set()) and retries < ARM_CALIB_SYNC_MAX_RETRIES and not arm_overlay_sync_done:
                    ok = _sync_arm_calibration_once(_calib_path)
                    if ok:
                        pts, wh = _load_arm_reachable_overlay(_calib_path)
                        if pts is not None and wh is not None:
                            with arm_overlay_lock:
                                arm_reachable_poly_cam8 = pts
                                arm_reachable_calib_wh = wh
                            arm_overlay_sync_done = True
                            print(f"[ARM-OVERLAY] auto-sync complete ({len(pts)} pts); stop pulling")
                            break
                    wait_s = ARM_CALIB_SYNC_BACKOFF_SEC[min(retries, len(ARM_CALIB_SYNC_BACKOFF_SEC) - 1)]
                    retries += 1
                    arm_overlay_stop_event.wait(wait_s)
                if not arm_overlay_sync_done and retries >= ARM_CALIB_SYNC_MAX_RETRIES:
                    print("[ARM-OVERLAY] auto-sync stopped (max retries reached), using local-only mode")

            arm_overlay_sync_thread = threading.Thread(target=_sync_worker, daemon=True, name="arm_overlay_sync")
            arm_overlay_sync_thread.start()

    DETECTION_THRESHOLD = 10
    merge_kernel = None

    initialized = False
    initialized_w = 0
    initialized_h = 0
    gpu_bufs = []
    tmp_diff = None; tmp_res = None; tmp_raw = None
    gpu_display_frame = None
    gpu_resized_frame = None
    # GPU buffers สำหรับ threshold และ mask operations
    gpu_sky_mask = None
    gpu_ground_mask = None
    gpu_mask_sky = None
    gpu_mask_ground = None
    gpu_mask_final = None
    # GPU buffers สำหรับ downsampled processing
    gpu_mask_downsampled = None
    gpu_mask_dilated_downsampled = None
    gpu_mask_sky_ds = None
    gpu_mask_ground_ds = None
    gpu_sky_mask_downsampled = None
    gpu_ground_mask_downsampled = None
    processing_scale_factor = 1.0
    processing_w = 0
    processing_h = 0

    HORIZON_FILE = "horizon_poly.npy"
    IGNORE_MOTION_FILE = "motion_ignore_mask.npy"
    IGNORE_MOTION_LINE_THICKNESS = 18
    HORIZON_MOTION_BUFFER_PX = 12
    WINDOW_NAME = "Horizon Tracker V8 (Smooth Validator)"
    sky_mask = None
    motion_sky_mask = None
    ignore_motion_mask = None
    ignore_motion_mask_downsampled = None
    ignore_motion_allow_mask_downsampled = None
    motion_sky_mask_downsampled = None
    show_boxes = False
    show_non_drone_paths = False  # default: เฉพาะโดรน; กด V = แสดง path/bbox อย่างอื่นด้วย
    show_guides = True  # G = toggle horizon + arm limit guides
    show_heavy_ui = True  # H = toggle heavy HUD/thumbnail overlays for better FPS
    show_profile_hud = False  # F = toggle per-stage timing (motion / YOLO / track+graph / display)

    # FPS and Latency tracking
    fps_times = deque(maxlen=30)  # Store last 30 frame times for FPS calculation
    current_fps = 0.0
    frame_start_time = None
    prof_motion_ms = prof_yolo_ms = prof_track_ms = prof_disp_ms = 0.0
    PROF_AVG_FRAMES = 20
    prof_hist_motion = deque(maxlen=PROF_AVG_FRAMES)
    prof_hist_yolo = deque(maxlen=PROF_AVG_FRAMES)
    prof_hist_track = deque(maxlen=PROF_AVG_FRAMES)
    prof_hist_disp = deque(maxlen=PROF_AVG_FRAMES)

    # Memory management
    frame_counter = 0
    PERIODIC_MEMORY_CLEANUP = 200  # Cleanup every 200 frames

    # Session drone count (per zone): นับโดรนที่ยืนยันแล้วใน session + ลดนับซ้ำตัวเดิมบินไปมา
    SESSION_DRONE_T_LOST_SEC = 18  # วินาที: ถ้าหายไปนานกว่านี้ถือว่าลำใหม่
    SESSION_DRONE_MAX_NEAR_PX = 250  # พิกเซล: โดรนใหม่อยู่ใกล้ตำแหน่ง "เพิ่งหาย" ภายในนี้ถือว่าอาจเป็นตัวเดิม
    SESSION_DRONE_RECENTLY_LOST_MAX = 25  # จำกัดจำนวนรายการ recently_lost
    SESSION_PATH_RED_STABLE_FRAMES = 28  # path ต้องแดงต่อเนื่องครบ N เฟรมถึงนับ session (เดิม 20)
    SESSION_YOLO_ONLY_MIN_HITS = 2  # YOLO-only ต้องเจอในบริเวณนั้นอย่างน้อย N ครั้งถึงนับ
    SESSION_YOLO_PENDING_WINDOW_SEC = 1.5  # หน้าต่างเวลา (วินาที) สำหรับนับ hit YOLO-only
    confirmed_drone_ids_this_session = set()
    session_drone_count = 0
    recently_lost_red_tracks = []  # list of (last_cx, last_cy, lost_time)
    # YOLO-only: ตำแหน่งที่เคยนับ session แล้ว (เจอๆ หายๆ ไม่นับซ้ำ)
    yolo_only_recent_positions = []  # list of (cx, cy, time)
    # Path: นับเฟรมที่ path แดงต่อเนื่อง (obj_id -> เฟรม)
    red_consecutive_frames = {}
    # YOLO-only: รอยืนยันก่อนนับ (cx, cy, time, count)
    yolo_only_pending = []  # list of (cx, cy, time, count)
    # YOLO-only distance EMA keyed by coarse pixel cell (no track id)
    yolo_ws_dist_ema = {}
    yolo_ws_ema_last_frame = {}

    # YOLO model
    yolo_model = None
    yolo_frame_counter = 0
    if YOLO_AVAILABLE:
        base_dir = os.path.dirname(__file__)
        model_path, model_name = select_yolo_model_path(base_dir)
        if model_path and os.path.exists(model_path):
            try:
                yolo_input_size = YOLO_INPUT_SIZE
                if model_name:
                    if "imgsz640" in model_name:
                        yolo_input_size = 640
                    elif "imgsz1280" in model_name:
                        yolo_input_size = 1280
                globals()["YOLO_INPUT_SIZE"] = yolo_input_size
                yolo_model = YOLO(model_path, task='detect')
                # Warm up
                dummy_frame = np.zeros((yolo_input_size, yolo_input_size, 3), dtype=np.uint8)
                yolo_model(dummy_frame, imgsz=yolo_input_size, verbose=False, device=0, half=True)
                print(f"✅ YOLO model loaded ({model_name})")
            except Exception as e:
                print(f"⚠️ YOLO model load failed: {e}")
        else:
            print(f"⚠️ YOLO model file not found: {model_name}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = _get_screen_size()
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_global)

    if ACTIVE_LABEL_AVAILABLE:
        try:
            label_trainer = ActiveLabelTrainer(
                dataset_dir=ACTIVE_LABEL_DATA_DIR,
                model_filename=ACTIVE_LABEL_MODEL_FILENAME,
                model_dir=ACTIVE_LABEL_MODEL_DIR,
                min_samples=ACTIVE_LABEL_MIN_SAMPLES,
                prompt_enabled=ACTIVE_LABELING_DEFAULT,
                confirm_conf_threshold=ACTIVE_LABEL_CONFIRM_CONF,
                confirm_frames=ACTIVE_LABEL_CONFIRM_FRAMES,
            )
            print("Active Label Trainer: ready (t=all, u=uncertain, 1-4=label, 0=skip)")
        except Exception as e:
            print(f"Active Label Trainer init failed: {e}")
            label_trainer = None

    label_trainer_global = label_trainer

    def rebuild_sky_mask(h, w, points):
        """ใช้แค่ส่วนที่อยู่บนเส้นขอบฟ้า (ระหว่างจุดแรก–จุดสุดท้าย) เป็นบริเวณท้องฟ้า ไม่เติมทั้งความกว้างเฟรม"""
        mask = np.zeros((h, w), dtype=np.uint8)
        if not points or len(points) < 2:
            return None
        pts = np.array(points, dtype=np.int32)
        first_pt, last_pt = pts[0].copy(), pts[-1].copy()
        x_left = int(np.clip(first_pt[0], 0, w - 1))
        x_right = int(np.clip(last_pt[0], 0, w - 1))
        upper_poly = np.vstack([
            [x_left, 0], [x_right, 0],
            [last_pt[0], last_pt[1]], pts[::-1], [first_pt[0], first_pt[1]]
        ])
        cv2.fillPoly(mask, [upper_poly], 255)
        return mask

    def rebuild_motion_sky_mask(base_sky_mask, buffer_px):
        """mask สำหรับ motion เท่านั้น: อนุญาตเฉพาะเหนือเส้น และถอยขึ้นอีกเล็กน้อยกัน motion รั่วใกล้เส้น."""
        if base_sky_mask is None:
            return None
        motion_mask = base_sky_mask.copy().astype(np.uint8)
        buffer_px = max(0, int(buffer_px))
        if buffer_px <= 0:
            return motion_mask
        kernel_size = (buffer_px * 2) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        return motion_mask

    def ensure_mask_size(mask, h, w):
        if mask is None:
            return None
        if mask.shape[:2] == (h, w):
            return mask.astype(np.uint8)
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    def rebuild_ignore_motion_processing_masks():
        nonlocal ignore_motion_mask_downsampled, ignore_motion_allow_mask_downsampled
        if ignore_motion_mask is None or processing_w <= 0 or processing_h <= 0:
            ignore_motion_mask_downsampled = None
            ignore_motion_allow_mask_downsampled = None
            return
        ignore_motion_mask_downsampled = cv2.resize(
            ignore_motion_mask,
            (processing_w, processing_h),
            interpolation=cv2.INTER_NEAREST
        )
        ignore_motion_allow_mask_downsampled = cv2.bitwise_not(ignore_motion_mask_downsampled)

    def ray_intersect_frame(px, py, dx, dy, w, h):
        """หาจุดที่รังสีจาก (px,py) ตามทิศ (dx,dy) ชนขอบเฟรม [0,w-1]x[0,h-1] ครั้งแรก (t>0)"""
        x, y = float(px), float(py)
        dx, dy = float(dx), float(dy)
        t_candidates = []
        if abs(dx) > 1e-9:
            t0 = (0 - x) / dx
            if t0 > 1e-9:
                t_candidates.append(t0)
            t1 = (w - 1 - x) / dx
            if t1 > 1e-9:
                t_candidates.append(t1)
        if abs(dy) > 1e-9:
            t2 = (0 - y) / dy
            if t2 > 1e-9:
                t_candidates.append(t2)
            t3 = (h - 1 - y) / dy
            if t3 > 1e-9:
                t_candidates.append(t3)
        if not t_candidates:
            return (int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1))))
        t_min = min(t_candidates)
        x2 = x + t_min * dx
        y2 = y + t_min * dy
        return (int(round(np.clip(x2, 0, w - 1))), int(round(np.clip(y2, 0, h - 1))))

    def extend_horizon_to_edges(points, w, h):
        """ต่อเส้นขอบฟ้าจากปลายทั้งสองตามทิศที่เส้นชี้ จนชนขอบเฟรม (ไม่ต่อไปซ้าย/ขวา x=0,w-1 โดยตรง)"""
        if not points or len(points) < 2:
            return points
        pts = np.array(points, dtype=np.float64)
        # ทิศจากจุดที่สองไปจุดแรก = ทิศที่ "ออกจากปลายซ้าย"
        x0, y0 = pts[0][0], pts[0][1]
        x1, y1 = pts[1][0], pts[1][1]
        dx_left = x0 - x1
        dy_left = y0 - y1
        norm_left = (dx_left * dx_left + dy_left * dy_left) ** 0.5
        if norm_left < 1e-9:
            p_left = (int(round(x0)), int(round(np.clip(y0, 0, h - 1))))
        else:
            dx_left /= norm_left
            dy_left /= norm_left
            p_left = ray_intersect_frame(x0, y0, dx_left, dy_left, w, h)

        # ทิศจากจุดรองสุดท้ายไปจุดสุดท้าย = ทิศที่ "ออกจากปลายขวา"
        xn2, yn2 = pts[-2][0], pts[-2][1]
        xn1, yn1 = pts[-1][0], pts[-1][1]
        dx_right = xn1 - xn2
        dy_right = yn1 - yn2
        norm_right = (dx_right * dx_right + dy_right * dy_right) ** 0.5
        if norm_right < 1e-9:
            p_right = (int(round(xn1)), int(round(np.clip(yn1, 0, h - 1))))
        else:
            dx_right /= norm_right
            dy_right /= norm_right
            p_right = ray_intersect_frame(xn1, yn1, dx_right, dy_right, w, h)

        return [list(p_left)] + [p.tolist() if hasattr(p, "tolist") else list(p) for p in points] + [list(p_right)]

    if os.path.exists(HORIZON_FILE):
        try:
            horizon_points = np.load(HORIZON_FILE).astype(np.int32).tolist()
            print(f"✅ Loaded horizon: {len(horizon_points)} pts")
        except: pass

    if os.path.exists(IGNORE_MOTION_FILE):
        try:
            loaded_ignore_mask = np.load(IGNORE_MOTION_FILE)
            ignore_motion_mask = loaded_ignore_mask.astype(np.uint8)
            print(f"✅ Loaded motion ignore mask: {ignore_motion_mask.shape[1]}x{ignore_motion_mask.shape[0]}")
        except:
            ignore_motion_mask = None

    try:
        while True:
            frame_start_time = time.time()  # Record frame capture start time
            frame_counter += 1  # Increment frame counter for periodic cleanup
            active, frame, _ = cam.read()
            if not active or frame is None:
                time.sleep(0.01)
                continue

            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            raw_frame = frame.copy()
            prev_raw_frame = last_raw_frame
            last_raw_frame = raw_frame

            h, w = frame.shape[:2]
            current_frame_w, current_frame_h = w, h
            # Fullscreen fit: ขนาดแสดงผลพอดีจอ (path/การคำนวณยังใช้เฟรมต้นฉบับ w,h)
            scale = min(screen_w / w, screen_h / h) if w > 0 and h > 0 else 1.0
            display_w = int(w * scale)
            display_h = int(h * scale)
            display_w = max(1, display_w)
            display_h = max(1, display_h)
            try:
                cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
            except cv2.error:
                pass
            # UI scale ตาม resolution (ใช้ทั้ง HUD และ label)
            ui_scale = min(w / 1920.0, h / 1080.0)
            ui_scale = max(0.2, min(1.2, ui_scale))

            # Dynamic GPU Buffer Reallocation - ตรวจสอบและ reallocate เมื่อ resolution เปลี่ยน
            if not initialized or w != initialized_w or h != initialized_h:
                # Cleanup old buffers ถ้ามี
                if initialized:
                    try:
                        del gpu_bufs
                        if tmp_diff is not None: del tmp_diff
                        if tmp_res is not None: del tmp_res
                        if tmp_raw is not None: del tmp_raw
                        if gpu_display_frame is not None: del gpu_display_frame
                        if gpu_resized_frame is not None: del gpu_resized_frame
                        if gpu_sky_mask is not None: del gpu_sky_mask
                        if gpu_ground_mask is not None: del gpu_ground_mask
                        if gpu_mask_sky is not None: del gpu_mask_sky
                        if gpu_mask_ground is not None: del gpu_mask_ground
                        if gpu_mask_final is not None: del gpu_mask_final
                        if gpu_mask_downsampled is not None: del gpu_mask_downsampled
                        if gpu_mask_dilated_downsampled is not None: del gpu_mask_dilated_downsampled
                        if gpu_mask_sky_ds is not None: del gpu_mask_sky_ds
                        if gpu_mask_ground_ds is not None: del gpu_mask_ground_ds
                        if gpu_sky_mask_downsampled is not None: del gpu_sky_mask_downsampled
                        if gpu_ground_mask_downsampled is not None: del gpu_ground_mask_downsampled
                    except:
                        pass
                    gc.collect()

                TARGET_GRID_SIZE = 120
                cols = max(4, int(w / TARGET_GRID_SIZE))
                rows = max(4, int(h / TARGET_GRID_SIZE))

                YOLO_START_PX = compute_yolo_start_px(w, h)

                grid_filter = AdaptiveSensitivityGrid(w, h, grid_rows=rows, grid_cols=cols)
                # Fast Cleanup (10 frames = 0.3s)
                # Gate ใหญ่ขึ้นตามความละเอียด — ลดโอกาสแยก ID เมื่อเฟรมกระตุก/กระโดด
                _track_match_dist = max(80, min(280, int(0.07 * min(w, h))))
                tracker = FastKalmanTracker(max_disappeared=30, max_distance=_track_match_dist)
                graph_manager = GraphTrajectoryManager(
                    max_nodes=PATH_MAX_NODES,
                    min_dist=PATH_MIN_DIST_PX,
                    enable_kinematic_rules=ENABLE_KINEMATIC_RULES,
                    smooth_window=PATH_SMOOTH_WINDOW,
                )
                merge_kernel = np.ones((9, 9), np.uint8)
                ignore_motion_brush_thickness = max(8, int(min(w, h) * 0.012))

                # Initialize sound alert system (ถ้ายังไม่ได้ initialize)
                if sound_alert is None:
                    try:
                        sound_alert = SoundAlert("alarm_loud.wav")
                        sound_alert.start()
                    except Exception as e:
                        print(f"⚠️ Sound Alert initialization failed: {e}")
                        sound_alert = None

                if horizon_points:
                    sky_mask = rebuild_sky_mask(h, w, horizon_points)
                    motion_sky_mask = rebuild_motion_sky_mask(sky_mask, HORIZON_MOTION_BUFFER_PX)
                else:
                    motion_sky_mask = None
                if ignore_motion_mask is not None:
                    ignore_motion_mask = ensure_mask_size(ignore_motion_mask, h, w)
                if drawing_mode and drawing_target == "ignore_motion":
                    if ignore_motion_preview_mask is not None:
                        ignore_motion_preview_mask = ensure_mask_size(ignore_motion_preview_mask, h, w)
                    else:
                        ignore_motion_preview_mask = (
                            ignore_motion_mask.copy()
                            if ignore_motion_mask is not None
                            else np.zeros((h, w), dtype=np.uint8)
                        )

                # คำนวณ adaptive scale factor สำหรับ downsampling
                # ใช้ 0.75 สำหรับวัตถุเล็ก (min_area <= 2) เพื่อรักษาความแม่นยำ
                # ใช้ 0.5 สำหรับวัตถุใหญ่ (min_area > 2) เพื่อความเร็ว
                MIN_AREA_GLOBAL = grid_filter.MIN_AREA_GLOBAL
                if MIN_AREA_GLOBAL <= 2:
                    processing_scale_factor = 0.75  # รักษา 1-2 pixel objects
                else:
                    processing_scale_factor = 0.5  # เร็วขึ้น 4 เท่า

                processing_w = int(w * processing_scale_factor)
                processing_h = int(h * processing_scale_factor)
                if motion_sky_mask is not None:
                    motion_sky_mask_downsampled = cv2.resize(
                        motion_sky_mask,
                        (processing_w, processing_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    motion_sky_mask_downsampled = None
                rebuild_ignore_motion_processing_masks()

                try:
                    gpu_bufs = [cv2.cuda_GpuMat(h, w, cv2.CV_8UC1) for _ in range(3)]
                    tmp_diff = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    tmp_res = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    tmp_raw = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
                    gpu_display_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
                    gpu_resized_frame = cv2.cuda_GpuMat(display_h, display_w, cv2.CV_8UC3)
                    # GPU buffers สำหรับ threshold และ mask operations
                    gpu_mask_sky = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_mask_ground = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_mask_final = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    # GPU buffers สำหรับ downsampled processing
                    gpu_mask_downsampled = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                    gpu_mask_dilated_downsampled = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                    # Temporary buffers สำหรับ downsampled threshold operations
                    gpu_mask_sky_ds = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                    gpu_mask_ground_ds = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                    # Upload motion_sky_mask ขึ้น GPU ถ้ามี (ใช้เฉพาะ motion detection)
                    if motion_sky_mask is not None:
                        gpu_sky_mask = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                        gpu_sky_mask.upload(motion_sky_mask)
                        # สร้าง ground_mask บน GPU ไว้เฉยๆ เผื่อใช้ต่อภายหลัง
                        gpu_ground_mask = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                        cv2.cuda.bitwise_not(gpu_sky_mask, gpu_ground_mask)
                        # สร้าง downsampled motion sky mask สำหรับ threshold operations
                        gpu_sky_mask_downsampled = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                        gpu_ground_mask_downsampled = cv2.cuda_GpuMat(processing_h, processing_w, cv2.CV_8UC1)
                        cv2.cuda.resize(gpu_sky_mask, (processing_w, processing_h), gpu_sky_mask_downsampled, interpolation=cv2.INTER_NEAREST)
                        cv2.cuda.resize(gpu_ground_mask, (processing_w, processing_h), gpu_ground_mask_downsampled, interpolation=cv2.INTER_NEAREST)
                    else:
                        gpu_sky_mask = None
                        gpu_ground_mask = None
                        gpu_sky_mask_downsampled = None
                        gpu_ground_mask_downsampled = None
                    # dilate kernel ใช้ numpy array ปกติ (OpenCV CUDA รับ numpy array)
                    # ไม่ต้องสร้างใหม่ เพราะ merge_kernel เป็น numpy array อยู่แล้ว
                    was_initialized = initialized
                    initialized_w, initialized_h = w, h
                    initialized = True
                    if was_initialized:
                        print(f"✅ Reinitialized: {w}x{h}")
                    else:
                        print(f"✅ Init: {w}x{h}")
                except Exception as e:
                    print(f"❌ GPU initialization error: {e}")
                    break

            try:
                prof_motion_ms = prof_yolo_ms = prof_track_ms = prof_disp_ms = 0.0
                _t_prof_m0 = time.perf_counter()
                tmp_raw.upload(frame)
                # Shift buffers: ใช้ copyTo() แทน download/upload เพื่อความเร็ว
                gpu_bufs[1].copyTo(gpu_bufs[2])
                gpu_bufs[0].copyTo(gpu_bufs[1])
                cv2.cuda.cvtColor(tmp_raw, cv2.COLOR_BGR2GRAY, gpu_bufs[0])
                cv2.cuda.absdiff(gpu_bufs[0], gpu_bufs[1], tmp_diff)
                cv2.cuda.absdiff(gpu_bufs[1], gpu_bufs[2], tmp_res)
                cv2.cuda.bitwise_and(tmp_diff, tmp_res, tmp_res)

                # Downsample tmp_res ก่อน threshold เพื่อความเร็ว (ใช้ INTER_NEAREST เพื่อรักษา edges)
                cv2.cuda.resize(tmp_res, (processing_w, processing_h), gpu_mask_downsampled, interpolation=cv2.INTER_NEAREST)

                # Threshold และ mask operations บน GPU (ใช้ downsampled mask)
                if gpu_sky_mask_downsampled is not None:
                    # motion detection: ใช้เฉพาะพื้นที่เหนือเส้นขอบฟ้า + buffer เท่านั้น
                    cv2.cuda.threshold(gpu_mask_downsampled, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY, gpu_mask_sky_ds)
                    cv2.cuda.bitwise_and(gpu_mask_sky_ds, gpu_sky_mask_downsampled, gpu_mask_sky_ds)
                    gpu_mask_sky_ds.copyTo(gpu_mask_dilated_downsampled)
                else:
                    # ถ้ายังไม่มี horizon line ใช้ threshold เดิม
                    cv2.cuda.threshold(gpu_mask_downsampled, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY, gpu_mask_dilated_downsampled)

                # Download downsampled mask เพื่อทำ dilate บน CPU (เร็วกว่า download full resolution มาก)
                mask_cpu_downsampled = gpu_mask_dilated_downsampled.download()

                # ตัด motion ตาม ignore zone ที่ผู้ใช้ลาก แต่ปล่อยให้ YOLO ตรวจทั้งเฟรมตามเดิม
                if ignore_motion_allow_mask_downsampled is not None:
                    mask_cpu_downsampled = cv2.bitwise_and(mask_cpu_downsampled, ignore_motion_allow_mask_downsampled)

                # Dilate บน CPU (จำเป็นเพราะ OpenCV CUDA บน Jetson ไม่รองรับ)
                if merge_kernel is not None:
                    mask_cpu_downsampled = cv2.dilate(mask_cpu_downsampled, merge_kernel, iterations=1)

                # กัน motion รั่วข้ามเส้นหลัง dilate อีกรอบ
                if motion_sky_mask_downsampled is not None:
                    mask_cpu_downsampled = cv2.bitwise_and(mask_cpu_downsampled, motion_sky_mask_downsampled)

                # connectedComponents บน downsampled mask (เร็วกว่ามาก)
                num, _, stats, _ = cv2.connectedComponentsWithStats(mask_cpu_downsampled, 8)
                candidates = []
                # Scale factor สำหรับ scale back coordinates
                scale_back = 1.0 / processing_scale_factor
                for i in range(1, num):
                    # Scale back bounding boxes และ coordinates กลับเป็น full resolution
                    x = int(stats[i, cv2.CC_STAT_LEFT] * scale_back)
                    y = int(stats[i, cv2.CC_STAT_TOP] * scale_back)
                    w_obj = int(stats[i, cv2.CC_STAT_WIDTH] * scale_back)
                    h_obj = int(stats[i, cv2.CC_STAT_HEIGHT] * scale_back)
                    # Scale back area (area scales with scale_factor²)
                    area = int(stats[i, cv2.CC_STAT_AREA] * scale_back * scale_back)
                    candidates.append((x, y, w_obj, h_obj, area))

                # Cleanup temporary objects
                del mask_cpu_downsampled
                del stats

                # ลด TTL ของ YOLO open cells (ใช้สำหรับ gating ใต้เส้นขอบฟ้า)
                if yolo_open_cells_ttl > 0:
                    yolo_open_cells_ttl -= 1
                    if yolo_open_cells_ttl == 0:
                        yolo_open_cells.clear()

                # สร้าง handoff open cells จากสถานะก่อนหน้า
                handoff_open_cells = set()
                if handoff_state and grid_filter is not None:
                    for state in handoff_state.values():
                        frames_left = state.get("frames_left", 0)
                        if frames_left > 0:
                            handoff_open_cells.update(
                                get_cells_in_radius(state.get("last_cell"), HANDOFF_GRID_RADIUS, grid_filter)
                            )

                # กรอง motion ใต้เส้นขอบฟ้าก่อนเข้า grid_filter
                if sky_mask is not None and grid_filter is not None:
                    gated_candidates = []
                    is_dense_motion = len(candidates) >= DENSE_MOTION_THRESHOLD

                    for det in candidates:
                        x, y, w_obj, h_obj, area = det
                        cx = min(max(x + w_obj // 2, 0), w - 1)
                        cy = min(max(y + h_obj // 2, 0), h - 1)
                        cell = get_grid_cell_from_point(cx, cy, grid_filter)
                        if cell is None:
                            continue

                        is_sky = sky_mask[cy, cx] > 0
                        if is_sky:
                            # ถ้า motion หนาแน่น → ต้องเริ่มจาก YOLO bbox เท่านั้น
                            if is_dense_motion:
                                if yolo_open_cells_ttl <= 0:
                                    continue
                                if cell not in yolo_open_cells:
                                    continue
                            # ถ้าโล่ง → อนุญาตวัตถุเล็กได้ (บังคับให้ผ่าน min area)
                            if area < SKY_MIN_AREA:
                                area = SKY_MIN_AREA
                            # 🔥 ไม่บังคับให้เป็น MIN_AREA_GLOBAL (20) สำหรับพื้นที่บนฟ้า - อนุญาต 2 pixels
                            gated_candidates.append((x, y, w_obj, h_obj, area))
                            continue

                        # ด้านล่างขอบฟ้า: ใช้เฉพาะ handoff (object ที่มาจากบนฟ้า) — motion ใต้ฟ้าไม่สร้าง ID เอง
                        if cell in handoff_open_cells:
                            gated_candidates.append(det)
                            continue

                    candidates = gated_candidates

                grid_filter.update()
                # 🔥 แยก candidates เป็นบนฟ้าและใต้ฟ้าเพื่อใช้ min_area ต่างกัน (2 vs 20)
                sky_candidates = []
                ground_candidates = []

                if sky_mask is not None:
                    for det in candidates:
                        x, y, w_obj, h_obj, area = det
                        cx = x + w_obj // 2
                        cy = y + h_obj // 2
                        if cy < sky_mask.shape[0] and cx < sky_mask.shape[1]:
                            if sky_mask[cy, cx] > 0:
                                sky_candidates.append(det)
                            else:
                                ground_candidates.append(det)
                        else:
                            ground_candidates.append(det)
                else:
                    ground_candidates = candidates

                # ใช้ min_area = 2 สำหรับบนฟ้า, 20 สำหรับใต้ฟ้า
                valid_dets_sky, congested_sky = grid_filter.filter_and_update(sky_candidates, min_area=2)
                valid_dets_ground, congested_ground = grid_filter.filter_and_update(ground_candidates, min_area=20)
                valid_dets = valid_dets_sky + valid_dets_ground
                # merge กล่อง motion ที่ซ้อนกัน (IoU >= threshold) ให้เหลือ 1 rect ต่อกลุ่ม
                if valid_dets:
                    valid_dets = merge_overlapping_bboxes(valid_dets, MOTION_BBOX_MERGE_IOU)
                congested = congested_sky + congested_ground

                prof_motion_ms = (time.perf_counter() - _t_prof_m0) * 1000.0
                _t_prof_y0 = time.perf_counter()
                # รัน YOLO ก่อน tracker แล้วรวม YOLO dets เข้า input ของ tracker — ให้ bbox ใต้เส้นขอบฟ้าสร้าง path ได้และได้ตำแหน่งอัปเดต
                yolo_origin_ids = set()
                yolo_dets_this_frame = []
                total_yolo_dets = 0
                yolo_dets = []
                yolo_mode_label = "wide-split" if should_use_wide_split(camera_name, w) else "full-frame"
                if yolo_model and yolo_frame_counter % YOLO_INTERVAL == 0:
                    if should_use_wide_split(camera_name, w):
                        yolo_mode_label = "wide-split"
                        yolo_dets = detect_yolo_wide_split(yolo_model, frame, LOCAL_YOLO_CONF_THRESHOLD)
                    else:
                        yolo_mode_label = "full-frame"
                        yolo_dets = detect_yolo_full_frame(yolo_model, frame, LOCAL_YOLO_CONF_THRESHOLD)
                    if yolo_dets:
                        yolo_dets = merge_overlapping_bboxes(yolo_dets, YOLO_BBOX_MERGE_IOU)
                        yolo_dets = nms_bboxes(yolo_dets, YOLO_BBOX_MERGE_IOU)
                        last_yolo_full_dets = yolo_dets
                        total_yolo_dets = len(yolo_dets)
                        max_conf = max(d[4] for d in yolo_dets)
                        print(f"✅ YOLO: Found {total_yolo_dets} detection(s) {yolo_mode_label} (max conf: {max_conf:.2f})")
                        if grid_filter is not None:
                            yolo_open_cells = set()
                            for det in yolo_dets:
                                yolo_open_cells.update(
                                    get_grid_cells_for_bbox(det, grid_filter, YOLO_OPEN_GRID_RADIUS)
                                )
                            yolo_open_cells_ttl = YOLO_OPEN_GRID_TTL_FRAMES
                    else:
                        last_yolo_full_dets = []
                        yolo_open_cells.clear()
                        yolo_open_cells_ttl = 0

                prof_yolo_ms = (time.perf_counter() - _t_prof_y0) * 1000.0
                _t_prof_tr0 = time.perf_counter()
                # รวม YOLO bbox (บนฟ้า+ใต้ฟ้า) เข้า input ของ tracker เพื่อสร้าง/อัปเดต path ได้ทุกที่
                # merge motion กับ YOLO ที่เป็นวัตถุเดียวกันให้เหลือ 1 rect เพื่อไม่ให้แยกสอง path
                if yolo_dets:
                    yolo_rects = [(int(x), int(y), int(w), int(h), int(w * h))
                                  for x, y, w, h, _ in yolo_dets]
                    _merge_center = max(
                        MOTION_YOLO_MAX_CENTER_DIST,
                        min(200, int(0.05 * min(w, h))),
                    )
                    all_dets = merge_motion_and_yolo_for_tracker(
                        valid_dets, yolo_rects, max_center_dist=_merge_center
                    )
                else:
                    all_dets = valid_dets

                # ใช้ YOLO เฟรมนี้ หรือชุดล่าสุด (เฟรมที่ไม่รัน YOLO) เพื่อให้ conf ช่วยคะแนน top-K
                _yolo_for_topk = yolo_dets if yolo_dets else (last_yolo_full_dets or [])
                all_dets_for_tracker = select_topk_tracker_dets(
                    tracker, all_dets, _yolo_for_topk, w, h
                )
                tracked_objs = tracker.update(all_dets_for_tracker)

                # สร้าง yolo_origin_ids: แต่ละ YOLO det ผูกกับ tracked object ที่ใกล้ที่สุด (tracker ได้รับ YOLO จาก all_dets แล้ว — บนฟ้า+ใต้ฟ้า)
                if yolo_dets:
                    tracked_centers = {obj_id: (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                                       for obj_id, (rect, _) in tracked_objs.items()}
                    YOLO_MATCH_DIST = 100
                    for yolo_x, yolo_y, yolo_w, yolo_h, yolo_conf in yolo_dets:
                        yolo_cx = yolo_x + yolo_w // 2
                        yolo_cy = yolo_y + yolo_h // 2
                        best_obj_id = None
                        min_dist = float('inf')
                        for obj_id, (cx, cy) in tracked_centers.items():
                            dist = math.sqrt((yolo_cx - cx)**2 + (yolo_cy - cy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_obj_id = obj_id
                        if best_obj_id is not None and min_dist < YOLO_MATCH_DIST:
                            yolo_origin_ids.add(best_obj_id)
                    yolo_dets_this_frame = yolo_dets

                # ส่ง frame_center สำหรับ threat assessment และ frame_counter สำหรับ gap detection
                frame_center = (w // 2, h // 2)
                # Snapshot red paths ก่อน update เพื่อ detect "lost red" หลัง update
                red_paths_snapshot = {}
                for pid, pdata in list(graph_manager.paths.items()):
                    if not (pdata.get('kinematic_confirmed', False) or pdata.get('yolo_state') == 'red'):
                        continue
                    pts = pdata.get('points') or []
                    if pts:
                        last_pt = pts[-1]
                        red_paths_snapshot[pid] = (last_pt[0], last_pt[1])
                    else:
                        rect = tracked_objs.get(pid)
                        if rect:
                            r = rect[0]
                            red_paths_snapshot[pid] = (r[0] + r[2] // 2, r[1] + r[3] // 2)
                        else:
                            red_paths_snapshot[pid] = (w // 2, h // 2)
                graph_manager.update(tracked_objs, congested, sky_mask, frame_center, frame_counter,
                                   frame_width=w, frame_height=h, valid_detections=valid_dets,
                                   yolo_origin_ids=yolo_origin_ids)
                # Per-frame FOV: cache config + resolver (distance loop, WS, HUD — avoid N lookups)
                _cam_cfg_fov = _ws_cam_config
                _fov_resolver = getattr(cam, "resolve_source_point", None)
                # ระยะแบบ median หลายเฟรมต่อ track (ลดกระโดดจาก bbox)
                for _oid, (_rect, _is_real) in tracked_objs.items():
                    if _oid not in graph_manager.paths:
                        continue
                    _pdata = graph_manager.paths[_oid]
                    _x, _y, _wb, _hb = _rect[0], _rect[1], _rect[2], _rect[3]
                    _ccx = _x + _wb // 2
                    _ccy = _y + _hb // 2
                    _inst = None
                    try:
                        _fh, _fv, _sw, _sh = get_fov_for_canvas_point(
                            _ccx, _ccy, _cam_cfg_fov,
                            resolver=_fov_resolver,
                        )
                        _inst = estimate_distance_m(
                            _wb, _hb, _sw or w, _sh or h, _fh, _fv, get_target_size_m("drone")
                        )
                    except Exception:
                        pass
                    update_path_distance_smoothed(_pdata, _inst)
                for _ek in list(yolo_ws_ema_last_frame.keys()):
                    if frame_counter - yolo_ws_ema_last_frame[_ek] > YOLO_WS_EMA_PRUNE_FRAMES:
                        yolo_ws_ema_last_frame.pop(_ek, None)
                        yolo_ws_dist_ema.pop(_ek, None)
                # เพิ่ม red paths ที่หายไปเข้า recently_lost_red_tracks
                now_t = time.time()
                for pid, (lcx, lcy) in red_paths_snapshot.items():
                    if pid not in graph_manager.paths:
                        recently_lost_red_tracks.append((lcx, lcy, now_t))
                # Prune: เอา entry ที่เก่ากว่า T_LOST_SEC และจำกัดจำนวน
                recently_lost_red_tracks[:] = [
                    (cx, cy, t) for (cx, cy, t) in recently_lost_red_tracks
                    if now_t - t <= SESSION_DRONE_T_LOST_SEC
                ][-SESSION_DRONE_RECENTLY_LOST_MAX:]

                # อัปเดต handoff state สำหรับเฟรมถัดไป
                if sky_mask is not None and grid_filter is not None:
                    new_handoff_state = {}
                    for obj_id, (rect, is_real) in tracked_objs.items():
                        cx = min(max(rect[0] + rect[2] // 2, 0), w - 1)
                        cy = min(max(rect[1] + rect[3] // 2, 0), h - 1)
                        cell = get_grid_cell_from_point(cx, cy, grid_filter)
                        if cell is None:
                            continue
                        is_above = sky_mask[cy, cx] > 0
                        if is_above:
                            new_handoff_state[obj_id] = {
                                "last_cell": cell,
                                "frames_left": HANDOFF_FRAMES,
                            }
                        else:
                            prev = handoff_state.get(obj_id)
                            if prev and prev.get("frames_left", 0) > 0:
                                frames_left = prev["frames_left"] - 1
                                if frames_left > 0:
                                    new_handoff_state[obj_id] = {
                                        "last_cell": cell,
                                        "frames_left": frames_left,
                                    }
                    handoff_state = new_handoff_state
                else:
                    handoff_state = {}

                label_predictions = {}
                label_confirmed = set()
                label_non_drone = set()
                if label_trainer is not None:
                    # Use previous-round FPS as snapshot (current_fps computed later in loop).
                    # Clamped to [5, 120] with fallback 30.0 to avoid division by zero in
                    # time-normalized feature calculation.
                    snapshot_fps = max(5.0, min(120.0, current_fps)) if current_fps > 0 else 30.0
                    canonical_context = {
                        "camera_name": camera_name,
                        "horizon_points": horizon_points,
                        "point_resolver": getattr(cam, "resolve_source_point", None),
                        "processing_fps": snapshot_fps,
                    }
                    label_predictions, label_confirmed = label_trainer.update(
                        tracked_objs,
                        graph_manager,
                        frame.shape,
                        raw_frame=raw_frame,
                        prev_frame=prev_raw_frame,
                        canonical_context=canonical_context,
                    )
                    for obj_id in label_confirmed:
                        path_data = graph_manager.paths.get(obj_id)
                        if not path_data:
                            continue
                        if not path_data.get("kinematic_confirmed", False):
                            path_data["kinematic_confirmed"] = True
                        if not path_data.get("drone_status"):
                            path_data["drone_status"] = "ACTIVE_LABEL_DRONE"
                        path_data["yolo_state"] = "red"
                    for obj_id, (pred_label, pred_conf) in label_predictions.items():
                        if pred_label is not None and pred_label != "drone":
                            label_non_drone.add(obj_id)
                            path_data = graph_manager.paths.get(obj_id)
                            if path_data:
                                path_data["kinematic_confirmed"] = False
                                path_data["drone_status"] = None
                                path_data["yolo_state"] = "orange"

                # Match YOLO detections กับ tracked objects และอัปเดต confidence (ใช้ yolo_dets_this_frame จากก่อน graph)
                if yolo_dets_this_frame:
                    eligible_centers = {}
                    for obj_id, (rect, is_real) in tracked_objs.items():
                        if obj_id not in graph_manager.paths:
                            continue
                        path_data = graph_manager.paths[obj_id]
                        is_big_for_yolo = YOLO_START_PX is not None and max(rect[2], rect[3]) >= YOLO_START_PX
                        if not path_data['validated'] and not is_big_for_yolo:
                            continue
                        if path_data.get('kinematic_confirmed', False):
                            continue
                        yolo_state = path_data.get('yolo_state', 'yellow')
                        if yolo_state == 'red':
                            continue
                        center_x = rect[0] + rect[2] // 2
                        center_y = rect[1] + rect[3] // 2
                        eligible_centers[obj_id] = (center_x, center_y)

                    best_conf_by_obj = {}
                    for yolo_x, yolo_y, yolo_w, yolo_h, yolo_conf in yolo_dets_this_frame:
                        yolo_center_x = yolo_x + yolo_w // 2
                        yolo_center_y = yolo_y + yolo_h // 2
                        best_obj_id = None
                        min_dist = float('inf')
                        for obj_id, (obj_cx, obj_cy) in eligible_centers.items():
                            dist = math.sqrt((yolo_center_x - obj_cx)**2 + (yolo_center_y - obj_cy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_obj_id = obj_id
                        if best_obj_id is not None and min_dist < 100:
                            prev = best_conf_by_obj.get(best_obj_id, 0.0)
                            if yolo_conf > prev:
                                best_conf_by_obj[best_obj_id] = yolo_conf

                    for obj_id, conf in best_conf_by_obj.items():
                        graph_manager.update_yolo_confidence(obj_id, conf)

                    if show_boxes:
                        yolo_box_thick = max(1, int(2 * ui_scale))
                        for yolo_x, yolo_y, yolo_w, yolo_h, yolo_conf in yolo_dets_this_frame:
                            cv2.rectangle(frame, (yolo_x, yolo_y),
                                          (yolo_x + yolo_w, yolo_y + yolo_h),
                                          (255, 0, 0), yolo_box_thick)  # Blue for YOLO
                            cv2.putText(frame, f"DETECTION:{yolo_conf:.2f}",
                                        (yolo_x, yolo_y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.25, 0.5 * ui_scale), (255, 0, 0), max(1, int(1 * ui_scale)))

                    if total_yolo_dets > 0:
                        print(f"📊 YOLO: Total {total_yolo_dets} detection(s) this frame")

                yolo_frame_counter += 1

                prof_track_ms = (time.perf_counter() - _t_prof_tr0) * 1000.0
                _t_prof_d0 = time.perf_counter()
                # ภาพต้นฉบับไม่มี overlay: ใช้ raw_frame เดียวกัน (หลีกเลี่ยง copy เต็มเฟรม — ROI crop ยัง .copy() ตามเดิม)
                original_frame = raw_frame

                # วาด grid เฉพาะเมื่อ show_boxes = True (แต่ grid ยังทำงานอยู่)
                if show_boxes:
                    grid_filter.draw_debug(frame)

                graph_manager.draw(frame, ui_scale=ui_scale, show_non_drone_paths=show_non_drone_paths)

                # FOV: use _cam_cfg_fov / _fov_resolver (set once per frame after graph_manager.update)
                is_night = not is_daytime()

                # วาด bbox ตาม path เท่านั้น: แสดง path ก็มี bbox ไม่แสดง path ก็ไม่มี bbox (ใช้ V สลับ DRONE/ALL)
                for obj_id, (rect, is_real) in tracked_objs.items():
                    if obj_id not in graph_manager.paths:
                        continue
                    if not graph_manager.paths[obj_id]['validated']:
                        continue
                    # โหมด "เฉพาะโดรน" = ไม่วาด path/bbox ตัวที่ไม่ใช่โดรน
                    if not show_non_drone_paths and not graph_manager.paths[obj_id].get('kinematic_confirmed', False):
                        continue

                    # ข้ามถ้าอยู่ใน congestion zone
                    center_x = rect[0] + rect[2] // 2
                    center_y = rect[1] + rect[3] // 2
                    in_congestion = False
                    for z in congested:
                        if z[0] <= center_x <= z[2] and z[1] <= center_y <= z[3]:
                            in_congestion = True
                            break
                    if in_congestion:
                        continue

                    # 🔥 ตรวจสอบ kinematic_confirmed ก่อน yolo_state
                    conf_history = list(graph_manager.paths[obj_id].get('yolo_conf_history', []))
                    if len(conf_history) >= 3:
                        avg_conf = sum(conf_history) / len(conf_history)
                    elif len(conf_history) > 0:
                        avg_conf = sum(conf_history) / len(conf_history)
                    else:
                        avg_conf = 0.0

                    if graph_manager.paths[obj_id].get('kinematic_confirmed', False):
                        validated_color = (0, 0, 255)  # Red (BGR)
                        display_cls = 'drone'
                    else:
                        state = graph_manager.paths[obj_id].get('yolo_state', 'yellow')
                        if state == 'red':
                            validated_color = (0, 0, 255)  # Red (BGR)
                            display_cls = 'drone'
                        elif state == 'orange':
                            validated_color = (0, 165, 255)  # Orange (BGR)
                            display_cls = 'obj'
                        else:
                            validated_color = (0, 255, 255)  # Yellow (BGR)
                            display_cls = 'obj'

                    if label_trainer is not None and obj_id in label_predictions:
                        pred_label, pred_conf = label_predictions[obj_id]
                        if pred_label is not None:
                            display_cls = pred_label
                            avg_conf = pred_conf if pred_conf is not None else avg_conf
                    label_text = f"ID:{obj_id} [{display_cls}:{avg_conf:.2f}]"

                    bbox_thick = max(2, int(3 * ui_scale))
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                                  validated_color, bbox_thick)

                    # แสดง ID และ label พร้อม background rectangle เพื่อให้อ่านง่าย
                    label_x = rect[0]
                    label_y = rect[1] - 5
                    lbl_scale = max(0.28, 0.8 * ui_scale)
                    lbl_thick = max(1, int(2 * ui_scale))
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, lbl_thick
                    )
                    cv2.rectangle(frame,
                                 (label_x - 2, label_y - text_height - 2),
                                 (label_x + text_width + 2, label_y + baseline + 2),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, validated_color, lbl_thick)

                # 🔥 วาด YOLO detections ที่ conf >= 0.6 แต่ยังไม่ match กับ validated path
                if last_yolo_full_dets:
                    for yolo_x, yolo_y, yolo_w, yolo_h, yolo_conf in last_yolo_full_dets:
                        if yolo_conf < LOCAL_YOLO_DRONE_CONF_THRESHOLD:
                            continue

                        # Match กับ validated path: IoU / IoMin / จุดกลางในกล่องใดกล่องหนึ่ง
                        # (ก่อนหน้านี้ใช้แค่จุดกลาง YOLO ใน path rect — พลาดเมื่อกล่องซ้อนแต่กลางไม่ตรง)
                        is_matched = False

                        _min_side = min(w, h)
                        for obj_id, (rect, is_real) in tracked_objs.items():
                            if obj_id not in graph_manager.paths:
                                continue
                            if not graph_manager.paths[obj_id]['validated']:
                                continue

                            pdata = graph_manager.paths[obj_id]
                            pts = pdata.get("smoothed_points") or pdata.get("points") or []
                            if yolo_matches_validated_path(
                                yolo_x, yolo_y, yolo_w, yolo_h, rect, pts, _min_side
                            ):
                                is_matched = True
                                break

                        # ถ้ายังไม่ match กับ validated path → วาด bbox สีแดง
                        if not is_matched:
                            yolo_bbox_thick = max(2, int(3 * ui_scale))
                            cv2.rectangle(frame, (yolo_x, yolo_y),
                                          (yolo_x + yolo_w, yolo_y + yolo_h),
                                          (0, 0, 255), yolo_bbox_thick)  # Red (BGR)

                            # แสดง label
                            label_text = f"DETECTION:{yolo_conf:.2f}"
                            label_x = yolo_x
                            label_y = yolo_y - 5
                            lbl_scale = max(0.28, 0.8 * ui_scale)
                            lbl_thick = max(1, int(2 * ui_scale))
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, lbl_thick
                            )
                            cv2.rectangle(frame,
                                         (label_x - 2, label_y - text_height - 2),
                                         (label_x + text_width + 2, label_y + baseline + 2),
                                         (0, 0, 0), -1)
                            cv2.putText(frame, label_text, (label_x, label_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, (0, 0, 255), lbl_thick)

                # วาดกล่อง congested zones เฉพาะเมื่อ show_boxes = True (แต่ยังทำงานอยู่)
                if show_boxes:
                    zone_thick = max(1, int(2 * ui_scale))
                    for z in congested: cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (0, 165, 255), zone_thick)

                for obj_id, (rect, is_real) in tracked_objs.items():
                    center_x = rect[0] + rect[2]//2
                    center_y = rect[1] + rect[3]//2

                    in_congestion = False
                    for z in congested:
                        if z[0] <= center_x <= z[2] and z[1] <= center_y <= z[3]:
                            in_congestion = True; break
                    if in_congestion: continue

                    # 🔥 CHECK TOGGLE - เช็คก่อนเพื่อให้วาดกล่องได้แม้ยังไม่ validated
                    if not show_boxes: continue

                    color = (0, 255, 0) if is_real else (0, 150, 0)
                    track_thick = max(1, int(2 * ui_scale))
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, track_thick)
                    if is_real:
                        cv2.putText(frame, f"ID:{obj_id}", (rect[0], rect[1]-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, max(0.22, 0.5 * ui_scale), color, max(1, int(1 * ui_scale)))

                if show_guides and horizon_points:
                    draw_pts = extend_horizon_to_edges(horizon_points, w, h)
                    pts = np.array(draw_pts, dtype=np.int32).reshape((-1, 1, 2))
                    horizon_thick = max(1, int(2 * ui_scale))
                    cv2.polylines(frame, [pts], False, (255, 0, 0), horizon_thick)

                # Arm reachable-zone overlay (cam8 only) — draw before HUD so text stays on top.
                with arm_overlay_lock:
                    _poly_snapshot = None if arm_reachable_poly_cam8 is None else arm_reachable_poly_cam8.copy()
                    _wh_snapshot = arm_reachable_calib_wh
                if show_guides and _poly_snapshot is not None and _wh_snapshot is not None:
                    _ow, _oh = _wh_snapshot
                    if _ow > 0 and _oh > 0 and w > 0 and h > 0:
                        _sx = float(w) / float(_ow)
                        _sy = float(h) / float(_oh)
                        _pts = np.array(
                            [[int(p[0] * _sx), int(p[1] * _sy)] for p in _poly_snapshot],
                            dtype=np.int32,
                        ).reshape(-1, 1, 2)
                        cv2.polylines(frame, [_pts], True, ARM_REACHABLE_OVERLAY_COLOR, ARM_REACHABLE_OVERLAY_THICKNESS)

                active_ignore_mask = ignore_motion_preview_mask if (drawing_mode and drawing_target == "ignore_motion" and ignore_motion_preview_mask is not None) else ignore_motion_mask
                show_ignore_overlay = show_boxes or (drawing_mode and drawing_target == "ignore_motion")
                if show_ignore_overlay and active_ignore_mask is not None:
                    overlay_mask = active_ignore_mask > 0
                    if np.any(overlay_mask):
                        red_overlay = np.full_like(frame, (0, 0, 255))
                        blended_overlay = cv2.addWeighted(frame, 0.35, red_overlay, 0.65, 0)
                        frame[overlay_mask] = blended_overlay[overlay_mask]
                    cv2.putText(frame, "MOTION IGNORE: ON", (max(4, int(20*ui_scale)), max(20, int(82*ui_scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, max(0.28, 0.7*ui_scale), (0, 0, 255), max(1, int(2*ui_scale)))

                if drawing_mode:
                    if temp_draw_points:
                        t_pts = np.array(temp_draw_points, dtype=np.int32).reshape((-1, 1, 2))
                        draw_line_thick = max(1, int(2 * ui_scale))
                        draw_color = (0, 255, 255)
                        cv2.polylines(frame, [t_pts], False, draw_color, draw_line_thick)
                    draw_mode_text = "DRAW IGNORE MOTION: Left=Paint, Right=Erase, Enter=Save"
                    if drawing_target == "horizon":
                        draw_mode_text = "DRAW HORIZON: Hold Left=Draw, Right=Undo, Enter=Save"
                    cv2.putText(frame, draw_mode_text, (max(4, int(20*ui_scale)), max(20, int(50*ui_scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, max(0.28, 0.8*ui_scale), (0, 0, 255), max(1, int(2*ui_scale)))

                if label_trainer is not None:
                    label_trainer.draw_prompt(frame, raw_frame)

                # Calculate FPS and Latency
                current_time = time.time()
                if frame_start_time is not None:
                    frame_time = current_time - frame_start_time
                    fps_times.append(frame_time)
                    if len(fps_times) > 0:
                        avg_frame_time = sum(fps_times) / len(fps_times)
                        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                    # Calculate latency (time from frame capture to display)
                    latency_ms = (current_time - frame_start_time) * 1000.0  # Convert to milliseconds
                else:
                    latency_ms = 0.0

                red_drone_thumbnails = []
                red_drone_count = 0
                red_path_bboxes = []
                has_red_drone_visual = False

                if show_heavy_ui:
                    # Display HUD at bottom — ปรับ font_scale ให้ทุกช่องอยู่ภายในความกว้างเฟรม
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    _prof_line_h = int(20 * ui_scale)
                    hud_bar_height = max(20, int(60 * ui_scale)) + (
                        (_prof_line_h + 4) if show_profile_hud else 0
                    )
                    y_pos = h - max(6, int(20 * ui_scale))
                    y_pos_prof = (y_pos - _prof_line_h - 2) if show_profile_hud else None
                    x_start = max(4, int(20 * ui_scale))
                    margin_right = max(20, int(40 * ui_scale))
                    # Reserve space for ARM CUE badge at bottom-right to avoid HUD overlap.
                    arm_cue_reserved_w = max(180, int(300 * ui_scale))
                    available_width = w - x_start - margin_right - arm_cue_reserved_w
                    n_slots = 11
                    font_scale = max(0.22, round(1.0 * ui_scale, 2))
                    thickness = max(1, int(round(2 * ui_scale)))
                    spacing = max(3, int(28 * ui_scale))
                    min_font_scale = 0.18
                    label_sample = "LABEL: UNCERTAIN (t=all u=uncertain o=off) (PAUSED)"
                    while font_scale >= min_font_scale:
                        (w_box, _), _ = cv2.getTextSize("BOXES: OFF", font, font_scale, thickness)
                        (wp, _), _ = cv2.getTextSize("PATHS: ALL", font, font_scale, thickness)
                        w_paths = max(wp, cv2.getTextSize("PATHS: DRONE", font, font_scale, thickness)[0][0])
                        (w_guides, _), _ = cv2.getTextSize("GUIDES(G): ON", font, font_scale, thickness)
                        (w_fps, _), _ = cv2.getTextSize("FPS: 999.9", font, font_scale, thickness)
                        (w_lat, _), _ = cv2.getTextSize("Latency: 999.9ms", font, font_scale, thickness)
                        (w_yolo, _), _ = cv2.getTextSize("DETECTION: 999", font, font_scale, thickness)
                        (w_mode, _), _ = cv2.getTextSize("MODE: KIN+MODEL+DL", font, font_scale, thickness)
                        (w_label, _), _ = cv2.getTextSize(label_sample, font, font_scale, thickness)
                        (w_queue, _), _ = cv2.getTextSize("QUEUE: 999", font, font_scale, thickness)
                        (w_acc, _), _ = cv2.getTextSize("Acc: 0.00 (n=999)", font, font_scale, thickness)
                        (w_yolo_px, _), _ = cv2.getTextSize("START_PX: 999.9", font, font_scale, thickness)
                        total_w = w_box + w_paths + w_guides + w_fps + w_lat + w_yolo + w_mode + w_label + w_queue + w_acc + w_yolo_px + (n_slots - 1) * spacing
                        if total_w <= available_width:
                            break
                        font_scale = round(font_scale - 0.02, 2)
                        if font_scale < min_font_scale:
                            font_scale = min_font_scale
                            break
                    thickness = max(1, int(round(2 * font_scale)))

                    # พื้นหลัง HUD สีดำ (แถบเต็มความกว้างด้านล่างจอ)
                    cv2.rectangle(frame, (0, h - hud_bar_height), (w, h), (0, 0, 0), -1)

                    # แถวสอง: เวลาแยกช่วง (สลับด้วย F)
                    if show_profile_hud and y_pos_prof is not None:
                        _pf_scale = max(0.18, round(font_scale * 0.92, 2))
                        _pf_th = max(1, int(round(2 * _pf_scale)))
                        _pm = (
                            sum(prof_hist_motion) / len(prof_hist_motion)
                            if prof_hist_motion
                            else 0.0
                        )
                        _py = (
                            sum(prof_hist_yolo) / len(prof_hist_yolo)
                            if prof_hist_yolo
                            else 0.0
                        )
                        _ptg = (
                            sum(prof_hist_track) / len(prof_hist_track)
                            if prof_hist_track
                            else 0.0
                        )
                        _pd = (
                            sum(prof_hist_disp) / len(prof_hist_disp)
                            if prof_hist_disp
                            else 0.0
                        )
                        prof_text = (
                            f"PROF(F): avg{PROF_AVG_FRAMES}  "
                            f"M:{_pm:.1f} Y:{_py:.1f} TG:{_ptg:.1f} D:{_pd:.1f}ms"
                        )
                        cv2.putText(
                            frame,
                            prof_text,
                            (x_start, y_pos_prof),
                            font,
                            _pf_scale,
                            (180, 220, 255),
                            _pf_th,
                        )

                    # ตำแหน่ง x ตายตัว: BOXES | PATHS | FPS | Latency | ...
                    (w_box, _), _ = cv2.getTextSize("BOXES(B): OFF", font, font_scale, thickness)
                    (wp, _), _ = cv2.getTextSize("PATHS(V): ALL", font, font_scale, thickness)
                    w_paths = max(wp, cv2.getTextSize("PATHS(V): DRONE", font, font_scale, thickness)[0][0])
                    (w_guides, _), _ = cv2.getTextSize("GUIDES(G): ON", font, font_scale, thickness)
                    (w_fps, _), _ = cv2.getTextSize("FPS: 999.9", font, font_scale, thickness)
                    (w_lat, _), _ = cv2.getTextSize("Latency: 999.9ms", font, font_scale, thickness)
                    (w_yolo, _), _ = cv2.getTextSize("Det: SPLIT Frame", font, font_scale, thickness)
                    (w_mode, _), _ = cv2.getTextSize("MODE: KIN+MODEL+DL", font, font_scale, thickness)
                    (w_label, _), _ = cv2.getTextSize(label_sample, font, font_scale, thickness)
                    (w_queue, _), _ = cv2.getTextSize("QUEUE: 999", font, font_scale, thickness)
                    (w_acc, _), _ = cv2.getTextSize("Acc: 0.00 (n=999)", font, font_scale, thickness)
                    (w_yolo_px, _), _ = cv2.getTextSize("START_PX: 999.9", font, font_scale, thickness)
                    x_paths = x_start + w_box + spacing
                    x_guides = x_paths + w_paths + spacing
                    x_fps = x_guides + w_guides + spacing
                    x_lat = x_fps + w_fps + spacing
                    x_yolo = x_lat + w_lat + spacing
                    x_mode = x_yolo + w_yolo + spacing
                    x_label = x_mode + w_mode + spacing
                    x_queue = x_label + w_label + spacing
                    x_acc = x_queue + w_queue + spacing
                    x_yolo_px = x_acc + w_acc + spacing
                    # Shift right-side HUD slots left so they stay clear of ARM CUE.
                    x_queue = max(x_mode + w_mode + spacing, x_queue - arm_cue_reserved_w)
                    x_acc = max(x_queue + w_queue + spacing, x_acc - arm_cue_reserved_w)
                    x_yolo_px = max(x_acc + w_acc + spacing, x_yolo_px - arm_cue_reserved_w)

                    # Status text (BOXES)
                    status_text = "BOXES(B): ON" if show_boxes else "BOXES(B): OFF"
                    color_text = (0, 255, 0) if show_boxes else (0, 0, 255)
                    cv2.putText(frame, status_text, (x_start, y_pos), font, font_scale, color_text, thickness)

                    # PATHS: DRONE only (default) / ALL (กด V)
                    paths_text = "PATHS(V): ALL" if show_non_drone_paths else "PATHS(V): DRONE"
                    paths_color = (0, 255, 200) if show_non_drone_paths else (0, 200, 255)
                    cv2.putText(frame, paths_text, (x_paths, y_pos), font, font_scale, paths_color, thickness)

                    guides_text = "GUIDES(G): ON" if show_guides else "GUIDES(G): OFF"
                    guides_color = (0, 255, 0) if show_guides else (0, 0, 255)
                    cv2.putText(frame, guides_text, (x_guides, y_pos), font, font_scale, guides_color, thickness)

                    # FPS text
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(frame, fps_text, (x_fps, y_pos), font, font_scale, (0, 255, 255), thickness)

                    # Latency text
                    latency_text = f"Latency: {latency_ms:.1f}ms"
                    cv2.putText(frame, latency_text, (x_lat, y_pos), font, font_scale, (255, 200, 0), thickness)

                    # Detection status text (แสดงจำนวน detections)
                    if yolo_model:
                        yolo_status_text = "Det: SPLIT Frame" if yolo_mode_label == "wide-split" else "Det: FULL Frame"
                        yolo_color = (0, 255, 0) if total_yolo_dets > 0 else (128, 128, 128)
                        cv2.putText(frame, yolo_status_text, (x_yolo, y_pos), font, font_scale, yolo_color, thickness)

                    # Drone analysis mode HUD
                    clf_name = label_trainer.get_classifier_name() if label_trainer is not None else "MODEL"
                    mode_text = f"MODE: KIN+{clf_name}+DL" if ENABLE_KINEMATIC_RULES else f"MODE: {clf_name}+DL"
                    cv2.putText(frame, mode_text, (x_mode, y_pos), font, font_scale, (100, 200, 255), thickness)

                    # Active label trainer HUD
                    if label_trainer is not None:
                        label_status_text = label_trainer.get_hud_text()
                        if label_trainer.pause_labeling:
                            label_status_text += " (PAUSED)"
                        cv2.putText(frame, label_status_text, (x_label, y_pos), font, font_scale, (100, 255, 100), thickness)

                        queue_text = f"QUEUE: {len(label_trainer.queue)}"
                        cv2.putText(frame, queue_text, (x_queue, y_pos), font, font_scale, (255, 150, 0), thickness)

                        # Model accuracy HUD (hold-out test) - แสดงเฉพาะบรรทัดแรก
                        acc_lines = label_trainer.get_accuracy_lines(min_per_class=5)
                        if acc_lines:
                            acc_text = acc_lines[0]
                            cv2.putText(frame, acc_text, (x_acc, y_pos), font, font_scale, (255, 100, 255), thickness)

                    if SHOW_YOLO_START_PX and YOLO_START_PX is not None:
                        yolo_px_text = f"START_PX: {YOLO_START_PX:.1f}"
                        # เมื่อไม่มี label_trainer ให้วางต่อจาก MODE เลย
                        x_yolo_px_pos = x_yolo_px if label_trainer is not None else (x_mode + w_mode + spacing)
                        cv2.putText(frame, yolo_px_text, (x_yolo_px_pos, y_pos), font, font_scale, (0, 255, 200), thickness)

                    # มุมขวาบน: THINGS (สไตล์เดียว detect_thermal_lock_track, สีฟ้า/น้ำเงิน)
                    s = min(h, w) / 1080.0
                    s = max(0.3, min(1.5, s))
                    things_label = "THINGS"
                    things_color = (255, 128, 0)  # BGR bluish
                    font_things = cv2.FONT_HERSHEY_SIMPLEX
                    scale_things = 2.0 * s
                    th_things = max(1, int(3 * s))
                    (tw_things, _), _ = cv2.getTextSize(things_label, font_things, scale_things, th_things)
                    x_things = w - tw_things - int(20 * s)
                    y_things = int(50 * s)
                    cv2.putText(frame, things_label, (x_things, y_things), font_things, scale_things, things_color, th_things)

                # กำหนดขนาด thumbnail ตาม resolution — สัดส่วนความสูงเฟรม, 4K ได้ PIP ใหญ่ขึ้น
                target_height = max(56, min(420, int(h * 0.17)))
                # กำหนด padding รอบ bbox (เพิ่มพื้นที่รอบๆ)
                padding_ratio = 0.15  # 15% ของขนาด bbox หรืออย่างน้อย 20px

                # รวบรวมวัตถุที่ path สีแดง
                # _ws_confirmed_candidates: list of (center_x, center_y, obj_id, distance_m, bbox_w, bbox_h)
                #   bbox_w/bbox_h = ขนาด bounding box บน frame detection จริง (สอดคล้อง w, h)
                # สำหรับ build drone records ส่ง websocket + arm cue หลังลูปเสร็จ
                _ws_confirmed_candidates = []
                # arm cue candidates (แยกจาก WS): 8-tuple มี tier + confidence
                # (cx, cy, obj_id, dist_m, bbox_w, bbox_h, tier, conf)
                _arm_cue_candidates = []

                red_obj_ids_this_frame = set()
                for obj_id, (rect, is_real) in tracked_objs.items():
                    if obj_id in label_non_drone:
                        continue
                    if obj_id not in graph_manager.paths:
                        continue
                    if not graph_manager.paths[obj_id]['validated']:
                        continue

                    path_data = graph_manager.paths[obj_id]
                    is_red = path_data.get('kinematic_confirmed', False) or path_data.get('yolo_state') == 'red'

                    if not is_red:
                        # possible tier: validated path ที่ยังไม่ confirmed แต่เคยมี
                        # YOLO conf พอสมควร (หรือ state orange) → arm cue เท่านั้น
                        _p_conf = float(path_data.get('max_yolo_conf', 0.0) or 0.0)
                        if (_p_conf >= ARM_CUE_POSSIBLE_PATH_MIN_CONF
                                or path_data.get('yolo_state') == 'orange'):
                            _px, _py, _pw, _ph = rect[0], rect[1], rect[2], rect[3]
                            _pcx = _px + _pw // 2
                            _pcy = _py + _ph // 2
                            _ppts = path_data.get("smoothed_points") or path_data.get("points") or []
                            if _ppts:
                                _pcx = float(_ppts[-1][0])
                                _pcy = float(_ppts[-1][1])
                            _arm_cue_candidates.append((
                                _pcx, _pcy, obj_id, path_data.get('distance_m', None),
                                int(_pw), int(_ph), "possible", _p_conf or None,
                            ))

                    if is_red:
                        has_red_drone_visual = True
                        red_drone_count += 1
                        red_obj_ids_this_frame.add(obj_id)
                        red_consecutive_frames[obj_id] = red_consecutive_frames.get(obj_id, 0) + 1
                        x, y, w_box, h_box = rect[0], rect[1], rect[2], rect[3]
                        # Arm cue / WS: kinematic_confirmed -> last path point (avoid bbox leading ahead);
                        # YOLO red only -> bbox center.
                        cur_cx = x + w_box // 2
                        cur_cy = y + h_box // 2
                        if path_data.get("kinematic_confirmed", False):
                            _pts = path_data.get("smoothed_points") or path_data.get("points") or []
                            if _pts:
                                _lp = _pts[-1]
                                cur_cx = float(_lp[0])
                                cur_cy = float(_lp[1])
                        # เก็บสำหรับ websocket export
                        _ws_dist = path_data.get('distance_m', None)
                        if _ws_dist is None:
                            # ลองประมาณจาก bbox + FOV ถ้ายังไม่มีในค่า path
                            try:
                                _fov_h, _fov_v, _src_w, _src_h = get_fov_for_canvas_point(
                                    cur_cx, cur_cy, _cam_cfg_fov, resolver=_fov_resolver
                                )
                                _sz = get_target_size_m('drone')
                                _ws_dist = estimate_distance_m(w_box, h_box, _src_w or w, _src_h or h, _fov_h, _fov_v, _sz)
                            except Exception:
                                _ws_dist = None
                        _ws_confirmed_candidates.append((cur_cx, cur_cy, obj_id, _ws_dist, int(w_box), int(h_box)))
                        _arm_cue_candidates.append((
                            cur_cx, cur_cy, obj_id, _ws_dist,
                            int(w_box), int(h_box), "confirmed",
                            float(path_data.get('max_yolo_conf', 0.0) or 0.0) or None,
                        ))
                        # Session: นับเมื่อ path แดงต่อเนื่องครบ SESSION_PATH_RED_STABLE_FRAMES เฟรม และไม่ใช่ reappeared
                        if (obj_id not in confirmed_drone_ids_this_session
                                and red_consecutive_frames[obj_id] >= SESSION_PATH_RED_STABLE_FRAMES):
                            confirmed_drone_ids_this_session.add(obj_id)
                            now_t = time.time()
                            max_near = max(SESSION_DRONE_MAX_NEAR_PX, int(0.25 * min(w, h)))
                            is_reappeared = False
                            for (lcx, lcy, lost_t) in recently_lost_red_tracks:
                                if now_t - lost_t > SESSION_DRONE_T_LOST_SEC:
                                    continue
                                dist = math.sqrt((cur_cx - lcx)**2 + (cur_cy - lcy)**2)
                                if dist <= max_near:
                                    is_reappeared = True
                                    break
                            if not is_reappeared:
                                session_drone_count += 1
                                now_pt = time.time()
                                yolo_only_recent_positions.append((cur_cx, cur_cy, now_pt))
                                yolo_only_recent_positions[:] = [
                                    (cx, cy, t) for (cx, cy, t) in yolo_only_recent_positions
                                    if now_pt - t <= SESSION_DRONE_T_LOST_SEC
                                ][-SESSION_DRONE_RECENTLY_LOST_MAX:]
                        red_path_bboxes.append((x, y, w_box, h_box))
                        if not show_heavy_ui:
                            continue

                        # คำนวณ padding
                        pad_x = max(20, int(w_box * padding_ratio))
                        pad_y = max(20, int(h_box * padding_ratio))

                        # เพิ่ม padding ไปที่ bbox (ขยายพื้นที่ crop)
                        x_padded = max(0, x - pad_x)
                        y_padded = max(0, y - pad_y)
                        w_padded = min(w - x_padded, w_box + pad_x * 2)
                        h_padded = min(h - y_padded, h_box + pad_y * 2)

                        if w_padded > 0 and h_padded > 0:
                            # 🔥 Crop bbox จาก original_frame (ภาพที่ยังไม่มี bbox)
                            bbox_roi = original_frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded].copy()

                            # Resize เป็น target_height (รักษาอัตราส่วน)
                            if h_padded > 0:
                                aspect_ratio = w_padded / h_padded
                                target_width = int(target_height * aspect_ratio)
                                bbox_resized = cv2.resize(bbox_roi, (target_width, target_height), interpolation=cv2.INTER_AREA)

                                # เพิ่ม text ID (ไม่ต้องมี border สีแดง)
                                id_text = f"ID:{obj_id}"
                                text_scale = 0.7 if target_height >= 200 else 0.6
                                text_thickness = max(1, int(2 * ui_scale))
                                (text_w, text_h), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

                                # วาด background สำหรับ text
                                text_y = text_h + 5
                                cv2.rectangle(bbox_resized,
                                            (0, 0),
                                            (text_w + 4, text_y + baseline + 2),
                                            (0, 0, 0), -1)  # background สีดำ

                                # วาด text ID
                                cv2.putText(bbox_resized, id_text,
                                          (2, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                                          (0, 0, 255), text_thickness)  # สีแดง

                                red_drone_thumbnails.append((obj_id, bbox_resized))

                # path ที่ไม่แดงเฟรมนี้ → รีเซ็ตจำนวนเฟรมแดงต่อเนื่อง
                for oid in list(red_consecutive_frames.keys()):
                    if oid not in red_obj_ids_this_frame:
                        red_consecutive_frames[oid] = 0

                # เพิ่มโดรนจาก YOLO full-frame (conf >= threshold) โดยไม่ซ้ำกับ path ที่แดงอยู่แล้ว
                has_confirmed_yolo_still_visible = False
                if last_yolo_full_dets:
                    _yolo_ws_sort_id = 10_000_000
                    for yolo_x, yolo_y, yolo_w, yolo_h, yolo_conf in last_yolo_full_dets:
                        if yolo_conf < LOCAL_YOLO_DRONE_POSSIBLE_CONF_THRESHOLD:
                            continue
                        skip = False
                        _min_side_ct = min(w, h)
                        for obj_id, (rect, is_real) in tracked_objs.items():
                            if obj_id not in graph_manager.paths:
                                continue
                            pdata = graph_manager.paths[obj_id]
                            if not pdata.get("validated"):
                                continue
                            is_red = pdata.get("kinematic_confirmed", False) or pdata.get("yolo_state") == "red"
                            if not is_red:
                                continue
                            pts = pdata.get("smoothed_points") or pdata.get("points") or []
                            if yolo_matches_validated_path(
                                yolo_x, yolo_y, yolo_w, yolo_h, rect, pts, _min_side_ct
                            ):
                                skip = True
                                break
                        if skip:
                            continue

                        yolo_cx = yolo_x + yolo_w // 2
                        yolo_cy = yolo_y + yolo_h // 2

                        if yolo_conf < LOCAL_YOLO_DRONE_CONF_THRESHOLD:
                            # possible tier: conf ต่ำกว่าเกณฑ์ confirmed —
                            # ส่งเป็น arm cue เท่านั้น ไม่นับ red/alert/session/WS
                            _arm_cue_candidates.append((
                                yolo_cx, yolo_cy, _yolo_ws_sort_id, None,
                                int(yolo_w), int(yolo_h), "possible", float(yolo_conf),
                            ))
                            _yolo_ws_sort_id += 1
                            continue

                        has_red_drone_visual = True
                        red_drone_count += 1
                        # WebSocket / แผนที่: ให้ตรงกับ Drones ที่นับจาก YOLO-only (เดิมมีแค่ path แดง)
                        _yolo_ws_dist = None
                        try:
                            _fov_h, _fov_v, _src_w, _src_h = get_fov_for_canvas_point(
                                yolo_cx, yolo_cy, _cam_cfg_fov,
                                resolver=_fov_resolver,
                            )
                            _sz = get_target_size_m('drone')
                            _yolo_ws_dist = estimate_distance_m(
                                yolo_w, yolo_h, _src_w or w, _src_h or h, _fov_h, _fov_v, _sz
                            )
                        except Exception:
                            pass
                        _cell_k = (yolo_cx // YOLO_WS_DIST_CELL_PX, yolo_cy // YOLO_WS_DIST_CELL_PX)
                        _prev_e = yolo_ws_dist_ema.get(_cell_k)
                        if _yolo_ws_dist is not None and _yolo_ws_dist > 0:
                            if _prev_e is None:
                                _yolo_sm = float(_yolo_ws_dist)
                            else:
                                _yolo_sm = (
                                    YOLO_WS_DIST_EMA_ALPHA * float(_yolo_ws_dist)
                                    + (1.0 - YOLO_WS_DIST_EMA_ALPHA) * float(_prev_e)
                                )
                            yolo_ws_dist_ema[_cell_k] = _yolo_sm
                            yolo_ws_ema_last_frame[_cell_k] = frame_counter
                        else:
                            _yolo_sm = _prev_e
                        _ws_confirmed_candidates.append((yolo_cx, yolo_cy, _yolo_ws_sort_id, _yolo_sm, int(yolo_w), int(yolo_h)))
                        _arm_cue_candidates.append((
                            yolo_cx, yolo_cy, _yolo_ws_sort_id, _yolo_sm,
                            int(yolo_w), int(yolo_h), "confirmed", float(yolo_conf),
                        ))
                        _yolo_ws_sort_id += 1
                        # Session YOLO-only: นับเมื่อเจอในบริเวณนั้นอย่างน้อย SESSION_YOLO_ONLY_MIN_HITS ครั้งภายในหน้าต่างเวลา
                        now_t = time.time()
                        max_near = max(SESSION_DRONE_MAX_NEAR_PX, int(0.25 * min(w, h)))
                        # เสียงต่อเนื่องแบบ 1: ยังเห็น YOLO ใกล้ตำแหน่งที่เคยนับ session
                        for (ox, oy, ot) in yolo_only_recent_positions:
                            if now_t - ot <= SESSION_DRONE_T_LOST_SEC:
                                dist = math.sqrt((yolo_cx - ox)**2 + (yolo_cy - oy)**2)
                                if dist <= max_near:
                                    has_confirmed_yolo_still_visible = True
                                    break
                        is_reappeared_yolo = False
                        for (ox, oy, ot) in yolo_only_recent_positions:
                            if now_t - ot > SESSION_DRONE_T_LOST_SEC:
                                continue
                            dist = math.sqrt((yolo_cx - ox)**2 + (yolo_cy - oy)**2)
                            if dist <= max_near:
                                is_reappeared_yolo = True
                                break
                        if not is_reappeared_yolo:
                            # หา pending ที่ใกล้และยังไม่เกินหน้าต่างเวลา
                            matched_idx = None
                            for idx, (px, py, pt, cnt) in enumerate(yolo_only_pending):
                                if now_t - pt > SESSION_YOLO_PENDING_WINDOW_SEC:
                                    continue
                                dist = math.sqrt((yolo_cx - px)**2 + (yolo_cy - py)**2)
                                if dist <= max_near:
                                    matched_idx = idx
                                    break
                            if matched_idx is not None:
                                px, py, pt, cnt = yolo_only_pending[matched_idx]
                                cnt += 1
                                yolo_only_pending[matched_idx] = (px, py, now_t, cnt)
                                if cnt >= SESSION_YOLO_ONLY_MIN_HITS:
                                    session_drone_count += 1
                                    yolo_only_recent_positions.append((yolo_cx, yolo_cy, now_t))
                                    yolo_only_pending.pop(matched_idx)
                            else:
                                yolo_only_pending.append((yolo_cx, yolo_cy, now_t, 1))
                        # Prune pending และ recent
                        yolo_only_pending[:] = [
                            (cx, cy, t, c) for (cx, cy, t, c) in yolo_only_pending
                            if now_t - t <= SESSION_YOLO_PENDING_WINDOW_SEC
                        ][-SESSION_DRONE_RECENTLY_LOST_MAX:]
                        yolo_only_recent_positions[:] = [
                            (cx, cy, t) for (cx, cy, t) in yolo_only_recent_positions
                            if now_t - t <= SESSION_DRONE_T_LOST_SEC
                        ][-SESSION_DRONE_RECENTLY_LOST_MAX:]

                        if not show_heavy_ui:
                            continue

                        # คำนวณ padding
                        pad_x = max(20, int(yolo_w * padding_ratio))
                        pad_y = max(20, int(yolo_h * padding_ratio))

                        # เพิ่ม padding ไปที่ bbox (ขยายพื้นที่ crop)
                        x_padded = max(0, yolo_x - pad_x)
                        y_padded = max(0, yolo_y - pad_y)
                        w_padded = min(w - x_padded, yolo_w + pad_x * 2)
                        h_padded = min(h - y_padded, yolo_h + pad_y * 2)

                        if w_padded > 0 and h_padded > 0:
                            bbox_roi = original_frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded].copy()

                            if h_padded > 0:
                                aspect_ratio = w_padded / h_padded
                                target_width = int(target_height * aspect_ratio)
                                bbox_resized = cv2.resize(bbox_roi, (target_width, target_height), interpolation=cv2.INTER_AREA)

                                conf_text = f"DETECTION:{yolo_conf:.2f}"
                                text_scale = 0.7 if target_height >= 200 else 0.6
                                text_thickness = max(1, int(2 * ui_scale))
                                (text_w, text_h), baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

                                # วาด background สำหรับ text
                                text_y = text_h + 5
                                cv2.rectangle(bbox_resized,
                                            (0, 0),
                                            (text_w + 4, text_y + baseline + 2),
                                            (0, 0, 0), -1)

                                # วาด text conf
                                cv2.putText(bbox_resized, conf_text,
                                          (2, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                                          (0, 0, 255), text_thickness)

                                red_drone_thumbnails.append(("DETECTION", bbox_resized))

                # 🔊 Sound alert: เล่นต่อเนื่องเฉพาะเมื่อยังติดตามโดรนที่เคยนับ session ได้ (path แดงหรือ YOLO ใกล้ตำแหน่งที่เคยนับ)
                has_confirmed_red_drone = bool(confirmed_drone_ids_this_session & red_obj_ids_this_frame)
                has_confirmed_still_visible = has_confirmed_red_drone or has_confirmed_yolo_still_visible
                if has_confirmed_still_visible:
                    sound_confirm_counter = min(sound_confirm_counter + 1, SOUND_CONFIRM_FRAMES)
                else:
                    sound_confirm_counter = 0

                confirm_sound = sound_confirm_counter >= SOUND_CONFIRM_FRAMES
                if sound_alert is not None:
                    sound_alert.update(confirm_sound)

                # แสดง thumbnails ที่มุมล่างขวา
                if show_heavy_ui and red_drone_thumbnails:
                    thumbnail_spacing = max(4, int(10 * ui_scale))
                    margin_right = max(6, int(20 * ui_scale))
                    margin_bottom = max(10, int(40 * ui_scale))  # วางเหนือ HUD text

                    # คำนวณตำแหน่งเริ่มต้นจากขวา
                    total_width = sum(thumb.shape[1] for _, thumb in red_drone_thumbnails) + thumbnail_spacing * (len(red_drone_thumbnails) - 1)
                    thumbnail_x_start = w - total_width - margin_right

                    # ตำแหน่ง Y (ด้านล่าง)
                    max_thumb_height = max(thumb.shape[0] for _, thumb in red_drone_thumbnails)
                    thumbnail_y_start = h - max_thumb_height - margin_bottom

                    # ตรวจสอบว่าตำแหน่งถูกต้อง
                    if thumbnail_x_start < 0:
                        thumbnail_x_start = margin_right
                    if thumbnail_y_start < 0:
                        thumbnail_y_start = margin_bottom

                    current_x = thumbnail_x_start
                    for obj_id, thumbnail in red_drone_thumbnails:
                        thumb_h, thumb_w = thumbnail.shape[:2]

                        # ตรวจสอบว่าพื้นที่พอหรือไม่
                        if current_x + thumb_w > w - margin_right:
                            break  # ไม่พอที่จะแสดง thumbnail ถัดไป

                        # วาง thumbnail
                        if thumbnail_y_start >= 0 and thumbnail_y_start + thumb_h <= h:
                            frame[thumbnail_y_start:thumbnail_y_start+thumb_h,
                                  current_x:current_x+thumb_w] = thumbnail

                        current_x += thumb_w + thumbnail_spacing

                    # แสดงจำนวนโดรนที่เจอ (ด้านบน thumbnails) + session total
                    count_text = f"Drones: {red_drone_count} | Session: {session_drone_count}"
                    count_y = thumbnail_y_start - max(10, int(35 * ui_scale))
                    if count_y >= 0:
                        (count_w, _), _ = cv2.getTextSize(count_text, font, font_scale, thickness)
                        count_x = w - count_w - margin_right
                        rect_h = max(8, int(25 * ui_scale))
                        cv2.rectangle(frame,
                                     (count_x - 2, count_y - rect_h),
                                     (count_x + count_w + 2, count_y + max(4, int(10 * ui_scale))),
                                     (0, 0, 0), -1)  # background สีดำ
                        cv2.putText(frame, count_text, (count_x, count_y),
                                  font, font_scale, (0, 0, 255), thickness)

                # GPU Resize for display - ใช้ GPU resize
                # Note: ต้อง upload frame ที่วาดเสร็จแล้ว (มี boxes, text, etc.) ไม่สามารถ reuse tmp_raw ได้
                if w != display_w or h != display_h:
                    try:
                        # ใช้ gpu_display_frame สำหรับ upload frame ที่วาดเสร็จแล้ว
                        if gpu_display_frame is not None and gpu_resized_frame is not None:
                            # ตรวจสอบว่า buffers มีขนาดถูกต้อง
                            if gpu_display_frame.size() == (h, w) and gpu_resized_frame.size() == (display_h, display_w):
                                gpu_display_frame.upload(frame)
                                cv2.cuda.resize(gpu_display_frame, (display_w, display_h), gpu_resized_frame)
                                display_frame = gpu_resized_frame.download()
                            else:
                                # Buffers มีขนาดไม่ถูกต้อง - fallback to CPU
                                raise ValueError(f"GPU buffer size mismatch: display={gpu_display_frame.size() if gpu_display_frame is not None else None}, resized={gpu_resized_frame.size() if gpu_resized_frame is not None else None}")
                        else:
                            # GPU buffers ไม่พร้อม - fallback to CPU
                            raise ValueError("GPU display buffers not initialized")
                    except Exception as e:
                        # Fallback to CPU resize if GPU resize fails
                        display_frame = cv2.resize(frame, (display_w, display_h))
                else:
                    display_frame = frame

                DRONE_HUD_COUNTS["session"] = session_drone_count
                DRONE_HUD_COUNTS["current"] = red_drone_count

                # --- Websocket export: build records + push (non-blocking) ---
                # ordering: sort ตาม center_x ซ้าย→ขวา, tie-break center_y, obj_id
                _ws_sorted = sorted(_ws_confirmed_candidates, key=lambda t: (t[0], t[1], t[2]))
                _ws_cam_geo_now = _get_camera_geo(camera_name)
                _ws_cam_cfg_now = _ws_cam_config
                _ws_records = [
                    build_drone_record(
                        order=i + 1,
                        cx=_cx,
                        cy=_cy,
                        frame_w=w,
                        frame_h=h,
                        cam_geo=_ws_cam_geo_now,
                        cam_config=_ws_cam_cfg_now,
                        distance_m=_dist,
                    )
                    for i, (_cx, _cy, _oid, _dist, *_bwh) in enumerate(_ws_sorted)
                ]
                DRONE_HUD_COUNTS["drone_records"] = _ws_records
                ws_exporter.push(_ws_records)

                # Arm cue: ส่ง candidates (confirmed + possible) ไปยัง Jetson แขน
                #   confirmed: kinematic_confirmed OR yolo_state=='red' OR YOLO conf >= เกณฑ์หลัก
                #   possible : validated path / YOLO conf ต่ำกว่าเกณฑ์ — AUTO ชี้รอ human กด LOCK
                # เรียง confirmed ก่อนแล้วตัดที่ ARM_CUE_MAX_CANDIDATES (กัน datagram บวม)
                if cue_sender is not None and _arm_cue_candidates:
                    _arm_cue_candidates.sort(
                        key=lambda t: (0 if t[6] == "confirmed" else 1, -(t[7] or 0.0))
                    )
                    cue_sender.push(_arm_cue_candidates[:ARM_CUE_MAX_CANDIDATES], w, h)

                # HUD: show arm cue sender status at bottom-right (avoid overlap)
                if cue_sender is not None:
                    _cue_label = f"ARM {cue_sender.status_label()}"
                    _cue_color = (0, 220, 0) if cue_sender.last_send_ok else (0, 100, 255)
                    _cue_font = cv2.FONT_HERSHEY_SIMPLEX
                    _cue_scale = 0.55
                    _cue_thickness = 1
                    (_cue_w, _cue_h), _cue_base = cv2.getTextSize(
                        _cue_label, _cue_font, _cue_scale, _cue_thickness
                    )
                    _cue_margin = 12
                    _cue_x = max(0, display_w - _cue_w - _cue_margin)
                    _cue_y = max(_cue_h + _cue_margin, display_h - _cue_margin)
                    # Dark background box for readability; keeps this HUD isolated from other overlays.
                    cv2.rectangle(
                        display_frame,
                        (_cue_x - 6, _cue_y - _cue_h - 4),
                        (_cue_x + _cue_w + 6, _cue_y + _cue_base + 2),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        _cue_label,
                        (_cue_x, _cue_y),
                        _cue_font,
                        _cue_scale,
                        _cue_color,
                        _cue_thickness,
                    )

                cv2.imshow(WINDOW_NAME, display_frame)
                prof_disp_ms = (time.perf_counter() - _t_prof_d0) * 1000.0
                prof_hist_motion.append(prof_motion_ms)
                prof_hist_yolo.append(prof_yolo_ms)
                prof_hist_track.append(prof_track_ms)
                prof_hist_disp.append(prof_disp_ms)

                # Cleanup display_frame
                del display_frame

            except cv2.error: continue

            # Periodic Memory Cleanup
            if frame_counter % PERIODIC_MEMORY_CLEANUP == 0:
                try:
                    from memory_manager import clear_cuda_memory
                    clear_cuda_memory()
                except:
                    pass
                gc.collect()

                # Log memory usage (optional, every 600 frames)
                if frame_counter % (PERIODIC_MEMORY_CLEANUP * 3) == 0:
                    try:
                        import psutil
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        mem_mb = mem_info.rss / 1024 / 1024
                        mem_percent = psutil.virtual_memory().percent
                        print(f"📊 Memory: {mem_mb:.1f} MB ({mem_percent:.1f}%)")
                        if mem_percent > 85:
                            print(f"⚠️ High memory usage detected!")
                    except:
                        pass

            key = cv2.waitKey(1) & 0xFF
            if label_trainer is not None:
                label_trainer.handle_key(key, original_frame)

            if key == ord('q'): break
            elif key in (ord('y'), ord('Y')):
                if ARM_REACHABLE_OVERLAY_ENABLED and str(camera_name).lower() == "cam8":
                    if arm_overlay_sync_done:
                        print("[ARM-OVERLAY] already synced; no pull needed")
                    else:
                        ok = _sync_arm_calibration_once(_calib_path)
                        if ok:
                            _pts, _wh = _load_arm_reachable_overlay(_calib_path)
                            if _pts is not None and _wh is not None:
                                with arm_overlay_lock:
                                    arm_reachable_poly_cam8 = _pts
                                    arm_reachable_calib_wh = _wh
                                arm_overlay_sync_done = True
                                print(f"[ARM-OVERLAY] manual sync complete ({len(_pts)} pts)")
                            else:
                                print("[ARM-OVERLAY] manual sync fetched file but polygon invalid")
                        else:
                            print("[ARM-OVERLAY] manual sync failed")
            elif key == ord('t'):
                if label_trainer is not None:
                    label_trainer.set_prompt_mode("all")
            elif key == ord('u'):
                if label_trainer is not None:
                    label_trainer.set_prompt_mode("uncertain")
            elif key == ord('o'):
                if label_trainer is not None:
                    label_trainer.set_prompt_mode("off")
            elif key == ord('p'):
                if label_trainer is not None:
                    label_trainer.toggle_pause()
            elif key == ord('l'):
                if drawing_mode and drawing_target == "horizon":
                    drawing_mode = False
                    drawing_target = None
                else:
                    drawing_mode = True
                    drawing_target = "horizon"
                temp_draw_points = []
                print("🟦 Draw mode: HORIZON")
            elif key == ord('i'):
                if drawing_mode and drawing_target == "ignore_motion":
                    drawing_mode = False
                    drawing_target = None
                    ignore_motion_preview_mask = None
                else:
                    drawing_mode = True
                    drawing_target = "ignore_motion"
                    ignore_motion_preview_mask = (
                        ignore_motion_mask.copy()
                        if ignore_motion_mask is not None
                        else np.zeros((h, w), dtype=np.uint8)
                    )
                temp_draw_points = []
                ignore_motion_last_point = None
                ignore_motion_is_painting = False
                ignore_motion_is_erasing = False
                print("🟥 Draw mode: IGNORE MOTION")
            elif key == ord('b'):
                show_boxes = not show_boxes
                print(f"📦 Boxes: {show_boxes}")
            elif key in (ord('g'), ord('G')):
                show_guides = not show_guides
                print(f"🧭 Guides (horizon + arm limit): {show_guides}")
            elif key == ord('v'):
                show_non_drone_paths = not show_non_drone_paths
                print(f"🛤️ Paths: {'ALL' if show_non_drone_paths else 'DRONE ONLY'}")
            elif key in (ord('h'), ord('H')):
                show_heavy_ui = not show_heavy_ui
                print(f"🖥️ Heavy UI overlays: {show_heavy_ui}")
            elif key in (ord('f'), ord('F')):
                show_profile_hud = not show_profile_hud
                print(f"⏱️ Profile HUD (F): {show_profile_hud}")
            elif key == ord('c'):
                ignore_motion_mask = None
                ignore_motion_mask_downsampled = None
                ignore_motion_allow_mask_downsampled = None
                ignore_motion_preview_mask = None
                ignore_motion_last_point = None
                ignore_motion_is_painting = False
                ignore_motion_is_erasing = False
                if drawing_target == "ignore_motion":
                    drawing_mode = False
                    drawing_target = None
                try:
                    if os.path.exists(IGNORE_MOTION_FILE):
                        os.remove(IGNORE_MOTION_FILE)
                except Exception as e:
                    print(f"⚠️ Clear motion ignore failed: {e}")
                else:
                    print("🧹 Cleared motion ignore mask")
            elif key == 13:
                if drawing_mode and drawing_target == "ignore_motion" and ignore_motion_preview_mask is not None:
                    ignore_motion_mask = ignore_motion_preview_mask.copy()
                    np.save(IGNORE_MOTION_FILE, ignore_motion_mask.astype(np.uint8))
                    rebuild_ignore_motion_processing_masks()
                    drawing_mode = False
                    drawing_target = None
                    ignore_motion_preview_mask = None
                    ignore_motion_last_point = None
                    ignore_motion_is_painting = False
                    ignore_motion_is_erasing = False
                    print("💾 Saved motion ignore mask!")
                elif drawing_mode and len(temp_draw_points) > 1:
                    if drawing_target == "horizon":
                        horizon_points = temp_draw_points[:]
                        np.save(HORIZON_FILE, np.array(horizon_points, dtype=np.int32))
                        sky_mask = rebuild_sky_mask(h, w, horizon_points)
                        motion_sky_mask = rebuild_motion_sky_mask(sky_mask, HORIZON_MOTION_BUFFER_PX)
                        if processing_w > 0 and processing_h > 0 and motion_sky_mask is not None:
                            motion_sky_mask_downsampled = cv2.resize(
                                motion_sky_mask,
                                (processing_w, processing_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        else:
                            motion_sky_mask_downsampled = None
                        try:
                            if gpu_sky_mask is not None and motion_sky_mask is not None:
                                gpu_sky_mask.upload(motion_sky_mask)
                                if gpu_ground_mask is not None:
                                    cv2.cuda.bitwise_not(gpu_sky_mask, gpu_ground_mask)
                                if gpu_sky_mask_downsampled is not None:
                                    cv2.cuda.resize(gpu_sky_mask, (processing_w, processing_h), gpu_sky_mask_downsampled, interpolation=cv2.INTER_NEAREST)
                                if gpu_ground_mask is not None and gpu_ground_mask_downsampled is not None:
                                    cv2.cuda.resize(gpu_ground_mask, (processing_w, processing_h), gpu_ground_mask_downsampled, interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            pass
                        print("💾 Saved horizon!")
                    drawing_mode = False
                    drawing_target = None

    except KeyboardInterrupt: pass
    finally:
        if 'arm_overlay_stop_event' in locals() and arm_overlay_stop_event is not None:
            arm_overlay_stop_event.set()
        if 'arm_overlay_sync_thread' in locals() and arm_overlay_sync_thread is not None:
            try:
                arm_overlay_sync_thread.join(timeout=0.5)
            except Exception:
                pass
        # Stop sound alert
        if 'sound_alert' in locals() and sound_alert is not None:
            sound_alert.stop()
        if 'label_trainer' in locals() and label_trainer is not None:
            label_trainer.save_if_dirty()
        if 'ws_exporter' in locals() and ws_exporter is not None:
            ws_exporter.stop()
        if 'cue_sender' in locals() and cue_sender is not None:
            cue_sender.stop()
        if 'cam' in locals(): cam.release()
        cv2.destroyAllWindows()
        gc.collect()

_single_camera_main = main


DEFAULT_WINDOW_NAME = "Horizon Tracker V8 (Smooth Validator)"
DEFAULT_HORIZON_FILE = "horizon_poly.npy"
DEFAULT_IGNORE_MOTION_FILE = "motion_ignore_mask.npy"


def _resolve_local_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path_value)


def _normalize_path(path_like):
    if path_like is None:
        return None
    try:
        return os.fspath(path_like)
    except TypeError:
        return path_like


def _build_fullscreen_layout():
    screen_w, screen_h = _get_screen_size()
    return {"x": 0, "y": 0, "width": screen_w, "height": screen_h}


def _build_top_bottom_layouts(top_ratio=0.5):
    screen_w, screen_h = _get_screen_size()
    top_height = max(1, int(screen_h * float(top_ratio)))
    bottom_height = max(1, screen_h - top_height)
    return {
        "top": {"x": 0, "y": 0, "width": screen_w, "height": top_height},
        "bottom": {"x": 0, "y": top_height, "width": screen_w, "height": bottom_height},
    }


class _HookProxy:
    def __init__(self, stream=None):
        self.stream = stream

    def on_display(self, module, image):
        if self.stream is None:
            return image
        return self.stream.overlay_display(image)

    def on_key(self, key):
        if self.stream is None:
            return False
        return self.stream.handle_key(key)

    def on_mouse(self, module, event, x, y, flags, userdata):
        if self.stream is None:
            return False
        return self.stream.handle_mouse(module, event, x, y, flags, userdata)


def _queue_put_latest(frame_queue, payload):
    if frame_queue is None:
        return
    while True:
        try:
            frame_queue.put_nowait(payload)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return


def _patch_runtime_for_worker(module, worker_config, hook_proxy=None):
    layout = worker_config["layout"]
    window_name = worker_config["window_name"]
    horizon_file = _resolve_local_path(worker_config["horizon_file"])
    ignore_motion_file = _resolve_local_path(worker_config["ignore_motion_file"])
    frame_queue = worker_config.get("frame_queue")
    control_queue = worker_config.get("control_queue")
    queue_display_mode = frame_queue is not None
    mouse_callback_state = {"callback": None, "userdata": None}

    for worker_queue in (frame_queue, control_queue):
        if worker_queue is None:
            continue
        try:
            worker_queue.cancel_join_thread()
        except Exception:
            pass

    module._get_screen_size = lambda: (layout["width"], layout["height"])

    def remap_window(name):
        return window_name if name == DEFAULT_WINDOW_NAME else name

    def remap_path(path_like):
        path = _normalize_path(path_like)
        if path == DEFAULT_HORIZON_FILE:
            return horizon_file
        if path == DEFAULT_IGNORE_MOTION_FILE:
            return ignore_motion_file
        return path_like

    cv2_mod = module.cv2
    original_named_window = cv2_mod.namedWindow
    original_set_window_property = cv2_mod.setWindowProperty
    original_resize_window = cv2_mod.resizeWindow
    original_move_window = getattr(cv2_mod, "moveWindow", None)
    original_set_mouse_callback = cv2_mod.setMouseCallback
    original_imshow = cv2_mod.imshow
    original_wait_key = cv2_mod.waitKey

    def apply_window_layout(name):
        if queue_display_mode:
            return
        mapped_name = remap_window(name)
        original_resize_window(mapped_name, layout["width"], layout["height"])
        if original_move_window is not None:
            original_move_window(mapped_name, layout["x"], layout["y"])

    def named_window(name, *args, **kwargs):
        if queue_display_mode:
            return None
        result = original_named_window(remap_window(name), *args, **kwargs)
        apply_window_layout(name)
        return result

    def set_window_property(name, prop_id, prop_value, *args, **kwargs):
        if queue_display_mode:
            return None
        if prop_id == cv2_mod.WND_PROP_FULLSCREEN:
            return original_set_window_property(remap_window(name), prop_id, cv2_mod.WINDOW_NORMAL, *args, **kwargs)
        return original_set_window_property(remap_window(name), prop_id, prop_value, *args, **kwargs)

    def resize_window(name, *args, **kwargs):
        if queue_display_mode:
            return None
        apply_window_layout(name)
        return None

    def move_window(name, *args, **kwargs):
        if queue_display_mode:
            return None
        if original_move_window is None:
            return None
        return original_move_window(remap_window(name), layout["x"], layout["y"])

    def set_mouse_callback(name, callback, userdata=None):
        def wrapped_callback(event, x, y, flags, cb_userdata):
            if hook_proxy is not None and hook_proxy.on_mouse(module, event, x, y, flags, cb_userdata):
                return
            return callback(event, x, y, flags, cb_userdata)

        if queue_display_mode:
            mouse_callback_state["callback"] = wrapped_callback
            mouse_callback_state["userdata"] = userdata
            return None
        return original_set_mouse_callback(remap_window(name), wrapped_callback, userdata)

    def imshow(name, image):
        output = image
        if hook_proxy is not None:
            output = hook_proxy.on_display(module, image)
        if queue_display_mode:
            hud = getattr(module, "DRONE_HUD_COUNTS", None)
            payload = {
                "frame": output,
                "width": output.shape[1],
                "height": output.shape[0],
                "window_name": remap_window(name),
            }
            if isinstance(hud, dict):
                payload["session_drone_count"] = hud.get("session", 0)
                payload["current_drone_count"] = hud.get("current", 0)
                payload["drone_records"] = hud.get("drone_records", [])
            _queue_put_latest(frame_queue, payload)
            return None
        return original_imshow(remap_window(name), output)

    def wait_key(delay=0):
        if queue_display_mode:
            if control_queue is None:
                return -1
            end_time = time.time() + max(0.0, delay / 1000.0)
            pending_key = -1
            while True:
                remaining = max(0.0, end_time - time.time())
                timeout = remaining if pending_key == -1 else 0.0
                try:
                    event = control_queue.get(timeout=timeout if timeout > 0 else 0.0)
                except queue.Empty:
                    break
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if event_type == "mouse" and mouse_callback_state["callback"] is not None:
                    # Compositor sends (x,y) in frame/source space; on_mouse_global expects
                    # display space and scales to frame. Convert so callback's scale yields frame coords.
                    fx = int(event.get("x", 0))
                    fy = int(event.get("y", 0))
                    cw = getattr(module, "current_frame_w", 1280)
                    ch = getattr(module, "current_frame_h", 720)
                    dw = max(1, getattr(module, "display_w", 1280))
                    dh = max(1, getattr(module, "display_h", 720))
                    dx = int(fx * dw / cw) if cw else fx
                    dy = int(fy * dh / ch) if ch else fy
                    mouse_callback_state["callback"](
                        int(event.get("event", 0)),
                        dx,
                        dy,
                        int(event.get("flags", 0)),
                        mouse_callback_state["userdata"],
                    )
                    continue
                if event_type == "key":
                    key = int(event.get("key", -1))
                    if hook_proxy is not None and hook_proxy.on_key(key):
                        pending_key = -1
                        continue
                    pending_key = key
                    break
                if event_type == "shutdown":
                    pending_key = ord("q")
                    break
                if remaining <= 0:
                    break
            return pending_key
        key = original_wait_key(delay)
        if hook_proxy is not None and hook_proxy.on_key(key):
            return -1
        return key

    cv2_mod.namedWindow = named_window
    cv2_mod.setWindowProperty = set_window_property
    cv2_mod.resizeWindow = resize_window
    if original_move_window is not None:
        cv2_mod.moveWindow = move_window
    cv2_mod.setMouseCallback = set_mouse_callback
    cv2_mod.imshow = imshow
    cv2_mod.waitKey = wait_key

    original_exists = module.os.path.exists
    original_remove = module.os.remove
    original_np_load = module.np.load
    original_np_save = module.np.save

    def patched_exists(path_like):
        return original_exists(remap_path(path_like))

    def patched_remove(path_like, *args, **kwargs):
        return original_remove(remap_path(path_like), *args, **kwargs)

    def patched_np_load(file, *args, **kwargs):
        return original_np_load(remap_path(file), *args, **kwargs)

    def patched_np_save(file, arr, *args, **kwargs):
        return original_np_save(remap_path(file), arr, *args, **kwargs)

    module.os.path.exists = patched_exists
    module.os.remove = patched_remove
    module.np.load = patched_np_load
    module.np.save = patched_np_save


def _apply_worker_thresholds(module, worker_config):
    if "yolo_conf_threshold" in worker_config:
        module.LOCAL_YOLO_CONF_THRESHOLD = worker_config["yolo_conf_threshold"]
    if "yolo_drone_conf_threshold" in worker_config:
        module.LOCAL_YOLO_DRONE_CONF_THRESHOLD = worker_config["yolo_drone_conf_threshold"]


def _run_single_camera_worker(worker_config):
    module = sys.modules[__name__]
    _apply_worker_thresholds(module, worker_config)
    module.LOCAL_ACTIVE_CAMERA = worker_config["camera_name"]
    _patch_runtime_for_worker(module, worker_config, hook_proxy=None)
    print(f"[{worker_config['instance_name']}] camera={worker_config['camera_name']}")
    print(f"[{worker_config['instance_name']}] window={worker_config['window_name']}")
    _single_camera_main(active_camera_name=worker_config["camera_name"])


def _run_bottom_stitch_worker(worker_config):
    module = sys.modules[__name__]
    _run_bottom_stitch_worker_impl(
        module=module,
        worker_config=worker_config,
        apply_worker_thresholds=_apply_worker_thresholds,
        patch_runtime_for_worker=_patch_runtime_for_worker,
        hook_proxy_cls=_HookProxy,
        single_camera_main=_single_camera_main,
    )


def worker_entrypoint(kind, worker_config):
    try:
        if kind == "single":
            _run_single_camera_worker(worker_config)
        elif kind == "bottom_stitch":
            _run_bottom_stitch_worker(worker_config)
        else:
            raise ValueError(f"Unknown worker kind: {kind}")
    except KeyboardInterrupt:
        pass
    except Exception:
        print(f"[{worker_config.get('instance_name', kind)}] worker crashed")
        traceback.print_exc()
        raise


def _build_multi_worker_configs():
    layouts = _build_top_bottom_layouts(TOP_PANEL_HEIGHT_RATIO)
    top_config = {
        "instance_name": "top_rgb",
        "camera_name": TOP_CAMERA_NAME,
        "window_name": f"{TOP_CAMERA_NAME}-top",
        "layout": layouts["top"],
        "horizon_file": TOP_HORIZON_FILE,
        "ignore_motion_file": TOP_IGNORE_MOTION_FILE,
        "yolo_conf_threshold": TOP_YOLO_CONF_THRESHOLD,
        "yolo_drone_conf_threshold": TOP_YOLO_DRONE_CONF_THRESHOLD,
    }
    if BOTTOM_SOURCE_MODE == "single":
        bottom_config = {
            "worker_kind": "single",
            "instance_name": "bottom_single",
            "camera_name": BOTTOM_CAMERA_NAME,
            "window_name": f"{BOTTOM_CAMERA_NAME}-bottom",
            "layout": layouts["bottom"],
            "horizon_file": BOTTOM_HORIZON_FILE,
            "ignore_motion_file": BOTTOM_IGNORE_MOTION_FILE,
            "yolo_conf_threshold": BOTTOM_YOLO_CONF_THRESHOLD,
            "yolo_drone_conf_threshold": BOTTOM_YOLO_DRONE_CONF_THRESHOLD,
        }
    elif BOTTOM_SOURCE_MODE == "stitched":
        bottom_config = {
            "worker_kind": "bottom_stitch",
            "instance_name": "bottom_stitched",
            "window_name": "bottom-stitched",
            "layout": layouts["bottom"],
            "horizon_file": BOTTOM_HORIZON_FILE,
            "ignore_motion_file": BOTTOM_IGNORE_MOTION_FILE,
            "left_camera": BOTTOM_LEFT_CAMERA,
            "right_camera": BOTTOM_RIGHT_CAMERA,
            "preset_path": BOTTOM_STITCH_PRESET,
            "clone_right_from_left": BOTTOM_CLONE_RIGHT_FROM_LEFT,
            "synthetic_camera_name": "bottom_stitched",
            "yolo_conf_threshold": BOTTOM_YOLO_CONF_THRESHOLD,
            "yolo_drone_conf_threshold": BOTTOM_YOLO_DRONE_CONF_THRESHOLD,
        }
    else:
        raise ValueError(
            f"Unsupported BOTTOM_SOURCE_MODE '{BOTTOM_SOURCE_MODE}'. Use 'single' or 'stitched'."
        )
    return layouts, top_config, bottom_config


def _spawn_multi_worker_processes(ctx, top_config, bottom_config):
    bottom_kind = bottom_config.get("worker_kind", "bottom_stitch")
    bottom_name = "worker-bottom-single" if bottom_kind == "single" else "worker-bottom-stitched"
    return [
        ctx.Process(target=worker_entrypoint, args=("single", top_config), name="worker-top-rgb"),
        ctx.Process(target=worker_entrypoint, args=(bottom_kind, bottom_config), name=bottom_name),
    ]


def _drain_latest_frame(frame_queue, current_frame):
    latest = current_frame
    while True:
        try:
            payload = frame_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(payload, dict):
            latest = payload
    return latest


def _resize_to_layout(frame, layout):
    if frame is None:
        return np.zeros((layout["height"], layout["width"], 3), dtype=np.uint8)
    if frame.shape[1] == layout["width"] and frame.shape[0] == layout["height"]:
        return frame
    return cv2.resize(frame, (layout["width"], layout["height"]))


def _fit_frame_to_layout(frame, layout):
    canvas = np.zeros((layout["height"], layout["width"], 3), dtype=np.uint8)
    if frame is None:
        return canvas, None

    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return canvas, None

    scale = min(layout["width"] / src_w, layout["height"] / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    offset_x = max(0, (layout["width"] - target_w) // 2)
    offset_y = max(0, (layout["height"] - target_h) // 2)

    if target_w == src_w and target_h == src_h:
        resized = frame
    else:
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas[offset_y:offset_y + target_h, offset_x:offset_x + target_w] = resized
    return canvas, {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "display_w": target_w,
        "display_h": target_h,
        "src_w": src_w,
        "src_h": src_h,
    }


def _extract_frame_payload(payload):
    if isinstance(payload, dict):
        return payload.get("frame")
    return payload


def _offset_display_mapping(mapping, layout):
    if mapping is None:
        return None
    return {
        **mapping,
        "offset_x": mapping["offset_x"] + layout["x"],
        "offset_y": mapping["offset_y"] + layout["y"],
    }


def _map_display_coords_to_source(mapping, x, y):
    if mapping is None:
        return None
    local_x = x - mapping["offset_x"]
    local_y = y - mapping["offset_y"]
    if (
        local_x < 0
        or local_y < 0
        or local_x >= mapping["display_w"]
        or local_y >= mapping["display_h"]
    ):
        return None
    frame_x = int(local_x * mapping["src_w"] / mapping["display_w"])
    frame_y = int(local_y * mapping["src_h"] / mapping["display_h"])
    frame_x = max(0, min(mapping["src_w"] - 1, frame_x))
    frame_y = max(0, min(mapping["src_h"] - 1, frame_y))
    return frame_x, frame_y


def _send_control(control_queue, message):
    if control_queue is None:
        return
    while True:
        try:
            control_queue.put_nowait(message)
            return
        except queue.Full:
            try:
                control_queue.get_nowait()
            except queue.Empty:
                return


def _broadcast_shutdown(*control_queues):
    for control_queue in control_queues:
        _send_control(control_queue, {"type": "shutdown"})


def _cleanup_mp_queues(*queues):
    for mp_queue in queues:
        if mp_queue is None:
            continue
        try:
            mp_queue.close()
        except Exception:
            pass
        try:
            mp_queue.cancel_join_thread()
        except Exception:
            pass


def _force_stop_processes(processes, join_timeout=0.3, terminate_timeout=1.0, kill_timeout=1.0):
    for process in processes:
        if process.is_alive():
            process.join(timeout=join_timeout)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.is_alive():
            process.join(timeout=terminate_timeout)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in processes:
        if process.is_alive():
            process.join(timeout=kill_timeout)


def _get_window_visible_state(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except Exception:
        return None


def _run_multi_mode_split():
    _, top_config, bottom_config = _build_multi_worker_configs()
    ctx = mp.get_context("spawn")
    processes = _spawn_multi_worker_processes(ctx, top_config, bottom_config)
    exit_code = 0
    try:
        for process in processes:
            process.start()
        while processes:
            remaining = []
            for process in processes:
                process.join(timeout=0.2)
                if process.exitcode is None:
                    remaining.append(process)
                    continue
                if process.exitcode != 0 and exit_code == 0:
                    exit_code = process.exitcode or 1
                    raise RuntimeError(
                        f"Worker '{process.name}' exited with code {process.exitcode}"
                    )
            processes = remaining
    except KeyboardInterrupt:
        print("Stopping top/bottom workers...")
        exit_code = 130
    finally:
        print("Waiting for workers to save model (up to %ds)..." % GRACEFUL_SHUTDOWN_JOIN_TIMEOUT, flush=True)
        _force_stop_processes(
            processes,
            join_timeout=GRACEFUL_SHUTDOWN_JOIN_TIMEOUT,
            terminate_timeout=1.0,
            kill_timeout=1.0,
        )
    return exit_code


def _run_multi_mode_single_window():
    layouts, top_config, bottom_config = _build_multi_worker_configs()
    ctx = mp.get_context("spawn")
    top_frame_queue = ctx.Queue(maxsize=2)
    bottom_frame_queue = ctx.Queue(maxsize=2)
    top_control_queue = ctx.Queue(maxsize=64)
    bottom_control_queue = ctx.Queue(maxsize=128)

    top_config = {**top_config, "frame_queue": top_frame_queue, "control_queue": top_control_queue}
    bottom_config = {**bottom_config, "frame_queue": bottom_frame_queue, "control_queue": bottom_control_queue}

    processes = _spawn_multi_worker_processes(ctx, top_config, bottom_config)
    active_panel = "bottom"
    latest_top = None
    latest_bottom = None
    canvas_h = layouts["top"]["height"] + layouts["bottom"]["height"]
    canvas_w = max(layouts["top"]["width"], layouts["bottom"]["width"])
    window_close_check_after = time.time() + 2.0
    window_was_visible = False
    top_mapping = None
    bottom_mapping = None

    def mouse_callback(event, x, y, flags, userdata):
        nonlocal active_panel
        target_panel = None
        if layouts["top"]["y"] <= y < layouts["top"]["y"] + layouts["top"]["height"]:
            target_panel = "top"
        elif layouts["bottom"]["y"] <= y < layouts["bottom"]["y"] + layouts["bottom"]["height"]:
            target_panel = "bottom"
        if target_panel is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            active_panel = target_panel
        queue_ref = top_control_queue if target_panel == "top" else bottom_control_queue
        mapping = top_mapping if target_panel == "top" else bottom_mapping
        mapped_coords = _map_display_coords_to_source(mapping, x, y)
        if mapped_coords is None:
            return
        frame_x, frame_y = mapped_coords
        _send_control(
            queue_ref,
            {"type": "mouse", "event": int(event), "x": int(frame_x), "y": int(frame_y), "flags": int(flags)},
        )

    cv2.namedWindow(COMPOSITE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(COMPOSITE_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(COMPOSITE_WINDOW_NAME, canvas_w, canvas_h)
    cv2.setMouseCallback(COMPOSITE_WINDOW_NAME, mouse_callback)

    exit_code = 0
    try:
        for process in processes:
            process.start()
        while True:
            latest_top = _drain_latest_frame(top_frame_queue, latest_top)
            latest_bottom = _drain_latest_frame(bottom_frame_queue, latest_bottom)

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            top_frame, top_mapping_local = _fit_frame_to_layout(_extract_frame_payload(latest_top), layouts["top"])
            bottom_frame, bottom_mapping_local = _fit_frame_to_layout(_extract_frame_payload(latest_bottom), layouts["bottom"])
            top_mapping = _offset_display_mapping(top_mapping_local, layouts["top"])
            bottom_mapping = _offset_display_mapping(bottom_mapping_local, layouts["bottom"])

            top_y = layouts["top"]["y"]
            bottom_y = layouts["bottom"]["y"]
            canvas[top_y:top_y + layouts["top"]["height"], 0:layouts["top"]["width"]] = top_frame
            canvas[bottom_y:bottom_y + layouts["bottom"]["height"], 0:layouts["bottom"]["width"]] = bottom_frame

            active_rect = layouts["top"] if active_panel == "top" else layouts["bottom"]
            cv2.rectangle(
                canvas,
                (active_rect["x"], active_rect["y"]),
                (active_rect["x"] + active_rect["width"] - 1, active_rect["y"] + active_rect["height"] - 1),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                f"ACTIVE PANEL: {active_panel.upper()}  TAB=switch  q/Esc=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            # Session drone count ต่อโซน (จาก payload ของแต่ละ worker)
            if isinstance(latest_top, dict):
                ts = latest_top.get("session_drone_count", 0)
                tc = latest_top.get("current_drone_count", 0)
                cv2.putText(canvas, f"T Session: {ts} | Now: {tc}", (20, top_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if isinstance(latest_bottom, dict):
                bs = latest_bottom.get("session_drone_count", 0)
                bc = latest_bottom.get("current_drone_count", 0)
                cv2.putText(canvas, f"B Session: {bs} | Now: {bc}", (20, bottom_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(COMPOSITE_WINDOW_NAME, canvas)

            window_visible_state = _get_window_visible_state(COMPOSITE_WINDOW_NAME)
            if window_visible_state is not None and window_visible_state >= 1:
                window_was_visible = True
            can_check_window_close = (
                time.time() >= window_close_check_after
                and (latest_top is not None or latest_bottom is not None)
                and window_was_visible
            )
            if (
                can_check_window_close
                and window_visible_state is not None
                and window_visible_state < 1
            ):
                _broadcast_shutdown(top_control_queue, bottom_control_queue)
                break

            for process in processes:
                if process.exitcode is not None and process.exitcode != 0 and exit_code == 0:
                    exit_code = process.exitcode or 1
                    raise RuntimeError(
                        f"Worker '{process.name}' exited with code {process.exitcode}"
                    )
            if all(process.exitcode is not None for process in processes):
                break

            key_raw = cv2.waitKeyEx(1)
            key = -1 if key_raw == -1 else (key_raw & 0xFF)
            if key in (ord("q"), ord("Q"), 27):
                _broadcast_shutdown(top_control_queue, bottom_control_queue)
                break
            if key in (9,):
                active_panel = "bottom" if active_panel == "top" else "top"
                continue
            if key not in (-1, 255):
                target_queue = top_control_queue if active_panel == "top" else bottom_control_queue
                _send_control(target_queue, {"type": "key", "key": int(key)})
    except KeyboardInterrupt:
        print("Stopping compositor/workers...")
        exit_code = 130
    finally:
        _broadcast_shutdown(top_control_queue, bottom_control_queue)
        try:
            cv2.destroyWindow(COMPOSITE_WINDOW_NAME)
        except Exception:
            pass
        print("Waiting for workers to save model (up to %ds)..." % GRACEFUL_SHUTDOWN_JOIN_TIMEOUT, flush=True)
        _force_stop_processes(
            processes,
            join_timeout=GRACEFUL_SHUTDOWN_JOIN_TIMEOUT,
            terminate_timeout=1.0,
            kill_timeout=1.0,
        )
        _cleanup_mp_queues(
            top_frame_queue,
            bottom_frame_queue,
            top_control_queue,
            bottom_control_queue,
        )
    return exit_code


def _build_single_worker_config():
    return {
        "instance_name": "single_camera",
        "camera_name": SINGLE_CAMERA_NAME,
        "window_name": f"{SINGLE_CAMERA_NAME}-single",
        "layout": _build_fullscreen_layout(),
        "horizon_file": f"horizon_poly_{SINGLE_CAMERA_NAME}_single.npy",
        "ignore_motion_file": f"motion_ignore_mask_{SINGLE_CAMERA_NAME}_single.npy",
        "yolo_conf_threshold": SINGLE_YOLO_CONF_THRESHOLD,
        "yolo_drone_conf_threshold": SINGLE_YOLO_DRONE_CONF_THRESHOLD,
    }


def _run_single_mode_direct():
    _run_single_camera_worker(_build_single_worker_config())


def _run_single_mode_single_window():
    worker_config = _build_single_worker_config()
    layout = worker_config["layout"]
    window_name = worker_config["window_name"]
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=2)
    control_queue = ctx.Queue(maxsize=128)
    worker_config = {
        **worker_config,
        "frame_queue": frame_queue,
        "control_queue": control_queue,
    }

    process = ctx.Process(
        target=worker_entrypoint,
        args=("single", worker_config),
        name="worker-single-camera",
    )
    latest_frame = None
    window_close_check_after = time.time() + 2.0
    window_was_visible = False
    exit_code = 0
    display_mapping = None

    def mouse_callback(event, x, y, flags, userdata):
        mapped_coords = _map_display_coords_to_source(display_mapping, x, y)
        if mapped_coords is None:
            return
        frame_x, frame_y = mapped_coords
        _send_control(
            control_queue,
            {
                "type": "mouse",
                "event": int(event),
                "x": int(frame_x),
                "y": int(frame_y),
                "flags": int(flags),
            },
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, layout["width"], layout["height"])
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        process.start()
        while True:
            latest_frame = _drain_latest_frame(frame_queue, latest_frame)
            display_frame, display_mapping = _fit_frame_to_layout(_extract_frame_payload(latest_frame), layout)
            cv2.imshow(window_name, display_frame)

            window_visible_state = _get_window_visible_state(window_name)
            if window_visible_state is not None and window_visible_state >= 1:
                window_was_visible = True
            can_check_window_close = (
                time.time() >= window_close_check_after
                and latest_frame is not None
                and window_was_visible
            )
            if (
                can_check_window_close
                and window_visible_state is not None
                and window_visible_state < 1
            ):
                _broadcast_shutdown(control_queue)
                break

            if process.exitcode is not None:
                if process.exitcode != 0:
                    exit_code = process.exitcode or 1
                    raise RuntimeError(
                        f"Worker '{process.name}' exited with code {process.exitcode}"
                    )
                break

            key_raw = cv2.waitKeyEx(1)
            key = -1 if key_raw == -1 else (key_raw & 0xFF)
            if key in (ord("q"), ord("Q"), 27):
                _broadcast_shutdown(control_queue)
                break
            if key not in (-1, 255):
                _send_control(control_queue, {"type": "key", "key": int(key)})
    except KeyboardInterrupt:
        print("Stopping single-camera compositor/worker...")
        exit_code = 130
    finally:
        _broadcast_shutdown(control_queue)
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        print("Waiting for worker to save model (up to %ds)..." % GRACEFUL_SHUTDOWN_JOIN_TIMEOUT, flush=True)
        _force_stop_processes(
            [process],
            join_timeout=GRACEFUL_SHUTDOWN_JOIN_TIMEOUT,
            terminate_timeout=1.0,
            kill_timeout=1.0,
        )
        _cleanup_mp_queues(frame_queue, control_queue)
    return exit_code


def _run_single_mode():
    if SINGLE_CAMERA_VIEW_MODE == "direct":
        _run_single_mode_direct()
        return 0
    if SINGLE_CAMERA_VIEW_MODE == "single_window":
        return _run_single_mode_single_window()
    raise ValueError(
        f"Unsupported SINGLE_CAMERA_VIEW_MODE '{SINGLE_CAMERA_VIEW_MODE}'. Use 'single_window' or 'direct'."
    )


def main():
    if RUN_MODE == "single_camera":
        return _run_single_mode()
    if RUN_MODE == "multi_camera_top_bottom":
        if MULTI_CAMERA_VIEW_MODE == "split_windows":
            return _run_multi_mode_split()
        if MULTI_CAMERA_VIEW_MODE == "single_window":
            return _run_multi_mode_single_window()
        raise ValueError(
            f"Unsupported MULTI_CAMERA_VIEW_MODE '{MULTI_CAMERA_VIEW_MODE}'. Use 'single_window' or 'split_windows'."
        )
    raise ValueError(
        f"Unsupported RUN_MODE '{RUN_MODE}'. Use 'single_camera' or 'multi_camera_top_bottom'."
    )


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())



