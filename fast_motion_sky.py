import cv2
import numpy as np
import time
import platform
import threading
import os
import importlib
import sys
import math

from memory_manager import init_memory_management, kill_old_instances
# เรียกใช้งานเพื่อเตรียมแรมสำหรับการตรวจจับ Motion ขนาดใหญ่
init_memory_management()

# Clear config cache and reload config module to ensure fresh config
config_cache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
if os.path.exists(config_cache_dir):
    # ลบไฟล์ .pyc ของ config module
    for file in os.listdir(config_cache_dir):
        if file.startswith('config') and file.endswith('.pyc'):
            try:
                os.remove(os.path.join(config_cache_dir, file))
            except:
                pass

# Reload config module if already imported
if 'config' in sys.modules:
    importlib.reload(sys.modules['config'])

# Import all config parameters
try:
    from config import (
        ACTIVE_CAMERA, get_camera_config, get_pixel_to_degree, pixel_to_angle,
        has_zoom, get_fov_at_zoom,
        # Global settings
        OPENCV_NUM_THREADS, OMP_NUM_THREADS,
        DEBUG_MODE, SHOW_GRID, BBOX_PADDING, BBOX_THICKNESS,
        GRID_ROWS, GRID_COLS, LEARNING_RATE, GRID_NOISE_FILTER_THRESHOLD, MERGE_DISTANCE,
        MIN_AREA_BASE, MIN_AREA_NOISE_MULTIPLIER,
        MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
        MORPH_KERNEL_SIZE, MOTION_AREA_MARGIN,
        CAM_MOVE_DETECTION_ENABLED, CAM_MOVE_DETECTION_INTERVAL,
        CAM_MOVE_THRESHOLD, CAM_MOVE_RESET_BACKGROUND,
        CAM_MOVE_LEARNING_RATE, CAM_MOVE_LEARNING_RATE_FRAMES,
        ADAPTIVE_ENABLED, ADAPTIVE_LEARNING_RATE,
        BASE_VAR_THRESHOLD, MAX_VAR_THRESHOLD,
        BASE_MIN_AREA_REF, MAX_MIN_AREA_REF,
        CONTOUR_COUNT_THRESHOLD_LOW, CONTOUR_COUNT_THRESHOLD_HIGH,
        MORPH_KERNEL_SIZE_MIN, MORPH_KERNEL_SIZE_MAX,
        ADAPTIVE_UPDATE_INTERVAL, MOG2_RECREATE_THRESHOLD,
        SKY_GATING_ENABLED,
        EDGE_EXCLUSION_PIXELS, HORIZON_EXCLUSION_PIXELS,
        MAX_MISS_FRAMES, PATH_HISTORY_LENGTH, MAX_REID_DISTANCE,
        ENABLE_OBJECT_CLASSIFICATION,
        DRONE_MIN_AREA, DRONE_MAX_AREA, DRONE_MIN_SPEED, DRONE_MAX_SPEED,
        DRONE_MIN_PATH_FRAMES, DRONE_AREA_WEIGHT, DRONE_SPEED_WEIGHT,
        DRONE_PATH_WEIGHT, DRONE_MIN_SCORE, DRONE_CONFIRMATION_MIN_SCORE,
        DRONE_MIN_STRAIGHTNESS, DRONE_MAX_STRAIGHTNESS, DRONE_MIN_SMOOTHNESS,
        DRONE_MIN_VELOCITY_CV, DRONE_MAX_VELOCITY_CV, DRONE_MIN_DIRECTION_CONSISTENCY,
        DRONE_YELLOW_DURATION_THRESHOLD, DRONE_MIN_SMOOTHNESS_SIMPLE, DRONE_MIN_DIRECTION_CONSISTENCY_SIMPLE,
        DRONE_MIN_PATH_FRAMES_FOR_ORANGE, DRONE_MIN_VELOCITY_FOR_ORANGE, DRONE_MIN_VELOCITY_FOR_HOVER,
        DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE, DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER,
        DRONE_MIN_MOVEMENT_FRAMES_FOR_ORANGE, DRONE_MIN_MOVEMENT_RATIO,
        MAX_RECT_SIZE_CHANGE_RATIO, MAX_MERGE_AREA_RATIO,
        MAX_ASPECT_RATIO, MIN_ASPECT_RATIO,
        DRONE_MIN_YELLOW_PATH_POINTS, DRONE_MIN_PATH_TOTAL_DISTANCE, DRONE_PATH_MAX_GAP_RATIO,
        DRONE_YELLOW_PATH_HISTORY_LIMIT, DRONE_HOVER_SMOOTHNESS_BONUS, DRONE_HOVER_CONSISTENCY_BONUS,
        INSECT_MAX_VELOCITY, INSECT_MIN_STRAIGHTNESS, INSECT_MAX_CONTINUITY,
        CLOUD_DETECTION_ENABLED, CLOUD_MIN_VELOCITY, CLOUD_MAX_VELOCITY,
        CLOUD_MIN_AREA, CLOUD_MAX_STRAIGHTNESS, CLOUD_MAX_VELOCITY_CV, CLOUD_MAX_SMOOTHNESS,
        HOVER_VELOCITY_THRESHOLD, HOVER_MIN_FRAMES, HOVER_RATIO_THRESHOLD,
        CLASSIFICATION_UPDATE_INTERVAL, PATH_QUALITY_HISTORY_LIMIT, CHECK_STATUS_UPDATE_INTERVAL,
        VELOCITY_HISTORY_LIMIT, HOVER_HISTORY_LIMIT,
        ALERT_COLOR, NORMAL_COLOR, ORANGE_COLOR, RED_COLOR, HORIZON_COLOR, DRAWING_COLOR,
        SOUND_ALERT_ENABLED, SOUND_FILE, SOUND_CHECK_INTERVAL, SOUND_RED_FRAME_THRESHOLD,
        COLOR_TEXT, HUD_BACKGROUND_COLOR, HUD_ENABLED, HUD_MARGIN, HUD_WIDTH_RATIO, HUD_MIN_WIDTH, HUD_MAX_WIDTH,
        FPS_GOOD_COLOR, FPS_BAD_COLOR, STATS_TEXT_COLOR, ALGO_TIME_COLOR,
        DEFAULT_FPS, MAX_FPS, FPS_GOOD_THRESHOLD,
        SIZE_CHANGE_MAX_MULTIPLIER,
        PATH_VISUALIZATION_ENABLED, PATH_MAX_POINTS,
        HEAT_POINT_ENABLED, HEAT_POINT_RADIUS,
        DRAW_GREEN_BOX, DRAW_YELLOW_BOX, DRAW_ORANGE_BOX, DRAW_RED_BOX,
        DRAW_PATH, DRAW_HEAT_POINT, DRAW_LABELS,
        DRAW_GREEN_PATH, DRAW_YELLOW_PATH, DRAW_ORANGE_PATH, DRAW_RED_PATH,
        DRAW_MOTION_BOXES,
        HYBRID_ROI_PADDING, HYBRID_MIN_MOTION_AREA, HYBRID_DIST_THRESHOLD, HYBRID_MAX_MISS_ALLOWED, HYBRID_ROI_WAIT_FRAMES,
        HYBRID_MODEL_FILE, HYBRID_SEARCH_CONF, HYBRID_YOLO_INTERVAL, HYBRID_BASE_CONF, HYBRID_BASE_SEARCH_CONF,
        HYBRID_SEARCH_CONF_MEDIUM, HYBRID_SEARCH_CONF_LOW,
        HYBRID_SEARCH_INTERVAL_NO_TARGET, HYBRID_SEARCH_INTERVAL_WITH_TARGET,
        HYBRID_MOTION_PARTIAL_DIST, HYBRID_MOTION_PARTIAL_BOOST,
        MIN_CONFIDENCE_FOR_TRACKING_SEARCH,
        MAX_ROI_TRACKING,
        ADAPTIVE_HYBRID_MIN_AREA_ENABLED, ADAPTIVE_HYBRID_MIN_AREA_BASE,
        ADAPTIVE_HYBRID_MIN_AREA_MAX, ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_LOW,
        ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_HIGH, ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_LOW,
        ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_HIGH, ADAPTIVE_HYBRID_MIN_AREA_TIME_THRESHOLD_HIGH,
        ADAPTIVE_HYBRID_MIN_AREA_UPDATE_INTERVAL, ADAPTIVE_HYBRID_MIN_AREA_ADJUSTMENT_STEP,
        # Size-based min motion area settings
        TINY_OBJECT_MIN_MOTION_AREA_BASE, TINY_OBJECT_MIN_MOTION_AREA_MAX,
        SMALL_OBJECT_MIN_MOTION_AREA_BASE, SMALL_OBJECT_MIN_MOTION_AREA_MAX,
        MEDIUM_OBJECT_MIN_MOTION_AREA_BASE, MEDIUM_OBJECT_MIN_MOTION_AREA_MAX,
        ADAPTIVE_TINY_OBJECT_MIN_AREA_ENABLED,
        STATIONARY_DETECTION_ENABLED, MAX_STATIONARY_DISTANCE, STATIONARY_CHECK_FRAMES,
        BLACKLIST_DURATION_SECONDS, BLACKLIST_BBOX_PADDING, STATIONARY_CENTER_THRESHOLD,
        PERMANENT_BLACKLIST_ENABLED, PERMANENT_BLACKLIST_THRESHOLD, PERMANENT_BLACKLIST_WINDOW_SECONDS,
        # Resolution-dependent thresholds (ratios)
        REFERENCE_RESOLUTION, REID_DISTANCE_RATIO, MERGE_DISTANCE_RATIO,
        HYBRID_DIST_THRESHOLD_RATIO, HYBRID_ROI_PADDING_RATIO, SMALL_OBJECT_REID_DISTANCE_RATIO,
        MAX_STATIONARY_DISTANCE_RATIO, STATIONARY_CENTER_THRESHOLD_RATIO,
        BLACKLIST_BBOX_PADDING_RATIO, BLACKLIST_MIN_VELOCITY_RATIO,
        EDGE_EXCLUSION_PIXELS_RATIO, HORIZON_EXCLUSION_PIXELS_RATIO,
        MOTION_AREA_MARGIN_RATIO, MIN_AREA_BASE_RATIO,
        DRONE_MIN_VELOCITY_FOR_ORANGE_RATIO, DRONE_MIN_VELOCITY_FOR_HOVER_RATIO,
        DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE_RATIO, DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER_RATIO,
        DRONE_MIN_AREA_RATIO, DRONE_MAX_AREA_RATIO, DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO,
        DRONE_MIN_PATH_TOTAL_DISTANCE_RATIO,
        BBOX_PADDING_RATIO, HUD_MARGIN_RATIO, HUD_MIN_WIDTH_RATIO, HUD_MAX_WIDTH_RATIO,
        CAM_MOVE_THRESHOLD_RATIO,
        PREDICTED_POSITION_LOOKBACK, SMALL_OBJECT_PATH_QUALITY_BONUS,
        MIN_PATH_QUALITY_FOR_SMALL_OBJECT, MAX_MISS_FRAMES_FOR_SMALL_OBJECT,
        PERMANENT_BLACKLIST_DURATION_SECONDS, BLACKLIST_OVERLAP_THRESHOLD,
        BLACKLIST_ALLOW_MOVEMENT, BLACKLIST_MIN_VELOCITY, BLACKLIST_MOTION_IOU_THRESHOLD,
        HYBRID_MOTION_IOU_THRESHOLD, HYBRID_MOTION_CONF_BOOST, HYBRID_MOTION_CONF_MAX,
        ADAPTIVE_HYBRID_MODE_ENABLED, ADAPTIVE_HYBRID_HYSTERESIS_FRAMES,
        ADAPTIVE_HYBRID_MIN_STATUS_FOR_LOCK, ADAPTIVE_HYBRID_SEARCH_TIMEOUT,
        ADAPTIVE_HYBRID_UPDATE_INTERVAL, GRID_CONFIDENCE_CACHE_FRAMES,
        GRID_NOISE_THRESHOLD_FOR_HYBRID, GRID_NOISE_THRESHOLD_FOR_MULTI, GRID_CONFIDENCE_WEIGHT,
        RED_ACCUM_WINDOW_FRAMES, RED_LOCK_SCORE, RED_DECAY_FACTOR,
        LOCK_YOLO_INTERVAL, LOCK_ROI_WAIT_MULTIPLIER, LOCK_DIST_MULTIPLIER,
        MOTION_STATIONARY_CHECK_INTERVAL, MOTION_STATIONARY_MOVEMENT_THRESHOLD_RATIO, MOTION_STATIONARY_FRAMES_TO_EXIT,
        DRONE_PATH_CONTINUITY_ENABLED, DRONE_SIZE_STABILITY_ENABLED,
        DRONE_PATH_CONTINUITY_MIN_POINTS, DRONE_PATH_MAX_GAP_RATIO,
        DRONE_PATH_SMOOTH_TRANSITION_THRESHOLD, DRONE_PATH_CONTINUITY_BOOST,
        DRONE_SIZE_STABILITY_MIN_POINTS, DRONE_SIZE_STABILITY_CV_THRESHOLD,
        DRONE_SIZE_STABILITY_BOOST, DRONE_ADAPTIVE_GAP_RATIO_LARGE,
        DRONE_ADAPTIVE_GAP_RATIO_SMALL, DRONE_ADAPTIVE_SMOOTH_THRESHOLD_LARGE,
        DRONE_ADAPTIVE_SMOOTH_THRESHOLD_SMALL, DRONE_MOTION_BOX_SIZE_THRESHOLD_LARGE,
        DRONE_MOTION_BOX_SIZE_THRESHOLD_SMALL,
        DRONE_BACKGROUND_NOISE_CHECK_ENABLED, DRONE_BACKGROUND_NOISE_THRESHOLD,
        DRONE_BACKGROUND_MOTION_IN_ROI_THRESHOLD,
        DRONE_PATH_HISTORY_OPTIMIZATION_ENABLED, DRONE_GREEN_PATH_HISTORY_ENABLED,
        DRONE_TINY_PATH_HISTORY_MAX_FRAMES, DRONE_SMALL_PATH_HISTORY_MAX_FRAMES,
        DRONE_MEDIUM_PATH_HISTORY_MAX_FRAMES, DRONE_LARGE_PATH_HISTORY_MAX_FRAMES,
        DRONE_GREEN_TINY_PATH_INTERVAL, DRONE_GREEN_SMALL_PATH_INTERVAL,
        DRONE_GREEN_MEDIUM_PATH_INTERVAL, DRONE_GREEN_LARGE_PATH_INTERVAL,
        DRONE_PATH_HISTORY_CLEANUP_INTERVAL,
        DRONE_PATH_BASED_ORANGE_ENABLED, DRONE_PATH_BASED_ORANGE_MIN_HISTORY_SECONDS,
        DRONE_PATH_BASED_ORANGE_MIN_CONFIDENCE, DRONE_PATH_BASED_ORANGE_MIN_TRACKING_DURATION_SECONDS,
        DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS, DRONE_PATH_BASED_ORANGE_MOTION_CLEAR_THRESHOLD,
        # Path validation settings
        PATH_VALIDATION_ENABLED, MIN_MOTION_SCORE_THRESHOLD, OUTLIER_DISTANCE_THRESHOLD_RATIO,
        OUTLIER_DIRECTION_THRESHOLD, OUTLIER_SIZE_THRESHOLD_RATIO,
        # Temporal continuity check settings
        TEMPORAL_CONTINUITY_ENABLED, TEMPORAL_CONTINUITY_MIN_PATH_POINTS,
        TEMPORAL_CONTINUITY_DISTANCE_THRESHOLD_RATIO, VELOCITY_CHANGE_THRESHOLD_RATIO,
        VELOCITY_DIRECTION_CHANGE_THRESHOLD,
        # Path smoothing settings
        PATH_SMOOTHING_ENABLED, PATH_SMOOTHING_WINDOW_SIZE,
        # ROI priority system settings
        MAX_ROI_DRAW_LIMIT, ROI_PRIORITY_CONTINUITY_WEIGHT, ROI_PRIORITY_PATH_LENGTH_WEIGHT,
        ROI_PRIORITY_TEMPORARY_LOSS_BONUS, ROI_PRIORITY_TRACKING_DURATION_WEIGHT,
        ROI_VALIDATION_ENABLED, ROI_MIN_IOU_THRESHOLD, ROI_MIN_MOVEMENT_THRESHOLD,
        ROI_MAX_DIRECTION_CHANGE, ROI_HISTORY_MAX_FRAMES, ROI_VALIDATION_MIN_HISTORY,
        ROI_VALIDATION_SKIP_RED_ORANGE, ROI_VALIDATION_SKIP_FIRST_FRAMES, ROI_VALIDATION_MOTION_ONLY_MIN_PATH,
        MOTION_ONLY_ORANGE_PATH_QUALITY_THRESHOLD, MOTION_ONLY_RED_PATH_QUALITY_THRESHOLD,
        MOTION_ONLY_RED_MIN_PATH_POINTS, MOTION_ONLY_STATUS_CHECK_INTERVAL, ROI_PATH_HISTORY_MAX_POINTS,
        # ROI dense motion filter settings
        ROI_DENSE_MOTION_FILTER_ENABLED, ROI_DENSE_MOTION_MAX_BOXES,
        ROI_DENSE_MOTION_COVERAGE_THRESHOLD, ROI_DENSE_MOTION_EXTENDED_AREA_MULTIPLIER,
        # Acceleration check settings
        ACCELERATION_CHECK_ENABLED, MAX_ACCELERATION_RATIO, MAX_DECELERATION_RATIO, ACCELERATION_CHECK_MIN_PATH_POINTS,
        # Motion box freshness check settings
        MOTION_BOX_FRESHNESS_CHECK_ENABLED, MOTION_BOX_MAX_AGE_FRAMES, MOTION_BOX_FRESHNESS_ROI_CHECK,
        # Path smoothness check settings
        PATH_SMOOTHNESS_CHECK_ENABLED, MAX_PATH_JUMP_RATIO, MAX_PATH_JUMP_RATIO_TINY,
        DRONE_PATH_BASED_ORANGE_CHECK_INTERVAL,
        MOTION_ONLY_TARGET_ENABLED, MOTION_ONLY_MIN_PATH_POINTS, MOTION_ONLY_MIN_PATH_QUALITY,
        MOTION_ONLY_STORE_PATH_EVERY_FRAME,
        FOOTPRINTS_MODULE_ENABLED, FOOTPRINTS_HISTORY_FRAMES,
        MOTION_ONLY_MIN_TRACKING_DURATION, MOTION_ONLY_CHECK_INTERVAL, MOTION_ONLY_MAX_TARGETS,
        MOTION_ONLY_PATH_HISTORY_MAX_FRAMES,
        # YOLO immediate RED lock settings
        YOLO_IMMEDIATE_RED_ENABLED, YOLO_IMMEDIATE_RED_CONF_THRESHOLD,
        # Path classification settings
        PATH_CLASSIFICATION_ENABLED, PATH_CLASSIFICATION_MIN_POINTS, PATH_CLASSIFICATION_RECOMMENDED_POINTS,
        PATH_CLASSIFICATION_UPDATE_INTERVAL, PATH_CLASSIFICATION_USE_RESOLUTION_ADAPTIVE,
        PATH_CLASSIFICATION_USE_FPS_AWARE, PATH_VELOCITY_FPS_MULTIPLIER, PATH_VELOCITY_FPS_REFERENCE,
        DRONE_STRAIGHTNESS_THRESHOLD, DRONE_SMOOTHNESS_THRESHOLD, DRONE_VELOCITY_CONSISTENCY_THRESHOLD,
        DRONE_VELOCITY_MIN, DRONE_VELOCITY_MAX,
        BIRD_STRAIGHTNESS_THRESHOLD, BIRD_SMOOTHNESS_THRESHOLD, BIRD_VELOCITY_CONSISTENCY_THRESHOLD,
        BIRD_VELOCITY_MIN, BIRD_VELOCITY_MAX,
        INSECT_STRAIGHTNESS_THRESHOLD, INSECT_SMOOTHNESS_THRESHOLD, INSECT_VELOCITY_CONSISTENCY_THRESHOLD,
        INSECT_VELOCITY_MIN, INSECT_VELOCITY_MAX,
        AIRPLANE_STRAIGHTNESS_THRESHOLD, AIRPLANE_SMOOTHNESS_THRESHOLD, AIRPLANE_VELOCITY_CONSISTENCY_THRESHOLD,
        AIRPLANE_VELOCITY_MIN, AIRPLANE_VELOCITY_MAX,
        NOISE_STRAIGHTNESS_THRESHOLD, NOISE_SMOOTHNESS_THRESHOLD, NOISE_VELOCITY_CONSISTENCY_THRESHOLD,
        TINY_MOTION_PATH_WEIGHT, TINY_YOLO_WEIGHT, TINY_MIN_PATH_POINTS, TINY_MIN_PATH_QUALITY,
        TINY_MIN_TRACKING_DURATION, TINY_YOLO_IMMEDIATE_LOCK_CONF, TINY_RED_YOLO_WEIGHT, TINY_RED_MOTION_WEIGHT,
        SMALL_MOTION_PATH_WEIGHT, SMALL_YOLO_WEIGHT, SMALL_MIN_PATH_POINTS, SMALL_MIN_PATH_QUALITY,
        SMALL_MIN_TRACKING_DURATION, SMALL_YOLO_IMMEDIATE_LOCK_CONF, SMALL_RED_YOLO_WEIGHT, SMALL_RED_MOTION_WEIGHT,
        MEDIUM_MOTION_PATH_WEIGHT, MEDIUM_YOLO_WEIGHT, MEDIUM_MIN_PATH_POINTS, MEDIUM_MIN_PATH_QUALITY,
        MEDIUM_MIN_TRACKING_DURATION, MEDIUM_YOLO_IMMEDIATE_LOCK_CONF, MEDIUM_RED_YOLO_WEIGHT, MEDIUM_RED_MOTION_WEIGHT,
        LARGE_MOTION_PATH_WEIGHT, LARGE_YOLO_WEIGHT, LARGE_MIN_PATH_POINTS, LARGE_MIN_PATH_QUALITY,
        LARGE_MIN_TRACKING_DURATION, LARGE_YOLO_IMMEDIATE_LOCK_CONF, LARGE_RED_YOLO_WEIGHT, LARGE_RED_MOTION_WEIGHT,
        BLENDED_OBJECT_MOTION_BOOST, BLENDED_OBJECT_YOLO_PENALTY, BLENDED_DETECTION_ENABLED,
        MAX_TOTAL_TARGETS,
        TINY_SIZE_STABILITY_CV_THRESHOLD, SMALL_SIZE_STABILITY_CV_THRESHOLD,
        MEDIUM_SIZE_STABILITY_CV_THRESHOLD, LARGE_SIZE_STABILITY_CV_THRESHOLD,
        TINY_SIZE_STABILITY_WEIGHT, SMALL_SIZE_STABILITY_WEIGHT,
        MEDIUM_SIZE_STABILITY_WEIGHT, LARGE_SIZE_STABILITY_WEIGHT,
        CAM2_PTZ_ENABLED, cam1_pixel_to_cam2_ptz_units, get_camera_config,
        CAM2_PTZ_PIP_ENABLED, CAM2_PTZ_PIP_WIDTH, CAM2_PTZ_PIP_HEIGHT,
        CAM2_PTZ_PIP_MARGIN, CAM2_PTZ_PIP_BORDER_COLOR, CAM2_PTZ_PIP_BORDER_THICKNESS,
        CAM2_PTZ_SEARCH_CONF, CAM2_PTZ_BLACKLIST_DURATION, CAM2_PTZ_BLACKLIST_RADIUS,
        CAM2_PTZ_REMOVE_TARGET_ON_FAIL, CAM2_PTZ_REMOVE_TARGET_CONF_THRESHOLD,
        # Object size definitions
        TINY_OBJECT_AREA_THRESHOLD_RATIO, TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO,
        SMALL_OBJECT_DIAGONAL_THRESHOLD_RATIO, TINY_OBJECT_YOLO_INTERVAL,
        SMALL_OBJECT_YOLO_INTERVAL,
        # Path-based analysis settings
        SMALL_OBJECT_PATH_ONLY_MODE, SMALL_OBJECT_MIN_PATH_FRAMES,
        SMALL_OBJECT_PATH_FRAMES_PER_SECOND, AIRPLANE_STRAIGHTNESS_THRESHOLD,
        DRONE_CURVATURE_THRESHOLD, SMALL_OBJECT_MAX_VELOCITY_RATIO,
        SMALL_OBJECT_MIN_VELOCITY_RATIO,         SMALL_OBJECT_PATH_SMOOTHING_WINDOW,
        SMALL_OBJECT_CURVATURE_WINDOW, SMALL_OBJECT_MIN_CURVATURE_FOR_DRONE,
        SMALL_OBJECT_CONFIDENCE_THRESHOLD,
        # Tiny objects path analysis settings
        TINY_OBJECT_PATH_HISTORY_LENGTH, TINY_OBJECT_AIRPLANE_STRAIGHTNESS_THRESHOLD,
        TINY_OBJECT_STAR_CLUSTERING_THRESHOLD,
        # Path-based RED detection settings
        TINY_OBJECT_PATH_BASED_RED_ENABLED, TINY_OBJECT_PATH_BASED_RED_MIN_HISTORY_SECONDS,
        TINY_OBJECT_PATH_BASED_RED_MIN_CONFIDENCE, TINY_OBJECT_PATH_BASED_RED_MIN_TRACKING_DURATION_SECONDS,
        # Airplane detection settings
        TINY_OBJECT_AIRPLANE_PATH_MIN_POINTS, TINY_OBJECT_AIRPLANE_SIZE_STABILITY_THRESHOLD,
        TINY_OBJECT_AIRPLANE_SIZE_HISTORY_MIN_POINTS,
        # Path history settings
        TINY_OBJECT_PATH_HISTORY_SECONDS,
        # Resolution-dependent parameters
        RES_4K_FRAME_SKIP_MOG2, RES_4K_MOG2_HISTORY, RES_4K_MOG2_HISTORY_4K,
        RES_4K_PERIODIC_MEMORY_CLEANUP, RES_4K_DISPLAY_UPDATE_INTERVAL,
        RES_4K_MAX_CONTOURS_TO_PROCESS, RES_4K_FULL_FRAME_YOLO_INTERVAL,
        RES_4K_FULL_FRAME_YOLO_CONF, RES_4K_MOG2_VAR_THRESHOLD,
        RES_4K_BASE_VAR_THRESHOLD, RES_4K_GRID_ROWS, RES_4K_GRID_COLS,
        RES_4K_BASE_MIN_AREA_REF, RES_4K_MAX_MIN_AREA_REF,
        RES_4K_HYBRID_ROI_PADDING, RES_4K_HYBRID_DIST_THRESHOLD,
        RES_4K_HYBRID_BASE_SEARCH_CONF, RES_4K_HYBRID_MIN_MOTION_AREA,
        RES_4K_HEIGHT_FRAME_SKIP_MOG2, RES_4K_HEIGHT_MOG2_HISTORY,
        RES_4K_HEIGHT_MOG2_HISTORY_4K,
        RES_2K_FRAME_SKIP_MOG2, RES_2K_MOG2_HISTORY, RES_2K_MOG2_HISTORY_4K,
        RES_2K_PERIODIC_MEMORY_CLEANUP, RES_2K_DISPLAY_UPDATE_INTERVAL,
        RES_2K_MAX_CONTOURS_TO_PROCESS, RES_2K_FULL_FRAME_YOLO_INTERVAL,
        RES_2K_FULL_FRAME_YOLO_CONF, RES_2K_MOG2_VAR_THRESHOLD,
        RES_2K_BASE_VAR_THRESHOLD, RES_2K_GRID_ROWS, RES_2K_GRID_COLS,
        RES_2K_BASE_MIN_AREA_REF, RES_2K_MAX_MIN_AREA_REF,
        RES_2K_HYBRID_ROI_PADDING, RES_2K_HYBRID_DIST_THRESHOLD,
        RES_2K_HYBRID_BASE_SEARCH_CONF, RES_2K_HYBRID_MIN_MOTION_AREA,
        RES_1080P_FRAME_SKIP_MOG2, RES_1080P_MOG2_HISTORY, RES_1080P_MOG2_HISTORY_4K,
        RES_1080P_PERIODIC_MEMORY_CLEANUP, RES_1080P_DISPLAY_UPDATE_INTERVAL,
        RES_1080P_MAX_CONTOURS_TO_PROCESS, RES_1080P_FULL_FRAME_YOLO_INTERVAL,
        RES_1080P_FULL_FRAME_YOLO_CONF, RES_1080P_MOG2_VAR_THRESHOLD,
        RES_1080P_BASE_VAR_THRESHOLD, RES_1080P_GRID_ROWS, RES_1080P_GRID_COLS,
        RES_1080P_BASE_MIN_AREA_REF, RES_1080P_MAX_MIN_AREA_REF,
        RES_1080P_HYBRID_ROI_PADDING, RES_1080P_HYBRID_DIST_THRESHOLD,
        RES_LOWER_FRAME_SKIP_MOG2, RES_LOWER_MOG2_HISTORY, RES_LOWER_MOG2_HISTORY_4K,
        RES_LOWER_PERIODIC_MEMORY_CLEANUP, RES_LOWER_DISPLAY_UPDATE_INTERVAL,
        RES_LOWER_MAX_CONTOURS_TO_PROCESS, RES_LOWER_FULL_FRAME_YOLO_INTERVAL,
        RES_LOWER_FULL_FRAME_YOLO_CONF, RES_LOWER_MOG2_VAR_THRESHOLD,
        RES_LOWER_BASE_VAR_THRESHOLD, RES_LOWER_GRID_ROWS, RES_LOWER_GRID_COLS,
        RES_LOWER_BASE_MIN_AREA_REF, RES_LOWER_MAX_MIN_AREA_REF,
        RES_LOWER_HYBRID_ROI_PADDING, RES_LOWER_HYBRID_DIST_THRESHOLD
    )

    # Get active camera config
    cam_config = get_camera_config()
    CAMERA_NAME = cam_config["name"]
    CAMERA_WIDTH = cam_config["width"]
    CAMERA_HEIGHT = cam_config["height"]
    VIDEO_FILENAME = cam_config["video_filename"]
    USE_VIDEO_FILE = cam_config["use_video_file"]
    RTSP_URL = cam_config["rtsp_url"]
    WINDOW_NAME = cam_config["window_name"]
    DISPLAY_MAX_WIDTH = cam_config["display_max_width"]
    DISPLAY_MAX_HEIGHT = cam_config["display_max_height"]
    HORIZON_FILE = cam_config["horizon_file"]
    FOV_HORIZONTAL = cam_config["fov_horizontal"]
    FOV_VERTICAL = cam_config["fov_vertical"]

    print(f"✅ Config loaded for camera: {CAMERA_NAME} (FOV: {FOV_HORIZONTAL}°H × {FOV_VERTICAL}°V)")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Config file not found or incomplete: {e}")
    print("❌ Cannot continue without config.py. Please create config.py file in the same directory.")
    raise ImportError("config.py is required. Please create the config file.") from e

# ---- OpenCV threading/optimizations ----
cv2.setUseOptimized(True)
cv2.setNumThreads(OPENCV_NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)

# --- LIBRARIES CHECK ---
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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

# Screen size detection
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

def get_screen_size():
    """Get screen resolution using tkinter (cross-platform)"""
    if TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except Exception as e:
            print(f"⚠️ Failed to get screen size via tkinter: {e}")

    # Fallback: return common screen size or use config max
    if DISPLAY_MAX_WIDTH and DISPLAY_MAX_HEIGHT:
        return DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT
    # Default fallback
    return 1920, 1080

def get_resolution_dependent_thresholds(frame_w, frame_h):
    """
    คำนวณ thresholds ทั้งหมดตาม resolution ของกล้อง
    ใช้ ratios จาก config เพื่อให้ thresholds ปรับตาม resolution อัตโนมัติ

    Args:
        frame_w: ความกว้างของเฟรม (pixels)
        frame_h: ความสูงของเฟรม (pixels)

    Returns:
        dict: Dictionary ของ thresholds ทั้งหมดที่คำนวณแล้ว
    """
    # คำนวณ frame diagonal และ area
    frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)
    frame_area = frame_w * frame_h

    thresholds = {}

    # Re-identification & Tracking (ratios of frame diagonal)
    thresholds['MAX_REID_DISTANCE'] = REID_DISTANCE_RATIO * frame_diagonal
    thresholds['MERGE_DISTANCE'] = MERGE_DISTANCE_RATIO * frame_diagonal
    thresholds['HYBRID_DIST_THRESHOLD'] = HYBRID_DIST_THRESHOLD_RATIO * frame_diagonal
    thresholds['HYBRID_ROI_PADDING'] = HYBRID_ROI_PADDING_RATIO * frame_diagonal
    thresholds['SMALL_OBJECT_REID_DISTANCE'] = SMALL_OBJECT_REID_DISTANCE_RATIO * frame_diagonal

    # Stationary Detection & Blacklist (ratios of frame diagonal)
    thresholds['MAX_STATIONARY_DISTANCE'] = MAX_STATIONARY_DISTANCE_RATIO * frame_diagonal
    thresholds['STATIONARY_CENTER_THRESHOLD'] = STATIONARY_CENTER_THRESHOLD_RATIO * frame_diagonal
    thresholds['BLACKLIST_BBOX_PADDING'] = BLACKLIST_BBOX_PADDING_RATIO * frame_diagonal
    thresholds['BLACKLIST_MIN_VELOCITY'] = BLACKLIST_MIN_VELOCITY_RATIO * frame_diagonal

    # Motion Detection (ratios of frame dimensions)
    thresholds['EDGE_EXCLUSION_PIXELS'] = EDGE_EXCLUSION_PIXELS_RATIO * frame_w
    thresholds['HORIZON_EXCLUSION_PIXELS'] = HORIZON_EXCLUSION_PIXELS_RATIO * frame_w
    thresholds['MOTION_AREA_MARGIN'] = MOTION_AREA_MARGIN_RATIO * frame_h
    thresholds['MIN_AREA_BASE'] = MIN_AREA_BASE_RATIO * frame_area

    # Drone Detection (ratios)
    thresholds['DRONE_MIN_VELOCITY_FOR_ORANGE'] = DRONE_MIN_VELOCITY_FOR_ORANGE_RATIO * frame_diagonal
    thresholds['DRONE_MIN_VELOCITY_FOR_HOVER'] = DRONE_MIN_VELOCITY_FOR_HOVER_RATIO * frame_diagonal
    thresholds['DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE'] = DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE_RATIO * frame_diagonal
    thresholds['DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER'] = DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER_RATIO * frame_diagonal
    thresholds['DRONE_MIN_AREA'] = DRONE_MIN_AREA_RATIO * frame_area
    thresholds['DRONE_MAX_AREA'] = DRONE_MAX_AREA_RATIO * frame_area
    thresholds['DRONE_SMALL_OBJECT_AREA_THRESHOLD'] = DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO * frame_area
    thresholds['DRONE_MIN_PATH_TOTAL_DISTANCE'] = DRONE_MIN_PATH_TOTAL_DISTANCE_RATIO * frame_diagonal

    # Display & UI (ratios)
    thresholds['BBOX_PADDING'] = BBOX_PADDING_RATIO * frame_diagonal
    thresholds['HUD_MARGIN'] = HUD_MARGIN_RATIO * frame_w
    thresholds['HUD_MIN_WIDTH'] = HUD_MIN_WIDTH_RATIO * frame_w
    thresholds['HUD_MAX_WIDTH'] = HUD_MAX_WIDTH_RATIO * frame_w

    # Camera Movement (ratios)
    thresholds['CAM_MOVE_THRESHOLD'] = CAM_MOVE_THRESHOLD_RATIO * frame_diagonal

    # New thresholds for ROI motion detection and smart path tracking (ratios)
    from config import (
        ROI_PADDING_FAST_VELOCITY_THRESHOLD_RATIO,
        RED_STATIONARY_VELOCITY_THRESHOLD_RATIO, RED_STATIONARY_DISTANCE_THRESHOLD_RATIO,
        EARLY_BLACKLIST_STATIONARY_THRESHOLD_RATIO
    )

    # Dynamic ROI padding velocity threshold
    thresholds['ROI_PADDING_FAST_VELOCITY_THRESHOLD'] = ROI_PADDING_FAST_VELOCITY_THRESHOLD_RATIO * frame_diagonal

    # RED stationary check thresholds
    thresholds['RED_STATIONARY_VELOCITY_THRESHOLD'] = RED_STATIONARY_VELOCITY_THRESHOLD_RATIO * frame_diagonal
    thresholds['RED_STATIONARY_DISTANCE_THRESHOLD'] = RED_STATIONARY_DISTANCE_THRESHOLD_RATIO * frame_diagonal

    # Early blacklist detection threshold
    thresholds['EARLY_BLACKLIST_STATIONARY_THRESHOLD'] = EARLY_BLACKLIST_STATIONARY_THRESHOLD_RATIO * frame_diagonal

    # Convert to integers where appropriate
    int_keys = ['EDGE_EXCLUSION_PIXELS', 'HORIZON_EXCLUSION_PIXELS', 'MOTION_AREA_MARGIN',
                'MIN_AREA_BASE', 'DRONE_MIN_AREA', 'DRONE_MAX_AREA', 'DRONE_SMALL_OBJECT_AREA_THRESHOLD',
                'BLACKLIST_BBOX_PADDING', 'HYBRID_ROI_PADDING', 'BBOX_PADDING',
                'HUD_MARGIN', 'HUD_MIN_WIDTH', 'HUD_MAX_WIDTH']
    for key in int_keys:
        if key in thresholds:
            thresholds[key] = int(round(thresholds[key]))

    return thresholds

# All configuration is now loaded from config.py

# --- CLASS: SYSTEM MONITOR ---
class SystemMonitor:
    def __init__(self):
        self.stats = {"GPU": 0, "CPU": 0, "RAM": 0, "TEMP": 0}
        self.jetson = None
        self.frame_count = 0

        if JTOP_AVAILABLE:
            try:
                self.jetson = jtop()
                self.jetson.start()
                print("✅ JTOP Service Connected")
            except Exception as e:
                print(f"⚠️ JTOP Connection Failed: {e}")
                self.jetson = None

    def _read_temp_fallback(self):
        max_temp = 0
        try:
            for i in range(5):
                path = f"/sys/devices/virtual/thermal/thermal_zone{i}/temp"
                if os.path.exists(path):
                    with open(path, "r") as f:
                        t = int(f.read().strip()) // 1000
                        if t > max_temp: max_temp = t
        except:
            pass
        return max_temp

    def update(self):
        self.frame_count += 1
        if self.frame_count % 30 != 0: return

        if PSUTIL_AVAILABLE:
            self.stats["CPU"] = int(psutil.cpu_percent())
            self.stats["RAM"] = int(psutil.virtual_memory().percent)

        if self.jetson and self.jetson.ok():
            try:
                s = self.jetson.stats
                self.stats["GPU"] = int(s.get('GPU', 0))

                temps = s.get('Temp', {})
                valid_temps = [v for k, v in temps.items() if isinstance(v, (int, float))]
                if valid_temps:
                    self.stats["TEMP"] = int(max(valid_temps))

                if not PSUTIL_AVAILABLE:
                    cpus = s.get('CPU', [])
                    if cpus: self.stats["CPU"] = int(sum(cpus) / len(cpus))
                    self.stats["RAM"] = int(s.get('RAM', 0) / s.get('RAM_MAX', 1) * 100)
            except:
                pass

        if self.stats["TEMP"] == 0:
            self.stats["TEMP"] = self._read_temp_fallback()

    def close(self):
        if self.jetson: self.jetson.close()

# --- CLASS: SOUND ALERT (Parallel Thread) ---
class SoundAlert:
    """ระบบเสียงเตือนแบบ parallel thread - ไม่กระทบความเร็วโปรแกรม"""
    def __init__(self, sound_file=SOUND_FILE):
        self.sound_file = sound_file
        self.is_playing = False
        self.should_play = False
        self.last_orange_detected_time = 0.0
        self.thread = None
        self.lock = threading.Lock()
        self.running = True

        # เก็บ counter สำหรับสะสม RED frames ของแต่ละ target
        # Format: {target_id: {'red_frames': int, 'last_frame': int}}
        self.target_red_frames = {}  # Thread-safe access with lock

        # Initialize sound library
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                if os.path.exists(sound_file):
                    self.sound = pygame.mixer.Sound(sound_file)
                    self.sound_method = 'pygame'
                    print(f"✅ Sound Alert: Using pygame (file: {sound_file})")
                else:
                    print(f"⚠️ Sound Alert: File not found: {sound_file}")
                    self.sound_method = None
            except Exception as e:
                print(f"⚠️ Sound Alert: pygame init failed: {e}")
                self.sound_method = None
        elif PLAYSOUND_AVAILABLE:
            if os.path.exists(sound_file):
                self.sound_method = 'playsound'
                print(f"✅ Sound Alert: Using playsound (file: {sound_file})")
            else:
                print(f"⚠️ Sound Alert: File not found: {sound_file}")
                self.sound_method = None
        else:
            self.sound_method = None
            print("⚠️ Sound Alert: No sound library available (install pygame or playsound)")

    def update(self, has_orange_drone):
        """อัปเดตสถานะ - มีโดรน RED หรือไม่"""
        with self.lock:
            if has_orange_drone:
                self.should_play = True
                self.last_orange_detected_time = time.time()
            else:
                # ลด delay จาก 0.5 วินาที เป็น 0.1 วินาที เพื่อให้ sync กับ display เร็วขึ้น
                if time.time() - self.last_orange_detected_time > 0.1:  # เปลี่ยนจาก SOUND_CHECK_INTERVAL
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

                time.sleep(SOUND_CHECK_INTERVAL)  # ตรวจสอบทุก 0.5 วินาที

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
                    time.sleep(1.0)  # เล่นทุก 1 วินาที (เพราะ beep_2x.wav อาจสั้น)
                else:
                    time.sleep(SOUND_CHECK_INTERVAL)

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

    def update_red_frame_count(self, target_id, is_red, current_frame):
        """
        อัปเดต counter สำหรับสะสม RED frames ของ target

        Args:
            target_id: ID ของ target
            is_red: True ถ้า target เป็น RED ในเฟรมนี้
            current_frame: Frame number ปัจจุบัน
        """
        with self.lock:
            if target_id not in self.target_red_frames:
                self.target_red_frames[target_id] = {'red_frames': 0, 'last_frame': current_frame}

            target_info = self.target_red_frames[target_id]

            # ถ้าเป็นเฟรมใหม่ (ไม่ใช่เฟรมเดิม)
            if current_frame != target_info['last_frame']:
                if is_red:
                    # เพิ่ม counter ถ้าเป็น RED (แต่ไม่เกิน threshold เพื่อประหยัด memory)
                    if target_info['red_frames'] < SOUND_RED_FRAME_THRESHOLD:
                        target_info['red_frames'] += 1
                    # ถ้า counter >= threshold แล้ว → คงค่าไว้ (เพื่อให้ส่งเสียงต่อ)
                else:
                    # Reset counter ถ้าไม่ใช่ RED → หยุดส่งเสียง
                    target_info['red_frames'] = 0

                target_info['last_frame'] = current_frame

    def get_red_frame_count(self, target_id):
        """ดึงจำนวน RED frames ที่สะสมไว้ของ target"""
        with self.lock:
            if target_id in self.target_red_frames:
                return self.target_red_frames[target_id]['red_frames']
            return 0

    def clear_target(self, target_id):
        """ลบ target ออกจาก counter (เมื่อ target หายไป)"""
        with self.lock:
            if target_id in self.target_red_frames:
                del self.target_red_frames[target_id]

# --- CLASS: CAMERA STREAM (FIXED: SYNC FPS) ---
# จำกัดอัตราอ่านเฟรมจากไฟล์วิดีโอ (.mp4) สำหรับทดสอบ (ไม่เกินค่านี้)
VIDEO_FILE_MAX_INPUT_FPS = 25.0


class CameraStream:
    def __init__(self, source=None, width=None, height=None, use_video_file=None, camera_name=None,
                 udp_ip=None, udp_port=None, use_udp_direct=None, stream_format=None):
        self.cap = None
        self.frame = None
        self.frame_timestamp = None  # Timestamp ของ frame ล่าสุด
        self.running = False
        self.lock = threading.Lock()
        self.fps = DEFAULT_FPS  # Default fallback
        self.frame_delay = 1.0 / DEFAULT_FPS
        self.stream_health = "connecting"
        self.last_frame_size = None
        self._cap_src = None
        self._cap_backend = None
        self._live_reopen = False

        # Store parameters for multi-camera support
        self.source = source
        self.width = width
        self.height = height
        self.use_video_file = use_video_file if use_video_file is not None else USE_VIDEO_FILE
        self.camera_name = camera_name if camera_name is not None else CAMERA_NAME

        # UDP direct stream parameters
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.use_udp_direct = use_udp_direct
        self.stream_format = stream_format if stream_format else "h264"  # Default H.264

    def _apply_capture_buffer(self, cap, width, use_video, use_udp_direct):
        if cap is None:
            return
        try:
            if width >= 3840:
                if use_video:
                    buffer_size = 10
                elif use_udp_direct:
                    buffer_size = 1
                else:
                    buffer_size = 3
            else:
                buffer_size = 1
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except Exception:
            pass

    def start(self):
        # Use provided parameters or fallback to global config
        use_video = self.use_video_file if self.use_video_file is not None else USE_VIDEO_FILE
        source = self.source if self.source is not None else (VIDEO_FILENAME if use_video else RTSP_URL)
        width = self.width if self.width is not None else CAMERA_WIDTH
        height = self.height if self.height is not None else CAMERA_HEIGHT
        camera_name = self.camera_name if self.camera_name is not None else CAMERA_NAME

        # Get UDP parameters from instance or config
        use_udp_direct = self.use_udp_direct
        udp_ip = self.udp_ip
        udp_port = self.udp_port
        stream_format = self.stream_format

        # Fallback to config if not provided
        if use_udp_direct is None:
            try:
                from config import get_camera_config
                cam_config = get_camera_config(camera_name)
                use_udp_direct = cam_config.get("use_udp_direct", False)
                udp_ip = udp_ip or cam_config.get("udp_ip")
                udp_port = udp_port or cam_config.get("udp_port", 5004)
                stream_format = stream_format or cam_config.get("stream_format", "h264")
            except:
                use_udp_direct = False

        if use_video:
            if not os.path.isfile(source):
                print(f"❌ ERROR: File not found: {source}")
                return
            print(f"📂 [{camera_name}] Opening Video File: {source}")
            src = source
            backend = cv2.CAP_FFMPEG
        elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Webcam (source เป็น integer หรือ string ที่เป็นตัวเลข)
            src = int(source) if isinstance(source, str) else source
            print(f"📷 [{camera_name}] Opening Webcam: {src}")
            backend = cv2.CAP_ANY  # ใช้ default backend สำหรับ webcam
        elif use_udp_direct and udp_ip and udp_port:
            # ใช้ RTSP แต่ optimize สำหรับ FPS สูงสุด (กล้องส่ง UDP ผ่าน RTSP protocol)
            print(f"🚀 [{camera_name}] Using RTSP with UDP protocol (optimized for max FPS)... (IP: {udp_ip}, Port: {udp_port}, Format: {stream_format})")

            max_surfaces = 2

            if stream_format == "h265":
                depay_parse = "rtph265depay ! h265parse ! "
            else:
                depay_parse = "rtph264depay ! h264parse ! "

            src = (
                f"rtspsrc location={source} latency=0 protocols=udp "
                f"drop-on-latency=true max-lateness=-1 buffer-mode=none ! "
                f"{depay_parse}"
                f"nvv4l2decoder enable-max-performance=1 disable-dpb=true max-surface-count={max_surfaces} ! "
                f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
                "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
                "appsink max-buffers=1 drop=true sync=false"
            )
            backend = cv2.CAP_GSTREAMER
        else:
            # GStreamer pipeline for Jetson optimized RTSP stream
            if width >= 3840:
                latency_ms = 100
                max_surfaces = 2
            elif width >= 2560:
                latency_ms = 100
                max_surfaces = 2
            else:
                latency_ms = 0
                max_surfaces = 2

            if stream_format == "h265":
                depay_parse = "rtph265depay ! h265parse ! "
            else:
                depay_parse = "rtph264depay ! h264parse ! "

            src = (
                f"rtspsrc location={source} latency={latency_ms} protocols=udp "
                f"drop-on-latency=true max-lateness=-1 buffer-mode=none ! "
                f"{depay_parse}"
                f"nvv4l2decoder enable-max-performance=1 disable-dpb=true max-surface-count={max_surfaces} ! "
                f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
                "queue leaky=downstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
                "appsink max-buffers=3 drop=false sync=false"
            )
            print(f"📡 [{camera_name}] Connecting RTSP Stream... (Resolution: {width}x{height}, Latency: {latency_ms}ms, Format: {stream_format})")
            backend = cv2.CAP_GSTREAMER

        self._cap_src = src
        self._cap_backend = backend
        self._live_reopen = not use_video
        self.cap = cv2.VideoCapture(src, backend)

        # ตั้งค่า resolution สำหรับ webcam
        if isinstance(src, int):
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass

        self._apply_capture_buffer(self.cap, width, use_video, use_udp_direct)

        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > MAX_FPS:
                self.fps = DEFAULT_FPS
            if use_video:
                self.fps = min(float(self.fps), VIDEO_FILE_MAX_INPUT_FPS)
            self.frame_delay = 1.0 / max(self.fps, 1e-3)
            print(f"✅ Video Started. Target FPS: {self.fps:.2f} (Delay: {self.frame_delay*1000:.1f}ms)")

            self.running = True
            self.stream_health = "ok"
            threading.Thread(target=self.update, args=(), daemon=True).start()
        else:
            self.stream_health = "failed"
            print("❌ Camera/Video Failed to Open")

    def update(self):
        use_video = self.use_video_file if self.use_video_file is not None else USE_VIDEO_FILE
        reconnect_wait = 0.25

        while self.running:
            start_read = time.time()

            if self.cap is None:
                if use_video or not self._live_reopen:
                    self.stream_health = "failed"
                    break
                self.stream_health = "reconnecting"
                time.sleep(reconnect_wait)
                try:
                    self.cap = cv2.VideoCapture(self._cap_src, self._cap_backend)
                    self._apply_capture_buffer(
                        self.cap, self.width or CAMERA_WIDTH, use_video, self.use_udp_direct
                    )
                except Exception:
                    self.cap = None
                if self.cap is not None and self.cap.isOpened():
                    self.stream_health = "ok"
                    reconnect_wait = 0.25
                else:
                    reconnect_wait = min(8.0, reconnect_wait * 1.8)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if use_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.stream_health = "reconnecting"
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(reconnect_wait)
                    reconnect_wait = min(8.0, reconnect_wait * 1.8)
                    continue

            frame_timestamp = time.time()
            with self.lock:
                self.frame = frame
                self.frame_timestamp = frame_timestamp
                self.stream_health = "ok"
                self.last_frame_size = (frame.shape[1], frame.shape[0])
            reconnect_wait = 0.25

            if use_video:
                process_time = time.time() - start_read
                wait_time = self.frame_delay - process_time
                if wait_time > 0 and process_time < self.frame_delay * 0.8:
                    time.sleep(wait_time)

        print("📷 Camera Thread Stopped")

    def read(self):
        with self.lock:
            stream_health = getattr(self, "stream_health", "ok")
            if stream_health != "ok" or self.frame is None:
                return False, None, getattr(self, "frame_timestamp", None)
            return self.running, self.frame, getattr(self, "frame_timestamp", None)

    def get_stream_health(self):
        with self.lock:
            return getattr(self, "stream_health", "ok")

    def get_last_frame_size(self):
        with self.lock:
            return self.last_frame_size

    def release(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()

# --- GRID CLASS (CPU Optimized) ---
class GridMotionFilter:
    def __init__(self, width, height, rows, cols, min_area_base=None):
        self.rows, self.cols = rows, cols
        self.cell_w, self.cell_h = width // cols, height // rows
        self.noise_map = np.zeros((rows, cols), dtype=np.float32)
        # เก็บ min_area_base สำหรับ resolution-dependent threshold
        self.min_area_base = min_area_base if min_area_base is not None else MIN_AREA_BASE
        # Removed GPU memory allocation here as we will use CPU resize

        # Performance optimization: Cache
        self._noise_level_cache = {}  # Cache สำหรับ noise level
        self._confidence_cache = {}  # Cache สำหรับ confidence
        self._low_noise_regions_cache = None  # Cache สำหรับ low noise regions
        self._cache_frame = -1  # Frame ที่ cache ไว้

    def update_heatmap(self, mask_cpu):
        # Perform resize on CPU (Very fast for binary mask of this size)
        small_mask_cpu = cv2.resize(mask_cpu, (self.cols, self.rows), interpolation=cv2.INTER_AREA)

        # Exponential moving average for noise map
        self.noise_map = (small_mask_cpu.astype(np.float32)/255.0 * LEARNING_RATE) + (self.noise_map * (1.0 - LEARNING_RATE))

    def get_min_area_for_object(self, x, y, w, h, adaptive_min_area_ref=None):
        """
        คำนวณ min area สำหรับกรอง contour (pixels²)

        Args:
            x, y, w, h: Bounding box coordinates
            adaptive_min_area_ref: ค่า adaptive min area จาก adaptive filter (optional)

        Returns:
            (min_area, noise) - min area ที่ต้องมีและ noise level
        """
        cx, cy = x + w // 2, y + h // 2
        gx, gy = min(cx // self.cell_w, self.cols - 1), min(cy // self.cell_h, self.rows - 1)
        noise = self.noise_map[gy, gx]

        # ใช้ adaptive min area ถ้ามี (เมื่อเปิด adaptive mode)
        if adaptive_min_area_ref is not None:
            base_area = adaptive_min_area_ref
        else:
            # ใช้ resolution-dependent threshold
            base_area = self.min_area_base

        # Base min area + area multiplier based on local noise
        min_area = base_area + (noise * MIN_AREA_NOISE_MULTIPLIER)
        return min_area, noise

    def get_grid_noise_level(self, x, y, w, h, current_frame, cache_frames=3):
        """
        คำนวณ noise level เฉลี่ยในพื้นที่ bounding box พร้อม caching (เร็วขึ้น)

        Returns:
            (avg_noise, max_noise, min_noise): ค่า noise เฉลี่ย, สูงสุด, ต่ำสุด
        """
        # ตรวจสอบ cache
        cache_key = (x // 10, y // 10, w // 10, h // 10)  # Quantize เพื่อลด cache size
        if (cache_key in self._noise_level_cache and
            current_frame - self._cache_frame < cache_frames):
            return self._noise_level_cache[cache_key]

        # คำนวณ (ใช้ numpy vectorized operations - เร็วมาก)
        cx, cy = x + w // 2, y + h // 2
        gx, gy = min(cx // self.cell_w, self.cols - 1), min(cy // self.cell_h, self.rows - 1)

        # ใช้ numpy slicing แทน loop (เร็วมาก)
        gx1 = max(0, (x - w // 2) // self.cell_w)
        gy1 = max(0, (y - h // 2) // self.cell_h)
        gx2 = min(self.cols - 1, (x + w + w // 2) // self.cell_w)
        gy2 = min(self.rows - 1, (y + h + h // 2) // self.cell_h)

        # Vectorized operation (เร็วมาก)
        noise_region = self.noise_map[gy1:gy2+1, gx1:gx2+1]

        if noise_region.size == 0:
            result = (0.0, 0.0, 0.0)
        else:
            result = (float(np.mean(noise_region)),
                     float(np.max(noise_region)),
                     float(np.min(noise_region)))

        # Cache result
        self._noise_level_cache[cache_key] = result
        self._cache_frame = current_frame

        return result

    def get_grid_confidence(self, x, y, w, h, current_frame, cache_frames=3):
        """
        คำนวณ confidence จาก grid noise (ต่ำ = confident มาก) พร้อม caching (เร็วขึ้น)

        Returns:
            confidence: 0.0-1.0 (1.0 = confident มาก, 0.0 = ไม่ confident)
        """
        # ตรวจสอบ cache
        cache_key = (x // 10, y // 10, w // 10, h // 10)
        if (cache_key in self._confidence_cache and
            current_frame - self._cache_frame < cache_frames):
            return self._confidence_cache[cache_key]

        # ใช้ get_grid_noise_level ที่มี cache แล้ว
        avg_noise, _, _ = self.get_grid_noise_level(x, y, w, h, current_frame, cache_frames)
        confidence = 1.0 - min(avg_noise, 1.0)

        # Cache result
        self._confidence_cache[cache_key] = confidence

        return confidence

    def find_low_noise_regions(self, threshold=0.3, min_cells=4, current_frame=None, cache_frames=5):
        """
        หาพื้นที่ noise ต่ำ พร้อม caching (เร็วขึ้น)

        Returns:
            List of (gx, gy, noise_level) tuples
        """
        # ตรวจสอบ cache
        if (self._low_noise_regions_cache is not None and
            current_frame is not None and
            current_frame - self._cache_frame < cache_frames):
            return self._low_noise_regions_cache

        # ใช้ numpy vectorized operations (เร็วมาก)
        low_noise_mask = self.noise_map < threshold
        low_noise_indices = np.argwhere(low_noise_mask)

        if len(low_noise_indices) == 0:
            result = []
        else:
            # สร้าง list ของ (gx, gy, noise)
            result = [(int(idx[1]), int(idx[0]), float(self.noise_map[idx[0], idx[1]]))
                      for idx in low_noise_indices[:min_cells*2]]  # เก็บมากกว่า min_cells เพื่อ sort

            # เรียงตาม noise (ต่ำสุดก่อน)
            result.sort(key=lambda x: x[2])
            result = result[:min_cells]

        # Cache result
        self._low_noise_regions_cache = result
        if current_frame is not None:
            self._cache_frame = current_frame

        return result

    def clear_cache(self):
        """ล้าง cache (เรียกเมื่อ grid update)"""
        self._noise_level_cache.clear()
        self._confidence_cache.clear()
        self._low_noise_regions_cache = None

    def draw_grid(self, frame):
        for r in range(self.rows):
            for c in range(self.cols):
                noise = self.noise_map[r, c]
                if noise > 0.1:
                    x, y = c * self.cell_w, r * self.cell_h
                    color = (0, 255 - int(min(noise*3,1)*255), int(min(noise*3,1)*255), 255)
                    cv2.rectangle(frame, (x, y), (x + self.cell_w, y + self.cell_h), color, 1)

# --- CLASS: ADAPTIVE BACKGROUND FILTER ---
class AdaptiveBackgroundFilter:
    """ปรับค่าตามการขยับของพื้นหลังอัตโนมัติ - OPTIMIZED"""
    def __init__(self):
        self.current_var_threshold = BASE_VAR_THRESHOLD
        self.current_min_area_ref = BASE_MIN_AREA_REF
        self.current_morph_kernel_size = MORPH_KERNEL_SIZE_MIN
        self.contour_count_history = []  # เก็บประวัติจำนวน contour
        self.history_size = 10  # เก็บประวัติ 10 เฟรม
        self.last_update_frame = -1  # Cache: อัปเดตทุก N เฟรม

    def update(self, contour_count, current_frame):
        """ปรับค่า threshold ตามจำนวน contour - OPTIMIZED"""
        if not ADAPTIVE_ENABLED:
            return BASE_VAR_THRESHOLD, BASE_MIN_AREA_REF, MORPH_KERNEL_SIZE_MIN

        # Cache: อัปเดตทุก ADAPTIVE_UPDATE_INTERVAL เฟรม (ไม่ใช่ทุกเฟรม)
        if (self.last_update_frame >= 0 and
            current_frame - self.last_update_frame < ADAPTIVE_UPDATE_INTERVAL):
            return self.current_var_threshold, self.current_min_area_ref, self.current_morph_kernel_size

        self.last_update_frame = current_frame

        # เก็บประวัติ (จำกัดแค่ 10 เฟรม)
        self.contour_count_history.append(contour_count)
        if len(self.contour_count_history) > self.history_size:
            self.contour_count_history.pop(0)

        # คำนวณค่าเฉลี่ย (เร็วมาก - แค่ sum/len)
        if len(self.contour_count_history) == 0:
            avg_contour = 0
        else:
            avg_contour = sum(self.contour_count_history) / len(self.contour_count_history)

        # คำนวณ adaptive factor (0.0 = นิ่ง, 1.0 = ขยับมาก)
        if avg_contour < CONTOUR_COUNT_THRESHOLD_LOW:
            adaptive_factor = 0.0
        elif avg_contour > CONTOUR_COUNT_THRESHOLD_HIGH:
            adaptive_factor = 1.0
        else:
            # Linear interpolation
            adaptive_factor = (avg_contour - CONTOUR_COUNT_THRESHOLD_LOW) / \
                            (CONTOUR_COUNT_THRESHOLD_HIGH - CONTOUR_COUNT_THRESHOLD_LOW)
            adaptive_factor = max(0.0, min(1.0, adaptive_factor))

        # ปรับค่า varThreshold (exponential moving average - เร็วมาก)
        target_var = BASE_VAR_THRESHOLD + (MAX_VAR_THRESHOLD - BASE_VAR_THRESHOLD) * adaptive_factor
        self.current_var_threshold = (ADAPTIVE_LEARNING_RATE * target_var) + \
                                    ((1 - ADAPTIVE_LEARNING_RATE) * self.current_var_threshold)

        # ปรับค่า min_area (exponential moving average)
        target_min_area = BASE_MIN_AREA_REF + (MAX_MIN_AREA_REF - BASE_MIN_AREA_REF) * adaptive_factor
        self.current_min_area_ref = (ADAPTIVE_LEARNING_RATE * target_min_area) + \
                                   ((1 - ADAPTIVE_LEARNING_RATE) * self.current_min_area_ref)

        # ปรับค่า morphology kernel size (exponential moving average)
        target_kernel = MORPH_KERNEL_SIZE_MIN + (MORPH_KERNEL_SIZE_MAX - MORPH_KERNEL_SIZE_MIN) * adaptive_factor
        self.current_morph_kernel_size = (ADAPTIVE_LEARNING_RATE * target_kernel) + \
                                        ((1 - ADAPTIVE_LEARNING_RATE) * self.current_morph_kernel_size)

        return int(self.current_var_threshold), self.current_min_area_ref, int(self.current_morph_kernel_size)

# --- CLASS: ADAPTIVE MIN AREA MANAGER ---
class AdaptiveMinAreaManager:
    """
    จัดการ HYBRID_MIN_MOTION_AREA แบบ adaptive ตามจำนวน motion boxes และประสิทธิภาพ
    เพื่อป้องกันการกระตุกเมื่อมี motion เยอะ (Jetson Orin Nano Optimized)
    """
    def __init__(self, min_area_base=1, min_area_max=10,
                 motion_box_threshold_low=50, motion_box_threshold_high=400,
                 fps_threshold_low=10.0, fps_threshold_high=25.0,
                 algo_time_threshold_high=150.0,  # milliseconds
                 update_interval=10,  # อัปเดตทุก N เฟรม
                 adjustment_step=1):  # เพิ่ม/ลดทีละกี่ pixel
        self.min_area_base = min_area_base  # เริ่มต้นที่ 1 (ค่าปัจจุบันที่ได้ผลดี)
        self.min_area_max = min_area_max
        self.current_min_area = min_area_base  # เริ่มต้นด้วย 1

        # Thresholds สำหรับจำนวน motion boxes
        self.motion_box_threshold_low = motion_box_threshold_low
        self.motion_box_threshold_high = motion_box_threshold_high

        # Thresholds สำหรับ FPS
        self.fps_threshold_low = fps_threshold_low
        self.fps_threshold_high = fps_threshold_high

        # Threshold สำหรับเวลา processing
        self.algo_time_threshold_high = algo_time_threshold_high

        # History สำหรับ smoothing
        self.motion_box_history = []
        self.fps_history = []
        self.algo_time_history = []
        self.history_size = 10  # เก็บ 10 ค่าล่าสุด

        self.update_interval = update_interval
        self.adjustment_step = adjustment_step
        self.last_update_frame = 0

    def update(self, motion_box_count, fps, algo_time_ms, frame_counter):
        """
        อัปเดตและปรับ HYBRID_MIN_MOTION_AREA ตามประสิทธิภาพ

        Args:
            motion_box_count: จำนวน motion boxes ในเฟรมปัจจุบัน
            fps: FPS ปัจจุบัน
            algo_time_ms: เวลา processing (milliseconds)
            frame_counter: เฟรม counter

        Returns:
            min_area: ค่า HYBRID_MIN_MOTION_AREA ที่ปรับแล้ว
        """
        # เพิ่มเข้า history
        self.motion_box_history.append(motion_box_count)
        self.fps_history.append(fps)
        self.algo_time_history.append(algo_time_ms)

        # จำกัดขนาด history
        if len(self.motion_box_history) > self.history_size:
            self.motion_box_history.pop(0)
        if len(self.fps_history) > self.history_size:
            self.fps_history.pop(0)
        if len(self.algo_time_history) > self.history_size:
            self.algo_time_history.pop(0)

        # ตรวจสอบว่าถึงเวลาอัปเดตหรือไม่
        if frame_counter - self.last_update_frame < self.update_interval:
            return self.current_min_area

        self.last_update_frame = frame_counter

        # คำนวณค่าเฉลี่ย
        avg_motion_boxes = sum(self.motion_box_history) / len(self.motion_box_history) if self.motion_box_history else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_algo_time = sum(self.algo_time_history) / len(self.algo_time_history) if self.algo_time_history else 0

        # ตรวจสอบประสิทธิภาพและปรับ min_area
        # สถานการณ์ที่ต้องเพิ่ม min_area (ลดจำนวน motion boxes):
        # 1. จำนวน motion boxes เยอะมาก (> threshold_high)
        # 2. FPS ต่ำ (< threshold_low)
        # 3. เวลา processing สูง (> threshold_high)

        should_increase = False
        should_decrease = False

        # เงื่อนไขสำหรับเพิ่ม min_area (ลดความ sensitive)
        if (avg_motion_boxes > self.motion_box_threshold_high or
            avg_fps < self.fps_threshold_low or
            avg_algo_time > self.algo_time_threshold_high):
            should_increase = True

        # เงื่อนไขสำหรับลด min_area (เพิ่มความ sensitive) - เมื่อประสิทธิภาพดี
        # ใช้ OR แทน AND - ถ้าประสิทธิภาพดีในด้านใดด้านหนึ่งก็ลดได้ (ถ้าไม่ต้องเพิ่ม min_area)
        # เพิ่มความเร็วในการลด: ใช้ 80% ของ time threshold (120ms แทน 105ms)
        time_threshold_low = self.algo_time_threshold_high * 0.8  # 120ms
        fps_threshold_low_for_decrease = self.fps_threshold_high * 0.8  # 20 FPS (80% ของ 25)

        # ลดได้ถ้าไม่ต้องเพิ่ม min_area และประสิทธิภาพดีในด้านใดด้านหนึ่ง
        should_decrease = False
        should_decrease_fast = False

        if not should_increase:  # ถ้าไม่ต้องเพิ่ม min_area
            if (avg_motion_boxes < self.motion_box_threshold_low or
                avg_fps > fps_threshold_low_for_decrease or
                avg_algo_time < time_threshold_low):
                should_decrease = True

                # เพิ่มความเร็ว: ถ้าประสิทธิภาพดีมาก → ลดได้ 2 steps
                if (avg_fps > self.fps_threshold_high and avg_algo_time < time_threshold_low * 0.8):
                    # ประสิทธิภาพดีมาก → ลดได้เร็วขึ้น (2 steps)
                    should_decrease_fast = True

        # ปรับ min_area
        if should_increase and self.current_min_area < self.min_area_max:
            self.current_min_area = min(self.min_area_max,
                                       self.current_min_area + self.adjustment_step)
            if DEBUG_MODE:
                print(f"📈 Adaptive HYBRID_MIN_MOTION_AREA: เพิ่มเป็น {self.current_min_area} (motion={avg_motion_boxes:.0f}, fps={avg_fps:.1f}, time={avg_algo_time:.1f}ms)")

        elif should_decrease and self.current_min_area > self.min_area_base:
            # คำนวณจำนวน steps ที่จะลด (1 หรือ 2)
            decrease_steps = self.adjustment_step * 2 if should_decrease_fast else self.adjustment_step
            self.current_min_area = max(self.min_area_base,
                                       self.current_min_area - decrease_steps)
            if DEBUG_MODE:
                speed_text = " (fast)" if should_decrease_fast else ""
                print(f"📉 Adaptive HYBRID_MIN_MOTION_AREA: ลดเป็น {self.current_min_area}{speed_text} (motion={avg_motion_boxes:.0f}, fps={avg_fps:.1f}, time={avg_algo_time:.1f}ms)")

        return self.current_min_area

    def get_min_area_for_size(self, object_size_category):
        """
        คืนค่า min_area ตาม object size

        Args:
            object_size_category: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None

        Returns:
            min_area: ค่า min_area ที่เหมาะสมตาม object size
        """
        if object_size_category is None:
            # ถ้าไม่มี object_size_category ใช้ current_min_area ปกติ
            return self.current_min_area
        elif object_size_category == 'TINY':
            # สำหรับ tiny objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 2)
            if ADAPTIVE_TINY_OBJECT_MIN_AREA_ENABLED:
                # ถ้าเปิด adaptive ให้ปรับตาม current_min_area แต่ไม่เกิน max
                return min(TINY_OBJECT_MIN_MOTION_AREA_MAX,
                          max(TINY_OBJECT_MIN_MOTION_AREA_BASE, self.current_min_area))
            else:
                # ถ้าปิด adaptive ให้ใช้ base เสมอ
                return TINY_OBJECT_MIN_MOTION_AREA_BASE
        elif object_size_category == 'SMALL':
            # สำหรับ small objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 5)
            return min(SMALL_OBJECT_MIN_MOTION_AREA_MAX,
                      max(SMALL_OBJECT_MIN_MOTION_AREA_BASE, self.current_min_area))
        elif object_size_category == 'MEDIUM':
            # สำหรับ medium objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 10)
            return min(MEDIUM_OBJECT_MIN_MOTION_AREA_MAX,
                      max(MEDIUM_OBJECT_MIN_MOTION_AREA_BASE, self.current_min_area))
        else:  # 'LARGE' หรืออื่นๆ
            # สำหรับ large objects: ใช้ current_min_area ปกติ
            return self.current_min_area

    def reset(self):
        """Reset เป็นค่าเริ่มต้น"""
        self.current_min_area = self.min_area_base
        self.motion_box_history = []
        self.fps_history = []
        self.algo_time_history = []
        self.last_update_frame = 0

# --- CLASS: TRACKED OBJECT ---
class TrackedObject:
    def __init__(self, rect, track_id):
        self.id = track_id
        self.rects = [rect] # List of (x, y, w, h)
        centroid = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        self.centroids = [centroid]
        # Initialize center_of_mass_history as empty (will be populated by update())
        self.center_of_mass_history = []
        self.missed_frames = 0
        self.status = 'GREEN' # 'GREEN', 'YELLOW', 'ORANGE', or 'RED'
        self.path_frames = 1 # Number of consecutive frames detected
        self.yellow_duration = 0  # จำนวนเฟรมที่อยู่ในสถานะ YELLOW
        self.object_type = 'UNKNOWN'
        self.classification_confidence = 0.0

        # Caching system for classification
        self._path_quality_cache = None
        self._cache_frame = -1

        # YOLO frame counter for interval-based detection
        self.yolo_frame_counter = 0

        # ⚠️ PERFORMANCE OPTIMIZATION: Caching for expensive calculations
        self._can_hover_cache = None
        self._can_hover_cache_frame = -1
        self._smoothness_cache = None
        self._smoothness_cache_frame = -1
        self._consistency_cache = None
        self._consistency_cache_frame = -1
        self._flight_trail_total_distance = None  # Cache total distance for flight trail
        self._flight_trail_cache_frame = -1
        self._last_check_status_frame = -1  # Frame counter สำหรับ interval-based check_status()

    @property
    def current_rect(self):
        return self.rects[-1]

    @property
    def velocity_mag(self):
        # Calculate velocity magnitude (pixels/frame) over the last 5 frames
        if len(self.centroids) < 5: return 0

        c1 = self.centroids[-5]
        c2 = self.centroids[-1]

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

        # Velocity magnitude: distance over 4 frames
        return (dx**2 + dy**2)**0.5 / 4.0

    def get_velocity_vector(self, use_average=False, lookback=5):
        """
        คำนวณ velocity vector (dx, dy) จากประวัติ centroids

        Args:
            use_average: ถ้า True ใช้ average velocity จากหลายจุด (สำหรับ prediction)
            lookback: จำนวนจุดที่ใช้สำหรับ average (ถ้า use_average=True)

        Returns:
            (dx, dy): velocity vector
        """
        if len(self.centroids) < 2:
            return (0.0, 0.0)

        if use_average and len(self.centroids) >= lookback:
            # ใช้ average velocity จากหลายจุด (แม่นยำกว่า)
            velocities = []
            for i in range(len(self.centroids) - lookback + 1, len(self.centroids)):
                if i > 0:
                    prev_center = self.centroids[i-1]
                    curr_center = self.centroids[i]
                    dx = curr_center[0] - prev_center[0]
                    dy = curr_center[1] - prev_center[1]
                    velocities.append((dx, dy))

            if velocities:
                avg_dx = sum(v[0] for v in velocities) / len(velocities)
                avg_dy = sum(v[1] for v in velocities) / len(velocities)
                return (avg_dx, avg_dy)

        # ใช้ 2 จุดล่าสุดเพื่อคำนวณทิศทาง (default)
        prev_center = self.centroids[-2]
        curr_center = self.centroids[-1]
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        return (dx, dy)

    def get_predicted_roi(self, frame_w, frame_h, wait_frames=1):
        """
        คำนวณ ROI ที่คาดการณ์ไว้เมื่อวัตถุหายไปชั่วครู่
        ใช้ velocity เพื่อคาดการณ์ตำแหน่งที่วัตถุควรจะอยู่

        Args:
            frame_w: ความกว้างของเฟรม
            frame_h: ความสูงของเฟรม
            wait_frames: จำนวนเฟรมที่หายไป (ใช้สำหรับคาดการณ์ตำแหน่ง)

        Returns:
            (roi_x1, roi_y1, roi_x2, roi_y2) หรือ None ถ้าไม่สามารถคาดการณ์ได้
        """
        if not self.centroids:
            return None

        # ใช้ rect ล่าสุดเป็นฐาน
        tx, ty, tw, th = self.current_rect

        # คำนวณ velocity vector
        vx, vy = self.get_velocity_vector()

        # คาดการณ์ตำแหน่งใหม่ตาม velocity (ถ้ามีการเคลื่อนที่)
        if abs(vx) > 0.1 or abs(vy) > 0.1:
            # คาดการณ์ตำแหน่งตาม velocity * wait_frames
            predicted_x = tx + vx * wait_frames
            predicted_y = ty + vy * wait_frames

            # จำกัดให้อยู่ในขอบเขตเฟรม
            predicted_x = max(0, min(frame_w - tw, predicted_x))
            predicted_y = max(0, min(frame_h - th, predicted_y))
        else:
            # ถ้าไม่มีการเคลื่อนที่ ใช้ตำแหน่งเดิม
            predicted_x = tx
            predicted_y = ty

        # คำนวณ ROI พร้อม padding
        roi_x1 = max(0, int(predicted_x - HYBRID_ROI_PADDING))
        roi_y1 = max(0, int(predicted_y - HYBRID_ROI_PADDING))
        roi_x2 = min(frame_w, int(predicted_x + tw + HYBRID_ROI_PADDING))
        roi_y2 = min(frame_h, int(predicted_y + th + HYBRID_ROI_PADDING))

        return (roi_x1, roi_y1, roi_x2, roi_y2)

    def get_predicted_centroid(self, missed_frames=0, frame_w=None, frame_h=None):
        """
        คำนวณ predicted centroid เมื่อวัตถุหายไปชั่วครู่
        ใช้ path history หลายจุดเพื่อทำนายตำแหน่งที่แม่นยำ

        Args:
            missed_frames: จำนวนเฟรมที่หายไป
            frame_w: ความกว้างของเฟรม (สำหรับจำกัดขอบเขต)
            frame_h: ความสูงของเฟรม (สำหรับจำกัดขอบเขต)

        Returns:
            (x, y) predicted centroid หรือ None ถ้าไม่สามารถคาดการณ์ได้
        """
        if not self.centroids:
            return None

        if len(self.centroids) < 2:
            return self.centroids[-1]

        # ใช้ average velocity จากหลายจุด (แม่นยำกว่า)
        vx, vy = self.get_velocity_vector(use_average=True, lookback=PREDICTED_POSITION_LOOKBACK)

        # ใช้ centroid ล่าสุดเป็นฐาน
        last_centroid = self.centroids[-1]

        # คำนวณ predicted position
        predicted_x = last_centroid[0] + vx * missed_frames
        predicted_y = last_centroid[1] + vy * missed_frames

        # จำกัดให้อยู่ในขอบเขตเฟรม (ถ้ามี)
        if frame_w is not None and frame_h is not None:
            predicted_x = max(0, min(frame_w - 1, predicted_x))
            predicted_y = max(0, min(frame_h - 1, predicted_y))

        return (int(predicted_x), int(predicted_y))

    def has_clear_path_for_small_object(self, min_path_frames=5):
        """
        ตรวจสอบว่า small object มี path ที่ชัดเจนพอสำหรับการติดตาม
        ใช้สำหรับ small object ที่อยู่ไกลมาก (เล็กมาก) แต่มี path ชัดเจน

        Args:
            min_path_frames: จำนวนเฟรมต่ำสุดที่ต้องมี path

        Returns:
            bool: True ถ้า path ชัดเจนพอ
        """
        if len(self.centroids) < min_path_frames:
            return False

        # ตรวจสอบว่า path มีความต่อเนื่อง (ไม่กระโดด)
        max_gap = 0
        for i in range(1, len(self.centroids)):
            if i < len(self.centroids):
                dx = self.centroids[i][0] - self.centroids[i-1][0]
                dy = self.centroids[i][1] - self.centroids[i-1][1]
                gap = math.sqrt(dx**2 + dy**2)
                max_gap = max(max_gap, gap)

        # ถ้า gap ใหญ่เกินไป → path ไม่ต่อเนื่อง
        if len(self.centroids) >= 2:
            avg_gap = max_gap / max(1, len(self.centroids) - 1)
            # ใช้ threshold ที่สัมพันธ์กับขนาดวัตถุ
            x, y, w, h = self.current_rect
            object_size = math.sqrt(w**2 + h**2)
            if avg_gap > object_size * 2.0:  # gap ใหญ่เกิน 2 เท่าของขนาดวัตถุ
                return False

        # ตรวจสอบว่า path มีทิศทางชัดเจน (direction consistency)
        if len(self.centroids) >= 3:
            directions = []
            for i in range(1, len(self.centroids)):
                if i < len(self.centroids):
                    dx = self.centroids[i][0] - self.centroids[i-1][0]
                    dy = self.centroids[i][1] - self.centroids[i-1][1]
                    if dx != 0 or dy != 0:
                        # Normalize direction vector
                        length = math.sqrt(dx**2 + dy**2)
                        directions.append((dx / length, dy / length))

            if len(directions) >= 2:
                # คำนวณ dot product ของทิศทางที่ติดกัน
                dot_products = []
                for i in range(1, len(directions)):
                    dot = directions[i][0] * directions[i-1][0] + directions[i][1] * directions[i-1][1]
                    dot_products.append(dot)

                if dot_products:
                    avg_dot = sum(dot_products) / len(dot_products)
                    # ถ้า avg_dot > 0.5 → ทิศทางสม่ำเสมอ
                    if avg_dot < 0.3:  # threshold ต่ำสำหรับ small object
                        return False

        return True

    def get_object_size_category(self, thresholds, frame_w=None, frame_h=None):
        """
        จำแนกขนาดวัตถุเป็น TINY, SMALL, MEDIUM, หรือ LARGE
        ใช้ทั้ง area และ diagonal thresholds เพื่อความแม่นยำ

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds
            frame_w: ความกว้างของเฟรม (ถ้า None ใช้จาก thresholds)
            frame_h: ความสูงของเฟรม (ถ้า None ใช้จาก thresholds)

        Returns:
            str: 'TINY', 'SMALL', 'MEDIUM', หรือ 'LARGE'
        """
        if thresholds is None:
            return 'MEDIUM'  # Default

        # ใช้ frame dimensions จาก thresholds ถ้าไม่ระบุ
        if frame_w is None or frame_h is None:
            ref_w, ref_h = REFERENCE_RESOLUTION
            if 'frame_w' in thresholds:
                frame_w = thresholds['frame_w']
            else:
                frame_w = ref_w
            if 'frame_h' in thresholds:
                frame_h = thresholds['frame_h']
            else:
                frame_h = ref_h

        # คำนวณ area และ diagonal ของวัตถุ
        x, y, w, h = self.current_rect
        area = w * h
        diagonal = math.sqrt(w**2 + h**2)

        # คำนวณ frame area และ diagonal
        frame_area = frame_w * frame_h
        frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)

        # คำนวณ thresholds
        tiny_area_threshold = TINY_OBJECT_AREA_THRESHOLD_RATIO * frame_area
        tiny_diagonal_threshold = TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        small_area_threshold = thresholds.get('SMALL_OBJECT_AREA_THRESHOLD',
                                               DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO * frame_area)
        small_diagonal_threshold = SMALL_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        # ตรวจสอบว่าเป็น tiny object หรือไม่ (ใช้ทั้ง area และ diagonal)
        if area < tiny_area_threshold and diagonal < tiny_diagonal_threshold:
            return 'TINY'

        # ตรวจสอบว่าเป็น small object หรือไม่ (ใช้ทั้ง area และ diagonal)
        if area < small_area_threshold and diagonal < small_diagonal_threshold:
            return 'SMALL'

        # ตรวจสอบว่าเป็น medium object หรือไม่ (ไม่เกิน DRONE_MAX_AREA_RATIO)
        max_area_threshold = thresholds.get('DRONE_MAX_AREA_THRESHOLD',
                                               DRONE_MAX_AREA_RATIO * frame_area)
        if area <= max_area_threshold:
            return 'MEDIUM'

        # ถ้าใหญ่กว่า max_area_threshold → LARGE
        return 'LARGE'

    def is_tiny_object(self, thresholds, frame_w=None, frame_h=None):
        """
        ตรวจสอบว่าเป็น tiny object หรือไม่ (ใช้ทั้ง area และ diagonal thresholds)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds
            frame_w: ความกว้างของเฟรม (ถ้า None ใช้จาก thresholds)
            frame_h: ความสูงของเฟรม (ถ้า None ใช้จาก thresholds)

        Returns:
            bool: True ถ้าเป็น tiny object
        """
        return self.get_object_size_category(thresholds, frame_w, frame_h) == 'TINY'

    def is_small_object(self, thresholds, frame_w=None, frame_h=None):
        """
        ตรวจสอบว่าเป็น small object หรือไม่ (ใช้ทั้ง area และ diagonal thresholds)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds
            frame_w: ความกว้างของเฟรม (ถ้า None ใช้จาก thresholds)
            frame_h: ความสูงของเฟรม (ถ้า None ใช้จาก thresholds)

        Returns:
            bool: True ถ้าเป็น small object
        """
        category = self.get_object_size_category(thresholds, frame_w, frame_h)
        return category == 'SMALL' or category == 'TINY'

    # --- MODIFIED METHOD: CHECK SUDDEN SIZE CHANGE based on 5-frame average ---
    def size_changed_too_much(self, new_rect, max_multiplier=3.0):

        # 1. Calculate Average Area of the last 5 frames
        # Use the minimum of PATH_HISTORY_LENGTH or 5 frames for averaging
        history_count = min(len(self.rects), 5)

        if history_count == 0:
            return False # ไม่มีประวัติให้เปรียบเทียบ

        # คำนวณพื้นที่ของ rects ในประวัติ
        historical_areas = [(r[2] * r[3]) for r in self.rects[-history_count:]]

        # ป้องกันค่าเฉลี่ยเป็นศูนย์
        avg_area = sum(historical_areas) / history_count
        if avg_area == 0:
            return False

        # 2. Calculate New Area
        _, _, w_new, h_new = new_rect
        area_new = w_new * h_new

        # 3. Check the condition: New Area > Max Multiplier * Avg Area
        # หรือ New Area < 1/Max Multiplier * Avg Area (ป้องกันขนาดลดฮวบ)

        # ตรวจสอบการเพิ่มขึ้น: ขนาดใหม่เกิน 2 เท่าของค่าเฉลี่ย
        if area_new > avg_area * max_multiplier:
            return True

        # ตรวจสอบการลดลง: ขนาดใหม่ต่ำกว่า 1/2 (50%) ของค่าเฉลี่ย
        if area_new < avg_area * (1.0 / max_multiplier):
            return True

        return False
    # --------------------------------------------------------------------------

    def update(self, new_rect, contour=None, current_fps=None, thresholds=None):
        """
        อัปเดต tracker ด้วย rect ใหม่

        Args:
            new_rect: Bounding box ใหม่ (x, y, w, h)
            contour: Contour ของวัตถุ (optional)
            current_fps: FPS ปัจจุบัน (optional, สำหรับ FPS-aware path counting)
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)
        """
        # ⚠️ CRITICAL: ตรวจสอบขนาด rect ก่อน update (ป้องกันกล่องขยายตัวเรื่อยๆ)
        if len(self.rects) > 0:
            old_area = self.current_rect[2] * self.current_rect[3]
            new_area = new_rect[2] * new_rect[3]

            # ถ้าขนาดเปลี่ยนมากเกินไป → ใช้ rect เดิมแทน (ป้องกันกล่องขยายตัว)
            if new_area > old_area * MAX_RECT_SIZE_CHANGE_RATIO or new_area < old_area / MAX_RECT_SIZE_CHANGE_RATIO:
                # ใช้ rect เดิมแทน (ไม่ update ขนาด)
                new_rect = self.current_rect
                if DEBUG_MODE and self.path_frames % 30 == 0:  # Log ทุก 30 เฟรม (ไม่ log ทุกเฟรม)
                    print(f"⚠️ Tracker {self.id}: Size change too large ({old_area:.0f} -> {new_area:.0f}), using old rect")

        # ⚠️ CRITICAL CHANGE: Check against 10-frame average size
        # ตรวจสอบขนาดเมื่อมีประวัติอย่างน้อย 1 เฟรม
        if self.path_frames > 0 and self.size_changed_too_much(new_rect, max_multiplier=SIZE_CHANGE_MAX_MULTIPLIER):
            # หากขนาดเปลี่ยนกระทันหันเกิน 2 เท่า หรือลดลงต่ำกว่า 1/2 เท่า
            # ให้ถือว่าการนับ path_frames ขาดความต่อเนื่อง และรีเซ็ต
            self.path_frames = 1
            self.status = 'GREEN' # กลับสู่สถานะเริ่มต้น
            self.yellow_duration = 0  # Reset yellow_duration ด้วย
        else:
            self.path_frames += 1

        self.rects.append(new_rect)
        centroid = (new_rect[0] + new_rect[2] // 2, new_rect[1] + new_rect[3] // 2)
        self.centroids.append(centroid)

        # ⚠️ PERFORMANCE OPTIMIZATION: ใช้ centroid สำหรับ GREEN/YELLOW (เร็วกว่า)
        # และ center_of_mass เฉพาะ ORANGE/RED (ต้องการความแม่นยำ)
        if self.status == 'ORANGE' or self.status == 'RED':
            # Calculate Center of Mass from contour (นิ่งกว่า centroid เมื่อ bounding box ไม่นิ่ง)
            # ใช้เฉพาะเมื่อ status เป็น ORANGE/RED เพื่อความแม่นยำ
            if contour is not None and len(contour) > 0:
                try:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        center_of_mass = (cx, cy)
                    else:
                        # Fallback to centroid if moments calculation fails
                        center_of_mass = centroid
                except:
                    # Fallback to centroid if any error occurs
                    center_of_mass = centroid
            else:
                # Fallback to centroid if no contour provided
                center_of_mass = centroid
        else:
            # สำหรับ GREEN/YELLOW ใช้ centroid (เร็วกว่า - ไม่ต้องคำนวณ moments)
            center_of_mass = centroid

        # ⚠️ OPTIMIZATION: เก็บ path history เฉพาะเมื่อ status เป็น YELLOW, ORANGE หรือ RED (ประหยัดทรัพยากร)
        if self.status == 'YELLOW' or self.status == 'ORANGE' or self.status == 'RED':
            self.center_of_mass_history.append(center_of_mass)
            # จำกัด history length สำหรับ YELLOW/ORANGE
            if len(self.center_of_mass_history) > DRONE_YELLOW_PATH_HISTORY_LIMIT:
                self.center_of_mass_history.pop(0)
        # ถ้าเป็น GREEN ไม่เก็บ path history (ประหยัดทรัพยากร)

        # Keep history limited
        if len(self.rects) > PATH_HISTORY_LENGTH:
            self.rects.pop(0)
            self.centroids.pop(0)

        self.missed_frames = 0

        # ⚠️ NEW: สำหรับ tiny/small objects ใช้ FPS-aware path counting
        if thresholds is not None:
            is_tiny = self.is_tiny_object(thresholds, frame_w=thresholds.get('frame_w'), frame_h=thresholds.get('frame_h'))
            is_small = self.is_small_object(thresholds, frame_w=thresholds.get('frame_w'), frame_h=thresholds.get('frame_h'))

            if (is_tiny or is_small) and current_fps is not None and current_fps > 0:
                # ใช้ FPS-aware path counting สำหรับ tiny/small objects
                required_path_frames = self.get_fps_aware_path_frames(current_fps)
                # ใช้ smoothed path สำหรับ tiny/small objects (จะใช้ใน check_status)
                # Note: path_frames ยังคงนับตามปกติ แต่จะใช้ required_path_frames ใน check_status
                pass

        # ⚠️ PERFORMANCE: เรียก check_status() เฉพาะเมื่อถึง interval หรือเมื่อ path_frames เปลี่ยน
        # หรือเมื่อ status เปลี่ยน (GREEN -> YELLOW)
        should_check_status = False
        if self._last_check_status_frame < 0:
            should_check_status = True  # ครั้งแรก
        elif self.path_frames % CHECK_STATUS_UPDATE_INTERVAL == 0:
            should_check_status = True  # ถึง interval
        elif self.path_frames == PATH_HISTORY_LENGTH:
            should_check_status = True  # เปลี่ยนเป็น YELLOW

        if should_check_status:
            self.check_status(thresholds=thresholds)
            self._last_check_status_frame = self.path_frames

    @property
    def current_center_of_mass(self):
        """Get current center of mass, fallback to centroid if not available"""
        if self.center_of_mass_history:
            return self.center_of_mass_history[-1]
        elif self.centroids:
            return self.centroids[-1]
        else:
            return None

    def _smooth_path_points_for_calculation(self, path_points, window_size=None):
        """
        Smooth path points สำหรับการคำนวณ path quality metrics (ใช้ smoothed path)

        Args:
            path_points: List of (x, y) tuples
            window_size: ขนาดของ window สำหรับ moving average (ถ้า None ใช้จาก config)

        Returns:
            list: smoothed path points [(x, y), ...] หรือ path_points เดิมถ้าปิด smoothing
        """
        from config import PATH_SMOOTHING_ENABLED, PATH_SMOOTHING_WINDOW_SIZE

        # Early exit: ถ้าปิดการ smoothing หรือมีจุดน้อยเกินไป
        if not PATH_SMOOTHING_ENABLED or len(path_points) < 2:
            return path_points.copy()

        if window_size is None:
            window_size = PATH_SMOOTHING_WINDOW_SIZE

        # Early exit: ถ้ามีจุดน้อยกว่า window size → ไม่ต้อง smooth
        if len(path_points) < window_size:
            return path_points.copy()

        # Moving average smoothing (O(n) - เร็วมาก)
        smoothed = []
        half_window = window_size // 2

        for i in range(len(path_points)):
            # คำนวณ window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(path_points), i + half_window + 1)

            # คำนวณ average ของจุดใน window
            window_points = path_points[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            smoothed.append((avg_x, avg_y))  # เก็บเป็น float เพื่อความแม่นยำ

        return smoothed

    def calculate_path_straightness(self):
        """คำนวณความตรงของ path (0-1, สูง=ตรง) - ใช้ smoothed path"""
        # ใช้ center_of_mass_history แต่จำกัดแค่ PATH_QUALITY_HISTORY_LIMIT จุดล่าสุด
        points = self.center_of_mass_history[-PATH_QUALITY_HISTORY_LIMIT:] if len(self.center_of_mass_history) > PATH_QUALITY_HISTORY_LIMIT else self.center_of_mass_history

        if len(points) < 3:
            return 0.5  # Early exit: ไม่มีข้อมูลเพียงพอ

        # ใช้ smoothed path สำหรับการคำนวณ
        smoothed_points = self._smooth_path_points_for_calculation(points)

        # คำนวณ total distance จาก smoothed path
        total_distance = 0.0
        for i in range(1, len(smoothed_points)):
            dx = smoothed_points[i][0] - smoothed_points[i-1][0]
            dy = smoothed_points[i][1] - smoothed_points[i-1][1]
            total_distance += (dx**2 + dy**2)**0.5

        if total_distance == 0:
            return 1.0

        # ระยะทางตรงจากจุดแรกถึงจุดสุดท้าย (ใช้ smoothed path)
        start_point = smoothed_points[0]
        end_point = smoothed_points[-1]
        straight_distance = ((end_point[0] - start_point[0])**2 +
                           (end_point[1] - start_point[1])**2)**0.5

        # ความตรง = straight_distance / total_distance
        straightness = straight_distance / total_distance if total_distance > 0 else 0.0
        return min(1.0, max(0.0, straightness))

    def calculate_velocity_consistency(self, processing_fps=None):
        """คำนวณความสม่ำเสมอของความเร็ว (coefficient of variation) - ใช้ smoothed path และ processing FPS"""
        from config import DEFAULT_FPS

        if processing_fps is None or processing_fps <= 0:
            processing_fps = DEFAULT_FPS

        # ใช้ center_of_mass_history (ใช้ smoothed path) แต่จำกัดแค่ VELOCITY_HISTORY_LIMIT จุดล่าสุด
        points = self.center_of_mass_history[-VELOCITY_HISTORY_LIMIT:] if len(self.center_of_mass_history) > VELOCITY_HISTORY_LIMIT else self.center_of_mass_history

        if len(points) < 5:
            return 1.0  # Early exit: ไม่สม่ำเสมอ

        # ใช้ smoothed path สำหรับการคำนวณ
        smoothed_points = self._smooth_path_points_for_calculation(points)

        # คำนวณ velocities จาก smoothed path (pixels per second)
        velocities = []
        for i in range(1, len(smoothed_points)):
            dx = smoothed_points[i][0] - smoothed_points[i-1][0]
            dy = smoothed_points[i][1] - smoothed_points[i-1][1]
            # แปลงเป็น pixels per second (ใช้ processing FPS)
            vel = (dx**2 + dy**2)**0.5 * processing_fps
            velocities.append(vel)

        if len(velocities) < 2:
            return 1.0

        # ใช้ median absolute deviation (MAD) แทน std (เร็วกว่า)
        velocities_sorted = sorted(velocities)
        median_vel = velocities_sorted[len(velocities_sorted) // 2]

        if median_vel == 0:
            return 1.0

        mad = sorted([abs(v - median_vel) for v in velocities])[len(velocities) // 2]
        cv = mad / median_vel if median_vel > 0 else 1.0
        return cv

    def can_hover(self):
        """ตรวจสอบว่าสามารถ hover ได้หรือไม่ - OPTIMIZED with caching"""
        # ⚠️ PERFORMANCE: ใช้ cache ถ้า centroids ไม่เปลี่ยน
        if (self._can_hover_cache is not None and
            self._can_hover_cache_frame == len(self.centroids)):
            return self._can_hover_cache

        # ตรวจสอบเฉพาะ HOVER_HISTORY_LIMIT เฟรมล่าสุด
        centroids = self.centroids[-HOVER_HISTORY_LIMIT:] if len(self.centroids) > HOVER_HISTORY_LIMIT else self.centroids

        if len(centroids) < HOVER_MIN_FRAMES:
            result = (False, 0.0)
            self._can_hover_cache = result
            self._can_hover_cache_frame = len(self.centroids)
            return result

        # คำนวณ velocity สำหรับแต่ละเฟรม
        low_velocity_frames = 0
        consecutive_low_velocity = 0
        max_consecutive = 0

        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            vel = (dx**2 + dy**2)**0.5

            if vel < HOVER_VELOCITY_THRESHOLD:
                low_velocity_frames += 1
                consecutive_low_velocity += 1
                max_consecutive = max(max_consecutive, consecutive_low_velocity)
            else:
                consecutive_low_velocity = 0

        check_frames = len(centroids) - 1
        if check_frames == 0:
            result = (False, 0.0)
            self._can_hover_cache = result
            self._can_hover_cache_frame = len(self.centroids)
            return result

        # คำนวณ hover ratio
        hover_ratio = low_velocity_frames / check_frames if check_frames > 0 else 0.0

        # ตรวจสอบว่าสามารถ hover ได้หรือไม่
        can_hover = (max_consecutive >= HOVER_MIN_FRAMES) and (hover_ratio >= HOVER_RATIO_THRESHOLD)

        result = (can_hover, hover_ratio)
        self._can_hover_cache = result
        self._can_hover_cache_frame = len(self.centroids)
        return result

    def calculate_path_smoothness(self):
        """คำนวณความราบเรียบของ path (0-1, สูง=ราบเรียบ) - ใช้ smoothed path"""
        # ⚠️ PERFORMANCE: ใช้ cache ถ้า center_of_mass_history ไม่เปลี่ยน
        if (self._smoothness_cache is not None and
            self._smoothness_cache_frame == len(self.center_of_mass_history)):
            return self._smoothness_cache

        # ใช้ center_of_mass_history แต่จำกัดแค่ 10 จุดล่าสุด
        points = self.center_of_mass_history[-10:] if len(self.center_of_mass_history) > 10 else self.center_of_mass_history

        if len(points) < 3:
            result = 0.5
            self._smoothness_cache = result
            self._smoothness_cache_frame = len(self.center_of_mass_history)
            return result

        # ใช้ smoothed path สำหรับการคำนวณ
        smoothed_points = self._smooth_path_points_for_calculation(points)

        # คำนวณ angle changes จาก smoothed path (จำกัดแค่ 5-10 จุด)
        angle_changes = []
        limit = min(len(smoothed_points) - 1, 10)
        for i in range(1, limit):
            # Vector 1: จากจุด i-1 ถึง i
            dx1 = smoothed_points[i][0] - smoothed_points[i-1][0]
            dy1 = smoothed_points[i][1] - smoothed_points[i-1][1]

            # Vector 2: จากจุด i ถึง i+1
            if i + 1 < len(smoothed_points):
                dx2 = smoothed_points[i+1][0] - smoothed_points[i][0]
                dy2 = smoothed_points[i+1][1] - smoothed_points[i][1]

                # คำนวณมุมระหว่าง vectors
                dot = dx1 * dx2 + dy1 * dy2
                mag1 = (dx1**2 + dy1**2)**0.5
                mag2 = (dx2**2 + dy2**2)**0.5

                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp
                    angle = np.arccos(cos_angle) * 180 / np.pi  # degrees
                    angle_changes.append(angle)

        if not angle_changes:
            result = 1.0
            self._smoothness_cache = result
            self._smoothness_cache_frame = len(self.center_of_mass_history)
            return result

        # Smoothness = 1 / (1 + mean_angle_change)
        mean_angle_change = sum(angle_changes) / len(angle_changes)
        smoothness = 1.0 / (1.0 + mean_angle_change / 180.0)  # Normalize
        result = min(1.0, max(0.0, smoothness))
        self._smoothness_cache = result
        self._smoothness_cache_frame = len(self.center_of_mass_history)
        return result

    def calculate_direction_consistency(self):
        """คำนวณความสม่ำเสมอของทิศทาง (0-1, สูง=สม่ำเสมอ) - OPTIMIZED with caching"""
        # ⚠️ PERFORMANCE: ใช้ cache ถ้า center_of_mass_history ไม่เปลี่ยน
        if (self._consistency_cache is not None and
            self._consistency_cache_frame == len(self.center_of_mass_history)):
            return self._consistency_cache

        # ใช้ center_of_mass_history แต่จำกัดแค่ 15-20 จุดล่าสุด
        points = self.center_of_mass_history[-20:] if len(self.center_of_mass_history) > 20 else self.center_of_mass_history

        if len(points) < 3:
            result = 0.5
            self._consistency_cache = result
            self._consistency_cache_frame = len(self.center_of_mass_history)
            return result

        # คำนวณ angles ระหว่าง consecutive points
        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx) * 180 / np.pi  # degrees
                angles.append(angle)

        if len(angles) < 2:
            result = 0.5
            self._consistency_cache = result
            self._consistency_cache_frame = len(self.center_of_mass_history)
            return result

        # คำนวณ angle differences (การเปลี่ยนแปลงทิศทาง)
        angle_diffs = []
        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i-1])
            # Normalize to 0-180 degrees
            if diff > 180:
                diff = 360 - diff
            angle_diffs.append(diff)

        if not angle_diffs:
            result = 1.0
            self._consistency_cache = result
            self._consistency_cache_frame = len(self.center_of_mass_history)
            return result

        # ความสม่ำเสมอ = 1 / (1 + mean_angle_diff)
        # ถ้า angle_diff ต่ำ = ทิศทางสม่ำเสมอ = consistency สูง
        mean_angle_diff = sum(angle_diffs) / len(angle_diffs)
        consistency = 1.0 / (1.0 + mean_angle_diff / 90.0)  # Normalize
        result = min(1.0, max(0.0, consistency))
        self._consistency_cache = result
        self._consistency_cache_frame = len(self.center_of_mass_history)
        return result

    def get_fps_aware_path_frames(self, current_fps):
        """
        คำนวณจำนวนเฟรมที่ต้องการตาม FPS สำหรับ path analysis

        Args:
            current_fps: FPS ปัจจุบัน

        Returns:
            int: จำนวนเฟรมที่ normalized ตาม FPS
        """
        if current_fps <= 0:
            return SMALL_OBJECT_MIN_PATH_FRAMES  # Fallback

        # คำนวณจำนวนเฟรมที่ต้องการตาม FPS
        required_frames = int(SMALL_OBJECT_PATH_FRAMES_PER_SECOND * current_fps)
        return max(SMALL_OBJECT_MIN_PATH_FRAMES, required_frames)

    def smooth_path(self, window_size=None):
        """
        Smooth path points โดยใช้ moving average เพื่อลด noise สำหรับ tiny/small objects

        Args:
            window_size: ขนาดของ window สำหรับ moving average (ถ้า None ใช้จาก config)

        Returns:
            list: smoothed path points [(x, y), ...]
        """
        if window_size is None:
            window_size = SMALL_OBJECT_PATH_SMOOTHING_WINDOW

        if len(self.center_of_mass_history) < window_size:
            return self.center_of_mass_history.copy()

        smoothed = []
        half_window = window_size // 2

        for i in range(len(self.center_of_mass_history)):
            # คำนวณ window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.center_of_mass_history), i + half_window + 1)

            # คำนวณ average ของจุดใน window
            window_points = self.center_of_mass_history[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            smoothed.append((int(avg_x), int(avg_y)))

        return smoothed

    def get_smoothed_path(self):
        """
        คืนค่า smoothed version ของ center_of_mass_history

        Returns:
            list: smoothed path points [(x, y), ...]
        """
        return self.smooth_path()

    def calculate_path_curvature(self, window_size=None):
        """
        คำนวณ curvature ของ path โดยใช้ angle changes
        ใช้ sliding window เพื่อคำนวณ curvature ในแต่ละส่วน

        Args:
            window_size: ขนาดของ window สำหรับ curvature calculation (ถ้า None ใช้จาก config)

        Returns:
            tuple: (average_curvature, max_curvature) - curvature values (0-1, สูง=โค้งมาก)
        """
        if window_size is None:
            window_size = SMALL_OBJECT_CURVATURE_WINDOW

        if len(self.center_of_mass_history) < window_size + 1:
            return (0.0, 0.0)

        curvatures = []

        # ใช้ sliding window เพื่อคำนวณ curvature ในแต่ละส่วน
        for i in range(len(self.center_of_mass_history) - window_size):
            window_points = self.center_of_mass_history[i:i+window_size+1]

            # คำนวณ angle changes ใน window
            angle_changes = []
            for j in range(1, len(window_points)):
                dx1 = window_points[j][0] - window_points[j-1][0]
                dy1 = window_points[j][1] - window_points[j-1][1]

                if j + 1 < len(window_points):
                    dx2 = window_points[j+1][0] - window_points[j][0]
                    dy2 = window_points[j+1][1] - window_points[j][1]

                    # คำนวณมุมระหว่าง vectors
                    dot = dx1 * dx2 + dy1 * dy2
                    mag1 = math.sqrt(dx1**2 + dy1**2)
                    mag2 = math.sqrt(dx2**2 + dy2**2)

                    if mag1 > 0 and mag2 > 0:
                        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                        angle = math.acos(cos_angle) * 180 / math.pi  # degrees
                        angle_changes.append(angle)

            if angle_changes:
                # Curvature = mean angle change normalized (0-1)
                mean_angle_change = sum(angle_changes) / len(angle_changes)
                curvature = min(1.0, mean_angle_change / 180.0)  # Normalize to 0-1
                curvatures.append(curvature)

        if not curvatures:
            return (0.0, 0.0)

        avg_curvature = sum(curvatures) / len(curvatures)
        max_curvature = max(curvatures)

        return (avg_curvature, max_curvature)

    def detect_sharp_turns(self, threshold=30.0):
        """
        ตรวจสอบการเลี้ยวที่แหลม (sharp turns) - บ่งชี้ว่าเป็นโดรน

        Args:
            threshold: มุมขั้นต่ำที่ถือว่าเป็น sharp turn (องศา)

        Returns:
            bool: True ถ้ามี sharp turns
        """
        if len(self.center_of_mass_history) < 3:
            return False

        # ตรวจสอบ angle changes
        for i in range(1, len(self.center_of_mass_history) - 1):
            # Vector 1: จากจุด i-1 ถึง i
            dx1 = self.center_of_mass_history[i][0] - self.center_of_mass_history[i-1][0]
            dy1 = self.center_of_mass_history[i][1] - self.center_of_mass_history[i-1][1]

            # Vector 2: จากจุด i ถึง i+1
            dx2 = self.center_of_mass_history[i+1][0] - self.center_of_mass_history[i][0]
            dy2 = self.center_of_mass_history[i+1][1] - self.center_of_mass_history[i][1]

            # คำนวณมุมระหว่าง vectors
            dot = dx1 * dx2 + dy1 * dy2
            mag1 = math.sqrt(dx1**2 + dy1**2)
            mag2 = math.sqrt(dx2**2 + dy2**2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                angle = math.acos(cos_angle) * 180 / math.pi  # degrees

                # ถ้ามุมมากกว่า threshold → sharp turn
                if angle > threshold:
                    return True

        return False

    def is_likely_airplane(self, thresholds=None):
        """
        ตรวจสอบว่าวัตถุเป็นเครื่องบินหรือไม่ (เส้นตรง, ทิศทางสม่ำเสมอ, ความเร็วสม่ำเสมอ)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            tuple: (is_airplane: bool, confidence_score: float)
        """
        if len(self.center_of_mass_history) < 5:
            return (False, 0.0)

        # ใช้ smoothed path สำหรับ tiny/small objects
        smoothed_path = self.get_smoothed_path()
        if len(smoothed_path) < 5:
            smoothed_path = self.center_of_mass_history

        # ตรวจสอบ path straightness
        # คำนวณ straightness จาก smoothed path
        total_distance = 0.0
        for i in range(1, len(smoothed_path)):
            dx = smoothed_path[i][0] - smoothed_path[i-1][0]
            dy = smoothed_path[i][1] - smoothed_path[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)

        if total_distance == 0:
            return (False, 0.0)

        straight_distance = math.sqrt(
            (smoothed_path[-1][0] - smoothed_path[0][0])**2 +
            (smoothed_path[-1][1] - smoothed_path[0][1])**2
        )
        straightness = straight_distance / total_distance if total_distance > 0 else 0.0

        # ตรวจสอบ direction consistency
        direction_consistency = self.calculate_direction_consistency()

        # ตรวจสอบ velocity consistency (ใช้ processing_fps ถ้ามี)
        processing_fps = getattr(self, '_current_processing_fps', None)
        velocity_consistency = 1.0 / (1.0 + self.calculate_velocity_consistency(processing_fps=processing_fps))

        # ตรวจสอบ curvature (ต่ำ = ไม่โค้ง = เป็นเครื่องบิน)
        avg_curvature, max_curvature = self.calculate_path_curvature()
        low_curvature = 1.0 - avg_curvature  # Invert: ต่ำ = สูง

        # ตรวจสอบ sharp turns (ไม่มี = เป็นเครื่องบิน)
        no_sharp_turns = not self.detect_sharp_turns(threshold=30.0)

        # คำนวณ confidence score
        confidence = 0.0
        if straightness >= AIRPLANE_STRAIGHTNESS_THRESHOLD:
            confidence += 0.3
        if direction_consistency >= 0.7:
            confidence += 0.25
        if velocity_consistency >= 0.7:
            confidence += 0.25
        if low_curvature >= 0.7:
            confidence += 0.1
        if no_sharp_turns:
            confidence += 0.1

        is_airplane = confidence >= 0.6  # Threshold สำหรับยืนยันว่าเป็นเครื่องบิน

        return (is_airplane, min(1.0, confidence))

    def is_likely_drone_by_path(self, thresholds=None):
        """
        ตรวจสอบว่าวัตถุเป็นโดรนหรือไม่ (โค้ง/เลี้ยว, มี sharp turns)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            tuple: (is_drone: bool, confidence_score: float)
        """
        if len(self.center_of_mass_history) < 5:
            return (False, 0.0)

        # ใช้ smoothed path สำหรับ tiny/small objects
        smoothed_path = self.get_smoothed_path()
        if len(smoothed_path) < 5:
            smoothed_path = self.center_of_mass_history

        # ตรวจสอบ path straightness (ต่ำ = โค้ง = เป็นโดรน)
        total_distance = 0.0
        for i in range(1, len(smoothed_path)):
            dx = smoothed_path[i][0] - smoothed_path[i-1][0]
            dy = smoothed_path[i][1] - smoothed_path[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)

        if total_distance == 0:
            return (False, 0.0)

        straight_distance = math.sqrt(
            (smoothed_path[-1][0] - smoothed_path[0][0])**2 +
            (smoothed_path[-1][1] - smoothed_path[0][1])**2
        )
        straightness = straight_distance / total_distance if total_distance > 0 else 0.0
        low_straightness = 1.0 - straightness  # Invert: ต่ำ = สูง

        # ตรวจสอบ direction consistency (ต่ำ = มีการเลี้ยว = เป็นโดรน)
        direction_consistency = self.calculate_direction_consistency()
        low_consistency = 1.0 - direction_consistency  # Invert

        # ตรวจสอบ curvature (สูง = โค้งมาก = เป็นโดรน)
        avg_curvature, max_curvature = self.calculate_path_curvature()

        # ตรวจสอบ sharp turns (มี = เป็นโดรน)
        has_sharp_turns = self.detect_sharp_turns(threshold=30.0)

        # ตรวจสอบ velocity consistency (ปานกลาง = แปรผัน = เป็นโดรน) (ใช้ processing_fps ถ้ามี)
        processing_fps = getattr(self, '_current_processing_fps', None)
        velocity_cv = self.calculate_velocity_consistency(processing_fps=processing_fps)
        moderate_velocity_variation = 0.2 <= velocity_cv <= 0.5

        # คำนวณ confidence score
        confidence = 0.0
        if low_straightness >= DRONE_CURVATURE_THRESHOLD:
            confidence += 0.25
        if low_consistency >= 0.3:
            confidence += 0.2
        if avg_curvature >= SMALL_OBJECT_MIN_CURVATURE_FOR_DRONE:
            confidence += 0.25
        if has_sharp_turns:
            confidence += 0.2
        if moderate_velocity_variation:
            confidence += 0.1

        is_drone = confidence >= SMALL_OBJECT_CONFIDENCE_THRESHOLD

        return (is_drone, min(1.0, confidence))

    def calculate_drone_confidence_by_path(self, thresholds=None):
        """
        คำนวณ confidence score ว่าวัตถุเป็นโดรนหรือไม่
        ใช้หลาย factors: straightness, curvature, direction consistency, velocity profile

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            float: confidence score (0.0-1.0)
        """
        is_drone, confidence = self.is_likely_drone_by_path(thresholds)
        return confidence

    def calculate_airplane_confidence_by_path(self, thresholds=None):
        """
        คำนวณ confidence score ว่าวัตถุเป็นเครื่องบินหรือไม่
        ใช้หลาย factors: straightness, direction consistency, velocity consistency

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            float: confidence score (0.0-1.0)
        """
        is_airplane, confidence = self.is_likely_airplane(thresholds)
        return confidence

    def analyze_velocity_profile(self, thresholds=None):
        """
        วิเคราะห์ velocity profile (acceleration, deceleration, consistency)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            dict: velocity metrics (mean, std, consistency, acceleration, deceleration)
        """
        if len(self.centroids) < 5:
            return {
                'mean': 0.0,
                'std': 0.0,
                'consistency': 0.0,
                'acceleration': 0.0,
                'deceleration': 0.0
            }

        # คำนวณ velocities
        velocities = []
        for i in range(1, len(self.centroids)):
            dx = self.centroids[i][0] - self.centroids[i-1][0]
            dy = self.centroids[i][1] - self.centroids[i-1][1]
            vel = math.sqrt(dx**2 + dy**2)
            velocities.append(vel)

        if not velocities:
            return {
                'mean': 0.0,
                'std': 0.0,
                'consistency': 0.0,
                'acceleration': 0.0,
                'deceleration': 0.0
            }

        # คำนวณ mean และ std
        mean_vel = sum(velocities) / len(velocities)
        variance = sum((v - mean_vel)**2 for v in velocities) / len(velocities)
        std_vel = math.sqrt(variance)

        # คำนวณ consistency (CV = std/mean, consistency = 1/(1+CV))
        cv = std_vel / mean_vel if mean_vel > 0 else 1.0
        consistency = 1.0 / (1.0 + cv)

        # คำนวณ acceleration/deceleration (rate of change of velocity)
        if len(velocities) >= 2:
            velocity_changes = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
            acceleration = sum(v for v in velocity_changes if v > 0) / max(1, sum(1 for v in velocity_changes if v > 0)) if any(v > 0 for v in velocity_changes) else 0.0
            deceleration = abs(sum(v for v in velocity_changes if v < 0) / max(1, sum(1 for v in velocity_changes if v < 0))) if any(v < 0 for v in velocity_changes) else 0.0
        else:
            acceleration = 0.0
            deceleration = 0.0

        return {
            'mean': mean_vel,
            'std': std_vel,
            'consistency': consistency,
            'acceleration': acceleration,
            'deceleration': deceleration
        }

    def analyze_path_multi_scale(self, thresholds=None):
        """
        วิเคราะห์ path ในหลาย scale (short-term, medium-term, long-term)

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)

        Returns:
            dict: analysis results สำหรับแต่ละ scale
        """
        if len(self.center_of_mass_history) < 5:
            return {
                'short_term': {'curvature': 0.0, 'straightness': 0.0},
                'medium_term': {'curvature': 0.0, 'straightness': 0.0},
                'long_term': {'curvature': 0.0, 'straightness': 0.0}
            }

        # Short-term: 5-10 frames
        short_term_points = self.center_of_mass_history[-10:] if len(self.center_of_mass_history) >= 10 else self.center_of_mass_history
        short_term_curvature = self._calculate_curvature_for_points(short_term_points)
        short_term_straightness = self._calculate_straightness_for_points(short_term_points)

        # Medium-term: 15-20 frames
        medium_term_points = self.center_of_mass_history[-20:] if len(self.center_of_mass_history) >= 20 else self.center_of_mass_history
        medium_term_curvature = self._calculate_curvature_for_points(medium_term_points)
        medium_term_straightness = self._calculate_straightness_for_points(medium_term_points)

        # Long-term: 30+ frames
        long_term_points = self.center_of_mass_history[-30:] if len(self.center_of_mass_history) >= 30 else self.center_of_mass_history
        long_term_curvature = self._calculate_curvature_for_points(long_term_points)
        long_term_straightness = self._calculate_straightness_for_points(long_term_points)

        return {
            'short_term': {'curvature': short_term_curvature, 'straightness': short_term_straightness},
            'medium_term': {'curvature': medium_term_curvature, 'straightness': medium_term_straightness},
            'long_term': {'curvature': long_term_curvature, 'straightness': long_term_straightness}
        }

    def _calculate_curvature_for_points(self, points):
        """Helper method สำหรับคำนวณ curvature ของจุด"""
        if len(points) < 3:
            return 0.0

        angle_changes = []
        for i in range(1, len(points) - 1):
            dx1 = points[i][0] - points[i-1][0]
            dy1 = points[i][1] - points[i-1][1]
            dx2 = points[i+1][0] - points[i][0]
            dy2 = points[i+1][1] - points[i][1]

            dot = dx1 * dx2 + dy1 * dy2
            mag1 = math.sqrt(dx1**2 + dy1**2)
            mag2 = math.sqrt(dx2**2 + dy2**2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                angle = math.acos(cos_angle) * 180 / math.pi
                angle_changes.append(angle)

        if not angle_changes:
            return 0.0

        mean_angle = sum(angle_changes) / len(angle_changes)
        return min(1.0, mean_angle / 180.0)

    def _calculate_straightness_for_points(self, points):
        """Helper method สำหรับคำนวณ straightness ของจุด"""
        if len(points) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)

        if total_distance == 0:
            return 1.0

        straight_distance = math.sqrt(
            (points[-1][0] - points[0][0])**2 +
            (points[-1][1] - points[0][1])**2
        )
        return straight_distance / total_distance if total_distance > 0 else 0.0

    def is_likely_cloud(self):
        """ตรวจสอบว่าอาจเป็นก้อนเมฆ - OPTIMIZED (early exit, ไม่กระทบความเร็ว)"""
        if not CLOUD_DETECTION_ENABLED:
            return False

        # Early exit: ต้องมี path_frames เพียงพอ
        if self.path_frames < 15:
            return False

        # ตรวจสอบ velocity ก่อน (เร็วที่สุด - O(1))
        # NOTE: velocity_mag เป็น method; อย่าใส่วงเล็บซ้ำ (จะกลายเป็น float callable)
        velocity = self.velocity_mag
        if not (CLOUD_MIN_VELOCITY <= velocity <= CLOUD_MAX_VELOCITY):
            return False  # Early exit: ไม่ใช่ cloud

        # ตรวจสอบ area (O(1))
        area = self.current_rect[2] * self.current_rect[3]

        # ตรวจสอบก้อนเมฆใหญ่
        if area >= CLOUD_MIN_AREA:
            # คำนวณเฉพาะเมื่อจำเป็น (ถ้า area ผ่าน) (ใช้ processing_fps ถ้ามี)
            path_straightness = self.calculate_path_straightness()
            can_hover, _ = self.can_hover()
            processing_fps = getattr(self, '_current_processing_fps', None)
            velocity_cv = self.calculate_velocity_consistency(processing_fps=processing_fps)

            return (path_straightness <= CLOUD_MAX_STRAIGHTNESS and
                    not can_hover and
                    velocity_cv <= CLOUD_MAX_VELOCITY_CV)

        # ตรวจสอบจุดเล็กๆในก้อนเมฆ (velocity สูง + size เล็ก + ไม่ตรง)
        elif area < CLOUD_MIN_AREA:
            path_straightness = self.calculate_path_straightness()
            can_hover, _ = self.can_hover()
            smoothness = self.calculate_path_smoothness()

            return (path_straightness <= CLOUD_MAX_STRAIGHTNESS and
                    not can_hover and
                    smoothness <= CLOUD_MAX_SMOOTHNESS)

        return False

    def has_valid_flight_trail(self, thresholds=None):
        """
        ตรวจสอบว่ามีรอยเท้าการบินที่ยืนยันได้ว่าเป็นวัตถุจริง (ไม่ใช่ก้อนเมฆ/ไอพ่น) - OPTIMIZED with caching

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)
        """
        # Early exit: ต้องมี path history จาก YELLOW/ORANGE
        if len(self.center_of_mass_history) < DRONE_MIN_YELLOW_PATH_POINTS:
            return False, "Path too short"

        # ใช้ resolution-dependent threshold ถ้ามี
        min_path_total_distance = thresholds.get('DRONE_MIN_PATH_TOTAL_DISTANCE', DRONE_MIN_PATH_TOTAL_DISTANCE) if thresholds else DRONE_MIN_PATH_TOTAL_DISTANCE

        # ⚠️ PERFORMANCE: จำกัดจำนวนจุดที่ตรวจสอบ (ใช้แค่ 30-40 จุดล่าสุด)
        points_to_check = self.center_of_mass_history[-40:] if len(self.center_of_mass_history) > 40 else self.center_of_mass_history

        # ⚠️ PERFORMANCE: Cache total_distance (อัปเดตเมื่อเพิ่มจุดใหม่)
        current_history_len = len(self.center_of_mass_history)
        if (self._flight_trail_total_distance is None or
            self._flight_trail_cache_frame != current_history_len):
            # คำนวณ total_distance ใหม่
            total_distance = 0.0
            distances = []
            # ใช้แค่จุดที่จำกัด
            check_points = points_to_check if len(points_to_check) >= 2 else self.center_of_mass_history
            for i in range(1, len(check_points)):
                dx = check_points[i][0] - check_points[i-1][0]
                dy = check_points[i][1] - check_points[i-1][1]
                dist = (dx**2 + dy**2)**0.5
                total_distance += dist
                distances.append(dist)

            self._flight_trail_total_distance = total_distance
            self._flight_trail_distances = distances
            self._flight_trail_cache_frame = current_history_len
        else:
            # ใช้ cache
            total_distance = self._flight_trail_total_distance
            distances = self._flight_trail_distances

        # ตรวจสอบระยะทางรวม (ใช้ resolution-dependent threshold)
        if total_distance < min_path_total_distance:
            return False, "Path distance too short"

        # ตรวจสอบความต่อเนื่อง (gap ไม่ควรใหญ่เกินไป)
        if len(distances) > 0:
            avg_distance = total_distance / len(distances)
            max_gap = max(distances) if distances else 0.0
            if avg_distance > 0 and max_gap > avg_distance * DRONE_PATH_MAX_GAP_RATIO:
                return False, "Path not continuous"

        return True, "Valid flight trail"

    def is_likely_insect(self):
        """ตรวจสอบว่าอาจเป็นแมลง (บินเร็ว/เส้นตรง/ไม่ต่อเนื่อง) - OPTIMIZED"""
        # Early exit: ต้องมี path history
        if len(self.center_of_mass_history) < 10:
            return False

        # ตรวจสอบความเร็ว
        velocity = self.velocity_mag
        if velocity <= INSECT_MAX_VELOCITY:
            return False  # Early exit: ไม่เร็วพอ

        # ตรวจสอบความตรง
        path_straightness = self.calculate_path_straightness()
        if path_straightness < INSECT_MIN_STRAIGHTNESS:
            return False  # Early exit: ไม่ตรงพอ

        # ตรวจสอบความต่อเนื่อง (คำนวณจาก gap ratio)
        if len(self.center_of_mass_history) < 5:
            return False

        distances = []
        for i in range(1, len(self.center_of_mass_history)):
            dx = self.center_of_mass_history[i][0] - self.center_of_mass_history[i-1][0]
            dy = self.center_of_mass_history[i][1] - self.center_of_mass_history[i-1][1]
            distances.append((dx**2 + dy**2)**0.5)

        if len(distances) == 0:
            return False

        total_distance = sum(distances)
        avg_distance = total_distance / len(distances) if len(distances) > 0 else 0
        max_gap = max(distances) if distances else 0.0

        # ความต่อเนื่อง = 1 - (max_gap / avg_distance) ถ้า gap ใหญ่ = ไม่ต่อเนื่อง
        if avg_distance > 0:
            continuity = 1.0 - min(1.0, max_gap / (avg_distance * 2.0))
            if continuity > INSECT_MAX_CONTINUITY:
                return False  # ต่อเนื่องเกินไป = ไม่ใช่แมลง

        # ผ่านทุกเงื่อนไข = เป็นแมลง
        return True

    def check_status(self, thresholds=None):
        """
        ตรวจสอบและอัปเดต status ของ tracker
        ใช้ resolution-dependent thresholds ถ้ามี

        Args:
            thresholds: Dictionary ของ resolution-dependent thresholds (optional)
        """
        # Change status to ALERT (YELLOW) if detected for PATH_HISTORY_LENGTH continuous frames
        if self.path_frames >= PATH_HISTORY_LENGTH and self.status == 'GREEN':
            self.status = 'YELLOW'

        # Track yellow_duration
        if self.status == 'YELLOW':
            self.yellow_duration += 1
        else:
            self.yellow_duration = 0  # Reset เมื่อไม่ใช่ YELLOW

        # ⚠️ CRITICAL: ตรวจสอบว่าอาจเป็นก้อนเมฆ (กรองก่อนเปลี่ยนเป็น ORANGE - ไม่กระทบความเร็ว)
        if self.is_likely_cloud():
            # เป็นก้อนเมฆ → ไม่เปลี่ยนเป็น ORANGE/RED (หรือเปลี่ยนกลับเป็น YELLOW)
            if self.status == 'ORANGE' or self.status == 'RED':
                self.status = 'YELLOW'  # เปลี่ยนกลับเป็น YELLOW
            return  # Early exit: หยุดการตรวจสอบต่อ (ไม่คำนวณ path characteristics)

        # Change status to ORANGE if YELLOW duration >= threshold and path characteristics match
        if (self.status == 'YELLOW' and
            self.yellow_duration >= DRONE_YELLOW_DURATION_THRESHOLD and
            self.path_frames >= DRONE_MIN_PATH_FRAMES_FOR_ORANGE):

            # ⚠️ CRITICAL: กรองแมลงก่อน (บินเร็ว/เส้นตรง/ไม่ต่อเนื่อง)
            if self.is_likely_insect():
                return  # ไม่เปลี่ยนเป็น ORANGE ถ้าเป็นแมลง

            # ⚠️ CRITICAL: ตรวจสอบรอยเท้าการบินก่อน (ยืนยันว่าเป็นวัตถุจริง ไม่ใช่ก้อนเมฆ/ไอพ่น)
            has_trail, trail_reason = self.has_valid_flight_trail(thresholds=thresholds)
            if not has_trail:
                if DEBUG_MODE and self.yellow_duration % 30 == 0:  # Log ทุก 30 เฟรม
                    print(f"⚠️ Tracker {self.id}: No valid flight trail ({trail_reason}), "
                          f"path_points={len(self.center_of_mass_history)}")
                return  # ไม่เปลี่ยนเป็น ORANGE ถ้าไม่มีรอยเท้าการบิน

            # ใช้ resolution-dependent thresholds ถ้ามี
            if thresholds is None:
                drone_min_area = DRONE_MIN_AREA
                drone_max_area = DRONE_MAX_AREA
                min_velocity_orange = DRONE_MIN_VELOCITY_FOR_ORANGE
                min_velocity_hover = DRONE_MIN_VELOCITY_FOR_HOVER
                min_distance_orange = DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE
                min_distance_hover = DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER
                small_object_area_thresh = 200  # fallback
            else:
                drone_min_area = thresholds.get('DRONE_MIN_AREA', DRONE_MIN_AREA)
                drone_max_area = thresholds.get('DRONE_MAX_AREA', DRONE_MAX_AREA)
                min_velocity_orange = thresholds.get('DRONE_MIN_VELOCITY_FOR_ORANGE', DRONE_MIN_VELOCITY_FOR_ORANGE)
                min_velocity_hover = thresholds.get('DRONE_MIN_VELOCITY_FOR_HOVER', DRONE_MIN_VELOCITY_FOR_HOVER)
                min_distance_orange = thresholds.get('DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE', DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE)
                min_distance_hover = thresholds.get('DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER', DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER)
                small_object_area_thresh = thresholds.get('DRONE_SMALL_OBJECT_AREA_THRESHOLD', 200)

            # ตรวจสอบขนาดก่อน (ป้องกันกล่องใหญ่เกินไป)
            area = self.current_rect[2] * self.current_rect[3]
            if not (drone_min_area <= area <= drone_max_area):
                return  # ไม่เปลี่ยนเป็น ORANGE ถ้าขนาดไม่เหมาะสม

            # ⚠️ NEW: ตรวจสอบว่าเป็น tiny/small object หรือไม่ (ใช้ทั้ง area และ diagonal)
            is_tiny = self.is_tiny_object(thresholds, frame_w=thresholds.get('frame_w') if thresholds else None,
                                         frame_h=thresholds.get('frame_h') if thresholds else None)
            is_small = self.is_small_object(thresholds, frame_w=thresholds.get('frame_w') if thresholds else None,
                                           frame_h=thresholds.get('frame_h') if thresholds else None)
            is_small_object = area < small_object_area_thresh  # Legacy check for backward compatibility

            # ⚠️ NEW: สำหรับ tiny/small objects ใช้ path-based analysis
            if is_tiny or is_small:
                # ใช้ smoothed path สำหรับ tiny/small objects
                smoothed_path = self.get_smoothed_path()
                if len(smoothed_path) < 5:
                    return  # ต้องมี path เพียงพอ

                # คำนวณ frame dimensions สำหรับ velocity thresholds
                frame_w = thresholds.get('frame_w', REFERENCE_RESOLUTION[0]) if thresholds else REFERENCE_RESOLUTION[0]
                frame_h = thresholds.get('frame_h', REFERENCE_RESOLUTION[1]) if thresholds else REFERENCE_RESOLUTION[1]
                frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)

                # ตรวจสอบ velocity (ใช้ ratio-based thresholds)
                velocity = self.velocity_mag
                min_velocity_threshold = SMALL_OBJECT_MIN_VELOCITY_RATIO * frame_diagonal
                max_velocity_threshold = SMALL_OBJECT_MAX_VELOCITY_RATIO * frame_diagonal

                if velocity < min_velocity_threshold or velocity > max_velocity_threshold:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Tiny/Small object velocity out of range "
                              f"({velocity:.2f} not in [{min_velocity_threshold:.2f}, {max_velocity_threshold:.2f}])")
                    return  # Velocity ไม่เหมาะสม

                # ตรวจสอบ path quality (curvature, straightness)
                avg_curvature, max_curvature = self.calculate_path_curvature()

                # ตรวจสอบว่าเป็น airplane หรือ drone โดยใช้ confidence scoring
                is_airplane, airplane_confidence = self.is_likely_airplane(thresholds)
                is_drone, drone_confidence = self.is_likely_drone_by_path(thresholds)

                # ถ้าเป็น airplane → ไม่เปลี่ยนเป็น ORANGE (ไม่ส่งเสียง)
                if is_airplane and airplane_confidence >= 0.6:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Tiny/Small object is likely airplane (confidence={airplane_confidence:.2f})")
                    return  # ไม่เปลี่ยนเป็น ORANGE ถ้าเป็นเครื่องบิน

                # ถ้าเป็น drone และมี confidence เพียงพอ → เปลี่ยนเป็น ORANGE
                if is_drone and drone_confidence >= SMALL_OBJECT_CONFIDENCE_THRESHOLD:
                    self.status = 'ORANGE'
                    if DEBUG_MODE:
                        print(f"✅ Tracker {self.id}: Tiny/Small object changed to ORANGE (drone confidence={drone_confidence:.2f}, "
                              f"curvature={avg_curvature:.2f})")
                    return
                else:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Tiny/Small object not confirmed as drone "
                              f"(confidence={drone_confidence:.2f} < {SMALL_OBJECT_CONFIDENCE_THRESHOLD:.2f})")
                    return  # ไม่เปลี่ยนเป็น ORANGE ถ้า confidence ไม่เพียงพอ

            # ⚠️ NEW: Multi-layer Movement Validation (Early Exit - ประหยัด CPU)
            # 1. ตรวจสอบ hover mode ก่อน (O(1) - เร็ว)
            can_hover, hover_ratio = self.can_hover()
            is_hover_mode = can_hover and hover_ratio >= 0.5

            # 2. ใช้ threshold ตาม hover mode และ small object
            # Small object ที่มี clear path → ลด threshold
            if is_small_object and SMALL_OBJECT_PATH_QUALITY_BONUS and self.has_clear_path_for_small_object():
                # ลด threshold สำหรับ small object ที่มี path ชัดเจน
                min_velocity = min_velocity_hover if is_hover_mode else min_velocity_orange * 0.5
                min_distance = min_distance_hover if is_hover_mode else min_distance_orange * 0.5
            else:
                min_velocity = min_velocity_hover if is_hover_mode else min_velocity_orange
                min_distance = min_distance_hover if is_hover_mode else min_distance_orange

            # 3. ตรวจสอบ velocity (Early Exit - O(1))
            velocity = self.velocity_mag
            if velocity < min_velocity:
                # แต่ถ้าเป็น hover mode และมี total distance → อนุญาต
                if not is_hover_mode:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Velocity too low ({velocity:.2f} < {min_velocity:.2f})")
                    return  # ไม่เปลี่ยนเป็น ORANGE ถ้าไม่เคลื่อนที่

            # 4. ตรวจสอบ total distance (Early Exit - O(1))
            if len(self.centroids) >= 2:
                first_center = self.centroids[0]
                last_center = self.centroids[-1]
                total_distance = ((last_center[0] - first_center[0])**2 +
                                 (last_center[1] - first_center[1])**2)**0.5
                if total_distance < min_distance:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Total distance too low ({total_distance:.2f} < {min_distance:.2f})")
                    return  # ไม่เปลี่ยนเป็น ORANGE ถ้าเคลื่อนที่น้อยเกินไป
            else:
                return  # ไม่มี centroids เพียงพอ

            # 5. ตรวจสอบ movement frames (ยืนยันว่ามีการเคลื่อนที่จริง - O(n) โดย n จำกัด)
            movement_frames = 0
            if len(self.centroids) >= DRONE_MIN_MOVEMENT_FRAMES_FOR_ORANGE:
                check_frames = min(len(self.centroids), DRONE_MIN_MOVEMENT_FRAMES_FOR_ORANGE + 1)
                for i in range(1, check_frames):
                    prev = self.centroids[i-1]
                    curr = self.centroids[i]
                    frame_velocity = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
                    if frame_velocity >= min_velocity:
                        movement_frames += 1

                # ต้องมีการเคลื่อนที่อย่างน้อย DRONE_MIN_MOVEMENT_RATIO ของเฟรม
                min_movement_frames = int(DRONE_MIN_MOVEMENT_FRAMES_FOR_ORANGE * DRONE_MIN_MOVEMENT_RATIO)
                if movement_frames < min_movement_frames:
                    if DEBUG_MODE and self.yellow_duration % 30 == 0:
                        print(f"⚠️ Tracker {self.id}: Not enough movement frames ({movement_frames}/{min_movement_frames})")
                    return  # ไม่เปลี่ยนเป็น ORANGE ถ้าไม่มีการเคลื่อนที่จริง

            # 6. ตรวจสอบ path characteristics (คำนวณเฉพาะเมื่อผ่านเงื่อนไขเบาแล้ว)
            smoothness = self.calculate_path_smoothness()
            direction_consistency = self.calculate_direction_consistency()

            # ⚠️ OPTIMIZATION: ลด threshold สำหรับโดรนที่สามารถ hover (บินช้าๆ/หยุดบิน)
            if is_hover_mode:
                # ถ้าสามารถ hover ได้ → ลด threshold (รองรับโดรนที่บินช้าๆ)
                min_smoothness = DRONE_MIN_SMOOTHNESS_SIMPLE - DRONE_HOVER_SMOOTHNESS_BONUS
                min_consistency = DRONE_MIN_DIRECTION_CONSISTENCY_SIMPLE - DRONE_HOVER_CONSISTENCY_BONUS
            else:
                min_smoothness = DRONE_MIN_SMOOTHNESS_SIMPLE
                min_consistency = DRONE_MIN_DIRECTION_CONSISTENCY_SIMPLE

            # เงื่อนไข: smooth path และ direction consistent
            if (smoothness >= min_smoothness and
                direction_consistency >= min_consistency):
                self.status = 'ORANGE'
                if DEBUG_MODE:
                    print(f"✅ Tracker {self.id}: Changed to ORANGE (velocity={velocity:.2f}, distance={total_distance:.2f}, "
                          f"smoothness={smoothness:.2f}, consistency={direction_consistency:.2f}, hover={is_hover_mode})")
            elif DEBUG_MODE and self.yellow_duration % 30 == 0:  # Log ทุก 30 เฟรม
                print(f"⚠️ Tracker {self.id}: Not ORANGE yet (smoothness={smoothness:.2f} >= {min_smoothness:.2f}?, "
                      f"consistency={direction_consistency:.2f} >= {min_consistency:.2f}?, hover={is_hover_mode})")

    def is_close_for_reid(self, new_rect, frame_w=None, frame_h=None, thresholds=None):
        """
        ตรวจสอบว่า new_rect อยู่ใกล้พอสำหรับ re-identification
        ใช้ resolution-dependent thresholds และ predicted position เมื่อวัตถุหายไปชั่วครู่

        Args:
            new_rect: rectangle ใหม่ที่ต้องการตรวจสอบ
            frame_w: ความกว้างของเฟรม (สำหรับ resolution-dependent thresholds)
            frame_h: ความสูงของเฟรม (สำหรับ resolution-dependent thresholds)
            thresholds: dictionary ของ thresholds (ถ้ามีแล้ว ไม่ต้องคำนวณใหม่)

        Returns:
            bool: True ถ้าใกล้พอสำหรับ re-identification
        """
        # ใช้ thresholds ที่ส่งมา หรือคำนวณใหม่
        if thresholds is None and frame_w is not None and frame_h is not None:
            thresholds = get_resolution_dependent_thresholds(frame_w, frame_h)
        elif thresholds is None:
            # Fallback to legacy thresholds if resolution not available
            thresholds = {
                'MAX_REID_DISTANCE': MAX_REID_DISTANCE,
                'MERGE_DISTANCE': MERGE_DISTANCE,
                'SMALL_OBJECT_REID_DISTANCE': MAX_REID_DISTANCE * 1.5
            }

        x, y, w, h = self.current_rect
        vx_mag = self.velocity_mag

        # ตรวจสอบว่าเป็น small object หรือไม่
        area = w * h
        is_small_object = area < thresholds.get('DRONE_SMALL_OBJECT_AREA_THRESHOLD', 200)

        # ใช้ predicted position ถ้าวัตถุหายไปชั่วครู่
        if self.missed_frames > 0 and frame_w is not None and frame_h is not None:
            predicted_centroid = self.get_predicted_centroid(self.missed_frames, frame_w, frame_h)
            if predicted_centroid is not None:
                cx1, cy1 = predicted_centroid
            else:
                cx1, cy1 = self.centroids[-1] if self.centroids else (x + w//2, y + h//2)
        else:
            cx1, cy1 = self.centroids[-1] if self.centroids else (x + w//2, y + h//2)

        cx2, cy2 = new_rect[0] + new_rect[2] // 2, new_rect[1] + new_rect[3] // 2
        distance = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

        # ใช้ threshold ตามประเภทวัตถุ
        if is_small_object:
            # Small object: ใช้ tolerance สูงขึ้น
            if self.has_clear_path_for_small_object():
                # Small object ที่มี path ชัดเจน → tolerance สูงสุด
                MAX_REID_DIST = thresholds.get('SMALL_OBJECT_REID_DISTANCE', thresholds['MAX_REID_DISTANCE'] * 1.5)
            else:
                # Small object ปกติ → tolerance ปานกลาง
                MAX_REID_DIST = thresholds.get('MAX_REID_DISTANCE', MAX_REID_DISTANCE) * 1.2
        else:
            # 1. Size Factor (use diagonal length)
            size_factor = (w**2 + h**2)**0.5

            # 2. Dynamic Max Distance calculation (Base + Size * Velocity Multiplier)
            dynamic_tolerance = size_factor * vx_mag * 0.2

            # Min check to prevent tiny objects with zero velocity having no tolerance
            base_reid_dist = thresholds.get('MAX_REID_DISTANCE', MAX_REID_DISTANCE)
            MAX_REID_DIST = max(thresholds.get('MERGE_DISTANCE', MERGE_DISTANCE), dynamic_tolerance)

            # Limit the max distance to prevent matching across the screen
            MAX_REID_DIST = min(MAX_REID_DIST, base_reid_dist)

        return distance < MAX_REID_DIST

# --- MERGE FUNCTION (Unchanged) ---
def merge_close_rectangles(rects, dist_thresh=50, frame_w=None, frame_h=None):
    if not rects: return []
    # ... (Unchanged Disjoint Set Union logic) ...
    def is_close(r1, r2):
        # Checks if two rectangles are within dist_thresh of each other in either dimension
        return not (r1[0] + r1[2] + dist_thresh < r2[0] or r2[0] + r2[2] + dist_thresh < r1[0] or
                    r1[1] + r1[3] + dist_thresh < r2[1] or r2[1] + r2[3] + dist_thresh < r1[1])
    labels = list(range(len(rects)))
    def find(i):
        while labels[i] != i: labels[i] = labels[labels[i]]; i = labels[i]
        return i
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: labels[root_j] = root_i
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if is_close(rects[i], rects[j]): union(i, j)
    groups = {}
    for i in range(len(rects)):
        root = find(i)
        if root not in groups: groups[root] = []
        groups[root].append(rects[i])
    merged = []
    for g in groups.values():
        if len(g) == 1:
            # ตรวจสอบขนาดก่อนเพิ่ม
            x, y, w_rect, h_rect = g[0]
            area = w_rect * h_rect
            if frame_w and frame_h:
                max_area = min(DRONE_MAX_AREA, frame_w * frame_h * 0.1)
                if area > max_area or w_rect > frame_w * 0.5 or h_rect > frame_h * 0.5:
                    continue  # ข้าม rect ที่ใหญ่เกินไป
            merged.append(g[0])
            continue

        mx, my = min(r[0] for r in g), min(r[1] for r in g)
        mw, mh = max(r[0]+r[2] for r in g) - mx, max(r[1]+r[3] for r in g) - my

        # ตรวจสอบขนาด merged rect ก่อน merge
        merged_area = mw * mh
        if frame_w and frame_h:
            max_area = min(DRONE_MAX_AREA, frame_w * frame_h * 0.1)
            if merged_area > max_area or mw > frame_w * 0.5 or mh > frame_h * 0.5:
                # ใช้ rect ที่ใหญ่ที่สุดในกลุ่มแทน (แต่ต้องไม่ใหญ่เกินไป)
                largest_rect = max(g, key=lambda r: r[2] * r[3])
                lx, ly, lw, lh = largest_rect
                larea = lw * lh
                if larea <= max_area and lw <= frame_w * 0.5 and lh <= frame_h * 0.5:
                    merged.append(largest_rect)
                continue  # ข้ามถ้าใหญ่เกินไป

        # ตรวจสอบว่ากล่องใหญ่เกินไปหรือไม่ (ป้องกันกล่องขยายตัว)
        avg_area = sum(r[2] * r[3] for r in g) / len(g)

        # ถ้ากล่องใหญ่เกิน MAX_MERGE_AREA_RATIO เท่าของค่าเฉลี่ย → ใช้ rect ที่ใหญ่ที่สุดในกลุ่มแทน
        if merged_area > avg_area * MAX_MERGE_AREA_RATIO:
            # ใช้ rect ที่ใหญ่ที่สุดในกลุ่มแทน (ไม่ merge)
            largest_rect = max(g, key=lambda r: r[2] * r[3])
            # ตรวจสอบขนาด largest_rect ด้วย
            lx, ly, lw, lh = largest_rect
            larea = lw * lh
            if frame_w and frame_h:
                max_area = min(DRONE_MAX_AREA, frame_w * frame_h * 0.1)
                if larea <= max_area and lw <= frame_w * 0.5 and lh <= frame_h * 0.5:
                    merged.append(largest_rect)
            else:
                merged.append(largest_rect)
        else:
            merged.append((mx, my, mw, mh))
    return merged

# --- IOU CALCULATION FUNCTION ---
def calculate_iou(rect1, rect2):
    """
    Calculate Intersection over Union (IOU) between two rectangles.

    Args:
        rect1: (x1, y1, w1, h1) - first rectangle
        rect2: (x2, y2, w2, h2) - second rectangle

    Returns:
        iou: IOU value (0.0 to 1.0)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def find_motion_boxes(mask_cpu, min_area=None, area_mask=None, roi_boxes=None,
                     object_size_category=None, adaptive_min_area=None, hybrid_tracker=None):
    """
    Find all motion detection boxes (green boxes) from mask.
    Optionally filter by area_mask and allow exceptions for ROI boxes.

    Args:
        mask_cpu: Motion mask (binary image)
        min_area: Minimum area threshold for motion boxes (uses HYBRID_MIN_MOTION_AREA if None)
        area_mask: Optional - binary mask defining allowed area for motion detection (255 = allowed, 0 = not allowed)
        roi_boxes: Optional - list of ROI boxes [(x1, y1, x2, y2), ...] where motion is always allowed
        object_size_category: Optional - 'TINY', 'SMALL', 'MEDIUM', 'LARGE' (ใช้สำหรับคำนวณ min_area ตาม size)
        adaptive_min_area: Optional - ค่า adaptive min_area จาก AdaptiveMinAreaManager (ใช้เมื่อมี object_size_category)
        hybrid_tracker: Optional - HybridDroneTracker instance (ใช้สำหรับเรียก get_min_motion_area_for_size)

    Returns:
        List of (x, y, w, h) rectangles
    """
    # ถ้ามี object_size_category และ hybrid_tracker ให้ใช้ size-based min_area
    if object_size_category is not None and hybrid_tracker is not None:
        if adaptive_min_area is None:
            adaptive_min_area = HYBRID_MIN_MOTION_AREA
        min_area = hybrid_tracker.get_min_motion_area_for_size(object_size_category, adaptive_min_area, lock_mode=False)
    elif min_area is None:
        min_area = HYBRID_MIN_MOTION_AREA

    # OPTIMIZATION: ใช้ area_mask กับ mask_cpu ก่อนหา contours (ประหยัดการประมวลผล)
    # ทำงานเฉพาะบนเส้น horizon เท่านั้น
    if area_mask is not None:
        # ใช้ mask เฉพาะพื้นที่ที่อนุญาต (บนเส้น horizon)
        masked_cpu = cv2.bitwise_and(mask_cpu, area_mask)
    else:
        masked_cpu = mask_cpu

    # OPTIMIZATION: Adaptive Resize Mask ก่อนหา Contours (สำคัญที่สุด - ทุก Resolution)
    # ถ้า min_area <= 1 → ไม่ resize (รักษา 1 pixel targets)
    # ถ้า min_area > 1 และ resolution >= 2K → resize (เร็วขึ้น 2-4 เท่า)
    from config import (
        CONTOUR_DETECTION_RESIZE_4K,
        CONTOUR_DETECTION_RESIZE_2K,
        CONTOUR_DETECTION_RESIZE_MIN_AREA_THRESHOLD,
        CONTOUR_DETECTION_SCALE_4K,
        CONTOUR_DETECTION_SCALE_2K,
    )

    original_h, original_w = masked_cpu.shape[:2]
    should_resize = False
    scale_factor = 1.0

    # ตรวจสอบว่าควร resize หรือไม่
    if min_area is not None and min_area > CONTOUR_DETECTION_RESIZE_MIN_AREA_THRESHOLD:
        total_pixels = original_w * original_h
        if total_pixels >= 3840 * 2160 and CONTOUR_DETECTION_RESIZE_4K:  # 4K
            should_resize = True
            scale_factor = CONTOUR_DETECTION_SCALE_4K
        elif total_pixels >= 2560 * 1440 and CONTOUR_DETECTION_RESIZE_2K:  # 2K
            should_resize = True
            scale_factor = CONTOUR_DETECTION_SCALE_2K

    # Resize mask ถ้าจำเป็น
    if should_resize and scale_factor < 1.0:
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        masked_cpu_resized = cv2.resize(masked_cpu, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Scale min_area ตาม scale factor (อย่างน้อย 1)
        scaled_min_area = max(1, int(min_area * scale_factor * scale_factor))
    else:
        masked_cpu_resized = masked_cpu
        scaled_min_area = min_area

    contours, _ = cv2.findContours(masked_cpu_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > scaled_min_area:
            x, y, w, h = cv2.boundingRect(cnt)

            # Scale bounding boxes กลับเป็น resolution เดิม (ถ้า resize)
            if should_resize and scale_factor < 1.0:
                inv_scale = 1.0 / scale_factor
                x = int(x * inv_scale)
                y = int(y * inv_scale)
                w = int(w * inv_scale)
                h = int(h * inv_scale)

            # ถ้ามี ROI boxes → ตรวจสอบว่าใน ROI หรือไม่
            # (area_mask ถูกใช้แล้วตอนหา contours → ไม่ต้องตรวจสอบอีก)
            if roi_boxes is not None and len(roi_boxes) > 0:
                center_x = x + w // 2
                center_y = y + h // 2
                in_roi = False
                for roi_box in roi_boxes:
                    if len(roi_box) == 4:
                        rx1, ry1, rx2, ry2 = roi_box
                        # Fast check: center point within ROI box
                        if rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2:
                            in_roi = True
                            break
                        # Fallback: check bounds overlap (only if center check fails)
                        elif not (x + w < rx1 or x > rx2 or y + h < ry1 or y > ry2):
                            in_roi = True
                            break
                if not in_roi:
                    continue  # ไม่ใน ROI → ข้าม

            motion_boxes.append((x, y, w, h))

    return motion_boxes

# --- YOLO DETECTION IN ROI FUNCTION ---
def detect_yolo_in_roi(yolo_model, tracker, frame, roi_box, frame_w, frame_h):
    """
    ตรวจจับวัตถุใน ROI โดยใช้ YOLO

    Args:
        yolo_model: YOLO model instance
        tracker: TrackedObject instance
        frame: Full frame image
        roi_box: (x1, y1, x2, y2) ROI coordinates
        frame_w: Frame width
        frame_h: Frame height

    Returns:
        (yolo_rect, confidence): ((x, y, w, h), confidence) หรือ (None, None) ถ้าไม่เจอ
    """
    if yolo_model is None:
        return None, None

    rx1, ry1, rx2, ry2 = roi_box

    # ตรวจสอบว่า ROI อยู่ในขอบเขตเฟรม
    rx1 = max(0, min(rx1, frame_w))
    ry1 = max(0, min(ry1, frame_h))
    rx2 = max(rx1, min(rx2, frame_w))
    ry2 = max(ry1, min(ry2, frame_h))

    # ตรวจสอบว่า ROI มีขนาดพอ
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return None, None

    # Extract ROI area
    roi_area = frame[ry1:ry2, rx1:rx2]

    if roi_area.size == 0:
        return None, None

    # ตรวจสอบและแปลงเป็น BGR (3 channels) ถ้ามี 4 channels
    if len(roi_area.shape) == 3 and roi_area.shape[2] == 4:
        roi_area = cv2.cvtColor(roi_area, cv2.COLOR_BGRA2BGR)

    try:
        # Run YOLO detection in ROI
        results = yolo_model.predict(roi_area, device=0, verbose=False, conf=HYBRID_SEARCH_CONF)

        if len(results[0].boxes) == 0:
            return None, None

        # Get tracker center for distance calculation
        tracker_center = tracker.centroids[-1] if tracker.centroids else None
        if tracker_center is None:
            return None, None

        # เลือก YOLO box ที่ใกล้กับ tracker center มากที่สุด
        best_yolo_idx = -1
        min_yolo_dist = float('inf')

        for i, box in enumerate(results[0].boxes):
            b = box.xyxy[0].cpu().numpy()
            # Convert from ROI coordinates to full frame coordinates
            yolo_x = int(b[0]) + rx1
            yolo_y = int(b[1]) + ry1
            yolo_w = int(b[2] - b[0])
            yolo_h = int(b[3] - b[1])
            yolo_center = (yolo_x + yolo_w // 2, yolo_y + yolo_h // 2)

            # Calculate distance to tracker center
            dist = math.sqrt((yolo_center[0] - tracker_center[0])**2 +
                           (yolo_center[1] - tracker_center[1])**2)

            if dist < min_yolo_dist:
                min_yolo_dist = dist
                best_yolo_idx = i

        # ตรวจสอบว่ากล่องที่เจอไม่อยู่ไกลเกินไป
        if best_yolo_idx >= 0 and min_yolo_dist < HYBRID_DIST_THRESHOLD:
            b = results[0].boxes[best_yolo_idx].xyxy[0].cpu().numpy()
            confidence = float(results[0].boxes[best_yolo_idx].conf[0])  # ดึง confidence
            yolo_x = int(b[0]) + rx1
            yolo_y = int(b[1]) + ry1
            yolo_w = int(b[2] - b[0])
            yolo_h = int(b[3] - b[1])

            # Return as ((x, y, w, h), confidence)
            return (yolo_x, yolo_y, yolo_w, yolo_h), confidence

        return None, None
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ YOLO detection error: {e}")
        return None, None

# --- ADAPTIVE HYBRID MODE LOGIC (Grid-Based) ---
def adaptive_hybrid_mode_logic_enhanced(hybrid_tracker, active_trackers, grid_system,
                                       current_mode, frame_counter, w, h):
    """
    Enhanced Adaptive Hybrid Mode Logic with Grid-based decision making

    Returns:
        (new_mode, best_tracker_id, grid_confidence)
    """
    if not ADAPTIVE_HYBRID_MODE_ENABLED:
        return current_mode, None, 0.0

    status_priority = {'RED': 2, 'ORANGE': 1, 'YELLOW': 0, 'GREEN': 0}
    min_priority = status_priority.get(ADAPTIVE_HYBRID_MIN_STATUS_FOR_LOCK, 1)

    # คำนวณ grid noise เฉลี่ยทั้งเฟรม (ใช้ cache)
    frame_noise_regions = grid_system.find_low_noise_regions(
        threshold=GRID_NOISE_THRESHOLD_FOR_HYBRID, min_cells=1,
        current_frame=frame_counter, cache_frames=GRID_CONFIDENCE_CACHE_FRAMES
    )
    avg_frame_noise = np.mean([r[2] for r in frame_noise_regions]) if frame_noise_regions else 0.5

    if current_mode == 'hybrid':
        if hybrid_tracker is None:
            return 'hybrid', None, 0.0

        # ถ้า hybrid tracker lock อยู่ → ใช้ hybrid ต่อไป
        try:
            is_locked = hybrid_tracker.target_locked
        except (TypeError, AttributeError) as e:
            if DEBUG_MODE:
                print(f"⚠️ Error accessing target_locked: {e}, type: {type(hybrid_tracker.target_locked)}")
            return 'hybrid', None, 0.0

        if is_locked:
            # ตรวจสอบ grid confidence ใน ROI
            if hybrid_tracker.roi_box:
                roi_x1, roi_y1, roi_x2, roi_y2 = hybrid_tracker.roi_box
                grid_conf = grid_system.get_grid_confidence(
                    roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1,
                    frame_counter, GRID_CONFIDENCE_CACHE_FRAMES
                )
            else:
                grid_conf = 1.0 - avg_frame_noise

            return 'hybrid', None, grid_conf

        # ถ้า hybrid tracker unlock (หายเป้าหมาย)
        try:
            is_locked = hybrid_tracker.target_locked
        except (TypeError, AttributeError) as e:
            if DEBUG_MODE:
                print(f"⚠️ Error accessing target_locked: {e}, type: {type(hybrid_tracker.target_locked)}")
            return 'hybrid', None, 0.0

        if not is_locked:
            # ถ้า grid noise สูง → ใช้ multi mode ค้นหา (พื้นหลังขยับ)
            if avg_frame_noise > GRID_NOISE_THRESHOLD_FOR_MULTI:
                return 'multi', None, 1.0 - avg_frame_noise
            else:
                # ถ้า grid noise ต่ำ → ยังใช้ hybrid ต่อไป (อาจจะเจอเร็ว)
                return 'hybrid', None, 1.0 - avg_frame_noise

    elif current_mode == 'multi':
        # ตรวจสอบว่ามี tracker ที่เหมาะสมสำหรับ hybrid หรือไม่
        # ใช้ priority system เพื่อเลือก tracker ที่ดีที่สุด
        high_priority_trackers = [
            (tid, tr) for tid, tr in active_trackers.items()
            if status_priority.get(tr.status, 0) >= min_priority
        ]

        if len(high_priority_trackers) > 0:
            # คำนวณ priority score สำหรับทุก high priority tracker
            tracker_scores = {}
            for tid, tracker in high_priority_trackers:
                priority_score = calculate_priority_score(tracker, tracker=tracker, frame_counter=frame_counter)
                tracker_scores[tid] = (tracker, priority_score)

            # เลือก tracker ที่มี priority score สูงสุด
            if tracker_scores:
                best_tracker_id = max(tracker_scores.items(), key=lambda x: x[1][1])[0]
                best_tracker, best_score = tracker_scores[best_tracker_id]

                x, y, w_box, h_box = best_tracker.current_rect

            # คำนวณ grid confidence สำหรับ tracker นี้ (ใช้ cache)
            grid_conf = grid_system.get_grid_confidence(x, y, w_box, h_box,
                                                       frame_counter, GRID_CONFIDENCE_CACHE_FRAMES)
            avg_noise, max_noise, min_noise = grid_system.get_grid_noise_level(
                x, y, w_box, h_box, frame_counter, GRID_CONFIDENCE_CACHE_FRAMES
            )

            # ตรวจสอบ confidence (รวม grid confidence)
            tracker_conf = 0.5  # default
            if hasattr(best_tracker, 'classification_confidence'):
                tracker_conf = best_tracker.classification_confidence

            # รวม confidence: tracker confidence + grid confidence
            combined_conf = (tracker_conf * (1.0 - GRID_CONFIDENCE_WEIGHT) +
                           grid_conf * GRID_CONFIDENCE_WEIGHT)

            # ถ้า noise ต่ำและ confidence สูง → สลับกลับ hybrid
            # ใช้ priority score เป็นส่วนหนึ่งของการตัดสินใจ
            from config import MIN_PRIORITY_FOR_PRIMARY
            if avg_noise < GRID_NOISE_THRESHOLD_FOR_HYBRID and (combined_conf >= 0.5 or best_score >= MIN_PRIORITY_FOR_PRIMARY):
                if DEBUG_MODE:
                    print(f"🔄 Adaptive Hybrid: Switching to hybrid mode with tracker {best_tracker_id} (priority={best_score:.2f}, conf={combined_conf:.2f})")
                return 'hybrid', best_tracker_id, grid_conf

        # ถ้าใช้ multi mode ค้นหานานเกินไป → กลับ hybrid
        # (จะตรวจสอบใน main loop ผ่าน multi_mode_search_start_frame)
        pass

    return current_mode, None, 1.0 - avg_frame_noise

# --- CLASS: HYBRID DRONE TRACKER (Multi-target version) ---
class HybridDroneTracker:
    """
    Multi-target persistent tracker using YOLO + Motion (MOG2) hybrid approach.
    Adapted to work with frame input instead of VideoCapture.
    """
    def __init__(self, yolo_model, frame_w, frame_h, thresholds=None):
        """
        Initialize hybrid tracker.

        Args:
            yolo_model: YOLO model instance (already loaded)
            frame_w: Frame width
            frame_h: Frame height
            thresholds: Dictionary of resolution-dependent thresholds (optional)
        """
        self.model = yolo_model

        # Initialize CUDA MOG2 (ใช้ MOG2_HISTORY_4K สำหรับ 4K)
        # ตรวจสอบว่า MOG2_HISTORY_4K ถูก define หรือไม่ (ถ้ายังไม่ถูก define ใน main → ใช้ MOG2_HISTORY)
        mog2_history_4k = globals().get('MOG2_HISTORY_4K', MOG2_HISTORY)
        mog2_history_value = mog2_history_4k if (frame_w >= 3840) else MOG2_HISTORY
        self.back_sub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=mog2_history_value,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE_MIN, MORPH_KERNEL_SIZE_MIN))

        # Multi-target tracking: เก็บหลาย targets
        # {target_id: {'target_locked': bool, 'roi_box': [x1,y1,x2,y2], 'last_center': (x,y), 'missed_frames': int, 'confidence': float}}
        self.targets = {}  # Dictionary of tracked targets
        self.next_target_id = 0
        self.gpu_frame = cv2.cuda_GpuMat()
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Store resolution-dependent thresholds
        if thresholds is None:
            self.thresholds = get_resolution_dependent_thresholds(frame_w, frame_h)
        else:
            self.thresholds = thresholds

        # Cache for frame properties to avoid repeated checks
        self._cached_frame_shape = None  # (h, w, channels)
        self._cached_needs_bgr_conversion = None  # True/False if frame needs BGR conversion

        # Blacklist system สำหรับวัตถุนิ่ง - เก็บพื้นที่ในเฟรม (area) ที่ถูก blacklist
        # Format: [{'bbox': (x1, y1, x2, y2), 'timestamp': float, 'count': int, 'first_seen': float}, ...]
        # bbox นี้เป็นพื้นที่ในเฟรมที่ถูก blacklist (มี padding รอบ bbox ของวัตถุ)
        # count = จำนวนครั้งที่ถูก blacklist
        # first_seen = timestamp ครั้งแรกที่ถูก blacklist
        self.blacklist = []  # List of blacklisted areas in frame with timestamps (ระดับปกติ)
        self.permanent_blacklist = []  # List of permanent blacklisted areas (ระดับถาวร)

        # Motion-only target tracking system
        # สำหรับ track motion boxes ที่เคลื่อนที่ smooth แต่ยังไม่มี YOLO detection match
        self.motion_only_targets = {}  # {motion_id: {'centers_history': [...], 'last_rect': (x,y,w,h), 'last_center': (x,y), 'path_quality': float, 'created_frame': int, 'missed_frames': int, ...}}
        self.next_motion_id = 0

        # Footprints module สำหรับวาดรอยเท้า
        if FOOTPRINTS_MODULE_ENABLED:
            from footprints_module import FootprintsDrawer
            self.footprints_drawer = FootprintsDrawer(
                history_frames=FOOTPRINTS_HISTORY_FRAMES,
                frame_w=frame_w,
                frame_h=frame_h
            )
        else:
            self.footprints_drawer = None

        # Performance metrics
        from config import PERFORMANCE_METRICS_ENABLED, PERFORMANCE_METRICS_INTERVAL
        self.performance_metrics_enabled = PERFORMANCE_METRICS_ENABLED
        self.performance_metrics_interval = PERFORMANCE_METRICS_INTERVAL
        self.performance_metrics = {
            'roi_motion_detection_time': [],
            'motion_boxes_found': [],
            'motion_boxes_filtered': [],
            'hit_rate': [],
            'last_metrics_frame': -1
        }

    def reset(self):
        """Reset tracker state (used when switching modes)"""
        self.targets = {}
        self.next_target_id = 0
        # Reset background subtractor (ใช้ MOG2_HISTORY_4K สำหรับ 4K)
        # ตรวจสอบว่า MOG2_HISTORY_4K ถูก define หรือไม่ (ถ้ายังไม่ถูก define ใน main → ใช้ MOG2_HISTORY)
        mog2_history_4k = globals().get('MOG2_HISTORY_4K', MOG2_HISTORY)
        mog2_history_value = mog2_history_4k if (self.frame_w >= 3840) else MOG2_HISTORY
        self.back_sub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=mog2_history_value,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )

    @property
    def target_locked(self):
        """Backward compatibility: return True if any target is locked"""
        return len(self.targets) > 0

    @property
    def roi_box(self):
        """Backward compatibility: return first target's ROI"""
        if len(self.targets) > 0:
            first_target = next(iter(self.targets.values()))
            return first_target['roi_box']
        return None

    @property
    def last_center(self):
        """Backward compatibility: return first target's center"""
        if len(self.targets) > 0:
            first_target = next(iter(self.targets.values()))
            return first_target['last_center']
        return None

    @property
    def missed_frames(self):
        """Backward compatibility: return max missed frames"""
        if len(self.targets) > 0:
            return max([t['missed_frames'] for t in self.targets.values()])
        return 0

    def get_dist(self, p1, p2):
        if p1 is None or p2 is None:
            return 1000
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_tiny_object_for_target(self, target, frame_w=None, frame_h=None):
        """
        ตรวจสอบว่า target เป็น tiny object หรือไม่

        Args:
            target: Target dictionary
            frame_w: Frame width (optional, use self.frame_w if None)
            frame_h: Frame height (optional, use self.frame_h if None)

        Returns:
            bool: True ถ้าเป็น tiny object
        """
        if frame_w is None:
            frame_w = self.frame_w
        if frame_h is None:
            frame_h = self.frame_h

        # ตรวจสอบว่ามี rect หรือไม่
        if 'last_rect' not in target or target['last_rect'] is None:
            return False

        x, y, w, h = target['last_rect']
        area = w * h
        diagonal = math.sqrt(w**2 + h**2)

        # คำนวณ frame area และ diagonal
        frame_area = frame_w * frame_h
        frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)

        # คำนวณ thresholds
        tiny_area_threshold = TINY_OBJECT_AREA_THRESHOLD_RATIO * frame_area
        tiny_diagonal_threshold = TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        # ตรวจสอบว่าเป็น tiny object หรือไม่ (ใช้ทั้ง area และ diagonal)
        return area < tiny_area_threshold and diagonal < tiny_diagonal_threshold

    def get_object_size_category_for_target(self, target, frame_w=None, frame_h=None):
        """
        คำนวณ object_size_category สำหรับ target และเก็บไว้ใน cache

        Args:
            target: Target dictionary
            frame_w: Frame width (optional, use self.frame_w if None)
            frame_h: Frame height (optional, use self.frame_h if None)

        Returns:
            str: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None
        """
        if frame_w is None:
            frame_w = self.frame_w
        if frame_h is None:
            frame_h = self.frame_h

        # ตรวจสอบว่ามี rect หรือไม่
        if 'last_rect' not in target or target['last_rect'] is None:
            return None

        x, y, w, h = target['last_rect']
        area = w * h
        diagonal = math.sqrt(w**2 + h**2)

        # คำนวณ frame area และ diagonal
        frame_area = frame_w * frame_h
        frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)

        # คำนวณ thresholds
        tiny_area_threshold = TINY_OBJECT_AREA_THRESHOLD_RATIO * frame_area
        tiny_diagonal_threshold = TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        small_area_threshold = self.thresholds.get('DRONE_SMALL_OBJECT_AREA_THRESHOLD',
                                                   DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO * frame_area)
        small_diagonal_threshold = SMALL_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        max_area_threshold = self.thresholds.get('DRONE_MAX_AREA',
                                                 DRONE_MAX_AREA_RATIO * frame_area)

        # ตรวจสอบว่าเป็น tiny object หรือไม่
        if area < tiny_area_threshold and diagonal < tiny_diagonal_threshold:
            return 'TINY'

        # ตรวจสอบว่าเป็น small object หรือไม่
        if area < small_area_threshold and diagonal < small_diagonal_threshold:
            return 'SMALL'

        # ตรวจสอบว่าเป็น medium object หรือไม่
        if area <= max_area_threshold:
            return 'MEDIUM'

        # ถ้าใหญ่กว่า max_area_threshold → LARGE
        return 'LARGE'

    def get_min_motion_area_for_size(self, object_size_category, adaptive_min_area, lock_mode=False):
        """
        คำนวณ min_area ตาม object size

        Args:
            object_size_category: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None
            adaptive_min_area: ค่า adaptive min_area จาก AdaptiveMinAreaManager
            lock_mode: ถ้า True ให้ลดลง 50%

        Returns:
            min_area: ค่า min_area ที่เหมาะสม
        """
        if object_size_category is None:
            # ถ้าไม่มี object_size_category ใช้ adaptive_min_area ปกติ
            base_area = adaptive_min_area
        elif object_size_category == 'TINY':
            # สำหรับ tiny objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 2)
            if ADAPTIVE_TINY_OBJECT_MIN_AREA_ENABLED:
                # ถ้าเปิด adaptive ให้ปรับตาม adaptive_min_area แต่ไม่เกิน max
                base_area = min(TINY_OBJECT_MIN_MOTION_AREA_MAX,
                               max(TINY_OBJECT_MIN_MOTION_AREA_BASE, adaptive_min_area))
            else:
                # ถ้าปิด adaptive ให้ใช้ base เสมอ
                base_area = TINY_OBJECT_MIN_MOTION_AREA_BASE
        elif object_size_category == 'SMALL':
            # สำหรับ small objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 5)
            base_area = min(SMALL_OBJECT_MIN_MOTION_AREA_MAX,
                           max(SMALL_OBJECT_MIN_MOTION_AREA_BASE, adaptive_min_area))
        elif object_size_category == 'MEDIUM':
            # สำหรับ medium objects: ใช้ base ถึง max (ไม่เพิ่มเกิน 10)
            base_area = min(MEDIUM_OBJECT_MIN_MOTION_AREA_MAX,
                           max(MEDIUM_OBJECT_MIN_MOTION_AREA_BASE, adaptive_min_area))
        else:  # 'LARGE' หรืออื่นๆ
            # สำหรับ large objects: ใช้ adaptive_min_area ปกติ
            base_area = adaptive_min_area

        # ถ้า lock_mode ให้ลดลง 50% จากค่า base
        if lock_mode:
            return max(1, int(base_area * 0.5))

        return base_area

    def calculate_bbox_iou(self, bbox1, bbox2):
        """
        คำนวณ IOU (Intersection over Union) ระหว่าง 2 bboxes

        Args:
            bbox1: (x1, y1, x2, y2) หรือ (x, y, w, h)
            bbox2: (x1, y1, x2, y2) หรือ (x, y, w, h)

        Returns:
            IOU value (0.0-1.0)
        """
        # แปลงเป็น (x1, y1, x2, y2) ถ้าจำเป็น
        if len(bbox1) == 4:
            if bbox1[2] < bbox1[0] or bbox1[3] < bbox1[1]:
                # เป็น (x, y, w, h)
                x1_1, y1_1, w1, h1 = bbox1
                x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            else:
                # เป็น (x1, y1, x2, y2)
                x1_1, y1_1, x2_1, y2_1 = bbox1
        else:
            return 0.0

        if len(bbox2) == 4:
            if bbox2[2] < bbox2[0] or bbox2[3] < bbox2[1]:
                # เป็น (x, y, w, h)
                x1_2, y1_2, w2, h2 = bbox2
                x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            else:
                # เป็น (x1, y1, x2, y2)
                x1_2, y1_2, x2_2, y2_2 = bbox2
        else:
            return 0.0

        # คำนวณ intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # คำนวณ union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def find_overlapping_blacklist(self, bbox, blacklist_type='normal'):
        """
        หา blacklist entry ที่ซ้อนทับกับ bbox ที่กำหนด

        Args:
            bbox: (x1, y1, x2, y2) หรือ (x, y, w, h)
            blacklist_type: 'normal' หรือ 'permanent'

        Returns:
            Blacklist entry ที่ซ้อนทับ หรือ None
        """
        if blacklist_type == 'permanent':
            blacklist_to_check = self.permanent_blacklist
        else:
            blacklist_to_check = self.blacklist

        # แปลง bbox เป็น (x1, y1, x2, y2) ถ้าจำเป็น
        if len(bbox) == 4:
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                # เป็น (x, y, w, h)
                x, y, w, h = bbox
                bbox_normalized = (x, y, x + w, y + h)
            else:
                # เป็น (x1, y1, x2, y2)
                bbox_normalized = bbox
        else:
            return None

        for entry in blacklist_to_check:
            entry_bbox = entry['bbox']
            iou = self.calculate_bbox_iou(bbox_normalized, entry_bbox)
            if iou >= BLACKLIST_OVERLAP_THRESHOLD:
                return entry

        return None

    def check_movement_in_blacklist(self, bbox, motion_boxes=None, velocity=None):
        """
        ตรวจสอบว่าวัตถุมีการเคลื่อนที่หรือไม่ (สำหรับอนุญาตให้ตรวจจับใน blacklist)

        Args:
            bbox: (x1, y1, x2, y2) หรือ (x, y, w, h)
            motion_boxes: list of motion boxes (optional) สำหรับตรวจสอบ motion overlap
            velocity: velocity ของวัตถุ (optional) สำหรับ target ที่ติดตามอยู่แล้ว

        Returns:
            True ถ้ามีการเคลื่อนที่, False ถ้าไม่มี
        """
        # ตรวจสอบ motion overlap (ถ้ามี motion_boxes)
        if motion_boxes is not None and len(motion_boxes) > 0:
            # แปลง bbox เป็น (x1, y1, x2, y2) ถ้าจำเป็น
            if len(bbox) == 4:
                if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                    # เป็น (x, y, w, h)
                    x, y, w, h = bbox
                    bbox_normalized = (x, y, x + w, y + h)
                else:
                    # เป็น (x1, y1, x2, y2)
                    bbox_normalized = bbox
            else:
                bbox_normalized = None

            if bbox_normalized:
                for motion_box in motion_boxes:
                    # motion_box อาจเป็น (x, y, w, h) หรือ (x1, y1, x2, y2)
                    if len(motion_box) == 4:
                        if motion_box[2] < motion_box[0] or motion_box[3] < motion_box[1]:
                            # เป็น (x, y, w, h)
                            mx, my, mw, mh = motion_box
                            motion_bbox = (mx, my, mx + mw, my + mh)
                        else:
                            # เป็น (x1, y1, x2, y2)
                            motion_bbox = motion_box

                        iou = self.calculate_bbox_iou(bbox_normalized, motion_bbox)
                        if iou >= BLACKLIST_MOTION_IOU_THRESHOLD:
                            return True

        # ตรวจสอบ velocity (ถ้ามี velocity)
        if velocity is not None:
            # ตรวจสอบว่า velocity เป็น tuple (vx, vy) หรือ float (velocity magnitude)
            velocity_magnitude = 0.0
            if isinstance(velocity, (tuple, list)) and len(velocity) == 2:
                # เป็น tuple (vx, vy) → คำนวณ magnitude
                vx, vy = velocity
                if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                    velocity_magnitude = math.sqrt(vx**2 + vy**2)
            elif isinstance(velocity, (int, float)):
                # เป็น float (velocity magnitude) → ใช้ค่าโดยตรง
                velocity_magnitude = float(velocity)

            if velocity_magnitude >= BLACKLIST_MIN_VELOCITY:
                return True

        return False

    def _check_if_in_blacklist_area(self, center, bbox=None):
        """
        Helper method: ตรวจสอบว่าตำแหน่งอยู่ใน blacklist area หรือไม่ (ไม่ตรวจสอบ movement)
        สำหรับ debug message เท่านั้น
        """
        if not STATIONARY_DETECTION_ENABLED:
            return False

        current_time = time.time()

        # ตรวจสอบ permanent blacklist
        if PERMANENT_BLACKLIST_ENABLED:
            for entry in self.permanent_blacklist:
                # ตรวจสอบ duration (ถ้ามี)
                if PERMANENT_BLACKLIST_DURATION_SECONDS > 0:
                    if current_time - entry['timestamp'] >= PERMANENT_BLACKLIST_DURATION_SECONDS:
                        continue

                x1, y1, x2, y2 = entry['bbox']
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    return True

                if bbox is not None:
                    if len(bbox) == 4:
                        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                            bx, by, bw, bh = bbox
                            bbox_normalized = (bx, by, bx + bw, by + bh)
                        else:
                            bbox_normalized = bbox
                        if self.calculate_bbox_iou(bbox_normalized, entry['bbox']) > 0:
                            return True

        # ตรวจสอบ blacklist ระดับปกติ
        for entry in self.blacklist:
            if current_time - entry['timestamp'] < BLACKLIST_DURATION_SECONDS:
                x1, y1, x2, y2 = entry['bbox']
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    return True

                if bbox is not None:
                    if len(bbox) == 4:
                        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                            bx, by, bw, bh = bbox
                            bbox_normalized = (bx, by, bx + bw, by + bh)
                        else:
                            bbox_normalized = bbox
                        if self.calculate_bbox_iou(bbox_normalized, entry['bbox']) > 0:
                            return True

        return False

    def is_in_blacklist(self, center, bbox=None, motion_boxes=None, velocity=None):
        """
        ตรวจสอบว่าตำแหน่งหรือ bbox อยู่ในพื้นที่ blacklist หรือไม่

        Args:
            center: (x, y) center point
            bbox: (x1, y1, x2, y2) หรือ (x, y, w, h) optional bounding box
            motion_boxes: list of motion boxes (optional) สำหรับตรวจสอบ movement
            velocity: velocity ของวัตถุ (optional) สำหรับ target ที่ติดตามอยู่แล้ว

        Returns:
            True ถ้าตำแหน่งหรือ bbox อยู่ในพื้นที่ blacklist และไม่มี movement, False ถ้าไม่อยู่หรือมี movement
        """
        if not STATIONARY_DETECTION_ENABLED:
            return False

        current_time = time.time()

        # ตรวจสอบ permanent blacklist ก่อน (ถ้าเปิดใช้งาน)
        if PERMANENT_BLACKLIST_ENABLED:
            # ลบ permanent blacklist entries ที่หมดอายุ (ถ้ามี duration)
            self.permanent_blacklist = [
                entry for entry in self.permanent_blacklist
                if PERMANENT_BLACKLIST_DURATION_SECONDS == 0 or
                   (current_time - entry['timestamp'] < PERMANENT_BLACKLIST_DURATION_SECONDS)
            ]

            # ตรวจสอบ center point และ bbox overlap
            for entry in self.permanent_blacklist:
                x1, y1, x2, y2 = entry['bbox']

                # ตรวจสอบ center point
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    # อยู่ใน permanent blacklist
                    if BLACKLIST_ALLOW_MOVEMENT:
                        # ตรวจสอบ movement
                        if self.check_movement_in_blacklist(bbox if bbox else (center[0], center[1], 1, 1), motion_boxes, velocity):
                            return False  # มี movement → อนุญาตให้ตรวจจับ
                        else:
                            return True  # ไม่มี movement → กรอง
                    else:
                        return True  # กรองทันที

                # ตรวจสอบ bbox overlap (ถ้ามี)
                if bbox is not None:
                    if len(bbox) == 4:
                        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                            bx, by, bw, bh = bbox
                            bbox_normalized = (bx, by, bx + bw, by + bh)
                        else:
                            bbox_normalized = bbox
                        if self.calculate_bbox_iou(bbox_normalized, entry['bbox']) > 0:
                            # Overlap กับ permanent blacklist
                            if BLACKLIST_ALLOW_MOVEMENT:
                                if self.check_movement_in_blacklist(bbox_normalized, motion_boxes, velocity):
                                    return False
                                else:
                                    return True
                            else:
                                return True

        # ตรวจสอบ blacklist ระดับปกติ
        # ลบ blacklist entries ที่หมดอายุอัตโนมัติ
        self.blacklist = [
            entry for entry in self.blacklist
            if current_time - entry['timestamp'] < BLACKLIST_DURATION_SECONDS
        ]

        # ตรวจสอบ center point และ bbox overlap
        for entry in self.blacklist:
            x1, y1, x2, y2 = entry['bbox']

            # ตรวจสอบ center point
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                # อยู่ใน blacklist
                if BLACKLIST_ALLOW_MOVEMENT:
                    # ตรวจสอบ movement
                    if self.check_movement_in_blacklist(bbox if bbox else (center[0], center[1], 1, 1), motion_boxes, velocity):
                        return False  # มี movement → อนุญาตให้ตรวจจับ
                    else:
                        return True  # ไม่มี movement → กรอง
                else:
                    return True  # กรองทันที

            # ตรวจสอบ bbox overlap (ถ้ามี)
            if bbox is not None:
                if len(bbox) == 4:
                    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                        bx, by, bw, bh = bbox
                        bbox_normalized = (bx, by, bx + bw, by + bh)
                    else:
                        bbox_normalized = bbox
                    if self.calculate_bbox_iou(bbox_normalized, entry['bbox']) > 0:
                        # Overlap กับ blacklist
                        if BLACKLIST_ALLOW_MOVEMENT:
                            if self.check_movement_in_blacklist(bbox_normalized, motion_boxes, velocity):
                                return False
                            else:
                                return True
                        else:
                            return True

        return False  # ไม่อยู่ใน blacklist ใดๆ

    def is_likely_insect_for_target(self, target, frame_counter=None):
        """
        ตรวจสอบว่าอาจเป็นแมลง (บินเร็ว/เส้นตรง/ไม่ต่อเนื่อง) - สำหรับ Hybrid Tracker targets
        ใช้ centers_history ที่มีอยู่แล้ว (ใช้สำหรับ stationary detection)

        Args:
            target: Target dictionary จาก self.targets
            frame_counter: Frame counter (optional, for caching)

        Returns:
            bool: True ถ้าเป็นแมลง, False ถ้าไม่ใช่
        """
        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < 10:
            return False

        # Check cache (performance optimization)
        cache_key = 'insect_check_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 12-15 frames (เพิ่มจาก 8 เพื่อลดการคำนวณ)
                if frame_counter - cache_frame < 12:
                    return cache_result

        centers_history = target['centers_history']
        # ใช้แค่ 15 จุดล่าสุด (ตาม requirement)
        if len(centers_history) > 15:
            centers_history = centers_history[-15:]

        # Early exit: ต้องมี history เพียงพอ
        if len(centers_history) < 10:
            return False

        # คำนวณความเร็ว (velocity) - ใช้จุดแรกและจุดสุดท้าย
        first_frame, first_center = centers_history[0]
        last_frame, last_center = centers_history[-1]

        if first_center is None or last_center is None:
            return False

        frames_diff = last_frame - first_frame
        if frames_diff <= 0:
            return False

        distance = self.get_dist(first_center, last_center)
        velocity = distance / frames_diff

        # Early exit: ตรวจสอบความเร็วก่อน (ไม่เร็วพอ = ไม่ใช่แมลง)
        if velocity <= INSECT_MAX_VELOCITY:
            result = False
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return False

        # คำนวณความตรง (straightness) - ใช้ path_straightness calculation
        # ใช้วิธีเดียวกับ TrackedObject.is_likely_insect()
        if len(centers_history) < 5:
            return False

        # คำนวณ straightness จาก path
        total_distance = 0.0
        direct_distance = self.get_dist(first_center, last_center)

        for i in range(1, len(centers_history)):
            _, prev_center = centers_history[i-1]
            _, curr_center = centers_history[i]
            if prev_center and curr_center:
                segment_dist = self.get_dist(prev_center, curr_center)
                total_distance += segment_dist

        if total_distance == 0:
            return False

        path_straightness = direct_distance / total_distance if total_distance > 0 else 0.0

        # Early exit: ไม่ตรงพอ = ไม่ใช่แมลง
        if path_straightness < INSECT_MIN_STRAIGHTNESS:
            result = False
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return False

        # คำนวณความต่อเนื่อง (continuity) - ใช้ gap ratio
        distances = []
        for i in range(1, len(centers_history)):
            _, prev_center = centers_history[i-1]
            _, curr_center = centers_history[i]
            if prev_center and curr_center:
                dist = self.get_dist(prev_center, curr_center)
                distances.append(dist)

        if len(distances) == 0:
            return False

        total_distance_sum = sum(distances)
        avg_distance = total_distance_sum / len(distances) if len(distances) > 0 else 0
        max_gap = max(distances) if distances else 0.0

        # ความต่อเนื่อง = 1 - (max_gap / avg_distance) ถ้า gap ใหญ่ = ไม่ต่อเนื่อง
        if avg_distance > 0:
            continuity = 1.0 - min(1.0, max_gap / (avg_distance * 2.0))
            if continuity > INSECT_MAX_CONTINUITY:
                # ต่อเนื่องเกินไป = ไม่ใช่แมลง
                result = False
                if cache_key not in target:
                    target[cache_key] = (result, frame_counter)
                return False

        # ผ่านทุกเงื่อนไข = เป็นแมลง
        result = True
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)
        return True

    def is_likely_star_by_path(self, target, frame_counter=None):
        """
        ตรวจสอบว่าอาจเป็นดาว (กระจุกตัวมาก แทบไม่มีเส้น) - สำหรับ Tiny Objects
        ใช้ centers_history ที่มีอยู่แล้ว

        Args:
            target: Target dictionary จาก self.targets
            frame_counter: Frame counter (optional, for caching)

        Returns:
            bool: True ถ้าเป็นดาว, False ถ้าไม่ใช่
        """
        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < 10:
            return False

        # Check cache (performance optimization)
        cache_key = 'star_check_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 12-15 frames (เหมือน insect check)
                if frame_counter - cache_frame < 12:
                    return cache_result

        centers_history = target['centers_history']
        # ใช้แค่ 20 จุดล่าสุด (ไม่ใช่ทั้งหมด) เพื่อประหยัด CPU
        if len(centers_history) > 20:
            centers_history = centers_history[-20:]

        # Early exit: ต้องมี history เพียงพอ
        if len(centers_history) < 10:
            result = False
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return False

        # คำนวณ clustering ratio (ใช้ O(n) แทน O(n²))
        # เปรียบเทียบ: ระยะห่างเฉลี่ยจากจุดแรก vs ระยะห่างจากจุดแรกถึงจุดสุดท้าย
        first_frame, first_center = centers_history[0]
        last_frame, last_center = centers_history[-1]

        if first_center is None or last_center is None:
            result = False
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return False

        # ระยะห่างจากจุดแรกถึงจุดสุดท้าย (direct distance)
        direct_distance = self.get_dist(first_center, last_center)

        # ระยะห่างเฉลี่ยจากจุดแรก (average distance from first point)
        total_distance_from_first = 0.0
        valid_points = 0
        for _, center in centers_history:
            if center is not None:
                total_distance_from_first += self.get_dist(first_center, center)
                valid_points += 1

        if valid_points == 0:
            result = False
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return False

        avg_distance_from_first = total_distance_from_first / valid_points

        # Clustering ratio = direct_distance / avg_distance_from_first
        # ถ้า ratio ต่ำ = กระจุกตัวมาก (เป็นดาว)
        # ถ้า ratio สูง = มีการเคลื่อนที่ (ไม่ใช่ดาว)
        if avg_distance_from_first == 0:
            # ถ้า avg_distance = 0 แสดงว่าทุกจุดอยู่ที่เดียวกัน = เป็นดาวแน่นอน
            result = True
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return True

        clustering_ratio = direct_distance / avg_distance_from_first

        # ถ้า clustering_ratio < threshold = กระจุกตัวมาก = เป็นดาว
        result = clustering_ratio < TINY_OBJECT_STAR_CLUSTERING_THRESHOLD

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def is_likely_airplane_for_target(self, target, frame_counter=None):
        """
        ตรวจสอบว่าอาจเป็นเครื่องบิน (เส้นตรงมาก) - สำหรับ Tiny Objects
        ใช้ centers_history ที่มีอยู่แล้ว

        Args:
            target: Target dictionary จาก self.targets
            frame_counter: Frame counter (optional, for caching)

        Returns:
            tuple: (is_airplane: bool, confidence_score: float)
        """
        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < 10:
            return (False, 0.0)

        # Check cache (performance optimization)
        cache_key = 'airplane_check_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 12-15 frames (เหมือน insect check)
                if frame_counter - cache_frame < 12:
                    return cache_result

        centers_history = target['centers_history']
        # ต้องมีอย่างน้อย TINY_OBJECT_AIRPLANE_PATH_MIN_POINTS จุด (ใช้ path ยาวขึ้น)
        if len(centers_history) < TINY_OBJECT_AIRPLANE_PATH_MIN_POINTS:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # ใช้ path ทั้งหมด (ไม่ตัดเหลือ 20 จุด) เพื่อให้เห็นความแตกต่างชัดเจน

        # คำนวณ path straightness
        first_frame, first_center = centers_history[0]
        last_frame, last_center = centers_history[-1]

        if first_center is None or last_center is None:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # คำนวณ total distance (ระยะทางรวมของ path)
        total_distance = 0.0
        for i in range(1, len(centers_history)):
            _, prev_center = centers_history[i-1]
            _, curr_center = centers_history[i]
            if prev_center and curr_center:
                segment_dist = self.get_dist(prev_center, curr_center)
                total_distance += segment_dist

        if total_distance == 0:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # ระยะทางตรงจากจุดแรกถึงจุดสุดท้าย
        direct_distance = self.get_dist(first_center, last_center)

        # Straightness = direct_distance / total_distance
        straightness = direct_distance / total_distance if total_distance > 0 else 0.0

        # ตรวจสอบ size stability (ถ้ามีข้อมูล)
        size_stable = True
        size_confidence_boost = 0.0
        if 'size_history' in target and len(target['size_history']) >= TINY_OBJECT_AIRPLANE_SIZE_HISTORY_MIN_POINTS:
            # คำนวณ coefficient of variation (CV) แทน variance เพื่อ normalize
            sizes = [s[1] for s in target['size_history']]  # extract areas
            mean_size = sum(sizes) / len(sizes)
            if mean_size > 0:
                variance = sum((s - mean_size)**2 for s in sizes) / len(sizes)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_size
                size_stable = cv < TINY_OBJECT_AIRPLANE_SIZE_STABILITY_THRESHOLD
                # ถ้า size stable → confidence เพิ่มขึ้น
                if size_stable:
                    size_confidence_boost = 0.1 * (1.0 - cv / TINY_OBJECT_AIRPLANE_SIZE_STABILITY_THRESHOLD)
        else:
            # ไม่มีข้อมูลเพียงพอ → ไม่สามารถยืนยันได้
            size_stable = False

        # ตรวจสอบว่าเป็นเครื่องบินหรือไม่ (ต้องทั้ง straightness และ size stable)
        is_airplane = (straightness >= TINY_OBJECT_AIRPLANE_STRAIGHTNESS_THRESHOLD) and size_stable
        confidence = min(1.0, straightness + size_confidence_boost)  # confidence = straightness + size boost

        result = (is_airplane, confidence)

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def calculate_tiny_object_path_confidence(self, target, frame_counter=None):
        """
        คำนวณ confidence จาก path analysis สำหรับ tiny objects

        Args:
            target: Target dictionary จาก self.targets
            frame_counter: Frame counter (optional, for caching)

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < 10:
            return 0.0

        # Check cache (performance optimization)
        cache_key = 'path_confidence_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 6 frames (interval-based checking)
                if frame_counter - cache_frame < 6:
                    return cache_result

        centers_history = target['centers_history']
        if len(centers_history) < 10:
            result = 0.0
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # 1. Path Smoothness: คำนวณจากความเปลี่ยนแปลงของ direction
        smoothness = 1.0
        if len(centers_history) >= 3:
            directions = []
            for i in range(1, len(centers_history)):
                _, prev_center = centers_history[i-1]
                _, curr_center = centers_history[i]
                if prev_center and curr_center:
                    dx = curr_center[0] - prev_center[0]
                    dy = curr_center[1] - prev_center[1]
                    if dx != 0 or dy != 0:
                        direction = math.atan2(dy, dx)
                        directions.append(direction)

            if len(directions) >= 2:
                # คำนวณความเปลี่ยนแปลงของ direction
                direction_changes = []
                for i in range(1, len(directions)):
                    change = abs(directions[i] - directions[i-1])
                    # Normalize to [0, pi]
                    if change > math.pi:
                        change = 2 * math.pi - change
                    direction_changes.append(change)

                if direction_changes:
                    avg_change = sum(direction_changes) / len(direction_changes)
                    # Smoothness = 1.0 - normalized_change (0 = ไม่เปลี่ยนทิศ, pi = เปลี่ยนทิศมาก)
                    smoothness = max(0.0, 1.0 - (avg_change / math.pi))

        # 2. Direction Consistency: คำนวณจากความสม่ำเสมอของทิศทาง
        consistency = 1.0
        if len(centers_history) >= 5:
            first_frame, first_center = centers_history[0]
            last_frame, last_center = centers_history[-1]
            if first_center and last_center:
                # ทิศทางรวม
                total_dx = last_center[0] - first_center[0]
                total_dy = last_center[1] - first_center[1]
                total_distance = math.sqrt(total_dx**2 + total_dy**2)

                # ระยะทางรวมของ path
                path_distance = 0.0
                for i in range(1, len(centers_history)):
                    _, prev_center = centers_history[i-1]
                    _, curr_center = centers_history[i]
                    if prev_center and curr_center:
                        path_distance += self.get_dist(prev_center, curr_center)

                # Consistency = direct_distance / path_distance (เหมือน straightness)
                if path_distance > 0:
                    consistency = total_distance / path_distance

        # 3. Path Curvature: คำนวณจากความโค้งของ path
        curvature_score = 0.5  # Default (กลาง)
        if len(centers_history) >= 5:
            # คำนวณ curvature จากความเปลี่ยนแปลงของ direction
            if len(directions) >= 3:
                curvature_changes = []
                for i in range(1, len(directions)):
                    change = abs(directions[i] - directions[i-1])
                    if change > math.pi:
                        change = 2 * math.pi - change
                    curvature_changes.append(change)

                if curvature_changes:
                    avg_curvature = sum(curvature_changes) / len(curvature_changes)
                    # Curvature score: มี curvature = เป็นโดรน (score สูง)
                    # Normalize: 0 = ไม่โค้ง, pi = โค้งมาก
                    curvature_score = min(1.0, avg_curvature / (math.pi / 4))  # pi/4 = 45 degrees

        # 4. Velocity Consistency: คำนวณจากความสม่ำเสมอของความเร็ว
        velocity_consistency = 1.0
        if len(centers_history) >= 5:
            velocities = []
            for i in range(1, len(centers_history)):
                prev_frame, prev_center = centers_history[i-1]
                curr_frame, curr_center = centers_history[i]
                if prev_center and curr_center and curr_frame > prev_frame:
                    distance = self.get_dist(prev_center, curr_center)
                    frame_diff = curr_frame - prev_frame
                    if frame_diff > 0:
                        velocity = distance / frame_diff
                        velocities.append(velocity)

            if len(velocities) >= 3:
                # คำนวณ coefficient of variation ของ velocity
                mean_velocity = sum(velocities) / len(velocities)
                if mean_velocity > 0:
                    variance = sum((v - mean_velocity)**2 for v in velocities) / len(velocities)
                    std_dev = math.sqrt(variance)
                    cv = std_dev / mean_velocity
                    # Velocity consistency = 1.0 - normalized CV (ต่ำ = สม่ำเสมอ)
                    velocity_consistency = max(0.0, 1.0 - min(1.0, cv))

        # 5. Total Distance Traveled: ตรวจสอบว่าเคลื่อนที่จริง
        distance_score = 0.0
        if len(centers_history) >= 2:
            first_frame, first_center = centers_history[0]
            last_frame, last_center = centers_history[-1]
            if first_center and last_center:
                total_distance = self.get_dist(first_center, last_center)
                # Normalize distance (ใช้ frame diagonal เป็น reference)
                frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
                if frame_diagonal > 0:
                    # Distance score: 0.0 = ไม่เคลื่อนที่, 1.0 = เคลื่อนที่มาก
                    distance_score = min(1.0, total_distance / (frame_diagonal * 0.1))  # 10% ของ frame diagonal

        # Weighted average ของ metrics ทั้งหมด
        weights = {
            'smoothness': 0.2,
            'consistency': 0.25,
            'curvature': 0.25,
            'velocity': 0.15,
            'distance': 0.15
        }

        confidence = (
            smoothness * weights['smoothness'] +
            consistency * weights['consistency'] +
            curvature_score * weights['curvature'] +
            velocity_consistency * weights['velocity'] +
            distance_score * weights['distance']
        )

        result = min(1.0, max(0.0, confidence))

        # เพิ่มการตรวจสอบ path continuity และ size stability (NEW!)
        continuity_boost = 0.0
        if DRONE_PATH_CONTINUITY_ENABLED:
            # ดึง motion box area จาก last_rect
            motion_box_area = None
            if 'last_rect' in target and target['last_rect']:
                x, y, w, h = target['last_rect']
                motion_box_area = w * h

            has_continuity, continuity_score = self.calculate_path_continuity(
                target, frame_counter, motion_box_area
            )
            if has_continuity:
                continuity_boost = continuity_score * DRONE_PATH_CONTINUITY_BOOST

        size_stability_boost = 0.0
        if DRONE_SIZE_STABILITY_ENABLED:
            is_stable, stability_score = self.calculate_size_stability(target, frame_counter)
            if is_stable:
                size_stability_boost = stability_score * DRONE_SIZE_STABILITY_BOOST

        # เพิ่ม boosts เข้าไปใน confidence
        result += continuity_boost
        result += size_stability_boost
        result = min(1.0, max(0.0, result))

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def check_background_noise_in_roi(self, target, frame_counter=None, size_category=None):
        """
        ตรวจสอบว่า background ใน ROI area มี noise/motion ยุ่งเหยิงหรือไม่
        ตรวจสอบเฉพาะใน ROI area ของ target (ไม่ใช่ทั้งเฟรม)
        ถ้ามี noise/motion เยอะใน ROI → ไม่ควรใช้ path continuity/size stability

        Args:
            target: Target dictionary
            frame_counter: Frame counter (optional, for caching)
            size_category: Optional - 'TINY', 'SMALL', 'MEDIUM', 'LARGE' (ใช้สำหรับปรับ threshold)

        Returns:
            tuple: (is_low_noise: bool, noise_level: float, motion_boxes_in_roi: int)
                - is_low_noise: True ถ้า background noise ใน ROI ต่ำ (ใช้ path continuity/size stability ได้)
                - noise_level: ค่า noise level ใน ROI (0.0-1.0, ต่ำ=noise น้อย)
                - motion_boxes_in_roi: จำนวน motion boxes ใน ROI
        """
        # Early exit: ถ้าปิดการตรวจสอบ
        if not DRONE_BACKGROUND_NOISE_CHECK_ENABLED:
            return (True, 0.0, 0)  # ถ้าปิด → ถือว่า noise ต่ำ (ใช้ได้)

        # Early exit: ต้องมี ROI box
        if 'roi_box' not in target or not target['roi_box']:
            return (True, 0.0, 0)  # ถ้าไม่มี ROI → ถือว่า noise ต่ำ (ใช้ได้)

        # Check cache (performance optimization)
        cache_key = 'background_noise_roi_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 6 frames
                if frame_counter - cache_frame < 6:
                    return cache_result

        noise_level = 0.0
        motion_boxes_in_roi = 0

        roi_x1, roi_y1, roi_x2, roi_y2 = target['roi_box']
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        roi_center_x = roi_x1 + roi_w // 2
        roi_center_y = roi_y1 + roi_h // 2

        # 1. ตรวจสอบ grid noise ใน ROI area
        if hasattr(self, 'grid_system') and self.grid_system is not None:
            try:
                avg_noise, max_noise, min_noise = self.grid_system.get_grid_noise_level(
                    roi_center_x - roi_w // 2,
                    roi_center_y - roi_h // 2,
                    roi_w,
                    roi_h,
                    frame_counter if frame_counter is not None else 0,
                    cache_frames=6
                )
                noise_level = avg_noise
            except Exception:
                # ถ้าเกิด error → ถือว่า noise ต่ำ (ไม่บล็อก)
                noise_level = 0.0

        # 2. ตรวจสอบจำนวน motion boxes ใน ROI area
        if hasattr(self, 'motion_boxes') and self.motion_boxes is not None:
            # ตรวจสอบว่า motion box ของ target เองอยู่ใน ROI หรือไม่ (เพื่อไม่นับตัวเอง)
            target_motion_center_x = None
            target_motion_center_y = None
            if 'last_rect' in target and target['last_rect']:
                tx, ty, tw, th = target['last_rect']
                target_motion_center_x = tx + tw // 2
                target_motion_center_y = ty + th // 2

            for motion_box in self.motion_boxes:
                mx, my, mw, mh = motion_box
                motion_center_x = mx + mw // 2
                motion_center_y = my + mh // 2

                # ข้าม motion box ของ target เอง (ไม่นับตัวเอง)
                if target_motion_center_x is not None and target_motion_center_y is not None:
                    if (abs(motion_center_x - target_motion_center_x) < 5 and
                        abs(motion_center_y - target_motion_center_y) < 5):
                        continue  # ข้าม motion box ของตัวเอง

                # ตรวจสอบว่า motion box อยู่ใน ROI หรือไม่
                if (roi_x1 <= motion_center_x <= roi_x2 and
                    roi_y1 <= motion_center_y <= roi_y2):
                    motion_boxes_in_roi += 1

        # ตรวจสอบว่า background noise/motion ใน ROI ต่ำหรือไม่
        is_low_noise = True

        # เงื่อนไข 1: grid noise ใน ROI ต้องต่ำกว่า threshold
        if noise_level > DRONE_BACKGROUND_NOISE_THRESHOLD:
            is_low_noise = False

        # เงื่อนไข 2: จำนวน motion boxes ใน ROI ต้องไม่เกิน threshold
        # สำหรับ TINY objects: ยืดหยุ่นขึ้น (ยอมรับ motion boxes มากขึ้น)
        motion_threshold = DRONE_BACKGROUND_MOTION_IN_ROI_THRESHOLD
        if size_category == 'TINY':
            motion_threshold = DRONE_BACKGROUND_MOTION_IN_ROI_THRESHOLD * 2  # ยืดหยุ่น 2 เท่า

        if motion_boxes_in_roi > motion_threshold:
            is_low_noise = False

        result = (is_low_noise, noise_level, motion_boxes_in_roi)

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def _check_motion_boxes_stationary_in_roi(self, target, target_id, roi_box, w_orig, h_orig, current_frame):
        """
        ตรวจสอบว่า motion boxes ใน ROI ขยับหรือไม่
        ใช้ interval-based checking และ caching เพื่อประหยัด CPU

        Args:
            target: Target dictionary
            target_id: Target ID
            roi_box: ROI box (x1, y1, x2, y2)
            w_orig: Frame width
            h_orig: Frame height
            current_frame: Current frame counter

        Returns:
            (is_stationary: bool, stationary_frames: int)
        """
        # Interval-based checking (ไม่ตรวจทุก frame)
        check_interval = MOTION_STATIONARY_CHECK_INTERVAL
        last_check_frame = target.get('motion_stationary_last_check_frame', -check_interval)

        if current_frame - last_check_frame < check_interval:
            # ใช้ cached result
            return (
                target.get('motion_stationary_cached', False),
                target.get('motion_stationary_frames', 0)
            )

        # ตรวจสอบ motion boxes
        is_stationary = False
        stationary_frames = target.get('motion_stationary_frames', 0)

        if hasattr(self, 'motion_boxes') and self.motion_boxes:
            # เก็บตำแหน่ง motion boxes ใน ROI ปัจจุบัน
            current_motion_boxes_in_roi = []
            rx1, ry1, rx2, ry2 = roi_box

            for motion_box in self.motion_boxes:
                mx, my, mw, mh = motion_box
                motion_center_x = mx + mw // 2
                motion_center_y = my + mh // 2

                # ตรวจสอบว่า motion box อยู่ใน ROI หรือไม่
                if (rx1 <= motion_center_x <= rx2 and
                    ry1 <= motion_center_y <= ry2):
                    # Quantize เพื่อลดการคำนวณ (ปัดเศษเป็น 5 pixels)
                    quantized_x = (motion_center_x // 5) * 5
                    quantized_y = (motion_center_y // 5) * 5
                    current_motion_boxes_in_roi.append((quantized_x, quantized_y, mw, mh))

            # เปรียบเทียบกับเฟรมก่อนหน้า
            prev_motion_boxes_key = f'prev_motion_boxes_{target_id}'
            prev_motion_boxes = target.get(prev_motion_boxes_key, [])

            if len(prev_motion_boxes) > 0 and len(current_motion_boxes_in_roi) > 0:
                # ตรวจสอบว่า motion boxes ขยับหรือไม่
                max_movement = 0.0
                frame_diagonal = math.sqrt(w_orig**2 + h_orig**2)
                movement_threshold = frame_diagonal * MOTION_STATIONARY_MOVEMENT_THRESHOLD_RATIO

                for curr_box in current_motion_boxes_in_roi:
                    curr_x, curr_y, curr_w, curr_h = curr_box
                    min_dist = float('inf')

                    for prev_box in prev_motion_boxes:
                        prev_x, prev_y, prev_w, prev_h = prev_box
                        # เปรียบเทียบขนาดก่อน (ถ้าขนาดต่างกันมาก → ไม่ใช่ตัวเดียวกัน)
                        size_diff = abs(curr_w - prev_w) + abs(curr_h - prev_h)
                        if size_diff > max(curr_w, prev_w) * 0.5:  # ขนาดต่างกันมากกว่า 50%
                            continue

                        # คำนวณระยะทาง (ใช้ quantized coordinates)
                        dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                        min_dist = min(min_dist, dist)

                    max_movement = max(max_movement, min_dist)

                # ถ้า motion boxes ไม่ขยับเลย (หรือขยับน้อยมาก)
                if max_movement < movement_threshold:
                    is_stationary = True
                    stationary_frames += 1
                else:
                    # Reset counter ถ้า motion boxes ขยับ
                    stationary_frames = 0
            elif len(current_motion_boxes_in_roi) == 0:
                # ถ้าไม่มี motion boxes ใน ROI เลย → ถือว่า stationary
                is_stationary = True
                stationary_frames += 1

            # อัปเดตตำแหน่ง motion boxes สำหรับเฟรมถัดไป
            target[prev_motion_boxes_key] = current_motion_boxes_in_roi
        else:
            # ถ้าไม่มี motion boxes เลย → ถือว่า stationary
            is_stationary = True
            stationary_frames += 1

        # Cache result
        target['motion_stationary_cached'] = is_stationary
        target['motion_stationary_frames'] = stationary_frames
        target['motion_stationary_last_check_frame'] = current_frame

        return (is_stationary, stationary_frames)

    def calculate_path_continuity(self, target, frame_counter=None, motion_box_area=None):
        """
        ตรวจสอบความต่อเนื่องของ path (การเคลื่อนที่ต่อเนื่อง ไม่กระโดด)
        ใช้ adaptive thresholds ตามขนาดกล่อง motion
        ต้องตรวจสอบ background noise ใน ROI ก่อน (ถ้ามี noise เยอะ → ไม่ใช้)

        Args:
            target: Target dictionary
            frame_counter: Frame counter (optional, for caching)
            motion_box_area: พื้นที่กล่อง motion (pixels²) - ใช้สำหรับ adaptive threshold

        Returns:
            tuple: (is_continuous: bool, continuity_score: float)
        """
        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < DRONE_PATH_CONTINUITY_MIN_POINTS:
            return (False, 0.0)

        # ตรวจสอบ background noise ใน ROI ก่อน (NEW!)
        size_category = target.get('object_size_category')
        is_low_noise, noise_level, motion_boxes_in_roi = self.check_background_noise_in_roi(target, frame_counter, size_category)

        # สำหรับ TINY objects: ยืดหยุ่นขึ้น (ยอมรับ noise สูงขึ้น)
        if size_category == 'TINY':
            # ถ้า noise ไม่สูงมากเกินไป → ยังใช้ path continuity
            if noise_level <= DRONE_BACKGROUND_NOISE_THRESHOLD * 1.5:  # ยืดหยุ่น 1.5 เท่า
                is_low_noise = True
            # หรือถ้า motion boxes ใน ROI ไม่มากเกินไป
            if motion_boxes_in_roi <= DRONE_BACKGROUND_MOTION_IN_ROI_THRESHOLD * 2:  # ยืดหยุ่น 2 เท่า
                is_low_noise = True

        if not is_low_noise:
            # ถ้ามี noise/motion ยุ่งเหยิงใน ROI → ไม่ใช้ path continuity
            return (False, 0.0)

        # Check cache (performance optimization)
        cache_key = 'path_continuity_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 6 frames
                if frame_counter - cache_frame < 6:
                    return cache_result

        centers_history = target['centers_history']

        # 1. ตรวจสอบ gap (ระยะห่างระหว่างจุด)
        distances = []
        for i in range(1, len(centers_history)):
            _, prev_center = centers_history[i-1]
            _, curr_center = centers_history[i]
            if prev_center and curr_center:
                distance = self.get_dist(prev_center, curr_center)
                distances.append(distance)

        if not distances:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # คำนวณ gap statistics
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)

        # Adaptive threshold ตามขนาดกล่อง motion
        if motion_box_area is None and 'last_rect' in target and target['last_rect']:
            x, y, w, h = target['last_rect']
            motion_box_area = w * h

        # ตรวจสอบ size category
        size_category = target.get('object_size_category')

        if motion_box_area is not None:
            if motion_box_area >= DRONE_MOTION_BOX_SIZE_THRESHOLD_LARGE:
                max_gap_ratio = DRONE_ADAPTIVE_GAP_RATIO_LARGE
                smooth_threshold = DRONE_ADAPTIVE_SMOOTH_THRESHOLD_LARGE
            elif motion_box_area <= DRONE_MOTION_BOX_SIZE_THRESHOLD_SMALL:
                max_gap_ratio = DRONE_ADAPTIVE_GAP_RATIO_SMALL
                smooth_threshold = DRONE_ADAPTIVE_SMOOTH_THRESHOLD_SMALL

                # สำหรับ TINY objects: ยืดหยุ่นขึ้นสำหรับการเคลื่อนที่แบบ "กระดึบๆ"
                if size_category == 'TINY':
                    # ลด smooth threshold (ยอมรับการเปลี่ยนทิศทางมากขึ้น)
                    smooth_threshold = max(0.5, smooth_threshold - 0.2)  # ลดจาก 0.8 เป็น 0.6
                    # เพิ่ม gap ratio (ยอมรับ gap ที่ใหญ่ขึ้น)
                    max_gap_ratio = min(3.0, max_gap_ratio + 0.5)  # เพิ่มจาก 1.5 เป็น 2.0
            else:
                # Interpolate ระหว่าง large และ small
                ratio = (motion_box_area - DRONE_MOTION_BOX_SIZE_THRESHOLD_SMALL) / \
                        (DRONE_MOTION_BOX_SIZE_THRESHOLD_LARGE - DRONE_MOTION_BOX_SIZE_THRESHOLD_SMALL)
                max_gap_ratio = DRONE_ADAPTIVE_GAP_RATIO_SMALL + \
                               (DRONE_ADAPTIVE_GAP_RATIO_LARGE - DRONE_ADAPTIVE_GAP_RATIO_SMALL) * ratio
                smooth_threshold = DRONE_ADAPTIVE_SMOOTH_THRESHOLD_SMALL + \
                                  (DRONE_ADAPTIVE_SMOOTH_THRESHOLD_LARGE - DRONE_ADAPTIVE_SMOOTH_THRESHOLD_SMALL) * (1 - ratio)
        else:
            max_gap_ratio = DRONE_PATH_MAX_GAP_RATIO
            smooth_threshold = DRONE_PATH_SMOOTH_TRANSITION_THRESHOLD

            # สำหรับ TINY objects: ยืดหยุ่นขึ้น
            if size_category == 'TINY':
                smooth_threshold = max(0.5, smooth_threshold - 0.2)
                max_gap_ratio = min(3.0, max_gap_ratio + 0.5)

            # สำหรับ TINY objects: ยืดหยุ่นขึ้น
            if size_category == 'TINY':
                smooth_threshold = max(0.5, smooth_threshold - 0.2)
                max_gap_ratio = min(3.0, max_gap_ratio + 0.5)

        # ตรวจสอบว่า gap ไม่ใหญ่เกินไป
        gap_ratio = max_distance / avg_distance if avg_distance > 0 else float('inf')
        is_continuous = gap_ratio <= max_gap_ratio

        # 2. ตรวจสอบ smooth transitions (การเปลี่ยนทิศทางที่ราบเรียบ)
        smooth_transitions = 0
        total_transitions = 0

        if len(centers_history) >= 3:
            for i in range(1, len(centers_history) - 1):
                _, prev_center = centers_history[i-1]
                _, curr_center = centers_history[i]
                _, next_center = centers_history[i+1]

                if prev_center and curr_center and next_center:
                    # Vector 1: จากจุด i-1 ถึง i
                    dx1 = curr_center[0] - prev_center[0]
                    dy1 = curr_center[1] - prev_center[1]

                    # Vector 2: จากจุด i ถึง i+1
                    dx2 = next_center[0] - curr_center[0]
                    dy2 = next_center[1] - curr_center[1]

                    # คำนวณมุมระหว่าง vectors
                    dot = dx1 * dx2 + dy1 * dy2
                    mag1 = math.sqrt(dx1**2 + dy1**2)
                    mag2 = math.sqrt(dx2**2 + dy2**2)

                    if mag1 > 0 and mag2 > 0:
                        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                        angle = math.acos(cos_angle) * 180 / math.pi  # degrees

                        # Smooth transition = มุมไม่มาก (เปลี่ยนทิศทางทีละน้อย)
                        if angle <= 45.0:
                            smooth_transitions += 1
                        total_transitions += 1

        # คำนวณ smooth transition ratio
        smooth_ratio = smooth_transitions / total_transitions if total_transitions > 0 else 0.0
        is_smooth = smooth_ratio >= smooth_threshold

        # 3. ตรวจสอบความสม่ำเสมอของ distance (ไม่กระโดด)
        if len(distances) >= 3:
            mean_distance = sum(distances) / len(distances)
            variance = sum((d - mean_distance)**2 for d in distances) / len(distances)
            std_dev = math.sqrt(variance)
            cv_distance = std_dev / mean_distance if mean_distance > 0 else 1.0

            # CV ต่ำ = สม่ำเสมอ = ต่อเนื่อง
            distance_consistency = max(0.0, 1.0 - min(1.0, cv_distance))

            # สำหรับ TINY objects: ยืดหยุ่นขึ้น (ยอมรับ CV สูงขึ้น)
            if size_category == 'TINY':
                # ถ้า CV สูง แต่ยังไม่เกิน 1.0 → ยังให้คะแนนบางส่วน
                if cv_distance <= 1.0:
                    distance_consistency = max(0.3, distance_consistency)  # อย่างน้อย 0.3
        else:
            distance_consistency = 0.5

        # คำนวณ continuity score
        continuity_score = 0.0

        # Gap score
        if is_continuous:
            gap_score = 1.0 - (gap_ratio - 1.0) / (max_gap_ratio - 1.0) if max_gap_ratio > 1.0 else 1.0
            gap_score = max(0.0, min(1.0, gap_score))
            continuity_score += gap_score * 0.4

        # Smooth transition score
        if is_smooth:
            continuity_score += smooth_ratio * 0.3

        # Distance consistency score
        continuity_score += distance_consistency * 0.3

        continuity_score = min(1.0, continuity_score)

        # Path ต่อเนื่อง = gap ไม่ใหญ่ + smooth transitions + distance consistent
        # สำหรับ TINY objects: ยืดหยุ่นขึ้น
        if size_category == 'TINY':
            # ถ้า continuity_score สูงพอ → ถือว่าต่อเนื่องแม้ smooth หรือ distance consistency ต่ำ
            has_continuity = (
                (is_continuous and is_smooth and distance_consistency >= 0.5) or
                (continuity_score >= 0.5 and is_continuous)  # ยืดหยุ่น: ถ้า score สูงและ gap OK → ต่อเนื่อง
            )
        else:
            has_continuity = is_continuous and is_smooth and distance_consistency >= 0.5

        result = (has_continuity, continuity_score)

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def calculate_size_stability(self, target, frame_counter=None):
        """
        ตรวจสอบความคงที่ของขนาดกล่อง (size stability)
        โดรนจะมีขนาดกล่องค่อนข้างคงที่ แม้จะเคลื่อนที่
        ต้องตรวจสอบ background noise ใน ROI ก่อน (ถ้ามี noise เยอะ → ไม่ใช้)

        Args:
            target: Target dictionary
            frame_counter: Frame counter (optional, for caching)

        Returns:
            tuple: (is_stable: bool, stability_score: float)
        """
        # Early exit: ต้องมี size_history
        if 'size_history' not in target or len(target['size_history']) < DRONE_SIZE_STABILITY_MIN_POINTS:
            return (False, 0.0)

        # ตรวจสอบ background noise ใน ROI ก่อน (NEW!)
        size_category = target.get('object_size_category')
        is_low_noise, noise_level, motion_boxes_in_roi = self.check_background_noise_in_roi(target, frame_counter, size_category)
        if not is_low_noise:
            # ถ้ามี noise/motion ยุ่งเหยิงใน ROI → ไม่ใช้ size stability
            return (False, 0.0)

        # Check cache (performance optimization)
        cache_key = 'size_stability_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 6 frames
                if frame_counter - cache_frame < 6:
                    return cache_result

        size_history = target['size_history']

        # คำนวณ area ของแต่ละ size
        areas = [s[1] for s in size_history[-DRONE_SIZE_STABILITY_MIN_POINTS:]]

        if len(areas) < 2:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        # คำนวณ CV (coefficient of variation)
        mean_area = sum(areas) / len(areas)
        if mean_area == 0:
            result = (False, 0.0)
            if cache_key not in target:
                target[cache_key] = (result, frame_counter)
            return result

        variance = sum((a - mean_area)**2 for a in areas) / len(areas)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_area

        # ใช้ size-adaptive CV threshold (สำหรับ TINY objects: ยอมรับการเปลี่ยนแปลงขนาดสูง)
        size_category = target.get('object_size_category')
        if size_category == 'TINY':
            cv_threshold = TINY_SIZE_STABILITY_CV_THRESHOLD
        elif size_category == 'SMALL':
            cv_threshold = SMALL_SIZE_STABILITY_CV_THRESHOLD
        elif size_category == 'MEDIUM':
            cv_threshold = MEDIUM_SIZE_STABILITY_CV_THRESHOLD
        elif size_category == 'LARGE':
            cv_threshold = LARGE_SIZE_STABILITY_CV_THRESHOLD
        else:
            cv_threshold = DRONE_SIZE_STABILITY_CV_THRESHOLD  # Default

        # CV ต่ำ = stable มาก
        is_stable = cv <= cv_threshold

        # คำนวณ stability score (CV ต่ำ = score สูง)
        if cv_threshold > 0:
            stability_score = max(0.0, 1.0 - (cv / cv_threshold))
        else:
            stability_score = 1.0 if cv == 0 else 0.0

        result = (is_stable, stability_score)

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)

        return result

    def update_path_history(self, target, center, frame_counter, status=None):
        """
        อัปเดต path history แบบ optimize memory และ performance
        - ใช้ cached size category (ไม่คำนวณซ้ำ)
        - Early exit สำหรับกรณีที่ไม่ต้องเก็บ
        - Fast conditional checks
        - สำหรับ motion-only targets: เก็บทุกเฟรม (ถ้าเปิด MOTION_ONLY_STORE_PATH_EVERY_FRAME)

        Args:
            target: Target dictionary
            center: (x, y) center point
            frame_counter: Frame counter
            status: Status ของ target (optional, ถ้า None ใช้จาก target)
        """
        # สำหรับ motion-only targets: เก็บทุกเฟรม (ถ้าเปิด MOTION_ONLY_STORE_PATH_EVERY_FRAME)
        is_motion_only = target.get('is_motion_only', False)
        if is_motion_only and MOTION_ONLY_STORE_PATH_EVERY_FRAME:
            if 'centers_history' not in target:
                target['centers_history'] = []
            target['centers_history'].append((frame_counter, center))
            # จำกัดความยาว (performance)
            from config import MOTION_ONLY_PATH_HISTORY_MAX_FRAMES
            if len(target['centers_history']) > MOTION_ONLY_PATH_HISTORY_MAX_FRAMES:
                target['centers_history'] = target['centers_history'][-MOTION_ONLY_PATH_HISTORY_MAX_FRAMES:]
            return

        # Early exit: ถ้าปิดการ optimize
        if not DRONE_PATH_HISTORY_OPTIMIZATION_ENABLED:
            if 'centers_history' not in target:
                target['centers_history'] = []
            target['centers_history'].append((frame_counter, center))
            return

        # ใช้ cached size category (ไม่คำนวณซ้ำ)
        size_category = target.get('object_size_category')
        if size_category is None:
            # ถ้ายังไม่มี cache → คำนวณครั้งเดียว
            size_category = self.get_object_size_category_for_target(target)
            if size_category:
                target['object_size_category'] = size_category

        if status is None:
            status = target.get('status', 'GREEN')

        # Early exit: GREEN และปิดการเก็บ
        # แต่ถ้า path-based ORANGE detection เปิดอยู่ → ต้องเก็บเพื่อตรวจสอบ
        if status == 'GREEN' and not DRONE_GREEN_PATH_HISTORY_ENABLED:
            # ถ้า path-based ORANGE detection เปิดอยู่ → เก็บต่อไป (จะใช้ interval-based)
            if not DRONE_PATH_BASED_ORANGE_ENABLED:
                return  # ไม่เก็บเลย
            # ถ้าเปิด path-based ORANGE → เก็บต่อไป (ไม่ return)

        # ตรวจสอบว่าควรเก็บหรือไม่ (ตาม status และ size)
        should_store = self._should_store_path_history(status, size_category, frame_counter)

        # Debug logging (เฉพาะเมื่อ DEBUG_MODE เปิดและ status เป็น GREEN)
        if status == 'GREEN' and DEBUG_MODE and not should_store:
            # Log เมื่อไม่เก็บ (เพื่อ debug) - ทุก 60 เฟรมเพื่อไม่กระทบ performance
            if frame_counter % 60 == 0:
                target_id = target.get('id', 'unknown')
                print(f"⚠️ Target {target_id}: Not storing path (size_category={size_category}, "
                      f"GREEN_ENABLED={DRONE_GREEN_PATH_HISTORY_ENABLED}, "
                      f"ORANGE_ENABLED={DRONE_PATH_BASED_ORANGE_ENABLED})")

        if should_store:
            if 'centers_history' not in target:
                target['centers_history'] = []
            target['centers_history'].append((frame_counter, center))

            # จำกัดความยาว (ใช้ slicing - เร็วมาก)
            max_history = self._get_path_history_limit(size_category, status, target)
            if max_history is None and size_category == 'TINY':
                # TINY: ใช้ seconds-based
                processing_fps = getattr(self, '_current_processing_fps', None)
                if processing_fps is None or processing_fps <= 0:
                    from config import DEFAULT_FPS
                    processing_fps = DEFAULT_FPS

                from config import TINY_OBJECT_PATH_HISTORY_SECONDS
                max_history = int(TINY_OBJECT_PATH_HISTORY_SECONDS * processing_fps)
                max_history = max(20, max_history)

            if max_history and len(target['centers_history']) > max_history:
                target['centers_history'] = target['centers_history'][-max_history:]

    def _should_store_path_history(self, status, size_category, frame_counter):
        """
        ตรวจสอบว่าควรเก็บ path history หรือไม่

        Args:
            status: Status ของ target
            size_category: Size category ('TINY', 'SMALL', 'MEDIUM', 'LARGE', None)
            frame_counter: Frame counter

        Returns:
            bool: True ถ้าควรเก็บ
        """
        if status == 'RED' or status == 'DRONE':
            return True
        elif status == 'ORANGE':
            return True
        elif status == 'YELLOW':
            return True
        elif status == 'GREEN' or status == 'TRACKING':
            # GREEN: เก็บแบบ interval-based หรือไม่เก็บเลย
            if not DRONE_GREEN_PATH_HISTORY_ENABLED:
                # แต่ถ้า path-based ORANGE detection เปิดอยู่ → เก็บเพื่อตรวจสอบ
                if DRONE_PATH_BASED_ORANGE_ENABLED:
                    # ใช้ default interval (MEDIUM) เมื่อไม่รู้ขนาด
                    if size_category is None:
                        return frame_counter % DRONE_GREEN_MEDIUM_PATH_INTERVAL == 0
                else:
                    return False

            # ถ้า size_category เป็น None → ใช้ default interval (MEDIUM)
            if size_category is None:
                return frame_counter % DRONE_GREEN_MEDIUM_PATH_INTERVAL == 0

            # เก็บแบบ interval-based ตามขนาด
            if size_category == 'TINY':
                return frame_counter % DRONE_GREEN_TINY_PATH_INTERVAL == 0
            elif size_category == 'SMALL':
                return frame_counter % DRONE_GREEN_SMALL_PATH_INTERVAL == 0
            elif size_category == 'MEDIUM':
                return frame_counter % DRONE_GREEN_MEDIUM_PATH_INTERVAL == 0
            elif size_category == 'LARGE':
                return frame_counter % DRONE_GREEN_LARGE_PATH_INTERVAL == 0
            else:
                return False  # ไม่รู้ขนาด → ไม่เก็บ (fallback)

        return False

    def _get_path_history_limit(self, size_category, status, target):
        """
        คืนค่า max_history ตาม size category และ status

        Args:
            size_category: Size category ('TINY', 'SMALL', 'MEDIUM', 'LARGE', None)
            status: Status ของ target
            target: Target dictionary (สำหรับใช้กับ TINY ที่ต้องใช้ seconds-based)

        Returns:
            int: max_history (None สำหรับ TINY ที่ใช้ seconds-based)
        """
        if size_category == 'TINY':
            # TINY: ใช้ seconds-based
            return None  # จะคำนวณใน update_path_history
        elif size_category == 'SMALL':
            return DRONE_SMALL_PATH_HISTORY_MAX_FRAMES
        elif size_category == 'MEDIUM':
            return DRONE_MEDIUM_PATH_HISTORY_MAX_FRAMES
        elif size_category == 'LARGE':
            return DRONE_LARGE_PATH_HISTORY_MAX_FRAMES
        else:
            # Default: ใช้ค่าเดียวกับ MEDIUM
            return DRONE_MEDIUM_PATH_HISTORY_MAX_FRAMES

    def cleanup_path_history(self, frame_counter):
        """
        ทำ cleanup path history สำหรับทุก targets (ทุก N เฟรม)
        เพื่อประหยัด memory และ CPU

        Args:
            frame_counter: Frame counter
        """
        if not DRONE_PATH_HISTORY_OPTIMIZATION_ENABLED:
            return

        # ทำ cleanup ทุก N เฟรม
        if frame_counter % DRONE_PATH_HISTORY_CLEANUP_INTERVAL != 0:
            return

        for target_id, target in self.targets.items():
            if 'centers_history' not in target:
                continue

            status = target.get('status', 'GREEN')
            size_category = target.get('object_size_category')

            # กำหนด max_history
            if size_category == 'TINY':
                # TINY: ใช้ seconds-based
                processing_fps = getattr(self, '_current_processing_fps', None)
                if processing_fps is None or processing_fps <= 0:
                    from config import DEFAULT_FPS
                    processing_fps = DEFAULT_FPS

                from config import TINY_OBJECT_PATH_HISTORY_SECONDS
                max_history = int(TINY_OBJECT_PATH_HISTORY_SECONDS * processing_fps)
                max_history = max(20, max_history)
            else:
                max_history = self._get_path_history_limit(size_category, status, target)

            if max_history and len(target['centers_history']) > max_history:
                target['centers_history'] = target['centers_history'][-max_history:]

            # Cleanup size_history ถ้ามี
            if 'size_history' in target and max_history and len(target['size_history']) > max_history:
                target['size_history'] = target['size_history'][-max_history:]

    def calculate_motion_path_quality(self, target, frame_counter=None, is_motion_only=False):
        """
        ตรวจสอบคุณภาพของ motion path (แยกจากพื้นหลังชัดเจนหรือไม่)
        ใช้สำหรับกรณีที่ motion path ชัดเจนมาก แต่ YOLO confidence ต่ำ

        Args:
            target: Target dictionary
            frame_counter: Frame counter (optional, for caching)
            is_motion_only: True ถ้าเป็น motion-only target (ไม่มี size_history)

        Returns:
            tuple: (is_clear: bool, quality_score: float)
                - is_clear: True ถ้า motion path ชัดเจน (แยกจากพื้นหลัง)
                - quality_score: คะแนนคุณภาพ (0.0-1.0, สูง=ชัดเจน)
        """
        # ใช้ min_path_points ตาม target type
        min_path_points = MOTION_ONLY_MIN_PATH_POINTS if is_motion_only else DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS

        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < min_path_points:
            return (False, 0.0)

        # Check cache (performance optimization)
        cache_key = 'motion_path_quality_cache'
        if cache_key in target:
            cache_result, cache_frame = target[cache_key]
            if frame_counter is not None and cache_frame is not None:
                # Cache valid for 6 frames
                if frame_counter - cache_frame < 6:
                    return cache_result

        # 1. ตรวจสอบ path continuity
        has_continuity, continuity_score = self.calculate_path_continuity(target, frame_counter)

        # 2. ตรวจสอบ size stability (ข้ามถ้าเป็น motion-only target)
        is_stable = True
        stability_score = 1.0
        size_category = target.get('object_size_category')

        if not is_motion_only:
            is_stable, stability_score = self.calculate_size_stability(target, frame_counter)

            # สำหรับ TINY objects: ตรวจสอบ size variation pattern (ช่วยยืนยันว่าเป็นโดรนจริง)
            if size_category == 'TINY' and not is_stable:
                # ถ้า size ไม่ stable แต่ pattern smooth → ยังให้คะแนนบางส่วน
                is_valid_pattern, variation_score = self.check_tiny_size_variation_pattern(target, frame_counter)
                if is_valid_pattern:
                    # Pattern smooth → ให้คะแนนเพิ่ม (ช่วยชดเชย size stability)
                    stability_score = max(stability_score, variation_score * 0.5)  # ใช้ 50% ของ variation score
                    # ถ้า pattern smooth มาก → ถือว่า stable สำหรับ TINY
                    if variation_score >= 0.7:
                        is_stable = True

        # 3. ตรวจสอบ background noise ใน ROI
        is_low_noise, noise_level, motion_boxes_in_roi = self.check_background_noise_in_roi(target, frame_counter, size_category)

        # 4. ตรวจสอบ path consistency
        path_consistency = self._calculate_path_consistency_for_quality(target)

        # คำนวณ quality score
        quality_score = 0.0

        # Continuity weight: 30%
        # สำหรับ TINY objects: ใช้ continuity_score แทน has_continuity (ยืดหยุ่นขึ้น)
        if size_category == 'TINY' and is_motion_only:
            # สำหรับ TINY motion-only: ใช้ continuity_score แทน has_continuity
            if continuity_score >= 0.3:  # ถ้า continuity_score สูงพอ
                quality_score += continuity_score * 0.3
            # หรือถ้า path consistency สูง → ยังให้คะแนนบางส่วน
            elif path_consistency >= 0.7:
                quality_score += path_consistency * 0.2  # ใช้ path consistency แทน
        elif has_continuity:
            quality_score += continuity_score * 0.3

        # Stability weight: ใช้ size-adaptive weight (หรือ 0% ถ้าเป็น motion-only)
        if is_motion_only:
            # สำหรับ motion-only: เพิ่ม weight ให้ continuity และ consistency แทน
            # แต่ถ้าเป็น TINY และ continuity_score ต่ำ → ไม่เพิ่ม
            if not (size_category == 'TINY' and continuity_score < 0.3):
                quality_score += continuity_score * 0.1  # เพิ่มจาก continuity
                quality_score += path_consistency * 0.1  # เพิ่มจาก consistency
        else:
            # ใช้ size-adaptive weight สำหรับ size stability
            if size_category == 'TINY':
                stability_weight = TINY_SIZE_STABILITY_WEIGHT
            elif size_category == 'SMALL':
                stability_weight = SMALL_SIZE_STABILITY_WEIGHT
            elif size_category == 'MEDIUM':
                stability_weight = MEDIUM_SIZE_STABILITY_WEIGHT
            elif size_category == 'LARGE':
                stability_weight = LARGE_SIZE_STABILITY_WEIGHT
            else:
                stability_weight = 0.2  # Default

            if is_stable:
                quality_score += stability_score * stability_weight
            else:
                # สำหรับ TINY objects: ถ้าไม่ stable แต่ path smooth → ยังให้คะแนนบางส่วน
                if size_category == 'TINY' and has_continuity and path_consistency >= 0.6:
                    # ให้คะแนนบางส่วน (50% ของ stability weight)
                    quality_score += stability_score * stability_weight * 0.5

        # Low noise weight: 30% (สำคัญมาก - ถ้า noise สูง = ไม่ชัดเจน)
        # สำหรับ TINY objects: ลด weight ของ noise (ยืดหยุ่นขึ้น)
        if is_low_noise:
            noise_score = 1.0 - min(1.0, noise_level / DRONE_BACKGROUND_NOISE_THRESHOLD)
            if size_category == 'TINY' and is_motion_only:
                # สำหรับ TINY motion-only: ลด weight ของ noise (20% แทน 30%)
                quality_score += noise_score * 0.2
            else:
                quality_score += noise_score * 0.3
        elif size_category == 'TINY' and is_motion_only and path_consistency >= 0.7:
            # สำหรับ TINY motion-only: ถ้า path consistency สูง → ยังให้คะแนนบางส่วนแม้ noise สูง
            quality_score += path_consistency * 0.1  # ใช้ path consistency แทน noise

        # Path consistency weight: 20%
        quality_score += path_consistency * 0.2

        quality_score = min(1.0, quality_score)

        # Motion path ชัดเจน = continuity + (stability ถ้าไม่ใช่ motion-only) + low noise + consistency
        threshold = MOTION_ONLY_MIN_PATH_QUALITY if is_motion_only else DRONE_PATH_BASED_ORANGE_MOTION_CLEAR_THRESHOLD

        # สำหรับ TINY objects: ไม่ต้องผ่าน size stability check ถ้า path smooth
        if size_category == 'TINY' and not is_motion_only:
            # TINY objects: ถ้า path smooth → ไม่ต้องผ่าน size stability (ยอมรับการเปลี่ยนแปลงขนาดสูง)
            is_clear = (
                quality_score >= threshold and
                (has_continuity or continuity_score >= 0.3 or path_consistency >= 0.7) and  # ยืดหยุ่น: ไม่ต้อง continuity ถ้า score/consistency สูง
                (is_stable or (has_continuity and path_consistency >= 0.6)) and  # ยืดหยุ่นสำหรับ TINY
                (is_low_noise or path_consistency >= 0.7)  # ยืดหยุ่น: ไม่ต้อง low noise ถ้า path consistency สูง
            )
        elif size_category == 'TINY' and is_motion_only:
            # สำหรับ TINY motion-only: ยืดหยุ่นมาก
            is_clear = (
                quality_score >= threshold and
                (has_continuity or continuity_score >= 0.3 or path_consistency >= 0.7) and  # ยืดหยุ่น
                (is_low_noise or path_consistency >= 0.7)  # ยืดหยุ่น
            )
        else:
            # สำหรับขนาดอื่น: ใช้เงื่อนไขเดิม
            is_clear = (
                quality_score >= threshold and
                has_continuity and
                (is_stable if not is_motion_only else True) and
                is_low_noise
            )

        result = (is_clear, quality_score)

        # Cache result
        if cache_key not in target:
            target[cache_key] = (result, frame_counter)
        else:
            target[cache_key] = (result, frame_counter)

        return result

    def calculate_size_adaptive_path_quality(self, target, frame_counter, size_category, is_motion_only=False):
        """
        คำนวณ path quality แบบ size-adaptive

        Args:
            target: Target dictionary
            frame_counter: Frame counter
            size_category: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None
            is_motion_only: True ถ้าเป็น motion-only target

        Returns:
            tuple: (is_clear: bool, quality_score: float)
        """
        # ใช้ min_path_points ตาม size category
        if size_category == 'TINY':
            min_path_points = TINY_MIN_PATH_POINTS
            min_quality = TINY_MIN_PATH_QUALITY
        elif size_category == 'SMALL':
            min_path_points = SMALL_MIN_PATH_POINTS
            min_quality = SMALL_MIN_PATH_QUALITY
        elif size_category == 'MEDIUM':
            min_path_points = MEDIUM_MIN_PATH_POINTS
            min_quality = MEDIUM_MIN_PATH_QUALITY
        elif size_category == 'LARGE':
            min_path_points = LARGE_MIN_PATH_POINTS
            min_quality = LARGE_MIN_PATH_QUALITY
        else:
            # Default: ใช้ค่าเดิม
            min_path_points = MOTION_ONLY_MIN_PATH_POINTS if is_motion_only else DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS
            min_quality = MOTION_ONLY_MIN_PATH_QUALITY if is_motion_only else DRONE_PATH_BASED_ORANGE_MOTION_CLEAR_THRESHOLD

        # Early exit: ต้องมี centers_history
        if 'centers_history' not in target or len(target['centers_history']) < min_path_points:
            return (False, 0.0)

        # เรียก calculate_motion_path_quality() และปรับ threshold ตาม size
        is_clear, quality_score = self.calculate_motion_path_quality(target, frame_counter, is_motion_only)

        # ปรับ threshold ตาม size category
        is_clear = quality_score >= min_quality

        return (is_clear, quality_score)

    def get_immediate_lock_threshold(self, size_category):
        """
        ดึง YOLO confidence threshold สำหรับ immediate lock ตาม size category

        Args:
            size_category: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None

        Returns:
            float: Confidence threshold
        """
        if size_category == 'TINY':
            return TINY_YOLO_IMMEDIATE_LOCK_CONF
        elif size_category == 'SMALL':
            return SMALL_YOLO_IMMEDIATE_LOCK_CONF
        elif size_category == 'MEDIUM':
            return MEDIUM_YOLO_IMMEDIATE_LOCK_CONF
        elif size_category == 'LARGE':
            return LARGE_YOLO_IMMEDIATE_LOCK_CONF
        else:
            # Default: ใช้ค่า MEDIUM
            return MEDIUM_YOLO_IMMEDIATE_LOCK_CONF

    def get_size_adaptive_red_weights(self, size_category):
        """
        ดึง weights สำหรับ RED decision ตาม size category

        Args:
            size_category: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None

        Returns:
            tuple: (yolo_weight, motion_weight)
        """
        if size_category == 'TINY':
            return (TINY_RED_YOLO_WEIGHT, TINY_RED_MOTION_WEIGHT)
        elif size_category == 'SMALL':
            return (SMALL_RED_YOLO_WEIGHT, SMALL_RED_MOTION_WEIGHT)
        elif size_category == 'MEDIUM':
            return (MEDIUM_RED_YOLO_WEIGHT, MEDIUM_RED_MOTION_WEIGHT)
        elif size_category == 'LARGE':
            return (LARGE_RED_YOLO_WEIGHT, LARGE_RED_MOTION_WEIGHT)
        else:
            # Default: ใช้ค่า MEDIUM
            return (MEDIUM_RED_YOLO_WEIGHT, MEDIUM_RED_MOTION_WEIGHT)

    def detect_background_blending(self, target, frame_counter):
        """
        ตรวจสอบว่าวัตถุกลืนกับพื้นหลังหรือไม่

        Args:
            target: Target dictionary
            frame_counter: Frame counter

        Returns:
            bool: True ถ้า detect blending
        """
        if not BLENDED_DETECTION_ENABLED:
            return False

        # ตรวจสอบว่า YOLO confidence ต่ำ แต่ motion path quality สูง
        original_conf = target.get('original_confidence', target.get('confidence', 0.0))
        path_quality = target.get('path_quality', 0.0)

        # ถ้า YOLO confidence ต่ำ (< HYBRID_BASE_CONF) แต่ motion path quality สูง (>= 0.6)
        if original_conf < HYBRID_BASE_CONF and path_quality >= 0.6:
            return True

        return False

    def check_yolo_motion_overlap_for_red(self, target, motion_boxes, frame_counter):
        """
        ตรวจสอบว่า YOLO detection มี motion overlap และพร้อมเปลี่ยนเป็น RED เร็วหรือไม่

        Args:
            target: Target dictionary
            motion_boxes: List of motion boxes [(x, y, w, h), ...]
            frame_counter: Current frame counter

        Returns:
            tuple: (is_overlap, max_iou, motion_quality, should_fast_red)
        """
        from config import (
            YOLO_MOTION_OVERLAP_RED_ENABLED, YOLO_MOTION_OVERLAP_MIN_IOU,
            YOLO_MOTION_OVERLAP_MIN_CONF, YOLO_MOTION_OVERLAP_MIN_MOTION_QUALITY
        )

        if not YOLO_MOTION_OVERLAP_RED_ENABLED:
            return (False, 0.0, 0.0, False)

        # ตรวจสอบว่า target มี YOLO detection หรือไม่
        if 'last_rect' not in target or target['last_rect'] is None:
            return (False, 0.0, 0.0, False)

        original_conf = target.get('original_confidence', 0.0)
        if original_conf < YOLO_MOTION_OVERLAP_MIN_CONF:
            return (False, 0.0, 0.0, False)

        # ตรวจสอบ motion overlap
        yolo_rect = target['last_rect']
        max_iou = 0.0
        best_motion_box = None

        for motion_box in motion_boxes:
            iou = calculate_iou(yolo_rect, motion_box)
            if iou > max_iou:
                max_iou = iou
                best_motion_box = motion_box

        if max_iou < YOLO_MOTION_OVERLAP_MIN_IOU:
            return (False, max_iou, 0.0, False)

        # ตรวจสอบ motion path quality
        path_quality = target.get('path_quality', 0.0)
        if path_quality < YOLO_MOTION_OVERLAP_MIN_MOTION_QUALITY:
            return (True, max_iou, path_quality, False)

        # พร้อมเปลี่ยนเป็น RED เร็ว
        return (True, max_iou, path_quality, True)

    def check_tiny_size_variation_pattern(self, target, frame_counter=None):
        """
        ตรวจสอบ pattern การเปลี่ยนแปลงขนาดสำหรับ TINY objects
        TINY objects ที่เป็นโดรนจริงจะมีการเปลี่ยนแปลงขนาดแบบ smooth (ไม่กระโดด)
        ใช้สำหรับช่วยยืนยันว่าเป็นโดรนจริงแม้มีการเปลี่ยนแปลงขนาดสูง

        Args:
            target: Target dictionary
            frame_counter: Frame counter

        Returns:
            tuple: (is_valid_pattern: bool, variation_score: float)
                - is_valid_pattern: True ถ้า pattern การเปลี่ยนแปลงขนาดเป็นแบบ smooth
                - variation_score: คะแนน (0.0-1.0, สูง=smooth)
        """
        if 'size_history' not in target or len(target['size_history']) < 5:
            return (False, 0.0)

        size_history = target['size_history']
        areas = [s[1] for s in size_history[-10:]]  # ใช้ 10 จุดล่าสุด

        if len(areas) < 2:
            return (False, 0.0)

        # ตรวจสอบว่าเป็นการเปลี่ยนแปลงแบบ smooth (ไม่กระโดด)
        # คำนวณการเปลี่ยนแปลงระหว่างจุดติดกัน
        variations = []
        for i in range(1, len(areas)):
            if areas[i-1] > 0:
                variation = abs(areas[i] - areas[i-1]) / areas[i-1]
                variations.append(variation)

        if len(variations) == 0:
            return (False, 0.0)

        # คำนวณ CV ของ variations (CV ต่ำ = smooth)
        mean_var = sum(variations) / len(variations)
        if mean_var == 0:
            return (True, 1.0)

        variance = sum((v - mean_var)**2 for v in variations) / len(variations)
        std_dev = math.sqrt(variance)
        cv_variations = std_dev / mean_var if mean_var > 0 else 1.0

        # CV ต่ำ = smooth variation = valid pattern
        is_valid = cv_variations < 0.5  # threshold สำหรับ smooth variation
        variation_score = max(0.0, 1.0 - min(1.0, cv_variations))

        return (is_valid, variation_score)

    def _smooth_path_points(self, path_points, window_size=None):
        """
        Smooth path points using moving average

        Args:
            path_points: List of (x, y) tuples
            window_size: Window size for moving average (if None, use from config)

        Returns:
            list: Smoothed path points [(x, y), ...]
        """
        from config import PATH_SMOOTHING_ENABLED, PATH_SMOOTHING_WINDOW_SIZE

        if not PATH_SMOOTHING_ENABLED or len(path_points) < 2:
            return path_points.copy()

        if window_size is None:
            window_size = PATH_SMOOTHING_WINDOW_SIZE

        if len(path_points) < window_size:
            return path_points.copy()

        smoothed = []
        half_window = window_size // 2

        for i in range(len(path_points)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(path_points), i + half_window + 1)

            window_points = path_points[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            smoothed.append((avg_x, avg_y))

        return smoothed

    def _calculate_path_characteristics(self, path_points, target, frame_counter=None, processing_fps=None):
        """
        คำนวณ path characteristics สำหรับการแยกประเภท (ใช้ processing_fps สำหรับ velocity calculation)

        Args:
            path_points: List of (x, y) path points
            target: Target dictionary
            frame_counter: Frame counter (optional)
            processing_fps: Processing FPS (optional, for velocity calculation)

        Returns:
            dict: {
                'straightness': float,      # 0.0-1.0 (สูง=ตรง)
                'smoothness': float,        # 0.0-1.0 (สูง=ราบเรียบ)
                'velocity_consistency': float,  # 0.0-1.0 (สูง=สม่ำเสมอ)
                'velocity_magnitude': float,    # pixels per second (ใช้ processing_fps)
                'direction_change': float,       # degrees (สูง=เปลี่ยนทิศทางบ่อย)
                'acceleration_pattern': str,    # 'CONSTANT', 'ACCELERATING', 'DECELERATING', 'VARIABLE'
                'path_length': float,           # total path length
                'net_displacement': float,      # distance from start to end
                'curvature': float,             # average curvature
                'sharp_turns': int              # number of sharp turns (>30 degrees)
            }
        """
        if len(path_points) < 2:
            return {
                'straightness': 0.5,
                'smoothness': 0.5,
                'velocity_consistency': 0.5,
                'velocity_magnitude': 0.0,
                'direction_change': 0.0,
                'acceleration_pattern': 'UNKNOWN',
                'path_length': 0.0,
                'net_displacement': 0.0,
                'curvature': 0.0,
                'sharp_turns': 0
            }

        from config import DEFAULT_FPS

        if processing_fps is None or processing_fps <= 0:
            processing_fps = DEFAULT_FPS

        # 1. Calculate path length and net displacement
        path_length = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            path_length += math.sqrt(dx**2 + dy**2)

        start_point = path_points[0]
        end_point = path_points[-1]
        net_displacement = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

        # 2. Calculate straightness (net_displacement / path_length)
        straightness = net_displacement / path_length if path_length > 0 else 0.0

        # 3. Calculate velocities (pixels per second)
        velocities = []
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            vel = math.sqrt(dx**2 + dy**2) * processing_fps
            velocities.append(vel)

        # 4. Calculate velocity magnitude (average)
        velocity_magnitude = sum(velocities) / len(velocities) if velocities else 0.0

        # 5. Calculate velocity consistency (coefficient of variation)
        if len(velocities) >= 2 and velocity_magnitude > 0:
            mean_vel = velocity_magnitude
            variance = sum((v - mean_vel)**2 for v in velocities) / len(velocities)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_vel if mean_vel > 0 else 1.0
            velocity_consistency = max(0.0, 1.0 - min(1.0, cv))  # CV ต่ำ = consistency สูง
        else:
            velocity_consistency = 0.5

        # 6. Calculate direction changes
        direction_changes = []
        sharp_turns = 0
        for i in range(1, len(path_points)):
            dx1 = path_points[i][0] - path_points[i-1][0]
            dy1 = path_points[i][1] - path_points[i-1][1]
            if i < len(path_points) - 1:
                dx2 = path_points[i+1][0] - path_points[i][0]
                dy2 = path_points[i+1][1] - path_points[i][1]

                angle1 = math.degrees(math.atan2(dy1, dx1))
                angle2 = math.degrees(math.atan2(dy2, dx2))

                angle_diff = abs(angle2 - angle1)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                direction_changes.append(angle_diff)
                if angle_diff > 30:
                    sharp_turns += 1

        direction_change = sum(direction_changes) / len(direction_changes) if direction_changes else 0.0

        # 7. Calculate smoothness (inverse of direction change variation)
        if len(direction_changes) >= 2:
            mean_dir_change = direction_change
            variance_dir = sum((d - mean_dir_change)**2 for d in direction_changes) / len(direction_changes)
            std_dir = math.sqrt(variance_dir)
            smoothness = max(0.0, 1.0 - min(1.0, std_dir / 90.0))  # normalize by 90 degrees
        else:
            smoothness = 0.5

        # 8. Calculate acceleration pattern
        if len(velocities) >= 3:
            accels = []
            for i in range(1, len(velocities)):
                accel = velocities[i] - velocities[i-1]
                accels.append(accel)

            if len(accels) >= 2:
                mean_accel = sum(accels) / len(accels)
                variance_accel = sum((a - mean_accel)**2 for a in accels) / len(accels)
                std_accel = math.sqrt(variance_accel)

                if std_accel < 0.1 * abs(mean_accel) if mean_accel != 0 else std_accel < 1.0:
                    if mean_accel > 0.1:
                        acceleration_pattern = 'ACCELERATING'
                    elif mean_accel < -0.1:
                        acceleration_pattern = 'DECELERATING'
                    else:
                        acceleration_pattern = 'CONSTANT'
                else:
                    acceleration_pattern = 'VARIABLE'
            else:
                acceleration_pattern = 'UNKNOWN'
        else:
            acceleration_pattern = 'UNKNOWN'

        # 9. Calculate curvature (simplified - average angle change)
        curvature = direction_change / 180.0 if direction_changes else 0.0  # normalize

        return {
            'straightness': min(1.0, max(0.0, straightness)),
            'smoothness': min(1.0, max(0.0, smoothness)),
            'velocity_consistency': min(1.0, max(0.0, velocity_consistency)),
            'velocity_magnitude': velocity_magnitude,
            'direction_change': direction_change,
            'acceleration_pattern': acceleration_pattern,
            'path_length': path_length,
            'net_displacement': net_displacement,
            'curvature': min(1.0, max(0.0, curvature)),
            'sharp_turns': sharp_turns
        }

    def _get_classification_thresholds(self, target, processing_fps=None):
        """
        ดึง classification thresholds ที่ปรับตาม resolution และ FPS

        Args:
            target: Target dictionary
            processing_fps: Processing FPS (optional)

        Returns:
            dict: Thresholds dictionary
        """
        from config import (
            PATH_CLASSIFICATION_USE_RESOLUTION_ADAPTIVE, PATH_CLASSIFICATION_USE_FPS_AWARE,
            PATH_VELOCITY_FPS_MULTIPLIER, PATH_VELOCITY_FPS_REFERENCE,
            DRONE_STRAIGHTNESS_THRESHOLD, DRONE_SMOOTHNESS_THRESHOLD,
            DRONE_VELOCITY_CONSISTENCY_THRESHOLD, DRONE_VELOCITY_MIN, DRONE_VELOCITY_MAX,
            BIRD_STRAIGHTNESS_THRESHOLD, BIRD_SMOOTHNESS_THRESHOLD,
            BIRD_VELOCITY_CONSISTENCY_THRESHOLD, BIRD_VELOCITY_MIN, BIRD_VELOCITY_MAX,
            INSECT_STRAIGHTNESS_THRESHOLD, INSECT_SMOOTHNESS_THRESHOLD,
            INSECT_VELOCITY_CONSISTENCY_THRESHOLD, INSECT_VELOCITY_MIN, INSECT_VELOCITY_MAX,
            AIRPLANE_STRAIGHTNESS_THRESHOLD, AIRPLANE_SMOOTHNESS_THRESHOLD,
            AIRPLANE_VELOCITY_CONSISTENCY_THRESHOLD, AIRPLANE_VELOCITY_MIN, AIRPLANE_VELOCITY_MAX,
            NOISE_STRAIGHTNESS_THRESHOLD, NOISE_SMOOTHNESS_THRESHOLD,
            NOISE_VELOCITY_CONSISTENCY_THRESHOLD
        )

        # คำนวณ FPS multiplier (ถ้าใช้ FPS-aware)
        fps_multiplier = 1.0
        if PATH_CLASSIFICATION_USE_FPS_AWARE and processing_fps is not None and processing_fps > 0:
            fps_multiplier = processing_fps / PATH_VELOCITY_FPS_REFERENCE

        # ปรับ velocity thresholds ตาม FPS
        thresholds = {
            'DRONE': {
                'straightness': DRONE_STRAIGHTNESS_THRESHOLD,
                'smoothness': DRONE_SMOOTHNESS_THRESHOLD,
                'velocity_consistency': DRONE_VELOCITY_CONSISTENCY_THRESHOLD,
                'velocity_min': DRONE_VELOCITY_MIN * fps_multiplier,
                'velocity_max': DRONE_VELOCITY_MAX * fps_multiplier
            },
            'BIRD': {
                'straightness': BIRD_STRAIGHTNESS_THRESHOLD,
                'smoothness': BIRD_SMOOTHNESS_THRESHOLD,
                'velocity_consistency': BIRD_VELOCITY_CONSISTENCY_THRESHOLD,
                'velocity_min': BIRD_VELOCITY_MIN * fps_multiplier,
                'velocity_max': BIRD_VELOCITY_MAX * fps_multiplier
            },
            'INSECT': {
                'straightness': INSECT_STRAIGHTNESS_THRESHOLD,
                'smoothness': INSECT_SMOOTHNESS_THRESHOLD,
                'velocity_consistency': INSECT_VELOCITY_CONSISTENCY_THRESHOLD,
                'velocity_min': INSECT_VELOCITY_MIN * fps_multiplier,
                'velocity_max': INSECT_VELOCITY_MAX * fps_multiplier
            },
            'AIRPLANE': {
                'straightness': AIRPLANE_STRAIGHTNESS_THRESHOLD,
                'smoothness': AIRPLANE_SMOOTHNESS_THRESHOLD,
                'velocity_consistency': AIRPLANE_VELOCITY_CONSISTENCY_THRESHOLD,
                'velocity_min': AIRPLANE_VELOCITY_MIN * fps_multiplier,
                'velocity_max': AIRPLANE_VELOCITY_MAX * fps_multiplier
            },
            'NOISE': {
                'straightness': NOISE_STRAIGHTNESS_THRESHOLD,
                'smoothness': NOISE_SMOOTHNESS_THRESHOLD,
                'velocity_consistency': NOISE_VELOCITY_CONSISTENCY_THRESHOLD
            }
        }

        return thresholds

    def _classify_from_characteristics(self, characteristics, target, processing_fps=None):
        """
        แยกประเภท target จาก path characteristics (ใช้ thresholds ที่ปรับตาม resolution และ FPS)

        Args:
            characteristics: Path characteristics dictionary
            target: Target dictionary
            processing_fps: Processing FPS (optional, for FPS-aware thresholds)

        Returns:
            tuple: (classification: str, confidence: float)
        """
        # ดึง thresholds ที่ปรับตาม resolution และ FPS
        thresholds = self._get_classification_thresholds(target, processing_fps)

        straightness = characteristics.get('straightness', 0.5)
        smoothness = characteristics.get('smoothness', 0.5)
        velocity_consistency = characteristics.get('velocity_consistency', 0.5)
        velocity_mag = characteristics.get('velocity_magnitude', 0.0)
        direction_change = characteristics.get('direction_change', 0.0)
        sharp_turns = characteristics.get('sharp_turns', 0)

        # คำนวณ confidence score สำหรับแต่ละประเภท (ใช้ thresholds ที่ปรับแล้ว)
        scores = {}

        # DRONE: straightness ต่ำ-กลาง, smoothness สูง, velocity consistency สูง, มี sharp turns
        drone_th = thresholds['DRONE']
        drone_score = 0.0
        if straightness < drone_th['straightness']:
            drone_score += 0.3
        if smoothness >= drone_th['smoothness']:
            drone_score += 0.3
        if velocity_consistency >= drone_th['velocity_consistency']:
            drone_score += 0.2
        if drone_th['velocity_min'] <= velocity_mag <= drone_th['velocity_max']:
            drone_score += 0.1
        if sharp_turns > 0:
            drone_score += 0.1
        scores['DRONE'] = drone_score

        # BIRD: straightness ต่ำ, smoothness ต่ำ-กลาง, velocity consistency ต่ำ, direction change สูง
        bird_th = thresholds['BIRD']
        bird_score = 0.0
        if straightness < bird_th['straightness']:
            bird_score += 0.3
        if smoothness < bird_th['smoothness']:
            bird_score += 0.3
        if velocity_consistency < bird_th['velocity_consistency']:
            bird_score += 0.2
        if direction_change > 30.0:
            bird_score += 0.2
        scores['BIRD'] = bird_score

        # INSECT: straightness ต่ำมาก, smoothness ต่ำ, velocity consistency ต่ำมาก, direction change สูงมาก
        insect_th = thresholds['INSECT']
        insect_score = 0.0
        if straightness < insect_th['straightness']:
            insect_score += 0.3
        if smoothness < insect_th['smoothness']:
            insect_score += 0.3
        if velocity_consistency < insect_th['velocity_consistency']:
            insect_score += 0.2
        if direction_change > 60.0:
            insect_score += 0.2
        scores['INSECT'] = insect_score

        # AIRPLANE: straightness สูงมาก, smoothness สูง, velocity consistency สูงมาก, velocity สูง
        airplane_th = thresholds['AIRPLANE']
        airplane_score = 0.0
        if straightness >= airplane_th['straightness']:
            airplane_score += 0.4
        if smoothness >= airplane_th['smoothness']:
            airplane_score += 0.3
        if velocity_consistency >= airplane_th['velocity_consistency']:
            airplane_score += 0.2
        if velocity_mag >= airplane_th['velocity_min']:
            airplane_score += 0.1
        scores['AIRPLANE'] = airplane_score

        # NOISE: straightness ต่ำมาก, smoothness ต่ำมาก, velocity consistency ต่ำมาก
        noise_th = thresholds['NOISE']
        noise_score = 0.0
        if straightness < noise_th['straightness']:
            noise_score += 0.4
        if smoothness < noise_th['smoothness']:
            noise_score += 0.4
        if velocity_consistency < noise_th['velocity_consistency']:
            noise_score += 0.2
        scores['NOISE'] = noise_score

        # เลือกประเภทที่มีคะแนนสูงสุด
        best_classification = max(scores, key=scores.get)
        best_confidence = scores[best_classification]

        # ถ้า confidence ต่ำเกินไป → UNKNOWN
        if best_confidence < 0.5:
            return ('UNKNOWN', best_confidence)

        return (best_classification, best_confidence)

    def classify_target_from_path(self, target, frame_counter=None, processing_fps=None):
        """
        วิเคราะห์ path เพื่อแยกประเภท target (นก, แมลง, เครื่องบิน, โดรน, noise)

        **หมายเหตุ: ใช้แค่ path characteristics อย่างเดียว (ไม่ใช้ YOLO confidence)**

        Args:
            target: Target dictionary
            frame_counter: Frame counter (optional, for caching)
            processing_fps: Processing FPS (optional, for FPS-aware thresholds)

        Returns:
            dict: {
                'classification': str,  # 'DRONE', 'BIRD', 'INSECT', 'AIRPLANE', 'NOISE', 'UNKNOWN'
                'confidence': float,    # 0.0-1.0 (confidence จาก path characteristics อย่างเดียว)
                'characteristics': dict,  # path characteristics
                'reason': str  # reason for classification (optional)
            }
        """
        from config import PATH_CLASSIFICATION_MIN_POINTS, PATH_CLASSIFICATION_RECOMMENDED_POINTS

        # ตรวจสอบว่า path มีจุดเพียงพอ (อย่างน้อย 15-30 จุด)
        centers_history = target.get('centers_history', [])

        if len(centers_history) < PATH_CLASSIFICATION_MIN_POINTS:
            return {
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'characteristics': {},
                'reason': f'Insufficient path points ({len(centers_history)} < {PATH_CLASSIFICATION_MIN_POINTS})'
            }

        # ใช้ smoothed path สำหรับการวิเคราะห์ (ใช้ 30 จุดล่าสุด หรือทั้งหมดถ้าน้อยกว่า)
        path_points = [center for _, center in centers_history[-PATH_CLASSIFICATION_RECOMMENDED_POINTS:]]
        if len(path_points) < PATH_CLASSIFICATION_MIN_POINTS:
            return {
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'characteristics': {},
                'reason': f'Insufficient path points after filtering ({len(path_points)} < {PATH_CLASSIFICATION_MIN_POINTS})'
            }

        # Smooth path points
        smoothed_path = self._smooth_path_points(path_points)

        # คำนวณ path characteristics (ใช้ processing_fps สำหรับ velocity calculation)
        characteristics = self._calculate_path_characteristics(smoothed_path, target, frame_counter, processing_fps)

        # แยกประเภท target จาก characteristics (ใช้ thresholds ที่ปรับตาม resolution และ FPS)
        classification, confidence = self._classify_from_characteristics(characteristics, target, processing_fps)

        return {
            'classification': classification,
            'confidence': confidence,
            'characteristics': characteristics
        }

    def get_object_size_category_for_target_from_rect(self, rect, frame_w, frame_h):
        """
        คำนวณ size category จาก rect (ใช้สำหรับ YOLO box ที่ยังไม่เป็น target)

        Args:
            rect: (x, y, w, h) หรือ (x1, y1, x2, y2)
            frame_w: Frame width
            frame_h: Frame height

        Returns:
            str: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', หรือ None
        """
        if len(rect) == 4:
            if rect[2] > 100 and rect[3] > 100:  # ถ้า w, h ใหญ่ → เป็น (x, y, w, h)
                x, y, w, h = rect
            else:  # ถ้า w, h เล็ก → เป็น (x1, y1, x2, y2)
                x1, y1, x2, y2 = rect
                w = x2 - x1
                h = y2 - y1
        else:
            return None

        area = w * h
        diagonal = math.sqrt(w**2 + h**2)
        frame_area = frame_w * frame_h
        frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)

        # คำนวณ thresholds
        tiny_area_threshold = TINY_OBJECT_AREA_THRESHOLD_RATIO * frame_area
        tiny_diagonal_threshold = TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        small_area_threshold = self.thresholds.get('DRONE_SMALL_OBJECT_AREA_THRESHOLD',
                                                   DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO * frame_area)
        small_diagonal_threshold = SMALL_OBJECT_DIAGONAL_THRESHOLD_RATIO * frame_diagonal

        max_area_threshold = self.thresholds.get('DRONE_MAX_AREA',
                                                 DRONE_MAX_AREA_RATIO * frame_area)

        # ตรวจสอบ size category
        if area < tiny_area_threshold and diagonal < tiny_diagonal_threshold:
            return 'TINY'
        elif area < small_area_threshold and diagonal < small_diagonal_threshold:
            return 'SMALL'
        elif area <= max_area_threshold:
            return 'MEDIUM'
        else:
            return 'LARGE'

    def _calculate_path_consistency_for_quality(self, target):
        """
        คำนวณ path consistency สำหรับ quality assessment

        Args:
            target: Target dictionary

        Returns:
            float: Path consistency score (0.0-1.0)
        """
        if 'centers_history' not in target or len(target['centers_history']) < 5:
            return 0.0

        centers_history = target['centers_history']
        first_frame, first_center = centers_history[0]
        last_frame, last_center = centers_history[-1]

        if not first_center or not last_center:
            return 0.0

        direct_dist = self.get_dist(first_center, last_center)
        path_dist = 0.0

        for i in range(1, len(centers_history)):
            _, prev_center = centers_history[i-1]
            _, curr_center = centers_history[i]
            if prev_center and curr_center:
                path_dist += self.get_dist(prev_center, curr_center)

        if path_dist > 0:
            return min(1.0, direct_dist / path_dist)
        else:
            return 0.0

    def _check_temporal_continuity(self, motion_box, target, predicted_center=None,
                                    velocity_vector=None, last_rect=None, thresholds=None):
        """
        ตรวจสอบ temporal continuity ของ motion box กับ path history
        เพื่อป้องกันการเชื่อมกระโดดเป็นกบเต้น (เช่น motion ร่องลอยจากแมลงที่บินผ่านไปแล้ว)

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            predicted_center: Optional - (x, y) predicted center position
            velocity_vector: Optional - (vx, vy) velocity vector
            last_rect: Optional - (x, y, w, h) ขนาด target ก่อนหน้า
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds

        Returns:
            tuple: (is_continuous: bool, continuity_score: float, reason: str)
                - is_continuous: True ถ้า motion box ต่อเนื่องกับ path history
                - continuity_score: คะแนน continuity (0.0-1.0, สูง=continuous)
                - reason: เหตุผลที่ continuous/not continuous
        """
        from config import (
            TEMPORAL_CONTINUITY_ENABLED, TEMPORAL_CONTINUITY_MIN_PATH_POINTS,
            TEMPORAL_CONTINUITY_DISTANCE_THRESHOLD_RATIO, VELOCITY_CHANGE_THRESHOLD_RATIO,
            VELOCITY_DIRECTION_CHANGE_THRESHOLD
        )

        # Early exit: ถ้าปิดการตรวจสอบ temporal continuity
        if not TEMPORAL_CONTINUITY_ENABLED:
            return (True, 1.0, "Temporal continuity check disabled")

        # ตรวจสอบว่ามี path history เพียงพอหรือไม่
        if 'centers_history' not in target or len(target['centers_history']) < TEMPORAL_CONTINUITY_MIN_PATH_POINTS:
            # ถ้ายังไม่มี path history เพียงพอ → ใช้ validation แบบปกติ (ไม่ตรวจสอบ temporal continuity)
            return (True, 0.5, "Insufficient path history for temporal continuity check")

        # ใช้ resolution-dependent thresholds
        if thresholds is None:
            thresholds = self.thresholds

        frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
        max_dist_threshold = thresholds.get('HYBRID_DIST_THRESHOLD', HYBRID_DIST_THRESHOLD)
        temporal_dist_threshold = TEMPORAL_CONTINUITY_DISTANCE_THRESHOLD_RATIO * frame_diagonal

        mx, my, mw, mh = motion_box
        motion_center = (mx + mw // 2, my + mh // 2)

        centers_history = target['centers_history']

        # ใช้ path history ย้อนหลัง 5-10 จุด (เพื่อตรวจสอบ continuity)
        lookback_points = min(len(centers_history), 10)
        recent_history = centers_history[-lookback_points:]

        continuity_score = 0.0
        reasons = []

        # 1. ตรวจสอบ continuity กับจุดก่อนหน้า (สำคัญที่สุด)
        if len(recent_history) >= 2:
            last_frame, last_center = recent_history[-1]
            if last_center is not None:
                dist_to_last = self.get_dist(motion_center, last_center)

                if dist_to_last > temporal_dist_threshold:
                    return (False, 0.0, f"Too far from last point ({dist_to_last:.1f} > {temporal_dist_threshold:.1f})")

                # คะแนนตามระยะห่างจากจุดก่อนหน้า (ใกล้ = score สูง)
                normalized_dist = dist_to_last / max(1.0, temporal_dist_threshold)
                last_point_score = 1.0 - normalized_dist
                continuity_score += last_point_score * 0.4  # Weight: 40%
                reasons.append(f"last_dist={dist_to_last:.1f}")

        # 2. ตรวจสอบ continuity กับ path history (หลายจุด)
        if len(recent_history) >= 3:
            # คำนวณ average distance จาก path history
            distances = []
            for frame, center in recent_history[-5:]:  # ใช้ 5 จุดล่าสุด
                if center is not None:
                    dist = self.get_dist(motion_center, center)
                    distances.append(dist)

            if distances:
                avg_dist = sum(distances) / len(distances)
                if avg_dist > temporal_dist_threshold * 1.5:  # อนุญาตให้ห่างได้ 1.5 เท่า
                    return (False, 0.0, f"Too far from path history (avg={avg_dist:.1f} > {temporal_dist_threshold * 1.5:.1f})")

                # คะแนนตาม average distance (ใกล้ = score สูง)
                normalized_avg_dist = avg_dist / max(1.0, temporal_dist_threshold)
                path_history_score = 1.0 - min(1.0, normalized_avg_dist)
                continuity_score += path_history_score * 0.3  # Weight: 30%
                reasons.append(f"path_avg_dist={avg_dist:.1f}")

        # 3. ตรวจสอบ velocity consistency (ถ้ามี velocity)
        if velocity_vector is not None and len(recent_history) >= 3:
            vx, vy = velocity_vector
            if abs(vx) > 0.01 or abs(vy) > 0.01:  # มีการเคลื่อนที่
                # คำนวณ velocity จาก path history (2-3 จุดล่าสุด)
                prev_frame, prev_center = recent_history[-2]
                curr_frame, curr_center = recent_history[-1]

                if prev_center is not None and curr_center is not None:
                    # คำนวณ velocity จาก path history
                    hist_dx = curr_center[0] - prev_center[0]
                    hist_dy = curr_center[1] - prev_center[1]
                    hist_velocity_mag = math.sqrt(hist_dx**2 + hist_dy**2)
                    current_velocity_mag = math.sqrt(vx**2 + vy**2)

                    # ตรวจสอบว่า velocity magnitude ไม่เปลี่ยนมากเกินไป
                    if hist_velocity_mag > 0.1 and current_velocity_mag > 0.1:
                        velocity_change_ratio = abs(current_velocity_mag - hist_velocity_mag) / hist_velocity_mag
                        if velocity_change_ratio > VELOCITY_CHANGE_THRESHOLD_RATIO:
                            return (False, 0.0, f"Velocity change too large ({velocity_change_ratio:.2f} > {VELOCITY_CHANGE_THRESHOLD_RATIO:.2f})")

                        # คะแนนตาม velocity consistency (สอดคล้อง = score สูง)
                        velocity_consistency_score = 1.0 - min(1.0, velocity_change_ratio / VELOCITY_CHANGE_THRESHOLD_RATIO)
                        continuity_score += velocity_consistency_score * 0.2  # Weight: 20%
                        reasons.append(f"vel_change={velocity_change_ratio:.2f}")

                    # ตรวจสอบว่า velocity direction ไม่เปลี่ยนมากเกินไป
                    if hist_velocity_mag > 0.1 and current_velocity_mag > 0.1:
                        # Normalize vectors
                        hist_norm = math.sqrt(hist_dx**2 + hist_dy**2)
                        hist_vx = hist_dx / hist_norm
                        hist_vy = hist_dy / hist_norm
                        curr_norm = math.sqrt(vx**2 + vy**2)
                        curr_vx = vx / curr_norm
                        curr_vy = vy / curr_norm

                        # คำนวณมุมระหว่าง velocity vectors
                        dot_product = hist_vx * curr_vx + hist_vy * curr_vy
                        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp
                        angle_deg = math.degrees(math.acos(dot_product))

                        if angle_deg > VELOCITY_DIRECTION_CHANGE_THRESHOLD:
                            return (False, 0.0, f"Velocity direction change too large ({angle_deg:.1f}° > {VELOCITY_DIRECTION_CHANGE_THRESHOLD}°)")

                        # คะแนนตาม direction consistency (สอดคล้อง = score สูง)
                        direction_consistency_score = 1.0 - (angle_deg / VELOCITY_DIRECTION_CHANGE_THRESHOLD)
                        continuity_score += direction_consistency_score * 0.1  # Weight: 10%
                        reasons.append(f"vel_dir_change={angle_deg:.1f}°")

        # 4. ตรวจสอบ acceleration (ถ้ามี velocity และ path history)
        from config import ACCELERATION_CHECK_MIN_PATH_POINTS
        if velocity_vector is not None and len(recent_history) >= ACCELERATION_CHECK_MIN_PATH_POINTS:
            is_valid_accel, accel_score, accel_reason = self._check_acceleration(
                motion_box, target, velocity_vector, thresholds
            )

            if not is_valid_accel:
                return (False, 0.0, f"Acceleration check failed: {accel_reason}")

            continuity_score += accel_score * 0.15  # Weight: 15%
            reasons.append(accel_reason)

        # 5. ตรวจสอบ path smoothness (ระยะกระโดด)
        if len(recent_history) >= 2:
            is_smooth, smoothness_score, smoothness_reason = self._check_path_smoothness(
                motion_box, target, thresholds
            )

            if not is_smooth:
                return (False, 0.0, f"Path smoothness check failed: {smoothness_reason}")

            continuity_score += smoothness_score * 0.1  # Weight: 10%
            reasons.append(smoothness_reason)

        reason_str = ", ".join(reasons) if reasons else "continuous"
        return (True, continuity_score, reason_str)

    def _check_acceleration(self, motion_box, target, velocity_vector=None, thresholds=None):
        """
        ตรวจสอบ acceleration/deceleration ของ motion box กับ path history
        เพื่อป้องกันการดีดตัวเร็วเกินไป (ไม่สมจริง)

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            velocity_vector: Optional - (vx, vy) velocity vector
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds

        Returns:
            tuple: (is_valid: bool, acceleration_score: float, reason: str)
        """
        from config import (
            ACCELERATION_CHECK_ENABLED, MAX_ACCELERATION_RATIO,
            MAX_DECELERATION_RATIO, ACCELERATION_CHECK_MIN_PATH_POINTS
        )

        # Early exit: ถ้าปิดการตรวจสอบ acceleration
        if not ACCELERATION_CHECK_ENABLED:
            return (True, 1.0, "Acceleration check disabled")

        # Early exit: ถ้าไม่มี velocity หรือ path history ไม่เพียงพอ
        if velocity_vector is None:
            return (True, 0.5, "No velocity vector")

        if 'centers_history' not in target or len(target['centers_history']) < ACCELERATION_CHECK_MIN_PATH_POINTS:
            return (True, 0.5, "Insufficient path history")

        vx, vy = velocity_vector
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return (True, 1.0, "No movement")

        # คำนวณ velocity จาก path history (2 จุดล่าสุด)
        centers_history = target['centers_history']
        if len(centers_history) < 2:
            return (True, 0.5, "Insufficient history")

        prev_frame, prev_center = centers_history[-2]
        curr_frame, curr_center = centers_history[-1]

        if prev_center is None or curr_center is None:
            return (True, 0.5, "Invalid history")

        # คำนวณ velocity จาก path history (pixels per frame)
        hist_dx = curr_center[0] - prev_center[0]
        hist_dy = curr_center[1] - prev_center[1]
        hist_velocity_mag = math.sqrt(hist_dx**2 + hist_dy**2)
        current_velocity_mag = math.sqrt(vx**2 + vy**2)

        # Early exit: ถ้า velocity ต่ำเกินไป (ไม่ต้องตรวจสอบ acceleration)
        if hist_velocity_mag < 0.1 or current_velocity_mag < 0.1:
            return (True, 1.0, "Low velocity")

        # คำนวณ acceleration ratio (rate of change of velocity)
        acceleration_ratio = (current_velocity_mag - hist_velocity_mag) / hist_velocity_mag

        # ตรวจสอบ acceleration/deceleration
        if acceleration_ratio > MAX_ACCELERATION_RATIO:
            return (False, 0.0, f"Acceleration too large ({acceleration_ratio:.2f} > {MAX_ACCELERATION_RATIO:.2f})")
        if acceleration_ratio < -MAX_DECELERATION_RATIO:
            return (False, 0.0, f"Deceleration too large ({abs(acceleration_ratio):.2f} > {MAX_DECELERATION_RATIO:.2f})")

        # คะแนนตาม acceleration consistency
        if acceleration_ratio > 0:
            accel_score = 1.0 - (acceleration_ratio / MAX_ACCELERATION_RATIO)
        else:
            accel_score = 1.0 - (abs(acceleration_ratio) / MAX_DECELERATION_RATIO)

        return (True, accel_score, f"accel={acceleration_ratio:.2f}")

    def _check_motion_box_freshness(self, motion_box, target, frame_counter, roi_box=None):
        """
        ตรวจสอบว่า motion box ยัง "fresh" หรือไม่ (ไม่ใช่ noise ที่ค้างอยู่)

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            frame_counter: Current frame counter
            roi_box: Optional - ROI box (x1, y1, x2, y2) สำหรับตรวจสอบ

        Returns:
            tuple: (is_fresh: bool, freshness_score: float, reason: str)
        """
        from config import (
            MOTION_BOX_FRESHNESS_CHECK_ENABLED, MOTION_BOX_MAX_AGE_FRAMES,
            MOTION_BOX_FRESHNESS_ROI_CHECK
        )

        # Early exit: ถ้าปิดการตรวจสอบ freshness
        if not MOTION_BOX_FRESHNESS_CHECK_ENABLED:
            return (True, 1.0, "Freshness check disabled")

        mx, my, mw, mh = motion_box
        motion_center = (mx + mw // 2, my + mh // 2)

        # ตรวจสอบว่า motion box อยู่ใน ROI หรือไม่ (ถ้าเปิดใช้งาน)
        if MOTION_BOX_FRESHNESS_ROI_CHECK and roi_box is not None:
            if len(roi_box) == 4:
                rx1, ry1, rx2, ry2 = roi_box
                if not (rx1 <= motion_center[0] <= rx2 and ry1 <= motion_center[1] <= ry2):
                    return (False, 0.0, "Motion box outside ROI")

        # ตรวจสอบว่า motion box ไม่เก่าเกินไป (ถ้ามี last_update_frame)
        if 'last_update_frame' in target:
            age = frame_counter - target['last_update_frame']
            if age > MOTION_BOX_MAX_AGE_FRAMES:
                return (False, 0.0, f"Motion box too old (age={age} > {MOTION_BOX_MAX_AGE_FRAMES})")

        return (True, 1.0, "Fresh")

    def _check_path_smoothness(self, motion_box, target, thresholds=None):
        """
        ตรวจสอบ path smoothness (ระยะกระโดด) เพื่อป้องกัน path กระโดดมากเกินไป

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds

        Returns:
            tuple: (is_smooth: bool, smoothness_score: float, reason: str)
        """
        from config import (
            PATH_SMOOTHNESS_CHECK_ENABLED, MAX_PATH_JUMP_RATIO, MAX_PATH_JUMP_RATIO_TINY
        )

        # Early exit: ถ้าปิดการตรวจสอบ path smoothness
        if not PATH_SMOOTHNESS_CHECK_ENABLED:
            return (True, 1.0, "Path smoothness check disabled")

        # Early exit: ถ้าไม่มี path history
        if 'centers_history' not in target or len(target['centers_history']) < 2:
            return (True, 0.5, "Insufficient path history")

        # ใช้ resolution-dependent thresholds
        if thresholds is None:
            thresholds = self.thresholds

        frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)

        # ตรวจสอบว่าเป็น tiny object หรือไม่ (ใช้ threshold ที่ยืดหยุ่นกว่า)
        is_tiny = self.is_tiny_object_for_target(target, self.frame_w, self.frame_h)
        max_jump_ratio = MAX_PATH_JUMP_RATIO_TINY if is_tiny else MAX_PATH_JUMP_RATIO
        max_jump = max_jump_ratio * frame_diagonal

        mx, my, mw, mh = motion_box
        motion_center = (mx + mw // 2, my + mh // 2)

        centers_history = target['centers_history']
        last_frame, last_center = centers_history[-1]

        if last_center is None:
            return (True, 0.5, "Invalid last center")

        dist_to_last = self.get_dist(motion_center, last_center)

        if dist_to_last > max_jump:
            return (False, 0.0, f"Path jump too large ({dist_to_last:.1f} > {max_jump:.1f})")

        # คะแนนตาม path smoothness (ใกล้ = score สูง)
        smoothness_score = 1.0 - (dist_to_last / max_jump)
        return (True, smoothness_score, f"jump={dist_to_last:.1f}")

    def _check_roi_iou_with_history(self, target, new_roi_box, thresholds=None):
        """
        ตรวจสอบว่า ROI ใหม่มี IOU กับ ROI ก่อนหน้าหรือไม่

        Args:
            target: Target dictionary
            new_roi_box: (x1, y1, x2, y2) ROI box ใหม่
            thresholds: Resolution-dependent thresholds

        Returns:
            (is_valid, iou_score, reason): (bool, float, str)
        """
        from config import (
            ROI_MIN_IOU_THRESHOLD, ROI_VALIDATION_MIN_HISTORY,
            ROI_TINY_IOU_THRESHOLD_MULTIPLIER, ROI_TINY_USE_PATH_FIRST,
            MOTION_ONLY_MIN_PATH_POINTS
        )

        if thresholds is None:
            thresholds = self.thresholds

        # ตรวจสอบว่าเป็น tiny object หรือไม่
        is_tiny = self.is_tiny_object_for_target(target, self.frame_w, self.frame_h)

        # Early exit: สำหรับ tiny objects - ใช้ path history แทน ROI history
        if 'roi_box_history' not in target or len(target['roi_box_history']) < ROI_VALIDATION_MIN_HISTORY:
            if is_tiny and ROI_TINY_USE_PATH_FIRST:
                # สำหรับ tiny objects: ตรวจสอบ path history แทน
                centers_history = target.get('centers_history', [])
                if len(centers_history) >= MOTION_ONLY_MIN_PATH_POINTS:
                    # มี path history เพียงพอ → ผ่าน (แต่ให้คะแนนต่ำกว่า)
                    return (True, 0.7, "Tiny object: Using path history instead")
                else:
                    # ไม่มี path history → ผ่านในเฟรมแรกๆ (progressive validation)
                    return (True, 0.5, "Tiny object: Initial frame (no history)")
            else:
                return (True, 1.0, "Insufficient ROI history")

        roi_history = target['roi_box_history']

        # ใช้ ROI ล่าสุดเป็นตัวเปรียบเทียบ (performance: ตรวจสอบแค่ 1 ROI)
        last_frame, last_roi_box, last_center = roi_history[-1]

        # คำนวณ IOU
        # แปลง ROI format: (x1,y1,x2,y2) → (x,y,w,h) สำหรับ calculate_iou
        last_rect = (last_roi_box[0], last_roi_box[1],
                     last_roi_box[2] - last_roi_box[0],
                     last_roi_box[3] - last_roi_box[1])
        new_rect = (new_roi_box[0], new_roi_box[1],
                    new_roi_box[2] - new_roi_box[0],
                    new_roi_box[3] - new_roi_box[1])

        iou = calculate_iou(last_rect, new_rect)

        # ใช้ resolution-dependent threshold (ปรับตาม frame size)
        # สำหรับ tiny objects: ลด IOU threshold
        if is_tiny:
            min_iou = ROI_MIN_IOU_THRESHOLD * ROI_TINY_IOU_THRESHOLD_MULTIPLIER  # ลดลง 50% (0.15 แทน 0.3)
        else:
            min_iou = ROI_MIN_IOU_THRESHOLD  # ใช้ threshold คงที่ (30%)

        if iou < min_iou:
            return (False, 0.0, f"IOU too low ({iou:.2f} < {min_iou:.2f})")

        # คะแนนตาม IOU (ยิ่งสูง = ดี)
        iou_score = min(1.0, iou / min_iou)  # normalize

        return (True, iou_score, f"IOU={iou:.2f}")

    def _check_roi_directional_movement(self, target, new_roi_box, thresholds=None):
        """
        ตรวจสอบว่า ROI เคลื่อนที่ในทิศทางคงที่หรือไม่

        Args:
            target: Target dictionary
            new_roi_box: (x1, y1, x2, y2) ROI box ใหม่
            thresholds: Resolution-dependent thresholds

        Returns:
            (is_valid, movement_score, reason): (bool, float, str)
        """
        from config import (
            ROI_MIN_MOVEMENT_THRESHOLD, ROI_MAX_DIRECTION_CHANGE,
            ROI_VALIDATION_MIN_HISTORY, ROI_TINY_MOVEMENT_THRESHOLD_MULTIPLIER,
            ROI_TINY_DIRECTION_CHANGE_MULTIPLIER, ROI_TINY_USE_PATH_FIRST,
            MOTION_ONLY_MIN_PATH_POINTS
        )

        if thresholds is None:
            thresholds = self.thresholds

        # ตรวจสอบว่าเป็น tiny object หรือไม่
        is_tiny = self.is_tiny_object_for_target(target, self.frame_w, self.frame_h)

        # Early exit: สำหรับ tiny objects - ใช้ path history
        if 'roi_box_history' not in target or len(target['roi_box_history']) < ROI_VALIDATION_MIN_HISTORY:
            if is_tiny and ROI_TINY_USE_PATH_FIRST:
                # สำหรับ tiny objects: ตรวจสอบ path history แทน
                centers_history = target.get('centers_history', [])
                if len(centers_history) >= MOTION_ONLY_MIN_PATH_POINTS:
                    return (True, 0.7, "Tiny object: Using path history")
                else:
                    return (True, 0.5, "Tiny object: Initial frame")
            else:
                return (True, 1.0, "Insufficient ROI history")

        roi_history = target['roi_box_history']

        # ใช้ ROI 2 ตัวล่าสุดเพื่อคำนวณทิศทาง
        if len(roi_history) < 2:
            return (True, 0.5, "Insufficient history for direction check")

        # ROI ก่อนหน้า (2 ตัวล่าสุด)
        prev_frame, prev_roi_box, prev_center = roi_history[-2]
        last_frame, last_roi_box, last_center = roi_history[-1]

        # คำนวณ center ของ ROI ใหม่
        new_center = ((new_roi_box[0] + new_roi_box[2]) // 2,
                      (new_roi_box[1] + new_roi_box[3]) // 2)

        # คำนวณทิศทางจาก ROI ก่อนหน้า → ROI ล่าสุด
        prev_dx = last_center[0] - prev_center[0]
        prev_dy = last_center[1] - prev_center[1]
        prev_angle = math.degrees(math.atan2(prev_dy, prev_dx))

        # คำนวณทิศทางจาก ROI ล่าสุด → ROI ใหม่
        curr_dx = new_center[0] - last_center[0]
        curr_dy = new_center[1] - last_center[1]
        curr_angle = math.degrees(math.atan2(curr_dy, curr_dx))

        # คำนวณระยะทางเคลื่อนที่
        movement_dist = math.sqrt(curr_dx**2 + curr_dy**2)
        frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)

        # สำหรับ tiny objects: ลด movement threshold ลงมาก
        if is_tiny:
            min_movement_ratio = ROI_MIN_MOVEMENT_THRESHOLD * ROI_TINY_MOVEMENT_THRESHOLD_MULTIPLIER  # ลดลง 90% (0.002 แทน 0.02)
            max_direction_change = ROI_MAX_DIRECTION_CHANGE * ROI_TINY_DIRECTION_CHANGE_MULTIPLIER  # เพิ่มขึ้น 50% (90° แทน 60°)
        else:
            min_movement_ratio = ROI_MIN_MOVEMENT_THRESHOLD
            max_direction_change = ROI_MAX_DIRECTION_CHANGE

        min_movement = min_movement_ratio * frame_diagonal

        # ตรวจสอบการเคลื่อนที่
        if movement_dist < min_movement:
            return (False, 0.0, f"Movement too small ({movement_dist:.1f} < {min_movement:.1f})")

        # คำนวณมุมที่เปลี่ยน (absolute difference)
        angle_diff = abs(curr_angle - prev_angle)
        # ปรับให้อยู่ในช่วง 0-180 องศา
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # ตรวจสอบทิศทาง
        if angle_diff > max_direction_change:
            return (False, 0.0, f"Direction change too large ({angle_diff:.1f}° > {max_direction_change}°)")

        # คะแนนตาม movement และ direction consistency
        movement_score = min(1.0, movement_dist / min_movement)  # normalize
        direction_score = 1.0 - (angle_diff / max_direction_change)  # ยิ่งสอดคล้อง = สูง

        combined_score = (movement_score * 0.6 + direction_score * 0.4)

        return (True, combined_score, f"movement={movement_dist:.1f}, angle_diff={angle_diff:.1f}°")

    def _validate_roi_for_drawing(self, target, new_roi_box, frame_counter, thresholds=None):
        """
        ตรวจสอบ ROI ก่อนวาด (รวม IOU และการเคลื่อนที่)

        Args:
            target: Target dictionary
            new_roi_box: (x1, y1, x2, y2) ROI box ใหม่
            frame_counter: Current frame counter
            thresholds: Resolution-dependent thresholds

        Returns:
            (is_valid, validation_score, reason): (bool, float, str)
        """
        from config import (
            ROI_VALIDATION_ENABLED, ROI_VALIDATION_SKIP_RED_ORANGE,
            ROI_TINY_USE_PATH_FIRST, ROI_TINY_PATH_QUALITY_THRESHOLD,
            MOTION_ONLY_MIN_PATH_POINTS, ROI_VALIDATION_SKIP_FIRST_FRAMES,
            ROI_VALIDATION_MOTION_ONLY_MIN_PATH
        )

        if not ROI_VALIDATION_ENABLED:
            return (True, 1.0, "Validation disabled")

        # Skip validation สำหรับ RED/ORANGE status (ถ้าตั้งค่าไว้)
        if ROI_VALIDATION_SKIP_RED_ORANGE:
            status = target.get('status', 'GREEN')
            if status in ['RED', 'ORANGE']:
                return (True, 1.0, f"Skip validation for {status} status")

        if thresholds is None:
            thresholds = self.thresholds

        # สำหรับ motion-only targets: ยืดหยุ่นขึ้น (วาดได้เร็วขึ้น)
        is_motion_only = target.get('is_motion_only', False)
        if is_motion_only:
            centers_history = target.get('centers_history', [])
            created_frame = target.get('created_frame', frame_counter)
            frames_since_creation = frame_counter - created_frame

            # ข้าม validation ใน N เฟรมแรก (ให้วาดได้ทันที)
            if frames_since_creation < ROI_VALIDATION_SKIP_FIRST_FRAMES:
                if len(centers_history) >= ROI_VALIDATION_MOTION_ONLY_MIN_PATH:
                    return (True, 0.7, f"Motion-only: First {ROI_VALIDATION_SKIP_FIRST_FRAMES} frames, path={len(centers_history)}")

            # ถ้ามี path history เพียงพอ → ผ่าน validation
            if len(centers_history) >= ROI_VALIDATION_MOTION_ONLY_MIN_PATH:
                return (True, 0.8, f"Motion-only: Path history={len(centers_history)} points")

        # ตรวจสอบว่าเป็น tiny object หรือไม่
        is_tiny = self.is_tiny_object_for_target(target, self.frame_w, self.frame_h)

        # สำหรับ tiny objects: ใช้ path-first validation
        if is_tiny and ROI_TINY_USE_PATH_FIRST:
            centers_history = target.get('centers_history', [])

            # ถ้ามี path history เพียงพอ → ใช้ path-based validation
            if len(centers_history) >= MOTION_ONLY_MIN_PATH_POINTS:
                # ตรวจสอบ path quality
                is_clear, path_quality = self.calculate_motion_path_quality(
                    target, frame_counter, is_motion_only=True
                )

                # ถ้า path quality ดี → ผ่าน validation
                if path_quality >= ROI_TINY_PATH_QUALITY_THRESHOLD:
                    return (True, path_quality, f"Tiny object: Path quality={path_quality:.2f}")
                else:
                    return (False, path_quality, f"Tiny object: Low path quality={path_quality:.2f}")
            else:
                # ยังไม่มี path history เพียงพอ → ผ่านในเฟรมแรกๆ
                return (True, 0.6, "Tiny object: Building path history")

        validation_score = 0.0
        reasons = []

        # 1. ตรวจสอบ IOU
        is_valid_iou, iou_score, iou_reason = self._check_roi_iou_with_history(
            target, new_roi_box, thresholds
        )

        if not is_valid_iou:
            return (False, 0.0, f"IOU check failed: {iou_reason}")

        validation_score += iou_score * 0.5  # Weight: 50%
        reasons.append(iou_reason)

        # 2. ตรวจสอบการเคลื่อนที่
        is_valid_movement, movement_score, movement_reason = self._check_roi_directional_movement(
            target, new_roi_box, thresholds
        )

        if not is_valid_movement:
            return (False, 0.0, f"Movement check failed: {movement_reason}")

        validation_score += movement_score * 0.5  # Weight: 50%
        reasons.append(movement_reason)

        reason_str = ", ".join(reasons)
        return (True, validation_score, reason_str)

    def _validate_motion_box_basic(self, motion_box, target, predicted_center=None,
                                   velocity_vector=None, last_rect=None, thresholds=None):
        """
        ตรวจสอบ motion box แบบ basic (predicted position, previous position, direction, size)
        ก่อนเพิ่มเข้าไปใน path เพื่อป้องกัน noise จากพื้นหลัง

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            predicted_center: Optional - (x, y) predicted center position
            velocity_vector: Optional - (vx, vy) velocity vector
            last_rect: Optional - (x, y, w, h) ขนาด target ก่อนหน้า
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds

        Returns:
            tuple: (is_valid: bool, validation_score: float, reason: str)
                - is_valid: True ถ้า motion box valid
                - validation_score: คะแนน validation (0.0-1.0, สูง=valid)
                - reason: เหตุผลที่ valid/invalid
        """
        from config import (
            PATH_VALIDATION_ENABLED, OUTLIER_DISTANCE_THRESHOLD_RATIO,
            OUTLIER_DIRECTION_THRESHOLD, OUTLIER_SIZE_THRESHOLD_RATIO
        )

        # Early exit: ถ้าปิดการ validation
        if not PATH_VALIDATION_ENABLED:
            return (True, 1.0, "Validation disabled")

        # ใช้ resolution-dependent thresholds
        if thresholds is None:
            thresholds = self.thresholds

        frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
        max_dist_threshold = thresholds.get('HYBRID_DIST_THRESHOLD', HYBRID_DIST_THRESHOLD)

        mx, my, mw, mh = motion_box
        motion_center = (mx + mw // 2, my + mh // 2)

        validation_score = 0.0
        reasons = []

        # 1. ตรวจสอบ predicted position (ถ้ามี)
        if predicted_center is not None:
            dist_to_predicted = self.get_dist(motion_center, predicted_center)
            outlier_distance_threshold = OUTLIER_DISTANCE_THRESHOLD_RATIO * frame_diagonal

            if dist_to_predicted > outlier_distance_threshold:
                return (False, 0.0, f"Too far from predicted position ({dist_to_predicted:.1f} > {outlier_distance_threshold:.1f})")

            # คะแนนตามระยะห่างจาก predicted position (ใกล้ = score สูง)
            normalized_dist = dist_to_predicted / max(1.0, outlier_distance_threshold)
            predicted_score = 1.0 - normalized_dist
            validation_score += predicted_score * 0.4  # Weight: 40%
            reasons.append(f"predicted_dist={dist_to_predicted:.1f}")

        # 2. ตรวจสอบ previous position (ถ้ามี)
        last_center = target.get('last_center')
        if last_center is not None:
            dist_to_previous = self.get_dist(motion_center, last_center)

            if dist_to_previous > max_dist_threshold * 2.0:  # อนุญาตให้ห่างได้ 2 เท่าของ threshold
                return (False, 0.0, f"Too far from previous position ({dist_to_previous:.1f} > {max_dist_threshold * 2.0:.1f})")

            # คะแนนตามระยะห่างจาก previous position (ใกล้ = score สูง)
            normalized_dist = dist_to_previous / max(1.0, max_dist_threshold)
            previous_score = 1.0 - min(1.0, normalized_dist)
            validation_score += previous_score * 0.3  # Weight: 30%
            reasons.append(f"previous_dist={dist_to_previous:.1f}")

        # 3. ตรวจสอบ direction consistency (ถ้ามี velocity)
        if velocity_vector is not None and last_center is not None:
            vx, vy = velocity_vector
            if abs(vx) > 0.01 or abs(vy) > 0.01:  # มีการเคลื่อนที่
                # คำนวณทิศทางจาก previous ไปยัง motion center
                dx = motion_center[0] - last_center[0]
                dy = motion_center[1] - last_center[1]

                # คำนวณมุมระหว่าง velocity vector และ direction vector
                velocity_mag = math.sqrt(vx**2 + vy**2)
                direction_mag = math.sqrt(dx**2 + dy**2)

                if velocity_mag > 0.1 and direction_mag > 0.1:
                    dot_product = (vx * dx + vy * dy) / (velocity_mag * direction_mag)
                    dot_product = max(-1.0, min(1.0, dot_product))  # Clamp
                    angle_deg = math.degrees(math.acos(dot_product))

                    if angle_deg > OUTLIER_DIRECTION_THRESHOLD:
                        return (False, 0.0, f"Direction mismatch ({angle_deg:.1f}° > {OUTLIER_DIRECTION_THRESHOLD}°)")

                    # คะแนนตาม direction alignment (สอดคล้อง = score สูง)
                    direction_score = 1.0 - (angle_deg / OUTLIER_DIRECTION_THRESHOLD)
                    validation_score += direction_score * 0.2  # Weight: 20%
                    reasons.append(f"direction_angle={angle_deg:.1f}°")

        # 4. ตรวจสอบ size consistency (ถ้ามี last_rect)
        if last_rect is not None:
            lx, ly, lw, lh = last_rect
            last_area = lw * lh
            motion_area = mw * mh

            if last_area > 0:
                size_ratio = motion_area / last_area
                outlier_size_min = OUTLIER_SIZE_THRESHOLD_RATIO
                outlier_size_max = 1.0 / OUTLIER_SIZE_THRESHOLD_RATIO

                if size_ratio < outlier_size_min or size_ratio > outlier_size_max:
                    return (False, 0.0, f"Size mismatch (ratio={size_ratio:.2f}, expected {outlier_size_min:.2f}-{outlier_size_max:.2f})")

                # คะแนนตาม size similarity (ใกล้เคียง = score สูง)
                if size_ratio < 1.0:
                    size_score = size_ratio / outlier_size_min
                else:
                    size_score = outlier_size_max / size_ratio
                validation_score += size_score * 0.1  # Weight: 10%
                reasons.append(f"size_ratio={size_ratio:.2f}")

        reason_str = ", ".join(reasons) if reasons else "valid"
        return (True, validation_score, reason_str)

    def _validate_motion_box_for_path(self, motion_box, target, predicted_center=None,
                                       velocity_vector=None, last_rect=None, thresholds=None):
        """
        ตรวจสอบ motion box ก่อนเพิ่มเข้าไปใน path (รวม temporal continuity check - adaptive)

        Args:
            motion_box: (x, y, w, h) motion box ที่ต้องการตรวจสอบ
            target: Target dictionary
            predicted_center: Optional - (x, y) predicted center position
            velocity_vector: Optional - (vx, vy) velocity vector
            last_rect: Optional - (x, y, w, h) ขนาด target ก่อนหน้า
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds

        Returns:
            tuple: (is_valid: bool, validation_score: float, reason: str)
                - is_valid: True ถ้า motion box valid
                - validation_score: คะแนน validation (0.0-1.0, สูง=valid)
                - reason: เหตุผลที่ valid/invalid
        """
        from config import PATH_VALIDATION_ENABLED, MIN_MOTION_SCORE_THRESHOLD, TEMPORAL_CONTINUITY_MIN_PATH_POINTS

        # Early exit: ถ้าปิดการ validation
        if not PATH_VALIDATION_ENABLED:
            return (True, 1.0, "Validation disabled")

        # 1. Basic validation (predicted position, previous position, direction, size)
        is_valid_basic, validation_score_basic, reason_basic = self._validate_motion_box_basic(
            motion_box, target, predicted_center, velocity_vector, last_rect, thresholds
        )

        if not is_valid_basic:
            return (False, validation_score_basic, reason_basic)

        # 2. Motion box freshness check
        current_frame = getattr(self, '_current_frame_counter', 0)
        roi_box = target.get('roi_box')
        is_fresh, freshness_score, freshness_reason = self._check_motion_box_freshness(
            motion_box, target, current_frame, roi_box
        )

        if not is_fresh:
            return (False, freshness_score, f"Motion box freshness failed: {freshness_reason}")

        # 3. Temporal continuity check (adaptive ตาม path history)
        # ถ้ามี path history เพียงพอ → ใช้ temporal continuity check แบบเต็ม
        # ถ้ายังไม่มี path history เพียงพอ → ใช้ validation แบบเบา (ไม่ตรวจสอบ temporal continuity)
        has_sufficient_history = ('centers_history' in target and
                                  len(target['centers_history']) >= TEMPORAL_CONTINUITY_MIN_PATH_POINTS)

        if has_sufficient_history:
            # ใช้ temporal continuity check แบบเต็ม
            is_continuous, continuity_score, reason_continuous = self._check_temporal_continuity(
                motion_box, target, predicted_center, velocity_vector, last_rect, thresholds
            )

            if not is_continuous:
                return (False, continuity_score, f"Temporal continuity failed: {reason_continuous}")

            # รวมคะแนน (basic validation 60%, temporal continuity 25%, freshness 15%)
            final_score = validation_score_basic * 0.6 + continuity_score * 0.25 + freshness_score * 0.15
        else:
            # ใช้ validation แบบเบา (ไม่ตรวจสอบ temporal continuity)
            final_score = validation_score_basic * 0.7 + freshness_score * 0.3
            reason_continuous = "insufficient history"

        if final_score < MIN_MOTION_SCORE_THRESHOLD:
            return (False, final_score, f"Combined score too low ({final_score:.2f} < {MIN_MOTION_SCORE_THRESHOLD:.2f})")

        reason_combined = f"{reason_basic}, {freshness_reason}, {reason_continuous}"
        return (True, final_score, reason_combined)

    def _smooth_path_points(self, path_points, window_size=None):
        """
        Smooth path points โดยใช้ moving average เพื่อลด noise (เฉพาะสำหรับการวาด)
        ไม่ใช้ในการคำนวณ path quality, velocity, หรือการวิเคราะห์อื่นๆ

        Args:
            path_points: List of (x, y) tuples
            window_size: ขนาดของ window สำหรับ moving average (ถ้า None ใช้จาก config)

        Returns:
            list: smoothed path points [(x, y), ...] หรือ path_points เดิมถ้าปิด smoothing
        """
        from config import PATH_SMOOTHING_ENABLED, PATH_SMOOTHING_WINDOW_SIZE

        # Early exit: ถ้าปิดการ smoothing หรือมีจุดน้อยเกินไป
        if not PATH_SMOOTHING_ENABLED or len(path_points) < 2:
            return path_points.copy()

        if window_size is None:
            window_size = PATH_SMOOTHING_WINDOW_SIZE

        # Early exit: ถ้ามีจุดน้อยกว่า window size → ไม่ต้อง smooth
        if len(path_points) < window_size:
            return path_points.copy()

        # Moving average smoothing (O(n) - เร็วมาก)
        smoothed = []
        half_window = window_size // 2

        for i in range(len(path_points)):
            # คำนวณ window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(path_points), i + half_window + 1)

            # คำนวณ average ของจุดใน window
            window_points = path_points[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            smoothed.append((int(avg_x), int(avg_y)))

        return smoothed

    def get_velocity_from_path_history(self, centers_history, lookback=None, use_smoothing=False, cached_velocity=None, processing_fps=None):
        """
        คำนวณ velocity vector จาก path history ของ target (ใช้ smoothed path และ processing FPS)

        Args:
            centers_history: List of (frame, center) tuples จาก target
            lookback: จำนวนจุดที่ใช้สำหรับคำนวณ velocity (default: ใช้ PREDICTED_POSITION_LOOKBACK จาก config)
            use_smoothing: ถ้า True ใช้ exponential smoothing สำหรับ velocity (default: False)
            cached_velocity: Optional - cached velocity vector สำหรับ smoothing (vx, vy)
            processing_fps: Optional - processing FPS (ใช้สำหรับแปลง frame counter เป็น time)

        Returns:
            (vx, vy): velocity vector (pixels per second) หรือ (0.0, 0.0) ถ้าไม่มีข้อมูลเพียงพอ
        """
        from config import PATH_SMOOTHING_ENABLED, PREDICTED_POSITION_LOOKBACK, DEFAULT_FPS

        if not centers_history or len(centers_history) < 2:
            return (0.0, 0.0)

        # ใช้ processing FPS จาก parameter หรือจาก instance variable
        if processing_fps is None:
            processing_fps = getattr(self, '_current_processing_fps', None)
            if processing_fps is None or processing_fps <= 0:
                processing_fps = DEFAULT_FPS

        # ใช้ lookback จาก config ถ้าไม่ระบุ
        if lookback is None:
            lookback = PREDICTED_POSITION_LOOKBACK

        # ใช้หลายจุดล่าสุด (lookback จุด) เพื่อความแม่นยำ
        recent_points = centers_history[-lookback:] if len(centers_history) >= lookback else centers_history

        if len(recent_points) < 2:
            return (0.0, 0.0)

        # แปลงเป็น list of points และ smooth ถ้าเปิดใช้งาน
        path_points = []
        frame_numbers = []
        for frame, center in recent_points:
            if center:
                path_points.append(center)
                frame_numbers.append(frame)

        if len(path_points) < 2:
            return (0.0, 0.0)

        # ใช้ smoothed path ถ้าเปิดใช้งาน
        if PATH_SMOOTHING_ENABLED and len(path_points) >= 3:
            path_points = self._smooth_path_points(path_points)

        # คำนวณ velocity จาก smoothed path (ใช้ processing FPS)
        velocities = []
        total_weight = 0.0

        for i in range(1, len(path_points)):
            prev_center = path_points[i-1]
            curr_center = path_points[i]
            prev_frame = frame_numbers[i-1]
            curr_frame = frame_numbers[i]

            if prev_center is None or curr_center is None:
                continue

            frames_diff = curr_frame - prev_frame
            if frames_diff <= 0:
                continue

            # แปลง frame difference เป็น time (seconds) โดยใช้ processing FPS
            time_diff = frames_diff / processing_fps if processing_fps > 0 else frames_diff

            # คำนวณ velocity (pixels per second)
            dx = (curr_center[0] - prev_center[0]) / time_diff
            dy = (curr_center[1] - prev_center[1]) / time_diff

            # ตรวจสอบว่า velocity ไม่เป็น NaN หรือ Inf
            if math.isnan(dx) or math.isnan(dy) or math.isinf(dx) or math.isinf(dy):
                continue

            # ใช้ weight ตาม time difference (time ที่ใกล้เคียงกันมาก = weight สูง)
            weight = 1.0 / max(0.001, time_diff)  # ใช้ 0.001 เพื่อป้องกัน division by zero
            velocities.append((dx, dy, weight))
            total_weight += weight

        if not velocities:
            return (0.0, 0.0)

        # คำนวณ weighted average velocity (pixels per second)
        avg_vx = sum(vx * w for vx, dy, w in velocities) / total_weight
        avg_vy = sum(vy * w for vx, vy, w in velocities) / total_weight

        # ตรวจสอบว่า velocity ไม่เป็น NaN หรือ Inf
        if math.isnan(avg_vx) or math.isnan(avg_vy) or math.isinf(avg_vx) or math.isinf(avg_vy):
            return (0.0, 0.0)

        # ใช้ exponential smoothing ถ้าต้องการ
        if use_smoothing and cached_velocity is not None:
            # ใช้ default alpha = 0.7 ถ้าไม่มี config
            try:
                from config import VELOCITY_SMOOTHING_ALPHA
                alpha = VELOCITY_SMOOTHING_ALPHA
            except ImportError:
                alpha = 0.7  # Default smoothing alpha

            # ตรวจสอบว่า cached_velocity เป็น tuple (vx, vy) หรือ float (velocity magnitude)
            if isinstance(cached_velocity, (tuple, list)) and len(cached_velocity) == 2:
                old_vx, old_vy = cached_velocity
                # ตรวจสอบว่า cached velocity ไม่เป็น NaN หรือ Inf
                if not (math.isnan(old_vx) or math.isnan(old_vy) or math.isinf(old_vx) or math.isinf(old_vy)):
                    # ตรวจสอบว่า velocity เปลี่ยนทิศทางมากเกินไป (> 90 องศา) หรือไม่
                    velocity_mag = math.sqrt(avg_vx**2 + avg_vy**2)
                    old_velocity_mag = math.sqrt(old_vx**2 + old_vy**2)

                    if velocity_mag > 0.1 and old_velocity_mag > 0.1:
                        # คำนวณมุมระหว่าง velocity vectors
                        dot_product = (avg_vx * old_vx + avg_vy * old_vy) / (velocity_mag * old_velocity_mag)
                        # Clamp dot_product เพื่อป้องกัน NaN
                        dot_product = max(-1.0, min(1.0, dot_product))
                        angle_rad = math.acos(dot_product)
                        angle_deg = math.degrees(angle_rad)

                        # ถ้าเปลี่ยนทิศทางมากเกิน 90 องศา: ใช้ velocity ที่ smooth กว่า (alpha ต่ำกว่า)
                        if angle_deg > 90:
                            alpha = alpha * 0.5  # ใช้ smoothing ที่เข้มข้นขึ้น

                    # Exponential smoothing
                    smooth_vx = alpha * avg_vx + (1.0 - alpha) * old_vx
                    smooth_vy = alpha * avg_vy + (1.0 - alpha) * old_vy
                    return (smooth_vx, smooth_vy)
                else:
                    # cached_velocity ไม่ valid (NaN/Inf) → ใช้ velocity ใหม่โดยไม่ smooth
                    return (avg_vx, avg_vy)
            else:
                # cached_velocity เป็น float หรือไม่ใช่ tuple → ใช้ velocity ใหม่โดยไม่ smooth
                return (avg_vx, avg_vy)

        return (avg_vx, avg_vy)

    def predict_next_position(self, centers_history, num_points=3):
        """
        ทำนายตำแหน่งถัดไปจาก path history โดยใช้ linear prediction

        Args:
            centers_history: List of (frame, center) tuples
            num_points: จำนวนจุดที่ใช้สำหรับ prediction (default: 3)

        Returns:
            (predicted_x, predicted_y) หรือ None ถ้าไม่มีข้อมูลเพียงพอ
        """
        if not centers_history or len(centers_history) < 2:
            return None

        # ใช้แค่ num_points จุดล่าสุด
        recent_points = centers_history[-num_points:] if len(centers_history) >= num_points else centers_history

        if len(recent_points) < 2:
            return None

        # คำนวณ velocity vector จาก 2 จุดล่าสุด
        last_frame, last_center = recent_points[-1]
        prev_frame, prev_center = recent_points[-2]

        if last_center is None or prev_center is None:
            return None

        frames_diff = last_frame - prev_frame
        if frames_diff <= 0:
            return None

        # คำนวณ velocity (pixels per frame)
        dx = (last_center[0] - prev_center[0]) / frames_diff
        dy = (last_center[1] - prev_center[1]) / frames_diff

        # ทำนายตำแหน่งถัดไป (1 เฟรมถัดไป)
        predicted_x = last_center[0] + dx
        predicted_y = last_center[1] + dy

        return (predicted_x, predicted_y)

    def calculate_path_consistency(self, center, centers_history, predicted_center=None):
        """
        คำนวณความสอดคล้องของ center ใหม่กับ path เดิม

        Args:
            center: (x, y) center ของ detection ใหม่
            centers_history: List of (frame, center) tuples
            predicted_center: Predicted position (ถ้ามี) หรือ None เพื่อคำนวณใหม่

        Returns:
            consistency_score (0.0-1.0): สูง = สอดคล้องมาก, ต่ำ = ไม่สอดคล้อง
        """
        if not centers_history or len(centers_history) < 2:
            return 0.5  # Default score ถ้าไม่มีข้อมูล

        # คำนวณ predicted position ถ้ายังไม่มี
        if predicted_center is None:
            predicted_center = self.predict_next_position(centers_history)

        if predicted_center is None:
            return 0.5  # Default score

        # คำนวณ distance จาก center ใหม่ไปยัง predicted position
        dist_to_predicted = self.get_dist(center, predicted_center)

        # คำนวณ distance จาก last_center
        last_frame, last_center = centers_history[-1]
        if last_center is None:
            return 0.5

        dist_to_last = self.get_dist(center, last_center)

        # คำนวณ consistency score
        # ถ้าใกล้ predicted position = สอดคล้องมาก (score สูง)
        # ถ้าไกล predicted position = ไม่สอดคล้อง (score ต่ำ)

        # ใช้ threshold 40 pixels สำหรับ predicted distance
        predicted_threshold = 40.0
        if dist_to_predicted <= predicted_threshold:
            # ใกล้ predicted position = สอดคล้องมาก
            consistency = 1.0 - (dist_to_predicted / predicted_threshold) * 0.5
        else:
            # ไกล predicted position = ไม่สอดคล้อง
            consistency = 0.5 - min(0.4, (dist_to_predicted - predicted_threshold) / 100.0)

        # ปรับด้วย distance to last_center (ถ้าไกล last_center มาก = ไม่สอดคล้อง)
        last_threshold = 60.0
        if dist_to_last > last_threshold:
            # ลด consistency ถ้าไกล last_center มาก
            consistency *= 0.5

        return max(0.0, min(1.0, consistency))

    def is_likely_insect_in_roi(self, yolo_center, target):
        """
        ตรวจสอบว่า YOLO detection ใน ROI เป็นแมลงหรือไม่

        Args:
            yolo_center: (x, y) center ของ YOLO detection
            target: Target dictionary

        Returns:
            bool: True ถ้าเป็นแมลง, False ถ้าไม่ใช่
        """
        tracking_duration = target.get('tracking_duration', 0)
        centers_history = target.get('centers_history', [])

        # ต้องมี history อย่างน้อย 2 จุด ถึงจะประเมินได้
        if not centers_history or len(centers_history) < 2:
            return False

        # คำนวณค่าเบื้องต้นที่ใช้ร่วมกัน
        consistency_score = self.calculate_path_consistency(yolo_center, centers_history)
        last_frame, last_center = centers_history[-1]
        dist_to_last = self.get_dist(yolo_center, last_center) if last_center else 0.0

        # 1) ช่วงเริ่มติดตาม (สั้นมาก) < 5 เฟรม
        #    - ระวัง false positive จากแมลง / noise ที่เพิ่งเข้ามา
        if tracking_duration < 5:
            # ถ้า path ไม่สอดคล้องมากหรือตำแหน่งกระโดดแรง → ถือว่าเป็นแมลง
            if consistency_score < 0.4 or dist_to_last > 60.0:
                return True
            return False

        # 2) ช่วงกลาง 5–14 เฟรม
        #    - ใช้ path consistency + sudden change เป็นหลัก
        if tracking_duration < 15:
            if consistency_score < 0.35:
                return True
            if dist_to_last > 60.0:
                return True
            return False

        # 3) ช่วงติดตามนาน (>= 15 เฟรม)
        #    - ใช้ตัวตรวจจับแมลงเต็มรูปแบบ (velocity + straightness + continuity)
        current_frame = getattr(self, '_current_frame_counter', None)
        if self.is_likely_insect_for_target(target, current_frame):
            return True

        return False

    def add_to_blacklist(self, bbox):
        """
        เพิ่มพื้นที่ในเฟรมเข้า blacklist

        Args:
            bbox: (x1, y1, x2, y2) หรือ (x, y, w, h) ของวัตถุ
        """
        if not STATIONARY_DETECTION_ENABLED:
            return

        current_time = time.time()

        # แปลง bbox เป็น (x1, y1, x2, y2) ถ้าจำเป็น
        if len(bbox) == 4:
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                # เป็น (x, y, w, h)
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                # เป็น (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
        else:
            return

        # เพิ่ม padding รอบ bbox (ใช้ resolution-dependent threshold)
        padding = self.thresholds.get('BLACKLIST_BBOX_PADDING', BLACKLIST_BBOX_PADDING)
        x1_new = max(0, x1 - padding)
        y1_new = max(0, y1 - padding)
        x2_new = min(self.frame_w, x2 + padding)
        y2_new = min(self.frame_h, y2 + padding)

        bbox_with_padding = (x1_new, y1_new, x2_new, y2_new)

        # ตรวจสอบ permanent blacklist
        if PERMANENT_BLACKLIST_ENABLED:
            overlapping_entry = self.find_overlapping_blacklist(bbox_with_padding, 'permanent')
            if overlapping_entry:
                # เพิ่ม count และอัปเดต timestamp
                overlapping_entry['count'] += 1
                overlapping_entry['timestamp'] = current_time
                if DEBUG_MODE:
                    print(f"🚫 Updated permanent blacklist entry: count={overlapping_entry['count']}, bbox=({x1_new}, {y1_new}, {x2_new}, {y2_new})")
                return

        # ตรวจสอบ blacklist ระดับปกติ
        overlapping_entry = self.find_overlapping_blacklist(bbox_with_padding, 'normal')
        if overlapping_entry:
            # เพิ่ม count และอัปเดต timestamp
            overlapping_entry['count'] += 1
            overlapping_entry['timestamp'] = current_time

            # ตรวจสอบ Permanent Blacklist Threshold
            if PERMANENT_BLACKLIST_ENABLED:
                time_since_first = current_time - overlapping_entry['first_seen']
                if (overlapping_entry['count'] >= PERMANENT_BLACKLIST_THRESHOLD and
                    time_since_first <= PERMANENT_BLACKLIST_WINDOW_SECONDS):
                    # ย้ายไป permanent blacklist
                    permanent_entry = {
                        'bbox': overlapping_entry['bbox'],
                        'timestamp': current_time,
                        'count': overlapping_entry['count'],
                        'first_seen': overlapping_entry['first_seen']
                    }
                    self.permanent_blacklist.append(permanent_entry)
                    self.blacklist.remove(overlapping_entry)
                    if DEBUG_MODE:
                        print(f"🔄 Moved to permanent blacklist: count={permanent_entry['count']}, bbox={permanent_entry['bbox']}")
                    return

            if DEBUG_MODE:
                print(f"🚫 Updated blacklist entry: count={overlapping_entry['count']}, bbox=({x1_new}, {y1_new}, {x2_new}, {y2_new})")
        else:
            # สร้าง entry ใหม่
            new_entry = {
                'bbox': bbox_with_padding,
                'timestamp': current_time,
                'count': 1,
                'first_seen': current_time
            }
            self.blacklist.append(new_entry)
            if DEBUG_MODE:
                print(f"🚫 Added to blacklist: bbox=({x1_new}, {y1_new}, {x2_new}, {y2_new}) for {BLACKLIST_DURATION_SECONDS} seconds")

    def check_stationary(self, target_id, target, frame_counter):
        """
        ตรวจสอบว่าวัตถุนิ่งมากหรือไม่ (รวมกรณีกล่องยืด/หดอยู่ที่เดิม)

        Args:
            target_id: ID ของ target
            target: Target dictionary
            frame_counter: Frame counter ปัจจุบัน

        Returns:
            True ถ้าวัตถุนิ่งมาก, False ถ้าไม่นิ่ง
        """
        if not STATIONARY_DETECTION_ENABLED:
            return False

        # เก็บประวัติตำแหน่ง
        if 'centers_history' not in target:
            target['centers_history'] = []

        # เพิ่มตำแหน่งปัจจุบัน (ใช้ฟังก์ชัน optimize แทน)
        if target.get('last_center'):
            self.update_path_history(target, target['last_center'], frame_counter)

        history = target['centers_history']
        history_len = len(history)

        # ตรวจสอบว่า history มีข้อมูลหรือไม่ (ป้องกัน IndexError)
        if history_len == 0:
            return False  # ไม่มีประวัติ → ไม่สามารถตรวจสอบได้

        # Early blacklist detection: ตรวจสอบเร็วขึ้นเมื่อเริ่ม ROI
        from config import EARLY_BLACKLIST_CHECK_FRAMES, EARLY_BLACKLIST_STATIONARY_THRESHOLD_RATIO

        # ตรวจสอบว่า target ถูกสร้างเมื่อไหร่
        tracking_start_frame = target.get('tracking_start_frame', frame_counter)
        tracking_duration = frame_counter - tracking_start_frame

        # ใช้ early check frames สำหรับ target ใหม่ (เริ่ม ROI)
        use_early_check = tracking_duration < STATIONARY_CHECK_FRAMES * 2  # ใช้ early check ใน 2 เท่าของ normal check frames
        check_frames = EARLY_BLACKLIST_CHECK_FRAMES if use_early_check else STATIONARY_CHECK_FRAMES

        # ต้องมีประวัติอย่างน้อย check_frames เพื่อยืนยันว่า stationary จริง
        # แต่ถ้ามีอย่างน้อยครึ่งหนึ่งแล้ว สามารถ mark เป็น stationary_suspect ได้ก่อน
        half_frames = max(1, check_frames // 2)

        # คำนวณระยะทางที่เคลื่อนที่ในช่วงเวลานี้ (ใช้ทั้งสำหรับ suspect และยืนยันนิ่ง)
        first_frame, first_center = history[0]
        last_frame, last_center = history[-1]

        total_distance = 0.0
        max_distance = 0.0
        prev_center = None
        for _, center in history:
            if prev_center is not None and center is not None:
                step_dist = self.get_dist(prev_center, center)
                total_distance += step_dist
                max_distance = max(max_distance, step_dist, max_distance)
            prev_center = center

        # ค่า fallback ถ้า first/last เป็น None
        head_tail_distance = (
            self.get_dist(first_center, last_center)
            if first_center is not None and last_center is not None else total_distance
        )

        # ค่าประมาณความนิ่ง (เล็ก = นิ่งมาก)
        movement_score = max(head_tail_distance, max_distance)

        # ใช้ resolution-dependent thresholds
        max_stationary_dist = self.thresholds.get('MAX_STATIONARY_DISTANCE', MAX_STATIONARY_DISTANCE)
        stationary_center_thresh = self.thresholds.get('STATIONARY_CENTER_THRESHOLD', STATIONARY_CENTER_THRESHOLD)

        # ใช้ early threshold สำหรับ early detection (หลวมกว่าเล็กน้อย)
        if use_early_check:
            frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
            early_threshold = EARLY_BLACKLIST_STATIONARY_THRESHOLD_RATIO * frame_diagonal
            # ใช้ threshold ที่ใหญ่กว่าเล็กน้อยสำหรับ early detection
            max_stationary_dist = max(max_stationary_dist, early_threshold)
            stationary_center_thresh = max(stationary_center_thresh, early_threshold)

        # --- ระดับสงสัย (stationary_suspect) ---
        # ถ้ามี history อย่างน้อยครึ่งหนึ่งของ STATIONARY_CHECK_FRAMES และ movement ต่ำ
        if history_len >= half_frames:
            # ใช้ threshold หลวมกว่าเล็กน้อยสำหรับสงสัย
            suspect_threshold = max(max_stationary_dist * 1.5, stationary_center_thresh * 1.5)
            if movement_score <= suspect_threshold:
                target['stationary_suspect'] = True
            else:
                # ถ้ามีการเคลื่อนที่มากขึ้นระหว่างทาง ให้ยกเลิกสถานะสงสัย
                target['stationary_suspect'] = False

        # ถ้ายังมี history ไม่ถึง check_frames → ยังไม่ยืนยันว่า stationary
        if history_len < check_frames:
            return False

        # --- ระดับยืนยัน stationary (คืนค่า True → นำไป blacklist) ---
        if first_center and last_center:
            # แบบที่ 1: ตรวจสอบระยะทางรวม
            distance = self.get_dist(first_center, last_center)
            if distance <= max_stationary_dist:
                # ⚠️ NEW: ตรวจสอบ velocity profile (velocity = 0 หรือต่ำมาก → stationary)
                velocities = []
                for i in range(1, len(history)):
                    if i < len(history) and history[i-1][1] is not None and history[i][1] is not None:
                        vel = self.get_dist(history[i-1][1], history[i][1])
                        velocities.append(vel)

                if velocities:
                    mean_velocity = sum(velocities) / len(velocities)
                    max_velocity = max(velocities)
                    # ถ้า mean velocity และ max velocity ต่ำมาก → stationary
                    frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
                    min_velocity_threshold = self.thresholds.get('BLACKLIST_MIN_VELOCITY_THRESHOLD',
                                                                 BLACKLIST_MIN_VELOCITY_RATIO * frame_diagonal)
                    if mean_velocity < min_velocity_threshold and max_velocity < min_velocity_threshold * 2:
                        target['stationary_suspect'] = True
                        if DEBUG_MODE:
                            print(f"🚫 Target {target_id} is stationary (total distance={distance:.2f} pixels, "
                                  f"mean_velocity={mean_velocity:.2f} < {min_velocity_threshold:.2f} in {last_frame - first_frame} frames)")
                        return True
                else:
                    # ไม่มี velocity history → stationary
                    target['stationary_suspect'] = True
                    if DEBUG_MODE:
                        print(f"🚫 Target {target_id} is stationary (total distance={distance:.2f} pixels in {last_frame - first_frame} frames)")
                    return True

            # แบบที่ 2: ตรวจสอบกรณียืด/หด - หา center point ที่อยู่ไกลที่สุดและใกล้ที่สุด
            max_pair_distance = 0.0
            for i, (frame1, center1) in enumerate(history):
                for j, (frame2, center2) in enumerate(history):
                    if i != j and center1 is not None and center2 is not None:
                        dist = self.get_dist(center1, center2)
                        max_pair_distance = max(max_pair_distance, dist)

            if max_pair_distance <= stationary_center_thresh:
                # ⚠️ NEW: ตรวจสอบ velocity profile สำหรับกรณียืด/หด
                velocities = []
                for i in range(1, len(history)):
                    if i < len(history) and history[i-1][1] is not None and history[i][1] is not None:
                        vel = self.get_dist(history[i-1][1], history[i][1])
                        velocities.append(vel)

                if velocities:
                    mean_velocity = sum(velocities) / len(velocities)
                    frame_diagonal = math.sqrt(self.frame_w**2 + self.frame_h**2)
                    min_velocity_threshold = self.thresholds.get('BLACKLIST_MIN_VELOCITY_THRESHOLD',
                                                                 BLACKLIST_MIN_VELOCITY_RATIO * frame_diagonal)
                    if mean_velocity < min_velocity_threshold:
                        target['stationary_suspect'] = True
                        if DEBUG_MODE:
                            print(f"🚫 Target {target_id} is stationary (stretch/shrink case: max center distance={max_pair_distance:.2f} pixels, "
                                  f"mean_velocity={mean_velocity:.2f} < {min_velocity_threshold:.2f})")
                        return True
                else:
                    target['stationary_suspect'] = True
                    if DEBUG_MODE:
                        print(f"🚫 Target {target_id} is stationary (stretch/shrink case: max center distance={max_pair_distance:.2f} pixels)")
                    return True

        return False

    def select_primary_target(self, frame_counter=0):
        """
        เลือก primary target จาก targets ทั้งหมดตามคะแนนความสำคัญ

        Args:
            frame_counter: Frame counter สำหรับคำนวณ duration

        Returns:
            (primary_target_id, primary_target) หรือ (None, None) ถ้าไม่มี targets
        """
        from config import MIN_PRIORITY_FOR_PRIMARY

        if len(self.targets) == 0:
            return None, None

        # คำนวณ priority score สำหรับทุก target
        target_scores = {}
        for target_id, target in self.targets.items():
            # อัปเดต tracking_duration ถ้ายังไม่มี
            if 'tracking_start_frame' not in target:
                target['tracking_start_frame'] = frame_counter
            target['tracking_duration'] = frame_counter - target.get('tracking_start_frame', frame_counter) + 1

            # คำนวณ priority score
            priority_score = calculate_priority_score(target, tracker=None, frame_counter=frame_counter)
            target_scores[target_id] = priority_score
            target['priority_score'] = priority_score  # เก็บไว้สำหรับแสดงผล

        # เลือก target ที่มีคะแนนสูงสุด
        if not target_scores:
            return None, None

        # กรอง targets ที่มีคะแนนต่ำกว่า threshold
        valid_targets = {tid: score for tid, score in target_scores.items()
                        if score >= MIN_PRIORITY_FOR_PRIMARY}

        if not valid_targets:
            # ถ้าไม่มี target ที่ผ่าน threshold ให้เลือกตัวที่มีคะแนนสูงสุด
            valid_targets = target_scores

        primary_target_id = max(valid_targets.items(), key=lambda x: x[1])[0]
        primary_target = self.targets[primary_target_id]

        # Check if primary target changed
        old_primary_id = getattr(self, '_last_primary_target_id', None)
        if old_primary_id != primary_target_id:
            if DEBUG_MODE:
                old_score = self.targets[old_primary_id].get('priority_score', 0.0) if old_primary_id and old_primary_id in self.targets else 0.0
                new_score = primary_target.get('priority_score', 0.0)
                print(f"🔄 Primary target switched: {old_primary_id} -> {primary_target_id} (scores: {old_score:.2f} -> {new_score:.2f})")
            self._last_primary_target_id = primary_target_id

        primary_target['is_primary'] = True

        # Mark other targets as non-primary
        for tid in self.targets:
            if tid != primary_target_id:
                self.targets[tid]['is_primary'] = False

        return primary_target_id, primary_target

    def add_target_from_rect(self, rect, confidence=0.0, frame_w=None, frame_h=None):
        """
        Add a new target from a bounding box rectangle.

        Args:
            rect: (x, y, w, h) bounding box
            confidence: Initial confidence value
            frame_w: Frame width (uses self.frame_w if None)
            frame_h: Frame height (uses self.frame_h if None)

        Returns:
            target_id: ID of the newly created target
        """
        if frame_w is None:
            frame_w = self.frame_w
        if frame_h is None:
            frame_h = self.frame_h

        x, y, w_box, h_box = rect
        center = (x + w_box//2, y + h_box//2)
        target_id = self.next_target_id
        self.next_target_id += 1

        # ใช้ frame_counter จาก instance variable
        current_frame = getattr(self, '_current_frame_counter', 0)

        roi_box = [
                max(0, x - HYBRID_ROI_PADDING),
                max(0, y - HYBRID_ROI_PADDING),
                min(frame_w, x + w_box + HYBRID_ROI_PADDING),
                min(frame_h, y + h_box + HYBRID_ROI_PADDING)
        ]
        roi_center = ((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)

        self.targets[target_id] = {
            'target_locked': True,
            'roi_box': roi_box,
            'roi_box_history': [(current_frame, tuple(roi_box), roi_center)],  # เริ่มต้นด้วย ROI แรก
            'last_center': center,
            'missed_frames': 0,
            'confidence': confidence,
            'yolo_frame_counter': 0,  # สำหรับ interval-based YOLO detection
            'centers_history': [(current_frame, center)]  # เริ่มต้นด้วยจุดแรก (วาด path ได้ทันที)
        }

        return target_id

    def match_motion_box_to_motion_targets(self, motion_box, dist_threshold):
        """
        Match motion box กับ motion-only targets ที่มีอยู่

        Args:
            motion_box: (x, y, w, h) motion box
            dist_threshold: ระยะห่างสูงสุดที่ยอมรับได้ (pixels)

        Returns:
            motion_id: ID ของ matched target หรือ None
        """
        motion_center = (motion_box[0] + motion_box[2]//2, motion_box[1] + motion_box[3]//2)

        best_match_id = None
        min_dist = float('inf')

        for motion_id, motion_target in self.motion_only_targets.items():
            if 'last_center' in motion_target and motion_target['last_center']:
                dist = self.get_dist(motion_center, motion_target['last_center'])
                if dist < dist_threshold and dist < min_dist:
                    min_dist = dist
                    best_match_id = motion_id

        return best_match_id

    def track_motion_only_targets(self, motion_boxes, frame_counter, w_orig, h_orig):
        """
        Track motion boxes ที่ไม่ match กับ YOLO detection

        Args:
            motion_boxes: List of (x, y, w, h) motion boxes
            frame_counter: Frame counter
            w_orig: Frame width
            h_orig: Frame height
        """
        if not MOTION_ONLY_TARGET_ENABLED:
            return

        # Interval-based checking (เพื่อ performance)
        if frame_counter % MOTION_ONLY_CHECK_INTERVAL != 0:
            return

        processing_fps = getattr(self, '_current_processing_fps', None)
        if processing_fps is None or processing_fps <= 0:
            from config import DEFAULT_FPS
            processing_fps = DEFAULT_FPS

        min_duration_frames = int(MOTION_ONLY_MIN_TRACKING_DURATION * processing_fps)
        min_duration_frames = max(5, min_duration_frames)

        # Match motion boxes กับ existing motion-only targets
        matched_targets = set()
        for motion_box in motion_boxes:
            motion_id = self.match_motion_box_to_motion_targets(motion_box, HYBRID_DIST_THRESHOLD)
            if motion_id is not None:
                matched_targets.add(motion_id)
                # Update target
                motion_target = self.motion_only_targets[motion_id]
                tx, ty, tw, th = motion_box
                motion_center = (tx + tw//2, ty + th//2)

                # ตรวจสอบ motion box ก่อนเพิ่มเข้าไปใน path (ไม่ skip initial frames)
                last_center = motion_target.get('last_center')
                last_rect = motion_target.get('last_rect')

                # คำนวณ velocity vector (ถ้ามี)
                velocity_vector = None
                if 'centers_history' in motion_target and len(motion_target['centers_history']) >= 2:
                    velocity_vector = self.get_velocity_from_path_history(
                        motion_target['centers_history'], processing_fps=processing_fps
                    )

                # คำนวณ predicted center (ถ้ามี velocity)
                predicted_center = None
                if velocity_vector is not None and last_center is not None:
                    vx, vy = velocity_vector
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        predicted_center = (last_center[0] + vx, last_center[1] + vy)

                # Motion box ที่ถูกเลือกแล้ว → เก็บ path point ทุกครั้ง (ไม่ต้อง validate)
                # อัปเดต target state
                motion_target['last_rect'] = motion_box
                motion_target['last_center'] = motion_center
                motion_target['missed_frames'] = 0
                motion_target['last_update_frame'] = frame_counter
                motion_target['consecutive_predicted_frames'] = 0  # Reset เมื่อเจอ motion box จริง

                # Update path history (ใช้ update_path_history เพื่อให้ใช้ logic ที่ถูกต้อง)
                self.update_path_history(motion_target, motion_center, frame_counter, status=motion_target.get('status', 'GREEN'))

                # สร้าง roi_box สำหรับ motion-only target
                new_roi_box = [
                    max(0, tx - HYBRID_ROI_PADDING),
                    max(0, ty - HYBRID_ROI_PADDING),
                    min(w_orig, tx + tw + HYBRID_ROI_PADDING),
                    min(h_orig, ty + th + HYBRID_ROI_PADDING)
                ]
                motion_target['roi_box'] = new_roi_box
                roi_center = ((new_roi_box[0] + new_roi_box[2]) // 2, (new_roi_box[1] + new_roi_box[3]) // 2)

                # อัปเดต ROI history
                from config import ROI_HISTORY_MAX_FRAMES
                if 'roi_box_history' not in motion_target:
                    motion_target['roi_box_history'] = []
                motion_target['roi_box_history'].append((frame_counter, tuple(new_roi_box), roi_center))

                # จำกัดจำนวน history (performance)
                if len(motion_target['roi_box_history']) > ROI_HISTORY_MAX_FRAMES:
                    motion_target['roi_box_history'] = motion_target['roi_box_history'][-ROI_HISTORY_MAX_FRAMES:]

                # Update path history (เก็บทุกเฟรมเพื่อให้ path ต่อเนื่อง)
                # Note: update_path_history ถูกเรียกไปแล้วข้างบน (บรรทัด ~7277)

                # ประเมิน path quality และเปลี่ยน status
                from config import (
                    MOTION_ONLY_MIN_PATH_POINTS, MOTION_ONLY_ORANGE_PATH_QUALITY_THRESHOLD,
                    MOTION_ONLY_RED_PATH_QUALITY_THRESHOLD, MOTION_ONLY_RED_MIN_PATH_POINTS,
                    MOTION_ONLY_STATUS_CHECK_INTERVAL
                )

                # ตรวจสอบ status ทุก N เฟรม (เพื่อ performance)
                if frame_counter % MOTION_ONLY_STATUS_CHECK_INTERVAL == 0:
                    if len(motion_target['centers_history']) >= MOTION_ONLY_MIN_PATH_POINTS:
                        # คำนวณ path quality
                        is_clear, path_quality = self.calculate_motion_path_quality(
                            motion_target, frame_counter, is_motion_only=True
                        )
                        motion_target['path_quality'] = path_quality

                        # ใช้ path quality เพื่อเปลี่ยน status
                        # Status progression: GREEN → ORANGE → RED
                        current_status = motion_target.get('status', 'GREEN')

                        # เปลี่ยนเป็น ORANGE เมื่อ path quality >= threshold และมี path points เพียงพอ
                        if current_status == 'GREEN':
                            if (path_quality >= MOTION_ONLY_ORANGE_PATH_QUALITY_THRESHOLD and
                                len(motion_target['centers_history']) >= MOTION_ONLY_MIN_PATH_POINTS):
                                motion_target['status'] = 'ORANGE'
                                if DEBUG_MODE:
                                    print(f"🟠 Motion target {motion_id}: Status changed to ORANGE (path_quality={path_quality:.2f})")

                        # เปลี่ยนเป็น RED เมื่อ path quality สูงมากและมี path ยาว
                        elif current_status == 'ORANGE':
                            if (path_quality >= MOTION_ONLY_RED_PATH_QUALITY_THRESHOLD and
                                len(motion_target['centers_history']) >= MOTION_ONLY_RED_MIN_PATH_POINTS):
                                motion_target['status'] = 'RED'
                                if DEBUG_MODE:
                                    print(f"🔴 Motion target {motion_id}: Status changed to RED (path_quality={path_quality:.2f})")

                # วิเคราะห์ path เพื่อแยกประเภท target (ตรวจสอบว่า path ยาวพอ - อย่างน้อย 15-30 จุด)
                from config import PATH_CLASSIFICATION_ENABLED, PATH_CLASSIFICATION_MIN_POINTS, PATH_CLASSIFICATION_UPDATE_INTERVAL
                if PATH_CLASSIFICATION_ENABLED:
                    centers_history = motion_target.get('centers_history', [])
                    if len(centers_history) >= PATH_CLASSIFICATION_MIN_POINTS:
                        last_classification_frame = motion_target.get('last_classification_frame', -PATH_CLASSIFICATION_UPDATE_INTERVAL)
                        if frame_counter - last_classification_frame >= PATH_CLASSIFICATION_UPDATE_INTERVAL:
                            # ใช้ processing_fps สำหรับ FPS-aware thresholds
                            processing_fps = getattr(self, '_processing_fps', None)
                            classification_result = self.classify_target_from_path(motion_target, frame_counter, processing_fps)
                            motion_target['path_classification'] = classification_result['classification']
                            motion_target['path_classification_confidence'] = classification_result['confidence']
                            motion_target['path_characteristics'] = classification_result['characteristics']
                            motion_target['last_classification_frame'] = frame_counter

                # จำกัดความยาว path history ตาม size category
                size_category = motion_target.get('object_size_category')
                if size_category is None:
                    size_category = self.get_object_size_category_for_target_from_rect(motion_box, w_orig, h_orig)
                    if size_category:
                        motion_target['object_size_category'] = size_category

                # ใช้ size-adaptive path history limit
                max_history = MOTION_ONLY_PATH_HISTORY_MAX_FRAMES
                if size_category == 'TINY':
                    max_history = min(MOTION_ONLY_PATH_HISTORY_MAX_FRAMES, 20)
                elif size_category == 'SMALL':
                    max_history = min(MOTION_ONLY_PATH_HISTORY_MAX_FRAMES, 25)
                elif size_category == 'MEDIUM':
                    max_history = min(MOTION_ONLY_PATH_HISTORY_MAX_FRAMES, 30)
                elif size_category == 'LARGE':
                    max_history = min(MOTION_ONLY_PATH_HISTORY_MAX_FRAMES, 35)

                if len(motion_target['centers_history']) > max_history:
                    motion_target['centers_history'] = motion_target['centers_history'][-max_history:]

        # สร้าง motion-only target ใหม่สำหรับ unmatched motion boxes
        for motion_box in motion_boxes:
            motion_id = self.match_motion_box_to_motion_targets(motion_box, HYBRID_DIST_THRESHOLD)
            if motion_id is None:
                # ตรวจสอบว่าไม่เกิน MAX_TOTAL_TARGETS (รวม motion-only + YOLO targets)
                total_targets = len(self.motion_only_targets) + len(self.targets)
                if total_targets >= MAX_TOTAL_TARGETS:
                    # ลบ target ที่เก่าที่สุด (หรือ path quality ต่ำสุด)
                    if self.motion_only_targets:
                        oldest_id = min(self.motion_only_targets.keys(),
                                      key=lambda k: self.motion_only_targets[k].get('created_frame', frame_counter))
                        del self.motion_only_targets[oldest_id]

                motion_id = self.next_motion_id
                self.next_motion_id += 1
                tx, ty, tw, th = motion_box
                motion_center = (tx + tw//2, ty + th//2)

                # คำนวณ size category
                size_category = self.get_object_size_category_for_target_from_rect(motion_box, w_orig, h_orig)

                roi_box = [
                        max(0, tx - HYBRID_ROI_PADDING),
                        max(0, ty - HYBRID_ROI_PADDING),
                        min(w_orig, tx + tw + HYBRID_ROI_PADDING),
                        min(h_orig, ty + th + HYBRID_ROI_PADDING)
                ]
                roi_center = ((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)

                self.motion_only_targets[motion_id] = {
                    'centers_history': [(frame_counter, motion_center)],
                    'last_rect': motion_box,
                    'last_center': motion_center,
                    'roi_box': roi_box,
                    'roi_box_history': [(frame_counter, tuple(roi_box), roi_center)],  # เริ่มต้นด้วย ROI แรก
                    'created_frame': frame_counter,
                    'missed_frames': 0,
                    'last_update_frame': frame_counter,  # เริ่มต้นด้วย frame_counter
                    'path_quality': 0.0,
                    'path_quality_cache_frame': -1,
                    'object_size_category': size_category,
                    'status': 'GREEN',  # เริ่มต้นด้วย GREEN status
                    'is_motion_only': True,  # ระบุว่าเป็น motion-only target
                    'consecutive_predicted_frames': 0  # จำนวนเฟรมที่ใช้ predicted position ติดต่อกัน
                }

        # Update missed_frames สำหรับ unmatched targets และใช้ predicted position
        for motion_id in list(self.motion_only_targets.keys()):
            if motion_id not in matched_targets:
                motion_target = self.motion_only_targets[motion_id]
                motion_target['missed_frames'] += 1

                # ใช้ predicted position เมื่อ motion หายไป
                from config import PREDICTED_POSITION_MIN_HISTORY, PREDICTED_POSITION_MAX_CONSECUTIVE
                consecutive_predicted = motion_target.get('consecutive_predicted_frames', 0)

                if consecutive_predicted < PREDICTED_POSITION_MAX_CONSECUTIVE:
                    centers_history = motion_target.get('centers_history', [])
                    if len(centers_history) >= PREDICTED_POSITION_MIN_HISTORY:
                        predicted_center = self.predict_next_position(centers_history)
                        if predicted_center is not None:
                            self.update_path_history(motion_target, predicted_center, frame_counter, status=motion_target.get('status', 'GREEN'))
                            motion_target['last_center'] = predicted_center
                            motion_target['consecutive_predicted_frames'] = consecutive_predicted + 1
                    elif len(centers_history) >= 1:
                        # มี path history 1 จุด → ใช้ last_center เป็น predicted position (ไม่ขยับ)
                        last_center = motion_target.get('last_center')
                        if last_center:
                            self.update_path_history(motion_target, last_center, frame_counter, status=motion_target.get('status', 'GREEN'))
                            motion_target['consecutive_predicted_frames'] = consecutive_predicted + 1
                else:
                    # ใช้ predicted position มากเกินไป → ไม่สร้างต่อ
                    motion_target['consecutive_predicted_frames'] = 0  # Reset เพื่อให้ลองใหม่ในอนาคต

        # คำนวณ path quality และ cleanup (ใช้ size-adaptive thresholds)
        targets_to_remove = []
        for motion_id, motion_target in self.motion_only_targets.items():
            # คำนวณ path quality (cache เพื่อ performance)
            cache_frame = motion_target.get('path_quality_cache_frame', -1)
            if cache_frame != frame_counter:
                centers_history = motion_target.get('centers_history', [])
                size_category = motion_target.get('object_size_category')

                # ใช้ size-adaptive min_path_points
                if size_category == 'TINY':
                    min_path_points = TINY_MIN_PATH_POINTS
                    min_quality = TINY_MIN_PATH_QUALITY
                    min_duration = TINY_MIN_TRACKING_DURATION
                elif size_category == 'SMALL':
                    min_path_points = SMALL_MIN_PATH_POINTS
                    min_quality = SMALL_MIN_PATH_QUALITY
                    min_duration = SMALL_MIN_TRACKING_DURATION
                elif size_category == 'MEDIUM':
                    min_path_points = MEDIUM_MIN_PATH_POINTS
                    min_quality = MEDIUM_MIN_PATH_QUALITY
                    min_duration = MEDIUM_MIN_TRACKING_DURATION
                elif size_category == 'LARGE':
                    min_path_points = LARGE_MIN_PATH_POINTS
                    min_quality = LARGE_MIN_PATH_QUALITY
                    min_duration = LARGE_MIN_TRACKING_DURATION
                else:
                    min_path_points = MOTION_ONLY_MIN_PATH_POINTS
                    min_quality = MOTION_ONLY_MIN_PATH_QUALITY
                    min_duration = MOTION_ONLY_MIN_TRACKING_DURATION

                if len(centers_history) >= min_path_points:
                    # ใช้ calculate_size_adaptive_path_quality() สำหรับ motion-only targets
                    is_path_clear, path_quality = self.calculate_size_adaptive_path_quality(
                        motion_target, frame_counter, size_category, is_motion_only=True
                    )
                    motion_target['path_quality'] = path_quality
                    motion_target['path_quality_cache_frame'] = frame_counter
                else:
                    motion_target['path_quality'] = 0.0

            # Cleanup: ลบ targets ที่หายไปนานหรือ path quality ต่ำ (ใช้ size-adaptive thresholds)
            missed_frames = motion_target.get('missed_frames', 0)
            path_quality = motion_target.get('path_quality', 0.0)

            # ตรวจสอบว่า processing_fps ไม่เป็น None (fallback safety check)
            cleanup_processing_fps = processing_fps
            if cleanup_processing_fps is None or cleanup_processing_fps <= 0:
                from config import DEFAULT_FPS
                cleanup_processing_fps = DEFAULT_FPS

            tracking_duration_seconds = (frame_counter - motion_target.get('created_frame', frame_counter)) / cleanup_processing_fps
            size_category = motion_target.get('object_size_category')
            centers_history = motion_target.get('centers_history', [])

            # ใช้ size-adaptive min_duration และ cleanup threshold
            if size_category == 'TINY':
                min_duration = TINY_MIN_TRACKING_DURATION
                min_quality = TINY_MIN_PATH_QUALITY
                min_path_points = TINY_MIN_PATH_POINTS
                cleanup_quality_threshold = 0.3  # ยืดหยุ่นสำหรับ cleanup (ต่ำกว่า min_quality)
            elif size_category == 'SMALL':
                min_duration = SMALL_MIN_TRACKING_DURATION
                min_quality = SMALL_MIN_PATH_QUALITY
                min_path_points = SMALL_MIN_PATH_POINTS
                cleanup_quality_threshold = min_quality  # ใช้ min_quality เดิม
            elif size_category == 'MEDIUM':
                min_duration = MEDIUM_MIN_TRACKING_DURATION
                min_quality = MEDIUM_MIN_PATH_QUALITY
                min_path_points = MEDIUM_MIN_PATH_POINTS
                cleanup_quality_threshold = 0.5  # ยืดหยุ่นสำหรับ cleanup (ต่ำกว่า 0.65)
            elif size_category == 'LARGE':
                min_duration = LARGE_MIN_TRACKING_DURATION
                min_quality = LARGE_MIN_PATH_QUALITY
                min_path_points = LARGE_MIN_PATH_POINTS
                cleanup_quality_threshold = 0.5  # ยืดหยุ่นสำหรับ cleanup (ต่ำกว่า 0.7)
            else:
                min_duration = MOTION_ONLY_MIN_TRACKING_DURATION
                min_quality = MOTION_ONLY_MIN_PATH_QUALITY
                min_path_points = MOTION_ONLY_MIN_PATH_POINTS
                cleanup_quality_threshold = min_quality  # ใช้ min_quality เดิม

            # ตรวจสอบว่ามี path points เพียงพอหรือไม่
            has_sufficient_path_points = len(centers_history) >= min_path_points

            # Cleanup condition: ลบเฉพาะเมื่อ:
            # 1. หายไปนานเกินไป (missed_frames > HYBRID_MAX_MISS_ALLOWED)
            # 2. หรือ tracking duration ถึงแล้ว แต่ path quality ต่ำมาก (ต่ำกว่า cleanup threshold) และไม่มี path points เพียงพอ
            should_remove = False
            if missed_frames > HYBRID_MAX_MISS_ALLOWED:
                should_remove = True
            elif tracking_duration_seconds >= min_duration:
                # ถ้า path quality ต่ำกว่า cleanup threshold และไม่มี path points เพียงพอ → ลบ
                if path_quality < cleanup_quality_threshold and not has_sufficient_path_points:
                    should_remove = True
                # ถ้า path quality ต่ำมาก (ต่ำกว่า cleanup threshold * 0.5) → ลบเลย (แม้มี path points)
                elif path_quality < cleanup_quality_threshold * 0.5:
                    should_remove = True

            if should_remove:
                targets_to_remove.append(motion_id)

        # ลบ targets
        for motion_id in targets_to_remove:
            del self.motion_only_targets[motion_id]

    def find_best_motion(self, mask_cpu, roi, ref_center, min_motion_area=None,
                        object_size_category=None, adaptive_min_area=None,
                        velocity_vector=None, predicted_center=None, last_rect=None,
                        thresholds=None, use_fallback=False, target_id=None, debug_display_frame=None):
        """
        ค้นหา motion box ที่เหมาะสมที่สุดโดยใช้ scoring system ที่พิจารณา:
        - ตำแหน่งที่ทำนายจากความเร็วและทิศทาง
        - ระยะห่างจากตำแหน่งเฟรมก่อนหน้า
        - ทิศทางที่สอดคล้องกับ velocity vector
        - ขนาดที่ใกล้เคียงกับ target

        Args:
            mask_cpu: Motion mask (CPU)
            roi: ROI box (x1, y1, x2, y2)
            ref_center: Reference center point (x, y) - ตำแหน่งเฟรมก่อนหน้า
            min_motion_area: Minimum motion area threshold (ถ้า None ใช้ HYBRID_MIN_MOTION_AREA)
            object_size_category: Optional - 'TINY', 'SMALL', 'MEDIUM', 'LARGE' (ใช้สำหรับคำนวณ min_area ตาม size)
            adaptive_min_area: Optional - ค่า adaptive min_area จาก AdaptiveMinAreaManager (ใช้เมื่อมี object_size_category)
            velocity_vector: Optional - (vx, vy) velocity vector สำหรับทำนายตำแหน่ง
            predicted_center: Optional - (x, y) predicted center position (ถ้ามีจะใช้แทนการคำนวณ)
            last_rect: Optional - (x, y, w, h) ขนาด target ก่อนหน้า สำหรับ size filtering
            thresholds: Optional - Dictionary ของ resolution-dependent thresholds
            use_fallback: Optional - ถ้า True ใช้ fallback strategies (ลด thresholds)
            target_id: Optional - Target ID สำหรับ debug logging

        Returns:
            (x, y, w, h) motion rect หรือ None ถ้าไม่เจอ
        """
        from config import (
            MOTION_SIZE_MIN_RATIO, MOTION_SIZE_MAX_RATIO,
            MOTION_SCORE_PREDICTED_WEIGHT, MOTION_SCORE_PREVIOUS_WEIGHT,
            MOTION_SCORE_DIRECTION_WEIGHT, MOTION_SCORE_SIZE_WEIGHT,
            FALLBACK_REDUCE_MIN_AREA_RATIO, FALLBACK_REDUCE_SIZE_RATIO,
            SIZE_CHANGE_SMOOTHING_ALPHA, SIZE_CHANGE_MAX_RATIO
        )

        # Fallback: ลด thresholds ถ้า use_fallback=True
        size_min_ratio = MOTION_SIZE_MIN_RATIO
        size_max_ratio = MOTION_SIZE_MAX_RATIO
        if use_fallback:
            size_min_ratio = MOTION_SIZE_MIN_RATIO * FALLBACK_REDUCE_SIZE_RATIO
            size_max_ratio = MOTION_SIZE_MAX_RATIO / FALLBACK_REDUCE_SIZE_RATIO
            if min_motion_area is not None:
                min_motion_area = max(1, int(min_motion_area * FALLBACK_REDUCE_MIN_AREA_RATIO))

        # ถ้ามี object_size_category ให้ใช้ size-based min_area
        if object_size_category is not None:
            if adaptive_min_area is None:
                adaptive_min_area = HYBRID_MIN_MOTION_AREA
            min_motion_area = self.get_min_motion_area_for_size(object_size_category, adaptive_min_area, lock_mode=False)
        elif min_motion_area is None:
            min_motion_area = HYBRID_MIN_MOTION_AREA

        rx1, ry1, rx2, ry2 = roi
        # Ensure ROI is within frame bounds
        rx1 = max(0, min(rx1, self.frame_w))
        ry1 = max(0, min(ry1, self.frame_h))
        rx2 = max(rx1, min(rx2, self.frame_w))
        ry2 = max(ry1, min(ry2, self.frame_h))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        roi_mask = mask_cpu[ry1:ry2, rx1:rx2]
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ใช้ resolution-dependent thresholds
        if thresholds is None:
            thresholds = self.thresholds
        max_dist_threshold = thresholds.get('HYBRID_DIST_THRESHOLD', HYBRID_DIST_THRESHOLD)

        # Edge case handling: คำนวณ predicted center ถ้ายังไม่มี
        if predicted_center is None and velocity_vector is not None:
            vx, vy = velocity_vector
            # ตรวจสอบว่า velocity ไม่เป็น NaN หรือ Inf
            if (not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)) and
                (abs(vx) > 0.01 or abs(vy) > 0.01)):  # มีการเคลื่อนที่
                predicted_center = (ref_center[0] + vx, ref_center[1] + vy)
                # Clamp ให้อยู่ใน frame bounds
                predicted_center = (
                    max(0, min(self.frame_w - 1, predicted_center[0])),
                    max(0, min(self.frame_h - 1, predicted_center[1]))
                )
            else:
                predicted_center = ref_center  # ไม่มีการเคลื่อนที่หรือ velocity ไม่ valid ใช้ตำแหน่งเดิม

        # Edge case: ถ้าไม่มี predicted_center ใช้ ref_center แทน
        if predicted_center is None:
            predicted_center = ref_center

        # Edge case: ตรวจสอบว่า predicted_center อยู่ในขอบเขตที่สมเหตุสมผล
        if predicted_center is not None:
            px, py = predicted_center
            if (px < -self.frame_w * 0.1 or px > self.frame_w * 1.1 or
                py < -self.frame_h * 0.1 or py > self.frame_h * 1.1):
                # predicted position อยู่นอก frame มาก ใช้ ref_center แทน
                predicted_center = ref_center

        # คำนวณ target area สำหรับ size filtering (ใช้ smoothed size ถ้ามี)
        target_area = None
        smoothed_size = None
        if last_rect is not None:
            tx, ty, tw, th = last_rect
            target_area = tw * th

            # Size change handling: ใช้ smoothed size จาก size history
            if target_id is not None and target_id in self.targets:
                target = self.targets[target_id]
                size_history = target.get('size_history', [])
                if len(size_history) >= 2:
                    # คำนวณ smoothed size จาก size history
                    recent_sizes = size_history[-5:] if len(size_history) >= 5 else size_history
                    if recent_sizes:
                        # ใช้ exponential smoothing
                        smoothed_area = recent_sizes[-1][1]  # เริ่มจาก size ล่าสุด
                        for i in range(len(recent_sizes) - 2, -1, -1):
                            _, area = recent_sizes[i]
                            smoothed_area = SIZE_CHANGE_SMOOTHING_ALPHA * area + (1.0 - SIZE_CHANGE_SMOOTHING_ALPHA) * smoothed_area

                        # ตรวจสอบว่า smoothed size ไม่ต่างจาก current size มากเกินไป
                        size_change_ratio = smoothed_area / target_area if target_area > 0 else 1.0
                        if 1.0 / SIZE_CHANGE_MAX_RATIO <= size_change_ratio <= SIZE_CHANGE_MAX_RATIO:
                            target_area = int(smoothed_area)
                            smoothed_size = int(math.sqrt(smoothed_area))  # ประมาณ width/height

        best_rect = None
        best_score = -float('inf')

        # Performance metrics: เก็บจำนวน motion boxes ที่พบ
        total_motion_boxes = 0
        filtered_motion_boxes = 0
        start_time = None
        if self.performance_metrics_enabled:
            import time
            start_time = time.time()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_motion_area:
                filtered_motion_boxes += 1
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            curr_rect = (x + rx1, y + ry1, w, h)
            curr_center = (curr_rect[0] + w//2, curr_rect[1] + h//2)

            # Size filtering: กรอง motion boxes ที่ขนาดต่างจาก target มากเกินไป
            if target_area is not None and target_area > 0:
                motion_area = w * h
                if motion_area > 0:
                    size_ratio = motion_area / target_area
                    # Edge case: ตรวจสอบว่า size_ratio ไม่เป็น NaN หรือ Inf
                    if (math.isnan(size_ratio) or math.isinf(size_ratio) or
                        size_ratio < size_min_ratio or size_ratio > size_max_ratio):
                        filtered_motion_boxes += 1
                        continue  # ข้าม motion box นี้
                else:
                    continue  # ข้าม motion box ที่มี area = 0

            # Early exit: ถ้า motion box อยู่นอก ROI มากเกินไป
            dist_to_ref = self.get_dist(ref_center, curr_center)
            if dist_to_ref > max_dist_threshold * 2:
                filtered_motion_boxes += 1
                continue  # ข้าม motion box ที่ไกลเกินไป

            # คำนวณ score components
            score = 0.0

            # 1. Distance to predicted position (น้ำหนักสูง)
            if predicted_center is not None:
                dist_to_predicted = self.get_dist(predicted_center, curr_center)
                normalized_dist_predicted = min(dist_to_predicted / max_dist_threshold, 1.0)
                score += MOTION_SCORE_PREDICTED_WEIGHT * (1.0 - normalized_dist_predicted)

            # 2. Distance to previous position (น้ำหนักกลาง)
            dist_to_previous = dist_to_ref
            normalized_dist_previous = min(dist_to_previous / max_dist_threshold, 1.0)
            score += MOTION_SCORE_PREVIOUS_WEIGHT * (1.0 - normalized_dist_previous)

            # 3. Direction alignment (น้ำหนักต่ำ)
            if velocity_vector is not None:
                vx, vy = velocity_vector
                # Edge case: ตรวจสอบว่า velocity ไม่เป็น NaN หรือ Inf
                if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                    velocity_mag = math.sqrt(vx**2 + vy**2)
                    if velocity_mag > 0.1:  # มีการเคลื่อนที่
                        # คำนวณ motion direction
                        motion_dx = curr_center[0] - ref_center[0]
                        motion_dy = curr_center[1] - ref_center[1]
                        motion_mag = math.sqrt(motion_dx**2 + motion_dy**2)

                        if motion_mag > 0.1:
                            try:
                                # Dot product และ normalize
                                dot_product = (vx * motion_dx + vy * motion_dy) / (velocity_mag * motion_mag)
                                # Edge case: ตรวจสอบว่า dot_product ไม่เป็น NaN หรือ Inf
                                if not (math.isnan(dot_product) or math.isinf(dot_product)):
                                    # Normalize จาก [-1, 1] เป็น [0, 1]
                                    direction_score = (dot_product + 1.0) / 2.0
                                    score += MOTION_SCORE_DIRECTION_WEIGHT * direction_score
                            except (ZeroDivisionError, ValueError):
                                # Edge case: ถ้ามี error ในการคำนวณ ข้าม direction alignment
                                pass

            # 4. Size similarity (น้ำหนักต่ำ)
            if target_area is not None and target_area > 0:
                motion_area = w * h
                if motion_area > 0:
                    try:
                        size_similarity = min(motion_area, target_area) / max(motion_area, target_area)
                        # Edge case: ตรวจสอบว่า size_similarity ไม่เป็น NaN หรือ Inf
                        if not (math.isnan(size_similarity) or math.isinf(size_similarity)):
                            score += MOTION_SCORE_SIZE_WEIGHT * size_similarity
                    except (ZeroDivisionError, ValueError):
                        # Edge case: ถ้ามี error ในการคำนวณ ข้าม size similarity
                        pass

            # เลือก motion box ที่มีคะแนนสูงสุด
            if score > best_score:
                best_score = score
                best_rect = curr_rect

        # Performance metrics: เก็บข้อมูล
        if self.performance_metrics_enabled and start_time is not None:
            import time
            elapsed_time = time.time() - start_time
            current_frame = getattr(self, '_current_frame_counter', 0)

            if (current_frame - self.performance_metrics['last_metrics_frame']) >= self.performance_metrics_interval:
                self.performance_metrics['roi_motion_detection_time'].append(elapsed_time)
                self.performance_metrics['motion_boxes_found'].append(total_motion_boxes)
                self.performance_metrics['motion_boxes_filtered'].append(filtered_motion_boxes)
                hit_rate = 1.0 if best_rect is not None else 0.0
                self.performance_metrics['hit_rate'].append(hit_rate)
                self.performance_metrics['last_metrics_frame'] = current_frame

                # จำกัดขนาด metrics history
                max_history = 100
                for key in ['roi_motion_detection_time', 'motion_boxes_found', 'motion_boxes_filtered', 'hit_rate']:
                    if len(self.performance_metrics[key]) > max_history:
                        self.performance_metrics[key] = self.performance_metrics[key][-max_history:]

        # Debug visualization (ถ้ามี debug_display_frame)
        if debug_display_frame is not None:
            from config import (
                DEBUG_SHOW_PREDICTED_POSITION, DEBUG_SHOW_VELOCITY_VECTOR,
                DEBUG_SHOW_EXTENDED_ROI, DEBUG_SHOW_SCORE, DEBUG_MODE
            )
            if DEBUG_MODE:
                if DEBUG_SHOW_PREDICTED_POSITION and predicted_center is not None:
                    px, py = int(predicted_center[0]), int(predicted_center[1])
                    cv2.circle(debug_display_frame, (px, py), 5, (255, 0, 255), 2)  # Magenta circle
                    cv2.putText(debug_display_frame, "Pred", (px + 10, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                if DEBUG_SHOW_VELOCITY_VECTOR and velocity_vector is not None and ref_center is not None:
                    vx, vy = velocity_vector
                    if isinstance(velocity_vector, (tuple, list)) and len(velocity_vector) == 2:
                        if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                            start_x, start_y = int(ref_center[0]), int(ref_center[1])
                            end_x = int(start_x + vx * 10)  # Scale up for visibility
                            end_y = int(start_y + vy * 10)
                            cv2.arrowedLine(debug_display_frame, (start_x, start_y), (end_x, end_y),
                                          (0, 255, 255), 2, tipLength=0.3)  # Cyan arrow

                # ROI visualization จะทำใน update_tracking() แทน

                if DEBUG_SHOW_SCORE and best_rect is not None and best_score > -float('inf'):
                    bx, by, bw, bh = best_rect
                    score_text = f"Score: {best_score:.2f}"
                    cv2.putText(debug_display_frame, score_text, (bx, by - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return best_rect

    def process_frame(self, frame, mask_cpu, detect_mask=None, horizon_mask_upper=None, motion_detection_area_mask=None, frame_counter=0, min_motion_area=None, processing_fps=None):
        # เก็บ frame_counter และ processing_fps สำหรับใช้ใน methods อื่นๆ
        self._current_frame_counter = frame_counter
        self._current_processing_fps = processing_fps

        # ทำ cleanup path history (ทุก N เฟรม - ไม่กระทบทุกเฟรม)
        if DRONE_PATH_HISTORY_OPTIMIZATION_ENABLED:
            self.cleanup_path_history(frame_counter)

        """
        Process a single frame and return display frame with tracking results.
        Supports multiple targets.

        Args:
            frame: Input frame (BGR)
            mask_cpu: Motion mask from background subtraction (CPU)
            detect_mask: Optional - detect_mask from main loop (for motion boxes matching)
            horizon_mask_upper: Optional - mask for upper zone (above horizon) - ถ้ามีจะตรวจสอบว่า detection อยู่ในโซนบนเส้นฟ้าหรือไม่
            motion_detection_area_mask: Optional - mask defining allowed area for motion detection (with margin from horizon line)
            frame_counter: Frame counter for priority calculation
            min_motion_area: Optional - minimum motion area threshold (uses HYBRID_MIN_MOTION_AREA if None)

        Returns:
            display_frame: Frame with tracking visualization
            tracking_info: Dict with tracking status info
        """
        # ใช้ min_motion_area parameter ถ้ามี ไม่งั้นใช้ HYBRID_MIN_MOTION_AREA
        if min_motion_area is None:
            min_motion_area = HYBRID_MIN_MOTION_AREA
        h_orig, w_orig = frame.shape[:2]
        display_frame = frame.copy()
        self._current_frame_counter = frame_counter  # Store for use in target creation

        # Apply morphology to mask
        mask_cpu_processed = cv2.morphologyEx(mask_cpu, cv2.MORPH_OPEN, self.kernel)

        # Cache frame shape and BGR conversion check to avoid repeated operations
        current_frame_shape = frame.shape
        if self._cached_frame_shape != current_frame_shape:
            # Frame shape changed, update cache
            self._cached_frame_shape = current_frame_shape
            self._cached_needs_bgr_conversion = (frame.shape[2] == 4)

        # ตรวจสอบและแปลง frame เป็น BGR (3 channels) ถ้ามี 4 channels (ใช้ cache)
        if self._cached_needs_bgr_conversion:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            frame_bgr = frame

        # === SEARCH MODE: หาโดรนใหม่ด้วย confidence ต่ำ (ปรับปรุงใหม่) ===
        # ใช้ adaptive interval: บ่อยขึ้นเมื่อไม่มี target, น้อยลงเมื่อมี target
        has_targets = len(self.targets) > 0
        search_interval = HYBRID_SEARCH_INTERVAL_WITH_TARGET if has_targets else HYBRID_SEARCH_INTERVAL_NO_TARGET
        should_search = (not has_targets) or (frame_counter % search_interval == 0)

        # รวบรวม ROI boxes จาก targets ทั้งหมด (สำหรับข้อยกเว้น motion detection)
        roi_boxes = []
        for target_id, target in self.targets.items():
            if 'roi_box' in target and target['roi_box'] is not None:
                roi_box = target['roi_box']
                if len(roi_box) == 4:
                    roi_boxes.append(roi_box)

        # หา motion boxes (กล่องเขียว) จาก mask - ทำทุกเฟรมเพื่อแสดงผล
        # ใช้ detect_mask ถ้ามี (ตรงกับที่แสดงในหน้าจอ) ไม่งั้นใช้ mask_cpu_processed
        # ส่ง motion_detection_area_mask และ roi_boxes เพื่อกรอง motion boxes
        # สำหรับ search mode: ยังไม่รู้ object size → ใช้ min_motion_area ปกติ
        # สำหรับ tracking mode: จะส่ง object_size_category จาก target cache ใน update_tracking()
        if detect_mask is not None:
            motion_boxes = find_motion_boxes(detect_mask, min_motion_area,
                                            area_mask=motion_detection_area_mask,
                                            roi_boxes=roi_boxes if len(roi_boxes) > 0 else None,
                                            object_size_category=None,  # Search mode: ยังไม่รู้ object size
                                            adaptive_min_area=min_motion_area,
                                            hybrid_tracker=self)
            mask_source = "detect_mask (from main loop)"
            mask_for_count = detect_mask
        else:
            motion_boxes = find_motion_boxes(mask_cpu_processed, min_motion_area,
                                            area_mask=motion_detection_area_mask,
                                            roi_boxes=roi_boxes if len(roi_boxes) > 0 else None,
                                            object_size_category=None,  # Search mode: ยังไม่รู้ object size
                                            adaptive_min_area=min_motion_area,
                                            hybrid_tracker=self)
            mask_source = "mask_cpu_processed (hybrid tracker)"
            mask_for_count = mask_cpu_processed

        # เก็บ motion_boxes ไว้ใน instance variable สำหรับวาดบนหน้าจอ
        self.motion_boxes = motion_boxes

        # Update footprints module (เก็บ motion boxes history) - เรียกทุกเฟรม
        if FOOTPRINTS_MODULE_ENABLED and self.footprints_drawer:
            # หา horizon_mask และ motion_area_mask (ถ้ามี)
            horizon_mask = getattr(self, 'horizon_mask_upper', None)
            motion_area_mask = getattr(self, 'motion_detection_area_mask', None)

            # หา YOLO boxes จาก history ล่าสุด (ถ้ามี) เพื่อใช้เป็น reference สำหรับ size filtering
            yolo_boxes_reference = None
            if hasattr(self.footprints_drawer, 'yolo_boxes_history') and self.footprints_drawer.yolo_boxes_history:
                # ใช้ YOLO boxes จาก history ล่าสุด (เฟรมล่าสุดที่มี YOLO)
                last_yolo_frame, last_yolo_boxes = self.footprints_drawer.yolo_boxes_history[-1]
                # แปลงเป็น format (x, y, w, h, conf) สำหรับใช้ใน filtering
                yolo_boxes_reference = []
                for box in last_yolo_boxes:
                    if len(box) >= 5:
                        yolo_boxes_reference.append((box[0], box[1], box[2], box[3], box[4]))
                    elif len(box) >= 4:
                        yolo_boxes_reference.append((box[0], box[1], box[2], box[3], 0.0))

            # เรียก update_motion_boxes() ทุกเฟรม (ไม่ต้องรอ YOLO detection)
            # ใช้ YOLO boxes จาก history ล่าสุดเป็น reference (ถ้ามี)
            self.footprints_drawer.update_motion_boxes(
                frame_counter,
                motion_boxes,
                horizon_mask=horizon_mask,
                motion_area_mask=motion_area_mask,
                frame_w=w_orig,
                frame_h=h_orig,
                yolo_boxes=yolo_boxes_reference  # ใช้ YOLO boxes จาก history ล่าสุด (ถ้ามี)
            )

        # Track motion-only targets (เฉพาะเมื่อเปิดใช้งาน)
        if MOTION_ONLY_TARGET_ENABLED:
            # กรอง motion boxes ที่ไม่ match กับ existing targets
            unmatched_motion_boxes = []
            for motion_box in motion_boxes:
                is_matched = False
                for target_id, target in self.targets.items():
                    if 'last_center' in target:
                        motion_center = (motion_box[0] + motion_box[2]//2, motion_box[1] + motion_box[3]//2)
                        dist = self.get_dist(motion_center, target['last_center'])
                        if dist < HYBRID_DIST_THRESHOLD:
                            is_matched = True
                            break
                if not is_matched:
                    unmatched_motion_boxes.append(motion_box)

            # Track motion-only targets
            self.track_motion_only_targets(unmatched_motion_boxes, frame_counter, w_orig, h_orig)

        if should_search:
            # Performance optimization: Early exit ถ้าไม่มี motion boxes และไม่มี motion-only targets
            # (ถ้ามี motion-only targets อาจจะยังมีโอกาสเจอ YOLO)
            has_motion_only_targets = len(self.motion_only_targets) > 0 if hasattr(self, 'motion_only_targets') else False
            if len(motion_boxes) == 0 and not has_motion_only_targets:
                # ไม่มี motion boxes และไม่มี motion-only targets → ข้าม YOLO detection (ประหยัด performance)
                if DEBUG_MODE:
                    print(f"🔍 Search mode: Skipping YOLO detection (no motion boxes, no motion-only targets)")
            else:
                if DEBUG_MODE:
                    print(f"🔍 Search mode: No targets, starting search...")

                # Debug: แสดงจำนวน motion boxes และข้อมูล mask
                if DEBUG_MODE:
                    mask_nonzero = cv2.countNonZero(mask_for_count)
                    print(f"🔍 Search mode: {mask_source} - nonzero pixels: {mask_nonzero}, found {len(motion_boxes)} motion boxes (min_area={min_motion_area})")
                    if len(motion_boxes) > 0:
                        for i, mb in enumerate(motion_boxes[:3]):  # แสดงแค่ 3 ตัวแรก
                            print(f"   Motion box {i}: {mb}")

                try:
                    # frame_bgr is already BGR (3 channels) from earlier conversion, use directly
                    frame_bgr_check = frame_bgr

                    # Optimized search strategy: Motion-first detection to reduce predict calls
                    results = None
                    stage_used = None

                    # If motion boxes exist, try motion-first detection with low threshold first
                    # This is faster and more accurate when motion is detected
                    if len(motion_boxes) > 0:
                        # Motion-first: try low threshold first (most likely to find small/distant drones)
                        results = self.model.predict(frame_bgr_check, device=0, verbose=False, conf=HYBRID_SEARCH_CONF_LOW)
                        if len(results[0].boxes) > 0:
                            stage_used = "motion-first"
                            if DEBUG_MODE:
                                print(f"🔍 Search mode: Motion-first detection found {len(results[0].boxes)} boxes with conf={HYBRID_SEARCH_CONF_LOW:.3f}")
                        else:
                            # If motion-first fails, try normal threshold (for larger/clearer objects)
                            results = self.model.predict(frame_bgr_check, device=0, verbose=False, conf=HYBRID_BASE_SEARCH_CONF)
                            if len(results[0].boxes) > 0:
                                stage_used = "normal (after motion-first)"
                                if DEBUG_MODE:
                                    print(f"🔍 Search mode: Found {len(results[0].boxes)} YOLO boxes with normal threshold ({HYBRID_BASE_SEARCH_CONF:.3f})")
                    else:
                        # No motion boxes: use normal threshold first, then medium if needed
                        results = self.model.predict(frame_bgr_check, device=0, verbose=False, conf=HYBRID_BASE_SEARCH_CONF)
                        if len(results[0].boxes) > 0:
                            stage_used = "normal"
                            if DEBUG_MODE:
                                print(f"🔍 Search mode: Found {len(results[0].boxes)} YOLO boxes with normal threshold ({HYBRID_BASE_SEARCH_CONF:.3f})")
                        else:
                            # Try medium threshold as fallback
                            results = self.model.predict(frame_bgr_check, device=0, verbose=False, conf=HYBRID_SEARCH_CONF_MEDIUM)
                            if len(results[0].boxes) > 0:
                                stage_used = "medium"
                                if DEBUG_MODE:
                                    print(f"🔍 Search mode: Found {len(results[0].boxes)} YOLO boxes with medium threshold ({HYBRID_SEARCH_CONF_MEDIUM:.3f})")

                    if results is not None and len(results[0].boxes) > 0:
                        # เก็บ YOLO boxes พร้อม confidence สำหรับ footprints module
                        yolo_boxes_for_footprints = []  # [(x, y, w, h, conf), ...]
                        for box in results[0].boxes:
                            b = box.xyxy[0].cpu().numpy()
                            yolo_rect = (int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1]))
                            confidence = float(box.conf[0])
                            # เก็บ YOLO box พร้อม confidence สำหรับ footprints
                            yolo_boxes_for_footprints.append((yolo_rect[0], yolo_rect[1], yolo_rect[2], yolo_rect[3], confidence))
                            yolo_center = (yolo_rect[0] + yolo_rect[2]//2, yolo_rect[1] + yolo_rect[3]//2)
                            original_confidence = confidence  # เก็บค่าเดิมไว้สำหรับ debug

                            # YOLO Immediate Lock: ถ้า YOLO confidence สูงมาก → lock เป็น target ทันที (ไม่ต้องรอ path)
                            size_category = self.get_object_size_category_for_target_from_rect(yolo_rect, w_orig, h_orig)
                            immediate_lock_conf = self.get_immediate_lock_threshold(size_category)

                            # ตรวจสอบว่าไม่เกิน MAX_TOTAL_TARGETS (รวม motion-only + YOLO targets)
                            total_targets = len(self.targets) + len(self.motion_only_targets)
                            should_immediate_lock = (confidence >= immediate_lock_conf and total_targets < MAX_TOTAL_TARGETS)

                            # YOLO Immediate RED: ถ้า confidence สูงมาก (>= 0.6-0.8) → เปลี่ยนเป็น RED ทันทีโดยไม่ต้องรอ path
                            from config import YOLO_IMMEDIATE_RED_ENABLED, YOLO_IMMEDIATE_RED_CONF_THRESHOLD
                            should_immediate_red = False
                            if YOLO_IMMEDIATE_RED_ENABLED and confidence >= YOLO_IMMEDIATE_RED_CONF_THRESHOLD:
                                should_immediate_red = True
                                if DEBUG_MODE:
                                    print(f"🔴 YOLO Immediate RED: Target (conf={confidence:.3f} >= {YOLO_IMMEDIATE_RED_CONF_THRESHOLD:.3f})")

                            # ตรวจสอบว่า detection อยู่ในโซนบนเส้นฟ้าหรือไม่ (ถ้ามี horizon_mask_upper)
                            if horizon_mask_upper is not None:
                                # ตรวจสอบว่า center ของ YOLO box อยู่ในโซนบนเส้นฟ้าหรือไม่
                                center_x, center_y = yolo_center
                                if center_y < 0 or center_y >= horizon_mask_upper.shape[0] or center_x < 0 or center_x >= horizon_mask_upper.shape[1]:
                                    if DEBUG_MODE:
                                        print(f"🚫 Horizon filter: YOLO detection at ({center_x}, {center_y}) is outside frame bounds, skipping")
                                    continue

                                # ตรวจสอบว่า center อยู่ในโซนบนเส้นฟ้าหรือไม่
                                if horizon_mask_upper[center_y, center_x] == 0:
                                    if DEBUG_MODE:
                                        print(f"🚫 Horizon filter: YOLO detection at ({center_x}, {center_y}) is below horizon line, skipping")
                                    continue

                                if DEBUG_MODE:
                                    print(f"✅ Horizon filter: YOLO detection at ({center_x}, {center_y}) is above horizon line, allowing")

                            # เพิ่ม partial motion boost: ถ้า YOLO box อยู่ใกล้ motion box (ไม่ต้อง overlap เต็ม)
                            has_motion_overlap = False
                            has_partial_motion = False
                            max_iou = 0.0
                            min_motion_dist = float('inf')
                            best_motion_box = None

                            for motion_box in motion_boxes:
                                iou = calculate_iou(yolo_rect, motion_box)
                                max_iou = max(max_iou, iou)

                                # Full overlap
                                if iou > HYBRID_MOTION_IOU_THRESHOLD:
                                    has_motion_overlap = True
                                    best_motion_box = motion_box
                                    break

                                # Partial boost: คำนวณระยะห่างระหว่าง centers
                                motion_center = (motion_box[0] + motion_box[2]//2, motion_box[1] + motion_box[3]//2)
                                dist_to_motion = self.get_dist(yolo_center, motion_center)
                                if dist_to_motion < min_motion_dist:
                                    min_motion_dist = dist_to_motion
                                    best_motion_box = motion_box

                            # Partial boost: ถ้าอยู่ใกล้ motion box (แต่ไม่ overlap เต็ม)
                            if not has_motion_overlap and min_motion_dist < HYBRID_MOTION_PARTIAL_DIST:
                                has_partial_motion = True

                            # Boost confidence
                            if has_motion_overlap:
                                confidence = confidence * HYBRID_MOTION_CONF_BOOST
                                confidence = min(confidence, HYBRID_MOTION_CONF_MAX)
                                if DEBUG_MODE:
                                    print(f"✅ FULL BOOST! YOLO box overlaps with motion (IOU={max_iou:.4f}), conf: {original_confidence:.4f} -> {confidence:.4f}")
                            elif has_partial_motion:
                                confidence = confidence * HYBRID_MOTION_PARTIAL_BOOST
                                confidence = min(confidence, HYBRID_MOTION_CONF_MAX)
                                if DEBUG_MODE:
                                    print(f"✅ PARTIAL BOOST! YOLO box near motion (dist={min_motion_dist:.1f}px), conf: {original_confidence:.4f} -> {confidence:.4f}")
                            elif DEBUG_MODE and len(motion_boxes) > 0:
                                print(f"⚠️ No boost: YOLO box {yolo_rect} does NOT overlap with motion boxes (max IOU={max_iou:.4f} <= {HYBRID_MOTION_IOU_THRESHOLD:.2f}), conf stays {original_confidence:.4f}")

                            # ตรวจสอบว่าโดรนนี้ใกล้กับ target ที่มีอยู่แล้วหรือไม่
                            is_new_target = True
                            for target_id, target in self.targets.items():
                                last_center = target.get('last_center')
                                if last_center:
                                    dist = self.get_dist(yolo_center, last_center)
                                    # ใช้ threshold ใหญ่ขึ้นเพื่อป้องกัน duplicate แต่ไม่ใหญ่เกินไป
                                    # ถ้าโดรนอยู่ห่างกันมากกว่า 2.5 เท่าของ threshold = โดรนคนละตัว
                                    duplicate_threshold = HYBRID_DIST_THRESHOLD * 2.5  # เพิ่มจาก 2 เป็น 2.5
                                    if dist < duplicate_threshold:
                                        is_new_target = False
                                        if DEBUG_MODE:
                                            print(f"🔍 Search mode: YOLO detection at {yolo_center} is too close to existing target {target_id} (dist={dist:.1f} < {duplicate_threshold:.1f}), skipping")
                                        break

                            # ถ้าเป็นโดรนใหม่ ให้สร้าง target ใหม่
                            if is_new_target:
                                # ตรวจสอบ blacklist แบบ dynamic - ตรวจสอบว่าตำแหน่งอยู่ในพื้นที่ blacklist หรือไม่
                                # ถ้าอยู่ใน blacklist → ตรวจสอบ movement
                                # ถ้ามี movement → อนุญาตให้ตรวจจับ (โดรนจริงที่บินผ่าน)
                                # ถ้าไม่มี movement → กรอง (วัตถุนิ่ง)
                                # motion_boxes ถูกสร้างใน should_search scope แล้ว
                                if self.is_in_blacklist(yolo_center, yolo_rect, motion_boxes):
                                    if DEBUG_MODE:
                                        print(f"🚫 Search mode: YOLO detection at {yolo_center} is in blacklist area and has no movement, skipping")
                                    continue
                                elif DEBUG_MODE and self._check_if_in_blacklist_area(yolo_center, yolo_rect):
                                    print(f"✅ Search mode: YOLO detection at {yolo_center} is in blacklist area but has movement, allowing detection")

                                # ใช้ MIN_CONFIDENCE_FOR_TRACKING_SEARCH สำหรับ search mode
                                temp_target = {
                                    'original_confidence': original_confidence,
                                    'confidence': confidence
                                }
                                if not is_likely_drone(temp_target, min_confidence=MIN_CONFIDENCE_FOR_TRACKING_SEARCH):
                                    if DEBUG_MODE:
                                        print(f"🚫 Search mode: Skipping non-drone target (conf={original_confidence:.2f} < {MIN_CONFIDENCE_FOR_TRACKING_SEARCH:.2f})")
                                    continue

                                # ตรวจสอบ YOLO + motion overlap สำหรับ fast RED
                                from config import (
                                    YOLO_MOTION_OVERLAP_RED_ENABLED, YOLO_MOTION_OVERLAP_MIN_IOU,
                                    YOLO_MOTION_OVERLAP_MIN_CONF
                                )

                                should_fast_red = False
                                if YOLO_MOTION_OVERLAP_RED_ENABLED and has_motion_overlap:
                                    if original_confidence >= YOLO_MOTION_OVERLAP_MIN_CONF and max_iou >= YOLO_MOTION_OVERLAP_MIN_IOU:
                                        should_fast_red = True
                                        if DEBUG_MODE:
                                            print(f"🚀 Search mode: YOLO+motion overlap detected (conf={original_confidence:.2f}, IOU={max_iou:.2f}) → Fast RED")

                                target_id = self.next_target_id
                                self.next_target_id += 1
                                # ใช้ frame_counter จาก process_frame parameter
                                current_frame = getattr(self, '_current_frame_counter', 0)

                                # คำนวณ rect จาก yolo_rect สำหรับเก็บ last_rect
                                tx, ty, tw, th = yolo_rect[0], yolo_rect[1], yolo_rect[2], yolo_rect[3]
                                last_rect = (tx, ty, tw, th)  # เก็บ rect สำหรับวาด

                                # lock_mode ถูกตั้งไว้แล้วในส่วน initial_status
                                if should_immediate_lock and DEBUG_MODE:
                                    print(f"🔒 YOLO Immediate Lock: Target {target_id} (size={size_category}, conf={confidence:.3f} >= {immediate_lock_conf:.3f})")

                                roi_box = [
                                        max(0, yolo_rect[0] - HYBRID_ROI_PADDING),
                                        max(0, yolo_rect[1] - HYBRID_ROI_PADDING),
                                        min(w_orig, yolo_rect[0] + yolo_rect[2] + HYBRID_ROI_PADDING),
                                        min(h_orig, yolo_rect[1] + yolo_rect[3] + HYBRID_ROI_PADDING)
                                ]
                                roi_center = ((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)

                                # ตั้ง initial status:
                                # - ถ้า should_immediate_red → RED ทันที
                                # - ถ้า should_fast_red → ORANGE (จะเปลี่ยนเป็น RED เร็ว)
                                # - มิฉะนั้น → GREEN
                                if should_immediate_red:
                                    initial_status = 'RED'
                                    lock_mode = True  # ตั้ง lock_mode = True สำหรับ immediate RED
                                    red_score = 1.0  # ตั้ง red_score = 1.0 เพื่อให้คง RED status
                                elif should_fast_red:
                                    initial_status = 'ORANGE'
                                    lock_mode = should_immediate_lock
                                    red_score = 0.0
                                else:
                                    initial_status = 'GREEN'
                                    lock_mode = should_immediate_lock
                                    red_score = 0.0

                                self.targets[target_id] = {
                                    'target_locked': True,
                                    'lock_mode': lock_mode,  # ตั้ง lock_mode สำหรับ immediate lock
                                    'roi_box': roi_box,
                                    'roi_box_history': [(current_frame, tuple(roi_box), roi_center)],  # เริ่มต้นด้วย ROI แรก
                                    'last_center': yolo_center,
                                    'last_rect': last_rect,  # เก็บ rect สำหรับวาด
                                    'missed_frames': 0,
                                    'confidence': confidence,  # boosted confidence (ใช้สำหรับ lock/tracking)
                                    'original_confidence': original_confidence,  # original confidence (ใช้สำหรับ sound alert)
                                    'object_size_category': size_category,  # เก็บ size category
                                    'tracking_start_frame': current_frame,
                                    'status': initial_status,  # RED สำหรับ immediate RED, ORANGE สำหรับ fast RED, GREEN สำหรับปกติ
                                    'should_fast_red': should_fast_red,  # เก็บ flag สำหรับ fast RED
                                    'is_primary': False,
                                    'stationary_suspect': False,  # ยังไม่สงสัยว่าเป็นวัตถุนิ่ง
                                    # RED lock fields
                                    'red_score': red_score if should_immediate_red else 0.0,  # ตั้ง red_score = 1.0 สำหรับ immediate RED
                                    'red_window_start': current_frame,
                                    'yolo_frame_counter': 0,  # สำหรับ interval-based YOLO detection
                                    'centers_history': [(current_frame, yolo_center)],  # เริ่มเก็บ path history ทันที
                                    'last_update_frame': current_frame  # เก็บ last_update_frame
                                }
                                if DEBUG_MODE:
                                    if has_motion_overlap:
                                        print(f"🔍 Search mode: Found NEW target {target_id} at {yolo_center} with conf={confidence:.2f} (boosted from {original_confidence:.4f}, overlaps with motion), locking... (total targets: {len(self.targets)})")
                                    else:
                                        print(f"🔍 Search mode: Found NEW target {target_id} at {yolo_center} with conf={confidence:.2f} (no motion overlap), locking... (total targets: {len(self.targets)})")

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"⚠️ YOLO detection error in search mode: {e}")

                # Update footprints module (เก็บ YOLO boxes history) - หลังจากประมวลผล YOLO results
                # yolo_boxes_for_footprints เก็บเป็น (x, y, w, h, conf) แล้ว
                if FOOTPRINTS_MODULE_ENABLED and self.footprints_drawer and 'yolo_boxes_for_footprints' in locals() and yolo_boxes_for_footprints:
                    self.footprints_drawer.update_yolo_detections(
                        frame_counter,
                        yolo_boxes_for_footprints,
                        motion_boxes,
                        frame_w=w_orig,
                        frame_h=h_orig
                    )
                    # หมายเหตุ: update_motion_boxes() ถูกเรียกทุกเฟรมแล้วในส่วนบน (บรรทัด ~7975)
                    # และจะใช้ YOLO boxes จาก history ล่าสุดเป็น reference อัตโนมัติ

        # === LIMIT TARGETS TO MAX_TOTAL_TARGETS (รวม motion-only + YOLO targets) ===
        # เรียงลำดับ targets ตาม priority score (confidence + path quality) และเก็บไว้แค่ MAX_TOTAL_TARGETS ตัวแรก
        total_targets = len(self.targets) + len(self.motion_only_targets)
        if total_targets > MAX_TOTAL_TARGETS:
            # รวม targets ทั้งหมดและคำนวณ priority score
            all_targets_with_priority = []

            # เพิ่ม YOLO targets
            for tid, target in self.targets.items():
                confidence = target.get('confidence', target.get('original_confidence', 0.0))
                path_quality = target.get('path_quality', 0.0)
                priority = confidence * 0.7 + path_quality * 0.3  # Weight: YOLO 70%, Motion 30%
                all_targets_with_priority.append(('yolo', tid, priority))

            # เพิ่ม motion-only targets
            for mid, motion_target in self.motion_only_targets.items():
                path_quality = motion_target.get('path_quality', 0.0)
                priority = path_quality  # Motion-only: ใช้ path quality เท่านั้น
                all_targets_with_priority.append(('motion', mid, priority))

            # เรียงตาม priority ลดลง
            all_targets_with_priority.sort(key=lambda x: x[2], reverse=True)

            # ลบ targets ที่เกิน limit (เริ่มจาก priority ต่ำสุด)
            targets_to_remove = []
            motion_targets_to_remove = []

            for i, (target_type, target_id, priority) in enumerate(all_targets_with_priority):
                if i >= MAX_TOTAL_TARGETS:
                    if target_type == 'yolo':
                        targets_to_remove.append(target_id)
                    else:
                        motion_targets_to_remove.append(target_id)

            # ลบ targets
            for tid in targets_to_remove:
                if tid in self.targets:
                    removed_conf = self.targets[tid].get('confidence', self.targets[tid].get('original_confidence', 0.0))
                    del self.targets[tid]
                    if DEBUG_MODE:
                        print(f"🗑️ Removed excess YOLO target {tid} (conf={removed_conf:.2f}) to limit tracking to {MAX_TOTAL_TARGETS} total targets")

            for mid in motion_targets_to_remove:
                if mid in self.motion_only_targets:
                    removed_quality = self.motion_only_targets[mid].get('path_quality', 0.0)
                    del self.motion_only_targets[mid]
                    if DEBUG_MODE:
                        print(f"🗑️ Removed excess motion-only target {mid} (quality={removed_quality:.2f}) to limit tracking to {MAX_TOTAL_TARGETS} total targets")

        # Legacy: ยังคงใช้ MAX_ROI_TRACKING สำหรับ YOLO targets เท่านั้น (backward compatibility)
        if len(self.targets) > MAX_ROI_TRACKING:
            # สร้าง list ของ (target_id, confidence) และเรียงตาม confidence ลดลง
            targets_with_conf = [
                (tid, target.get('confidence', target.get('original_confidence', 0.0)))
                for tid, target in self.targets.items()
            ]
            # เรียงตาม confidence ลดลง (confidence สูงสุดก่อน)
            targets_with_conf.sort(key=lambda x: x[1], reverse=True)

            # เก็บ target_ids ที่จะลบ (ตัวที่ MAX_ROI_TRACKING+1 ขึ้นไป)
            targets_to_remove_excess = []
            for i, (tid, conf) in enumerate(targets_with_conf):
                if i >= MAX_ROI_TRACKING:  # เก็บแค่ MAX_ROI_TRACKING ตัวแรก
                    targets_to_remove_excess.append(tid)

            # ลบ targets ที่เกิน MAX_ROI_TRACKING ตัว
            for tid in targets_to_remove_excess:
                if tid in self.targets:
                    removed_conf = self.targets[tid].get('confidence', self.targets[tid].get('original_confidence', 0.0))
                    del self.targets[tid]
                    if DEBUG_MODE:
                        print(f"🗑️ Removed excess target {tid} (conf={removed_conf:.2f}) to limit tracking to {MAX_ROI_TRACKING} ROIs (total was {len(targets_with_conf)})")

        # === TRACKING MODE: ติดตามแต่ละ target ===
        targets_to_remove = []
        all_high_confidence = True
        max_confidence = 0.0
        any_target_locked = False

        for target_id, target in self.targets.items():
            if not target['target_locked']:
                continue

            any_target_locked = True
            rx1, ry1, rx2, ry2 = target['roi_box']
            rx1 = max(0, min(rx1, w_orig))
            ry1 = max(0, min(ry1, h_orig))
            rx2 = max(rx1, min(rx2, w_orig))
            ry2 = max(ry1, min(ry2, h_orig))

            # กำหนด base confidence threshold สำหรับ ROI นี้
            if rx2 > rx1 and ry2 > ry1:
                inference_area = frame_bgr[ry1:ry2, rx1:rx2]
                base_conf = HYBRID_SEARCH_CONF
            else:
                inference_area = frame_bgr
                rx1, ry1 = 0, 0
                base_conf = HYBRID_BASE_SEARCH_CONF

            # เริ่มต้น current_conf จาก base_conf
            current_conf = base_conf

            # Dynamic confidence สำหรับ lock target:
            # - ใช้ last_yolo_conf ของ target นั้นค่อย ๆ ลด threshold ลง
            # - แต่ไม่ต่ำกว่า 0.02 และไม่สูงกว่า base_conf
            if target.get('lock_mode', False):
                last_conf = target.get('last_yolo_conf', None)
                if last_conf is not None:
                    # ลดลงทีละ ~20% จาก conf ล่าสุด แต่ไม่ต่ำเกินไป
                    dynamic_conf = max(0.02, min(base_conf, last_conf * 0.8))
                    current_conf = dynamic_conf

            # YOLO Detection in ROI (ใช้ interval สำหรับ lock target)
            yolo_box = None
            if self.model is not None:
                # ตรวจสอบว่าควรเรียก YOLO ในเฟรมนี้หรือไม่
                if 'yolo_frame_counter' not in target:
                    target['yolo_frame_counter'] = 0

                # สำหรับ lock target: ใช้ LOCK_YOLO_INTERVAL (ทุกเฟรม), สำหรับ target ปกติ: ใช้ HYBRID_YOLO_INTERVAL
                yolo_interval = HYBRID_YOLO_INTERVAL
                if target.get('lock_mode', False):
                    yolo_interval = LOCK_YOLO_INTERVAL

                target['yolo_frame_counter'] += 1
                should_run_yolo = (target['yolo_frame_counter'] >= yolo_interval)

                if should_run_yolo:
                    target['yolo_frame_counter'] = 0

                try:
                    if should_run_yolo:
                        if inference_area.shape[2] != 3:
                            inference_area = cv2.cvtColor(inference_area, cv2.COLOR_BGRA2BGR)

                        # ใช้ confidence threshold ปกติเท่านั้น (ไม่ใช้ boost ใน tracking mode)
                        results = self.model.predict(inference_area, device=0, verbose=False, conf=current_conf)

                        if len(results[0].boxes) > 0:
                            best_yolo_idx = -1
                            min_yolo_dist = float('inf')
                            best_combined_score = float('inf')

                            # คำนวณ predicted position และ tracking duration ครั้งเดียว
                            predicted_center = None
                            if 'centers_history' in target and len(target['centers_history']) >= 2:
                                predicted_center = self.predict_next_position(target['centers_history'])

                            tracking_duration = frame_counter - target.get('tracking_start_frame', frame_counter) + 1
                            target['tracking_duration'] = tracking_duration

                            for i, box in enumerate(results[0].boxes):
                                b = box.xyxy[0].cpu().numpy()
                                curr_yolo_rect = (int(b[0]+rx1), int(b[1]+ry1), int(b[2]-b[0]), int(b[3]-b[1]))
                                curr_yolo_center = (curr_yolo_rect[0] + curr_yolo_rect[2]//2, curr_yolo_rect[1] + curr_yolo_rect[3]//2)

                                # ตรวจสอบว่าเป็นแมลงหรือไม่ (กรองก่อนเลือก)
                                # ถ้าอยู่ใน lock_mode แล้ว ให้ผ่อน insect filter (ไม่กรองซ้ำ)
                                if not target.get('lock_mode', False):
                                    if self.is_likely_insect_in_roi(curr_yolo_center, target):
                                        if DEBUG_MODE:
                                            print(f"🚫 Target {target_id}: Skipping YOLO box {i} (likely insect) at {curr_yolo_center}")
                                        continue  # ข้าม box ที่เป็นแมลง

                                # คำนวณ distance และ path consistency
                                dist_to_last = self.get_dist(target['last_center'], curr_yolo_center)

                                # คำนวณ path consistency score
                                consistency_score = 0.5  # Default
                                if 'centers_history' in target and len(target['centers_history']) >= 2:
                                    consistency_score = self.calculate_path_consistency(curr_yolo_center, target['centers_history'], predicted_center)

                                # ใช้ combined score: distance + consistency
                                # consistency สูง = ลด distance score
                                combined_score = dist_to_last * (1.0 - consistency_score * 0.5)

                                # เลือก box ที่มี combined score ต่ำสุด (ใกล้ + สอดคล้อง)
                                if combined_score < best_combined_score:
                                    best_combined_score = combined_score
                                    min_yolo_dist = dist_to_last
                                    best_yolo_idx = i

                            if best_yolo_idx >= 0:
                                # ใช้ threshold พิเศษสำหรับ lock target (ยอมให้ขยับได้มากขึ้น)
                                dist_threshold = HYBRID_DIST_THRESHOLD
                                if target.get('lock_mode', False):
                                    dist_threshold = int(HYBRID_DIST_THRESHOLD * LOCK_DIST_MULTIPLIER)
                                if min_yolo_dist < dist_threshold:
                                    yolo_box = results[0].boxes[best_yolo_idx]
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"⚠️ YOLO detection error for target {target_id}: {e}")

            # Hybrid Logic
            final_rect = None
            status_label = ""
            color = (0, 0, 0)
            confidence = 0.0

            # Fallback: ถ้าไม่มี yolo_box และอยู่ใน lock_mode → ใช้ predicted center + last_rect ช่วย
            if yolo_box is None and target.get('lock_mode', False):
                try:
                    if 'centers_history' in target and len(target['centers_history']) >= 2 and target.get('last_rect'):
                        pred_center = self.predict_next_position(target['centers_history'])
                        if pred_center is not None:
                            tx, ty, tw, th = target['last_rect']
                            px, py = int(pred_center[0]), int(pred_center[1])
                            final_rect = (px - tw//2, py - th//2, tw, th)
                            if DEBUG_MODE:
                                print(f"🔒 Target {target_id}: Using predicted center fallback at {pred_center}")
                except Exception:
                    pass

            if yolo_box is not None:
                b = yolo_box.xyxy[0].cpu().numpy()
                final_rect = (int(b[0]+rx1), int(b[1]+ry1), int(b[2]-b[0]), int(b[3]-b[1]))
                yolo_center = (final_rect[0] + final_rect[2]//2, final_rect[1] + final_rect[3]//2)

                # Quick check: ตรวจสอบ path consistency ทันที (ป้องกันแมลงที่หลุดผ่าน)
                is_insect_quick = False
                if 'centers_history' in target and len(target['centers_history']) >= 2:
                    tracking_duration = frame_counter - target.get('tracking_start_frame', frame_counter) + 1
                    if tracking_duration < 10:  # ตรวจสอบเฉพาะเมื่อ tracking duration < 10 เฟรม
                        is_insect_quick = self.is_likely_insect_in_roi(yolo_center, target)
                        if is_insect_quick and DEBUG_MODE:
                            print(f"🚫 Target {target_id}: Quick insect check failed (path inconsistency) at {yolo_center}")

                # ถ้าเป็นแมลงชัดเจน → ไม่ให้เป็น RED เลย
                if is_insect_quick:
                    color = (0, 255, 0)  # Green
                    status_label = "TRACKING"
                    target['status'] = 'GREEN'
                    target['is_insect'] = True
                    target['is_blacklisted'] = False
                    target['missed_frames'] = 0
                    target['confidence'] = float(yolo_box.conf[0])
                    target['original_confidence'] = target['confidence']
                    all_high_confidence = False
                    # อัปเดต position แต่ไม่ให้เป็น RED
                    target['last_rect'] = final_rect
                    target['last_center'] = yolo_center
                    target['roi_box'] = [
                        max(0, final_rect[0] - HYBRID_ROI_PADDING), max(0, final_rect[1] - HYBRID_ROI_PADDING),
                        min(w_orig, final_rect[0] + final_rect[2] + HYBRID_ROI_PADDING), min(h_orig, final_rect[1] + final_rect[3] + HYBRID_ROI_PADDING)
                    ]
                    # อัปเดต centers_history (ใช้ฟังก์ชัน optimize แทน)
                    self.update_path_history(target, yolo_center, frame_counter, status='GREEN')

                    # เก็บ size history สำหรับ tiny objects (interval-based storage)
                    # ใช้ cached size category
                    size_category = target.get('object_size_category')
                    if size_category is None:
                        size_category = self.get_object_size_category_for_target(target, w_orig, h_orig)
                        if size_category:
                            target['object_size_category'] = size_category
                    is_tiny = (size_category == 'TINY')

                    if is_tiny and final_rect:
                        # ใช้ interval-based storage (เก็บทุก 6 เฟรม) เพื่อไม่กระทบความเร็ว
                        if frame_counter % 6 == 0:  # เก็บทุก 6 เฟรม
                            if 'size_history' not in target:
                                target['size_history'] = []
                            tx, ty, tw, th = final_rect
                            area = tw * th
                            target['size_history'].append((frame_counter, area))
                            # จำกัดความยาว size history ให้เท่ากับ path history (เพื่อประหยัด memory)
                            # ใช้ max_history จาก size category
                            max_history = self._get_path_history_limit(size_category, target.get('status', 'GREEN'), target)
                            if max_history is None and size_category == 'TINY':
                                # TINY: ใช้ seconds-based
                                processing_fps = getattr(self, '_current_processing_fps', None)
                                if processing_fps is None or processing_fps <= 0:
                                    from config import DEFAULT_FPS
                                    processing_fps = DEFAULT_FPS
                                from config import TINY_OBJECT_PATH_HISTORY_SECONDS
                                max_history = int(TINY_OBJECT_PATH_HISTORY_SECONDS * processing_fps)
                                max_history = max(20, max_history)
                            if max_history and len(target['size_history']) > max_history:
                                target['size_history'] = target['size_history'][-max_history:]

                    continue  # ข้ามการตรวจสอบ RED status

                target['missed_frames'] = 0
                confidence = float(yolo_box.conf[0])
                original_confidence = confidence

                # คำนวณ size category (cache เพื่อ performance)
                size_category = target.get('object_size_category')
                if size_category is None:
                    size_category = self.get_object_size_category_for_target(target, w_orig, h_orig)
                    if size_category:
                        target['object_size_category'] = size_category

                # คำนวณ motion path quality สำหรับ size-adaptive RED decision
                # Performance optimization: Early exit ถ้าไม่มี path history เพียงพอ
                path_quality = 0.0
                is_path_clear = False
                centers_history = target.get('centers_history', [])
                if len(centers_history) >= DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS:
                    is_path_clear, path_quality = self.calculate_size_adaptive_path_quality(
                        target, frame_counter, size_category, is_motion_only=False
                    )
                # ถ้าไม่มี path history → ใช้ path_quality = 0.0 (ไม่ต้องคำนวณ)

                # Background blending detection
                is_blended = False
                if BLENDED_DETECTION_ENABLED and size_category:
                    # ตรวจสอบว่า YOLO confidence ต่ำ แต่ motion path quality สูง
                    if original_confidence < HYBRID_BASE_CONF and path_quality >= 0.6:
                        is_blended = True
                        if DEBUG_MODE:
                            print(f"🔍 Target {target_id}: Background blending detected (yolo={original_confidence:.3f}, motion={path_quality:.3f})")

                # คำนวณ composite score จาก YOLO และ motion path quality (size-adaptive weights)
                yolo_weight, motion_weight = self.get_size_adaptive_red_weights(size_category)

                # ปรับ weights ถ้า detect blending
                if is_blended:
                    yolo_weight = yolo_weight * BLENDED_OBJECT_YOLO_PENALTY
                    motion_weight = motion_weight * BLENDED_OBJECT_MOTION_BOOST
                    # Normalize weights
                    total_weight = yolo_weight + motion_weight
                    if total_weight > 0:
                        yolo_weight = yolo_weight / total_weight
                        motion_weight = motion_weight / total_weight

                # คำนวณ composite score
                composite_score = (original_confidence * yolo_weight) + (path_quality * motion_weight)

                # ใช้ composite score สำหรับ boost confidence (ถ้าจำเป็น)
                if is_path_clear and path_quality >= DRONE_PATH_BASED_ORANGE_MOTION_CLEAR_THRESHOLD:
                    # Boost confidence จาก composite score
                    path_boost = path_quality * 0.3  # boost สูงสุด 30%
                    confidence = min(1.0, confidence + path_boost)
                    if DEBUG_MODE:
                        print(f"✅ Path quality boost: conf {original_confidence:.4f} -> {confidence:.4f} "
                              f"(quality={path_quality:.2f}, composite={composite_score:.2f}, weights: yolo={yolo_weight:.2f}, motion={motion_weight:.2f})")

                target['confidence'] = confidence
                target['original_confidence'] = original_confidence  # เก็บค่าเดิมไว้
                target['composite_score'] = composite_score  # เก็บ composite score สำหรับใช้ใน RED decision
                target['path_quality'] = path_quality  # เก็บ path quality
                # เก็บ last_yolo_conf สำหรับ dynamic confidence ใน lock mode
                target['last_yolo_conf'] = confidence
                max_confidence = max(max_confidence, confidence)

                # ใช้ original_confidence สำหรับการตรวจสอบ sound alert
                original_conf = target.get('original_confidence', confidence)
                if original_conf >= HYBRID_BASE_CONF:
                    # ตรวจสอบแมลงและ blacklist ก่อนตั้ง status เป็น RED
                    # Interval-based checking: ตรวจสอบทุก 6 เฟรม (ไม่ใช่ทุกเฟรม) เพื่อลดการคำนวณ
                    check_interval = 6
                    last_check_frame = target.get('last_insect_check_frame', -check_interval)
                    current_frame = getattr(self, '_current_frame_counter', 0)
                    should_check = (current_frame - last_check_frame) >= check_interval

                    if should_check:
                        is_insect = False
                        is_blacklisted = False

                        # ตรวจสอบแมลง (เฉพาะเมื่อมี path history เพียงพอ)
                        if 'centers_history' in target and len(target['centers_history']) >= 10:
                            is_insect = self.is_likely_insect_for_target(target, current_frame)
                            target['last_insect_check_frame'] = current_frame

                        # ตรวจสอบ blacklist
                        if not is_insect:  # ถ้าเป็นแมลงแล้ว ไม่ต้องตรวจ blacklist
                            # ใช้ cached velocity ถ้ามี (อัปเดตทุก 10 เฟรม)
                            velocity = 0.0
                            cached_velocity_frame = target.get('cached_velocity_frame', -10)
                            if 'cached_velocity' in target and (current_frame - cached_velocity_frame) < 10:
                                cached_val = target['cached_velocity']
                                # ตรวจสอบว่า cached_velocity เป็น tuple (vx, vy) หรือ float (velocity magnitude)
                                if isinstance(cached_val, (tuple, list)) and len(cached_val) == 2:
                                    # เป็น tuple (vx, vy) → คำนวณ magnitude
                                    vx, vy = cached_val
                                    if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                                        velocity = math.sqrt(vx**2 + vy**2)
                                elif isinstance(cached_val, (int, float)):
                                    # เป็น float (velocity magnitude) → ใช้ค่าโดยตรง
                                    velocity = float(cached_val)
                            else:
                                # คำนวณ velocity สำหรับ blacklist check
                                if 'centers_history' in target and len(target['centers_history']) >= 2:
                                    first_frame, first_center = target['centers_history'][0]
                                    last_frame, last_center = target['centers_history'][-1]
                                    if first_center and last_center:
                                        distance = self.get_dist(first_center, last_center)
                                        frames_diff = last_frame - first_frame
                                        if frames_diff > 0:
                                            velocity = distance / frames_diff
                                # Cache velocity (เก็บเป็น float สำหรับ blacklist check)
                                target['cached_velocity'] = velocity
                                target['cached_velocity_frame'] = current_frame

                            # motion_boxes อาจไม่มีใน tracking mode, ใช้ None
                            is_blacklisted = self.is_in_blacklist(target['last_center'], final_rect, None, velocity)
                            target['last_blacklist_check_frame'] = current_frame

                        # เก็บผลลัพธ์ไว้ใน cache
                        target['is_insect'] = is_insect
                        target['is_blacklisted'] = is_blacklisted
                    else:
                        # ใช้ cached values ในเฟรมที่ไม่ตรวจสอบ
                        is_insect = target.get('is_insect', False)
                        is_blacklisted = target.get('is_blacklisted', False)

                    # ตรวจสอบสถานะ stationary_suspect (candidate สำหรับ blacklist)
                    is_suspect = target.get('stationary_suspect', False)

                    # ตรวจสอบ airplane/star สำหรับ tiny objects
                    is_airplane = target.get('is_airplane', False)
                    is_star = target.get('is_star', False)

                    # ถ้าเป็นแมลง, อยู่ใน blacklist, กำลังถูกพิจารณาเป็นวัตถุนิ่ง, เป็นเครื่องบิน, หรือเป็นดาว → ไม่ให้เป็น RED
                    if is_insect or is_blacklisted or is_suspect or is_airplane or is_star:
                        color = (0, 255, 0)  # Green
                        status_label = "TRACKING"
                        target['status'] = 'GREEN'
                        all_high_confidence = False
                        if DEBUG_MODE and should_check:  # แสดง debug เฉพาะเมื่อตรวจสอบจริง
                            if is_insect:
                                reason = "insect"
                            elif is_blacklisted:
                                reason = "blacklist"
                            elif is_suspect:
                                reason = "stationary_suspect"
                            elif is_airplane:
                                reason = f"airplane (conf={target.get('airplane_confidence', 0.0):.2f})"
                            elif is_star:
                                reason = "star"
                            else:
                                reason = "unknown"
                            print(f"🚫 Target {target_id}: Blocked RED status (conf={original_conf:.2f}) - {reason}")
                    else:
                        # --- แยก logic ก่อน lock / หลัง lock ---
                        if not target.get('lock_mode', False):
                            # ========== ก่อน lock: ใช้ logic เดิมทั้งหมด ==========
                            # --- เงื่อนไขเพิ่มเติมก่อนตั้ง RED + RED LOCK (ป้องกัน false + เพิ่มความมั่นใจ) ---
                            can_be_red = True
                            grid_conf = None
                            path_consistency = None

                            # 1) ระยะเวลาการติดตามต้องเพียงพอ
                            tracking_duration = current_frame - target.get('tracking_start_frame', current_frame) + 1
                            target['tracking_duration'] = tracking_duration
                            min_red_frames = 5
                            if tracking_duration < min_red_frames:
                                can_be_red = False

                            # 2) ตรวจสอบ path consistency (เฉพาะเมื่อมี history เพียงพอ)
                            centers_history = target.get('centers_history', [])
                            if can_be_red and centers_history and len(centers_history) >= 2:
                                try:
                                    last_center = centers_history[-1][1]
                                    if last_center:
                                        path_consistency = self.calculate_path_consistency(last_center, centers_history)
                                        if path_consistency is not None and path_consistency < 0.4:
                                            can_be_red = False
                                except Exception:
                                    pass

                            # 3) ตรวจสอบ grid noise (ถ้ามี grid_system และ duration ยังสั้น)
                            if can_be_red and hasattr(self, 'grid_system') and 'roi_box' in target and tracking_duration < 20:
                                try:
                                    rx1, ry1, rx2, ry2 = target['roi_box']
                                    gw = max(1, rx2 - rx1)
                                    gh = max(1, ry2 - ry1)
                                    grid_conf = self.grid_system.get_grid_confidence(
                                        rx1, ry1, gw, gh, current_frame, GRID_CONFIDENCE_CACHE_FRAMES
                                    )
                                    # ถ้า grid_conf ต่ำ แสดงว่าเป็นโซน noise สูง → ไม่ตั้ง RED
                                    if grid_conf is not None and grid_conf < GRID_NOISE_THRESHOLD_FOR_HYBRID:
                                        can_be_red = False
                                except Exception:
                                    pass

                            # --- อัปเดต RED accumulation score (สะสมเฟรม RED ไม่จำเป็นต้องต่อเนื่อง) ---
                            # เตรียมค่าเริ่มต้น
                            if 'red_score' not in target:
                                target['red_score'] = 0.0
                            if 'red_window_start' not in target:
                                target['red_window_start'] = current_frame

                            # จัดการหน้าต่างสะสมคะแนน
                            if current_frame - target['red_window_start'] > RED_ACCUM_WINDOW_FRAMES:
                                # ลดคะแนนลงบางส่วนเพื่อไม่ให้ลากยาวเกินไป แล้วเริ่มหน้าต่างใหม่
                                target['red_score'] *= 0.5
                                target['red_window_start'] = current_frame

                            # อัปเดตคะแนนตามสถานะในเฟรมนี้ (ใช้ composite score สำหรับ size-adaptive decision)
                            composite_score = target.get('composite_score', original_conf)

                            # ตรวจสอบ YOLO + motion overlap สำหรับ fast RED
                            motion_boxes_for_check = getattr(self, 'motion_boxes', [])
                            is_yolo_motion_overlap, max_iou, motion_quality, should_fast_red = self.check_yolo_motion_overlap_for_red(
                                target, motion_boxes_for_check, current_frame
                            )

                            # ใช้ composite score threshold ตาม size category (ใช้ config แทน hardcoded)
                            from config import (
                                TINY_RED_MIN_COMPOSITE, SMALL_RED_MIN_COMPOSITE,
                                MEDIUM_RED_MIN_COMPOSITE, LARGE_RED_MIN_COMPOSITE,
                                TINY_YOLO_MOTION_RED_MIN_COMPOSITE, SMALL_YOLO_MOTION_RED_MIN_COMPOSITE,
                                MEDIUM_YOLO_MOTION_RED_MIN_COMPOSITE, LARGE_YOLO_MOTION_RED_MIN_COMPOSITE,
                                TINY_YOLO_MOTION_RED_SCORE_BOOST, SMALL_YOLO_MOTION_RED_SCORE_BOOST,
                                MEDIUM_YOLO_MOTION_RED_SCORE_BOOST, LARGE_YOLO_MOTION_RED_SCORE_BOOST,
                                ROI_LOCK_RED_ENABLED, ROI_LOCK_RED_MIN_COMPOSITE, ROI_LOCK_RED_SCORE_BOOST
                            )

                            if size_category == 'TINY':
                                min_composite = TINY_RED_MIN_COMPOSITE
                                yolo_motion_min_composite = TINY_YOLO_MOTION_RED_MIN_COMPOSITE
                                yolo_motion_boost = TINY_YOLO_MOTION_RED_SCORE_BOOST
                            elif size_category == 'SMALL':
                                min_composite = SMALL_RED_MIN_COMPOSITE
                                yolo_motion_min_composite = SMALL_YOLO_MOTION_RED_MIN_COMPOSITE
                                yolo_motion_boost = SMALL_YOLO_MOTION_RED_SCORE_BOOST
                            elif size_category == 'MEDIUM':
                                min_composite = MEDIUM_RED_MIN_COMPOSITE
                                yolo_motion_min_composite = MEDIUM_YOLO_MOTION_RED_MIN_COMPOSITE
                                yolo_motion_boost = MEDIUM_YOLO_MOTION_RED_SCORE_BOOST
                            elif size_category == 'LARGE':
                                min_composite = LARGE_RED_MIN_COMPOSITE
                                yolo_motion_min_composite = LARGE_YOLO_MOTION_RED_MIN_COMPOSITE
                                yolo_motion_boost = LARGE_YOLO_MOTION_RED_SCORE_BOOST
                            else:
                                min_composite = 0.5  # Default
                                yolo_motion_min_composite = 0.4
                                yolo_motion_boost = 1.5

                            # สำหรับ YOLO + motion overlap: ใช้ threshold ต่ำกว่า
                            if should_fast_red and composite_score >= yolo_motion_min_composite:
                                # ใช้ threshold ต่ำกว่าสำหรับ fast RED
                                min_composite = yolo_motion_min_composite
                                # Boost red_score
                                if can_be_red:
                                    target['red_score'] = target['red_score'] * RED_DECAY_FACTOR + (1.0 * yolo_motion_boost)
                                if DEBUG_MODE:
                                    print(f"🚀 Target {target_id}: Fast RED (YOLO+motion overlap, IOU={max_iou:.2f}, composite={composite_score:.2f})")

                            # สำหรับ ROI lock mode: ใช้ threshold ต่ำกว่า
                            is_lock_mode = target.get('lock_mode', False)
                            if is_lock_mode and ROI_LOCK_RED_ENABLED:
                                # ใช้ threshold ต่ำกว่าสำหรับ lock mode
                                if composite_score >= ROI_LOCK_RED_MIN_COMPOSITE:
                                    min_composite = ROI_LOCK_RED_MIN_COMPOSITE
                                    # Boost red_score
                                    if can_be_red:
                                        target['red_score'] = target['red_score'] * RED_DECAY_FACTOR + (1.0 * ROI_LOCK_RED_SCORE_BOOST)
                                    if DEBUG_MODE:
                                        print(f"🔒 Target {target_id}: Lock mode RED boost (composite={composite_score:.2f})")

                            # ถ้า composite score สูงพอ → พร้อมเป็น RED
                            if can_be_red and composite_score >= min_composite:
                                # ถ้าพร้อมจะเป็น RED → เพิ่มคะแนนพร้อม decay
                                target['red_score'] = target['red_score'] * RED_DECAY_FACTOR + 1.0
                            else:
                                # ถ้าเงื่อนไข RED ไม่ผ่าน → ค่อย ๆ ลดคะแนนลง
                                target['red_score'] *= RED_DECAY_FACTOR

                            # ปรับ can_be_red ตาม composite score
                            if can_be_red and composite_score < min_composite:
                                can_be_red = False
                                if DEBUG_MODE:
                                    print(f"🚫 Target {target_id}: Blocked RED by composite score ({composite_score:.2f} < {min_composite:.2f}, size={size_category})")

                            # --- ตัดสินใจตั้ง RED / ORANGE ตามเงื่อนไขและคะแนนสะสม ---
                            if not can_be_red:
                                # ตั้งเป็น ORANGE หรือ GREEN แทน ขึ้นอยู่กับ duration
                                if tracking_duration >= min_red_frames:
                                    color = (0, 165, 255)  # Orange
                                    status_label = "TRACKING"
                                    target['status'] = 'ORANGE'
                                else:
                                    color = (0, 255, 0)  # Green
                                    status_label = "TRACKING"
                                    target['status'] = 'GREEN'
                                all_high_confidence = False

                                if DEBUG_MODE and should_check:
                                    debug_reason = []
                                    if tracking_duration < min_red_frames:
                                        debug_reason.append(f"duration<{min_red_frames}")
                                    if path_consistency is not None and path_consistency < 0.4:
                                        debug_reason.append(f"path={path_consistency:.2f}")
                                    if grid_conf is not None and grid_conf < GRID_NOISE_THRESHOLD_FOR_HYBRID:
                                        debug_reason.append(f"grid={grid_conf:.2f}")
                                    if debug_reason:
                                        print(f"🚫 Target {target_id}: Blocked RED by extra conditions ({', '.join(debug_reason)})")
                            else:
                                # เงื่อนไข RED ผ่าน → ตั้งเป็น RED/DRONE ปกติ
                                color = (0, 0, 255)  # Red
                                status_label = "DRONE"
                                target['status'] = 'RED'

                            # --- เงื่อนไขเข้า RED LOCK mode ---
                            # ตรวจสอบ airplane/star อีกครั้ง (อาจอัปเดตใหม่)
                            is_airplane_check = target.get('is_airplane', False)
                            is_star_check = target.get('is_star', False)

                            if (target.get('status') in ('RED', 'DRONE')
                                and target.get('red_score', 0.0) >= RED_LOCK_SCORE
                                and not is_suspect
                                and not is_insect
                                and not is_blacklisted
                                and not is_airplane_check
                                and not is_star_check):
                                target['lock_mode'] = True
                                if DEBUG_MODE:
                                    print(f"🔒 Target {target_id}: ENTER RED LOCK (score={target['red_score']:.1f})")
                        else:
                            # ========== หลัง lock: บังคับคง RED หนึบ ==========
                            # ตรวจสอบ airplane/star อีกครั้ง (อาจอัปเดตใหม่)
                            is_airplane_check = target.get('is_airplane', False)
                            is_star_check = target.get('is_star', False)

                            # สำหรับ lock mode: ใช้ threshold ต่ำกว่า (ถ้าไม่แย่มาก)
                            from config import ROI_LOCK_RED_ENABLED, ROI_LOCK_RED_MIN_COMPOSITE

                            composite_score = target.get('composite_score', original_conf)
                            is_lock_mode = target.get('lock_mode', False)

                            # ถ้าไม่ใช่ insect / blacklist / stationary_suspect / airplane / star → บังคับคง RED
                            if not (is_insect or is_blacklisted or is_suspect or is_airplane_check or is_star_check):
                                # สำหรับ lock mode: ใช้ threshold ต่ำกว่า
                                if is_lock_mode and ROI_LOCK_RED_ENABLED:
                                    if composite_score >= ROI_LOCK_RED_MIN_COMPOSITE:
                                        color = (0, 0, 255)  # Red
                                        status_label = "DRONE"
                                        target['status'] = 'RED'
                                        # คง red_score ไว้เหนือ threshold เพื่อไม่ให้หลุด lock ง่าย
                                        target['red_score'] = max(target.get('red_score', 0.0), RED_LOCK_SCORE)
                                        all_high_confidence = True  # ยังคง high confidence
                                        if DEBUG_MODE:
                                            print(f"🔒 Target {target_id}: Lock mode RED (composite={composite_score:.2f} >= {ROI_LOCK_RED_MIN_COMPOSITE:.2f})")
                                    else:
                                        # composite score ต่ำเกินไป → เปลี่ยนเป็น ORANGE
                                        color = (0, 165, 255)  # Orange
                                        status_label = "TRACKING"
                                        target['status'] = 'ORANGE'
                                        if DEBUG_MODE:
                                            print(f"🟠 Target {target_id}: Lock mode ORANGE (composite={composite_score:.2f} < {ROI_LOCK_RED_MIN_COMPOSITE:.2f})")
                                else:
                                    # ไม่ใช่ lock mode หรือปิดใช้งาน → ใช้ logic ปกติ
                                    color = (0, 0, 255)  # Red
                                    status_label = "DRONE"
                                    target['status'] = 'RED'
                                    target['red_score'] = max(target.get('red_score', 0.0), RED_LOCK_SCORE)
                                    all_high_confidence = True
                            else:
                                # เจอเหตุผลหนัก (แมลง/blacklist/stationary/airplane/star) → ปลด lock
                                target['lock_mode'] = False
                                target['red_score'] = 0.0
                                color = (0, 255, 0)  # Green
                                status_label = "TRACKING"
                                target['status'] = 'GREEN'
                                all_high_confidence = False
                                if DEBUG_MODE:
                                    reason = []
                                    if is_insect: reason.append("insect")
                                    if is_blacklisted: reason.append("blacklist")
                                    if is_suspect: reason.append("stationary_suspect")
                                    if is_airplane_check: reason.append(f"airplane(conf={target.get('airplane_confidence', 0.0):.2f})")
                                    if is_star_check: reason.append("star")
                                    print(f"🔓 Target {target_id}: EXIT RED LOCK (reason={'/'.join(reason)})")
                else:
                    # Boost lock mode: ถ้า lock แล้วและไม่มี yolo_box → ใช้ last_rect เดิมต่อ
                    if target.get('lock_mode', False):
                        max_keep_no_update = 5  # ทนได้อีก 5 เฟรมแบบ no-update
                        if target['missed_frames'] < max_keep_no_update and target.get('last_rect'):
                            # ใช้ last_rect เดิมต่อ (ไม่เพิ่ม missed_frames ในช่วง boost)
                            final_rect = target['last_rect']
                            if DEBUG_MODE and target['missed_frames'] == 0:
                                print(f"🔒 Target {target_id}: BOOST LOCK (no YOLO) - keeping last_rect")
                        else:
                            target['missed_frames'] += 1
                    else:
                        target['missed_frames'] += 1

                    # ตรวจสอบ path-based RED สำหรับ tiny objects (เมื่อไม่มี YOLO detection)
                    if TINY_OBJECT_PATH_BASED_RED_ENABLED:
                        is_tiny_for_red = self.is_tiny_object_for_target(target, w_orig, h_orig)
                        if is_tiny_for_red and 'centers_history' in target:
                            # ใช้ interval-based checking (ทุก 6 เฟรม) เพื่อประหยัด CPU
                            current_frame = getattr(self, '_current_frame_counter', 0)
                            check_interval = 6
                            last_path_red_check_frame = target.get('last_path_red_check_frame', -check_interval)
                            should_check_path_red = (current_frame - last_path_red_check_frame) >= check_interval

                            if should_check_path_red:
                                # ตรวจสอบเงื่อนไข path-based RED
                                processing_fps = getattr(self, '_current_processing_fps', None)
                                if processing_fps is None or processing_fps <= 0:
                                    from config import DEFAULT_FPS
                                    processing_fps = DEFAULT_FPS

                                # ตรวจสอบเงื่อนไขทั้งหมด
                                min_history_frames = int(TINY_OBJECT_PATH_BASED_RED_MIN_HISTORY_SECONDS * processing_fps)
                                min_history_frames = max(20, min_history_frames)  # minimum fallback
                                min_duration_frames = int(TINY_OBJECT_PATH_BASED_RED_MIN_TRACKING_DURATION_SECONDS * processing_fps)
                                min_duration_frames = max(10, min_duration_frames)  # minimum fallback

                                has_enough_history = len(target['centers_history']) >= min_history_frames
                                tracking_duration = current_frame - target.get('tracking_start_frame', current_frame) + 1
                                has_enough_duration = tracking_duration >= min_duration_frames

                                if has_enough_history and has_enough_duration:
                                    # คำนวณ path-based confidence
                                    path_confidence = self.calculate_tiny_object_path_confidence(target, current_frame)

                                    # ตรวจสอบเงื่อนไขอื่นๆ
                                    is_insect = target.get('is_insect', False)
                                    is_blacklisted = target.get('is_blacklisted', False)
                                    is_suspect = target.get('stationary_suspect', False)
                                    is_airplane = target.get('is_airplane', False)
                                    is_star = target.get('is_star', False)

                                    # ตรวจสอบ path consistency
                                    path_consistency = None
                                    if len(target['centers_history']) >= 2:
                                        first_frame, first_center = target['centers_history'][0]
                                        last_frame, last_center = target['centers_history'][-1]
                                        if first_center and last_center:
                                            direct_dist = self.get_dist(first_center, last_center)
                                            path_dist = 0.0
                                            for i in range(1, len(target['centers_history'])):
                                                _, prev_center = target['centers_history'][i-1]
                                                _, curr_center = target['centers_history'][i]
                                                if prev_center and curr_center:
                                                    path_dist += self.get_dist(prev_center, curr_center)
                                            if path_dist > 0:
                                                path_consistency = direct_dist / path_dist

                                    # เงื่อนไขสำหรับ path-based RED
                                    if (path_confidence >= TINY_OBJECT_PATH_BASED_RED_MIN_CONFIDENCE and
                                        not is_insect and not is_blacklisted and not is_suspect and
                                        not is_airplane and not is_star and
                                        (path_consistency is None or path_consistency >= 0.4)):
                                        # ตั้งเป็น RED
                                        color = (0, 0, 255)  # Red
                                        status_label = "DRONE"
                                        target['status'] = 'RED'
                                        target['path_based_confidence'] = path_confidence
                                        all_high_confidence = True

                                        if DEBUG_MODE:
                                            print(f"🎯 Target {target_id}: Path-based RED (conf={path_confidence:.2f}, fps={processing_fps:.1f})")
                                    else:
                                        color = (0, 255, 0)  # Green
                                        status_label = "TRACKING"
                                        target['status'] = 'GREEN'
                                        all_high_confidence = False
                                else:
                                    color = (0, 255, 0)  # Green
                                    status_label = "TRACKING"
                                    target['status'] = 'GREEN'
                                    all_high_confidence = False

                                target['last_path_red_check_frame'] = current_frame
                            else:
                                # ใช้ cached status
                                if target.get('status') == 'RED' and target.get('path_based_confidence', 0.0) >= TINY_OBJECT_PATH_BASED_RED_MIN_CONFIDENCE:
                                    color = (0, 0, 255)  # Red
                                    status_label = "DRONE"
                                    all_high_confidence = True
                                else:
                                    color = (0, 255, 0)  # Green
                                    status_label = "TRACKING"
                                    target['status'] = 'GREEN'
                                    all_high_confidence = False

                    # ตรวจสอบ path-based ORANGE สำหรับ non-tiny objects
                    # กรณีที่ motion path ชัดเจนมาก แต่ YOLO confidence ต่ำ
                    if DRONE_PATH_BASED_ORANGE_ENABLED:
                        current_status = target.get('status', 'GREEN')
                        original_conf = target.get('original_confidence', target.get('confidence', 0.0))

                        # ตรวจสอบเฉพาะเมื่อ status เป็น GREEN และ confidence ต่ำ
                        if current_status == 'GREEN' and original_conf < HYBRID_BASE_CONF:
                            # ใช้ interval-based checking (ทุก N เฟรม)
                            current_frame = getattr(self, '_current_frame_counter', 0)
                            check_interval = DRONE_PATH_BASED_ORANGE_CHECK_INTERVAL
                            last_check_frame = target.get('last_path_orange_check_frame', -check_interval)
                            should_check = (current_frame - last_check_frame) >= check_interval

                            if should_check:
                                # ตรวจสอบเงื่อนไขทั้งหมด
                                processing_fps = getattr(self, '_current_processing_fps', None)
                                if processing_fps is None or processing_fps <= 0:
                                    from config import DEFAULT_FPS
                                    processing_fps = DEFAULT_FPS

                                min_history_frames = int(DRONE_PATH_BASED_ORANGE_MIN_HISTORY_SECONDS * processing_fps)
                                min_history_frames = max(DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS, min_history_frames)
                                min_duration_frames = int(DRONE_PATH_BASED_ORANGE_MIN_TRACKING_DURATION_SECONDS * processing_fps)
                                min_duration_frames = max(5, min_duration_frames)

                                has_enough_history = 'centers_history' in target and len(target['centers_history']) >= min_history_frames
                                tracking_duration = current_frame - target.get('tracking_start_frame', current_frame) + 1
                                has_enough_duration = tracking_duration >= min_duration_frames

                                if has_enough_history and has_enough_duration:
                                    # คำนวณ path confidence
                                    path_confidence = self.calculate_tiny_object_path_confidence(target, current_frame)

                                    # ตรวจสอบ motion path quality
                                    is_path_clear, path_quality = self.calculate_motion_path_quality(target, current_frame)

                                    # ตรวจสอบเงื่อนไขอื่นๆ
                                    is_insect = target.get('is_insect', False)
                                    is_blacklisted = target.get('is_blacklisted', False)
                                    is_suspect = target.get('stationary_suspect', False)
                                    is_airplane = target.get('is_airplane', False)
                                    is_star = target.get('is_star', False)

                                    # ตรวจสอบ background noise ใน ROI
                                    size_category = target.get('object_size_category')
                                    is_low_noise, noise_level, motion_boxes_in_roi = self.check_background_noise_in_roi(target, current_frame, size_category)

                                    # เงื่อนไขสำหรับ path-based ORANGE
                                    if (path_confidence >= DRONE_PATH_BASED_ORANGE_MIN_CONFIDENCE and
                                        is_path_clear and
                                        not is_insect and not is_blacklisted and not is_suspect and
                                        not is_airplane and not is_star):
                                        # ตั้งเป็น ORANGE
                                        color = (0, 165, 255)  # Orange
                                        status_label = "TRACKING"
                                        target['status'] = 'ORANGE'
                                        target['path_based_confidence'] = path_confidence
                                        target['path_based_quality'] = path_quality
                                        all_high_confidence = False

                                        if DEBUG_MODE:
                                            print(f"🎯 Target {target_id}: Path-based ORANGE "
                                                  f"(conf={path_confidence:.2f}, quality={path_quality:.2f}, "
                                                  f"noise={noise_level:.2f}, motion_boxes={motion_boxes_in_roi})")

                                target['last_path_orange_check_frame'] = current_frame
                        else:
                            color = (0, 255, 0)  # Green
                            status_label = "TRACKING"
                            target['status'] = 'GREEN'
                            all_high_confidence = False
                    else:
                        color = (0, 255, 0)  # Green
                        status_label = "TRACKING"
                        target['status'] = 'GREEN'
                        all_high_confidence = False
            elif target['target_locked']:
                # สำหรับ lock target: ใช้ min_motion_area จาก parameter (หรือ HYBRID_MIN_MOTION_AREA)
                # ใช้ min_motion_area parameter ที่รับมาจาก process_frame()
                lock_min_motion_area = min_motion_area
                if target.get('lock_mode', False):
                    # ลด threshold สำหรับ lock target (ประมาณ 50% ของค่าเดิม)
                    lock_min_motion_area = max(5, int(min_motion_area * 0.5))

                # คำนวณ velocity และ predicted position จาก path history
                velocity_vector = None
                predicted_center = None
                centers_history = target.get('centers_history', [])

                # ใช้ cached velocity ถ้ามี
                cached_velocity = target.get('cached_velocity')
                cached_velocity_frame = target.get('cached_velocity_frame', -1)
                current_frame = getattr(self, '_current_frame_counter', 0)

                if (cached_velocity is not None and
                    cached_velocity_frame == current_frame and
                    len(centers_history) > 0):
                    # ใช้ cached velocity
                    velocity_vector = cached_velocity
                elif len(centers_history) >= 2:
                    # คำนวณ velocity ใหม่ (ใช้ smoothing ถ้ามี cached velocity)
                    use_smoothing = cached_velocity is not None
                    velocity_vector = self.get_velocity_from_path_history(
                        centers_history,
                        use_smoothing=use_smoothing,
                        cached_velocity=cached_velocity,
                        processing_fps=self._current_processing_fps
                    )
                    # Cache velocity
                    target['cached_velocity'] = velocity_vector
                    target['cached_velocity_frame'] = current_frame

                # คำนวณ predicted center จาก velocity
                if velocity_vector is not None:
                    vx, vy = velocity_vector
                    if abs(vx) > 0.01 or abs(vy) > 0.01:  # มีการเคลื่อนที่
                        last_center = target.get('last_center')
                        if last_center is not None:
                            predicted_center = (last_center[0] + vx, last_center[1] + vy)
                            # Clamp ให้อยู่ใน frame bounds
                            predicted_center = (
                                max(0, min(w_orig - 1, predicted_center[0])),
                                max(0, min(h_orig - 1, predicted_center[1]))
                            )

                # ดึง last_rect สำหรับ size filtering
                last_rect = target.get('last_rect')

                # ดึง object_size_category และ thresholds
                object_size_category = target.get('object_size_category')
                size_category = object_size_category  # สำหรับ backward compatibility

                # ใช้ resolution-dependent thresholds
                thresholds = self.thresholds

                # ใช้ ROI ปกติ (ไม่ขยาย) - คงที่ตาม resolution
                roi_box = target['roi_box']

                # Fallback strategy: ลองหา motion box ด้วย thresholds ปกติก่อน
                # กำหนด display_frame_for_debug สำหรับ debug visualization (ถ้าต้องการ)
                display_frame_for_debug = None  # ยังไม่มี access ถึง display_frame ใน update_tracking

                motion_rect = self.find_best_motion(
                    mask_cpu_processed,
                    roi_box,
                    target['last_center'],
                    lock_min_motion_area,
                    object_size_category=object_size_category,
                    adaptive_min_area=lock_min_motion_area,
                    velocity_vector=velocity_vector,
                    predicted_center=predicted_center,
                    last_rect=last_rect,
                    thresholds=thresholds,
                    use_fallback=False,
                    target_id=target_id,
                    debug_display_frame=display_frame_for_debug
                )

                # ถ้าไม่เจอ motion box: ใช้ fallback strategy (เฉพาะเมื่อไม่ใช่ lock mode หรือ lock mode แต่ยังไม่นาน)
                from config import FALLBACK_MAX_ATTEMPTS
                missed_frames = target.get('missed_frames', 0)

                # สำหรับ lock mode: ถ้าไม่มี motion ใน ROI ต่อเนื่อง → ไม่ใช้ fallback (จะออกจาก lock แทน)
                is_lock_mode = target.get('lock_mode', False)
                no_motion_frames = target.get('no_motion_in_roi_frames', 0)

                if motion_rect is None:
                    # นับจำนวนเฟรมที่ไม่มี motion ใน ROI
                    no_motion_frames += 1
                    target['no_motion_in_roi_frames'] = no_motion_frames

                    # สำหรับ lock mode: ถ้าไม่มี motion ต่อเนื่อง 2 เฟรม → ไม่ใช้ fallback
                    if is_lock_mode and no_motion_frames >= 2:
                        # ไม่ใช้ fallback - จะออกจาก lock แทน
                        if DEBUG_MODE:
                            print(f"🚫 Target {target_id}: No motion in ROI for {no_motion_frames} frames (lock mode) - skipping fallback")
                    else:
                        # ใช้ fallback strategy (สำหรับ non-lock หรือ lock ที่ยังไม่นาน)
                        if missed_frames >= 3:  # ใช้ threshold ต่ำกว่าเดิม (ไม่ต้องรอ ROI_EXTENDED_SEARCH_MIN_MISSED)
                            fallback_attempts = target.get('fallback_attempts', 0)
                            if fallback_attempts < FALLBACK_MAX_ATTEMPTS:
                                target['fallback_attempts'] = fallback_attempts + 1
                                motion_rect = self.find_best_motion(
                                    mask_cpu_processed,
                                    roi_box,
                                    target['last_center'],
                                    lock_min_motion_area,
                                    object_size_category=object_size_category,
                                    adaptive_min_area=lock_min_motion_area,
                                    velocity_vector=velocity_vector,
                                    predicted_center=predicted_center,
                                    last_rect=last_rect,
                                    thresholds=thresholds,
                                    use_fallback=True,
                                    target_id=target_id,
                                    debug_display_frame=display_frame_for_debug
                                )
                                if DEBUG_MODE and motion_rect:
                                    print(f"🔄 Target {target_id}: Fallback strategy succeeded (attempt {fallback_attempts + 1})")
                            else:
                                # Reset fallback attempts หลังจากพยายามครบแล้ว
                                target['fallback_attempts'] = 0
                else:
                    # Reset counters เมื่อเจอ motion box
                    target['fallback_attempts'] = 0
                    target['no_motion_in_roi_frames'] = 0

                if motion_rect:
                    final_rect = motion_rect

                    # Motion box ที่ถูกเลือกแล้วใน ROI → เก็บ path point ทุกครั้ง (ไม่ต้อง validate)
                    tx, ty, tw, th = motion_rect
                    center = (tx + tw // 2, ty + th // 2)
                    current_frame = getattr(self, '_current_frame_counter', 0)
                    self.update_path_history(target, center, current_frame, status=target.get('status'))

                    # อัปเดต target state
                    target['last_rect'] = motion_rect
                    target['last_center'] = center
                    target['missed_frames'] = 0
                    target['last_update_frame'] = current_frame

                    # อัปเดต roi_box_history
                    if 'roi_box' in target and target['roi_box']:
                        roi_box = target['roi_box']
                        roi_center = ((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)
                        if 'roi_box_history' not in target:
                            target['roi_box_history'] = []
                        target['roi_box_history'].append((current_frame, tuple(roi_box), roi_center))
                        from config import ROI_HISTORY_MAX_FRAMES
                        if len(target['roi_box_history']) > ROI_HISTORY_MAX_FRAMES:
                            target['roi_box_history'] = target['roi_box_history'][-ROI_HISTORY_MAX_FRAMES:]

                    color = (0, 165, 255)  # Orange
                    status_label = "TRACKING"
                    # Update status to ORANGE if tracking with motion (แต่ถ้า lock_mode=True จะถูกบังคับเป็น RED ในส่วนบน)
                    if target.get('status') != 'RED':
                        target['status'] = 'ORANGE'
                    all_high_confidence = False
                else:
                    # ถ้าไม่มี motion box → ใช้ predicted position
                    current_frame = getattr(self, '_current_frame_counter', 0)
                    from config import PREDICTED_POSITION_MIN_HISTORY, PREDICTED_POSITION_MAX_CONSECUTIVE

                    # ตรวจสอบว่าใช้ predicted position ต่อเนื่องมากเกินไปหรือไม่
                    consecutive_predicted = target.get('consecutive_predicted_frames', 0)
                    if consecutive_predicted >= PREDICTED_POSITION_MAX_CONSECUTIVE:
                        # ใช้ predicted position มากเกินไป → ไม่สร้างต่อ (ป้องกัน false positive)
                        target['missed_frames'] += 1
                    elif 'centers_history' in target and len(target['centers_history']) >= PREDICTED_POSITION_MIN_HISTORY:
                        # มี path history เพียงพอ → ใช้ predicted position
                        predicted_center = self.predict_next_position(target['centers_history'])
                        if predicted_center is not None:
                            # เก็บ predicted position เป็น path point
                            self.update_path_history(target, predicted_center, current_frame, status=target.get('status'))
                            # อัปเดต last_center แต่ไม่อัปเดต last_rect
                            target['last_center'] = predicted_center
                            target['missed_frames'] += 1
                            target['consecutive_predicted_frames'] = consecutive_predicted + 1
                        else:
                            target['missed_frames'] += 1
                            target['consecutive_predicted_frames'] = 0  # Reset เมื่อ predict ไม่ได้
                    elif 'centers_history' in target and len(target['centers_history']) >= 1:
                        # มี path history 1 จุด → ใช้ last_center เป็น predicted position (ไม่ขยับ)
                        last_center = target.get('last_center')
                        if last_center:
                            self.update_path_history(target, last_center, current_frame, status=target.get('status'))
                            target['missed_frames'] += 1
                            target['consecutive_predicted_frames'] = consecutive_predicted + 1
                        else:
                            target['missed_frames'] += 1
                            target['consecutive_predicted_frames'] = 0
                    else:
                        target['missed_frames'] += 1
                        target['consecutive_predicted_frames'] = 0
                    # Boost lock mode: ถ้า lock แล้ว ให้ทน no-update ได้มากขึ้น
                    if target.get('lock_mode', False):
                        # ใช้ last_rect เดิมต่อสักระยะ (ไม่เพิ่ม missed_frames เร็วเกินไป)
                        max_keep_no_update = 5  # ทนได้อีก 5 เฟรมแบบ no-update
                        if target['missed_frames'] < max_keep_no_update:
                            # ใช้ last_rect เดิมต่อ (ไม่เพิ่ม missed_frames ในช่วงนี้)
                            if target.get('last_rect'):
                                final_rect = target['last_rect']
                                # ไม่เพิ่ม missed_frames ในช่วง boost
                                if DEBUG_MODE and target['missed_frames'] == 0:
                                    print(f"🔒 Target {target_id}: BOOST LOCK - keeping last_rect (missed={target['missed_frames']})")
                            else:
                                target['missed_frames'] += 1
                        else:
                            # หลังจาก boost period แล้ว → เริ่มนับ missed_frames ปกติ
                            target['missed_frames'] += 1
                    else:
                        # Target ปกติ → นับ missed_frames ทันที
                        target['missed_frames'] += 1

                # Check if target lost (ถ้าเป็น lock_mode ให้รอได้นานขึ้น)
                max_miss_allowed = HYBRID_MAX_MISS_ALLOWED
                motion_stationary = False
                stationary_frames = 0

                if target.get('lock_mode', False):
                    from config import LOCK_ROI_WAIT_MULTIPLIER
                    max_miss_allowed = int(HYBRID_MAX_MISS_ALLOWED * LOCK_ROI_WAIT_MULTIPLIER)

                    # ตรวจสอบว่า motion boxes ใน ROI ขยับหรือไม่
                    current_frame = getattr(self, '_current_frame_counter', 0)
                    motion_stationary, stationary_frames = self._check_motion_boxes_stationary_in_roi(
                        target, target_id, roi_box, w_orig, h_orig, current_frame
                    )

                    # ถ้า motion boxes ไม่ขยับเลย → ลด wait time (ใช้ wait time ปกติแทน)
                    if motion_stationary:
                        max_miss_allowed = HYBRID_MAX_MISS_ALLOWED  # ใช้ wait time ปกติ
                        if DEBUG_MODE:
                            print(f"⏱️ Target {target_id}: Reducing wait time (motion stationary, frames={stationary_frames}, max_miss={max_miss_allowed})")

                # เงื่อนไขออกจาก lock: missed_frames สูงเกินไป หรือ red_score ต่ำมาก หรือ motion boxes ไม่ขยับ หรือไม่มี motion ใน ROI
                should_exit_lock = False
                exit_reason = []
                if target.get('lock_mode', False):
                    if target['missed_frames'] > max_miss_allowed:
                        should_exit_lock = True
                        exit_reason.append(f"missed_frames={target['missed_frames']}")
                    # ตรวจสอบ red_score ต่ำมาก (ต่ำกว่า 30% ของ threshold)
                    if target.get('red_score', 0.0) < RED_LOCK_SCORE * 0.3:
                        should_exit_lock = True
                        exit_reason.append(f"red_score={target.get('red_score', 0.0):.1f}")
                    # ตรวจสอบ motion boxes ไม่ขยับเลย (ถ้าไม่ขยับต่อเนื่อง N เฟรม → ออกจาก lock)
                    if motion_stationary and stationary_frames >= MOTION_STATIONARY_FRAMES_TO_EXIT:
                        should_exit_lock = True
                        exit_reason.append(f"motion_stationary={stationary_frames}frames")
                    # ตรวจสอบไม่มี motion ใน ROI (ถ้าไม่มี motion ต่อเนื่อง 3 เฟรม → ออกจาก lock)
                    no_motion_frames = target.get('no_motion_in_roi_frames', 0)
                    if no_motion_frames >= 3:
                        should_exit_lock = True
                        exit_reason.append(f"no_motion_in_roi={no_motion_frames}frames")

                    if should_exit_lock:
                        target['lock_mode'] = False
                        target['red_score'] = 0.0
                        target['motion_stationary_frames'] = 0
                        target['no_motion_in_roi_frames'] = 0
                        if DEBUG_MODE:
                            print(f"🔓 Target {target_id}: EXIT RED LOCK (reason={'/'.join(exit_reason)})")

                if target['missed_frames'] > max_miss_allowed:
                    targets_to_remove.append(target_id)
                    if DEBUG_MODE:
                        print(f"⚠️ Target {target_id} lost (missed_frames={target['missed_frames']} > {max_miss_allowed})")

                # วิเคราะห์ path เพื่อแยกประเภท target (ทุกเฟรม)
                from config import PATH_CLASSIFICATION_ENABLED, PATH_CLASSIFICATION_MIN_POINTS, PATH_CLASSIFICATION_UPDATE_INTERVAL

                if PATH_CLASSIFICATION_ENABLED:
                    centers_history = target.get('centers_history', [])
                    # ตรวจสอบว่า path ยาวพอ (อย่างน้อย 15-30 จุด)
                    if len(centers_history) >= PATH_CLASSIFICATION_MIN_POINTS:
                        # ตรวจสอบว่าเวลาอัปเดตหรือยัง
                        last_classification_frame = target.get('last_classification_frame', -PATH_CLASSIFICATION_UPDATE_INTERVAL)
                        current_frame = getattr(self, '_current_frame_counter', 0)
                        if current_frame - last_classification_frame >= PATH_CLASSIFICATION_UPDATE_INTERVAL:
                            # ใช้ processing_fps สำหรับ FPS-aware thresholds
                            processing_fps = getattr(self, '_processing_fps', None)
                            classification_result = self.classify_target_from_path(target, current_frame, processing_fps)
                            target['path_classification'] = classification_result['classification']
                            target['path_classification_confidence'] = classification_result['confidence']
                            target['path_characteristics'] = classification_result['characteristics']
                            target['last_classification_frame'] = current_frame

                            if DEBUG_MODE:
                                reason = classification_result.get('reason', '')
                                reason_str = f" ({reason})" if reason else ""
                                print(f"🎯 Target {target_id}: Classified as {classification_result['classification']} "
                                      f"(confidence={classification_result['confidence']:.2f}, path_points={len(centers_history)}){reason_str}")

            # Update target position
            if final_rect:
                tx, ty, tw, th = final_rect
                target['last_rect'] = final_rect  # Store rect for drawing
                target['last_center'] = (tx + tw//2, ty + th//2)

                # ใช้ resolution-dependent thresholds (กำหนดไว้ก่อนใช้งานเสมอ)
                thresholds = self.thresholds

                # Dynamic ROI padding ตามความเร็ว
                from config import ROI_PADDING_FAST_MULTIPLIER, ROI_PADDING_FAST_VELOCITY_THRESHOLD_RATIO

                # ดึง velocity_vector จาก target (ถ้ามี) หรือคำนวณใหม่
                velocity_vector = None
                if 'cached_velocity' in target:
                    cached_val = target.get('cached_velocity')
                    # ตรวจสอบว่า cached_velocity เป็น tuple (vx, vy) หรือ float (velocity magnitude)
                    if isinstance(cached_val, (tuple, list)) and len(cached_val) == 2:
                        velocity_vector = cached_val  # เป็น tuple (vx, vy)
                    # ถ้าเป็น float ให้คำนวณใหม่จาก centers_history

                # ถ้ายังไม่มี velocity_vector ให้คำนวณใหม่
                if velocity_vector is None and 'centers_history' in target and len(target['centers_history']) >= 2:
                    velocity_vector = self.get_velocity_from_path_history(
                        target['centers_history'],
                        processing_fps=self._current_processing_fps
                    )

                # คำนวณ velocity magnitude
                velocity_mag = 0.0
                if velocity_vector is not None:
                    # ตรวจสอบว่า velocity_vector เป็น tuple หรือไม่
                    if isinstance(velocity_vector, (tuple, list)) and len(velocity_vector) == 2:
                        vx, vy = velocity_vector
                        if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                            velocity_mag = math.sqrt(vx**2 + vy**2)
                    elif isinstance(velocity_vector, (int, float)):
                        # ถ้าเป็น float (velocity magnitude) ให้ใช้ค่าโดยตรง
                        velocity_mag = float(velocity_vector)

                # ใช้ ROI padding คงที่ (ไม่ขยายตามความเร็ว) - คงที่ตาม resolution
                base_roi_padding = thresholds.get('HYBRID_ROI_PADDING', HYBRID_ROI_PADDING)
                roi_padding = base_roi_padding  # ใช้ padding คงที่

                target['roi_box'] = [
                    max(0, tx - roi_padding), max(0, ty - roi_padding),
                    min(w_orig, tx + tw + roi_padding), min(h_orig, ty + th + roi_padding)
                ]

                # Cache object_size_category (คำนวณครั้งเดียวต่อเฟรมเมื่อ update target)
                target['object_size_category'] = self.get_object_size_category_for_target(target, w_orig, h_orig)

                # ใช้ cached size category แทน is_tiny
                size_category = target.get('object_size_category')
                is_tiny = (size_category == 'TINY')

                # อัปเดต centers_history (ใช้ฟังก์ชัน optimize แทน)
                # หมายเหตุ: การเพิ่มจุดเข้าไปใน centers_history จะทำในส่วนที่ตรวจสอบ motion_rect แล้ว
                # (ใช้ validation function เพื่อป้องกัน noise จากพื้นหลัง)
                # แต่ถ้าไม่มี motion_rect (ใช้ final_rect จากที่อื่น) ให้อัปเดต path history ด้วย
                if final_rect and target.get('last_center'):
                    current_frame = getattr(self, '_current_frame_counter', 0)
                    # ตรวจสอบว่ายังไม่ได้เพิ่มจุดเข้าไปใน path history แล้วหรือไม่
                    # (ถ้าเพิ่มแล้วในส่วน motion_rect validation จะไม่เพิ่มซ้ำ)
                    if 'centers_history' not in target or len(target['centers_history']) == 0 or target['centers_history'][-1][0] != current_frame:
                        self.update_path_history(target, target['last_center'], current_frame)

                # เก็บ size history สำหรับ tiny objects (interval-based storage)
                if is_tiny:
                    # ใช้ interval-based storage (เก็บทุก 6 เฟรม) เพื่อไม่กระทบความเร็ว
                    if current_frame % 6 == 0:  # เก็บทุก 6 เฟรม
                        if 'size_history' not in target:
                            target['size_history'] = []
                        area = tw * th
                        target['size_history'].append((current_frame, area))
                        # จำกัดความยาว size history ให้เท่ากับ path history (เพื่อประหยัด memory)
                        # ใช้ max_history จาก size category
                        max_history = self._get_path_history_limit(size_category, target.get('status', 'GREEN'), target)
                        if max_history is None and size_category == 'TINY':
                            # TINY: ใช้ seconds-based
                            processing_fps = getattr(self, '_current_processing_fps', None)
                            if processing_fps is None or processing_fps <= 0:
                                from config import DEFAULT_FPS
                                processing_fps = DEFAULT_FPS
                            from config import TINY_OBJECT_PATH_HISTORY_SECONDS
                            max_history = int(TINY_OBJECT_PATH_HISTORY_SECONDS * processing_fps)
                            max_history = max(20, max_history)
                        if max_history and len(target['size_history']) > max_history:
                            target['size_history'] = target['size_history'][-max_history:]

                # ตรวจสอบ airplane/star สำหรับ tiny objects (interval-based checking)
                if is_tiny and 'centers_history' in target and len(target['centers_history']) >= 10:
                    # ใช้ interval-based checking (ทุก 6 เฟรม) เพื่อประหยัด CPU (เหมือน insect check)
                    check_interval = 6
                    last_airplane_check_frame = target.get('last_airplane_check_frame', -check_interval)
                    last_star_check_frame = target.get('last_star_check_frame', -check_interval)
                    should_check_airplane = (current_frame - last_airplane_check_frame) >= check_interval
                    should_check_star = (current_frame - last_star_check_frame) >= check_interval

                    if should_check_airplane:
                        is_airplane, airplane_confidence = self.is_likely_airplane_for_target(target, current_frame)
                        target['is_airplane'] = is_airplane
                        target['airplane_confidence'] = airplane_confidence
                        target['last_airplane_check_frame'] = current_frame
                    else:
                        # ใช้ cached value
                        is_airplane = target.get('is_airplane', False)
                        airplane_confidence = target.get('airplane_confidence', 0.0)

                    if should_check_star:
                        is_star = self.is_likely_star_by_path(target, current_frame)
                        target['is_star'] = is_star
                        target['last_star_check_frame'] = current_frame
                    else:
                        # ใช้ cached value
                        is_star = target.get('is_star', False)

                # ตรวจสอบการเคลื่อนที่ของ RED bbox (ปล่อยถ้านิ่งนานเกินไป)
                current_status = target.get('status', 'GREEN')
                current_frame = getattr(self, '_current_frame_counter', 0)

                if current_status == 'RED':
                    from config import (
                        RED_STATIONARY_CHECK_INTERVAL, RED_STATIONARY_MAX_FRAMES,
                        RED_STATIONARY_VELOCITY_THRESHOLD_RATIO, RED_STATIONARY_DISTANCE_THRESHOLD_RATIO
                    )

                    # ตรวจสอบทุก N เฟรม
                    last_red_check_frame = target.get('last_red_stationary_check_frame', -RED_STATIONARY_CHECK_INTERVAL)
                    should_check_red = (current_frame - last_red_check_frame) >= RED_STATIONARY_CHECK_INTERVAL

                    if should_check_red:
                        target['last_red_stationary_check_frame'] = current_frame

                        # คำนวณ velocity และ total distance moved
                        centers_history = target.get('centers_history', [])
                        if len(centers_history) >= 2:
                            # คำนวณ velocity magnitude
                            velocity_mag = 0.0
                            if velocity_vector is not None:
                                vx, vy = velocity_vector
                                if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                                    velocity_mag = math.sqrt(vx**2 + vy**2)

                            # คำนวณ total distance moved
                            first_frame, first_center = centers_history[0]
                            last_frame, last_center = centers_history[-1]
                            total_distance = 0.0
                            if first_center is not None and last_center is not None:
                                total_distance = self.get_dist(first_center, last_center)

                            # ใช้ resolution-dependent thresholds
                            frame_diagonal = math.sqrt(w_orig**2 + h_orig**2)
                            velocity_threshold = RED_STATIONARY_VELOCITY_THRESHOLD_RATIO * frame_diagonal
                            distance_threshold = RED_STATIONARY_DISTANCE_THRESHOLD_RATIO * frame_diagonal

                            # ตรวจสอบจำนวนเฟรมที่นิ่ง
                            red_stationary_frames = target.get('red_stationary_frames', 0)
                            if velocity_mag < velocity_threshold and total_distance < distance_threshold:
                                red_stationary_frames += RED_STATIONARY_CHECK_INTERVAL
                            else:
                                red_stationary_frames = 0  # Reset ถ้ามีการเคลื่อนที่

                            target['red_stationary_frames'] = red_stationary_frames

                            # ถ้านิ่งนานเกิน threshold: ปล่อย target
                            if red_stationary_frames >= RED_STATIONARY_MAX_FRAMES:
                                targets_to_remove.append(target_id)
                                if DEBUG_MODE:
                                    print(f"🚫 Target {target_id} (RED) removed due to stationary for {red_stationary_frames} frames "
                                          f"(velocity={velocity_mag:.2f} < {velocity_threshold:.2f}, "
                                          f"distance={total_distance:.2f} < {distance_threshold:.2f})")

                # ตรวจสอบว่าวัตถุนิ่งมากหรือไม่ (รวมกรณียืด/หดอยู่ที่เดิม)
                if target_id not in targets_to_remove:  # ตรวจสอบเฉพาะถ้ายังไม่ถูก remove
                    if self.check_stationary(target_id, target, getattr(self, '_current_frame_counter', 0)):
                        # วัตถุนิ่งมาก - ลบ target และ blacklist พื้นที่ในเฟรมรอบๆ bbox
                        targets_to_remove.append(target_id)

                    # สร้าง bbox สำหรับ blacklist (พื้นที่ในเฟรม) โดยเพิ่ม padding
                    bbox_for_blacklist = (tx, ty, tx + tw, ty + th)

                    # เพิ่มพื้นที่รอบๆ bbox เข้า blacklist (จะเพิ่ม padding อัตโนมัติใน add_to_blacklist)
                    # ถ้าพื้นที่นี้ถูก blacklist หลายครั้ง → จะย้ายไป permanent blacklist
                    self.add_to_blacklist(bbox_for_blacklist)

                    if DEBUG_MODE:
                        print(f"🚫 Target {target_id} removed due to stationary detection (including stretch/shrink case) - blacklisted area around bbox ({tx}, {ty}, {tx+tw}, {ty+th})")

                # ตรวจสอบว่าวัตถุที่ติดตามอยู่แล้วอยู่ใน blacklist และไม่มี movement หรือไม่
                # (กรณีที่วัตถุเคลื่อนที่เข้าไปใน blacklist แล้วหยุดนิ่ง)
                elif BLACKLIST_ALLOW_MOVEMENT:
                    # คำนวณ velocity จากประวัติ (ถ้ามี)
                    velocity = 0.0
                    if 'centers_history' in target and len(target['centers_history']) >= 2:
                        first_frame, first_center = target['centers_history'][0]
                        last_frame, last_center = target['centers_history'][-1]
                        if first_center and last_center:
                            distance = self.get_dist(first_center, last_center)
                            frames_diff = last_frame - first_frame
                            if frames_diff > 0:
                                velocity = distance / frames_diff

                    # ตรวจสอบ blacklist และ movement
                    if self.is_in_blacklist(target['last_center'], final_rect, None, velocity):
                        # อยู่ใน blacklist และไม่มี movement → ลบ target
                        targets_to_remove.append(target_id)
                        if DEBUG_MODE:
                            print(f"🚫 Target {target_id} removed: in blacklist area with no movement (velocity={velocity:.2f})")
            else:
                # ถ้าไม่เจอ detection ให้ใช้ last_rect เดิม (ถ้ามี)
                # หรืออัปเดต ROI box จาก last_center
                if not target.get('last_rect') and target.get('last_center'):
                    # Reconstruct ROI box จาก last_center (ถ้าไม่มี last_rect)
                    center_x, center_y = target['last_center']
                    # ใช้ขนาดจาก ROI box เดิม
                    roi_x1, roi_y1, roi_x2, roi_y2 = target['roi_box']
                    estimated_w = max(20, (roi_x2 - roi_x1) // 3)  # ใช้ 1/3 ของ ROI width
                    estimated_h = max(20, (roi_y2 - roi_y1) // 3)  # ใช้ 1/3 ของ ROI height
                    tx = max(0, center_x - estimated_w // 2)
                    ty = max(0, center_y - estimated_h // 2)
                    tw = min(w_orig - tx, estimated_w)
                    th = min(h_orig - ty, estimated_h)
                    target['last_rect'] = (tx, ty, tw, th)

        # Filter out targets that don't meet drone characteristics (periodic check)
        # Check every 30 frames to avoid performance impact
        from config import MIN_CONFIDENCE_FOR_TRACKING
        targets_to_filter = []
        if frame_counter % 30 == 0:
            for target_id, target in self.targets.items():
                if not is_likely_drone(target):
                    # Check if confidence is too low for too long
                    tracking_duration = frame_counter - target.get('tracking_start_frame', frame_counter)
                    if tracking_duration > 60:  # ติดตามมานานกว่า 60 เฟรม
                        original_conf = target.get('original_confidence', target.get('confidence', 0.0))
                        if original_conf < MIN_CONFIDENCE_FOR_TRACKING:
                            targets_to_filter.append(target_id)
                            if DEBUG_MODE:
                                print(f"🚫 Filtering non-drone target {target_id} (conf={original_conf:.2f}, duration={tracking_duration})")

        # Remove filtered targets
        for target_id in targets_to_filter:
            if target_id in self.targets:
                del self.targets[target_id]

        # Remove lost targets (before selecting primary)
        for target_id in targets_to_remove:
            if target_id in self.targets:
                del self.targets[target_id]
                if DEBUG_MODE:
                    print(f"⚠️ Hybrid Tracker: Target {target_id} Lost")

        # === SELECT PRIMARY TARGET ===
        # Select primary target after processing all targets
        primary_target_id, primary_target = self.select_primary_target(frame_counter)

        # Store primary target info for tracking_info
        self._primary_target_id = primary_target_id

        # === DRAW ALL TARGETS ===
        # DISABLED: Temporarily disabled to test footprints module performance
        # Draw all targets, highlighting primary
        # if DEBUG_MODE and len(self.targets) > 0:
        #     print(f"🎯 Drawing {len(self.targets)} targets: {list(self.targets.keys())}")

        # DISABLED: Temporarily disabled to test footprints module performance
        # for target_id, target in self.targets.items():
        #     if not target.get('target_locked', False):
        #         if DEBUG_MODE:
        #             print(f"  ⏭️ Skipping target {target_id}: not locked")
        #         continue
        #
        #     if DEBUG_MODE:
        #         print(f"  🎨 Drawing target {target_id}: locked={target.get('target_locked')}, has_rect={target.get('last_rect') is not None}, has_center={target.get('last_center') is not None}, missed={target.get('missed_frames', 0)}")
        #
        #     # Get target rect (use stored rect or reconstruct from center)
        #     final_rect = target.get('last_rect')
        #     if not final_rect and target.get('last_center'):
        #         # Reconstruct rect from last_center and roi_box
        #         roi_x1, roi_y1, roi_x2, roi_y2 = target['roi_box']
        #         center_x, center_y = target['last_center']
        #
        #         # ใช้ขนาดจาก ROI box (ประมาณว่า object อยู่ตรงกลาง ROI)
        #         # ROI box = object + padding ดังนั้น object size ≈ ROI size / 3
        #         estimated_w = max(20, (roi_x2 - roi_x1) // 3)  # ใช้ 1/3 ของ ROI width (ไม่ใช่ 1/2)
        #         estimated_h = max(20, (roi_y2 - roi_y1) // 3)  # ใช้ 1/3 ของ ROI height
        #
        #         tx = max(0, center_x - estimated_w // 2)
        #         ty = max(0, center_y - estimated_h // 2)
        #         tw = min(w_orig - tx, estimated_w)
        #         th = min(h_orig - ty, estimated_h)
        #         final_rect = (tx, ty, tw, th)
        #
        #         # เก็บ reconstructed rect ไว้ด้วย
        #         target['last_rect'] = final_rect
        #
        #     if final_rect:
        #         tx, ty, tw, th = final_rect
        #         is_primary = target.get('is_primary', False)
        #         status_label = target.get('status', 'TRACKING')
        #         original_status_label = status_label  # เก็บ status เดิมสำหรับ path drawing
        #         priority_score = target.get('priority_score', 0.0)
        #
        #         # ตรวจสอบแมลงและ blacklist ก่อนวาดกล่องสีแดง
        #         is_insect = target.get('is_insect', False)
        #         is_blacklisted = target.get('is_blacklisted', False)
        #
        #         # ถ้า status เป็น RED แต่เป็นแมลงหรืออยู่ใน blacklist → เปลี่ยนเป็น GREEN
        #         if (status_label == 'RED' or status_label == 'DRONE') and (is_insect or is_blacklisted):
        #             status_label = 'GREEN'
        #             if DEBUG_MODE:
        #                 reason = "insect" if is_insect else "blacklist"
        #                 print(f"🚫 Target {target_id}: Blocked RED drawing - {reason}")
        #
        #         # Determine color based on status
        #         if status_label == 'RED' or status_label == 'DRONE':
        #             color = (0, 0, 255)  # Red
        #             display_status = "DRONE"
        #         elif status_label == 'ORANGE':
        #             color = (0, 165, 255)  # Orange
        #             display_status = "TRACKING"
        #         else:
        #             color = (0, 255, 0)  # Green
        #             display_status = "TRACKING"
        #
        #         # เพิ่ม padding ให้ bbox ห่างจากวัตถุ (ใช้ resolution-dependent threshold)
        #         padding = self.thresholds.get('BBOX_PADDING', BBOX_PADDING)
        #         bbox_x1 = max(0, tx - padding)
        #         bbox_y1 = max(0, ty - padding)
        #         bbox_x2 = min(w_orig, tx + tw + padding)
        #         bbox_y2 = min(h_orig, ty + th + padding)
        #
        #         # Highlight primary target with thinner border
        #         if is_primary:
        #             # Thinner border for primary (ลดจาก 3 เป็น BBOX_THICKNESS)
        #             cv2.rectangle(display_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, BBOX_THICKNESS)
        #             # Label with priority score (แสดงเฉพาะเมื่อเป็น RED/DRONE)
        #             if status_label == 'RED' or status_label == 'DRONE':
        #                 label_text = f"{display_status} #{target_id} [P:{priority_score:.2f}]"
        #             else:
        #                 label_text = None
        #         else:
        #             # Normal border for secondary targets (ลดจาก 2 เป็น BBOX_THICKNESS)
        #             cv2.rectangle(display_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, BBOX_THICKNESS)
        #             # Label (แสดงเฉพาะเมื่อเป็น RED/DRONE)
        #             if status_label == 'RED' or status_label == 'DRONE':
        #                 label_text = f"{display_status} #{target_id}"
        #             else:
        #                 label_text = None
        #
        #         # วาด text เฉพาะเมื่อเป็น RED/DRONE เท่านั้น
        #         if label_text:
        #             cv2.putText(display_frame, label_text, (bbox_x1, bbox_y1-10),
        #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #
        #         # Draw ROI box (ตรวจสอบ validation ก่อนวาด)
        #         if 'roi_box' in target and target['roi_box']:
        #             roi_box = target['roi_box']
        #             if len(roi_box) == 4:
        #                 # ตรวจสอบ validation ก่อนวาด
        #                 current_frame = getattr(self, '_current_frame_counter', 0)
        #                 is_valid_roi, validation_score, validation_reason = self._validate_roi_for_drawing(
        #                     target, roi_box, current_frame, self.thresholds
        #                 )
        #
        #                 if is_valid_roi:
        #                     if target.get('lock_mode', False):
        #                         roi_color = (0, 0, 255)  # แดง = LOCK แล้ว
        #                         roi_thickness = 1
        #                     else:
        #                         roi_color = (200, 200, 200)  # เทาจาง
        #                         roi_thickness = 1
        #
        #                     cv2.rectangle(display_frame, (roi_box[0], roi_box[1]),
        #                                  (roi_box[2], roi_box[3]), roi_color, roi_thickness)
        #                 # ถ้าไม่ผ่าน validation → ไม่วาด ROI
        #
        #         # --- Draw Path (if enabled) ---
        #         # ตรวจสอบว่าควรวาด path หรือไม่ตาม status (ใช้ original status)
        #         should_draw_path = False
        #         path_color = None
        #
        #         if original_status_label == 'RED' or original_status_label == 'DRONE':
        #             should_draw_path = DRAW_PATH and DRAW_RED_PATH
        #             path_color = RED_COLOR
        #         elif original_status_label == 'ORANGE':
        #             should_draw_path = DRAW_PATH and DRAW_ORANGE_PATH
        #             path_color = ORANGE_COLOR
        #         elif original_status_label == 'YELLOW':
        #             should_draw_path = DRAW_PATH and DRAW_YELLOW_PATH
        #             path_color = ALERT_COLOR
        #         else:  # GREEN หรือ TRACKING
        #             should_draw_path = DRAW_PATH and DRAW_GREEN_PATH
        #             path_color = NORMAL_COLOR
        #
        #         # วาด path ถ้าเปิดใช้งานและมี centers_history
        #         if should_draw_path and PATH_VISUALIZATION_ENABLED and 'centers_history' in target:
        #             centers_history = target['centers_history']
        #             if len(centers_history) >= 2:
        #                 # ใช้แค่ N จุดล่าสุด
        #                 path_points = []
        #                 for frame, center in centers_history[-PATH_MAX_POINTS:]:
        #                     if center and isinstance(center, (tuple, list)) and len(center) >= 2:
        #                         # แปลงเป็น int และตรวจสอบว่าไม่เป็น None หรือ NaN
        #                         try:
        #                             x, y = int(center[0]), int(center[1])
        #                             if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
        #                                 path_points.append((x, y))
        #                         except (ValueError, TypeError, IndexError):
        #                             continue  # ข้าม center ที่ไม่ valid
        #
        #                 if len(path_points) >= 2:
        #                     # วาดเส้น path
        #                     pts = np.array(path_points, dtype=np.int32).reshape((-1, 1, 2))
        #                     # Smooth path points ก่อนวาด
        #                     smoothed_path_points = self._smooth_path_points(path_points)
        #
        #                     # วาดเส้น path (ใช้ smoothed points)
        #                     pts = np.array(smoothed_path_points, dtype=np.int32).reshape((-1, 1, 2))
        #                     cv2.polylines(display_frame, [pts], False, path_color, 2)
        #
        #                     # วาด heat point (จุดล่าสุด - ใช้จุดจริงไม่ใช่ smoothed)
        #                     if DRAW_HEAT_POINT and HEAT_POINT_ENABLED and path_points:
        #                         last_center = path_points[-1]  # ใช้จุดจริง
        #                         # ตรวจสอบว่า last_center เป็น tuple (int, int)
        #                         if last_center and isinstance(last_center, (tuple, list)) and len(last_center) >= 2:
        #                             try:
        #                                 x, y = int(last_center[0]), int(last_center[1])
        #                                 if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
        #                                     cv2.circle(display_frame, (x, y), HEAT_POINT_RADIUS, path_color, -1)
        #                             except (ValueError, TypeError, IndexError):
        #                                 pass  # ข้ามการวาดถ้าไม่ valid

        # ============================================================================
        # วาดรอยเท้า (footprints) สำหรับ motion-only targets - คอมเม้นไว้ก่อน
        # ============================================================================
        # # วาดรอยเท้า (footprints) สำหรับ motion-only targets - วาดเป็นจุดๆ ทุกเฟรม
        # if MOTION_ONLY_TARGET_ENABLED and DRAW_GREEN_PATH:
        #     # สีเทาสำหรับรอยเท้า
        #     footprint_color = (128, 128, 128)  # สีเทา (BGR)
        #     footprint_radius = 3  # ขนาดจุดรอยเท้า
        #
        #     # วาดรอยเท้าทุก motion-only target
        #     for motion_id, motion_target in self.motion_only_targets.items():
        #         centers_history = motion_target.get('centers_history', [])
        #
        #         # วาดรอยเท้าเป็นจุดๆ ทุกจุดที่เก็บได้ (ไม่ต้องกรอง ไม่ต้องเชื่อมเส้น)
        #         if len(centers_history) > 0:
        #             # วาดทุกจุดใน history (ไม่ต้องกรอง ไม่ต้องคำนวณ)
        #             for frame, center in centers_history:
        #                 if center:
        #                     try:
        #                         # ตรวจสอบว่า center เป็น tuple/list
        #                         if isinstance(center, (tuple, list)) and len(center) >= 2:
        #                             x, y = int(float(center[0])), int(float(center[1]))
        #                             # วาดจุดรอยเท้า (ไม่ต้องตรวจสอบ NaN/Inf - วาดทุกจุด)
        #                             cv2.circle(display_frame, (x, y), footprint_radius, footprint_color, -1)
        #                         elif isinstance(center, (int, float)):
        #                             # ถ้า center เป็นตัวเลขเดียว → ข้าม (ไม่ valid)
        #                             continue
        #                     except (ValueError, TypeError, IndexError, OverflowError):
        #                         continue  # ข้าม center ที่แปลงค่าไม่ได้
        #
        #             # วาดจุดล่าสุดด้วย (ถ้ามี last_center)
        #             last_center = motion_target.get('last_center')
        #             if last_center:
        #                 try:
        #                     if isinstance(last_center, (tuple, list)) and len(last_center) >= 2:
        #                         x, y = int(float(last_center[0])), int(float(last_center[1]))
        #                         # วาดจุดล่าสุดด้วย (สีเข้มขึ้นเล็กน้อย)
        #                         cv2.circle(display_frame, (x, y), footprint_radius + 1, (100, 100, 100), -1)
        #                 except (ValueError, TypeError, IndexError, OverflowError):
        #                     pass

        # Draw footprints (footprints module) - วาดก่อน motion boxes
        # วาด footprints จากเฟรมก่อนหน้า (สีน้ำเงินสำหรับ motion, สีส้มสำหรับ YOLO)
        if FOOTPRINTS_MODULE_ENABLED and self.footprints_drawer:
            self.footprints_drawer.draw_footprints(display_frame, current_frame_counter=frame_counter)

        # วาด motion boxes เฟรมปัจจุบัน (สีเขียว)
        if DRAW_MOTION_BOXES and hasattr(self, 'motion_boxes') and self.motion_boxes:
            for motion_box in self.motion_boxes:
                mx, my, mw, mh = motion_box
                cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 1)  # Green, thin border

        # Remove lost targets (after drawing)
        for target_id in targets_to_remove:
            if target_id in self.targets:
                del self.targets[target_id]
                if DEBUG_MODE:
                    print(f"⚠️ Hybrid Tracker: Target {target_id} Lost")
                # If primary target was removed, reset primary
                if target_id == primary_target_id:
                    primary_target_id = None
                    primary_target = None

        # Return tracking info
        # ใช้ original_confidence สำหรับ sound alert (ไม่ใช้ boosted)
        max_original_confidence = max([t.get('original_confidence', t.get('confidence', 0.0)) for t in self.targets.values()] + [0.0])
        is_high_confidence_original = max_original_confidence >= HYBRID_BASE_CONF and len(self.targets) > 0

        # Get max path_based_confidence สำหรับ tiny objects
        max_path_based_confidence = max([t.get('path_based_confidence', 0.0) for t in self.targets.values()] + [0.0])

        # Get primary target info
        primary_conf = 0.0
        primary_status = 'TRACKING'
        if primary_target_id is not None and primary_target:
            primary_conf = primary_target.get('original_confidence', primary_target.get('confidence', 0.0))
            primary_status = primary_target.get('status', 'TRACKING')

        tracking_info = {
            'target_locked': any_target_locked,
            'num_targets': len(self.targets),
            'missed_frames': max([t['missed_frames'] for t in self.targets.values()] + [0]),
            'status': 'DRONE' if all_high_confidence and len(self.targets) > 0 else 'TRACKING',
            'confidence': max_confidence,  # boosted confidence (ใช้สำหรับ display)
            'original_confidence': max_original_confidence,  # original confidence (ใช้สำหรับ sound alert)
            'path_based_confidence': max_path_based_confidence,  # path-based confidence สำหรับ tiny objects
            'is_high_confidence': is_high_confidence_original,  # ใช้ original_confidence เท่านั้น
            'primary_target_id': primary_target_id,  # ID ของ primary target
            'primary_confidence': primary_conf,  # Confidence ของ primary target
            'primary_status': primary_status  # Status ของ primary target
        }

        return display_frame, tracking_info

# --- PRIORITY SYSTEM FUNCTIONS ---
def is_likely_drone(target, tracker=None, min_confidence=0.1):
    """
    ตรวจสอบว่าเป้าหมายน่าจะเป็นโดรนหรือไม่ โดยใช้คุณสมบัติต่างๆ

    Args:
        target: Target dictionary หรือ TrackedObject
        tracker: TrackedObject instance (ถ้าใช้กับ multi mode)
        min_confidence: Confidence ต่ำสุดที่ยอมรับได้ (default 0.1, แต่ search mode ใช้ 0.05)

    Returns:
        bool: True ถ้าน่าจะเป็นโดรน, False ถ้าไม่น่าจะเป็น
    """
    from config import MIN_CONFIDENCE_FOR_TRACKING, MIN_CONFIDENCE_FOR_TRACKING_SEARCH

    if isinstance(target, dict):
        # Hybrid mode
        confidence = target.get('original_confidence', target.get('confidence', 0.0))

        # ใช้ min_confidence ที่ส่งเข้ามา (อาจต่ำกว่า MIN_CONFIDENCE_FOR_TRACKING)
        if confidence < min_confidence:
            return False

        # ถ้า confidence สูงพอ → น่าจะเป็นโดรน
        if confidence >= 0.3:  # Threshold สำหรับยืนยันว่าเป็นโดรน
            return True

        # ถ้า confidence อยู่ในช่วงกลาง → อนุญาตให้ติดตามไว้ก่อน (สำหรับจุดเล็ก)
        return True
    else:
        # Multi mode: ใช้ TrackedObject
        tracker = target
        if hasattr(tracker, 'classification_confidence'):
            conf = tracker.classification_confidence
            if conf < MIN_CONFIDENCE_FOR_TRACKING:
                return False
            if conf >= 0.3:
                return True

        # ตรวจสอบ path characteristics
        if tracker.path_frames < 3:
            return True  # ยังใหม่เกินไป ต้องรอ

        # ตรวจสอบ velocity (โดรนไม่ควรเร็วเกินไปหรือช้าเกินไป)
        velocity = tracker.velocity_mag
        from config import DRONE_MIN_SPEED, DRONE_MAX_SPEED
        if velocity > 0 and (velocity < DRONE_MIN_SPEED or velocity > DRONE_MAX_SPEED):
            # อาจไม่ใช่โดรน แต่ยังไม่ลบทันที (อาจเป็นโดรนที่บินช้า/เร็ว)
            pass

        return True  # อนุญาตให้ติดตามไว้ก่อน

def calculate_priority_score(target, tracker=None, frame_counter=0):
    """
    คำนวณคะแนนความสำคัญของเป้าหมายสำหรับการเลือก primary target

    Args:
        target: Target dictionary จาก HybridDroneTracker (สำหรับ hybrid mode)
                หรือ TrackedObject instance (สำหรับ multi mode)
        tracker: TrackedObject instance (ถ้าใช้กับ multi mode)
        frame_counter: Frame counter สำหรับคำนวณ duration

    Returns:
        float: Priority score (0.0 - 1.0)
    """
    from config import (
        PRIORITY_YOLO_CONF_WEIGHT, PRIORITY_PATH_QUALITY_WEIGHT,
        PRIORITY_STATUS_WEIGHT, PRIORITY_DURATION_WEIGHT,
        PRIORITY_STATUS_RED, PRIORITY_STATUS_ORANGE,
        PRIORITY_STATUS_YELLOW, PRIORITY_STATUS_GREEN,
        PRIORITY_DURATION_MAX_FRAMES,
        PRIORITY_PATH_SMOOTHNESS_WEIGHT, PRIORITY_PATH_CONSISTENCY_WEIGHT
    )

    # 1. YOLO Confidence Score
    if isinstance(target, dict):
        # Hybrid mode: ใช้ original_confidence หรือ confidence
        yolo_conf = target.get('original_confidence', target.get('confidence', 0.0))
        # Duration: ใช้ missed_frames เป็นตัวบ่งชี้ (ถ้า missed_frames ต่ำ = ติดตามได้นาน)
        # แต่เราต้องเก็บ tracking_start_frame ใน target dict
        tracking_duration = target.get('tracking_duration', 1)
        status_str = target.get('status', 'TRACKING')
    else:
        # Multi mode: ใช้ TrackedObject
        tracker = target
        yolo_conf = getattr(tracker, 'classification_confidence', 0.0)
        tracking_duration = tracker.path_frames
        status_str = tracker.status

    yolo_score = min(1.0, max(0.0, yolo_conf))

    # 2. Path Quality Score (สำหรับ multi mode เท่านั้น)
    path_quality_score = 0.5  # Default
    if tracker is not None and isinstance(tracker, TrackedObject):
        try:
            smoothness = tracker.calculate_path_smoothness()
            consistency = tracker.calculate_direction_consistency()
            path_quality_score = (
                smoothness * PRIORITY_PATH_SMOOTHNESS_WEIGHT +
                consistency * PRIORITY_PATH_CONSISTENCY_WEIGHT
            )
        except:
            path_quality_score = 0.5
    elif isinstance(target, dict):
        # สำหรับ hybrid mode: ใช้ confidence เป็นตัวบ่งชี้ path quality
        # ถ้า confidence สูง = มีแนวโน้มว่าเป็นโดรนจริง
        path_quality_score = yolo_conf * 0.5 + 0.5

    # 3. Status Score
    status_scores = {
        'RED': PRIORITY_STATUS_RED,
        'ORANGE': PRIORITY_STATUS_ORANGE,
        'YELLOW': PRIORITY_STATUS_YELLOW,
        'GREEN': PRIORITY_STATUS_GREEN,
        'DRONE': PRIORITY_STATUS_RED,  # DRONE = RED
        'TRACKING': PRIORITY_STATUS_YELLOW  # TRACKING = YELLOW
    }
    status_score = status_scores.get(status_str, PRIORITY_STATUS_GREEN)

    # 4. Duration Score (normalized)
    duration_score = min(1.0, tracking_duration / PRIORITY_DURATION_MAX_FRAMES)

    # Calculate final priority score
    priority_score = (
        yolo_score * PRIORITY_YOLO_CONF_WEIGHT +
        path_quality_score * PRIORITY_PATH_QUALITY_WEIGHT +
        status_score * PRIORITY_STATUS_WEIGHT +
        duration_score * PRIORITY_DURATION_WEIGHT
    )

    return min(1.0, max(0.0, priority_score))

def _calculate_roi_priority_score(tracker, frame_counter, processing_fps=None):
    """
    คำนวณ priority score สำหรับ ROI box เพื่อใช้ในการจำกัดจำนวน ROI ที่วาด

    Args:
        tracker: TrackedObject instance
        frame_counter: Current frame counter
        processing_fps: Optional - processing FPS (ใช้สำหรับคำนวณ duration)

    Returns:
        float: Priority score (0.0-1.0, สูง=priority สูง)
    """
    from config import (
        ROI_PRIORITY_CONTINUITY_WEIGHT, ROI_PRIORITY_PATH_LENGTH_WEIGHT,
        ROI_PRIORITY_TEMPORARY_LOSS_BONUS, ROI_PRIORITY_TRACKING_DURATION_WEIGHT,
        DEFAULT_FPS
    )

    if processing_fps is None or processing_fps <= 0:
        processing_fps = DEFAULT_FPS

    priority_score = 0.0

    # 1. Path continuity score (น้ำหนักสูง)
    if tracker.center_of_mass_history and len(tracker.center_of_mass_history) >= 3:
        # คำนวณ path continuity แบบเบาๆ
        continuity_score = 1.0
        for i in range(1, min(5, len(tracker.center_of_mass_history))):
            if i < len(tracker.center_of_mass_history):
                prev_center = tracker.center_of_mass_history[-i-1]
                curr_center = tracker.center_of_mass_history[-i]
                if prev_center and curr_center:
                    dist = math.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    # ถ้าระยะทางมากเกินไป → ลด continuity score
                    if dist > 50:  # threshold สำหรับ continuity
                        continuity_score *= 0.9
        priority_score += continuity_score * ROI_PRIORITY_CONTINUITY_WEIGHT
    else:
        # ถ้ายังไม่มี path history → continuity score ต่ำ
        priority_score += 0.3 * ROI_PRIORITY_CONTINUITY_WEIGHT

    # 2. Path length score
    path_length = len(tracker.center_of_mass_history) if tracker.center_of_mass_history else 0
    path_length_score = min(1.0, path_length / 30.0)  # normalize ตาม 30 points
    priority_score += path_length_score * ROI_PRIORITY_PATH_LENGTH_WEIGHT

    # 3. Temporary loss bonus (ถ้า missed_frames > 0 แต่ยังไม่มาก → bonus)
    if tracker.missed_frames > 0 and tracker.missed_frames <= 3:
        priority_score += ROI_PRIORITY_TEMPORARY_LOSS_BONUS

    # 4. Tracking duration score
    tracking_duration = tracker.path_frames / processing_fps if processing_fps > 0 else tracker.path_frames
    duration_score = min(1.0, tracking_duration / 5.0)  # normalize ตาม 5 seconds
    priority_score += duration_score * ROI_PRIORITY_TRACKING_DURATION_WEIGHT

    # 5. Status priority (RED > ORANGE > YELLOW > GREEN)
    status_priority_map = {
        'RED': 0.5,
        'ORANGE': 0.3,
        'YELLOW': 0.2,
        'GREEN': 0.1
    }
    status_priority = status_priority_map.get(tracker.status, 0.0)
    priority_score += status_priority * 0.2  # Weight: 20%

    return min(1.0, max(0.0, priority_score))

def _check_roi_area_has_dense_motion(roi_box, motion_boxes, frame_w, frame_h):
    """
    ตรวจสอบว่า ROI area มี motion ยุ่งเหยิงหรือไม่ (ใช้สำหรับกรอง ROI สีขาว)

    Args:
        roi_box: (x1, y1, x2, y2) ROI box coordinates
        motion_boxes: List of (x, y, w, h) motion boxes
        frame_w: Frame width
        frame_h: Frame height

    Returns:
        bool: True ถ้า ROI area มี motion ยุ่งเหยิง (ไม่ควรวาด ROI)
    """
    from config import (
        ROI_DENSE_MOTION_FILTER_ENABLED, ROI_DENSE_MOTION_MAX_BOXES,
        ROI_DENSE_MOTION_COVERAGE_THRESHOLD, ROI_DENSE_MOTION_EXTENDED_AREA_MULTIPLIER
    )

    # Early exit: ถ้าปิดการกรอง
    if not ROI_DENSE_MOTION_FILTER_ENABLED:
        return False

    if len(roi_box) != 4:
        return False

    x1, y1, x2, y2 = roi_box
    roi_area = (x2 - x1) * (y2 - y1)

    if roi_area <= 0:
        return False

    # ขยาย ROI area สำหรับตรวจสอบ (extended area)
    extended_width = (x2 - x1) * ROI_DENSE_MOTION_EXTENDED_AREA_MULTIPLIER
    extended_height = (y2 - y1) * ROI_DENSE_MOTION_EXTENDED_AREA_MULTIPLIER
    extended_center_x = (x1 + x2) / 2
    extended_center_y = (y1 + y2) / 2

    extended_x1 = max(0, extended_center_x - extended_width / 2)
    extended_y1 = max(0, extended_center_y - extended_height / 2)
    extended_x2 = min(frame_w, extended_center_x + extended_width / 2)
    extended_y2 = min(frame_h, extended_center_y + extended_height / 2)

    # นับ motion boxes ใน extended area
    motion_boxes_in_area = []
    total_motion_area = 0.0

    for mx, my, mw, mh in motion_boxes:
        motion_center_x = mx + mw // 2
        motion_center_y = my + mh // 2

        # ตรวจสอบว่า motion box center อยู่ใน extended area หรือไม่
        if (extended_x1 <= motion_center_x <= extended_x2 and
            extended_y1 <= motion_center_y <= extended_y2):
            motion_boxes_in_area.append((mx, my, mw, mh))
            total_motion_area += mw * mh

    # ตรวจสอบจำนวน motion boxes
    if len(motion_boxes_in_area) > ROI_DENSE_MOTION_MAX_BOXES:
        return True  # มี motion ยุ่งเหยิง

    # ตรวจสอบ coverage ratio
    extended_area = (extended_x2 - extended_x1) * (extended_y2 - extended_y1)
    if extended_area > 0:
        coverage_ratio = total_motion_area / extended_area
        if coverage_ratio > ROI_DENSE_MOTION_COVERAGE_THRESHOLD:
            return True  # มี motion ยุ่งเหยิง

    return False  # ไม่มี motion ยุ่งเหยิง

def _smooth_path_points_for_tracker(path_points, window_size=None):
    """
    Smooth path points สำหรับ TrackedObject (ใช้ในการวาด path)

    Args:
        path_points: List of (x, y) tuples
        window_size: ขนาดของ window สำหรับ moving average (ถ้า None ใช้จาก config)

    Returns:
        list: smoothed path points [(x, y), ...] หรือ path_points เดิมถ้าปิด smoothing
    """
    from config import PATH_SMOOTHING_ENABLED, PATH_SMOOTHING_WINDOW_SIZE

    # Early exit: ถ้าปิดการ smoothing หรือมีจุดน้อยเกินไป
    if not PATH_SMOOTHING_ENABLED or len(path_points) < 2:
        return path_points.copy()

    if window_size is None:
        window_size = PATH_SMOOTHING_WINDOW_SIZE

    # Early exit: ถ้ามีจุดน้อยกว่า window size → ไม่ต้อง smooth
    if len(path_points) < window_size:
        return path_points.copy()

    # Moving average smoothing (O(n) - เร็วมาก)
    smoothed = []
    half_window = window_size // 2

    for i in range(len(path_points)):
        # คำนวณ window bounds
        start_idx = max(0, i - half_window)
        end_idx = min(len(path_points), i + half_window + 1)

        # คำนวณ average ของจุดใน window
        window_points = path_points[start_idx:end_idx]
        avg_x = sum(p[0] for p in window_points) / len(window_points)
        avg_y = sum(p[1] for p in window_points) / len(window_points)
        smoothed.append((int(avg_x), int(avg_y)))  # เก็บเป็น int สำหรับการวาด

    return smoothed

# --- FOOTPRINTS ROI TRACKING AND SWARMING DETECTION ---
class FootprintsROITracker:
    """
    Class สำหรับ track ROI จาก footprints และตรวจจับ swarming behavior
    """
    def __init__(self):
        self.roi_tracking = {}  # {roi_id: {...}}
        self.next_roi_id = 0
        self.frame_counter = 0

    def _generate_roi_id(self, roi_box):
        """สร้าง ROI ID จาก ROI box (ใช้ center เป็น key)"""
        x1, y1, x2, y2 = roi_box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # ใช้ center เป็น key (ปัดเศษเพื่อให้ใกล้เคียงกัน)
        key = (center[0] // 10 * 10, center[1] // 10 * 10)
        return key

    def update_roi(self, roi_box, frame_counter, merged_with=None):
        """
        อัปเดต ROI tracking

        Args:
            roi_box: (x1, y1, x2, y2) ROI box
            frame_counter: Current frame counter
            merged_with: List of ROI IDs ที่ merge เข้า ROI นี้ (optional)

        Returns:
            roi_id: ROI ID
        """
        roi_id = self._generate_roi_id(roi_box)
        center = ((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)

        if roi_id not in self.roi_tracking:
            # สร้าง ROI tracking ใหม่
            self.roi_tracking[roi_id] = {
                'roi_box_history': [],
                'merge_count': 0,
                'movement_score': 0.0,
                'power': 0.0,
                'status': 'GREEN',  # GREEN, YELLOW, FOCUS
                'yellow_duration': 0,
                'yolo_detections': [],
                'best_target': None,  # (conf, box) ของ target ที่ดีที่สุด
                'movement_frames': 0,  # จำนวนเฟรมที่ขยับติดต่อกัน
                'last_center': center,
                'yolo_frame_counter': 0  # สำหรับ focus mode interval
            }

        tracking = self.roi_tracking[roi_id]

        # อัปเดต ROI history
        tracking['roi_box_history'].append((frame_counter, roi_box, center))

        # จำกัด history
        from config import ROI_HISTORY_MAX_FRAMES
        if len(tracking['roi_box_history']) > ROI_HISTORY_MAX_FRAMES:
            tracking['roi_box_history'] = tracking['roi_box_history'][-ROI_HISTORY_MAX_FRAMES:]

        # ตรวจสอบ merge (อัปเดต merge_count)
        if merged_with:
            tracking['merge_count'] += len(merged_with)

        # ตรวจสอบการขยับ
        if tracking['last_center']:
            last_center = tracking['last_center']
            movement_dist = math.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            if movement_dist > 0:
                tracking['movement_frames'] += 1
            else:
                tracking['movement_frames'] = 0

        tracking['last_center'] = center

        return roi_id

    def increment_merge_count(self, roi_id, count=1):
        """เพิ่ม merge count สำหรับ ROI"""
        if roi_id in self.roi_tracking:
            self.roi_tracking[roi_id]['merge_count'] += count

    def get_roi_tracking(self, roi_id):
        """ดึง ROI tracking data"""
        return self.roi_tracking.get(roi_id, None)

    def cleanup_old_rois(self, current_frame_counter, max_age_frames=30):
        """ลบ ROI tracking ที่เก่าเกินไป"""
        rois_to_remove = []
        for roi_id, tracking in self.roi_tracking.items():
            if tracking['roi_box_history']:
                last_frame = tracking['roi_box_history'][-1][0]
                if current_frame_counter - last_frame > max_age_frames:
                    rois_to_remove.append(roi_id)

        for roi_id in rois_to_remove:
            del self.roi_tracking[roi_id]

# Global ROI tracker instance
_footprints_roi_tracker = None

def get_footprints_roi_tracker():
    """Get global footprints ROI tracker instance"""
    global _footprints_roi_tracker
    if _footprints_roi_tracker is None:
        _footprints_roi_tracker = FootprintsROITracker()
    return _footprints_roi_tracker

# --- FOOTPRINTS ROI FUNCTIONS ---
def detect_motion_groups(motion_boxes, frame_w, frame_h, thresholds):
    """
    ตรวจจับกลุ่ม motion (swarming) โดยใช้ fast distance-based grouping

    Args:
        motion_boxes: List of (x, y, w, h) motion boxes
        frame_w: Frame width
        frame_h: Frame height
        thresholds: Dictionary of thresholds (from resolution_thresholds)

    Returns:
        List of (group_bbox, group_size, density) tuples
        - group_bbox: (x1, y1, x2, y2) bounding box ที่ครอบคลุมทั้งกลุ่ม
        - group_size: จำนวน motion boxes ในกลุ่ม
        - density: ความหนาแน่นของกลุ่ม (0.0-1.0)
    """
    from config import (
        FOOTPRINTS_ROI_MOTION_GROUP_DISTANCE_RATIO,
        FOOTPRINTS_ROI_MIN_GROUP_SIZE,
        FOOTPRINTS_ROI_MAX_MOTION_BOXES_CHECK
    )

    # Early exit: ถ้าไม่มี motion boxes หรือมีน้อยเกินไป
    if not motion_boxes or len(motion_boxes) < FOOTPRINTS_ROI_MIN_GROUP_SIZE:
        return []

    # Limit: จำกัดจำนวน motion boxes ที่ตรวจสอบ (performance)
    check_boxes = motion_boxes[:FOOTPRINTS_ROI_MAX_MOTION_BOXES_CHECK]

    # คำนวณ distance threshold (ใช้ distance squared)
    frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)
    max_distance = frame_diagonal * FOOTPRINTS_ROI_MOTION_GROUP_DISTANCE_RATIO
    max_distance_sq = max_distance * max_distance  # ใช้ distance squared

    # Grid-based pre-filtering: แบ่งเฟรมเป็น grid (8x8)
    grid_cols, grid_rows = 8, 8
    cell_w = frame_w / grid_cols
    cell_h = frame_h / grid_rows

    # จัดกลุ่ม motion boxes ตาม grid cells
    grid_cells = {}
    for i, box in enumerate(check_boxes):
        if len(box) < 4:
            continue
        x, y, w, h = box[0], box[1], box[2], box[3]
        center_x = x + w // 2
        center_y = y + h // 2

        # หา grid cell
        cell_col = min(int(center_x / cell_w), grid_cols - 1)
        cell_row = min(int(center_y / cell_h), grid_rows - 1)
        cell_key = (cell_row, cell_col)

        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append(i)

    # ใช้ Union-Find สำหรับ grouping
    parent = list(range(len(check_boxes)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # ตรวจสอบ motion boxes ใน grid cells ที่อยู่ใกล้กัน
    for cell_key, box_indices in grid_cells.items():
        cell_row, cell_col = cell_key

        # ตรวจสอบ motion boxes ใน cell เดียวกัน
        for i in range(len(box_indices)):
            for j in range(i + 1, len(box_indices)):
                idx1, idx2 = box_indices[i], box_indices[j]
                box1, box2 = check_boxes[idx1], check_boxes[idx2]

                if len(box1) < 4 or len(box2) < 4:
                    continue

                # คำนวณ distance squared ระหว่าง centers
                cx1 = box1[0] + box1[2] // 2
                cy1 = box1[1] + box1[3] // 2
                cx2 = box2[0] + box2[2] // 2
                cy2 = box2[1] + box2[3] // 2

                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2

                if dist_sq <= max_distance_sq:
                    union(idx1, idx2)

        # ตรวจสอบ motion boxes ใน cells ที่อยู่ใกล้กัน (adjacent cells)
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            adj_row, adj_col = cell_row + dr, cell_col + dc
            if 0 <= adj_row < grid_rows and 0 <= adj_col < grid_cols:
                adj_key = (adj_row, adj_col)
                if adj_key in grid_cells:
                    for idx1 in box_indices:
                        for idx2 in grid_cells[adj_key]:
                            box1, box2 = check_boxes[idx1], check_boxes[idx2]

                            if len(box1) < 4 or len(box2) < 4:
                                continue

                            cx1 = box1[0] + box1[2] // 2
                            cy1 = box1[1] + box1[3] // 2
                            cx2 = box2[0] + box2[2] // 2
                            cy2 = box2[1] + box2[3] // 2

                            dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2

                            if dist_sq <= max_distance_sq:
                                union(idx1, idx2)

    # จัดกลุ่ม motion boxes ตาม parent
    groups = {}
    for i in range(len(check_boxes)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # กรองกลุ่มที่มีขนาดพอ (min_group_size)
    valid_groups = []
    for group_indices in groups.values():
        if len(group_indices) >= FOOTPRINTS_ROI_MIN_GROUP_SIZE:
            # คำนวณ bounding box ที่ครอบคลุมทั้งกลุ่ม
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            total_area = 0.0

            for idx in group_indices:
                box = check_boxes[idx]
                if len(box) < 4:
                    continue
                x, y, w, h = box[0], box[1], box[2], box[3]
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                total_area += w * h

            if min_x < float('inf'):
                group_bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
                group_size = len(group_indices)

                # คำนวณ density (coverage ratio)
                bbox_area = (max_x - min_x) * (max_y - min_y)
                density = total_area / bbox_area if bbox_area > 0 else 0.0

                valid_groups.append((group_bbox, group_size, density))

    # เรียงตาม group size และ density (ใหญ่และหนาแน่น = สูงกว่า)
    valid_groups.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return valid_groups

def detect_isolated_motion_with_path(footprints_drawer, motion_boxes, frame_w, frame_h, frame_counter, thresholds):
    """
    ตรวจจับ motion เดี่ยวที่มี path (isolated motion with path history)

    Args:
        footprints_drawer: FootprintsDrawer instance
        motion_boxes: List of (x, y, w, h) motion boxes
        frame_w: Frame width
        frame_h: Frame height
        frame_counter: Current frame counter
        thresholds: Dictionary of thresholds

    Returns:
        List of (motion_box, path_quality_score) tuples
        - motion_box: (x, y, w, h) motion box
        - path_quality_score: Path quality score (0.0-1.0)
    """
    from config import (
        FOOTPRINTS_ROI_ISOLATED_PATH_MIN_FRAMES,
        FOOTPRINTS_ROI_MAX_FOOTPRINTS_HISTORY,
        FOOTPRINTS_ISOLATED_MOTION_RADIUS_RATIO,
        FOOTPRINTS_ISOLATED_MOTION_MAX_NEIGHBORS
    )

    # Early exit: ถ้าไม่มี motion boxes
    if not motion_boxes:
        return []

    # Limit: ตรวจสอบเฉพาะ motion boxes ที่โดดเดี่ยวจริงๆ (max 20 boxes)
    max_check = 20
    check_boxes = motion_boxes[:max_check]

    # คำนวณ isolation radius (ใช้ distance squared)
    frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)
    isolation_radius = frame_diagonal * FOOTPRINTS_ISOLATED_MOTION_RADIUS_RATIO
    isolation_radius_sq = isolation_radius * isolation_radius

    isolated_motions = []

    for box in check_boxes:
        if len(box) < 4:
            continue

        x, y, w, h = box[0], box[1], box[2], box[3]
        center_x = x + w // 2
        center_y = y + h // 2

        # Fast isolation check: นับ neighbors (ใช้ distance squared)
        neighbor_count = 0
        for other_box in motion_boxes:
            if len(other_box) < 4 or other_box == box:
                continue

            ox, oy, ow, oh = other_box[0], other_box[1], other_box[2], other_box[3]
            other_center_x = ox + ow // 2
            other_center_y = oy + oh // 2

            dist_sq = (center_x - other_center_x)**2 + (center_y - other_center_y)**2

            if dist_sq <= isolation_radius_sq:
                neighbor_count += 1
                # Early exit: ถ้ามี neighbors มากเกินไป → ไม่ใช่ isolated
                if neighbor_count > FOOTPRINTS_ISOLATED_MOTION_MAX_NEIGHBORS:
                    break

        # ถ้าไม่ใช่ isolated → skip
        if neighbor_count > FOOTPRINTS_ISOLATED_MOTION_MAX_NEIGHBORS:
            continue

        # Path continuity check: ตรวจสอบเฉพาะเฟรมล่าสุด 2-3 เฟรม
        if hasattr(footprints_drawer, '_check_motion_continuity'):
            is_continuous, continuity_score, continuity_frames = footprints_drawer._check_motion_continuity(
                box, frame_counter, frame_w, frame_h
            )

            # ต้องมี continuity อย่างน้อย min_frames
            if is_continuous and continuity_frames >= FOOTPRINTS_ROI_ISOLATED_PATH_MIN_FRAMES:
                # คำนวณ path quality score (ใช้ continuity score)
                path_quality_score = min(1.0, continuity_score * 1.2)  # Boost score เล็กน้อย
                isolated_motions.append((box, path_quality_score))

    # เรียงตาม path quality score (สูง = สูงกว่า)
    isolated_motions.sort(key=lambda x: x[1], reverse=True)

    return isolated_motions

def detect_roi_swarming(roi_tracker, roi_id, frame_w, frame_h, thresholds):
    """
    ตรวจจับ swarming behavior ของ ROI (ROI merge และขยับ)

    Args:
        roi_tracker: FootprintsROITracker instance
        roi_id: ROI ID
        frame_w: Frame width
        frame_h: Frame height
        thresholds: Dictionary of thresholds

    Returns:
        tuple: (is_swarming: bool, movement_score: float, merge_count: int)
    """
    from config import (
        FOOTPRINTS_ROI_SWARMING_ENABLED,
        FOOTPRINTS_ROI_MOVEMENT_THRESHOLD_RATIO,
        FOOTPRINTS_ROI_MERGE_DETECTION_ENABLED,
        FOOTPRINTS_ROI_SWARMING_MIN_MOVEMENT_FRAMES
    )

    if not FOOTPRINTS_ROI_SWARMING_ENABLED:
        return (False, 0.0, 0)

    tracking = roi_tracker.get_roi_tracking(roi_id)
    if not tracking or len(tracking['roi_box_history']) < 2:
        return (False, 0.0, 0)

    # คำนวณ movement score
    movement_score = 0.0
    frame_diagonal = math.sqrt(frame_w**2 + frame_h**2)
    movement_threshold = frame_diagonal * FOOTPRINTS_ROI_MOVEMENT_THRESHOLD_RATIO

    # ตรวจสอบการขยับจาก history (ใช้ distance squared เพื่อความเร็ว)
    history = tracking['roi_box_history']
    total_movement = 0.0
    movement_frames = 0
    movement_threshold_sq = movement_threshold * movement_threshold  # ใช้ distance squared

    for i in range(1, min(len(history), 5)):  # ตรวจสอบ 5 เฟรมล่าสุด
        prev_frame, prev_box, prev_center = history[-i-1]
        curr_frame, curr_box, curr_center = history[-i]

        # ใช้ distance squared (ไม่ต้อง sqrt)
        movement_dist_sq = (
            (curr_center[0] - prev_center[0])**2 +
            (curr_center[1] - prev_center[1])**2
        )

        if movement_dist_sq > movement_threshold_sq:
            # sqrt เฉพาะเมื่อจำเป็น (เมื่อผ่าน threshold)
            movement_dist = math.sqrt(movement_dist_sq)
            total_movement += movement_dist
            movement_frames += 1

    # คำนวณ movement score (normalize)
    if movement_frames > 0:
        avg_movement = total_movement / movement_frames
        movement_score = min(1.0, avg_movement / (frame_diagonal * 0.1))  # Normalize to 0-1

    # ตรวจสอบ merge count
    merge_count = tracking['merge_count'] if FOOTPRINTS_ROI_MERGE_DETECTION_ENABLED else 0

    # ตรวจสอบว่าเป็น swarming หรือไม่
    is_swarming = (
        tracking['movement_frames'] >= FOOTPRINTS_ROI_SWARMING_MIN_MOVEMENT_FRAMES and
        (movement_score > 0.1 or merge_count > 0)
    )

    return (is_swarming, movement_score, merge_count)

def update_roi_power(roi_tracker, roi_id, is_swarming, movement_score, merge_count, has_yolo_detection=False):
    """
    อัปเดตพลังของ ROI ตาม swarming behavior

    Args:
        roi_tracker: FootprintsROITracker instance
        roi_id: ROI ID
        is_swarming: bool - มี swarming behavior หรือไม่
        movement_score: float - movement score (0.0-1.0)
        merge_count: int - จำนวนครั้งที่ merge
        has_yolo_detection: bool - มี YOLO detection หรือไม่

    Returns:
        float: พลังใหม่ของ ROI
    """
    from config import (
        FOOTPRINTS_ROI_POWER_MOVEMENT_BONUS,
        FOOTPRINTS_ROI_POWER_MERGE_BONUS,
        FOOTPRINTS_ROI_POWER_YOLO_BONUS,
        FOOTPRINTS_ROI_POWER_DECAY_RATE,
        FOOTPRINTS_ROI_POWER_MAX,
        FOOTPRINTS_ROI_POWER_MIN
    )

    tracking = roi_tracker.get_roi_tracking(roi_id)
    if not tracking:
        return 0.0

    power = tracking['power']

    # เพิ่มพลังเมื่อมี swarming behavior
    if is_swarming:
        # Movement bonus
        if movement_score > 0:
            power += FOOTPRINTS_ROI_POWER_MOVEMENT_BONUS * movement_score

        # Merge bonus
        if merge_count > 0:
            power += FOOTPRINTS_ROI_POWER_MERGE_BONUS * min(merge_count, 3)  # จำกัดที่ 3 ครั้ง

    # YOLO bonus
    if has_yolo_detection:
        power += FOOTPRINTS_ROI_POWER_YOLO_BONUS

    # Decay (ลดพลังเมื่อไม่มีการเคลื่อนไหว)
    if not is_swarming and not has_yolo_detection:
        power -= FOOTPRINTS_ROI_POWER_DECAY_RATE

    # Clamp power
    power = max(FOOTPRINTS_ROI_POWER_MIN, min(FOOTPRINTS_ROI_POWER_MAX, power))

    tracking['power'] = power
    return power

def update_roi_status(roi_tracker, roi_id):
    """
    อัปเดต status ของ ROI ตามพลัง (GREEN -> YELLOW -> FOCUS)

    Args:
        roi_tracker: FootprintsROITracker instance
        roi_id: ROI ID

    Returns:
        str: Status ใหม่ ('GREEN', 'YELLOW', 'FOCUS')
    """
    from config import (
        FOOTPRINTS_ROI_YELLOW_POWER_THRESHOLD,
        FOOTPRINTS_ROI_YELLOW_DURATION_THRESHOLD
    )

    tracking = roi_tracker.get_roi_tracking(roi_id)
    if not tracking:
        return 'GREEN'

    power = tracking['power']
    current_status = tracking['status']

    # ตรวจสอบว่าเปลี่ยนเป็น YELLOW หรือไม่
    if power >= FOOTPRINTS_ROI_YELLOW_POWER_THRESHOLD:
        if current_status == 'GREEN':
            tracking['status'] = 'YELLOW'
            tracking['yellow_duration'] = 0
        elif current_status == 'YELLOW':
            tracking['yellow_duration'] += 1
            # ถ้าอยู่ใน YELLOW นานพอ → เปลี่ยนเป็น FOCUS
            if tracking['yellow_duration'] >= FOOTPRINTS_ROI_YELLOW_DURATION_THRESHOLD:
                tracking['status'] = 'FOCUS'
    else:
        # พลังต่ำ → กลับเป็น GREEN
        if current_status in ('YELLOW', 'FOCUS'):
            tracking['status'] = 'GREEN'
            tracking['yellow_duration'] = 0

    return tracking['status']

def detect_yolo_in_footprints_roi(yolo_model, frame, roi_box, frame_w, frame_h):
    """
    ทำ YOLO detection ใน ROI สำหรับ footprints ROI focus mode

    Args:
        yolo_model: YOLO model instance
        frame: Full frame image
        roi_box: (x1, y1, x2, y2) ROI box coordinates
        frame_w: Frame width
        frame_h: Frame height

    Returns:
        List of (conf, box) tuples - YOLO detections sorted by confidence
    """
    from config import (
        FOOTPRINTS_ROI_FOCUS_MIN_CONF,
        FOOTPRINTS_ROI_FOCUS_TARGET_MIN_HEIGHT_RATIO,
        HYBRID_SEARCH_CONF
    )

    if yolo_model is None:
        return []

    rx1, ry1, rx2, ry2 = roi_box

    # ตรวจสอบว่า ROI อยู่ในขอบเขตเฟรม
    rx1 = max(0, min(rx1, frame_w))
    ry1 = max(0, min(ry1, frame_h))
    rx2 = max(rx1, min(rx2, frame_w))
    ry2 = max(ry1, min(ry2, frame_h))

    # ตรวจสอบว่า ROI มีขนาดพอ
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return []

    # Extract ROI area
    roi_area = frame[ry1:ry2, rx1:rx2]

    if roi_area.size == 0:
        return []

    # ตรวจสอบและแปลงเป็น BGR (3 channels) ถ้ามี 4 channels
    if len(roi_area.shape) == 3 and roi_area.shape[2] == 4:
        roi_area = cv2.cvtColor(roi_area, cv2.COLOR_BGRA2BGR)

    try:
        # Run YOLO detection in ROI
        results = yolo_model.predict(roi_area, device=0, verbose=False, conf=FOOTPRINTS_ROI_FOCUS_MIN_CONF)

        if len(results[0].boxes) == 0:
            return []

        detections = []
        min_height = frame_h * FOOTPRINTS_ROI_FOCUS_TARGET_MIN_HEIGHT_RATIO

        for box in results[0].boxes:
            b = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])

            # Convert from ROI coordinates to full frame coordinates
            yolo_x = int(b[0]) + rx1
            yolo_y = int(b[1]) + ry1
            yolo_w = int(b[2] - b[0])
            yolo_h = int(b[3] - b[1])

            # ตรวจสอบความสูง (กล่องตัวสูงๆ)
            if yolo_h >= min_height:
                detections.append((confidence, (yolo_x, yolo_y, yolo_w, yolo_h)))

        # เรียงตาม confidence (สูงสุดก่อน)
        detections.sort(key=lambda x: x[0], reverse=True)

        return detections
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ YOLO detection error in footprints ROI: {e}")
        return []

def update_focus_mode_roi(roi_tracker, roi_id, yolo_model, frame, frame_w, frame_h, frame_counter):
    """
    อัปเดต focus mode สำหรับ ROI (ทำ YOLO detection และเลือก target ที่ดีที่สุด)

    Args:
        roi_tracker: FootprintsROITracker instance
        roi_id: ROI ID
        yolo_model: YOLO model instance
        frame: Full frame image
        frame_w: Frame width
        frame_h: Frame height
        frame_counter: Current frame counter

    Returns:
        bool: True ถ้ามี YOLO detection
    """
    from config import (
        FOOTPRINTS_ROI_FOCUS_MODE_ENABLED,
        FOOTPRINTS_ROI_FOCUS_YOLO_INTERVAL,
        FOOTPRINTS_ROI_FOCUS_TARGET_MIN_CONF
    )

    if not FOOTPRINTS_ROI_FOCUS_MODE_ENABLED:
        return False

    tracking = roi_tracker.get_roi_tracking(roi_id)
    if not tracking or tracking['status'] != 'FOCUS':
        return False

    # ตรวจสอบ interval
    if 'yolo_frame_counter' not in tracking:
        tracking['yolo_frame_counter'] = 0

    tracking['yolo_frame_counter'] += 1

    if tracking['yolo_frame_counter'] < FOOTPRINTS_ROI_FOCUS_YOLO_INTERVAL:
        return len(tracking['yolo_detections']) > 0

    tracking['yolo_frame_counter'] = 0

    # ดึง ROI box จาก history
    if not tracking['roi_box_history']:
        return False

    _, roi_box, _ = tracking['roi_box_history'][-1]

    # ทำ YOLO detection
    detections = detect_yolo_in_footprints_roi(yolo_model, frame, roi_box, frame_w, frame_h)

    if detections:
        # เก็บ detections
        tracking['yolo_detections'].extend(detections)

        # จำกัดจำนวน detections
        if len(tracking['yolo_detections']) > 20:
            tracking['yolo_detections'] = tracking['yolo_detections'][-20:]

        # เลือก target ที่ดีที่สุด (conf สูงสุด)
        best_conf, best_box = max(tracking['yolo_detections'], key=lambda x: x[0])

        if best_conf >= FOOTPRINTS_ROI_FOCUS_TARGET_MIN_CONF:
            tracking['best_target'] = (best_conf, best_box)
            return True

    return len(tracking['yolo_detections']) > 0

def calculate_roi_from_footprints(footprints_drawer, frame_w, frame_h, thresholds):
    """
    คำนวณ ROI boxes จาก footprints (YOLO boxes, motion groups, isolated motion)

    Args:
        footprints_drawer: FootprintsDrawer instance
        frame_w: Frame width
        frame_h: Frame height
        thresholds: Dictionary of thresholds (from resolution_thresholds)

    Returns:
        List of (roi_box, priority, type, score) tuples
        - roi_box: (x1, y1, x2, y2) ROI box coordinates
        - priority: 1 (YOLO), 2 (motion group), 3 (isolated motion)
        - type: 'yolo', 'motion_group', 'isolated_motion'
        - score: Priority score (0.0-1.0)
    """
    from config import (
        FOOTPRINTS_ROI_ENABLED,
        FOOTPRINTS_ROI_MAX_FOOTPRINTS_HISTORY,
        FOOTPRINTS_ROI_MAX_ROI_COUNT,
        HYBRID_ROI_PADDING
    )

    # Early exit: ถ้าปิดการใช้งาน
    if not FOOTPRINTS_ROI_ENABLED:
        return []

    roi_list = []

    # 1. Priority 1: YOLO boxes (orange boxes)
    if hasattr(footprints_drawer, 'yolo_boxes_history') and footprints_drawer.yolo_boxes_history:
        # ใช้เฉพาะเฟรมปัจจุบัน (เฟรมล่าสุด)
        recent_frames = footprints_drawer.yolo_boxes_history[-1:]

        for frame_num, yolo_boxes in recent_frames:
            for box in yolo_boxes:
                if len(box) >= 4:
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    conf = box[4] if len(box) >= 5 else 0.5

                    # คำนวณ ROI box พร้อม padding
                    roi_x1 = max(0, x - HYBRID_ROI_PADDING)
                    roi_y1 = max(0, y - HYBRID_ROI_PADDING)
                    roi_x2 = min(frame_w, x + w + HYBRID_ROI_PADDING)
                    roi_y2 = min(frame_h, y + h + HYBRID_ROI_PADDING)

                    # Validation: ตรวจสอบว่า ROI อยู่ในขอบเขตเฟรม
                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
                        # ใช้ confidence เป็น score
                        score = min(1.0, conf * 1.5)  # Boost score เล็กน้อย
                        roi_list.append((roi_box, 1, 'yolo', score))

    # 2. Priority 2: Motion groups (swarming)
    if hasattr(footprints_drawer, 'motion_boxes_history') and footprints_drawer.motion_boxes_history:
        # ใช้เฉพาะเฟรมปัจจุบัน (เฟรมล่าสุด)
        recent_frames = footprints_drawer.motion_boxes_history[-1:]
        all_motion_boxes = []

        for frame_num, motion_boxes in recent_frames:
            all_motion_boxes.extend(motion_boxes)

        if all_motion_boxes:
            motion_groups = detect_motion_groups(all_motion_boxes, frame_w, frame_h, thresholds)

            for group_bbox, group_size, density in motion_groups:
                x1, y1, x2, y2 = group_bbox

                # คำนวณ ROI box พร้อม padding
                roi_x1 = max(0, x1 - HYBRID_ROI_PADDING)
                roi_y1 = max(0, y1 - HYBRID_ROI_PADDING)
                roi_x2 = min(frame_w, x2 + HYBRID_ROI_PADDING)
                roi_y2 = min(frame_h, y2 + HYBRID_ROI_PADDING)

                # Validation: ตรวจสอบว่า ROI อยู่ในขอบเขตเฟรม
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
                    # คำนวณ score จาก group size และ density
                    size_score = min(1.0, group_size / 10.0)  # Normalize to 0-1
                    score = (size_score * 0.6 + density * 0.4)  # Weight: size 60%, density 40%
                    roi_list.append((roi_box, 2, 'motion_group', score))

    # 3. Priority 3: Isolated motion with path
    if hasattr(footprints_drawer, 'motion_boxes_history') and footprints_drawer.motion_boxes_history:
        # ใช้เฟรมล่าสุด
        if footprints_drawer.motion_boxes_history:
            frame_num, current_motion_boxes = footprints_drawer.motion_boxes_history[-1]

            if current_motion_boxes:
                isolated_motions = detect_isolated_motion_with_path(
                    footprints_drawer, current_motion_boxes, frame_w, frame_h, frame_num, thresholds
                )

                for motion_box, path_quality_score in isolated_motions:
                    if len(motion_box) >= 4:
                        x, y, w, h = motion_box[0], motion_box[1], motion_box[2], motion_box[3]

                        # คำนวณ ROI box พร้อม padding
                        roi_x1 = max(0, x - HYBRID_ROI_PADDING)
                        roi_y1 = max(0, y - HYBRID_ROI_PADDING)
                        roi_x2 = min(frame_w, x + w + HYBRID_ROI_PADDING)
                        roi_y2 = min(frame_h, y + h + HYBRID_ROI_PADDING)

                        # Validation: ตรวจสอบว่า ROI อยู่ในขอบเขตเฟรม
                        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                            roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
                            roi_list.append((roi_box, 3, 'isolated_motion', path_quality_score))

    # เรียงตาม priority และ score (priority ต่ำ = สูงกว่า, score สูง = สูงกว่า)
    roi_list.sort(key=lambda x: (x[1], -x[3]))  # priority ascending, score descending

    # Limit: จำกัดจำนวน ROI ที่ return
    roi_list = roi_list[:FOOTPRINTS_ROI_MAX_ROI_COUNT]

    return roi_list

# --- MAIN SYSTEM ---
def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    horizon_points = []
    temp_draw_points = []
    drawing_mode = {"active": False}
    horizon_mask_upper = None
    horizon_mask_lower = None
    motion_detection_area_mask = None
    sky_gating_on = SKY_GATING_ENABLED
    scale_factor = 1.0  # Initialize scale factor (will be calculated after getting frame)

    # โหลดเส้นขอบฟ้าที่เคยบันทึกไว้
    if os.path.exists(HORIZON_FILE):
        try:
            horizon_points = np.load(HORIZON_FILE).astype(np.int32).tolist()
            print(f"✅ Loaded horizon polyline with {len(horizon_points)} points")
        except Exception as e:
            print(f"⚠️ Failed to load horizon file: {e}")

    # Mouse callback สำหรับวาดเส้นขอบฟ้า
    # Scale coordinates back to original resolution for horizon drawing
    def on_mouse(event, x, y, flags, userdata):
        nonlocal temp_draw_points, scale_factor
        if not drawing_mode["active"]:
            return
        # Scale coordinates back to original frame resolution
        if scale_factor > 0:
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
        else:
            orig_x, orig_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            temp_draw_points = [(orig_x, orig_y)]
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            temp_draw_points.append((orig_x, orig_y))
        elif event == cv2.EVENT_LBUTTONUP:
            temp_draw_points.append((orig_x, orig_y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    cam = CameraStream()
    cam.start()
    monitor = SystemMonitor()

    # Initialize sound alert system
    sound_alert = None
    if SOUND_ALERT_ENABLED:
        sound_alert = SoundAlert(SOUND_FILE)
        sound_alert.start()

    # Initialize PTZ verification (will be set later if enabled)
    ptz_verification = None
    cam2_stream = None

    # Initialize Camera 2 stream and PTZ Verification Manager if enabled
    if CAM2_PTZ_ENABLED:
        try:
            from ptz_verification import PTZVerificationManager

            # Initialize Camera 2 stream
            try:
                cam2_config = get_camera_config("cam2")
                cam2_stream = CameraStream(
                    cam2_config["rtsp_url"] if not cam2_config["use_video_file"] else cam2_config["video_filename"],
                    cam2_config["width"], cam2_config["height"], cam2_config["use_video_file"], "cam2"
                )
                cam2_stream.start()
                print("✅ Camera 2 stream initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Camera 2 stream: {e}")
                cam2_stream = None

            # Initialize PTZ Verification Manager (will be initialized after yolo_model is ready)
            # We'll set it up after yolo_model is loaded
            print("⏳ PTZ Verification Manager will be initialized after YOLO model is loaded")
        except ImportError as e:
            print(f"⚠️ Failed to import PTZVerificationManager: {e}")
            ptz_verification = None
            cam2_stream = None

    print("⏳ Waiting for camera...")
    while True:
        running, frame = cam.read()
        if running and frame is not None: break
        time.sleep(0.1)

    h, w = frame.shape[:2]
    print(f"✅ Video Source: {w}x{h} (Full Resolution Mode)")

    # --- CALCULATE RESOLUTION-DEPENDENT THRESHOLDS ---
    resolution_thresholds = get_resolution_dependent_thresholds(w, h)
    if DEBUG_MODE:
        print(f"📊 Resolution-dependent thresholds calculated for {w}x{h}")
        print(f"   MAX_REID_DISTANCE: {resolution_thresholds['MAX_REID_DISTANCE']:.1f}")
        print(f"   DRONE_MIN_AREA: {resolution_thresholds['DRONE_MIN_AREA']}")
        print(f"   DRONE_MAX_AREA: {resolution_thresholds['DRONE_MAX_AREA']}")

    # --- RESOLUTION-ADAPTIVE PARAMETERS FOR JETSON ---
    # Calculate parameters based on resolution (อ่านจาก config.py)
    total_pixels = w * h

    if w >= 3840 or h >= 2160:  # 4K or larger
        if w >= 3840:  # 4K width
            FRAME_SKIP_MOG2 = RES_4K_FRAME_SKIP_MOG2
            MOG2_HISTORY = RES_4K_MOG2_HISTORY
            MOG2_HISTORY_4K = RES_4K_MOG2_HISTORY_4K
        else:  # 4K height but not 4K width
            FRAME_SKIP_MOG2 = RES_4K_HEIGHT_FRAME_SKIP_MOG2
            MOG2_HISTORY = RES_4K_HEIGHT_MOG2_HISTORY
            MOG2_HISTORY_4K = RES_4K_HEIGHT_MOG2_HISTORY_4K
        PERIODIC_MEMORY_CLEANUP = RES_4K_PERIODIC_MEMORY_CLEANUP
        DISPLAY_UPDATE_INTERVAL = RES_4K_DISPLAY_UPDATE_INTERVAL
        MAX_CONTOURS_TO_PROCESS = RES_4K_MAX_CONTOURS_TO_PROCESS
        FULL_FRAME_YOLO_INTERVAL = RES_4K_FULL_FRAME_YOLO_INTERVAL
        FULL_FRAME_YOLO_CONF = RES_4K_FULL_FRAME_YOLO_CONF
        MOG2_VAR_THRESHOLD = RES_4K_MOG2_VAR_THRESHOLD
        BASE_VAR_THRESHOLD = RES_4K_BASE_VAR_THRESHOLD
        GRID_ROWS = RES_4K_GRID_ROWS
        GRID_COLS = RES_4K_GRID_COLS
        BASE_MIN_AREA_REF = RES_4K_BASE_MIN_AREA_REF
        MAX_MIN_AREA_REF = RES_4K_MAX_MIN_AREA_REF
        HYBRID_ROI_PADDING = RES_4K_HYBRID_ROI_PADDING
        HYBRID_DIST_THRESHOLD = RES_4K_HYBRID_DIST_THRESHOLD
        HYBRID_BASE_SEARCH_CONF = RES_4K_HYBRID_BASE_SEARCH_CONF
        HYBRID_MIN_MOTION_AREA = RES_4K_HYBRID_MIN_MOTION_AREA
        print("🔧 4K Mode: Balanced optimization enabled (smooth display)")
    elif w >= 2560 or h >= 1440:  # 2K
        FRAME_SKIP_MOG2 = RES_2K_FRAME_SKIP_MOG2
        MOG2_HISTORY = RES_2K_MOG2_HISTORY
        MOG2_HISTORY_4K = RES_2K_MOG2_HISTORY_4K
        PERIODIC_MEMORY_CLEANUP = RES_2K_PERIODIC_MEMORY_CLEANUP
        DISPLAY_UPDATE_INTERVAL = RES_2K_DISPLAY_UPDATE_INTERVAL
        MAX_CONTOURS_TO_PROCESS = RES_2K_MAX_CONTOURS_TO_PROCESS
        FULL_FRAME_YOLO_INTERVAL = RES_2K_FULL_FRAME_YOLO_INTERVAL
        FULL_FRAME_YOLO_CONF = RES_2K_FULL_FRAME_YOLO_CONF
        MOG2_VAR_THRESHOLD = RES_2K_MOG2_VAR_THRESHOLD
        BASE_VAR_THRESHOLD = RES_2K_BASE_VAR_THRESHOLD
        GRID_ROWS = RES_2K_GRID_ROWS
        GRID_COLS = RES_2K_GRID_COLS
        BASE_MIN_AREA_REF = RES_2K_BASE_MIN_AREA_REF
        MAX_MIN_AREA_REF = RES_2K_MAX_MIN_AREA_REF
        HYBRID_ROI_PADDING = RES_2K_HYBRID_ROI_PADDING
        HYBRID_DIST_THRESHOLD = RES_2K_HYBRID_DIST_THRESHOLD
        HYBRID_BASE_SEARCH_CONF = RES_2K_HYBRID_BASE_SEARCH_CONF
        HYBRID_MIN_MOTION_AREA = RES_2K_HYBRID_MIN_MOTION_AREA
        print("🔧 2K Mode: Moderate optimization enabled")
    elif w >= 1920 or h >= 1080:  # 1080p
        FRAME_SKIP_MOG2 = RES_1080P_FRAME_SKIP_MOG2
        MOG2_HISTORY = RES_1080P_MOG2_HISTORY
        MOG2_HISTORY_4K = RES_1080P_MOG2_HISTORY_4K
        PERIODIC_MEMORY_CLEANUP = RES_1080P_PERIODIC_MEMORY_CLEANUP
        DISPLAY_UPDATE_INTERVAL = RES_1080P_DISPLAY_UPDATE_INTERVAL
        MAX_CONTOURS_TO_PROCESS = RES_1080P_MAX_CONTOURS_TO_PROCESS
        FULL_FRAME_YOLO_INTERVAL = RES_1080P_FULL_FRAME_YOLO_INTERVAL
        FULL_FRAME_YOLO_CONF = RES_1080P_FULL_FRAME_YOLO_CONF
        MOG2_VAR_THRESHOLD = RES_1080P_MOG2_VAR_THRESHOLD
        BASE_VAR_THRESHOLD = RES_1080P_BASE_VAR_THRESHOLD
        GRID_ROWS = RES_1080P_GRID_ROWS
        GRID_COLS = RES_1080P_GRID_COLS
        BASE_MIN_AREA_REF = RES_1080P_BASE_MIN_AREA_REF
        MAX_MIN_AREA_REF = RES_1080P_MAX_MIN_AREA_REF
        HYBRID_ROI_PADDING = RES_1080P_HYBRID_ROI_PADDING
        HYBRID_DIST_THRESHOLD = RES_1080P_HYBRID_DIST_THRESHOLD
        # สำหรับ 1080p: HYBRID_BASE_SEARCH_CONF และ HYBRID_MIN_MOTION_AREA ใช้ค่าจาก config (ไม่ override)
        print("🔧 1080p Mode: Light optimization enabled")
    else:  # Lower than 1080p
        FRAME_SKIP_MOG2 = RES_LOWER_FRAME_SKIP_MOG2
        MOG2_HISTORY = RES_LOWER_MOG2_HISTORY
        MOG2_HISTORY_4K = RES_LOWER_MOG2_HISTORY_4K
        PERIODIC_MEMORY_CLEANUP = RES_LOWER_PERIODIC_MEMORY_CLEANUP
        DISPLAY_UPDATE_INTERVAL = RES_LOWER_DISPLAY_UPDATE_INTERVAL
        MAX_CONTOURS_TO_PROCESS = RES_LOWER_MAX_CONTOURS_TO_PROCESS
        FULL_FRAME_YOLO_INTERVAL = RES_LOWER_FULL_FRAME_YOLO_INTERVAL
        FULL_FRAME_YOLO_CONF = RES_LOWER_FULL_FRAME_YOLO_CONF
        MOG2_VAR_THRESHOLD = RES_LOWER_MOG2_VAR_THRESHOLD
        BASE_VAR_THRESHOLD = RES_LOWER_BASE_VAR_THRESHOLD
        GRID_ROWS = RES_LOWER_GRID_ROWS
        GRID_COLS = RES_LOWER_GRID_COLS
        BASE_MIN_AREA_REF = RES_LOWER_BASE_MIN_AREA_REF
        MAX_MIN_AREA_REF = RES_LOWER_MAX_MIN_AREA_REF
        HYBRID_ROI_PADDING = RES_LOWER_HYBRID_ROI_PADDING
        HYBRID_DIST_THRESHOLD = RES_LOWER_HYBRID_DIST_THRESHOLD
        print("🔧 Lower Resolution Mode: Minimal optimization")

    # Override MOG2_HISTORY and other parameters
    # Note: HYBRID_ROI_PADDING and HYBRID_DIST_THRESHOLD are set above and will override config values
    print(f"📊 MOG2 History: {MOG2_HISTORY} (adaptive for {w}x{h})")
    print(f"📊 Frame Skip: {FRAME_SKIP_MOG2} (process every {FRAME_SKIP_MOG2 + 1} frames)")
    print(f"📊 MOG2 Sensitivity: varThreshold={MOG2_VAR_THRESHOLD}, baseVarThreshold={BASE_VAR_THRESHOLD}")
    print(f"📊 Grid Resolution: {GRID_ROWS}x{GRID_COLS}")
    print(f"📊 ROI Padding: {HYBRID_ROI_PADDING}, Dist Threshold: {HYBRID_DIST_THRESHOLD}")

    # --- DISPLAY SCALING SETUP ---
    # Get screen size and calculate display scale
    screen_width, screen_height = get_screen_size()

    # Determine target display size
    if DISPLAY_MAX_WIDTH and DISPLAY_MAX_HEIGHT:
        # Use config max size if specified
        max_display_w = DISPLAY_MAX_WIDTH
        max_display_h = DISPLAY_MAX_HEIGHT
    else:
        # Auto-fit to screen (leave some margin for window borders)
        max_display_w = screen_width - 100
        max_display_h = screen_height - 100

    # Calculate scale factor to fit within max display size
    scale_w = max_display_w / w if w > max_display_w else 1.0
    scale_h = max_display_h / h if h > max_display_h else 1.0
    scale_factor = min(scale_w, scale_h)  # Use smaller scale to fit both dimensions

    # Calculate actual display dimensions
    display_w = int(w * scale_factor)
    display_h = int(h * scale_factor)

    print(f"📺 [{CAMERA_NAME}] Screen: {screen_width}x{screen_height}, Display: {display_w}x{display_h} (Scale: {scale_factor:.3f})")

    # Set window size
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

    # --- INITIALIZE YOLO MODEL ---
    yolo_model = None
    model_path = os.path.join(os.path.dirname(__file__), HYBRID_MODEL_FILE)
    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            print(f"🚀 Loading YOLO model: {model_path}")
            yolo_model = YOLO(model_path, task='detect')
            # Warm up model with dummy frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_model.predict(dummy_frame, verbose=False)
            print("✅ YOLO model loaded and warmed up")
        except ImportError:
            print("⚠️ ultralytics library not found. YOLO detection will be disabled.")
            print("   Install with: pip install ultralytics")
            yolo_model = None
        except Exception as e:
            print(f"⚠️ Failed to load YOLO model: {e}")
            print("   YOLO detection will be disabled")
            yolo_model = None
    else:
        print(f"⚠️ YOLO model file not found: {model_path}")
        print("   YOLO detection will be disabled")
        yolo_model = None

    # Initialize PTZ Verification Manager after YOLO model is loaded
    if CAM2_PTZ_ENABLED and yolo_model is not None:
        try:
            from ptz_verification import PTZVerificationManager
            ptz_verification = PTZVerificationManager(yolo_model=yolo_model)
            if ptz_verification.is_active:
                print("✅ PTZ Verification Manager initialized and active")
            else:
                print("⚠️ PTZ Verification Manager initialized but not active (check PTZ controller connection)")
        except Exception as e:
            print(f"⚠️ Failed to initialize PTZ Verification Manager: {e}")
            import traceback
            traceback.print_exc()
            ptz_verification = None
    elif CAM2_PTZ_ENABLED and yolo_model is None:
        print("⚠️ PTZ Verification disabled: YOLO model not available")
        ptz_verification = None

    def rebuild_horizon_masks():
        """สร้าง mask บน/ล่าง ตามเส้นขอบฟ้า (ทำครั้งเดียวตอนมีการเซฟเส้น)"""
        nonlocal horizon_mask_upper, horizon_mask_lower, motion_detection_area_mask
        horizon_mask_upper = None
        horizon_mask_lower = None
        motion_detection_area_mask = None
        if not horizon_points or len(horizon_points) < 2:
            return
        pts = np.array(horizon_points, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return
        horizon_mask_upper = np.zeros((h, w), dtype=np.uint8)
        horizon_mask_lower = np.zeros((h, w), dtype=np.uint8)
        upper_poly = np.vstack([pts, [pts[-1, 0], 0], [pts[0, 0], 0]])
        lower_poly = np.vstack([pts, [pts[-1, 0], h], [pts[0, 0], h]])
        cv2.fillPoly(horizon_mask_upper, [upper_poly], 255)
        cv2.fillPoly(horizon_mask_lower, [lower_poly], 255)

        # Create motion_detection_area_mask by eroding horizon_mask_upper to create margin from line
        # ใช้ resolution-dependent threshold
        motion_area_margin = resolution_thresholds.get('MOTION_AREA_MARGIN', MOTION_AREA_MARGIN)
        if motion_area_margin > 0 and horizon_mask_upper is not None:
            motion_detection_area_mask = horizon_mask_upper.copy()
            # Erode to create margin from horizon line
            kernel_size = int(motion_area_margin * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            motion_detection_area_mask = cv2.erode(motion_detection_area_mask, kernel, iterations=1)
        else:
            # If no margin needed, use horizon_mask_upper directly
            motion_detection_area_mask = horizon_mask_upper.copy() if horizon_mask_upper is not None else None

    def create_exclusion_mask():
        """สร้าง exclusion mask สำหรับขอบเฟรมและรอบเส้นขอบฟ้า"""
        exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255

        # ใช้ resolution-dependent thresholds
        edge_exclusion = resolution_thresholds.get('EDGE_EXCLUSION_PIXELS', EDGE_EXCLUSION_PIXELS)
        horizon_exclusion = resolution_thresholds.get('HORIZON_EXCLUSION_PIXELS', HORIZON_EXCLUSION_PIXELS)

        # 1. Exclude edges of frame
        if edge_exclusion > 0:
            exclusion_mask[:edge_exclusion, :] = 0  # Top edge
            exclusion_mask[-edge_exclusion:, :] = 0  # Bottom edge
            exclusion_mask[:, :edge_exclusion] = 0  # Left edge
            exclusion_mask[:, -edge_exclusion:] = 0  # Right edge

        # 2. Exclude area around horizon line
        if horizon_exclusion > 0 and horizon_points and len(horizon_points) >= 2:
            pts = np.array(horizon_points, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                # Create a thicker line by dilating the horizon line
                horizon_line_mask = np.zeros((h, w), dtype=np.uint8)
                # Draw the horizon line
                for i in range(len(pts) - 1):
                    cv2.line(horizon_line_mask, tuple(pts[i]), tuple(pts[i+1]), 255, 1)
                # Dilate to create exclusion zone
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                    (int(horizon_exclusion * 2 + 1), int(horizon_exclusion * 2 + 1)))
                horizon_exclusion_mask = cv2.dilate(horizon_line_mask, kernel, iterations=1)
                # Remove horizon exclusion area from valid detection area
                exclusion_mask = cv2.bitwise_and(exclusion_mask, cv2.bitwise_not(horizon_exclusion_mask))

        return exclusion_mask

    # ถ้ามีไฟล์เส้นขอบฟ้าแล้ว สร้าง mask ตั้งแต่ต้น
    if horizon_points:
        rebuild_horizon_masks()

    # สร้าง exclusion mask สำหรับขอบเฟรมและรอบเส้นขอบฟ้า
    exclusion_mask = create_exclusion_mask()

    # สร้าง Adaptive Background Filter
    # Note: BASE_VAR_THRESHOLD, BASE_MIN_AREA_REF, MAX_MIN_AREA_REF are set above based on resolution
    adaptive_filter = AdaptiveBackgroundFilter()
    # Override adaptive filter's initial values with resolution-adaptive values
    adaptive_filter.current_var_threshold = BASE_VAR_THRESHOLD
    adaptive_filter.current_min_area_ref = BASE_MIN_AREA_REF

    # สร้าง AdaptiveMinAreaManager สำหรับ HYBRID_MIN_MOTION_AREA
    if ADAPTIVE_HYBRID_MIN_AREA_ENABLED:
        adaptive_hybrid_min_area_manager = AdaptiveMinAreaManager(
            min_area_base=ADAPTIVE_HYBRID_MIN_AREA_BASE,  # เริ่มต้นที่ 1
            min_area_max=ADAPTIVE_HYBRID_MIN_AREA_MAX,
            motion_box_threshold_low=ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_LOW,
            motion_box_threshold_high=ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_HIGH,
            fps_threshold_low=ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_LOW,
            fps_threshold_high=ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_HIGH,
            algo_time_threshold_high=ADAPTIVE_HYBRID_MIN_AREA_TIME_THRESHOLD_HIGH,
            update_interval=ADAPTIVE_HYBRID_MIN_AREA_UPDATE_INTERVAL,
            adjustment_step=ADAPTIVE_HYBRID_MIN_AREA_ADJUSTMENT_STEP
        )
        # เริ่มต้นด้วยค่าปัจจุบัน (1)
        current_hybrid_min_motion_area = ADAPTIVE_HYBRID_MIN_AREA_BASE
    else:
        adaptive_hybrid_min_area_manager = None
        current_hybrid_min_motion_area = HYBRID_MIN_MOTION_AREA
    current_var_threshold = BASE_VAR_THRESHOLD if ADAPTIVE_ENABLED else MOG2_VAR_THRESHOLD
    current_morph_kernel_size = int(MORPH_KERNEL_SIZE_MIN if ADAPTIVE_ENABLED else MORPH_KERNEL_SIZE)  # Ensure integer

    if ADAPTIVE_ENABLED:
        print(f"🔄 Adaptive Mode: ENABLED (ปรับค่าตามการขยับของพื้นหลัง)")
    else:
        print(f"🔄 Adaptive Mode: DISABLED (ใช้ค่าเดิม)")

    gpu_frame = cv2.cuda_GpuMat()
    # Don't pre-allocate gpu_gray - let cvtColor create it with correct size/format
    # Note: gpu_fg_mask is NOT reused - back_sub.apply() creates its own output buffer
    # MOG2 parameters optimized for speed and detection
    # MOG2_HISTORY is set above based on resolution (adaptive)
    # ใช้ current_var_threshold (adaptive หรือค่าเดิม)

    # [เพิ่มใหม่] จอง Memory สำหรับ Temporal Filter
    # ใช้ CV_8UC1 เพราะเป็นภาพขาวดำ (Mask)
    gpu_prev_mask = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
    gpu_prev_mask.setTo(0)  # เคลียร์ค่าเริ่มต้นเป็นสีดำ
    gpu_clean_mask = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)

    mog2_history_value = MOG2_HISTORY_4K if (w >= 3840) else MOG2_HISTORY
    back_sub = cv2.cuda.createBackgroundSubtractorMOG2(
        history=mog2_history_value,  # Use adaptive value from resolution detection
        varThreshold=current_var_threshold,
        detectShadows=MOG2_DETECT_SHADOWS
    )

    # --- [MOVED TO CPU] Create kernel for CPU Morphology ---
    # We remove the GPU morphology filter to prevent NPP crashes
    # ใช้ current_morph_kernel_size (adaptive หรือค่าเดิม)
    kernel_cpu = cv2.getStructuringElement(cv2.MORPH_RECT, (current_morph_kernel_size, current_morph_kernel_size))

    # GRID_ROWS and GRID_COLS are set above based on resolution (adaptive)
    grid_system = GridMotionFilter(w, h, GRID_ROWS, GRID_COLS,
                                   min_area_base=resolution_thresholds.get('MIN_AREA_BASE', MIN_AREA_BASE))

    global_track_id = 0
    active_trackers = {} # {id: TrackedObject}
    frame_counter = 0  # Frame counter for caching

    # Tracking mode: 'multi' or 'hybrid'
    if ADAPTIVE_HYBRID_MODE_ENABLED:
        tracking_mode = 'hybrid'  # เริ่มต้นด้วย hybrid mode
    else:
        tracking_mode = 'multi'  # Default to multi-object tracking
    hybrid_tracker = None  # Will be initialized when switching to hybrid mode

    # Adaptive hybrid mode variables
    adaptive_hybrid_last_update = -1  # Frame ล่าสุดที่อัปเดต
    adaptive_hybrid_cached_mode = tracking_mode  # Cached mode
    adaptive_hybrid_cached_tracker = None  # Cached tracker
    adaptive_hybrid_cached_grid_conf = 0.0  # Cached grid confidence
    pending_mode_switch = None  # Mode ที่รอสลับ (None = ไม่มี)
    adaptive_hybrid_counter = 0  # Counter สำหรับ hysteresis
    last_mode_switch_frame = 0  # เฟรมล่าสุดที่สลับ mode
    multi_mode_search_start_frame = 0  # เฟรมที่เริ่มใช้ multi mode ค้นหา

    # HUD display variables (ประกาศก่อน main loop เพื่อให้ HUD เข้าถึงได้)
    adaptive_var_display = BASE_VAR_THRESHOLD
    adaptive_min_area_display = BASE_MIN_AREA_REF
    adaptive_morph_size_display = int(current_morph_kernel_size)
    avg_contour_count = 0
    contours_count_for_hud = 0

    # Camera movement detection variables
    prev_small_gray = None
    cam_move_reset_counter = 0  # Counter สำหรับใช้ learning rate สูงชั่วคราว

    prev_time = time.time()
    prev_display_time = time.time()  # เพิ่มตัวแปรสำหรับ Display FPS
    display_fps = 0.0  # เริ่มต้น Display FPS
    fps = DEFAULT_FPS  # เริ่มต้น Processing FPS (ใช้ DEFAULT_FPS เพื่อป้องกัน error ในเฟรมแรก)

    while True:
        frame_counter += 1
        t_start = cv2.getTickCount()

        # คำนวณ processing FPS ก่อนเรียก process_frame() (ใช้เวลาจากเฟรมก่อนหน้า)
        curr_time = time.time()
        if frame_counter > 1:  # เริ่มคำนวณตั้งแต่เฟรมที่ 2
            fps = 1 / (curr_time - prev_time + 1e-6)
        # สำหรับเฟรมแรก ใช้ DEFAULT_FPS ที่กำหนดไว้แล้ว
        prev_time = curr_time

        running, frame = cam.read()
        if not running or frame is None:
            print("⚠️ Frame lost or stream ended.")
            break

        # ตรวจสอบและจัดการการเปลี่ยนขนาดเฟรม
        h_current, w_current = frame.shape[:2]
        # ตรวจสอบว่า gpu_prev_mask มีขนาดหรือไม่ และขนาดเปลี่ยนหรือไม่
        if gpu_prev_mask.size() != (w_current, h_current):
            # ขนาดเฟรมเปลี่ยน - reinitialize GPU masks
            gpu_prev_mask = cv2.cuda_GpuMat(h_current, w_current, cv2.CV_8UC1)
            gpu_prev_mask.setTo(0)  # เคลียร์ค่าเริ่มต้นเป็นสีดำ
            gpu_clean_mask = cv2.cuda_GpuMat(h_current, w_current, cv2.CV_8UC1)

        gpu_frame.upload(frame)

        # --- CAMERA MOVEMENT DETECTION (ทุก 5 เฟรม) ---
        camera_moved = False  # ประกาศไว้ก่อนเพื่อใช้ใน HUD
        if CAM_MOVE_DETECTION_ENABLED and frame_counter % CAM_MOVE_DETECTION_INTERVAL == 0:
            # ใช้ frame ขนาดเล็ก (320px) เพื่อความเร็ว
            small_w = 320
            small_h = int(h * (small_w / w))
            frame_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            if prev_small_gray is not None:
                move_score = cv2.absdiff(gray_small, prev_small_gray).mean()
                camera_moved = move_score > CAM_MOVE_THRESHOLD

            prev_small_gray = gray_small

            # Reset background model เมื่อกล้องขยับ
            if camera_moved and CAM_MOVE_RESET_BACKGROUND:
                mog2_history_value = MOG2_HISTORY_4K if (w >= 3840) else MOG2_HISTORY
                back_sub = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=mog2_history_value,
                    varThreshold=current_var_threshold,
                    detectShadows=MOG2_DETECT_SHADOWS
                )
                cam_move_reset_counter = CAM_MOVE_LEARNING_RATE_FRAMES  # ใช้ learning rate สูง 10 เฟรม
                if DEBUG_MODE:
                    print(f"🔄 Camera moved detected (score: {move_score:.2f}) - Background model reset")

        # --- 1. CUDA Processing: Background Subtraction ---
        # FRAME_SKIP_MOG2 is set above based on resolution (adaptive)
        process_mog2 = (FRAME_SKIP_MOG2 == 0) or (frame_counter % (FRAME_SKIP_MOG2 + 1) == 0)

        if process_mog2:
            # Normal MOG2 processing
            gpu_frame.upload(frame)
            # Let cvtColor create gpu_gray with correct size/format (don't reuse)
            if gpu_frame.channels() == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)  # Create new buffer
            else:
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2GRAY)  # Create new buffer

            # 1. Background Subtraction (Keep on GPU)
            # ใช้ learning rate สูงชั่วคราวหลัง reset เพื่อเรียนรู้เร็วขึ้น
            learning_rate = CAM_MOVE_LEARNING_RATE if cam_move_reset_counter > 0 else -1
            if cam_move_reset_counter > 0:
                cam_move_reset_counter -= 1

            gpu_fg_mask = back_sub.apply(gpu_gray, learning_rate, stream=None)

            # [แทรกใหม่] --- STAGE 1: TEMPORAL FILTER ---
            # หลักการ: พิกเซลต้องเป็นสีขาว (Motion) ทั้งในเฟรมนี้ AND เฟรมที่แล้ว ถึงจะผ่าน
            cv2.cuda.bitwise_and(gpu_fg_mask, gpu_prev_mask, gpu_clean_mask)

            # อัปเดต Mask ปัจจุบันเข้าไปใน prev_mask เพื่อใช้รอบหน้า
            gpu_fg_mask.copyTo(gpu_prev_mask)
            # -------------------------------------------------------

            # --- 2. CPU Processing: Morphology & Grid ---
            # [CRITICAL FIX] Download immediately to avoid NPP GPU Errors
            # [เปลี่ยน] ดาวน์โหลด gpu_clean_mask แทน gpu_fg_mask
            mask_cpu = gpu_clean_mask.download()
        else:
            # Skip MOG2 - reuse previous mask
            if 'mask_cpu' not in locals() or mask_cpu is None:
                # First frame - must process
                gpu_frame.upload(frame)
                # Let cvtColor create gpu_gray with correct size/format (don't reuse)
                if gpu_frame.channels() == 3:
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)  # Create new buffer
                else:
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2GRAY)  # Create new buffer
                learning_rate = CAM_MOVE_LEARNING_RATE if cam_move_reset_counter > 0 else -1
                if cam_move_reset_counter > 0:
                    cam_move_reset_counter -= 1
                gpu_fg_mask = back_sub.apply(gpu_gray, learning_rate, stream=None)

                # [แทรกใหม่] --- STAGE 1: TEMPORAL FILTER ---
                # สำหรับเฟรมแรก: ยังไม่มี prev_mask → ใช้ mask ปัจจุบันเป็น prev_mask
                cv2.cuda.bitwise_and(gpu_fg_mask, gpu_prev_mask, gpu_clean_mask)
                gpu_fg_mask.copyTo(gpu_prev_mask)
                # -------------------------------------------------------

                mask_cpu = gpu_clean_mask.download()
                skip_processing_this_frame = False  # Process first frame
            else:
                # Reuse previous mask - skip new detections this frame
                # This saves GPU memory and processing time
                skip_processing_this_frame = True

        if not skip_processing_this_frame:
            # Apply Morphology on CPU (Very fast for binary images)
            # ใช้ adaptive kernel (อัปเดตใน adaptive update logic)
            mask_cpu = cv2.morphologyEx(mask_cpu, cv2.MORPH_OPEN, kernel_cpu)
            # เพิ่ม MORPH_CLOSE เพื่อลด noise เพิ่มเติม (ใช้ adaptive kernel)
            if ADAPTIVE_ENABLED:
                mask_cpu = cv2.morphologyEx(mask_cpu, cv2.MORPH_CLOSE, kernel_cpu)
                # เพิ่ม: ใช้ erosion เพื่อลดขนาดกล่องที่ขยายตัวจากเมฆ
                if current_morph_kernel_size >= 5:  # ถ้า kernel ใหญ่พอ
                    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_cpu = cv2.erode(mask_cpu, erosion_kernel, iterations=1)

            # Update Grid System (using CPU mask)
            grid_system.update_heatmap(mask_cpu)

            # กรองการตรวจจับใหม่เฉพาะโซนท้องฟ้า (ถ้ามี mask และเปิดใช้งาน)
            detect_mask = mask_cpu
            if sky_gating_on and horizon_mask_upper is not None:
                detect_mask = cv2.bitwise_and(mask_cpu, horizon_mask_upper)

            # กรองการตรวจจับที่ขอบเฟรมและรอบเส้นขอบฟ้า (ไม่ให้ตรวจจับ motion ที่ขอบ)
            detect_mask = cv2.bitwise_and(detect_mask, exclusion_mask)

            # Find Contours (Always runs on CPU anyway)
            # Optimize: If detect_mask == mask_cpu, use same contours for both
            # Otherwise, calculate from mask_cpu once and reuse for adaptive
            if np.array_equal(detect_mask, mask_cpu):
                # No filtering applied, use same contours for both
                contours, _ = cv2.findContours(detect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_for_adaptive = contours
            else:
                # Filtering applied: calculate from detect_mask for motion detection
                contours, _ = cv2.findContours(detect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # For adaptive: use mask_cpu (unfiltered) to get accurate environmental data
                # But optimize by calculating once and reusing
                contours_for_adaptive, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Optimize sorting: use numpy.argsort for better performance
            if len(contours_for_adaptive) > MAX_CONTOURS_TO_PROCESS:
                areas = np.array([cv2.contourArea(c) for c in contours_for_adaptive])
                sorted_indices = np.argsort(areas)[::-1][:MAX_CONTOURS_TO_PROCESS]
                contours_for_adaptive = [contours_for_adaptive[i] for i in sorted_indices]
        else:
            # Skip processing - use empty contours
            contours = []
            contours_for_adaptive = []
            detect_mask = None

        if not skip_processing_this_frame:
            # Limit number of contours processed (adaptive based on resolution)
            # MAX_CONTOURS_TO_PROCESS is set above based on resolution
            if len(contours) > MAX_CONTOURS_TO_PROCESS:
                # Optimize: Use numpy.argsort for better performance
                areas = np.array([cv2.contourArea(c) for c in contours])
                sorted_indices = np.argsort(areas)[::-1][:MAX_CONTOURS_TO_PROCESS]
                contours = [contours[i] for i in sorted_indices]
                if DEBUG_MODE and frame_counter % 60 == 0:
                    print(f"⚠️ Too many contours ({len(contours)}), limiting to {MAX_CONTOURS_TO_PROCESS}")

            # --- ADAPTIVE UPDATE: ปรับค่าตามจำนวน contour (ทุก 5 เฟรม) ---
            adaptive_min_area_ref = None  # ใช้สำหรับส่งให้ get_min_area_for_object()
            # เก็บค่า adaptive สำหรับแสดงใน HUD (อัปเดตตัวแปรใน main function scope)
            # ไม่ต้องใช้ nonlocal เพราะตัวแปรอยู่ใน main() function scope เดียวกัน

            if ADAPTIVE_ENABLED:
                # เรียก update() เพื่อให้ adaptive filter ประมวลผล (อาจ return ค่าเดิมถ้ายังไม่ถึง interval)
                adaptive_var, adaptive_min_area_ref, adaptive_morph_size = adaptive_filter.update(len(contours_for_adaptive), frame_counter)

                # อัปเดตตัวแปร HUD จาก instance โดยตรง (เพื่อให้ได้ค่าล่าสุดที่ adaptive filter เก็บไว้)
                # ใช้ค่าจาก instance แทนค่าที่ return กลับมา เพราะค่าที่ return อาจเป็นค่าเดิมเมื่อยังไม่ถึง interval
                adaptive_var_display = adaptive_filter.current_var_threshold
                adaptive_min_area_display = adaptive_filter.current_min_area_ref
                adaptive_morph_size_display = int(adaptive_filter.current_morph_kernel_size)

                # คำนวณค่าเฉลี่ย contour count
                if len(adaptive_filter.contour_count_history) > 0:
                    avg_contour_count = sum(adaptive_filter.contour_count_history) / len(adaptive_filter.contour_count_history)
                else:
                    avg_contour_count = 0

                # เก็บจำนวน contours สำหรับแสดงใน HUD (ใช้ contours_for_adaptive)
                contours_count_for_hud = len(contours_for_adaptive)
            else:
                # เมื่อ ADAPTIVE_ENABLED = False: ใช้ค่า default
                adaptive_var_display = current_var_threshold
                adaptive_min_area_display = BASE_MIN_AREA_REF
                adaptive_morph_size_display = int(current_morph_kernel_size)
                avg_contour_count = 0
                contours_count_for_hud = len(contours) if 'contours' in locals() else 0

                # อัปเดต morphology kernel (ถ้าเปลี่ยน)
                if adaptive_morph_size != current_morph_kernel_size:
                    current_morph_kernel_size = int(adaptive_morph_size)  # Ensure integer
                    kernel_cpu = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                          (current_morph_kernel_size, current_morph_kernel_size))

                # อัปเดต MOG2 threshold (ถ้าเปลี่ยนมากพอ - สร้างใหม่เพราะ CUDA MOG2 ไม่มี setVarThreshold)
                # OPTIMIZED: ตรวจสอบก่อนสร้างใหม่ (สร้างใหม่เมื่อเปลี่ยนมากกว่า threshold)
                if abs(adaptive_var - current_var_threshold) > MOG2_RECREATE_THRESHOLD:
                    current_var_threshold = adaptive_var
                mog2_history_value = MOG2_HISTORY_4K if (w >= 3840) else MOG2_HISTORY
                back_sub = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=mog2_history_value,
                    varThreshold=current_var_threshold,
                    detectShadows=MOG2_DETECT_SHADOWS
                )

        display_frame = np.ascontiguousarray(frame)
        if SHOW_GRID: grid_system.draw_grid(display_frame)

        # --- Enhanced Adaptive Hybrid Mode with Grid-based decision (OPTIMIZED) ---
        if ADAPTIVE_HYBRID_MODE_ENABLED:
            # ตรวจสอบว่าต้องอัปเดตหรือไม่ (ไม่ใช่ทุกเฟรม)
            should_update = (frame_counter - adaptive_hybrid_last_update >= ADAPTIVE_HYBRID_UPDATE_INTERVAL)

            if should_update:
                new_mode, best_tracker_id, grid_confidence = adaptive_hybrid_mode_logic_enhanced(
                    hybrid_tracker, active_trackers, grid_system,
                    tracking_mode, frame_counter, w, h
                )

                # Cache results
                adaptive_hybrid_cached_mode = new_mode
                adaptive_hybrid_cached_tracker = best_tracker_id
                adaptive_hybrid_cached_grid_conf = grid_confidence
                adaptive_hybrid_last_update = frame_counter

                # ตรวจสอบว่าต้องสลับ mode หรือไม่
                if new_mode != tracking_mode:
                    # ตรวจสอบ hysteresis
                    if pending_mode_switch != new_mode:
                        pending_mode_switch = new_mode
                        adaptive_hybrid_counter = 0

                    adaptive_hybrid_counter += 1

                    if adaptive_hybrid_counter >= ADAPTIVE_HYBRID_HYSTERESIS_FRAMES:
                        # สลับ mode
                        old_mode = tracking_mode
                        tracking_mode = new_mode
                        last_mode_switch_frame = frame_counter
                        pending_mode_switch = None
                        adaptive_hybrid_counter = 0

                        if new_mode == 'hybrid':
                            # สลับเป็น hybrid mode
                            if hybrid_tracker is None:
                                hybrid_tracker = HybridDroneTracker(yolo_model, w, h, resolution_thresholds)
                            else:
                                hybrid_tracker.reset()
                            # เชื่อม hybrid_tracker กับ grid_system สำหรับตรวจสอบ noise ใน ROI
                            hybrid_tracker.grid_system = grid_system

                            # ตั้งค่า hybrid tracker ด้วย tracker ที่เลือก (ถ้ามี)
                            if best_tracker_id is not None and best_tracker_id in active_trackers:
                                best_tracker = active_trackers[best_tracker_id]
                                x, y, w_box, h_box = best_tracker.current_rect
                                # ใช้ confidence จาก tracker status ถ้ามี
                                confidence = 0.0
                                if best_tracker.status == 'RED':
                                    confidence = HYBRID_BASE_CONF  # ใช้ base conf สำหรับ RED
                                elif best_tracker.status == 'ORANGE':
                                    confidence = HYBRID_BASE_CONF * 0.8  # ประมาณ 0.32
                                hybrid_tracker.add_target_from_rect((x, y, w_box, h_box), confidence, w, h)

                            if DEBUG_MODE:
                                print(f"🔄 Adaptive: Switched to HYBRID mode (from {old_mode}, grid_conf={grid_confidence:.2f})")

                        elif new_mode == 'multi':
                            # สลับเป็น multi mode (เพื่อค้นหา)
                            multi_mode_search_start_frame = frame_counter
                            if hybrid_tracker is not None:
                                hybrid_tracker.reset()

                            if DEBUG_MODE:
                                print(f"🔄 Adaptive: Switched to MULTI mode (searching, grid_conf={grid_confidence:.2f})")
                else:
                    # ไม่ต้องสลับ mode → reset counter
                    pending_mode_switch = None
                    adaptive_hybrid_counter = 0
            else:
                # ใช้ cached values (ไม่ต้องคำนวณใหม่)
                new_mode = adaptive_hybrid_cached_mode
                best_tracker_id = adaptive_hybrid_cached_tracker
                grid_confidence = adaptive_hybrid_cached_grid_conf

                # ตรวจสอบว่าต้องสลับ mode หรือไม่ (ใช้ cached values)
                if new_mode != tracking_mode:
                    # ตรวจสอบ hysteresis
                    if pending_mode_switch != new_mode:
                        pending_mode_switch = new_mode
                        adaptive_hybrid_counter = 0

                    adaptive_hybrid_counter += 1

                    if adaptive_hybrid_counter >= ADAPTIVE_HYBRID_HYSTERESIS_FRAMES:
                        # สลับ mode (เหมือนข้างบน)
                        old_mode = tracking_mode
                        tracking_mode = new_mode
                        last_mode_switch_frame = frame_counter
                        pending_mode_switch = None
                        adaptive_hybrid_counter = 0

                        if new_mode == 'hybrid':
                            if hybrid_tracker is None:
                                hybrid_tracker = HybridDroneTracker(yolo_model, w, h, resolution_thresholds)
                            else:
                                hybrid_tracker.reset()
                            # เชื่อม hybrid_tracker กับ grid_system สำหรับตรวจสอบ noise ใน ROI
                            hybrid_tracker.grid_system = grid_system

                            if best_tracker_id is not None and best_tracker_id in active_trackers:
                                best_tracker = active_trackers[best_tracker_id]
                                x, y, w_box, h_box = best_tracker.current_rect
                                # ใช้ confidence จาก tracker status ถ้ามี
                                confidence = 0.0
                                if best_tracker.status == 'RED':
                                    confidence = HYBRID_BASE_CONF  # ใช้ base conf สำหรับ RED
                                elif best_tracker.status == 'ORANGE':
                                    confidence = HYBRID_BASE_CONF * 0.8  # ประมาณ 0.32
                                hybrid_tracker.add_target_from_rect((x, y, w_box, h_box), confidence, w, h)

                            if DEBUG_MODE:
                                print(f"🔄 Adaptive: Switched to HYBRID mode (from {old_mode})")

                        elif new_mode == 'multi':
                            multi_mode_search_start_frame = frame_counter
                            if hybrid_tracker is not None:
                                hybrid_tracker.reset()

                            if DEBUG_MODE:
                                print(f"🔄 Adaptive: Switched to MULTI mode (searching)")
                else:
                    pending_mode_switch = None
                    adaptive_hybrid_counter = 0

            # ตรวจสอบ timeout สำหรับ multi mode search
            if tracking_mode == 'multi' and frame_counter - multi_mode_search_start_frame > ADAPTIVE_HYBRID_SEARCH_TIMEOUT:
                # ค้นหานานเกินไป → กลับ hybrid
                if pending_mode_switch != 'hybrid':
                    pending_mode_switch = 'hybrid'
                    adaptive_hybrid_counter = 0

                adaptive_hybrid_counter += 1

                if adaptive_hybrid_counter >= ADAPTIVE_HYBRID_HYSTERESIS_FRAMES:
                    tracking_mode = 'hybrid'
                    last_mode_switch_frame = frame_counter
                    pending_mode_switch = None
                    adaptive_hybrid_counter = 0

                    if hybrid_tracker is None:
                        hybrid_tracker = HybridDroneTracker(yolo_model, w, h)
                    else:
                        hybrid_tracker.reset()
                    # เชื่อม hybrid_tracker กับ grid_system สำหรับตรวจสอบ noise ใน ROI
                    hybrid_tracker.grid_system = grid_system

                    if DEBUG_MODE:
                        print("🔄 Adaptive: Timeout - Switched back to HYBRID mode")

        # Clear cache เมื่อ grid update (ทุก N เฟรม)
        if frame_counter % ADAPTIVE_HYBRID_UPDATE_INTERVAL == 0:
            grid_system.clear_cache()

        # --- MODE-SPECIFIC PROCESSING ---
        if tracking_mode == 'hybrid':
            # HYBRID MODE: Single-target persistent tracking
            if hybrid_tracker is None:
                # Initialize hybrid tracker if not already initialized
                hybrid_tracker = HybridDroneTracker(yolo_model, w, h)
                # เชื่อม hybrid_tracker กับ grid_system สำหรับตรวจสอบ noise ใน ROI
                hybrid_tracker.grid_system = grid_system

            # Update hybrid tracker's background subtractor with current frame
            hybrid_tracker.gpu_frame.upload(frame)
            if hybrid_tracker.gpu_frame.channels() == 3:
                hybrid_gpu_gray = cv2.cuda.cvtColor(hybrid_tracker.gpu_frame, cv2.COLOR_BGR2GRAY)
            else:
                hybrid_gpu_gray = cv2.cuda.cvtColor(hybrid_tracker.gpu_frame, cv2.COLOR_BGRA2GRAY)
            hybrid_gpu_mask = hybrid_tracker.back_sub.apply(hybrid_gpu_gray, -1, stream=cv2.cuda.Stream_Null())
            hybrid_mask_cpu = hybrid_gpu_mask.download()

            # Process frame through hybrid tracker (uses its own mask)
            # ส่ง detect_mask, horizon_mask_upper และ motion_detection_area_mask เข้าไปเพื่อกรอง detection ให้อยู่ในโซนบนเส้นฟ้าเท่านั้น
            display_frame, hybrid_info = hybrid_tracker.process_frame(
                frame, hybrid_mask_cpu,
                detect_mask=detect_mask,
                horizon_mask_upper=horizon_mask_upper if sky_gating_on else None,
                motion_detection_area_mask=motion_detection_area_mask,
                frame_counter=frame_counter,
                min_motion_area=current_hybrid_min_motion_area,
                processing_fps=fps  # ส่ง processing FPS
            )

            # เก็บ tracking info สำหรับ sound alert
            hybrid_tracker.last_tracking_info = hybrid_info
            if 'confidence' in hybrid_info:
                hybrid_tracker.last_confidence = hybrid_info['confidence']

            # Skip multi-object tracking logic for hybrid mode
            # Clear active trackers in hybrid mode (they won't be drawn anyway)
            active_trackers = {}

        # ประกาศ enhanced_roi_rects และ enhanced_roi_contours ไว้ก่อน (ใช้ได้ทั้ง hybrid และ multi mode)
        enhanced_roi_rects = []
        enhanced_roi_contours = []

        if tracking_mode == 'multi':
            # MULTI MODE: Multi-object tracking (original logic)
            # --- 3. ENHANCED MOTION DETECTION IN ROI FOR ORANGE TRACKERS ---
            # เพิ่ม contours จาก ROI ที่ sensitive มากขึ้นสำหรับ ORANGE trackers
            # รวมถึง trackers ที่หายไปชั่วครู่ (missed_frames > 0 แต่ <= HYBRID_ROI_WAIT_FRAMES)

            if active_trackers:
                # สร้าง ROI mask สำหรับ ORANGE/RED trackers (รวมถึงที่หายไปชั่วครู่)
                for tracker_id, tracker in active_trackers.items():
                    if tracker.status == 'ORANGE' or tracker.status == 'RED':
                        # ใช้ current rect เสมอ (ไม่ใช้ predicted ROI) - คงที่ตาม resolution
                        tx, ty, tw, th = tracker.current_rect
                        roi_x1 = max(0, tx - HYBRID_ROI_PADDING)
                        roi_y1 = max(0, ty - HYBRID_ROI_PADDING)
                        roi_x2 = min(w, tx + tw + HYBRID_ROI_PADDING)
                        roi_y2 = min(h, ty + th + HYBRID_ROI_PADDING)
                        tracker_center = tracker.centroids[-1] if tracker.centroids else None

                        # Extract ROI mask
                        roi_mask = mask_cpu[roi_y1:roi_y2, roi_x1:roi_x2]
                        if roi_mask.size > 0:
                            # Apply stronger morphology operations for better sensitivity
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
                            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_close)

                            # Find contours in ROI with lower threshold (more sensitive)
                            roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            for roi_cnt in roi_contours:
                                # ใช้ min_area ที่ต่ำกว่า (sensitive มากขึ้น) - ใช้ HYBRID_MIN_MOTION_AREA
                                min_area_roi = max(1, HYBRID_MIN_MOTION_AREA // 2)  # เพิ่มความ sensitive
                                if cv2.contourArea(roi_cnt) > min_area_roi:
                                    rx, ry, rw, rh = cv2.boundingRect(roi_cnt)
                                    # Convert ROI coordinates back to full frame
                                    full_x = rx + roi_x1
                                    full_y = ry + roi_y1

                                    # Check if this detection is close to tracker center (หรือ predicted center)
                                    if tracker_center:
                                        detection_center = (full_x + rw // 2, full_y + rh // 2)
                                        dist = ((detection_center[0] - tracker_center[0])**2 +
                                               (detection_center[1] - tracker_center[1])**2)**0.5

                                        # เพิ่ม tolerance สำหรับ trackers ที่หายไปชั่วครู่
                                        dist_threshold = HYBRID_DIST_THRESHOLD * 2.0 if tracker.missed_frames > 0 else HYBRID_DIST_THRESHOLD

                                        # Only add if close enough to tracker (prevent detecting other objects)
                                        if dist < dist_threshold:
                                            enhanced_roi_rects.append([full_x, full_y, rw, rh])
                                            # Create full-frame contour by offsetting
                                            roi_cnt_full = roi_cnt.copy()
                                            roi_cnt_full[:, :, 0] += roi_x1
                                            roi_cnt_full[:, :, 1] += roi_y1
                                            enhanced_roi_contours.append(roi_cnt_full)

        # --- 3. FILTER CONTOURS based on Grid Noise ---
        if skip_processing_this_frame:
            # Skip contour filtering and tracking when skipping MOG2
            valid_rects = []
            valid_contours = []
            merged_rects = []
            merged_contours = []
            # Still update existing trackers (increment missed frames)
            for tracker_id, tracker in list(active_trackers.items()):
                tracker.missed_frames += 1
                if tracker.missed_frames > MAX_MISS_FRAMES:
                    # Remove tracker
                    del active_trackers[tracker_id]
                else:
                    # Update tracker status even when skipping
                    tracker.check_status(thresholds=resolution_thresholds)
        else:
            valid_rects = []
            valid_contours = []  # Store contours for center of mass calculation
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
            # ใช้ adaptive_min_area_ref ถ้ามี (เมื่อเปิด adaptive mode)
            req_area, _ = grid_system.get_min_area_for_object(x, y, w_box, h_box, adaptive_min_area_ref=adaptive_min_area_ref)
            if cv2.contourArea(cnt) > req_area:
                valid_rects.append([x, y, w_box, h_box])
                valid_contours.append(cnt)

            # เพิ่ม enhanced ROI detections เข้าไปด้วย
            valid_rects.extend(enhanced_roi_rects)
            valid_contours.extend(enhanced_roi_contours)

            # --- 3.5. FILTER OUT OVERSIZED CONTOURS (ป้องกันกล่องใหญ่ทั้งจอ) ---
        # กรอง contours ที่ใหญ่เกินไป (เช่น > DRONE_MAX_AREA หรือ > 5% ของเฟรม - เข้มงวดขึ้น)
        filtered_rects = []
        filtered_contours = []
        max_frame_area = w * h
        max_allowed_area = min(DRONE_MAX_AREA, max_frame_area * 0.05)  # ลดจาก 0.1 เป็น 0.05 (เข้มงวดขึ้น)

        for i, rect in enumerate(valid_rects):
            x, y, w_box, h_box = rect
            area = w_box * h_box

            # กรอง contours ที่ใหญ่เกินไป
            if area > max_allowed_area:
                if DEBUG_MODE and len(filtered_rects) % 30 == 0:  # Log ทุก 30 ตัว
                    print(f"⚠️ Filtered oversized contour: area={area:.0f} > max={max_allowed_area:.0f}")
                continue

            # กรอง contours ที่กว้างหรือสูงเกินไป (เช่น > 30% ของเฟรม - ลดจาก 50%)
            if w_box > w * 0.3 or h_box > h * 0.3:
                if DEBUG_MODE and len(filtered_rects) % 30 == 0:
                    print(f"⚠️ Filtered oversized contour: size={w_box}x{h_box} > {w*0.3:.0f}x{h*0.3:.0f}")
                continue

            # ตรวจสอบอัตราส่วนกว้าง/สูง (ป้องกันกล่องแบนผิดปกติจากเมฆ)
            if h_box > 0:
                aspect_ratio = w_box / h_box
                if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
                    if DEBUG_MODE and len(filtered_rects) % 30 == 0:
                        print(f"⚠️ Filtered invalid aspect ratio: {aspect_ratio:.2f} (w={w_box}, h={h_box})")
                    continue

            filtered_rects.append(rect)
            if i < len(valid_contours):
                filtered_contours.append(valid_contours[i])

            valid_rects = filtered_rects
            valid_contours = filtered_contours

            # --- 4. FULL-FRAME YOLO SCAN (Periodic) ---
        # FULL_FRAME_YOLO_INTERVAL and FULL_FRAME_YOLO_CONF are set above based on resolution
        if yolo_model is not None and frame_counter % FULL_FRAME_YOLO_INTERVAL == 0:
            try:
                # Convert frame to BGR if it has 4 channels (BGRA)
                yolo_frame = frame
                if frame.shape[2] == 4:
                    yolo_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Full-frame YOLO scan
                scan_results = yolo_model.predict(
                    yolo_frame,  # Use converted frame
                    device=0,
                    verbose=False,
                    conf=FULL_FRAME_YOLO_CONF,
                    imgsz=640  # Use 640 for all resolutions (model max size)
                )

                # Add detections to tracking
                for box in scan_results[0].boxes:
                    if box.cls == 0:  # Drone class
                        b = box.xyxy[0].cpu().numpy()
                        x, y, w_box, h_box = int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])
                        conf = float(box.conf[0])

                        # Check if not duplicate with existing trackers
                        is_duplicate = False
                        for tracker_id, tracker in active_trackers.items():
                            tx, ty, tw, th = tracker.current_rect
                            # Check overlap
                            overlap_x = max(0, min(x + w_box, tx + tw) - max(x, tx))
                            overlap_y = max(0, min(y + h_box, ty + th) - max(y, ty))
                            overlap_area = overlap_x * overlap_y
                            box_area = w_box * h_box
                            tracker_area = tw * th
                            iou = overlap_area / (box_area + tracker_area - overlap_area) if (box_area + tracker_area - overlap_area) > 0 else 0

                            if iou > 0.3:  # Significant overlap
                                is_duplicate = True
                                break

                        # Check if not duplicate with valid_rects
                        if not is_duplicate:
                            for vx, vy, vw, vh in valid_rects:
                                overlap_x = max(0, min(x + w_box, vx + vw) - max(x, vx))
                                overlap_y = max(0, min(y + h_box, vy + vh) - max(y, vy))
                                overlap_area = overlap_x * overlap_y
                                box_area = w_box * h_box
                                rect_area = vw * vh
                                iou = overlap_area / (box_area + rect_area - overlap_area) if (box_area + rect_area - overlap_area) > 0 else 0

                                if iou > 0.3:
                                    is_duplicate = True
                                    break

                        if not is_duplicate and conf > 0.2:
                            # Create new tracker from full-frame detection
                            new_tracker = TrackedObject([x, y, w_box, h_box], global_track_id)
                            active_trackers[global_track_id] = new_tracker
                            global_track_id += 1
                            if DEBUG_MODE:
                                print(f"🔍 Full-frame YOLO detected drone: ({x}, {y}, {w_box}, {h_box}) conf={conf:.2f}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Full-frame YOLO scan error: {e}")

            # --- 4. MERGE close bounding boxes ---
            merged_rects = merge_close_rectangles(valid_rects, resolution_thresholds.get('MERGE_DISTANCE', MERGE_DISTANCE), frame_w=w, frame_h=h)

            # Create mapping from merged_rects to contours for center of mass calculation
            # Find the best matching contour for each merged rect
            merged_contours = []
            for merged_rect in merged_rects:
                mx, my, mw, mh = merged_rect
                mcx, mcy = mx + mw // 2, my + mh // 2  # Fix: Move inside for loop
                best_contour = None
                min_dist = float('inf')
                for i, (rect, cnt) in enumerate(zip(valid_rects, valid_contours)):
                    rx, ry, rw, rh = rect
                    rcx, rcy = rx + rw // 2, ry + rh // 2
                    dist = ((mcx - rcx)**2 + (mcy - rcy)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = cnt
                merged_contours.append(best_contour if best_contour is not None else None)

            # --- 5. TRACKING & ASSOCIATION LOGIC ---
            new_trackers = {}
            used_rects = [False] * len(merged_rects)

            # a. Associate existing trackers with new detections
            for tracker_id, tracker in list(active_trackers.items()):
                best_match_idx = -1
                min_dist = float('inf')  # Fix: Move inside for loop

                # Find best match using the re-ID distance logic
                for i, new_rect in enumerate(merged_rects):
                    if used_rects[i]: continue

                    if tracker.is_close_for_reid(new_rect, frame_w=w, frame_h=h, thresholds=resolution_thresholds):
                        # Use centroid distance for matching score
                        cx1, cy1 = tracker.centroids[-1]
                        cx2, cy2 = new_rect[0] + new_rect[2]//2, new_rect[1] + new_rect[3]//2
                        dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

                        if dist < min_dist:
                            min_dist = dist
                            best_match_idx = i

                if best_match_idx != -1:  # Fix: Move inside for loop
                    # Found a match: Update tracker with contour for center of mass calculation
                    contour = merged_contours[best_match_idx] if best_match_idx < len(merged_contours) else None
                    # ใช้ Process FPS ก่อน (FPS จริงของการประมวลผล) - สำคัญกว่า Input FPS
                    current_fps_value = fps if 'fps' in locals() else None
                    # ถ้ายังไม่มี ใช้ Input FPS (fallback)
                    if current_fps_value is None:
                        current_fps_value = getattr(cam, 'fps', None) if 'cam' in locals() else None
                    tracker.update(merged_rects[best_match_idx], contour=contour,
                                 current_fps=current_fps_value, thresholds=resolution_thresholds)
                    # Reset YOLO frame counter when matched
                    tracker.yolo_frame_counter = 0
                    new_trackers[tracker_id] = tracker
                    used_rects[best_match_idx] = True
                else:
                    # No match found: Try YOLO detection for ORANGE/RED trackers first
                    yolo_rect = None
                if ((tracker.status == 'ORANGE' or tracker.status == 'RED') and
                    tracker.missed_frames == 0 and
                    yolo_model is not None):
                    # Check if it's time to run YOLO (every N frames)
                    tracker.yolo_frame_counter += 1

                    # ⚠️ NEW: ใช้ interval พิเศษสำหรับ tiny/small objects
                    # ตรวจสอบว่าเป็น tiny/small object หรือไม่
                    is_tiny = tracker.is_tiny_object(resolution_thresholds, frame_w=w, frame_h=h)
                    is_small = tracker.is_small_object(resolution_thresholds, frame_w=w, frame_h=h)

                    # กำหนด YOLO interval ตามขนาดวัตถุ
                    if is_tiny:
                        yolo_interval = TINY_OBJECT_YOLO_INTERVAL  # 60 เฟรมสำหรับ tiny objects
                    elif is_small:
                        yolo_interval = SMALL_OBJECT_YOLO_INTERVAL  # 15 เฟรมสำหรับ small objects
                    else:
                        # ใช้ interval พิเศษสำหรับ tracker ที่อยู่ใน RED lock mode (ถ้ามีฟิลด์นี้)
                        yolo_interval = HYBRID_YOLO_INTERVAL
                        if hasattr(tracker, 'lock_mode') and tracker.lock_mode:
                            from config import LOCK_YOLO_INTERVAL
                            yolo_interval = max(1, LOCK_YOLO_INTERVAL)

                    if tracker.yolo_frame_counter >= yolo_interval:
                        tracker.yolo_frame_counter = 0

                        # Calculate ROI box
                        tx, ty, tw, th = tracker.current_rect
                        roi_x1 = max(0, tx - HYBRID_ROI_PADDING)
                        roi_y1 = max(0, ty - HYBRID_ROI_PADDING)
                        roi_x2 = min(w, tx + tw + HYBRID_ROI_PADDING)
                        roi_y2 = min(h, ty + th + HYBRID_ROI_PADDING)
                        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

                        # Run YOLO detection in ROI
                        yolo_rect, yolo_confidence = detect_yolo_in_roi(yolo_model, tracker, frame, roi_box, w, h)

                        if yolo_rect is not None:
                            # YOLO found the object: Update tracker with YOLO box
                            # ใช้ Process FPS ก่อน (FPS จริงของการประมวลผล) - สำคัญกว่า Input FPS
                            current_fps_value = fps if 'fps' in locals() else None
                            # ถ้ายังไม่มี ใช้ Input FPS (fallback)
                            if current_fps_value is None:
                                current_fps_value = getattr(cam, 'fps', None) if 'cam' in locals() else None
                            tracker.update(yolo_rect, contour=None,
                                         current_fps=current_fps_value, thresholds=resolution_thresholds)
                            tracker.missed_frames = 0

                            # ตรวจสอบ confidence และเปลี่ยน status
                            if yolo_confidence is not None:
                                if yolo_confidence >= HYBRID_BASE_CONF:
                                    # Confidence สูง → เปลี่ยนเป็น RED (หรือคงเป็น RED)
                                    if tracker.status != 'RED':
                                        tracker.status = 'RED'
                                        if DEBUG_MODE:
                                            print(f"🔴 Tracker {tracker_id}: Changed to RED (conf={yolo_confidence:.2f} >= {HYBRID_BASE_CONF:.2f})")
                                else:
                                    # Confidence ต่ำ → ถ้าเป็น RED เปลี่ยนกลับเป็น ORANGE
                                    if tracker.status == 'RED':
                                        tracker.status = 'ORANGE'
                                        if DEBUG_MODE:
                                            print(f"🟠 RED Tracker {tracker_id}: Changed back to ORANGE (conf={yolo_confidence:.2f} < {HYBRID_BASE_CONF:.2f})")

                            new_trackers[tracker_id] = tracker
                            if DEBUG_MODE:
                                status_name = "ORANGE" if tracker.status == 'ORANGE' else "RED"
                                conf_str = f"{yolo_confidence:.2f}" if yolo_confidence is not None else "N/A"
                                print(f"✅ {status_name} Tracker {tracker_id}: YOLO detection in ROI (conf={conf_str})")
                            continue  # Skip to next tracker
                        else:
                            # YOLO ไม่เจอ → ถ้าเป็น RED และหายไปนาน อาจเปลี่ยนกลับเป็น ORANGE
                            if tracker.status == 'RED' and tracker.missed_frames > 5:
                                tracker.status = 'ORANGE'
                                if DEBUG_MODE:
                                    print(f"🟠 RED Tracker {tracker_id}: Changed back to ORANGE (YOLO not found for {tracker.missed_frames} frames)")

                # No match found and YOLO didn't find it: Mark as missing
                tracker.missed_frames += 1

                # --- FALLBACK: ใช้กล่องเขียวใน ROI ช่วยอัปเดตกล่องส้ม/แดง ---
                # ขยาย wait period เป็น HYBRID_ROI_WAIT_FRAMES เพื่อให้ ROI รอได้นานขึ้น
                if (tracker.status == 'ORANGE' or tracker.status == 'RED') and tracker.missed_frames <= HYBRID_ROI_WAIT_FRAMES:
                    # ถ้ากล่องส้มไม่เจอ match แต่ยังไม่หายไปนาน (<= HYBRID_ROI_WAIT_FRAMES เฟรม)
                    # ตรวจสอบว่ามีกล่องเขียวใน ROI หรือไม่ (ใช้ current rect - ไม่ใช้ predicted ROI)
                    # ใช้ current rect เสมอ (ไม่ใช้ predicted ROI) - คงที่ตาม resolution
                    tx, ty, tw, th = tracker.current_rect
                    roi_x1 = max(0, tx - HYBRID_ROI_PADDING)
                    roi_y1 = max(0, ty - HYBRID_ROI_PADDING)
                    roi_x2 = min(w, tx + tw + HYBRID_ROI_PADDING)
                    roi_y2 = min(h, ty + th + HYBRID_ROI_PADDING)
                    tracker_center = tracker.centroids[-1] if tracker.centroids else None

                    if tracker_center:
                        # คำนวณทิศทางการเคลื่อนที่ของ ORANGE tracker จากประวัติ
                        orange_direction = None
                        if len(tracker.centroids) >= 2:
                            # ใช้ 2 จุดล่าสุดเพื่อคำนวณทิศทาง
                            prev_center = tracker.centroids[-2]
                            dx = tracker_center[0] - prev_center[0]
                            dy = tracker_center[1] - prev_center[1]
                            # Normalize direction vector
                            dir_mag = (dx**2 + dy**2)**0.5
                            if dir_mag > 0.1:  # มีการเคลื่อนที่
                                orange_direction = (dx / dir_mag, dy / dir_mag)

                        # หา GREEN tracker ที่อยู่ใน ROI และใกล้กับ ORANGE tracker
                        best_green_tracker = None
                        best_green_dist = float('inf')

                        for other_id, other_tracker in active_trackers.items():
                            if other_id == tracker_id:
                                continue

                            # ตรวจสอบเฉพาะ GREEN trackers
                            if other_tracker.status == 'GREEN':
                                other_center = other_tracker.centroids[-1] if other_tracker.centroids else None
                                if other_center:
                                    # ตรวจสอบว่าอยู่ใน ROI หรือไม่
                                    if (roi_x1 <= other_center[0] <= roi_x2 and
                                        roi_y1 <= other_center[1] <= roi_y2):
                                        # คำนวณระยะห่าง
                                        dist = ((other_center[0] - tracker_center[0])**2 +
                                               (other_center[1] - tracker_center[1])**2)**0.5

                                        # ตรวจสอบทิศทางการเคลื่อนที่
                                        direction_match = True
                                        if orange_direction is not None:
                                            # คำนวณทิศทางจาก ORANGE tracker ไปยัง GREEN tracker
                                            to_green_dx = other_center[0] - tracker_center[0]
                                            to_green_dy = other_center[1] - tracker_center[1]
                                            to_green_mag = (to_green_dx**2 + to_green_dy**2)**0.5

                                            if to_green_mag > 0.1:
                                                to_green_dir = (to_green_dx / to_green_mag, to_green_dy / to_green_mag)

                                                # คำนวณ dot product เพื่อตรวจสอบว่าทิศทางตรงกันหรือไม่
                                                dot_product = orange_direction[0] * to_green_dir[0] + orange_direction[1] * to_green_dir[1]

                                                # ตรวจสอบว่าทิศทางตรงกัน (dot product > 0.5 = มุม < 60 องศา)
                                                # หรือถ้าใกล้มาก (dist < threshold/2) ก็ยอมรับได้
                                                if dot_product < 0.5 and dist >= HYBRID_DIST_THRESHOLD:
                                                    direction_match = False

                                        # ตรวจสอบว่าใกล้พอ มี path ต่อเนื่อง และทิศทางตรงกัน
                                        # เพิ่ม tolerance สำหรับ trackers ที่หายไปชั่วครู่
                                        max_dist = HYBRID_DIST_THRESHOLD * 1.2 if tracker.missed_frames > 0 else HYBRID_DIST_THRESHOLD * 2.0
                                        if (dist < max_dist and
                                            other_tracker.path_frames >= 3 and
                                            direction_match):
                                            if dist < best_green_dist:
                                                best_green_dist = dist
                                                best_green_tracker = other_tracker

                        # ถ้าเจอ GREEN tracker ใน ROI ที่เหมาะสม ให้ใช้มันมาอัปเดต ORANGE tracker
                        if best_green_tracker is not None:
                            max_dist_threshold = HYBRID_DIST_THRESHOLD * 1.2 if tracker.missed_frames > 0 else HYBRID_DIST_THRESHOLD * 2.0
                            if best_green_dist < max_dist_threshold:
                                # ใช้ rect ของ GREEN tracker มาอัปเดต ORANGE tracker
                                green_rect = best_green_tracker.current_rect
                                # อัปเดต ORANGE tracker ด้วย rect ของ GREEN (ไม่ reset missed_frames)
                                # ใช้ Process FPS ก่อน (FPS จริงของการประมวลผล) - สำคัญกว่า Input FPS
                                current_fps_value = fps if 'fps' in locals() else None
                                # ถ้ายังไม่มี ใช้ Input FPS (fallback)
                                if current_fps_value is None:
                                    current_fps_value = getattr(cam, 'fps', None) if 'cam' in locals() else None
                                tracker.update(green_rect, contour=None,
                                             current_fps=current_fps_value, thresholds=resolution_thresholds)
                                tracker.missed_frames = 0  # Reset missed_frames เพราะเจอแล้ว
                                if DEBUG_MODE:
                                    print(f"✅ ORANGE Tracker {tracker_id}: Using GREEN Tracker {best_green_tracker.id} in ROI as fallback (dist={best_green_dist:.1f}, missed={tracker.missed_frames})")

                new_trackers[tracker_id] = tracker

            # b. Create new trackers for unmatched detections
            for i, new_rect in enumerate(merged_rects):
                if not used_rects[i]:
                    # ตรวจสอบขนาด rect ก่อนสร้าง tracker
                    x, y, w_box, h_box = new_rect
                    area = w_box * h_box
                    max_frame_area = w * h

                    # กรอง rects ที่ใหญ่เกินไป
                    if area > DRONE_MAX_AREA or area > max_frame_area * 0.1:
                        if DEBUG_MODE and i % 30 == 0:  # Log ทุก 30 ตัว
                            print(f"⚠️ Skipped oversized rect for new tracker: area={area:.0f}")
                        continue  # ข้าม rect ที่ใหญ่เกินไป

                    # กรอง rects ที่กว้างหรือสูงเกิน 50% ของเฟรม
                    if w_box > w * 0.5 or h_box > h * 0.5:
                        if DEBUG_MODE and i % 30 == 0:
                            print(f"⚠️ Skipped oversized rect for new tracker: size={w_box}x{h_box}")
                        continue  # ข้าม rect ที่ใหญ่เกินไป

                    # [เพิ่มใหม่] ตรวจสอบ noise level ใน grid cell - skip ถ้า noise สูงเกินไป
                    req_area, noise = grid_system.get_min_area_for_object(x, y, w_box, h_box, adaptive_min_area_ref=adaptive_min_area_ref)
                    if noise > GRID_NOISE_FILTER_THRESHOLD:
                        if DEBUG_MODE and i % 30 == 0:  # Log ทุก 30 ตัว
                            print(f"⚠️ Skipped creating tracker in high noise area: noise={noise:.3f} > {GRID_NOISE_FILTER_THRESHOLD}")
                        continue  # Skip creating tracker in high noise area

                    new_tracker = TrackedObject(new_rect, global_track_id)
                    # Initialize with contour if available
                    if i < len(merged_contours) and merged_contours[i] is not None:
                        # ใช้ Process FPS ก่อน (FPS จริงของการประมวลผล) - สำคัญกว่า Input FPS
                        current_fps_value = fps if 'fps' in locals() else None
                        # ถ้ายังไม่มี ใช้ Input FPS (fallback)
                        if current_fps_value is None:
                            current_fps_value = getattr(cam, 'fps', None) if 'cam' in locals() else None
                        new_tracker.update(new_rect, contour=merged_contours[i],
                                         current_fps=current_fps_value, thresholds=resolution_thresholds)
                    new_trackers[global_track_id] = new_tracker
                    global_track_id += 1

            # c. Clean up: remove trackers missed for too long
            active_trackers = {k: v for k, v in new_trackers.items() if v.missed_frames <= MAX_MISS_FRAMES}

        # --- 6. DRAW RESULTS ---
        # เก็บ ROI boxes ทั้งหมดก่อนวาดเพื่อตรวจสอบ overlap
        roi_boxes_to_draw = []  # [(roi_coords, tracker, is_waiting), ...]

        # เก็บ motion_boxes สำหรับใช้ใน dense motion check (ถ้ามี)
        current_motion_boxes_for_check = []
        if tracking_mode == 'hybrid' and 'hybrid_tracker' in locals() and hybrid_tracker is not None:
            if hasattr(hybrid_tracker, 'motion_boxes') and hybrid_tracker.motion_boxes:
                current_motion_boxes_for_check = hybrid_tracker.motion_boxes

        for tracker in active_trackers.values():
            x, y, w_box, h_box = tracker.current_rect

            # --- Enhanced classification (rule-based with path quality) ---
            def classify_tracker(tr, current_frame_idx):
                # Early exit checks (ประหยัด CPU)
                if tr.path_frames < DRONE_MIN_PATH_FRAMES:
                    return "UNKNOWN", 0.0

                area = tr.current_rect[2] * tr.current_rect[3]
                v = tr.velocity_mag

                # Early exit: ถ้า area หรือ velocity ไม่อยู่ในช่วง
                if not (DRONE_MIN_AREA <= area <= DRONE_MAX_AREA):
                    return "UNKNOWN", 0.0
                if not (DRONE_MIN_SPEED <= v <= DRONE_MAX_SPEED):
                    return "UNKNOWN", 0.0

                # Cached calculation: ตรวจสอบ cache ก่อน
                if (tr._cache_frame >= 0 and
                    current_frame_idx - tr._cache_frame < CLASSIFICATION_UPDATE_INTERVAL and
                    tr._path_quality_cache is not None):
                    return tr._path_quality_cache

                # Base score (เร็วมาก)
                is_drone_area = DRONE_MIN_AREA <= area <= DRONE_MAX_AREA
                is_drone_speed = DRONE_MIN_SPEED <= v <= DRONE_MAX_SPEED
                has_path = tr.path_frames >= DRONE_MIN_PATH_FRAMES
                base_score = (
                    (DRONE_AREA_WEIGHT if is_drone_area else 0) +
                    (DRONE_SPEED_WEIGHT if is_drone_speed else 0) +
                    (DRONE_PATH_WEIGHT if has_path else 0)
                )

                # Early exit: ถ้า base score ต่ำเกินไป
                if base_score < 0.3:
                    result = ("UNKNOWN", base_score * 0.3)
                    tr._path_quality_cache = result
                    tr._cache_frame = current_frame_idx
                    return result

                # Conditional Path Quality Calculation (คำนวณเฉพาะเมื่อจำเป็น)
                # คำนวณ path quality เฉพาะเมื่อ base score >= 0.5 และ status เป็น YELLOW, ORANGE หรือ RED
                final_score = base_score
                if base_score >= 0.5 and (tr.status == 'YELLOW' or tr.status == 'ORANGE' or tr.status == 'RED'):
                    # คำนวณ path quality metrics (ใช้ processing_fps ถ้ามี)
                    path_straightness = tr.calculate_path_straightness()
                    processing_fps = fps if 'fps' in locals() else None
                    velocity_cv = tr.calculate_velocity_consistency(processing_fps=processing_fps)
                    can_hover, hover_ratio = tr.can_hover()
                    path_smoothness = tr.calculate_path_smoothness()

                    # เพิ่มคะแนนจาก path quality
                    if can_hover:
                        final_score += 0.3

                    if DRONE_MIN_STRAIGHTNESS <= path_straightness <= DRONE_MAX_STRAIGHTNESS:
                        final_score += 0.2
                    elif path_straightness > DRONE_MAX_STRAIGHTNESS:
                        final_score -= 0.2  # ลดคะแนนถ้าเกิน (อาจเป็นเครื่องบิน)

                    if DRONE_MIN_VELOCITY_CV <= velocity_cv <= DRONE_MAX_VELOCITY_CV:
                        final_score += 0.2
                    elif velocity_cv > DRONE_MAX_VELOCITY_CV:
                        final_score -= 0.2  # ลดคะแนนถ้าไม่สม่ำเสมอ

                    if path_smoothness >= DRONE_MIN_SMOOTHNESS:
                        final_score += 0.1

                    # ลดคะแนนถ้าไม่สามารถ hover และ velocity ต่ำ
                    if not can_hover and v < 5.0:
                        final_score -= 0.2

                # Clamp score
                final_score = min(1.0, max(0.0, final_score))

                # Determine object type
                if final_score >= DRONE_MIN_SCORE:
                    obj_type = "DRONE"
                    conf = final_score
                else:
                    obj_type = "UNKNOWN"
                    conf = final_score * 0.3

                result = (obj_type, conf)

                # Cache result
                tr._path_quality_cache = result
                tr._cache_frame = current_frame_idx

                return result

            if ENABLE_OBJECT_CLASSIFICATION:
                obj_type, conf = classify_tracker(tracker, frame_counter)
                tracker.object_type = obj_type
                tracker.classification_confidence = conf
                # Update status based on classification
                tracker.check_status()

            # ตรวจสอบแมลงและ blacklist ก่อนวาดกล่องสีแดง
            is_insect = tracker.is_likely_insect() if hasattr(tracker, 'is_likely_insect') else False
            is_blacklisted = False

            # ตรวจสอบ blacklist (ถ้าไม่ใช่แมลง)
            if not is_insect:
                # ใช้ center และ rect สำหรับตรวจสอบ blacklist
                center = tracker.current_center_of_mass if hasattr(tracker, 'current_center_of_mass') else None
                if center is None:
                    center = (tracker.current_rect[0] + tracker.current_rect[2] // 2,
                             tracker.current_rect[1] + tracker.current_rect[3] // 2)
                rect = tracker.current_rect
                # ตรวจสอบ blacklist (ต้องมี hybrid_tracker หรือใช้ global blacklist)
                # สำหรับ multi mode เราไม่มี hybrid_tracker instance โดยตรง
                # ดังนั้นเราจะข้ามการตรวจสอบ blacklist ใน multi mode (จะตรวจใน hybrid mode เท่านั้น)
                # หรือถ้ามี global blacklist system ก็ใช้ที่นี่

            # ถ้าเป็นแมลงหรืออยู่ใน blacklist → ไม่ให้เป็น RED
            effective_status = tracker.status
            if tracker.status == 'RED' and (is_insect or is_blacklisted):
                effective_status = 'GREEN'
                if DEBUG_MODE:
                    reason = "insect" if is_insect else "blacklist"
                    print(f"🚫 Tracker {tracker.id}: Blocked RED drawing - {reason}")

            # Determine draw color/flags based on effective_status
            if effective_status == 'RED':
                draw_color = RED_COLOR  # สีแดง
                should_draw_box = DRAW_RED_BOX
                should_draw_path = DRAW_PATH and DRAW_RED_PATH
            elif effective_status == 'ORANGE':
                draw_color = ORANGE_COLOR  # สีส้ม
                should_draw_box = DRAW_ORANGE_BOX
                should_draw_path = DRAW_PATH and DRAW_ORANGE_PATH
            elif effective_status == 'YELLOW':
                draw_color = ALERT_COLOR  # สีเหลือง
                should_draw_box = DRAW_YELLOW_BOX
                should_draw_path = DRAW_PATH and DRAW_YELLOW_PATH
            else:  # GREEN status
                draw_color = NORMAL_COLOR  # สีเขียว
                should_draw_box = DRAW_GREEN_BOX
                should_draw_path = DRAW_PATH and DRAW_GREEN_PATH

            # --- Draw Path (if enabled) ---
            if should_draw_path and PATH_VISUALIZATION_ENABLED and tracker.center_of_mass_history:
                path_points = tracker.center_of_mass_history[-PATH_MAX_POINTS:]
                if len(path_points) >= 2:
                    # Smooth path points ก่อนวาด
                    smoothed_path_points = _smooth_path_points_for_tracker(path_points)

                    # วาดเส้น path (ใช้ smoothed points)
                    pts = np.array(smoothed_path_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], False, draw_color, 2)

            # --- Draw Heat Point (if enabled) ---
            if should_draw_path and DRAW_HEAT_POINT and HEAT_POINT_ENABLED:
                center_of_mass = tracker.current_center_of_mass
                if center_of_mass is not None:
                    # ตรวจสอบว่า center_of_mass เป็น tuple (int, int)
                    if isinstance(center_of_mass, (tuple, list)) and len(center_of_mass) >= 2:
                        try:
                            x, y = int(center_of_mass[0]), int(center_of_mass[1])
                            if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                                cv2.circle(display_frame, (x, y), HEAT_POINT_RADIUS, draw_color, -1)
                        except (ValueError, TypeError, IndexError):
                            pass  # ข้ามการวาดถ้าไม่ valid

            # --- Draw Bounding Box (if enabled for this status) ---
            if should_draw_box:
                # เพิ่ม padding ให้ bbox ห่างจากวัตถุ (ใช้ resolution-dependent threshold)
                padding = resolution_thresholds.get('BBOX_PADDING', BBOX_PADDING)
                bbox_x1 = max(0, x - padding)
                bbox_y1 = max(0, y - padding)
                bbox_x2 = min(display_frame.shape[1], x + w_box + padding)
                bbox_y2 = min(display_frame.shape[0], y + h_box + padding)
                cv2.rectangle(display_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), draw_color, BBOX_THICKNESS)

            # --- Collect ROI Box for ORANGE/RED trackers (เก็บไว้ก่อนวาด) ---
            if tracker.status == 'ORANGE' or tracker.status == 'RED':
                # Calculate ROI box with padding (คงที่ตาม resolution - ไม่ใช้ predicted ROI)
                h_frame, w_frame = display_frame.shape[:2]
                # ใช้ current rect เสมอ (ไม่ใช้ predicted ROI)
                roi_x1 = max(0, x - HYBRID_ROI_PADDING)
                roi_y1 = max(0, y - HYBRID_ROI_PADDING)
                roi_x2 = min(w_frame, x + w_box + HYBRID_ROI_PADDING)
                roi_y2 = min(h_frame, y + h_box + HYBRID_ROI_PADDING)
                roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

                # ตรวจสอบว่า ROI area มี motion ยุ่งเหยิงหรือไม่ (กรอง ROI สีขาว)
                # ใช้ motion_boxes ที่เก็บไว้ก่อนหน้า
                has_dense_motion = _check_roi_area_has_dense_motion(
                    roi_box, current_motion_boxes_for_check, w_frame, h_frame
                )

                if not has_dense_motion:
                    # ตรวจสอบ ROI validation (ถ้าเปิดใช้งานและเป็น hybrid mode)
                    is_valid_roi = True
                    validation_reason = "Validation disabled or not hybrid mode"

                    if tracking_mode == 'hybrid' and hybrid_tracker is not None:
                        # หา target ที่ตรงกับ tracker
                        target_id = None
                        for tid, tgt in hybrid_tracker.targets.items():
                            if 'tracker_id' in tgt and tgt['tracker_id'] == tracker.id:
                                target_id = tid
                                break

                        if target_id is not None:
                            target = hybrid_tracker.targets[target_id]

                            # ตรวจสอบ validation
                            is_valid_roi, validation_score, validation_reason = hybrid_tracker._validate_roi_for_drawing(
                                target, roi_box, frame_counter
                            )

                            if is_valid_roi:
                                # ผ่าน validation → อัปเดต ROI history
                                roi_center = ((roi_x1 + roi_x2) // 2, (roi_y1 + roi_y2) // 2)
                                if 'roi_box_history' not in target:
                                    target['roi_box_history'] = []
                                target['roi_box_history'].append((frame_counter, roi_box, roi_center))

                                # จำกัดจำนวน history
                                from config import ROI_HISTORY_MAX_FRAMES
                                if len(target['roi_box_history']) > ROI_HISTORY_MAX_FRAMES:
                                    target['roi_box_history'] = target['roi_box_history'][-ROI_HISTORY_MAX_FRAMES:]

                    if is_valid_roi:
                        # คำนวณ priority score สำหรับ ROI
                        processing_fps = None
                        try:
                            if tracking_mode == 'hybrid' and hybrid_tracker is not None:
                                processing_fps = getattr(hybrid_tracker, '_current_processing_fps', None)
                        except (NameError, AttributeError):
                            pass
                        priority_score = _calculate_roi_priority_score(tracker, frame_counter, processing_fps)
                        roi_boxes_to_draw.append(((roi_x1, roi_y1, roi_x2, roi_y2), tracker, False, priority_score))
                    else:
                        # Skip ROI นี้ (ไม่วาด)
                        if DEBUG_MODE:
                            print(f"⚠️ ROI validation failed for tracker {tracker.id}: {validation_reason}")

            # --- Draw Labels (if enabled) ---
            if DRAW_LABELS:
                label = f"ID:{tracker.id} | {tracker.status}"
                if ENABLE_OBJECT_CLASSIFICATION and tracker.object_type != 'UNKNOWN':
                    label = f"{tracker.object_type} ({tracker.classification_confidence:.0%}) | {label}"
                cv2.putText(display_frame,
                            f"{label} | Path:{tracker.path_frames}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

        # --- Add ROI boxes from footprints (interval-based update) ---
        from config import (
            FOOTPRINTS_ROI_ENABLED, FOOTPRINTS_ROI_UPDATE_INTERVAL,
            FOOTPRINTS_MODULE_ENABLED
        )

        if FOOTPRINTS_ROI_ENABLED and FOOTPRINTS_MODULE_ENABLED:
            # หา footprints_drawer จาก hybrid_tracker หรือ global variable
            footprints_drawer = None
            if tracking_mode == 'hybrid' and 'hybrid_tracker' in locals() and hybrid_tracker is not None:
                if hasattr(hybrid_tracker, 'footprints_drawer'):
                    footprints_drawer = hybrid_tracker.footprints_drawer

            if footprints_drawer is not None:
                # Interval-based update: อัปเดตทุก N เฟรม
                if not hasattr(main, '_footprints_roi_frame_counter'):
                    main._footprints_roi_frame_counter = 0
                    main._cached_footprints_roi = []

                main._footprints_roi_frame_counter += 1

                if main._footprints_roi_frame_counter >= FOOTPRINTS_ROI_UPDATE_INTERVAL:
                    main._footprints_roi_frame_counter = 0

                    # เรียก calculate_roi_from_footprints()
                    footprints_roi_list = calculate_roi_from_footprints(
                        footprints_drawer, w, h, resolution_thresholds
                    )
                    main._cached_footprints_roi = footprints_roi_list

                # ใช้ cached ROI (หรือ ROI ที่เพิ่งคำนวณ)
                # ดึง ROI tracker
                roi_tracker = get_footprints_roi_tracker()

                # อัปเดต ROI tracking และตรวจสอบ swarming (interval-based)
                from config import (
                    FOOTPRINTS_ROI_SWARMING_CHECK_INTERVAL,
                    FOOTPRINTS_ROI_MAX_FOCUS_YOLO_ROI
                )

                # เก็บ swarming check counter
                if not hasattr(main, '_swarming_check_counter'):
                    main._swarming_check_counter = 0

                main._swarming_check_counter += 1
                should_check_swarming = (main._swarming_check_counter >= FOOTPRINTS_ROI_SWARMING_CHECK_INTERVAL)

                if should_check_swarming:
                    main._swarming_check_counter = 0

                processed_footprints_roi = []
                focus_roi_count = 0  # นับจำนวน ROI ที่ทำ YOLO detection

                for roi_box, priority, roi_type, score in main._cached_footprints_roi:
                    # อัปเดต ROI tracking
                    roi_id = roi_tracker.update_roi(roi_box, frame_counter)

                    # ตรวจจับ swarming (interval-based)
                    if should_check_swarming:
                        is_swarming, movement_score, merge_count = detect_roi_swarming(
                            roi_tracker, roi_id, w, h, resolution_thresholds
                        )
                        # เก็บค่า swarming สำหรับใช้ในเฟรมถัดไป
                        tracking = roi_tracker.get_roi_tracking(roi_id)
                        if tracking:
                            tracking['last_swarming'] = is_swarming
                            tracking['movement_score'] = movement_score
                    else:
                        # ใช้ค่าเก่า (ไม่ต้องคำนวณใหม่)
                        tracking = roi_tracker.get_roi_tracking(roi_id)
                        if tracking:
                            is_swarming = tracking.get('last_swarming', False)
                            movement_score = tracking.get('movement_score', 0.0)
                            merge_count = tracking.get('merge_count', 0)
                        else:
                            is_swarming, movement_score, merge_count = False, 0.0, 0

                    # ตรวจสอบ YOLO detection (ถ้ามี yolo_model และยังไม่เกิน limit)
                    has_yolo = False
                    if ('yolo_model' in locals() and yolo_model is not None and
                        focus_roi_count < FOOTPRINTS_ROI_MAX_FOCUS_YOLO_ROI):
                        # ตรวจสอบว่า ROI อยู่ใน FOCUS status หรือไม่
                        tracking = roi_tracker.get_roi_tracking(roi_id)
                        if tracking and tracking.get('status') == 'FOCUS':
                            # อัปเดต focus mode
                            has_yolo = update_focus_mode_roi(
                                roi_tracker, roi_id, yolo_model, frame, w, h, frame_counter
                            )
                            if has_yolo:
                                focus_roi_count += 1

                    # อัปเดตพลัง
                    power = update_roi_power(
                        roi_tracker, roi_id, is_swarming, movement_score, merge_count, has_yolo
                    )

                    # อัปเดต status
                    status = update_roi_status(roi_tracker, roi_id)

                    # เก็บ ROI พร้อม tracking data
                    tracking = roi_tracker.get_roi_tracking(roi_id)
                    processed_footprints_roi.append((
                        roi_box, priority, roi_type, score, roi_id, status, power, tracking
                    ))

                # Cleanup old ROIs
                roi_tracker.cleanup_old_rois(frame_counter)

                # ROI Limiting: ถ้ามี YELLOW/FOCUS ROI → จำกัดให้เหลือเฉพาะ ROI เหล่านั้น
                from config import (
                    FOOTPRINTS_ROI_LIMIT_TO_FOCUS_ENABLED,
                    FOOTPRINTS_ROI_MAX_FOCUS_ROI
                )

                if FOOTPRINTS_ROI_LIMIT_TO_FOCUS_ENABLED:
                    # แยก ROI ตาม status
                    focus_rois = [r for r in processed_footprints_roi if r[5] in ('YELLOW', 'FOCUS')]
                    other_rois = [r for r in processed_footprints_roi if r[5] == 'GREEN']

                    if focus_rois:
                        # เรียงตาม power (สูงสุดก่อน)
                        focus_rois.sort(key=lambda x: x[6], reverse=True)
                        # จำกัดจำนวน focus ROI
                        focus_rois = focus_rois[:FOOTPRINTS_ROI_MAX_FOCUS_ROI]
                        # ใช้เฉพาะ focus ROI
                        processed_footprints_roi = focus_rois

                # สร้าง dummy tracker objects และเพิ่มเข้า roi_boxes_to_draw
                for roi_box, priority, roi_type, score, roi_id, status, power, tracking in processed_footprints_roi:
                    class DummyTracker:
                        def __init__(self, prio, roi_t, sc, st, pwr, trk):
                            self.id = -1  # ใช้ ID -1 สำหรับ footprints ROI
                            self.status = st  # ใช้ status จาก tracking
                            self.priority = prio
                            self.roi_type = roi_t
                            self.score = sc
                            self.power = pwr
                            self.tracking = trk

                    dummy_tracker = DummyTracker(priority, roi_type, score, status, power, tracking)
                    # เพิ่ม ROI จาก footprints เข้าไปใน roi_boxes_to_draw
                    # Format: (roi_coords, tracker, is_waiting, priority_score)
                    roi_boxes_to_draw.append((roi_box, dummy_tracker, False, score))

        # --- Filter overlapping ROI boxes (ซ้ำกันมากกว่า 80% ให้เหลืออันเดียว) - OPTIMIZED ---
        def calculate_overlap_ratio(roi1, roi2):
            """คำนวณ overlap ratio ระหว่าง 2 ROI boxes (0.0-1.0) - OPTIMIZED with early exit"""
            x1_1, y1_1, x2_1, y2_1 = roi1
            x1_2, y1_2, x2_2, y2_2 = roi2

            # ⚠️ PERFORMANCE: Early exit - ตรวจสอบว่าไม่ overlap เลยก่อน
            if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
                return 0.0  # ไม่มี overlap เลย

            # คำนวณ intersection
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0  # ไม่มี overlap

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            # ใช้ area ที่เล็กกว่าเป็นตัวหาร (เพื่อให้ได้ overlap ratio ที่ถูกต้อง)
            min_area = min(area1, area2)
            if min_area == 0:
                return 0.0

            return inter_area / min_area

        # --- Filter/Merge overlapping ROI boxes (รองรับ footprints ROI priority) ---
        filtered_roi_boxes = []
        status_priority = {'RED': 2, 'ORANGE': 1, 'YELLOW': 0, 'GREEN': 0}

        from config import (
            MAX_ROI_DRAW_LIMIT,
            FOOTPRINTS_ROI_OVERLAP_IOU_THRESHOLD,
            FOOTPRINTS_ROI_OVERLAP_MERGE_ENABLED,
            FOOTPRINTS_ROI_OVERLAP_MERGE_IOU_THRESHOLD
        )

        # เรียงตาม priority: footprints ROI (priority 1, 2, 3) > tracker ROI
        # สำหรับ footprints ROI: ใช้ priority (1=YOLO, 2=motion group, 3=isolated) และ score
        # สำหรับ tracker ROI: ใช้ priority_score และ status priority
        def get_sort_key(roi_item):
            if len(roi_item) >= 4:
                roi_coords, tracker, is_waiting, priority_score = roi_item
            else:
                roi_coords, tracker, is_waiting = roi_item
                priority_score = status_priority.get(tracker.status, 0) / 10.0

            # ตรวจสอบว่าเป็น footprints ROI (tracker.id == -1)
            if hasattr(tracker, 'id') and tracker.id == -1:
                # Footprints ROI: ใช้ priority (1, 2, 3) และ score
                footprints_priority = getattr(tracker, 'priority', 3)
                footprints_score = getattr(tracker, 'score', 0.0)
                # Priority ต่ำ = สูงกว่า (1 > 2 > 3), score สูง = สูงกว่า
                return (0, -footprints_priority, footprints_score)  # 0 = footprints ROI (สูงสุด)
            else:
                # Tracker ROI: ใช้ priority_score และ status
                return (1, priority_score, status_priority.get(tracker.status, 0))  # 1 = tracker ROI

        roi_boxes_sorted = sorted(roi_boxes_to_draw, key=get_sort_key, reverse=False)

        # จำกัดจำนวน ROI ที่วาด (MAX_ROI_DRAW_LIMIT)
        roi_boxes_sorted = roi_boxes_sorted[:MAX_ROI_DRAW_LIMIT]

        def get_roi_priority_value(tracker):
            """คืนค่า priority value สำหรับเปรียบเทียบ (ต่ำกว่า = สูงกว่า)"""
            if hasattr(tracker, 'id') and tracker.id == -1:
                # Footprints ROI: ใช้ priority (1, 2, 3)
                return getattr(tracker, 'priority', 3)
            else:
                # Tracker ROI: ใช้ status priority (แปลงเป็น priority value ที่ต่ำกว่า = สูงกว่า)
                # RED=2, ORANGE=1, YELLOW=0, GREEN=0 → แปลงเป็น 0, 1, 2, 2 (ต่ำกว่า = สูงกว่า)
                status_val = status_priority.get(tracker.status, 0)
                return 3 - status_val  # Invert: RED(2) → 1, ORANGE(1) → 2, YELLOW(0) → 3

        def merge_roi_boxes(roi1, roi2):
            """รวม ROI boxes 2 อันเป็นอันเดียว"""
            x1_1, y1_1, x2_1, y2_1 = roi1
            x1_2, y1_2, x2_2, y2_2 = roi2
            return (
                min(x1_1, x1_2),
                min(y1_1, y1_2),
                max(x2_1, x2_2),
                max(y2_1, y2_2)
            )

        # Process ROI boxes with improved merge/filter logic
        processed_roi_boxes = []  # [(roi_coords, tracker, is_waiting), ...]

        for i, roi_item in enumerate(roi_boxes_sorted):
            # Handle both old format (3 elements) and new format (4 elements)
            if len(roi_item) >= 4:
                roi_coords, tracker, is_waiting, priority_score = roi_item
            else:
                roi_coords, tracker, is_waiting = roi_item

            should_keep = True
            current_roi = roi_coords
            current_priority = get_roi_priority_value(tracker)

            # ตรวจสอบ overlap กับ ROI ที่ผ่านมาแล้ว (รวมถึง ROI ที่ merge แล้ว)
            check_limit = min(len(processed_roi_boxes), 20)  # เพิ่ม limit เป็น 20 เพื่อรองรับหลาย ROI
            merged_with_indices = []  # เก็บ index ของ ROI ที่ merge เข้า current

            for j in range(max(0, len(processed_roi_boxes) - check_limit), len(processed_roi_boxes)):
                other_roi_coords, other_tracker, other_is_waiting = processed_roi_boxes[j]

                overlap_ratio = calculate_overlap_ratio(current_roi, other_roi_coords)

                if overlap_ratio > FOOTPRINTS_ROI_OVERLAP_IOU_THRESHOLD:
                    # มี overlap มาก
                    other_priority = get_roi_priority_value(other_tracker)

                    if FOOTPRINTS_ROI_OVERLAP_MERGE_ENABLED and overlap_ratio > FOOTPRINTS_ROI_OVERLAP_MERGE_IOU_THRESHOLD:
                        # Merge mode: รวม ROI ที่ overlap กัน
                        if current_priority < other_priority:
                            # Current ROI มี priority สูงกว่า → merge other เข้า current
                            current_roi = merge_roi_boxes(current_roi, other_roi_coords)
                            merged_with_indices.append(j)
                        elif current_priority == other_priority:
                            # Priority เท่ากัน → merge other เข้า current
                            current_roi = merge_roi_boxes(current_roi, other_roi_coords)
                            merged_with_indices.append(j)
                        else:
                            # Other ROI มี priority สูงกว่า → skip current (จะใช้ other แทน)
                            should_keep = False
                            break
                    else:
                        # Filter mode: กรอง ROI ที่ overlap (เก็บอันที่มี priority สูงกว่า)
                        if other_priority < current_priority:
                            # Other ROI มี priority สูงกว่า → ไม่ต้องวาด current
                            should_keep = False
                            break
                        elif other_priority == current_priority:
                            # Priority เท่ากัน → เลือก tracker ID ที่ต่ำกว่า
                            if hasattr(other_tracker, 'id') and hasattr(tracker, 'id'):
                                if other_tracker.id < tracker.id:
                                    should_keep = False
                                    break

            if should_keep:
                # อัปเดต merge count สำหรับ footprints ROI (ถ้า merge)
                if hasattr(tracker, 'id') and tracker.id == -1 and merged_with_indices:
                    # หา ROI ID ของ current ROI
                    if hasattr(tracker, 'tracking') and tracker.tracking:
                        roi_tracker = get_footprints_roi_tracker()
                        for rid, trk in roi_tracker.roi_tracking.items():
                            if trk == tracker.tracking:
                                # เพิ่ม merge count
                                roi_tracker.increment_merge_count(rid, len(merged_with_indices))
                                break

                # ลบ ROI ที่ merge เข้า current ออกจาก processed_roi_boxes
                # เรียง indices จากมากไปน้อยเพื่อลบจากหลังไปหน้า (ไม่กระทบ index)
                for idx in sorted(merged_with_indices, reverse=True):
                    if 0 <= idx < len(processed_roi_boxes):
                        processed_roi_boxes.pop(idx)

                # ตรวจสอบ overlap ซ้ำกับ ROI ที่เหลืออยู่ (หลังจาก merge แล้ว ROI อาจใหญ่ขึ้น)
                # เพื่อให้แน่ใจว่า ROI ที่ merge แล้วไม่ overlap กับ ROI อื่นอีก
                if FOOTPRINTS_ROI_OVERLAP_MERGE_ENABLED:
                    final_check_limit = min(len(processed_roi_boxes), 10)
                    for k in range(max(0, len(processed_roi_boxes) - final_check_limit), len(processed_roi_boxes)):
                        other_roi_coords, other_tracker, _ = processed_roi_boxes[k]
                        overlap_ratio = calculate_overlap_ratio(current_roi, other_roi_coords)

                        if overlap_ratio > FOOTPRINTS_ROI_OVERLAP_MERGE_IOU_THRESHOLD:
                            # ยังมี overlap มาก → merge อีกครั้ง
                            other_priority = get_roi_priority_value(other_tracker)
                            if current_priority <= other_priority:
                                # Current มี priority สูงกว่าหรือเท่ากัน → merge other เข้า current
                                current_roi = merge_roi_boxes(current_roi, other_roi_coords)
                                # ลบ other ออก
                                processed_roi_boxes.pop(k)
                                break

                # เพิ่ม current ROI (ที่อาจ merge แล้ว) เข้าไป
                processed_roi_boxes.append((current_roi, tracker, is_waiting))

        # ตรวจสอบ overlap ซ้ำอีกครั้งหลังจาก merge ทั้งหมด (เพื่อให้แน่ใจว่าไม่มี overlap มาก)
        filtered_roi_boxes = []
        for i, (roi_coords, tracker, is_waiting) in enumerate(processed_roi_boxes):
            should_keep = True

            # ตรวจสอบ overlap กับ ROI ที่ผ่านมาแล้ว
            check_limit = min(i, 10)
            for j in range(max(0, i - check_limit), i):
                other_roi_coords, other_tracker, _ = processed_roi_boxes[j]

                overlap_ratio = calculate_overlap_ratio(roi_coords, other_roi_coords)

                if overlap_ratio > FOOTPRINTS_ROI_OVERLAP_IOU_THRESHOLD:
                    # ยังมี overlap มาก → กรองออก (เก็บอันแรก)
                    current_priority = get_roi_priority_value(tracker)
                    other_priority = get_roi_priority_value(other_tracker)

                    if other_priority < current_priority:
                        # Other ROI มี priority สูงกว่า
                        should_keep = False
                        break
                    elif other_priority == current_priority:
                        # Priority เท่ากัน → เลือก tracker ID ที่ต่ำกว่า
                        if hasattr(other_tracker, 'id') and hasattr(tracker, 'id'):
                            if other_tracker.id < tracker.id:
                                should_keep = False
                                break

            if should_keep:
                filtered_roi_boxes.append((roi_coords, tracker, is_waiting))

        # --- Draw filtered ROI boxes ---
        h_frame, w_frame = display_frame.shape[:2]
        from config import ALERT_COLOR, YELLOW_COLOR, ORANGE_COLOR, RED_COLOR, NORMAL_COLOR

        for roi_coords, tracker, is_waiting in filtered_roi_boxes:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords

            # กำหนดสีตาม status (สำหรับ footprints ROI)
            if hasattr(tracker, 'id') and tracker.id == -1:
                # Footprints ROI: ใช้ status จาก tracking
                if hasattr(tracker, 'status'):
                    if tracker.status == 'FOCUS':
                        roi_color = RED_COLOR  # สีแดงสำหรับ FOCUS
                        roi_label = f"FOCUS ROI (P:{getattr(tracker, 'power', 0):.1f})"
                    elif tracker.status == 'YELLOW':
                        roi_color = YELLOW_COLOR  # สีเหลืองสำหรับ YELLOW
                        roi_label = f"SWARMING ROI (P:{getattr(tracker, 'power', 0):.1f})"
                    else:
                        roi_color = NORMAL_COLOR  # สีเขียวสำหรับ GREEN
                        roi_label = f"FOOTPRINTS ROI (P:{getattr(tracker, 'power', 0):.1f})"
                else:
                    roi_color = NORMAL_COLOR
                    roi_label = "FOOTPRINTS ROI"
            else:
                # Tracker ROI: ใช้สีขาวตามเดิม
                roi_color = (255, 255, 255)
                roi_label = "FOCUS ROI" if not is_waiting else f"FOCUS ROI (WAIT {tracker.missed_frames})"

            cv2.rectangle(display_frame, (max(0, roi_x1), max(0, roi_y1)),
                           (min(w_frame, roi_x2), min(h_frame, roi_y2)), roi_color, 2)
            cv2.putText(display_frame, roi_label, (max(0, roi_x1), max(0, roi_y1)-5),
                       0, 0.5, roi_color, 2)

            # วาด best target ถ้ามี (สำหรับ FOCUS ROI)
            if hasattr(tracker, 'id') and tracker.id == -1:
                if hasattr(tracker, 'tracking') and tracker.tracking:
                    best_target = tracker.tracking.get('best_target')
                    if best_target:
                        conf, (tx, ty, tw, th) = best_target
                        # วาด target box (สีเขียว)
                        cv2.rectangle(display_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Target (conf:{conf:.2f})", (tx, ty - 5),
                                   0, 0.5, (0, 255, 0), 1)

        t_end = cv2.getTickCount()
        algo_time = (t_end - t_start) / cv2.getTickFrequency() * 1000

        # คำนวณ display FPS (ใช้เวลาจากเฟรมก่อนหน้า)
        curr_display_time = time.time()
        if frame_counter > 1:
            display_fps = 1 / (curr_display_time - prev_display_time + 1e-6)
        prev_display_time = curr_display_time

        # อัปเดต adaptive HYBRID_MIN_MOTION_AREA (ถ้าเปิดใช้งาน)
        if ADAPTIVE_HYBRID_MIN_AREA_ENABLED and 'adaptive_hybrid_min_area_manager' in locals() and adaptive_hybrid_min_area_manager is not None:
            # นับจำนวน motion boxes จาก hybrid_tracker
            motion_box_count = 0
            if hybrid_tracker is not None and hasattr(hybrid_tracker, 'motion_boxes'):
                motion_box_count = len(hybrid_tracker.motion_boxes)

            # อัปเดตและรับค่า min_area ใหม่
            current_hybrid_min_motion_area = adaptive_hybrid_min_area_manager.update(
                motion_box_count, fps, algo_time, frame_counter
            )

            # อัปเดต HYBRID_MIN_MOTION_AREA global variable (ใช้ใน find_motion_boxes)
            HYBRID_MIN_MOTION_AREA = current_hybrid_min_motion_area
        elif 'current_hybrid_min_motion_area' not in locals():
            current_hybrid_min_motion_area = HYBRID_MIN_MOTION_AREA

        monitor.update()

        # --- PTZ Verification: ตรวจสอบและลบ target ที่อยู่ใน blacklist area ---
        if 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active and hybrid_tracker is not None:
            # ดึง blacklist information
            blacklist_info = ptz_verification.get_blacklist_info()

            if blacklist_info:
                # ตรวจสอบทุก target ว่าอยู่ใน blacklist area หรือไม่
                targets_to_remove = []

                for target_id, target in hybrid_tracker.targets.items():
                    roi_x1, roi_y1, roi_x2, roi_y2 = target.get('roi_box', [0, 0, 0, 0])
                    roi_center_x = (roi_x1 + roi_x2) // 2
                    roi_center_y = (roi_y1 + roi_y2) // 2

                    # ตรวจสอบว่าอยู่ใน blacklist area หรือไม่
                    for bl_x, bl_y, bl_expiry in blacklist_info:
                        dist = ((roi_center_x - bl_x)**2 + (roi_center_y - bl_y)**2)**0.5
                        if dist < CAM2_PTZ_BLACKLIST_RADIUS:
                            # Target อยู่ใน blacklist area → ลบออก
                            targets_to_remove.append(target_id)

                            if DEBUG_MODE:
                                remaining_time = bl_expiry - time.time()
                                print(f"🚫 Blacklist: Removing target {target_id} (center=({roi_center_x}, {roi_center_y}), dist={dist:.1f}px < {CAM2_PTZ_BLACKLIST_RADIUS}px, remaining={remaining_time:.1f}s)")
                            break

                # ลบ targets ที่อยู่ใน blacklist
                for target_id in targets_to_remove:
                    if target_id in hybrid_tracker.targets:
                        del hybrid_tracker.targets[target_id]
                        if DEBUG_MODE:
                            print(f"✅ Blacklist: Target {target_id} removed from hybrid_tracker")

        # --- PTZ Verification: อัปเดต frame info และ YOLO status ของ cam1 (non-blocking) ---
        cam1_yolo_active = False
        if CAM2_PTZ_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active:
            try:
                # ตรวจสอบว่า cam1 กำลังใช้ YOLO หรือไม่
                # 1. Full-frame YOLO
                if yolo_model is not None and frame_counter % FULL_FRAME_YOLO_INTERVAL == 0:
                    cam1_yolo_active = True

                # 2. ROI YOLO จาก hybrid_tracker
                if hybrid_tracker is not None:
                    for target_id, target in hybrid_tracker.targets.items():
                        if target.get('target_locked', False):
                            # ตรวจสอบว่าควรเรียก YOLO ในเฟรมนี้หรือไม่
                            yolo_frame_counter = target.get('yolo_frame_counter', 0)
                            yolo_interval = HYBRID_YOLO_INTERVAL
                            if target.get('lock_mode', False):
                                from config import LOCK_YOLO_INTERVAL
                                yolo_interval = LOCK_YOLO_INTERVAL

                            if yolo_frame_counter >= yolo_interval:
                                cam1_yolo_active = True
                                break

                # ส่งข้อมูลไปยัง PTZ verification (non-blocking)
                ptz_verification.update_cam1_frame_info(frame_counter, cam1_yolo_active)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ PTZ Verification: Error updating cam1 frame info (non-critical): {e}")

        # --- PTZ Verification: ตรวจสอบ ROI ที่สงสัย ---
        if CAM2_PTZ_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active and hybrid_tracker is not None:
            try:
                # ตรวจสอบ ROI ที่สงสัย (conf < HYBRID_BASE_CONF แต่กำลัง tracking)
                for target_id, target in hybrid_tracker.targets.items():
                    # Read-only access - ไม่แก้ไข target
                    original_conf = target.get('original_confidence', target.get('confidence', 0.0))

                    # ส่งไปตรวจสอบเฉพาะ target ที่:
                    # 1. กำลัง tracking (target_locked = True)
                    # 2. confidence ยังไม่ถึง threshold (< HYBRID_BASE_CONF)
                    # 3. confidence สูงพอที่จะไม่ใช่ noise (> 0.1)
                    if (target.get('target_locked', False) and
                        original_conf < HYBRID_BASE_CONF and
                        original_conf > 0.1):

                        roi_x1, roi_y1, roi_x2, roi_y2 = target.get('roi_box', (0, 0, 0, 0))
                        roi_center_x = (roi_x1 + roi_x2) // 2
                        roi_center_y = (roi_y1 + roi_y2) // 2

                        # เพิ่มเข้า queue สำหรับตรวจสอบ (non-blocking)
                        ptz_verification.add_verification_request(
                            target_id, roi_center_x, roi_center_y, original_conf
                        )
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ PTZ Verification: Error adding verification request (non-critical): {e}")

        # --- PTZ Verification: อัปเดต frame จากกล้อง 2 (non-blocking) ---
        if CAM2_PTZ_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active and 'cam2_stream' in locals() and cam2_stream is not None:
            try:
                ret2, frame2 = cam2_stream.read()
                if ret2 and frame2 is not None:
                    # อัปเดต frame buffer (non-blocking, thread-safe)
                    ptz_verification.update_cam2_frame(frame2)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ PTZ Verification: Error updating cam2 frame (non-critical): {e}")

        # --- PTZ Verification: ตรวจสอบผลลัพธ์ (non-blocking) ---
        if CAM2_PTZ_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active:
            try:
                verification_result = ptz_verification.get_verification_result()

                if verification_result:
                    if verification_result['verified']:
                        # Action เมื่อยืนยันว่าเป็นโดรน
                        target_id = verification_result['target_id']
                        verified_confidence = verification_result.get('confidence', 0.0)

                        # ตรวจสอบว่า verified_confidence >= HYBRID_BASE_CONF ก่อนอัปเดต
                        if verified_confidence >= HYBRID_BASE_CONF:
                            if hybrid_tracker is not None and target_id in hybrid_tracker.targets:
                                # Read-only access แล้วอัปเดต (safe update)
                                # อัปเดตทั้ง original_confidence และ confidence
                                hybrid_tracker.targets[target_id]['original_confidence'] = verified_confidence
                                hybrid_tracker.targets[target_id]['confidence'] = verified_confidence

                                # เปลี่ยน status เป็น RED (ถ้ายังไม่ใช่)
                                current_status = hybrid_tracker.targets[target_id].get('status', 'GREEN')
                                if current_status != 'RED':
                                    hybrid_tracker.targets[target_id]['status'] = 'RED'

                                if DEBUG_MODE:
                                    print(f"✅ PTZ Verification: Target {target_id} verified with conf={verified_confidence:.2f} >= {HYBRID_BASE_CONF:.2f} → Status changed to RED (DRONE)")
                            else:
                                if DEBUG_MODE:
                                    print(f"⚠️ PTZ Verification: Target {target_id} not found in hybrid_tracker.targets")
                        else:
                            if DEBUG_MODE:
                                print(f"⚠️ PTZ Verification: Target {target_id} verified but conf={verified_confidence:.2f} < {HYBRID_BASE_CONF:.2f} → No sound alert")
                    else:
                        # Action เมื่อตรวจสอบแล้วไม่ใช่โดรน
                        target_id = verification_result['target_id']
                        reason = verification_result.get('reason', 'unknown')

                        # ลบ target ถ้า confidence ต่ำมาก
                        remove_on_fail = CAM2_PTZ_REMOVE_TARGET_ON_FAIL
                        remove_conf_threshold = CAM2_PTZ_REMOVE_TARGET_CONF_THRESHOLD

                        if remove_on_fail and hybrid_tracker is not None and target_id in hybrid_tracker.targets:
                            current_conf = hybrid_tracker.targets[target_id].get('original_confidence',
                                                                                  hybrid_tracker.targets[target_id].get('confidence', 0.0))
                            if current_conf < remove_conf_threshold:
                                # ลบ target (read-only access แล้วลบ)
                                del hybrid_tracker.targets[target_id]
                                if DEBUG_MODE:
                                    print(f"🗑️ PTZ Verification: Target {target_id} removed (verification failed, conf={current_conf:.2f} < {remove_conf_threshold:.2f}, reason={reason})")
                            else:
                                if DEBUG_MODE:
                                    print(f"⚠️ PTZ Verification: Target {target_id} verification failed but keeping (conf={current_conf:.2f} >= {remove_conf_threshold:.2f}, reason={reason})")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ PTZ Verification: Error processing verification result (non-critical): {e}")

        # --- Sound Alert: Update based on RED drones only ---
        # ส่งเสียงเฉพาะเมื่อ:
        # 1. กล้อง 1 แสดงกล่องสีแดง (RED status / DRONE status)
        # 2. original_confidence >= HYBRID_BASE_CONF
        # 3. กล้อง 2 ไม่ได้กำลังทำงาน (ไม่มี ROI search/tracking)
        if sound_alert is not None:
            # ตรวจสอบว่ากล้อง 2 กำลังทำงานหรือไม่ (มี ROI)
            ptz_verifying = False
            if 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active:
                ptz_stats = ptz_verification.get_stats()
                ptz_verifying = ptz_stats.get('is_verifying', False) and ptz_stats.get('current_roi') is not None

            # ถ้ากล้อง 2 กำลังทำงาน (มี ROI) → ไม่ส่งเสียง
            if ptz_verifying:
                if DEBUG_MODE:
                    print(f"🔇 Sound Alert: Suppressed (PTZ verification active with ROI)")
                sound_alert.update(False)
            else:
                has_red_drone = False

                if tracking_mode == 'hybrid':
                    # In hybrid mode, check if status is 'DRONE' (RED) AND original_confidence >= HYBRID_BASE_CONF
                    if hybrid_tracker is not None:
                        # ตรวจสอบจาก tracking_info ก่อน (มี status และ original_confidence)
                        if hasattr(hybrid_tracker, 'last_tracking_info'):
                            tracking_info = hybrid_tracker.last_tracking_info

                            # ตรวจสอบจาก individual targets โดยตรง (ไม่พึ่ง tracking_info status)
                            has_red_drone = False

                            if hybrid_tracker.targets:
                                for target_id, target in hybrid_tracker.targets.items():
                                    target_status = target.get('status', 'TRACKING')
                                    target_locked = target.get('target_locked', False)

                                    # เช็คเงื่อนไขง่ายๆ: มี ROI ครอบ + bbox สีแดง
                                    is_red_with_roi = (
                                        target_locked and  # ต้องมี ROI ครอบ
                                        (target_status == 'RED' or target_status == 'DRONE')  # ต้องเป็นสีแดง
                                    )

                                    # อัปเดต RED frame counter
                                    sound_alert.update_red_frame_count(target_id, is_red_with_roi, frame_counter)

                                    if is_red_with_roi:
                                        # เช็คว่าสะสม RED frames เพียงพอหรือไม่ (3 เฟรม)
                                        red_frame_count = sound_alert.get_red_frame_count(target_id)
                                        if red_frame_count >= SOUND_RED_FRAME_THRESHOLD:
                                            # ส่งเสียง (และจะส่งต่อถ้ายังเป็น RED)
                                            has_red_drone = True
                                            if DEBUG_MODE:
                                                print(f"🔊 Sound Alert: RED drone detected (target {target_id}, red_frames={red_frame_count})")
                                            break
                                        elif DEBUG_MODE:
                                            print(f"🔇 Sound Alert: Suppressed (target {target_id} red_frames={red_frame_count} < {SOUND_RED_FRAME_THRESHOLD})")
                                    else:
                                        # ไม่ใช่ RED หรือไม่มี ROI → reset counter → หยุดส่งเสียง
                                        sound_alert.update_red_frame_count(target_id, False, frame_counter)

                            # ⚠️ FIX: ลบ fallback path - ถ้าไม่มี targets = ไม่มี ROI = ไม่ส่งเสียง
                        # Fallback: ตรวจสอบจาก target dictionary โดยตรง
                        elif hybrid_tracker.target_locked and len(hybrid_tracker.targets) > 0:
                            has_red_drone = False

                            # ตรวจสอบทุก target โดยตรง
                            for target_id, target in hybrid_tracker.targets.items():
                                target_status = target.get('status', 'TRACKING')
                                target_locked = target.get('target_locked', False)

                                # เช็คเงื่อนไขง่ายๆ: มี ROI ครอบ + bbox สีแดง
                                is_red_with_roi = (
                                    target_locked and  # ต้องมี ROI ครอบ
                                    (target_status == 'RED' or target_status == 'DRONE')  # ต้องเป็นสีแดง
                                )

                                # อัปเดต RED frame counter
                                sound_alert.update_red_frame_count(target_id, is_red_with_roi, frame_counter)

                                if is_red_with_roi:
                                    # เช็คว่าสะสม RED frames เพียงพอหรือไม่ (3 เฟรม)
                                    red_frame_count = sound_alert.get_red_frame_count(target_id)
                                    if red_frame_count >= SOUND_RED_FRAME_THRESHOLD:
                                        # ส่งเสียง (และจะส่งต่อถ้ายังเป็น RED)
                                        has_red_drone = True
                                        if DEBUG_MODE:
                                            print(f"🔊 Sound Alert: RED drone detected (target {target_id}, fallback, red_frames={red_frame_count})")
                                        break
                                    elif DEBUG_MODE:
                                        print(f"🔇 Sound Alert: Suppressed (target {target_id} red_frames={red_frame_count} < {SOUND_RED_FRAME_THRESHOLD})")
                                else:
                                    # ไม่ใช่ RED หรือไม่มี ROI → reset counter → หยุดส่งเสียง
                                    sound_alert.update_red_frame_count(target_id, False, frame_counter)
                else:
                    # In multi mode, check for RED trackers only (และไม่ใช่แมลง, stationary, หรือ airplane)
                    has_red_drone = False
                    for tracker in active_trackers.values():
                        if tracker.status == 'RED':
                            # ตรวจสอบแมลง
                            is_insect = tracker.is_likely_insect() if hasattr(tracker, 'is_likely_insect') else False

                            # ⚠️ NEW: ตรวจสอบว่าเป็น airplane หรือไม่ (ใช้ confidence score)
                            is_airplane = False
                            airplane_confidence = 0.0
                            if hasattr(tracker, 'is_likely_airplane'):
                                is_airplane, airplane_confidence = tracker.is_likely_airplane(thresholds=resolution_thresholds)

                            # ⚠️ NEW: ตรวจสอบว่าเป็น stationary object หรือไม่ (ใช้ velocity profile)
                            is_stationary = False
                            if hasattr(tracker, 'analyze_velocity_profile'):
                                velocity_profile = tracker.analyze_velocity_profile(thresholds=resolution_thresholds)
                                # ถ้า mean velocity ต่ำมาก → stationary
                                frame_diagonal = math.sqrt(w**2 + h**2)
                                min_velocity_threshold = BLACKLIST_MIN_VELOCITY_RATIO * frame_diagonal
                                if velocity_profile['mean'] < min_velocity_threshold:
                                    is_stationary = True

                            if not is_insect and not is_airplane and not is_stationary:
                                # ไม่ใช่แมลง, ไม่ใช่เครื่องบิน, ไม่ใช่ stationary → นับเป็น red drone
                                has_red_drone = True
                                break
                            elif DEBUG_MODE:
                                if is_insect:
                                    print(f"🔇 Sound Alert: Suppressed (tracker {tracker.id} is insect)")
                                elif is_airplane:
                                    print(f"🔇 Sound Alert: Suppressed (tracker {tracker.id} is airplane, confidence={airplane_confidence:.2f})")
                                elif is_stationary:
                                    print(f"🔇 Sound Alert: Suppressed (tracker {tracker.id} is stationary, velocity={velocity_profile['mean']:.2f})")

                    if DEBUG_MODE and has_red_drone:
                        print(f"🔊 Sound Alert: RED drone detected (multi mode, RED tracker found)")
                    elif DEBUG_MODE:
                        statuses = [tracker.status for tracker in active_trackers.values()]
                        if statuses:
                            print(f"🔇 Sound Alert: Suppressed (multi mode, statuses={statuses}, no RED)")

                sound_alert.update(has_red_drone)

                # ลบ target ที่หายไปจาก counter
                if hybrid_tracker is not None and hybrid_tracker.targets:
                    current_target_ids = set(hybrid_tracker.targets.keys())
                    with sound_alert.lock:
                        target_ids_to_remove = [tid for tid in sound_alert.target_red_frames.keys() if tid not in current_target_ids]
                        for tid in target_ids_to_remove:
                            del sound_alert.target_red_frames[tid]

        # วาดเส้นขอบฟ้า (ถ้ามี)
        if horizon_points:
            pts = np.array(horizon_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, HORIZON_COLOR, 2)
        if drawing_mode["active"] and temp_draw_points:
            pts_tmp = np.array(temp_draw_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts_tmp], False, DRAWING_COLOR, 2)

        # --- DISPLAY SCALING: Resize only for display (not for processing) ---
        # This ensures performance is not affected - processing still uses full resolution
        if scale_factor < 1.0:
            display_frame_scaled = cv2.resize(display_frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame_scaled = display_frame

        # --- HUD: แสดงผลแบบแนวนอนชิดขอบล่าง ---
        if HUD_ENABLED:
            # คำนวณขนาดฟอนต์ให้สัดส่วนกับ resolution (คำนวณครั้งเดียวต่อเฟรม)
            base_font_scale = display_w / 1920.0  # normalize to 1920p
            font_scale_small = 0.5 * base_font_scale
            font_scale_medium = 0.6 * base_font_scale
            font_scale_large = 0.7 * base_font_scale

            # คำนวณความสูง HUD (1-2 บรรทัด)
            hud_height = int(35 * base_font_scale)

            # คำนวณความกว้าง HUD (เว้นที่สำหรับ PIP ถ้ามี)
            pip_width = 0
            if CAM2_PTZ_PIP_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active:
                ptz_stats_check = ptz_verification.get_stats()
                is_verifying_check = ptz_stats_check.get('is_verifying', False)
                if is_verifying_check:
                    pip_width = CAM2_PTZ_PIP_WIDTH + CAM2_PTZ_PIP_MARGIN * 2
            # ใช้ resolution-dependent thresholds สำหรับ HUD
            hud_margin = resolution_thresholds.get('HUD_MARGIN', HUD_MARGIN)
            hud_min_width = resolution_thresholds.get('HUD_MIN_WIDTH', HUD_MIN_WIDTH)
            hud_max_width = resolution_thresholds.get('HUD_MAX_WIDTH', HUD_MAX_WIDTH)

            # คำนวณความกว้าง HUD (จำกัดด้วย min/max)
            hud_width_calc = display_w - hud_margin * 2 - pip_width
            hud_width = max(hud_min_width, min(hud_max_width, hud_width_calc))

            # ตำแหน่ง (ชิดขอบล่างสุด)
            hud_x = hud_margin  # มุมซ้าย
            hud_y = display_h - hud_height  # ชิดขอบล่างสุด (ไม่มี margin)

            # คำนวณ spacing สำหรับแนวนอน
            text_offset_x = int(10 * base_font_scale)
            text_offset_y = int(22 * base_font_scale)  # กลางแนวตั้งของ HUD
            column_spacing = int(180 * base_font_scale)  # ระยะห่างระหว่างคอลัมน์

            # วาด HUD background (ชิดขอบล่าง)
            cv2.rectangle(display_frame_scaled, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), HUD_BACKGROUND_COLOR, -1)

            # เริ่มต้นตำแหน่ง X สำหรับแต่ละคอลัมน์
            current_x = hud_x + text_offset_x
            current_y = hud_y + text_offset_y

            # Column 1: Mode
            mode_color = (0, 255, 255) if tracking_mode == 'hybrid' else (0, 255, 0)
            mode_text = "HYBRID" if tracking_mode == 'hybrid' else "MULTI"
            if ADAPTIVE_HYBRID_MODE_ENABLED:
                mode_text += " (A)"
                if tracking_mode == 'multi':
                    search_frames = frame_counter - multi_mode_search_start_frame
                    mode_text += f" [{search_frames}/{ADAPTIVE_HYBRID_SEARCH_TIMEOUT}]"
            cv2.putText(display_frame_scaled, f"Mode: {mode_text}",
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, mode_color, 1)
            current_x += column_spacing

            # Column 2: Processing FPS
            status_color = FPS_GOOD_COLOR if fps >= FPS_GOOD_THRESHOLD else FPS_BAD_COLOR
            cv2.putText(display_frame_scaled, f"Proc: {fps:.1f} FPS",
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, status_color, 1)
            current_x += column_spacing

            # Column 3: Display FPS
            display_status_color = FPS_GOOD_COLOR if display_fps >= FPS_GOOD_THRESHOLD else FPS_BAD_COLOR
            cv2.putText(display_frame_scaled, f"Display: {display_fps:.1f} FPS",
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, display_status_color, 1)
            current_x += column_spacing

            # Column 4: Algo Time
            cv2.putText(display_frame_scaled, f"Algo: {algo_time:.1f}ms",
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, ALGO_TIME_COLOR, 1)
            current_x += column_spacing

            # Column 5: GPU/CPU/Temp (รวมกัน)
            gpu_text = f"GPU: {monitor.stats['GPU']}%"
            cpu_text = f"CPU: {monitor.stats['CPU']}%"
            temp_text = f"Temp: {monitor.stats['TEMP']}C"  # เปลี่ยนจาก °C เป็น C เพื่อหลีกเลี่ยงปัญหา encoding

            # คำนวณความกว้างของข้อความจริงๆ เพื่อไม่ให้ทับกัน
            (gpu_w, gpu_h), _ = cv2.getTextSize(gpu_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 1)
            (cpu_w, cpu_h), _ = cv2.getTextSize(cpu_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 1)

            cv2.putText(display_frame_scaled, gpu_text,
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, STATS_TEXT_COLOR, 1)
            current_x += gpu_w + int(15 * base_font_scale)  # ใช้ความกว้างจริง + spacing

            cv2.putText(display_frame_scaled, cpu_text,
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, STATS_TEXT_COLOR, 1)
            current_x += cpu_w + int(15 * base_font_scale)  # ใช้ความกว้างจริง + spacing

            cv2.putText(display_frame_scaled, temp_text,
                       (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, STATS_TEXT_COLOR, 1)
            current_x += column_spacing

            # Column 6: Adaptive Status (ถ้าเปิดใช้งาน)
            if ADAPTIVE_ENABLED:
                adaptive_text = f"Adaptive: V={adaptive_var_display} A={adaptive_min_area_display:.1f} K={adaptive_morph_size_display}"
                cv2.putText(display_frame_scaled, adaptive_text,
                           (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 0), 1)
                current_x += int(column_spacing * 1.5)
                contours_text = f"Contours: {contours_count_for_hud} (avg: {avg_contour_count:.0f})"
                cv2.putText(display_frame_scaled, contours_text,
                           (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 0), 1)
                current_x += int(column_spacing * 1.2)

            # Column 7: Adaptive HYBRID_MIN_MOTION_AREA (ถ้าเปิดใช้งาน)
            if ADAPTIVE_HYBRID_MIN_AREA_ENABLED:
                adaptive_min_area_value = current_hybrid_min_motion_area if 'current_hybrid_min_motion_area' in locals() else HYBRID_MIN_MOTION_AREA

                # แสดง min_area แยกตาม object size (ถ้ามี adaptive_hybrid_min_area_manager)
                if 'adaptive_hybrid_min_area_manager' in locals() and adaptive_hybrid_min_area_manager is not None:
                    tiny_min = adaptive_hybrid_min_area_manager.get_min_area_for_size('TINY')
                    small_min = adaptive_hybrid_min_area_manager.get_min_area_for_size('SMALL')
                    medium_min = adaptive_hybrid_min_area_manager.get_min_area_for_size('MEDIUM')
                    min_area_text = f"MinArea: T={tiny_min} S={small_min} M={medium_min}"
                else:
                    # Fallback: แสดงค่าเดียว
                    min_area_text = f"MinArea: {adaptive_min_area_value}"

                cv2.putText(display_frame_scaled, min_area_text,
                           (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, COLOR_TEXT, 1)
                current_x += column_spacing

            # Column 8: Camera Movement Status (ถ้าเปิดใช้งาน)
            if CAM_MOVE_DETECTION_ENABLED:
                cam_status = "MOVED" if camera_moved else "STABLE"
                cam_status_color = (0, 255, 255) if camera_moved else (0, 255, 0)
                cv2.putText(display_frame_scaled, f"Cam: {cam_status}",
                           (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, cam_status_color, 1)

        # --- PTZ Verification: แสดง PIP window (มุมขวาล่าง) เมื่อกำลังทำงาน ---
        # วาง PIP หลังจาก resize เพื่อให้ขนาดถูกต้องตาม display resolution
        if CAM2_PTZ_PIP_ENABLED and 'ptz_verification' in locals() and ptz_verification is not None and ptz_verification.is_active:
            # ตรวจสอบว่า PTZ verification กำลังทำงานหรือไม่
            ptz_stats = ptz_verification.get_stats()
            is_verifying = ptz_stats.get('is_verifying', False)

            if is_verifying and 'cam2_stream' in locals() and cam2_stream is not None:
                # ดึง frame จากกล้อง 2
                ret2, frame2 = cam2_stream.read()
                if ret2 and frame2 is not None:
                    h_display, w_display = display_frame_scaled.shape[:2]

                    # คำนวณตำแหน่ง PIP (มุมขวาล่าง) - ปรับตาม display resolution
                    pip_w = CAM2_PTZ_PIP_WIDTH
                    pip_h = CAM2_PTZ_PIP_HEIGHT
                    pip_x = w_display - pip_w - CAM2_PTZ_PIP_MARGIN
                    pip_y = h_display - pip_h - CAM2_PTZ_PIP_MARGIN

                    # ตรวจสอบว่าตำแหน่งอยู่ในขอบเขต
                    if pip_x >= 0 and pip_y >= 0 and pip_x + pip_w <= w_display and pip_y + pip_h <= h_display:
                        # ดึง ROI box (ถ้ามี)
                        current_roi = ptz_stats.get('current_roi', None)
                        search_mode = ptz_stats.get('search_mode', True)

                        # Resize frame จากกล้อง 2
                        frame2_h, frame2_w = frame2.shape[:2]
                        frame2_resized = cv2.resize(frame2, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

                        # วาด ROI box บน frame2_resized (ถ้ามี)
                        if current_roi is not None:
                            roi_box = current_roi.get('box', None)
                            if roi_box is not None and len(roi_box) == 4:
                                try:
                                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box

                                    # ตรวจสอบว่า ROI box ถูกต้อง (x2 > x1 และ y2 > y1)
                                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                                        # แปลง ROI coordinates จาก frame2 ขนาดจริงไปเป็น frame2_resized
                                        scale_x = pip_w / frame2_w
                                        scale_y = pip_h / frame2_h

                                        roi_x1_scaled = int(roi_x1 * scale_x)
                                        roi_y1_scaled = int(roi_y1 * scale_y)
                                        roi_x2_scaled = int(roi_x2 * scale_x)
                                        roi_y2_scaled = int(roi_y2 * scale_y)

                                        # ตรวจสอบว่าอยู่ในขอบเขต
                                        roi_x1_scaled = max(0, min(roi_x1_scaled, pip_w - 1))
                                        roi_y1_scaled = max(0, min(roi_y1_scaled, pip_h - 1))
                                        roi_x2_scaled = max(roi_x1_scaled + 1, min(roi_x2_scaled, pip_w))
                                        roi_y2_scaled = max(roi_y1_scaled + 1, min(roi_y2_scaled, pip_h))

                                        # ตรวจสอบอีกครั้งว่า ROI box ถูกต้องหลัง scaling
                                        if roi_x2_scaled > roi_x1_scaled and roi_y2_scaled > roi_y1_scaled:
                                            # เลือกสีตาม mode
                                            if search_mode:
                                                # Search mode: สีเขียว (เหมือนกล้อง 1)
                                                roi_color = (0, 255, 0)  # BGR: เขียว
                                                roi_thickness = 2
                                            else:
                                                # Tracking mode: สีเหลือง
                                                roi_color = (0, 255, 255)  # BGR: เหลือง
                                                roi_thickness = 2

                                            # วาด ROI box
                                            cv2.rectangle(frame2_resized,
                                                        (roi_x1_scaled, roi_y1_scaled),
                                                        (roi_x2_scaled, roi_y2_scaled),
                                                        roi_color,
                                                        roi_thickness)

                                            # วาด center point
                                            roi_cx_scaled = (roi_x1_scaled + roi_x2_scaled) // 2
                                            roi_cy_scaled = (roi_y1_scaled + roi_y2_scaled) // 2
                                            cv2.circle(frame2_resized, (roi_cx_scaled, roi_cy_scaled), 3, roi_color, -1)

                                            # วาด label (ถ้าพอที่)
                                            if roi_y1_scaled > 15:
                                                label_text = "SEARCH" if search_mode else "TRACKING"
                                                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                                cv2.rectangle(frame2_resized,
                                                            (roi_x1_scaled, roi_y1_scaled - text_h - 4),
                                                            (roi_x1_scaled + text_w + 2, roi_y1_scaled),
                                                            (0, 0, 0),
                                                            -1)
                                                cv2.putText(frame2_resized, label_text,
                                                          (roi_x1_scaled + 1, roi_y1_scaled - 2),
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                          roi_color, 1)

                                            if DEBUG_MODE:
                                                print(f"✅ PIP: ROI box drawn at ({roi_x1_scaled}, {roi_y1_scaled}) to ({roi_x2_scaled}, {roi_y2_scaled}), mode={'SEARCH' if search_mode else 'TRACKING'}")
                                        else:
                                            if DEBUG_MODE:
                                                print(f"⚠️ PIP: Invalid ROI box after scaling: ({roi_x1_scaled}, {roi_y1_scaled}) to ({roi_x2_scaled}, {roi_y2_scaled})")
                                    else:
                                        if DEBUG_MODE:
                                            print(f"⚠️ PIP: Invalid ROI box coordinates: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
                                except Exception as e:
                                    if DEBUG_MODE:
                                        print(f"⚠️ PIP: Error drawing ROI box: {e}, roi_box={roi_box}")
                            else:
                                if DEBUG_MODE:
                                    print(f"⚠️ PIP: current_roi exists but no valid 'box' key, current_roi={current_roi}")
                        else:
                            if DEBUG_MODE and is_verifying:
                                print(f"⚠️ PIP: current_roi is None (is_verifying={is_verifying})")

                        # วาง frame ใน display_frame_scaled
                        display_frame_scaled[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = frame2_resized

                        # วาดขอบและ label
                        cv2.rectangle(display_frame_scaled,
                                    (pip_x, pip_y),
                                    (pip_x + pip_w, pip_y + pip_h),
                                    CAM2_PTZ_PIP_BORDER_COLOR,
                                    CAM2_PTZ_PIP_BORDER_THICKNESS)

                        # แสดง label และ zoom level
                        current_zoom = ptz_stats.get('current_zoom_level', None)
                        mode_text = "SEARCH" if search_mode else "TRACKING"
                        if current_zoom is not None:
                            label_text = f"CAM2 PTZ ({mode_text}, Zoom: {current_zoom:.1f}x)"
                        else:
                            label_text = f"CAM2 PTZ ({mode_text})"

                        # วาด background สำหรับ text
                        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_frame_scaled,
                                    (pip_x, pip_y - text_h - 5),
                                    (pip_x + text_w + 4, pip_y),
                                    (0, 0, 0),
                                    -1)

                        # วาด text
                        cv2.putText(display_frame_scaled, label_text,
                                  (pip_x + 2, pip_y - 3),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  CAM2_PTZ_PIP_BORDER_COLOR, 1)

        # --- PERIODIC MEMORY CLEANUP ---
        # PERIODIC_MEMORY_CLEANUP is set above based on resolution
        if frame_counter % PERIODIC_MEMORY_CLEANUP == 0:
            # Periodic CUDA memory cleanup
            try:
                from memory_manager import clear_cuda_memory
                clear_cuda_memory()
            except:
                pass

            # Force garbage collection
            import gc
            gc.collect()

            # Log memory usage (less frequently)
            if frame_counter % (PERIODIC_MEMORY_CLEANUP * 3) == 0:
                try:
                    import psutil
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    mem_mb = mem_info.rss / 1024 / 1024
                    mem_percent = psutil.virtual_memory().percent
                    print(f"📊 Memory: {mem_mb:.1f} MB ({mem_percent:.1f}%)")
                    if mem_percent > 85:
                        print(f"⚠️ High memory - consider increasing FRAME_SKIP_MOG2")
                except:
                    pass

        # --- DISPLAY UPDATE (with interval) ---
        # DISPLAY_UPDATE_INTERVAL is set above based on resolution
        if frame_counter % DISPLAY_UPDATE_INTERVAL == 0:
            cv2.imshow(WINDOW_NAME, display_frame_scaled)
            # คำนวณ Display FPS (FPS จริงของการแสดงผล)
            curr_display_time = time.time()
            display_fps = 1 / (curr_display_time - prev_display_time + 1e-6)
            prev_display_time = curr_display_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 User requested exit...")
            break
        elif key == ord('h'):  # Switch between multi and hybrid tracking modes
            if tracking_mode == 'multi':
                # Switch to hybrid mode
                tracking_mode = 'hybrid'
                # Initialize hybrid tracker if not already initialized
                if hybrid_tracker is None:
                    hybrid_tracker = HybridDroneTracker(yolo_model, w, h)
                    print("🔄 Switched to HYBRID tracking mode (single-target persistent)")
                else:
                    hybrid_tracker.reset()
                    print("🔄 Switched to HYBRID tracking mode (reset)")
                # Reset multi-tracker state
                active_trackers = {}
                global_track_id = 0
            else:
                # Switch to multi mode
                tracking_mode = 'multi'
                # Reset hybrid tracker state
                if hybrid_tracker is not None:
                    hybrid_tracker.reset()
                # Reset multi-tracker state
                active_trackers = {}
                global_track_id = 0
                print("🔄 Switched to MULTI tracking mode (multi-object)")
        elif key == ord('l'):
            drawing_mode["active"] = True
            temp_draw_points = []
            print("✏️ Draw horizon: click+drag to sketch, press Enter to save")
        elif key in (10, 13):  # Enter key
            if drawing_mode["active"] and temp_draw_points:
                horizon_points = temp_draw_points[:]
                try:
                    np.save(HORIZON_FILE, np.array(horizon_points, dtype=np.int32))
                    print(f"💾 Horizon saved ({len(horizon_points)} points) -> {HORIZON_FILE}")
                except Exception as e:
                    print(f"⚠️ Save failed: {e}")
                rebuild_horizon_masks()
                # Rebuild exclusion mask after horizon changes
                exclusion_mask = create_exclusion_mask()
            drawing_mode["active"] = False
            temp_draw_points = []

        # horizon_mask_upper / lower พร้อมใช้แบ่งบนล่าง (เบา ไม่กระทบ FPS)
        # ตัวอย่างการใช้งาน (คอมเมนต์ไว้): ใช้ mask กรองคอนทัวร์/พิกเซลเฉพาะด้านบนหรือด้านล่าง
        # if horizon_mask_upper is not None:
        #     upper_only = cv2.bitwise_and(mask_cpu, mask_cpu, mask=horizon_mask_upper)
        #     lower_only = cv2.bitwise_and(mask_cpu, mask_cpu, mask=horizon_mask_lower)

    # CLEANUP
    print("Cleaning up resources...")
    if sound_alert is not None:
        sound_alert.stop()
    if 'ptz_verification' in locals() and ptz_verification is not None:
        ptz_verification.stop()  # จะกลับไปที่ preset position อัตโนมัติ
    if 'cam2_stream' in locals() and cam2_stream is not None:
        cam2_stream.release()
    cam.release()
    monitor.close()
    cv2.destroyAllWindows()
    print("👋 Force Exiting...")
    os._exit(0)

if __name__ == "__main__":
    # 1. ฆ่าร่างเก่าก่อนทำอะไรทั้งสิ้น (เรียกจาก memory_manager)
    # ใส่ชื่อไฟล์ตัวเองลงไป
    kill_old_instances("fast_motion_sky.py")

    # 2. เริ่มการทำงานปกติ
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by User")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 3. ล้างแรมทิ้งท้ายเมื่อปิดโปรแกรม
        print("👋 Exiting...")
        try:
            from memory_manager import clear_cuda_memory
            clear_cuda_memory()
        except:
            pass
        import sys
        sys.exit(0)






