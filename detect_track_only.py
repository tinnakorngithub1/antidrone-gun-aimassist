"""
detect_track_only.py — Gun Aim Assist with fast async YOLO26 detection.

Detection-only: no Kalman, no egomotion, no tracking state machine.
Each frame is detected independently — bbox comes directly from YOLO26.
Camera rotation/shake has zero impact on accuracy.

Architecture:
  CameraStream (threaded) → Main Loop
    ├─ DetectorWorker  (async YOLO26 640, pre-resize, NMS-free)
    ├─ DetectSystem    (submit every frame, pick best, temporal hold)
    └─ Full gun-aim HUD (reticle, READY/THINGS, sound, arrows, distance)
"""

import cv2
import datetime
import json
import math
import numpy as np
import os
import sys
import threading
import time
from collections import deque

try:
    from fast_motion_sky import CameraStream
    from config import get_camera_config, ACTIVE_CAMERA
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from smart_detection_yolo_only import load_yolo_model, detect_yolo_full_frame

# Sound (optional)
try:
    import pygame
    pygame.mixer.init()
    PYGAME_SOUND_AVAILABLE = True
except Exception:
    PYGAME_SOUND_AVAILABLE = False
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

_sound_ready = None

def _play_ready_sound():
    global _sound_ready
    if not SOUND_ON_READY:
        return
    path = os.path.join(os.path.dirname(__file__), SOUND_FILE)
    if not os.path.isfile(path):
        return
    try:
        if PYGAME_SOUND_AVAILABLE:
            if _sound_ready is None:
                _sound_ready = pygame.mixer.Sound(path)
            _sound_ready.play()
        elif PLAYSOUND_AVAILABLE:
            playsound(path, block=False)
    except Exception:
        pass

# =============================================================================
# Parameters
# =============================================================================

CAMERA_NAME = None

# --- Detection ---
YOLO_CONF_MIN = 0.01
YOLO_ENGINE = "last_imgsz640.engine"
YOLO_IMGSZ = 640

# --- Frame enhancement before YOLO ---
ENHANCE_SHARPEN = False       # Unsharp mask to counter motion blur
SHARPEN_AMOUNT = 1.5          # >1 = sharper (thermal: use 1.3)
ENHANCE_CLAHE = True         # CLAHE contrast boost (helps thermal/low-light)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# --- Temporal hold (keep last bbox when YOLO misses due to blur) ---
MAX_HOLD_FRAMES = 5           # keep bbox for N frames if YOLO finds nothing

# --- YOLO + Thermal Fusion (PTZ-safe) ---
FUSION_ENABLED = True
YOLO_CONF_CANDIDATE = 0.05      # YOLO conf ต่ำเพื่อหา candidates
FUSION_THERMAL_ROI_PAD = 20
FUSION_YOLO_WEIGHT = 0.5
FUSION_THERMAL_WEIGHT = 0.5
FUSION_THRESHOLD = 0.35
THERMAL_BLOB_MIN_AREA = 10
THERMAL_BLOB_MAX_AREA = 5000
THERMAL_ROI_MIN_SIDE = 10       # ROI ขนาดต่ำสุด (px) ถ้าเล็กกว่านี้ return 0
THERMAL_BLOB_CENTER_RATIO = 0.4 # blob centroid ต้องอยู่ภายใน center 40% ของ ROI

# --- Phase 2: Thermal-only fallback ---
THERMAL_FALLBACK_ENABLED = True   # เมื่อ YOLO ไม่เจอ ใช้ thermal blob รอบ last position
THERMAL_FALLBACK_SEARCH_PAD = 80  # ขยาย ROI สำหรับค้นหา (px)
THERMAL_FALLBACK_CONF = 0.5       # confidence ที่ให้เมื่อใช้ thermal-only

# --- Phase 2: Motion compensation ---
MOTION_COMPENSATION_ENABLED = True
MOTION_FLOW_DOWNSCALE = 4       # 640/4=160, 480/4=120
MOTION_GRID_ROWS = 8
MOTION_GRID_COLS = 8
MOTION_PREDICTED_SEARCH_PAD = 40  # ROI เล็กรอบ predicted (แม่นกว่า)
MOTION_LK_WINSIZE = (15, 15)    # Lucas-Kanade window size (เล็กลง = เร็วขึ้น)
MOTION_LK_MAX_LEVEL = 2         # pyramid levels
MOTION_MIN_VALID_POINTS = 8     # ถ้า valid points น้อยกว่านี้ → return (0, 0)

# --- Reticle / HUD ---
CENTER_RADIUS_RATIO = 0.05
CENTER_RADIUS_PX = 0
RETICLE_RADIUS_RATIOS = (0.03, 0.05, 0.08, 0.20)
ARROW_LEN = 120
ARROW_THICKNESS = 6

# --- Sound ---
SOUND_ON_READY = True
SOUND_FILE = "beep_2x.wav"
SOUND_READY_INTERVAL = 0.6
SOUND_APPROACH_ON = True
SOUND_BEEP_SLOW_INTERVAL = 1.2
SOUND_BEEP_FAST_INTERVAL = 0.35
APPROACH_BEEP_OUTER_SCALE = 2.5

# --- Distance ---
DRONE_SIZE_M = 0.30

# --- Day/Night ---
DAY_HOUR_START = 6
DAY_HOUR_END = 18

# --- Display ---
DISPLAY_MAX_W = 1920
DISPLAY_MAX_H = 1080
WINDOW_NAME = "Gun Aim Assist"
HUD_REF_WIDTH = 1920.0

# --- Keys ---
KEY_ADJUST_CENTER = "C"
KEY_SETTINGS = "S"
KEY_QUIT = "Q"
AIM_CENTER_STEP_PX = 10

# =============================================================================
# ShooterConfig — aim center + ballistics (save/load JSON)
# =============================================================================

class ShooterConfig:
    FILENAME = "gun_aim_assist_config.json"

    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.effective_range_m = 100
        self.muzzle_velocity_ms = 900
        self.bullet_weight_g = 9
        self.target_size_m = 0.30
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FILENAME)

    def load(self):
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.offset_x = int(d.get("offset_x", 0))
                self.offset_y = int(d.get("offset_y", 0))
                self.effective_range_m = float(d.get("effective_range_m", 100))
                self.muzzle_velocity_ms = float(d.get("muzzle_velocity_ms", 900))
                self.bullet_weight_g = float(d.get("bullet_weight_g", 9))
                self.target_size_m = float(d.get("target_size_m", 0.30))
            except Exception:
                pass

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({
                    "offset_x": self.offset_x,
                    "offset_y": self.offset_y,
                    "effective_range_m": self.effective_range_m,
                    "muzzle_velocity_ms": self.muzzle_velocity_ms,
                    "bullet_weight_g": self.bullet_weight_g,
                    "target_size_m": self.target_size_m,
                }, f, indent=2)
        except Exception:
            pass

    def get_center(self, frame_w, frame_h):
        cx = frame_w // 2 + self.offset_x
        cy = frame_h // 2 + self.offset_y
        return max(0, min(frame_w - 1, cx)), max(0, min(frame_h - 1, cy))

    def move(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

    def reset(self):
        self.offset_x = 0
        self.offset_y = 0

# =============================================================================
# Helper functions
# =============================================================================

def is_daytime():
    h = datetime.datetime.now().hour
    return DAY_HOUR_START <= h < DAY_HOUR_END

def estimate_distance_m(w_px, h_px, frame_w, frame_h, fov_h_deg, fov_v_deg, size_m=0.35):
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

def compute_guide_direction(target_center, cx_frame, cy_frame, radius_px, ready_to_fire):
    if target_center is None or ready_to_fire:
        return None, None
    cx_t, cy_t = target_center
    dx = cx_t - cx_frame
    dy = cy_t - cy_frame
    if abs(dx) <= radius_px and abs(dy) <= radius_px:
        return None, None
    guide_h = "RIGHT" if dx > 0 else "LEFT"
    guide_v = "DOWN" if dy > 0 else "UP"
    return guide_h, guide_v

def compute_approach_beep_interval(d, radius_px, w_d, h_d, slow_interval, fast_interval):
    d_bbox = 0.5 * math.sqrt(w_d * w_d + h_d * h_d)
    d_outer = d_bbox * APPROACH_BEEP_OUTER_SCALE
    d_outer = max(d_outer, radius_px + 1.0)
    if d >= d_outer:
        return slow_interval
    if d <= radius_px:
        return None
    t = (d - radius_px) / (d_outer - radius_px)
    t = max(0.0, min(1.0, t))
    return fast_interval + t * (slow_interval - fast_interval)

def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return DISPLAY_MAX_W, DISPLAY_MAX_H

def hud_scale(frame_w):
    """Scale factor for HUD text/elements relative to 1920px reference."""
    return max(0.4, frame_w / HUD_REF_WIDTH)

# =============================================================================
# AsyncLatestValue — thread-safe single-slot mailbox
# =============================================================================

class AsyncLatestValue:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._new = False

    def put(self, value):
        with self._lock:
            self._value = value
            self._new = True

    def get(self):
        with self._lock:
            v = self._value
            n = self._new
            self._new = False
            return v, n

# =============================================================================
# Frame enhancement — sharpen + CLAHE before YOLO
# =============================================================================

def enhance_frame(frame):
    """Apply unsharp mask and/or CLAHE to improve detection on blurry/low-contrast frames."""
    enhanced = frame

    if ENHANCE_SHARPEN:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, SHARPEN_AMOUNT, blurred, 1.0 - SHARPEN_AMOUNT, 0)

    if ENHANCE_CLAHE:
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                    tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                    tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
            enhanced = clahe.apply(enhanced)

    return enhanced

# =============================================================================
# YOLO + Thermal Fusion
# =============================================================================

def compute_thermal_score(frame, bbox, pad):
    """
    คำนวณ thermal_score (0-1) สำหรับ bbox — มี blob ร้อนใน ROI หรือไม่
    """
    try:
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        if (x2 - x1) < THERMAL_ROI_MIN_SIDE or (y2 - y1) < THERMAL_ROI_MIN_SIDE:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        if len(roi.shape) == 2:
            roi_gray = roi
        else:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_w, roi_h = x2 - x1, y2 - y1
        center_x, center_y = roi_w / 2.0, roi_h / 2.0
        max_dist = THERMAL_BLOB_CENTER_RATIO * min(roi_w, roi_h) / 2.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if THERMAL_BLOB_MIN_AREA <= area <= THERMAL_BLOB_MAX_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    d = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                    if d <= max_dist:
                        return 1.0
        return 0.0
    except Exception:
        return 0.0

def fuse_yolo_thermal(dets, frame, pad):
    """Fuse YOLO detections กับ thermal blob score"""
    fused = []
    for x, y, w, h, conf in dets:
        thermal_score = compute_thermal_score(frame, (x, y, w, h), pad)
        fusion_score = FUSION_YOLO_WEIGHT * conf + FUSION_THERMAL_WEIGHT * thermal_score
        if fusion_score >= FUSION_THRESHOLD:
            fused.append((x, y, w, h, fusion_score))
    return fused

def detect_thermal_blobs_in_roi(frame, center_bbox, search_pad):
    """
    Phase 2: ค้นหา thermal blob ใน ROI รอบ center_bbox
    คืน [(x, y, w, h, THERMAL_FALLBACK_CONF)] หรือ []
    """
    try:
        x, y, w, h = center_bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, x - search_pad)
        y1 = max(0, y - search_pad)
        x2 = min(fw, x + w + search_pad)
        y2 = min(fh, y + h + search_pad)
        if (x2 - x1) < THERMAL_ROI_MIN_SIDE or (y2 - y1) < THERMAL_ROI_MIN_SIDE:
            return []
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        if len(roi.shape) == 2:
            roi_gray = roi
        else:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_cx, center_cy = x + w / 2.0, y + h / 2.0
        best_blob = None
        best_dist = float("inf")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if THERMAL_BLOB_MIN_AREA <= area <= THERMAL_BLOB_MAX_AREA:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                bcx = x1 + bx + bw / 2.0
                bcy = y1 + by + bh / 2.0
                d = math.sqrt((bcx - center_cx) ** 2 + (bcy - center_cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_blob = (x1 + bx, y1 + by, bw, bh)
        if best_blob is not None:
            bx, by, bw, bh = best_blob
            return [(bx, by, bw, bh, THERMAL_FALLBACK_CONF)]
        return []
    except Exception:
        return []


def compute_scene_shift(prev_gray, curr_gray):
    """
    Sparse Lucas-Kanade optical flow: คำนวณ scene shift (dx, dy) จาก prev → curr.
    คืน (dx, dy) หรือ (0, 0) ถ้า valid points น้อยเกินไป / exception.
    """
    try:
        if prev_gray.shape != curr_gray.shape:
            return (0.0, 0.0)
        ds = MOTION_FLOW_DOWNSCALE
        prev_small = cv2.resize(prev_gray, None, fx=1.0 / ds, fy=1.0 / ds, interpolation=cv2.INTER_LINEAR)
        curr_small = cv2.resize(curr_gray, None, fx=1.0 / ds, fy=1.0 / ds, interpolation=cv2.INTER_LINEAR)
        H, W = prev_small.shape[0], prev_small.shape[1]
        if min(H, W) < 20:
            return (0.0, 0.0)
        xs = np.linspace(0, W - 1, MOTION_GRID_COLS).astype(np.float32)
        ys = np.linspace(0, H - 1, MOTION_GRID_ROWS).astype(np.float32)
        prev_pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float32).reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_small, curr_small, prev_pts, None,
            winSize=MOTION_LK_WINSIZE, maxLevel=MOTION_LK_MAX_LEVEL
        )
        if next_pts is None or status is None:
            return (0.0, 0.0)
        ok = status.ravel() == 1
        if np.sum(ok) < MOTION_MIN_VALID_POINTS:
            return (0.0, 0.0)
        flow = next_pts - prev_pts
        flow_x = flow[ok, 0, 0]
        flow_y = flow[ok, 0, 1]
        dx = float(np.median(flow_x)) * ds
        dy = float(np.median(flow_y)) * ds
        return (dx, dy)
    except Exception:
        return (0.0, 0.0)


def apply_motion_to_bbox(bbox, dx, dy, fw, fh):
    """
    คาดการณ์ตำแหน่ง bbox หลัง scene shift (dx, dy).
    predicted_center = last_center - (dx, dy)
    """
    x, y, w, h = bbox
    center = (x + w / 2.0, y + h / 2.0)
    predicted_cx = center[0] - dx
    predicted_cy = center[1] - dy
    px = predicted_cx - w / 2.0
    py = predicted_cy - h / 2.0
    px = float(np.clip(px, 0, max(0, fw - w)))
    py = float(np.clip(py, 0, max(0, fh - h)))
    return (px, py, w, h)


# =============================================================================
# DetectorWorker — async YOLO 640
# =============================================================================

class DetectorWorker:
    """
    Runs YOLO inference in a background thread (single 640 engine).
    Applies frame enhancement (sharpen + CLAHE) before inference.
    Supports YOLO + Thermal fusion (Phase 1) and thermal-only fallback (Phase 2).
    """

    def __init__(self, model, imgsz=640, conf_min=YOLO_CONF_MIN, fusion_enabled=False):
        self.model = model
        self.imgsz = imgsz
        self.conf_min = conf_min
        self.fusion_enabled = fusion_enabled

        self._task = AsyncLatestValue()
        self._result = AsyncLatestValue()
        self._prev_gray = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame, frame_idx, last_bbox=None):
        self._task.put((frame, frame_idx, last_bbox))

    def get_result(self):
        val, is_new = self._result.get()
        if val is None:
            return [], -1, 0.0, False
        dets, fidx, lat = val
        return dets, fidx, lat, is_new

    def _loop(self):
        while True:
            task, is_new = self._task.get()
            if task is None or not is_new:
                time.sleep(0.001)
                continue

            frame = task[0]
            fidx = task[1]
            last_bbox = task[2] if len(task) > 2 else None
            t0 = time.time()

            if len(frame.shape) == 2:
                curr_gray = frame
            else:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fh, fw = frame.shape[:2]

            inp = enhance_frame(frame) if (ENHANCE_SHARPEN or ENHANCE_CLAHE) else frame
            if fw > self.imgsz or fh > self.imgsz:
                ratio = self.imgsz / max(fw, fh)
                inp = cv2.resize(inp, (int(fw * ratio), int(fh * ratio)), interpolation=cv2.INTER_LINEAR)
            dets = detect_yolo_full_frame(self.model, inp, self.conf_min, imgsz=self.imgsz)
            if fw > self.imgsz or fh > self.imgsz:
                inv = 1.0 / ratio
                dets = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv), c) for x, y, w, h, c in dets]

            if self.fusion_enabled:
                if dets:
                    dets = fuse_yolo_thermal(dets, frame, FUSION_THERMAL_ROI_PAD)
                elif THERMAL_FALLBACK_ENABLED and last_bbox is not None:
                    if MOTION_COMPENSATION_ENABLED and self._prev_gray is not None:
                        x, y, w, h = last_bbox
                        if w > 0 and h > 0:
                            dx, dy = compute_scene_shift(self._prev_gray, curr_gray)
                            predicted_bbox = apply_motion_to_bbox(last_bbox, dx, dy, fw, fh)
                            dets = detect_thermal_blobs_in_roi(frame, predicted_bbox, MOTION_PREDICTED_SEARCH_PAD)
                        else:
                            dets = detect_thermal_blobs_in_roi(frame, last_bbox, THERMAL_FALLBACK_SEARCH_PAD)
                    else:
                        dets = detect_thermal_blobs_in_roi(frame, last_bbox, THERMAL_FALLBACK_SEARCH_PAD)

            self._prev_gray = curr_gray.copy()
            latency = (time.time() - t0) * 1000.0
            self._result.put((dets, fidx, latency))

# =============================================================================
# DetectSystem — pure detection, no tracking
# =============================================================================

class DetectSystem:
    """
    Submit every frame to async YOLO, pick best detection, return raw bbox.
    Temporal hold: keep last good detection for MAX_HOLD_FRAMES if YOLO
    misses (e.g. due to motion blur). bbox is always raw YOLO output —
    never interpolated or shifted.
    """

    def __init__(self, detector_worker):
        self.detector = detector_worker
        self.frame_idx = 0

        self.all_detections = []
        self.best_detection = None
        self.det_latency_ms = 0.0
        self._hold_age = 0

    def process(self, frame, frame_ts=None):
        self.frame_idx += 1
        last_bbox = None
        if self.best_detection is not None:
            x, y, w, h, _ = self.best_detection
            last_bbox = (x, y, w, h)
        self.detector.submit(frame, self.frame_idx, last_bbox)

        dets, det_fidx, det_lat, det_new = self.detector.get_result()
        if det_new:
            self.all_detections = dets if dets else []
            self.det_latency_ms = det_lat
            if dets:
                self.best_detection = self._pick_best(dets)
                self._hold_age = 0
            else:
                self._hold_age += 1

        if self._hold_age > MAX_HOLD_FRAMES:
            self.best_detection = None

        target_bbox = None
        target_center = None
        confidence = 0.0
        if self.best_detection is not None:
            x, y, w, h, c = self.best_detection
            target_bbox = (x, y, w, h)
            target_center = (x + w / 2.0, y + h / 2.0)
            confidence = c

        return {
            "target_bbox": target_bbox,
            "target_center": target_center,
            "all_detections": self.all_detections,
            "det_latency_ms": self.det_latency_ms,
            "confidence": confidence,
            "frame_age_ms": (time.time() - frame_ts) * 1000.0 if frame_ts else 0.0,
            "det_new": det_new,
            "hold_age": self._hold_age,
        }

    @staticmethod
    def _pick_best(dets):
        if not dets:
            return None
        return max(dets, key=lambda d: d[4])

# =============================================================================
# Camera builder
# =============================================================================

def build_camera_from_config(camera_name=None):
    name = camera_name if camera_name is not None else ACTIVE_CAMERA
    cfg = get_camera_config(name)
    use_video = cfg.get("use_video_file", False)
    source = cfg["video_filename"] if use_video else cfg["rtsp_url"]
    if use_video and source and not os.path.isabs(source):
        source = os.path.join(os.path.dirname(__file__), source)
    return CameraStream(
        source=source,
        width=cfg["width"],
        height=cfg["height"],
        use_video_file=use_video,
        camera_name=cfg.get("name", name),
        udp_ip=cfg.get("udp_ip"),
        udp_port=cfg.get("udp_port"),
        use_udp_direct=cfg.get("use_udp_direct"),
        stream_format=cfg.get("stream_format"),
    )

# =============================================================================
# Full gun-aim HUD
# =============================================================================

def draw_hud(frame, cx_frame, cy_frame, radius_px, target_bbox, target_center,
             ready_to_fire, guide_h=None, guide_v=None, is_day=None,
             distance_m=None, all_detections=None, confidence=0.0):
    h, w = frame.shape[:2]
    if is_day is None:
        is_day = is_daytime()
    if is_day:
        color_ready = (0, 0, 255)
        color_aim = (0, 0, 0)
    else:
        color_ready = (0, 0, 255)
        color_aim = (0, 255, 0)

    color_use = color_ready if ready_to_fire else color_aim

    cv2.line(frame, (0, cy_frame), (w, cy_frame), color_use, 1)
    cv2.line(frame, (cx_frame, 0), (cx_frame, h), color_use, 1)

    min_side = min(h, w)
    for ratio in RETICLE_RADIUS_RATIOS:
        r = max(4, int(min_side * ratio))
        cv2.circle(frame, (cx_frame, cy_frame), r, color_use, 2)
    cv2.circle(frame, (cx_frame, cy_frame), 2, color_use, -1)

    color_other = (128, 128, 128)
    if all_detections:
        for (dx, dy, dw, dh, dc) in all_detections:
            cv2.rectangle(frame, (int(dx), int(dy)), (int(dx+dw), int(dy+dh)), color_other, 1)

    s = hud_scale(w)

    if target_bbox is not None:
        bx, by, bw, bh = [int(v) for v in target_bbox]
        color = color_use
        color_text_hud = (255, 255, 255) if (color == (0, 0, 0)) else color
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, max(1, int(4 * s)))
        if target_center is not None:
            cx_t, cy_t = int(target_center[0]), int(target_center[1])
            cv2.circle(frame, (cx_t, cy_t), max(2, int(6 * s)), color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_label = 1.2 * s
        th_label = max(1, int(3 * s))
        pad = max(4, int(10 * s))

        conf_pct = min(99, max(0, int(round(confidence * 100))))
        label_drone = f"DRONE {conf_pct}%"
        (tw, th_t), _ = cv2.getTextSize(label_drone, font, scale_label, th_label)
        ty_baseline = max(th_t + pad, by - int(6 * s))
        r_y1 = max(0, ty_baseline - th_t - pad)
        r_x2 = min(w, bx + tw + int(24 * s))
        cv2.rectangle(frame, (bx, r_y1), (r_x2, ty_baseline + pad), (0, 0, 0), -1)
        tx, ty = bx + pad, ty_baseline
        cv2.putText(frame, label_drone, (tx, ty), font, scale_label, (0, 0, 0), th_label + 2)
        cv2.putText(frame, label_drone, (tx, ty), font, scale_label, color_text_hud, th_label)

        if distance_m is not None and distance_m > 0:
            dist_text = f"~{distance_m:.0f} m"
            scale_dist = 1.3 * s
            th_dist = max(1, int(3 * s))
            dy_bottom = min(h - 8, by + bh + int(28 * s))
            cv2.putText(frame, dist_text, (bx, dy_bottom), font, scale_dist, (0, 0, 0), th_dist + 2)
            cv2.putText(frame, dist_text, (bx, dy_bottom), font, scale_dist, color_text_hud, th_dist)

    label = "READY" if ready_to_fire else "THINGS"
    if ready_to_fire:
        color_text = (255, 255, 255) if (color_use == (0, 0, 0)) else color_use
    else:
        color_text = (255, 0, 0)
    cv2.putText(frame, label, (int(20 * s), int(58 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, color_text, max(1, int(3 * s)))

    arrow_r = max(4, int(min_side * RETICLE_RADIUS_RATIOS[-2]))
    arrow_y_center = cy_frame + arrow_r + int(35 * s)
    al = min(int(ARROW_LEN * s), w // 5)
    thickness = max(1, int(ARROW_THICKNESS * s))
    tip_len = 0.25
    if guide_h == "RIGHT":
        cv2.arrowedLine(frame, (cx_frame - al, arrow_y_center), (cx_frame + al, arrow_y_center), color_use, thickness, tipLength=tip_len)
    elif guide_h == "LEFT":
        cv2.arrowedLine(frame, (cx_frame + al, arrow_y_center), (cx_frame - al, arrow_y_center), color_use, thickness, tipLength=tip_len)
    if guide_v == "DOWN":
        cv2.arrowedLine(frame, (cx_frame, arrow_y_center - al), (cx_frame, arrow_y_center + al), color_use, thickness, tipLength=tip_len)
    elif guide_v == "UP":
        cv2.arrowedLine(frame, (cx_frame, arrow_y_center + al), (cx_frame, arrow_y_center - al), color_use, thickness, tipLength=tip_len)

    return frame


def draw_det_info(frame, det_result, display_fps, det_fps):
    fh, fw = frame.shape[:2]
    s = hud_scale(fw)
    det_lat = det_result.get("det_latency_ms", 0)
    frame_age = det_result.get("frame_age_ms", 0)
    has_target = det_result["target_bbox"] is not None

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_color = (0, 255, 0) if has_target else (0, 255, 255)

    status = "DETECTED" if has_target else "SCANNING"
    lines = [
        f"{status}  FPS:{display_fps:.0f}  Det:{det_fps:.0f}",
        f"Lat:{det_lat:.0f}ms  Age:{frame_age:.0f}ms",
    ]
    font_scale = 0.6 * s
    line_gap = max(16, int(26 * s))
    y0 = fh - int(100 * s)
    for i, line in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(frame, line, (int(12 * s), y), font, font_scale, (0, 0, 0), max(1, int(3 * s)))
        cv2.putText(frame, line, (int(12 * s), y), font, font_scale, status_color, max(1, int(1 * s)))


def draw_hint_keys_normal(frame, w, h):
    s = hud_scale(w)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = 1.0 * s
    scale2 = 0.85 * s
    thickness = max(1, int(2 * s))
    pad_box = int(20 * s)
    line1 = f"Press [{KEY_ADJUST_CENTER}] Aim center  [{KEY_SETTINGS}] Ballistics  [{KEY_QUIT}] Quit"
    line2 = "Settings: C = aim center, S = ballistics, Enter = save & exit"
    (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thickness)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thickness)
    x = int(20 * s)
    y1 = h - int(85 * s)
    y2 = h - int(44 * s)
    box_w = min(w - x - 10, max(tw1, tw2) + pad_box)
    cv2.rectangle(frame, (x, y1 - th1 - int(6 * s)), (x + box_w, y2 + int(6 * s)), (0, 0, 0), -1)
    cv2.putText(frame, line1, (x + int(6 * s), y1), font, scale1, (255, 255, 255), thickness)
    cv2.putText(frame, line2, (x + int(6 * s), y2), font, scale2, (200, 200, 200), thickness)


def draw_adjust_center_hud(frame, config, w, h):
    s = hud_scale(w)
    cx, cy = config.get_center(w, h)
    color = (0, 255, 0)
    min_side = min(h, w)
    for ratio in RETICLE_RADIUS_RATIOS:
        r = max(4, int(min_side * ratio))
        cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 2, color, -1)
    cv2.line(frame, (0, cy), (w, cy), color, 1)
    cv2.line(frame, (cx, 0), (cx, h), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95 * s
    thickness = max(1, int(2 * s))
    lines = [
        "Aim center setup",
        "Arrow / W A S D: Move reticle",
        "R: Reset to screen center",
        "Enter: Save and exit",
    ]
    y0 = int(68 * s)
    step = max(18, int(36 * s))
    for i, line in enumerate(lines):
        y = y0 + i * step
        cv2.putText(frame, line, (int(20 * s), y), font, scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, line, (int(20 * s), y), font, scale, (255, 255, 255), thickness)
    bottom_hint = "Keys: Arrow/WASD move, R reset, Enter save & exit"
    hint_scale = 0.9 * s
    (tw, th), _ = cv2.getTextSize(bottom_hint, font, hint_scale, thickness)
    by = h - int(32 * s)
    bx = int(20 * s)
    box_w = min(w - bx - 10, tw + int(20 * s))
    cv2.rectangle(frame, (bx, by - th - int(6 * s)), (bx + box_w, by + int(6 * s)), (0, 0, 0), -1)
    cv2.putText(frame, bottom_hint, (bx + int(6 * s), by), font, hint_scale, (255, 255, 255), thickness)


def draw_settings_overlay(frame, config, selected_field):
    h, w = frame.shape[:2]
    s = hud_scale(w)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.05 * s
    thickness = max(1, int(2 * s))
    pad_x = int(56 * s)
    title = "Ballistics (marksman)"
    title_scale = 1.2 * s
    title_th = max(1, int(2 * s))
    fields = [
        ("Effective range (m)", config.effective_range_m),
        ("Muzzle velocity (m/s)", config.muzzle_velocity_ms),
        ("Bullet weight (g)", config.bullet_weight_g),
        ("Target size (m)", config.target_size_m),
    ]
    footer = "1/2/3/4: Select  Up/Down: +/-  Enter: Save"
    footer_scale = 0.85 * s
    (tw_title, th_title), _ = cv2.getTextSize(title, font, title_scale, title_th)
    max_fw = tw_title
    for label, value in fields:
        (fw_t, _), _ = cv2.getTextSize(f"  > {label}: {value:.1f}", font, scale, thickness)
        max_fw = max(max_fw, fw_t)
    (tw_foot, th_foot), _ = cv2.getTextSize(footer, font, footer_scale, title_th)
    max_fw = max(max_fw, tw_foot)
    line_h = max(24, int(48 * s))
    panel_w = max_fw + pad_x
    panel_h = th_title + int(24 * s) + len(fields) * line_h + int(20 * s) + th_foot + int(16 * s)
    panel_w = min(w - int(40 * s), max(int(420 * s), panel_w))
    panel_h = min(h - int(80 * s), max(int(320 * s), panel_h))
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, title, (x0 + int(24 * s), y0 + int(40 * s)), font, title_scale, (255, 255, 255), title_th)
    y = y0 + int(82 * s)
    for i, (label, value) in enumerate(fields):
        marker = ">" if i == selected_field else " "
        text = f"  {marker} {label}: {value:.2f}" if i == 3 else f"  {marker} {label}: {value:.1f}"
        color = (0, 255, 255) if i == selected_field else (220, 220, 220)
        cv2.putText(frame, text, (x0 + int(24 * s), y), font, scale, color, thickness)
        y += line_h
    cv2.putText(frame, footer, (x0 + int(24 * s), y0 + panel_h - int(32 * s)), font, footer_scale, (180, 180, 180), title_th)
    bottom_hint = "Keys: 1/2/3/4 select  Up/Down +/-  Enter save & exit"
    hint_scale = 0.95 * s
    (tw, th), _ = cv2.getTextSize(bottom_hint, font, hint_scale, thickness)
    by = h - int(32 * s)
    bx = int(20 * s)
    box_w = min(w - bx - 10, tw + int(20 * s))
    cv2.rectangle(frame, (bx, by - th - int(6 * s)), (bx + box_w, by + int(6 * s)), (0, 0, 0), -1)
    cv2.putText(frame, bottom_hint, (bx + int(6 * s), by), font, hint_scale, (255, 255, 255), thickness)


# =============================================================================
# main()
# =============================================================================

def main():
    camera_name = CAMERA_NAME if CAMERA_NAME is not None else ACTIVE_CAMERA
    cam_config = get_camera_config(camera_name)
    fov_h = cam_config.get("fov_horizontal", 60.0)
    fov_v = cam_config.get("fov_vertical", 36.0)
    print(f"Gun Aim Assist: starting camera '{camera_name}'")
    cam = build_camera_from_config(camera_name)
    cam.start()
    time.sleep(0.5)

    print("Loading YOLO model...")
    engine_path = os.path.join(os.path.dirname(__file__), YOLO_ENGINE)
    yolo_model = None
    if os.path.exists(engine_path):
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            yolo_model = UltralyticsYOLO(engine_path, task="detect")
            dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
            yolo_model.predict(dummy, verbose=False, device=0, imgsz=YOLO_IMGSZ)
            print(f"Gun Aim Assist: YOLO {YOLO_IMGSZ} loaded (fast mode)")
        except Exception as e:
            print(f"YOLO engine load failed: {e}")
            yolo_model = None

    if yolo_model is None:
        result = load_yolo_model()
        if result is None or result[0] is None:
            print("YOLO model not available. Exiting.")
            cam.release()
            return
        yolo_model, _ = result
        print(f"Gun Aim Assist: YOLO loaded (fallback model)")

    conf_min = YOLO_CONF_CANDIDATE if FUSION_ENABLED else YOLO_CONF_MIN
    detector = DetectorWorker(
        model=yolo_model,
        imgsz=YOLO_IMGSZ,
        conf_min=conf_min,
        fusion_enabled=FUSION_ENABLED,
    )
    system = DetectSystem(detector)

    config = ShooterConfig()
    config.load()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()

    app_mode = "normal"
    settings_selected_field = 0
    settings_step = (10, 50, 0.5, 0.05)
    settings_min_max = ((10, 500), (200, 1500), (1, 50), (0.1, 2.0))

    fps_times = deque(maxlen=60)
    det_timestamps = deque(maxlen=30)
    prev_ready_to_fire = False
    last_ready_sound_time = 0.0
    last_approach_beep_time = 0.0
    loop_frames = 0
    det_fps = 0.0
    GRACE_FRAMES = 60

    print("Entering main loop... (Q quit, C aim center, S ballistics)")

    while True:
        t0 = time.time()
        active, frame, frame_ts = cam.read()
        if not active or frame is None:
            time.sleep(0.01)
            continue

        loop_frames += 1

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        fh, fw = frame.shape[:2]
        cx_frame, cy_frame = config.get_center(fw, fh)

        min_side = min(fh, fw)
        radius_px = int(min_side * RETICLE_RADIUS_RATIOS[0])
        if CENTER_RADIUS_PX > 0:
            radius_px = CENTER_RADIUS_PX
        radius_px = max(10, radius_px)

        if app_mode == "normal":
            det_result = system.process(frame, frame_ts)

            elapsed = time.time() - t0
            fps_times.append(elapsed)
            display_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            if det_result.get("det_new") and det_result["det_latency_ms"] > 0:
                det_timestamps.append(time.time())
            if len(det_timestamps) >= 2:
                span = det_timestamps[-1] - det_timestamps[0]
                det_fps = (len(det_timestamps) - 1) / span if span > 0 else 0
            else:
                det_fps = 0.0

            target_bbox = det_result["target_bbox"]
            target_center = det_result["target_center"]
            all_dets = det_result.get("all_detections", [])
            confidence = det_result.get("confidence", 0.0)

            ready_to_fire = False
            if target_center is not None:
                cx_t, cy_t = target_center
                d = math.sqrt((cx_t - cx_frame) ** 2 + (cy_t - cy_frame) ** 2)
                ready_to_fire = d <= radius_px

            now = time.time()
            if ready_to_fire and SOUND_ON_READY:
                if not prev_ready_to_fire:
                    _play_ready_sound()
                    last_ready_sound_time = now
                elif now - last_ready_sound_time >= SOUND_READY_INTERVAL:
                    _play_ready_sound()
                    last_ready_sound_time = now
            elif target_bbox is not None and target_center is not None and SOUND_APPROACH_ON:
                bx, by, bw, bh = target_bbox
                cx_t, cy_t = target_center
                d = math.sqrt((cx_t - cx_frame) ** 2 + (cy_t - cy_frame) ** 2)
                interval = compute_approach_beep_interval(
                    d, radius_px, bw, bh,
                    SOUND_BEEP_SLOW_INTERVAL, SOUND_BEEP_FAST_INTERVAL
                )
                if interval is not None and (now - last_approach_beep_time) >= interval:
                    _play_ready_sound()
                    last_approach_beep_time = now
            prev_ready_to_fire = ready_to_fire

            guide_h, guide_v = compute_guide_direction(
                target_center, cx_frame, cy_frame, radius_px, ready_to_fire
            )

            distance_m = None
            if target_bbox is not None:
                bx, by, bw, bh = target_bbox
                distance_m = estimate_distance_m(bw, bh, fw, fh, fov_h, fov_v, config.target_size_m)

            draw_hud(frame, cx_frame, cy_frame, radius_px, target_bbox, target_center,
                     ready_to_fire, guide_h, guide_v, is_day=is_daytime(),
                     distance_m=distance_m, all_detections=all_dets, confidence=confidence)
            draw_det_info(frame, det_result, display_fps, det_fps)
            draw_hint_keys_normal(frame, fw, fh)

        elif app_mode == "adjust_center":
            draw_adjust_center_hud(frame, config, fw, fh)
        elif app_mode == "settings":
            draw_settings_overlay(frame, config, settings_selected_field)

        s_fps = hud_scale(fw)
        fps_text = f"DET FPS: {det_fps:.1f}"
        font_fps = cv2.FONT_HERSHEY_SIMPLEX
        fps_scale = 1.2 * s_fps
        fps_th = max(1, int(2 * s_fps))
        (fps_tw, _), _ = cv2.getTextSize(fps_text, font_fps, fps_scale, fps_th)
        cv2.putText(frame, fps_text, (fw - fps_tw - int(28 * s_fps), int(42 * s_fps)), font_fps, fps_scale, (255, 255, 255), fps_th)

        max_w = min(screen_w, DISPLAY_MAX_W)
        max_h = min(screen_h, DISPLAY_MAX_H)
        scale = min(max_w / fw, max_h / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        if (dw, dh) != (fw, fh):
            display_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame

        try:
            cv2.resizeWindow(WINDOW_NAME, dw, dh)
        except cv2.error:
            pass
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1)
        if key == -1:
            pass
        elif key in (ord("q"), ord("Q")) and loop_frames >= GRACE_FRAMES:
            break
        elif app_mode == "normal":
            if key in (ord("c"), ord("C")):
                app_mode = "adjust_center"
            elif key in (ord("s"), ord("S")):
                app_mode = "settings"
        elif app_mode == "adjust_center":
            step = AIM_CENTER_STEP_PX
            if key in (83, 65363, ord("d")):
                config.move(step, 0)
            elif key in (81, 65361, ord("a")):
                config.move(-step, 0)
            elif key in (82, 65362, ord("w")):
                config.move(0, -step)
            elif key in (84, 65364, ord("s")):
                config.move(0, step)
            elif key in (ord("r"), ord("R")):
                config.reset()
            elif key in (13, 10):
                config.save()
                app_mode = "normal"
        elif app_mode == "settings":
            if key in (ord("1"),):
                settings_selected_field = 0
            elif key in (ord("2"),):
                settings_selected_field = 1
            elif key in (ord("3"),):
                settings_selected_field = 2
            elif key in (ord("4"),):
                settings_selected_field = 3
            elif key in (82, 65362):
                idx = settings_selected_field
                st = settings_step[idx]
                _min, _max = settings_min_max[idx]
                if idx == 0:
                    config.effective_range_m = min(_max, config.effective_range_m + st)
                elif idx == 1:
                    config.muzzle_velocity_ms = min(_max, config.muzzle_velocity_ms + st)
                elif idx == 2:
                    config.bullet_weight_g = min(_max, config.bullet_weight_g + st)
                else:
                    config.target_size_m = min(_max, config.target_size_m + st)
            elif key in (84, 65364):
                idx = settings_selected_field
                st = settings_step[idx]
                _min, _max = settings_min_max[idx]
                if idx == 0:
                    config.effective_range_m = max(_min, config.effective_range_m - st)
                elif idx == 1:
                    config.muzzle_velocity_ms = max(_min, config.muzzle_velocity_ms - st)
                elif idx == 2:
                    config.bullet_weight_g = max(_min, config.bullet_weight_g - st)
                else:
                    config.target_size_m = max(_min, config.target_size_m - st)
            elif key in (13, 10):
                config.save()
                app_mode = "normal"

    cam.release()
    cv2.destroyAllWindows()
    print("Gun Aim Assist: stopped.")


if __name__ == "__main__":
    main()

