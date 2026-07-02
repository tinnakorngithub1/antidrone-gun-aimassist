"""
detect_motion_yolo_fusion.py — Motion residual + YOLO full-frame + fusion + thermal fallback

- Motion residual (Farneback GPU) → mask สีเขียว = moving
- YOLO full-frame → detections
- Fusion: boost score เมื่อ bbox overlap กับ motion mask
- Thermal fallback: เมื่อ YOLO conf ต่ำ ใช้ detect_motion_shape (contours จาก mask)
"""

import cv2
import numpy as np
import time
import os
import sys
from collections import deque

try:
    from fast_motion_sky import CameraStream
    from config import get_camera_config, ACTIVE_CAMERA
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from detect_motion_residual_visual import (
    has_cuda_farneback,
    create_farneback_gpu,
    compute_residual_mask_gpu,
    compute_residual_mask_cpu,
    build_camera,
    PROCESS_MAX_W,
    PROCESS_MAX_H,
    RESIDUAL_THRESH,
    FARNEBACK_PYR_SCALE,
    FARNEBACK_LEVELS,
    FARNEBACK_WIN_SIZE,
    FARNEBACK_ITERS,
    FARNEBACK_POLY_N,
    FARNEBACK_POLY_SIGMA,
    MEDIAN_SAMPLE_STEP,
    OVERLAY_ALPHA,
    OVERLAY_COLOR,
)
from smart_detection_yolo_only import load_yolo_model, detect_yolo_full_frame

# =============================================================================
# Parameters
# =============================================================================

CAMERA_NAME = None  # หรือ "cam6" สำหรับ thermal

# --- YOLO ---
YOLO_ENGINE = "last_imgsz640.engine"
YOLO_IMGSZ = 640
YOLO_CONF_MIN = 0.2
ATTENTION_WEIGHT = 0.5
YOLO_ENABLED = True  # ปิดได้เพื่อดู motion อย่างเดียว

# --- Thermal fallback ---
YOLO_CONF_THERMAL_THRESH = 0.4  # ถ้า YOLO best < ค่านี้ → ใช้ shape fallback
SHAPE_MIN_AREA = 50
SHAPE_MAX_AREA = 5000
SHAPE_ASPECT_MIN = 0.4
SHAPE_ASPECT_MAX = 2.5
SHAPE_CONF = 0.5  # confidence ที่ให้เมื่อใช้ shape

# =============================================================================
# Fusion: motion_overlap_ratio, fuse_motion_yolo
# =============================================================================

def motion_overlap_ratio(bbox, mask):
    """สัดส่วนของ bbox ที่ overlap กับ motion mask (0-1)"""
    x, y, w, h = bbox
    fh, fw = mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return np.sum(roi > 0) / roi.size


def fuse_motion_yolo(dets, mask, attention_weight):
    """
    รับ: dets [(x,y,w,h,conf), ...], mask (binary), attention_weight
    คืน (best_det, fused_dets) — best คืออันที่ fusion_score สูงสุด
    """
    if not dets or mask is None:
        best = max(dets, key=lambda d: d[4]) if dets else None
        return best, dets
    fused = []
    for det in dets:
        x, y, w, h, conf = det
        overlap = motion_overlap_ratio((x, y, w, h), mask)
        fusion_score = conf * (1.0 + attention_weight * overlap)
        fused.append((x, y, w, h, fusion_score))
    best = max(fused, key=lambda d: d[4])
    # คืน best ในรูปแบบ (x, y, w, h, conf) โดยใช้ conf เดิมของ detection นั้น
    best_det = (best[0], best[1], best[2], best[3], dets[fused.index(best)][4])
    return best_det, fused


# =============================================================================
# Thermal fallback: detect_motion_shape
# =============================================================================

def detect_motion_shape(mask, fw, fh):
    """
    จาก motion mask หา contour ที่น่าจะเป็นเป้า (area, aspect ratio)
    คืน [(x, y, w, h, score), ...] เรียงจากคะแนนสูงไปต่ำ
    """
    if mask is None or mask.size == 0:
        return []
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    center_x, center_y = fw / 2.0, fh / 2.0
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2) or 1.0  # ระยะจากศูนย์ถึงมุม
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < SHAPE_MIN_AREA or area > SHAPE_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        aspect = w / float(h)
        if aspect < SHAPE_ASPECT_MIN or aspect > SHAPE_ASPECT_MAX:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        score = 0.5 + 0.5 * (1.0 - dist / max_dist)
        score = min(1.0, max(0.0, score))
        candidates.append((x, y, w, h, score))
    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates


# =============================================================================
# YOLO loading (engine ก่อน แล้ว fallback load_yolo_model)
# =============================================================================

def load_yolo_for_fusion():
    """โหลด YOLO: engine ก่อน ถ้าไม่มีใช้ load_yolo_model()"""
    engine_path = os.path.join(os.path.dirname(__file__), YOLO_ENGINE)
    yolo_model = None
    imgsz = YOLO_IMGSZ
    if os.path.exists(engine_path):
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            yolo_model = UltralyticsYOLO(engine_path, task="detect")
            dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
            yolo_model.predict(dummy, verbose=False, device=0, imgsz=YOLO_IMGSZ)
            print(f"Motion YOLO Fusion: YOLO {YOLO_IMGSZ} loaded (engine)")
        except Exception as e:
            print(f"YOLO engine load failed: {e}")
            yolo_model = None
    if yolo_model is None:
        result = load_yolo_model()
        if result is None or result[0] is None:
            return None, YOLO_IMGSZ
        yolo_model, imgsz = result[0], result[1] if result[1] is not None else YOLO_IMGSZ
        print(f"Motion YOLO Fusion: YOLO loaded (fallback, imgsz={imgsz})")
    return yolo_model, imgsz


# =============================================================================
# main()
# =============================================================================

def main():
    camera_name = CAMERA_NAME or ACTIVE_CAMERA
    print(f"Motion YOLO Fusion: camera '{camera_name}'")
    print("Motion residual + YOLO + fusion; thermal fallback = Motion+Shape")
    print("Q = quit, M = toggle mask only")

    cam = build_camera(camera_name)
    cam.start()
    time.sleep(0.5)

    use_gpu = has_cuda_farneback()
    farn = create_farneback_gpu() if use_gpu else None
    use_gpu = use_gpu and farn is not None
    if use_gpu:
        print("✅ Using GPU Farneback optical flow")
    else:
        print("⚠️ Using CPU Farneback (slower)")

    gpu_prev = gpu_curr = gpu_flow = None
    prev_gray = None
    mask_only = False
    fps_times = deque(maxlen=30)
    motion_times = deque(maxlen=30)
    yolo_times = deque(maxlen=30)

    yolo_model = None
    yolo_imgsz = YOLO_IMGSZ
    if YOLO_ENABLED:
        yolo_model, yolo_imgsz = load_yolo_for_fusion()
        if yolo_model is None:
            print("⚠️ YOLO not available; using Motion+Shape only")

    cv2.namedWindow("Motion YOLO Fusion", cv2.WINDOW_NORMAL)

    while True:
        t0 = time.time()
        active, frame, _ = cam.read()
        if not active or frame is None:
            time.sleep(0.01)
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        fh, fw = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if fw > PROCESS_MAX_W or fh > PROCESS_MAX_H:
            ratio = min(PROCESS_MAX_W / fw, PROCESS_MAX_H / fh)
            new_w = int(fw * ratio)
            new_h = int(fh * ratio)
            curr_s = cv2.resize(curr_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            curr_s = curr_gray

        mask = None
        lat_motion = 0.0

        if prev_gray is not None and prev_gray.shape == curr_s.shape:
            if use_gpu and farn:
                if gpu_prev is None:
                    h, w = curr_s.shape[:2]
                    gpu_prev = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_curr = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_flow = cv2.cuda_GpuMat(h, w, cv2.CV_32FC2)
                mask, lat_motion = compute_residual_mask_gpu(
                    prev_gray, curr_s, farn, gpu_prev, gpu_curr, gpu_flow
                )
            else:
                mask, lat_motion = compute_residual_mask_cpu(prev_gray, curr_s)

        prev_gray = curr_s.copy()

        if mask is not None and curr_s.shape[:2] != (fh, fw):
            mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        best_det = None
        t_yolo0 = time.time()
        if YOLO_ENABLED and yolo_model is not None:
            dets = detect_yolo_full_frame(
                yolo_model, frame, YOLO_CONF_MIN, imgsz=yolo_imgsz
            )
            lat_yolo = (time.time() - t_yolo0) * 1000.0
            yolo_times.append(lat_yolo)
            if dets:
                best_yolo = max(dets, key=lambda d: d[4])
                if best_yolo[4] >= YOLO_CONF_THERMAL_THRESH:
                    best_det, _ = fuse_motion_yolo(dets, mask, ATTENTION_WEIGHT)
                else:
                    shape_dets = detect_motion_shape(mask, fw, fh)
                    best_det = shape_dets[0] if shape_dets else None
            else:
                shape_dets = detect_motion_shape(mask, fw, fh)
                best_det = shape_dets[0] if shape_dets else None
        else:
            lat_yolo = 0.0
            shape_dets = detect_motion_shape(mask, fw, fh)
            best_det = shape_dets[0] if shape_dets else None

        motion_times.append(lat_motion)

        if mask_only and mask is not None:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            if mask is not None:
                overlay = np.zeros_like(frame)
                overlay[mask > 0] = OVERLAY_COLOR
                display = cv2.addWeighted(frame, 1.0, overlay, OVERLAY_ALPHA, 0)
            else:
                display = frame.copy()

        if best_det is not None:
            x, y, w, h, conf = best_det
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                display, f"DRONE {conf*100:.0f}%",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        avg_motion = sum(motion_times) / len(motion_times) if motion_times else 0
        avg_yolo = sum(yolo_times) / len(yolo_times) if yolo_times else 0

        cv2.putText(
            display, f"FPS: {fps:.0f}  Motion: {avg_motion:.0f}ms  YOLO: {avg_yolo:.0f}ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        cv2.putText(
            display, "Green = moving | Red = best detection",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("Motion YOLO Fusion", display)

        key = cv2.waitKey(1)
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("m"), ord("M")):
            mask_only = not mask_only

    cam.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

