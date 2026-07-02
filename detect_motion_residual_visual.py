"""
detect_motion_residual_visual.py — แสดงบริเวณที่ขยับต่างจากพื้นหลัง (Farneback GPU + residual)

หลักการ: flow - global_motion = residual → พื้นที่ที่ residual สูง = วัตถุเคลื่อนที่
- ใช้ GPU Farneback optical flow
- แสดง overlay สีเขียวบนบริเวณที่ขยับ (ยังไม่ใช้ YOLO)
- เน้น FPS สูงสุด
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

# =============================================================================
# Parameters
# =============================================================================

CAMERA_NAME = None  # หรือ "cam6" สำหรับ thermal

# --- Process resolution (thermal max 640x512) ---
PROCESS_MAX_W = 480
PROCESS_MAX_H = 384

# --- Residual threshold ---
RESIDUAL_THRESH = 1.5   # px — residual magnitude เกินนี้ = moving

# --- Farneback params (winSize ต้องเป็นเลขคี่) ---
FARNEBACK_PYR_SCALE = 0.5
FARNEBACK_LEVELS = 1
FARNEBACK_WIN_SIZE = 7
FARNEBACK_ITERS = 1
FARNEBACK_POLY_N = 5
FARNEBACK_POLY_SIGMA = 1.1
MEDIAN_SAMPLE_STEP = 8

# --- Display ---
OVERLAY_ALPHA = 0.6     # ความโปร่งใสของ overlay
OVERLAY_COLOR = (0, 255, 0)  # BGR — สีเขียว = moving

# =============================================================================
# GPU Farneback + Residual
# =============================================================================

def has_cuda_farneback():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, 'FarnebackOpticalFlow')
    except Exception:
        return False

def create_farneback_gpu():
    """สร้าง Farneback optical flow บน GPU"""
    try:
        if hasattr(cv2.cuda, 'FarnebackOpticalFlow'):
            farn = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=FARNEBACK_LEVELS,
                pyrScale=FARNEBACK_PYR_SCALE,
                fastPyramids=False,
                winSize=FARNEBACK_WIN_SIZE,
                numIters=FARNEBACK_ITERS,
                polyN=FARNEBACK_POLY_N,
                polySigma=FARNEBACK_POLY_SIGMA,
                flags=0
            )
            return farn
    except Exception as e:
        print(f"Farneback GPU create error: {e}")
    return None

def compute_residual_mask_gpu(prev_gray, curr_gray, farn, gpu_prev, gpu_curr, gpu_flow):
    """คำนวณ mask ของบริเวณที่ขยับต่างจากพื้นหลัง (GPU)"""
    try:
        t0 = time.time()
        gpu_prev.upload(prev_gray)
        gpu_curr.upload(curr_gray)
        farn.calc(gpu_prev, gpu_curr, gpu_flow)

        flow = gpu_flow.download()
        if flow is None or flow.size == 0:
            return None, 0

        fx = flow[:, :, 0]
        fy = flow[:, :, 1]
        step = MEDIAN_SAMPLE_STEP
        gx = np.median(fx[::step, ::step].ravel())
        gy = np.median(fy[::step, ::step].ravel())
        rx = fx - gx
        ry = fy - gy
        mag = np.sqrt(rx * rx + ry * ry)

        _, binary = cv2.threshold(mag.astype(np.float32), RESIDUAL_THRESH, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel)

        return binary, (time.time() - t0) * 1000.0
    except Exception as e:
        print(f"Residual GPU error: {e}")
        return None, 0

# =============================================================================
# CPU fallback
# =============================================================================

def compute_residual_mask_cpu(prev_gray, curr_gray):
    """Fallback CPU Farneback"""
    try:
        t0 = time.time()
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=FARNEBACK_PYR_SCALE,
            levels=FARNEBACK_LEVELS,
            winsize=FARNEBACK_WIN_SIZE,
            iterations=FARNEBACK_ITERS,
            poly_n=FARNEBACK_POLY_N,
            poly_sigma=FARNEBACK_POLY_SIGMA,
            flags=0
        )
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        step = MEDIAN_SAMPLE_STEP
        gx = np.median(fx[::step, ::step].ravel())
        gy = np.median(fy[::step, ::step].ravel())
        rx, ry = fx - gx, fy - gy
        mag = np.sqrt(rx * rx + ry * ry)
        _, binary = cv2.threshold(mag.astype(np.float32), RESIDUAL_THRESH, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel)
        return binary, (time.time() - t0) * 1000.0
    except Exception as e:
        print(f"Residual CPU error: {e}")
        return None, 0

# =============================================================================
# Camera
# =============================================================================

def build_camera(camera_name=None):
    name = camera_name or ACTIVE_CAMERA
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
# main()
# =============================================================================

def main():
    camera_name = CAMERA_NAME or ACTIVE_CAMERA
    print(f"Motion Residual Visual: camera '{camera_name}'")
    print("แสดงบริเวณที่ขยับต่างจากพื้นหลัง (สีเขียว = moving)")
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
    proc_times = deque(maxlen=30)

    cv2.namedWindow("Motion Residual", cv2.WINDOW_NORMAL)

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
        lat_ms = 0.0

        if prev_gray is not None and prev_gray.shape == curr_s.shape:
            if use_gpu and farn:
                if gpu_prev is None:
                    h, w = curr_s.shape[:2]
                    gpu_prev = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_curr = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                    gpu_flow = cv2.cuda_GpuMat(h, w, cv2.CV_32FC2)
                mask, lat_ms = compute_residual_mask_gpu(prev_gray, curr_s, farn, gpu_prev, gpu_curr, gpu_flow)
            else:
                mask, lat_ms = compute_residual_mask_cpu(prev_gray, curr_s)

        prev_gray = curr_s.copy()

        if mask is not None:
            if curr_s.shape[:2] != (fh, fw):
                mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
            if mask_only:
                display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                overlay = np.zeros_like(frame)
                overlay[mask > 0] = OVERLAY_COLOR
                display = cv2.addWeighted(frame, 1.0, overlay, OVERLAY_ALPHA, 0)
        else:
            display = frame.copy()

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        proc_times.append(lat_ms)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        avg_lat = sum(proc_times) / len(proc_times) if proc_times else 0

        cv2.putText(display, f"FPS: {fps:.0f}  Det: {avg_lat:.0f}ms  {'GPU' if use_gpu else 'CPU'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, "Green = moving (residual)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Motion Residual", display)

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

