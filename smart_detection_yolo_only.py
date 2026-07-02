import cv2
import numpy as np
import time
import sys
import os
from collections import deque

try:
    from fast_motion_sky import CameraStream
    from config import get_camera_config
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics library not found. YOLO detection will be disabled.")
    print("   Install with: pip install ultralytics")

# ============================================================================
# Configuration Parameters (ของไฟล์นี้เอง)
# ============================================================================

# ชื่อ class ตรงกับ CLASS_NAMES ในโน้ตบุ๊กเทรน multiclass RGB (id 0..7)
YOLO_CLASS_NAMES = (
    "drone",
    "person",
    "dog",
    "bird",
    "airplane",
    "car",
    "motorcycle",
    "boat",
)


def _canonical_class_label(cls_id):
    try:
        i = int(cls_id)
    except (TypeError, ValueError):
        return "unknown"
    if 0 <= i < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[i]
    return "unknown"


# YOLO Model Path
YOLO_MODEL_FILE = "yolo_11n_day_night_200_2_imgsz640.engine"  # หรือระบุ path เอง

# YOLO Input Size (detect from filename or set manually)
YOLO_IMG_SIZE = 1280  # 640, 1280, or 1920

# YOLO Detection Parameters
YOLO_CONF_THRESHOLD = 0.001  # 🔥 ลดลงเป็น 0.001 สำหรับ detect 1-2 pixel
YOLO_MAX_DETECTIONS = 30
YOLO_INTERVAL = 2  # 🔥 ทำ YOLO ทุก 5 เฟรม (เพิ่ม FPS)

# Display Parameters
WINDOW_NAME = "YOLO Detection Only"
SHOW_CONFIDENCE = True  # แสดง confidence score
BBOX_COLOR = (0, 0, 255)  # Red color (BGR)
BBOX_THICKNESS = 2

# 🔥 Display resolution (แสดงผลแค่ Full HD เพื่อเพิ่ม FPS)
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# ============================================================================
# YOLO Functions
# ============================================================================

def load_yolo_model():
    """
    โหลด YOLO model จาก YOLO_MODEL_FILE
    Returns:
        yolo_model: YOLO model object หรือ None ถ้าโหลดไม่ได้
    """
    if not YOLO_AVAILABLE:
        return None

    model_path = os.path.join(os.path.dirname(__file__), YOLO_MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"⚠️ YOLO model file not found: {model_path}")
        print("   YOLO detection will be disabled")
        return None

    try:
        print(f"🚀 Loading YOLO model: {model_path}")
        yolo_model = YOLO(model_path, task='detect')

        # 🔥 Detect imgsz from filename or use default
        imgsz = YOLO_IMG_SIZE
        if 'imgsz640' in model_path:
            imgsz = 640
        elif 'imgsz1280' in model_path:
            imgsz = 1280
        elif 'imgsz1920' in model_path:
            imgsz = 1920

        # Warm up model with dummy frame (ใช้ขนาดที่ถูกต้อง)
        dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        yolo_model.predict(dummy_frame, verbose=False, device=0, imgsz=imgsz)
        print(f"✅ YOLO model loaded and warmed up (imgsz={imgsz})")
        return yolo_model, imgsz  # 🔥 Return imgsz ด้วย
    except Exception as e:
        print(f"⚠️ Failed to load YOLO model: {e}")
        print("   YOLO detection will be disabled")
        return None, None

def detect_yolo_full_frame(yolo_model, frame, conf_threshold, imgsz=1280, max_det=None):
    """
    ตรวจจับวัตถุทั้งเฟรมด้วย YOLO

    Args:
        yolo_model: YOLO model object
        frame: Input frame (numpy array)
        conf_threshold: Confidence threshold
        imgsz: Input image size (default: 1280)
        max_det: Maximum detections (None = use YOLO_MAX_DETECTIONS)

    Returns:
        List of detections: [(x, y, w, h, conf, cls_id, cls_name), ...] หรือ [] ถ้าไม่เจอ
    """
    if yolo_model is None:
        return []

    if max_det is None:
        max_det = YOLO_MAX_DETECTIONS

    try:
        yolo_frame = frame
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            yolo_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = yolo_model.predict(
            yolo_frame,
            conf=conf_threshold,
            verbose=False,
            device=0,
            max_det=max_det,
            imgsz=imgsz,
            half=True,
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            # 🔥 Optimize: Batch process boxes (ลด CPU-GPU transfer)
            if len(boxes) > 0:
                # Get all boxes data at once (เร็วกว่า loop)
                xyxy = boxes.xyxy.cpu().numpy()  # Batch transfer
                conf = boxes.conf.cpu().numpy()  # Batch transfer
                cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(boxes),), dtype=np.float32)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]
                    c = float(conf[i])
                    x = int(x1)
                    y = int(y1)
                    w = max(1, int(x2 - x1))  # 🔥 รับ 1 pixel minimum
                    h = max(1, int(y2 - y1))  # 🔥 รับ 1 pixel minimum
                    cls_id = int(cls[i]) if i < len(cls) else 0
                    cls_name = _canonical_class_label(cls_id)
                    detections.append((x, y, w, h, c, cls_id, cls_name))

        return detections
    except Exception as e:
        print(f"⚠️ YOLO detection error: {e}")
        return []

def draw_yolo_detections(frame, yolo_detections):
    """
    วาด red bbox สำหรับ YOLO detections (optimized for tiny objects)

    Args:
        frame: Input frame
        yolo_detections: List of (x, y, w, h, conf, ...)
    """
    for det in yolo_detections:
        x, y, w, h, conf = det[:5]
        # วาด red bbox (รับ 1-2 pixel)
        if w >= 1 and h >= 1:  # รับ 1 pixel minimum
            cv2.rectangle(frame, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
            # แสดง confidence score (optional) - แสดง 3 decimal places สำหรับ conf ต่ำ
            if SHOW_CONFIDENCE:
                if len(det) > 6:
                    conf_text = f"{det[6]} {conf:.3f}"
                else:
                    conf_text = f"{conf:.3f}"
                cv2.putText(frame, conf_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BBOX_COLOR, 1)

# ============================================================================
# Main Function
# ============================================================================

def main():
    print("📷 Starting camera...")
    cam_config = get_camera_config()
    cam = CameraStream()
    cam.start()
    print("✅ Camera started")
    time.sleep(1.0)
    print("✅ Sleep completed")

    # Load YOLO model
    print("🤖 Loading YOLO model...")
    yolo_model, yolo_imgsz = load_yolo_model()  # 🔥 รับ imgsz ด้วย
    if yolo_model is None:
        print("⚠️ YOLO model not available - exiting")
        cam.release()
        return

    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    print("✅ Window created")

    # 🔥 GPU buffers สำหรับ resize (process ภาพเต็ม แต่แสดงผล Full HD)
    gpu_display_frame = None
    gpu_resized_frame = None
    initialized = False
    initialized_w = 0
    initialized_h = 0

    # FPS tracking (ใช้ deque เหมือน 2_vibe_grid.py)
    fps_timer = time.time()
    frames = 0
    fps = 0.0
    frame_time_history = deque(maxlen=30)  # Store last 30 frame times
    frame_count = 0
    yolo_frame_counter = 0  # 🔥 เพิ่ม YOLO frame counter

    print("🔄 Entering main loop...")

    while True:
        active, frame, _ = cam.read()  # 🔥 แก้ไข: เพิ่ม _ เพื่อรับ timestamp (แม้จะไม่ใช้)
        if not active or frame is None:
            print("⚠️ No frame received")
            break

        frame_count += 1
        frame_start_time = time.time()

        if frame_count == 1:
            print(f"📹 First frame received: {frame.shape}")

        h, w = frame.shape[:2]

        # 🔥 Initialize GPU buffers สำหรับ resize (ถ้ายังไม่ได้ init หรือ resolution เปลี่ยน)
        if not initialized or w != initialized_w or h != initialized_h:
            try:
                # Cleanup old buffers
                if gpu_display_frame is not None:
                    del gpu_display_frame
                if gpu_resized_frame is not None:
                    del gpu_resized_frame

                # Create new GPU buffers
                gpu_display_frame = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
                gpu_resized_frame = cv2.cuda_GpuMat(DISPLAY_HEIGHT, DISPLAY_WIDTH, cv2.CV_8UC3)

                initialized_w, initialized_h = w, h
                initialized = True
                print(f"✅ GPU buffers initialized: Process {w}x{h}, Display {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
            except Exception as e:
                print(f"⚠️ GPU buffer initialization failed: {e}")
                initialized = False

        # 🔥 YOLO detection (ทำทุก YOLO_INTERVAL เฟรม เพื่อเพิ่ม FPS)
        yolo_detections = []
        if yolo_model is not None and yolo_frame_counter % YOLO_INTERVAL == 0:
            yolo_detections = detect_yolo_full_frame(
                yolo_model, frame, YOLO_CONF_THRESHOLD, yolo_imgsz
            )

            # วาด bbox บน full resolution frame
            if len(yolo_detections) > 0:
                draw_yolo_detections(frame, yolo_detections)

        # 🔥 Increment YOLO frame counter
        yolo_frame_counter += 1

        # FPS Calculation (ใช้ deque เหมือน 2_vibe_grid.py)
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_time_history.append(frame_duration)
        if len(frame_time_history) > 0:
            avg_frame_time = np.mean(frame_time_history)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0

        frames += 1
        if time.time() - fps_timer >= 5.0:
            fps_old, frames, fps_timer = frames / 5.0, 0, time.time()
            if frame_count > 5:
                print(f"📊 FPS (5s avg): {fps_old:.1f} | Real-time FPS: {fps:.1f} | Detections: {len(yolo_detections)} | Conf Threshold: {YOLO_CONF_THRESHOLD}")

        # Display FPS and info
        fps_text = f"FPS: {fps:.1f}"
        if fps >= 25:
            fps_color = (0, 255, 0)
        elif fps >= 15:
            fps_color = (0, 165, 255)
        else:
            fps_color = (0, 0, 255)
        cv2.putText(frame, fps_text, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, fps_color, 3)

        info_text = f"Detections: {len(yolo_detections)} | Conf: {YOLO_CONF_THRESHOLD:.2f}"
        cv2.putText(frame, info_text, (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 🔥 GPU Resize for display - ใช้ GPU resize เพื่อเพิ่ม FPS
        # Process ภาพเต็ม resolution แต่แสดงผลแค่ Full HD
        display_frame = None
        if w != DISPLAY_WIDTH or h != DISPLAY_HEIGHT:
            try:
                if initialized and gpu_display_frame is not None and gpu_resized_frame is not None:
                    # ตรวจสอบว่า buffers มีขนาดถูกต้อง
                    if gpu_display_frame.size() == (h, w) and gpu_resized_frame.size() == (DISPLAY_HEIGHT, DISPLAY_WIDTH):
                        # Upload frame ที่วาดเสร็จแล้ว (มี boxes, text, etc.) ไป GPU
                        gpu_display_frame.upload(frame)
                        # Resize บน GPU (เร็วกว่า CPU มาก)
                        cv2.cuda.resize(gpu_display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), gpu_resized_frame)
                        # Download เฉพาะ display resolution
                        display_frame = gpu_resized_frame.download()
                    else:
                        # Buffer size ไม่ตรง - fallback to CPU
                        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                else:
                    # GPU buffers ไม่พร้อม - fallback to CPU
                    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            except Exception as e:
                # Fallback to CPU resize if GPU resize fails
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        else:
            display_frame = frame

        # Show frame (ตรวจสอบว่า display_frame ไม่เป็น None)
        if display_frame is not None and display_frame.size > 0:
            cv2.imshow(WINDOW_NAME, display_frame)
        else:
            print("⚠️ display_frame is None or empty, skipping display")
            continue

        # Cleanup display_frame
        del display_frame

        # Keyboard control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("👋 Program exited")

if __name__ == "__main__":
    main()




