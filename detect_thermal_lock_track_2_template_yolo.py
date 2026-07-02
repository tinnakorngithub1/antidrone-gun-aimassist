"""
detect_thermal_lock_track.py — ล็อกเป้าด้วยลากเมาส์ แล้วตามด้วย template matching ทุกเฟรม

- ล็อก: คลิกซ้ายลากเมาส์คลุมวัตถุ → เก็บ patch เป็น template
- ทุกเฟรม: cv2.matchTemplate ใน ROI รอบตำแหน่งเดิม → อัปเดต bbox สีเหลือง (เร็ว ไม่หลุด)
- ไม่ใช้ motion, ไม่ใช้ blob/signature
"""

import cv2
import numpy as np
import time
from collections import deque

try:
    from config import ACTIVE_CAMERA
except ImportError:
    ACTIVE_CAMERA = "cam0"

from detect_motion_residual_visual import build_camera

# =============================================================================
# Parameters
# =============================================================================

CAMERA_NAME = None

# --- Template matching: บริเวณใกล้ตำแหน่ง lock เป็น priority สูงสุด ---
TM_METHOD = cv2.TM_CCOEFF_NORMED  # เร็วและทนแสงเปลี่ยน
TM_MIN_SCORE = 0.25  # ต่ำกว่านี้ไม่อัปเดต; ลดไว้เพื่ออย่าให้หลุดแม้วัตถุเปลี่ยนมุม (เฉดสีรอบๆ ต่างจากวัตถุ)
SEARCH_MARGIN = 180  # px ขยายจาก bbox เดิมเป็น ROI ค้นเท่านั้น (ไม่ fallback ทั้งเฟรม)

# --- Drag ---
DRAG_MIN_SIZE = 5

# --- Display ---
LOCKED_COLOR = (0, 255, 255)
DRAG_PREVIEW_COLOR = (255, 255, 255)


def main():
    camera_name = CAMERA_NAME or ACTIVE_CAMERA
    print("Thermal Lock Track: ลากเมาส์คลุมวัตถุ → ตามด้วย template matching ทุกเฟรม")
    print("Q = quit  L/U = unlock")

    cam = build_camera(camera_name)
    cam.start()
    time.sleep(0.5)

    fps_times = deque(maxlen=30)
    locked = False
    stored_template = None  # gray patch (h, w)
    stored_tw = 0
    stored_th = 0
    locked_bbox = None
    drawing = False
    drag_start = None
    drag_end = None
    curr_gray_ref = [None]

    window_name = "Thermal Lock Track"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, drag_start, drag_end, locked, stored_template, stored_tw, stored_th, locked_bbox
        gray = curr_gray_ref[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if not drawing or drag_start is None or gray is None:
                drag_start = None
                drag_end = None
                drawing = False
                return
            x1, y1 = drag_start
            x2, y2 = drag_end
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            drag_start = None
            drag_end = None
            drawing = False
            if w < DRAG_MIN_SIZE or h < DRAG_MIN_SIZE:
                return
            fh, fw = gray.shape[:2]
            x2_c = min(fw, x_min + w)
            y2_c = min(fh, y_min + h)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            if x2_c <= x_min or y2_c <= y_min:
                return
            template = gray[y_min:y2_c, x_min:x2_c].copy()
            if template.size == 0:
                return
            stored_template = template
            stored_tw = template.shape[1]
            stored_th = template.shape[0]
            locked_bbox = (x_min, y_min, stored_tw, stored_th)
            locked = True

    cv2.setMouseCallback(window_name, on_mouse)

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
        curr_gray_ref[0] = curr_gray

        # ค้นเฉพาะใน ROI รอบตำแหน่ง lock (priority สูงสุด), อัปเดตตำแหน่งไปเรื่อยๆ อย่าให้หลุด
        if locked and stored_template is not None and locked_bbox is not None:
            tw, th = stored_tw, stored_th
            if tw > 0 and th > 0 and curr_gray.shape[0] >= th and curr_gray.shape[1] >= tw:
                px, py, pw, ph = locked_bbox
                margin = SEARCH_MARGIN
                rx1 = max(0, px - margin)
                ry1 = max(0, py - margin)
                rx2 = min(fw, px + pw + margin)
                ry2 = min(fh, py + ph + margin)
                # ใช้เฉพาะ ROI — ไม่ fallback ทั้งเฟรม; ถ้า ROI เล็กเกินขยายให้พอ
                roi_w = rx2 - rx1
                roi_h = ry2 - ry1
                if roi_w < tw or roi_h < th:
                    rx1 = max(0, px - max(margin, tw))
                    ry1 = max(0, py - max(margin, th))
                    rx2 = min(fw, px + pw + max(margin, tw))
                    ry2 = min(fh, py + ph + max(margin, th))
                    roi_w = rx2 - rx1
                    roi_h = ry2 - ry1
                if roi_w >= tw and roi_h >= th:
                    roi = curr_gray[ry1:ry2, rx1:rx2]
                    res = cv2.matchTemplate(roi, stored_template, TM_METHOD)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= TM_MIN_SCORE:
                        nx = rx1 + max_loc[0]
                        ny = ry1 + max_loc[1]
                        locked_bbox = (nx, ny, tw, th)
                    # score ต่ำก็คง bbox เดิม (ไม่อัปเดต) — ไม่กระโดดไปทั้งเฟรม

        display = frame.copy()

        if locked and locked_bbox is not None:
            x, y, w, h = locked_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), LOCKED_COLOR, 2)
            cv2.putText(display, "LOCKED", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LOCKED_COLOR, 2)

        if drawing and drag_start is not None and drag_end is not None:
            cv2.rectangle(display, drag_start, drag_end, DRAG_PREVIEW_COLOR, 2)

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(display, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, "LOCKED" if locked else "UNLOCKED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not locked:
            cv2.putText(display, "ลากเมาส์คลุมวัตถุเพื่อล็อก", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1)
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("l"), ord("L"), ord("u"), ord("U")) and locked:
            locked = False
            stored_template = None
            stored_tw = stored_th = 0
            locked_bbox = None

    cam.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

