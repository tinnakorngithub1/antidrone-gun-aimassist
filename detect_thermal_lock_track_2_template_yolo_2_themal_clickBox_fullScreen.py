"""
detect_thermal_lock_track.py — ล็อกเป้าด้วยลากเมาส์ หรือจาก bbox โดรน (YOLO) แล้วตามด้วย template

- ล็อก: (1) ลากเมาส์คลุมวัตถุ หรือ (2) กด L เพื่อ lock โดรนจาก YOLO หรือ (3) คลิกที่ bbox โดรน
- หลัง lock: ตามด้วย template matching ใน ROI (กล่องสีเหลือง), ทำเหมือนกันทุกกรณี
- รัน YOLO ได้เหมือน detect_motion_yolo_fusion
"""

import cv2
import numpy as np
import time
import os
from collections import deque

def _get_screen_size():
    """ได้ (width, height) ของจอหลัก (พิกเซล)"""
    try:
        from tkinter import Tk
        r = Tk()
        r.withdraw()
        w = r.winfo_screenwidth()
        h = r.winfo_screenheight()
        r.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080

try:
    from config import ACTIVE_CAMERA
except ImportError:
    ACTIVE_CAMERA = "cam0"

from detect_motion_residual_visual import build_camera
from smart_detection_yolo_only import load_yolo_model, detect_yolo_full_frame

# =============================================================================
# Parameters
# =============================================================================

CAMERA_NAME = None

# --- YOLO ---
YOLO_ENGINE = "last_imgsz640.engine"
YOLO_IMGSZ = 640
YOLO_CONF_MIN = 0.1
YOLO_ENABLED = True

# --- Template matching ---
TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_MIN_SCORE = 0.5          # ต่ำสุดที่ยอมรับ (ลดการติดจุดที่รูปร่างไม่เหมือน)
TM_MIN_SCORE_STRICT = 0.65  # ถ้ากระโดดไกล ต้องได้คะแนนสูงถึงค่อยย้าย
MAX_JUMP_RATIO = 1.2        # ถ้าเคลื่อนเกิน ratio เท่าของขนาด template ถือว่ากระโดดไกล
SEARCH_MARGIN = 180

# --- Drag / Click ---
DRAG_MIN_SIZE = 5
CLICK_MAX_MOVE = 8  # ถ้าเลื่อนเมาส์ไม่เกินนี้ถือว่าเป็น "คลิก" (สำหรับ lock จาก bbox โดรน)

# --- Display ---
LOCKED_COLOR = (0, 255, 255)
DRAG_PREVIEW_COLOR = (255, 255, 255)
YOLO_BBOX_COLOR = (0, 0, 255)  # สีแดง


def load_yolo():
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
            print(f"Thermal Lock Track: YOLO {YOLO_IMGSZ} loaded (engine)")
        except Exception as e:
            print(f"YOLO engine load failed: {e}")
            yolo_model = None
    if yolo_model is None:
        result = load_yolo_model()
        if result is None or result[0] is None:
            return None, YOLO_IMGSZ
        yolo_model, imgsz = result[0], result[1] if result[1] is not None else YOLO_IMGSZ
        print(f"Thermal Lock Track: YOLO loaded (fallback, imgsz={imgsz})")
    return yolo_model, imgsz


def lock_from_bbox(gray, bbox, stored_template_ref, stored_tw_ref, stored_th_ref, locked_bbox_ref, locked_ref):
    """ตั้ง lock จาก bbox (x,y,w,h); เก็บ template จาก gray ใน bbox"""
    x, y, w, h = bbox
    fh, fw = gray.shape[:2]
    if w < 2 or h < 2:
        return
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    x = max(0, x)
    y = max(0, y)
    if x2 <= x or y2 <= y:
        return
    template = gray[y:y2, x:x2].copy()
    if template.size == 0:
        return
    stored_template_ref[0] = template
    stored_tw_ref[0] = template.shape[1]
    stored_th_ref[0] = template.shape[0]
    locked_bbox_ref[0] = (x, y, stored_tw_ref[0], stored_th_ref[0])
    locked_ref[0] = True


def main():
    camera_name = CAMERA_NAME or ACTIVE_CAMERA
    print("Thermal Lock Track: ลากเมาส์ / กด L lock โดรน / คลิกที่ bbox โดรน → ตามด้วย template")
    print("Q = quit  L = lock โดรนจาก YOLO หรือ unlock")

    cam = build_camera(camera_name)
    cam.start()
    time.sleep(0.5)

    yolo_model = None
    yolo_imgsz = YOLO_IMGSZ
    if YOLO_ENABLED:
        yolo_model, yolo_imgsz = load_yolo()
        if yolo_model is None:
            print("YOLO not available; lock ได้แค่ลากเมาส์")

    fps_times = deque(maxlen=30)
    locked = False
    stored_template = None
    stored_tw = 0
    stored_th = 0
    locked_bbox = None
    drawing = False
    drag_start = None
    drag_end = None
    curr_gray_ref = [None]
    yolo_dets_ref = [[]]  # ให้ mouse callback อ่าน dets ล่าสุด

    # ref สำหรับส่งเข้า lock_from_bbox (ให้ฟังก์ชันอัปเดต locked state ได้)
    stored_template_ref = [None]
    stored_tw_ref = [0]
    stored_th_ref = [0]
    locked_bbox_ref = [None]
    locked_ref = [False]

    window_name = "Thermal Lock Track"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = _get_screen_size()
    frame_size_ref = [None]   # (fw, fh)
    display_size_ref = [None]  # (display_w, display_h) หลัง resize fit จอ

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, drag_start, drag_end, locked, stored_template, stored_tw, stored_th, locked_bbox
        gray = curr_gray_ref[0]
        dets = yolo_dets_ref[0] if yolo_dets_ref else []
        # แปลงพิกัดจากจอ (หลัง fit) เป็นพิกัดเฟรมต้นฉบับ
        fsize = frame_size_ref[0]
        dsize = display_size_ref[0]
        if gray is not None and fsize is not None and dsize is not None:
            fw, fh = fsize
            dw, dh = dsize
            if dw > 0 and dh > 0:
                x = max(0, min(fw - 1, int(x * fw / dw)))
                y = max(0, min(fh - 1, int(y * fh / dh)))
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
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            drag_start = None
            drag_end = None
            drawing = False

            if dx <= CLICK_MAX_MOVE and dy <= CLICK_MAX_MOVE and dets:
                # ถือว่าเป็นคลิก — ตรวจว่าคลิกอยู่ภายใน bbox โดรนใดหรือไม่
                for det in dets:
                    bx, by, bw, bh, _ = det
                    if bx <= x2 <= bx + bw and by <= y2 <= by + bh:
                        lock_from_bbox(gray, (bx, by, bw, bh), stored_template_ref, stored_tw_ref, stored_th_ref, locked_bbox_ref, locked_ref)
                        stored_template = stored_template_ref[0]
                        stored_tw = stored_tw_ref[0]
                        stored_th = stored_th_ref[0]
                        locked_bbox = locked_bbox_ref[0]
                        locked = locked_ref[0]
                        return
                return
            # ลากเมาส์จริง
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
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
            # sync ref ด้วย — ไม่ให้ลูปหลักเขียนทับ lock ใหม่ด้วยข้อมูลเก่า (ลากเมาส์ = target ใหม่ตลอด)
            stored_template_ref[0] = stored_template
            stored_tw_ref[0] = stored_tw
            stored_th_ref[0] = stored_th
            locked_bbox_ref[0] = locked_bbox
            locked_ref[0] = True

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

        dets = []
        if YOLO_ENABLED and yolo_model is not None:
            dets = detect_yolo_full_frame(yolo_model, frame, YOLO_CONF_MIN, imgsz=yolo_imgsz)
        yolo_dets_ref[0] = dets

        # Sync ref กลับมา (กรณี lock จาก callback)
        if locked_ref[0]:
            locked = True
            stored_template = stored_template_ref[0]
            stored_tw = stored_tw_ref[0]
            stored_th = stored_th_ref[0]
            locked_bbox = locked_bbox_ref[0]

        if locked and stored_template is not None and locked_bbox is not None:
            tw, th = stored_tw, stored_th
            if tw > 0 and th > 0 and curr_gray.shape[0] >= th and curr_gray.shape[1] >= tw:
                px, py, pw, ph = locked_bbox
                margin = SEARCH_MARGIN
                rx1 = max(0, px - margin)
                ry1 = max(0, py - margin)
                rx2 = min(fw, px + pw + margin)
                ry2 = min(fh, py + ph + margin)
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
                        # ตรวจระยะกระโดด: ถ้าไกลจากตำแหน่งเดิมมาก ต้องได้คะแนนสูงถึงยอมรับ
                        dx = nx - px
                        dy = ny - py
                        dist = (dx * dx + dy * dy) ** 0.5
                        template_diag = (pw * pw + ph * ph) ** 0.5
                        if template_diag < 1:
                            template_diag = 1
                        if dist <= MAX_JUMP_RATIO * template_diag:
                            accept = True
                        else:
                            accept = max_val >= TM_MIN_SCORE_STRICT
                        if accept:
                            locked_bbox = (nx, ny, tw, th)
                            locked_bbox_ref[0] = locked_bbox

        display = frame.copy()

        for det in dets:
            x, y, w, h, conf = det
            cv2.rectangle(display, (x, y), (x + w, y + h), YOLO_BBOX_COLOR, 1)
            cv2.putText(display, f"{conf:.2f}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YOLO_BBOX_COLOR, 1)

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
            msg = "Drag box / L=lock drone / Click red box"
            cv2.putText(display, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # ย่อให้พอดีจอ (รักษาอัตราส่วน) แบบ detect_track_only
        fh, fw = display.shape[:2]
        frame_size_ref[0] = (fw, fh)
        scale = min(screen_w / fw, screen_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        display_size_ref[0] = (new_w, new_h)
        display_fit = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        try:
            cv2.resizeWindow(window_name, new_w, new_h)
        except cv2.error:
            pass
        cv2.imshow(window_name, display_fit)
        key = cv2.waitKey(1)
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("l"), ord("L"), ord("u"), ord("U")):
            if locked:
                locked = False
                locked_ref[0] = False
                stored_template = None
                stored_template_ref[0] = None
                stored_tw = stored_th = 0
                stored_tw_ref[0] = stored_th_ref[0] = 0
                locked_bbox = None
                locked_bbox_ref[0] = None
            elif dets:
                best = max(dets, key=lambda d: d[4])
                bx, by, bw, bh, _ = best
                lock_from_bbox(curr_gray, (bx, by, bw, bh), stored_template_ref, stored_tw_ref, stored_th_ref, locked_bbox_ref, locked_ref)
                locked = True
                stored_template = stored_template_ref[0]
                stored_tw = stored_tw_ref[0]
                stored_th = stored_th_ref[0]
                locked_bbox = locked_bbox_ref[0]

    cam.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

