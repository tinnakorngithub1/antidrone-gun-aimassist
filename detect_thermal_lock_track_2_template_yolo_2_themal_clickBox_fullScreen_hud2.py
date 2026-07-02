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
import json
import math
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

# --- Reticle / HUD (from detect_track_only) ---
CENTER_RADIUS_RATIO = 0.05
RETICLE_RADIUS_RATIOS = (0.03, 0.05, 0.08, 0.20)
ARROW_LEN = 120
ARROW_THICKNESS = 6
HUD_REF_WIDTH = 1920.0

# --- Keys ---
KEY_ADJUST_CENTER = "C"
KEY_SETTINGS = "S"
KEY_QUIT = "Q"
AIM_CENTER_STEP_PX = 10


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


# =============================================================================
# ShooterConfig — aim center + ballistics (save/load JSON), from detect_track_only
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


def hud_scale(frame_w):
    return max(0.4, frame_w / HUD_REF_WIDTH)


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


def estimate_distance_m(w_px, h_px, frame_w, frame_h, fov_h_deg, fov_v_deg, size_m=0.30):
    if w_px < 3 or h_px < 3 or frame_w <= 0 or frame_h <= 0 or fov_h_deg <= 0 or fov_v_deg <= 0:
        return None
    deg2rad = math.pi / 180.0
    theta_h = (w_px / frame_w) * (fov_h_deg * deg2rad)
    theta_v = (h_px / frame_h) * (fov_v_deg * deg2rad)
    if theta_h < 1e-6 and theta_v < 1e-6:
        return None
    dist_h = size_m / (2.0 * math.tan(theta_h / 2.0)) if theta_h >= 1e-6 else None
    dist_v = size_m / (2.0 * math.tan(theta_v / 2.0)) if theta_v >= 1e-6 else None
    if dist_h is not None and dist_v is not None:
        return (dist_h + dist_v) / 2.0
    return dist_h if dist_h is not None else dist_v


def draw_hud(frame, cx_frame, cy_frame, radius_px, target_bbox, target_center,
              ready_to_fire, guide_h=None, guide_v=None, distance_m=None, all_detections=None):
    """Reticle at center, locked target bbox, READY/THINGS, arrows to target."""
    h, w = frame.shape[:2]
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
        for det in all_detections:
            dx, dy, dw, dh = det[0], det[1], det[2], det[3]
            cv2.rectangle(frame, (int(dx), int(dy)), (int(dx + dw), int(dy + dh)), color_other, 1)

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
        label_drone = "LOCKED"
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
    color_text = (255, 255, 255) if (color_use == (0, 0, 0)) else color_use
    if not ready_to_fire:
        color_text = (255, 128, 0)  # BGR: bluish (จ้ำเงิน)
    font_label = cv2.FONT_HERSHEY_SIMPLEX
    scale_label = 2.0 * s
    th_label = max(1, int(3 * s))
    (tw_label, th_label_h), _ = cv2.getTextSize(label, font_label, scale_label, th_label)
    x_label = w - tw_label - int(20 * s)
    y_label = int(50 * s)
    cv2.putText(frame, label, (x_label, y_label), font_label, scale_label, color_text, th_label)

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


def main():
    camera_name = CAMERA_NAME or ACTIVE_CAMERA
    print("Thermal Lock Track: Drag box / L=lock / Click red box | C=aim center S=ballistics")
    print("Q=quit  L=lock/unlock  C=aim center  S=ballistics  U=unlock")

    cam = build_camera(camera_name)
    cam.start()
    time.sleep(0.5)

    try:
        from config import get_camera_config
        cam_config = get_camera_config(camera_name)
        fov_h = cam_config.get("fov_horizontal", 60.0)
        fov_v = cam_config.get("fov_vertical", 36.0)
    except Exception:
        fov_h, fov_v = 60.0, 36.0

    config = ShooterConfig()
    config.load()
    app_mode = "normal"
    settings_selected_field = 0
    settings_step = (10, 50, 0.5, 0.05)
    settings_min_max = ((10, 500), (200, 1500), (1, 50), (0.1, 2.0))

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

        if drawing and drag_start is not None and drag_end is not None:
            cv2.rectangle(display, drag_start, drag_end, DRAG_PREVIEW_COLOR, 2)

        if app_mode == "normal":
            cx_frame, cy_frame = config.get_center(fw, fh)
            radius_px = max(4, int(min(fw, fh) * CENTER_RADIUS_RATIO))
            target_bbox = locked_bbox if locked and locked_bbox is not None else None
            target_center = None
            if target_bbox is not None:
                bx, by, bw, bh = target_bbox
                target_center = (bx + bw / 2.0, by + bh / 2.0)
            ready_to_fire = False
            if target_center is not None:
                d = math.sqrt((target_center[0] - cx_frame) ** 2 + (target_center[1] - cy_frame) ** 2)
                ready_to_fire = d <= radius_px
            guide_h, guide_v = compute_guide_direction(target_center, cx_frame, cy_frame, radius_px, ready_to_fire)
            distance_m = None
            if target_bbox is not None and fov_h and fov_v:
                bx, by, bw, bh = target_bbox
                distance_m = estimate_distance_m(bw, bh, fw, fh, fov_h, fov_v, config.target_size_m)
            draw_hud(display, cx_frame, cy_frame, radius_px, target_bbox, target_center,
                     ready_to_fire, guide_h, guide_v, distance_m=distance_m, all_detections=None)
        elif app_mode == "adjust_center":
            draw_adjust_center_hud(display, config, fw, fh)
        elif app_mode == "settings":
            draw_settings_overlay(display, config, settings_selected_field)

        elapsed = time.time() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(display, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if app_mode == "normal":
            msg = "C=aim center  S=ballistics  L=lock  U=unlock  Drag box=lock  Click red box=lock"
            fh_d, fw_d = display.shape[:2]
            font_msg = cv2.FONT_HERSHEY_SIMPLEX
            scale_msg = 0.5
            th_msg = 2
            (tw_msg, th_msg_h), _ = cv2.getTextSize(msg, font_msg, scale_msg, th_msg)
            while tw_msg > fw_d - 24 and scale_msg > 0.3:
                scale_msg -= 0.05
                (tw_msg, th_msg_h), _ = cv2.getTextSize(msg, font_msg, scale_msg, th_msg)
            y_bottom = fh_d - 12
            cv2.putText(display, msg, (10, y_bottom), font_msg, scale_msg, (200, 200, 200), th_msg)

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
        elif app_mode == "normal":
            if key in (ord("c"), ord("C")):
                app_mode = "adjust_center"
            elif key in (ord("s"), ord("S")):
                app_mode = "settings"
            elif key in (ord("l"), ord("L"), ord("u"), ord("U")):
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
    print("Stopped.")


if __name__ == "__main__":
    main()

