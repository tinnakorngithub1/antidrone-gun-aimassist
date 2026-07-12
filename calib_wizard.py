"""
Calibration Wizard — คาลิเบรตกล้องใหม่ให้จบในแอป (ปุ่ม W)

ปัญหาที่แก้: เปลี่ยนกล้องที่แขนกลแล้วต้องคาลิเบรตใหม่ทั้งชุด แต่ของเดิม
  - ปุ่ม G (px/deg) บังคับ resize เฟรมเป็น 3840×2160 เสมอ (REF_OUTPUT_W/H) → คาลิเบรตกล้อง
    ความละเอียดอื่นออกมาผิด
  - ego_comp_latency เป็นเลข 0.06 ที่ 'เดา' ไว้ — ทั้ง repo ไม่มีโค้ดวัดมันเลย
  - noise floor (LOCK_MEAS_SIGMA_PX) เป็นค่าคงที่ในโค้ด ทั้งที่คอมเมนต์เขียนเองว่า
    'ต้องวัดใหม่ต่อกล้อง'
  - ทุกอย่างต้องคลิกทีละจุดด้วยมือ

วิธีวัด (อัตโนมัติ — ไม่ต้องคลิก):
  ppd      : แขนหมุนเป็นองศาที่รู้ค่า (อ่านกลับจาก GRBL) → วัดว่าภาพเลื่อนกี่พิกเซล
             (cv2.phaseCorrelate) → fit เส้นตรง shift_px = -ppd × Δdeg ด้วย lstsq
             ใช้ได้เมื่อฉากมี texture (ต้นไม้/อาคาร). ท้องฟ้าเปล่า → ถอยไปโหมดคลิก
  latency  : แขนสะบัดกลับไปกลับมา (ω เปลี่ยนเครื่องหมาย) ระหว่างนั้น 'ทิศจริง' ของฉากนิ่ง
             ต้องคงที่ → หา L ที่ทำให้ pose_at(t−L) + px_offset/ppd แปรปรวนน้อยสุด
             *ต้องสะบัด* — ถ้าหมุนอัตราคงที่ L จะหาไม่ได้ (มันกลายเป็นค่าคงที่ ไม่กระทบ variance)
  noise    : แขนนิ่ง จับเป้า → σ ของ bbox center (หักการดริฟท์เชิงเส้นออกก่อน)
  boresight: วัดเองไม่ได้ — ความจริงอยู่นอกระบบกล้อง+แขน (ลำกล้องชี้ไปไหนจริง) ต้องมีคนชี้

งานหนัก (สั่งแขน + รอ GRBL) รันใน worker thread — ห้าม block main loop เด็ดขาด
(เคยทำให้โปรแกรมโดน SIGKILL มาแล้ว)
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

_DIR = Path(__file__).resolve().parent
_CALIB_DIR = _DIR / "calibration_data"

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_BG = (20, 20, 24)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 155)
C_OK = (80, 220, 120)
C_WARN = (60, 200, 255)
C_ERR = (80, 80, 255)
C_ACC = (0, 140, 255)

# --- พารามิเตอร์การวัด ---
PPD_STEP_DEG = 1.5        # ขยับทีละกี่องศา (เล็กพอให้ phaseCorrelate ไม่หลุด)
PPD_STEPS = 6             # กี่ก้าวต่อทิศ → กวาด ±9°
PPD_SETTLE_SEC = 0.35     # รอภาพนิ่งหลังแขนหยุด (กัน motion blur เข้าไปในการวัด)
PPD_MIN_R2 = 0.985        # ความเป็นเส้นตรงขั้นต่ำ — ต่ำกว่านี้ = ฉากไม่มี texture/แทร็กหลุด
PROC_W = 640              # ย่อภาพก่อน phaseCorrelate (เร็ว + shift เล็กลง = แม่นขึ้น)

# ขับแขนเป็น 'ไซน์' ไม่ใช่สะบัดไปกลับ: ω ต้องเปลี่ยนตลอดเวลาถึงจะหา L ได้
#   bearing_est(L) − bearing_จริง ≈ ω(t)·(L_จริง − L)   → variance = (L_จริง−L)²·Var(ω)
#   ω คงที่ → Var(ω)=0 → variance ไม่ขึ้นกับ L เลย → หา L ไม่ได้ (นี่คือกับดัก)
#   ไซน์ → ω เปลี่ยนต่อเนื่อง + ได้เฟรมเยอะ (สะบัดเร็ว ๆ ที่ 30fps ได้ไม่กี่เฟรม = สัญญาณไม่พอ)
LAT_AMP_DEG = 5.0         # แอมพลิจูด (องศา)
LAT_FREQ_HZ = 0.6         # ยิ่งเร็ว ω ยิ่งแกว่ง แต่เฟรมต่อคาบยิ่งน้อย — 0.6Hz สมดุลที่ 30fps
LAT_DURATION_SEC = 6.0
LAT_CMD_HZ = 50.0         # อัตราส่งคำสั่งแขน
LAT_CANDIDATES = np.arange(0.0, 0.251, 0.0025)  # ลอง L 0–250ms ทีละ 2.5ms

NOISE_SEC = 8.0           # เก็บ bbox center กี่วินาที

STEPS = ["เตรียม", "ppd", "latency", "noise", "boresight", "สรุป"]


class _W:
    """สถานะ wizard (module-level singleton — เหมือน embed ตัวอื่น)"""
    active = False
    step = 0
    camera = "cam4"
    frame_w = 0
    frame_h = 0

    busy = False              # worker กำลังทำงาน
    worker: Optional[threading.Thread] = None
    log: List[str] = []
    err = ""

    latest_gray: Optional[np.ndarray] = None   # เฟรมล่าสุด (ย่อแล้ว) — worker อ่านจากตรงนี้
    latest_t = 0.0
    # ท่าแขน ณ 'เวลาที่เฟรมมาถึง' — ต้องเก็บคู่กับเฟรมตรงนี้เลย ห้ามให้ worker ไปอ่าน arm.pos_x
    # ทีหลัง เพราะกว่า worker จะประมวลผลเสร็จ แขนขยับไปแล้ว → ท่าที่ได้ 'ล้ำหน้า' ภาพ
    # → latency ที่วัดได้จะบวกเกินเท่ากับเวลาประมวลผลพอดี (เจอมาแล้ว: เกินคงที่ ~33ms)
    latest_pose: Tuple[float, float] = (0.0, 0.0)
    frame_lock = threading.Lock()

    # ผลลัพธ์
    ppd_x: Optional[float] = None
    ppd_y: Optional[float] = None
    ppd_r2: Optional[Tuple[float, float]] = None
    latency: Optional[float] = None
    sigma_px: Optional[float] = None
    boresight: Optional[Tuple[float, float]] = None

    # โหมดคลิก (fallback ของ ppd)
    click_mode = False
    click_pts: List[Tuple[float, float, float, float]] = []   # (px, py, pan, tilt)
    pending_click: Optional[Tuple[float, float]] = None

    display = (0, 0, 0, 0)
    display_wh = (0, 0)
    get_detection_center: Optional[Callable[[], Optional[Tuple[float, float]]]] = None


def is_active() -> bool:
    return _W.active


def is_busy() -> bool:
    return _W.busy


def enter(window_name: str, camera_name: str, frame_w: int, frame_h: int,
          get_detection_center: Optional[Callable] = None) -> None:
    _W.active = True
    _W.step = 0
    _W.camera = camera_name
    _W.frame_w, _W.frame_h = int(frame_w), int(frame_h)
    _W.log = [f"กล้อง: {camera_name}  ({frame_w}×{frame_h})"]
    _W.err = ""
    _W.busy = False
    _W.ppd_x = _W.ppd_y = _W.ppd_r2 = None
    _W.latency = _W.sigma_px = _W.boresight = None
    _W.click_mode = False
    _W.click_pts = []
    _W.pending_click = None
    _W.get_detection_center = get_detection_center
    cv2.setMouseCallback(window_name, _on_mouse)


def leave(window_name: str) -> None:
    _W.active = False
    _W.busy = False
    try:
        cv2.setMouseCallback(window_name, lambda *a: None)
    except cv2.error:
        pass


def set_display_size(display_w: int, display_h: int,
                     content_rect: Optional[Tuple[int, int, int, int]] = None) -> None:
    _W.display_wh = (int(display_w), int(display_h))
    _W.display = tuple(int(v) for v in content_rect) if content_rect else (0, 0, int(display_w), int(display_h))


def hud_label() -> str:
    return f"WIZARD [{STEPS[_W.step]}]" + ("  …กำลังวัด" if _W.busy else "")


def _say(msg: str) -> None:
    _W.log.append(msg)
    if len(_W.log) > 12:
        del _W.log[0]
    print(f"[WIZARD] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Image shift (phaseCorrelate)
# ---------------------------------------------------------------------------
def _prep(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """คืน (grayscale float32 ที่ย่อแล้ว, scale) — scale = เท่าไรที่ย่อลง"""
    h, w = frame.shape[:2]
    sc = PROC_W / float(w)
    small = cv2.resize(frame, (PROC_W, max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32), sc


def _shift(prev: np.ndarray, cur: np.ndarray) -> Tuple[float, float, float]:
    """เลื่อนกี่พิกเซล (ในภาพย่อ) + response (0-1, ต่ำ = ไม่น่าเชื่อถือ)"""
    win = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
    return float(dx), float(dy), float(resp)


def _grab() -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """คืน (ภาพย่อ, เวลาที่เฟรมมาถึง, ท่าแขน ณ เวลานั้น) — ทั้งสามต้องมาจากช็อตเดียวกัน"""
    with _W.frame_lock:
        if _W.latest_gray is None:
            return None
        return _W.latest_gray.copy(), _W.latest_t, _W.latest_pose


def _wait_fresh(after_t: float, timeout: float = 3.0) -> Optional[np.ndarray]:
    """รอเฟรมที่ 'ถ่ายหลัง' เวลาที่กำหนด — กันเอาเฟรมเก่าค้างท่อมาวัด"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = _grab()
        if g is not None and g[1] > after_t:
            return g[0]
        time.sleep(0.01)
    return None


# ---------------------------------------------------------------------------
# STEP: ppd อัตโนมัติ
# ---------------------------------------------------------------------------
def _fit_through_origin(dx: List[float], dy: List[float]) -> Tuple[float, float]:
    """fit y = m·x ผ่านจุดกำเนิด (lstsq) → คืน (slope, r²)
    ใช้ lstsq ไม่ใช่ mean-of-ratios แบบโค้ดเดิม: mean-of-ratios ให้น้ำหนักจุดใกล้ศูนย์มากเกิน
    (หาร Δ เล็ก ๆ → ratio เหวี่ยง) ส่วน lstsq ให้น้ำหนักตามระยะที่ขยับจริง = ถูกต้องกว่า"""
    x = np.asarray(dx, dtype=np.float64)
    y = np.asarray(dy, dtype=np.float64)
    denom = float(np.dot(x, x))
    if denom < 1e-9:
        return 0.0, 0.0
    m = float(np.dot(x, y) / denom)
    resid = y - m * x
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y, y))     # ผ่านจุดกำเนิด → เทียบกับ 0 ไม่ใช่ค่าเฉลี่ย
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return m, r2


def _goto(arm, axis: int, val: float, other: float) -> float:
    """ขยับแกนหนึ่งไปตำแหน่งหนึ่ง รอนิ่ง แล้วอ่านองศา 'จริง' กลับจาก GRBL"""
    if axis == 0:
        arm.move_absolute(val, other, blocking=True)
    else:
        arm.move_absolute(other, val, blocking=True)
    time.sleep(PPD_SETTLE_SEC)
    try:
        arm.sync_position_from_grbl()
    except Exception:
        pass
    return float(arm.pos_x if axis == 0 else arm.pos_y)


def _sweep_axis(arm, axis: int, scale: float) -> Tuple[List[float], List[float]]:
    """หมุนแขนทีละก้าวเล็ก ๆ ในแกนหนึ่ง วัด shift ของภาพสะสม. axis 0=pan 1=tilt
    คืน (Δองศาจริงเทียบจุดเริ่ม[], shift พิกเซลสะสมในภาพเต็ม[])

    สำคัญ: ต้องไปยืนที่ 'จุดเริ่มของช่วงกวาด' ก่อน แล้วค่อยถ่ายภาพอ้างอิง
    ถ้าถ่ายอ้างอิงที่ home แล้วกระโดดไปปลายช่วงทีเดียว ก้าวแรกจะใหญ่เกินกว่า phaseCorrelate
    จะวัดแม่น → ความผิดก้อนนั้นติดไปกับทุกจุดที่สะสมต่อ → fit ผ่านจุดกำเนิดเอียงทั้งเส้น
    (เคยทำให้ได้ 83.9 แทนที่จะเป็น 87.1 = ต่ำไป 4%)
    """
    lim = arm._effective_x_limits if axis == 0 else arm._effective_y_limits
    other = 0.0
    steps = [k * PPD_STEP_DEG for k in range(-PPD_STEPS, PPD_STEPS + 1)]
    steps = [s for s in steps if lim[0] + 1.0 <= s <= lim[1] - 1.0]
    if len(steps) < 4:
        raise RuntimeError("ลิมิตแกนนี้แคบเกินกว่าจะกวาดวัดได้")

    start_deg = _goto(arm, axis, steps[0], other)
    prev_img = _wait_fresh(time.time())
    if prev_img is None:
        raise RuntimeError("ไม่ได้เฟรมจากกล้อง")
    _check_texture(prev_img)

    d_deg: List[float] = [0.0]
    d_px: List[float] = [0.0]
    acc_px = 0.0

    for tgt in steps[1:]:            # ก้าวละ PPD_STEP_DEG เท่ากันหมด → shift เล็ก วัดแม่น
        actual = _goto(arm, axis, tgt, other)
        img = _wait_fresh(time.time())
        if img is None:
            raise RuntimeError("กล้องหยุดส่งเฟรมระหว่างวัด")
        dx, dy, resp = _shift(prev_img, img)
        if resp < 0.05:
            raise RuntimeError(f"ภาพจับคู่ไม่ได้ (response {resp:.2f}) — ฉากไม่มี texture?")
        acc_px += (dx if axis == 0 else dy) / scale     # กลับเป็นพิกเซลของภาพเต็ม
        prev_img = img
        d_deg.append(actual - start_deg)
        d_px.append(acc_px)
        _say(f"  {'pan' if axis==0 else 'tilt'} {actual - start_deg:+.2f}° → {acc_px:+.1f}px")

    arm.move_absolute(0.0, 0.0, blocking=True)
    return d_deg, d_px


def _check_texture(gray: np.ndarray) -> None:
    """ฉากต้องมีรายละเอียดพอให้จับคู่ภาพได้ — ท้องฟ้าเปล่า/มืดสนิท วัดไม่ได้

    เช็คด้วย Laplacian variance (พลังงานขอบ): ต่ำ = ภาพเรียบ ไม่มีอะไรให้เกาะ
    ถ้าไม่เช็ค phaseCorrelate จะคืน 'ตัวเลขอะไรสักอย่าง' ที่ดูสมเหตุสมผลแต่มั่ว
    """
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if lap < 40.0:
        raise RuntimeError(
            f"ฉากเรียบเกินไป (edge energy {lap:.0f}) — ท้องฟ้าเปล่า/มืด/เลนส์ปิด? "
            "หันไปทางที่มีต้นไม้หรืออาคาร หรือกด C ใช้โหมดคลิก"
        )


def _worker_ppd(arm) -> None:
    try:
        g = _grab()
        if g is None:
            raise RuntimeError("ยังไม่มีเฟรมจากกล้อง")
        # scale ที่ใช้ย่อ (ต้องรู้เพื่อแปลง shift กลับเป็นพิกเซลจริง)
        scale = PROC_W / float(_W.frame_w)

        _say("วัด pan …")
        dpan, dpx = _sweep_axis(arm, 0, scale)
        _say("วัด tilt …")
        dtilt, dpy = _sweep_axis(arm, 1, scale)

        if len(dpan) < 4 or len(dtilt) < 4:
            raise RuntimeError("จุดวัดไม่พอ (ลิมิตแขนแคบไป?)")

        # กล้องอยู่บนแขน: แขนหมุน +Δ → ฉากเลื่อน −Δ·ppd ในภาพ  (convention เดียวกับ LOCK)
        mx, r2x = _fit_through_origin(dpan, dpx)
        my, r2y = _fit_through_origin(dtilt, dpy)
        ppd_x, ppd_y = -mx, -my

        if r2x < PPD_MIN_R2 or r2y < PPD_MIN_R2:
            raise RuntimeError(
                f"ความสัมพันธ์ไม่เป็นเส้นตรงพอ (r²={r2x:.3f}/{r2y:.3f}) — "
                "ฉากอาจเป็นท้องฟ้าเปล่า/มืด/มีของเคลื่อนไหว. กด C ใช้โหมดคลิกแทน"
            )
        if abs(ppd_x) < 1.0 or abs(ppd_y) < 1.0:
            raise RuntimeError(f"ppd ผิดปกติ ({ppd_x:.1f}/{ppd_y:.1f}) — แขนขยับจริงหรือเปล่า?")

        _W.ppd_x, _W.ppd_y = ppd_x, ppd_y
        _W.ppd_r2 = (r2x, r2y)
        fh = _W.frame_w / abs(ppd_x)
        fv = _W.frame_h / abs(ppd_y)
        _say(f"✓ ppd = {ppd_x:.2f} / {ppd_y:.2f} px/°  (r² {r2x:.4f}/{r2y:.4f})")
        _say(f"  → FOV จริง {fh:.1f}° × {fv:.1f}°")
        if ppd_y > 0:
            _say("⚠ ppd_y เป็นบวก — ผิดจาก convention (ระบบสมมติว่าติดลบ) ตรวจการติดตั้งกล้อง")
    except Exception as e:
        _W.err = str(e)
        _say(f"✗ {e}")
    finally:
        _W.busy = False


# ---------------------------------------------------------------------------
# STEP: ego_comp_latency อัตโนมัติ
# ---------------------------------------------------------------------------
def _worker_latency(arm) -> None:
    try:
        if not _W.ppd_x:
            raise RuntimeError("ต้องวัด ppd ก่อน")
        scale = PROC_W / float(_W.frame_w)
        lim = arm._effective_x_limits
        amp = min(LAT_AMP_DEG, (lim[1] - lim[0]) / 2.0 - 1.0)
        if amp < 1.0:
            raise RuntimeError("ลิมิต pan แคบเกินกว่าจะกวาดวัดได้")

        arm.move_absolute(0.0, 0.0, blocking=True)
        time.sleep(0.5)

        samples: List[Tuple[float, float, float]] = []   # (t_เฟรม, pan ที่ 'สั่ง', feat_px สะสม)
        stop = threading.Event()
        state = {"prev": None, "px": 0.0, "bad": 0}

        def _sampler():
            last_t = 0.0
            while not stop.is_set():
                g = _grab()
                if g is None or g[1] <= last_t:     # ยังไม่มีเฟรมใหม่ → อย่าวัดซ้ำเฟรมเดิม
                    time.sleep(0.003)
                    continue
                img, t, pose = g
                last_t = t
                if state["prev"] is None:
                    state["prev"] = img
                    continue
                dx, _dy, resp = _shift(state["prev"], img)
                if resp < 0.03:
                    state["bad"] += 1
                else:
                    state["px"] += dx / scale
                state["prev"] = img
                # ใช้ pose ที่เก็บ 'คู่กับเฟรม' (ไม่ใช่ arm.pos_x ตอนนี้ ซึ่งล้ำหน้าไปแล้ว)
                # pose = ท่าที่ controller 'สั่ง' — ตรงกับที่ pose_hist เก็บใน LOCK พอดี
                # → L ที่วัดได้จึงกลืนทั้ง camera latency และ servo lag = ค่าที่ ego-comp ต้องการจริง
                samples.append((t, pose[0], state["px"]))

        th = threading.Thread(target=_sampler, daemon=True)
        th.start()

        t0 = time.time()
        dt_cmd = 1.0 / LAT_CMD_HZ
        while time.time() - t0 < LAT_DURATION_SEC:
            tt = time.time() - t0
            tgt = amp * math.sin(2.0 * math.pi * LAT_FREQ_HZ * tt)
            arm.move_absolute(tgt, 0.0, blocking=False)
            time.sleep(dt_cmd)
        stop.set()
        th.join(timeout=2.0)
        arm.move_absolute(0.0, 0.0, blocking=True)

        if len(samples) < 40:
            raise RuntimeError(f"เก็บได้แค่ {len(samples)} เฟรม — เฟรมเรตต่ำเกินไป")
        if state["bad"] > len(samples) * 0.3:
            raise RuntimeError("ภาพจับคู่ไม่ได้บ่อย — ฉากไม่มี texture?")

        ts = np.array([s[0] for s in samples]) - samples[0][0]
        pans = np.array([s[1] for s in samples])
        pxs = np.array([s[2] for s in samples])
        if float(np.ptp(pans)) < 1.0:
            raise RuntimeError("แขนแทบไม่ขยับ — ต่ออยู่และปลดล็อกแล้วหรือยัง?")

        # ทิศจริงของฉาก 'นิ่ง' ต้องคงที่: bearing(t) = pan(t−L) + feat_px(t)/ppd
        # หา L ที่ทำให้มันคงที่ที่สุด (variance ต่ำสุด)
        def _var_at(L: float) -> float:
            pan_at = np.interp(ts - L, ts, pans)
            return float(np.var(pan_at + pxs / _W.ppd_x))

        vars_ = np.array([_var_at(float(L)) for L in LAT_CANDIDATES])
        i = int(np.argmin(vars_))
        best_L = float(LAT_CANDIDATES[i])
        best_var = float(vars_[i])

        if i == 0 or i == len(LAT_CANDIDATES) - 1:
            _say(f"⚠ ค่าออกมาชนขอบช่วงที่ลอง ({best_L*1000:.0f}ms) — ไม่น่าเชื่อถือ")
        else:
            # พาราโบลาผ่าน 3 จุดรอบจุดต่ำสุด → ได้ความละเอียดกว่า grid
            y0, y1, y2 = vars_[i - 1], vars_[i], vars_[i + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-18:
                step = float(LAT_CANDIDATES[1] - LAT_CANDIDATES[0])
                best_L += 0.5 * step * float(y0 - y2) / den
        best_L = max(0.0, best_L)

        # โค้งต้องมี 'ก้น' ชัด — ถ้าแบน แปลว่า ω แทบไม่เปลี่ยน (แขนไม่ตามไซน์) → เชื่อไม่ได้
        contrast = float(vars_.max() / max(1e-12, vars_.min()))
        _W.latency = best_L
        _say(f"✓ ego_comp_latency = {best_L*1000:.0f} ms "
             f"(σ ทิศ {math.sqrt(best_var):.3f}°, {len(samples)} เฟรม, contrast {contrast:.1f}×)")
        if contrast < 2.0:
            _say("⚠ โค้งแบน (contrast ต่ำ) — แขนอาจตามไซน์ไม่ทัน ผลไม่น่าเชื่อถือ")
    except Exception as e:
        _W.err = str(e)
        _say(f"✗ {e}")
    finally:
        _W.busy = False


# ---------------------------------------------------------------------------
# STEP: noise floor
# ---------------------------------------------------------------------------
def _worker_noise(arm) -> None:
    try:
        if _W.get_detection_center is None:
            raise RuntimeError("ไม่มีตัวป้อน detection")
        _say(f"เล็งไปที่เป้า (แขนนิ่ง) — เก็บ {NOISE_SEC:.0f} วิ …")
        t0 = time.time()
        xs: List[float] = []
        ys: List[float] = []
        tt: List[float] = []
        while time.time() - t0 < NOISE_SEC:
            c = _W.get_detection_center()
            if c is not None:
                xs.append(float(c[0]))
                ys.append(float(c[1]))
                tt.append(time.time() - t0)
            time.sleep(0.02)
        if len(xs) < 20:
            raise RuntimeError(f"จับเป้าได้แค่ {len(xs)} เฟรม — ต้องมีเป้าอยู่ในภาพและ LOCK ติด")

        # หักการเคลื่อนที่เชิงเส้นออกก่อน (เป้าอาจลอยช้า ๆ) → เหลือเฉพาะ 'jitter'
        t = np.asarray(tt)
        rx = np.asarray(xs) - np.polyval(np.polyfit(t, xs, 1), t)
        ry = np.asarray(ys) - np.polyval(np.polyfit(t, ys, 1), t)
        sigma = float(math.hypot(np.std(rx), np.std(ry)) / math.sqrt(2.0))
        _W.sigma_px = sigma
        ppd = abs(_W.ppd_x or 87.0)
        _say(f"✓ σ = {sigma:.1f} px  (= {sigma/ppd:.3f}° ที่ ppd {ppd:.0f}) จาก {len(xs)} เฟรม")
        _say(f"  → LOCK_MEAS_SIGMA_PX = {sigma:.1f}  (Kalman R = (σ/ppd)²)")
    except Exception as e:
        _W.err = str(e)
        _say(f"✗ {e}")
    finally:
        _W.busy = False


# ---------------------------------------------------------------------------
# Mouse (โหมดคลิก ppd + boresight)
# ---------------------------------------------------------------------------
def _on_mouse(event, x, y, flags, param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or not _W.active or _W.busy:
        return
    cx, cy, cw, ch = _W.display
    if cw <= 0 or ch <= 0:
        return
    fx = (float(x) - cx) * _W.frame_w / cw
    fy = (float(y) - cy) * _W.frame_h / ch
    if not (0 <= fx < _W.frame_w and 0 <= fy < _W.frame_h):
        return
    _W.pending_click = (fx, fy)


def _consume_click(arm) -> None:
    """โหมดคลิก: คลิกจุดหนึ่ง → แขนหมุนไปให้จุดนั้นมาอยู่กลางจอ → บันทึกคู่ (px, องศาจริง)"""
    if _W.pending_click is None:
        return
    px, py = _W.pending_click
    _W.pending_click = None

    if _W.step == 4:   # boresight
        if not _W.ppd_x or not _W.ppd_y:
            _say("✗ ต้องมี ppd ก่อน")
            return
        cx, cy = _W.frame_w / 2.0, _W.frame_h / 2.0
        _W.boresight = ((px - cx) / _W.ppd_x, (py - cy) / _W.ppd_y)
        _say(f"✓ boresight = {_W.boresight[0]:+.2f}° / {_W.boresight[1]:+.2f}° "
             f"(จากจุดที่คลิก {px:.0f},{py:.0f})")
        return

    if not _W.click_mode or _W.step != 1:
        return
    # ใช้ ppd ปัจจุบัน (หรือค่าเดา) ทำนายว่าต้องหมุนกี่องศา แล้วให้คนจูนต่อ
    ppx = _W.ppd_x or 60.0
    ppy = _W.ppd_y or -60.0
    cx, cy = _W.frame_w / 2.0, _W.frame_h / 2.0
    pan = (px - cx) / ppx
    tilt = (py - cy) / ppy
    _W.busy = True

    def _go():
        try:
            arm.move_absolute(0.0, 0.0, blocking=True)
            arm.move_absolute(pan, tilt, blocking=True)
            time.sleep(0.3)
            try:
                arm.sync_position_from_grbl()
            except Exception:
                pass
            _W.click_pts.append((px, py, float(arm.pos_x), float(arm.pos_y)))
            _say(f"จุดที่ {len(_W.click_pts)}: px({px:.0f},{py:.0f}) ← แขน({arm.pos_x:+.2f},{arm.pos_y:+.2f})")
            _say("  จูนด้วยลูกศรให้จุดนั้นตรงกลางเป๊ะ แล้วกด Enter ยืนยัน (หรือคลิกจุดใหม่)")
        finally:
            _W.busy = False

    threading.Thread(target=_go, daemon=True).start()


def _fit_click_pts() -> bool:
    if len(_W.click_pts) < 3:
        _say("✗ ต้องมีอย่างน้อย 3 จุด")
        return False
    cx, cy = _W.frame_w / 2.0, _W.frame_h / 2.0
    dpan = [p[2] for p in _W.click_pts]
    dpx = [p[0] - cx for p in _W.click_pts]
    dtilt = [p[3] for p in _W.click_pts]
    dpy = [p[1] - cy for p in _W.click_pts]
    # คลิกที่พิกเซล px แล้วแขนหมุน pan ไปเอามาไว้กลาง → px − cx = ppd_x · pan  (ไม่มีลบ)
    mx, r2x = _fit_through_origin(dpan, dpx)
    my, r2y = _fit_through_origin(dtilt, dpy)
    if abs(mx) < 1.0 or abs(my) < 1.0:
        _say("✗ ค่าที่ได้ผิดปกติ")
        return False
    _W.ppd_x, _W.ppd_y, _W.ppd_r2 = mx, my, (r2x, r2y)
    _say(f"✓ ppd (โหมดคลิก) = {mx:.2f} / {my:.2f}  (r² {r2x:.3f}/{r2y:.3f}, {len(_W.click_pts)} จุด)")
    return True


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_all() -> List[str]:
    """เขียนผลลัพธ์ลงไฟล์ — คืนรายการไฟล์ที่เขียน"""
    written: List[str] = []
    _CALIB_DIR.mkdir(parents=True, exist_ok=True)

    if _W.ppd_x and _W.ppd_y:
        p = _CALIB_DIR / f"{_W.camera}_pixel_per_degree.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "output_width": _W.frame_w,
                "output_height": _W.frame_h,
                "crosshair": {"x": _W.frame_w / 2.0, "y": _W.frame_h / 2.0},
                "pixel_per_degree_x": _W.ppd_x,
                "pixel_per_degree_y": _W.ppd_y,
                "measured_by": "calib_wizard",
                "r2": list(_W.ppd_r2) if _W.ppd_r2 else None,
            }, f, indent=2)
        written.append(p.name)

    if _W.boresight:
        p = _CALIB_DIR / f"{_W.camera}_boresight.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "camera": _W.camera,
                "offset_pan_deg": _W.boresight[0],
                "offset_tilt_deg": _W.boresight[1],
            }, f, indent=2)
        written.append(p.name)

    # latency + sigma ไปอยู่ใน runtime_config (เป็น 'ค่าตั้ง' ไม่ใช่ 'คาลิเบรตเชิงเรขาคณิต')
    if _W.latency is not None or _W.sigma_px is not None:
        try:
            import runtime_config as rc
            data = rc.load()
            if _W.latency is not None:
                data.setdefault("cameras", {}).setdefault(_W.camera, {})["ego_comp_latency_sec"] = round(_W.latency, 4)
            if _W.sigma_px is not None:
                data.setdefault("globals", {})["LOCK_MEAS_SIGMA_PX"] = round(_W.sigma_px, 2)
            if rc.save(data):
                written.append(rc.JSON_PATH.name)
        except Exception as e:
            _say(f"⚠ เขียน runtime_config ไม่ได้: {e}")
    return written


# ---------------------------------------------------------------------------
# tick + keys
# ---------------------------------------------------------------------------
def tick(frame: np.ndarray, arm) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    # ประทับเวลา + ท่าแขน 'ตอนเฟรมมาถึง' พร้อมกัน (convention เดียวกับ pose_hist ใน LOCK)
    t_cap = time.time()
    pose = (float(getattr(arm, "pos_x", 0.0)), float(getattr(arm, "pos_y", 0.0))) if arm else (0.0, 0.0)
    small, _sc = _prep(frame)
    with _W.frame_lock:
        _W.latest_gray = small
        _W.latest_t = t_cap
        _W.latest_pose = pose
    if not _W.busy:
        _consume_click(arm)
    _draw(frame)
    return frame


def _draw(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    s = max(0.35, min(2.0, min(h, w) / 1080.0))
    fs = 0.6 * s
    th = max(1, int(round(1.6 * s)))

    pw = int(w * 0.52)
    x0, y0 = int(24 * s), int(24 * s)
    x1, y1 = x0 + pw, h - int(24 * s)
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), C_BG, -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), C_ACC, max(1, int(2 * s)))

    yy = y0 + int(40 * s)
    cv2.putText(frame, f"CALIBRATION WIZARD — {_W.camera}", (x0 + int(18 * s), yy),
                FONT, 0.78 * s, C_TEXT, max(1, int(2 * s)), cv2.LINE_AA)

    # แถบขั้นตอน
    yy += int(32 * s)
    tx = x0 + int(18 * s)
    for i, name in enumerate(STEPS):
        done = (i == 1 and _W.ppd_x) or (i == 2 and _W.latency is not None) or \
               (i == 3 and _W.sigma_px is not None) or (i == 4 and _W.boresight)
        col = C_OK if done else (C_ACC if i == _W.step else C_DIM)
        txt = f"{i}.{name}"
        cv2.putText(frame, txt, (tx, yy), FONT, fs * 0.9, col, th, cv2.LINE_AA)
        (tw, _th), _base = cv2.getTextSize(txt, FONT, fs * 0.9, th)
        tx += tw + int(14 * s)

    # ผลลัพธ์
    yy += int(34 * s)
    def row(label, val, col=C_TEXT):
        nonlocal yy
        cv2.putText(frame, label, (x0 + int(18 * s), yy), FONT, fs, C_DIM, th, cv2.LINE_AA)
        cv2.putText(frame, val, (x0 + int(240 * s), yy), FONT, fs, col, th, cv2.LINE_AA)
        yy += int(26 * s)

    if _W.ppd_x:
        fh = _W.frame_w / abs(_W.ppd_x)
        fv = _W.frame_h / abs(_W.ppd_y)
        row("ppd (px/องศา)", f"{_W.ppd_x:.2f} / {_W.ppd_y:.2f}", C_OK)
        row("→ FOV จริง", f"{fh:.1f}° × {fv:.1f}°", C_OK)
    else:
        row("ppd", "ยังไม่วัด", C_DIM)
    row("ego latency", f"{_W.latency*1000:.0f} ms" if _W.latency is not None else "ยังไม่วัด",
        C_OK if _W.latency is not None else C_DIM)
    row("σ bbox jitter", f"{_W.sigma_px:.1f} px" if _W.sigma_px is not None else "ยังไม่วัด",
        C_OK if _W.sigma_px is not None else C_DIM)
    row("boresight", f"{_W.boresight[0]:+.2f}° / {_W.boresight[1]:+.2f}°" if _W.boresight else "ยังไม่ตั้ง",
        C_OK if _W.boresight else C_DIM)

    # log
    yy += int(10 * s)
    cv2.line(frame, (x0 + int(18 * s), yy), (x1 - int(18 * s), yy), C_DIM, 1)
    yy += int(22 * s)
    for line in _W.log[-9:]:
        col = C_ERR if line.startswith("✗") else (C_OK if line.startswith("✓") else
              (C_WARN if line.startswith("⚠") else C_DIM))
        cv2.putText(frame, line[:78], (x0 + int(18 * s), yy), FONT, fs * 0.82, col, th, cv2.LINE_AA)
        yy += int(22 * s)

    # คำสั่ง
    hint = _step_hint()
    cv2.putText(frame, hint, (x0 + int(18 * s), y1 - int(44 * s)),
                FONT, fs * 0.85, C_WARN if _W.busy else C_TEXT, th, cv2.LINE_AA)
    cv2.putText(frame, "Space=รัน  N=ถัดไป  B=ย้อน  C=สลับโหมดคลิก  S=บันทึก  Esc/W=ออก",
                (x0 + int(18 * s), y1 - int(18 * s)), FONT, fs * 0.78, C_DIM, th, cv2.LINE_AA)

    # crosshair (โหมดคลิก/boresight ต้องเล็งกลางจอ)
    if _W.step in (1, 4):
        cx, cy = w // 2, h // 2
        r = int(28 * s)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), C_ACC, max(1, int(1.5 * s)))
        cv2.line(frame, (cx, cy - r), (cx, cy + r), C_ACC, max(1, int(1.5 * s)))


def _step_hint() -> str:
    if _W.busy:
        return "กำลังวัด… อย่าขยับกล้อง/แขน"
    if _W.step == 0:
        return "เตรียม: แขนต้องต่ออยู่+ปลดล็อก, ฉากมี texture (ต้นไม้/อาคาร) ไม่ใช่ท้องฟ้าเปล่า → Space"
    if _W.step == 1:
        if _W.click_mode:
            return f"โหมดคลิก: คลิกจุดเด่นในภาพ (มี {len(_W.click_pts)} จุด) → Enter=fit  |  C=กลับโหมดออโต้"
        return "Space = วัด ppd อัตโนมัติ (แขนจะกวาด ±9°)  |  C = ถ้าฉากไม่มี texture ใช้โหมดคลิก"
    if _W.step == 2:
        return "Space = วัด ego latency (แขนจะสะบัด ±4° สองสามรอบ)"
    if _W.step == 3:
        return "เล็งให้เป้าอยู่ในภาพ + LOCK ติด แล้ว Space = วัด noise (แขนต้องนิ่ง)"
    if _W.step == 4:
        return "boresight: เล็งลำกล้องไปจุดอ้างอิง แล้วคลิกจุดที่ 'ลำกล้องชี้จริง' ในภาพ"
    return "S = บันทึกทั้งหมด (ppd/boresight → calibration_data, latency/σ → runtime_config)"


def handle_key(key: int, arm) -> str:
    """คืน 'none' | 'exit' | 'saved'"""
    if key < 0:
        return "none"
    if key in (27, ord("w"), ord("W")):
        return "exit" if not _W.busy else "none"
    if _W.busy:
        return "none"

    if key in (ord("n"), ord("N")):
        _W.step = min(len(STEPS) - 1, _W.step + 1)
        _W.err = ""
    elif key in (ord("b"), ord("B")):
        _W.step = max(0, _W.step - 1)
        _W.err = ""
    elif key in (ord("c"), ord("C")) and _W.step == 1:
        _W.click_mode = not _W.click_mode
        _W.click_pts = []
        _say("โหมดคลิก" if _W.click_mode else "โหมดอัตโนมัติ")
    elif key in (13, 10) and _W.step == 1 and _W.click_mode:
        _fit_click_pts()
    elif key in (ord("s"), ord("S")):
        files = save_all()
        _say(("✓ บันทึก: " + ", ".join(files)) if files else "✗ ไม่มีอะไรให้บันทึก")
        if files:
            _say("  ppd เปลี่ยน → FOV/Kalman R/boresight คำนวณใหม่ให้อัตโนมัติตอนออก")
            return "saved"
    elif key == 32:   # Space = รันขั้นตอนปัจจุบัน
        if arm is None:
            _say("✗ ไม่มีแขนกล — คาลิเบรตไม่ได้")
            return "none"
        if getattr(arm, "is_simulation_mode", False):
            _say("✗ แขนอยู่โหมดจำลอง — ต้องใช้แขนจริง")
            return "none"
        if _W.step == 0:
            _say("แขนพร้อม — ไปขั้น ppd (N)")
            _W.step = 1
        elif _W.step == 1 and not _W.click_mode:
            _W.busy = True
            _W.err = ""
            _W.worker = threading.Thread(target=_worker_ppd, args=(arm,), daemon=True)
            _W.worker.start()
        elif _W.step == 2:
            _W.busy = True
            _W.err = ""
            _W.worker = threading.Thread(target=_worker_latency, args=(arm,), daemon=True)
            _W.worker.start()
        elif _W.step == 3:
            _W.busy = True
            _W.err = ""
            _W.worker = threading.Thread(target=_worker_noise, args=(arm,), daemon=True)
            _W.worker.start()
    return "none"


def results() -> Dict[str, Any]:
    return {
        "ppd_x": _W.ppd_x, "ppd_y": _W.ppd_y,
        "latency": _W.latency, "sigma_px": _W.sigma_px,
        "boresight": _W.boresight,
    }
