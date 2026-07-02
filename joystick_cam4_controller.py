"""
joystick_cam4_controller.py
---------------------------
อินเทอร์เฟซสำหรับควบคุมแขนกล cam4 ด้วยจอยสติ๊ก (เช่น Logitech Extreme 3D Pro)

แนวคิด:
- ใช้ joystick เป็นตัวควบคุมความเร็วหมุน (angular velocity) ของแขน ไม่ใช่มุม absolute
- map ค่าแกน X/Y [-1, 1] → อัตราหมุน (deg/s) → delta_deg ต่อเฟรม (ใช้ dt จาก loop หลัก)
- มี deadzone เพื่อไม่ให้แขนสั่นเมื่อไม่จับ joystick
- มี enum โหมดแขน: AUTO / MANUAL / SAFE ให้ import ไปใช้ใน gun_aim_assist

ข้อกำหนด:
- ต้องมี pygame ติดตั้งและระบบรู้จัก joystick device (เช่น Extreme 3D Pro)
- ถ้าไม่มี pygame หรือไม่พบ joystick, JoystickReader.enabled จะเป็น False และ manual mode จะถูกปิดโดยอัตโนมัติ

แนวทางทดสอบ (lab / ภาคสนามเบื้องต้น):
- ตรวจว่าแขนกลและกล้อง cam4 ทำงานในโหมด AUTO ปกติก่อน (ไม่เสีย calibration)
- เสียบ joystick แล้วรัน gun_aim_assist.py:
    - กดปุ่ม/คีย์ A → MODE: AUTO, ให้ YOLO คุมแขน ตรวจว่าหมุนตามโดรนถูกต้อง
    - กดปุ่ม/คีย์ M → MODE: MANUAL, ขยับ joystick ช้า/เร็ว ดูว่ากล้องหมุนตามลื่น ไม่มี jitter หรือ overshoot รุนแรง
    - กดปุ่ม/คีย์ F → MODE: SAFE, ยืนยันว่าไม่ว่าดัน joystick แรงแค่ไหน แขนก็ไม่ขยับ
- ทดสอบสลับโหมด AUTO ↔ MANUAL หลาย ๆ ครั้ง ขณะกำลังตามเป้า เพื่อดูว่าจุดเล็งไม่กระโดดผิดปกติและ latency ยังคงต่ำ

รันเฉพาะจอยสติ๊ก (ไม่เปิดกล้อง/YOLO):
- จากโฟลเดอร์โปรเจกต์: python joystick_cam4_controller.py
- ใช้ config: CAM4_ARM_SIMULATION_MODE (True=จำลองแขน), CAM4_ARM_ENABLED (แขนจริงต้อง True)
- ออกด้วย Ctrl+C
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pygame

    _HAS_PYGAME = True
except ImportError:
    pygame = None  # type: ignore
    _HAS_PYGAME = False


# โหมดควบคุมแขน
MODE_AUTO = 0
MODE_MANUAL = 1
MODE_SAFE = 2
MODE_LOCK = 3

# โหมดความเร็วจอยสติ๊ก (ช้า/กลาง/สูง) — สลับด้วยปุ่ม 12 (index 11)
SENSITIVITY_LOW = 0
SENSITIVITY_MEDIUM = 1
SENSITIVITY_HIGH = 2
# preset ต่อโหมด: (rate_multiplier, stick_exponent)
SENSITIVITY_PRESETS = {
    SENSITIVITY_LOW: (0.3, 3.0),
    SENSITIVITY_MEDIUM: (0.55, 2.5),
    SENSITIVITY_HIGH: (1.0, 2.0),
}
SENSITIVITY_LABELS = ("Slow", "Medium", "High")

# Joystick hot-plug: consecutive read failures before disconnect; poll/reconnect intervals (sec)
JOY_READ_FAIL_DISCONNECT = 3
JOY_CONNECT_CONFIRM_READS = 3
JOY_RECONNECT_INTERVAL_SEC = 1.5
JOY_LINK_CHECK_INTERVAL_SEC = 2.0


def _joy_config_float(name: str, default: float) -> float:
    try:
        import config as _cfg
        return float(getattr(_cfg, name, default))
    except Exception:
        return default


def _joy_event_type(name: str):
    if not _HAS_PYGAME:
        return None
    return getattr(pygame, name, None)


def _parse_sensitivity_mode(value: str) -> int:
    """แปลง 'low'/'medium'/'high' หรือตัวเลข → 0/1/2."""
    if isinstance(value, int) and 0 <= value <= 2:
        return value
    s = str(value).strip().lower()
    if s in ("low", "ช้า", "0"):
        return SENSITIVITY_LOW
    if s in ("high", "สูง", "2"):
        return SENSITIVITY_HIGH
    return SENSITIVITY_MEDIUM  # "medium", "กลาง", "1", default


@dataclass
class JoystickState:
    """สถานะล่าสุดของ joystick."""

    axis_x: float = 0.0
    axis_y: float = 0.0
    speed_axis: float = 0.0  # axis 3: [-1,1] → scale 0.1–10 (ละเอียด→หยาบ)
    mode_switch: Optional[int] = None  # None หรือ MODE_AUTO / MODE_MANUAL / MODE_SAFE / MODE_LOCK
    fire_pressed: bool = False         # ปุ่มยิง (trigger): เฉพาะเมื่อกด Hat ค้าง + ปุ่ม 1 (index 0)
    confirm_pressed: bool = False      # ปุ่ม 1 (index 0) ล้วน — ใช้โหมดเทส Px/Deg: ยืนยัน/กลับ home
    sensitivity_cycle_pressed: bool = False  # ปุ่ม 12 (index 11): สลับโหมดความเร็ว ช้า→กลาง→สูง
    csrt_center_bbox_pressed: bool = False   # ปุ่ม 4 (index 3): สร้าง CSRT bbox รอบศูนย์เล็ง (โหมด LOCK)
    unlock_pressed: bool = False             # ปุ่ม 3 (index 2): ปลด LOCK → MANUAL
    lock_csrt_move_pressed: bool = False    # ปุ่ม 6 (index 5): ขยับแขนไป CSRT bbox (เหมือนปุ่ม 5 สำหรับ YOLO)
    class_cycle_pressed: bool = False       # ปุ่ม CLASS_CYCLE (ค่าเริ่มต้น 8): สลับ YOLO class — กดค้างเลื่อนต่อเนื่อง
    detection_engine_toggle_pressed: bool = False  # ปุ่ม DETECTION_TOGGLE (ค่าเริ่มต้น 10): สลับ engine RGB/thermal
    hat_held: bool = False                  # มี Hat (ทุกทิศ บน/ล่าง/กลาง/ซ้าย/ขวา) = เปิดโหมดทำนายจุด 0.3 s


class JoystickReader:
    """
    อ่านค่าจาก joystick ตัวแรกในระบบ (index 0).
    หมายเลขปุ่มใช้ตาม check_joystick_buttons.py: ปุ่มหมายเลข N = pygame index (N-1).
    ปุ่มโหมด/ยิงอ่านจาก config (CAM4_ARM_JOYSTICK_BUTTON_*).

    - ใช้ axis 0 (ซ้าย/ขวา) เป็น pan, axis 1 (หน้า/หลัง) เป็น tilt
    - ยิง: กด Hat ค้าง + ปุ่ม FIRE_AND_LOCK (ปุ่ม 1); LOCK: ปุ่ม LOCK (ปุ่ม 5, index 4)
    - โหมด: ปุ่ม AUTO / MANUAL / SAFE ตาม config (ค่าเริ่มต้น 7, 9, 11)
    """

    def __init__(self) -> None:
        self.enabled = False
        self.joystick: Optional["pygame.joystick.Joystick"] = None
        self._read_fail_count = 0
        self._last_link_check_t = 0.0
        self._last_reconnect_t = 0.0
        self._reconnect_in_progress = False
        self._pending_probe = False
        self._connect_confirm_count = 0
        self._tracked_instance_id: Optional[int] = None
        self.pygame_available = _HAS_PYGAME

        if not _HAS_PYGAME:
            print("JoystickReader: pygame not available, manual mode disabled.")
            return

        if not self.try_reconnect() and not self._pending_probe:
            print("JoystickReader: no joystick detected — will retry while running.")

    def _release_joystick(self) -> None:
        if self.joystick is not None:
            try:
                self.joystick.quit()
            except Exception:
                pass
        self.joystick = None
        self._tracked_instance_id = None

    def _mark_disconnected(self, reason: str = "") -> None:
        """Mark joystick offline only — does not touch arm/camera."""
        was_link = self.enabled or self._pending_probe
        self.enabled = False
        self._pending_probe = False
        self._connect_confirm_count = 0
        self._release_joystick()
        self._read_fail_count = 0
        if was_link:
            msg = "JoystickReader: disconnected"
            if reason:
                msg += f" ({reason})"
            print(f"{msg} — auto-reconnecting (arm unchanged).")

    def _remember_instance_id(self) -> None:
        if self.joystick is None:
            self._tracked_instance_id = None
            return
        try:
            self._tracked_instance_id = int(self.joystick.get_instance_id())
        except Exception:
            self._tracked_instance_id = None

    def _probe_axis_ok(self) -> bool:
        if self.joystick is None:
            return False
        try:
            if not self.joystick.get_init():
                return False
            pygame.event.pump()
            self.joystick.get_axis(0)
            if self.joystick.get_numaxes() > 1:
                self.joystick.get_axis(1)
            return True
        except Exception:
            return False

    def _verify_handle_alive(self) -> bool:
        return self._probe_axis_ok()

    def _refresh_sdl_device_count(self) -> int:
        if not _HAS_PYGAME:
            return 0
        if not pygame.get_init():
            pygame.init()
        pygame.joystick.quit()
        pygame.joystick.init()
        return int(pygame.joystick.get_count())

    def _instance_still_listed(self) -> bool:
        """Heavy check only after handle read fails — avoids OK/DISC flicker."""
        if self.joystick is None or self._tracked_instance_id is None:
            return False
        tracked = int(self._tracked_instance_id)
        self._release_joystick()
        try:
            count = self._refresh_sdl_device_count()
            if count == 0:
                return False
            for i in range(count):
                js = pygame.joystick.Joystick(i)
                js.init()
                try:
                    if int(js.get_instance_id()) == tracked:
                        self.joystick = js
                        self._tracked_instance_id = tracked
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def _open_first_joystick(self) -> bool:
        if not _HAS_PYGAME:
            return False
        try:
            if not pygame.get_init():
                pygame.init()
            pygame.event.pump()
            count = self._refresh_sdl_device_count()
            if count == 0:
                return False
            js = pygame.joystick.Joystick(0)
            js.init()
            self.joystick = js
            self._remember_instance_id()
            return True
        except Exception:
            self._release_joystick()
            return False

    def _finish_connect(self) -> None:
        self.enabled = True
        self._pending_probe = False
        self._connect_confirm_count = 0
        self._read_fail_count = 0
        name = "?"
        n_axes = 0
        if self.joystick is not None:
            try:
                name = self.joystick.get_name()
                n_axes = self.joystick.get_numaxes()
            except Exception:
                pass
        print(f"JoystickReader: connected '{name}' ({n_axes} axes).")

    def _advance_pending_probe(self) -> None:
        if not self._pending_probe or self.joystick is None:
            return
        if self._probe_axis_ok():
            self._connect_confirm_count += 1
            need = int(_joy_config_float("JOY_CONNECT_CONFIRM_READS", JOY_CONNECT_CONFIRM_READS))
            need = max(1, need)
            if self._connect_confirm_count >= need:
                self._finish_connect()
        else:
            self._connect_confirm_count = 0
            self._release_joystick()
            self._pending_probe = False

    def _process_hotplug_events(self) -> None:
        if not _HAS_PYGAME:
            return
        joy_removed = _joy_event_type("JOYDEVICEREMOVED")
        joy_added = _joy_event_type("JOYDEVICEADDED")
        for event in pygame.event.get():
            if joy_removed is not None and event.type == joy_removed:
                if self.joystick is None:
                    self._mark_disconnected("removed")
                    break
                try:
                    if int(event.instance_id) == int(self.joystick.get_instance_id()):
                        self._mark_disconnected("unplugged")
                        break
                except Exception:
                    self._mark_disconnected("removed")
            elif (
                joy_added is not None
                and event.type == joy_added
                and not self.enabled
                and not self._pending_probe
            ):
                self._last_reconnect_t = 0.0

    def tick(self, now: Optional[float] = None) -> None:
        """
        Hot-plug poll + auto-reconnect (call once per main loop).
        Independent from arm reconnect/homing.
        """
        if not _HAS_PYGAME:
            return
        if now is None:
            now = time.monotonic()
        self._process_hotplug_events()

        if self._pending_probe:
            self._advance_pending_probe()
            return

        link_iv = _joy_config_float("JOY_LINK_CHECK_INTERVAL_SEC", JOY_LINK_CHECK_INTERVAL_SEC)
        reconnect_iv = _joy_config_float("JOY_RECONNECT_INTERVAL_SEC", JOY_RECONNECT_INTERVAL_SEC)

        if self.enabled:
            if now - self._last_link_check_t >= link_iv:
                self._last_link_check_t = now
                if not self._verify_handle_alive():
                    if not self._instance_still_listed():
                        self._mark_disconnected("device gone")
        elif now - self._last_reconnect_t >= reconnect_iv:
            self._last_reconnect_t = now
            self._reconnect_in_progress = True
            try:
                self.try_reconnect()
            finally:
                self._reconnect_in_progress = False

    def try_reconnect(self) -> bool:
        """Probe open index 0; JOY:OK only after consecutive confirmed reads."""
        if self.enabled and self.joystick is not None and self._verify_handle_alive():
            return True
        if self.enabled:
            self._mark_disconnected("stale handle")
        if not _HAS_PYGAME:
            return False
        self._release_joystick()
        if not self._open_first_joystick():
            self.enabled = False
            self._pending_probe = False
            return False
        self.enabled = False
        self._pending_probe = True
        self._connect_confirm_count = 0
        self._advance_pending_probe()
        return self.enabled

    def read(self) -> JoystickState:
        """
        อ่านค่าปัจจุบันจาก joystick.
        หมายเลขปุ่มใช้ตาม check_joystick_buttons.py: ปุ่มหมายเลข N = pygame index (N-1).
        ปุ่มโหมด/ยิงอ่านจาก config (CAM4_ARM_JOYSTICK_BUTTON_*).

        Returns:
            JoystickState (axes ในช่วง [-1,1], mode_switch อาจเป็น None หรือ MODE_*).
        """
        state = JoystickState()
        if not self.enabled or self.joystick is None:
            return state

        try:
            import config as _cfg
            btn_fire_lock = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_FIRE_AND_LOCK", 1)
            btn_lock = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_LOCK", 5)
            btn_auto = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_AUTO", 7)
            btn_manual = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_MANUAL", 9)
            btn_safe = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_SAFE", 11)
            idx_fire_lock = max(0, int(btn_fire_lock) - 1)
            idx_lock = max(0, int(btn_lock) - 1)
            idx_auto = max(0, int(btn_auto) - 1)
            idx_manual = max(0, int(btn_manual) - 1)
            idx_safe = max(0, int(btn_safe) - 1)
        except Exception:
            idx_fire_lock, idx_lock, idx_auto, idx_manual, idx_safe = 0, 1, 6, 8, 10

        try:
            pygame.event.pump()
            # axis 0 = left/right, axis 1 = forward/back ค่านี้อาจกลับด้านตาม OS/driver
            axis_x = float(self.joystick.get_axis(0))
            axis_y = float(self.joystick.get_axis(1))
            state.axis_x = axis_x
            state.axis_y = axis_y
            # axis 3 = คันปรับ step (ละเอียด 0.1 mm ↔ หยาบ 10 mm)
            n_axes = self.joystick.get_numaxes()
            if n_axes > 3:
                try:
                    state.speed_axis = float(self.joystick.get_axis(3))
                except Exception:
                    state.speed_axis = 0.0
            else:
                state.speed_axis = 0.0  # scale = 1.0

            # อ่าน Hat (ถ้ามี) — ใช้ร่วมกับปุ่มยิง และแยก LOCK กับยิง
            hat = (0, 0)
            if self.joystick.get_numhats() > 0:
                try:
                    hat = self.joystick.get_hat(0)
                except Exception:
                    pass
            # โหมดทำนาย: บน ล่าง กลาง ซ้าย ขวา (มี Hat = เปิดโหมดทำนาย)
            state.hat_held = self.joystick.get_numhats() > 0

            # ปุ่มยิง + เปลี่ยนโหมด (หมายเลขปุ่มตรงกับ check_joystick_buttons: หมายเลข N = index N-1)
            n_buttons = self.joystick.get_numbuttons()
            try:
                btn_fire = self.joystick.get_button(idx_fire_lock) if idx_fire_lock < n_buttons else False
                # ปุ่ม 1 (index 0) ล้วน — สำหรับโหมดเทส confirm / กลับ home
                if n_buttons > 0:
                    state.confirm_pressed = bool(self.joystick.get_button(0))
                # ยิง: กด Hat ด้านใดก็ได้ค้าง + ปุ่มยิง (ปุ่ม 1)
                if hat != (0, 0) and btn_fire:
                    state.fire_pressed = True
                # ปุ่ม LOCK (ปุ่ม 5, index 4)
                if idx_lock < n_buttons and self.joystick.get_button(idx_lock):
                    state.mode_switch = MODE_LOCK
                # ปุ่ม AUTO / MANUAL / SAFE (ตาม config)
                elif idx_auto < n_buttons and self.joystick.get_button(idx_auto):
                    state.mode_switch = MODE_AUTO
                elif idx_manual < n_buttons and self.joystick.get_button(idx_manual):
                    state.mode_switch = MODE_MANUAL
                elif idx_safe < n_buttons and self.joystick.get_button(idx_safe):
                    state.mode_switch = MODE_SAFE
                # ปุ่ม 12 (index 11): สลับโหมดความเร็วจอยสติ๊ก (ช้า/กลาง/สูง)
                idx_sensitivity = 11  # pygame index for button 12
                try:
                    btn_sens = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_SENSITIVITY_CYCLE", 12)
                    idx_sensitivity = max(0, int(btn_sens) - 1)
                except Exception:
                    pass
                if idx_sensitivity < n_buttons and self.joystick.get_button(idx_sensitivity):
                    state.sensitivity_cycle_pressed = True
                # ปุ่ม 4 (index 3): สร้าง CSRT bbox รอบศูนย์เล็ง (ใช้ใน gun_aim_assist_vector โหมด LOCK)
                if 3 < n_buttons and self.joystick.get_button(3):
                    state.csrt_center_bbox_pressed = True
                # ปุ่ม 3 (index 2): ปลด LOCK → MANUAL
                try:
                    idx_unlock = max(0, int(getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_UNLOCK", 3)) - 1)
                except Exception:
                    idx_unlock = 2
                if idx_unlock < n_buttons and self.joystick.get_button(idx_unlock):
                    state.unlock_pressed = True
                # ปุ่ม 6 (index 5): ขยับแขนไป CSRT bbox (เหมือนปุ่ม 5 สำหรับ YOLO)
                try:
                    idx_lock_csrt = max(0, int(getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_LOCK_CSRT", 6)) - 1)
                except Exception:
                    idx_lock_csrt = 5
                if idx_lock_csrt < n_buttons and self.joystick.get_button(idx_lock_csrt):
                    state.lock_csrt_move_pressed = True
                # ปุ่มสลับ YOLO class (ค่าเริ่มต้น 8, index 7) — แยกจากปุ่มยิง/โหมด
                try:
                    btn_cc = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_CLASS_CYCLE", 8)
                    idx_class_cycle = max(0, int(btn_cc) - 1)
                except Exception:
                    idx_class_cycle = 7
                if idx_class_cycle < n_buttons and self.joystick.get_button(idx_class_cycle):
                    state.class_cycle_pressed = True
                try:
                    btn_det = getattr(_cfg, "CAM4_ARM_JOYSTICK_BUTTON_DETECTION_TOGGLE", 10)
                    idx_det_toggle = max(0, int(btn_det) - 1)
                except Exception:
                    idx_det_toggle = 9
                if idx_det_toggle < n_buttons and self.joystick.get_button(idx_det_toggle):
                    state.detection_engine_toggle_pressed = True
            except Exception:
                pass
            self._read_fail_count = 0
        except Exception:
            self._read_fail_count += 1
            if self._read_fail_count >= JOY_READ_FAIL_DISCONNECT:
                self._mark_disconnected("read failed")
        return state


class JoystickArmMapper:
    """
    แปลง JoystickState + dt → delta pan/tilt (หน่วยเดียวกับ limit, มักเป็น mm).

    - คันหลัก (axis 0,1): โยกมากเร็วมาก โยกน้อยเร็วน้อย
    - Axis 3: scale ต่อเนื่อง 0.1–10 (ละเอียด↔หยาบ) แบบ smooth
    - โหมดความเร็ว (ช้า/กลาง/สูง): สลับด้วยปุ่ม 12; ใช้ multiplier + exponent ตาม SENSITIVITY_PRESETS
    """

    def __init__(
        self,
        arm_controller,
        max_pan_rate_deg: float = 80.0,
        max_tilt_rate_deg: float = 60.0,
        deadzone: float = 0.05,
        stick_exponent: float = 2.0,
        speed_axis_scale_min: float = 0.1,
        speed_axis_scale_max: float = 10.0,
        speed_smooth_alpha: float = 0.82,
        initial_sensitivity_mode: Optional[int] = None,
    ) -> None:
        self.arm = arm_controller
        self._base_max_pan_rate_deg = max_pan_rate_deg
        self._base_max_tilt_rate_deg = max_tilt_rate_deg
        self.max_pan_rate_deg = max_pan_rate_deg
        self.max_tilt_rate_deg = max_tilt_rate_deg
        self.deadzone = deadzone
        self.stick_exponent = max(0.5, min(4.0, stick_exponent))
        self.speed_axis_scale_min = speed_axis_scale_min
        self.speed_axis_scale_max = speed_axis_scale_max
        self.speed_smooth_alpha = max(0.01, min(0.99, speed_smooth_alpha))
        self._speed_axis_smooth: Optional[float] = None
        if initial_sensitivity_mode is not None:
            self.sensitivity_mode = max(0, min(2, int(initial_sensitivity_mode)))
        else:
            self.sensitivity_mode = SENSITIVITY_MEDIUM

    def _get_effective_params(self) -> Tuple[float, float, float]:
        """คืน (effective_pan_rate, effective_tilt_rate, stick_exponent) ตาม sensitivity_mode."""
        mult, exp = SENSITIVITY_PRESETS.get(
            self.sensitivity_mode, SENSITIVITY_PRESETS[SENSITIVITY_MEDIUM]
        )
        pan = self._base_max_pan_rate_deg * mult
        tilt = self._base_max_tilt_rate_deg * mult
        exp = max(0.5, min(4.0, exp))
        return pan, tilt, exp

    def cycle_sensitivity_mode(self) -> None:
        """สลับโหมดความเร็ว: ช้า → กลาง → สูง → ช้า."""
        self.sensitivity_mode = (self.sensitivity_mode + 1) % 3

    def get_sensitivity_label(self) -> str:
        """คืนข้อความโหมดความเร็วสำหรับ HUD: 'ช้า' / 'กลาง' / 'สูง'."""
        return SENSITIVITY_LABELS[self.sensitivity_mode]

    def _speed_scale_from_axis(self, speed_axis: float) -> float:
        """แมป speed_axis [-1,1] → scale [scale_min, scale_max] แบบ log (0.1, 1, 10 เนียน)."""
        s = max(-1.0, min(1.0, speed_axis))
        # log10(scale) แปรเชิงเส้นกับ s: s=-1 → 0.1, s=0 → 1, s=1 → 10
        log_min = math.log10(self.speed_axis_scale_min)
        log_max = math.log10(self.speed_axis_scale_max)
        log_scale = log_min + (s + 1.0) * 0.5 * (log_max - log_min)
        return float(max(self.speed_axis_scale_min, min(self.speed_axis_scale_max, 10.0 ** log_scale)))

    def compute_delta(self, state: JoystickState, dt: float) -> Tuple[float, float]:
        """
        คำนวณ delta จาก joystick state. คันหลักแปรผันความเร็ว; axis 3 ปรับ scale 0.1–10 แบบ smooth.
        """
        if dt <= 0.0:
            return 0.0, 0.0

        x = state.axis_x
        y = state.axis_y

        # deadzone
        if abs(x) < self.deadzone:
            x = 0.0
        if abs(y) < self.deadzone:
            y = 0.0

        if x == 0.0 and y == 0.0:
            return 0.0, 0.0

        eff_pan_rate, eff_tilt_rate, eff_exponent = self._get_effective_params()
        # ระยะโยก → ความเร็ว: ใช้ power curve โยกนิดเดียวไม่ถึงความเร็วสูง (exponent > 1)
        def apply_curve(v: float, exp: float) -> float:
            if v == 0.0:
                return 0.0
            return (1.0 if v > 0 else -1.0) * (abs(v) ** exp)

        x_eff = apply_curve(x, eff_exponent)
        y_eff = apply_curve(y, eff_exponent)
        pan_rate = eff_pan_rate * x_eff
        tilt_rate = eff_tilt_rate * y_eff

        delta_pan = pan_rate * dt
        delta_tilt = tilt_rate * dt

        # Axis 3: scale 0.1–10 (ละเอียด↔หยาบ) + low-pass ให้เนียน
        raw_scale = self._speed_scale_from_axis(state.speed_axis)
        if self._speed_axis_smooth is None:
            self._speed_axis_smooth = raw_scale
        else:
            self._speed_axis_smooth = (
                self.speed_smooth_alpha * self._speed_axis_smooth
                + (1.0 - self.speed_smooth_alpha) * raw_scale
            )
        scale = self._speed_axis_smooth
        delta_pan *= scale
        delta_tilt *= scale

        return float(delta_pan), float(delta_tilt)

    def apply(self, state: JoystickState, dt: float) -> None:
        """คำนวณและสั่ง move_relative ไปที่แขนกล (ถ้ามี delta)."""
        if self.arm is None or getattr(self.arm, "is_simulation_mode", False):
            return
        dx, dy = self.compute_delta(state, dt)
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return
        self.arm.move_relative(dx, dy, blocking=False)


class ArmModeManager:
    """ตัวจัดการโหมดแขนกล (AUTO / MANUAL / SAFE / LOCK)."""

    def __init__(self, has_arm: bool) -> None:
        self.mode: int = MODE_MANUAL if has_arm else MODE_SAFE

    def set_mode(self, new_mode: int) -> None:
        if new_mode in (MODE_AUTO, MODE_MANUAL, MODE_SAFE, MODE_LOCK):
            self.mode = new_mode

    def label_and_color(self) -> Tuple[str, Tuple[int, int, int]]:
        """คืน (ข้อความ, สี BGR) สำหรับแสดงบน HUD."""
        if self.mode == MODE_AUTO:
            return "MODE: AUTO", (0, 255, 0)  # เขียว
        if self.mode == MODE_MANUAL:
            return "MODE: MANUAL", (0, 165, 255)  # ส้ม
        if self.mode == MODE_LOCK:
            return "MODE: LOCK", (255, 0, 255)  # ม่วง (BGR)
        return "MODE: SAFE", (0, 0, 255)  # แดง


# ---------------------------------------------------------------------------
# Standalone: รันเฉพาะจอยสติ๊กควบคุมแขน (ไม่เปิดกล้อง/YOLO)
# ---------------------------------------------------------------------------
def _run_joystick_arm_only() -> None:
    """
    โหมดทดสอบ: เปิดแขนกล + จอยสติ๊กอย่างเดียว
    - อ่าน config (CAM4_ARM_*, CAM4_ARM_SIMULATION_MODE)
    - ถ้า simulation = True ใช้ SimCam4ArmController (ไม่ต่อ Serial)
    - ลูป: อ่านจอย → move_relative; กด Q หรือ ESC ออก
    - ออกปกติหรือกระทันหัน (Ctrl+C / kill): แขนกลับ home ก่อนปิด
    """
    import signal
    import sys
    import time

    try:
        import config as cfg_mod
    except ImportError:
        print("joystick_cam4_controller: config not found. Run from project root.")
        sys.exit(1)

    use_sim = getattr(cfg_mod, "CAM4_ARM_SIMULATION_MODE", False)
    use_arm = getattr(cfg_mod, "CAM4_ARM_ENABLED", True)

    try:
        from cam4_arm_controller import Cam4ArmController, SimCam4ArmController
    except ImportError as e:
        print(f"joystick_cam4_controller: cam4_arm_controller not found: {e}")
        sys.exit(1)

    if use_sim:
        print("Joystick-only mode: using SimCam4ArmController (no serial).")
        arm_controller = SimCam4ArmController()
    else:
        if not use_arm:
            print("CAM4_ARM_ENABLED is False. Set to True in config for real arm, or CAM4_ARM_SIMULATION_MODE=True for sim.")
        print("Joystick-only mode: using Cam4ArmController (real arm).")
        arm_controller = Cam4ArmController()

    # อ้างอิงสำหรับ signal handler (ออกกระทันหัน → กลับ home แล้วปิด)
    _arm_ref = [arm_controller]

    def _exit_return_home(sig, frame):
        print("\n⚠️ สัญญาณออก — ส่งแขนกลับ home ก่อนปิด...")
        try:
            if _arm_ref[0] is not None:
                _arm_ref[0].disconnect()
                _arm_ref[0] = None
        except Exception as e:
            print(f"   ⚠️ {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_return_home)
    signal.signal(signal.SIGTERM, _exit_return_home)

    if not arm_controller.connect():
        print("Arm connect() failed. Exit.")
        sys.exit(1)

    reader = JoystickReader()
    if not getattr(reader, "enabled", False):
        print("No joystick detected. Plug in joystick and run again.")
        arm_controller.disconnect()
        sys.exit(1)

    max_pan = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG", 80.0)
    max_tilt = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG", 60.0)
    deadzone = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_DEADZONE", 0.05)
    stick_exp = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_STICK_EXPONENT", 2.0)
    scale_min = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MIN", 0.1)
    scale_max = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_SPEED_SCALE_MAX", 10.0)
    speed_smooth = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_SPEED_SMOOTH_ALPHA", 0.82)
    sens_str = getattr(cfg_mod, "CAM4_ARM_JOYSTICK_SENSITIVITY", "medium")
    initial_sens = _parse_sensitivity_mode(sens_str)
    mapper = JoystickArmMapper(
        arm_controller,
        max_pan_rate_deg=max_pan,
        max_tilt_rate_deg=max_tilt,
        deadzone=deadzone,
        stick_exponent=stick_exp,
        speed_axis_scale_min=scale_min,
        speed_axis_scale_max=scale_max,
        speed_smooth_alpha=speed_smooth,
        initial_sensitivity_mode=initial_sens,
    )

    print("--- Joystick-only arm control ---")
    print("  Axis 0,1: pan/tilt (โยกมากเร็วมาก). Axis 3: scale 0.1–10 (ละเอียด↔หยาบ).")
    print("  Button 12: cycle sensitivity (ช้า/กลาง/สูง).")
    print("  Press Q or ESC in terminal to quit (or close window if using a window).")
    print("  This script has no window; use Ctrl+C to stop if needed.")
    last_print = 0.0
    last_loop = time.time()
    prev_sensitivity_cycle_pressed = False

    try:
        while True:
            t = time.time()
            dt = t - last_loop
            last_loop = t
            if dt <= 0 or dt > 0.5:
                dt = 0.02

            state = reader.read()
            if state.sensitivity_cycle_pressed and not prev_sensitivity_cycle_pressed:
                mapper.cycle_sensitivity_mode()
            prev_sensitivity_cycle_pressed = state.sensitivity_cycle_pressed

            dx, dy = mapper.compute_delta(state, dt)
            # ส่งคำสั่งทั้งแขนจริงและจำลอง (mapper.apply ข้ามโหมด sim จึงเรียก move_relative ตรงๆ)
            if abs(dx) >= 1e-3 or abs(dy) >= 1e-3:
                arm_controller.move_relative(dx, dy, blocking=False)

            if t - last_print >= 0.5:
                px = getattr(arm_controller, "pos_x", 0.0)
                py = getattr(arm_controller, "pos_y", 0.0)
                joy_label = mapper.get_sensitivity_label()
                print(f"  arm X={px:.2f}° Y={py:.2f}°  axis_x={state.axis_x:.2f} axis_y={state.axis_y:.2f}  JOY: {joy_label}")
                last_print = t

            # ปุ่ม 3 = SAFE ใช้เป็น "หยุดไม่ส่งคำสั่ง" ก็ได้; ออกด้วย Ctrl+C
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if _arm_ref[0] is not None:
            arm_controller.disconnect()
            _arm_ref[0] = None


if __name__ == "__main__":
    _run_joystick_arm_only()


