"""
Cam4 Arm Controller
-------------------
- ควบคุมแขนกล GRBL (cam4) ผ่าน Serial / G-code
- Homing: $X → $HY → $HX → G53 → G92 X0 Y0 (ผ่าน cam4_arm_homing)
- จำกัดการหมุนด้วย CAM4_ARM_X_LIMITS, CAM4_ARM_Y_LIMITS (หน่วย mm) — clip ก่อนส่ง G0 ทุกครั้ง
"""

import atexit
import os
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
import serial

import config

try:
    import cam4_arm_homing
except ImportError:
    cam4_arm_homing = None

try:
    from cam4_arm_homing import parse_grbl_status
except ImportError:
    parse_grbl_status = None


_global_cam4_arm_controller = None


class Cam4ArmController:
    """
    Controller แขน GRBL สำหรับ cam4.
    - อ่าน limit จาก config และ clip ตำแหน่งก่อนส่ง G0 (กันหมุนเกิน limit ค้าง)
    """

    def __init__(self) -> None:
        global _global_cam4_arm_controller
        _global_cam4_arm_controller = self

        self.enabled = getattr(config, "CAM4_ARM_ENABLED", False)
        self.is_simulation_mode = not self.enabled

        self.port: str = getattr(config, "CAM4_ARM_SERIAL_PORT", "/dev/ttyUSB0")
        self.baud: int = getattr(config, "CAM4_ARM_BAUD_RATE", 115200)
        self.timeout: float = getattr(config, "CAM4_ARM_SERIAL_TIMEOUT", 0.1)

        # ขีดจำกัด X,Y หน่วย mm จาก home — clip ทุกครั้งก่อนส่ง G0
        self.x_limits = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0))
        self.y_limits = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0))
        # safety margin ด้านใน limit เพื่อกัน drift เกิน (หน่วยเดียวกับ limit)
        self.limit_safety_margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0)
        self._effective_x_limits = (
            self.x_limits[0] + self.limit_safety_margin,
            self.x_limits[1] - self.limit_safety_margin,
        )
        self._effective_y_limits = (
            self.y_limits[0] + self.limit_safety_margin,
            self.y_limits[1] - self.limit_safety_margin,
        )
        # ซิงก์ตำแหน่งจาก GRBL ทุก N ครั้งที่เรียก move_relative (ลด drift)
        self.sync_every_n_moves = getattr(config, "CAM4_ARM_SYNC_POSITION_EVERY_N_MOVES", 8)
        self._move_relative_count = 0
        # throttle: ส่ง G0 ได้อย่างมากทุก min_move_interval วินาที (ตอนไม่ blocking)
        self._min_move_interval_sec = getattr(config, "CAM4_ARM_MIN_MOVE_INTERVAL_MS", 60) / 1000.0
        self._last_g0_send_time: float = 0.0

        self.ref_pan_deg = getattr(config, "CAM4_ARM_REF_PAN_DEG", 0.0)
        self.ref_tilt_deg = getattr(config, "CAM4_ARM_REF_TILT_DEG", 0.0)
        self.grbl_units_mm = getattr(config, "CAM4_ARM_GRBL_UNITS_MM", True)
        self.mm_per_deg_pan = getattr(config, "CAM4_ARM_MM_PER_DEG_PAN", 1.0)
        self.mm_per_deg_tilt = getattr(config, "CAM4_ARM_MM_PER_DEG_TILT", 1.0)

        ref_pan_mm_cfg = getattr(config, "CAM4_ARM_REF_PAN_MM", None)
        ref_tilt_mm_cfg = getattr(config, "CAM4_ARM_REF_TILT_MM", None)
        if ref_pan_mm_cfg is not None and ref_tilt_mm_cfg is not None:
            self.ref_pan_mm = float(ref_pan_mm_cfg)
            self.ref_tilt_mm = float(ref_tilt_mm_cfg)
        else:
            self.ref_pan_mm = self.ref_pan_deg * self.mm_per_deg_pan
            self.ref_tilt_mm = self.ref_tilt_deg * self.mm_per_deg_tilt

        self.default_feed_rate = getattr(config, "CAM4_ARM_FEED_RATE", 10000)
        self.current_feed_rate = self.default_feed_rate

        self.pos_x: float = 0.0
        self.pos_y: float = 0.0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.ser: Optional[serial.Serial] = None
        self.is_healthy: bool = True
        self.motion_verified: bool = True
        self.last_arm_activity_t: float = 0.0
        self.last_motion_probe_t: float = 0.0
        self._probe_in_progress: bool = False
        self._reconnect_in_progress: bool = False
        self._last_reconnect_attempt_t: float = 0.0
        self._reconnect_fail_count: int = 0
        self._last_connect_ok_t: float = 0.0
        self._stall_since_t: float = 0.0
        self._grbl_status_fail_streak: int = 0
        self._last_stall_recovery_t: float = 0.0
        self._last_stall_query_t: float = 0.0
        self._last_fault_recovery_t: float = 0.0
        self._idle_probe_fail_streak: int = 0
        self._grbl_read_fail_streak: int = 0
        self._homing_fault: Optional[str] = None  # "no_motion" | "homing_fail"
        self._operator_joystick_active: bool = False
        self._camera_operator_lock: bool = False
        self._camera_probe_grace_until_t: float = 0.0
        self._drive_verify_pan0: Optional[float] = None
        self._drive_verify_tilt0: Optional[float] = None
        self._drive_verify_cmd_deg: float = 0.0
        self._drive_verify_due_t: float = 0.0
        self._joy_session_pan0: Optional[float] = None
        self._joy_session_tilt0: Optional[float] = None
        self._joy_session_cmd: float = 0.0
        self._joy_session_verify_due_t: float = 0.0
        self._last_grbl_read_ok_t: float = 0.0
        self._last_grbl_read_fail_log_t: float = 0.0
        self._last_liveness_check_t: float = 0.0
        self._liveness_fail_streak: int = 0
        self._serial_lock = threading.RLock()
        self._bg_lock = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._probe_thread: Optional[threading.Thread] = None
        self._shutdown_requested: bool = False

        if self.is_simulation_mode:
            print("✅ Cam4ArmController initialized in Simulation Mode.")
        else:
            print(f"✅ Cam4ArmController initialized (port={self.port}, baud={self.baud})")
            print(f"   Limits: X={self.x_limits}, Y={self.y_limits} (effective clip: X={self._effective_x_limits}, Y={self._effective_y_limits})")
            print(f"   Sync every {self.sync_every_n_moves} move_relative; G0 throttle {self._min_move_interval_sec*1000:.0f} ms")

        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        try:
            if _global_cam4_arm_controller is self and self.ser and self.ser.is_open:
                self.disconnect()
        except Exception:
            pass

    def _open_serial(self) -> bool:
        if not os.path.exists(self.port):
            if self.ser is not None:
                self._mark_serial_lost("device path missing")
            return False
        if self.ser is not None and self.ser.is_open:
            return True
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(0.2)
            self.ser.write(b"\r\n\r\n")
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            return True
        except (serial.SerialException, OSError) as e:
            print(f"⚠️ Cam4ArmController: cannot open {self.port}: {e}")
            self._mark_serial_lost("open failed")
            return False

    def request_shutdown(self) -> None:
        """Signal background workers to stop; wait briefly."""
        self._shutdown_requested = True
        t_rec = self._reconnect_thread
        t_probe = self._probe_thread
        if t_rec is not None and t_rec.is_alive():
            t_rec.join(timeout=3.0)
        if t_probe is not None and t_probe.is_alive():
            t_probe.join(timeout=5.0)

    def _send_gcode(self, cmd: str, wait_ok: bool = False) -> Optional[str]:
        if self.is_simulation_mode:
            return None
        with self._serial_lock:
            return self._send_gcode_unlocked(cmd, wait_ok)

    def _send_gcode_unlocked(self, cmd: str, wait_ok: bool = False) -> Optional[str]:
        if not self._open_serial():
            return None
        if getattr(config, "CAM4_ARM_VERBOSE_GCODE", False):
            print(f"    [GRBL] >>> {cmd}")
        try:
            self.ser.write((cmd + "\n").encode("ascii", errors="ignore"))
        except (serial.SerialException, OSError):
            self._mark_serial_lost("gcode write failed")
            return None
        if not wait_ok:
            return None
        lines = []
        start = time.time()
        while time.time() - start < 3.0:
            try:
                line = self.ser.readline()
            except serial.SerialException:
                break
            if not line:
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue
            lines.append(decoded)
            if decoded.startswith("ok") or decoded.startswith("error") or decoded.startswith("ALARM"):
                break
        resp = "\n".join(lines)
        if getattr(config, "CAM4_ARM_VERBOSE_GCODE", False) and resp:
            print(f"    [GRBL] <<< {resp.strip()[:80]}")
        if resp and ("error" in resp.lower() or "alarm" in resp.lower()):
            self.is_healthy = False
        return resp

    def _wait_grbl_idle(self, timeout_sec: Optional[float] = None) -> bool:
        """รอ GRBL Idle หลังคำสั่งขยับ (blocking move / homing verify)."""
        if self.is_simulation_mode or not self.ser or not self.ser.is_open:
            return True
        if cam4_arm_homing is None:
            time.sleep(getattr(config, "CAM4_ARM_BLOCKING_MOVE_DELAY", 0.2))
            return True
        t = timeout_sec
        if t is None:
            t = float(getattr(config, "CAM4_ARM_HOMING_IDLE_TIMEOUT_SEC", 60.0))
        verbose = bool(getattr(config, "CAM4_ARM_VERBOSE_GCODE", False))
        wait_idle = bool(getattr(config, "CAM4_ARM_HOMING_WAIT_IDLE", True))
        if not wait_idle:
            time.sleep(getattr(config, "CAM4_ARM_BLOCKING_MOVE_DELAY", 0.2))
            return True
        return cam4_arm_homing.wait_grbl_idle(
            self.ser, timeout_sec=t, verbose=verbose
        )

    def _run_connect_selftest(self) -> bool:
        """ขยับทดสอบหลัง homing เพื่อยืนยันว่าแขนตอบสนองจริง."""
        if self.is_simulation_mode:
            return True
        if not getattr(config, "CAM4_ARM_RUN_SELFTEST_ON_CONNECT", False):
            return True
        if cam4_arm_homing is None or not self.ser or not self.ser.is_open:
            return False
        move_mm = float(getattr(config, "CAM4_ARM_SELFTEST_MOVE_MM", 2.0))
        feed = float(getattr(config, "CAM4_ARM_HOME_FEED_RATE", self.default_feed_rate))
        idle_t = float(getattr(config, "CAM4_ARM_HOMING_IDLE_TIMEOUT_SEC", 60.0))
        wait_idle = bool(getattr(config, "CAM4_ARM_HOMING_WAIT_IDLE", True))
        verbose = bool(getattr(config, "CAM4_ARM_VERBOSE_GCODE", False))
        print(f"🔧 Cam4ArmController: self-test (±{move_mm:.1f} mm)...")
        ok = cam4_arm_homing.run_connect_selftest(
            self.ser,
            move_mm=move_mm,
            feed_rate=feed,
            verbose=verbose,
            idle_timeout=idle_t,
            wait_idle=wait_idle,
        )
        if not ok:
            print("❌ Cam4ArmController: self-test failed — arm did not complete test moves.")
        else:
            print("✅ Cam4ArmController: self-test passed.")
        return ok

    def arm_is_stall(self) -> bool:
        """Serial open + GRBL healthy but motion verify failed (ARM:STALL)."""
        return (
            not self.is_simulation_mode
            and self._serial_is_open()
            and self.is_healthy
            and not self.motion_verified
        )

    def _reset_stall_timing(self) -> None:
        """Reset STALL timers / GRBL query counters — keep probe warn streak for HUD."""
        self._stall_since_t = 0.0
        self._grbl_status_fail_streak = 0
        self._last_stall_query_t = 0.0
        self._clear_drive_verify()

    def _reset_stall_tracking(self) -> None:
        """Full reset including probe warn streak (after successful connect/probe)."""
        self._reset_stall_timing()
        self._idle_probe_fail_streak = 0
        self._grbl_read_fail_streak = 0

    def _probe_warn_active(self) -> bool:
        return self._idle_probe_fail_streak > 0 or self._grbl_read_fail_streak > 0

    def _clear_probe_warn_streaks(self) -> None:
        """Operator/AUTO drive confirmed — drop ARM:WARN back to OK."""
        with self._bg_lock:
            self._grbl_read_fail_streak = 0
            self._idle_probe_fail_streak = 0
        self.motion_verified = True

    def note_joystick_operator_active(self, active: bool) -> None:
        """True while joystick axes outside deadzone — pauses idle probe."""
        was = self._operator_joystick_active
        self._operator_joystick_active = bool(active)
        if active:
            self.last_arm_activity_t = time.monotonic()
            if not was:
                self._joy_session_pan0 = float(self.pos_x)
                self._joy_session_tilt0 = float(self.pos_y)
                self._joy_session_cmd = 0.0
                self._joy_session_verify_due_t = 0.0
        elif was:
            delay = float(getattr(config, "CAM4_ARM_DRIVE_VERIFY_DELAY_SEC", 0.45))
            self._joy_session_verify_due_t = time.monotonic() + delay

    def set_camera_operator_lock(self, locked: bool) -> None:
        """True while CAM:DISC/WAIT/FAIL — pauses probe and optional auto-reconnect."""
        was_locked = self._camera_operator_lock
        self._camera_operator_lock = bool(locked)
        if was_locked and not locked:
            grace = float(getattr(config, "CAM4_ARM_POST_CAMERA_OK_PROBE_GRACE_SEC", 30.0))
            self._camera_probe_grace_until_t = time.monotonic() + max(0.0, grace)

    def _probe_paused(self) -> bool:
        if getattr(config, "CAM4_ARM_PAUSE_PROBE_ON_CAMERA_LOSS", True):
            if self._camera_operator_lock:
                return True
            if time.monotonic() < self._camera_probe_grace_until_t:
                return True
        if getattr(config, "CAM4_ARM_PAUSE_PROBE_ON_JOYSTICK", True):
            if self._operator_joystick_active:
                return True
        return False

    def _clear_drive_verify(self) -> None:
        self._drive_verify_pan0 = None
        self._drive_verify_tilt0 = None
        self._drive_verify_cmd_deg = 0.0
        self._drive_verify_due_t = 0.0

    def _trigger_stall_reconnect(self, reason: str) -> None:
        """motion_verified=False + background homing (joy/AUTO/probe fail 2)."""
        if self.is_simulation_mode or self._shutdown_requested:
            return
        if self._reconnect_in_progress:
            self.motion_verified = False
            return
        self.motion_verified = False
        self._clear_drive_verify()
        print(f"[ARM] motion stall ({reason}) → homing reconnect...")
        self.request_reconnect(f"stall-{reason}")

    def _mark_serial_lost(self, reason: str = "serial lost") -> None:
        """USB/serial gone — force ARM:DISC (ser.is_open can stay True until I/O)."""
        if self.is_simulation_mode:
            return
        self.is_healthy = False
        self.motion_verified = False
        self._liveness_fail_streak = 0
        self._close_serial_only()
        print(f"[ARM] serial lost ({reason})")

    def _note_grbl_read_ok(self) -> None:
        self._last_grbl_read_ok_t = time.monotonic()
        with self._bg_lock:
            self._grbl_read_fail_streak = 0

    def _note_drive_command(self, dx_deg: float, dy_deg: float) -> None:
        if self.is_simulation_mode or self._probe_in_progress or self._reconnect_in_progress:
            return
        cmd_mag = float((dx_deg ** 2 + dy_deg ** 2) ** 0.5)
        if cmd_mag < 1e-5:
            return
        # G-code ส่งจริง = serial ยังใช้ได้; เคลียร์ motion WARN (ไม่ bypass liveness/DISC)
        if self._probe_warn_active() and self._idle_probe_fail_streak > 0:
            with self._bg_lock:
                self._idle_probe_fail_streak = 0
            self.motion_verified = True
        if self._operator_joystick_active:
            self._joy_session_cmd += cmd_mag
        if not getattr(config, "CAM4_ARM_DRIVE_VERIFY_ENABLED", True):
            return
        if self._drive_verify_pan0 is None:
            self._drive_verify_pan0 = float(self.pos_x)
            self._drive_verify_tilt0 = float(self.pos_y)
            self._drive_verify_cmd_deg = 0.0
        self._drive_verify_cmd_deg += cmd_mag
        delay = float(getattr(config, "CAM4_ARM_DRIVE_VERIFY_DELAY_SEC", 0.45))
        self._drive_verify_due_t = time.monotonic() + delay

    def _maybe_verify_joy_session(self, now: float) -> None:
        """After joystick release: successful GRBL read clears ARM:WARN (no penalty on fail)."""
        if self._joy_session_verify_due_t <= 0.0:
            return
        if now < self._joy_session_verify_due_t:
            return
        if self._joy_session_pan0 is None:
            self._joy_session_verify_due_t = 0.0
            return

        cmd_mag = float(self._joy_session_cmd)
        self._joy_session_verify_due_t = 0.0
        self._joy_session_pan0 = None
        self._joy_session_tilt0 = None
        self._joy_session_cmd = 0.0

        min_cmd = float(getattr(config, "CAM4_ARM_DRIVE_VERIFY_MIN_CMD_DEG", 0.02))
        if cmd_mag + 1e-6 < min_cmd:
            return

        if self._sync_position_from_grbl_retries():
            if self._probe_warn_active():
                print(f"[ARM] operator joy OK — cleared WARN (cmd={cmd_mag:.2f}°)")
            self._clear_probe_warn_streaks()

    def maybe_verify_drive_motion(self, now: Optional[float] = None) -> None:
        """Verify joy/AUTO/socket drive; clear WARN on success, never penalize during recovery."""
        if self.is_simulation_mode or self._probe_in_progress or self._reconnect_in_progress:
            return
        if now is None:
            now = time.monotonic()
        self._maybe_verify_joy_session(now)
        if not getattr(config, "CAM4_ARM_DRIVE_VERIFY_ENABLED", True):
            return
        if self._drive_verify_pan0 is None or self._drive_verify_due_t <= 0.0:
            return
        if now < self._drive_verify_due_t:
            return

        pan0 = float(self._drive_verify_pan0)
        tilt0 = float(self._drive_verify_tilt0)
        cmd_mag = float(self._drive_verify_cmd_deg)
        recovering = self._probe_warn_active()
        self._clear_drive_verify()

        min_cmd = float(getattr(config, "CAM4_ARM_DRIVE_VERIFY_MIN_CMD_DEG", 0.02))
        if cmd_mag + 1e-6 < min_cmd:
            return

        if not self._sync_position_from_grbl_retries():
            return

        dx = self.pos_x - pan0
        dy = self.pos_y - tilt0
        moved = float((dx ** 2 + dy ** 2) ** 0.5)
        min_moved = float(getattr(config, "CAM4_ARM_DRIVE_VERIFY_MIN_MOVED_DEG", 0.04))
        min_moved = max(min_moved, cmd_mag * 0.12)
        if moved + 1e-6 >= min_moved:
            if recovering:
                print(f"[ARM] drive verify OK — cleared WARN (cmd={cmd_mag:.2f}°)")
            self._clear_probe_warn_streaks()
            return

        if recovering:
            return

        self._trigger_stall_reconnect(
            f"drive-no-motion cmd={cmd_mag:.3f}° moved={moved:.3f}°"
        )

    def _probe_min_moved_deg(self, delta: float) -> float:
        floor = float(getattr(config, "CAM4_ARM_IDLE_PROBE_MIN_MOVED_DEG", 0.15))
        return max(floor, abs(float(delta)) * 0.12)

    def _sync_position_from_grbl_retries(self) -> bool:
        retries = int(getattr(config, "CAM4_ARM_IDLE_PROBE_GRBL_RETRIES", 3))
        delay = float(getattr(config, "CAM4_ARM_IDLE_PROBE_GRBL_RETRY_DELAY_SEC", 0.1))
        timeout = float(getattr(config, "CAM4_ARM_IDLE_PROBE_GRBL_TIMEOUT_SEC", 2.5))
        for attempt in range(max(1, retries)):
            if self.sync_position_from_grbl(status_timeout_sec=timeout):
                return True
            if attempt + 1 < retries:
                time.sleep(delay)
        return False

    def _probe_measure_motion(
        self,
        pan0: float,
        tilt0: float,
        dpan: float,
        dtilt: float,
        min_moved: float,
    ) -> Tuple[Optional[bool], float, float]:
        """
        หลังสั่งขยับ + รอ Idle แล้ว — อ่าน WPos หลายรอบ (รอบแรกอาจเร็วไป).
        Returns (moved_ok, pan_delta, tilt_delta). moved_ok is None = อ่าน GRBL ไม่ได้.
        """
        attempts = int(getattr(config, "CAM4_ARM_IDLE_PROBE_VERIFY_ATTEMPTS", 2))
        retry_delay = float(getattr(config, "CAM4_ARM_IDLE_PROBE_VERIFY_RETRY_DELAY_SEC", 0.2))
        post_idle = float(getattr(config, "CAM4_ARM_IDLE_PROBE_POST_IDLE_DELAY_SEC", 0.15))
        if post_idle > 0:
            time.sleep(post_idle)

        last_dx, last_dy = 0.0, 0.0
        read_any = False
        for attempt in range(max(1, attempts)):
            if not self._sync_position_from_grbl_retries():
                if attempt + 1 < attempts:
                    time.sleep(retry_delay)
                    continue
                if not read_any:
                    return None, last_dx, last_dy
                break

            read_any = True
            last_dx = self.pos_x - pan0
            last_dy = self.pos_y - tilt0
            pan_ok = abs(dpan) < 1e-9 or abs(last_dx) >= min_moved
            tilt_ok = abs(dtilt) < 1e-9 or abs(last_dy) >= min_moved
            if pan_ok and tilt_ok:
                return True, last_dx, last_dy
            if attempt + 1 < attempts:
                time.sleep(retry_delay)

        return False, last_dx, last_dy

    @staticmethod
    def _is_grbl_read_reason(reason: str) -> bool:
        return "grbl-read" in str(reason)

    def _probe_record_grbl_read_fail(self, reason: str) -> bool:
        """
        GRBL ?/WPos read timeout — log only, never homing, never HUD WARN.
        Skip when operator recently drove the arm (G-code works; ? may be flaky).
        """
        now = time.monotonic()
        if self._operator_joystick_active or (now - self.last_arm_activity_t) < 30.0:
            return True
        if (
            self._grbl_read_fail_streak > 0
            and (now - self._last_grbl_read_fail_log_t) < 60.0
        ):
            return True
        with self._bg_lock:
            self._grbl_read_fail_streak += 1
            streak = self._grbl_read_fail_streak
        self._last_grbl_read_fail_log_t = now
        print(f"[ARM] GRBL read fail {streak} ({reason}) — log only, no homing/HUD")
        return True

    def _probe_record_fail(
        self,
        reason: str,
        *,
        dpan: float = 0.0,
        dtilt: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> bool:
        """
        Motion fail: fail 1 → ARM:WARN, fail 2 → STALL + homing.
        GRBL read reasons are routed to _probe_record_grbl_read_fail (no homing).
        Returns False when homing reconnect was triggered.
        """
        if self._is_grbl_read_reason(reason):
            return self._probe_record_grbl_read_fail(reason)

        need = int(getattr(config, "CAM4_ARM_IDLE_PROBE_FAILS_BEFORE_HOMING", 2))
        need = max(1, need)
        with self._bg_lock:
            self._idle_probe_fail_streak += 1
            streak = self._idle_probe_fail_streak
        detail = ""
        if abs(dpan) > 1e-9 or abs(dtilt) > 1e-9 or abs(dx) > 1e-9 or abs(dy) > 1e-9:
            detail = (
                f" cmd dpan={dpan:+.3f} dtilt={dtilt:+.3f}"
                f" measured d=({dx:+.3f},{dy:+.3f})"
            )
        print(f"[ARM] arm check fail {streak}/{need} ({reason}){detail}")
        if streak >= need:
            self._trigger_stall_reconnect(f"probe-{reason}")
            return False
        return True

    def _probe_record_motion_fail(self, reason: str, dpan: float, dtilt: float, dx: float, dy: float) -> bool:
        return self._probe_record_fail(reason, dpan=dpan, dtilt=dtilt, dx=dx, dy=dy)

    def _finalize_connect(self) -> bool:
        """After homing: optional self-test, then mark ARM:OK."""
        if not self._run_connect_selftest():
            if getattr(config, "CAM4_ARM_ACCEPT_HOMING_WITHOUT_SELFTEST", True):
                print(
                    "⚠️ Cam4ArmController: self-test failed — "
                    "accepting homing-only connect (ARM:OK)."
                )
            else:
                self.is_healthy = False
                self.motion_verified = False
                self._send_gcode("$X", wait_ok=True)
                return False
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self._mark_connect_ok()
        return True

    def _query_grbl_status_unlocked(self, timeout_sec: float = 0.6) -> Optional[str]:
        """Send ? and read status line — caller must hold _serial_lock."""
        if self.is_simulation_mode or not self._open_serial():
            return None
        try:
            # ล้าง ok/ค้างจาก G0 ก่อน — สาเหตุหลัก idle-probe-grbl-read-before-move
            self.ser.reset_input_buffer()
            self.ser.write(b"?")
        except (serial.SerialException, OSError):
            self._mark_serial_lost("status query write failed")
            return None
        time.sleep(0.03)
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                line = self.ser.readline()
            except (serial.SerialException, OSError):
                self._mark_serial_lost("status query read failed")
                break
            if not line:
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded.startswith("<") and "|" in decoded:
                return decoded
        return None

    def _query_grbl_status(self, timeout_sec: float = 0.6) -> Optional[str]:
        """Thread-safe GRBL status read."""
        if self.is_simulation_mode:
            return None
        with self._serial_lock:
            return self._query_grbl_status_unlocked(timeout_sec=timeout_sec)

    def sync_position_from_grbl(self, status_timeout_sec: float = 0.6) -> bool:
        """
        อ่านตำแหน่งจริงจาก GRBL (WPos) แล้วอัปเดต pos_x, pos_y (clip อยู่ใน effective limit).
        ลดปัญหา drift เมื่อคุมแขนนานแล้วค่าบวกลบพลาดจนเกิน limit.
        คืน True ถ้าซิงก์ได้ ไม่ใช่ simulation และ parse_grbl_status มีให้ใช้
        """
        if self.is_simulation_mode:
            return False
        if parse_grbl_status is None:
            return False
        with self._serial_lock:
            raw = self._query_grbl_status_unlocked(timeout_sec=status_timeout_sec)
        if not raw:
            return False
        info = parse_grbl_status(raw)
        wpos_x = info.get("wpos_x")
        wpos_y = info.get("wpos_y")
        if wpos_x is None or wpos_y is None:
            return False
        # แปลง WPos (mm) → pos หน่วยเดียวกับ limit: pos = ref_deg + (wpos - ref_mm) / mm_per_deg
        pos_x = self.ref_pan_deg + (float(wpos_x) - self.ref_pan_mm) / self.mm_per_deg_pan
        pos_y = self.ref_tilt_deg + (float(wpos_y) - self.ref_tilt_mm) / self.mm_per_deg_tilt
        x_clipped = float(np.clip(pos_x, self._effective_x_limits[0], self._effective_x_limits[1]))
        y_clipped = float(np.clip(pos_y, self._effective_y_limits[0], self._effective_y_limits[1]))
        self.pos_x = x_clipped
        self.pos_y = y_clipped
        self.target_x = x_clipped
        self.target_y = y_clipped
        self._note_grbl_read_ok()
        if self._operator_joystick_active:
            self._clear_probe_warn_streaks()
        return True

    def touch_activity(self) -> None:
        """Mark recent arm command (suppresses idle probe until quiet period elapses)."""
        self.last_arm_activity_t = time.monotonic()

    @staticmethod
    def _probe_nudge_axis(pos: float, lim_lo: float, lim_hi: float, delta: float) -> float:
        """Pick +delta or -delta nudge that stays inside effective limits, or 0 if no room."""
        if pos + delta <= lim_hi - 1e-6:
            return float(delta)
        if pos - delta >= lim_lo + 1e-6:
            return float(-delta)
        return 0.0

    def run_idle_motion_probe(self) -> bool:
        """
        สั่งขยับ ~1° (config), รอ GRBL Idle, อ่าน WPos 2 รอบเพื่อยืนยันว่าขยับจริง แล้วกลับตำแหน่งเดิม.
        GRBL read fail → WARN only (no homing). Motion fail → WARN then STALL+homing.
        """
        self.last_motion_probe_t = time.monotonic()
        if self.is_simulation_mode:
            self.motion_verified = True
            return True
        if not getattr(config, "CAM4_ARM_IDLE_PROBE_ENABLED", True):
            return True
        if not self.ser or not self.ser.is_open:
            self.motion_verified = False
            return False

        delta = float(getattr(config, "CAM4_ARM_IDLE_PROBE_DELTA_DEG", 1.0))
        min_moved = self._probe_min_moved_deg(delta)

        if not self._sync_position_from_grbl_retries():
            return self._probe_record_grbl_read_fail("idle-probe-grbl-read-before-move")
        self._note_grbl_read_ok()

        pan0, tilt0 = self.pos_x, self.pos_y
        dpan = self._probe_nudge_axis(
            pan0, self._effective_x_limits[0], self._effective_x_limits[1], delta
        )
        dtilt = self._probe_nudge_axis(
            tilt0, self._effective_y_limits[0], self._effective_y_limits[1], delta
        )

        if abs(dpan) < 1e-9 and abs(dtilt) < 1e-9:
            # อยู่ limit — อย่าล้าง streak ที่สะสมไว้ (กัน fail 1/2 วนซ้ำ)
            if self._idle_probe_fail_streak > 0:
                return True
            self.motion_verified = True
            return True

        prev_feed = self.current_feed_rate
        self.current_feed_rate = float(
            getattr(config, "CAM4_ARM_IDLE_PROBE_FEED_RATE", self.default_feed_rate)
        )
        ok = True
        try:
            self.move_absolute(pan0 + dpan, tilt0 + dtilt, blocking=True)

            moved_ok, dx, dy = self._probe_measure_motion(
                pan0, tilt0, dpan, dtilt, min_moved
            )
            self.move_absolute(pan0, tilt0, blocking=True)

            if moved_ok is None:
                ok = self._probe_record_grbl_read_fail("idle-probe-grbl-read-after-move")
            elif moved_ok:
                if self._idle_probe_fail_streak == 0:
                    self.motion_verified = True
                self._idle_probe_fail_streak = 0
                self._grbl_read_fail_streak = 0
                ok = True
            else:
                ok = self._probe_record_motion_fail("insufficient motion", dpan, dtilt, dx, dy)
        finally:
            self.current_feed_rate = prev_feed
            self.last_motion_probe_t = time.monotonic()
        return ok

    def maybe_check_arm_liveness(self) -> None:
        """Periodic GRBL ? + device path check — detect unplug while HUD still shows OK."""
        if self.is_simulation_mode or self._shutdown_requested:
            return
        if not getattr(config, "CAM4_ARM_LIVENESS_CHECK_ENABLED", True):
            return
        if self._reconnect_in_progress or self._probe_in_progress:
            return
        if self._operator_joystick_active:
            return
        if not self._serial_is_open():
            return

        now = time.monotonic()
        interval = float(getattr(config, "CAM4_ARM_LIVENESS_INTERVAL_SEC", 8.0))
        if now - self._last_liveness_check_t < interval:
            return
        self._last_liveness_check_t = now

        if self._query_grbl_status(
            timeout_sec=float(getattr(config, "CAM4_ARM_LIVENESS_GRBL_TIMEOUT_SEC", 1.5))
        ) is None:
            self._liveness_fail_streak += 1
            need = int(getattr(config, "CAM4_ARM_LIVENESS_FAILS_BEFORE_DISC", 2))
            need = max(1, need)
            print(
                f"[ARM] liveness fail {self._liveness_fail_streak}/{need} "
                "(GRBL ? no response — unplugged?)"
            )
            if self._liveness_fail_streak >= need:
                self._mark_serial_lost("liveness timeout")
        else:
            self._liveness_fail_streak = 0
            self._note_grbl_read_ok()

    def maybe_run_idle_probe(self) -> None:
        """Run idle motion probe when interval elapsed and no recent arm commands."""
        if self.is_simulation_mode:
            return
        if not getattr(config, "CAM4_ARM_IDLE_PROBE_ENABLED", True):
            return
        if self._probe_paused():
            return
        if self._probe_in_progress or self._reconnect_in_progress:
            return
        if not self._serial_is_open() or not self.is_healthy:
            return
        now = time.monotonic()
        grace = float(getattr(config, "CAM4_ARM_POST_CONNECT_PROBE_GRACE_SEC", 90.0))
        if self._last_connect_ok_t > 0 and (now - self._last_connect_ok_t) < grace:
            return
        interval = float(getattr(config, "CAM4_ARM_IDLE_PROBE_INTERVAL_SEC", 25.0))
        if self._idle_probe_fail_streak > 0:
            interval = float(
                getattr(config, "CAM4_ARM_PROBE_FAIL_RETRY_INTERVAL_SEC", 5.0)
            )
        elif self._grbl_read_fail_streak > 0:
            interval = float(
                getattr(config, "CAM4_ARM_GRBL_READ_PROBE_INTERVAL_SEC", 120.0)
            )
        quiet = float(getattr(config, "CAM4_ARM_IDLE_PROBE_MIN_QUIET_SEC", 4.0))
        if now - self.last_motion_probe_t < interval:
            return
        if now - self.last_arm_activity_t < quiet:
            return
        if getattr(config, "CAM4_ARM_IDLE_PROBE_IN_BACKGROUND", True):
            self.request_idle_probe()
        else:
            self._probe_in_progress = True
            try:
                with self._serial_lock:
                    self.run_idle_motion_probe()
            finally:
                self._probe_in_progress = False

    def request_idle_probe(self) -> None:
        """Start idle motion probe in background (non-blocking UI)."""
        if self.is_simulation_mode or self._shutdown_requested:
            return
        with self._bg_lock:
            if self._probe_in_progress or self._reconnect_in_progress:
                return
            t = self._probe_thread
            if t is not None and t.is_alive():
                return
            self._probe_in_progress = True
            self._probe_thread = threading.Thread(
                target=self._idle_probe_worker,
                daemon=True,
                name="Cam4ArmIdleProbe",
            )
            self._probe_thread.start()

    def _idle_probe_worker(self) -> None:
        ok = False
        try:
            with self._serial_lock:
                ok = self.run_idle_motion_probe()
        except Exception as exc:
            print(f"[ARM] idle probe worker exception: {exc}")
        finally:
            self._probe_in_progress = False
        if (
            not ok
            and getattr(config, "CAM4_ARM_PROBE_FAIL_TRIGGERS_HOMING", True)
            and not self._shutdown_requested
            and not self._reconnect_in_progress
        ):
            print("[ARM] idle probe failed → full homing reconnect (limit → home)...")
            self.request_reconnect("idle-probe-fail")

    def _arm_fault_label(self) -> Optional[str]:
        """Return DISC/ERR/STALL when arm is not fully OK, else None."""
        if self.is_simulation_mode or self.arm_link_ok():
            return None
        if not self._serial_is_open():
            return "DISC"
        if not self.is_healthy:
            return "ERR"
        if not self.motion_verified:
            return "STALL"
        return "FAULT"

    def maybe_handle_arm_fault_recovery(self, *, camera_operator_lock: bool = False) -> None:
        """
        ทุก fault → full homing reconnect เดียวกับเปิดโปรแกรม ($X→$HY→$HX→G53→G92).
        22_gun บังคับ MODE:SAFE ระหว่าง homing; กลับโหมดเดิมเมื่อ ARM:OK.
        """
        if self.is_simulation_mode:
            return
        if camera_operator_lock and getattr(
            config, "CAM4_ARM_PAUSE_ARM_RECONNECT_ON_CAMERA_LOSS", True
        ):
            return
        if not getattr(config, "CAM4_ARM_FAULT_AUTO_HOMING_ENABLED", True):
            self._maybe_handle_stall_recovery_legacy()
            return
        if self._reconnect_in_progress or self._probe_in_progress:
            return
        if self.arm_link_ok():
            if self._idle_probe_fail_streak == 0 and self._grbl_read_fail_streak == 0:
                self._reset_stall_tracking()
            else:
                self._reset_stall_timing()
            return
        if not getattr(config, "CAM4_ARM_AUTO_RECONNECT_ENABLED", True):
            return

        max_fail = int(getattr(config, "CAM4_ARM_RECONNECT_MAX_ATTEMPTS", 5))
        if self._reconnect_fail_count >= max_fail:
            return

        fault = self._arm_fault_label()
        if fault is None:
            return

        now = time.monotonic()
        cooldown = float(getattr(config, "CAM4_ARM_FAULT_RECOVERY_COOLDOWN_SEC", 2.0))
        if now - self._last_fault_recovery_t < cooldown:
            return

        immediate = bool(getattr(config, "CAM4_ARM_IMMEDIATE_FAULT_HOMING", True))
        if not immediate and fault in ("STALL", "ERR"):
            self._maybe_handle_stall_recovery_legacy()
            return

        if fault == "DISC":
            delay = float(getattr(config, "CAM4_ARM_RECONNECT_DELAY", 3.0))
            if now - self._last_reconnect_attempt_t < delay:
                return

        self._last_fault_recovery_t = now
        self._reset_stall_timing()
        print(
            f"[ARM] fault ({fault}): full homing reconnect (limit switches → home)..."
        )
        self.request_reconnect(f"fault-{fault.lower()}")

    def maybe_handle_stall_recovery(self, *, camera_operator_lock: bool = False) -> None:
        """Backward-compatible alias → unified fault recovery."""
        self.maybe_handle_arm_fault_recovery(camera_operator_lock=camera_operator_lock)

    def _maybe_handle_stall_recovery_legacy(self) -> None:
        """
        STALL ค้าง (legacy) → ปิด serial → auto-reconnect homing.
        ใช้เมื่อ CAM4_ARM_IMMEDIATE_FAULT_HOMING=False.
        """
        if self.is_simulation_mode:
            return
        if not getattr(config, "CAM4_ARM_STALL_AUTO_RECONNECT_ENABLED", True):
            return
        if self._reconnect_in_progress or self._probe_in_progress:
            return
        if self.arm_link_ok():
            if self._idle_probe_fail_streak == 0 and self._grbl_read_fail_streak == 0:
                self._reset_stall_tracking()
            else:
                self._reset_stall_timing()
            return
        if not self.arm_is_stall():
            return

        now = time.monotonic()
        if self._stall_since_t <= 0.0:
            self._stall_since_t = now

        query_interval = 2.0
        if now - self._last_stall_query_t >= query_interval:
            self._last_stall_query_t = now
            if self._query_grbl_status() is None:
                self._grbl_status_fail_streak += 1
            else:
                self._grbl_status_fail_streak = 0

        stall_duration = now - self._stall_since_t
        stall_sec = float(getattr(config, "CAM4_ARM_STALL_AUTO_RECONNECT_SEC", 30.0))
        fail_need = int(getattr(config, "CAM4_ARM_STALL_GRBL_FAIL_COUNT", 3))
        force_sec = float(getattr(config, "CAM4_ARM_STALL_FORCE_RECONNECT_SEC", 45.0))
        cooldown = float(getattr(config, "CAM4_ARM_STALL_RECOVERY_COOLDOWN_SEC", 60.0))

        query_dead = stall_duration >= stall_sec and self._grbl_status_fail_streak >= fail_need
        probe_stuck = stall_duration >= force_sec
        if not (query_dead or probe_stuck):
            return
        if now - self._last_stall_recovery_t < cooldown:
            return

        reason = "GRBL not responding" if query_dead else "STALL timeout (homing reset)"
        print(
            f"[ARM] STALL recovery ({reason}, {stall_duration:.0f}s): "
            "closing serial → auto-reconnect homing..."
        )
        self._last_stall_recovery_t = now
        self._reset_stall_tracking()
        self._reconnect_fail_count = 0
        self._close_serial_only()
        self.motion_verified = False

    def _serial_is_open(self) -> bool:
        ser = self.ser
        if ser is None or not getattr(ser, "is_open", False):
            return False
        if not os.path.exists(self.port):
            self._mark_serial_lost("device path missing")
            return False
        return True

    def _close_serial_only(self) -> None:
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def arm_link_ok(self) -> bool:
        """True when serial open, healthy, and last motion verify passed."""
        return (
            not self.is_simulation_mode
            and self._serial_is_open()
            and self.is_healthy
            and self.motion_verified
        )

    def arm_requires_safe_mode(self) -> bool:
        """True when operator must not drive arm (DISC/ERR/BUSY/STALL/reconnecting/probing)."""
        if self.is_simulation_mode:
            return False
        if self._reconnect_in_progress or self._probe_in_progress:
            return True
        if not self._serial_is_open():
            return True
        if not self.is_healthy:
            return True
        if not self.motion_verified:
            return True  # ARM:STALL — serial OK but motion verify failed
        return False

    def _reconnect_body(self) -> bool:
        """Close serial and full connect/homing — caller should hold _serial_lock."""
        self._close_serial_only()
        self.is_healthy = False
        self.motion_verified = False
        return self._connect_impl()

    def request_reconnect(self, reason: str = "auto") -> bool:
        """Non-blocking: spawn background reconnect (homing + self-test)."""
        if self.is_simulation_mode:
            return True
        if self._shutdown_requested:
            return False
        if not getattr(config, "CAM4_ARM_RECONNECT_IN_BACKGROUND", True):
            return self.try_reconnect()
        with self._bg_lock:
            if self._reconnect_in_progress:
                return False
            t = self._reconnect_thread
            if t is not None and t.is_alive():
                return False
            self._reconnect_in_progress = True
            self._last_reconnect_attempt_t = time.monotonic()
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_worker,
                args=(reason,),
                daemon=True,
                name="Cam4ArmReconnect",
            )
            self._reconnect_thread.start()
        return True

    def _reconnect_worker(self, reason: str) -> None:
        try:
            print(
                f"[ARM] reconnect ({reason}): background homing + home + self-test..."
            )
            with self._serial_lock:
                ok = self._reconnect_body()
            if ok:
                self._reconnect_fail_count = 0
                print("[ARM] reconnect OK — arm homed (mode restores if previously saved)")
            else:
                self._reconnect_fail_count += 1
                print(f"[ARM] reconnect failed ({self._reconnect_fail_count})")
        except Exception as exc:
            self._reconnect_fail_count += 1
            self.is_healthy = False
            self.motion_verified = False
            print(f"[ARM] reconnect exception: {exc}")
        finally:
            self._reconnect_in_progress = False

    def try_reconnect(self) -> bool:
        """
        Full reconnect after power loss / serial drop: close port, homing, home, self-test.
        Blocking — used at startup or when CAM4_ARM_RECONNECT_IN_BACKGROUND=False.
        """
        if self.is_simulation_mode:
            return True
        self._reconnect_in_progress = True
        self._last_reconnect_attempt_t = time.monotonic()
        print("[ARM] reconnect: closing serial → homing + home + self-test...")
        try:
            with self._serial_lock:
                ok = self._reconnect_body()
            if ok:
                self._reconnect_fail_count = 0
                print("[ARM] reconnect OK — arm homed (mode restores if previously saved)")
            else:
                self._reconnect_fail_count += 1
                print(f"[ARM] reconnect failed ({self._reconnect_fail_count})")
            return ok
        except Exception as exc:
            self._reconnect_fail_count += 1
            self.is_healthy = False
            self.motion_verified = False
            print(f"[ARM] reconnect exception: {exc}")
            return False
        finally:
            self._reconnect_in_progress = False

    def maybe_auto_reconnect(self) -> None:
        """Alias → unified fault recovery (DISC/ERR/STALL)."""
        self.maybe_handle_arm_fault_recovery()

    def fire(self) -> None:
        """ส่ง G-code ยิง: M3 S255 → sleep(duration) → M5 (อ้างอิง v34). โหมด simulation แค่ log."""
        duration = getattr(config, "CAM4_ARM_FIRE_DURATION", 0.1)
        if self.is_simulation_mode:
            print(f"[SIMULATE] FIRE command (Duration: {duration:.2f}s)")
            return
        if not self._probe_in_progress:
            self.touch_activity()
        self._send_gcode("M3 S255")
        time.sleep(duration)
        self._send_gcode("M5")

    def connect(self) -> bool:
        with self._serial_lock:
            return self._connect_impl()

    def _connect_impl(self) -> bool:
        if self.is_simulation_mode:
            self.pos_x = self.ref_pan_deg
            self.pos_y = self.ref_tilt_deg
            self.target_x = self.pos_x
            self.target_y = self.pos_y
            print("✅ Cam4ArmController (simulation) connected.")
            return True

        if not self._open_serial():
            return False

        self._send_gcode("$X", wait_ok=True)
        # รอให้ GRBL พร้อมก่อน homing (ลดโอกาส homing ล้มเหลวครั้งแรก)
        delay_after_unlock = getattr(config, "CAM4_ARM_DELAY_AFTER_UNLOCK_SEC", 0.8)
        if delay_after_unlock > 0:
            time.sleep(delay_after_unlock)

        skip_homing = getattr(config, "CAM4_ARM_SKIP_HOMING_USE_G92_ONLY", False)
        if skip_homing:
            self._send_gcode("G92 X0 Y0", wait_ok=True)
            self._send_gcode("G90", wait_ok=True)
            if self.grbl_units_mm:
                self._send_gcode("G21", wait_ok=True)
            self.pos_x = 0.0
            self.pos_y = 0.0
            self.target_x = 0.0
            self.target_y = 0.0
            if not self._finalize_connect():
                return False
            print("✅ Cam4ArmController connected (skip homing, G92 only).")
            return True

        use_g53 = getattr(config, "CAM4_ARM_USE_G53_REF_AFTER_HOMING", True)
        ref_mpos_x = getattr(config, "CAM4_ARM_REF_MPOS_X", None)
        ref_mpos_y = getattr(config, "CAM4_ARM_REF_MPOS_Y", None)

        if use_g53 and ref_mpos_x is not None and ref_mpos_y is not None and cam4_arm_homing:
            print("🏠 Cam4ArmController: homing + connect ($X → $HY → $HX → G53 → G92 X0 Y0)...")
            max_homing_retries = getattr(config, "CAM4_ARM_HOMING_MAX_RETRIES", 2)
            homing_ok = False
            for attempt in range(max(1, max_homing_retries)):
                if cam4_arm_homing.run_homing_cycle(
                    self.ser,
                    ref_mpos_x=ref_mpos_x,
                    ref_mpos_y=ref_mpos_y,
                ):
                    homing_ok = True
                    break
                if attempt < max_homing_retries - 1:
                    print("🔄 Homing retry in 1s (unlock + wait)...")
                    self._send_gcode("$X", wait_ok=True)
                    time.sleep(1.0)
            if not homing_ok:
                print("❌ Homing failed after {} attempt(s).".format(max_homing_retries))
                self._send_gcode("$X", wait_ok=True)
                self._mark_homing_failed()
                return False
        else:
            if cam4_arm_homing:
                max_homing_retries = getattr(config, "CAM4_ARM_HOMING_MAX_RETRIES", 2)
                homing_ok = False
                for attempt in range(max(1, max_homing_retries)):
                    if cam4_arm_homing.run_homing_cycle(self.ser):
                        homing_ok = True
                        break
                    if attempt < max_homing_retries - 1:
                        print("🔄 Homing retry in 1s (unlock + wait)...")
                        self._send_gcode("$X", wait_ok=True)
                        time.sleep(1.0)
                if not homing_ok:
                    print("❌ Homing failed after {} attempt(s).".format(max_homing_retries))
                    self._send_gcode("$X", wait_ok=True)
                    self._mark_homing_failed()
                    return False
            else:
                self._send_gcode("$H", wait_ok=True)
                time.sleep(getattr(config, "CAM4_ARM_HOMING_COMPLETE_DELAY", 15.0))

        if self.grbl_units_mm:
            self._send_gcode("G21", wait_ok=True)
        self._send_gcode("G90", wait_ok=True)
        self.current_feed_rate = self.default_feed_rate

        if not self._finalize_connect():
            return False

        print("✅ Cam4ArmController connected and moved to reference pose.")
        return True

    def clear_homing_fault(self) -> None:
        """Clear homing-failure HUD state (manual reconnect / successful homing)."""
        self._homing_fault = None

    def _mark_homing_failed(self) -> None:
        """
        Homing cycle failed — GRBL alive but no motion → NOPWR; else HOMEFAIL.
        Keeps MODE:SAFE until operator fixes issue or manual reconnect succeeds.
        """
        self.is_healthy = False
        self.motion_verified = False
        if self._serial_is_open() and self._query_grbl_status(timeout_sec=1.0):
            self._homing_fault = "no_motion"
            print(
                "[ARM] homing failed — GRBL responds but arm did not home "
                "(check motor power, E-stop, stepper drivers)"
            )
        else:
            self._homing_fault = "homing_fail"
            print("[ARM] homing failed — check USB cable, limits, GRBL alarm")

    def _mark_connect_ok(self) -> None:
        """After successful connect/selftest: healthy and defer first idle probe."""
        self.is_healthy = True
        self.motion_verified = True
        now = time.monotonic()
        self._last_connect_ok_t = now
        self.last_arm_activity_t = now
        self.last_motion_probe_t = now
        self._reset_stall_tracking()
        self._reconnect_fail_count = 0
        self.clear_homing_fault()

    def disconnect(self) -> None:
        self.request_shutdown()
        if self.is_simulation_mode:
            return
        with self._serial_lock:
            if getattr(config, "CAM4_ARM_RETURN_TO_REF_ON_DISCONNECT", True) and self.is_healthy and self.ser and self.ser.is_open:
                self.go_home(blocking=True)
                time.sleep(getattr(config, "CAM4_ARM_RETURN_TO_REF_WAIT_SEC", 3.0))
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None

    def go_home(self, blocking: bool = True) -> None:
        prev_feed = self.current_feed_rate
        self.current_feed_rate = getattr(config, "CAM4_ARM_HOME_FEED_RATE", self.default_feed_rate)
        try:
            self.move_absolute(0.0, 0.0, blocking=blocking)
        finally:
            self.current_feed_rate = prev_feed

    def move_absolute(self, x_deg: float, y_deg: float, blocking: bool = False) -> bool:
        """ขยับไปตำแหน่ง absolute. คืน True ถ้าส่ง G-code จริง (หรือ simulation)."""
        x_clipped = float(np.clip(x_deg, self._effective_x_limits[0], self._effective_x_limits[1]))
        y_clipped = float(np.clip(y_deg, self._effective_y_limits[0], self._effective_y_limits[1]))
        self.pos_x = x_clipped
        self.pos_y = y_clipped
        self.target_x = x_clipped
        self.target_y = y_clipped

        if self.is_simulation_mode:
            if blocking:
                time.sleep(getattr(config, "CAM4_ARM_BLOCKING_MOVE_DELAY", 0.2))
            return True

        if not blocking:
            if not self._serial_lock.acquire(blocking=False):
                return False
        else:
            self._serial_lock.acquire()
        try:
            if not blocking and self._min_move_interval_sec > 0:
                now = time.monotonic()
                if (now - self._last_g0_send_time) < self._min_move_interval_sec:
                    return False
                self._last_g0_send_time = now

            if self.grbl_units_mm:
                x_mm = self.ref_pan_mm + (x_clipped - self.ref_pan_deg) * self.mm_per_deg_pan
                y_mm = self.ref_tilt_mm + (y_clipped - self.ref_tilt_deg) * self.mm_per_deg_tilt
                cmd = f"G0 X{x_mm:.3f} Y{y_mm:.3f} F{self.current_feed_rate:.1f}"
            else:
                cmd = f"G0 X{x_clipped:.3f} Y{y_clipped:.3f}"
            if not self._probe_in_progress:
                self.touch_activity()
            self._send_gcode_unlocked(cmd, wait_ok=blocking)
            if blocking:
                if not self._wait_grbl_idle():
                    if not self._probe_in_progress:
                        self.is_healthy = False
                else:
                    time.sleep(getattr(config, "CAM4_ARM_BLOCKING_MOVE_DELAY", 0.05))
            return True
        finally:
            self._serial_lock.release()

    def move_relative(self, dx_deg: float, dy_deg: float, blocking: bool = False) -> None:
        """ขยับแบบ relative. ทุก sync_every_n_moves ครั้งจะซิงก์ตำแหน่งจาก GRBL ก่อน แล้ว clip อยู่ใน effective limit."""
        if not self.is_simulation_mode and self.sync_every_n_moves > 0 and parse_grbl_status is not None:
            self._move_relative_count += 1
            if self._move_relative_count >= self.sync_every_n_moves:
                self._move_relative_count = 0
                self.sync_position_from_grbl()
        actual_dx = float(dx_deg)
        actual_dy = float(dy_deg)
        new_x = self.pos_x + actual_dx
        new_y = self.pos_y + actual_dy
        x_clipped = float(np.clip(new_x, self._effective_x_limits[0], self._effective_x_limits[1]))
        y_clipped = float(np.clip(new_y, self._effective_y_limits[0], self._effective_y_limits[1]))
        clipped_dx = x_clipped - self.pos_x
        clipped_dy = y_clipped - self.pos_y
        if self.move_absolute(x_clipped, y_clipped, blocking=blocking):
            self._note_drive_command(clipped_dx, clipped_dy)

    def __enter__(self) -> "Cam4ArmController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class SimCam4ArmController:
    """จำลองแขน — ไม่ส่ง G-code, เก็บ pos_x/pos_y กับ limit ไว้ให้ joystick/tracker ใช้."""

    def __init__(self) -> None:
        self.is_simulation_mode = True
        self.x_limits = getattr(config, "CAM4_ARM_X_LIMITS", (-65.0, 65.0))
        self.y_limits = getattr(config, "CAM4_ARM_Y_LIMITS", (-35.0, 35.0))
        margin = getattr(config, "CAM4_ARM_LIMIT_SAFETY_MARGIN", 2.0)
        self._effective_x_limits = (self.x_limits[0] + margin, self.x_limits[1] - margin)
        self._effective_y_limits = (self.y_limits[0] + margin, self.y_limits[1] - margin)
        self.ref_pan_deg = getattr(config, "CAM4_ARM_REF_PAN_DEG", 0.0)
        self.ref_tilt_deg = getattr(config, "CAM4_ARM_REF_TILT_DEG", 0.0)
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.is_healthy: bool = True
        self.motion_verified: bool = True
        self.last_arm_activity_t: float = 0.0
        self.last_motion_probe_t: float = 0.0

    def connect(self) -> bool:
        self.pos_x = self.ref_pan_deg
        self.pos_y = self.ref_tilt_deg
        self.target_x = self.pos_x
        self.target_y = self.pos_y
        return True

    def disconnect(self) -> None:
        pass

    def go_home(self, blocking: bool = True) -> None:
        self.move_absolute(self.ref_pan_deg, self.ref_tilt_deg, blocking=blocking)

    def move_absolute(self, x_deg: float, y_deg: float, blocking: bool = False) -> None:
        x_clipped = float(np.clip(x_deg, self._effective_x_limits[0], self._effective_x_limits[1]))
        y_clipped = float(np.clip(y_deg, self._effective_y_limits[0], self._effective_y_limits[1]))
        self.pos_x = x_clipped
        self.pos_y = y_clipped
        self.target_x = x_clipped
        self.target_y = y_clipped

    def move_relative(self, dx_deg: float, dy_deg: float, blocking: bool = False) -> None:
        new_x = self.pos_x + dx_deg
        new_y = self.pos_y + dy_deg
        x_clipped = float(np.clip(new_x, self._effective_x_limits[0], self._effective_x_limits[1]))
        y_clipped = float(np.clip(new_y, self._effective_y_limits[0], self._effective_y_limits[1]))
        self.pos_x = x_clipped
        self.pos_y = y_clipped
        self.target_x = x_clipped
        self.target_y = y_clipped

    def fire(self) -> None:
        """โหมด simulation: แค่ log ไม่ส่ง G-code."""
        duration = getattr(config, "CAM4_ARM_FIRE_DURATION", 0.1)
        print(f"[SIMULATE] FIRE command (Duration: {duration:.2f}s)")

    def touch_activity(self) -> None:
        self.last_arm_activity_t = time.monotonic()

    def note_joystick_operator_active(self, active: bool) -> None:
        pass

    def set_camera_operator_lock(self, locked: bool) -> None:
        pass

    def maybe_check_arm_liveness(self) -> None:
        pass

    def maybe_run_idle_probe(self) -> None:
        pass

    def maybe_verify_drive_motion(self, now: Optional[float] = None) -> None:
        pass

    def try_reconnect(self) -> bool:
        return True

    def request_reconnect(self, reason: str = "manual") -> bool:
        return True

    def request_shutdown(self) -> None:
        pass

    def request_idle_probe(self) -> None:
        pass

    def maybe_auto_reconnect(self) -> None:
        pass

    def maybe_handle_stall_recovery(self) -> None:
        pass

    def maybe_handle_arm_fault_recovery(self) -> None:
        pass

    def arm_is_stall(self) -> bool:
        return False

    def arm_link_ok(self) -> bool:
        return True

    def arm_requires_safe_mode(self) -> bool:
        return False

    def sync_position_from_grbl(self) -> bool:
        return False

    def __enter__(self) -> "SimCam4ArmController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

