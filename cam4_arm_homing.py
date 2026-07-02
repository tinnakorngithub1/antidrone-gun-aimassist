"""
โมดูลร่วม: หา home position จาก limit switch (GRBL)
------------------------------------------------------
- ลำดับที่ใช้: $X → $HY → $HX → G53 X Y → G92 X0 Y0 (ส่งทีละคำสั่ง รอ ok ก่อนส่งถัดไป)
- หลังคำสั่งขยับ: รอ GRBL Idle + (ถ้าเปิด) เช็ค MPos หลัง G53
- ใช้ได้ทั้งจาก controller (ส่ง ser เข้ามา) และสคริปต์ standalone (เปิด/ปิด serial เอง)
"""

import time
from typing import Any, Dict, Optional, Tuple

import serial

try:
    import config
except ImportError:
    config = None


def _drain_serial(ser: serial.Serial, max_time: float = 0.4, idle_time: float = 0.1) -> None:
    """อ่านและทิ้งข้อมูลที่ค้างใน serial จนไม่มีข้อมูลมา idle_time วินาที หรือครบ max_time."""
    if not ser or not ser.is_open:
        return
    start = time.time()
    last_data = time.time()
    try:
        while time.time() - start < max_time and (time.time() - last_data) < idle_time:
            if ser.in_waiting:
                ser.read(ser.in_waiting)
                last_data = time.time()
            else:
                time.sleep(0.02)
    except (serial.SerialException, OSError):
        pass


def _send_gcode_line(
    ser: serial.Serial,
    cmd: str,
    timeout: float = 3.0,
    verbose: bool = False,
) -> Optional[str]:
    """
    ส่งคำสั่ง G-code หนึ่งบรรทัดไปยัง GRBL รอจนเจอ 'ok' หรือ 'error' หรือ 'ALARM' แล้วคืน response string.
    """
    if not ser or not ser.is_open:
        return None
    if verbose:
        print(f"    [GRBL] >>> {cmd}")
    try:
        ser.write((cmd + "\n").encode("ascii", errors="ignore"))
    except (serial.SerialException, OSError):
        return None

    lines = []
    start = time.time()
    while time.time() - start < timeout:
        try:
            line = ser.readline()
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
    if verbose and resp:
        print(f"    [GRBL] <<< {resp.strip()[:80]}")
    return resp if resp else None


def _response_failed(resp: Optional[str]) -> bool:
    if not resp:
        return False
    low = resp.lower()
    return "error" in low or "alarm" in low


def parse_grbl_status(line: str) -> Dict[str, Any]:
    """แยกค่าจากข้อความตอบกลับ ? ของ GRBL. คืน dict: state, mpos, wpos, etc."""
    out: Dict[str, Any] = {
        "state": None,
        "mpos_x": None,
        "mpos_y": None,
        "mpos_z": None,
        "wpos_x": None,
        "wpos_y": None,
        "wpos_z": None,
        "raw": line,
    }
    if not line or not line.startswith("<") or "|" not in line:
        return out
    s = line[1:].strip()
    parts = s.split("|")
    for p in parts:
        p = p.strip()
        if ":" not in p:
            if out["state"] is None:
                out["state"] = p
            continue
        key, val = p.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "MPos":
            try:
                nums = [float(x.strip()) for x in val.split(",")[:3]]
                if len(nums) >= 3:
                    out["mpos_x"], out["mpos_y"], out["mpos_z"] = nums
            except (ValueError, IndexError):
                pass
        elif key == "WPos":
            try:
                nums = [float(x.strip()) for x in val.split(",")[:3]]
                if len(nums) >= 3:
                    out["wpos_x"], out["wpos_y"], out["wpos_z"] = nums
            except (ValueError, IndexError):
                pass
    if out["state"] is None and parts:
        out["state"] = parts[0].strip()
    return out


def query_grbl_status(ser: serial.Serial, timeout: float = 0.6) -> Optional[Dict[str, Any]]:
    """ส่ง ? แล้วคืน parsed status dict หรือ None."""
    if not ser or not ser.is_open:
        return None
    try:
        ser.write(b"?\n")
    except (serial.SerialException, OSError):
        return None
    start = time.time()
    while time.time() - start < timeout:
        try:
            line = ser.readline()
        except serial.SerialException:
            break
        if not line:
            continue
        decoded = line.decode("utf-8", errors="ignore").strip()
        if decoded.startswith("<") and "|" in decoded:
            return parse_grbl_status(decoded)
    return None


def _is_motion_command(cmd: str) -> bool:
    c = cmd.strip().upper()
    if c in ("$HX", "$HY", "$H"):
        return True
    return c.startswith("G0") or c.startswith("G1") or c.startswith("G53")


def wait_grbl_idle(
    ser: serial.Serial,
    timeout_sec: float = 60.0,
    poll_interval: float = 0.12,
    verbose: bool = False,
) -> bool:
    """รอจน GRBL รายงาน state=Idle (หรือค้างใน Run/Jog แล้วกลับ Idle)."""
    if not ser or not ser.is_open:
        return False
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_state = None
    while time.time() < deadline:
        info = query_grbl_status(ser, timeout=0.5)
        if info:
            state = str(info.get("state") or "").strip().upper()
            if state and state != last_state and verbose:
                print(f"    [GRBL] state={state}")
                last_state = state
            if state == "IDLE":
                return True
            if "ALARM" in state:
                if verbose:
                    print("    [GRBL] ALARM during wait for Idle")
                return False
        time.sleep(poll_interval)
    if verbose:
        print(f"    [GRBL] timeout waiting for Idle ({timeout_sec:.0f}s)")
    return False


def _wait_after_motion(
    ser: serial.Serial,
    cmd: str,
    verbose: bool,
    idle_timeout: float,
    wait_idle: bool,
) -> bool:
    if not wait_idle or not _is_motion_command(cmd):
        return True
    if verbose:
        print(f"    [GRBL] waiting Idle after: {cmd.strip()}")
    return wait_grbl_idle(ser, timeout_sec=idle_timeout, verbose=verbose)


def verify_mpos_near(
    ser: serial.Serial,
    target_x: float,
    target_y: float,
    tolerance_mm: float,
    verbose: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """เช็คว่า MPos ใกล้ target ภายใน tolerance_mm."""
    time.sleep(0.15)
    info = query_grbl_status(ser, timeout=0.8)
    if info is None:
        print("    [GRBL] cannot read status for MPos verify")
        return False, None
    mx = info.get("mpos_x")
    my = info.get("mpos_y")
    if mx is None or my is None:
        print("    [GRBL] MPos missing in status — skip MPos verify")
        return True, info
    dx = abs(float(mx) - float(target_x))
    dy = abs(float(my) - float(target_y))
    ok = dx <= tolerance_mm and dy <= tolerance_mm
    msg = (
        f"    [GRBL] MPos verify: ({mx:.3f},{my:.3f}) "
        f"target ({target_x:.3f},{target_y:.3f}) tol={tolerance_mm:.1f}mm -> {'OK' if ok else 'FAIL'}"
    )
    if verbose or not ok:
        print(msg)
    if not ok:
        print(
            f"    [GRBL] hint: set CAM4_ARM_REF_MPOS_X={mx:.3f} "
            f"CAM4_ARM_REF_MPOS_Y={my:.3f} in config.py if this is your real home"
        )
    return ok, info


def verify_wpos_near_zero(
    ser: serial.Serial,
    tolerance_mm: float,
    verbose: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """หลัง G92 X0 Y0 — เช็คว่า WPos (work) อยู่ใกล้ 0,0."""
    time.sleep(0.15)
    info = query_grbl_status(ser, timeout=0.8)
    if info is None:
        print("    [GRBL] cannot read status for WPos verify")
        return False, None
    wx = info.get("wpos_x")
    wy = info.get("wpos_y")
    if wx is None or wy is None:
        print("    [GRBL] WPos missing — skip WPos verify")
        return True, info
    dx = abs(float(wx))
    dy = abs(float(wy))
    ok = dx <= tolerance_mm and dy <= tolerance_mm
    msg = (
        f"    [GRBL] WPos home verify: ({wx:.3f},{wy:.3f}) "
        f"target (0,0) tol={tolerance_mm:.1f}mm -> {'OK' if ok else 'FAIL'}"
    )
    if verbose or not ok:
        print(msg)
    return ok, info


def run_connect_selftest(
    ser: serial.Serial,
    *,
    move_mm: float = 2.0,
    feed_rate: float = 3000.0,
    verbose: bool = False,
    idle_timeout: float = 30.0,
    wait_idle: bool = True,
) -> bool:
    """
    ขยับทดสอบ ±move_mm บน X/Y แล้วกลับ work home (0,0).
    ยืนยันว่าแขนตอบสนองจริงหลัง homing.
    """
    if not ser or not ser.is_open:
        return False
    move_mm = max(0.5, float(move_mm))
    if verbose:
        print(f"    [GRBL] self-test: ±{move_mm:.1f} mm on X/Y")

    steps = [
        "G90",
        "G21",
        f"G0 X0 Y0 F{feed_rate:.1f}",
        "G91",
        f"G0 X{move_mm:.3f} F{feed_rate:.1f}",
        f"G0 X{-move_mm:.3f} F{feed_rate:.1f}",
        f"G0 Y{move_mm:.3f} F{feed_rate:.1f}",
        f"G0 Y{-move_mm:.3f} F{feed_rate:.1f}",
        "G90",
        f"G0 X0 Y0 F{feed_rate:.1f}",
    ]
    for cmd in steps:
        r = _send_gcode_line(ser, cmd, timeout=15.0, verbose=verbose)
        if _response_failed(r):
            if verbose:
                print(f"    [GRBL] self-test failed at: {cmd}")
            return False
        if not _wait_after_motion(ser, cmd, verbose, idle_timeout, wait_idle):
            if verbose:
                print(f"    [GRBL] self-test Idle timeout at: {cmd}")
            return False
        time.sleep(0.05)
    if verbose:
        print("    [GRBL] self-test OK")
    return True


def run_homing_cycle(
    ser: serial.Serial,
    *,
    homing_commands: Optional[list] = None,
    ref_mpos_x: Optional[float] = None,
    ref_mpos_y: Optional[float] = None,
    verbose: Optional[bool] = None,
    homing_delay: Optional[float] = None,
    unlock_first: bool = False,
    feed_rate: Optional[float] = None,
    use_sequential: Optional[bool] = None,
) -> bool:
    """
    รัน homing ตามลำดับ: ส่งทีละคำสั่ง รอ ok/error/ALARM ก่อนส่งถัดไป.
    ลำดับ: homing_commands (เช่น $X, $HY, $HX) → G53 G0 → (verify MPos) → G92 X0 Y0.
    """
    if not ser or not ser.is_open:
        return False

    if homing_commands is None and config is not None:
        homing_commands = getattr(
            config, "CAM4_ARM_HOMING_COMMANDS", ["$X", "$HY", "$HX"]
        )
    if homing_commands is None:
        homing_commands = ["$X", "$HY", "$HX"]

    if verbose is None and config is not None:
        verbose = bool(getattr(config, "CAM4_ARM_VERBOSE_GCODE", False))

    wait_idle = True
    idle_timeout = 60.0
    mpos_tol = 15.0
    verify_mpos = False
    verify_wpos = True
    wpos_tol = 3.0
    if config is not None:
        wait_idle = bool(getattr(config, "CAM4_ARM_HOMING_WAIT_IDLE", True))
        idle_timeout = float(getattr(config, "CAM4_ARM_HOMING_IDLE_TIMEOUT_SEC", idle_timeout))
        mpos_tol = float(getattr(config, "CAM4_ARM_HOMING_MPOS_TOLERANCE_MM", mpos_tol))
        verify_mpos = bool(getattr(config, "CAM4_ARM_HOMING_VERIFY_MPOS", False))
        verify_wpos = bool(getattr(config, "CAM4_ARM_HOMING_VERIFY_WPOS_HOME", True))
        wpos_tol = float(getattr(config, "CAM4_ARM_HOMING_WPOS_TOLERANCE_MM", wpos_tol))

    _drain_serial(ser, max_time=0.3)

    def timeout_for(cmd: str) -> float:
        if cmd.strip().upper() in ("$HX", "$HY", "$H"):
            return 5.0
        if cmd.strip().upper() == "$X":
            return 2.0
        return 5.0

    for cmd in homing_commands:
        t = timeout_for(cmd)
        r = _send_gcode_line(ser, cmd.strip(), timeout=t, verbose=verbose)
        if _response_failed(r):
            if verbose:
                print(f"    [GRBL] <<< ข้อผิดพลาดที่คำสั่ง: {cmd}")
            return False
        if not _wait_after_motion(ser, cmd, verbose, idle_timeout, wait_idle):
            if verbose:
                print(f"    [GRBL] Idle timeout after: {cmd}")
            return False
        time.sleep(0.08)

    if ref_mpos_x is not None and ref_mpos_y is not None:
        home_feed = 3000.0
        if config is not None:
            home_feed = float(getattr(config, "CAM4_ARM_HOME_FEED_RATE", home_feed))
        if feed_rate is not None:
            home_feed = float(feed_rate)
        prep_cmds = ["G90"]
        if config is None or getattr(config, "CAM4_ARM_GRBL_UNITS_MM", True):
            prep_cmds.append("G21")
        for prep in prep_cmds:
            r = _send_gcode_line(ser, prep, timeout=2.0, verbose=verbose)
            if _response_failed(r):
                if verbose:
                    print(f"    [GRBL] <<< ข้อผิดพลาดที่คำสั่ง: {prep}")
                return False
            time.sleep(0.05)
        g53 = f"G53 G0 X{ref_mpos_x:.3f} Y{ref_mpos_y:.3f} F{home_feed:.1f}"
        if verbose:
            print("    [GRBL] >>> moving to home position (machine coords)")
        r = _send_gcode_line(ser, g53, timeout=5.0, verbose=verbose)
        if _response_failed(r):
            if verbose:
                print(f"    [GRBL] <<< ข้อผิดพลาดที่คำสั่ง: {g53}")
            return False
        if not _wait_after_motion(ser, g53, verbose, idle_timeout, wait_idle):
            if verbose:
                print("    [GRBL] Idle timeout after G53 home move")
            return False
        ok_mpos, mpos_info = verify_mpos_near(
            ser, ref_mpos_x, ref_mpos_y, mpos_tol, verbose=verbose
        )
        if verify_mpos and not ok_mpos:
            print(
                f"❌ Homing verify failed: MPos not near ({ref_mpos_x}, {ref_mpos_y}) "
                f"within {mpos_tol} mm"
            )
            return False
        if not verify_mpos and mpos_info is not None:
            mx = mpos_info.get("mpos_x")
            my = mpos_info.get("mpos_y")
            if mx is not None and my is not None:
                print(
                    f"    [GRBL] MPos after G53 (info only): ({mx:.3f},{my:.3f}) "
                    f"REF_MPOS=({ref_mpos_x:.3f},{ref_mpos_y:.3f})"
                )
        r = _send_gcode_line(ser, "G92 X0 Y0", timeout=2.0, verbose=verbose)
        if _response_failed(r):
            if verbose:
                print("    [GRBL] <<< ข้อผิดพลาดที่คำสั่ง: G92 X0 Y0")
            return False
        time.sleep(0.1)
        if verify_wpos:
            ok_wpos, _ = verify_wpos_near_zero(ser, wpos_tol, verbose=verbose)
            if not ok_wpos:
                print(
                    f"❌ Homing verify failed: WPos not near (0,0) within {wpos_tol} mm after G92"
                )
                return False

    return True


def run_homing_cycle_standalone(
    port: Optional[str] = None,
    baud: int = 115200,
    timeout: float = 0.1,
    wake_sleep: float = 2.0,
    verbose: Optional[bool] = None,
    homing_delay: Optional[float] = None,
    enable_homing_if_disabled: bool = False,
) -> bool:
    """
    เปิด serial เอง, ปลุก GRBL, รัน homing cycle ($X → $HY → $HX → G53 → G92), ปิด serial.
    """
    if port is None and config is not None:
        port = getattr(config, "CAM4_ARM_SERIAL_PORT", "/dev/ttyUSB0")
    if port is None:
        port = "/dev/ttyUSB0"
    if verbose is None and config is not None:
        verbose = bool(getattr(config, "CAM4_ARM_VERBOSE_GCODE", False))

    ref_mpos_x = getattr(config, "CAM4_ARM_REF_MPOS_X", -77.001) if config else -77.001
    ref_mpos_y = getattr(config, "CAM4_ARM_REF_MPOS_Y", -41.988) if config else -41.988

    try:
        print(f"🔌 Opening {port} at {baud} baud...")
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(wake_sleep)
        ser.write(b"\r\n\r\n")
        time.sleep(wake_sleep)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"❌ Failed to open serial: {e}")
        return False

    try:
        ok = run_homing_cycle(
            ser,
            ref_mpos_x=ref_mpos_x,
            ref_mpos_y=ref_mpos_y,
            verbose=verbose,
        )
        if ok:
            print("✅ Homing completed ($X → $HY → $HX → G53 → G92 X0 Y0).")
        else:
            print("❌ Homing reported error or alarm.")
        return ok
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    v = "--verbose" in sys.argv or "-v" in sys.argv
    raise SystemExit(0 if run_homing_cycle_standalone(verbose=v) else 1)
