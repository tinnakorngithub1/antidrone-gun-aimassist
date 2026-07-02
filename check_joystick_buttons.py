"""
check_joystick_buttons.py
-------------------------
ตรวจสอบและยืนยัน mapping จอยสติ๊ก (เช่น Logitech Extreme 3D Pro):
- ปุ่ม 1–12, ปุ่มกลาง (hat/ปุ่มเล็กสีดำ), คันโยงเล็ก (บวก/ลบ), คันจอยหลัก
- โหมดสแกน: แสดง real-time ว่ากด/โยกอะไรอยู่
- โหมด confirm: บอกให้กดปุ่มหมายเลข N ตามลำดับ แล้วรอรับรู้
- บันทึก event ลงไฟล์ (JSON หรือ CSV) เพื่อใช้อ้างอิงและพัฒนาต่อ

Usage:
  python check_joystick_buttons.py              # โหมดสแกน (ไม่บันทึก)
  python check_joystick_buttons.py --log        # โหมดสแกน + บันทึก event
  python check_joystick_buttons.py --log --format csv --output events.csv
  python check_joystick_buttons.py --confirm     # โหมด confirm
  python check_joystick_buttons.py --confirm --log

ปุ่มคีย์บอร์ดในโหมดสแกน: S=สแกน, C=confirm, L=สลับบันทึก, Q=ออก
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Optional

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    pygame = None  # type: ignore
    _HAS_PYGAME = False


# --- Config (ปรับได้ตามจอยจริง) ---
NUM_BUTTONS = 12
LEVER_AXIS = 2
LEVER_POSITIVE_THRESHOLD = 0.5
LEVER_NEGATIVE_THRESHOLD = -0.5
JOY_AXIS_X = 0
JOY_AXIS_Y = 1
USE_HAT_CENTRE = True  # True = ใช้ hat (0,0) เป็นปุ่มกลาง; False = ใช้ CENTRE_BUTTON_INDEX
CENTRE_BUTTON_INDEX: Optional[int] = None  # ถ้าไม่ใช้ hat ใส่ button index ที่เป็นปุ่มกลาง
AXIS_LOG_CHANGE_THRESHOLD = 0.15  # บันทึกแกนเมื่อเปลี่ยนเกินค่านี้ (ลดการบันทึกถี่)
AXIS_LOG_MIN_INTERVAL_SEC = 0.2  # บันทึกแกนห่างกันอย่างน้อยกี่วินาที
DEBOUNCE_RELEASE_MS = 200  # ปล่อยปุ่ม/คันก่อนนับในโหมด confirm (มิลลิวินาที)

DEFAULT_LOG_PATH = "joystick_events"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def write_event(
    filepath: str,
    event_type: str,
    index: int,
    value: Any,
    label: str,
    log_format: str = "json",
    write_header: bool = False,
    button_number: Optional[int] = None,
) -> None:
    """เขียน event หนึ่งรายการลงไฟล์ (append). รองรับ JSON lines และ CSV. button_number = หมายเลขปุ่ม 1-12 (ใช้กับ type=button)."""
    timestamp = get_timestamp()
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        if log_format == "csv":
            if write_header:
                f.write("timestamp,type,index,button_number,value,label\n")
            btn = "" if button_number is None else str(button_number)
            value_str = str(value).replace(",", ";") if not isinstance(value, list) else ";".join(map(str, value))
            f.write(f'"{timestamp}","{event_type}",{index},"{btn}","{value_str}","{label}"\n')
        else:
            record = {
                "time": timestamp,
                "type": event_type,
                "index": index,
                "value": value,
                "label": label,
            }
            if button_number is not None:
                record["button_number"] = button_number
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def button_label(index: int) -> str:
    if 0 <= index < NUM_BUTTONS:
        return f"ปุ่ม {index + 1}"
    if CENTRE_BUTTON_INDEX is not None and index == CENTRE_BUTTON_INDEX:
        return "ปุ่มกลาง"
    return f"ปุ่ม index {index}"


def hat_value_to_label(hx: int, hy: int) -> str:
    if hx == 0 and hy == 0:
        return "Hat กลาง"
    labels = []
    if hy == 1:
        labels.append("บน")
    elif hy == -1:
        labels.append("ล่าง")
    if hx == 1:
        labels.append("ขวา")
    elif hx == -1:
        labels.append("ซ้าย")
    return "Hat " + "".join(labels) if labels else f"Hat ({hx},{hy})"


def axis_label(axis_index: int, value: float) -> str:
    if axis_index == JOY_AXIS_X:
        return "จอย X"
    if axis_index == JOY_AXIS_Y:
        return "จอย Y"
    if axis_index == LEVER_AXIS:
        return "คันโยงเล็ก บวก" if value >= LEVER_POSITIVE_THRESHOLD else "คันโยงเล็ก ลบ" if value <= LEVER_NEGATIVE_THRESHOLD else "คันโยงเล็ก"
    return f"แกน {axis_index}"


def run_scan(
    js: pygame.joystick.Joystick,
    log_enabled: bool,
    log_path: str,
    log_format: str,
) -> None:
    """โหมดสแกน: แสดง real-time และบันทึก event เมื่อมีการเปลี่ยนแปลง."""
    n_buttons = js.get_numbuttons()
    n_hats = js.get_numhats()
    n_axes = js.get_numaxes()

    prev_buttons = [False] * n_buttons if n_buttons else [False]
    prev_hat = (0, 0)
    prev_axes = [0.0] * n_axes if n_axes else [0.0]
    axis_last_log_time = [0.0] * n_axes if n_axes else [0.0]
    csv_header_written = os.path.exists(log_path) and log_format == "csv"

    print("โหมดสแกน — กดปุ่มหรือโยกคันใดก็ได้ (S=สแกน C=confirm L=สลับบันทึก Q=ออก)")
    if log_enabled:
        print(f"  บันทึกไปที่: {log_path} (รูปแบบ: {log_format})")
    else:
        print("  บันทึก: ปิด (กด L เพื่อเปิด)")

    clock = pygame.time.Clock()
    while True:
        pygame.event.pump()
        t = pygame.time.get_ticks() / 1000.0

        # ปุ่ม
        for i in range(n_buttons):
            try:
                pressed = bool(js.get_button(i))
            except Exception:
                pressed = False
            if pressed != prev_buttons[i]:
                prev_buttons[i] = pressed
                label = button_label(i)
                val = 1 if pressed else 0
                num_display = f"หมายเลข {i + 1}" if i < NUM_BUTTONS else f"index {i}"
                print(f"  ปุ่ม: {label} ({num_display}) = {'กด' if pressed else 'ปล่อย'}")
                if log_enabled:
                    btn_num = (i + 1) if i < NUM_BUTTONS else None
                    if log_format == "csv" and not csv_header_written:
                        write_event(log_path, "button", i, val, label, log_format, write_header=True, button_number=btn_num)
                        csv_header_written = True
                    else:
                        write_event(log_path, "button", i, val, label, log_format, button_number=btn_num)

        # Hat
        if n_hats > 0:
            try:
                hat = js.get_hat(0)
            except Exception:
                hat = (0, 0)
            if hat != prev_hat:
                prev_hat = hat
                label = hat_value_to_label(hat[0], hat[1])
                print(f"  Hat: {label} value={hat}")
                if log_enabled:
                    if log_format == "csv" and not csv_header_written:
                        write_event(log_path, "hat", 0, list(hat), label, log_format, write_header=True)
                        csv_header_written = True
                    else:
                        write_event(log_path, "hat", 0, list(hat), label, log_format)

        # แกน (จอย X/Y + คันโยงเล็ก และอื่นๆ)
        for ax in range(n_axes):
            try:
                v = float(js.get_axis(ax))
            except Exception:
                v = 0.0
            prev = prev_axes[ax]
            if abs(v - prev) >= AXIS_LOG_CHANGE_THRESHOLD and (t - axis_last_log_time[ax]) >= AXIS_LOG_MIN_INTERVAL_SEC:
                prev_axes[ax] = v
                axis_last_log_time[ax] = t
                label = axis_label(ax, v)
                print(f"  แกน: {label} (axis {ax}) = {v:.3f}")
                if log_enabled:
                    if log_format == "csv" and not csv_header_written:
                        write_event(log_path, "axis", ax, round(v, 4), label, log_format, write_header=True)
                        csv_header_written = True
                    else:
                        write_event(log_path, "axis", ax, round(v, 4), label, log_format)
            else:
                prev_axes[ax] = v

        # คีย์บอร์ด
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    print("ออกจากโหมดสแกน")
                    return None
                if e.key == pygame.K_l:
                    log_enabled = not log_enabled
                    print(f"  บันทึก: {'เปิด' if log_enabled else 'ปิด'}")
                if e.key == pygame.K_c:
                    print("สลับไปโหมด confirm...")
                    return "confirm"

        clock.tick(30)


def run_confirm(
    js: pygame.joystick.Joystick,
    log_enabled: bool,
    log_path: str,
    log_format: str,
) -> None:
    """โหมด confirm: บอกให้กดปุ่ม/โยกตามลำดับ แล้วรอรับรู้."""
    n_buttons = js.get_numbuttons()
    n_hats = js.get_numhats()
    n_axes = js.get_numaxes()
    csv_header_written = os.path.exists(log_path) and log_format == "csv"

    steps: list[tuple[str, Any]] = []
    for i in range(1, NUM_BUTTONS + 1):
        steps.append(("button", i))
    if USE_HAT_CENTRE:
        steps.append(("centre_hat", None))
    elif CENTRE_BUTTON_INDEX is not None:
        steps.append(("centre_button", CENTRE_BUTTON_INDEX))
    steps.append(("lever", "up"))
    steps.append(("lever", "down"))
    steps.append(("joystick", "any"))

    for step_type, step_value in steps:
        if step_type == "button":
            num = step_value
            prompt = f"กดปุ่มหมายเลข {num}"
            target_index = num - 1
        elif step_type == "centre_hat":
            prompt = "กดปุ่มกลาง (หรือ Hat กลาง)"
            target_index = -1
        elif step_type == "centre_button":
            prompt = "กดปุ่มกลาง"
            target_index = step_value
        elif step_type == "lever":
            prompt = f"โยกคันเล็ก{'ขึ้น (บวก)' if step_value == 'up' else 'ลง (ลบ)'}"
            target_index = -1
        else:
            prompt = "โยกจอยหลักไปทิศใดก็ได้"
            target_index = -1

        print(prompt + " ... ", end="", flush=True)

        # รอจนกด/โยกตรงตาม step
        clock = pygame.time.Clock()
        done = False
        logged = False
        release_after = 0

        while not done:
            pygame.event.pump()
            t = pygame.time.get_ticks()

            if step_type == "button" and 0 <= target_index < n_buttons:
                try:
                    if js.get_button(target_index):
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                label = button_label(target_index)
                                btn_num = (target_index + 1) if target_index < NUM_BUTTONS else None
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "button", target_index, 1, label, log_format, write_header=True, button_number=btn_num)
                                    csv_header_written = True
                                else:
                                    write_event(log_path, "button", target_index, 1, label, log_format, button_number=btn_num)
                                logged = True
                    else:
                        if done and release_after > 0:
                            pass
                        release_after = 0
                except Exception:
                    pass

            elif step_type == "centre_hat" and n_hats > 0:
                try:
                    hat = js.get_hat(0)
                    if hat == (0, 0):
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                label = "Hat กลาง"
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "hat", 0, [0, 0], label, log_format, write_header=True)
                                    csv_header_written = True
                                else:
                                    write_event(log_path, "hat", 0, [0, 0], label, log_format)
                                logged = True
                    else:
                        release_after = 0
                except Exception:
                    pass

            elif step_type == "centre_button" and target_index < n_buttons:
                try:
                    if js.get_button(target_index):
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "button", target_index, 1, "ปุ่มกลาง", log_format, write_header=True)
                                    csv_header_written = True
                                else:
                                    write_event(log_path, "button", target_index, 1, "ปุ่มกลาง", log_format)
                                logged = True
                    else:
                        release_after = 0
                except Exception:
                    pass

            elif step_type == "lever" and n_axes > LEVER_AXIS:
                try:
                    v = float(js.get_axis(LEVER_AXIS))
                    if step_value == "up" and v >= LEVER_POSITIVE_THRESHOLD:
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                label = "คันโยงเล็ก บวก"
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "axis", LEVER_AXIS, round(v, 4), label, log_format, write_header=True)
                                    csv_header_written = True
                                else:
                                    write_event(log_path, "axis", LEVER_AXIS, round(v, 4), label, log_format)
                                logged = True
                    elif step_value == "down" and v <= LEVER_NEGATIVE_THRESHOLD:
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                label = "คันโยงเล็ก ลบ"
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "axis", LEVER_AXIS, round(v, 4), label, log_format, write_header=True)
                                    csv_header_written = True
                                else:
                                    write_event(log_path, "axis", LEVER_AXIS, round(v, 4), label, log_format)
                                logged = True
                    else:
                        release_after = 0
                except Exception:
                    pass

            elif step_type == "joystick":
                try:
                    x = float(js.get_axis(JOY_AXIS_X))
                    y = float(js.get_axis(JOY_AXIS_Y))
                    if abs(x) > 0.2 or abs(y) > 0.2:
                        if release_after == 0:
                            release_after = t + DEBOUNCE_RELEASE_MS
                        elif t >= release_after:
                            done = True
                            print("รับรู้แล้ว")
                            if log_enabled and not logged:
                                if log_format == "csv" and not csv_header_written:
                                    write_event(log_path, "axis", JOY_AXIS_X, round(x, 4), "จอย X", log_format, write_header=True)
                                    csv_header_written = True
                                    write_event(log_path, "axis", JOY_AXIS_Y, round(y, 4), "จอย Y", log_format)
                                else:
                                    write_event(log_path, "axis", JOY_AXIS_X, round(x, 4), "จอย X", log_format)
                                    write_event(log_path, "axis", JOY_AXIS_Y, round(y, 4), "จอย Y", log_format)
                                logged = True
                    else:
                        release_after = 0
                except Exception:
                    pass

            clock.tick(30)

    print("ยืนยันครบทุกขั้นแล้ว")


def main() -> int:
    parser = argparse.ArgumentParser(description="ตรวจสอบและยืนยัน mapping จอยสติ๊ก")
    parser.add_argument("--log", action="store_true", help="เปิดการบันทึก event ตั้งแต่ต้น (โหมดสแกนยังกด L สลับได้)")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="รูปแบบไฟล์บันทึก (default: json)")
    parser.add_argument("--output", default="", help="path ไฟล์บันทึก (default: joystick_events.json หรือ .csv)")
    parser.add_argument("--confirm", action="store_true", help="รันโหมด confirm แทนโหมดสแกน")
    args = parser.parse_args()

    if not _HAS_PYGAME:
        print("ต้องติดตั้ง pygame ก่อน: pip install pygame")
        return 1

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ไม่พบจอยสติ๊ก")
        return 1

    js = pygame.joystick.Joystick(0)
    js.init()
    n_axes = js.get_numaxes()
    n_buttons = js.get_numbuttons()
    n_hats = js.get_numhats()
    print(f"จอย: {js.get_name()}")
    print(f"  แกน: {n_axes}, ปุ่ม: {n_buttons}, Hat: {n_hats}")

    if args.output:
        log_path = args.output
    else:
        ext = "json" if args.format == "json" else "csv"
        log_path = os.path.join(SCRIPT_DIR, f"{DEFAULT_LOG_PATH}.{ext}")

    log_enabled = args.log

    if args.confirm:
        run_confirm(js, log_enabled, log_path, args.format)
        return 0

    result = run_scan(js, log_enabled, log_path, args.format)
    while result == "confirm":
        run_confirm(js, log_enabled, log_path, args.format)
        result = run_scan(js, log_enabled, log_path, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
