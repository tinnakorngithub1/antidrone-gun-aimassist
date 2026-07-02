#!/usr/bin/env python3
"""
สคริปต์รัน homing อย่างเดียว (หา home จาก limit switch)
-------------------------------------------------------
- เปิด serial → $X, $H → รอจบ → ปิด serial
- ไม่โหลด controller เต็มชุด ใช้สำหรับทดสอบ limit switch / ตั้งเครื่อง

Usage:
  python run_cam4_homing.py
  python run_cam4_homing.py --port /dev/ttyUSB1
"""

import argparse
import sys

from cam4_arm_homing import run_homing_cycle_standalone


def main() -> int:
    p = argparse.ArgumentParser(description="Run GRBL homing cycle ($H) for Cam4 arm.")
    p.add_argument("--port", type=str, default=None, help="Serial port (default: from config)")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--delay", type=float, default=None, help="Homing wait seconds (default: from config)")
    p.add_argument("--enable-homing", action="store_true", help="Send $22=1 if homing is disabled (so $H will move the arm)")
    p.add_argument("--no-verbose", action="store_true", help="Disable GRBL response verbose")
    args = p.parse_args()

    ok = run_homing_cycle_standalone(
        port=args.port,
        baud=args.baud,
        homing_delay=args.delay,
        verbose=not args.no_verbose,
        enable_homing_if_disabled=args.enable_homing,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
