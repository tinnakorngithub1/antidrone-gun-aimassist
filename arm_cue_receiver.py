"""
arm_cue_receiver.py
-------------------
UDP cue receiver: รับ confirmed drone target cue จาก 11/cam8 ใน background thread.
ใช้ใน 22_gun_aim_assist_vector.py ฝั่ง Jetson แขน (192.168.144.66).

Design:
  - Network thread รับ UDP แล้วเก็บ latest_cue ไว้ใน _latest_cue (lock-protected)
  - Main loop เรียก get_latest_cue() — คืน None ถ้า stale (อายุเกิน cue_ttl_ms)
  - Network thread ห้ามแตะ serial/arm ตรงๆ ทุกการตัดสินใจอยู่ที่ main loop

Cue record fields (ที่อาจมีใน dict):
  cx, cy            : float  — จุดกลางเป้าหมายบน frame
  frame_w, frame_h  : int    — ขนาด frame ที่ cx/cy อ้างอิง
  bbox_w, bbox_h    : int|null   — ขนาด bbox บน frame detection (พิกเซล); None ถ้าไม่มี
  bbox_w_norm, bbox_h_norm : float|null — bbox_w/frame_w, bbox_h/frame_h (0–1); None ถ้าไม่มี
  distance_m        : float|null
  tier              : str    — "confirmed" | "possible"; ไม่มี field = "confirmed" (backward compat)
  confidence        : float|null — YOLO confidence (0–1); null = ยืนยันด้วย kinematics
  source_camera, target_id, timestamp, sequence, cue_ttl_ms
  target_lat, target_lng : float|null — optional (Phase 2, จาก Jetson #1)
  recv_timestamp    : float  — stamped local clock เมื่อรับ (ใช้คำนวณ TTL)

การเลือก primary target จาก list: เลือก confirmed ก่อน possible,
ภายใน tier เดียวกันเลือก confidence สูงสุด (ไม่พึ่งลำดับจากผู้ส่ง).
"""

import json
import socket
import threading
import time
from typing import Any, Dict, Optional

DEFAULT_PORT = 5765
DEFAULT_CUE_TTL_MS = 500


def _cue_rank(cue: Dict[str, Any]) -> tuple:
    """sort key เลือก primary cue: confirmed ก่อน possible, confidence มาก่อน."""
    tier = str(cue.get("tier", "confirmed")).lower()
    conf = cue.get("confidence")
    return (0 if tier == "confirmed" else 1, -(conf if conf is not None else 0.0))


class ArmCueReceiver:
    """
    รับ UDP cue payloads ใน background thread.
    Main loop เรียก get_latest_cue() เพื่อดึง cue ล่าสุดที่ยังสด.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        default_ttl_ms: int = DEFAULT_CUE_TTL_MS,
        enabled: bool = True,
    ) -> None:
        self.port = port
        self.default_ttl_ms = int(default_ttl_ms)
        self.enabled = bool(enabled)

        self._lock = threading.Lock()
        self._latest_cue: Optional[Dict[str, Any]] = None
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._recv_count: int = 0
        self._last_recv_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("", self.port))
            self._sock.settimeout(0.5)
            print(f"[ArmCueReceiver] listening UDP port {self.port}")
        except Exception as e:
            print(f"[ArmCueReceiver] bind error: {e}")
            self._sock = None
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="arm_cue_receiver"
        )
        self._thread.start()

    def get_latest_cue(self) -> Optional[Dict[str, Any]]:
        """คืน cue ล่าสุดที่ยังไม่ stale; None ถ้าไม่มีหรืออายุเกิน TTL."""
        with self._lock:
            cue = self._latest_cue
        if cue is None:
            return None
        ttl_ms = cue.get("cue_ttl_ms", self.default_ttl_ms)
        # ใช้ recv_timestamp (local clock) แทน timestamp จากผู้ส่ง เพื่อหลีก clock skew
        age_ms = (time.time() - cue.get("recv_timestamp", 0.0)) * 1000.0
        if age_ms > ttl_ms:
            return None
        return cue

    def get_cue_age_ms(self) -> Optional[float]:
        """คืนอายุของ cue ล่าสุด (ms) โดยไม่คำนึงถึง TTL; None ถ้าไม่เคยรับ."""
        with self._lock:
            cue = self._latest_cue
        if cue is None:
            return None
        return (time.time() - cue.get("recv_timestamp", 0.0)) * 1000.0

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def status_label(self) -> str:
        """คืน string สั้นสำหรับ HUD แสดงสถานะ receiver."""
        if not self.enabled:
            return "RCV:off"
        age = self.get_cue_age_ms()
        if age is None:
            return "RCV:no cue"
        cue = self.get_latest_cue()
        ttl_ms = (cue or {}).get("cue_ttl_ms", self.default_ttl_ms) if cue else self.default_ttl_ms
        raw_age = self.get_cue_age_ms() or 0.0
        if raw_age > ttl_ms:
            return f"RCV:stale {raw_age:.0f}ms"
        _tier = str((cue or {}).get("tier", "confirmed")).lower()
        _tag = "" if _tier == "confirmed" else f" [{_tier.upper()}]"
        return f"RCV:{raw_age:.0f}ms{_tag}"

    # ------------------------------------------------------------------
    # Background thread — ห้ามแตะ arm/serial ที่นี่
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        while self._running:
            if self._sock is None:
                time.sleep(0.1)
                continue
            try:
                data, _ = self._sock.recvfrom(65535)
                parsed = json.loads(data.decode("utf-8"))
                # parsed อาจเป็น list (หลาย candidates) หรือ dict (single)
                if isinstance(parsed, list) and parsed:
                    _dicts = [c for c in parsed if isinstance(c, dict)]
                    if not _dicts:
                        continue
                    # primary target: confirmed ก่อน possible, แล้ว confidence สูงสุด
                    cue = min(_dicts, key=_cue_rank)
                elif isinstance(parsed, dict):
                    cue = parsed
                else:
                    continue
                now = time.time()
                cue["recv_timestamp"] = now  # stamp local clock เพื่อ TTL
                with self._lock:
                    self._latest_cue = cue
                self._recv_count += 1
                self._last_recv_time = now
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    print(f"[ArmCueReceiver] recv error: {e}")
                time.sleep(0.01)
