"""
arm_cue_sender.py
-----------------
UDP cue sender: ส่ง confirmed drone target จาก 11/cam8 ไปยัง Jetson แขน.

Protocol:
  - JSON list ต่อ datagram, UTF-8
  - แต่ละ element คือ cue record สำหรับ confirmed drone 1 ตัว
  - ฝั่ง receiver (arm_cue_receiver.py ใน 22) ใช้ element แรกเป็น primary target

Record fields (ทุก field เสมอ):
  cx, cy         : float  — จุดกลางเป้าหมายบน frame (พิกเซล)
  frame_w, frame_h: int   — ขนาด frame ที่ cx/cy อ้างอิง
  bbox_w, bbox_h : int|null — ขนาด bounding box บน frame detection (พิกเซล); null ถ้าไม่มีข้อมูล
  bbox_w_norm, bbox_h_norm: float|null — bbox_w/frame_w, bbox_h/frame_h (0–1); null ถ้า bbox ไม่มี
  distance_m     : float|null
  target_id, source_camera, timestamp, sequence, cue_ttl_ms
"""

import json
import socket
import threading
import time
from typing import Any, List, Optional, Tuple

DEFAULT_HOST = "192.168.144.66"
DEFAULT_PORT = 5765
DEFAULT_SEND_INTERVAL_HZ = 10.0   # ส่งได้สูงสุด 10 ครั้ง/วินาที
DEFAULT_CUE_TTL_MS = 500          # ฝั่ง receiver จะทิ้ง cue เมื่ออายุเกิน 500 ms


class ArmCueSender:
    """
    ส่ง confirmed target cue ผ่าน UDP ไปยัง Jetson แขน.
    Thread-safe: เรียก push() จาก main loop ของ 11, background thread ส่งออก.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        send_interval_hz: float = DEFAULT_SEND_INTERVAL_HZ,
        cue_ttl_ms: int = DEFAULT_CUE_TTL_MS,
        source_camera: str = "cam8",
        enabled: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.min_interval = 1.0 / max(float(send_interval_hz), 0.5)
        self.cue_ttl_ms = int(cue_ttl_ms)
        self.source_camera = source_camera
        self.enabled = bool(enabled)

        self._lock = threading.Lock()
        self._pending: Optional[List[dict]] = None
        self._seq: int = 0
        self._last_send_time: float = 0.0
        self._last_send_ok: bool = False
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(0.1)
        except Exception as e:
            print(f"[ArmCueSender] socket init error: {e}")
            self._sock = None
        self._running = True
        self._thread = threading.Thread(
            target=self._send_loop, daemon=True, name="arm_cue_sender"
        )
        self._thread.start()
        print(f"[ArmCueSender] started -> {self.host}:{self.port}")

    def push(
        self,
        candidates: List[Tuple],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        candidates: list of tuples จาก _ws_confirmed_candidates
          - 4-tuple: (cx, cy, obj_id, distance_m)                       — backward compat
          - 6-tuple: (cx, cy, obj_id, distance_m, bbox_w, bbox_h)       — ส่งขนาดกล่องด้วย
        บันทึก payload ล่าสุดไว้; background thread จะส่งออกตาม rate limit.
        """
        if not self.enabled or not candidates:
            return
        now = time.time()
        fw = int(frame_w)
        fh = int(frame_h)
        with self._lock:
            self._seq += 1
            seq = self._seq
            payloads = []
            for entry in candidates:
                cx, cy, obj_id, dist_m = entry[0], entry[1], entry[2], entry[3]
                bw = int(entry[4]) if len(entry) > 4 and entry[4] is not None else None
                bh = int(entry[5]) if len(entry) > 5 and entry[5] is not None else None
                bw_norm = float(bw) / fw if (bw is not None and fw > 0) else None
                bh_norm = float(bh) / fh if (bh is not None and fh > 0) else None
                payloads.append(
                    {
                        "source_camera": self.source_camera,
                        "target_id": str(obj_id),
                        "cx": float(cx),
                        "cy": float(cy),
                        "frame_w": fw,
                        "frame_h": fh,
                        "bbox_w": bw,
                        "bbox_h": bh,
                        "bbox_w_norm": bw_norm,
                        "bbox_h_norm": bh_norm,
                        "distance_m": float(dist_m) if dist_m is not None else None,
                        "timestamp": now,
                        "sequence": seq,
                        "cue_ttl_ms": self.cue_ttl_ms,
                    }
                )
            self._pending = payloads

    def set_enabled(self, enabled: bool) -> None:
        """สลับเปิด/ปิดการส่ง UDP — ปิดแล้วเคลียร์คิว pending (ประหยัด LAN)."""
        with self._lock:
            self.enabled = bool(enabled)
            if not self.enabled:
                self._pending = None

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    @property
    def last_send_ok(self) -> bool:
        return self._last_send_ok

    @property
    def last_send_time(self) -> float:
        return self._last_send_time

    def status_label(self) -> str:
        """คืน string สั้นสำหรับ HUD ต่อท้าย ARM CUE (A) ON …"""
        if not self.enabled:
            return "LAN off"
        if not self._last_send_ok:
            return f"{self.host}:ERR"
        age = time.time() - self._last_send_time
        return f"{self.host}:{age*1000:.0f}ms"

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _send_loop(self) -> None:
        while self._running:
            with self._lock:
                pending = self._pending
                self._pending = None

            now = time.time()
            if pending is not None and (now - self._last_send_time) >= self.min_interval:
                if self._sock is not None:
                    try:
                        data = json.dumps(pending, separators=(",", ":")).encode("utf-8")
                        self._sock.sendto(data, (self.host, self.port))
                        self._last_send_ok = True
                        self._last_send_time = now
                    except Exception as e:
                        self._last_send_ok = False
                        print(f"[ArmCueSender] send error: {e}")
            time.sleep(0.01)
