"""
antidrone_ws_exporter.py
========================
Websocket publisher สำหรับส่งข้อมูลโดรนที่ยืนยันแล้วออก endpoint ภายนอก

Schema ที่ส่งออก (top-level):
    {
        "lat": float,          # lat ของกล้อง
        "lng": float,          # lng ของกล้อง
        "heading": float,      # heading ของกล้อง (องศาจากทิศเหนือ)
        "fov": float,          # horizontal FOV ของกล้อง (องศา)
        "timestamp": str,      # ISO-8601 UTC เช่น "2026-03-07T15:11:00Z"
        "drones": [            # รายการโดรนที่กำลังเห็นตอนนี้ ([] ถ้าไม่มี)
            {
                "drone_id": str,       # "DRONE_1", "DRONE_2", ...
                "lat": float | null,
                "lng": float | null,
                "distance": float | null,
                "altitude": null,      # เฟสแรกยังไม่พร้อม
                "speed": null,         # เฟสแรกยังไม่พร้อม
            },
            ...
        ]
    }

confirmed rule: path แดง (kinematic_confirmed OR yolo_state == 'red') หรือ YOLO full-frame
                conf >= drone threshold ที่ไม่ match path แดงเดิม (สอดคล้อง HUD red_drone_count)
ordering rule:  sort ตาม center_x ซ้าย→ขวา, tie-break ด้วย center_y แล้ว obj_id / synthetic id
drone_id:       "DRONE_1" .. "DRONE_N"  (ลำดับปัจจุบัน reset ทุก frame)
empty policy:   ส่ง drones=[] เสมอถ้าไม่มีโดรน (ไม่ตัด field)
null policy:    field ที่ยังไม่พร้อมส่งเป็น null
"""

from __future__ import annotations

import json
import logging
import math
import queue
import socket
import ssl
import base64
import os
import struct
import threading
import time
import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal synchronous websocket frame encoder/decoder (ไม่ต้องพึ่ง library)
# ---------------------------------------------------------------------------

def _ws_handshake(sock: ssl.SSLSocket, host: str, path: str) -> None:
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Origin: https://thingsanalytic.com\r\n"
        "\r\n"
    )
    sock.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed before completing handshake")
        resp += chunk
    if b"101 Switching Protocols" not in resp:
        first_line = resp.split(b"\r\n")[0].decode("utf-8", "replace")
        raise ConnectionError(f"Unexpected handshake response: {first_line}")


def _ws_send_text(sock: ssl.SSLSocket, text: str) -> None:
    payload = text.encode("utf-8")
    n = len(payload)
    mask_key = os.urandom(4)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    header = bytearray()
    header.append(0x81)  # FIN + opcode=text
    if n <= 125:
        header.append(0x80 | n)
    elif n <= 65535:
        header.append(0x80 | 126)
        header += struct.pack("!H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", n)
    header += mask_key
    sock.sendall(bytes(header) + masked)


def _ws_recv_close_check(sock: ssl.SSLSocket) -> bool:
    """Non-blocking check for incoming close/ping frame. Returns True if server closed."""
    import select
    r, _, _ = select.select([sock], [], [], 0)
    if not r:
        return False
    try:
        header = sock.recv(2)
        if len(header) < 2:
            return True
        opcode = header[0] & 0x0F
        if opcode == 0x8:
            return True
    except Exception:
        return True
    return False


# ---------------------------------------------------------------------------
# Geo helper: คำนวณ lat/lng ปลายทางจากจุดต้น + bearing + ระยะ (เมตร)
# ---------------------------------------------------------------------------
_EARTH_R = 6_371_000.0


def _destination_latlon(lat_deg: float, lng_deg: float, bearing_deg: float, dist_m: float):
    """Haversine destination point. Returns (lat, lng) degrees."""
    if dist_m is None or dist_m <= 0:
        return None, None
    lat = math.radians(lat_deg)
    lng = math.radians(lng_deg)
    ang = dist_m / _EARTH_R
    brng = math.radians(bearing_deg % 360)
    lat2 = math.asin(math.sin(lat) * math.cos(ang) + math.cos(lat) * math.sin(ang) * math.cos(brng))
    lng2 = lng + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat),
        math.cos(ang) - math.sin(lat) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 7), round(math.degrees(lng2), 7)


# ---------------------------------------------------------------------------
# Main publisher thread
# ---------------------------------------------------------------------------

class AntidroneWsExporter:
    """
    สร้างด้วย camera_geo และ cam_config แล้วเรียก start()
    ส่งข้อมูลผ่าน push(drone_export_list) — non-blocking

    Parameters
    ----------
    camera_geo:  dict จาก config.get_camera_geo()  { camera_id, site_lat, site_lng, heading_deg }
    cam_config:  dict จาก config.get_camera_config()  { fov_horizontal, ... }
    ws_url:      endpoint URL (wss://...)
    interval:    วินาทีระหว่างการส่ง (ควบคุม cadence)
    backoff_init/backoff_max: เวลา reconnect backoff (วินาที)
    enabled:     ถ้า False จะไม่เชื่อมต่อหรือส่งอะไรทั้งนั้น
    """

    def __init__(
        self,
        camera_geo: dict,
        cam_config: dict,
        ws_url: str,
        interval: float = 0.4,
        backoff_init: float = 2.0,
        backoff_max: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self._geo = camera_geo
        self._fov = float(cam_config.get("fov_horizontal", 60.0))
        self._url = ws_url
        self._interval = interval
        self._backoff_init = backoff_init
        self._backoff_max = backoff_max
        self._enabled = enabled
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_send_time: float = 0.0
        self._connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._enabled:
            logger.info("AntidroneWsExporter disabled — not starting thread")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="antidrone-ws-exporter", daemon=True
        )
        self._thread.start()
        logger.info("AntidroneWsExporter started → %s", self._url)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def push(self, drone_records: list[dict]) -> None:
        """Non-blocking: ส่งรายการโดรนเข้า queue; ถ้า queue เต็มให้ทิ้ง payload เก่า"""
        if not self._enabled:
            return
        payload = self._build_payload(drone_records)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_payload(self, drone_records: list[dict]) -> str:
        geo = self._geo
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        data: dict[str, Any] = {
            "lat": geo.get("site_lat", 0.0),
            "lng": geo.get("site_lng", 0.0),
            "heading": geo.get("heading_deg", 0.0),
            "fov": self._fov,
            "timestamp": ts,
            "drones": drone_records,
        }
        return json.dumps(data, ensure_ascii=False)

    def _parse_url(self):
        url = self._url
        if url.startswith("wss://"):
            host_path = url[len("wss://"):]
            use_tls = True
        elif url.startswith("ws://"):
            host_path = url[len("ws://"):]
            use_tls = False
        else:
            raise ValueError(f"Unsupported scheme in URL: {url}")
        if "/" in host_path:
            host, path = host_path.split("/", 1)
            path = "/" + path
        else:
            host = host_path
            path = "/"
        port = 443 if use_tls else 80
        if ":" in host:
            host, port_str = host.rsplit(":", 1)
            port = int(port_str)
        return host, port, path, use_tls

    def _connect(self) -> ssl.SSLSocket | socket.socket:
        host, port, path, use_tls = self._parse_url()
        raw = socket.create_connection((host, port), timeout=10)
        if use_tls:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(raw, server_hostname=host)
        else:
            sock = raw
        sock.settimeout(10)
        _ws_handshake(sock, host, path)
        self._connected = True
        logger.info("AntidroneWsExporter connected to %s", self._url)
        return sock

    def _run(self) -> None:
        backoff = self._backoff_init
        sock = None
        while not self._stop_event.is_set():
            # ---- connect ------------------------------------------------
            try:
                sock = self._connect()
                backoff = self._backoff_init
            except Exception as exc:
                self._connected = False
                logger.warning("AntidroneWsExporter connect failed: %s — retry in %.1fs", exc, backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, self._backoff_max)
                continue

            # ---- send loop ----------------------------------------------
            while not self._stop_event.is_set():
                try:
                    payload = self._queue.get(timeout=self._interval)
                except queue.Empty:
                    payload = None

                # cadence throttle
                now = time.monotonic()
                elapsed = now - self._last_send_time
                if payload is not None and elapsed < self._interval:
                    time.sleep(self._interval - elapsed)

                if payload is None:
                    continue

                try:
                    if _ws_recv_close_check(sock):
                        raise ConnectionError("Server sent close frame")
                    _ws_send_text(sock, payload)
                    self._last_send_time = time.monotonic()
                except Exception as exc:
                    self._connected = False
                    logger.warning("AntidroneWsExporter send failed: %s — reconnecting", exc)
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                    break

        self._connected = False
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        logger.info("AntidroneWsExporter stopped")


# ---------------------------------------------------------------------------
# Helper: สร้าง drone record 1 ตัวจากข้อมูลที่มี
# ---------------------------------------------------------------------------

def build_drone_record(
    order: int,
    cx: int,
    cy: int,
    frame_w: int,
    frame_h: int,
    cam_geo: dict,
    cam_config: dict,
    distance_m: float | None = None,
) -> dict:
    """
    สร้าง record โดรน 1 ตัวสำหรับใส่ใน drones list

    Parameters
    ----------
    order      : ลำดับปัจจุบัน (1-based)
    cx, cy     : pixel center ของ bbox
    frame_w/h  : ขนาด frame จริง (ก่อน display resize)
    cam_geo    : dict จาก get_camera_geo()
    cam_config : dict จาก get_camera_config()
    distance_m : ระยะถึงโดรน (เมตร) จาก estimate_distance_m() หรือ None
    """
    fov_h = float(cam_config.get("fov_horizontal", 60.0))
    heading = float(cam_geo.get("heading_deg", 0.0))
    site_lat = cam_geo.get("site_lat", 0.0)
    site_lng = cam_geo.get("site_lng", 0.0)

    # มุมเบี่ยงจากกึ่งกลางภาพ
    offset_x = cx - frame_w / 2.0
    angle_x_deg = (offset_x / frame_w) * fov_h if frame_w > 0 else 0.0
    bearing = (heading + angle_x_deg) % 360

    drone_lat, drone_lng = None, None
    if distance_m is not None and distance_m > 0:
        drone_lat, drone_lng = _destination_latlon(site_lat, site_lng, bearing, distance_m)

    return {
        "drone_id": f"DRONE_{order}",
        "lat": drone_lat,
        "lng": drone_lng,
        "distance": round(distance_m, 1) if distance_m is not None else None,
        "altitude": None,
        "speed": None,
    }

