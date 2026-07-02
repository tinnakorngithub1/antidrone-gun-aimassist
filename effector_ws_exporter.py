"""
effector_ws_exporter.py
=======================
Websocket publisher สำหรับส่ง telemetry แขน effector ไป C2 dashboard.

Schema (device → C2):
    {
        "lat": float,
        "lng": float,
        "heading": float,
        "fov": float,
        "lock_state": "searching" | "acquiring" | "locked",
        "lock_progress": float,
        "target_id": str | null,
        "target_lat": float | null,
        "target_lng": float | null,
        "timestamp": str,
    }
"""

from __future__ import annotations

import datetime
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
from typing import Any

logger = logging.getLogger(__name__)

_EARTH_R = 6_371_000.0

try:
    from joystick_cam4_controller import MODE_AUTO, MODE_LOCK, MODE_MANUAL, MODE_SAFE
except ImportError:
    MODE_AUTO = 0
    MODE_MANUAL = 1
    MODE_SAFE = 2
    MODE_LOCK = 3


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
    header.append(0x81)
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


def _destination_latlon(lat_deg: float, lng_deg: float, bearing_deg: float, dist_m: float):
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


def format_target_id(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower().startswith("uav_"):
        return s
    try:
        n = int(s)
        return f"uav_{n:03d}"
    except ValueError:
        return s


def angular_diff(a: float, b: float) -> float:
    d = (float(a) - float(b)) % 360.0
    if d > 180.0:
        d -= 360.0
    return abs(d)


def effector_heading(pos_x: float, home_arm_pan: float, geo: dict) -> float:
    return (float(geo.get("heading_deg", 0.0)) + (float(pos_x) - float(home_arm_pan))) % 360.0


def cue_target_bearing(cue: dict, cam8_geo: dict, fov_h: float) -> float | None:
    cx = cue.get("cx")
    fw = cue.get("frame_w")
    if cx is None or fw is None or float(fw) <= 0:
        return None
    heading = float(cam8_geo.get("heading_deg", 0.0))
    offset_x = float(cx) - float(fw) / 2.0
    angle_x_deg = (offset_x / float(fw)) * float(fov_h)
    return (heading + angle_x_deg) % 360.0


def target_latlng_from_cue(
    cue: dict,
    cam8_geo: dict,
    fov_h: float,
    distance_m: float | None,
) -> tuple[float | None, float | None]:
    bearing = cue_target_bearing(cue, cam8_geo, fov_h)
    if bearing is None:
        return None, None
    dist = distance_m
    if dist is None:
        dist = cue.get("distance_m")
    if dist is None or float(dist) <= 0:
        return None, None
    return _destination_latlon(
        float(cam8_geo.get("site_lat", 0.0)),
        float(cam8_geo.get("site_lng", 0.0)),
        bearing,
        float(dist),
    )


def _cue_fresh(cue: dict | None, cue_ttl_ms: int) -> bool:
    if cue is None:
        return False
    age_ms = (time.time() - float(cue.get("recv_timestamp", 0.0))) * 1000.0
    ttl = int(cue.get("cue_ttl_ms", cue_ttl_ms))
    return age_ms <= ttl


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_effector_payload(
    pos_x: float,
    pos_y: float,
    home_arm_pan: float,
    last_target_pan: float | None,
    last_target_tilt: float | None,
    ready_to_fire: bool,
    arm_mode: int,
    latest_cue: dict | None,
    target_det,
    distance_m: float | None,
    lock_csrt_initialized: bool,
    lock_csrt_lost: bool,
    lock_csrt_smooth_px: float | None,
    lock_csrt_smooth_py: float | None,
    cx_frame: float,
    cy_frame: float,
    w: int,
    h: int,
    px_per_deg_x: float | None,
    px_per_deg_y: float | None,
    cue_ttl_ms: int,
    effector_geo: dict,
    cam8_geo: dict,
    fov_deg: float,
    cam8_fov_h: float | None = None,
) -> dict[str, Any]:
    """สร้าง effector telemetry dict ตาม mode-specific lock rules."""
    del pos_y, last_target_tilt  # tilt ไม่อยู่ใน bearing 2D ของเฟสนี้

    half_fov = float(fov_deg) / 2.0
    heading = effector_heading(pos_x, home_arm_pan, effector_geo)
    fov_h = float(cam8_fov_h) if cam8_fov_h is not None else 180.0

    lock_state = "searching"
    lock_progress = 0.0
    target_id = None
    target_lat = None
    target_lng = None

    cue_fresh = _cue_fresh(latest_cue, cue_ttl_ms)
    cue_bearing = None
    cue_in_cone = False
    if cue_fresh and latest_cue is not None:
        cue_bearing = cue_target_bearing(latest_cue, cam8_geo, fov_h)
        if cue_bearing is not None:
            cue_in_cone = angular_diff(heading, cue_bearing) <= half_fov

    lock_csrt_active = (
        bool(lock_csrt_initialized)
        and not bool(lock_csrt_lost)
        and lock_csrt_smooth_px is not None
        and lock_csrt_smooth_py is not None
    )

    if ready_to_fire:
        lock_state = "locked"
        lock_progress = 1.0
    elif arm_mode == MODE_AUTO and cue_fresh and cue_in_cone and cue_bearing is not None:
        lock_progress = _clamp(1.0 - angular_diff(heading, cue_bearing) / half_fov, 0.0, 1.0)
        lock_state = "acquiring" if lock_progress > 0 else "searching"
    elif arm_mode == MODE_AUTO and last_target_pan is not None:
        lock_progress = _clamp(1.0 - abs(float(pos_x) - float(last_target_pan)) / half_fov, 0.0, 1.0)
        lock_state = "acquiring" if lock_progress > 0 else "searching"
    elif arm_mode == MODE_LOCK and lock_csrt_active:
        if px_per_deg_x and px_per_deg_y and float(px_per_deg_x) > 0 and float(px_per_deg_y) > 0:
            angle_off = math.hypot(
                (float(lock_csrt_smooth_px) - float(cx_frame)) * float(px_per_deg_x),
                (float(lock_csrt_smooth_py) - float(cy_frame)) * float(px_per_deg_y),
            )
            lock_progress = _clamp(1.0 - angle_off / half_fov, 0.0, 1.0)
            lock_state = "acquiring" if lock_progress > 0 else "searching"
    else:
        lock_state = "searching"
        lock_progress = 0.0

    if lock_progress > 0:
        if latest_cue is not None:
            target_id = format_target_id(latest_cue.get("target_id"))
        cue_lat = latest_cue.get("target_lat") if latest_cue else None
        cue_lng = latest_cue.get("target_lng") if latest_cue else None
        if cue_lat is not None and cue_lng is not None:
            target_lat = float(cue_lat)
            target_lng = float(cue_lng)
        elif arm_mode == MODE_AUTO and cue_fresh and latest_cue is not None:
            target_lat, target_lng = target_latlng_from_cue(
                latest_cue, cam8_geo, fov_h, distance_m
            )
        elif arm_mode == MODE_LOCK and lock_csrt_active:
            dist = distance_m
            if dist is None and latest_cue is not None:
                dist = latest_cue.get("distance_m")
            if (
                dist is not None
                and float(dist) > 0
                and px_per_deg_x
                and px_per_deg_y
                and float(px_per_deg_x) > 0
                and float(px_per_deg_y) > 0
            ):
                off_x = (float(lock_csrt_smooth_px) - float(cx_frame)) / float(px_per_deg_x)
                off_y = (float(lock_csrt_smooth_py) - float(cy_frame)) / float(px_per_deg_y)
                bearing = (heading + off_x) % 360.0
                target_lat, target_lng = _destination_latlon(
                    float(effector_geo.get("site_lat", 0.0)),
                    float(effector_geo.get("site_lng", 0.0)),
                    bearing,
                    float(dist),
                )

    if lock_state == "searching" or lock_progress <= 0:
        target_id = None
        target_lat = None
        target_lng = None

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "lat": float(effector_geo.get("site_lat", 0.0)),
        "lng": float(effector_geo.get("site_lng", 0.0)),
        "heading": round(heading, 2),
        "fov": float(fov_deg),
        "lock_state": lock_state,
        "lock_progress": round(lock_progress, 4),
        "target_id": target_id,
        "target_lat": target_lat,
        "target_lng": target_lng,
        "timestamp": ts,
    }


class EffectorWsExporter:
    """Background WS publisher — push(dict) non-blocking."""

    def __init__(
        self,
        ws_url: str,
        interval: float = 0.4,
        backoff_init: float = 2.0,
        backoff_max: float = 30.0,
        enabled: bool = True,
        log_payload: bool = False,
    ) -> None:
        self._url = ws_url
        self._interval = interval
        self._backoff_init = backoff_init
        self._backoff_max = backoff_max
        self._enabled = enabled
        self._log_payload = log_payload
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_send_time: float = 0.0
        self._connected = False
        self._last_lock_state: str = "searching"

    def start(self) -> None:
        if not self._enabled:
            logger.info("EffectorWsExporter disabled — not starting thread")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="effector-ws-exporter", daemon=True
        )
        self._thread.start()
        logger.info("EffectorWsExporter started → %s", self._url)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def push(self, payload: dict) -> None:
        if not self._enabled:
            if self._log_payload:
                logger.info("EffectorWS (disabled): %s", json.dumps(payload, ensure_ascii=False))
            return
        self._last_lock_state = str(payload.get("lock_state", "searching"))
        text = json.dumps(payload, ensure_ascii=False)
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_lock_state(self) -> str:
        return self._last_lock_state

    def _parse_url(self):
        url = self._url
        if url.startswith("wss://"):
            host_path = url[len("wss://") :]
            use_tls = True
        elif url.startswith("ws://"):
            host_path = url[len("ws://") :]
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
        logger.info("EffectorWsExporter connected to %s", self._url)
        return sock

    def _run(self) -> None:
        backoff = self._backoff_init
        sock = None
        while not self._stop_event.is_set():
            try:
                sock = self._connect()
                backoff = self._backoff_init
            except Exception as exc:
                self._connected = False
                logger.warning(
                    "EffectorWsExporter connect failed: %s — retry in %.1fs", exc, backoff
                )
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, self._backoff_max)
                continue

            while not self._stop_event.is_set():
                try:
                    payload = self._queue.get(timeout=self._interval)
                except queue.Empty:
                    payload = None

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
                    logger.warning("EffectorWsExporter send failed: %s — reconnecting", exc)
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
        logger.info("EffectorWsExporter stopped")
