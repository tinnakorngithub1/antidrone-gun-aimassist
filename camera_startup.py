"""
Camera startup helpers — wait for first frame, retry open, graceful shutdown.
ใช้กับ GStreamer/RTSP บน Jetson หลัง USB glitch หรือ process เก่าค้าง.
"""

from __future__ import annotations

import atexit
import signal
import time
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def get_camera_startup_wait_sec(camera_name: Optional[str] = None) -> float:
    try:
        import config
        from config import ACTIVE_CAMERA, get_camera_config

        override = float(getattr(config, "CAMERA_STARTUP_WAIT_SEC", 0.0) or 0.0)
        if override > 0:
            return override
        name = camera_name or ACTIVE_CAMERA
        cfg = get_camera_config(name)
        w = int(cfg.get("width", 1280))
        if w >= 3840:
            return float(getattr(config, "CAMERA_STARTUP_WAIT_4K_SEC", 30.0))
        if w >= 2560:
            return float(getattr(config, "CAMERA_STARTUP_WAIT_2K_SEC", 15.0))
        return float(getattr(config, "CAMERA_STARTUP_WAIT_HD_SEC", 8.0))
    except Exception:
        return 15.0


def wait_for_camera_first_frame(cam, camera_name: Optional[str] = None):
    """Poll cam.read() until first frame or timeout. Returns frame or None."""
    try:
        import config

        poll = float(getattr(config, "CAMERA_STARTUP_POLL_SEC", 0.033))
    except Exception:
        poll = 0.033
    deadline = time.time() + get_camera_startup_wait_sec(camera_name)
    while time.time() < deadline:
        active, frame, _ = cam.read()
        if active and frame is not None:
            return frame
        time.sleep(poll)
    return None


def open_camera_with_retries(
    build_camera_fn: Callable[[Optional[str]], Any],
    camera_name: Optional[str] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    build → start → wait first frame; retry on failure.
    Returns (cam, first_frame). first_frame is None if all attempts failed.
    """
    try:
        import config

        retries = max(1, int(getattr(config, "CAMERA_STARTUP_RETRIES", 3)))
        retry_delay = float(getattr(config, "CAMERA_STARTUP_RETRY_DELAY_SEC", 2.0))
    except Exception:
        retries = 3
        retry_delay = 2.0

    last_cam = None
    wait_sec = get_camera_startup_wait_sec(camera_name)
    for attempt in range(1, retries + 1):
        cam = build_camera_fn(camera_name)
        last_cam = cam
        try:
            cam.start()
        except Exception as exc:
            print(f"⚠️ Camera start exception (attempt {attempt}/{retries}): {exc}")
            try:
                if hasattr(cam, "release"):
                    cam.release()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(retry_delay)
            continue

        frame = wait_for_camera_first_frame(cam, camera_name)
        if frame is not None:
            if attempt > 1:
                print(f"✅ Camera ready on attempt {attempt}/{retries} (wait≤{wait_sec:.0f}s)")
            return cam, frame

        health = getattr(cam, "get_stream_health", lambda: "?")()
        print(
            f"⚠️ Camera not ready (attempt {attempt}/{retries}, "
            f"waited {wait_sec:.0f}s, health={health}) — retrying..."
        )
        try:
            if hasattr(cam, "release"):
                cam.release()
        except Exception:
            pass
        if attempt < retries:
            time.sleep(retry_delay)

    return last_cam, None


class RuntimeCleanup:
    """Register SIGINT/SIGTERM/atexit handlers to release camera + arm."""

    def __init__(self) -> None:
        self._resources: Dict[str, Any] = {}
        self._cleaned = False
        self._installed = False

    def set_resources(self, **kwargs) -> None:
        self._resources.update(kwargs)

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True

        def _handler(signum=None, frame=None):
            self.cleanup()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass
        atexit.register(self.cleanup)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True

        arm = self._resources.get("arm_controller")
        if arm is not None:
            try:
                if hasattr(arm, "request_shutdown"):
                    arm.request_shutdown()
                if hasattr(arm, "disconnect"):
                    arm.disconnect()
            except Exception:
                pass

        for key in ("cam", "cam8_stream"):
            cam = self._resources.get(key)
            if cam is None:
                continue
            try:
                if hasattr(cam, "release"):
                    cam.release()
            except Exception:
                pass

        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


_runtime_cleanup = RuntimeCleanup()


def get_runtime_cleanup() -> RuntimeCleanup:
    return _runtime_cleanup


def get_stream_health(cam) -> str:
    if cam is None:
        return "failed"
    try:
        if hasattr(cam, "get_stream_health"):
            return str(cam.get_stream_health() or "unknown").strip().lower()
    except Exception:
        pass
    return "unknown"


def cam_hud_label_and_color(cam, active: bool = True, frame=None) -> Tuple[str, Tuple[int, int, int]]:
    """
    HUD label for live camera stream (like ARM:OK / JOY:OK).
    CAM:OK | CAM:WAIT (reconnecting) | CAM:DISC | CAM:FAIL
    """
    health = get_stream_health(cam)
    if health == "failed":
        return "CAM:FAIL", (0, 80, 255)
    if not active or frame is None:
        if health in ("reconnecting", "connecting"):
            return "CAM:WAIT", (0, 200, 255)
        if health == "ok":
            return "CAM:DISC", (0, 100, 255)
        return "CAM:DISC", (0, 100, 255)
    if health == "reconnecting":
        return "CAM:WAIT", (0, 200, 255)
    return "CAM:OK", (0, 220, 0)


def network_loss_placeholder(width, height, camera_name, status_text):
    """Full-frame placeholder while camera reconnects (GStreamer/RTSP drop)."""
    import numpy as np

    if cv2 is None:
        raise RuntimeError("opencv required for network_loss_placeholder")
    w = max(160, int(width))
    h = max(120, int(height))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 3)
    title = f"{camera_name} - CAMERA OFFLINE"
    subtitle = f"STATUS: {str(status_text).upper()}"
    hint = "Auto-reconnecting..."
    title_scale = max(0.55, min(1.2, w / 1300.0))
    subtitle_scale = max(0.5, min(1.0, w / 1400.0))
    hint_scale = max(0.45, min(0.85, w / 1500.0))
    title_y = max(40, int(h * 0.42))
    subtitle_y = max(70, int(h * 0.52))
    hint_y = max(100, int(h * 0.62))
    cv2.putText(frame, title, (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (20, subtitle_y), cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (0, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, hint, (20, hint_y), cv2.FONT_HERSHEY_SIMPLEX, hint_scale, (180, 180, 180), 1, cv2.LINE_AA)
    return frame
