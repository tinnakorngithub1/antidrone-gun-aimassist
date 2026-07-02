"""
rtmp_display_streamer.py
────────────────────────
Push display frames (numpy BGR) to an RTMP server via ffmpeg subprocess.
Designed for Jetson Orin NX: uses h264_nvenc with JetPack-compatible preset
fallback chain (ll → p1 → fast → libx264).

Key design decisions:
- Rate-limiting is done in the caller (compositor loop), NOT here.
- submit_frame() is non-blocking; old frames are dropped when the queue is full.
- BrokenPipeError / OSError in the writer thread triggers reconnect.
- stderr is routed to DEVNULL by default; set debug=True to log error lines.
- Reconnect is handled in a daemon thread so stop() is never blocked.
"""

import os
import queue
import shutil
import subprocess
import threading
import time


_NVENC_PRESET_CANDIDATES = ["ll", "p1", "fast"]
_RECONNECT_SENTINEL = object()


def _build_nvenc_cmd(ffmpeg_bin, width, height, fps, bitrate, preset, url):
    bufsize = _double_bitrate(bitrate)
    gop = max(1, int(fps * 2))
    return [
        ffmpeg_bin,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-preset", preset,
        "-tune", "ull",
        "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bufsize,
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-an",
        "-f", "flv",
        url,
    ]


def _build_x264_cmd(ffmpeg_bin, width, height, fps, bitrate, url):
    gop = max(1, int(fps * 2))
    return [
        ffmpeg_bin,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-b:v", bitrate, "-maxrate", bitrate,
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-an",
        "-f", "flv",
        url,
    ]


def _double_bitrate(bitrate_str):
    """'2M' -> '4M', '1500k' -> '3000k', fallback doubles the string prefix."""
    s = bitrate_str.strip()
    try:
        if s[-1].upper() == "M":
            return f"{int(float(s[:-1]) * 2)}M"
        if s[-1].upper() == "K":
            return f"{int(float(s[:-1]) * 2)}k"
        return str(int(s) * 2)
    except Exception:
        return s


def _log(msg):
    print(f"[RTMP] {msg}", flush=True)


def _safe_log_url(url):
    """Strip password from URL for logging: pass=xxx → pass=***"""
    import re
    return re.sub(r"(pass=)[^&]+", r"\1***", url)


def _probe_nvenc_preset(ffmpeg_bin, width, height, fps, bitrate, url):
    """
    Try each NVENC preset with a 0-second null encode to find the first
    one supported by this ffmpeg/JetPack build.
    Returns (preset_name, cmd_list) or None if NVENC unavailable.
    """
    for preset in _NVENC_PRESET_CANDIDATES:
        probe_cmd = [
            ffmpeg_bin,
            "-f", "lavfi", "-i", "nullsrc=s=16x16:r=1",
            "-t", "0",
            "-c:v", "h264_nvenc",
            "-preset", preset,
            "-an",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0:
                cmd = _build_nvenc_cmd(ffmpeg_bin, width, height, fps, bitrate, preset, url)
                return preset, cmd
        except Exception:
            continue
    return None


class RtmpDisplayStreamer:
    """
    Streams BGR numpy frames to an RTMP URL via ffmpeg.

    Usage:
        streamer = RtmpDisplayStreamer(url="rtmp://...", fps=10)
        streamer.start(canvas_w, canvas_h)
        # in compositor loop (rate-limited by caller):
        streamer.submit_frame(canvas)
        # on shutdown:
        streamer.stop()
    """

    def __init__(
        self,
        url,
        fps=10.0,
        bitrate="2M",
        codec="auto",
        reconnect_max=5,
        reconnect_delay=5.0,
        debug=False,
    ):
        self._url = url
        self._fps = float(fps)
        self._bitrate = bitrate
        self._codec = codec
        self._reconnect_max = int(reconnect_max)
        self._reconnect_delay = float(reconnect_delay)
        self._debug = debug

        self._width = 0
        self._height = 0
        self._cmd = None

        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._reconnect_count = 0

        self._proc = None
        self._writer_thr = None
        self._stderr_thr = None
        self._reconnect_thr = None

        self._lock = threading.Lock()
        self._active = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self, width, height):
        """Build ffmpeg command, spawn process, start threads."""
        self._width = width
        self._height = height

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            _log("ffmpeg not found in PATH — RTMP streaming disabled")
            return

        cmd = self._resolve_cmd(ffmpeg_bin, width, height)
        if cmd is None:
            _log("No suitable encoder found — RTMP streaming disabled")
            return

        self._cmd = cmd
        if not self._spawn_ffmpeg():
            return

        self._active = True
        codec_name = "h264_nvenc" if "h264_nvenc" in cmd else "libx264"
        _log(
            f"publishing to {_safe_log_url(self._url)} "
            f"({self._width}x{self._height} @ {self._fps:.0f} fps, {codec_name})"
        )

    def submit_frame(self, bgr_frame):
        """Non-blocking. Drops oldest frame if queue is full. Thread-safe."""
        if not self._active:
            return
        frame_bytes = bgr_frame.tobytes()
        while True:
            try:
                self._frame_queue.put_nowait(frame_bytes)
                return
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    return

    def stop(self):
        """Signal stop, flush, join all threads gracefully."""
        self._stop_event.set()
        self._active = False
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        self._join_threads(timeout=6.0)
        self._kill_proc()
        _log("stopped")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_cmd(self, ffmpeg_bin, width, height):
        codec = self._codec.strip().lower()
        if codec == "h264_nvenc":
            result = _probe_nvenc_preset(ffmpeg_bin, width, height, self._fps, self._bitrate, self._url)
            if result:
                preset, cmd = result
                _log(f"using h264_nvenc preset={preset}")
                return cmd
            _log("h264_nvenc unavailable, falling back to libx264")
            return _build_x264_cmd(ffmpeg_bin, width, height, self._fps, self._bitrate, self._url)
        if codec == "libx264":
            return _build_x264_cmd(ffmpeg_bin, width, height, self._fps, self._bitrate, self._url)
        # auto: try NVENC first
        result = _probe_nvenc_preset(ffmpeg_bin, width, height, self._fps, self._bitrate, self._url)
        if result:
            preset, cmd = result
            _log(f"auto: using h264_nvenc preset={preset}")
            return cmd
        _log("auto: NVENC unavailable, using libx264")
        return _build_x264_cmd(ffmpeg_bin, width, height, self._fps, self._bitrate, self._url)

    def _spawn_ffmpeg(self):
        """Spawn ffmpeg subprocess and start writer + optional stderr thread."""
        stderr_pipe = subprocess.PIPE if self._debug else subprocess.DEVNULL
        try:
            proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_pipe,
            )
        except Exception as exc:
            _log(f"failed to spawn ffmpeg: {exc}")
            return False

        with self._lock:
            self._proc = proc

        self._writer_thr = threading.Thread(
            target=self._writer_thread, daemon=True, name="rtmp-writer"
        )
        self._writer_thr.start()

        if self._debug and proc.stderr is not None:
            self._stderr_thr = threading.Thread(
                target=self._stderr_reader_thread,
                args=(proc.stderr,),
                daemon=True,
                name="rtmp-stderr",
            )
            self._stderr_thr.start()

        return True

    def _writer_thread(self):
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            with self._lock:
                proc = self._proc
            if proc is None or proc.stdin is None:
                break
            try:
                proc.stdin.write(item)
            except (BrokenPipeError, OSError):
                _log("pipe broken — scheduling reconnect")
                self._schedule_reconnect()
                return

        # Flush stdin so ffmpeg can finalize
        with self._lock:
            proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def _stderr_reader_thread(self, stderr_handle):
        try:
            for raw_line in stderr_handle:
                if self._stop_event.is_set():
                    break
                try:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if "error" in line.lower():
                    _log(f"ffmpeg: {line}")
        except Exception:
            pass
        finally:
            try:
                stderr_handle.close()
            except Exception:
                pass

    def _schedule_reconnect(self):
        if self._stop_event.is_set():
            return
        thr = threading.Thread(
            target=self._reconnect_loop, daemon=True, name="rtmp-reconnect"
        )
        thr.start()
        self._reconnect_thr = thr

    def _reconnect_loop(self):
        while not self._stop_event.is_set():
            if self._reconnect_count >= self._reconnect_max:
                _log(
                    f"reconnect limit ({self._reconnect_max}) reached — "
                    "RTMP streaming stopped; detection continues normally"
                )
                self._active = False
                return

            self._reconnect_count += 1
            _log(
                f"reconnecting ({self._reconnect_count}/{self._reconnect_max}) "
                f"in {self._reconnect_delay:.0f}s ..."
            )
            # Wait before reconnect, but honour stop_event
            deadline = time.monotonic() + self._reconnect_delay
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    return
                time.sleep(0.2)

            self._kill_proc()

            # Drain stale frames
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            if not self._spawn_ffmpeg():
                _log("respawn failed — will retry")
                continue

            _log(f"reconnect {self._reconnect_count} succeeded")
            return

    def _kill_proc(self):
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def _join_threads(self, timeout=6.0):
        deadline = time.monotonic() + timeout
        for thr in (self._writer_thr, self._stderr_thr, self._reconnect_thr):
            if thr is None or not thr.is_alive():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thr.join(timeout=remaining)
