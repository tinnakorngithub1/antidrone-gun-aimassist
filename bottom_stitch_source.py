#bottom_stitch_source.py

import copy
import json
import os
import queue
import sys
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

from canonical_path import stitched_point_to_source_camera as resolve_stitched_point


STANDALONE_LEFT_CAMERA = "cam7"
STANDALONE_RIGHT_CAMERA = "cam7"
STANDALONE_PRESET_PATH = "bottom_stitch.jetson"
STANDALONE_WINDOW_NAME = "Bottom Stitch Editor"
STANDALONE_MAX_WIDTH = 2560
STANDALONE_MAX_HEIGHT = 1080
STANDALONE_SCREEN_MARGIN_X = 80
STANDALONE_SCREEN_MARGIN_Y = 140
SHOW_STITCH_RUNTIME_HUD = False


def _resolve_local_path(path_like):
    if not path_like:
        return path_like
    if isinstance(path_like, str) and os.path.isabs(path_like):
        return path_like
    return os.path.join(os.path.dirname(__file__), str(path_like))


def _default_stitch_preset(left_camera, right_camera):
    return {
        "version": 1,
        "left_camera": left_camera,
        "right_camera": right_camera,
        "output_width": 0,
        "output_height": 0,
        "blend_width": 0,
        "left_offset": {"x": 0, "y": 0},
        "right_offset": {"x": 0, "y": 0},
        "left_crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "right_crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
    }


def load_stitch_preset(preset_path, left_camera, right_camera):
    preset = _default_stitch_preset(left_camera, right_camera)
    preset_path = _resolve_local_path(preset_path)
    loaded_from_disk = False
    if preset_path and os.path.exists(preset_path):
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                preset.update(payload)
                loaded_from_disk = True
        except Exception as exc:
            print(f"⚠️ Failed to load stitch preset '{preset_path}': {exc}")
    preset["left_camera"] = left_camera
    preset["right_camera"] = right_camera
    return preset, loaded_from_disk


def save_stitch_preset(preset_path, preset):
    preset_path = _resolve_local_path(preset_path)
    if not preset_path:
        return
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2)


def _normalize_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _clamp_crop(crop, frame_shape):
    h, w = frame_shape[:2]
    left = max(0, int(crop.get("left", 0)))
    top = max(0, int(crop.get("top", 0)))
    right = max(0, int(crop.get("right", 0)))
    bottom = max(0, int(crop.get("bottom", 0)))
    max_x = max(0, w - 2)
    max_y = max(0, h - 2)
    left = min(left, max_x)
    right = min(right, max(0, w - left - 1))
    top = min(top, max_y)
    bottom = min(bottom, max(0, h - top - 1))
    return {"left": left, "top": top, "right": right, "bottom": bottom}


class StitchedCameraStream:
    HANDLE_RADIUS = 24
    FRAME_REUSE_GRACE_SECONDS = 1.0

    def __init__(
        self,
        base_module,
        left_camera,
        right_camera,
        preset_path,
        clone_right_from_left=False,
    ):
        self.base_module = base_module
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.preset_path = _resolve_local_path(preset_path)
        self.clone_right_from_left = bool(clone_right_from_left)
        if self.left_camera == self.right_camera and not self.clone_right_from_left:
            # A single RTSP source often refuses two concurrent connections.
            self.clone_right_from_left = True
            print(
                f"[bottom_stitch] left/right both use '{self.left_camera}', "
                "enabling clone_right_from_left automatically"
            )
        self.left_stream = None
        self.right_stream = None
        self.last_layout = {}
        self.last_canvas_shape = None
        self.last_display_shape = None
        self.last_source_shapes = {}
        self.edit_mode = False
        self.selected_target = "right"
        self.edit_action = "move"
        self.drag_state = None
        self.awaiting_mouse_rearm = False
        self._fps_window_seconds = 1.5
        self._left_frame_times = deque(maxlen=240)
        self._right_frame_times = deque(maxlen=240)
        self._stitched_frame_times = deque(maxlen=240)
        self._last_left_timestamp = None
        self._last_right_timestamp = None
        self._last_left_frame = None
        self._last_right_frame = None
        self._last_left_seen_at = 0.0
        self._last_right_seen_at = 0.0
        self._last_fps_console_log_at = 0.0
        self.preset, self.loaded_from_disk = load_stitch_preset(
            self.preset_path, left_camera, right_camera
        )
        self._loaded_snapshot = copy.deepcopy(self.preset)
        self._auto_initialized = False
        self.synthetic_camera_name = getattr(base_module, "LOCAL_ACTIVE_CAMERA", None)

    def _build_stream(self, camera_name):
        cfg = self.base_module.LOCAL_CAMERAS[camera_name]
        source = cfg["video_filename"] if cfg.get("use_video_file", False) else cfg["rtsp_url"]
        if cfg.get("use_video_file", False) and isinstance(source, str) and source and not os.path.isabs(source):
            source = os.path.join(os.path.dirname(self.base_module.__file__), source)
        return self.base_module.CameraStream(
            source=source,
            width=cfg["width"],
            height=cfg["height"],
            use_video_file=cfg.get("use_video_file", False),
            camera_name=cfg.get("name", camera_name),
            udp_ip=cfg.get("udp_ip"),
            udp_port=cfg.get("udp_port"),
            use_udp_direct=cfg.get("use_udp_direct", False),
            stream_format=cfg.get("stream_format", "h264"),
        )

    def start(self):
        self.left_stream = self._build_stream(self.left_camera)
        self.left_stream.start()
        if self.clone_right_from_left:
            self.right_stream = None
        else:
            self.right_stream = self._build_stream(self.right_camera)
            self.right_stream.start()
        return self

    def release(self):
        if self.left_stream is not None and hasattr(self.left_stream, "release"):
            self.left_stream.release()
        if self.right_stream is not None and hasattr(self.right_stream, "release"):
            self.right_stream.release()

    def _read_stream(self, stream):
        if stream is None:
            return False, None, None
        active, frame, frame_timestamp = stream.read()
        return bool(active and frame is not None), _normalize_bgr(frame), frame_timestamp

    def _read_stream_with_cache(self, side, stream):
        active, frame, frame_timestamp = self._read_stream(stream)
        now = time.time()
        if active and frame is not None:
            if side == "left":
                self._last_left_frame = frame.copy()
                self._last_left_seen_at = now
            else:
                self._last_right_frame = frame.copy()
                self._last_right_seen_at = now
            return True, frame, frame_timestamp

        if side == "left":
            cached_frame = self._last_left_frame
            seen_at = self._last_left_seen_at
        else:
            cached_frame = self._last_right_frame
            seen_at = self._last_right_seen_at

        if cached_frame is not None and (now - seen_at) <= self.FRAME_REUSE_GRACE_SECONDS:
            return True, cached_frame.copy(), frame_timestamp
        return False, None, frame_timestamp

    def _record_input_timestamp(self, side, frame_timestamp):
        if frame_timestamp is None:
            return
        if side == "left":
            if frame_timestamp == self._last_left_timestamp:
                return
            self._last_left_timestamp = frame_timestamp
            self._append_fps_timestamp(self._left_frame_times, frame_timestamp)
            return
        if frame_timestamp == self._last_right_timestamp:
            return
        self._last_right_timestamp = frame_timestamp
        self._append_fps_timestamp(self._right_frame_times, frame_timestamp)

    def _append_fps_timestamp(self, timestamp_store, timestamp_value):
        timestamp_store.append(float(timestamp_value))
        cutoff = float(timestamp_value) - self._fps_window_seconds
        while len(timestamp_store) >= 2 and timestamp_store[0] < cutoff:
            timestamp_store.popleft()

    def _fps_value(self, timestamp_store):
        if len(timestamp_store) < 2:
            return 0.0
        duration = float(timestamp_store[-1] - timestamp_store[0])
        if duration <= 0:
            return 0.0
        return (len(timestamp_store) - 1) / duration

    def _fps_text(self, image_shape):
        return (
            f"INPUT L: {self._fps_value(self._left_frame_times):.1f} fps"
            f"  INPUT R: {self._fps_value(self._right_frame_times):.1f} fps"
            f"  STITCH OUT: {self._fps_value(self._stitched_frame_times):.1f} fps"
            f"  OUT: {image_shape[1]}x{image_shape[0]}"
        )

    def _maybe_log_fps(self):
        now = time.time()
        if now - self._last_fps_console_log_at < 1.0:
            return
        self._last_fps_console_log_at = now
        print(
            "[bottom_stitch_fps] "
            f"L={self._fps_value(self._left_frame_times):.1f} "
            f"R={self._fps_value(self._right_frame_times):.1f} "
            f"OUT={self._fps_value(self._stitched_frame_times):.1f}"
        )

    def _apply_crop(self, frame, crop):
        crop = _clamp_crop(crop, frame.shape)
        h, w = frame.shape[:2]
        x1 = crop["left"]
        y1 = crop["top"]
        x2 = max(x1 + 1, w - crop["right"])
        y2 = max(y1 + 1, h - crop["bottom"])
        return frame[y1:y2, x1:x2].copy(), crop

    def _resize_to_height(self, image, target_height):
        current_h, current_w = image.shape[:2]
        target_height = max(1, int(target_height))
        if current_h == target_height:
            return image
        target_width = max(1, int(round(current_w * (target_height / float(max(1, current_h))))))
        interpolation = cv2.INTER_AREA if target_height < current_h else cv2.INTER_LINEAR
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    def _normalize_pair_height(self, left_img, right_img):
        left_h = left_img.shape[0]
        right_h = right_img.shape[0]
        target_height = max(1, min(left_h, right_h))
        left_norm = self._resize_to_height(left_img, target_height)
        right_norm = self._resize_to_height(right_img, target_height)
        return left_norm, right_norm

    def _ensure_defaults(self, left_frame, right_frame):
        if self._auto_initialized:
            return
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_frame.shape[:2]
        if not self.loaded_from_disk:
            self.preset["right_offset"]["x"] = left_w
            self.preset["right_offset"]["y"] = 0
            self.preset["output_width"] = left_w + right_w
            self.preset["output_height"] = max(left_h, right_h)
        else:
            if int(self.preset.get("output_width", 0)) <= 0:
                self.preset["output_width"] = left_w + right_w
            if int(self.preset.get("output_height", 0)) <= 0:
                self.preset["output_height"] = max(left_h, right_h)
        self._loaded_snapshot = copy.deepcopy(self.preset)
        self._auto_initialized = True

    def _compute_paste_rect(self, image, pos_x, pos_y, canvas_shape):
        h, w = image.shape[:2]
        canvas_h, canvas_w = canvas_shape[:2]
        x1 = max(0, int(pos_x))
        y1 = max(0, int(pos_y))
        x2 = min(canvas_w, int(pos_x + w))
        y2 = min(canvas_h, int(pos_y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        src_x1 = x1 - int(pos_x)
        src_y1 = y1 - int(pos_y)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        return {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "src_x1": src_x1,
            "src_y1": src_y1,
            "src_x2": src_x2,
            "src_y2": src_y2,
        }

    def _resolve_runtime_canvas(self, left_img, right_img, left_offset, right_offset, preset_width, preset_height):
        left_x = int(left_offset.get("x", 0))
        left_y = int(left_offset.get("y", 0))
        right_x = int(right_offset.get("x", 0))
        right_y = int(right_offset.get("y", 0))

        min_x = min(0, left_x, right_x)
        min_y = min(0, left_y, right_y)
        max_x = max(
            int(max(0, preset_width)),
            left_x + int(left_img.shape[1]),
            right_x + int(right_img.shape[1]),
        )
        max_y = max(
            int(max(0, preset_height)),
            left_y + int(left_img.shape[0]),
            right_y + int(right_img.shape[0]),
        )

        shift_x = -min_x if min_x < 0 else 0
        shift_y = -min_y if min_y < 0 else 0
        canvas_w = max(1, int(max_x + shift_x))
        canvas_h = max(1, int(max_y + shift_y))
        return {
            "canvas_shape": (canvas_h, canvas_w, 3),
            "left_pos": (left_x + shift_x, left_y + shift_y),
            "right_pos": (right_x + shift_x, right_y + shift_y),
        }

    def _update_runtime_camera_shape(self, width, height):
        try:
            camera_name = self.synthetic_camera_name or getattr(self.base_module, "LOCAL_ACTIVE_CAMERA", None)
            camera_registry = getattr(self.base_module, "LOCAL_CAMERAS", None)
            if camera_name and isinstance(camera_registry, dict) and camera_name in camera_registry:
                camera_registry[camera_name]["width"] = int(width)
                camera_registry[camera_name]["height"] = int(height)
        except Exception:
            pass

    def _blend_alpha_cols(self, overlap_width, blend_width):
        if overlap_width <= 0:
            return None
        feather = overlap_width
        if int(blend_width) > 0:
            feather = max(1, min(int(blend_width), overlap_width))
        if feather <= 1:
            return np.full(overlap_width, 0.5, dtype=np.float32)
        base = np.linspace(0.0, 1.0, feather, dtype=np.float32)
        if overlap_width > feather:
            return np.concatenate([base, np.ones(overlap_width - feather, dtype=np.float32)])
        return base[:overlap_width]

    def _paste_direct(self, canvas, image, rect):
        if rect is None:
            return None
        canvas[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"]] = image[
            rect["src_y1"]:rect["src_y2"], rect["src_x1"]:rect["src_x2"]
        ]
        return {"x": rect["x"], "y": rect["y"], "w": rect["w"], "h": rect["h"]}

    def _paste_with_overlap_blend(self, canvas, image, rect, overlap_rect, blend_width):
        if rect is None:
            return None

        dst_region = canvas[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"]]
        src_region = image[rect["src_y1"]:rect["src_y2"], rect["src_x1"]:rect["src_x2"]]

        if overlap_rect is None or int(blend_width) <= 0:
            dst_region[:] = src_region
            return {"x": rect["x"], "y": rect["y"], "w": rect["w"], "h": rect["h"]}

        overlap_x1 = overlap_rect["x"] - rect["x"]
        overlap_y1 = overlap_rect["y"] - rect["y"]
        overlap_x2 = overlap_x1 + overlap_rect["w"]
        overlap_y2 = overlap_y1 + overlap_rect["h"]

        dst_overlap_before = dst_region[overlap_y1:overlap_y2, overlap_x1:overlap_x2].copy()
        dst_region[:] = src_region

        overlap_width = overlap_rect["w"]
        alpha_cols = self._blend_alpha_cols(overlap_width, blend_width)
        if alpha_cols is None:
            return {"x": rect["x"], "y": rect["y"], "w": rect["w"], "h": rect["h"]}

        src_overlap = src_region[overlap_y1:overlap_y2, overlap_x1:overlap_x2].astype(np.float32)
        dst_overlap_before = dst_overlap_before.astype(np.float32)
        alpha = alpha_cols.reshape(1, overlap_width, 1)
        blended = (dst_overlap_before * (1.0 - alpha)) + (src_overlap * alpha)
        dst_region[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return {"x": rect["x"], "y": rect["y"], "w": rect["w"], "h": rect["h"]}

    def _paste(self, canvas, coverage, image, pos_x, pos_y, blend_width):
        h, w = image.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        x1 = max(0, int(pos_x))
        y1 = max(0, int(pos_y))
        x2 = min(canvas_w, int(pos_x + w))
        y2 = min(canvas_h, int(pos_y + h))
        if x2 <= x1 or y2 <= y1:
            return None

        src_x1 = x1 - int(pos_x)
        src_y1 = y1 - int(pos_y)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        src_region = image[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        dst_region = canvas[y1:y2, x1:x2]
        cov_region = coverage[y1:y2, x1:x2]

        empty_mask = cov_region == 0
        if np.any(empty_mask):
            dst_region[empty_mask] = src_region[empty_mask]
            cov_region[empty_mask] = 1

        overlap_mask = (cov_region > 0) & (~empty_mask)
        if np.any(overlap_mask):
            overlap_cols = np.where(np.any(overlap_mask, axis=0))[0]
            if overlap_cols.size > 0:
                overlap_width = int(overlap_cols[-1] - overlap_cols[0] + 1)
                feather = overlap_width
                if int(blend_width) > 0:
                    feather = max(1, min(int(blend_width), overlap_width))
                if feather <= 1:
                    alpha_cols = np.full(overlap_width, 0.5, dtype=np.float32)
                else:
                    base = np.linspace(0.0, 1.0, feather, dtype=np.float32)
                    if overlap_width > feather:
                        alpha_cols = np.concatenate([base, np.ones(overlap_width - feather, dtype=np.float32)])
                    else:
                        alpha_cols = base[:overlap_width]
                for idx, col in enumerate(range(int(overlap_cols[0]), int(overlap_cols[-1]) + 1)):
                    col_mask = overlap_mask[:, col]
                    if not np.any(col_mask):
                        continue
                    alpha = float(alpha_cols[idx])
                    dst_region[col_mask, col] = (
                        dst_region[col_mask, col] * (1.0 - alpha)
                        + src_region[col_mask, col] * alpha
                    )
            cov_region[:, :] = 1

        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

    def compose_frames(self, left_frame, right_frame):
        left_crop = self.preset.get("left_crop", {})
        right_crop = self.preset.get("right_crop", {})
        left_img, left_crop = self._apply_crop(left_frame, left_crop)
        right_img, right_crop = self._apply_crop(right_frame, right_crop)
        left_img, right_img = self._normalize_pair_height(left_img, right_img)
        self._ensure_defaults(left_img, right_img)
        self.preset["left_crop"] = left_crop
        self.preset["right_crop"] = right_crop

        left_offset = self.preset.get("left_offset", {"x": 0, "y": 0})
        right_offset = self.preset.get("right_offset", {"x": 0, "y": 0})
        output_width = max(1, int(self.preset.get("output_width", 0)))
        output_height = max(1, int(self.preset.get("output_height", 0)))
        if output_width <= 1:
            output_width = left_img.shape[1] + right_img.shape[1]
        if output_height <= 1:
            output_height = max(left_img.shape[0], right_img.shape[0])
        blend_width = int(self.preset.get("blend_width", 0))
        # Runtime should behave like a single stitched camera source, so disable seam blending
        # unless the user is actively editing alignment.
        effective_blend_width = blend_width if self.edit_mode else 0

        runtime_canvas = self._resolve_runtime_canvas(
            left_img,
            right_img,
            left_offset,
            right_offset,
            output_width,
            output_height,
        )
        canvas_shape = runtime_canvas["canvas_shape"]
        left_pos_x, left_pos_y = runtime_canvas["left_pos"]
        right_pos_x, right_pos_y = runtime_canvas["right_pos"]
        left_rect_data = self._compute_paste_rect(
            left_img,
            left_pos_x,
            left_pos_y,
            canvas_shape,
        )
        right_rect_data = self._compute_paste_rect(
            right_img,
            right_pos_x,
            right_pos_y,
            canvas_shape,
        )

        left_rect = None if left_rect_data is None else {
            "x": left_rect_data["x"],
            "y": left_rect_data["y"],
            "w": left_rect_data["w"],
            "h": left_rect_data["h"],
        }
        right_rect = None if right_rect_data is None else {
            "x": right_rect_data["x"],
            "y": right_rect_data["y"],
            "w": right_rect_data["w"],
            "h": right_rect_data["h"],
        }

        # Fast path: when the two cropped images sit side-by-side with no overlap or blending.
        if (
            effective_blend_width <= 0
            and left_rect_data is not None
            and right_rect_data is not None
            and left_rect_data["y"] == 0
            and right_rect_data["y"] == 0
            and left_rect_data["h"] == canvas_shape[0]
            and right_rect_data["h"] == canvas_shape[0]
            and left_rect_data["x"] == 0
            and right_rect_data["x"] == left_rect_data["w"]
            and canvas_shape[1] == left_rect_data["w"] + right_rect_data["w"]
        ):
            stitched = np.hstack(
                [
                    left_img[left_rect_data["src_y1"]:left_rect_data["src_y2"], left_rect_data["src_x1"]:left_rect_data["src_x2"]],
                    right_img[right_rect_data["src_y1"]:right_rect_data["src_y2"], right_rect_data["src_x1"]:right_rect_data["src_x2"]],
                ]
            )
            self.last_layout = {"left": left_rect, "right": right_rect}
            self.last_canvas_shape = (stitched.shape[0], stitched.shape[1])
            self.last_source_shapes = {
                "left": (left_img.shape[0], left_img.shape[1]),
                "right": (right_img.shape[0], right_img.shape[1]),
            }
            self._update_runtime_camera_shape(stitched.shape[1], stitched.shape[0])
            return stitched

        canvas = np.zeros(canvas_shape, dtype=np.uint8)
        self._paste_direct(canvas, left_img, left_rect_data)
        overlap_rect = None
        if left_rect is not None and right_rect is not None:
            overlap_rect = {
                "x": max(left_rect["x"], right_rect["x"]),
                "y": max(left_rect["y"], right_rect["y"]),
                "w": min(left_rect["x"] + left_rect["w"], right_rect["x"] + right_rect["w"]) - max(left_rect["x"], right_rect["x"]),
                "h": min(left_rect["y"] + left_rect["h"], right_rect["y"] + right_rect["h"]) - max(left_rect["y"], right_rect["y"]),
            }
            if overlap_rect["w"] <= 0 or overlap_rect["h"] <= 0:
                overlap_rect = None
        self._paste_with_overlap_blend(canvas, right_img, right_rect_data, overlap_rect, effective_blend_width)

        self.last_layout = {"left": left_rect, "right": right_rect}
        self.last_canvas_shape = (canvas_shape[0], canvas_shape[1])
        self.last_source_shapes = {
            "left": (left_img.shape[0], left_img.shape[1]),
            "right": (right_img.shape[0], right_img.shape[1]),
        }
        self._update_runtime_camera_shape(canvas_shape[1], canvas_shape[0])
        return canvas

    def read(self):
        left_active, left_frame, left_timestamp = self._read_stream_with_cache("left", self.left_stream)
        if not left_active or left_frame is None:
            return False, None, None
        self._record_input_timestamp("left", left_timestamp)
        if self.clone_right_from_left:
            right_active = True
            right_frame = left_frame.copy()
            right_timestamp = left_timestamp
        else:
            right_active, right_frame, right_timestamp = self._read_stream_with_cache("right", self.right_stream)
        if not right_active or right_frame is None:
            return False, None, None
        self._record_input_timestamp("right", right_timestamp)
        stitched_frame = self.compose_frames(left_frame, right_frame)
        self._append_fps_timestamp(self._stitched_frame_times, time.time())
        return True, stitched_frame, None

    def _get_display_shape(self, module=None, image=None):
        if image is not None:
            return image.shape[:2]
        if self.last_display_shape is not None:
            return self.last_display_shape
        canvas_h, canvas_w = self.last_canvas_shape if self.last_canvas_shape else (1, 1)
        if module is None:
            return (canvas_h, canvas_w)
        display_w = max(1, int(getattr(module, "display_w", canvas_w)))
        display_h = max(1, int(getattr(module, "display_h", canvas_h)))
        return (display_h, display_w)

    def _canvas_point_from_display(self, module, x, y):
        canvas_h, canvas_w = self.last_canvas_shape if self.last_canvas_shape else (1, 1)
        display_h, display_w = self._get_display_shape(module)
        max_x = max(0, display_w - 1)
        max_y = max(0, display_h - 1)
        x = min(max(0, int(x)), max_x)
        y = min(max(0, int(y)), max_y)
        return (
            int(round(x * canvas_w / float(max(1, display_w)))),
            int(round(y * canvas_h / float(max(1, display_h)))),
        )

    def _point_in_rect(self, px, py, rect):
        if not rect:
            return False
        return rect["x"] <= px <= rect["x"] + rect["w"] and rect["y"] <= py <= rect["y"] + rect["h"]

    def _iter_target_priority(self):
        if self.selected_target == "left":
            return ("left", "right")
        if self.selected_target == "right":
            return ("right", "left")
        return ("right", "left")

    def _display_rects(self, module):
        display_h, display_w = self._get_display_shape(module)
        display_shape = (display_h, display_w, 3)
        return {
            "left": self._scale_rect(self.last_layout.get("left"), display_shape),
            "right": self._scale_rect(self.last_layout.get("right"), display_shape),
            "seam": self._scale_rect(self._overlap_rect(), display_shape),
        }

    def _display_handle_radius(self, module):
        canvas_h, canvas_w = self.last_canvas_shape if self.last_canvas_shape else (1, 1)
        display_h, display_w = self._get_display_shape(module)
        sx = display_w / float(max(1, canvas_w))
        sy = display_h / float(max(1, canvas_h))
        return max(10, int(round(self.HANDLE_RADIUS * max(sx, sy))))

    def _point_near_rect(self, px, py, rect, pad=0):
        if not rect:
            return False
        return (
            rect["x"] - pad <= px <= rect["x"] + rect["w"] + pad
            and rect["y"] - pad <= py <= rect["y"] + rect["h"] + pad
        )

    def _detect_display_edges(self, px, py, rect, radius):
        if not rect or not self._point_near_rect(px, py, rect, radius):
            return None
        edges = []
        if abs(px - rect["x"]) <= radius:
            edges.append("left")
        elif abs(px - (rect["x"] + rect["w"])) <= radius:
            edges.append("right")
        if abs(py - rect["y"]) <= radius:
            edges.append("top")
        elif abs(py - (rect["y"] + rect["h"])) <= radius:
            edges.append("bottom")
        return edges or None

    def _overlap_rect(self):
        left_rect = self.last_layout.get("left")
        right_rect = self.last_layout.get("right")
        if not left_rect or not right_rect:
            return None
        x1 = max(left_rect["x"], right_rect["x"])
        y1 = max(left_rect["y"], right_rect["y"])
        x2 = min(left_rect["x"] + left_rect["w"], right_rect["x"] + right_rect["w"])
        y2 = min(left_rect["y"] + left_rect["h"], right_rect["y"] + right_rect["h"])
        if x2 <= x1 or y2 <= y1:
            return None
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

    def resolve_source_point(self, x, y, preferred_source=None):
        payload = resolve_stitched_point(
            x_value=x,
            y_value=y,
            stitched_layout=self.last_layout,
            source_camera_names={
                "left": self.left_camera,
                "right": self.right_camera,
            },
            preferred_source=preferred_source,
        )
        if payload is None:
            return None
        resolved_region = payload.get("resolved_region")
        source_shape = self.last_source_shapes.get(resolved_region)
        if source_shape is not None:
            payload["source_frame_h"] = int(source_shape[0])
            payload["source_frame_w"] = int(source_shape[1])
        return payload

    def _pick_target_from_display(self, x, y, display_rects):
        if self._point_in_rect(x, y, display_rects.get("seam")):
            return "seam"
        for target in self._iter_target_priority():
            if self._point_in_rect(x, y, display_rects.get(target)):
                return target
        return None

    def _pick_mouse_hit(self, x, y, display_rects, handle_radius, target_order):
        for target in target_order:
            rect = display_rects.get(target)
            edges = self._detect_display_edges(x, y, rect, handle_radius)
            if edges is not None:
                return {"mode": "crop", "target": target, "edges": edges}
        if self._point_in_rect(x, y, display_rects.get("seam")):
            return {"mode": "blend", "target": "seam"}
        for target in target_order:
            rect = display_rects.get(target)
            if self._point_in_rect(x, y, rect):
                return {"mode": "move", "target": target}
        return None

    def _set_edit_mode(self, enabled, save_on_exit=False):
        enabled = bool(enabled)
        if self.edit_mode and not enabled and save_on_exit:
            self.save_preset()
        self.edit_mode = enabled
        self.drag_state = None
        self.awaiting_mouse_rearm = False
        if not self.edit_mode and self.selected_target == "seam":
            self.selected_target = "right"
            self.edit_action = "move"
        print(f"Stitch edit mode: {'ON' if self.edit_mode else 'OFF'}")

    def _set_selected_target(self, target):
        if target not in ("left", "right", "seam"):
            return
        self.selected_target = target
        if target == "seam":
            self.edit_action = "blend"
        elif self.edit_action == "blend":
            self.edit_action = "move"

    def _set_edit_action(self, action):
        if action not in ("move", "crop", "blend"):
            return
        if self.selected_target == "seam":
            self.edit_action = "blend"
            return
        if action == "blend":
            self.edit_action = "move"
            return
        self.edit_action = action

    def _nudge_selected(self, dx, dy):
        if self.selected_target not in ("left", "right"):
            return
        offset_key = f"{self.selected_target}_offset"
        self.preset[offset_key]["x"] = int(self.preset[offset_key].get("x", 0)) + int(dx)
        self.preset[offset_key]["y"] = int(self.preset[offset_key].get("y", 0)) + int(dy)

    def _adjust_selected_crop(self, edge, amount):
        if self.selected_target not in ("left", "right"):
            return
        crop_key = f"{self.selected_target}_crop"
        crop = self.preset[crop_key]
        crop[edge] = max(0, int(crop.get(edge, 0)) + int(amount))
        self.preset[crop_key] = crop

    def _adjust_uniform_crop(self, target, amount):
        if target not in ("left", "right"):
            return
        self._set_selected_target(target)
        for edge in ("left", "top", "right", "bottom"):
            self._adjust_selected_crop(edge, amount)

    def _adjust_selected_action(self, dx, dy):
        if self.selected_target == "seam":
            if dx != 0:
                self.preset["blend_width"] = max(0, int(self.preset.get("blend_width", 32)) + int(dx))
            return
        if self.edit_action == "crop":
            if dx < 0:
                self._adjust_selected_crop("left", -dx)
            elif dx > 0:
                self._adjust_selected_crop("right", dx)
            if dy < 0:
                self._adjust_selected_crop("top", -dy)
            elif dy > 0:
                self._adjust_selected_crop("bottom", dy)
            return
        self._nudge_selected(dx, dy)

    def _default_target_offset(self, target):
        if target == "left":
            return {"x": 0, "y": 0}
        if target == "right":
            left_cfg = self.base_module.LOCAL_CAMERAS.get(self.left_camera, {})
            left_width = int(left_cfg.get("width", 1280))
            return {"x": left_width, "y": 0}
        return {"x": 0, "y": 0}

    def save_preset(self):
        save_stitch_preset(self.preset_path, self.preset)
        self._loaded_snapshot = copy.deepcopy(self.preset)
        print(f"💾 Saved stitch preset: {os.path.basename(self.preset_path)}")

    def reload_preset(self):
        self.preset, self.loaded_from_disk = load_stitch_preset(self.preset_path, self.left_camera, self.right_camera)
        self._loaded_snapshot = copy.deepcopy(self.preset)
        self._auto_initialized = False
        print(f"🔄 Reloaded stitch preset: {os.path.basename(self.preset_path)}")

    def reset_selected(self):
        if self.selected_target in ("left", "right"):
            crop_key = f"{self.selected_target}_crop"
            offset_key = f"{self.selected_target}_offset"
            self.preset[crop_key] = {"left": 0, "top": 0, "right": 0, "bottom": 0}
            self.preset[offset_key] = self._default_target_offset(self.selected_target)
        elif self.selected_target == "seam":
            self.preset["blend_width"] = 32
        print(f"↩️ Reset target: {self.selected_target.upper()}")

    def handle_key(self, key):
        if key in (ord("m"), ord("M")):
            self._set_edit_mode(not self.edit_mode, save_on_exit=self.edit_mode)
            return True
        if not self.edit_mode:
            return False
        if key == ord("1"):
            self._set_selected_target("left")
            print("Stitch target: LEFT")
            return True
        if key == ord("2"):
            self._set_selected_target("right")
            print("Stitch target: RIGHT")
            return True
        if key == ord("3"):
            self._set_selected_target("seam")
            print("Stitch target: SEAM")
            return True
        if key == 9:
            order = ("left", "right", "seam")
            try:
                idx = order.index(self.selected_target)
            except ValueError:
                idx = 0
            self._set_selected_target(order[(idx + 1) % len(order)])
            print(f"Stitch target: {self.selected_target.upper()}")
            return True
        if key in (ord("c"), ord("C")):
            self._set_edit_action("crop")
            print(f"Stitch action: {self.edit_action.upper()}")
            return True
        if key in (ord("v"), ord("V")):
            self._set_edit_action("move")
            print(f"Stitch action: {self.edit_action.upper()}")
            return True
        # Use 'e' for blend so 'b' is free for main app (show_boxes)
        if key in (ord("e"), ord("E")):
            self._set_edit_action("blend")
            print(f"Stitch action: {self.edit_action.upper()}")
            return True
        if key in (ord("["), ord("{")):
            self.preset["blend_width"] = max(0, int(self.preset.get("blend_width", 32)) - 1)
            return True
        if key in (ord("]"), ord("}")):
            self.preset["blend_width"] = int(self.preset.get("blend_width", 32)) + 1
            return True
        if key == 13:
            self.save_preset()
            self.drag_state = None
            self.awaiting_mouse_rearm = True
            print("Stitch saved. Click mouse to adjust again.")
            return True
        if key in (ord("g"), ord("G")):
            self.reload_preset()
            return True
        if key in (ord("r"), ord("R")):
            self.reset_selected()
            return True
        key_map = {
            ord("a"): (-1, 0), ord("d"): (1, 0), ord("w"): (0, -1), ord("s"): (0, 1),
            ord("A"): (-10, 0), ord("D"): (10, 0), ord("W"): (0, -10), ord("S"): (0, 10),
        }
        if key in key_map and not self.awaiting_mouse_rearm:
            dx, dy = key_map[key]
            self._adjust_selected_action(dx, dy)
            return True
        return False

    def _update_crop(self, target, edges, dx, dy):
        crop_key = f"{target}_crop"
        crop = self.preset[crop_key]
        for edge in edges:
            if edge == "left":
                crop["left"] = max(0, int(self.drag_state["start_crop"]["left"] + dx))
            elif edge == "right":
                crop["right"] = max(0, int(self.drag_state["start_crop"]["right"] - dx))
            elif edge == "top":
                crop["top"] = max(0, int(self.drag_state["start_crop"]["top"] + dy))
            elif edge == "bottom":
                crop["bottom"] = max(0, int(self.drag_state["start_crop"]["bottom"] - dy))
        self.preset[crop_key] = crop

    def handle_mouse(self, module, event, x, y, flags, userdata):
        if self.last_canvas_shape is None:
            return False
        display_rects = self._display_rects(module)
        handle_radius = self._display_handle_radius(module)

        if event == cv2.EVENT_RBUTTONDOWN:
            if not self.edit_mode:
                target = self._pick_target_from_display(x, y, display_rects)
                if target is not None:
                    self._set_selected_target(target)
                self._set_edit_mode(True)
            else:
                self._set_edit_mode(False, save_on_exit=True)
            return True

        if not self.edit_mode:
            return False

        px, py = self._canvas_point_from_display(module, x, y)
        overlap_rect = display_rects.get("seam")
        target_order = self._iter_target_priority()

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.awaiting_mouse_rearm = False

        if event == cv2.EVENT_MBUTTONDOWN:
            target = self._pick_target_from_display(x, y, display_rects)
            if target is not None:
                self._set_selected_target(target)
                self.reset_selected()
            else:
                self.reload_preset()
            return True

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._pick_mouse_hit(x, y, display_rects, handle_radius, target_order)
            if hit is None:
                return False
            self._set_selected_target(hit["target"])
            if hit["mode"] == "crop":
                self._set_edit_action("crop")
                self.drag_state = {
                    "mode": "crop",
                    "target": hit["target"],
                    "edges": hit["edges"],
                    "start_x": px,
                    "start_y": py,
                    "start_crop": copy.deepcopy(self.preset[f"{hit['target']}_crop"]),
                }
                return True
            if hit["mode"] == "blend":
                self.drag_state = {"mode": "blend", "start_x": px, "start_blend": int(self.preset.get("blend_width", 32))}
                return True
            self._set_edit_action("move")
            self.drag_state = {
                "mode": "move",
                "target": hit["target"],
                "start_x": px,
                "start_y": py,
                "start_offset": copy.deepcopy(self.preset[f"{hit['target']}_offset"]),
            }
            return True
        if event == cv2.EVENT_MOUSEMOVE and self.drag_state is not None:
            dx = px - self.drag_state["start_x"]
            dy = py - self.drag_state.get("start_y", py)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                if abs(dx) >= abs(dy):
                    dy = 0
                else:
                    dx = 0
            if flags & cv2.EVENT_FLAG_ALTKEY:
                dx = int(round(dx * 0.2))
                dy = int(round(dy * 0.2))
            mode = self.drag_state["mode"]
            if mode == "move":
                target = self.drag_state["target"]
                offset_key = f"{target}_offset"
                start_offset = self.drag_state["start_offset"]
                self.preset[offset_key]["x"] = int(start_offset["x"] + dx)
                self.preset[offset_key]["y"] = int(start_offset["y"] + dy)
            elif mode == "crop":
                self._update_crop(self.drag_state["target"], self.drag_state["edges"], dx, dy)
            elif mode == "blend":
                self.preset["blend_width"] = max(0, int(self.drag_state["start_blend"] + dx))
            return True
        if event == cv2.EVENT_LBUTTONUP and self.drag_state is not None:
            self.drag_state = None
            return True
        return False

    def _scale_rect(self, rect, display_shape):
        if not rect or not self.last_canvas_shape:
            return None
        canvas_h, canvas_w = self.last_canvas_shape
        disp_h, disp_w = display_shape[:2]
        sx = disp_w / float(max(1, canvas_w))
        sy = disp_h / float(max(1, canvas_h))
        return {
            "x": int(round(rect["x"] * sx)),
            "y": int(round(rect["y"] * sy)),
            "w": int(round(rect["w"] * sx)),
            "h": int(round(rect["h"] * sy)),
        }

    def _draw_hud_line(self, image, text, x, y, font_scale=0.65):
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad_x = 8
        pad_y = 6
        top_left = (max(0, x - pad_x), max(0, y - text_h - pad_y))
        bottom_right = (
            min(image.shape[1] - 1, x + text_w + pad_x),
            min(image.shape[0] - 1, y + baseline + pad_y),
        )
        cv2.rectangle(image, top_left, bottom_right, (235, 235, 235), -1)
        cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    def overlay_display(self, image):
        self.last_display_shape = image.shape[:2]
        overlay = image.copy()
        if SHOW_STITCH_RUNTIME_HUD:
            stitch_h, stitch_w = self.last_canvas_shape if self.last_canvas_shape else image.shape[:2]
            fps_line_1 = (
                f"L: {self._fps_value(self._left_frame_times):.1f} fps"
                f"  R: {self._fps_value(self._right_frame_times):.1f} fps"
            )
            fps_line_2 = (
                f"OUT FPS: {self._fps_value(self._stitched_frame_times):.1f}"
                f"  STITCH: {stitch_w}x{stitch_h}"
                f"  SHOW: {image.shape[1]}x{image.shape[0]}"
            )
            self._draw_hud_line(overlay, fps_line_1, 12, 28, font_scale=0.55)
            self._draw_hud_line(overlay, fps_line_2, 12, 52, font_scale=0.55)
            self._maybe_log_fps()
        if not self.edit_mode:
            return overlay
        colors = {"left": (0, 255, 255), "right": (255, 255, 0), "seam": (0, 255, 0)}
        for target in ("left", "right"):
            rect = self._scale_rect(self.last_layout.get(target), overlay.shape)
            if not rect:
                continue
            color = colors[target]
            thickness = 3 if self.selected_target == target else 2
            cv2.rectangle(overlay, (rect["x"], rect["y"]), (rect["x"] + rect["w"], rect["y"] + rect["h"]), color, thickness)
            if self.selected_target == target:
                handles = [
                    (rect["x"], rect["y"]),
                    (rect["x"] + rect["w"], rect["y"]),
                    (rect["x"], rect["y"] + rect["h"]),
                    (rect["x"] + rect["w"], rect["y"] + rect["h"]),
                    (rect["x"], rect["y"] + rect["h"] // 2),
                    (rect["x"] + rect["w"], rect["y"] + rect["h"] // 2),
                    (rect["x"] + rect["w"] // 2, rect["y"]),
                    (rect["x"] + rect["w"] // 2, rect["y"] + rect["h"]),
                ]
                for hx, hy in handles:
                    cv2.circle(overlay, (hx, hy), 6, color, -1, lineType=cv2.LINE_AA)
        seam_rect = self._scale_rect(self._overlap_rect(), overlay.shape)
        if seam_rect:
            seam_color = colors["seam"]
            cv2.rectangle(
                overlay,
                (seam_rect["x"], seam_rect["y"]),
                (seam_rect["x"] + seam_rect["w"], seam_rect["y"] + seam_rect["h"]),
                seam_color,
                2 if self.selected_target == "seam" else 1,
            )
        lines = [
            f"STITCH MODE: ON  TARGET: {self.selected_target.upper()}  ACTION: {self.edit_action.upper()}  PRESET: {os.path.basename(self.preset_path)}",
            (
                f"L ofs=({self.preset['left_offset']['x']},{self.preset['left_offset']['y']}) "
                f"crop={self.preset['left_crop']}  "
                f"R ofs=({self.preset['right_offset']['x']},{self.preset['right_offset']['y']}) "
                f"crop={self.preset['right_crop']}"
            ),
            "Mouse: RightClick=enter edit / save+exit  LeftDrag inside=move  LeftDrag edge/corner=adjust only while held",
            "Keys: 1/2/3=target  C/V/E=action(crop/move/blend)  WASD=adjust  Shift+WASD=step10  Enter=save+lock  G=reload R=reset-full M=toggle",
            f"Blend: {int(self.preset.get('blend_width', 0))}  Mouse adjust=hold+drag only  MiddleClick=reset full target  Tab=cycle target",
        ]
        y = 84
        for line in lines:
            self._draw_hud_line(overlay, line, 12, y)
            y += 26
        return overlay


def build_bottom_synthetic_config(module, worker_config):
    left_cfg = copy.deepcopy(module.LOCAL_CAMERAS[worker_config["left_camera"]])
    right_cfg = copy.deepcopy(module.LOCAL_CAMERAS[worker_config["right_camera"]])
    preset, _ = load_stitch_preset(
        worker_config["preset_path"],
        worker_config["left_camera"],
        worker_config["right_camera"],
    )
    left_offset = preset.get("left_offset", {"x": 0, "y": 0})
    right_offset = preset.get("right_offset", {"x": int(left_cfg.get("width", 1280)), "y": 0})
    left_x = int(left_offset.get("x", 0))
    left_y = int(left_offset.get("y", 0))
    right_x = int(right_offset.get("x", 0))
    right_y = int(right_offset.get("y", 0))
    min_x = min(0, left_x, right_x)
    min_y = min(0, left_y, right_y)
    required_width = max(
        left_x + int(left_cfg.get("width", 1280)),
        right_x + int(right_cfg.get("width", 1280)),
    ) - min_x
    required_height = max(
        left_y + int(left_cfg.get("height", 720)),
        right_y + int(right_cfg.get("height", 720)),
    ) - min_y
    width = max(int(preset.get("output_width", 0)), required_width)
    height = max(int(preset.get("output_height", 0)), required_height)
    # fov_by_source stores per-source FOV so callers can look up the correct FOV
    # for any canvas point rather than always using left_cfg FOV for the whole canvas.
    fov_by_source = {
        "left": {
            "h": float(left_cfg.get("fov_horizontal", 60.0)),
            "v": float(left_cfg.get("fov_vertical", 36.0)),
        },
        "right": {
            "h": float(right_cfg.get("fov_horizontal", 60.0)),
            "v": float(right_cfg.get("fov_vertical", 36.0)),
        },
    }
    return {
        **left_cfg,
        "name": worker_config.get("synthetic_camera_name", "bottom_stitched"),
        "width": width,
        "height": height,
        "source_camera_names": [
            worker_config["left_camera"],
            worker_config["right_camera"],
        ],
        "fov_by_source": fov_by_source,
        "canonical_schema_version": "canonical_v2",
        "use_video_file": False,
        "rtsp_url": None,
        "udp_ip": None,
        "udp_port": None,
        "use_udp_direct": False,
        "stream_format": "h264",
    }


def run_bottom_stitch_worker(
    module,
    worker_config,
    apply_worker_thresholds,
    patch_runtime_for_worker,
    hook_proxy_cls,
    single_camera_main,
):
    apply_worker_thresholds(module, worker_config)
    synthetic_camera_name = worker_config.get("synthetic_camera_name", "bottom_stitched")
    module.LOCAL_CAMERAS[synthetic_camera_name] = build_bottom_synthetic_config(module, worker_config)
    module.LOCAL_ACTIVE_CAMERA = synthetic_camera_name

    stitched_stream = StitchedCameraStream(
        module,
        worker_config["left_camera"],
        worker_config["right_camera"],
        worker_config["preset_path"],
        clone_right_from_left=worker_config.get("clone_right_from_left", False),
    )
    hook_proxy = hook_proxy_cls(stream=stitched_stream)
    patch_runtime_for_worker(module, worker_config, hook_proxy=hook_proxy)

    print(
        f"[{worker_config['instance_name']}] bottom stitch source={worker_config['left_camera']} + {worker_config['right_camera']}"
    )
    print(f"[{worker_config['instance_name']}] synthetic camera={synthetic_camera_name}")
    print(f"[{worker_config['instance_name']}] window={worker_config.get('window_name', synthetic_camera_name)}")

    single_camera_main(
        camera_override=stitched_stream,
        active_camera_name=synthetic_camera_name,
    )


def _load_standalone_camera_registry():
    try:
        from config import CAMERAS as config_cameras
    except Exception as exc:
        raise RuntimeError(f"Failed to import camera registry from config.py: {exc}") from exc
    return copy.deepcopy(config_cameras)


def _fit_display_size(frame_w, frame_h, max_w=STANDALONE_MAX_WIDTH, max_h=STANDALONE_MAX_HEIGHT):
    frame_w = max(1, int(frame_w))
    frame_h = max(1, int(frame_h))
    scale = min(float(max_w) / float(frame_w), float(max_h) / float(frame_h), 1.0)
    return max(1, int(round(frame_w * scale))), max(1, int(round(frame_h * scale)))


def _get_screen_size_fallback():
    return STANDALONE_MAX_WIDTH, STANDALONE_MAX_HEIGHT


def _get_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
        root.destroy()
        if screen_w > 0 and screen_h > 0:
            return screen_w, screen_h
    except Exception:
        pass
    return _get_screen_size_fallback()


def _fit_display_size_to_screen(frame_w, frame_h):
    screen_w, screen_h = _get_screen_size()
    max_w = max(320, int(screen_w) - STANDALONE_SCREEN_MARGIN_X)
    max_h = max(240, int(screen_h) - STANDALONE_SCREEN_MARGIN_Y)
    return _fit_display_size(frame_w, frame_h, max_w=max_w, max_h=max_h)


def _get_window_visible_state(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except Exception:
        return None


def _build_standalone_module():
    try:
        from fast_motion_sky import CameraStream
    except Exception as exc:
        raise RuntimeError(f"Failed to import CameraStream from fast_motion_sky.py: {exc}") from exc

    return SimpleNamespace(
        __file__=__file__,
        CameraStream=CameraStream,
        LOCAL_CAMERAS=_load_standalone_camera_registry(),
        display_w=1,
        display_h=1,
        current_frame_w=1,
        current_frame_h=1,
    )


def run_standalone_editor(
    left_camera=STANDALONE_LEFT_CAMERA,
    right_camera=STANDALONE_RIGHT_CAMERA,
    preset_path=STANDALONE_PRESET_PATH,
    clone_right_from_left=False,
    start_in_edit_mode=True,
    window_name=STANDALONE_WINDOW_NAME,
):
    module = _build_standalone_module()
    if left_camera not in module.LOCAL_CAMERAS:
        raise KeyError(f"Unknown left camera '{left_camera}'")
    if right_camera not in module.LOCAL_CAMERAS:
        raise KeyError(f"Unknown right camera '{right_camera}'")

    stitched_stream = StitchedCameraStream(
        base_module=module,
        left_camera=left_camera,
        right_camera=right_camera,
        preset_path=preset_path,
        clone_right_from_left=clone_right_from_left,
    ).start()

    if start_in_edit_mode:
        stitched_stream._set_edit_mode(True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_close_check_after = time.time() + 2.0
    window_was_visible = False

    def mouse_callback(event, x, y, flags, userdata):
        stitched_stream.handle_mouse(module, event, x, y, flags, userdata)

    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            active, frame, _ = stitched_stream.read()
            if not active or frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            module.current_frame_w = int(frame.shape[1])
            module.current_frame_h = int(frame.shape[0])
            display_w, display_h = _fit_display_size_to_screen(frame.shape[1], frame.shape[0])
            module.display_w = display_w
            module.display_h = display_h

            display_frame = frame
            if display_w != frame.shape[1] or display_h != frame.shape[0]:
                display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            cv2.resizeWindow(window_name, display_w, display_h)
            display_frame = stitched_stream.overlay_display(display_frame)
            cv2.imshow(window_name, display_frame)

            visible = _get_window_visible_state(window_name)
            if visible is not None and visible >= 1:
                window_was_visible = True
            can_check_window_close = time.time() >= window_close_check_after and window_was_visible
            if can_check_window_close and visible is not None and visible < 1:
                break

            key = cv2.waitKeyEx(1)
            key = key & 0xFF if key != -1 else -1
            if key != -1 and stitched_stream.handle_key(key):
                continue
            if key in (ord("q"), 27):
                if stitched_stream.edit_mode:
                    stitched_stream.save_preset()
                break
        return 0
    finally:
        stitched_stream.release()
        cv2.destroyWindow(window_name)


def _parse_cli_args(argv):
    left_camera = STANDALONE_LEFT_CAMERA
    right_camera = STANDALONE_RIGHT_CAMERA
    preset_path = STANDALONE_PRESET_PATH
    clone_right_from_left = False
    start_in_edit_mode = True

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--left" and idx + 1 < len(argv):
            idx += 1
            left_camera = argv[idx]
        elif arg == "--right" and idx + 1 < len(argv):
            idx += 1
            right_camera = argv[idx]
        elif arg == "--preset" and idx + 1 < len(argv):
            idx += 1
            preset_path = argv[idx]
        elif arg == "--clone-right-from-left":
            clone_right_from_left = True
        elif arg == "--no-edit":
            start_in_edit_mode = False
        elif arg in ("-h", "--help"):
            print("Usage: python3 bottom_stitch_source.py [--left cam7] [--right cam7] [--preset bottom_stitch.jetson] [--clone-right-from-left] [--no-edit]")
            return None
        else:
            raise ValueError(f"Unknown argument: {arg}")
        idx += 1

    return {
        "left_camera": left_camera,
        "right_camera": right_camera,
        "preset_path": preset_path,
        "clone_right_from_left": clone_right_from_left,
        "start_in_edit_mode": start_in_edit_mode,
    }


__all__ = [
    "StitchedCameraStream",
    "build_bottom_synthetic_config",
    "load_stitch_preset",
    "run_standalone_editor",
    "run_bottom_stitch_worker",
    "save_stitch_preset",
]


if __name__ == "__main__":
    cli_config = _parse_cli_args(sys.argv[1:])
    if cli_config is not None:
        sys.exit(run_standalone_editor(**cli_config))

