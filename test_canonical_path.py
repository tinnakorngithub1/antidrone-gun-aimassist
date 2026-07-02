#test_canonical_path.py

import os
import tempfile
import unittest

from canonical_path import (
    CANONICAL_PATH_SCHEMA_VERSION,
    build_path_ml_bundle,
    canonicalize_detection,
    ensure_path_ml_bundle,
    ensure_storage_layout,
    normalized_to_source,
    sample_horizon_y,
    source_to_normalized,
    stitched_point_to_source_camera,
)


class CanonicalPathTests(unittest.TestCase):
    def test_normalized_roundtrip(self):
        x_norm, y_norm = source_to_normalized(640, 360, 1280, 720)
        self.assertAlmostEqual(x_norm, 0.5)
        self.assertAlmostEqual(y_norm, 0.5)
        x_src, y_src = normalized_to_source(x_norm, y_norm, 1280, 720)
        self.assertEqual((x_src, y_src), (640, 360))

    def test_horizon_sampling_interpolation(self):
        horizon = [(0, 100), (100, 200)]
        self.assertAlmostEqual(sample_horizon_y(horizon, 50), 150.0)
        self.assertAlmostEqual(sample_horizon_y(horizon, -50), 50.0)
        self.assertAlmostEqual(sample_horizon_y(horizon, 150), 250.0)

    def test_stitched_point_to_source_camera(self):
        layout = {
            "left": {"x": 0, "y": 0, "w": 100, "h": 50},
            "right": {"x": 80, "y": 0, "w": 100, "h": 50},
        }
        left_payload = stitched_point_to_source_camera(20, 10, layout, {"left": "cam3", "right": "cam7"})
        self.assertEqual(left_payload["source_camera_name"], "cam3")
        self.assertEqual(left_payload["resolved_region"], "left")
        overlap_payload = stitched_point_to_source_camera(90, 10, layout, {"left": "cam3", "right": "cam7"})
        self.assertTrue(overlap_payload["crosses_seam"])
        self.assertIn(overlap_payload["resolved_region"], ("left", "right"))

    def test_canonicalize_detection_with_stitched_resolver(self):
        layout = {
            "left": {"x": 0, "y": 0, "w": 100, "h": 50},
            "right": {"x": 100, "y": 0, "w": 100, "h": 50},
        }

        def resolver(x_value, y_value):
            return stitched_point_to_source_camera(
                x_value,
                y_value,
                layout,
                {"left": "cam3", "right": "cam7"},
            )

        meta = canonicalize_detection(
            rect=(120, 10, 20, 10),
            frame_shape=(50, 200, 3),
            horizon_points=[(0, 5), (200, 5)],
            camera_name="bottom_stitched",
            point_resolver=resolver,
        )
        self.assertEqual(meta["source_camera_name"], "cam7")
        self.assertAlmostEqual(meta["x_norm"], 0.3, places=3)
        self.assertIsNotNone(meta["y_from_horizon_norm"])

    def test_ensure_storage_layout_writes_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = ensure_storage_layout(tmpdir, CANONICAL_PATH_SCHEMA_VERSION)
            self.assertTrue(os.path.exists(layout["metadata"]["path_schema"]))
            self.assertTrue(os.path.isdir(layout["datasets"]["canonical_new_train"]))

    def test_path_ml_bundle_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            b = build_path_ml_bundle(tmpdir, "path_ml_v1")
            self.assertIn("train", b)
            self.assertTrue(b["train"].endswith(os.path.join("path_ml_v1", "train")))
            layout = ensure_path_ml_bundle(tmpdir, "path_ml_v1")
            self.assertTrue(os.path.isdir(layout["train"]))
            self.assertTrue(os.path.isdir(layout["test"]))
            self.assertTrue(os.path.isdir(layout["models"]))


if __name__ == "__main__":
    unittest.main()

