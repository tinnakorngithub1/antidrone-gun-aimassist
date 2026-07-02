#active_label_trainer.py

import os
import sys
import time
import csv
import json
import numpy as np
import cv2
import pandas as pd

from canonical_path import (
    CANONICAL_PATH_SCHEMA_VERSION,
    canonicalize_detection,
    canonicalize_trajectory,
)

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    HistGradientBoostingClassifier = None
    train_test_split = None
    StandardScaler = None
    compute_sample_weight = None
    joblib = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


class ActiveLabelTrainer:
    FEATURE_DIM = 20
    FEATURE_NAMES = [
        "vel_mag_per_sec",          # f0: mean speed (normalized/sec)
        "acc_mag_per_sec",          # f1: mean acceleration (normalized/sec^2)
        "smoothness_index",         # f2
        "area_variance",            # f3
        "bbox_area_norm",           # f4
        "avg_area_norm",            # f5
        "bbox_aspect_ratio",        # f6
        "y_from_horizon_norm",      # f7
        "turn_rate_norm",           # f8
        "velocity_cv",              # f9
        "jitter_index_norm",        # f10
        "area_vs_sky_y",            # f11
        "path_straightness",        # f12
        "hover_ratio",              # f13
        "vertical_velocity_ratio",  # f14
        "area_oscillation_rate",    # f15
        "cumulative_length_norm",   # f16: total path arc-length / frame diagonal (plane >> drone)
        "direction_variance",       # f17: variance of direction angles (erratic = bird/drone)
        "speed_range_norm",         # f18: (max-min speed) / (mean+eps) — plane constant; drone variable
        "path_lateral_spread",      # f19: std of cross-track distance from mean direction (zigzag vs straight)
    ]

    def __init__(
        self,
        dataset_dir,
        class_names=None,
        model_filename=None,
        model_dir=None,
        min_samples=12,
        prompt_enabled=False,
        prompt_interval=1.0,
        obj_cooldown=6.0,
        skip_conf_threshold=0.85,
        confirm_conf_threshold=0.85,
        confirm_frames=3,
        padding_ratio=0.15,
        # When upstream path sampling uses a lower px threshold, paths grow faster; raise these
        # after field test if labeling prompts become noisy (see 11_ PATH_MIN_DIST_PX).
        min_path_points=15,
        test_ratio=0.2,
        eval_every_n=20,
        min_path_points_slow=20,
        min_path_points_fast=12,
        min_path_points_very_fast=8,
        fast_vel_thresh=0.18,       # norm/sec (was 0.006/frame × 30fps reference)
        very_fast_vel_thresh=0.36,  # norm/sec (was 0.012/frame × 30fps reference)
        short_window=8,
        long_window=12,
    ):
        if model_filename is None:
            model_filename = f"path_model_{CANONICAL_PATH_SCHEMA_VERSION}.joblib"
        self.enabled = True
        self.prompt_mode = "all" if prompt_enabled else "off"  # off | all | uncertain
        self.prompt_enabled = self.prompt_mode != "off"
        self.class_names = class_names or ["drone", "bird", "plane", "other"]
        self.label_keys = {
            ord("1"): self.class_names[0],
            ord("2"): self.class_names[1],
            ord("3"): self.class_names[2],
            ord("4"): self.class_names[3],
        }
        self.skip_key = ord("0")

        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.labels_csv = os.path.join(dataset_dir, "labels.csv")
        self.model_dir = model_dir
        if model_dir is not None:
            self.model_path = os.path.join(model_dir, os.path.basename(model_filename))
        else:
            self.model_path = os.path.join(dataset_dir, model_filename)
        self.schema_path = os.path.join(dataset_dir, "schema.json")
        self.schema_version = CANONICAL_PATH_SCHEMA_VERSION

        self.min_samples = min_samples
        self.prompt_interval = prompt_interval
        self.obj_cooldown = obj_cooldown
        self.skip_conf_threshold = skip_conf_threshold
        self.confirm_conf_threshold = confirm_conf_threshold
        self.confirm_frames = confirm_frames
        self.padding_ratio = padding_ratio
        self.min_path_points = min_path_points
        self.test_ratio = float(test_ratio)
        self.eval_every_n = int(eval_every_n)
        self.min_path_points_slow = int(min_path_points_slow)
        self.min_path_points_fast = int(min_path_points_fast)
        self.min_path_points_very_fast = int(min_path_points_very_fast)
        self.fast_vel_thresh = float(fast_vel_thresh)
        self.very_fast_vel_thresh = float(very_fast_vel_thresh)
        self.short_window = int(short_window)
        self.long_window = int(long_window)

        self.samples = []
        self.sample_labels = []
        self.test_samples = []
        self.test_labels = []
        self.scaler = None
        self.clf = None
        self.is_fitted = False
        self.model_dirty = False
        self.eval_accuracy = None
        self.eval_total = 0
        self.eval_per_class = {}
        self.new_labels_since_eval = 0
        self.has_new_data = False  # Flag ติดตามว่ามีข้อมูลใหม่ที่ยังไม่ได้ retrain

        self.predictions = {}
        self.current_candidate = None
        self.last_prompt_time = 0.0
        self.recent_labels = {}
        self.drone_streaks = {}
        self.confirmed_ids = set()
        self.button_regions = []
        self.flash_label = None
        self.flash_until = 0.0
        self.defer_clear = False
        self.pause_labeling = False
        self.queue = []
        self.max_queue = 5
        self.pending_pop = False
        self.pending_pop_id = None

        self._ensure_dirs()
        self._load_samples()
        self._load_or_build_model()

    def set_prompt_mode(self, mode):
        if mode not in ("off", "all", "uncertain"):
            return
        self.prompt_mode = mode
        self.prompt_enabled = mode != "off"
        if not self.prompt_enabled:
            self.current_candidate = None

    def toggle_all_prompt(self):
        new_mode = "all" if self.prompt_mode != "all" else "off"
        self.set_prompt_mode(new_mode)
        print(f"Active Label Trainer prompt: {self.prompt_mode.upper()}")

    def toggle_uncertain_prompt(self):
        new_mode = "uncertain" if self.prompt_mode != "uncertain" else "off"
        self.set_prompt_mode(new_mode)
        print(f"Active Label Trainer prompt: {self.prompt_mode.upper()}")

    def toggle(self):
        # backward compatibility
        self.toggle_all_prompt()

    def get_hud_text(self):
        status = self.prompt_mode.upper()
        if status == "OFF":
            return "LABEL: OFF (t=all u=uncertain o=off)"
        if self.prompt_mode == "uncertain":
            if (not self.is_fitted) or (len(self.samples) < self.min_samples):
                return "LABEL: UNCERTAIN (model not ready) (t=all u=uncertain o=off)"
        if self.prompt_mode == "all":
            return f"LABEL: ALL (human first, no HUD pred) (t=all u=uncertain o=off)"
        return f"LABEL: {status} (t=all u=uncertain o=off)"

    def get_classifier_name(self):
        """Return short name for HUD: HGB, LGBM, RF, or MODEL."""
        if self.clf is None:
            return "MODEL"
        name = type(self.clf).__name__
        if "HistGradient" in name:
            return "HGB"
        if "LGBM" in name:
            return "LGBM"
        if "RandomForest" in name:
            return "RF"
        return "MODEL"

    def toggle_pause(self):
        self.pause_labeling = not self.pause_labeling
        print(f"Labeling paused: {self.pause_labeling}")

    def get_accuracy_text(self):
        if not SKLEARN_AVAILABLE:
            return "ACC: N/A"
        if self.eval_accuracy is None or self.eval_total <= 0:
            return "ACC: N/A"
        return f"ACC: {self.eval_accuracy * 100:.1f}% (test={self.eval_total})"

    def get_accuracy_lines(self, min_per_class=5):
        lines = []
        if not SKLEARN_AVAILABLE or self.eval_accuracy is None or self.eval_total <= 0:
            return lines
        lines.append(f"ACC: {self.eval_accuracy * 100:.1f}% (test={self.eval_total})")
        for cls in self.class_names:
            if cls not in self.eval_per_class:
                continue
            correct, total = self.eval_per_class[cls]
            if total < min_per_class:
                continue
            pct = (correct / max(1, total)) * 100.0
            lines.append(f"{cls}: {pct:.1f}% (n={total})")
        return lines

    def _ensure_dirs(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        if self.model_dir is not None:
            os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        for name in self.class_names:
            os.makedirs(os.path.join(self.images_dir, name), exist_ok=True)
        schema_payload = {
            "schema_version": self.schema_version,
            "feature_dim": self.FEATURE_DIM,
            "feature_names": self.FEATURE_NAMES,
            "labels_csv": os.path.basename(self.labels_csv),
            "model_filename": os.path.basename(self.model_path),
            "space": "normalized_horizon_relative_per_sec",
            "fps_normalized": True,
            "notes": {
                "f0_f1": "vel_mag_per_sec / acc_mag_per_sec are time-normalized using actual processing FPS and frame gaps",
                "hover_ratio_threshold": "0.02 normalized/sec",
                "legacy_compatibility": "v1 data is not loaded; clean start with schema_version=canonical_v2",
                "f16_f19": "phase-2 path features: cumulative_length_norm, direction_variance, speed_range_norm, path_lateral_spread",
                "classifier": "HistGradientBoostingClassifier (primary) → LGBMClassifier → RandomForestClassifier",
            },
        }
        try:
            with open(self.schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_payload, f, indent=2)
        except Exception as e:
            print(f"Active Label Trainer: schema write failed: {e}")

    def _assign_split(self, timestamp_ms=None, idx=None):
        """Fallback split used only for new samples saved at runtime."""
        if self.test_ratio <= 0:
            return "train"
        if timestamp_ms is not None:
            try:
                bucket = int(timestamp_ms) % 100
                return "test" if bucket < int(self.test_ratio * 100) else "train"
            except Exception:
                pass
        if idx is not None:
            period = max(1, int(round(1.0 / max(1e-6, self.test_ratio))))
            return "test" if (idx % period) == 0 else "train"
        return "train"

    def _read_labels_csv_rows(self):
        """Read all labeled rows from labels.csv."""
        all_feats = []
        all_labels = []
        if not os.path.exists(self.labels_csv):
            return all_feats, all_labels
        try:
            with open(self.labels_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "")
                    if not label:
                        continue
                    feats = [float(row.get(f"f{i}", 0.0)) for i in range(self.FEATURE_DIM)]
                    all_feats.append(feats)
                    all_labels.append(label)
        except Exception as e:
            print(f"Active Label Trainer: read labels CSV failed: {e}")
        return all_feats, all_labels

    def _stratified_split_last_fraction_per_class(self, all_feats, all_labels):
        """Fallback when train_test_split cannot stratify: last test_ratio per class -> test."""
        class_indices = {}
        for idx, lbl in enumerate(all_labels):
            class_indices.setdefault(lbl, []).append(idx)
        test_set = set()
        for _, idxs in class_indices.items():
            n_test = max(1, int(round(len(idxs) * self.test_ratio)))
            for i in idxs[-n_test:]:
                test_set.add(i)
        for idx, (feats, lbl) in enumerate(zip(all_feats, all_labels)):
            if idx in test_set:
                self.test_samples.append(feats)
                self.test_labels.append(lbl)
            else:
                self.samples.append(feats)
                self.sample_labels.append(lbl)

    def _apply_stratified_split(self, all_feats, all_labels):
        """
        Build train/test from full row lists. Prefer sklearn stratified shuffle split
        (same data -> same split with random_state=42); fallback to last-fraction-per-class.
        """
        self.samples = []
        self.sample_labels = []
        self.test_samples = []
        self.test_labels = []
        if not all_feats:
            return
        if self.test_ratio <= 0:
            self.samples = list(all_feats)
            self.sample_labels = list(all_labels)
            return
        n = len(all_labels)
        if n < 2:
            self.samples = list(all_feats)
            self.sample_labels = list(all_labels)
            return
        X = np.asarray(all_feats, dtype=np.float32)
        y = np.asarray(all_labels)
        used_sklearn = False
        if SKLEARN_AVAILABLE and train_test_split is not None:
            try:
                _, counts = np.unique(y, return_counts=True)
                min_per_class = int(np.min(counts))
                stratify_arg = y if (len(np.unique(y)) >= 2 and min_per_class >= 2) else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.test_ratio,
                    stratify=stratify_arg,
                    random_state=42,
                    shuffle=True,
                )
                self.samples = X_train.tolist()
                self.sample_labels = list(y_train)
                self.test_samples = X_test.tolist()
                self.test_labels = list(y_test)
                used_sklearn = True
            except ValueError:
                used_sklearn = False
        if not used_sklearn:
            self._stratified_split_last_fraction_per_class(all_feats, all_labels)

    def _reload_train_test_from_csv(self):
        """Reload labels.csv and rebuild train/test with a fresh split from all rows (for retrain on exit)."""
        all_feats, all_labels = self._read_labels_csv_rows()
        self._apply_stratified_split(all_feats, all_labels)
        print(
            f"Active Label Trainer: reloaded split from full CSV — "
            f"{len(self.samples)} train / {len(self.test_samples)} test",
            flush=True,
        )

    def _load_samples(self):
        """Load all samples from CSV then stratified train/test split."""
        try:
            all_feats, all_labels = self._read_labels_csv_rows()
            self._apply_stratified_split(all_feats, all_labels)
            if all_feats:
                print(
                    f"Active Label Trainer: loaded {len(self.samples)} train / "
                    f"{len(self.test_samples)} test samples"
                )
        except Exception as e:
            print(f"Active Label Trainer: load samples failed: {e}")

    def _init_model(self):
        if not SKLEARN_AVAILABLE:
            return False
        self.scaler = StandardScaler()
        # Priority: HistGradientBoosting (sklearn, no extra deps, fast predict)
        #           → LGBM (if installed, tuned small)
        #           → RandomForest (fallback)
        if HistGradientBoostingClassifier is not None:
            self.clf = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=8,
                max_leaf_nodes=31,
                learning_rate=0.08,
                min_samples_leaf=10,
                l2_regularization=0.1,
                random_state=42,
                early_stopping=False,
            )
        elif LGBMClassifier is not None:
            self.clf = LGBMClassifier(
                class_weight='balanced',
                n_estimators=100,
                max_depth=7,
                num_leaves=31,
                learning_rate=0.1,
                min_child_samples=10,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        else:
            self.clf = RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
            )
        self.is_fitted = False
        return True

    def _load_or_build_model(self):
        if not SKLEARN_AVAILABLE:
            print("Active Label Trainer: sklearn not available (pip install scikit-learn joblib)")
            return
        loaded = False
        _expected_clf = (
            "HGB" if HistGradientBoostingClassifier is not None
            else ("LGBM" if LGBMClassifier is not None else "RF")
        )
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                dim_ok = isinstance(data, dict) and data.get("feature_dim") == self.FEATURE_DIM
                cls_ok = data.get("class_names") == self.class_names
                clf_ok = data.get("clf_type", _expected_clf) == _expected_clf
                if dim_ok and cls_ok and clf_ok:
                    self.scaler = data.get("scaler")
                    self.clf = data.get("clf")
                    self.is_fitted = bool(data.get("is_fitted", False))
                    self.eval_accuracy = data.get("eval_accuracy")
                    self.eval_total = data.get("eval_total", 0)
                    self.eval_per_class = data.get("eval_per_class", {})
                    loaded = self.scaler is not None and self.clf is not None
                    if loaded:
                        print(f"Active Label Trainer: model loaded ({data.get('clf_type', '?')})")
                        if self.eval_accuracy is None and self.is_fitted:
                            self._evaluate_accuracy()
                else:
                    reasons = []
                    if not dim_ok:
                        reasons.append(f"feature_dim {data.get('feature_dim')} != {self.FEATURE_DIM}")
                    if not cls_ok:
                        reasons.append("class_names mismatch")
                    if not clf_ok:
                        reasons.append(f"clf_type {data.get('clf_type')} != {_expected_clf}")
                    print(f"Active Label Trainer: model mismatch ({'; '.join(reasons)}), retrain required")
            except Exception as e:
                print(f"Active Label Trainer: model load failed: {e}")
        if not loaded:
            self._init_model()
            self._fit_from_samples()

    def _fit_from_samples(self):
        if not SKLEARN_AVAILABLE:
            return
        if not self.samples or len(self.samples) < max(2, self.min_samples):
            return
        if self.scaler is None or self.clf is None:
            self._init_model()
        X = np.array(self.samples, dtype=np.float32)
        y = np.array(self.sample_labels)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        # HistGBM and RF accept ndarray directly; use sample_weight for balanced training
        # (HistGBM class_weight='balanced' requires sklearn ≥1.2; use sample_weight for safety)
        clf_name = type(self.clf).__name__
        use_df = "LGBM" in clf_name  # LGBM benefits from named columns
        try:
            sw = compute_sample_weight("balanced", y) if compute_sample_weight is not None else None
        except Exception:
            sw = None
        try:
            if use_df:
                feature_cols = [f"f{i}" for i in range(self.FEATURE_DIM)]
                Xs_fit = pd.DataFrame(Xs, columns=feature_cols)
            else:
                Xs_fit = Xs
            if sw is not None and "LGBM" not in clf_name:
                self.clf.fit(Xs_fit, y, sample_weight=sw)
            else:
                self.clf.fit(Xs_fit, y)
        except TypeError:
            # fallback if sample_weight not accepted
            self.clf.fit(Xs_fit, y)
        self.is_fitted = True
        self.model_dirty = True
        print(f"Active Label Trainer: model trained ({clf_name}) n={len(y)}")
        self._evaluate_accuracy()

    def retrain_and_save(self):
        """
        Retrain เมื่อปิดโปรแกรม: โหลด labels.csv ทั้งหมด แบ่ง train/test ใหม่จากข้อมูลทั้งก้อน
        (stratified shuffle + random_state=42 เมื่อใช้ sklearn) แล้ว fit บน train และ eval บน test
        """
        if not self.has_new_data:
            self.save_if_dirty()  # Save ตามปกติถ้าไม่มีข้อมูลใหม่
            return

        print("กำลัง retrain model path โปรดรอสักครู่...", flush=True)
        self._reload_train_test_from_csv()
        if len(self.samples) < self.min_samples:
            print("Active Label Trainer: ไม่มีข้อมูลเพียงพอสำหรับ retrain", flush=True)
            self.save_if_dirty()  # Save model ที่มีอยู่
            return

        self._fit_from_samples()  # เรียก _evaluate_accuracy() ภายใน
        self.has_new_data = False  # รีเซ็ต flag ก่อนเรียก save_if_dirty() เพื่อป้องกัน infinite loop
        self.save_if_dirty()  # Save พร้อม accuracy

        if self.eval_accuracy is not None:
            print(f"Active Label Trainer: retrain เสร็จสิ้น (ACC: {self.eval_accuracy * 100:.1f}%)", flush=True)
        else:
            print("Active Label Trainer: retrain เสร็จสิ้น", flush=True)

    def save_if_dirty(self):
        if not SKLEARN_AVAILABLE:
            return

        # ถ้ามีข้อมูลใหม่และมีข้อมูลเพียงพอ → retrain ก่อน save
        if self.has_new_data and len(self.samples) >= self.min_samples:
            self.retrain_and_save()  # retrain_and_save() จะเรียก save_if_dirty() อีกทีหลังจาก retrain
            return

        # Save model ตามปกติ (ไม่มีข้อมูลใหม่ หรือมีข้อมูลไม่เพียงพอ)
        if not self.model_dirty:
            return
        try:
            clf_type = (
                "HGB" if HistGradientBoostingClassifier is not None
                else ("LGBM" if LGBMClassifier is not None else "RF")
            )
            data = {
                "scaler": self.scaler,
                "clf": self.clf,
                "clf_type": clf_type,
                "class_names": self.class_names,
                "feature_dim": self.FEATURE_DIM,
                "is_fitted": self.is_fitted,
                "eval_accuracy": self.eval_accuracy,
                "eval_total": self.eval_total,
                "eval_per_class": self.eval_per_class,
            }
            joblib.dump(data, self.model_path)
            self.model_dirty = False
            print("Active Label Trainer: model saved", flush=True)
        except Exception as e:
            print(f"Active Label Trainer: model save failed: {e}", flush=True)

    def _append_sample(self, timestamp_ms, label, obj_id, rect, frame_shape, features, image_path, canonical_meta=None):
        is_new = not os.path.exists(self.labels_csv)
        header = [
            "schema_version",
            "timestamp",
            "label",
            "obj_id",
            "camera_name",
            "source_camera_name",
            "source_region",
            "crosses_seam",
            "global_frame_w",
            "global_frame_h",
            "frame_w",
            "frame_h",
            "x",
            "y",
            "w",
            "h",
            "x_src",
            "y_src",
            "x_norm",
            "y_norm",
            "horizon_y_at_x",
            "y_from_horizon_norm",
            "bbox_w_norm",
            "bbox_h_norm",
            "processing_fps",
            "median_dt_frames",
        ] + [f"f{i}" for i in range(self.FEATURE_DIM)] + ["image_path"]
        meta = canonical_meta or {}
        source_frame_w = int(meta.get("frame_w", frame_shape[1]))
        source_frame_h = int(meta.get("frame_h", frame_shape[0]))

        try:
            with open(self.labels_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(header)
                row = [
                    self.schema_version,
                    timestamp_ms,
                    label,
                    obj_id,
                    meta.get("camera_name", ""),
                    meta.get("source_camera_name", ""),
                    meta.get("source_region", ""),
                    int(bool(meta.get("crosses_seam", False))),
                    frame_shape[1],
                    frame_shape[0],
                    source_frame_w,
                    source_frame_h,
                    rect[0],
                    rect[1],
                    rect[2],
                    rect[3],
                    f"{float(meta.get('x_src', 0.0)):.6f}",
                    f"{float(meta.get('y_src', 0.0)):.6f}",
                    f"{float(meta.get('x_norm', 0.0)):.6f}",
                    f"{float(meta.get('y_norm', 0.0)):.6f}",
                    "" if meta.get("horizon_y_at_x") is None else f"{float(meta.get('horizon_y_at_x')):.6f}",
                    "" if meta.get("y_from_horizon_norm") is None else f"{float(meta.get('y_from_horizon_norm')):.6f}",
                    f"{float(meta.get('bbox_w_norm', 0.0)):.6f}",
                    f"{float(meta.get('bbox_h_norm', 0.0)):.6f}",
                    f"{float(meta.get('processing_fps', 30.0)):.4f}",
                    f"{float(meta.get('median_dt_frames', 1.0)):.4f}",
                ] + [f"{v:.6f}" for v in features] + [image_path]
                writer.writerow(row)
        except Exception as e:
            print(f"Active Label Trainer: write failed: {e}")

    def _evaluate_accuracy(self):
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            self.eval_accuracy = None
            self.eval_total = 0
            self.eval_per_class = {}
            return
        if not self.test_samples or len(self.test_samples) < 5:
            self.eval_accuracy = None
            self.eval_total = len(self.test_samples)
            self.eval_per_class = {}
            return
        try:
            X = np.array(self.test_samples, dtype=np.float32)
            y = np.array(self.test_labels)
            Xs = self.scaler.transform(X)
            clf_name = type(self.clf).__name__
            if "LGBM" in clf_name:
                feature_cols = [f"f{i}" for i in range(self.FEATURE_DIM)]
                Xs_pred = pd.DataFrame(Xs, columns=feature_cols)
            else:
                Xs_pred = Xs
            preds = self.clf.predict(Xs_pred)
            correct = np.sum(preds == y)
            self.eval_total = len(y)
            self.eval_accuracy = float(correct) / float(max(1, self.eval_total))
            self.eval_per_class = {}
            present_classes = [c for c in self.class_names if np.any(y == c)]

            # per-class recall
            for cls in self.class_names:
                mask = (y == cls)
                total = int(np.sum(mask))
                if total == 0:
                    continue
                cls_correct = int(np.sum(preds[mask] == y[mask]))
                self.eval_per_class[cls] = (cls_correct, total)

            # log confusion matrix and per-class recall to console
            print(f"\n=== Path classifier eval (test={self.eval_total}) ACC={self.eval_accuracy*100:.1f}% ===")
            header = f"{'':>8}" + "".join(f"{c:>8}" for c in present_classes)
            print(header)
            for true_cls in present_classes:
                row_mask = (y == true_cls)
                row_total = int(np.sum(row_mask))
                if row_total == 0:
                    continue
                cells = ""
                for pred_cls in present_classes:
                    count = int(np.sum(preds[row_mask] == pred_cls))
                    cells += f"{count:>8}"
                recall = self.eval_per_class.get(true_cls, (0, row_total))
                recall_pct = recall[0] / max(1, recall[1]) * 100
                print(f"{true_cls:>8}{cells}  recall={recall_pct:.0f}%")
            print("=" * max(40, 8 + 8 * len(present_classes)))
        except Exception as e:
            print(f"Active Label Trainer: eval failed: {e}")
            self.eval_accuracy = None
            self.eval_total = 0
            self.eval_per_class = {}

    def _predict(self, features):
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return None, None
        if len(self.samples) < self.min_samples:
            return None, None
        if self.scaler is None or self.clf is None:
            return None, None

        feats = np.array(features, dtype=np.float32).reshape(1, -1)
        feats = self.scaler.transform(feats)
        clf_name = type(self.clf).__name__
        if "LGBM" in clf_name:
            feature_cols = [f"f{i}" for i in range(self.FEATURE_DIM)]
            feats_in = pd.DataFrame(feats, columns=feature_cols)
        else:
            feats_in = feats  # HistGBM / RF accept ndarray directly → avoids pd overhead
        try:
            proba = self.clf.predict_proba(feats_in)[0]
            idx = int(np.argmax(proba))
            label = self.clf.classes_[idx]
            conf = float(proba[idx])
            return label, conf
        except Exception:
            try:
                label = self.clf.predict(feats_in)[0]
                return label, None
            except Exception:
                return None, None

    def _predict_batch(self, features_list):
        """
        ทำนายหลายแถวในครั้งเดียว — StandardScaler + predict_proba แบบ batch
        เทียบเท่าการเรียก _predict ทีละแถว (แต่ละแถวของ proba อิสระจากกัน)
        """
        n = len(features_list)
        if n == 0:
            return []
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return [(None, None)] * n
        if len(self.samples) < self.min_samples:
            return [(None, None)] * n
        if self.scaler is None or self.clf is None:
            return [(None, None)] * n
        try:
            X = np.asarray(features_list, dtype=np.float32).reshape(n, -1)
        except Exception:
            return [self._predict(f) for f in features_list]
        if X.shape[1] != self.FEATURE_DIM:
            return [self._predict(f) for f in features_list]
        Xs = self.scaler.transform(X)
        clf_name = type(self.clf).__name__
        if "LGBM" in clf_name:
            feature_cols = [f"f{i}" for i in range(self.FEATURE_DIM)]
            feats_in = pd.DataFrame(Xs, columns=feature_cols)
        else:
            feats_in = Xs
        try:
            proba = self.clf.predict_proba(feats_in)
            out = []
            for i in range(n):
                row = proba[i]
                idx = int(np.argmax(row))
                label = self.clf.classes_[idx]
                conf = float(row[idx])
                out.append((label, conf))
            return out
        except Exception:
            out = []
            for i in range(n):
                sub = feats_in[i : i + 1]
                try:
                    proba = self.clf.predict_proba(sub)[0]
                    idx = int(np.argmax(proba))
                    label = self.clf.classes_[idx]
                    conf = float(proba[idx])
                    out.append((label, conf))
                except Exception:
                    try:
                        label = self.clf.predict(sub)[0]
                        out.append((label, None))
                    except Exception:
                        out.append((None, None))
            return out

    def _extract_features(self, rect, path_data, frame_shape, canonical_context=None):
        canonical_context = canonical_context or {}
        h, w = frame_shape[:2]
        fw = max(w, 1)
        fh = max(h, 1)
        fd = float(max(fw, fh))

        x, y, bw, bh = rect[0], rect[1], rect[2], rect[3]
        point_resolver = canonical_context.get("point_resolver")
        horizon_points = canonical_context.get("horizon_points")
        camera_name = canonical_context.get("camera_name")

        canonical_meta = canonicalize_detection(
            rect=rect,
            frame_shape=frame_shape,
            horizon_points=horizon_points,
            camera_name=camera_name,
            point_resolver=point_resolver,
        )

        bbox_area = float(canonical_meta.get("bbox_w_norm", 0.0)) * float(canonical_meta.get("bbox_h_norm", 0.0))
        aspect = bw / max(1.0, float(bh))
        y_from_horizon_norm = canonical_meta.get("y_from_horizon_norm")
        if y_from_horizon_norm is None:
            y_from_horizon_norm = float(canonical_meta.get("y_norm", 0.0))
        y_from_horizon_norm = float(y_from_horizon_norm)

        kin = path_data.get("kinematic_history", {})
        smoothness = float(kin.get("smoothness_index", 0.0))
        area_var = float(kin.get("area_variance", 0.0))

        areas = path_data.get("smoothed_areas") or path_data.get("areas") or []
        areas_arr = np.array(list(areas), dtype=np.float32) if len(areas) > 0 else None
        bbox_denom = float(
            max(
                1,
                int(canonical_meta.get("frame_w", fw)) * int(canonical_meta.get("frame_h", fh)),
            )
        )
        if areas_arr is not None:
            avg_area = float(np.mean(areas_arr)) / bbox_denom
        else:
            avg_area = bbox_area

        # --- Additional features --- (reuse pts_arr_use, pts_arr_full, areas_arr)
        turn_rate = 0.0
        direction_changes = 0.0
        smoothed_pts = path_data.get("smoothed_points", path_data.get("points", []))
        canonical_points = canonicalize_trajectory(
            points=list(smoothed_pts) if smoothed_pts else [],
            frame_shape=frame_shape,
            horizon_points=horizon_points,
            camera_name=camera_name,
            point_resolver=point_resolver,
        )
        if canonical_points:
            pts_list = [(pt["x_norm"], pt["y_norm"]) for pt in canonical_points]
            use_pts = pts_list if len(pts_list) >= self.long_window else pts_list[-self.short_window:]
        else:
            pts_list = []
            use_pts = []

        pts_arr_use = np.array(list(use_pts), dtype=np.float32) if use_pts and len(use_pts) >= 4 else None
        pts_arr_full = np.array(list(pts_list), dtype=np.float32) if pts_list and len(pts_list) >= 2 else None

        # --- Time normalization setup (v2) ---
        # Build per-step dt array from points_with_frame (x, y, frame_no).
        # frame_gap is clamped to [1, 5] to avoid inflated dt from tracking gaps.
        # Falls back to 1/fps (one-frame step) when points_with_frame is unavailable.
        processing_fps = float(canonical_context.get("processing_fps") or 30.0)
        processing_fps = max(5.0, min(120.0, processing_fps))

        pwf = list(path_data.get("points_with_frame", []))  # list of (x, y, frame_no)
        n_pts_full = len(pts_list)

        def _build_dt_array(n_steps):
            """Return array of Δt (seconds) of length n_steps aligned to the last n_steps+1 points."""
            if len(pwf) >= n_steps + 1:
                tail = pwf[-(n_steps + 1):]
                dt_arr = np.ones(n_steps, dtype=np.float32) / processing_fps
                for i in range(n_steps):
                    gap = abs(int(tail[i + 1][2]) - int(tail[i][2]))
                    gap = max(1, min(gap, 5))  # clamp: ignore large tracking gaps
                    dt_arr[i] = float(gap) / processing_fps
                return dt_arr
            return np.full(n_steps, 1.0 / processing_fps, dtype=np.float32)

        # f0: vel_mag_per_sec, f1: acc_mag_per_sec (time-normalized, schema v2)
        vel_mag = 0.0
        acc_mag = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 2:
            n_steps = len(pts_arr_full) - 1
            dt_full = _build_dt_array(n_steps)
            diffs = pts_arr_full[1:] - pts_arr_full[:-1]
            speed_series = np.linalg.norm(diffs, axis=1) / np.maximum(dt_full, 1e-6)
            if len(speed_series) > 0:
                vel_mag = float(np.mean(speed_series))
            if len(speed_series) >= 2:
                dt_mid = (dt_full[:-1] + dt_full[1:]) / 2.0
                acc_mag = float(np.mean(np.abs(np.diff(speed_series)) / np.maximum(dt_mid, 1e-6)))

        # 1) Turn rate / direction change frequency (dimensionless ratios — no time normalization needed)
        if pts_arr_use is not None and len(pts_arr_use) >= 4:
            vecs = pts_arr_use[1:] - pts_arr_use[:-1]
            mags = np.linalg.norm(vecs, axis=1)
            valid = mags > 1e-6
            if np.any(valid) and np.sum(valid) >= 2:
                dirs = np.zeros_like(vecs)
                dirs[valid] = vecs[valid] / mags[valid][:, None]
                dots = np.sum(dirs[1:] * dirs[:-1], axis=1)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                turn_rate = float(np.mean(angles)) / np.pi
                direction_changes = float(np.mean(angles > (20.0 * np.pi / 180.0)))

        # 2) Velocity consistency (CV) — use time-normalized speeds for consistency with f0
        velocity_cv = 0.0
        if pts_arr_use is not None and len(pts_arr_use) >= 4:
            n_use = len(pts_arr_use) - 1
            dt_use = _build_dt_array(n_use)
            use_diffs = pts_arr_use[1:] - pts_arr_use[:-1]
            speeds_use = np.linalg.norm(use_diffs, axis=1) / np.maximum(dt_use, 1e-6)
            if len(speeds_use) >= 2:
                mean_v = float(np.mean(speeds_use))
                std_v = float(np.std(speeds_use))
                if mean_v > 1e-6:
                    velocity_cv = std_v / mean_v

        # 3) Hover / jitter index (position jitter — spatial, dimensionless)
        jitter_index = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 5:
            recent = pts_arr_full[-10:] if len(pts_arr_full) >= 10 else pts_arr_full
            if len(recent) >= 2:
                center = np.mean(recent, axis=0)
                dists = np.linalg.norm(recent - center, axis=1)
                jitter_index = float(np.mean(dists)) / fd

        # 4) Area vs sky position
        area_vs_sky_y = bbox_area * (1.0 - float(np.clip(canonical_meta.get("y_norm", 0.0), 0.0, 1.0)))

        # 5) Path straightness (displacement / total length — dimensionless)
        path_straightness = 0.0
        if pts_arr_use is not None and len(pts_arr_use) >= 3:
            segs = np.linalg.norm(pts_arr_use[1:] - pts_arr_use[:-1], axis=1)
            total_len = float(np.sum(segs))
            displacement = float(np.linalg.norm(pts_arr_use[-1] - pts_arr_use[0]))
            if total_len > 1e-6:
                path_straightness = displacement / total_len

        # 6) Hover ratio: fraction of path where time-normalized speed < threshold (norm/sec)
        HOVER_SPEED_THRESH = 0.02  # normalized/sec; drone hovering moves < 2% of frame per second
        hover_ratio = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 2:
            n_steps = len(pts_arr_full) - 1
            dt_full_h = _build_dt_array(n_steps)
            diffs_h = pts_arr_full[1:] - pts_arr_full[:-1]
            speeds_h = np.linalg.norm(diffs_h, axis=1) / np.maximum(dt_full_h, 1e-6)
            if len(speeds_h) > 0:
                hover_ratio = float(np.sum(speeds_h < HOVER_SPEED_THRESH)) / float(len(speeds_h))

        # 7) Vertical velocity ratio: |vy| / (|vx| + |vy| + eps) (plane more horizontal)
        if pts_arr_full is not None and len(pts_arr_full) >= 2:
            last_diff = pts_arr_full[-1] - pts_arr_full[-2]
            abs_vx = abs(float(last_diff[0]))
            abs_vy = abs(float(last_diff[1]))
        else:
            abs_vx = 0.0
            abs_vy = 0.0
        vertical_velocity_ratio = abs_vy / (abs_vx + abs_vy + 1e-6)

        # 8) Area oscillation rate: sign changes in area diff / path length (bird flapping)
        area_oscillation_rate = 0.0
        if areas_arr is not None and len(areas_arr) >= 3:
            diff_arr = np.diff(areas_arr)
            if len(diff_arr) >= 2:
                sign_changes = np.sum((diff_arr[1:] * diff_arr[:-1]) < 0)
                area_oscillation_rate = float(sign_changes) / max(len(areas_arr) - 1, 1)

        # --- Phase-2 features (f16–f19) ---

        # f16: cumulative arc-length / frame diagonal — plane travels far more than drone/bird
        cumulative_length_norm = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 2:
            segs_full = np.linalg.norm(pts_arr_full[1:] - pts_arr_full[:-1], axis=1)
            cumulative_length_norm = float(np.sum(segs_full))  # already in normalized coords

        # f17: variance of direction angles across the path — erratic = bird/drone; steady = plane
        direction_variance = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 3:
            vecs_f = pts_arr_full[1:] - pts_arr_full[:-1]
            mags_f = np.linalg.norm(vecs_f, axis=1)
            valid_f = mags_f > 1e-6
            if np.sum(valid_f) >= 2:
                # angle of each step relative to x-axis, in [-π, π]
                angles_f = np.arctan2(vecs_f[valid_f, 1], vecs_f[valid_f, 0])
                # circular variance: 1 - |mean of unit complex|
                c = np.mean(np.cos(angles_f))
                s = np.mean(np.sin(angles_f))
                direction_variance = float(1.0 - np.sqrt(c * c + s * s))

        # f18: speed range normalized by mean — plane constant ≈ 0; drone/bird variable > 0
        speed_range_norm = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 3:
            n_s = len(pts_arr_full) - 1
            dt_s = _build_dt_array(n_s)
            diffs_s = pts_arr_full[1:] - pts_arr_full[:-1]
            speeds_s = np.linalg.norm(diffs_s, axis=1) / np.maximum(dt_s, 1e-6)
            if len(speeds_s) >= 2:
                mean_s = float(np.mean(speeds_s))
                speed_range_norm = float(np.ptp(speeds_s)) / (mean_s + 1e-6)

        # f19: lateral spread — std of cross-track distance from the overall displacement axis
        # plane: very small (straight line); bird/drone: larger (zigzag/hovering)
        path_lateral_spread = 0.0
        if pts_arr_full is not None and len(pts_arr_full) >= 4:
            start_pt = pts_arr_full[0]
            end_pt = pts_arr_full[-1]
            axis = end_pt - start_pt
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-6:
                axis_unit = axis / axis_len
                # perpendicular unit (rotate 90°)
                perp = np.array([-axis_unit[1], axis_unit[0]])
                offsets = pts_arr_full - start_pt
                cross_track = np.abs(offsets @ perp)
                path_lateral_spread = float(np.std(cross_track))

        # Compute median_dt_frames for audit (median frame gap across all steps)
        median_dt_frames = 1.0
        if len(pwf) >= 2:
            all_gaps = [max(1, min(abs(int(pwf[i + 1][2]) - int(pwf[i][2])), 5)) for i in range(len(pwf) - 1)]
            median_dt_frames = float(np.median(all_gaps))

        # Annotate canonical_meta with v2 timing fields so _append_sample can write them
        canonical_meta = dict(canonical_meta)  # don't mutate original
        canonical_meta["processing_fps"] = processing_fps
        canonical_meta["median_dt_frames"] = median_dt_frames

        return [
            vel_mag, acc_mag, smoothness, area_var,
            bbox_area, avg_area, aspect, y_from_horizon_norm,
            turn_rate, velocity_cv, jitter_index, area_vs_sky_y,
            path_straightness,
            hover_ratio, vertical_velocity_ratio, area_oscillation_rate,
            cumulative_length_norm, direction_variance, speed_range_norm, path_lateral_spread,
        ], canonical_meta

    def _normalize_rect(self, rect):
        if rect is None:
            return None
        if len(rect) >= 4:
            return rect[0], rect[1], rect[2], rect[3]
        return None

    def _estimate_speed_norm(self, path_data, frame_shape, canonical_context=None):
        """Return mean speed in normalized units per second (v2: time-normalized)."""
        canonical_context = canonical_context or {}
        processing_fps = float(canonical_context.get("processing_fps") or 30.0)
        processing_fps = max(5.0, min(120.0, processing_fps))

        smoothed_pts = path_data.get("smoothed_points", path_data.get("points", []))
        if not smoothed_pts or len(smoothed_pts) < 2:
            return 0.0
        pts_payload = canonicalize_trajectory(
            points=list(smoothed_pts),
            frame_shape=frame_shape,
            horizon_points=canonical_context.get("horizon_points"),
            camera_name=canonical_context.get("camera_name"),
            point_resolver=canonical_context.get("point_resolver"),
        )
        if len(pts_payload) < 2:
            return 0.0
        pts_arr = np.array([(pt["x_norm"], pt["y_norm"]) for pt in pts_payload], dtype=np.float32)
        recent = pts_arr[-5:] if len(pts_arr) >= 5 else pts_arr
        if len(recent) < 2:
            return 0.0

        # Build dt array using points_with_frame if available
        pwf = list(path_data.get("points_with_frame", []))
        n_steps = len(recent) - 1
        dt_arr = np.ones(n_steps, dtype=np.float32) / processing_fps
        if len(pwf) >= n_steps + 1:
            tail = pwf[-(n_steps + 1):]
            for i in range(n_steps):
                gap = abs(int(tail[i + 1][2]) - int(tail[i][2]))
                gap = max(1, min(gap, 5))
                dt_arr[i] = float(gap) / processing_fps

        diffs = recent[1:] - recent[:-1]
        speeds = np.linalg.norm(diffs, axis=1) / np.maximum(dt_arr, 1e-6)
        if len(speeds) == 0:
            return 0.0
        return float(np.mean(speeds))

    def update(self, tracked_objs, graph_manager, frame_shape, raw_frame=None, prev_frame=None, canonical_context=None):
        self.predictions = {}
        self.confirmed_ids = set()
        if self.pause_labeling:
            self.current_candidate = None
            return self.predictions, self.confirmed_ids

        current_ids = set()
        candidates = []
        now = time.time()
        work = []

        for obj_id, (rect, _is_real) in tracked_objs.items():
            norm_rect = self._normalize_rect(rect)
            if norm_rect is None:
                continue
            current_ids.add(obj_id)
            path_data = graph_manager.paths.get(obj_id)
            if not path_data:
                continue

            speed_norm = self._estimate_speed_norm(path_data, frame_shape, canonical_context=canonical_context)
            if speed_norm >= self.very_fast_vel_thresh:
                min_points = self.min_path_points_very_fast
            elif speed_norm >= self.fast_vel_thresh:
                min_points = self.min_path_points_fast
            else:
                min_points = self.min_path_points_slow

            allow_unvalidated = self.prompt_mode == "all" or speed_norm >= self.very_fast_vel_thresh
            if not path_data.get("validated", False) and not allow_unvalidated:
                continue

            features, canonical_meta = self._extract_features(
                norm_rect,
                path_data,
                frame_shape,
                canonical_context=canonical_context,
            )
            if self.current_candidate and obj_id == self.current_candidate["obj_id"]:
                if not self.current_candidate.get("frozen", False):
                    self.current_candidate["rect"] = norm_rect
                    self.current_candidate["features"] = features
                    self.current_candidate["frame_shape"] = frame_shape
                    self.current_candidate["canonical_meta"] = canonical_meta
            work.append(
                (obj_id, norm_rect, path_data, features, canonical_meta, min_points)
            )

        preds = self._predict_batch([w[3] for w in work])

        for (obj_id, norm_rect, path_data, features, canonical_meta, min_points), (
            pred_label,
            pred_conf,
        ) in zip(work, preds):
            # โหมด ALL (กด t): ไม่ส่งผลโมเดลไป HUD / ไม่ยืนยันโดรนจากโมเดล — ให้ถามมนุษย์ก่อน
            # ยังเรียก _predict เพื่อใช้คะแนนจัดคิว (pred_conf) และโหมด uncertain
            if self.prompt_mode == "all":
                self.drone_streaks[obj_id] = 0
            else:
                if pred_label is not None:
                    self.predictions[obj_id] = (pred_label, pred_conf)

                if pred_label == self.class_names[0] and pred_conf is not None and pred_conf >= self.confirm_conf_threshold:
                    self.drone_streaks[obj_id] = self.drone_streaks.get(obj_id, 0) + 1
                else:
                    self.drone_streaks[obj_id] = 0

                if self.drone_streaks.get(obj_id, 0) >= self.confirm_frames:
                    self.confirmed_ids.add(obj_id)

            if self.prompt_mode != "off":
                points = path_data.get("points") or []
                path_len = len(points)
                can_prompt = path_len >= min_points
                if not can_prompt:
                    continue
                if self.prompt_mode == "uncertain":
                    if (not self.is_fitted) or (len(self.samples) < self.min_samples):
                        continue
                    if pred_conf is None or pred_conf >= self.skip_conf_threshold:
                        continue
                if obj_id in self.recent_labels and (now - self.recent_labels[obj_id]) < self.obj_cooldown:
                    continue
                if self.prompt_mode != "all":
                    if pred_label is not None and pred_conf is not None and pred_conf >= self.skip_conf_threshold:
                        continue
                area = norm_rect[2] * norm_rect[3]
                score = area * (1.0 + (1.0 - (pred_conf if pred_conf is not None else 0.0)))
                candidates.append((score, obj_id, norm_rect, features, canonical_meta))

        for obj_id in list(self.drone_streaks.keys()):
            if obj_id not in current_ids:
                del self.drone_streaks[obj_id]

        if self.prompt_mode != "off":
            if not self.defer_clear:
                if self.queue:
                    self.current_candidate = self.queue[0]
                if candidates and len(self.queue) < self.max_queue:
                    candidates.sort(reverse=True, key=lambda x: x[0])
                    for _, obj_id, rect, features, cmeta in candidates:
                        if any(item["obj_id"] == obj_id for item in self.queue):
                            continue
                        if len(self.queue) >= self.max_queue:
                            break
                        snapshot = {
                            "obj_id": obj_id,
                            "rect": rect,
                            "features": features,
                            "frame_shape": frame_shape,
                            "canonical_meta": cmeta,
                            "frozen": False,
                        }
                        if raw_frame is not None:
                            if prev_frame is None or self._has_motion_in_rect(prev_frame, raw_frame, rect):
                                snapshot["frozen"] = True
                                snapshot["raw_frame"] = raw_frame.copy()
                                snapshot["prev_frame"] = prev_frame.copy() if prev_frame is not None else None
                            else:
                                continue
                        self.queue.append(snapshot)
                        if self.current_candidate is None:
                            self.current_candidate = self.queue[0]
        if self.prompt_mode != "off" and not self.queue:
            self.current_candidate = None
        if self.prompt_mode == "off":
            self.current_candidate = None
        if self.defer_clear and time.time() >= self.flash_until:
            if self.pending_pop and self.queue and self.queue[0].get("obj_id") == self.pending_pop_id:
                self.queue.pop(0)
            self.pending_pop = False
            self.pending_pop_id = None
            self.current_candidate = self.queue[0] if self.queue else None
            self.defer_clear = False
            self.flash_label = None

        return self.predictions, self.confirmed_ids

    def draw_prompt(self, frame, raw_frame=None):
        self.button_regions = []
        if self.pause_labeling:
            return
        if self.prompt_mode == "off" or self.current_candidate is None:
            return

        h, w = frame.shape[:2]
        x, y, bw, bh = self.current_candidate["rect"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)

        labels = [
            ("DRONE", self.class_names[0], (0, 0, 255)),
            ("BIRD", self.class_names[1], (255, 200, 0)),
            ("PLANE", self.class_names[2], (0, 255, 255)),
            ("OTHER", self.class_names[3], (200, 200, 200)),
        ]
        btn_w, btn_h, gap = 140, 44, 10
        text_scale = 0.9
        text_thickness = 2
        total_w = len(labels) * btn_w + (len(labels) - 1) * gap

        start_x = max(5, min(x, w - total_w - 5))
        start_y = max(5, y - btn_h - 10)
        if start_y < 5:
            start_y = min(h - btn_h - 5, y + bh + 10)

        flash_active = self.flash_label is not None and time.time() < self.flash_until
        for i, (txt, label, color) in enumerate(labels):
            x1 = start_x + i * (btn_w + gap)
            y1 = start_y
            x2 = x1 + btn_w
            y2 = y1 + btn_h
            self.button_regions.append((x1, y1, x2, y2, label))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            border_color = (0, 255, 0) if (flash_active and label == self.flash_label) else (0, 0, 0)
            border_thickness = 3 if (flash_active and label == self.flash_label) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            tx = x1 + (btn_w - tw) // 2
            ty = y1 + (btn_h + th) // 2 - 4
            cv2.putText(frame, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness)

        source_frame = raw_frame
        if self.current_candidate is not None and self.current_candidate.get("raw_frame") is not None:
            source_frame = self.current_candidate["raw_frame"]
        if source_frame is not None:
            crop = self._crop_with_padding(source_frame, self.current_candidate["rect"])
            if crop is not None and crop.size > 0:
                target_h = 160 if h >= 1080 else 120
                aspect = crop.shape[1] / max(1, crop.shape[0])
                target_w = int(target_h * aspect)

                if target_w > 0 and target_h > 0:
                    crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    x1 = start_x - target_w - 10
                    y1 = start_y
                    if x1 < 5:
                        x1 = start_x + total_w + 10
                    if x1 + target_w > w:
                        x1 = max(5, w - target_w - 5)
                    if y1 + target_h > h:
                        y1 = max(5, h - target_h - 5)

                    frame[y1:y1+target_h, x1:x1+target_w] = crop_resized
                    cv2.rectangle(frame, (x1, y1), (x1 + target_w, y1 + target_h), (0, 0, 0), 2)

    def handle_key(self, key, original_frame):
        if self.prompt_mode == "off" or self.current_candidate is None:
            return False
        if self.pause_labeling:
            return False
        if self.defer_clear and time.time() < self.flash_until:
            return False

        obj_id = self.current_candidate["obj_id"]
        if key == self.skip_key:
            self.recent_labels[obj_id] = time.time()
            if self.queue and self.queue[0].get("obj_id") == obj_id:
                self.queue.pop(0)
            self.current_candidate = self.queue[0] if self.queue else None
            return True

        if key not in self.label_keys:
            return False

        label = self.label_keys[key]
        frame_for_save = original_frame
        if self.current_candidate.get("raw_frame") is not None:
            frame_for_save = self.current_candidate["raw_frame"]
        self._save_sample(label, self.current_candidate, frame_for_save)
        self.recent_labels[obj_id] = time.time()
        self.flash_label = label
        self.flash_until = time.time() + 0.6
        self.defer_clear = True
        self.pending_pop = True
        self.pending_pop_id = obj_id
        return True

    def handle_mouse_click(self, x, y, original_frame, prev_frame=None):
        if self.prompt_mode == "off" or self.current_candidate is None:
            return False
        if self.pause_labeling:
            return False
        if self.defer_clear and time.time() < self.flash_until:
            return False

        rect = self.current_candidate["rect"]

        check_curr = original_frame
        check_prev = prev_frame
        if self.current_candidate.get("raw_frame") is not None:
            check_curr = self.current_candidate["raw_frame"]
            check_prev = self.current_candidate.get("prev_frame")

        if check_prev is not None:
            if not self._has_motion_in_rect(check_prev, check_curr, rect):
                print("Active Label Trainer: motion too low, skip save")
                return False

        for x1, y1, x2, y2, label in self.button_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                obj_id = self.current_candidate["obj_id"]
                frame_for_save = original_frame
                if self.current_candidate.get("raw_frame") is not None:
                    frame_for_save = self.current_candidate["raw_frame"]
                self._save_sample(label, self.current_candidate, frame_for_save)
                self.recent_labels[obj_id] = time.time()
                self.flash_label = label
                self.flash_until = time.time() + 0.6
                self.defer_clear = True
                self.pending_pop = True
                self.pending_pop_id = obj_id
                return True
        return False

    def handle_mouse_skip(self):
        if self.prompt_mode == "off" or self.current_candidate is None:
            return False
        obj_id = self.current_candidate["obj_id"]
        self.recent_labels[obj_id] = time.time()
        if self.queue and self.queue[0].get("obj_id") == obj_id:
            self.queue.pop(0)
        self.current_candidate = self.queue[0] if self.queue else None
        return True

    def _has_motion_in_rect(self, prev_frame, curr_frame, rect, min_ratio=0.01):
        if prev_frame is None or curr_frame is None:
            return True
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        if w <= 0 or h <= 0:
            return False

        roi_prev = prev_frame[y:y+h, x:x+w]
        roi_curr = curr_frame[y:y+h, x:x+w]
        if roi_prev.size == 0 or roi_curr.size == 0:
            return False

        g_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
        g_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g_prev, g_curr)
        _, mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)

        motion_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return motion_ratio >= min_ratio

    def _crop_with_padding(self, frame, rect):
        h, w = frame.shape[:2]
        x, y, bw, bh = rect

        pad_x = max(20, int(bw * self.padding_ratio))
        pad_y = max(20, int(bh * self.padding_ratio))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _save_sample(self, label, candidate, original_frame):
        timestamp_ms = int(time.time() * 1000)
        obj_id = candidate["obj_id"]
        rect = candidate["rect"]
        features = candidate["features"]
        frame_shape = candidate["frame_shape"]
        canonical_meta = candidate.get("canonical_meta") or {}

        image_path = ""
        if original_frame is not None:
            crop = self._crop_with_padding(original_frame, rect)
            if crop is not None and crop.size > 0:
                label_dir = os.path.join(self.images_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                filename = f"{timestamp_ms}_id{obj_id}.jpg"
                image_path = os.path.join(label_dir, filename)
                cv2.imwrite(image_path, crop)

        self._append_sample(
            timestamp_ms,
            label,
            obj_id,
            rect,
            frame_shape,
            features,
            image_path,
            canonical_meta=canonical_meta,
        )
        split = self._assign_split(timestamp_ms=timestamp_ms)
        if split == "test":
            self.test_samples.append(features)
            self.test_labels.append(label)
        else:
            self.samples.append(features)
            self.sample_labels.append(label)
            self._train_incremental(features, label)

        self.has_new_data = True  # ตั้งค่า flag เมื่อมีข้อมูลใหม่
        self.new_labels_since_eval += 1
        if self.new_labels_since_eval >= self.eval_every_n:
            self._evaluate_accuracy()
            self.new_labels_since_eval = 0

    def _train_incremental(self, features, label):
        # ไม่ retrain ระหว่างใช้งาน แค่ตั้งค่า flag (has_new_data ถูกตั้งค่าใน _save_sample() แล้ว)
        # Retrain จะทำเมื่อปิดโปรแกรมใน retrain_and_save()
        pass




