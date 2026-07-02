"""eval_path_model_v2.py — Evaluate path-classification model from canonical_v2 labels.csv

Usage:
    python eval_path_model_v2.py [--csv PATH] [--model PATH] [--min-per-class N]

The script reads labels.csv (schema_version == canonical_v2), splits train/test
deterministically (20 % test, same logic as ActiveLabelTrainer._assign_split),
trains a Random Forest, and prints accuracy broken down by camera_name and by class.

For grouped train/test split (leave-one-camera-out), pass --grouped.
"""

import argparse
import os
import sys
import csv
import json
from collections import defaultdict

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

FEATURE_DIM = 16
SCHEMA_VERSION = "canonical_v2"
TEST_RATIO = 0.20
MIN_SAMPLES_DEFAULT = 12


def load_csv(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print(f"[eval] CSV not found: {csv_path}")
        return rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sv = row.get("schema_version", "")
            if sv != SCHEMA_VERSION:
                continue
            label = row.get("label", "").strip()
            if not label:
                continue
            try:
                feats = [float(row.get(f"f{i}", 0.0)) for i in range(FEATURE_DIM)]
            except (ValueError, TypeError):
                continue
            rows.append({
                "label": label,
                "feats": feats,
                "camera_name": row.get("camera_name", "unknown"),
                "source_camera_name": row.get("source_camera_name", ""),
                "timestamp": row.get("timestamp", ""),
                "processing_fps": float(row.get("processing_fps", 30.0) or 30.0),
            })
    return rows


def assign_split(timestamp_str, idx, test_ratio=TEST_RATIO):
    try:
        bucket = int(float(timestamp_str)) % 100
        return "test" if bucket < int(test_ratio * 100) else "train"
    except Exception:
        period = max(1, int(round(1.0 / max(1e-6, test_ratio))))
        return "test" if (idx % period) == 0 else "train"


def print_accuracy_by_camera(rows, preds, labels, camera_names):
    cameras = sorted(set(camera_names))
    print("\n=== Accuracy by camera ===")
    for cam in cameras:
        idxs = [i for i, c in enumerate(camera_names) if c == cam]
        if not idxs:
            continue
        cam_preds = [preds[i] for i in idxs]
        cam_labels = [labels[i] for i in idxs]
        correct = sum(p == l for p, l in zip(cam_preds, cam_labels))
        total = len(idxs)
        print(f"  {cam:30s}  {correct}/{total}  ({100.0*correct/max(1,total):.1f}%)")


def run_standard_split(rows, min_per_class, grouped):
    if not SKLEARN_OK:
        print("[eval] scikit-learn not available")
        return

    train_X, train_y = [], []
    test_X, test_y, test_cam = [], [], []

    if grouped:
        # Leave-one-camera-out: hold out one camera at a time for test
        cameras = sorted(set(r["camera_name"] for r in rows))
        print(f"\n[eval] Grouped (leave-one-camera-out) using cameras: {cameras}")
        all_preds, all_labels, all_cams = [], [], []
        for hold_cam in cameras:
            tr_X = [r["feats"] for r in rows if r["camera_name"] != hold_cam]
            tr_y = [r["label"] for r in rows if r["camera_name"] != hold_cam]
            te_X = [r["feats"] for r in rows if r["camera_name"] == hold_cam]
            te_y = [r["label"] for r in rows if r["camera_name"] == hold_cam]
            if not tr_X or not te_X:
                continue
            classes = sorted(set(tr_y))
            if len(classes) < 2:
                print(f"  [skip {hold_cam}] not enough classes in train")
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(np.array(tr_X, dtype=np.float32))
            Xte = scaler.transform(np.array(te_X, dtype=np.float32))
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
            clf.fit(Xtr, tr_y)
            preds = clf.predict(Xte)
            correct = np.sum(preds == np.array(te_y))
            print(f"  hold_out={hold_cam:30s}  {correct}/{len(te_y)}  ({100.0*correct/max(1,len(te_y)):.1f}%)")
            all_preds.extend(preds)
            all_labels.extend(te_y)
            all_cams.extend([hold_cam] * len(te_y))
        if all_labels:
            overall = sum(p == l for p, l in zip(all_preds, all_labels))
            print(f"\nOverall LOCO accuracy: {overall}/{len(all_labels)}  ({100.0*overall/max(1,len(all_labels)):.1f}%)")
        return

    # Standard random split
    for idx, row in enumerate(rows):
        split = assign_split(row["timestamp"], idx)
        if split == "test":
            test_X.append(row["feats"])
            test_y.append(row["label"])
            test_cam.append(row["camera_name"])
        else:
            train_X.append(row["feats"])
            train_y.append(row["label"])

    print(f"\n[eval] Train: {len(train_X)}  Test: {len(test_X)}")
    if len(train_X) < min_per_class * 2:
        print(f"[eval] Not enough train samples (need at least {min_per_class * 2})")
        return
    if len(test_X) < 1:
        print("[eval] No test samples")
        return

    classes = sorted(set(train_y))
    class_counts = {c: train_y.count(c) for c in classes}
    print(f"[eval] Classes in train: {class_counts}")
    sparse = [c for c, n in class_counts.items() if n < min_per_class]
    if sparse:
        print(f"[eval] WARNING: sparse classes (< {min_per_class} samples): {sparse}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.array(train_X, dtype=np.float32))
    Xte = scaler.transform(np.array(test_X, dtype=np.float32))

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(Xtr, train_y)
    preds = clf.predict(Xte)

    correct = int(np.sum(preds == np.array(test_y)))
    total = len(test_y)
    print(f"\n=== Overall accuracy: {correct}/{total} ({100.0*correct/max(1,total):.1f}%) ===")

    print("\n=== Classification report ===")
    print(classification_report(test_y, preds, zero_division=0))

    print_accuracy_by_camera(rows, preds, test_y, test_cam)

    # Per-class FPS distribution (helps spot domain shift)
    print("\n=== Median processing_fps by camera (all data) ===")
    fps_by_cam = defaultdict(list)
    for r in rows:
        fps_by_cam[r["camera_name"]].append(r["processing_fps"])
    for cam in sorted(fps_by_cam):
        vals = fps_by_cam[cam]
        print(f"  {cam:30s}  median={np.median(vals):.1f}  min={np.min(vals):.1f}  max={np.max(vals):.1f}  n={len(vals)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate path model v2")
    parser.add_argument("--csv", default=None, help="Path to labels.csv")
    parser.add_argument("--dir", default=None, help="Dataset directory (auto-discovers labels.csv)")
    parser.add_argument("--min-per-class", type=int, default=MIN_SAMPLES_DEFAULT)
    parser.add_argument("--grouped", action="store_true", help="Leave-one-camera-out evaluation")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        # Auto-discover from canonical layout
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "datasets", "path", "canonical_v2_train", "labels.csv"),
            os.path.join(base, "datasets", "path", "canonical_v2_val", "labels.csv"),
        ]
        if args.dir:
            candidates.insert(0, os.path.join(args.dir, "labels.csv"))
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        if csv_path is None:
            print(f"[eval] Could not find labels.csv. Searched: {candidates}")
            sys.exit(1)

    print(f"[eval] Loading: {csv_path}")
    rows = load_csv(csv_path)
    print(f"[eval] Loaded {len(rows)} v2 rows")

    if not rows:
        print("[eval] No data. Collect labels with the main app first.")
        sys.exit(0)

    run_standard_split(rows, args.min_per_class, args.grouped)


if __name__ == "__main__":
    main()
