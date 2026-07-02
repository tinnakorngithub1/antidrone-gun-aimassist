"""
เทรน MLP 2→32→2 จาก teach log (cam4_teach_*.jsonl) แล้ว save weights เป็น .npz
ใช้เมื่อกด R ใน run_cam4_sim_only หรือรัน: python train_mlp.py
Dependency: sklearn (pip install scikit-learn)
"""

import json
from pathlib import Path

import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
except ImportError:
    MLPRegressor = None

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration_data"
MLP_SAVE_PATH = CALIBRATION_DIR / "cam4_arm_mlp.npz"


def train_and_save() -> bool:
    """โหลด cam4_teach_*.jsonl ทั้งหมด, fit MLPRegressor(32), save .npz. คืน True ถ้าสำเร็จ."""
    if MLPRegressor is None:
        print("train_mlp: sklearn not installed. pip install scikit-learn")
        return False

    paths = sorted(CALIBRATION_DIR.glob("cam4_teach_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        print("train_mlp: no cam4_teach_*.jsonl in", CALIBRATION_DIR)
        return False

    X_list = []
    Y_list = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    px = r.get("px")
                    py = r.get("py")
                    pan = r.get("pan_deg")
                    tilt = r.get("tilt_deg")
                    if px is None or py is None or pan is None or tilt is None:
                        continue
                    X_list.append([float(px), float(py)])
                    Y_list.append([float(pan), float(tilt)])
                except (json.JSONDecodeError, TypeError):
                    continue

    if not X_list:
        print("train_mlp: no valid records in teach logs")
        return False

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)
    n = len(X_list)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    mlp.fit(X, Y)

    W1 = mlp.coefs_[0]   # (2, 32)
    b1 = mlp.intercepts_[0]  # (32,)
    W2 = mlp.coefs_[1]   # (32, 2)
    b2 = mlp.intercepts_[1]  # (2,)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MLP_SAVE_PATH, W1=W1, b1=b1, W2=W2, b2=b2)
    print("train_mlp: saved", n, "samples to", MLP_SAVE_PATH)
    return True


if __name__ == "__main__":
    train_and_save()
