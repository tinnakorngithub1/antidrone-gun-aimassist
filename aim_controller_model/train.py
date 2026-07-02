"""
ฝึก MLP จากข้อมูล transition ที่เก็บระหว่างรัน (behavioral cloning จาก good transitions).
เรียกจาก atexit เมื่อปิดโปรแกรม: โหลด buffer ที่เขียนลง .npz แล้วเทรน แล้วบันทึก .pt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# โปรเจกต์ root = parent ของ aim_controller_model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aim_controller_model.constants import (
    D_MAX_DEG_PER_SEC,
    DT_MAX_SEC,
    E_MAX_DEG,
    ERROR_DEG_MAX,
    DELTA_MAX_DEG,
)
from aim_controller_model.model import (
    STATE_DIM,
    create_model,
    load_model,
    normalize_state,
    save_model,
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _is_good_transition(
    next_error_deg: float,
    error_deg: float,
    threshold_red_deg: float,
    threshold_orange_deg: float,
) -> bool:
    """Good = แดง (next ≤ red) หรือ ส้ม (next ≤ orange) หรือ error ลดลง."""
    if next_error_deg <= threshold_red_deg:
        return True
    if next_error_deg <= threshold_orange_deg:
        return True
    if next_error_deg < error_deg:
        return True
    return False


def load_transitions_from_npz(npz_path: Path) -> List[Dict[str, Any]]:
    """โหลด list ของ transition จากไฟล์ .npz ที่บันทึกจาก buffer."""
    if not npz_path.exists():
        return []
    data = np.load(npz_path, allow_pickle=True)
    # รูปแบบ: states (N, 8), actions (N, 2), next_states (N, 3) = (err_pan, err_tilt, error_deg)
    states = data.get("states")
    actions = data.get("actions")
    next_states = data.get("next_states")
    if states is None or actions is None or next_states is None:
        return []
    out = []
    for i in range(len(states)):
        st = states[i]
        ns = next_states[i]
        next_err_deg = float(ns[2]) if len(ns) >= 3 else float(np.hypot(ns[0], ns[1]))
        err_deg = float(st[2]) if len(st) >= 3 else float(np.hypot(st[0], st[1]))
        out.append({
            "state": st.tolist() if hasattr(st, "tolist") else list(st),
            "action": actions[i].tolist() if hasattr(actions[i], "tolist") else list(actions[i]),
            "next_state": ns.tolist() if hasattr(ns, "tolist") else list(ns),
            "next_error_deg": next_err_deg,
            "error_deg": err_deg,
        })
    return out


def filter_good_transitions(
    transitions: List[Dict[str, Any]],
    threshold_red_deg: float,
    threshold_orange_deg: float,
) -> List[Dict[str, Any]]:
    good = []
    for t in transitions:
        next_err = t.get("next_error_deg")
        err = t.get("error_deg")
        if next_err is None or err is None:
            continue
        if _is_good_transition(next_err, err, threshold_red_deg, threshold_orange_deg):
            good.append(t)
    return good


def train_from_npz(
    npz_path: Union[str, Path],
    model_path: Union[str, Path],
    threshold_red_deg: float = 0.35,
    threshold_orange_deg: float = 0.7,
    min_transitions: int = 500,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_red: float = 1.5,
    weight_orange: float = 1.0,
) -> bool:
    """
    โหลด transition จาก npz, กรอง good, เทรน behavioral cloning (MSE), บันทึก model.
    คืน True ถ้าเทรนและบันทึกสำเร็จ.
    """
    if not TORCH_AVAILABLE:
        print("aim_controller_model.train: PyTorch not installed, skip training.")
        return False

    npz_path = Path(npz_path)
    model_path = Path(model_path)
    if not npz_path.exists():
        print(f"aim_controller_model.train: no data file {npz_path}, skip.")
        return False

    transitions = load_transitions_from_npz(npz_path)
    if len(transitions) < min_transitions:
        print(f"aim_controller_model.train: only {len(transitions)} transitions (need {min_transitions}), skip.")
        return False

    good = filter_good_transitions(transitions, threshold_red_deg, threshold_orange_deg)
    if len(good) == 0:
        print("aim_controller_model.train: no good transitions after filter, skip.")
        return False

    # น้ำหนักต่อ sample: แดง (แม่นมาก) > ส้ม (แม่นรอง) > ตามทัน
    def _weight_for(t: Dict[str, Any]) -> float:
        ne = t.get("next_error_deg") or 0.0
        if ne <= threshold_red_deg:
            return weight_red
        if ne <= threshold_orange_deg:
            return weight_orange
        return 1.0

    # Build state/action/weight arrays; state ใน buffer เป็น raw 8 หรือ 10 ค่า → normalize ก่อนเทรน
    states_norm = []
    actions = []
    weights = []
    for t in good:
        st = t["state"]
        if len(st) >= 10:
            st = normalize_state(
                float(st[0]), float(st[1]), float(st[2]), float(st[3]),
                float(st[4]), float(st[5]), float(st[6]), float(st[7]),
                float(st[8]), float(st[9]),
            )
        elif len(st) >= 8:
            st = normalize_state(
                float(st[0]), float(st[1]), float(st[2]), float(st[3]),
                float(st[4]), float(st[5]), float(st[6]), float(st[7]),
                0.0, 0.0,
            )
        else:
            continue
        states_norm.append(st)
        actions.append(t["action"])
        weights.append(_weight_for(t))

    X = torch.tensor(states_norm, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)
    W = torch.tensor(weights, dtype=torch.float32)

    model = load_model(model_path)
    if model is None:
        model = create_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X, Y, W)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for xi, yi, wi in loader:
            opt.zero_grad()
            pred = model(xi)
            # Weighted MSE: ให้น้ำหนัก transition แดงมากกว่าส้ม
            sq = (pred - yi) ** 2
            loss = (wi * sq.sum(dim=1)).sum() / (wi.sum() + 1e-9)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"  aim_controller epoch {epoch+1}/{epochs} loss={total_loss/max(1,n_batches):.6f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)
    print(f"aim_controller_model: saved model to {model_path} ({len(good)} good transitions)")
    return True


def run_retrain(
    buffer_npz_path: Union[str, Path],
    model_path: Union[str, Path],
    threshold_red_deg: float,
    threshold_orange_deg: float,
    min_transitions: int,
    weight_red: float = 1.5,
    weight_orange: float = 1.0,
) -> bool:
    """เรียกจาก atexit: เทรนจาก buffer ที่บันทึกแล้ว."""
    return train_from_npz(
        buffer_npz_path,
        model_path,
        threshold_red_deg=threshold_red_deg,
        threshold_orange_deg=threshold_orange_deg,
        min_transitions=min_transitions,
        weight_red=weight_red,
        weight_orange=weight_orange,
    )


