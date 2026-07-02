"""
MLP เล็กสำหรับ learned aim controller.
Input: state 10 มิติ (err_pan, err_tilt, error_deg, last_delta_*, dt, d_pan, d_tilt, current_pan_deg, current_tilt_deg)
Output: (delta_pan, delta_tilt) หน่วย degree
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

STATE_DIM = 10  # + current_pan_deg, current_tilt_deg
OUTPUT_DIM = 2

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

from .constants import (
    D_MAX_DEG_PER_SEC,
    DT_MAX_SEC,
    E_MAX_DEG,
    ERROR_DEG_MAX,
    DELTA_MAX_DEG,
    PAN_RANGE_DEG,
    TILT_RANGE_DEG,
)


def _clip(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def normalize_state(
    err_pan: float,
    err_tilt: float,
    error_deg: float,
    last_delta_pan: float,
    last_delta_tilt: float,
    dt: float,
    d_pan: float,
    d_tilt: float,
    current_pan_deg: float = 0.0,
    current_tilt_deg: float = 0.0,
) -> List[float]:
    """
    Clip แล้ว scale เป็นช่วงประมาณ [-1, 1] สำหรับแต่ละมิติ.
    คืนค่า list ความยาว STATE_DIM (10).
    """
    ep = _clip(err_pan, -E_MAX_DEG, E_MAX_DEG) / E_MAX_DEG
    et = _clip(err_tilt, -E_MAX_DEG, E_MAX_DEG) / E_MAX_DEG
    e = _clip(error_deg, 0, ERROR_DEG_MAX) / ERROR_DEG_MAX
    ldp = _clip(last_delta_pan, -DELTA_MAX_DEG, DELTA_MAX_DEG) / DELTA_MAX_DEG
    ldt = _clip(last_delta_tilt, -DELTA_MAX_DEG, DELTA_MAX_DEG) / DELTA_MAX_DEG
    dt_n = _clip(dt, 0, DT_MAX_SEC) / DT_MAX_SEC
    dp = _clip(d_pan, -D_MAX_DEG_PER_SEC, D_MAX_DEG_PER_SEC) / D_MAX_DEG_PER_SEC
    dt_ = _clip(d_tilt, -D_MAX_DEG_PER_SEC, D_MAX_DEG_PER_SEC) / D_MAX_DEG_PER_SEC
    cp = _clip(current_pan_deg, -PAN_RANGE_DEG, PAN_RANGE_DEG) / PAN_RANGE_DEG
    ct = _clip(current_tilt_deg, -TILT_RANGE_DEG, TILT_RANGE_DEG) / TILT_RANGE_DEG
    return [ep, et, e, ldp, ldt, dt_n, dp, dt_, cp, ct]


if TORCH_AVAILABLE:

    class AimControllerMLP(nn.Module):
        def __init__(
            self,
            input_dim: int = STATE_DIM,
            hidden_dims: Tuple[int, ...] = (64, 32),
            output_dim: int = OUTPUT_DIM,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU(inplace=True))
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

    def create_model(
        input_dim: int = STATE_DIM,
        hidden_dims: Tuple[int, ...] = (64, 32),
    ) -> AimControllerMLP:
        return AimControllerMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=OUTPUT_DIM)

    def load_model(path: Union[str, Path]) -> Optional[AimControllerMLP]:
        path = Path(path)
        if not path.exists() or path.suffix.lower() == ".onnx":
            return None
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and "state_dict" in loaded:
                state_dict = loaded["state_dict"]
                input_dim = int(loaded.get("input_dim", STATE_DIM))
            else:
                state_dict = loaded
                # Infer input_dim from first layer weight (รองรับ .pt เก่า 8 หรือ 10 dim)
                input_dim = int(state_dict["mlp.0.weight"].shape[1])
            model = create_model(input_dim=input_dim)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception:
            return None

    def save_model(model: AimControllerMLP, path: Union[str, Path]) -> None:
        torch.save({"state_dict": model.state_dict(), "input_dim": model.input_dim}, path)

    def predict_delta(
        model: AimControllerMLP,
        state_vector: List[float],
        max_step_deg: float,
    ) -> Tuple[float, float]:
        """Inference: state_vector (ความยาว STATE_DIM, normalized) → (delta_pan, delta_tilt) แล้ว clip ด้วย max_step_deg."""
        x = torch.tensor([state_vector], dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
        delta_pan = float(out[0, 0])
        delta_tilt = float(out[0, 1])
        magnitude = math.hypot(delta_pan, delta_tilt)
        if magnitude > max_step_deg and magnitude > 1e-9:
            scale = max_step_deg / magnitude
            delta_pan *= scale
            delta_tilt *= scale
        return delta_pan, delta_tilt

else:
    AimControllerMLP = None  # type: ignore

    def create_model(*args, **kwargs):  # type: ignore
        return None

    def load_model(path: Union[str, Path]) -> Optional[None]:  # type: ignore
        return None

    def save_model(model: None, path: Union[str, Path]) -> None:
        pass

    def predict_delta(
        model: None,
        state_vector: List[float],
        max_step_deg: float,
    ) -> Tuple[float, float]:
        return 0.0, 0.0


def load_onnx(path: Union[str, Path]) -> Any:
    """โหลด model จาก .onnx คืน InferenceSession หรือ None."""
    if not ONNXRUNTIME_AVAILABLE:
        return None
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".onnx":
        return None
    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception:
        return None


def predict_delta_onnx(
    session: Any,
    state_vector: List[float],
    max_step_deg: float,
) -> Tuple[float, float]:
    """Inference ด้วย ONNX: state_vector (ความยาว STATE_DIM) → (delta_pan, delta_tilt) แล้ว clip."""
    if session is None or not ONNXRUNTIME_AVAILABLE:
        return 0.0, 0.0
    inp = session.get_inputs()[0]
    name = inp.name
    x = np.array([state_vector], dtype=np.float32)
    out = session.run(None, {name: x})
    delta_pan = float(out[0][0, 0])
    delta_tilt = float(out[0][0, 1])
    magnitude = math.hypot(delta_pan, delta_tilt)
    if magnitude > max_step_deg and magnitude > 1e-9:
        scale = max_step_deg / magnitude
        delta_pan *= scale
        delta_tilt *= scale
    return delta_pan, delta_tilt
