"""
Export PyTorch .pt model เป็น .onnx สำหรับ inference เบา (optional).
รัน: python -m aim_controller_model.export_onnx [path_to.pt] [output.onnx]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aim_controller_model.model import STATE_DIM, create_model, load_model


def export(pt_path: Path, onnx_path: Path) -> bool:
    try:
        import torch
    except ImportError:
        print("PyTorch required for export.")
        return False

    model = load_model(pt_path)
    if model is None:
        model = create_model()
    model.eval()

    dummy = torch.zeros(1, STATE_DIM)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["state"],
        output_names=["delta"],
        dynamic_axes={"state": {0: "batch"}, "delta": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported {pt_path} -> {onnx_path}")
    return True


if __name__ == "__main__":
    pt = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "aim_controller_model" / "aim_model.pt"
    onnx = Path(sys.argv[2]) if len(sys.argv) > 2 else pt.with_suffix(".onnx")
    export(pt, onnx)
