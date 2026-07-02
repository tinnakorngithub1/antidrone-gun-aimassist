#!/usr/bin/env python3
"""
Re-save a YOLO checkpoint with current ultralytics symbols.

Useful when an older/custom .pt checkpoint contains pickled loss classes
such as E2ELoss or v10DetectLoss that newer ultralytics versions no longer
expose directly.

python3 resave_yolo_checkpoint.py yolo26_60epoch_thermal.pt
python3 convert_yolo_to_tensorrt.py yolo26_60epoch_thermal_clean.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Install with: pip install ultralytics")
    sys.exit(1)


def install_ultralytics_checkpoint_compat(verbose=True):
    """Alias old loss names to the new E2EDetectLoss symbol when available."""
    aliases_applied = []
    try:
        from ultralytics.utils import loss as lossmod
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not import ultralytics loss module for compatibility patch: {e}")
        return aliases_applied

    replacement = getattr(lossmod, "E2EDetectLoss", None)
    if replacement is None:
        return aliases_applied

    if not hasattr(lossmod, "E2ELoss"):
        lossmod.E2ELoss = replacement
        aliases_applied.append("E2ELoss -> E2EDetectLoss")
    if not hasattr(lossmod, "v10DetectLoss"):
        lossmod.v10DetectLoss = replacement
        aliases_applied.append("v10DetectLoss -> E2EDetectLoss")

    if verbose and aliases_applied:
        print("🔧 Applied ultralytics checkpoint compatibility aliases:")
        for alias in aliases_applied:
            print(f"   - {alias}")
    return aliases_applied


def resave_checkpoint(model_path: Path, output_path: Path) -> Path:
    install_ultralytics_checkpoint_compat(verbose=True)
    print(f"📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"💾 Saving clean checkpoint: {output_path}")
    model.save(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Re-save a YOLO checkpoint with compatibility aliases applied."
    )
    parser.add_argument("model", type=str, help="Path to input .pt model")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output .pt path (default: <input>_clean.pt)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else model_path.with_name(f"{model_path.stem}_clean.pt")
    saved = resave_checkpoint(model_path, output_path)
    print(f"✅ Saved: {saved}")


if __name__ == "__main__":
    main()

