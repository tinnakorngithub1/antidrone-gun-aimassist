#!/usr/bin/env python3
"""
YOLO to TensorRT Engine Converter
สำหรับ Jetson Orin Nano

รองรับการสร้าง engine หลายขนาด (640, 1280, 1920 ฯลฯ)

python3 resave_yolo_checkpoint.py yolo26_60epoch_thermal.pt
python3 convert_yolo_to_tensorrt.py yolo26_60epoch_thermal_clean.pt --imgsz 640

"""

import os
import sys
import argparse
import gc
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Install with: pip install ultralytics torch")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Default image sizes สำหรับ YOLO
DEFAULT_IMG_SIZES = [640, 1280, 1920]

# TensorRT export parameters
TENSORRT_EXPORT_ARGS = {
    'format': 'engine',  # Export as TensorRT engine
    'half': True,        # FP16 precision (เร็วกว่า FP32)
    'int8': False,       # INT8 quantization (เร็วที่สุดแต่ต้องมี calibration data)
    'dynamic': False,    # Dynamic shapes (ถ้า True จะรองรับหลายขนาดแต่ช้ากว่า)
    'simplify': True,    # Simplify ONNX model
    'workspace': 4,      # Workspace size in GB
}

# ============================================================================
# Functions
# ============================================================================

def install_ultralytics_checkpoint_compat(verbose=True):
    """
    ติดตั้ง alias สำหรับ checkpoint รุ่นเก่าที่ pickle class loss ชื่อเดิมไว้
    เช่น E2ELoss / v10DetectLoss แต่ ultralytics รุ่นใหม่เปลี่ยนชื่อเป็น E2EDetectLoss.
    """
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


def load_yolo_model_compat(model_path):
    """โหลด YOLO model พร้อม compatibility patch สำหรับ checkpoint รุ่นเก่า."""
    install_ultralytics_checkpoint_compat(verbose=True)
    return YOLO(str(model_path))

def clear_gpu_memory():
    """ล้าง GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def clear_all_caches():
    """Clear all caches (GPU, Python, TensorRT)"""
    print("🧹 Clearing all caches...")

    # 1. GPU Memory Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # 2. Python garbage collection
    gc.collect()

    # 3. TensorRT Cache
    try:
        from pathlib import Path
        tensorrt_cache = Path.home() / '.nv' / 'ComputeCache'
        if tensorrt_cache.exists():
            import shutil
            cache_size = sum(f.stat().st_size for f in tensorrt_cache.rglob('*') if f.is_file()) / 1024**3
            shutil.rmtree(tensorrt_cache)
            print(f"   TensorRT cache cleared ({cache_size:.2f} GB)")
        else:
            print("   TensorRT cache: not found")
    except Exception as e:
        print(f"   TensorRT cache: could not clear ({e})")

    print("✅ All caches cleared")

def check_environment():
    """ตรวจสอบ environment และ dependencies"""
    print("🔍 Checking environment...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - TensorRT export may fail")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        # Show GPU memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
    except ImportError:
        print("⚠️ TensorRT not found in Python - ultralytics will use TensorRT from system")

    # Check ultralytics
    try:
        from ultralytics import __version__
        print(f"✅ Ultralytics version: {__version__}")
    except:
        print("⚠️ Could not get ultralytics version")

    print()

def convert_to_tensorrt(model_path, output_dir=None, imgsz_list=None, half_precision=True, int8_quantization=False):
    """
    Convert YOLO .pt model เป็น TensorRT .engine

    Args:
        model_path: Path to .pt model file
        output_dir: Output directory (default: same as model file)
        imgsz_list: List of image sizes to export (default: [640, 1280, 1920])
        half_precision: Use FP16 precision (default: True)
        int8_quantization: Use INT8 quantization (default: False, requires calibration)

    Returns:
        List of exported engine file paths
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return []

    if not model_path.suffix == '.pt':
        print(f"⚠️ Warning: Expected .pt file, got {model_path.suffix}")

    # Setup output directory
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Default image sizes
    if imgsz_list is None:
        imgsz_list = DEFAULT_IMG_SIZES

    print(f"📦 Model: {model_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📐 Image sizes: {imgsz_list}")
    print(f"🔧 FP16: {half_precision}, INT8: {int8_quantization}")
    print()

    # Clear all caches before starting
    clear_all_caches()

    exported_files = []

    # Export for each image size
    for idx, imgsz in enumerate(imgsz_list):
        print(f"\n{'='*60}")
        print(f"🔄 Exporting for imgsz={imgsz} ({idx+1}/{len(imgsz_list)})...")
        print(f"{'='*60}")

        # Generate output filename
        model_name = model_path.stem
        output_filename = f"{model_name}_imgsz{imgsz}.engine"
        output_path = output_dir / output_filename

        # Skip if already exists
        if output_path.exists():
            print(f"⏭️  File already exists: {output_filename}")
            print(f"   Delete it first if you want to regenerate")
            exported_files.append(str(output_path))
            continue

        # Clear GPU memory before each export
        print("🧹 Clearing GPU memory before export...")
        clear_gpu_memory()

        # Load model fresh for each export (to avoid memory leaks)
        print("🔄 Loading model...")
        try:
            model = load_yolo_model_compat(model_path)
            # Move model to CPU to avoid GPU memory issues during fuse
            if hasattr(model.model, 'to'):
                model.model = model.model.to('cpu')
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue

        try:
            # 🔥 Strategy: Export to ONNX first (uses less memory), then convert to engine
            print("📤 Step 1: Exporting to ONNX (CPU, low memory)...")
            onnx_args = {
                'format': 'onnx',
                'imgsz': imgsz,
                'simplify': False,
                'device': 'cpu',  # Use CPU for ONNX export to avoid GPU memory issues
            }
            model.export(**onnx_args)

            # Find ONNX file
            model_name = model_path.stem
            onnx_file = model_path.parent / f"{model_name}.onnx"
            if not onnx_file.exists():
                # Try alternative name or find most recent ONNX file
                onnx_files = list(model_path.parent.glob("*.onnx"))
                if onnx_files:
                    onnx_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    onnx_file = onnx_files[0]

            if not onnx_file.exists():
                print("❌ ONNX file not found after export")
                del model
                clear_gpu_memory()
                continue

            print(f"✅ ONNX exported: {onnx_file.name}")

            # Clear memory before TensorRT conversion
            del model
            clear_gpu_memory()

            # Step 2: Convert ONNX to TensorRT engine using TensorRT API directly
            print("📤 Step 2: Converting ONNX to TensorRT engine (GPU)...")
            workspace_gb = 1 if imgsz <= 640 else 2  # 1GB for 640, 2GB for larger

            # Use TensorRT API directly to convert ONNX to engine
            try:
                import tensorrt as trt
            except ImportError as e:
                print(f"❌ TensorRT not found: {e}")
                print("   TensorRT should be installed with Jetson")
                continue

            # TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Build TensorRT engine from ONNX
            print("🔧 Building TensorRT engine from ONNX...")

            builder = trt.Builder(TRT_LOGGER)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX model
            print(f"📖 Parsing ONNX file: {onnx_file.name}")
            with open(str(onnx_file), 'rb') as model:
                if not parser.parse(model.read()):
                    print("❌ Failed to parse ONNX file")
                    for error in range(parser.num_errors):
                        print(f"   Error {error}: {parser.get_error(error)}")
                    continue

            print("✅ ONNX parsed successfully")

            # Configure builder
            config = builder.create_builder_config()

            # Set workspace size (TensorRT 10.x uses set_memory_pool_limit)
            workspace_bytes = workspace_gb * 1024 * 1024 * 1024
            if hasattr(config, 'set_memory_pool_limit'):
                # TensorRT 10.x API
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
                print(f"   Workspace: {workspace_gb} GB (TensorRT 10.x API)")
            elif hasattr(config, 'max_workspace_size'):
                # TensorRT 8.x/9.x API (fallback)
                config.max_workspace_size = workspace_bytes
                print(f"   Workspace: {workspace_gb} GB (TensorRT 8.x/9.x API)")
            else:
                print("⚠️  Could not set workspace size (using default)")

            # Set precision
            if half_precision:
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("✅ FP16 precision enabled")
                else:
                    print("⚠️  FP16 not supported on this platform, using FP32")

            if int8_quantization:
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("✅ INT8 precision enabled")
                else:
                    print("⚠️  INT8 not supported on this platform, using FP16/FP32")

            # Build engine
            print("🔨 Building TensorRT engine (this may take a while)...")
            print(f"   Workspace: {workspace_gb} GB")
            print(f"   Input size: {imgsz}x{imgsz}")

            # TensorRT 10.x uses build_serialized_network, older versions use build_engine
            engine_filename = output_dir / output_filename

            if hasattr(builder, 'build_serialized_network'):
                # TensorRT 10.x API - returns serialized engine directly
                print("   Using TensorRT 10.x API (build_serialized_network)")
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    print("❌ Failed to build TensorRT engine")
                    continue

                # Save engine directly
                print(f"💾 Saving engine to: {engine_filename.name}")
                with open(str(engine_filename), 'wb') as f:
                    f.write(serialized_engine)

                file_size_mb = engine_filename.stat().st_size / (1024 * 1024)
                print(f"✅ TensorRT engine saved: {engine_filename.name} ({file_size_mb:.2f} MB)")

                # Clean up
                del serialized_engine
            elif hasattr(builder, 'build_engine'):
                # TensorRT 8.x/9.x API - returns engine object
                print("   Using TensorRT 8.x/9.x API (build_engine)")
                engine = builder.build_engine(network, config)

                if engine is None:
                    print("❌ Failed to build TensorRT engine")
                    continue

                # Save engine
                print(f"💾 Saving engine to: {engine_filename.name}")
                with open(str(engine_filename), 'wb') as f:
                    f.write(engine.serialize())

                file_size_mb = engine_filename.stat().st_size / (1024 * 1024)
                print(f"✅ TensorRT engine saved: {engine_filename.name} ({file_size_mb:.2f} MB)")

                # Clean up
                del engine
            else:
                print("❌ Unsupported TensorRT version")
                continue

            # Clean up builder objects
            del builder, network, parser, config
            clear_gpu_memory()

            # Engine file is already saved to the correct location
            if engine_filename.exists():
                exported_files.append(str(engine_filename))
            else:
                print(f"⚠️  Engine file not found: {engine_filename}")

        except Exception as e:
            print(f"❌ Failed to export imgsz={imgsz}: {e}")
            import traceback
            traceback.print_exc()
            # Clear memory on error
            try:
                del model
            except:
                pass
            clear_gpu_memory()
            continue

    # Final cleanup - clear all caches
    print("\n🧹 Final cleanup...")
    clear_all_caches()

    return exported_files

def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO .pt model to TensorRT .engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default sizes (640, 1280, 1920)
  python convert_yolo_to_tensorrt.py yolo_11n_day_night_200_2.pt

  # Convert with custom sizes
  python convert_yolo_to_tensorrt.py yolo_11n_day_night_200_2.pt --imgsz 640 1280
  python3 convert_yolo_to_tensorrt.py best.pt --imgsz 640 1280

  # Convert with FP32 (no half precision)
  python convert_yolo_to_tensorrt.py yolo_11n_day_night_200_2.pt --no-half

  # Specify output directory
  python convert_yolo_to_tensorrt.py yolo_11n_day_night_200_2.pt --output engines/
        """
    )

    parser.add_argument('model', type=str, help='Path to YOLO .pt model file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: same as model file)')
    parser.add_argument('--imgsz', type=int, nargs='+', default=None,
                       help=f'Image sizes to export (default: {DEFAULT_IMG_SIZES})')
    parser.add_argument('--no-half', action='store_true',
                       help='Disable FP16 half precision (use FP32)')
    parser.add_argument('--int8', action='store_true',
                       help='Enable INT8 quantization (requires calibration data)')
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment and exit')

    args = parser.parse_args()

    # Check environment
    check_environment()

    if args.check_env:
        print("✅ Environment check complete")
        return

    # Convert model
    exported_files = convert_to_tensorrt(
        model_path=args.model,
        output_dir=args.output,
        imgsz_list=args.imgsz,
        half_precision=not args.no_half,
        int8_quantization=args.int8
    )

    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary")
    print(f"{'='*60}")
    if exported_files:
        print(f"✅ Successfully exported {len(exported_files)} engine file(s):")
        for f in exported_files:
            print(f"   - {f}")
    else:
        print("❌ No engine files were exported")
    print()

if __name__ == "__main__":
    main()



