# Learned Aim Controller

MLP เล็กที่เรียนจาก state 10 มิติ (err_pan, err_tilt, error_deg, last_delta, dt, d_pan, d_tilt, current_pan_deg, current_tilt_deg) → (delta_pan, delta_tilt) ใช้ในโหมด AUTO/LOCK แทนหรือเสริม PD.

## การใช้

1. **เก็บข้อมูล**: ตั้ง `CAM4_ARM_AIM_COLLECT_DATA = True` ใน config แล้วรันโหมด AUTO/LOCK เล็งเป้าตามปกติ
2. **ปิดโปรแกรม**: atexit จะเขียน buffer ลง `aim_buffer.npz` (ถ้ามี ≥ 500 transition) แล้วรีเทรนและบันทึก `aim_model.pt` (ใช้น้ำหนัก transition แดง > ส้ม ตาม `CAM4_ARM_LEARNED_AIM_WEIGHT_RED/ORANGE`)
3. **ใช้ model**: ตั้ง `CAM4_ARM_USE_LEARNED_AIM_MODEL = True` และ path เป็น `aim_controller_model/aim_model.pt` หรือ `aim_model.onnx` รันครั้งถัดไปจะโหลด model (รองรับทั้ง .pt และ .onnx)

## โครงสร้าง

- `model.py`: MLP (10→64→32→2), normalize_state, load/save .pt, load_onnx, predict_delta / predict_delta_onnx
- `train.py`: โหลด .npz, กรอง good transitions, weighted MSE (แดง>ส้ม), บันทึก .pt
- `export_onnx.py`: export .pt → .onnx (optional)

## Dependencies

- **PyTorch** (`torch`): สำหรับเทรนและ inference แบบ .pt (บน Jetson ใช้ torch จาก NVIDIA)
- **onnxruntime** (optional): สำหรับ inference แบบ .onnx เบากว่า
