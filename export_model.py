import torch
from ultralytics import YOLO
import gc

# 1. เคลียร์ทุกอย่างออกจาก GPU ก่อน
torch.cuda.empty_cache()
gc.collect()

# 2. โหลดโมเดลไปที่ CPU โดยตรง (เพื่อเลี่ยง cuBLAS บน GPU ตอน Fuse)
model = YOLO("best.pt")
model.to('cpu') 

print("Starting export on CPU-to-GPU pipeline...")

# 3. สั่ง Export
# การใส่ device=0 ตรงนี้จะทำให้ TensorRT Builder ใช้ GPU เฉพาะตอนสร้าง Engine
# แต่การ Fuse (จุดที่ค้าง) จะพยายามทำบน CPU ก่อนเพราะเราย้ายโมเดลไปแล้ว
model.export(
    format="engine", 
    device=0, 
    half=True, 
    imgsz=640, 
    simplify=True
)
