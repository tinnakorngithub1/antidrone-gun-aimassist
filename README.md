# Anti-Drone Gun Aim-Assist (cam4 arm-mounted)

ระบบเล็ง/ยิงต่อต้านโดรน — กล้อง cam4 ติดบนแขนกล (GRBL) + YOLO (TensorRT) + โหมด LOCK ติดตามเป้า
พร้อมโหมดทดสอบด้วย "โดรนเสมือน" (ฉีดลงเฟรมกล้องสด) โดยไม่ต้องบินโดรนจริง

## โครงสร้างหลัก
| ไฟล์ | หน้าที่ |
|---|---|
| `22_gun_aim_assist_vector.py` | **main** — camera + YOLO + โหมด AUTO/LOCK/MANUAL/SAFE, firing solution, HUD |
| `config.py` | ค่าคอนฟิกทั้งหมด (กล้อง, แขน, โมเดล, LOCK/fire params, โดรนเสมือน) |
| `cam4_arm_controller.py` / `cam4_arm_tracker.py` | คุมแขน GRBL + tracker |
| `cam4_arm_mouse_grid_calibrator.py` | `_variable_step_toward_target` (ตัวขับแขน) + calibrator |
| `lock_sim_target.py` | โดรนเสมือน (realistic flight + explosion) — ทดสอบ LOCK บนแขนจริง |
| `33_lock_sim_closedloop.py` | sim harness รันโค้ด LOCK จริง (`--selftest / --firetest / --campaign`) |
| `fast_motion_sky.py` / `camera_startup.py` | กล้อง/สตรีม RTSP |
| `arm_cue_receiver.py` / `effector_ws_exporter.py` | รับ cue จาก cam8 (UDP) + telemetry → C2 |
| `REVIEW_ANTIDRONE_LOCK_MODE.md` | เอกสารรีวิว + ผลวิจัย (tracking/firing envelope) |

## โมเดล (TensorRT engine)
- `yolo_11n_day_night_200_2_imgsz640.engine` — โดรน day/night (โหลดตอน start)
- `rgb_multiclass_imgsz640.engine` / `thermal_multiclass_imgsz640.engine` — multiclass RGB/thermal

## รัน
```bash
source ~/antidrone_v2/bin/activate     # หรือ venv ที่ใช้
python3 22_gun_aim_assist_vector.py
```
ปุ่ม: **L**=เข้า LOCK, **K**=ล็อกเป้าใต้ crosshair, **ยิง 1 ครั้ง**=อนุญาต (โปรแกรมลั่นไกเองตอนโดนแน่)

## ทดสอบ (sim)
```bash
python3 33_lock_sim_closedloop.py --selftest    # 7/7
python3 33_lock_sim_closedloop.py --firetest
python3 33_lock_sim_closedloop.py --campaign    # envelope → lock_campaign.csv
```

## โดรนเสมือน (ทดสอบบนแขนจริงโดยไม่ต้องบินโดรน)
ตั้งใน `config.py`: `LOCK_SIM_TARGET_ENABLED = True` (⚠️ ปิดเป็น `False` ก่อนใช้งานจริง)
ปุ่มลูกศร=บังคับโดรน, **v**=สลับ pattern

## หมายเหตุ
- ไฟล์วิดีโอ (.mp4), โมเดลดิบ (.pt/.onnx), datasets, runs → ไม่เก็บใน git (ดู `.gitignore`)
- ต้องมี `~/antidrone_v2` (venv) + dependencies ตาม `requirements.txt` + GRBL arm + RTSP camera
