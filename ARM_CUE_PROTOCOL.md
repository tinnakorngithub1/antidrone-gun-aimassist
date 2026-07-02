# ARM CUE Protocol — cam8 (detection) → Jetson แขน (aim-assist)

เอกสารสัญญา interface ระหว่างสองทีม:
- **ทีม cam8** — ระบบ detection / track / alert (`11_AntidroneAlert_*.py` + `arm_cue_sender.py`)
- **ทีมแขน** — ระบบเล็ง/ยิง (`22_gun_aim_assist_vector.py` + `arm_cue_receiver.py`)

> **กติกา:** การแก้ protocol นี้ต้องแก้เอกสารนี้ใน commit เดียวกัน และต้อง backward compatible
> (receiver เก่าต้องไม่พังเมื่อเจอ field ใหม่ / sender เก่าต้องใช้กับ receiver ใหม่ได้)

## Transport

| รายการ | ค่า |
|---|---|
| Protocol | UDP, JSON UTF-8, 1 datagram = 1 payload |
| ปลายทาง | `192.168.144.66:5765` (Jetson แขน) |
| อัตราส่ง | สูงสุด 10 Hz |
| TTL | `cue_ttl_ms` = 500 ms — receiver ทิ้ง cue ที่อายุเกิน (นับจาก local clock ตอนรับ ไม่ใช้ timestamp ผู้ส่ง กัน clock skew) |

## Payload

JSON **list** ของ cue record (หรือ dict เดี่ยว — backward compat)
เรียง **confirmed ก่อน possible**, ภายใน tier เรียง confidence มาก→น้อย, ตัดที่ 8 ตัว

```json
[
  {
    "source_camera": "cam8",
    "target_id": "42",
    "cx": 1520.5, "cy": 380.0,
    "frame_w": 3840, "frame_h": 2160,
    "bbox_w": 48, "bbox_h": 32,
    "bbox_w_norm": 0.0125, "bbox_h_norm": 0.0148,
    "distance_m": 120.5,
    "tier": "confirmed",
    "confidence": 0.87,
    "timestamp": 1782990000.123,
    "sequence": 1234,
    "cue_ttl_ms": 500
  }
]
```

| Field | Type | ความหมาย |
|---|---|---|
| `cx`, `cy` | float | จุดกลางเป้าบน frame cam8 (พิกเซล) |
| `frame_w`, `frame_h` | int | ขนาด frame ที่ cx/cy อ้างอิง (สำคัญ — ฝั่งแขนใช้ scale เข้า grid calibration) |
| `bbox_w`, `bbox_h` | int\|null | ขนาด bbox บน frame detection |
| `bbox_w_norm`, `bbox_h_norm` | float\|null | bbox หารด้วยขนาด frame (0–1) |
| `distance_m` | float\|null | ระยะประมาณ (จาก bbox + FOV) |
| **`tier`** | str | `"confirmed"` = ยืนยันเป็นโดรน / `"possible"` = มีความเป็นไปได้ — **ไม่มี field นี้ = `"confirmed"`** (payload เวอร์ชันเก่า) |
| **`confidence`** | float\|null | YOLO conf (0–1); `null` = ยืนยันด้วย kinematic signature ล้วน ไม่มีค่า YOLO |
| `target_id` | str | id ของ track ฝั่ง cam8 (YOLO-only ใช้ id ≥ 10,000,000) |
| `timestamp`, `sequence` | float, int | เวลา + ลำดับฝั่งผู้ส่ง (debug) |
| `cue_ttl_ms` | int | อายุ cue ที่ receiver ยอมรับ |

## เกณฑ์ tier (ฝั่ง cam8)

| Tier | เกณฑ์ |
|---|---|
| `confirmed` | path `kinematic_confirmed` **หรือ** `yolo_state == 'red'` **หรือ** YOLO conf ≥ `LOCAL_YOLO_DRONE_CONF_THRESHOLD` (0.5) |
| `possible` | validated path ที่เคยมี YOLO conf ≥ 0.30 หรือ `yolo_state == 'orange'` **หรือ** YOLO det ที่ conf อยู่ใน [0.30, 0.5) |

ค่าปรับได้ใน `11_..._hudFPS.py`: `LOCAL_YOLO_DRONE_POSSIBLE_CONF_THRESHOLD`,
`ARM_CUE_POSSIBLE_PATH_MIN_CONF`, `ARM_CUE_MAX_CANDIDATES`

**หมายเหตุ:** possible tier ส่งเป็น arm cue เท่านั้น — **ไม่**เข้า alert / session count / websocket export ของ cam8

## พฤติกรรมฝั่งแขน (โหมด AUTO)

1. เลือก primary cue: confirmed ก่อน possible → confidence สูงสุด
2. แปลง `cx,cy` → องศาแขนด้วย grid calibration (`cam8_mouse_grid_lookup.json`) + residual bias
3. ขับแขนชี้ตาม แล้ว**รอ human ตัดสินใจ** กด `K` (ล็อกเป้าใต้ crosshair) / `L` (โหมด LOCK) — โปรแกรมไม่ยิงเองจาก cue
4. `ARM_CUE_FOLLOW_POSSIBLE` (ใน `22_...py`, default `True`):
   - `True` — ชี้ตาม possible ด้วย (HUD แสดง `SRC:cam8(poss)` + `CUE:..ms[POSS]` สีส้ม)
   - `False` — ชี้เฉพาะ confirmed; possible จะขึ้น `SRC:cam8_wait(possible)`
5. Online residual-bias learning เรียนรู้จาก **confirmed cue เท่านั้น**
6. cue หมดอายุ/ไม่มี → แขนหยุดรอ (ไม่เดา ไม่ fallback)

## Versioning / compatibility

| ฝั่งส่ง | ฝั่งรับ | ผล |
|---|---|---|
| เก่า (ไม่มี tier) | ใหม่ | ทุก cue ถูกมองเป็น `confirmed` — พฤติกรรมเดิม |
| ใหม่ | เก่า | receiver เก่าใช้ element แรกของ list = confirmed ที่ดีที่สุด (เพราะ sender เรียงให้) — ไม่พัง แต่ไม่เห็น possible |
| ใหม่ | ใหม่ | ครบทุก feature |

## ทดสอบข้ามทีม

```bash
# ฝั่งแขน: ฟัง cue ที่เข้ามา (ไม่ต้องรันโปรแกรมหลัก)
python3 -c "
from arm_cue_receiver import ArmCueReceiver; import time
r = ArmCueReceiver(); r.start()
while True: print(r.status_label(), r.get_latest_cue()); time.sleep(0.5)"

# ฝั่ง cam8: ยิง cue ทดสอบ 1 นัด
python3 -c "
from arm_cue_sender import ArmCueSender; import time
s = ArmCueSender(host='192.168.144.66'); s.start()
s.push([(960, 540, 1, 100.0, 40, 30, 'possible', 0.42)], 1920, 1080); time.sleep(1)"
```
