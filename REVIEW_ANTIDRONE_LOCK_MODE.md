# รีวิวระบบ Antidrone — โหมด LOCK (กล้องติดบนแขนกล)

วันที่รีวิว: 2 ก.ค. 2026
ไฟล์หลักที่ตรวจ: `22_gun_aim_assist_vector.py` (4,705 บรรทัด), `cam4_arm_mouse_grid_calibrator.py`, `cam4_arm_controller.py`, `config.py` และชุดไฟล์ทดลอง `23–32_*.py` พร้อมผล CSV ทั้งหมด (ลงวันที่ 1 ก.ค. 2026)

---

## 1. บทสรุปผู้บริหาร (TL;DR)

อาการ "LOCK ติดตามโดรนไม่ได้เลย" **ไม่ได้เกิดจากปัญหา ego-motion อย่างเดียว** — พบบั๊กที่ทำให้ `_tick_lock` **ไม่ขับแขนเลยตั้งแต่ต้น**:

> **B1 (วิกฤตที่สุด):** ไฟล์ calibration `calibration_data/cam4_pixel_per_degree.json` มีค่า `pixel_per_degree_y = -89.73` (ค่า**ลบ**เป็น convention ปกติของระบบ — ค่า default ก็คือ `-50.0`) แต่ guard ใน `_tick_lock` บรรทัด 2712–2713 เขียนว่า `not (px_per_deg_y > 0)` → **return ทันทีทุกเฟรม** ผลคือใน LOCK แขนขยับแค่ครั้งเดียวตอนกดปุ่มล็อก (ผ่าน `move_absolute` ที่บรรทัด 4135 ซึ่งไม่มี guard นี้) แล้วหลังจากนั้น auto-track ไม่ทำงานเลย

นอกจากนี้ สถาปัตยกรรมปัจจุบันติดตามเป้าใน **พิกัด pixel ดิบ** ทั้งที่กล้องติดบนแขน — การเคลื่อนแขนจึงปนเข้าไปใน velocity ของ Kalman (ego-motion ไม่ถูกหักออก) ทางแก้ที่ถูกต้อง**คุณเขียนไว้แล้วใน simulator `23_arm_chase_sim.py`** (แปลง pixel error เป็น absolute bearing โดยใช้ท่าแขน ณ เวลาเก็บภาพ) แต่**ยังไม่ได้ port กลับเข้าไฟล์จริง 22**

ผลการทดลอง 27–32 ชี้ชัดว่า**คอขวดไม่ใช่แขนกล แต่คือ detection pipeline** (latency + det-fps) — ข้อสรุปนี้น่าเชื่อถือและควรกำหนดทิศทางงานต่อ

---

## ✅ สถานะการแก้ P0 (2 ก.ค. 2026 — แก้แล้วใน `22_gun_aim_assist_vector.py`)

| ข้อ | สถานะ | รายละเอียด |
|---|---|---|
| B1 guard ค่าลบ | **แก้แล้ว** | `_tick_lock` + bias learner ใช้ `abs(ppd) > 1e-6`; เครื่องหมายลบของ ppd_y ทำหน้าที่กลับทิศ tilt ตามเดิม |
| B2 ego-motion | **แก้แล้ว** | เพิ่ม `_ArmPoseHistory` (บันทึกท่าแขนทุก loop + interpolate) → main loop แปลงเป้าเป็น **absolute bearing** = pose(t_capture) + px_offset/ppd แล้วป้อน `lock_kalman` ซึ่งเปลี่ยนเป็น **bearing space (องศา)** — Q/R scale ด้วย (1/ppd)² (`LOCK_BEARING_KALMAN_Q/R`); `_tick_lock` เขียนใหม่: ขับแขนเข้าหา bearing + deadzone `LOCK_DEADZONE_DEG` |
| B3 stale/coast | **แก้แล้ว** | measurement เข้า filter เฉพาะตอน IoU tracker **match จริง** → เป้าหลุดแล้ว filter coast ตามเวลา (bounded, τ=0.30) จนครบ 0.5 s แล้ว hold; เพิ่ม hysteresis `IOT_MISS_FRAMES_TO_LOST=3` + tracker เก็บ `last_conf` ส่งเข้า Kalman (conf-weighted R ทำงานแล้ว) |
| B5 SWAP_PAN_TILT | **ตรวจแล้ว — ไม่ใช่บั๊ก** | `config.py:28 CAM4_ARM_SWAP_PAN_TILT = False` → mapping pan↔pos_x↔x-limits ถูกต้องสอดคล้องทั้ง acquire/step/move_relative |

การทดสอบ (venv `antidrone_v2`): unit test 16 ข้อ + closed-loop sim ผ่านทั้งหมด —
แขนลู่เข้า bearing นิ่งภายใน deadzone (0.49°), **ghost velocity จากการขยับแขน = 0.0 deg/s**,
coast เคลื่อนต่อแบบ bounded (1.2° ที่เป้า 5°/s) แล้ว **hold สนิท**หลังหมดหน้าต่าง 0.5 s
(สคริปต์ทดสอบ: scratchpad `test_lock_p0.py`, `test_lock_closedloop.py`)

ผลพลอยได้: ปัญหา "สั่งซ้ำสองเด้งหลัง acquire" (B4) หายไปโดยอัตโนมัติ — bearing ของเป้าคำนวณจาก pose ณ เวลาเก็บภาพ จึงไม่สั่ง slew ซ้ำแม้ frame จะเก่าก่อน `move_absolute` (แต่ blocking freeze ยังอยู่ = งาน P1)

---

## 🧪 Closed-loop simulation — `33_lock_sim_closedloop.py` (ใหม่ 2 ก.ค.)

ไฟล์ 23–24 ใช้ controller **จำลอง** ของตัวเอง จึงไม่ได้ทดสอบโค้ดจริง และไฟล์ 24 "จริง" ก็**ไม่มีกล้องใน loop** (ช่องโหว่ที่รายงานไว้) — ไฟล์ใหม่นี้อุดช่องนั้น:

- **รันโค้ด production ตัวจริง**: `import` `_SimpleIoUTracker`, `_TargetKalman`, `_lock_feed_bearing_measurement`, `_tick_lock`, `_variable_step_toward_target` จาก `22_gun_aim_assist_vector.py` โดยตรง (ไม่ก็อปมา) → ทดสอบตรงกับที่รันจริง 100%
- **จำลองกล้องบนแขน**: pixel ของโดรน = (bearing_โดรน − มุมแขน**จริง**)·ppd + ศูนย์กลาง → พอแขนขยับ ภาพเลื่อน (ego-motion) ซึ่งเป็นโจทย์จริง
- **แยก commanded vs actual**: โค้ดอ่าน `pos_x` (commanded, อัปเดตทันที) แต่กล้องเห็นมุม**จริง**ที่ lag ตาม rate 200°/s + accel 650°/s² (ค่า GRBL จริงจาก config) + sync ทุก 10 moves → จับ residual mechanical lag ที่ ego-comp ลบไม่ได้
- pipeline latency + det-fps + bbox noise + miss-rate ปรับได้

**การใช้งาน:**
```
python3 33_lock_sim_closedloop.py --selftest      # assert P0 (CI) — ผ่าน 5/5
python3 33_lock_sim_closedloop.py --sweep         # ตาราง omega × latency → lock_sim_sweep.csv
python3 33_lock_sim_closedloop.py --omega 8 --latency-ms 150 --det-fps 15
```

**ผล sweep (แขนจริง, det 15fps, noise 8px, miss 20%, ยังไม่มี lead comp):**

| ω (°/s) | v@150m | median err (lat 50/150/300 ms) | สรุป |
|---|---|---|---|
| 5  | 13 m/s | 0.80 / 1.16 / 1.69° | เกาะได้ดี (ต่ำกว่า reticle ที่ latency ต่ำ) |
| 8  | 21 m/s | 1.26 / 1.74 / 2.45° | ยังเกาะได้ |
| 10 | 26 m/s | 1.99 / 2.60 / 3.68° | เริ่มหลุด reticle |
| 15 | 39 m/s | 3.49 / 4.62 / 5.83° | fail |
| 25+ | 65+ m/s | 9.5–20° | แขน/ภาพตามไม่ทัน เป้าออก FOV |

**ข้อสรุปที่ยืนยัน:** envelope จาก closed-loop (โค้ดจริง) มี**รูปทรงเดียวกับ** sim 28–32 (เกาะได้ ω ≲ 8–10°/s, พังที่ ω ≳ 15) — เป็นการ cross-validate ว่าโค้ด production หลังแก้ P0 ทำงานตรงกับ controller ที่ validate ไว้ ตัวเลข median สูงกว่า latency_sensitivity.csv เล็กน้อยเพราะ**ยังไม่ได้ทำ lead compensation (P1)** — measurement ถูก timestamp ณ เวลา deliver ไม่ใช่ t_capture จึงมี staleness = latency คงเหลือ ซึ่งเป็นงานถัดไปที่จะดันขอบ envelope

**Before/after (แก้ B1):** ก่อนแก้ guard ค่าลบ `_tick_lock` return ทุกเฟรม → แขนแช่นิ่ง → error = RMS ของ sweep ±20° ≈ 14° ทุก ω; หลังแก้ ω=5 เหลือ median 0.8° — พิสูจน์ว่าแขนติดตามจริง

---

## 🚁 โดรนเสมือนบนกล้องจริง — `lock_sim_target.py` (ใหม่ 2 ก.ค.) — ทดสอบแขนจริงโดยไม่ต้องบินโดรน

ทดสอบ LOCK บน**ฮาร์ดแวร์จริง (แขน+กล้อง)** ได้เลยโดยไม่ต้องมีโดรนจริง — ฉีดโดรนเสมือนลงเฟรมกล้องสด

**หลักการ (สำคัญ — กล้องติดบนแขน):** โดรนมีตำแหน่งใน bearing โลกจริง แล้ว project ลงภาพด้วย
`px = cx + (bearing − มุมแขนจริง)·ppd` → เมื่อแขนหมุนตาม ภาพเลื่อนจริง (ego-motion แท้) เหมือนเป้าจริง
detection ไหลผ่าน pipeline จริง (real YOLO latency) → tracker → LOCK → ขับแขนจริง โดย bearing คิดจาก
pose ณ เวลาเก็บภาพ (ไม่ double-count)

**วิธีเปิด** (แก้ `config.py`):
```python
LOCK_SIM_TARGET_ENABLED = True          # เปิดโดรนเสมือน (⚠️ ปิดตอนใช้งานจริง)
LOCK_SIM_TARGET_PATTERN = "hover"       # เริ่มด้วย hover (ง่ายสุด) → sine → figure8
LOCK_SIM_TARGET_OMEGA_DEG_S = 8.0       # ความเร็ว (เริ่ม 5 แล้วค่อยเพิ่ม)
LOCK_SIM_TARGET_MISS_RATE = 0.15
LOCK_SIM_TARGET_INJECT_DETECTION = True # True=ฉีด detection ตรง (เชื่อถือได้ ทดสอบ tracking/แขน)
                                        # False=วาดสไปรต์อย่างเดียว ให้ YOLO จริงตรวจ (end-to-end)
```
แล้วรัน `python3 22_gun_aim_assist_vector.py` ตามปกติ → กด **L** เข้า LOCK → **K** (หรือปุ่ม 5) ล็อกเป้า → แขนติดตามโดรนเสมือน

**ปุ่มบังคับโดรนเสมือน (ในหน้าต่างโปรแกรม):** ลูกศร ←↑↓→ = ขยับ (โหมด manual), **v** = สลับ pattern
(ไม่ชนปุ่ม k/l ที่เป็น acquire/LOCK)

**ลำดับทดสอบแนะนำ:** hover (เกาะนิ่ง) → sine ω=5 → เพิ่ม ω → figure8 → ตั้ง `INJECT_DETECTION=False`
ทดสอบว่า YOLO จริงตรวจสไปรต์ได้ไหม (end-to-end) ดู `[LOCK] bearing/age/COAST` ยืนยันพฤติกรรม
(unit test wiring ผ่าน: hover เกาะ 100% median 0.44°, sine ω5 median 0.85°)

---

## 2. บั๊กที่ต้องแก้ใน `22_gun_aim_assist_vector.py` (เรียงตามความรุนแรง)

### B1 — Guard ปฏิเสธ `px_per_deg_y` ค่าลบ → LOCK auto-track ตายสนิท 🔴

- ตำแหน่ง: `_tick_lock` บรรทัด 2712–2713 (`not (px_per_deg_x > 0) or not (px_per_deg_y > 0)`)
- ข้อเท็จจริง: `_load_px_per_deg()` (บรรทัด 330–353) คืนค่าดิบจาก JSON โดยไม่ abs(); calibration จริงของ cam4 คือ `(87.14, -89.73)`; ค่า fallback ของระบบเองคือ `PX_PER_DEG_Y_DEFAULT = -50.0` — ค่าลบคือเรื่องปกติ (แกน y ภาพกลับทิศกับ tilt)
- โค้ดส่วนอื่นรู้เรื่องนี้อยู่แล้ว: `get_velocity_deg_s` (บรรทัด 499) ใช้ `abs(px_per_deg_y) < 1e-6` อย่างถูกต้อง
- **จุดที่โดนบั๊กเดียวกัน:** `ResidualBiasLearner` บรรทัด 2500 (`px_per_deg_y > 0`) → bias fine-step ใน AUTO ก็ไม่เคยทำงานด้วย
- **วิธีแก้:** เปลี่ยนทั้งสองจุดเป็น `abs(px_per_deg_x) > 1e-6 and abs(px_per_deg_y) > 1e-6` แล้ว**อย่า** abs() ตัวค่า — เครื่องหมายลบคือข้อมูลทิศทางที่ทำให้ `dy_px / px_per_deg_y` ออกมาถูกทิศเอง (และจะทำให้ hack `LOCK_PAN_SIGN`/`LOCK_TILT_SIGN` ไม่จำเป็น)
- วิธีตรวจว่าใช่: debug print `[LOCK dbg]` (บรรทัด 2708) พิมพ์ `ppd=(87.1..., -89.7...)` อยู่แล้ว — ถ้าเห็น ppd ลบและแขนนิ่ง = ยืนยัน

### B2 — ไม่มี ego-motion compensation: ติดตามใน pixel space ทั้งที่กล้องอยู่บนแขน 🔴

นี่คือต้นเหตุเชิงสถาปัตยกรรมของอาการ "พอแขนขยับ ภาพเปลี่ยน ตำแหน่งเป้าเพี้ยน":

- `_tick_lock` (2732–2742) คำนวณ `target = pos_ปัจจุบัน + pixel_error/ppd` โดย pixel error มาจาก**เฟรมเก่า** (pipeline latency วัดได้จริงในโค้ด ~150–400 ms) → ระหว่างนั้นแขนขยับไปแล้ว แต่ error เก่ายังถูกนับซ้ำ (double-count) → overshoot/แกว่ง ซึ่งตอนนี้กดอาการด้วย `LOCK_TRACK_STEP_SCALE = 0.7` (บรรทัด 405) — เป็น band-aid ไม่ใช่การแก้
- Kalman `lock_kalman` ถูกป้อนพิกัด pixel ดิบ → velocity ที่ประมาณได้ = (การเคลื่อนของโดรน) − (การเคลื่อนของแขน) ปนกัน → coast/predict_ahead ทำนายผิดทิศ
- **ทางแก้ที่พิสูจน์แล้วมีอยู่ใน `23_arm_chase_sim.py` บรรทัด 274–278** (`apan_delayed`): แปลง pixel error เป็น **absolute bearing** ด้วยท่าแขน ณ เวลาเก็บภาพ — ใน sim วิธีนี้ทำให้ latency ไม่ double-count
- **สิ่งที่ต้อง port เข้า 22:**
  1. เก็บ ring buffer `(timestamp, pos_x, pos_y)` ของแขนทุก loop (มี `_t_sent` ติดไปกับเฟรมที่ส่งเข้า YOLO อยู่แล้ว — บรรทัด 3611)
  2. เมื่อผล YOLO กลับมา: หา arm pose ที่ interpolate ณ `_t_sent` แล้วคำนวณ `target_bearing_pan = arm_pan(t_capture) + dx_px/ppd_x`, `target_bearing_tilt = arm_tilt(t_capture) + dy_px/ppd_y`
  3. ให้ `lock_kalman` ติดตามใน **angle space (bearing)** แทน pixel space → velocity ที่ได้คือความเร็วเชิงมุมของโดรนจริง ไม่ปนการเคลื่อนแขน → coast มีความหมายจริง
  4. `_tick_lock` ขับแขนเข้าหา bearing (+ lead) — จุดนี้ `LOCK_TRACK_STEP_SCALE` ยกกลับไปใกล้ 1.0 ได้
- ประโยชน์พลอยได้: IoU tracker gate จะไม่หลุดตอนแขน slew เร็ว เพราะ predicted bbox เลื่อนตามการเคลื่อนแขนที่คาดไว้ได้ (ชดเชย `pred_cx/cy` ด้วย Δarm × ppd)

### B3 — เป้าหลุดแล้วแขนไล่ "ตำแหน่งผี" ไม่หยุด / coast ไม่เคยทำงาน 🔴

- `_SimpleIoUTracker` เมื่อหาเป้าไม่เจอ set แค่ `lost=True` (บรรทัด 688–690) แต่ **ไม่เคลียร์ `smooth_cx/cy`** — ค่าค้างเป็นตำแหน่งสุดท้าย
- Main loop sync ค่านี้เข้า `lock_csrt_smooth_px/py` ทุกเฟรม (3669–3675) → ใน `_tick_lock` เงื่อนไข `if lock_smooth_px is not None` (2720) เป็นจริง**ตลอด** → `lock_kalman.update(ค่าค้าง, conf=1.0)` ทุกเฟรม → `_last_update_time` refresh ตลอด → **branch coast (2724–2728) เป็น dead code** และไม่มี timeout
- ผล: โดรนหลุดเฟรม → แขนขับเข้าหา pixel ที่แช่แข็งไปเรื่อย ๆ ไม่ coast ไม่ hold
- **วิธีแก้:** เมื่อ `lost=True` ให้หยุดป้อน update (ข้าม branch 2720 เมื่อ `lock_lost`) เพื่อให้เข้า coast ตาม `LOCK_KALMAN_COAST_MAX_SEC=0.5` + `KALMAN_COAST_DECAY_TAU=0.30` ที่ตั้งไว้แล้ว (bounded coast ที่คุณ validate ใน 31/32 ถูก port มาแล้วแต่**ไม่เคยถูกเรียกจริง**เพราะบั๊กนี้)
- เสริม: เพิ่ม miss-count hysteresis ใน tracker — ตอนนี้ detection ว่างแค่ **1 เฟรม** ก็ `lost=True` ทันที (บรรทัด 633–635)

### B4 — Acquire แบบ blocking + สั่งซ้ำสองเด้ง 🟠

- ตอนกดล็อก: `arm_controller.move_absolute(..., blocking=True)` (บรรทัด 4135) **แช่ main loop ทั้งเส้น** ระหว่างแขน slew — ไม่มีเฟรม/detection ประมวลผลเลยในช่วงที่ภาพกำลังกวาด (ช่วงที่เสี่ยงหลุดที่สุด)
- Tracker ถูก seed ด้วย bbox center **ก่อน slew** (บรรทัด 4111) → หลัง slew เสร็จ ค่า `smooth_cx` ยังเป็นตำแหน่งเก่า → `_tick_lock` เฟรมถัดไปสั่งแขนเคลื่อนด้วย offset เดิม**ซ้ำอีกรอบ**จนกว่า YOLO ผลใหม่จะมา
- **วิธีแก้:** เปลี่ยนเป็น non-blocking แล้วปล่อยให้ `_tick_lock` (หลังแก้ B1/B2) เป็นตัวพาเข้าเป้าเอง — ถ้าใช้ bearing space ตาม B2 ปัญหา seed เก่าจะหายไปเองเพราะ bearing ของเป้าไม่เปลี่ยนตามการเคลื่อนแขน

### B5 — `SWAP_PAN_TILT` default = True แต่โค้ด LOCK คิดว่าเป็น False 🟠

- `cam4_arm_mouse_grid_calibrator.py` ประกาศ `SWAP_PAN_TILT = True` (default) แต่คอมเมนต์ที่ `_tick_lock` บรรทัด 2743 เขียนว่า "SWAP=False อยู่แล้ว → กิ่งนี้ไม่ทำงาน" — ถ้าค่าจริงเป็น True แกน pan จะถูก clip ด้วยลิมิต y (±35°) และ tilt ด้วยลิมิต x (±65°) สลับกัน
- **ต้องยืนยันค่าจริงที่รันไทม์** แล้วทำให้ mapping กับคอมเมนต์ตรงกัน — จุดนี้บวกกับ feedback บนแขนคือผู้ต้องสงสัยของอาการ "แขนวิ่งหนีเป้า" ที่ `LOCK_PAN_SIGN/TILT_SIGN` ถูกสร้างมา patch

### B6 — โค้ดที่ตั้งใจไว้แต่ไม่เคยทำงาน (dead code) 🟡

| สิ่งที่ตั้งใจ | สถานะจริง |
|---|---|
| `get_velocity_deg_s()` — velocity feedforward | นิยามไว้ (497–504) **ไม่มีใครเรียก** — ทั้งที่ผลทดลอง 29 ชี้ว่า lead=latency ช่วยมาก |
| `LEAD_TIME_SEC=0.20` + `avg_pipeline_latency` ที่วัดจริง (3636–3640) | ใช้แค่โชว์ `lat_ms` บน HUD — **ไม่ถูกใช้ predict-ahead** |
| `lock_coast_start_time` (3175) | ประกาศแล้วไม่ถูกอ่าน/เขียนอีกเลย |
| Confidence-weighted Kalman (`R = R/conf²`) | LOCK เรียก `update(..., 1.0)` เสมอ (2723) — ควรส่ง conf จริงของ detection |
| Fallback distance gate ใน tracker (668, `0.4×max_dim`) | ซ้ำซ้อน — pre-filter กรองที่ `0.25×max_dim` ไปก่อนแล้ว (649) |

---

## 3. รีวิวชุดการทดลอง 23–32 (งานเมื่อ 1 ก.ค.)

### สิ่งที่ทำได้ดีและข้อสรุปที่ใช้ได้

การออกแบบชุดทดลองโดยรวม**ดีมากในเชิงวิธีวิจัย** — แยกตัวแปร (mechanical / latency / det-fps / noise / controller) เป็นระบบ และมีการ validate twin กับจุดวัดจริง ข้อสรุปที่ตัวเลขรองรับ:

1. **แขนกลไม่ใช่คอขวด** — mechanical twin (27, envelope_rate*.csv): ที่ rate 50–100 Hz + accel ≥ 200°/s² ตาม ω ≤ 15°/s ได้ 100% on-target
2. **คอขวดคือ detection pipeline** (28, 29): ที่ latency 150 ms + 15 fps ระบบเอาอยู่แค่ ω ≲ 5–10°/s (≈ v 13–26 m/s ที่ระยะ 150 m); ω=15°/s **fail แม้ latency = 0** ถ้า det ยัง 15 fps
3. **งบ latency จำกัดมาก**: ω=10°/s เหลือ budget เพียง ~75 ms แม้ชดเชย lead เต็มที่ (latency_sensitivity.csv)
4. **det-fps คือกำแพง, bbox noise แทบไม่มีผล** (30): ω=15 ต้องการ ≥30 fps; ω=20 ต้องการ 60 fps — ลงแรงกับ frame rate/latency ไม่ใช่ความละเอียด bbox
5. **`kalman_coast_bounded` (τ=0.30) คือ controller ที่ควรใช้** (31, 32): ปิด danger zone (ω สูง + fps ต่ำ ที่ coast เชิงเส้น overshoot เช่น ω30fps3 แย่ลง −1.58°) โดยรักษา ~72% ของข้อได้เปรียบใน sweet zone

### ช่องโหว่ของการทดลองที่ต้องระวังก่อนอ้างในงานวิจัย

1. **"การทดลองจริง" (24) ยังไม่มีกล้องใน loop** — เป้าคือ setpoint สคริปต์ และ error = ตำแหน่งสั่ง − ตำแหน่งแขนจริง (24:572) → วัดได้แค่ mechanical lag; ปัญหา image-shift/YOLO latency ที่เป็นโจทย์จริง**ยังไม่เคยถูกทดสอบบนฮาร์ดแวร์เลย** ข้อสรุปฝั่ง pipeline ทั้งหมดมาจาก sim ล้วน
2. **Twin ถูก back-fit ไม่ใช่วัดจริง**: accel 100°/s² ใน sim มาจากค่า fallback (key `CAM4_ARM_MAX_PAN_ACCEL_DEG` ไม่มีใน config.py) แล้วบังเอิญ fit median จริงได้ ขณะที่ config จริงระบุ accel 650 mm/s² และ G0 rapid วิ่งได้ ~200 mm/s — ควรอ่านค่า `$120/$121`, `$110/$111` จาก GRBL จริงมาใส่ twin แล้วหาสาเหตุ offset ~1.0° ที่เหลือ (น่าจะเป็น serial/planner latency ซึ่ง `--sim-latency-ms` มีอยู่แล้วแต่ default 0 และ sweep ไม่เคยตั้ง)
3. **การทดลอง 28–32 ใช้สมมติฐานดีกว่าที่ validate ไว้**: default accel 200 / rate 50 Hz ทั้งที่ twin ยืนยันกับของจริงที่ accel≈100 / rate 20 → ผลจึง optimistic เล็กน้อย ควรระบุใน paper
4. **ข้อมูลเสีย 2 จุด**: `tune.csv` header/label เพี้ยน (`v\R,0,0,...` — `_write_matrix` ที่ 23:789 รับ label ผิด) และ `accel_sweep.csv` คอลัมน์ `real_on%` ค้างที่ 20.0 เป๊ะทั้งสามค่า accel ทั้งที่ median ดีขึ้นชัด — เชื่อถือไม่ได้ ควรรันเก็บใหม่
5. **ผลการทดลองยังไม่ถูกตั้งเป็น default**: 23 ยัง default `plead`, 28–30 ยัง `kalman_coast` ทั้งที่ 32 สรุปแล้วว่า bounded ดีกว่า — และใน 22 ตัวจริง bounded coast มีโค้ดแล้วแต่ unreachable (บั๊ก B3)

---

## 4. สิ่งที่ต้องปรับแก้ — เรียงลำดับ

### P0 — ทำก่อน (ปลดล็อกให้ LOCK ทำงานได้จริง)

1. แก้ guard ค่าลบ 2 จุด (B1): `_tick_lock` 2712–2713 และ bias learner 2500 → ใช้ `abs(ppd) > 1e-6`
2. หยุดป้อนค่าค้างเข้า Kalman ตอน `lost` (B3) → ให้ bounded coast ที่มีอยู่แล้วทำงานจริง + เพิ่ม miss hysteresis (~3–5 เฟรม) + เคลียร์/timeout เมื่อ coast หมดหน้าต่าง
3. Port แนวทาง absolute-bearing จาก `23_arm_chase_sim.py:274-278` เข้า 22 (B2): arm-pose ring buffer + แปลง detection เป็น bearing ณ `_t_sent` + Kalman ใน angle space
4. ยืนยัน `SWAP_PAN_TILT` และ mapping ลิมิตแกน (B5)

### P1 — ทำต่อทันทีหลัง P0

5. เปลี่ยน acquire เป็น non-blocking (B4)
6. ต่อสาย feedforward + lead: ใช้ `get_velocity_deg_s()` กับ `avg_pipeline_latency` ที่วัดจริง (มีของครบแล้ว แค่ยังไม่ต่อ) — ผลทดลอง 29 ยืนยันว่าคุ้ม
7. ส่ง conf จริงของ detection เข้า `lock_kalman.update()` แทน 1.0
8. หลังแก้ B2 แล้ว ทดลองยก `LOCK_TRACK_STEP_SCALE` 0.7 → ~1.0 และถอด `LOCK_PAN_SIGN/TILT_SIGN` hack

### P2 — ระดับระบบ/งานวิจัย

9. **ลด latency / เพิ่ม det-fps** — ข้อสรุปหลักจากการทดลองของคุณเอง: RTSP 4K 15fps คือคอขวด; พิจารณา CSI/USB camera + ลด resolution ฝั่ง detect + วัด end-to-end latency ใหม่ (เป้าหมาย: ≥30 fps, latency ≤ 100 ms เพื่อขยาย envelope ไป ω≈15°/s)
10. สร้างการทดลองจริงแบบ **closed-loop มีกล้องใน loop** (ช่องโหว่ใหญ่สุดของชุดทดลอง): เป้าเคลื่อนที่จริง (จอ/ราง/โดรนผูกเชือก) + log `(t_capture, detection, arm_pose)` เพื่อวัด error เชิงมุมจริงเทียบ sim
11. อ่านค่า GRBL จริง (`$$`) มาตั้ง twin แทน back-fit + ใส่ transport latency ใน `PhysicsSimArmController` (ตอนนี้ default 0) เพื่อปิด gap ~1.0°
12. รันเก็บ `accel_sweep` ใหม่ (real_on% เสีย) และแก้ label `tune.csv`
13. ตั้ง `kalman_coast_bounded` เป็น default controller ในทุกสคริปต์ทดลองให้ตรงกับข้อสรุปของ 32

---

## 5. เช็กลิสต์ทดสอบยืนยันหลังแก้

- [ ] เปิด LOCK กับเป้านิ่ง: debug `[LOCK]` ต้องพิมพ์ `move=(...)` ไม่ใช่เงียบ (พิสูจน์ B1 หาย)
- [ ] เป้านิ่ง + โยกแขนด้วยมือ (จอย MANUAL แล้วสลับ LOCK): crosshair ต้องดึงกลับเข้าเป้าโดยไม่แกว่ง (พิสูจน์ B2 — bearing ไม่เปลี่ยนตามแขน)
- [ ] บังเป้าชั่วคราว 0.3 s: แขน coast ตามทิศเดิมแล้วหยุด ไม่ไล่จุดค้าง; บังนาน >0.5 s: แขนหยุดนิ่ง (พิสูจน์ B3)
- [ ] กดล็อกเป้าที่ขอบเฟรม: ระหว่าง slew ภาพต้องยังเดิน (ไม่ freeze) และไม่มีการ slew ซ้ำรอบสอง (พิสูจน์ B4)
- [ ] เป้าเคลื่อน ω≈5–10°/s: median error ≤ 1° ตาม envelope ที่ sim ทำนาย — ถ้าจริงแย่กว่า sim มาก แปลว่ายังมี latency ที่ไม่ได้ model
