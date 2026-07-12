import os

# ============================================================================
# CAM4 ARM SIMULATION MODE
# ============================================================================
# ใช้สำหรับทดสอบบนวิดีโอ .mp4 โดยไม่ส่ง G-code จริงไปยัง GRBL
# - False = ใช้แขนจริงผ่าน Serial/G-code (โหมดภาคสนาม)
# - True  = ใช้ SimCam4ArmController (ไม่แตะ Serial, แค่จำลองมุมแขนในซอฟต์แวร์)
CAM4_ARM_SIMULATION_MODE = False

# ----------------------------------------------------------------------------
# CAM4 ARM: แขนจริง (GRBL) — ใช้เมื่อ CAM4_ARM_SIMULATION_MODE = False
# ----------------------------------------------------------------------------
# เปิดใช้แขนจริง (ต้อง True เพื่อส่ง G-code ไป Serial)
CAM4_ARM_ENABLED = True  # ต้อง True บน Jetson ถึงจะ homing แขนก่อนรัน (ถ้า False จะไม่มีข้อความ Cam4ArmController ใน log)
# Serial — udev symlink คงที่ (Jetson: /etc/udev/rules.d/*cam4_arm* → ไม่ขึ้นกับชื่อโฟลเดอร์โปรเจกต์)
# fallback ถ้าไม่มี udev: /dev/ttyUSB0 หรือ /dev/ttyACM0
CAM4_ARM_SERIAL_PORT = "/dev/cam4_arm"
CAM4_ARM_BAUD_RATE = 115200
CAM4_ARM_SERIAL_TIMEOUT = 0.1
# ขีดจำกัด X,Y หน่วย mm จาก home — ใส่ safety margin ไม่ให้สั่งไปถึง limit จริง (กันค้าง)
# จริงวัดได้ ±70 mm X, ±40 mm Y จาก OpenBuilds; ใช้ ±65, ±35 เพื่อกันชน limit
CAM4_ARM_X_LIMITS = (-65.0, 65.0)
CAM4_ARM_Y_LIMITS = (-35.0, 35.0)
# Safety margin ด้านใน limit (หน่วยเดียวกับ limit) — clip ไม่ให้ถึงขอบจริง ลดโอกาสเกิน limit เมื่อ drift
CAM4_ARM_LIMIT_SAFETY_MARGIN = 2.0
# สลับ pan/tilt ตอนส่ง G0 ใน calibrator Test: True = ส่ง tilt→X, pan→Y (ใช้เมื่อแขนได้แค่ขึ้น-ลง ไม่ได้ซ้าย-ขวา)
CAM4_ARM_SWAP_PAN_TILT = False

# ============================================================================
# โดรนเสมือน (virtual target) — ทดสอบโหมด LOCK กับแขนจริง+กล้องจริง โดยไม่ต้องบินโดรนจริง
# โดรนถูกฉีดลงเฟรมกล้องสด: ตำแหน่งอิง bearing โลกจริง project ด้วยมุมแขนจริง
# → แขนขยับ ภาพเลื่อนจริง (ego-motion) เหมือนติดตามเป้าจริง (ดู lock_sim_target.py)
# ⚠️ ต้อง False ในการใช้งานจริง — เปิดเฉพาะตอนทดสอบ
# ============================================================================
LOCK_SIM_TARGET_ENABLED = False       # default โหมดจริง (ไม่มีโดรนจำลอง) — กดปุ่ม I ในแอปเพื่อเปิด/ปิดตอน runtime
LOCK_SIM_TARGET_PATTERN = "realistic" # realistic | sine | hover | figure8 | manual (i/j/k/l)
                                      # realistic = บินอิสระ waypoint (เร่ง/หยุด/หักหลบ/โฉบ) เหมือนจริง
LOCK_SIM_TARGET_OMEGA_DEG_S = 8.0     # peak angular rate สำหรับ sine/figure8 (deg/s)
LOCK_SIM_TARGET_AMP_DEG = 15.0        # แอมพลิจูดการกวาด pan (deg)
LOCK_SIM_TARGET_TILT_AMP_DEG = 5.0    # แอมพลิจูด tilt (deg)
LOCK_SIM_TARGET_BOX_DEG = 0.6         # ใช้เมื่อไม่กำหนดระยะ (ปกติคำนวณจากระยะ)
# realistic flight: ระยะ + ความเร็วโดรนจริง → ขนาดเชิงมุม/ความเร็วเชิงมุมคำนวณเอง
LOCK_SIM_TARGET_RANGE_M = 200.0       # ระยะโดรน (m) — 200m สมจริง (มุมกวาดช้า ~5°/s ลื่นต่อเนื่อง)
LOCK_SIM_TARGET_SIZE_M = 0.30         # ขนาดจริงโดรน (m)
LOCK_SIM_TARGET_MAX_SPEED_MS = 18.0   # ความเร็วบินสูงสุด (m/s)
LOCK_SIM_TARGET_MAX_ACCEL_MS2 = 12.0  # อัตราเร่ง (m/s²)
LOCK_SIM_TARGET_MISS_RATE = 0.15      # อัตรา YOLO miss จำลอง (0–1)
LOCK_SIM_TARGET_CLASS_ID = 0          # class id ของ detection ที่ฉีด (ให้ตรง active class)
LOCK_SIM_TARGET_INJECT_DETECTION = True   # True=ฉีด detection ตรง ๆ (ไม่พึ่ง YOLO ตรวจสไปรต์)
                                          # False=วาดสไปรต์อย่างเดียว ให้ YOLO จริงตรวจเอง (end-to-end)
LOCK_SIM_TARGET_DRAW = True           # วาดสไปรต์โดรนบนจอ (ให้ผู้ควบคุมเห็น)
# ซิงก์ตำแหน่งจาก GRBL ทุก N ครั้งที่เรียก move_relative (0 = ปิด) — ลด drift ค่ามุมเมื่อคุมแขนนาน
CAM4_ARM_SYNC_POSITION_EVERY_N_MOVES = 8
# ช่วงเวลาขั้นต่ำ (ms) ระหว่างส่ง G0 ตอนไม่ blocking — ลดการยิงคำสั่งรัวๆ ตอนจอยสติ๊กสั่งถี่ (กัน overshoot ที่ limit)
# โหมด P zone step: ใช้ 20 เพื่อให้ throttle 0.02 s ทำงานจริง (เนียน เร็ว); ถ้า GRBL รับไม่ทันให้ใช้ 60
CAM4_ARM_MIN_MOVE_INTERVAL_MS = 10  # ลดจาก 20 → ส่ง G-code ถี่ขึ้นเมื่อ throttle < 20ms (ถ้า GRBL กระตุกให้คืนเป็น 20)
# Reference pose (home) หน่วยองศา — ใช้หลัง homing
CAM4_ARM_REF_PAN_DEG = 0.0
CAM4_ARM_REF_TILT_DEG = 0.0
# Home ทุกครั้ง: เปิดโปรแกรม = homing + ไป home; ปิดโปรแกรม = กลับ home ก่อนตัด serial (ไม่แนะนำปิด)
CAM4_ARM_RUN_HOMING_ON_START = True
CAM4_ARM_RETURN_TO_REF_ON_DISCONNECT = True
# วินาทีที่รอหลังส่งคำสั่งกลับ home ก่อนปิด serial (ให้แขนขยับกลับทัน)
CAM4_ARM_RETURN_TO_REF_WAIT_SEC = 3.0
# หน่วย GRBL เป็น mm (G21); สเกลองศา -> mm (ปรับตามเครื่องจริง / OpenBuilds)
CAM4_ARM_GRBL_UNITS_MM = True
CAM4_ARM_MM_PER_DEG_PAN = 1.0
CAM4_ARM_MM_PER_DEG_TILT = 1.0
# Reference position ใน mm (ถ้าไม่ตั้ง จะใช้ REF_PAN_DEG/TILT_DEG * MM_PER_DEG)
CAM4_ARM_REF_PAN_MM = None
CAM4_ARM_REF_TILT_MM = None
# ใช้คำสั่งจ็อก $J= สำหรับ move_relative (เหมือน OpenBuilds)
CAM4_ARM_USE_JOG_FOR_RELATIVE = True
# Feed rate (mm/min) — ใส่ในทุกคำสั่งเคลื่อนที่
CAM4_ARM_FEED_RATE = 12000  # mm/min (167 mm/s) — ทดสอบเพิ่มทีละ 5000: 10000→15000→20000 ถ้าแขนไม่กระตุก
# ความเร็วตอนกลับ home เท่านั้น (ช้ากว่า FEED_RATE ปกติ)
CAM4_ARM_HOME_FEED_RATE = 3000
# อื่นๆ (G92 ที่ ref, blocking delay, reconnect)
CAM4_ARM_USE_G92_AT_REF = True
CAM4_ARM_BLOCKING_MOVE_DELAY = 0.2
CAM4_ARM_RECONNECT_MAX_ATTEMPTS = 5
CAM4_ARM_RECONNECT_DELAY = 3.0
# หลังแขนหลุด/power-cycle: ลอง homing+home ใหม่โดยไม่ต้องปิดโปรแกรม (กด P = บังคับ reconnect)
CAM4_ARM_AUTO_RECONNECT_ENABLED = True
# หลังไป home แล้ว: ส่งคำสั่งทดสอบ (ขยับเล็กน้อยทุกทิศ) รอตอบกลับ แล้วกลับ home — ใช้ตรวจว่าแขนตอบสนอง
CAM4_ARM_RUN_SELFTEST_ON_CONNECT = False  # ปิด: self-test ขยับ ±2mm ทุก connect/reconnect ทำแขนกระตุกตอน liveness วนลูป
CAM4_ARM_SELFTEST_MOVE_MM = 2.0  # ขยับทดสอบ ±mm หลัง homing (ลดจาก 10 ถ้าแขนไม่ขยับ)
# ตรวจแขนขณะ idle (ไม่มีคำสั่งขยับจาก AUTO/MANUAL/mapping): nudge แล้วรอ Idle + อ่าน WPos
CAM4_ARM_IDLE_PROBE_ENABLED = True
CAM4_ARM_IDLE_PROBE_INTERVAL_SEC = 60.0   # อย่างน้อยเท่านี้ระหว่าง probe แต่ละครั้ง
CAM4_ARM_IDLE_PROBE_MIN_QUIET_SEC = 8.0   # ต้องไม่มีคำสั่ง G0/joy มานานเท่านี้ก่อน probe
CAM4_ARM_IDLE_PROBE_DELTA_DEG = 1.0       # สั่งขยับทดสอบ (องศา)
CAM4_ARM_IDLE_PROBE_MIN_MOVED_DEG = 0.15  # ต้องขยับอย่างน้อยเท่านี้ถึงถือว่าตอบสนอง
CAM4_ARM_IDLE_PROBE_VERIFY_ATTEMPTS = 2   # วัด WPos หลังขยับ 2 รอบ (รอบแรกอาจเร็วไป)
CAM4_ARM_IDLE_PROBE_VERIFY_RETRY_DELAY_SEC = 0.2  # ห่างระหว่างรอบวัด
CAM4_ARM_IDLE_PROBE_POST_IDLE_DELAY_SEC = 0.15    # หลัง GRBL Idle ก่อนอ่าน WPos
CAM4_ARM_IDLE_PROBE_FEED_RATE = 3000      # mm/min ช้าๆ ตอน probe
CAM4_ARM_IDLE_PROBE_GRBL_RETRIES = 3      # retry อ่าน ? ก่อนถือว่า fail
CAM4_ARM_IDLE_PROBE_GRBL_RETRY_DELAY_SEC = 0.15
CAM4_ARM_IDLE_PROBE_GRBL_TIMEOUT_SEC = 2.5
CAM4_ARM_IDLE_PROBE_FAILS_BEFORE_HOMING = 2  # motion fail เท่านั้น → STALL+homing
CAM4_ARM_PROBE_FAIL_RETRY_INTERVAL_SEC = 5.0  # หลัง motion WARN แล้ว probe ถี่ขึ้น
CAM4_ARM_GRBL_READ_PROBE_INTERVAL_SEC = 120.0  # หลัง grbl-read fail อย่า probe ถี่ (log only)
# ตรวจ GRBL เป็นระยะ — จับถอด USB/ปลั๊กแม้ ser.is_open ยัง True
# ⚠️ ปิด liveness: probe ? ล้ม 100% (false positive) แม้ GRBL มีชีวิต (reconnect สำเร็จทุกครั้ง +
# เทสต์ตรงได้ <Idle|..>) — น่าจะชนกับ thread อื่นที่ใช้ serial ร่วม → วน reconnect + reset โหมดทุก ~20 วิ
# การจับ fault จริงยังมี motion-probe ตอนสั่ง move (ไม่เสียการป้องกัน). เปิดใหม่ได้ถ้าแก้สาเหตุ probe แล้ว
CAM4_ARM_LIVENESS_CHECK_ENABLED = False
CAM4_ARM_LIVENESS_INTERVAL_SEC = 8.0
CAM4_ARM_LIVENESS_GRBL_TIMEOUT_SEC = 3.0   # เดิม 1.5 — ให้เวลา GRBL ตอบ ? มากขึ้น (กัน false "serial lost")
CAM4_ARM_LIVENESS_FAILS_BEFORE_DISC = 4    # เดิม 2 — ทนหลาย probe ก่อนตัดสินว่าหลุด (กัน reconnect วนลูป)
CAM4_ARM_PAUSE_PROBE_ON_CAMERA_LOSS = True
CAM4_ARM_PAUSE_ARM_RECONNECT_ON_CAMERA_LOSS = True
CAM4_ARM_POST_CAMERA_OK_PROBE_GRACE_SEC = 30.0  # หลัง CAM:OK ก่อน idle-probe รอบแรก
CAM4_ARM_PAUSE_PROBE_ON_JOYSTICK = True
# หลัง joy/AUTO/socket สั่งขยับ: รอแล้วเช็ค WPos — ไม่หมุน → STALL+homing ทันที
CAM4_ARM_DRIVE_VERIFY_ENABLED = True
CAM4_ARM_DRIVE_VERIFY_MIN_CMD_DEG = 0.02   # รวม delta ต่อเฟรมจากจoy (ไม่ใช่ต่อครั้ง ≥ ค่านี้)
CAM4_ARM_DRIVE_VERIFY_DELAY_SEC = 0.45
CAM4_ARM_DRIVE_VERIFY_MIN_MOVED_DEG = 0.04
# หลัง connect/reconnect สำเร็จ: ไม่ idle-probe ช่วงนี้ (กัน ERR/homing วนหลังเพิ่ง home)
CAM4_ARM_POST_CONNECT_PROBE_GRACE_SEC = 90.0
# STALL ค้าง → ปิด serial ให้ auto-reconnect homing (หลังถอด/เสียบไฟ ไม่ต้องกด P)
CAM4_ARM_STALL_AUTO_RECONNECT_ENABLED = True
CAM4_ARM_STALL_AUTO_RECONNECT_SEC = 30.0       # STALL อย่างน้อยเท่านี้ + GRBL ? ไม่ตอบ
CAM4_ARM_STALL_GRBL_FAIL_COUNT = 3             # อ่าน ? ไม่ได้ติดกัน (ทุก ~2s)
CAM4_ARM_STALL_FORCE_RECONNECT_SEC = 45.0      # STALL นานเท่านี้ → reconnect แม้ ? ตอบ
CAM4_ARM_STALL_RECOVERY_COOLDOWN_SEC = 60.0    # อย่างน้อยเท่านี้ระหว่าง stall-recovery
# homing ผ่านแต่ self-test ไม่ผ่an → ยัง ARM:OK (กัน STALL หลังกด P ที่แขนหมุน home แล้ว)
CAM4_ARM_ACCEPT_HOMING_WITHOUT_SELFTEST = True
# Reconnect / idle probe ใน background thread — ไม่ block OpenCV main loop
CAM4_ARM_RECONNECT_IN_BACKGROUND = True
CAM4_ARM_IDLE_PROBE_IN_BACKGROUND = True
# ทุก fault (DISC/ERR/STALL/probe fail) → homing limit เต็mทันที + MODE:SAFE จน ARM:OK
CAM4_ARM_FAULT_AUTO_HOMING_ENABLED = True
CAM4_ARM_IMMEDIATE_FAULT_HOMING = True       # True = ไม่รอ STALL 30–45s
CAM4_ARM_FAULT_RECOVERY_COOLDOWN_SEC = 2.0   # กันยิง reconnect ซ้ำถี่เกิน
CAM4_ARM_PROBE_FAIL_TRIGGERS_HOMING = True   # idle probe ล้ม → homing เต็m (ไม่ค้าง STALL)
# รอ GRBL state=Idle หลังคำสั่งขยับ ($HY/$HX/G53) ก่อนถือว่า homing สำเร็จ
CAM4_ARM_HOMING_WAIT_IDLE = True
CAM4_ARM_HOMING_IDLE_TIMEOUT_SEC = 25.0   # เดิม 60 — ลดให้ตรวจ homing ค้างเร็วขึ้นแล้ว fall back
# homing ล้ม/ค้าง → เชื่อมต่อแบบ MANUAL JOG แทนออกโปรแกรม (เข้าหน้าจอ + ขยับจอยปรับตำแหน่งได้)
# True = ไม่ค้าง/ไม่ออก ใช้งานต่อได้ (ยังไม่ได้ home จน home ใหม่สำเร็จ); False = พฤติกรรมเดิม
CAM4_ARM_HOMING_FAIL_FALLBACK_MANUAL = True
# เช็ค MPos หลัง G53 (ปิดได้ถ้า GRBL/G53 ไม่ตรง REF — ใช้ WPos verify แทน)
CAM4_ARM_HOMING_VERIFY_MPOS = False
CAM4_ARM_HOMING_MPOS_TOLERANCE_MM = 15.0  # ใช้เมื่อ VERIFY_MPOS=True
CAM4_ARM_HOMING_VERIFY_WPOS_HOME = True   # หลัง G92 ต้องได้ WPos ≈ 0,0 (สำคัญกว่า MPos)
CAM4_ARM_HOMING_WPOS_TOLERANCE_MM = 3.0
# homing/connect ล้มเหลว → ออกจาก 22_gun_aim_assist_vector ทันที (ไม่รันต่อโดยไม่มีแขน)
# False = ไม่ออก ให้ fallback เป็น MANUAL JOG เข้าหน้าจอได้ แล้วใช้ปุ่ม J jog / H home แก้เอง
# (แขนตัวนี้ limit switch เสีย ถ้าเผลอเปิด homing จริงจะ fail — ตั้ง False กันโดนล็อกออกจากโปรแกรม)
CAM4_ARM_EXIT_IF_HOMING_FAILS = False
# พิมพ์คำสั่ง G-code ที่ส่งไป GRBL ทุกคำสั่ง (ใช้เทียบกับ OpenBuilds ได้)
CAM4_ARM_VERBOSE_GCODE = False
# แบบ v34: True = ไม่ทำ homing (แขนไม่ขยับตอนเปิด). False = รัน homing แล้วไป home ($X → $HY → $HX → G53 → G92 X0 Y0)
CAM4_ARM_SKIP_HOMING_USE_G92_ONLY = True   # แขนนี้ homing switch ไม่ทำงาน → ใช้ $X+G92 (ตำแหน่งปัจจุบัน=ศูนย์) เหมือน 24 --skip-homing
# ใช้แค่ G0 X Y สำหรับขยับ (ไม่ใช้ $J=) แบบ v34
CAM4_ARM_USE_G0_ONLY = True
# หลัง $H รอวินาทีก่อนส่งคำสั่งถัดไป (ใช้เมื่อไม่ skip homing)
CAM4_ARM_HOMING_COMPLETE_DELAY = 15.0

# ชุดคำสั่ง HOME: ส่งทีละคำสั่ง รอ ok — ลำดับ $X → $HY → $HX → G53 X Y → G92 X0 Y0 (ตาม GRBL ที่ใช้)
CAM4_ARM_HOMING_COMMANDS = ["$X", "$HY", "$HX"]
# ตำแหน่ง preset ใน machine coordinate (หลัง homing ส่ง G53 X Y แล้ว G92 X0 Y0)
CAM4_ARM_REF_MPOS_X = -77.001
CAM4_ARM_REF_MPOS_Y = -41.988
# ใช้ชุด G53 + G92 หลัง homing (ต้องตั้ง REF_MPOS_X, REF_MPOS_Y ด้วย)
CAM4_ARM_USE_G53_REF_AFTER_HOMING = True
# ถ้าไม่มีไฟล์ pixel_per_degree: False = เริ่ม MANUAL ได้ (ไม่บังคับ SAFE), True = บังคับ SAFE เหมือนเดิม
CAM4_ARM_FORCE_SAFE_WITHOUT_CALIB = False
# การขยับแบบ relative (เมื่อ USE_G0_ONLY): ใช้ G91 G21 แล้ว G0 X Y F แล้ว G90 (ไม่สนใจ home, ละเอียด 0.1 mm)
CAM4_ARM_RELATIVE_STEP_MM = 0.1  # ขั้นละเอียดสุดสำหรับ G91 (mm, 0.1 = 0.1 องศา ถ้า MM_PER_DEG=1)

# จอยสติ๊กจริง (axis 0,1 = pan/tilt; axis 3 = scale ละเอียด↔หยาบ) — ไม่ใช้กับโหมด P
CAM4_ARM_JOYSTICK_MAX_PAN_RATE_DEG = 80.0   # ความเร็วสูงสุดจอยสติกจริง (deg/s)
CAM4_ARM_JOYSTICK_MAX_TILT_RATE_DEG = 60.0
CAM4_ARM_JOYSTICK_DEADZONE = 0.05
# ระยะโยก→ความเร็ว: 2 = แบบจอยสติก (โยกน้อยช้า โยกมากเร็ว)
CAM4_ARM_JOYSTICK_STICK_EXPONENT = 2.0
CAM4_ARM_JOYSTICK_SPEED_SCALE_MIN = 0.1   # axis 3 = -1 → scale 0.1 (ละเอียด)
CAM4_ARM_JOYSTICK_SPEED_SCALE_MAX = 10.0  # axis 3 = +1 → scale 10 (หยาบ)
CAM4_ARM_JOYSTICK_SPEED_SMOOTH_ALPHA = 0.82  # low-pass เนียน (0.8–0.9 = เนียนมาก)
# โหมดความเร็วจอยสติ๊กเริ่มต้น: "low" | "medium" | "high" (กดปุ่ม 12 สลับได้ขณะรัน)
CAM4_ARM_JOYSTICK_SENSITIVITY = "medium"
# ระยะเวลายิง (วินาที) สำหรับ G-code M3 S255 → sleep → M5 (อ้างอิง v34 MANUAL_FIRE_DURATION)
CAM4_ARM_FIRE_DURATION = 0.1
# หน่วงระหว่างนัดยิง (วินาที) — ห้ามส่ง G-code ยิงซ้ำก่อนครบเวลานี้ (ให้ไก/กลไกกลับตัว)
CAM4_ARM_FIRE_COOLDOWN_SEC = 0.5
# หมายเลขปุ่มจอย (1–12) ตรงกับ check_joystick_buttons.py: ปุ่ม N = pygame index (N-1)
# ยิง = Hat ค้าง + ปุ่ม FIRE_AND_LOCK; LOCK = ปุ่ม LOCK (แยกจากปุ่มยิง)
CAM4_ARM_JOYSTICK_BUTTON_FIRE_AND_LOCK = 1   # ปุ่ม 1 (trigger) — ยิงเมื่อกดร่วมกับ Hat
CAM4_ARM_JOYSTICK_BUTTON_LOCK = 5            # ปุ่ม 5 (index 4) — สลับโหมด LOCK / ขยับแขนไป YOLO bbox
CAM4_ARM_JOYSTICK_BUTTON_UNLOCK = 3          # ปุ่ม 3 (index 2) — ปลด LOCK → MANUAL
CAM4_ARM_JOYSTICK_BUTTON_LOCK_CSRT = 6       # ปุ่ม 6 (index 5) — ขยับแขนไป CSRT bbox (เหมือนปุ่ม 5 สำหรับ YOLO)
CAM4_ARM_JOYSTICK_BUTTON_AUTO = 7            # ปุ่ม 7
CAM4_ARM_JOYSTICK_BUTTON_MANUAL = 9          # ปุ่ม 9
CAM4_ARM_JOYSTICK_BUTTON_SAFE = 11           # ปุ่ม 11
CAM4_ARM_JOYSTICK_BUTTON_SENSITIVITY_CYCLE = 12  # ปุ่ม 12 — สลับโหมดความเร็ว (ช้า/กลาง/สูง)
# สลับ YOLO target class ทีละตัว — กดค้างแล้วเลื่อนซ้ำ (ไม่ใช้ปุ่ม 1 เพื่อไม่ปนกับยิง)
# ห้ามใช้เลขเดียวกับ CAM4_ARM_JOYSTICK_BUTTON_FIRE_AND_LOCK — จะทับกันที่ pygame index
CAM4_ARM_JOYSTICK_BUTTON_CLASS_CYCLE = 8  # ปุ่ม 8 (ว่างในแผนที่ปุ่มเริ่มต้น; ไม่ชนกับ 7=AUTO / 9=MANUAL)
CAM4_ARM_JOYSTICK_CLASS_CYCLE_REPEAT_INITIAL_SEC = 0.35  # หลังขั้นแรก รอแค่ไหนก่อนเริ่มเลื่อนต่อเนื่อง
CAM4_ARM_JOYSTICK_CLASS_CYCLE_REPEAT_INTERVAL_SEC = 0.14  # ระยะห่างระหว่างขั้นขณะกดค้าง
# สลับ TensorRT detection engine (edge trigger): multiclass = RGB↔Thermal | drone_only = 640↔1280
CAM4_ARM_JOYSTICK_BUTTON_DETECTION_TOGGLE = 10  # ปุ่ม 10 (index 9) — ไม่ชนกับปุ่มอื่นในแผนที่เริ่มต้น

# --- YOLO detection สำหรับ 22_gun_aim_assist_vector ---
# โหมด "multiclass" = rgb/thermal 8 class + สลับ class ด้วยจอย
# โหมด "drone_only" = yolo_11n_day_night โดรนอย่างเดียว + สลับ 640/1280 ด้วยปุ่ม 10 / T
CAM4_ARM_YOLO_DETECTION_MODE = "drone_only"
# multiclass engines (ใช้เมื่อ CAM4_ARM_YOLO_DETECTION_MODE = "multiclass")
CAM4_ARM_YOLO_ENGINE_RGB_640 = "rgb_multiclass_imgsz640.engine"
CAM4_ARM_YOLO_ENGINE_THERMAL_640 = "thermal_multiclass_imgsz640.engine"
CAM4_ARM_YOLO_ENABLE_CLASS_CYCLE = True
# drone_only engines (ใช้เมื่อ CAM4_ARM_YOLO_DETECTION_MODE = "drone_only")
CAM4_ARM_YOLO_ENGINE_DRONE_640 = "yolo_11n_day_night_200_2_imgsz640.engine"
CAM4_ARM_YOLO_ENGINE_DRONE_1280 = "yolo_11n_day_night_200_2_imgsz1280.engine"
CAM4_ARM_YOLO_USE_FAST = True  # True = เริ่มที่ 640 (เร็ว), False = เริ่มที่ 1280 (แม่นขึ้น); ปุ่ม 10/T สลับอีกขนาด
CAM4_ARM_YOLO_LOAD_ALT_ON_START = False  # False = เปิดเร็ว (โหลด engine สำรองตอนกดปุ่ม 10/T ครั้งแรก)

# โหมด AUTO/LOCK: step ตามระยะศูนย์เล็ง–เป้า (ใกล้ละเอียด ไกลเร็ว) + exponential smooth
CAM4_ARM_AIM_STEP_VERY_NEAR_MM = 0.05  # step เมื่อใกล้มาก (ลดส่ายละเอียด)
CAM4_ARM_AIM_STEP_NEAR_MM = 0.15  # step ต่อครั้งเมื่อใกล้ศูนย์ (mm) — เร็วขึ้น
CAM4_ARM_AIM_STEP_FAR_MM = 1.5    # step ต่อครั้งเมื่อห่าง (mm) — ตามวัตถุเคลื่อนที่เร็ว
CAM4_ARM_AIM_ERROR_VERY_NEAR_MM = 0.2  # ใต้ค่านี้ใช้ STEP_VERY_NEAR
CAM4_ARM_AIM_ERROR_NEAR_MM = 0.3   # ใต้ค่านี้ถือว่า "ใกล้" ใช้ STEP_NEAR — ใช้ step ใหญ่บ่อยขึ้น
# โหมด P (กด P แล้วคลิก): error เกิน REF_SCALE = ความเร็วเต็ม (เล็กลง = ใกล้ก็เร็ว)
CAM4_ARM_AIM_REF_SCALE_DEG = 6.0  # โหมด P: error เกินนี้ = ความเร็วเต็ม
CAM4_ARM_AIM_SMOOTH_ALPHA = 0.85   # exponential smoothing บน delta (ยิ่งสูงยิ่งเนียน)
# Low-pass บน error ใน tracker — ยิ่งสูง error กระตุกน้อย
CAM4_ARM_FILTER_ALPHA = 0.82

# โหมด P/L continuous: ค่าของโหมด P เอง (ไม่แชร์กับจอยสติกจริง) — ใช้ใน calibrator Test
CAM4_ARM_CONTINUOUS_MAX_RATE_DEG = 240.0    # อัตราสูงสุดโหมด P (deg/s) — เร็วสุด
CAM4_ARM_CONTINUOUS_STICK_EXPONENT = 2.0    # power curve โหมด P (ห่างมากโยกมาก)
CAM4_ARM_CONTINUOUS_MAX_SPEED_MM_S = CAM4_ARM_FEED_RATE / 60.0   # cruise สูงสุด (mm/s) = อ้างอิง feed rate
CAM4_ARM_CONTINUOUS_ACCEL_MM_S2 = 650.0    # อัตราเร่ง (mm/s²) — สูงขึ้นเข้าเป้าเร็ว
CAM4_ARM_CONTINUOUS_DECEL_MM_S2 = 650.0   # อัตราเบรก (mm/s²)
CAM4_ARM_CONTINUOUS_THROTTLE_SEC = 0.01   # ช่วงส่ง move (ต่ำ = ส่งบ่อย = เร็ว)
CAM4_ARM_CONTINUOUS_STEP_SCALE_PX = 18.0  # scale สำหรับ max_step_mm = error_px/scale (เล็กลง = step ใหญ่ = เร็ว)
CAM4_ARM_CONTINUOUS_DEADZONE_MM = 0.5    # หยุดส่ง move เมื่อ error_mm ≤ ค่านี้ (0.5=ลดเด้ง LOCK; 0.01=แม่นยำสูงแต่สั่นได้)
# ใกล้เป้า: ความเร็วขั้นต่ำเพื่อตามวัตถุทัน (smooth ไม่กระตุก)
CAM4_ARM_CONTINUOUS_MIN_NEAR_SPEED_MM_S = 50.0   # ความเร็วขั้นต่ำเมื่ออยู่ใกล้เป้า (mm/s)
CAM4_ARM_CONTINUOUS_NEAR_THRESHOLD_DEG = 3.0     # error_deg ≤ ค่านี้ = โซนใกล้ ใช้ min speed
# ไกล: เพิ่ม step ให้ปิดระยะเร็ว (แนวทาง 2 ไม่ยุ่ง smooth_target)
CAM4_ARM_CONTINUOUS_FAST_THRESHOLD_DEG = 5.0    # error_deg > ค่านี้ = โซนไกล ใช้ step ใหญ่ขึ้น
CAM4_ARM_CONTINUOUS_FAST_STEP_BOOST = 2.0       # ตัวคูณ step เมื่อไกล (เร็วขึ้น แบบ homography test)
CAM4_ARM_CONTINUOUS_ERROR_SMOOTH_ALPHA = 0.85   # smoothing error (0.85 = ตอบสนองเร็ว, 0.4 = เนียนมาก)

# โหมด L sim โดรน: เป้ารอบมุมแขน 0,0 ไม่เกินกี่องศา
CAM4_ARM_DRONE_SIM_MAX_DEG = 15.0   # ขอบเขตมุมสูงสุดรอบ 0,0 (องศา)

# โหมด P zone step (ตามโดรนทัน มั่นยำ ไม่กระตุก): step ตามระยะ + throttle 0.02 s
CAM4_ARM_CONTINUOUS_ZONE_FAR_DEG = 5.0    # โซนไกล (step 3 mm)
CAM4_ARM_CONTINUOUS_VERY_FAR_DEG = 10.0   # โซนห่างมาก → step ใหญ่ เร่งตามโดรน (L/P)
CAM4_ARM_CONTINUOUS_ZONE_NEAR_DEG = 0.5   # โซนใกล้ (step 0.1 mm)
CAM4_ARM_CONTINUOUS_STEP_FAR_MM = 3.0     # 3 mm/0.02s = 150 mm/s < feed
CAM4_ARM_CONTINUOUS_STEP_VERY_FAR_MM = 8.0   # ห่างมาก (error > VERY_FAR_DEG) → step ใหญ่ ตามทัน
CAM4_ARM_CONTINUOUS_STEP_MID_MM = 1.0
CAM4_ARM_CONTINUOUS_STEP_NEAR_MM = 0.1
CAM4_ARM_CONTINUOUS_THROTTLE_FAR_SEC = 0.01   # ลดจาก 0.02 → FAR zone ส่ง G0 ถี่ 2x
CAM4_ARM_CONTINUOUS_THROTTLE_MID_SEC = 0.015  # ลดจาก 0.02 → MID zone เร็วขึ้น
CAM4_ARM_CONTINUOUS_THROTTLE_NEAR_SEC = 0.02  # คงเดิม → NEAR ไม่สั่น
# แขนต้องเร็วกว่าเป้า (โดรน): step อย่างน้อย = ความเร็วเป้า × dt × ตัวคูณนี้
CAM4_ARM_CONTINUOUS_ARM_FASTER_THAN_TARGET_FACTOR = 1.5  # เพิ่มจาก 1.2 → เร็วกว่าเป้า 50%

# Deadzone + hysteresis เพื่อไม่ให้ศูนย์เล็งสั่นเมื่ออยู่ในกรอบเป้า (สีส้ม)
CAM4_ARM_DEADZONE_DEG = 0.25       # deadzone ปกติ (องศา)
CAM4_ARM_DEADZONE_NEAR_DEG = 0.65  # ขยาย deadzone เมื่อใกล้ศูนย์ (โซนสีส้ม) เพื่อไม่สั่น
CAM4_ARM_DEADZONE_HOLD_DEG = 0.28  # hysteresis: ใต้ค่านี้ถือว่า "นิ่ง" หยุดขยับ
CAM4_ARM_DEADZONE_TRACK_DEG = 0.6  # hysteresis: เกินค่านี้ถึงค่อยขยับอีก (ต้อง > HOLD)

# Learned aim controller (MLP เรียนจากโหมด AUTO/LOCK)
# ปิดไว้ก่อน — เปิดเมื่อมีข้อมูลเทรนมากพอและ model ทำงานได้ดี (ถ้าแย่กว่า PD ให้ใช้ BLEND_PD หรือปิด)
CAM4_ARM_USE_LEARNED_AIM_MODEL = False
CAM4_ARM_LEARNED_AIM_MODEL_PATH = "aim_controller_model/aim_model.pt"
# เมื่อเปิดใช้ model: ผสมกับ PD เพื่อความเสถียร (0 = ใช้แค่ model, 0.5 = ครึ่ง model ครึ่ง PD, 1 = ใช้แค่ PD)
CAM4_ARM_LEARNED_AIM_BLEND_PD = 0.5
CAM4_ARM_AIM_COLLECT_DATA = True  # เก็บ state/action/next_state เพื่อรีเทรนตอนปิด
CAM4_ARM_AIM_COLLECT_DATA_MODES = "lock"  # เก็บเฉพาะโหมดที่ระบุ: "lock" | "auto" | "auto,lock"
CAM4_ARM_LEARNED_AIM_MIN_TRANSITIONS = 20  # ขั้นต่ำจำนวน transition ก่อนจะเขียนไฟล์และรีเทรน
CAM4_ARM_LEARNED_AIM_THRESHOLD_RED_DEG = 0.35   # good = next_error_deg ≤ ค่านี้ (แม่นมาก)
CAM4_ARM_LEARNED_AIM_THRESHOLD_ORANGE_DEG = 0.7  # good = next_error_deg ≤ ค่านี้ (แม่นรอง) หรือ error ลดลง
CAM4_ARM_LEARNED_AIM_WEIGHT_RED = 1.5    # น้ำหนัก sample แดง (แม่นมาก) ในการเทรน
CAM4_ARM_LEARNED_AIM_WEIGHT_ORANGE = 1.0  # น้ำหนัก sample ส้ม (แม่นรอง)

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
# Startup — รอเฟรมแรก + retry ก่อน exit (Jetson GStreamer หลัง USB glitch / process เก่าค้าง)
# กล้อง DISC/WAIT/FAIL → บังคับ MODE:SAFE ห้ามสลับโหมด/ขยับแขน/ยิงจน CAM:OK
CAM_FORCE_SAFE_ON_CAMERA_LOSS = True
CAMERA_STARTUP_RETRIES = 3
CAMERA_STARTUP_RETRY_DELAY_SEC = 2.0
CAMERA_STARTUP_POLL_SEC = 0.033
CAMERA_STARTUP_WAIT_SEC = 0.0       # 0 = auto จากความละเอียดกล้อง
CAMERA_STARTUP_WAIT_HD_SEC = 8.0
CAMERA_STARTUP_WAIT_2K_SEC = 15.0
CAMERA_STARTUP_WAIT_4K_SEC = 30.0   # cam4 3840×2160

# กล้องที่ใช้งานปัจจุบัน (เปลี่ยนที่นี่เพื่อสลับกล้อง)
ACTIVE_CAMERA = "cam4"  # เลือก: "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"

# Camera configurations dictionary - แบ่ง parameter ตามชื่อกล้อง
CAMERAS = {
    "cam1": {
        "name": "cam1",
        "width": 2560,
        "height": 1440,
        "video_filename": "55.mp4",
        "use_video_file": True,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.1.203:554/Streaming/channels/201",
        "window_name": "FAST HUD DETECTOR - cam1",
        "display_max_width": None,  # None = auto-detect from screen
        "display_max_height": None,  # None = auto-detect from screen
        "horizon_file": "horizon_poly_cam1.npy",  # ไฟล์เส้นขอบฟ้าสำหรับกล้องนี้
        # FOV (Field of View) - อัปเดตตามข้อมูลจริง
        "fov_horizontal": 96.1,  # องศา (horizontal FOV)
        "fov_vertical": 52.1,    # องศา (vertical FOV)
        "fov_diagonal": 113.3,   # องศา (diagonal FOV)
        # Zoom parameters - ใส่เฉพาะกล้องที่มี zoom capability
        # ถ้ากล้องซูมไม่ได้ ไม่ต้องใส่ 3 parameters นี้
        # "fov_tele_horizontal": None,  # FOV ที่ tele/zoom max
        # "fov_tele_vertical": None,    # FOV ที่ tele/zoom max
        # "zoom_max": None,             # ระดับ zoom สูงสุด (เช่น 25x)
    },
    "cam2": {
        "name": "cam2",
        "width": 2560,
        "height": 1440,
        "video_filename": "DroneNighttime.mp4",
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.1.203:554/Streaming/channels/101",
        "window_name": "FAST HUD DETECTOR - cam2",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam2.npy",
        # FOV (Field of View) - อัปเดตตามข้อมูลจริง
        "fov_horizontal": 55.0,  # FOV ที่ wide (1x zoom)
        "fov_vertical": 33.0,    # FOV ที่ wide (1x zoom)
        "fov_diagonal": 61.5,    # FOV ที่ wide (1x zoom)
        # Zoom parameters - เพิ่มใหม่สำหรับ PTZ camera
        "fov_tele_horizontal": 2.4,  # FOV ที่ tele (zoom max)
        "fov_tele_vertical": 1.4,    # FOV ที่ tele (zoom max)
        "fov_tele_diagonal": 2.8,    # FOV ที่ tele (zoom max)
        "zoom_max": 25.0,  # 25x optical zoom (ประมาณจาก 55/2.4 ≈ 22.9x)
    },
    "cam3": {
        "name": "cam3",
        "width": 1280,
        "height": 720,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Passw0rd@192.168.144.201:554/Streaming/channels/201",
        "udp_ip": "192.168.144.201",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "window_name": "FAST HUD DETECTOR - cam3",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam3.npy",
        "fov_horizontal": 66.0,  # FOV ที่ wide (1x zoom)
        "fov_vertical": 33.0,
        # มี zoom - ใส่ parameters
        "fov_tele_horizontal": 2.4,  # FOV ที่ tele (zoom max)
        "fov_tele_vertical": 1.4,
        "zoom_max": 25.0,  # 25x optical zoom
    },
    "cam4": {
        "name": "cam4",
        "width": 3840,
        "height": 2160,
        "video_filename": "55.mp4",
        "use_video_file": False,  # เปลี่ยนเป็น False เพื่อใช้ UDP stream
        "rtsp_url": "rtsp://admin:Things22@192.168.144.15/11",
        "udp_ip": "192.168.144.15",
        "udp_port": 6600,
        "use_udp_direct": True,
        "stream_format": "h264",
        "window_name": "FAST HUD DETECTOR - cam4",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam4.npy",
        # FOV ต้องตรงกับ pixel_per_degree ที่คาลิเบรตจริง: fov = width/|ppd|
        # (เดิมตั้ง 60/36 ตามสเปคเลนส์ แต่ ppd ที่วัดได้ 87.138/-89.734 บอกว่าจริง ๆ คือ 44/24
        #  → estimate_distance_m() อ่านระยะสั้นกว่าจริง ~30% → hit_radius_deg พองเกิน → fire gate หลวม)
        # ค่าที่ใช้จริงตอนรัน derive จาก ppd โดยตรง (ดู _effective_fov_deg) — ค่าที่นี่เป็น fallback
        "fov_horizontal": 44.1,  # องศา = 3840 / 87.138
        "fov_vertical": 24.1,    # องศา = 2160 / 89.734
        # cam4 = 4K H264 UDP @ 10fps บน Jetson (sensor readout cap 10fps ที่ฮาร์ดแวร์)
        # → ภาพหน่วง ~330ms (3 เฟรม buffer+decode) วัดด้วย wizard (W) บนเครื่องจริง
        # ego_comp = latency กล้อง + servo lag แขน (หาท่าแขนตอนเก็บภาพ) — วัดได้ 377ms
        "ego_comp_latency_sec": 0.33,
        # cam_latency = latency กล้องล้วน (ไม่รวม servo) — ใช้ทำนายชดเชย 'ตำแหน่งเป้าเก่า'
        # ตั้งค่าวัดจริงไว้ (ไม่ใช่ 0) → lead ชดเชยตั้งแต่เปิดโปรแกรม ไม่ต้องรอกด W
        # sim: โดน% ที่ 20m/2°s พุ่งจาก 75% (ชดเชย 0) เป็น 100% (ชดเชย 330ms)
        # กด W เพื่อวัดใหม่แม่นขึ้น (แยก servo lag ออกให้อัตโนมัติ)
        "cam_latency_sec": 0.33,
        # ไม่มี zoom - ไม่ต้องใส่ zoom parameters
    },
    "cam5": {
        "name": "cam5",
        "width": 1280,
        "height": 720,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": 0,  # Webcam device index (0 = default webcam)
        "window_name": "FAST HUD DETECTOR - cam5",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": None,  # ไม่มี horizon file สำหรับ webcam
        "fov_horizontal": 60.0,  # องศา (ปรับตาม webcam จริง)
        "fov_vertical": 36.0,    # องศา (ปรับตาม webcam จริง)
        "is_webcam": True,  # ระบุว่าเป็น webcam
    },
    "cam6": {
        "name": "cam6",
        "width": 1280,
        "height": 720,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": "rtsp://192.168.144.108:554/stream=1",
        "udp_ip": "192.168.144.108",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "window_name": "FAST HUD DETECTOR - cam6",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam6.npy",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam7": {
        "name": "cam7",
        "width": 1280,
        "height": 720,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": "rtsp://192.168.144.108:555/stream=2",
        "udp_ip": "192.168.144.108",
        "udp_port": 555,
        "use_udp_direct": True,
        "stream_format": "h265",
        "window_name": "FAST HUD DETECTOR - cam6",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam7.npy",
        "fov_horizontal": 60.0,
        "fov_vertical": 36.0,
    },
    "cam8": {
        "name": "cam8",
        "width": 5120,
        "height": 1440,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Things22@192.168.144.112:554/Streaming/channels/101",
        "udp_ip": "192.168.144.112",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "window_name": "FAST HUD DETECTOR - cam8",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam8.npy",
        "fov_horizontal": 180.0,
        "fov_vertical": 40.0,

    },
    "cam9": {
        "name": "cam9",
        "width": 5120,
        "height": 1440,
        "video_filename": None,
        "use_video_file": False,
        "rtsp_url": "rtsp://admin:Things22@192.168.144.113:554/Streaming/channels/101",
        "udp_ip": "192.168.144.113",
        "udp_port": 554,
        "use_udp_direct": True,
        "stream_format": "h265",
        "window_name": "FAST HUD DETECTOR - cam9",
        "display_max_width": None,
        "display_max_height": None,
        "horizon_file": "horizon_poly_cam9.npy",
        "fov_horizontal": 180.0,
        "fov_vertical": 40.0,

    },
}

# ============================================================================
# CAMERA CONFIGURATION FUNCTIONS
# ============================================================================
def get_camera_config(camera_name=None):
    """
    Get configuration for a specific camera, or active camera if None

    Args:
        camera_name: ชื่อกล้อง ("cam1", "cam2", etc.) หรือ None เพื่อใช้ active camera

    Returns:
        dict: Camera configuration dictionary
    """
    if camera_name is None:
        camera_name = ACTIVE_CAMERA
    if camera_name not in CAMERAS:
        raise ValueError(f"Camera '{camera_name}' not found in CAMERAS config. Available: {list(CAMERAS.keys())}")
    return CAMERAS[camera_name]

def has_zoom(camera_name=None):
    """
    ตรวจสอบว่ากล้องมี zoom capability หรือไม่

    Args:
        camera_name: ชื่อกล้อง (None = ใช้ active camera)

    Returns:
        bool: True ถ้ากล้องมี zoom, False ถ้าไม่มี
    """
    cam_config = get_camera_config(camera_name)
    return (
        "fov_tele_horizontal" in cam_config and
        cam_config["fov_tele_horizontal"] is not None and
        "zoom_max" in cam_config and
        cam_config["zoom_max"] is not None
    )

def get_pixel_to_degree(camera_name=None):
    """
    คำนวณ pixel-to-degree conversion factors สำหรับกล้องที่เลือก

    Args:
        camera_name: ชื่อกล้อง (None = ใช้ active camera)

    Returns:
        (pixel_to_degree_x, pixel_to_degree_y) - องศาต่อพิกเซล
    """
    cam_config = get_camera_config(camera_name)
    width = cam_config["width"]
    height = cam_config["height"]
    fov_h = cam_config["fov_horizontal"]
    fov_v = cam_config["fov_vertical"]

    # คำนวณ: องศาต่อพิกเซล = FOV / resolution
    pixel_to_degree_x = fov_h / width   # องศาต่อพิกเซล (แกน X)
    pixel_to_degree_y = fov_v / height  # องศาต่อพิกเซล (แกน Y)

    return pixel_to_degree_x, pixel_to_degree_y

def pixel_to_angle(pixel_x, pixel_y, camera_name=None):
    """
    แปลง pixel coordinates เป็นมุมองศา (relative to center)

    Args:
        pixel_x, pixel_y: ตำแหน่ง pixel
        camera_name: ชื่อกล้อง (None = ใช้ active camera)

    Returns:
        (angle_x_deg, angle_y_deg) - มุมองศา relative to center
    """
    cam_config = get_camera_config(camera_name)
    width = cam_config["width"]
    height = cam_config["height"]
    fov_h = cam_config["fov_horizontal"]
    fov_v = cam_config["fov_vertical"]

    # คำนวณ center
    center_x = width / 2.0
    center_y = height / 2.0

    # คำนวณ offset จาก center
    offset_x = pixel_x - center_x
    offset_y = pixel_y - center_y

    # แปลงเป็นองศา: angle = (offset / frame_size) * FOV
    angle_x_deg = (offset_x / width) * fov_h
    angle_y_deg = (offset_y / height) * fov_v

    return angle_x_deg, angle_y_deg

def get_fov_at_zoom(camera_name=None, zoom_level=1.0):
    """
    คำนวณ FOV ที่ zoom level ที่กำหนด (สำหรับกล้องที่มี zoom)

    Args:
        camera_name: ชื่อกล้อง (None = ใช้ active camera)
        zoom_level: ระดับ zoom (1.0 = wide, zoom_max = tele)

    Returns:
        (fov_h, fov_v) - FOV ที่ zoom level นั้น
    """
    cam_config = get_camera_config(camera_name)
    fov_wide_h = cam_config["fov_horizontal"]
    fov_wide_v = cam_config["fov_vertical"]

    # ถ้าไม่มี zoom ให้ return FOV wide
    if not has_zoom(camera_name):
        return fov_wide_h, fov_wide_v

    # Interpolate ระหว่าง wide และ tele
    fov_tele_h = cam_config["fov_tele_horizontal"]
    fov_tele_v = cam_config["fov_tele_vertical"]
    zoom_max = cam_config["zoom_max"]

    # คำนวณ zoom factor (1.0 = wide, zoom_max = tele)
    zoom_factor = (zoom_level - 1.0) / (zoom_max - 1.0) if zoom_max > 1.0 else 0.0
    zoom_factor = max(0.0, min(1.0, zoom_factor))  # Clamp 0-1

    # Interpolate
    fov_h = fov_wide_h + (fov_tele_h - fov_wide_h) * zoom_factor
    fov_v = fov_wide_v + (fov_tele_v - fov_wide_v) * zoom_factor

    return fov_h, fov_v

def cam1_pixel_to_cam2_ptz(pixel_x, pixel_y, zoom_level=1.0, pan_offset_deg=0.0, tilt_offset_deg=0.0):
    """
    แปลง pixel coordinates จากกล้อง 1 เป็น Pan/Tilt/Zoom สำหรับกล้อง 2 (PTZ)

    Args:
        pixel_x, pixel_y: ตำแหน่ง pixel ในกล้อง 1
        zoom_level: ระดับ zoom สำหรับกล้อง 2 (1.0 = wide, zoom_max = tele)
        pan_offset_deg: offset ของ Pan (องศา) - ใช้เมื่อกล้อง 1 และ 2 ไม่ตั้งในทิศทางเดียวกัน
        tilt_offset_deg: offset ของ Tilt (องศา) - ใช้เมื่อกล้อง 1 และ 2 ไม่ตั้งในทิศทางเดียวกัน

    Returns:
        (pan_deg, tilt_deg, zoom_units) - มุม Pan/Tilt เป็นองศา, Zoom เป็น units
    """
    cam1_config = get_camera_config("cam1")
    cam2_config = get_camera_config("cam2")

    # ดึง FOV ของกล้อง 1
    cam1_width = cam1_config["width"]
    cam1_height = cam1_config["height"]
    cam1_fov_h = cam1_config["fov_horizontal"]
    cam1_fov_v = cam1_config["fov_vertical"]

    # คำนวณ center ของกล้อง 1
    cam1_center_x = cam1_width / 2.0
    cam1_center_y = cam1_height / 2.0

    # คำนวณ offset จาก center ของกล้อง 1
    offset_x = pixel_x - cam1_center_x
    offset_y = pixel_y - cam1_center_y

    # แปลงเป็นมุมองศา (relative to center ของกล้อง 1)
    angle_x_deg = (offset_x / cam1_width) * cam1_fov_h
    angle_y_deg = (offset_y / cam1_height) * cam1_fov_v

    # Pan/Tilt สำหรับกล้อง 2 (บวก offset ถ้ามี)
    pan_deg = angle_x_deg + pan_offset_deg
    tilt_deg = angle_y_deg + tilt_offset_deg

    # คำนวณ Zoom units จาก zoom_level
    zoom_max = cam2_config.get("zoom_max", 25.0)
    zoom_units = int(round(zoom_level))
    zoom_units = max(1, min(int(zoom_max), zoom_units))  # Clamp 1 to zoom_max

    return pan_deg, tilt_deg, zoom_units

def cam1_pixel_to_cam2_ptz_units(pixel_x, pixel_y, zoom_level=1.0, pan_offset_deg=0.0, tilt_offset_deg=0.0):
    """
    แปลง pixel coordinates จากกล้อง 1 เป็น Pan/Tilt/Zoom units สำหรับกล้อง Hikvision PTZ

    Args:
        pixel_x, pixel_y: ตำแหน่ง pixel ในกล้อง 1
        zoom_level: ระดับ zoom สำหรับกล้อง 2 (1.0 = wide, zoom_max = tele)
        pan_offset_deg: offset ของ Pan (องศา)
        tilt_offset_deg: offset ของ Tilt (องศา)

    Returns:
        (pan_units, tilt_units, zoom_units) - units สำหรับ Hikvision PTZ
        pan_units: 0-3599 (0.1° per unit)
        tilt_units: -900 to 900 (-90° to +90°, 0.1° per unit)
        zoom_units: 1 to zoom_max
    """
    pan_deg, tilt_deg, zoom_units = cam1_pixel_to_cam2_ptz(
        pixel_x, pixel_y, zoom_level, pan_offset_deg, tilt_offset_deg
    )

    # แปลงเป็น units สำหรับกล้อง Hikvision
    pan_units = int(round(pan_deg * 10.0))
    pan_units = (pan_units + 3600) % 3600  # Wrap 0-3599
    tilt_units = int(round(tilt_deg * 10.0))
    tilt_units = max(-900, min(900, tilt_units))  # Clamp -90° to +90°

    return pan_units, tilt_units, zoom_units

# ============================================================================
# RUNTIME OVERRIDES (in-app settings)
# ============================================================================
# ค่าที่ operator แก้ในแอป (หน้า Settings) เก็บใน calibration_data/runtime_config.json
# ทับค่าตั้งต้นด้านบน — ต้องทำ 'ตรงนี้' คือหลัง CAMERAS/ACTIVE_CAMERA ครบ แต่ก่อน flatten
# ด้านล่าง (CAMERA_WIDTH/RTSP_URL/... คำนวณจาก CAMERAS[ACTIVE_CAMERA] ตอน import)
# ลบไฟล์ JSON ทิ้ง = กลับสู่ค่าตั้งต้นในโค้ดทันที
try:
    import runtime_config as _runtime_config

    _rc_applied = _runtime_config.apply_to_config(globals())
    if _rc_applied:
        print(f"config: runtime override {len(_rc_applied)} ค่า -> " + ", ".join(_rc_applied[:6])
              + (" ..." if len(_rc_applied) > 6 else ""))
except Exception as _e:   # ไม่มีไฟล์/พัง → ใช้ค่าตั้งต้น ไม่ทำให้โปรแกรมไม่ขึ้น
    print(f"config: runtime_config ใช้ไม่ได้ ({_e}) — ใช้ค่าตั้งต้นจาก config.py")

# ============================================================================
# ACTIVE CAMERA CONFIGURATION (for backward compatibility)
# ============================================================================
# Get active camera config
_active_cam = get_camera_config()
CAMERA_NAME = _active_cam["name"]
CAMERA_WIDTH = _active_cam["width"]
CAMERA_HEIGHT = _active_cam["height"]
VIDEO_FILENAME = _active_cam["video_filename"]
USE_VIDEO_FILE = _active_cam["use_video_file"]
RTSP_URL = _active_cam["rtsp_url"]
WINDOW_NAME = _active_cam["window_name"]
DISPLAY_MAX_WIDTH = _active_cam["display_max_width"]
DISPLAY_MAX_HEIGHT = _active_cam["display_max_height"]
HORIZON_FILE = _active_cam["horizon_file"]
FOV_HORIZONTAL = _active_cam["fov_horizontal"]
FOV_VERTICAL = _active_cam["fov_vertical"]

# ============================================================================
# GLOBAL SETTINGS (ใช้ร่วมกันทุกกล้อง)
# ============================================================================
# FPS settings
DEFAULT_FPS = 30.0  # Default FPS fallback
MAX_FPS = 120  # Maximum FPS threshold
FPS_GOOD_THRESHOLD = 20  # FPS threshold for good performance (แสดงสีเขียว)

# ============================================================================
# OPENCV THREADING OPTIMIZATION
# ============================================================================
OPENCV_NUM_THREADS = 2  # Reduced for Jetson (was 6)
OMP_NUM_THREADS = 2     # กัน OMP/BLAS spawn เกินเหตุ

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
DEBUG_MODE = False
SHOW_GRID = True  # เปิด grid เพื่อเห็นว่า adaptive filter ทำงาน
SHOW_PERSISTENCE_PATHS = False  # ปิดการแสดงวงกลมสีแดงของ persistence paths (วัตถุที่หายไปแต่ยังทำนายตำแหน่ง)

# HUD (Heads-Up Display) settings
HUD_ENABLED = True  # เปิด/ปิดการแสดง HUD
HUD_MARGIN = 10  # ระยะห่างจากขอบ (pixels)
HUD_WIDTH_RATIO = 0.15  # สัดส่วนความกว้าง HUD (15% ของความกว้างหน้าจอ)
HUD_MIN_WIDTH = 280  # ความกว้างต่ำสุดของ HUD (pixels)
HUD_MAX_WIDTH = 400  # ความกว้างสูงสุดของ HUD (pixels)

# Bounding box display settings
BBOX_PADDING = 5  # จำนวนพิกเซลที่เพิ่มรอบ bbox เพื่อให้เห็นวัตถุชัดเจนขึ้น
BBOX_THICKNESS = 2  # ความหนาของเส้นขอบ bbox (ลดจาก 2-3 เป็น 1)

# ============================================================================
# MOTION DETECTION SETTINGS
# ============================================================================
# Grid system for noise filtering
GRID_ROWS = 16
GRID_COLS = 18
LEARNING_RATE = 0.01  # Exponential moving average learning rate (เพิ่มขึ้นเพื่อเรียนรู้เร็วขึ้น)
GRID_NOISE_FILTER_THRESHOLD = 0.2  # Skip creating trackers in cells with noise > this value (0.0 - 1.0) - กรองพื้นที่ที่มี noise สูง
MERGE_DISTANCE = 50     # Distance threshold for merging close rectangles

# Contour filtering - Min area for motion detection (pixels²)
# ค่านี้ใช้กรอง contour ที่เล็กเกินไป (ลด noise)
MIN_AREA_BASE = 1  # Base min area (pixels²) - ค่าต่ำสุด
MIN_AREA_NOISE_MULTIPLIER = 3000  # Multiplier สำหรับ noise (ยิ่ง noise สูง ยิ่งต้องมี area มาก) - เพิ่มจาก 2000 เป็น 3000 สำหรับ stricter filtering

# Background subtraction (MOG2)
MOG2_HISTORY = 300  # ลดจาก 1500 → 500 (เรียนรู้เร็วขึ้น, sensitive มากขึ้นเมื่อกล้องนิ่ง)
MOG2_VAR_THRESHOLD = 10  # ลดจาก 10 เป็น 5 (sensitive มากขึ้น, ตรวจจับ 1 pixel ได้ดีขึ้น)
MOG2_DETECT_SHADOWS = False

# Morphology kernel size
MORPH_KERNEL_SIZE = 3  # 3x3 kernel

# Motion detection area restriction
MOTION_AREA_MARGIN = 10  # จำนวนพิกเซลที่ห่างจากเส้น horizon ที่จะไม่ตรวจจับ motion (เพื่อประหยัดหน่วยประมวลผล)

# ============================================================================
# RESOLUTION-DEPENDENT PARAMETERS
# ============================================================================
# Parameters ที่ปรับตาม resolution ของกล้อง
# แบ่งเป็น 4K, 2K, 1080p, และ lower resolution

# --- 4K Resolution (3840x2160 or larger) ---
RES_4K_FRAME_SKIP_MOG2 = 1  # Process MOG2 every 3 frames (skip 2)
RES_4K_MOG2_HISTORY = 300  # ลด history สำหรับ 4K (ประหยัด GPU memory)
RES_4K_MOG2_HISTORY_4K = 300  # สำหรับ 4K width
RES_4K_PERIODIC_MEMORY_CLEANUP = 200  # Cleanup every 200 frames
RES_4K_DISPLAY_UPDATE_INTERVAL = 1  # Update every frame
RES_4K_MAX_CONTOURS_TO_PROCESS = 500  # Limit contours
RES_4K_FULL_FRAME_YOLO_INTERVAL = 1  # Full-frame scan every frame (ทุกเฟรมเพื่อตามทันวัตถุ)
RES_4K_FULL_FRAME_YOLO_CONF = 0.2  # Lower for 4K (distant drones)
RES_4K_MOG2_VAR_THRESHOLD = 14  # Sensitive มากขึ้น
RES_4K_BASE_VAR_THRESHOLD = 20
RES_4K_GRID_ROWS = 24  # Finer grid for 4K
RES_4K_GRID_COLS = 32
RES_4K_BASE_MIN_AREA_REF = 2  # Sensitive มากขึ้น
RES_4K_MAX_MIN_AREA_REF = 30
RES_4K_HYBRID_ROI_PADDING = 300  # Larger for 4K
RES_4K_HYBRID_DIST_THRESHOLD = 200
RES_4K_HYBRID_BASE_SEARCH_CONF = 0.15  # ต่ำกว่า 1080p (เพราะโดรนไกลกว่า)
RES_4K_HYBRID_MIN_MOTION_AREA = 10  # ต่ำกว่า 1080p (ตรวจจับจุดเล็กได้ดีขึ้น)

# --- 4K Height but not 4K Width (2160p but width < 3840) ---
RES_4K_HEIGHT_FRAME_SKIP_MOG2 = 1  # Process MOG2 every 2 frames (skip 1)
RES_4K_HEIGHT_MOG2_HISTORY = 500  # Reduced history to save GPU memory
RES_4K_HEIGHT_MOG2_HISTORY_4K = 500  # ใช้ค่าเดียวกับ MOG2_HISTORY

# --- 2K Resolution (2560x1440) ---
RES_2K_FRAME_SKIP_MOG2 = 1  # Process MOG2 every 2 frames (skip 1)
RES_2K_MOG2_HISTORY = 500  # Medium history
RES_2K_MOG2_HISTORY_4K = 500  # ใช้ค่าเดียวกับ MOG2_HISTORY
RES_2K_PERIODIC_MEMORY_CLEANUP = 150  # Cleanup every 150 frames
RES_2K_DISPLAY_UPDATE_INTERVAL = 1  # Update every frame
RES_2K_MAX_CONTOURS_TO_PROCESS = 800  # More contours allowed
RES_2K_FULL_FRAME_YOLO_INTERVAL = 1  # Full-frame scan every frame (ทุกเฟรมเพื่อตามทันวัตถุ)
RES_2K_FULL_FRAME_YOLO_CONF = 0.2
RES_2K_MOG2_VAR_THRESHOLD = 12
RES_2K_BASE_VAR_THRESHOLD = 20
RES_2K_GRID_ROWS = 20
RES_2K_GRID_COLS = 24
RES_2K_BASE_MIN_AREA_REF = 3
RES_2K_MAX_MIN_AREA_REF = 30
RES_2K_HYBRID_ROI_PADDING = 200
RES_2K_HYBRID_DIST_THRESHOLD = 150
RES_2K_HYBRID_BASE_SEARCH_CONF = 0.15
RES_2K_HYBRID_MIN_MOTION_AREA = 11

# --- 1080p Resolution (1920x1080) ---
RES_1080P_FRAME_SKIP_MOG2 = 1  # Process MOG2 every 2 frames (skip 1)
RES_1080P_MOG2_HISTORY = 500  # Full history
RES_1080P_MOG2_HISTORY_4K = 500  # ใช้ค่าเดียวกับ MOG2_HISTORY
RES_1080P_PERIODIC_MEMORY_CLEANUP = 200  # Cleanup every 200 frames
RES_1080P_DISPLAY_UPDATE_INTERVAL = 1  # Update every frame
RES_1080P_MAX_CONTOURS_TO_PROCESS = 1000  # More contours allowed
RES_1080P_FULL_FRAME_YOLO_INTERVAL = 1  # Full-frame scan every frame (ถี่กว่าเพื่อความเสถียร)
RES_1080P_FULL_FRAME_YOLO_CONF = 0.2
RES_1080P_MOG2_VAR_THRESHOLD = 12
RES_1080P_BASE_VAR_THRESHOLD = 20
RES_1080P_GRID_ROWS = 20
RES_1080P_GRID_COLS = 24
RES_1080P_BASE_MIN_AREA_REF = 2
RES_1080P_MAX_MIN_AREA_REF = 30
RES_1080P_HYBRID_ROI_PADDING = 120
RES_1080P_HYBRID_DIST_THRESHOLD = 100
# RES_1080P_HYBRID_BASE_SEARCH_CONF = ใช้ค่าจาก config (HYBRID_BASE_SEARCH_CONF)
# RES_1080P_HYBRID_MIN_MOTION_AREA = ใช้ค่าจาก config (HYBRID_MIN_MOTION_AREA)

# --- Lower than 1080p ---
RES_LOWER_FRAME_SKIP_MOG2 = 0  # No skipping - process every frame
RES_LOWER_MOG2_HISTORY = 1500  # Full history
RES_LOWER_MOG2_HISTORY_4K = 1500  # ใช้ค่าเดียวกับ MOG2_HISTORY
RES_LOWER_PERIODIC_MEMORY_CLEANUP = 300  # Cleanup every 300 frames
RES_LOWER_DISPLAY_UPDATE_INTERVAL = 1  # Update every frame
RES_LOWER_MAX_CONTOURS_TO_PROCESS = 1000  # More contours allowed
RES_LOWER_FULL_FRAME_YOLO_INTERVAL = 1  # Full-frame scan every frame (ถี่กว่าเพื่อความเสถียร)
RES_LOWER_FULL_FRAME_YOLO_CONF = 0.2
RES_LOWER_MOG2_VAR_THRESHOLD = 20
RES_LOWER_BASE_VAR_THRESHOLD = 35
RES_LOWER_GRID_ROWS = 16
RES_LOWER_GRID_COLS = 18
RES_LOWER_BASE_MIN_AREA_REF = 5
RES_LOWER_MAX_MIN_AREA_REF = 50
RES_LOWER_HYBRID_ROI_PADDING = 120
RES_LOWER_HYBRID_DIST_THRESHOLD = 100

# ============================================================================
# CAMERA MOVEMENT DETECTION SETTINGS
# ============================================================================
# การตรวจสอบการขยับกล้องเพื่อ reset background model (ป้องกันภาพค้าง)
CAM_MOVE_DETECTION_ENABLED = True  # เปิด/ปิดการตรวจสอบกล้องขยับ
CAM_MOVE_DETECTION_INTERVAL = 8  # ตรวจสอบทุก 8 เฟรม (ไม่ใช่ทุกเฟรม) - PERFORMANCE OPTIMIZATION (เร็วขึ้น 50%)
CAM_MOVE_THRESHOLD = 3.0  # mean absdiff threshold (pixels) - ถ้าเกินนี้ถือว่ากล้องขยับ
CAM_MOVE_RESET_BACKGROUND = True  # Reset background model เมื่อกล้องขยับ
CAM_MOVE_LEARNING_RATE = 0.1  # Learning rate สูงชั่วคราวหลัง reset (เพื่อเรียนรู้เร็วขึ้น)
CAM_MOVE_LEARNING_RATE_FRAMES = 10  # จำนวนเฟรมที่ใช้ learning rate สูงหลัง reset

# ============================================================================
# ADAPTIVE BACKGROUND FILTER SETTINGS
# ============================================================================
# Adaptive filter สำหรับจัดการพื้นหลังที่ขยับหรือกล้องสั่นไหว
ADAPTIVE_ENABLED = True  # เปิด/ปิด adaptive mode
ADAPTIVE_LEARNING_RATE = 0.1  # ความเร็วในการปรับค่า (0.1 = ปรับเร็ว, 0.05 = ปรับช้า)
BASE_VAR_THRESHOLD = 12  # ลดจาก 35 → 12 (sensitive มากขึ้นเมื่อพื้นหลังนิ่ง)
MAX_VAR_THRESHOLD = 50  # ลดจาก 90 → 50 (ยังคงกรอง noise เมื่อพื้นหลังขยับ)
BASE_MIN_AREA_REF = 1  # ค่าเริ่มต้นของ min area (at reference resolution)
MAX_MIN_AREA_REF = 30  # ลดจาก 50 → 30 (sensitive มากขึ้น)
CONTOUR_COUNT_THRESHOLD_LOW = 10  # ลดจาก 30 → 10 (ปรับให้ sensitive มากขึ้นเมื่อพื้นหลังนิ่ง)
CONTOUR_COUNT_THRESHOLD_HIGH = 200  # ลดจาก 300 → 200 (ปรับให้ sensitive มากขึ้น)
MORPH_KERNEL_SIZE_MIN = 3  # kernel size ต่ำสุดสำหรับ morphology
MORPH_KERNEL_SIZE_MAX = 7  # ลดจาก 9 → 7 (sensitive มากขึ้น)
ADAPTIVE_UPDATE_INTERVAL = 8  # อัปเดต adaptive filter ทุก 8 เฟรม (ไม่ใช่ทุกเฟรม) - PERFORMANCE OPTIMIZATION (เร็วขึ้น 50%)
MOG2_RECREATE_THRESHOLD = 5  # สร้าง MOG2 ใหม่เมื่อ threshold เปลี่ยนมากกว่า 10

# ============================================================================
# HORIZON LINE SETTINGS
# ============================================================================
SKY_GATING_ENABLED = True  # กรองตรวจจับใหม่เฉพาะโซนท้องฟ้า (เหนือเส้น)

# Exclusion zones: ไม่ให้ตรวจจับ motion ที่ขอบเส้นและขอบเฟรม
EDGE_EXCLUSION_PIXELS = 10  # จำนวนพิกเซลที่ขอบเฟรมที่จะไม่ตรวจจับ
HORIZON_EXCLUSION_PIXELS = 5  # จำนวนพิกเซลรอบเส้นขอบฟ้าที่จะไม่ตรวจจับ

# ============================================================================
# RESOLUTION-DEPENDENT THRESHOLDS (Ratios)
# ============================================================================
# Thresholds ทั้งหมดจะคำนวณจาก ratios ตาม resolution ของกล้อง
# ใช้ frame diagonal = sqrt(width^2 + height^2) สำหรับ distance thresholds
# ใช้ frame area = width * height สำหรับ area thresholds
# ใช้ frame width/height สำหรับ dimension-based thresholds

REFERENCE_RESOLUTION = (2560, 1440)  # Resolution อ้างอิง (cam1 default)

# Re-identification & Tracking (ratios of frame diagonal)
REID_DISTANCE_RATIO = 0.12  # 12% ของ frame diagonal
MERGE_DISTANCE_RATIO = 0.02  # 2% ของ frame diagonal
HYBRID_DIST_THRESHOLD_RATIO = 0.04  # 4% ของ frame diagonal
HYBRID_ROI_PADDING_RATIO = 0.05  # 5% ของ frame diagonal
SMALL_OBJECT_REID_DISTANCE_RATIO = 0.20  # 20% สำหรับ small object

# Stationary Detection & Blacklist (ratios of frame diagonal)
MAX_STATIONARY_DISTANCE_RATIO = 0.002  # 0.2% ของ frame diagonal
STATIONARY_CENTER_THRESHOLD_RATIO = 0.004  # 0.4% ของ frame diagonal
BLACKLIST_BBOX_PADDING_RATIO = 0.01  # 1% ของ frame diagonal
BLACKLIST_MIN_VELOCITY_RATIO = 0.0008  # 0.08% ของ frame diagonal per frame

# Motion Detection (ratios of frame dimensions)
EDGE_EXCLUSION_PIXELS_RATIO = 0.004  # 0.4% ของ frame width
HORIZON_EXCLUSION_PIXELS_RATIO = 0.002  # 0.2% ของ frame width
MOTION_AREA_MARGIN_RATIO = 0.004  # 0.4% ของ frame height
MIN_AREA_BASE_RATIO = 0.00015  # 0.015% ของ frame area

# Drone Detection (ratios)
DRONE_MIN_VELOCITY_FOR_ORANGE_RATIO = 0.0008  # 0.08% ของ frame diagonal per frame
DRONE_MIN_VELOCITY_FOR_HOVER_RATIO = 0.0002  # 0.02% ของ frame diagonal per frame
DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE_RATIO = 0.012  # 1.2% ของ frame diagonal
DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER_RATIO = 0.006  # 0.6% ของ frame diagonal
DRONE_MIN_AREA_RATIO = 0.000006  # 0.0006% ของ frame area
DRONE_MAX_AREA_RATIO = 0.0033  # 0.33% ของ frame area
DRONE_SMALL_OBJECT_AREA_THRESHOLD_RATIO = 0.00003  # 0.003% ของ frame area
DRONE_MIN_PATH_TOTAL_DISTANCE_RATIO = 0.012  # 1.2% ของ frame diagonal

# Display & UI (ratios)
BBOX_PADDING_RATIO = 0.002  # 0.2% ของ frame diagonal
HUD_MARGIN_RATIO = 0.004  # 0.4% ของ frame width
HUD_MIN_WIDTH_RATIO = 0.11  # 11% ของ frame width
HUD_MAX_WIDTH_RATIO = 0.16  # 16% ของ frame width

# Camera Movement (ratios)
CAM_MOVE_THRESHOLD_RATIO = 0.0012  # 0.12% ของ frame diagonal

# Additional settings for small object path detection
PREDICTED_POSITION_LOOKBACK = 5  # ใช้ประวัติ 5 จุดล่าสุดสำหรับ prediction
PREDICTED_POSITION_MIN_HISTORY = 1  # จำนวน path points ต่ำสุดที่ต้องมีก่อนใช้ predicted position (1 = ใช้ได้ทันที)
PREDICTED_POSITION_MAX_CONSECUTIVE = 5  # จำนวนเฟรมสูงสุดที่สร้าง predicted position ติดต่อกัน (ป้องกัน false positive)
SMALL_OBJECT_PATH_QUALITY_BONUS = True  # ให้ bonus สำหรับ small object ที่มี path ชัดเจน
MIN_PATH_QUALITY_FOR_SMALL_OBJECT = 0.3  # threshold ต่ำสำหรับ small object ที่มี path ชัดเจน
MAX_MISS_FRAMES_FOR_SMALL_OBJECT = 15  # อนุญาตให้หายไปนานขึ้นสำหรับ small object

# ============================================================================
# OBJECT SIZE DEFINITIONS (Ratios)
# ============================================================================
# Size thresholds สำหรับแยกแยะ tiny objects, small objects, และ medium objects
# ใช้ทั้ง area และ diagonal thresholds เพื่อความแม่นยำ

# Tiny Objects (Path-Only Mode) - วัตถุเล็กมาก (<37 pixels² สำหรับ 2560x1440)
TINY_OBJECT_AREA_THRESHOLD_RATIO = 0.00001  # 0.001% ของ frame area
TINY_OBJECT_DIAGONAL_THRESHOLD_RATIO = 0.0008  # 0.08% ของ frame diagonal
TINY_OBJECT_YOLO_INTERVAL = 15  # เรียก YOLO ทุก 60 เฟรมสำหรับ tiny objects (ลดการใช้ YOLO มาก)

# Small Objects (Path + YOLO) - วัตถุเล็ก (37-110 pixels² สำหรับ 2560x1440)
SMALL_OBJECT_DIAGONAL_THRESHOLD_RATIO = 0.0015  # 0.15% ของ frame diagonal
SMALL_OBJECT_YOLO_INTERVAL = 5  # เรียก YOLO ทุก 15 เฟรมสำหรับ small objects

# Object Size Definitions (Pixels) - สำหรับ hybrid tracker
# ค่านี้ใช้ใน hybrid_drone_tracker.py เพื่อเลือกโหมดการทำงาน
TINY_OBJECT_MAX_SIZE = 40   # วัตถุเล็กมาก (ไกล) -> เน้น Path
SMALL_OBJECT_MAX_SIZE = 200 # วัตถุเล็ก -> เริ่มใช้ YOLO

# ============================================================================
# PATH-BASED ANALYSIS SETTINGS (สำหรับ Tiny/Small Objects)
# ============================================================================
# Path analysis สำหรับ tiny/small objects ที่ใช้ path และความเร็วในการวิเคราะห์มากกว่า YOLO

SMALL_OBJECT_PATH_ONLY_MODE = True  # เปิด/ปิด path-only mode สำหรับ tiny objects
SMALL_OBJECT_MIN_PATH_FRAMES = 15  # จำนวนเฟรมต่ำสุดสำหรับ path analysis (สัมพันธ์กับ FPS)
SMALL_OBJECT_PATH_FRAMES_PER_SECOND = 0.5  # จำนวนเฟรมต่อวินาทีที่ต้องการ (0.5 = 15 frames @ 30 FPS)

# Airplane vs Drone Detection Thresholds
AIRPLANE_STRAIGHTNESS_THRESHOLD = 0.9  # threshold สำหรับเครื่องบิน (เส้นตรงมาก)
DRONE_CURVATURE_THRESHOLD = 0.3  # threshold สำหรับโดรน (มีการโค้ง/เลี้ยว)

# Velocity thresholds (ratios of frame diagonal per frame)
SMALL_OBJECT_MAX_VELOCITY_RATIO = 0.0015  # 0.15% ของ frame diagonal per frame
SMALL_OBJECT_MIN_VELOCITY_RATIO = 0.0003  # 0.03% ของ frame diagonal per frame

# Path Smoothing & Curvature Analysis
SMALL_OBJECT_PATH_SMOOTHING_WINDOW = 3  # window size สำหรับ path smoothing
SMALL_OBJECT_CURVATURE_WINDOW = 5  # window size สำหรับ curvature calculation
SMALL_OBJECT_MIN_CURVATURE_FOR_DRONE = 0.15  # curvature ต่ำสุดที่ถือว่าเป็นโดรน
SMALL_OBJECT_CONFIDENCE_THRESHOLD = 0.6  # confidence threshold สำหรับยืนยันว่าเป็นโดรน

# Tiny Objects Path Analysis Settings (สำหรับกรองเครื่องบินและดาว)
TINY_OBJECT_PATH_HISTORY_LENGTH = 90  # จำนวนเฟรมที่เก็บ path history สำหรับ tiny objects (ยาวขึ้นเพื่อวิเคราะห์ได้แม่นยำ) - DEPRECATED: ใช้ TINY_OBJECT_PATH_HISTORY_SECONDS แทน
TINY_OBJECT_AIRPLANE_STRAIGHTNESS_THRESHOLD = 0.85  # threshold สำหรับตรวจสอบเครื่องบิน (เส้นตรงมาก)
TINY_OBJECT_STAR_CLUSTERING_THRESHOLD = 0.4  # threshold สำหรับตรวจสอบดาว (กระจุกตัวมาก แทบไม่มีเส้น) - ค่าต่ำ = กระจุกตัวมาก

# Path-Based RED Detection Settings (สำหรับ Tiny Objects)
TINY_OBJECT_PATH_BASED_RED_ENABLED = True  # เปิด/ปิดฟีเจอร์ path-based RED detection
TINY_OBJECT_PATH_BASED_RED_MIN_HISTORY_SECONDS = 2.0  # เวลาต่ำสุด (วินาที) สำหรับ path analysis (คำนวณจาก processing FPS)
TINY_OBJECT_PATH_BASED_RED_MIN_CONFIDENCE = 0.75  # confidence ต่ำสุดจาก path analysis (0.75 = 75%)
TINY_OBJECT_PATH_BASED_RED_MIN_TRACKING_DURATION_SECONDS = 1.0  # ระยะเวลาติดตามต่ำสุด (วินาที) (คำนวณจาก processing FPS)

# Airplane Detection Settings (สำหรับ Tiny Objects)
TINY_OBJECT_AIRPLANE_PATH_MIN_POINTS = 20  # จำนวน path points ต่ำสุดสำหรับตรวจสอบเครื่องบิน (20 จุดเพื่อให้แน่ใจว่าเป็นเครื่องบินจริงๆ)
TINY_OBJECT_AIRPLANE_SIZE_STABILITY_THRESHOLD = 0.15  # threshold สำหรับตรวจสอบขนาด bbox ที่คงที่ (15% CV = เครื่องบินไกลมาก)
TINY_OBJECT_AIRPLANE_SIZE_HISTORY_MIN_POINTS = 15  # จำนวน size history points ต่ำสุดสำหรับตรวจสอบ (15 จุด)

# Path History Settings (ใช้ Processing FPS)
TINY_OBJECT_PATH_HISTORY_SECONDS = 3.0  # เวลาเก็บ path history (วินาที) - คำนวณจาก processing FPS

# ============================================================================
# TRACKING SETTINGS (Legacy - will be replaced by resolution-dependent)
# ============================================================================
MAX_MISS_FRAMES = 10        # อนุญาตให้หายไปได้สูงสุด 5 เฟรม
PATH_HISTORY_LENGTH = 20    # ใช้ประวัติ 20 เฟรมเพื่อยืนยันวิถีการบินต่อเนื่อง (เร็วขึ้น 50% จาก 40)
MAX_REID_DISTANCE = 300     # ระยะสูงสุดสำหรับ re-identification (legacy - use REID_DISTANCE_RATIO)

# ============================================================================
# MULTI-TARGET PRIORITY SYSTEM SETTINGS
# ============================================================================
# Priority score calculation weights
PRIORITY_YOLO_CONF_WEIGHT = 0.4      # น้ำหนักของ YOLO confidence (สูง = สำคัญมาก)
PRIORITY_PATH_QUALITY_WEIGHT = 0.3    # น้ำหนักของ path quality (smoothness, consistency)
PRIORITY_STATUS_WEIGHT = 0.2          # น้ำหนักของ status (RED > ORANGE > YELLOW > GREEN)
PRIORITY_DURATION_WEIGHT = 0.1        # น้ำหนักของ duration tracking (ติดตามนาน = น่าเชื่อถือ)

# Status scores for priority calculation
PRIORITY_STATUS_RED = 1.0              # คะแนนสำหรับ RED status
PRIORITY_STATUS_ORANGE = 0.7           # คะแนนสำหรับ ORANGE status
PRIORITY_STATUS_YELLOW = 0.4           # คะแนนสำหรับ YELLOW status
PRIORITY_STATUS_GREEN = 0.1             # คะแนนสำหรับ GREEN status

# Duration normalization (max frames for full score)
PRIORITY_DURATION_MAX_FRAMES = 60     # จำนวนเฟรมสูงสุดสำหรับ duration score (normalized to 1.0)

# Priority thresholds
MIN_PRIORITY_FOR_PRIMARY = 0.5        # คะแนนต่ำสุดสำหรับเป็น primary target
PRIORITY_SWITCH_THRESHOLD = 0.15      # ต้องต่างกันอย่างน้อยเท่านี้เพื่อสลับ primary target
MIN_CONFIDENCE_FOR_TRACKING = 0.1     # confidence ต่ำสุดสำหรับติดตาม (กรอง noise)

# Path quality calculation settings
PRIORITY_PATH_SMOOTHNESS_WEIGHT = 0.5  # น้ำหนักของ smoothness ใน path quality
PRIORITY_PATH_CONSISTENCY_WEIGHT = 0.5 # น้ำหนักของ consistency ใน path quality

# ============================================================================
# OBJECT CLASSIFICATION SETTINGS
# ============================================================================
ENABLE_OBJECT_CLASSIFICATION = True  # เปิด rule-based เบา ๆ สำหรับโดรน

# Drone classification thresholds
DRONE_MIN_AREA = 40
DRONE_MAX_AREA = 12000
DRONE_MIN_SPEED = 1.0
DRONE_MAX_SPEED = 120.0
DRONE_MIN_PATH_FRAMES = 5

# Classification score weights
DRONE_AREA_WEIGHT = 0.4
DRONE_SPEED_WEIGHT = 0.4
DRONE_PATH_WEIGHT = 0.2
DRONE_MIN_SCORE = 0.7  # Minimum score to classify as DRONE
DRONE_CONFIRMATION_MIN_SCORE = 0.75  # คะแนนต่ำสุดสำหรับยืนยันว่าเป็นโดรน (เปลี่ยนเป็น ORANGE)

# Simplified Drone Detection (based on YELLOW duration and path characteristics)
DRONE_YELLOW_DURATION_THRESHOLD = 25  # จำนวนเฟรมที่ต้องอยู่ในสถานะ YELLOW เพื่อยืนยันว่าเป็นโดรน (เพิ่มจาก 17 - ลด false alarm)
DRONE_MIN_SMOOTHNESS_SIMPLE = 0.5  # ความราบเรียบของ path ต่ำสุด (เพิ่มจาก 0.4 - เข้มงวดขึ้น กรอง noise)
DRONE_MIN_DIRECTION_CONSISTENCY_SIMPLE = 0.6  # ความสม่ำเสมอของทิศทางต่ำสุด (เพิ่มจาก 0.5 - เข้มงวดขึ้น กรอง noise)
DRONE_MIN_PATH_FRAMES_FOR_ORANGE = 30  # ต้องมี path_frames >= 30 (เพิ่มจาก 20 - ช้าลง ยืนยันได้ดีขึ้น)

# Minimum movement requirements for ORANGE status (กรองวัตถุที่แทบไม่เคลื่อนที่)
DRONE_MIN_VELOCITY_FOR_ORANGE = 2.0  # ความเร็วต่ำสุด (pixels/frame) สำหรับโดรนที่บินปกติ
DRONE_MIN_VELOCITY_FOR_HOVER = 0.5   # ความเร็วต่ำสุด (pixels/frame) สำหรับโดรนที่ hover (ต่ำกว่า)
DRONE_MIN_TOTAL_DISTANCE_FOR_ORANGE = 30.0  # ระยะทางรวมต่ำสุด (pixels) สำหรับโดรนที่บินปกติ
DRONE_MIN_TOTAL_DISTANCE_FOR_HOVER = 15.0   # ระยะทางรวมต่ำสุด (pixels) สำหรับโดรนที่ hover (ต่ำกว่า)
DRONE_MIN_MOVEMENT_FRAMES_FOR_ORANGE = 10   # จำนวนเฟรมที่ต้องมีการเคลื่อนที่จริง
DRONE_MIN_MOVEMENT_RATIO = 0.5  # สัดส่วนเฟรมที่ต้องมีการเคลื่อนที่ (50%)

# Flight trail verification (ยืนยันรอยเท้าการบิน - เก็บเฉพาะ YELLOW/ORANGE)
DRONE_MIN_YELLOW_PATH_POINTS = 20  # ต้องมี path history จาก YELLOW อย่างน้อย 20 จุด (ลดจาก 25)
DRONE_MIN_PATH_TOTAL_DISTANCE = 30.0  # ระยะทางรวมของ path ต้องมากกว่า 30 pixels (ลดจาก 40)
DRONE_HOVER_SMOOTHNESS_BONUS = 0.1  # ลด threshold สำหรับโดรนที่สามารถ hover (smoothness)
DRONE_HOVER_CONSISTENCY_BONUS = 0.1  # ลด threshold สำหรับโดรนที่สามารถ hover (consistency)
DRONE_PATH_MAX_GAP_RATIO = 3.0  # gap สูงสุดที่อนุญาต (เท่าของค่าเฉลี่ย)
DRONE_YELLOW_PATH_HISTORY_LIMIT = 60  # จำกัด path history สำหรับ YELLOW/ORANGE (ประหยัดทรัพยากร)

# Insect filtering (กรองแมลงที่บินเร็ว/เส้นตรง/ไม่ต่อเนื่อง)
# หมายเหตุ: ใช้เป็น threshold ความเร็วขั้นต่ำที่ \"น่าสงสัยว่าเป็นแมลง\"
INSECT_MAX_VELOCITY = 35.0  # ความเร็ว (pixels/frame) ที่มากกว่านี้จะพิจารณาว่าอาจเป็นแมลง (ลดจาก 50 เพื่อจับแมลงที่บินช้าขึ้น)
INSECT_MIN_STRAIGHTNESS = 0.85  # ความตรงต่ำสุดของแมลง (สูง=ตรงมาก)
INSECT_MAX_CONTINUITY = 0.6  # ความต่อเนื่องสูงสุดของแมลง (ต่ำ=ไม่ต่อเนื่อง)

# Bounding box size limits (ป้องกันกล่องขยายตัวเรื่อยๆ)
MAX_RECT_SIZE_CHANGE_RATIO = 1.5  # อนุญาตให้ขนาดเปลี่ยนได้สูงสุด 1.5 เท่า (ลดจาก 2.0 เพื่อป้องกันกล่องขยายตัวจากเมฆ)
MAX_MERGE_AREA_RATIO = 1.5  # อนุญาตให้ merge ถ้า area ไม่เกิน 1.5 เท่าของค่าเฉลี่ย (ลดจาก 2.0 เพื่อป้องกัน merge กล่องใหญ่เกินไป)
MAX_ASPECT_RATIO = 3.0  # อัตราส่วนกว้าง/สูงสูงสุด (ป้องกันกล่องแบนผิดปกติ)
MIN_ASPECT_RATIO = 0.33  # อัตราส่วนกว้าง/สูงต่ำสุด (1/3)

# Cloud detection thresholds (กรองก้อนเมฆ - ไม่กระทบความเร็ว)
CLOUD_DETECTION_ENABLED = True
CLOUD_MIN_VELOCITY = 40.0   # ความเร็วต่ำสุดของก้อนเมฆ (เร็ว - ลมพัด)
CLOUD_MAX_VELOCITY = 200.0  # ความเร็วสูงสุดของก้อนเมฆ (เร็วมาก)
CLOUD_MIN_AREA = 1000       # ขนาดต่ำสุดของก้อนเมฆ (ใหญ่ - pixels²)
CLOUD_MAX_STRAIGHTNESS = 0.4    # ความตรงของ path สูงสุด (ไม่ตรง - แปรผันตามลม)
CLOUD_MAX_VELOCITY_CV = 0.5     # CV สูงสุด (แปรผัน - ลมไม่สม่ำเสมอ)
CLOUD_MAX_SMOOTHNESS = 0.5      # ความราบเรียบของ path สูงสุด (ไม่ราบเรียบมาก)

# ============================================================================
# PATH QUALITY ANALYSIS SETTINGS
# ============================================================================
# Path quality thresholds for drone detection
DRONE_MIN_STRAIGHTNESS = 0.55  # ความตรงของ path ต่ำสุด (0-1, สูง=ตรง)
DRONE_MAX_STRAIGHTNESS = 0.85  # ความตรงของ path สูงสุด (โดรนไม่ตรงมากเท่าเครื่องบิน)
DRONE_MIN_SMOOTHNESS = 0.5     # ความราบเรียบของ path ต่ำสุด (0-1, สูง=ราบเรียบ)
DRONE_MIN_VELOCITY_CV = 0.15   # CV (coefficient of variation) ต่ำสุดของความเร็ว (สม่ำเสมอ)
DRONE_MAX_VELOCITY_CV = 0.35   # CV สูงสุดของความเร็ว (สม่ำเสมอ)
DRONE_MIN_DIRECTION_CONSISTENCY = 0.6  # ความสม่ำเสมอของทิศทางต่ำสุด (0-1, สูง=สม่ำเสมอ)

# Hover detection settings
HOVER_VELOCITY_THRESHOLD = 2.0  # velocity threshold สำหรับ hover detection (pixels/frame)
HOVER_MIN_FRAMES = 3            # จำนวนเฟรมต่ำสุดที่ต้อง hover
HOVER_RATIO_THRESHOLD = 0.3     # hover ratio threshold (30% ของเฟรมต้องมี velocity ต่ำ)

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================
CLASSIFICATION_UPDATE_INTERVAL = 15  # คำนวณ classification ใหม่ทุก 15 เฟรม (ไม่ใช่ทุกเฟรม) (เร็วขึ้น 50%)
PATH_QUALITY_HISTORY_LIMIT = 20      # จำกัด history length สำหรับ path quality (20 จุด)
VELOCITY_HISTORY_LIMIT = 15         # จำกัด history length สำหรับ velocity (15 จุด)
HOVER_HISTORY_LIMIT = 10            # จำกัด history length สำหรับ hover detection (10 เฟรม)
CHECK_STATUS_UPDATE_INTERVAL = 3    # เรียก check_status() ทุก 3 เฟรม (ไม่ใช่ทุกเฟรม) - ประหยัด CPU

# ============================================================================
# DISPLAY COLORS (BGR format)
# ============================================================================
ALERT_COLOR = (0, 255, 255)  # สีเหลือง (BGR) สำหรับวัตถุเฝ้าระวัง
YELLOW_COLOR = ALERT_COLOR   # alias สำหรับสีเหลือง (ใช้ร่วมกับโค้ดเก่า)
NORMAL_COLOR = (0, 255, 0)   # สีเขียว (BGR) สำหรับวัตถุทั่วไป
ORANGE_COLOR = (0, 165, 255)  # สีส้ม (BGR) สำหรับโดรนที่ยืนยันแล้ว
RED_COLOR = (0, 0, 255)  # สีแดง (BGR) สำหรับโดรนที่ยืนยันด้วย YOLO confidence สูง
HORIZON_COLOR = (255, 255, 0)  # สีเหลืองสำหรับเส้นขอบฟ้า
DRAWING_COLOR = (0, 255, 255)  # สีฟ้าสำหรับวาดเส้น
COLOR_TEXT = (0, 255, 255)  # สีสำหรับข้อความทั่วไป (cyan)
HUD_BACKGROUND_COLOR = (0, 0, 0)  # สีพื้นหลัง HUD (ดำ)
FPS_GOOD_COLOR = (0, 255, 0)  # สีเขียวเมื่อ FPS ดี
FPS_BAD_COLOR = (0, 0, 255)  # สีแดงเมื่อ FPS ต่ำ
STATS_TEXT_COLOR = (255, 255, 255)  # สีขาวสำหรับข้อความสถิติ
ALGO_TIME_COLOR = (0, 255, 255)  # สี cyan สำหรับ algo time

# ============================================================================
# SIZE CHANGE DETECTION
# ============================================================================
SIZE_CHANGE_MAX_MULTIPLIER = 3.0  # ตรวจสอบการเปลี่ยนแปลงขนาด (3 เท่า)

# ============================================================================
# PATH AND HEAT POINT VISUALIZATION
# ============================================================================
PATH_VISUALIZATION_ENABLED = True  # เปิด/ปิดการแสดง path
PATH_MAX_POINTS = 20  # จำนวนจุดสูงสุดที่แสดงใน path
HEAT_POINT_ENABLED = True  # เปิด/ปิด heat point
HEAT_POINT_RADIUS = 5  # ขนาดจุด heat point (pixels)

# Drawing control settings (ควบคุมการวาดกล่องแต่ละสีและ path)
DRAW_GREEN_BOX = False   # วาดกล่องสีเขียว (GREEN status)
DRAW_YELLOW_BOX = False  # วาดกล่องสีเหลือง (YELLOW status)
DRAW_ORANGE_BOX = True  # วาดกล่องสีส้ม (ORANGE status)
DRAW_RED_BOX = True    # วาดกล่องสีแดง (RED status - YOLO confidence สูง)
DRAW_PATH = True        # วาด path การบิน (เส้น path และ heat point)
DRAW_HEAT_POINT = True  # วาดจุด heat point (ใช้ร่วมกับ DRAW_PATH และ HEAT_POINT_ENABLED)
DRAW_LABELS = False      # วาดข้อความ label/ID
# Path per status (เปิด/ปิด path ตามสี)
DRAW_GREEN_PATH = True
DRAW_YELLOW_PATH = False
DRAW_ORANGE_PATH = True
DRAW_RED_PATH = True    # วาด path สีแดง (RED status)

# Motion boxes display (สำหรับ hybrid mode)
DRAW_MOTION_BOXES = True  # วาดกล่องเขียว (motion detection boxes) ใน hybrid mode

# ============================================================================
# SOUND ALERT SETTINGS
# ============================================================================
SOUND_ALERT_ENABLED = True  # เปิด/ปิดเสียงเตือน
SOUND_FILE = "beep_2x.wav"  # ไฟล์เสียงเตือน
SOUND_CHECK_INTERVAL = 0.5  # ตรวจสอบทุก 0.5 วินาที (ไม่ใช่ทุกเฟรม)
SOUND_RED_FRAME_THRESHOLD = 4  # จำนวนเฟรมที่ target ต้องเป็น RED ต่อเนื่องกันก่อนส่งเสียง (3-5 เฟรม)

# ============================================================================
# HYBRID TRACKER SETTINGS (สำหรับ hybrid_tracker.py)
# ============================================================================
# เปิด/ปิดการใช้งาน Hybrid Tracker (ทำงานใน thread แยก)
HYBRID_TRACKER_ENABLED = False  # ตั้งเป็น True เพื่อเปิดใช้งาน

# การตั้งค่าสำหรับการติดตามแบบ Hybrid (YOLO + Motion)
HYBRID_BASE_CONF = 0.5       # มั่นใจปกติ (confidence threshold สำหรับ YOLO เมื่อยังไม่ lock)
HYBRID_BASE_SEARCH_CONF = 0.06  # ลดจาก 0.1 → 0.06 (ตรวจจับจุดเล็กได้เร็วขึ้น)
HYBRID_SEARCH_CONF = 0.06       # ลดจาก 0.1 → 0.06 (ให้สอดคล้องกัน)

# Multi-stage search thresholds
HYBRID_SEARCH_CONF_MEDIUM = 0.03  # Stage 2: threshold กลาง (50% ของ base)
HYBRID_SEARCH_CONF_LOW = 0.01     # Stage 3: threshold ต่ำสุด (with motion)

# Search interval settings
HYBRID_SEARCH_INTERVAL_NO_TARGET = 2  # ค้นหาทุก 2 เฟรมเมื่อไม่มี target
HYBRID_SEARCH_INTERVAL_WITH_TARGET = 4  # ค้นหาทุก 4 เฟรมเมื่อมี target

# Partial motion boost (สำหรับ YOLO box ที่อยู่ใกล้ motion แต่ไม่ overlap เต็ม)
HYBRID_MOTION_PARTIAL_DIST = 50  # ระยะห่างสูงสุด (pixels) สำหรับ partial boost
HYBRID_MOTION_PARTIAL_BOOST = 2.0  # ตัวคูณสำหรับ partial boost (ต่ำกว่า full boost)

# Minimum confidence for tracking (search mode only)
MIN_CONFIDENCE_FOR_TRACKING_SEARCH = 0.05  # ลดจาก 0.1 → 0.05 สำหรับ search mode

# Motion-based confidence boost settings (สำหรับ search mode)
HYBRID_MOTION_IOU_THRESHOLD = 0.4  # IOU threshold สำหรับตรวจสอบ overlap กับ motion box (30%)
HYBRID_MOTION_CONF_BOOST = 30.0    # ตัวคูณสำหรับ boost confidence เมื่อ IOU > threshold (50 เท่า)
HYBRID_MOTION_CONF_MAX = 1.0        # ค่า confidence สูงสุดหลัง boost (clamp ไม่ให้เกินนี้)
HYBRID_MAX_MISS_ALLOWED = 45 # ยอมให้ใช้ Motion ตามได้นานขึ้น (จำนวนเฟรมที่หายไปก่อน unlock)
HYBRID_ROI_PADDING = 120     # เพิ่มพื้นที่ค้นหาให้กว้างขึ้น (pixels)
HYBRID_DIST_THRESHOLD = 100  # ระยะพิกเซลที่ยอมให้เป้าหมายขยับได้ใน 1 เฟรม (pixels)
HYBRID_MIN_MOTION_AREA = 1   # ลดจาก 5 → 1 (ตรวจจับจุดเล็กๆ ได้ดีขึ้น, sensitive มากขึ้น)
HYBRID_ROI_WAIT_FRAMES = 8   # จำนวนเฟรมที่ ROI จะรอวัตถุที่หายไปชั่วครู่ (ใช้คาดการณ์ตำแหน่ง)
HYBRID_YOLO_INTERVAL = 1     # จำนวนเฟรมที่ใช้ YOLO ตรวจจับใน ROI (ทุก N เฟรม เพื่อประหยัด performance)
MAX_ROI_TRACKING = 10         # จำนวน ROI สูงสุดที่ติดตามพร้อมกัน (เรียงตาม confidence สูงสุดก่อน) - เพื่อป้องกันการกระตุก

# Size similarity thresholds for motion box selection (ratios)
MOTION_SIZE_MIN_RATIO = 0.3  # motion box ต้องมีขนาดอย่างน้อย 30% ของ target
MOTION_SIZE_MAX_RATIO = 3.0  # motion box ต้องมีขนาดไม่เกิน 3 เท่าของ target

# Scoring weights for motion box selection
MOTION_SCORE_PREDICTED_WEIGHT = 0.4  # น้ำหนักสำหรับ predicted position
MOTION_SCORE_PREVIOUS_WEIGHT = 0.25  # น้ำหนักสำหรับ previous position
MOTION_SCORE_DIRECTION_WEIGHT = 0.2  # น้ำหนักสำหรับ direction alignment
MOTION_SCORE_SIZE_WEIGHT = 0.15  # น้ำหนักสำหรับ size similarity

# Dynamic ROI padding multipliers
ROI_PADDING_FAST_MULTIPLIER = 1.8  # คูณ padding เมื่อวัตถุเคลื่อนที่เร็ว
ROI_PADDING_FAST_VELOCITY_THRESHOLD_RATIO = 0.001  # 0.1% ของ frame diagonal per frame

# Extended ROI search (เมื่อไม่เจอ motion box ใน ROI ปกติ)
ROI_EXTENDED_SEARCH_ENABLED = True  # เปิด/ปิด extended ROI search
ROI_EXTENDED_SEARCH_MULTIPLIER = 2.0  # คูณ ROI padding เมื่อทำ extended search
ROI_EXTENDED_SEARCH_MIN_MISSED = 3  # จำนวนเฟรมที่หายไปขั้นต่ำก่อนทำ extended search

# ROI overlap priority
ROI_OVERLAP_IOU_THRESHOLD = 0.3  # IOU threshold สำหรับตรวจสอบว่า motion box อยู่ในหลาย ROI

# Size change handling
SIZE_CHANGE_SMOOTHING_ALPHA = 0.7  # Exponential smoothing alpha สำหรับ size history
SIZE_CHANGE_MAX_RATIO = 2.0  # ขนาดเปลี่ยนได้สูงสุด 2 เท่า (สำหรับ size similarity check)

# Velocity smoothing
VELOCITY_SMOOTHING_ALPHA = 0.7  # Exponential smoothing alpha สำหรับ velocity (0.0-1.0, สูง = smooth มาก)

# Fallback strategies
FALLBACK_REDUCE_MIN_AREA_RATIO = 0.5  # ลด min_area ลง 50% เมื่อไม่เจอ motion box
FALLBACK_REDUCE_SIZE_RATIO = 0.7  # ลด size ratio threshold ลง 30% เมื่อไม่เจอ motion box
FALLBACK_MAX_ATTEMPTS = 3  # จำนวนครั้งสูงสุดที่ทำ fallback

# Performance metrics
PERFORMANCE_METRICS_ENABLED = True  # เปิด/ปิด performance metrics
PERFORMANCE_METRICS_INTERVAL = 60  # เก็บ metrics ทุก N เฟรม

# Debug visualization
DEBUG_SHOW_PREDICTED_POSITION = True  # แสดง predicted position
DEBUG_SHOW_VELOCITY_VECTOR = True  # แสดง velocity vector
DEBUG_SHOW_EXTENDED_ROI = True  # แสดง extended ROI
DEBUG_SHOW_SCORE = False  # แสดง score ของ motion box (อาจทำให้ช้า)

# ============================================================================
# PATH VALIDATION SETTINGS (ป้องกัน noise จากพื้นหลัง)
# ============================================================================
PATH_VALIDATION_ENABLED = True  # เปิด/ปิด path validation
MIN_MOTION_SCORE_THRESHOLD = 0.5  # คะแนน validation ต่ำสุดที่ยอมรับได้ (0.0-1.0)
OUTLIER_DISTANCE_THRESHOLD_RATIO = 0.05  # ระยะห่างสูงสุดจาก predicted position (5% ของ frame diagonal)
OUTLIER_DIRECTION_THRESHOLD = 90.0  # มุมสูงสุดที่ยอมรับได้ (degrees)
OUTLIER_SIZE_THRESHOLD_RATIO = 0.3  # ขนาดเปลี่ยนได้สูงสุด (30% ของขนาดเดิม)

# ============================================================================
# TEMPORAL CONTINUITY CHECK SETTINGS (ป้องกันการเชื่อมกระโดดเป็นกบเต้น)
# ============================================================================
TEMPORAL_CONTINUITY_ENABLED = True  # เปิด/ปิด temporal continuity check
TEMPORAL_CONTINUITY_MIN_PATH_POINTS = 3  # จำนวน path points ต่ำสุดที่ต้องมีก่อนตรวจสอบ temporal continuity
TEMPORAL_CONTINUITY_DISTANCE_THRESHOLD_RATIO = 0.03  # ระยะห่างสูงสุดจาก path history (3% ของ frame diagonal)
VELOCITY_CHANGE_THRESHOLD_RATIO = 0.5  # Velocity magnitude เปลี่ยนได้สูงสุด (50% ของ velocity ก่อนหน้า)
VELOCITY_DIRECTION_CHANGE_THRESHOLD = 60.0  # มุมสูงสุดที่ velocity direction เปลี่ยนได้ (degrees)

# ============================================================================
# PATH SMOOTHING SETTINGS (ใช้ทั้งการวาดและการคำนวณ)
# ============================================================================
PATH_SMOOTHING_ENABLED = True  # เปิด/ปิด path smoothing
PATH_SMOOTHING_WINDOW_SIZE = 3  # ขนาด window สำหรับ moving average (3 = smooth ปานกลาง, 5 = smooth มาก)

# ============================================================================
# ROI PRIORITY SYSTEM SETTINGS
# ============================================================================
MAX_ROI_DRAW_LIMIT = 10  # จำนวน ROI สูงสุดที่วาดได้ (10 ROI)
ROI_PRIORITY_CONTINUITY_WEIGHT = 1.0  # น้ำหนักสำหรับ path continuity score
ROI_PRIORITY_PATH_LENGTH_WEIGHT = 0.5  # น้ำหนักสำหรับ path length score
ROI_PRIORITY_TEMPORARY_LOSS_BONUS = 0.8  # Bonus สำหรับ temporary loss (motion หายไปแวบนึง)
ROI_PRIORITY_TRACKING_DURATION_WEIGHT = 0.3  # น้ำหนักสำหรับ tracking duration score
ROI_PATH_HISTORY_MAX_POINTS = 50  # จำนวน path points สูงสุดที่วาดใน ROI (path ยาวๆ เพื่อแยก target)

# ============================================================================
# ROI DENSE MOTION FILTER SETTINGS (กรอง ROI สีขาวที่ไม่สร้างบนพื้นที่ที่มี motion ยุ่งเหยิง)
# ============================================================================
ROI_DENSE_MOTION_FILTER_ENABLED = True  # เปิด/ปิดการกรอง ROI ที่มี motion ยุ่งเหยิง
ROI_DENSE_MOTION_MAX_BOXES = 8  # จำนวน motion boxes สูงสุดใน extended area ที่ยอมรับได้ (ถ้าเกินนี้ = motion ยุ่งเหยิง)
ROI_DENSE_MOTION_COVERAGE_THRESHOLD = 0.4  # Coverage ratio สูงสุดของ motion boxes ใน ROI area ที่ยอมรับได้ (0.0-1.0, สูง=motion เต็มไปหมด)
ROI_DENSE_MOTION_EXTENDED_AREA_MULTIPLIER = 1.5  # Multiplier สำหรับ extended area (1.5 = ขยาย 1.5 เท่าของ ROI size)

# ============================================================================
# ROI VALIDATION SETTINGS (ตรวจสอบ IOU และการเคลื่อนที่ก่อนวาด ROI)
# ============================================================================
ROI_VALIDATION_ENABLED = True  # เปิด/ปิด ROI validation
ROI_MIN_IOU_THRESHOLD = 0.3  # IOU ต่ำสุดที่ยอมรับได้ (30% overlap) - resolution-dependent
ROI_MIN_MOVEMENT_THRESHOLD = 0.02  # การเคลื่อนที่ต่ำสุด (2% ของ frame diagonal)
ROI_MAX_DIRECTION_CHANGE = 60.0  # มุมสูงสุดที่ทิศทางเปลี่ยนได้ (degrees)
ROI_HISTORY_MAX_FRAMES = 10  # จำนวนเฟรมสูงสุดที่เก็บ ROI history (เพื่อ performance)
ROI_VALIDATION_MIN_HISTORY = 2  # จำนวน ROI history ต่ำสุดที่ต้องมีก่อนตรวจสอบ (2 = ต้องมีอย่างน้อย 2 ROI)
ROI_VALIDATION_SKIP_RED_ORANGE = False  # ข้าม validation สำหรับ RED/ORANGE status (True = skip, False = validate)
ROI_VALIDATION_SKIP_FIRST_FRAMES = 3  # ข้าม validation ใน N เฟรมแรกสำหรับ motion-only targets (ให้วาดได้เร็วขึ้น)
ROI_VALIDATION_MOTION_ONLY_MIN_PATH = 1  # จำนวน path points ต่ำสุดสำหรับ motion-only targets (1 = วาดได้ทันทีเมื่อมี path)

# ============================================================================
# ROI VALIDATION FOR TINY OBJECTS (1-pixel targets)
# ============================================================================
ROI_TINY_IOU_THRESHOLD_MULTIPLIER = 0.5  # ลด IOU threshold ลง 50% สำหรับ tiny objects (0.15 แทน 0.3)
ROI_TINY_MOVEMENT_THRESHOLD_MULTIPLIER = 0.1  # ลด movement threshold ลง 90% สำหรับ tiny objects (0.002 แทน 0.02)
ROI_TINY_DIRECTION_CHANGE_MULTIPLIER = 1.5  # เพิ่ม direction change threshold ขึ้น 50% สำหรับ tiny objects (90° แทน 60°)
ROI_TINY_PATH_QUALITY_THRESHOLD = 0.5  # path quality threshold สำหรับ tiny objects (ต่ำกว่า)
ROI_TINY_USE_PATH_FIRST = True  # ใช้ path-first validation สำหรับ tiny objects

# ============================================================================
# ACCELERATION CHECK SETTINGS (ป้องกันการดีดตัวเร็วเกินไป)
# ============================================================================
ACCELERATION_CHECK_ENABLED = True  # เปิด/ปิด acceleration check
MAX_ACCELERATION_RATIO = 0.3  # Acceleration สูงสุดที่ยอมรับได้ (30% ของ velocity ก่อนหน้า per frame)
MAX_DECELERATION_RATIO = 0.5  # Deceleration สูงสุดที่ยอมรับได้ (50% ของ velocity ก่อนหน้า per frame)
ACCELERATION_CHECK_MIN_PATH_POINTS = 3  # จำนวน path points ต่ำสุดที่ต้องมีก่อนตรวจสอบ acceleration

# ============================================================================
# MOTION BOX FRESHNESS CHECK SETTINGS (ป้องกัน motion box เก่าค้างอยู่)
# ============================================================================
MOTION_BOX_FRESHNESS_CHECK_ENABLED = True  # เปิด/ปิด motion box freshness check
MOTION_BOX_MAX_AGE_FRAMES = 2  # อายุสูงสุดของ motion box (เฟรม) - ถ้าเก่ากว่านี้ = noise
MOTION_BOX_FRESHNESS_ROI_CHECK = True  # ตรวจสอบว่า motion box อยู่ใน ROI หรือไม่

# ============================================================================
# PATH SMOOTHNESS CHECK SETTINGS (ป้องกัน path กระโดดมากเกินไป)
# ============================================================================
PATH_SMOOTHNESS_CHECK_ENABLED = True  # เปิด/ปิด path smoothness check
MAX_PATH_JUMP_RATIO = 0.05  # ระยะกระโดดสูงสุด (5% ของ frame diagonal) - สำหรับ tiny objects ใช้ 0.08
MAX_PATH_JUMP_RATIO_TINY = 0.08  # ระยะกระโดดสูงสุดสำหรับ tiny objects (8% ของ frame diagonal)

# ============================================================================
# ADAPTIVE HYBRID_MIN_MOTION_AREA SETTINGS (Jetson Orin Nano Optimized)
# ============================================================================
# ระบบปรับ HYBRID_MIN_MOTION_AREA แบบอัตโนมัติตามประสิทธิภาพ
ADAPTIVE_HYBRID_MIN_AREA_ENABLED = True  # เปิด/ปิด adaptive HYBRID_MIN_MOTION_AREA
ADAPTIVE_HYBRID_MIN_AREA_BASE = 1  # ค่าเริ่มต้น (ได้ผลดีกับ Jetson เมื่อกล้องนิ่ง)
ADAPTIVE_HYBRID_MIN_AREA_MAX = 10  # ค่าสูงสุด (pixels)
ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_LOW = 50  # จำนวน motion boxes ต่ำ (ลด min_area)
ADAPTIVE_HYBRID_MIN_AREA_MOTION_THRESHOLD_HIGH = 400  # จำนวน motion boxes สูง (เพิ่ม min_area - conservative)
ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_LOW = 10.0  # FPS ต่ำ (เพิ่ม min_area - อนุญาตให้ต่ำได้)
ADAPTIVE_HYBRID_MIN_AREA_FPS_THRESHOLD_HIGH = 25.0  # FPS สูง (ลด min_area)
ADAPTIVE_HYBRID_MIN_AREA_TIME_THRESHOLD_HIGH = 150.0  # เวลา processing สูง (milliseconds) - เพิ่ม min_area (ตามที่สังเกต - Jetson ทนได้ถึง 150ms)
ADAPTIVE_HYBRID_MIN_AREA_UPDATE_INTERVAL = 10  # อัปเดตทุก N เฟรม
ADAPTIVE_HYBRID_MIN_AREA_ADJUSTMENT_STEP = 1  # เพิ่ม/ลดทีละกี่ pixel

# ============================================================================
# SIZE-BASED MIN MOTION AREA SETTINGS (สำหรับแยก min_area ตาม object size)
# ============================================================================
# ระบบแยก min_area ตาม object size เพื่อให้ตรวจจับโดรนเล็กมากๆ (1 pixel) ได้แม้ motion เยอะ
TINY_OBJECT_MIN_MOTION_AREA_BASE = 1  # min_area base สำหรับ tiny objects (ไม่เพิ่มเกินนี้)
TINY_OBJECT_MIN_MOTION_AREA_MAX = 1  # min_area สูงสุดสำหรับ tiny objects (ลดจาก 2 เป็น 1 - ตรวจจับ 1 pixel ได้เสมอ)
SMALL_OBJECT_MIN_MOTION_AREA_BASE = 2  # min_area base สำหรับ small objects
SMALL_OBJECT_MIN_MOTION_AREA_MAX = 5  # min_area สูงสุดสำหรับ small objects (แม้ motion เยอะ)
MEDIUM_OBJECT_MIN_MOTION_AREA_BASE = 5  # min_area base สำหรับ medium objects
MEDIUM_OBJECT_MIN_MOTION_AREA_MAX = 10  # min_area สูงสุดสำหรับ medium objects
ADAPTIVE_TINY_OBJECT_MIN_AREA_ENABLED = False  # เปิด/ปิด adaptive สำหรับ tiny objects (ควรปิดเพื่อให้ตรวจจับได้เสมอ)

# Stationary object detection (กรองแสงไฟ/วัตถุนิ่ง)
STATIONARY_DETECTION_ENABLED = True  # เปิด/ปิดการตรวจสอบวัตถุนิ่ง
MAX_STATIONARY_DISTANCE = 5.0  # ระยะทางสูงสุดที่ยอมรับได้ว่าวัตถุนิ่ง (pixels) ในช่วงเวลาที่กำหนด
STATIONARY_CHECK_FRAMES = 60  # จำนวนเฟรมที่ต้องตรวจสอบ (30 เฟรม ≈ 1 วินาทีที่ 30 FPS)
BLACKLIST_DURATION_SECONDS = 60  # ระยะเวลา blacklist (วินาที) - อย่างน้อย 1 นาที
BLACKLIST_BBOX_PADDING = 25  # เพิ่ม padding รอบ bbox เมื่อเพิ่ม blacklist (pixels) - เพื่อ blacklist พื้นที่ในเฟรมรอบๆ
# การตรวจสอบกรณีกล่องยืด/หดอยู่ที่เดิม
STATIONARY_CENTER_THRESHOLD = 10.0  # ระยะทางสูงสุดของ center point ที่ยอมรับได้ว่ายังอยู่ที่เดิม (pixels) - สำหรับกรณียืด/หด

# Early blacklist detection (ตรวจสอบเร็วขึ้นเมื่อเริ่ม ROI)
EARLY_BLACKLIST_CHECK_FRAMES = 20  # ตรวจสอบ stationary หลังจาก 20 เฟรม (เร็วขึ้น)
EARLY_BLACKLIST_STATIONARY_THRESHOLD_RATIO = 0.002  # 0.2% ของ frame diagonal

# RED stationary check (ตรวจสอบการเคลื่อนที่ของ RED bbox)
RED_STATIONARY_CHECK_INTERVAL = 10  # ตรวจสอบทุก 10 เฟรม
RED_STATIONARY_MAX_FRAMES = 60  # ถ้านิ่งเกิน 60 เฟรม (2 วินาที @ 30 FPS) → ปล่อย
RED_STATIONARY_VELOCITY_THRESHOLD_RATIO = 0.0005  # 0.05% ของ frame diagonal per frame
RED_STATIONARY_DISTANCE_THRESHOLD_RATIO = 0.005  # 0.5% ของ frame diagonal

# Permanent blacklist สำหรับพื้นที่ที่ถูก blacklist ซ้ำๆ (ดาว/ไฟ/background)
PERMANENT_BLACKLIST_ENABLED = True  # เปิด/ปิด permanent blacklist
PERMANENT_BLACKLIST_THRESHOLD = 3  # จำนวนครั้งที่ต้องถูก blacklist ก่อนย้ายไป permanent (เช่น 3 ครั้ง)
PERMANENT_BLACKLIST_WINDOW_SECONDS = 300  # หน้าต่างเวลาสำหรับนับจำนวนครั้ง (วินาที) - เช่น 5 นาที
PERMANENT_BLACKLIST_DURATION_SECONDS = 3600  # ระยะเวลา permanent blacklist (วินาที) - เช่น 1 ชั่วโมง หรือ 0 = ถาวร
BLACKLIST_OVERLAP_THRESHOLD = 0.5  # IOU threshold สำหรับตรวจสอบว่า bbox ซ้อนทับกัน (0.0-1.0) - 0.5 = 50% overlap

# การอนุญาตให้ตรวจจับวัตถุที่เคลื่อนที่ผ่าน blacklist
BLACKLIST_ALLOW_MOVEMENT = True  # อนุญาตให้ตรวจจับวัตถุที่เคลื่อนที่ผ่าน blacklist
BLACKLIST_MIN_VELOCITY = 2.0  # ความเร็วต่ำสุด (pixels/frame) ที่ถือว่าวัตถุเคลื่อนที่ (สำหรับอนุญาตให้ตรวจจับ)
BLACKLIST_MOTION_IOU_THRESHOLD = 0.3  # IOU threshold สำหรับตรวจสอบ motion overlap (ถ้ามี motion overlap = มีการเคลื่อนที่)

# ============================================================================
# RED LOCK (ACCUMULATED RED FRAMES) SETTINGS
# ============================================================================
# ใช้เฟรม RED แบบสะสม (ไม่จำเป็นต้องต่อเนื่อง) เพื่อยกระดับเป็น RED lock mode
RED_ACCUM_WINDOW_FRAMES = 90   # จำนวนเฟรมที่ใช้เป็นหน้าต่างสะสมคะแนน RED (ขยายเพื่อให้สะสมได้นานขึ้น)
RED_LOCK_SCORE          = 6.0  # คะแนนขั้นต่ำสำหรับเข้า RED lock (ลดลงเพื่อให้เข้า lock ได้ง่ายขึ้น)
RED_DECAY_FACTOR        = 0.95 # ปัจจัยลดคะแนนต่อเฟรม (เพิ่มขึ้นเพื่อให้คะแนนไม่ตกเร็วเกินไป)

# พารามิเตอร์สำหรับ lock mode
LOCK_YOLO_INTERVAL        = 1      # ใช้ YOLO ใน ROI ทุกเฟรมเมื่ออยู่ใน RED lock
LOCK_ROI_WAIT_MULTIPLIER  = 2.0    # คูณ HYBRID_ROI_WAIT_FRAMES เมื่อเป็น RED lock (รอวัตถุได้นานขึ้น)
LOCK_DIST_MULTIPLIER      = 2.0    # lock target ยอมให้ YOLO center ขยับได้ 2 เท่าของ HYBRID_DIST_THRESHOLD

# Motion stationarity detection for RED lock optimization
MOTION_STATIONARY_CHECK_INTERVAL = 3  # ตรวจสอบทุก 3 เฟรม (เพื่อประหยัด CPU)
MOTION_STATIONARY_MOVEMENT_THRESHOLD_RATIO = 0.01  # 1% ของ frame diagonal (threshold สำหรับการขยับ)
MOTION_STATIONARY_FRAMES_TO_EXIT = 3  # จำนวนเฟรมที่ stationary ก่อนออกจาก lock

# Model file path (relative to script directory)
HYBRID_MODEL_FILE = "yolo_11n_day_night_200_2_imgsz640.engine"  # หรือ "yolo11n_400epoch_uav_day.engine" ถ้าใช้ TensorRT

# ============================================================================
# PTZ CAMERA 2 VERIFICATION SETTINGS
# ============================================================================
# ตั้งค่าสำหรับการตรวจสอบ ROI ด้วยกล้อง 2 (PTZ)
CAM2_PTZ_ENABLED = False  # เปิด/ปิดการใช้งาน PTZ verification
CAM2_PTZ_IP = '192.168.1.203'  # IP ของกล้อง 2 (PTZ)
CAM2_PTZ_USER = 'admin'
CAM2_PTZ_PASS = 'Passw0rd'
CAM2_PTZ_CHANNEL = 1
CAM2_PTZ_PAN_OFFSET_DEG = 0.0  # Offset ของ Pan (องศา) - ปรับตามการติดตั้งจริง
CAM2_PTZ_TILT_OFFSET_DEG = 0.0  # Offset ของ Tilt (องศา) - ปรับตามการติดตั้งจริง

# ระดับ zoom ที่จะลอง (จากน้อยไปมาก)
CAM2_PTZ_ZOOM_LEVELS = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]  # ระดับ zoom
CAM2_PTZ_VERIFICATION_FRAMES = 30  # จำนวนเฟรมที่ตรวจสอบต่อ zoom level
CAM2_PTZ_STABILIZATION_FRAMES = 10  # จำนวนเฟรมที่รอให้กล้องเสถียรก่อนตรวจสอบ
CAM2_PTZ_VERIFICATION_CONF_THRESHOLD = 0.3  # Confidence threshold สำหรับ YOLO ในกล้อง 2 (สำหรับ tracking)
CAM2_PTZ_SEARCH_CONF = 0.1  # Confidence threshold สำหรับ search mode (หา ROI ก่อน)
CAM2_PTZ_BLACKLIST_DURATION = 30.0  # ระยะเวลา blacklist (วินาที)
CAM2_PTZ_BLACKLIST_RADIUS = 100  # รัศมี blacklist (pixels)

# Preset position สำหรับกล้อง 2 (ตำแหน่งที่กล้องจะกลับไปเมื่อเริ่ม/ปิดโปรแกรม)
CAM2_PTZ_PRESET_PAN_UNITS = 0  # Pan units (0-3599, 0.1° per unit)
CAM2_PTZ_PRESET_TILT_UNITS = 0  # Tilt units (-900 to 900, -90° to +90°, 0.1° per unit)
CAM2_PTZ_PRESET_ZOOM_UNITS = 1  # Zoom units (1 to zoom_max)
CAM2_PTZ_PRESET_NUMBER = 1  # Preset number (ถ้าใช้ preset function แทน absolute_move)
CAM2_PTZ_USE_PRESET_FUNCTION = False  # ใช้ preset function (True) หรือ absolute_move (False)

# Timeout และ Retry Settings
CAM2_PTZ_MOVE_TIMEOUT = 5.0  # Timeout สำหรับการหมุนกล้อง (วินาที)
CAM2_PTZ_MAX_RETRIES = 3  # จำนวนครั้งที่ลองใหม่เมื่อล้มเหลว
CAM2_PTZ_VERIFICATION_TIMEOUT = 60.0  # Timeout สำหรับการตรวจสอบแต่ละ ROI (วินาที)

# Rate Limiting
CAM2_PTZ_MAX_MOVES_PER_SECOND = 2.0  # จำกัดการหมุนกล้องไม่เกิน 2 ครั้ง/วินาที
CAM2_PTZ_MIN_MOVE_INTERVAL = 0.5  # ระยะเวลาขั้นต่ำระหว่างการหมุน (วินาที)

# Safety Limits
CAM2_PTZ_MAX_PAN_DEG = 180.0  # จำกัดการหมุน Pan ไม่เกิน ±180°
CAM2_PTZ_MAX_TILT_DEG = 90.0  # จำกัดการหมุน Tilt ไม่เกิน ±90°
CAM2_PTZ_SAFETY_ENABLED = True  # เปิด/ปิด safety limits

# Queue Management
CAM2_PTZ_MAX_QUEUE_SIZE = 10  # จำกัดขนาด queue เพื่อไม่ให้ queue เต็ม
CAM2_PTZ_DUPLICATE_DISTANCE_THRESHOLD = 50  # ระยะห่างขั้นต่ำ (pixels) สำหรับตรวจสอบ duplicate

# Frame Monitoring
CAM2_PTZ_FRAME_TIMEOUT = 5.0  # ถ้า frame ไม่อัปเดตเกิน 5 วินาที → ผิดปกติ

# PIP Window Settings (Picture-in-Picture สำหรับแสดงกล้อง 2)
CAM2_PTZ_PIP_ENABLED = True  # เปิด/ปิด PIP window
CAM2_PTZ_PIP_WIDTH = 320  # ความกว้างของ PIP window (pixels)
CAM2_PTZ_PIP_HEIGHT = 180  # ความสูงของ PIP window (pixels)
CAM2_PTZ_PIP_MARGIN = 10  # ระยะห่างจากขอบ (pixels)
CAM2_PTZ_PIP_BORDER_COLOR = (0, 255, 255)  # สีขอบ PIP (BGR format - cyan)
CAM2_PTZ_PIP_BORDER_THICKNESS = 2  # ความหนาของขอบ

# Camera movement optimization
CAM2_PTZ_YOLO_DEAD_ZONE = 15  # ระยะต่ำสุดที่ต้องขยับกล้อง (pixels, default: 15)
CAM2_PTZ_SMOOTHING_FACTOR = 0.3  # ค่า smoothing สำหรับการขยับกล้อง (0.0-1.0, default: 0.3)
CAM2_PTZ_MOVE_INTERVAL = 2  # จำนวนเฟรมขั้นต่ำระหว่างการขยับกล้อง (default: 2)

# ROI-based YOLO optimization
CAM2_PTZ_USE_ROI_YOLO = True  # ใช้ ROI-based YOLO (default: True)
CAM2_PTZ_ROI_PADDING = 1.5  # Padding สำหรับ ROI (เท่าของ detection box, default: 1.5)

# YOLO resolution optimization
CAM2_PTZ_YOLO_IMGSZ = 416  # Resolution สำหรับ cam2 YOLO (default: 416)

# Motion detection
CAM2_PTZ_USE_MOTION_DETECTION = True  # ใช้ motion detection (default: True)

# Predictive tracking
CAM2_PTZ_PREDICTIVE_TRACKING = True  # ใช้ predictive tracking (default: True)
CAM2_PTZ_PREDICTIVE_YOLO_INTERVAL = 5  # เรียก YOLO ทุก N เฟรมใน tracking mode (default: 5)

# Adaptive confidence threshold
CAM2_PTZ_ADAPTIVE_CONF_THRESHOLD = True  # ใช้ adaptive confidence threshold (default: True)

# Early exit
CAM2_PTZ_EARLY_EXIT_CONF = 0.7  # Confidence สำหรับ early exit (default: 0.7)

# Verified ROI tracking
CAM2_PTZ_VERIFIED_ROI_TIMEOUT = 30.0  # เวลาที่ผ่านไปก่อนตรวจสอบซ้ำ (วินาที, default: 30)
CAM2_PTZ_VERIFIED_ROI_DISTANCE = 50  # ระยะห่างสูงสุดที่ถือว่าเป็น ROI เดิม (pixels, default: 50)
CAM2_PTZ_REMOVE_TARGET_ON_FAIL = True  # ลบ target เมื่อ verification ล้มเหลว (default: True)
CAM2_PTZ_REMOVE_TARGET_CONF_THRESHOLD = 0.1  # Confidence threshold สำหรับลบ target (default: 0.1)

# Jetson Orin Nano optimizations
CAM2_PTZ_JETSON_MODE = True  # เปิด Jetson optimization mode (default: True)
CAM2_PTZ_MAX_FRAME_BUFFER_SIZE = 2  # จำนวน frames สูงสุดใน buffer (default: 2)
CAM2_PTZ_FRAME_RESIZE = (1280, 720)  # Resize frame ก่อนประมวลผล (None = ไม่ resize, default: (1280, 720))
CAM2_PTZ_YOLO_IMGSZ_JETSON = 320  # Resolution สำหรับ YOLO ใน Jetson (default: 320)
CAM2_PTZ_MAX_TEMPERATURE = 75  # อุณหภูมิสูงสุดก่อน throttle (องศาเซลเซียส, default: 75)
CAM2_PTZ_TEMPERATURE_THROTTLE = True  # เปิด thermal throttling (default: True)
CAM2_PTZ_USE_TENSORRT = True  # ใช้ TensorRT engine (ถ้ามี, default: True)
CAM2_PTZ_TENSORRT_PRECISION = "FP16"  # TensorRT precision (FP16 หรือ INT8, default: "FP16")
CAM2_PTZ_WORKER_THREADS = 1  # จำนวน worker threads (default: 1 สำหรับ Jetson)
CAM2_PTZ_ADAPTIVE_FPS_THRESHOLD_LOW = 15.0  # FPS ต่ำสุดก่อนลด YOLO frequency (default: 15.0)
CAM2_PTZ_ADAPTIVE_FPS_THRESHOLD_CRITICAL = 10.0  # FPS ต่ำสุดก่อน skip verification (default: 10.0)

# ============================================================================
# GRID-BASED ADAPTIVE HYBRID MODE SETTINGS (OPTIMIZED)
# ============================================================================
ADAPTIVE_HYBRID_MODE_ENABLED = True  # เปิด Adaptive Hybrid Mode (ใช้ hybrid เป็นหลัก)
ADAPTIVE_HYBRID_HYSTERESIS_FRAMES = 20  # รอ 20 เฟรมก่อนสลับ (ป้องกันการสลับบ่อย)
ADAPTIVE_HYBRID_MIN_STATUS_FOR_LOCK = 'ORANGE'  # Status ต่ำสุดที่อนุญาตให้ lock ('ORANGE' หรือ 'RED')
ADAPTIVE_HYBRID_SEARCH_TIMEOUT = 60  # จำนวนเฟรมที่ใช้ multi mode ค้นหา (ถ้าเกินนี้ยังไม่เจอ → กลับ hybrid)

# Performance optimization
ADAPTIVE_HYBRID_UPDATE_INTERVAL = 5  # อัปเดตทุก 5 เฟรม (ไม่ใช่ทุกเฟรม) - PERFORMANCE OPTIMIZATION
GRID_CONFIDENCE_CACHE_FRAMES = 3  # Cache grid confidence เป็นเวลา 3 เฟรม

# Grid-based switching parameters
GRID_NOISE_THRESHOLD_FOR_HYBRID = 0.3  # ถ้า grid noise ต่ำกว่า → ใช้ hybrid (พื้นหลังนิ่ง)
GRID_NOISE_THRESHOLD_FOR_MULTI = 0.6  # ถ้า grid noise สูงกว่า → ใช้ multi (พื้นหลังขยับ)
GRID_CONFIDENCE_WEIGHT = 0.3  # น้ำหนักของ grid confidence (0.0-1.0)

# ============================================================================
# DRONE PATH CONTINUITY & SIZE STABILITY DETECTION (Hybrid Mode)
# ============================================================================
# ตรวจจับโดรนจากลักษณะการเคลื่อนที่ต่อเนื่องและขนาดคงที่
DRONE_PATH_CONTINUITY_ENABLED = True  # เปิด/ปิดการตรวจสอบ path continuity
DRONE_SIZE_STABILITY_ENABLED = True  # เปิด/ปิดการตรวจสอบ size stability

# Path Continuity Settings
DRONE_PATH_CONTINUITY_MIN_POINTS = 10  # จำนวนจุดต่ำสุดสำหรับการวิเคราะห์ continuity
DRONE_PATH_MAX_GAP_RATIO = 2.0  # gap สูงสุดที่ยอมรับได้ (เท่าของค่าเฉลี่ย)
DRONE_PATH_SMOOTH_TRANSITION_THRESHOLD = 0.7  # threshold สำหรับ smooth transition (0.0-1.0)
DRONE_PATH_CONTINUITY_BOOST = 0.25  # เพิ่ม confidence เมื่อ path ต่อเนื่องมาก

# Size Stability Settings
DRONE_SIZE_STABILITY_MIN_POINTS = 10  # จำนวนจุดต่ำสุดสำหรับการวิเคราะห์ size stability
DRONE_SIZE_STABILITY_CV_THRESHOLD = 0.25  # CV สูงสุดที่ยอมรับได้ (25% = ค่อนข้างคงที่)
DRONE_SIZE_STABILITY_BOOST = 0.2  # เพิ่ม confidence เมื่อ size stable

# Adaptive Thresholds ตามขนาดกล่อง Motion
# กล่องใหญ่ = บินเร็ว → threshold หลวมขึ้น
# กล่องเล็ก = บินช้า → threshold เข้มงวดขึ้น
DRONE_ADAPTIVE_GAP_RATIO_LARGE = 2.5  # gap ratio สำหรับกล่องใหญ่ (หลวมขึ้น)
DRONE_ADAPTIVE_GAP_RATIO_SMALL = 1.5  # gap ratio สำหรับกล่องเล็ก (เข้มงวดขึ้น)
DRONE_ADAPTIVE_SMOOTH_THRESHOLD_LARGE = 0.6  # smooth threshold สำหรับกล่องใหญ่ (หลวมขึ้น)
DRONE_ADAPTIVE_SMOOTH_THRESHOLD_SMALL = 0.8  # smooth threshold สำหรับกล่องเล็ก (เข้มงวดขึ้น)
DRONE_MOTION_BOX_SIZE_THRESHOLD_LARGE = 100  # พื้นที่กล่อง motion ที่ถือว่าใหญ่ (pixels²)
DRONE_MOTION_BOX_SIZE_THRESHOLD_SMALL = 30  # พื้นที่กล่อง motion ที่ถือว่าเล็ก (pixels²)

# Background Noise/Motion Check Settings (สำหรับ Path Continuity & Size Stability)
# ตรวจสอบเฉพาะใน ROI area ของ target (ไม่ใช่ทั้งเฟรม)
DRONE_BACKGROUND_NOISE_CHECK_ENABLED = True  # เปิด/ปิดการตรวจสอบ background noise
DRONE_BACKGROUND_NOISE_THRESHOLD = 0.3  # noise threshold สูงสุดที่ยอมรับได้ใน ROI (0.0-1.0, ต่ำ=noise น้อย)
DRONE_BACKGROUND_MOTION_IN_ROI_THRESHOLD = 5  # จำนวน motion boxes สูงสุดใน ROI ที่ยอมรับได้ (ถ้าเกินนี้ = motion ยุ่งเหยิง)

# Path History Memory Optimization Settings
DRONE_PATH_HISTORY_OPTIMIZATION_ENABLED = True  # เปิด/ปิดการ optimize
DRONE_GREEN_PATH_HISTORY_ENABLED = True  # เปิดการเก็บ path history สำหรับ GREEN

# ขีดจำกัด path history ตามขนาด (หน่วย: เฟรม)
DRONE_TINY_PATH_HISTORY_MAX_FRAMES = None  # ใช้ seconds-based
DRONE_SMALL_PATH_HISTORY_MAX_FRAMES = 30  # จำกัดที่ 30 เฟรมสำหรับ small objects
DRONE_MEDIUM_PATH_HISTORY_MAX_FRAMES = 20  # จำกัดที่ 20 เฟรมสำหรับ medium objects
DRONE_LARGE_PATH_HISTORY_MAX_FRAMES = 15  # จำกัดที่ 15 เฟรมสำหรับ large objects

# Path history สำหรับ GREEN status แบบ interval-based (ตามขนาด)
DRONE_GREEN_TINY_PATH_INTERVAL = 10  # เก็บทุก 10 เฟรมสำหรับ tiny objects
DRONE_GREEN_SMALL_PATH_INTERVAL = 15  # เก็บทุก 15 เฟรมสำหรับ small objects
DRONE_GREEN_MEDIUM_PATH_INTERVAL = 20  # เก็บทุก 20 เฟรมสำหรับ medium objects
DRONE_GREEN_LARGE_PATH_INTERVAL = 30  # เก็บทุก 30 เฟรมสำหรับ large objects

# Cleanup interval
DRONE_PATH_HISTORY_CLEANUP_INTERVAL = 30  # ทำ cleanup ทุก 30 เฟรม

# Path-Based ORANGE Detection Settings (สำหรับ Non-Tiny Objects)
# สำหรับกรณีที่ motion path ชัดเจนมาก แต่ YOLO confidence ต่ำ
DRONE_PATH_BASED_ORANGE_ENABLED = True  # เปิด/ปิด path-based ORANGE detection
DRONE_PATH_BASED_ORANGE_MIN_HISTORY_SECONDS = 1.0  # เวลาต่ำสุด (วินาที)
DRONE_PATH_BASED_ORANGE_MIN_CONFIDENCE = 0.6  # confidence ต่ำสุดจาก path analysis (60%)
DRONE_PATH_BASED_ORANGE_MIN_TRACKING_DURATION_SECONDS = 0.5  # ระยะเวลาติดตามต่ำสุด
DRONE_PATH_BASED_ORANGE_MIN_PATH_POINTS = 15  # จำนวน path points ต่ำสุด
DRONE_PATH_BASED_ORANGE_MOTION_CLEAR_THRESHOLD = 0.7  # threshold สำหรับ motion path ที่ชัดเจน
DRONE_PATH_BASED_ORANGE_CHECK_INTERVAL = 6  # ตรวจสอบทุก N เฟรม (เพื่อ performance)

# Motion-Only Target Tracking Settings
# สำหรับ track motion boxes ที่เคลื่อนที่ smooth แต่ยังไม่มี YOLO detection match
MOTION_ONLY_TARGET_ENABLED = True  # เปิด/ปิด motion-only target tracking
MOTION_ONLY_MIN_PATH_POINTS = 10  # จำนวน path points ต่ำสุดก่อนสร้าง target
MOTION_ONLY_STORE_PATH_EVERY_FRAME = True  # เก็บ path ทุกเฟรมสำหรับ motion-only targets (ไม่ใช้ interval-based)
MOTION_ONLY_MIN_PATH_QUALITY = 0.6  # path quality threshold (0.0-1.0)
MOTION_ONLY_MIN_TRACKING_DURATION = 0.5  # ระยะเวลาติดตามต่ำสุด (วินาที)
MOTION_ONLY_CHECK_INTERVAL = 1  # ตรวจสอบทุกเฟรม (เพื่อให้ตรวจจับจุดเล็กๆ ได้ดีขึ้น)
MOTION_ONLY_MAX_TARGETS = 5  # จำนวน motion-only targets สูงสุด
MOTION_ONLY_PATH_HISTORY_MAX_FRAMES = 30  # จำกัด path history (เฟรม)

# ============================================================================
# MOTION-ONLY STATUS PROGRESSION SETTINGS
# ============================================================================
MOTION_ONLY_ORANGE_PATH_QUALITY_THRESHOLD = 0.6  # path quality threshold สำหรับเปลี่ยนเป็น ORANGE
MOTION_ONLY_RED_PATH_QUALITY_THRESHOLD = 0.75  # path quality threshold สำหรับเปลี่ยนเป็น RED
MOTION_ONLY_RED_MIN_PATH_POINTS = 20  # จำนวน path points ต่ำสุดสำหรับเปลี่ยนเป็น RED
MOTION_ONLY_STATUS_CHECK_INTERVAL = 3  # ตรวจสอบ status ทุก N เฟรม (เพื่อ performance)

# ============================================================================
# SIZE-ADAPTIVE TARGET DETECTION SETTINGS
# ============================================================================
# สำหรับแบ่งเงื่อนไขตามขนาดวัตถุ (TINY, SMALL, MEDIUM, LARGE)

# === TINY Objects (เล็กมาก - YOLO อาจพลาด) ===
TINY_MOTION_PATH_WEIGHT = 0.8  # น้ำหนัก motion path (80%)
TINY_YOLO_WEIGHT = 0.2  # น้ำหนัก YOLO (20%)
TINY_MIN_PATH_POINTS = 8  # จำนวน path points ต่ำสุด (น้อยกว่า small)
TINY_MIN_PATH_QUALITY = 0.5  # path quality threshold (ต่ำกว่า small)
TINY_MIN_TRACKING_DURATION = 0.3  # ระยะเวลาติดตามต่ำสุด (วินาที)
TINY_YOLO_IMMEDIATE_LOCK_CONF = 0.7  # YOLO confidence สำหรับ immediate lock (สูง)
TINY_RED_YOLO_WEIGHT = 0.3  # น้ำหนัก YOLO สำหรับ RED decision (30%)
TINY_RED_MOTION_WEIGHT = 0.7  # น้ำหนัก motion สำหรับ RED decision (70%)

# === Size-Adaptive Size Stability Settings ===
# สำหรับ TINY objects: ยอมรับการเปลี่ยนแปลงขนาดสูง (2-3 เท่า)
TINY_SIZE_STABILITY_CV_THRESHOLD = 0.7  # CV สูงสุดสำหรับ TINY (70% = ยอมรับการเปลี่ยนแปลงขนาดสูง)
SMALL_SIZE_STABILITY_CV_THRESHOLD = 0.4  # CV สูงสุดสำหรับ SMALL (40%)
MEDIUM_SIZE_STABILITY_CV_THRESHOLD = 0.3  # CV สูงสุดสำหรับ MEDIUM (30%)
LARGE_SIZE_STABILITY_CV_THRESHOLD = 0.25  # CV สูงสุดสำหรับ LARGE (25% = เดิม)

# Size-Adaptive Size Stability Weights (สำหรับ path quality calculation)
TINY_SIZE_STABILITY_WEIGHT = 0.1  # น้ำหนักต่ำสำหรับ TINY (10% แทน 20%)
SMALL_SIZE_STABILITY_WEIGHT = 0.15  # น้ำหนักสำหรับ SMALL (15%)
MEDIUM_SIZE_STABILITY_WEIGHT = 0.2  # น้ำหนักสำหรับ MEDIUM (20% = เดิม)
LARGE_SIZE_STABILITY_WEIGHT = 0.2  # น้ำหนักสำหรับ LARGE (20% = เดิม)

# === SMALL Objects ===
SMALL_MOTION_PATH_WEIGHT = 0.6  # น้ำหนัก motion path (60%)
SMALL_YOLO_WEIGHT = 0.4  # น้ำหนัก YOLO (40%)
SMALL_MIN_PATH_POINTS = 10  # จำนวน path points ต่ำสุด
SMALL_MIN_PATH_QUALITY = 0.6  # path quality threshold
SMALL_MIN_TRACKING_DURATION = 0.5  # ระยะเวลาติดตามต่ำสุด (วินาที)
SMALL_YOLO_IMMEDIATE_LOCK_CONF = 0.6  # YOLO confidence สำหรับ immediate lock
SMALL_RED_YOLO_WEIGHT = 0.5  # น้ำหนัก YOLO สำหรับ RED decision (50%)
SMALL_RED_MOTION_WEIGHT = 0.5  # น้ำหนัก motion สำหรับ RED decision (50%)

# === MEDIUM Objects ===
MEDIUM_MOTION_PATH_WEIGHT = 0.4  # น้ำหนัก motion path (40%)
MEDIUM_YOLO_WEIGHT = 0.6  # น้ำหนัก YOLO (60%)
MEDIUM_MIN_PATH_POINTS = 12  # จำนวน path points ต่ำสุด
MEDIUM_MIN_PATH_QUALITY = 0.65  # path quality threshold
MEDIUM_MIN_TRACKING_DURATION = 0.5  # ระยะเวลาติดตามต่ำสุด (วินาที)
MEDIUM_YOLO_IMMEDIATE_LOCK_CONF = 0.5  # YOLO confidence สำหรับ immediate lock
MEDIUM_RED_YOLO_WEIGHT = 0.7  # น้ำหนัก YOLO สำหรับ RED decision (70%)
MEDIUM_RED_MOTION_WEIGHT = 0.3  # น้ำหนัก motion สำหรับ RED decision (30%)

# === LARGE Objects ===
LARGE_MOTION_PATH_WEIGHT = 0.2  # น้ำหนัก motion path (20%)
LARGE_YOLO_WEIGHT = 0.8  # น้ำหนัก YOLO (80%)
LARGE_MIN_PATH_POINTS = 15  # จำนวน path points ต่ำสุด
LARGE_MIN_PATH_QUALITY = 0.7  # path quality threshold
LARGE_MIN_TRACKING_DURATION = 0.5  # ระยะเวลาติดตามต่ำสุด (วินาที)
LARGE_YOLO_IMMEDIATE_LOCK_CONF = 0.4  # YOLO confidence สำหรับ immediate lock (ต่ำ)
LARGE_RED_YOLO_WEIGHT = 0.85  # น้ำหนัก YOLO สำหรับ RED decision (85%)
LARGE_RED_MOTION_WEIGHT = 0.15  # น้ำหนัก motion สำหรับ RED decision (15%)

# === Background Blending Detection ===
# สำหรับกรณีที่วัตถุกลืนกับพื้นหลัง
BLENDED_OBJECT_MOTION_BOOST = 1.5  # ตัวคูณสำหรับ motion weight เมื่อ detect blending
BLENDED_OBJECT_YOLO_PENALTY = 0.7  # ตัวคูณสำหรับ YOLO weight เมื่อ detect blending
BLENDED_DETECTION_ENABLED = True  # เปิด/ปิดการตรวจสอบ blending

# === Target Limit ===
MAX_TOTAL_TARGETS = 5  # จำนวน targets สูงสุด (รวม motion-only + YOLO targets)

# ============================================================================
# MOTION BOX FOOTPRINTS MODULE SETTINGS
# ============================================================================
FOOTPRINTS_MODULE_ENABLED = True  # เปิด/ปิด footprints module
FOOTPRINTS_HISTORY_FRAMES = 10  # จำนวนเฟรมที่เก็บ history (10 เฟรม)
FOOTPRINTS_MOTION_COLOR = (255, 0, 0)  # สีน้ำเงิน (BGR) สำหรับ motion footprints (เฟรมก่อนหน้า)
FOOTPRINTS_YOLO_COLOR = (0, 165, 255)  # สีส้ม (BGR) สำหรับ YOLO boxes
FOOTPRINTS_THICKNESS = 1  # ความหนาของเส้นขอบ

# Background Stability Check
FOOTPRINTS_BG_STABILITY_ENABLED = True  # เปิด/ปิดการตรวจสอบพื้นหลังนิ่ง
FOOTPRINTS_BG_MOTION_THRESHOLD = 50  # จำนวน motion boxes สูงสุดที่ยอมรับได้ (ถ้าเกิน = พื้นหลังยุ่งเหยิง) - DEPRECATED: ใช้ค่าเฉลี่ยแทน
FOOTPRINTS_BG_CHECK_INTERVAL = 1  # ตรวจสอบทุก N เฟรม

# Background stability check (ใช้ค่าเฉลี่ย)
FOOTPRINTS_BG_MOTION_HISTORY_SIZE = 30  # เก็บประวัติ 30 เฟรมสำหรับคำนวณค่าเฉลี่ย
FOOTPRINTS_BG_MOTION_DEVIATION_FACTOR = 2.0  # ค่าเบี่ยงเบนที่ยอมรับได้ (2.0x = 100% มากกว่าค่าเฉลี่ย) - เพิ่มจาก 1.5 → 2.0 (ยอมรับได้มากขึ้น)
FOOTPRINTS_BG_MOTION_MIN_SPREAD_RATIO = 0.3  # สัดส่วนของจอที่ต้องมี motion (30% = มี motion กระจายทั่วจอ)
FOOTPRINTS_BG_MOTION_MIN_COUNT = 1  # จำนวน motion boxes ต่ำสุดที่ยอมรับได้ - ลดจาก 5 → 1 (ยอมรับได้แม้มี motion น้อย)

# Resolution-dependent YOLO interval (เฟรมที่วาด YOLO footprints)
FOOTPRINTS_YOLO_INTERVAL_4K = 5  # 4K (3840x2160): ทุก 5 เฟรม
FOOTPRINTS_YOLO_INTERVAL_2K = 3  # 2K (2560x1440): ทุก 3 เฟรม
FOOTPRINTS_YOLO_INTERVAL_1080P = 2  # 1080p (1920x1080): ทุก 2 เฟรม
FOOTPRINTS_YOLO_INTERVAL_720P = 1  # 720p: ทุกเฟรม

# Size-based confidence thresholds สำหรับ YOLO footprints
FOOTPRINTS_YOLO_CONF_TINY = 0.001  # TINY objects: conf >= 0.001 (ลดจาก 0.01)
FOOTPRINTS_YOLO_CONF_SMALL = 0.01  # SMALL objects: conf >= 0.01 (ลดจาก 0.10)
FOOTPRINTS_YOLO_CONF_MEDIUM = 0.05  # MEDIUM objects: conf >= 0.05 (ลดจาก 0.20)
FOOTPRINTS_YOLO_CONF_LARGE = 0.10  # LARGE objects: conf >= 0.10 (ลดจาก 0.30)

# ============================================================================
# FOOTPRINTS MODULE - LOCAL DENSITY CHECK SETTINGS
# ============================================================================
# ตรวจสอบความหนาแน่นของ motion boxes ในบริเวณรอบๆ แต่ละ box
# ถ้า motion หนาแน่นเกินไป → ไม่วาด (แสดงว่ากล้องขยับ)
FOOTPRINTS_LOCAL_DENSITY_CHECK_ENABLED = True  # เปิด/ปิดการตรวจสอบความหนาแน่น
FOOTPRINTS_LOCAL_DENSITY_RADIUS_RATIO = 0.1  # รัศมีตรวจสอบ (10% ของ frame diagonal)
FOOTPRINTS_LOCAL_DENSITY_MAX_BOXES = 5  # จำนวน motion boxes สูงสุดในรัศมีที่ยอมรับได้
FOOTPRINTS_LOCAL_DENSITY_COVERAGE_THRESHOLD = 0.3  # Coverage ratio สูงสุด (30% = motion เต็มไปหมด)

# Thresholds สำหรับ TINY และ SMALL objects (เข้มงวดกว่า - ตรวจเฉพาะบริเวณใกล้ๆ)
FOOTPRINTS_LOCAL_DENSITY_TINY_MAX_BOXES = 2  # TINY: จำนวน motion boxes สูงสุดในรัศมีใกล้ๆ (เข้มงวดกว่า)
FOOTPRINTS_LOCAL_DENSITY_SMALL_MAX_BOXES = 3  # SMALL: จำนวน motion boxes สูงสุดในรัศมีใกล้ๆ (เข้มงวดกว่า)
FOOTPRINTS_LOCAL_DENSITY_TINY_COVERAGE_THRESHOLD = 0.15  # TINY: Coverage ratio สูงสุด (15% = เข้มงวดกว่า)
FOOTPRINTS_LOCAL_DENSITY_SMALL_COVERAGE_THRESHOLD = 0.2  # SMALL: Coverage ratio สูงสุด (20% = เข้มงวดกว่า)

# ============================================================================
# FOOTPRINTS MOTION CONTINUITY CHECK SETTINGS
# ============================================================================
# ตรวจสอบความต่อเนื่องของ motion เพื่อกรองเฉพาะ target ที่เคลื่อนที่ได้แน่นอน
FOOTPRINTS_MOTION_CONTINUITY_ENABLED = True  # เปิด/ปิดการตรวจสอบความต่อเนื่อง
FOOTPRINTS_MOTION_MIN_CONTINUITY_FRAMES = 2  # จำนวนเฟรมติดกันขั้นต่ำ (2 เฟรม)
FOOTPRINTS_MOTION_MAX_DISTANCE_RATIO = 0.1  # ระยะทางสูงสุดระหว่างเฟรม (10% ของ frame diagonal)
FOOTPRINTS_MOTION_SIZE_CONSISTENCY_RATIO = 0.3  # อัตราส่วนขนาดที่ยอมรับได้ (30% = ขนาดใกล้เคียง)
FOOTPRINTS_MOTION_MAX_FOOTPRINTS_PER_FRAME = 15  # จำนวน footprints สูงสุดต่อเฟรม (วาดเฉพาะที่สำคัญ)

# ============================================================================
# FOOTPRINTS MODULE - SIMPLIFIED FAST MODE
# ============================================================================
# Simplified Fast Mode: ใช้ MOG2 เป็นหลัก + YOLO ช่วยกรอง target จริงๆ
# ลดการกรองลงมาก → เร็วขึ้นมาก → ได้รอยเท้า 1 pixel
FOOTPRINTS_SIMPLIFIED_MODE = True  # เปิด/ปิด simplified mode (เร็วมาก)
FOOTPRINTS_SIMPLIFIED_MAX_FOOTPRINTS = 50  # เพิ่ม limit (ให้เห็นมากขึ้น)

# ============================================================================
# CONTOUR DETECTION RESIZE SETTINGS (สำหรับเพิ่มความเร็ว)
# ============================================================================
# Resize mask ก่อนหา contours เพื่อเพิ่มความเร็ว (เฉพาะเมื่อ min_area > 1)
CONTOUR_DETECTION_RESIZE_4K = True  # เปิด/ปิดการ resize สำหรับ 4K
CONTOUR_DETECTION_RESIZE_2K = True  # เปิด/ปิดการ resize สำหรับ 2K
CONTOUR_DETECTION_RESIZE_MIN_AREA_THRESHOLD = 1  # ถ้า min_area <= threshold → ไม่ resize (รักษา 1 pixel)
CONTOUR_DETECTION_SCALE_4K = 0.5  # Scale factor สำหรับ 4K (50% = เร็วขึ้น 4 เท่า)
CONTOUR_DETECTION_SCALE_2K = 0.66  # Scale factor สำหรับ 2K (66% = เร็วขึ้น 2.25 เท่า)

# ============================================================================
# FOOTPRINTS YOLO-GUIDED FILTERING SETTINGS
# ============================================================================
# YOLO Size Filter (กรองขนาด - เร็วสุด)
FOOTPRINTS_YOLO_SIZE_FILTER_ENABLED = True  # เปิด/ปิด YOLO size filter
FOOTPRINTS_YOLO_SIZE_FILTER_USE_MAX = True  # ใช้ YOLO box ที่ใหญ่ที่สุดในเฟรม

# YOLO Proximity Check (ตรวจสอบระยะทาง - วาดทุกเฟรมถ้าใกล้)
FOOTPRINTS_YOLO_PROXIMITY_ENABLED = True  # เปิด/ปิด YOLO proximity check
FOOTPRINTS_YOLO_PROXIMITY_DISTANCE_RATIO = 0.12  # ระยะทางสูงสุดที่ถือว่าใกล้ YOLO (12% ของ frame diagonal)

# Isolated Small Motion Detection (ตรวจจับ motion เล็กๆ ที่โดดเดี่ยว)
FOOTPRINTS_ISOLATED_SMALL_MOTION_ENABLED = True  # เปิด/ปิด isolated small motion detection
FOOTPRINTS_ISOLATED_MOTION_RADIUS_RATIO = 0.12  # รัศมีตรวจสอบ isolation (12% ของ frame diagonal)
FOOTPRINTS_ISOLATED_MOTION_MAX_NEIGHBORS = 2  # จำนวน motion boxes สูงสุดในรัศมีที่ยอมรับได้ (2 = แทบไม่มี)

# Resolution-Dependent Draw Interval (วาดบางเฟรมตาม resolution)
FOOTPRINTS_FAR_YOLO_DRAW_INTERVAL_4K = 4  # วาดทุก 4 เฟรมสำหรับ motion ที่ไม่ใกล้ YOLO (4K)
FOOTPRINTS_FAR_YOLO_DRAW_INTERVAL_2K = 3  # วาดทุก 3 เฟรมสำหรับ motion ที่ไม่ใกล้ YOLO (2K)
FOOTPRINTS_FAR_YOLO_DRAW_INTERVAL_1080P = 2  # วาดทุก 2 เฟรมสำหรับ motion ที่ไม่ใกล้ YOLO (1080p)
FOOTPRINTS_FAR_YOLO_DRAW_INTERVAL_720P = 1  # วาดทุกเฟรมสำหรับ motion ที่ไม่ใกล้ YOLO (720p)

# Motion Clustering Check (ตรวจสอบการกระจายตัว)
FOOTPRINTS_CLUSTERING_CHECK_ENABLED = True  # เปิด/ปิดการตรวจสอบ clustering (กระจุก)

# Continuity Check Interval (skip บางเฟรม)
FOOTPRINTS_MOTION_CONTINUITY_CHECK_INTERVAL_4K = 2  # ตรวจสอบทุก 2 เฟรม (สำหรับ 4K)

# Local Density Check Limit (จำกัดจำนวน)
FOOTPRINTS_MAX_DENSITY_CHECK_BOXES_4K = 50  # ตรวจสอบเฉพาะ 50 อันแรก (สำหรับ 4K)

# ============================================================================
# FOOTPRINTS ROI COVERAGE SETTINGS
# ============================================================================
# ROI coverage สำหรับ footprints (YOLO boxes, motion groups, isolated motion)
FOOTPRINTS_ROI_ENABLED = True  # เปิด/ปิด ROI จาก footprints
FOOTPRINTS_ROI_UPDATE_INTERVAL = 2  # อัปเดตทุก N เฟรม (2 = ทุก 2 เฟรม) - performance optimization
FOOTPRINTS_ROI_MOTION_GROUP_DISTANCE_RATIO = 0.05  # ระยะห่างสำหรับจัดกลุ่ม motion (5% ของ frame diagonal)
FOOTPRINTS_ROI_MIN_GROUP_SIZE = 3  # จำนวน motion boxes ต่ำสุดสำหรับเป็นกลุ่ม
FOOTPRINTS_ROI_MAX_MOTION_BOXES_CHECK = 50  # จำกัดจำนวน motion boxes ที่ตรวจสอบ (performance)
FOOTPRINTS_ROI_ISOLATED_PATH_MIN_FRAMES = 3  # จำนวนเฟรมต่ำสุดสำหรับ isolated motion path (ลดจาก 5)
FOOTPRINTS_ROI_MAX_FOOTPRINTS_HISTORY = 3  # จำนวนเฟรมล่าสุดที่ใช้จาก history (3 เฟรม)
FOOTPRINTS_ROI_MAX_ROI_COUNT = 10  # จำนวน ROI สูงสุดจาก footprints

# ROI Overlap Handling Settings
FOOTPRINTS_ROI_OVERLAP_IOU_THRESHOLD = 0.3  # IOU threshold สำหรับ overlap (30% overlap = filter)
FOOTPRINTS_ROI_OVERLAP_MERGE_ENABLED = True  # เปิด/ปิดการ merge ROI ที่ overlap (False = filter, True = merge)
FOOTPRINTS_ROI_OVERLAP_MERGE_IOU_THRESHOLD = 0.2  # IOU threshold สำหรับ merge (20% overlap = merge)

# ============================================================================
# ROI SWARMING DETECTION AND FOCUS MODE SETTINGS
# ============================================================================
# Swarming Detection Settings
FOOTPRINTS_ROI_SWARMING_ENABLED = True  # เปิด/ปิดการตรวจจับ swarming behavior
FOOTPRINTS_ROI_MOVEMENT_THRESHOLD_RATIO = 0.005  # ระยะทางขยับต่ำสุด (0.5% ของ frame diagonal) - ลดจาก 0.01
FOOTPRINTS_ROI_MERGE_DETECTION_ENABLED = True  # เปิด/ปิดการตรวจจับ ROI merge
FOOTPRINTS_ROI_SWARMING_MIN_MOVEMENT_FRAMES = 2  # จำนวนเฟรมต่ำสุดที่ต้องขยับเพื่อนับว่าเป็น swarming - ลดจาก 3
FOOTPRINTS_ROI_SWARMING_CHECK_INTERVAL = 2  # ตรวจสอบ swarming ทุก N เฟรม (2 = ทุก 2 เฟรม) - performance optimization

# Power Accumulation Settings
FOOTPRINTS_ROI_POWER_MOVEMENT_BONUS = 0.2  # เพิ่มพลังเมื่อ ROI ขยับ (0.2 per frame) - เพิ่มจาก 0.1
FOOTPRINTS_ROI_POWER_MERGE_BONUS = 1.0  # เพิ่มพลังเมื่อ ROI merge (1.0 per merge) - เพิ่มจาก 0.5
FOOTPRINTS_ROI_POWER_YOLO_BONUS = 0.5  # เพิ่มพลังเมื่อมี YOLO detection (0.5 per detection) - เพิ่มจาก 0.3
FOOTPRINTS_ROI_POWER_DECAY_RATE = 0.03  # ลดพลังเมื่อไม่มีการเคลื่อนไหว (0.03 per frame) - ลดจาก 0.05
FOOTPRINTS_ROI_POWER_MAX = 10.0  # พลังสูงสุด
FOOTPRINTS_ROI_POWER_MIN = 0.0  # พลังต่ำสุด

# YELLOW Status Settings
FOOTPRINTS_ROI_YELLOW_POWER_THRESHOLD = 2.0  # พลังต่ำสุดสำหรับเปลี่ยนเป็น YELLOW - ลดจาก 3.0
FOOTPRINTS_ROI_YELLOW_DURATION_THRESHOLD = 5  # จำนวนเฟรมต่ำสุดที่ต้องอยู่ใน YELLOW ก่อนเริ่ม focus mode

# Focus Mode Settings
FOOTPRINTS_ROI_FOCUS_MODE_ENABLED = True  # เปิด/ปิด focus mode
FOOTPRINTS_ROI_FOCUS_YOLO_INTERVAL = 2  # ทำ YOLO detection ทุก N เฟรม (2 = ทุก 2 เฟรม)
FOOTPRINTS_ROI_FOCUS_MIN_CONF = 0.1  # Confidence ต่ำสุดสำหรับ YOLO detection (10%)
FOOTPRINTS_ROI_FOCUS_TARGET_MIN_CONF = 0.3  # Confidence ต่ำสุดสำหรับเลือกเป็น target (30%)
FOOTPRINTS_ROI_FOCUS_TARGET_MIN_HEIGHT_RATIO = 0.02  # ความสูงต่ำสุดของ target (2% ของ frame height) - กล่องตัวสูงๆ
FOOTPRINTS_ROI_MAX_FOCUS_YOLO_ROI = 3  # จำนวน ROI สูงสุดที่ทำ YOLO detection พร้อมกัน (3 ROI) - performance optimization

# ROI Limiting Settings
FOOTPRINTS_ROI_LIMIT_TO_FOCUS_ENABLED = True  # เปิด/ปิดการจำกัด ROI ให้เหลือเฉพาะที่ focus
FOOTPRINTS_ROI_MAX_FOCUS_ROI = 3  # จำนวน ROI สูงสุดที่ focus พร้อมกัน (3 ROI)

# ============================================================================
# SIZE-ADAPTIVE RED STATUS DECISION SETTINGS
# ============================================================================
# สำหรับกรณี YOLO conf ต่ำ + motion overlap → RED เร็ว

# YOLO conf ต่ำ + motion overlap thresholds (ต่ำกว่า normal thresholds)
YOLO_MOTION_OVERLAP_RED_ENABLED = True  # เปิด/ปิดการเปลี่ยน RED เร็วสำหรับ YOLO + motion
YOLO_MOTION_OVERLAP_MIN_IOU = 0.3  # IOU ต่ำสุดสำหรับ motion overlap (30%)
YOLO_MOTION_OVERLAP_MIN_CONF = 0.15  # YOLO conf ต่ำสุดที่ยอมรับได้ (15%)
YOLO_MOTION_OVERLAP_MIN_MOTION_QUALITY = 0.5  # Motion path quality ต่ำสุด (50%)

# Size-adaptive thresholds สำหรับ YOLO + motion overlap → RED
TINY_YOLO_MOTION_RED_MIN_COMPOSITE = 0.35  # TINY: composite score ต่ำสุด (35%)
SMALL_YOLO_MOTION_RED_MIN_COMPOSITE = 0.4   # SMALL: composite score ต่ำสุด (40%)
MEDIUM_YOLO_MOTION_RED_MIN_COMPOSITE = 0.45 # MEDIUM: composite score ต่ำสุด (45%)
LARGE_YOLO_MOTION_RED_MIN_COMPOSITE = 0.5   # LARGE: composite score ต่ำสุด (50%)

# Size-adaptive red_score accumulation สำหรับ YOLO + motion overlap
TINY_YOLO_MOTION_RED_SCORE_BOOST = 2.0  # TINY: boost red_score (2.0 เท่า)
SMALL_YOLO_MOTION_RED_SCORE_BOOST = 1.8  # SMALL: boost red_score (1.8 เท่า)
MEDIUM_YOLO_MOTION_RED_SCORE_BOOST = 1.5  # MEDIUM: boost red_score (1.5 เท่า)
LARGE_YOLO_MOTION_RED_SCORE_BOOST = 1.3  # LARGE: boost red_score (1.3 เท่า)

# ROI Lock Mode RED Decision Settings
ROI_LOCK_RED_ENABLED = True  # เปิด/ปิดการเปลี่ยน RED สำหรับ targets ใน lock mode
ROI_LOCK_RED_MIN_COMPOSITE = 0.4  # composite score ต่ำสุดสำหรับ lock mode (40% - ต่ำกว่า normal)
ROI_LOCK_RED_SCORE_BOOST = 1.5  # boost red_score สำหรับ lock mode (1.5 เท่า)
ROI_LOCK_RED_MIN_DURATION = 0.2  # ระยะเวลาติดตามต่ำสุดสำหรับ lock mode (0.2 วินาที - ต่ำกว่า normal)

# Size-adaptive min_composite thresholds (ปรับจาก hardcoded values)
TINY_RED_MIN_COMPOSITE = 0.35  # TINY: ต่ำกว่าเดิม (0.4 → 0.35)
SMALL_RED_MIN_COMPOSITE = 0.45  # SMALL: ต่ำกว่าเดิม (0.5 → 0.45)
MEDIUM_RED_MIN_COMPOSITE = 0.55  # MEDIUM: ต่ำกว่าเดิม (0.6 → 0.55)
LARGE_RED_MIN_COMPOSITE = 0.6   # LARGE: ต่ำกว่าเดิม (0.65 → 0.6)

# ============================================================================
# YOLO IMMEDIATE RED LOCK SETTINGS (เปลี่ยนเป็น RED ทันทีโดยไม่ต้องรอ path)
# ============================================================================
YOLO_IMMEDIATE_RED_ENABLED = True  # เปิด/ปิด YOLO immediate RED lock
YOLO_IMMEDIATE_RED_CONF_THRESHOLD = 0.6  # YOLO confidence threshold สำหรับ immediate RED (0.6 = 60%)
# สำหรับโดรนที่เห็นชัดเจน (conf >= 0.6) → เปลี่ยนเป็น RED ทันทีโดยไม่ต้องรอ path history

# ============================================================================
# TARGET CLASSIFICATION FROM PATH SETTINGS
# ============================================================================
PATH_CLASSIFICATION_ENABLED = True  # เปิด/ปิดการแยกประเภท target จาก path
PATH_CLASSIFICATION_MIN_POINTS = 15  # จำนวน path points ต่ำสุดสำหรับการแยกประเภท (15 จุด)
PATH_CLASSIFICATION_RECOMMENDED_POINTS = 30  # จำนวน path points ที่แนะนำสำหรับการแยกประเภท (30 จุด)
PATH_CLASSIFICATION_UPDATE_INTERVAL = 1  # อัปเดตทุก N เฟรม (1 = ทุกเฟรม)

# Resolution-dependent และ FPS-aware thresholds
PATH_CLASSIFICATION_USE_RESOLUTION_ADAPTIVE = True  # ใช้ resolution-dependent thresholds
PATH_CLASSIFICATION_USE_FPS_AWARE = True  # ใช้ FPS-aware thresholds (ปรับ velocity thresholds ตาม FPS)

# DRONE Characteristics (base values - จะปรับตาม resolution และ FPS)
DRONE_STRAIGHTNESS_THRESHOLD = 0.6  # straightness ต่ำ-กลาง (< 0.6)
DRONE_SMOOTHNESS_THRESHOLD = 0.7  # smoothness สูง (>= 0.7)
DRONE_VELOCITY_CONSISTENCY_THRESHOLD = 0.7  # velocity consistency สูง (>= 0.7)
DRONE_VELOCITY_MIN = 1.0  # pixels per second (base - จะปรับตาม FPS)
DRONE_VELOCITY_MAX = 120.0  # pixels per second (base - จะปรับตาม FPS)

# BIRD Characteristics (base values)
BIRD_STRAIGHTNESS_THRESHOLD = 0.5  # straightness ต่ำ (< 0.5)
BIRD_SMOOTHNESS_THRESHOLD = 0.6  # smoothness ต่ำ-กลาง (< 0.6)
BIRD_VELOCITY_CONSISTENCY_THRESHOLD = 0.5  # velocity consistency ต่ำ (< 0.5)
BIRD_VELOCITY_MIN = 5.0  # pixels per second (base - จะปรับตาม FPS)
BIRD_VELOCITY_MAX = 80.0  # pixels per second (base - จะปรับตาม FPS)

# INSECT Characteristics (base values)
INSECT_STRAIGHTNESS_THRESHOLD = 0.4  # straightness ต่ำมาก (< 0.4)
INSECT_SMOOTHNESS_THRESHOLD = 0.5  # smoothness ต่ำ (< 0.5)
INSECT_VELOCITY_CONSISTENCY_THRESHOLD = 0.3  # velocity consistency ต่ำมาก (< 0.3)
INSECT_VELOCITY_MIN = 10.0  # pixels per second (base - จะปรับตาม FPS)
INSECT_VELOCITY_MAX = 200.0  # pixels per second (base - จะปรับตาม FPS)

# AIRPLANE Characteristics (base values)
AIRPLANE_STRAIGHTNESS_THRESHOLD = 0.9  # straightness สูงมาก (>= 0.9)
AIRPLANE_SMOOTHNESS_THRESHOLD = 0.8  # smoothness สูง (>= 0.8)
AIRPLANE_VELOCITY_CONSISTENCY_THRESHOLD = 0.9  # velocity consistency สูงมาก (>= 0.9)
AIRPLANE_VELOCITY_MIN = 50.0  # pixels per second (base - จะปรับตาม FPS)
AIRPLANE_VELOCITY_MAX = 500.0  # pixels per second (base - จะปรับตาม FPS)

# NOISE Characteristics (base values)
NOISE_STRAIGHTNESS_THRESHOLD = 0.3  # straightness ต่ำมาก (< 0.3)
NOISE_SMOOTHNESS_THRESHOLD = 0.4  # smoothness ต่ำมาก (< 0.4)
NOISE_VELOCITY_CONSISTENCY_THRESHOLD = 0.2  # velocity consistency ต่ำมาก (< 0.2)

# Resolution-dependent thresholds (ratios)
# Thresholds จะถูกปรับตาม frame diagonal
PATH_STRAIGHTNESS_THRESHOLD_RATIO = 0.6  # ratio สำหรับ straightness threshold
PATH_SMOOTHNESS_THRESHOLD_RATIO = 0.7  # ratio สำหรับ smoothness threshold
PATH_VELOCITY_CONSISTENCY_THRESHOLD_RATIO = 0.7  # ratio สำหรับ velocity consistency threshold

# FPS-aware velocity thresholds (multipliers)
# Velocity thresholds จะถูกปรับตาม processing FPS
PATH_VELOCITY_FPS_MULTIPLIER = 1.0  # multiplier สำหรับ velocity thresholds (1.0 = ไม่ปรับ)
PATH_VELOCITY_FPS_REFERENCE = 30.0  # FPS reference (30 FPS)

# ============================================================================
# HYBRID MODE PERFORMANCE OPTIMIZATION
# ============================================================================
# เพิ่ม YOLO interval เพื่อเพิ่มความเร็ว
HYBRID_YOLO_INTERVAL_FAST = 8  # เพิ่มจากค่าปัจจุบัน (เร็วขึ้น)
HYBRID_SEARCH_INTERVAL_NO_TARGET_FAST = 3  # เพิ่มจาก 2
HYBRID_SEARCH_INTERVAL_WITH_TARGET_FAST = 6  # เพิ่มจาก 4

# ============================================================================
# FOOTPRINTS MODULE PERFORMANCE OPTIMIZATION
# ============================================================================
# Interval-based processing (ไม่ประมวลผลทุกเฟรม)
FOOTPRINTS_UPDATE_INTERVAL = 1  # อัปเดตทุก N เฟรม (1 = ทุกเฟรม, 2 = ทุก 2 เฟรม)
FOOTPRINTS_BG_STABILITY_CHECK_INTERVAL = 2  # ตรวจสอบ background stability ทุก N เฟรม
FOOTPRINTS_LOCAL_DENSITY_CHECK_INTERVAL = 1  # ตรวจสอบ local density ทุก N เฟรม
FOOTPRINTS_MOTION_CONTINUITY_CHECK_INTERVAL = 1  # ตรวจสอบ continuity ทุก N เฟรม

# Tiny motion settings (สำหรับ 1 pixel detection)
FOOTPRINTS_TINY_MOTION_EXEMPT = True  # ยกเว้น tiny motion จาก filtering
FOOTPRINTS_TINY_MOTION_MAX_SIZE = 4  # ขนาดสูงสุดที่ถือว่าเป็น tiny (pixels²)
FOOTPRINTS_BG_MOTION_DEVIATION_FACTOR_TINY = 3.0  # Deviation factor สำหรับ tiny motion (ยืดหยุ่นกว่า)
FOOTPRINTS_MOTION_MAX_DISTANCE_RATIO_TINY = 0.2  # Distance ratio สำหรับ tiny motion (20% ของ frame diagonal)
FOOTPRINTS_TINY_MOTION_PREDICTION_ENABLED = True  # ใช้ motion prediction สำหรับ tiny motion

# Noise filtering
FOOTPRINTS_NOISE_FILTERING_MODE = 'adaptive'  # 'strict', 'adaptive', 'relaxed'
FOOTPRINTS_NOISE_ADAPTIVE_THRESHOLD = 0.5  # Threshold สำหรับ adaptive mode

# ROI tracking improvements
FOOTPRINTS_ROI_PREDICTION_ENABLED = True  # ใช้ velocity prediction
FOOTPRINTS_ROI_EXPAND_RATIO = 1.5  # ขยาย ROI เมื่อตาม target
FOOTPRINTS_ROI_MIN_OVERLAP = 0.1  # Minimum overlap สำหรับ ROI matching

# ============================================================================
# SWARM BEHAVIOR SETTINGS (แมลงวันตอมขี้)
# ============================================================================
# Fast initial detection (เจอเร็วมาก - ไม่รอ checks)
FOOTPRINTS_FAST_INITIAL_DETECTION = True  # เปิด fast initial detection
FOOTPRINTS_FAST_INITIAL_FRAMES = 5  # จำนวนเฟรมแรกที่ไม่ต้องรอ checks (5 เฟรม)
FOOTPRINTS_FAST_INITIAL_SKIP_BG_CHECK = True  # ข้าม background stability check
FOOTPRINTS_FAST_INITIAL_SKIP_CONTINUITY = True  # ข้าม continuity check

# Motion clustering (หาจุดที่มี motion มาก)
FOOTPRINTS_MOTION_CLUSTERING_ENABLED = True  # เปิด motion clustering
FOOTPRINTS_CLUSTER_DISTANCE_RATIO = 0.05  # ระยะห่างสูงสุดสำหรับ clustering (5% ของ frame diagonal)
FOOTPRINTS_CLUSTER_MIN_SIZE = 2  # จำนวน motion boxes ต่ำสุดสำหรับ cluster (2 boxes)
FOOTPRINTS_CLUSTER_PRIORITY_MULTIPLIER = 2.0  # ตัวคูณ priority สำหรับ cluster (ยิ่งมาก = priority สูง)

# Swarm ROI generation (รุมตอม)
FOOTPRINTS_SWARM_ENABLED = True  # เปิด swarm behavior
FOOTPRINTS_SWARM_ROI_COUNT = 3  # จำนวน ROI ที่ส่งไปรุม cluster (3-5 ROI)
FOOTPRINTS_SWARM_ROI_SIZE_RATIO = 0.8  # ขนาด ROI เทียบกับ cluster size (80% = เล็กกว่า)
FOOTPRINTS_SWARM_MIN_CLUSTER_SIZE = 3  # จำนวน motion boxes ต่ำสุดสำหรับ swarm (3 boxes)

# Rapid ROI expansion (scope ให้เล็กลงเรื่อยๆ)
FOOTPRINTS_ROI_RAPID_EXPANSION = True  # เปิด rapid expansion
FOOTPRINTS_ROI_EXPANSION_INTERVAL = 1  # ขยาย ROI ทุก N เฟรม (1 = ทุกเฟรม)
FOOTPRINTS_SUB_ROI_COUNT = 2  # จำนวน sub-ROI ที่สร้างภายใน ROI หลัก (2-3 sub-ROI)
FOOTPRINTS_SUB_ROI_SIZE_RATIO = 0.6  # ขนาด sub-ROI เทียบกับ ROI หลัก (60% = เล็กกว่า)

# Parallel tracking (หลายตัวช่วยกัน)
FOOTPRINTS_PARALLEL_TRACKING_ENABLED = True  # เปิด parallel tracking
FOOTPRINTS_PARALLEL_TRACKERS_MAX = 5  # จำนวน trackers สูงสุดที่ทำงาน parallel (5 trackers)
FOOTPRINTS_PARALLEL_SEARCH_RADIUS_RATIO = 0.02  # Search radius สำหรับแต่ละ tracker (2% ของ frame diagonal)
FOOTPRINTS_PARALLEL_SHARE_INFO = True  # Share information ระหว่าง trackers

# ============================================================================
# PATH-BASED TRACKING SETTINGS (สุนัขดมกลิ่น - ตามรอยเท้า)
# ============================================================================
# Footprints persistence (รอยเท้าไม่หาย)
FOOTPRINTS_PERSISTENCE_ENABLED = True  # เปิด footprints persistence
FOOTPRINTS_PERSISTENCE_HISTORY_FRAMES = 20  # เก็บ footprints history นานขึ้น (20 เฟรม)
FOOTPRINTS_PERSISTENCE_DECAY_RATE = 0.9  # Decay rate สำหรับ footprints (0.9 = fade out ช้า)
FOOTPRINTS_PERSISTENCE_MIN_STRENGTH = 0.1  # ความแรงต่ำสุดก่อนลบ (0.1 = เก็บไว้จนเกือบหาย)

# Path following (ตาม path ไปเรื่อยๆ)
FOOTPRINTS_PATH_FOLLOWING_ENABLED = True  # เปิด path following
FOOTPRINTS_PATH_PREDICTION_ENABLED = True  # ใช้ velocity prediction
FOOTPRINTS_PATH_SEARCH_RADIUS_RATIO = 0.15  # Search radius ตาม path (15% ของ frame diagonal)
FOOTPRINTS_PATH_EXTENSION_FRAMES = 5  # จำนวนเฟรมที่ขยาย path เมื่อไม่เจอ motion (5 เฟรม)

# Isolated motion path detection (หาก้อนเล็กๆ ที่มี path)
FOOTPRINTS_ISOLATED_PATH_DETECTION_ENABLED = True  # เปิด isolated path detection
FOOTPRINTS_ISOLATED_PATH_MIN_FRAMES = 3  # จำนวนเฟรมต่ำสุดสำหรับ path (3 เฟรม)
FOOTPRINTS_ISOLATED_PATH_MIN_QUALITY = 0.4  # Path quality ต่ำสุด (0.4 = ยืดหยุ่น)
FOOTPRINTS_ISOLATED_PATH_PRIORITY_BOOST = 1.5  # Priority boost สำหรับ isolated path (1.5x)

# Continuous tracking (ตามได้เรื่อยๆ)
FOOTPRINTS_CONTINUOUS_TRACKING_ENABLED = True  # เปิด continuous tracking
FOOTPRINTS_CONTINUOUS_PREDICTIVE_FRAMES = 10  # จำนวนเฟรมที่ predict เมื่อไม่เจอ motion (10 เฟรม)
FOOTPRINTS_CONTINUOUS_SEARCH_EXPANSION = 1.5  # ตัวคูณสำหรับขยาย search radius (1.5x)
FOOTPRINTS_CONTINUOUS_REACQUISITION_ENABLED = True  # ใช้ path เพื่อหา target ใหม่

# Path quality metrics (วัดคุณภาพ path)
FOOTPRINTS_PATH_QUALITY_METRICS_ENABLED = True  # เปิด path quality metrics
FOOTPRINTS_PATH_CONSISTENCY_WEIGHT = 0.3  # น้ำหนัก path consistency (30%)
FOOTPRINTS_PATH_SMOOTHNESS_WEIGHT = 0.3  # น้ำหนัก path smoothness (30%)
FOOTPRINTS_PATH_DIRECTION_WEIGHT = 0.2  # น้ำหนัก path direction stability (20%)
FOOTPRINTS_PATH_VELOCITY_WEIGHT = 0.2  # น้ำหนัก path velocity consistency (20%)
FOOTPRINTS_PATH_MIN_QUALITY = 0.3  # Path quality ต่ำสุดสำหรับ tracking (0.3 = ยืดหยุ่น)

# ============================================================================
# DUAL-MODE HIGH-SENSITIVITY MOTION DETECTION SETTINGS
# ============================================================================
# Search Mode: Full Frame + 1 Pixel Detection (หา target ใหม่)
FOOTPRINTS_SEARCH_MODE_ENABLED = True  # เปิด search mode
FOOTPRINTS_SEARCH_FULL_FRAME_INTERVAL = 5  # Full frame search ทุก N เฟรม (5 เฟรม = หาบ่อย)
FOOTPRINTS_SEARCH_MIN_MOTION_AREA = 1  # min_motion_area ใน search mode (1 pixel)
FOOTPRINTS_SEARCH_MOG2_VAR_THRESHOLD = 3  # MOG2 varThreshold ใน search mode (sensitive มาก)
FOOTPRINTS_SEARCH_MORPH_KERNEL_SIZE = 1  # Morphology kernel size ใน search mode (1x1 = ไม่ blur)
FOOTPRINTS_SEARCH_SKIP_DENSITY_CHECK = True  # ข้าม local density check ใน search mode
FOOTPRINTS_SEARCH_PRIORITY_BASED = True  # เน้นบริเวณที่มี motion มากก่อน

# Tracking Mode: Predicted ROI + 1 Pixel Detection (ติดตาม target)
FOOTPRINTS_TRACKING_MODE_ENABLED = True  # เปิด tracking mode
FOOTPRINTS_TRACKING_PREDICTIVE_ROI_ENABLED = True  # ใช้ predictive ROI
FOOTPRINTS_TRACKING_ROI_EXPAND_RATIO = 1.3  # ขยาย predicted ROI (1.3x = 30% padding)
FOOTPRINTS_TRACKING_ROI_MIN_SIZE = 50  # ขนาดต่ำสุดของ predicted ROI (pixels)
FOOTPRINTS_TRACKING_ROI_MAX_SIZE = 300  # ขนาดสูงสุดของ predicted ROI (pixels)
FOOTPRINTS_TRACKING_MIN_MOTION_AREA = 1  # min_motion_area ใน tracking mode (1 pixel)
FOOTPRINTS_TRACKING_MOG2_VAR_THRESHOLD = 3  # MOG2 varThreshold ใน tracking mode (sensitive มาก)
FOOTPRINTS_TRACKING_MORPH_KERNEL_SIZE = 1  # Morphology kernel size ใน tracking mode (1x1 = ไม่ blur)
FOOTPRINTS_TRACKING_SKIP_DENSITY_CHECK = True  # ข้าม local density check ใน tracking mode
FOOTPRINTS_TRACKING_SAFETY_CHECK_INTERVAL = 10  # Full frame safety check ทุก N เฟรม (10 เฟรม)

# Mode Switching Logic (สลับระหว่าง Search และ Tracking)
FOOTPRINTS_MODE_SWITCHING_ENABLED = True  # เปิด mode switching
FOOTPRINTS_SWITCH_TO_TRACKING_MIN_PATH = 3  # จำนวนเฟรม path ต่ำสุดก่อนสลับเป็น tracking (3 เฟรม)
FOOTPRINTS_SWITCH_TO_TRACKING_MIN_CONFIDENCE = 0.6  # Confidence ต่ำสุดก่อนสลับเป็น tracking (0.6)
FOOTPRINTS_SWITCH_TO_SEARCH_MISSED_FRAMES = 3  # จำนวนเฟรมที่ไม่ได้เจอก่อนสลับเป็น search (3 เฟรม)
FOOTPRINTS_SWITCH_TO_SEARCH_LOW_CONFIDENCE = 0.4  # Confidence ต่ำสุดก่อนสลับเป็น search (0.4)

# Resource Optimization Strategy
FOOTPRINTS_RESOURCE_OPTIMIZATION_MODE = 'adaptive'  # 'adaptive' = ปรับตาม mode
FOOTPRINTS_SEARCH_MODE_COST = 1.0  # Cost ของ search mode (1.0 = full cost)
FOOTPRINTS_TRACKING_MODE_COST = 0.1  # Cost ของ tracking mode (0.1 = 10% ของ search mode)

# Prediction Confidence & Fallback
FOOTPRINTS_PREDICTION_CONFIDENCE_THRESHOLD = 0.6  # Confidence ต่ำสุดสำหรับ tracking mode (0.6)
FOOTPRINTS_PREDICTION_CONFIDENCE_PATH_WEIGHT = 0.4  # น้ำหนัก path consistency (40%)
FOOTPRINTS_PREDICTION_CONFIDENCE_LENGTH_WEIGHT = 0.3  # น้ำหนัก path length (30%)
FOOTPRINTS_PREDICTION_CONFIDENCE_RECENT_WEIGHT = 0.3  # น้ำหนัก recent detections (30%)
FOOTPRINTS_PREDICTION_FALLBACK_EXPAND_RATIO = 2.0  # ขยาย predicted ROI เมื่อไม่เจอ (2.0x)
FOOTPRINTS_PREDICTION_FALLBACK_FRAMES = 3  # จำนวนเฟรมที่ไม่ได้เจอก่อน fallback (3 เฟรม)

# ============================================================================
# MULTI-TARGET SUPPORT SETTINGS (หาเจอทุก target พร้อมกัน)
# ============================================================================
# Multi-target detection ใน search mode
FOOTPRINTS_MULTI_TARGET_ENABLED = True  # เปิด multi-target support
FOOTPRINTS_MULTI_TARGET_SEARCH_ALL = True  # หาทุก targets ใน search mode (ไม่หยุดที่ target แรก)
FOOTPRINTS_TARGET_SEPARATION_DISTANCE_RATIO = 0.1  # ระยะห่างต่ำสุดสำหรับแยก targets (10% ของ frame diagonal)
FOOTPRINTS_TARGET_CLUSTERING_ENABLED = True  # เปิด target clustering
FOOTPRINTS_TARGET_CLUSTERING_DISTANCE_RATIO = 0.05  # ระยะห่างสูงสุดสำหรับ clustering (5% ของ frame diagonal)

# Multiple predicted ROIs (หลาย targets = หลาย ROIs)
FOOTPRINTS_MULTI_ROI_ENABLED = True  # เปิด multiple ROIs
FOOTPRINTS_MULTI_ROI_PARALLEL = True  # ประมวลผลหลาย ROIs พร้อมกัน
FOOTPRINTS_MULTI_ROI_OVERLAP_THRESHOLD = 0.3  # IOU threshold สำหรับ ROI overlap (30% = merge)
FOOTPRINTS_MULTI_ROI_PRIORITY_ENABLED = True  # ใช้ priority สำหรับ ROIs

# Parallel tracking สำหรับหลาย targets
FOOTPRINTS_PARALLEL_MULTI_TARGET_TRACKING = True  # เปิด parallel multi-target tracking
FOOTPRINTS_MULTI_TARGET_INDEPENDENT_TRACKING = True  # แต่ละ target ถูก track แยกกัน
FOOTPRINTS_MULTI_TARGET_SHARE_INFO = True  # Share information ระหว่าง trackers
FOOTPRINTS_MULTI_TARGET_RESOURCE_ALLOCATION = 'priority'  # 'priority' = แบ่งตาม priority, 'equal' = แบ่งเท่ากัน

# Target management (เพิ่ม/ลบ/อัปเดต targets)
FOOTPRINTS_TARGET_MANAGEMENT_ENABLED = True  # เปิด target management
FOOTPRINTS_TARGET_REGISTRATION_MIN_FRAMES = 2  # จำนวนเฟรมต่ำสุดก่อนลงทะเบียน target (2 เฟรม)
FOOTPRINTS_TARGET_REMOVAL_MISSED_FRAMES = 5  # จำนวนเฟรมที่ไม่ได้เจอก่อนลบ target (5 เฟรม)
FOOTPRINTS_TARGET_MERGING_ENABLED = True  # เปิด target merging
FOOTPRINTS_TARGET_MERGING_IOU_THRESHOLD = 0.5  # IOU threshold สำหรับ merge targets (50%)
FOOTPRINTS_TARGET_SPLITTING_ENABLED = True  # เปิด target splitting
FOOTPRINTS_TARGET_SPLITTING_SIZE_RATIO = 2.0  # ขนาด ROI เทียบกับ target size (2.0x = แยก)

# Target separation & clustering
FOOTPRINTS_TARGET_SEPARATION_ENABLED = True  # เปิด target separation
FOOTPRINTS_TARGET_SEPARATION_DISTANCE = 0.1  # ระยะห่างต่ำสุดสำหรับแยก targets (10% ของ frame diagonal)
FOOTPRINTS_TARGET_MOTION_CLUSTERING = True  # ใช้ motion direction สำหรับ clustering
FOOTPRINTS_TARGET_PATH_CLUSTERING = True  # ใช้ path direction สำหรับ clustering
FOOTPRINTS_TARGET_SIZE_SEPARATION = True  # ใช้ขนาดสำหรับแยก targets

# Multi-target resource optimization
FOOTPRINTS_MULTI_TARGET_RESOURCE_OPTIMIZATION = True  # เปิด resource optimization
FOOTPRINTS_MULTI_TARGET_MAX_TARGETS = 10  # จำนวน targets สูงสุด (10 targets)
FOOTPRINTS_MULTI_TARGET_PRIORITY_ALLOCATION = True  # แบ่งทรัพยากรตาม priority
FOOTPRINTS_MULTI_TARGET_LOW_PRIORITY_THROTTLE = 0.5  # ลดความถี่การประมวลผลสำหรับ low-priority targets (50%)
FOOTPRINTS_MULTI_TARGET_CULLING_ENABLED = True  # เปิด target culling
FOOTPRINTS_MULTI_TARGET_CULLING_PRIORITY_THRESHOLD = 0.2  # Priority ต่ำสุดก่อน cull (0.2)


# ============================================================================
# WEBSOCKET ANTIDRONE EXPORT
# ============================================================================
# เปิด/ปิด websocket export (ปิดเพื่อทดสอบในเครื่องโดยไม่กระทบ endpoint ภายนอก)
WS_EXPORT_ENABLED = True

# Endpoint template — {camera_id} จะถูกแทนด้วย camera_id ของกล้องนั้น
WS_EXPORT_URL_TEMPLATE = "wss://bbdc-api.thingsanalytic.com/ws/antidrone/{camera_id}/data/send"

# ความถี่ส่ง (วินาที): ส่งได้มากสุด 1 ครั้งต่อ interval นี้ (0.4 s ≈ 2.5 Hz)
WS_EXPORT_INTERVAL_SEC = 0.4

# Reconnect backoff (วินาที) — ถ้าเชื่อมต่อหลุดรอ delay นี้ก่อนลองใหม่, cap สูงสุดที่ WS_EXPORT_BACKOFF_MAX
WS_EXPORT_BACKOFF_INIT_SEC = 2.0
WS_EXPORT_BACKOFF_MAX_SEC = 30.0

# ============================================================================
# CAMERA GEO METADATA  (ใช้สำหรับ websocket payload)
# ตั้งค่า lat/lng/heading ตามตำแหน่งและทิศทางจริงของกล้องในสนาม
# heading_deg: ทิศที่กึ่งกลางภาพชี้ไป (องศาจากทิศเหนือ, 0=N, 90=E, 180=S, 270=W)
# camera_id  : ชื่อ id ที่จะใส่ใน websocket URL และ payload
# ============================================================================
CAMERA_GEO = {
    "cam1": {
        "camera_id": "CAM001",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam2": {
        "camera_id": "CAM002",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam3": {
        "camera_id": "CAM003",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam4": {
        "camera_id": "CAM004",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam5": {
        "camera_id": "CAM005",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam6": {
        "camera_id": "CAM006",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam7": {
        "camera_id": "CAM007",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
    "cam8": {
        "camera_id": "CAM008",
        "site_lat": 14.29083,
        "site_lng": 105.13274,
        "heading_deg": 118.5,
    },
    "cam9": {
        "camera_id": "CAM009",
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    },
}


def get_camera_geo(camera_name):
    """คืน geo metadata dict สำหรับกล้องที่ระบุ; fallback ถ้าไม่มีใน CAMERA_GEO"""
    default = {
        "camera_id": camera_name.upper(),
        "site_lat": 0.0,
        "site_lng": 0.0,
        "heading_deg": 0.0,
    }
    return CAMERA_GEO.get(camera_name, default)


# ============================================================================
# WEBSOCKET EFFECTOR EXPORT (Jetson #2 — 22_gun_aim_assist_vector)
# ============================================================================
EFFECTOR_WS_ENABLED = True
EFFECTOR_WS_LOG_PAYLOAD = False  # True = log payload แม้ WS ปิด (ทด local)

EFFECTOR_WS_URL_TEMPLATE = "wss://c2-api.thingsanalytic.com/ws/effector/{effector_id}/data/send"
EFFECTOR_WS_INTERVAL_SEC = 0.4
EFFECTOR_WS_BACKOFF_INIT_SEC = 2.0
EFFECTOR_WS_BACKOFF_MAX_SEC = 30.0

# จุดติดตั้งแขน — วางข้าง cam8 (sync กับ CAMERA_GEO["cam8"] / Jetson #1)
_EFFECTOR_SITE_LAT = 14.29083
_EFFECTOR_SITE_LNG = 105.13274
_CAM8_SURVEY_HEADING_DEG = 118.5

EFFECTOR_GEO = {
    "effector_id": "effector_id_1",
    "site_lat": _EFFECTOR_SITE_LAT,
    "site_lng": _EFFECTOR_SITE_LNG,
    "heading_deg": _CAM8_SURVEY_HEADING_DEG,
}

EFFECTOR_FOV_DEG = 30.0

# copy lat/lng/heading จาก CAMERA_GEO["cam8"] บน Jetson #1
CAM8_SURVEY_GEO = {
    "site_lat": _EFFECTOR_SITE_LAT,
    "site_lng": _EFFECTOR_SITE_LNG,
    "heading_deg": _CAM8_SURVEY_HEADING_DEG,
}


def get_effector_geo():
    """คืน effector geo metadata dict (site + effector_id + heading)."""
    return dict(EFFECTOR_GEO)


def get_cam8_survey_geo():
    """คืน cam8 survey site geo สำหรับคำนวณ bearing จาก UDP cue."""
    return dict(CAM8_SURVEY_GEO)


def get_cam8_fov_for_cue():
    """Horizontal FOV ของ cam8 จาก camera config (180° — ห้าม hardcode 60°)."""
    cfg = get_camera_config("cam8")
    return float(cfg.get("fov_horizontal", 180.0))


# ============================================================================
# RTMP DISPLAY STREAMING (Jetson #2 — 22_gun_aim_assist_vector)
# ============================================================================
# Disable temporarily: RTMP_STREAM_ENABLED=0 python3 22_gun_aim_assist_vector.py
# Debug ffmpeg errors: RTMP_STREAM_DEBUG=1 python3 22_gun_aim_assist_vector.py
_RTMP_STREAM_URL_DEFAULT = "rtmp://202.129.205.4/live/camnx2?user=admin&pass=Things22"
RTMP_STREAM_ENABLED = os.environ.get("RTMP_STREAM_ENABLED", "1") == "1"
RTMP_STREAM_URL = os.environ.get("RTMP_STREAM_URL", _RTMP_STREAM_URL_DEFAULT).strip()
RTMP_STREAM_FPS = float(os.environ.get("RTMP_STREAM_FPS", "10"))
RTMP_STREAM_BITRATE = os.environ.get("RTMP_STREAM_BITRATE", "2M")
RTMP_STREAM_CODEC = os.environ.get("RTMP_STREAM_CODEC", "auto")
RTMP_STREAM_RECONNECT_MAX = int(os.environ.get("RTMP_STREAM_RECONNECT_MAX", "5"))
RTMP_STREAM_RECONNECT_DELAY = float(os.environ.get("RTMP_STREAM_RECONNECT_DELAY", "5.0"))
RTMP_STREAM_DEBUG = os.environ.get("RTMP_STREAM_DEBUG", "0") == "1"

