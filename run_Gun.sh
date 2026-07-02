#!/bin/bash -i
set -e

# เข้าโฟลเดอร์โปรเจกต์ (ไม่ผูกชื่อ path — rename โฟลเดอร์แล้วรันได้)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# โหลด Environment (Jetson: ~/antidrone_v2)
if [[ -f "${HOME}/antidrone_v2/bin/activate" ]]; then
  source "${HOME}/antidrone_v2/bin/activate"
fi

# รันโค้ด Python
python3 "$SCRIPT_DIR/22_gun_aim_assist_vector.py"
