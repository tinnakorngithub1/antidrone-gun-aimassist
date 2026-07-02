#!/bin/bash
# แชร์อินเทอร์เน็ตจากเครื่องนี้ (WiFi wlP1p1s0) ให้วง LAN 192.168.144.0/24 (eno1)
# ใช้กับ Jetson อีกเครื่องที่ต่อ WiFi ไม่ได้
#   เปิดตอนนี้:        sudo bash share_internet_lan.sh
#   ติดตั้งถาวร (boot): sudo bash share_internet_lan.sh install
#   ถอนถาวร:           sudo bash share_internet_lan.sh uninstall
#   ปิดชั่วคราว:        sudo bash share_internet_lan.sh off
set -e

WIFI_IF="wlP1p1s0"
LAN_IF="eno1"
LAN_NET="192.168.144.0/24"
UNIT="/etc/systemd/system/share-internet-lan.service"
SCRIPT_DST="/usr/local/sbin/share_internet_lan.sh"

if [ "$1" = "install" ]; then
    cp "$(readlink -f "$0")" "$SCRIPT_DST"
    chmod 755 "$SCRIPT_DST"
    cat > "$UNIT" <<'EOF'
[Unit]
Description=Share internet (WiFi) to LAN 192.168.144.0/24 for cam8 Jetson
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/share_internet_lan.sh
RemainAfterExit=yes
ExecStop=/usr/local/sbin/share_internet_lan.sh off

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now share-internet-lan.service
    echo ""
    echo "✅ ติดตั้งถาวรแล้ว — เปิดแชร์อัตโนมัติทุกครั้งที่บูต"
    systemctl status share-internet-lan.service --no-pager -l | head -5
    exit 0
fi

if [ "$1" = "uninstall" ]; then
    systemctl disable --now share-internet-lan.service 2>/dev/null || true
    rm -f "$UNIT" "$SCRIPT_DST"
    systemctl daemon-reload
    bash "$(readlink -f "$0")" off
    echo "ถอนการติดตั้งถาวรแล้ว"
    exit 0
fi

if [ "$1" = "off" ]; then
    sysctl -w net.ipv4.ip_forward=0
    iptables -t nat -D POSTROUTING -s "$LAN_NET" -o "$WIFI_IF" -j MASQUERADE 2>/dev/null || true
    iptables -D FORWARD -i "$LAN_IF" -o "$WIFI_IF" -j ACCEPT 2>/dev/null || true
    iptables -D FORWARD -i "$WIFI_IF" -o "$LAN_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true
    echo "ปิดการแชร์เน็ตแล้ว"
    exit 0
fi

# 1) เปิด IP forwarding
sysctl -w net.ipv4.ip_forward=1

# 2) NAT: ปลอมที่อยู่ขาออกของวง LAN เป็นของเครื่องนี้ (idempotent — เช็คก่อนเพิ่ม)
iptables -t nat -C POSTROUTING -s "$LAN_NET" -o "$WIFI_IF" -j MASQUERADE 2>/dev/null \
    || iptables -t nat -A POSTROUTING -s "$LAN_NET" -o "$WIFI_IF" -j MASQUERADE

# 3) อนุญาต forward ระหว่าง LAN ↔ WiFi (จำเป็นเพราะ docker ตั้ง FORWARD policy = DROP)
iptables -C FORWARD -i "$LAN_IF" -o "$WIFI_IF" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$LAN_IF" -o "$WIFI_IF" -j ACCEPT
iptables -C FORWARD -i "$WIFI_IF" -o "$LAN_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$WIFI_IF" -o "$LAN_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT

echo ""
echo "✅ แชร์เน็ตแล้ว: $LAN_NET ($LAN_IF) → $WIFI_IF"
echo ""
echo "ไปตั้งที่ Jetson อีกเครื่อง (ครั้งเดียว):"
echo "  sudo ip route replace default via 192.168.144.66"
echo "  echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf"
echo "ทดสอบ:  ping -c 2 8.8.8.8 && ping -c 2 github.com"
echo ""
echo "ทำให้ถาวร (รันเองทุกบูต): sudo bash share_internet_lan.sh install"
