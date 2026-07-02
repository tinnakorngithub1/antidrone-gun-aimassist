#!/usr/bin/env python3
"""
claude_link.py — สายคุยตรง Claude↔Claude ระหว่างสองเครื่องในวง LAN (UDP)
ใช้คุยสด ๆ ระหว่าง dev สองทีม (เร็วกว่ารอ git push/pull)

ใช้งาน:
  ฟังข้อความเข้า:   python3 claude_link.py listen [--port 5766]
  ส่งข้อความ:       python3 claude_link.py send --to <IP> [--port 5766] "ข้อความ"

Wire format (UDP datagram, UTF-8, JSON บรรทัดเดียว):
  {"from": "<hostname>", "role": "arm|cam8", "msg": "...", "ts": <epoch>}
receiver ทนต่อ payload ที่ไม่ใช่ JSON — ถ้า decode ไม่ได้จะพิมพ์ raw text
ทุกข้อความที่รับได้ append ลง claude_link_inbox.log (ให้ Claude อ่านย้อนหลังได้)
"""
import argparse
import json
import socket
import sys
import time

DEFAULT_PORT = 5766
ROLE = "arm"  # เครื่องนี้คือฝั่งแขน
INBOX_LOG = "claude_link_inbox.log"


def _hostname():
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def do_listen(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    print(f"[claude_link] listening UDP :{port} (role={ROLE}, host={_hostname()})",
          flush=True)
    while True:
        try:
            data, addr = s.recvfrom(65535)
        except KeyboardInterrupt:
            print("\n[claude_link] stopped", flush=True)
            break
        except Exception as e:
            print(f"[claude_link] recv error: {e}", flush=True)
            continue
        raw = data.decode("utf-8", errors="replace")
        ts_local = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            obj = json.loads(raw)
            frm = obj.get("from", "?")
            role = obj.get("role", "?")
            msg = obj.get("msg", "")
            line = f"[{ts_local}] {addr[0]} ({role}/{frm}): {msg}"
        except Exception:
            line = f"[{ts_local}] {addr[0]} (raw): {raw}"
        print(line, flush=True)
        try:
            with open(INBOX_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def do_send(to, port, msg):
    payload = {
        "from": _hostname(),
        "role": ROLE,
        "msg": msg,
        "ts": time.time(),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(data, (to, port))
        print(f"[claude_link] sent -> {to}:{port}  msg={msg!r}", flush=True)
    except Exception as e:
        print(f"[claude_link] send error: {e}", flush=True)
        sys.exit(1)
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description="Claude↔Claude LAN link")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_listen = sub.add_parser("listen", help="ฟังข้อความเข้า")
    p_listen.add_argument("--port", type=int, default=DEFAULT_PORT)

    p_send = sub.add_parser("send", help="ส่งข้อความ")
    p_send.add_argument("--to", required=True, help="IP ปลายทาง")
    p_send.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_send.add_argument("message", help="ข้อความ")

    args = ap.parse_args()
    if args.cmd == "listen":
        do_listen(args.port)
    elif args.cmd == "send":
        do_send(args.to, args.port, args.message)


if __name__ == "__main__":
    main()
