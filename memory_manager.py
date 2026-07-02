import os
import subprocess
import torch
import gc
import atexit
import time
import sys

# ลอง import psutil (ถ้าไม่มีให้ข้ามไป)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Warning: 'psutil' module not found. Auto-kill feature disabled.")


def kill_old_instances(script_name_to_kill=None):
    """
    ค้นหาและฆ่า Process เก่าที่รันไฟล์ Python ชื่อเดียวกัน
    arg: script_name_to_kill (str) -> ชื่อไฟล์ที่ต้องการฆ่า (เช่น 'fast_motion_sky.py')
    """
    if not PSUTIL_AVAILABLE:
        return

    # ถ้าไม่ระบุชื่อไฟล์มา ให้พยายามหาจาก process ปัจจุบัน (แต่อาจจะไม่แม่นเท่าระบุเอง)
    if script_name_to_kill is None:
        # ใช้ชื่อไฟล์ของคนเรียกฟังก์ชันนี้
        import __main__
        if hasattr(__main__, '__file__'):
            script_name_to_kill = os.path.basename(__main__.__file__)
        else:
            return

    current_pid = os.getpid()
    killed_count = 0

    print(f"🔍 Checking for ghost instances of '{script_name_to_kill}'...")

    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # เช็คว่าเป็น Python process
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    # เช็คว่า command line มีชื่อไฟล์เป้าหมายอยู่
                    if cmdline and any(script_name_to_kill in arg for arg in cmdline):
                        # ต้องไม่ใช่ตัวเราเอง (current process)
                        if proc.info['pid'] != current_pid:
                            print(f"🔪 Killing zombie process PID: {proc.info['pid']}")
                            proc.kill()
                            killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"⚠️ Auto-kill error: {e}")

    if killed_count > 0:
        print(f"✅ Killed {killed_count} old instances. Waiting for RAM release...")
        # พัก 3 วินาที เพื่อให้ OS คืน RAM และ Camera Handle
        time.sleep(3)
    else:
        print("✅ No ghost instances found. System clean.")


def clear_system_memory():
    """ล้างขยะในแรมระดับ System (ต้องรันโปรแกรมด้วย sudo)"""
    try:
        # print("🧹 Cleaning System Memory...") # ปิด print เพื่อลด noise
        # 1. ล้าง Cache ของระบบ
        subprocess.run("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True, check=True, stderr=subprocess.DEVNULL)
        # 2. ล้าง Swap
        subprocess.run("sudo swapoff -a && sudo swapon -a", shell=True, check=True, stderr=subprocess.DEVNULL)
        # 3. รีสตาร์ทเซอร์วิสกล้อง (Nvargus) - สำคัญสำหรับ Jetson
        subprocess.run("sudo systemctl restart nvargus-daemon", shell=True, check=True, stderr=subprocess.DEVNULL)
        print("✅ System Memory & Camera Service Cleared")
    except Exception as e:
        print(f"⚠️ System Memory Clear Warning: {e} (Run with sudo to fix)")

def clear_cuda_memory():
    """ล้างขยะในหน่วยความจำ GPU/CUDA"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print("✅ CUDA/GC Memory Cleared")
    except Exception as e:
        print(f"⚠️ CUDA Memory Clear Warning: {e}")

def init_memory_management():
    """เรียกใช้ตอนเริ่มรันและลงทะเบียนการล้างข้อมูลตอนปิดโปรแกรม"""
    clear_cuda_memory()
    # หมายเหตุ: clear_system_memory() อาจจะทำให้ start ช้าลงเล็กน้อย ถ้าไม่จำเป็นให้ปิดไว้
    # clear_system_memory()

    atexit.register(clear_cuda_memory)
    # atexit.register(clear_system_memory)

