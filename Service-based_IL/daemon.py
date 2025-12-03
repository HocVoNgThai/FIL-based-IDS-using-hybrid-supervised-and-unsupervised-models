import sys
import time
import queue
import threading
from run_daily import run_incremental_job   # tách logic ra function

# ===== Support Libs =====
import gc

def supports_color():
    return sys.stdout.isatty()

if supports_color():
    GREEN = "\033[0;32m"
    RESET = "\033[0m"
else:
    GREEN = RESET = ""

job_queue = queue.Queue()
# JOB_INTERVAL = 24 * 3600   # 1 ngày
JOB_INTERVAL = 60

def scheduler_thread():
    while True:
        print("\n[SCHEDULER] Adding daily incremental job into queue...")
        job_queue.put("incremental_job")
        time.sleep(JOB_INTERVAL)

def worker_thread():
    while True:
        job = job_queue.get()  # block until job exists
        print("[WORKER] Starting job: ", job)
        try:
            run_incremental_job()
            gc.collect()
            
        except Exception as e:
            print("[WORKER] ERROR:", e)
            
        print("[WORKER] Job finished.")
        gc.collect()
        job_queue.task_done()

if __name__ == "__main__":
    print("[DAEMON] Starting job scheduler service...")

    # Start background threads
    t1 = threading.Thread(target=scheduler_thread, daemon=True)
    t2 = threading.Thread(target=worker_thread, daemon=True)
    
    t1.start()
    t2.start()

    # Keep service alive
    while True:
        time.sleep(1)
