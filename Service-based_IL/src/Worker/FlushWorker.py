# STANDARD LIBS
import os, sys
import threading
import queue
import time

# 3rd LIBS
import zmq
import gc


# MINE
sys.path.append("../")
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS

class FlushWorker(threading.Thread):
    def __init__(self, worker_id, task_queue):
        super().__init__(daemon=True)

        # WORKER - QUEUE
        self.worker_id = worker_id
        self.q = task_queue

        # FlowTransform
        self.flowFlushTransformer = FlowFlushTransformer(
            MINMAX_SCALER_PATH, STANDARD_SCALER_PATH, MINMAX_COLS, STANDARD_COLS, decimal_bin=6
        )
                
        self.running = True

    def run(self):
        print(f"[Flush-{self.worker_id}] started")
        try:
            while self.running: # self.running
                try:
                    self.flush()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Flush-{self.worker_id}] error:", e)
            
            print(f"[Flush-{self.worker_id}] Exiting Signal received! Flushing data left..." )
            while not self.q.empty():
                try:
                    self.flush()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Flush-{self.worker_id}] error:", e)

        finally:
            print(f"[Flush-{self.worker_id}] Exit cleanly!")
    
    def flush(self):
        df = self.q.get(timeout=1)
        # if df.empty:
        #     self.q.task_done()
        #     break

        fname = self.flowFlushTransformer.flush(df)
        
        if fname != None:
            print(f"[Flush-{self.worker_id}] ✔ Flushed {len(df)} flows → {fname}")

        self.q.task_done()
        gc.collect()


    def stop(self):
        self.running = False