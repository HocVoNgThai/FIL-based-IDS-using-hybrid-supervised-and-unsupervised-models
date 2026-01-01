#
# 1rd libs
import os, sys
import threading
import queue


# 3rd libs
import zmq, time
import sqlite3
import json
import gc
from datetime import datetime
from pathlib import Path

# ===== ALGO ===== 
import pandas as pd


# Local Import
from src.config.settings import settings

class LogWorker(threading.Thread):
    def __init__(self, worker_id, log_queue):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.save_dir = Path(settings.ALERTS_DIR)
        self.log_queue = log_queue
        
        self.save_dir = settings.ALERTS_DIR
        if not  Path.exists(self.save_dir):
            Path.mkdir(self.save_dir, exist_ok= True)
        
        # SQLITE
        self.conn = None 
        self.log_file = self.save_dir / f"{datetime.now().date()}.db"
        Path(self.log_file).touch(exist_ok=True)
        
        self.running = True
    def run(self):
        print(f"[Logger-{self.worker_id}] started")
        
        # LOAD LẦN ĐẦU
        self.reload_conn()
        try:
            while self.running:
                try:
                    df = self.log_queue.get(timeout=0.1)
                    
                    self.save_log(df)
                    self.log_queue.task_done()
                    gc.collect()

                except queue.Empty:
                    continue

                except Exception as e:
                    print(f"[Logger-{self.worker_id}] error:", e)
        
        finally:
            print(f"[Logger-{self.worker_id}] exited cleanly")
            
    def stop(self):
        """
        Báo hiệu worker dừng lại.
        KHÔNG đóng socket ở đây.
        """
        self.running = False
        print(f"[Logger-{self.worker_id}] SQLITE Conn Closed!")
        self.conn.close()
        
        return
    
    def reload_conn(self):
        if self.conn is not None:
            self.conn.close()
        
        # mở mới
        self.conn = sqlite3.connect(self.log_file, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL") # Chế độ ghi cực nhanh, cho phép đọc/ghi đồng thời
                
    def save_log(self, indexdf):
        log_file = self.save_dir / f"{datetime.now().date()}.db"
        # Nếu khác reload
        if log_file != self.log_file:
            Path(log_file).touch(exist_ok=True)
            self.log_file = log_file
            self.reload_conn()
        
        
        indexdf.to_sql("logs", self.conn, if_exists = 'append', index = False)
        
    # def save_log_old(self, indexdf):
    #     log_file = self.save_dir / f"{datetime.now().date()}.db"
        
    #     if not Path.exists(log_file):
    #         Path(log_file).touch()
            
    #     # def convert_time_to_int(x):
    #     #     dt = datetime.strptime(x, "%d/%m/%Y %H:%M:%S.%f")
    #     #     return int(dt.timestamp() * 1000)
        
    #     end_time_ms = int(time.time() * 1000)
        
    #     try:
    #         ts_numeric = pd.to_numeric(indexdf["Timestamp"], errors='coerce')
    #         indexdf["Pipetime"] = end_time_ms - ts_numeric
    #     except Exception as e:
    #         print(f"[Logger-Error] Tính Pipetime thất bại: {e}")
    #         indexdf["Pipetime"] = 0 
        
    #     indexdf.to_json(
    #         log_file, 
    #         orient='records', 
    #         lines=True,
    #         mode='a', 
    #         force_ascii=False
    #     )
    #     print(f"[Logger-{self.worker_id}] Logged {len(indexdf)} alerts → {log_file.name}")      