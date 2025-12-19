import os, sys

import threading
import queue

from datetime import datetime
import zmq, time

import json
import gc
from pathlib import Path

# ===== ALGO ===== 
import pandas as pd

# IMPORT
sys.path.append("../")
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS, COLS_TO_DROP

class DetectorWorker(threading.Thread):
    def __init__(
        self,
        worker_id,
        task_queue,
        pipeline,
        flowFlushTransformer,
        alert_pub_addr="tcp://127.0.0.1:5570"
    ):
        super().__init__(daemon=True)

        # ===== ZMQ =====
        self.ctx = zmq.Context()
        self.alert_pub = self.ctx.socket(zmq.PUSH)
        self.alert_pub.connect(alert_pub_addr)

        # ===== WORKER =====
        self.worker_id = worker_id
        self.q = task_queue
        self.pipeline = pipeline

        self.flowFlushTransformer = FlowFlushTransformer(
            MINMAX_SCALER_PATH,
            STANDARD_SCALER_PATH,
            MINMAX_COLS,
            STANDARD_COLS,
            decimal_bin=6
        )
        
        # Lưu log
        self.save_dir = Path.cwd() / "app_logs"
        if not  Path.exists(self.save_dir):
            Path.mkdir(self.save_dir, exist_ok= True)
        
        self.running = True

    def run(self):
        print(f"[Detector-{self.worker_id}] started")

        try:
            while self.running:
                try:
                    df = self.q.get(timeout=1)

                    indexdf, X, _ = self.flowFlushTransformer.detect(df)
                    preds = self.pipeline.simple_predict(X)

                    # self.alert_callback(indexdf, preds, X)
                    self.save_alert(indexdf, preds)
                    self.q.task_done()

                    gc.collect()

                except queue.Empty:
                    continue

                except Exception as e:
                    print(f"[Detector-{self.worker_id}] error:", e)

        finally:
            # 🔥 ĐẢM BẢO ZMQ ĐƯỢC ĐÓNG
            self._close_zmq()
            print(f"[Detector-{self.worker_id}] exited cleanly")
        

    def stop(self):
        """
        Báo hiệu worker dừng lại.
        KHÔNG đóng socket ở đây.
        """
        self.running = False

    def _close_zmq(self):
        try:
            if self.alert_pub:
                self.alert_pub.close(linger=1000)  # cho gửi nốt (ms)
            if self.ctx:
                self.ctx.term()
        except Exception as e:
            print(f"[Detector-{self.worker_id}] ZMQ close error:", e)

    def alert_callback(self, indexdf, preds, X = None):
        print(f"[Detector-{self.worker_id}] pushing alert ...")

        for i, p in enumerate(preds):
            alert = {
                "flow_id": indexdf.at[i, COLS_TO_DROP[0]],
                "ts": indexdf.at[i, COLS_TO_DROP[1]],
                "label": p
            }
            self.alert_pub.send_json(alert)

        gc.collect()
        
    def save_alert_old(self, indexdf, preds, X=None):
        filePath = self.save_dir / f"{datetime.now().date()}.parquet"
        if not Path.exists(filePath):
            Path(filePath).touch()
        df = pd.DataFrame({
            f"{COLS_TO_DROP[0]}": indexdf[COLS_TO_DROP[0]].values,       # lấy toàn bộ cột nhanh hơn .at
            f"{COLS_TO_DROP[1]}":      indexdf[COLS_TO_DROP[1]].values,
            "Label":   preds                             # thêm thông tin worker nếu cần
        })
        
        df = df.astype(str)
        
        if filePath.exists():
            try:
                existing_df = pd.read_parquet(filePath)
                df = pd.concat ([existing_df, df], ignore_index= True)
                del existing_df
                gc.collect()
            except Exception as e:
                print(f"[Detector-{self.worker_id}] Reading Saved File Error:", e)
            
        df.to_parquet(filePath, index = False)
        print(f"[Detector-{self.worker_id}] Đã gửi ZMQ + lưu alerts → {filePath.name}")
        
        del df
        gc.collect()

    def save_alert(self, indexdf, preds, X= None):
        log_file = self.save_dir / f"{datetime.now().date()}.jsonl"
        
        if not Path.exists(log_file):
            Path(log_file).touch()
        records = []
        for i, p in enumerate(preds):
            records.append({
                f"{COLS_TO_DROP[0]}": str(indexdf.iloc[i][COLS_TO_DROP[0]]),
                f"{COLS_TO_DROP[1]}": str(indexdf.iloc[i][COLS_TO_DROP[1]]),
                "Label": str(p)
            })

        # 🔥 append-safe
        with open(log_file, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"[Detector-{self.worker_id}] Logged {len(records)} alerts → {log_file.name}")
        
        
        