import sys, os
import zmq

# TIME
import time # Cần thiết cho việc kiểm tra RCVTIMEO
from datetime import datetime

# THR
import queue
import threading
import signal
import gc

# IO
import json
import pandas as pd
from pathlib import Path

# ========= TOÀN CỤC =========
CURR_DIR = Path.cwd()
# Kích thước timeout (miligiây)
TIMEOUT_MS = 500
BATCH_SIZE = 500000

# ========= IMPORT FROM MINE =====
from src.config.settings import settings
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, COLS_TO_DROP, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS, SAMPLE_COLS_TO_REMOVE, DO_NOT_TOUCH_COLS, DECIMAL_BIN

from src.Worker.DetectorWorker import DetectorWorker
from src.Worker.FlushWorker import FlushWorker

from src.Components.Detector import Detectors_Pipeline
from src.Components.Models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.Components.Manager import Manager


# ===========================
# CLASS 
# ===========================
class FlowZmqServer:
    def __init__( self, 
                 bind_addr="tcp://*:5555", 
                 detect_batch_size=10, 
                 flush_batch_size=1000, 
                 n_workers= 2, 
                 output_dir="flows_parquet"):
        
        # self.bind_addr = bind_addr
        
        # BATCH SIZE
        self.detect_batch_size = detect_batch_size
        self.flush_batch_size = flush_batch_size
        
        # OUT
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SOCKET
        self.ctx = zmq.Context() #zmq.Content.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.bind(bind_addr)
        
        # Điều này khiến sock.recv() thoát ra sau 500ms nếu không có tin nhắn
        self.sock.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)

        self.flowFlushTransformer = FlowFlushTransformer(
            MINMAX_SCALER_PATH, STANDARD_SCALER_PATH, MINMAX_COLS, STANDARD_COLS, decimal_bin=6
        )
        
        self.curr_index_parquet = 0
        
        # BUFFER 
        self.flush_buffer = []
        self.detect_buffer = []
        
        # HEADER - RUNNING STATE
        self.header = None
        self.running = True

        # QUEUE
        self.detect_queue = queue.Queue(maxsize = 5000)
        self.flush_queue = queue.Queue(maxsize=100)
        
        # PIPE
        ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.75)
        self.mgr = Manager(settings.MODEL_DIR)
        self.mgr.load_models([xgb, ocsvm, ae])
        self.pipelineDetector = Detectors_Pipeline(xgb= xgb, ocsvm=ocsvm , ae= ae)
        
        # WORKERS INITIALIZATION
        self.detect_workers = [
            DetectorWorker(i, self.detect_queue, self.pipelineDetector, self.flowFlushTransformer) #, self.alert_callback
            for i in range(n_workers)
        ]
        
        self.flush_workers = [
            FlushWorker(i, self.flush_queue) for i in range(1)
        ]
        
        #
        print(f"[PY] ZMQ listening on {bind_addr}")

    # =========================
    # Core loop
    # =========================
    def run(self):
        print("[PY] Server running, press Ctrl+C to stop...")
        
        # START WORKER THREAD
        self.start_workers()
        
        # LOOP
        while self.running:
            try:
                msg = self.sock.recv(flags=0)
                self.handle_message(msg)
                
            except zmq.error.Again:
                # Xảy ra khi timeout (RCVTIMEO) hết hạn và không có tin nhắn. 
                # Đây là cách chúng ta cho phép vòng lặp kiểm tra self.running.
                continue 
                
            except zmq.ZMQError as e:
                # Xảy ra khi socket bị đóng từ bên ngoài (ví dụ: ZContext.term())
                # Hoặc khi có lỗi ZMQ khác
                if self.running: # Nếu lỗi không phải do lệnh tắt, log nó
                     print(f"[ERROR] ZMQ Error: {e}")
                break
                
            except Exception as e:
                print(f"[ERROR] General Error: {e}")
                self.running = False
        
        self.detect()
        self.flush()
        self.close()

    # =========================
    # Message handler
    # =========================
    def handle_message(self, msg: bytes):
        # Giả định dữ liệu là JSON chứa header và flowDump
        data = json.loads(msg.decode("utf-8"))

        # # GỢI Ý: Kiểm tra xem message có phải là tin nhắn control để tắt server không
        # if "command" in data and data["command"] == "shutdown":
        #     self.stop()
        #     return
        
        # Logic trích xuất Flow
        if self.header is None and "header" in data:
            self.header = data["header"].split(",")

        # Dump
        if "flowDump" not in data:
            return
        
        flow_row = data["flowDump"].split(",")
        
        # BUFFER DETECT
        self.detect_buffer.append(flow_row)
        if len(self.detect_buffer) >= self.detect_batch_size:
            self.detect()
        
        # BUFFER FLUSH
        self.flush_buffer.append(flow_row)
        if len(self.flush_buffer) >= self.flush_batch_size:
            self.flush()

    # =========================
    # WORKERS
    # =========================
    def start_workers(self):
        for w in self.detect_workers:
            w.start()
        for w in self.flush_workers:
            w.start()
            
    # =========================
    # FLUSH WORKER
    # =========================
    def flush_old(self):
        if not self.flush_buffer or self.header is None:
            return

        # Đảm bảo header có số lượng cột khớp với buffer
        try:
            df = pd.DataFrame(self.flush_buffer, columns=self.header)
        except ValueError as e:
            print(f"[ERROR] Column/Data mismatch during DataFrame creation: {e}")
            self.flush_buffer.clear()
            return
        
        fname = self.flowFlushTransformer.flush(df)
        
        if fname != None:
            print(f"[PY] ✔ Flushed {len(self.flush_buffer)} flows → {fname}")
        
        self.flush_buffer.clear()
    
    def flush(self):
        if not self.flush_buffer or self.header is None:
            return

        # Đảm bảo header có số lượng cột khớp với buffer
        try:
            df = pd.DataFrame(self.flush_buffer, columns=self.header)
        except ValueError as e:
            print(f"[ERROR] Column/Data mismatch during DataFrame creation: {e}")
            self.flush_buffer.clear()
            return
        
        self.flush_queue.put(df)
        print(f"[PY] ✔ Push into Flush Queue {len(self.flush_buffer)} flows")
        self.flush_buffer.clear()

    
    # =========================
    # DETECT WORKER
    # =========================         
    def detect(self):
        if not self.detect_buffer or self.header is None:
            return

        # Đảm bảo header có số lượng cột khớp với buffer
        try:
            df = pd.DataFrame(self.detect_buffer, columns=self.header)
        except ValueError as e:
            print(f"[ERROR] Column/Data mismatch during DataFrame creation: {e}")
            self.detect_buffer.clear()
            return

        self.detect_queue.put(df)
        print(f"[PY] ✔ Push into Queue {len(self.detect_buffer)} flows")
        self.detect_buffer.clear()
    
    # =========================
    # Alert Callback
    # =========================
    # def alert_callback(self, alert):
    #     """
    #     Gửi cho Streamlit (file, socket, redis, memory, ...)
    #     """
    #     print(f"[ALERT] {alert}")
        
    # =========================
    # Graceful shutdown
    # =========================
    def stop(self, *_):
        if self.running:
            print("\n[PY] Shutdown signal received. Starting graceful termination...")
            self.running = False
            # Nếu ZMQ.term() được gọi, nó sẽ hủy bỏ sock.recv()
            # Tùy thuộc vào phiên bản ZMQ/Python, có thể cần gọi ctx.term()
        
    def close(self):
        
        print("[PY] Closing worker, zmq, ...")
        # for _ in self.flush_workers:
        #     self.flush_queue.put(pd.DataFrame(None))

        # for _ in self.detect_workers:
        #     self.detect_queue.put(pd.DataFrame(None))
        
        # chờ queue hết
        # self.detect_queue.join()
        # self.flush_queue.join()

        # CLOSE WORKER
        # self.detect_queue.clear()
        # self.flush_queue.clear()
            
        for worker in self.detect_workers:
            worker.stop()
        for worker in self.flush_workers:
            worker.stop()
            
        for w in self.detect_workers:
            w.join(timeout=5)

        for w in self.flush_workers:
            w.join(timeout=5)
        
        print("[PY] All detector workers stopped")
        
        # CLOSE SOCKET
        try:
            self.sock.close(linger=1000) #(mili sec)
        except ValueError as e:
            print("[PY] ZMQ Socket close error: ", e)
            
        try:
            # ctx.term() sẽ hủy bất kỳ tác vụ chặn nào (như sock.recv())
            self.ctx.term()
        except ValueError as e:
            print("[PY] ZMQ Content terminate error: ", e)
            
        gc.collect()
        print("[PY] ZMQ closed")
        return

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    server = FlowZmqServer(
        bind_addr="tcp://*:5555",
        detect_batch_size=10,
        flush_batch_size= 2000,
        n_workers=2
    )

    # Đăng ký hàm stop cho các tín hiệu
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)

    server.run()