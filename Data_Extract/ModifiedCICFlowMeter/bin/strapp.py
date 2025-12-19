# dashboard/app.py
import streamlit as st
import zmq
import time
# Không cần import threading nữa

import atexit
import gc


# --- Cấu hình Trang ---
st.set_page_config(layout="wide")
st.title("🚨 Realtime IDS Alerts")

TIMEOUT_MS = 100
# Giảm TIMEOUT_MS xuống mức nhỏ hơn để việc Polling hiệu quả hơn
# Nếu quá cao (ví dụ: 500ms), giao diện có thể bị trì hoãn

# --- 1. Khởi tạo Trạng thái và ZMQ Socket An toàn ---

# Sử dụng st.session_state để lưu trữ alerts và socket
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'zmq_ctx' not in st.session_state:
    st.session_state.zmq_ctx = zmq.Context() # .instance()

if 'zmq_sub' not in st.session_state:
    sub = st.session_state.zmq_ctx.socket(zmq.PULL)
    sub.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
    sub.bind("tcp://*:5570")
    st.session_state.zmq_sub = sub
    

# --- 2. Hàm Lắng nghe Non-Blocking trong Luồng Chính ---
def check_for_alert():
    """Kiểm tra alert mới mà không làm Blocking luồng chính."""
    new_alerts_count = 0
    while True:
        try:
            # Nhận tin nhắn. Nếu hết RCVTIMEO, nó sẽ ném ra zmq.Again
            # flags=0 (blocking) kết hợp với RCVTIMEO tạo ra hành vi timeout
            msg = sub.recv_json(flags=0) 
            st.session_state.alerts.append(msg)
            new_alerts_count += 1
        except zmq.Again:
            # Không có tin nhắn trong thời gian timeout, thoát vòng lặp nhận
            break
        except zmq.ZMQError as e:
            # Xử lý lỗi khác của ZMQ
            print(f"Lỗi ZMQ khi nhận: {e}")
            break
    
    if new_alerts_count > 0:
        print(f"[STREAMLIT] Nhận thành công {new_alerts_count} tin nhắn.")
        return True # Đã cập nhật trạng thái

    return False

# CLEAN up zmq
def cleanup_zmq():
    print("[ZMQ] Cleaning up...")

    if 'zmq_sub' in st.session_state:
        try:
            st.session_state.zmq_sub.close(linger=0) # nhận nốt 100
            print("[ZMQ] Socket closed")
        except Exception as e:
            print("[ZMQ] Socket close error:", e)
            
    if 'zmq_ctx' in st.session_state:
        try:
            st.session_state.zmq_ctx.term()
            print("[ZMQ] Context terminated")
        except Exception as e:
            print("[ZMQ] Context term error:", e)
            
atexit.register(cleanup_zmq)


# --- 3. Vòng lặp Cập nhật Streamlit (Polling) ---
placeholder = st.empty()

if 'running' not in st.session_state:
        st.session_state.running = True
        
if __name__ == "__main__":
    while st.session_state.running:
        check_for_alert()

        with placeholder.container():
            total_alerts = len(st.session_state.alerts)
            st.metric("Total Alerts", total_alerts)

            display_alerts = st.session_state.alerts[-100:][::-1]
            if display_alerts:
                st.dataframe(display_alerts, use_container_width=True)
            else:
                st.info("Đang chờ nhận alert...")

        time.sleep(0.2)
        gc.collect()
        # st.rerun()
    
    st.session_state.alerts.clear()
    cleanup_zmq()
    gc.collect()
    st.stop()
    
