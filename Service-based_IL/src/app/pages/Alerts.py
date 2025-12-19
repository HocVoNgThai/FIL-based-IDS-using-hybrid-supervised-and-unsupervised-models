# pages/Alerts.py
import sys,os
import time

# 3rd libs
import streamlit as st
from pathlib import Path

#
from src.config.settings import settings
from core.Load_Alerts import Load_Data, COLS_TO_DROP

# CONST
HOME = Path.cwd()


# ======================
# MAIN
# ======================

def main():
    st.title("🚨 Realtime IDS Alerts Log")

    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = Load_Data(
            dir_in=Path(HOME/settings.ALERTS_DIR),
            last_update_time=0,
            refresh_interval=60,
            auto_refresh=True,
            file=None
        )

    data_loader = st.session_state.data_loader  
    
    # === Tùy chọn refresh trên giao diện ===
    col1, col2, col3 = st.columns(3)
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        data_loader.auto_refresh = auto_refresh
        
    with col2:
        if auto_refresh:
            interval = st.selectbox("Thời gian refresh", [5, 10, 30, 60], index=3)
            data_loader.refresh_interval = interval
            
    with col3:
        if st.button("🔄 Refresh Ngay"):
            data_loader.enable_reload_immediate = True

    st.markdown("---")

    # === Load và hiển thị dữ liệu ===
    df = data_loader.load_alerts(limit=100)

    if df.empty:
        st.info("Chưa có alert nào được ghi.")
        st.stop()

    # Metric
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giới hạn page - Alerts: ", len(df))
    col2.metric("Attack", len(df[df["Label"] != "BENIGN"]))
    col3.metric("Unknown", len(df[df.get('Label', []) =="UNKNOWN"]))
    col4.metric("Mới nhất", df.iloc[0][COLS_TO_DROP[1]])

    st.markdown("---")

    # Bảng
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            COLS_TO_DROP[1]: st.column_config.DatetimeColumn("Thời gian", format="DD/MM/YYYY HH:mm:ss"),
        }
    )

    # Auto rerun nếu bật
    if auto_refresh:
        time.sleep(1)
        st.rerun()