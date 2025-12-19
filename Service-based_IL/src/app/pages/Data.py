# pages/Data.py
import os, sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st


#
import time
from datetime import datetime
import gc

#
from src.config.settings import settings
from src.app.core.Data import Data_Labeling
from src.app.config.func_convert import round_decimal
from src.app.config.config import COLS_TO_DROP, MINMAX_COLS, STANDARD_COLS, DECIMAL_BIN

# CONFIG ST
st.set_page_config(layout="wide")

# ===== CONST =====
HOME = Path.cwd()
LABEL_OPTIONS = ["Benign", "Unknown", "DDoS", "DoS", "Reconnaisance", "MITM_ArpSpoofing", "DNS_Spoofing", "NeedManualLabel"]

def main():
    st.title("🧠 Data Management & Labeling")
    # ===== SESSION STATE =====
    if "current_file" not in st.session_state:
        st.session_state.current_file = Path(HOME / settings.DATA_DIR / f"{datetime.now().date()}" / "batch_0.parquet")

    if "jsonl_file" not in st.session_state:
        st.session_state.jsonl_file = Path(HOME /settings.ALERTS_DIR / f"{datetime.now().date()}.jsonl")
        
    if "dataLabeling" not in st.session_state:
        st.session_state.dataLabeling = Data_Labeling(st.session_state.current_file)

    if "df" not in st.session_state:
        st.session_state.df = None
        
    if "alerts_df_view" not in st.session_state:
        st.session_state.alerts_df_view = None

    if "selected_flow_id" not in st.session_state:
        st.session_state.selected_flow_id = None
        
    if "overwrite_label" not in st.session_state:
        st.session_state.overwrite_label = True
    # ---- LLOAD DATA----
    if st.session_state.dataLabeling:
        dataLabeling = st.session_state.dataLabeling

    # Đảm bảo không load lại nhiều lần
    if st.session_state.df is None and Path.exists(st.session_state.current_file):  
        st.session_state.df = dataLabeling.load_data()
        st.success("Data Loaded!")


    # =========================================================
    # SECTION A — ALERT DATA (PARQUET)
    # =========================================================
    st.subheader(f"📦 Preprocessed Dataset (Parquet, Csv) | File hiện tại: {st.session_state.current_file.name}")
    colA1, colA2, colA3 = st.columns([4, 1, 1])

    with colA1:
        # choosed_file = st.file_uploader(
        #     "📂 Chọn file alerts (.parquet)",
        #     type=["parquet"],
        #     key="alerts_uploader"
        # )
        choosed_file = st.text_input(
            "📂 Nhập đường dẫn file Parquet/Csv",
            value=st.session_state.current_file,
            placeholder="Ví dụ: C:/data/alerts.parquet hoặc /home/user/app_logs/batch_1.parquet",
            key="file_path_input"
        )
        # choosed_file = choosed_file.replace('\\', '/')
        # st.text(choosed_file)
        
    with colA2:
        load_button = st.button("🔄 Load File", use_container_width=True, type="primary")

        # --- Chỉ load khi người dùng bấm nút Load ---
        if load_button:
            if not choosed_file:
                st.error("Vui lòng nhập đường dẫn file!")
                        
            elif choosed_file is not None and dataLabeling.check_dir(Path(choosed_file)):
                if Path(choosed_file) != st.session_state.current_file:
                    st.session_state.current_file = Path(choosed_file)
                    st.session_state.dataLabeling = Data_Labeling(st.session_state.current_file)
                    st.success(f"Đã chọn file: {st.session_state.current_file.name}")
                else:
                    st.success("File đã load từ lần nhấp gần nhất!")
                
            time.sleep(0.2)
            st.rerun()
        
    with colA3:
        save_mode = st.radio(
            "Chế độ lưu",
            ["Overwrite", "Save As"],
            horizontal=True
        )


    # ---- SHOW & EDIT ----
    if st.session_state.df is not None:
        df = st.session_state.df
        # ROUND PHẦN THẬP PHÂN
        df = round_decimal(df, min_max_cols= MINMAX_COLS, standard_cols= STANDARD_COLS, minmax_decimal_bin=8, standard_decimal_bin=DECIMAL_BIN)
        
        st.subheader("📝 Chỉnh sửa Label")
        df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Label": st.column_config.SelectboxColumn(
                    "Label",
                    options=LABEL_OPTIONS,
                    default="NeedManualLabel",
                    required=True
                )
            }
        )

        st.session_state.df = df

        # ---- LABEL MAPPING ----
        with st.expander("🔁 Mapping Label hàng loạt"):
            colM1, colM2, colM3 = st.columns(3)
            with colM1:
                from_label = st.selectbox("From", LABEL_OPTIONS, key="map_from")
            with colM2:
                to_label = st.selectbox("To", LABEL_OPTIONS, key="map_to")
            with colM3:
                if st.button("Apply Mapping"):
                    mask = df["Label"] == from_label
                    df.loc[mask, "Label"] = to_label
                    st.success(f"Mapped {mask.sum()} samples")

        # ---- SAVE ----
        # ---- Save Controls ----
        st.markdown("### 💾 Save")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save (Overwrite)"):
                fname = dataLabeling.save_data(df)
                if fname is not None:
                    st.success(f"✔ Saved (overwrite) to {fname}")

        with col2:
            fname = st.text_input(
                "Save As (filename.parquet)",
                value=f"labeled_.parquet?"
            )

            if st.button("💾 Save As"):
                fname = dataLabeling.save_data(df, fname)
                if fname is not None:
                    st.success(f"Saved → {fname}")

    # =========================================================
    # SECTION B — RAW FLOW DATA (JSONL)
    # =========================================================
    st.markdown("---")
    st.subheader(f"📜 B. Raw Flow Logs (JSONL) | File hiện tại: {st.session_state.jsonl_file.name}")

    if st.session_state.alerts_df_view is None and Path.exists(st.session_state.jsonl_file):
        alerts_df = pd.read_json(st.session_state.jsonl_file, lines= True, nrows=None)
                
        # VIEWED
        df_flow_ids = set(st.session_state.df["Flow ID"])
        st.session_state.alerts_df_view = alerts_df[alerts_df["Flow ID"].isin(df_flow_ids)]
        st.success("Alerts Loaded!")
        
        del alerts_df
        gc.collect()
        
    colB1, colB2, colB3= st.columns([4, 1, 1]) 

    with colB1:
        
        jsonl_file = st.text_input(
            "📂 Nhập đường dẫn file Parquet/Csv",
            value=st.session_state.jsonl_file,  
            placeholder="Ví dụ: C:/data/alerts.parquet hoặc /home/user/app_logs/batch_1.parquet",
            key="json_file"
        )

    
    with colB2:
        load_jsonl_button = st.button("🔄 Load File", use_container_width=True, type="primary", key="jsonloadButton")

        # --- Chỉ load khi người dùng bấm nút Load ---
        if load_jsonl_button:
            if not jsonl_file:
                st.error("Vui lòng nhập đường dẫn file!")
                        
            elif jsonl_file is not None and Path.exists(Path(jsonl_file)):
                # if Path(jsonl_file) != st.session_state.jsonl_file:
                    
                st.session_state.jsonl_file = Path(jsonl_file)
                alerts_df = pd.read_json(st.session_state.jsonl_file, lines= True, nrows=None)
                
                # VIEWED
                df_flow_ids = set(st.session_state.df["Flow ID"])
                st.session_state.alerts_df_view = alerts_df[alerts_df["Flow ID"].isin(df_flow_ids)]
                
                st.success(f"Đã chọn file: {st.session_state.current_file.name}")
                del df_flow_ids, alerts_df
                gc.collect()
                
            else:
                st.error("Không thể tìm thấy file jsonl!")
            time.sleep(0.2)
            st.rerun()
    
    # Không thay đổi thì cứu thế mà hiển thị/sử dụng
    alerts_df_view = st.session_state.alerts_df_view
    
    with colB3:
        if st.button("🚀 Apply Mapping từ Alerts", use_container_width= True, type = "primary", key="LabelMapping"):
            if alerts_df_view is None:
                st.error("Không flow nào trùng id hoặc chưa load được!")
            elif "Flow ID" not in df.columns or "Flow ID" not in alerts_df_view.columns:
                st.error("Không tìm thấy cột Flow ID")
            elif "Label" not in alerts_df_view.columns:
                st.error("alerts_df không có cột Label")
            else:
                # --- Build map ---
                flowid_to_label = (
                    alerts_df_view
                    .dropna(subset=["Flow ID", "Label"])
                    .drop_duplicates("Flow ID")
                    .set_index("Flow ID")["Label"]
                    .to_dict()
                )

                # --- Mask ---
                if st.session_state.overwrite_label:
                    mask = df["Flow ID"].isin(flowid_to_label)
                else:
                    mask = (
                        df["Flow ID"].isin(flowid_to_label)
                        & df["Label"].isin(["NeedManualLabel", "Unknown"])
                    )

                # --- Apply ---
                before = mask.sum()
                df.loc[mask, "Label"] = df.loc[mask, "Flow ID"].map(flowid_to_label)

                st.session_state.df = df
                st.success(f"✔ Mapped {before} flows từ alerts")
                
                time.sleep(1)
                st.rerun()
                
                
        overwrite_label = st.checkbox(
            "Ghi đè label đã gán thủ công",
            value=True
        )
        
        st.session_state.overwrite_label = overwrite_label

    if alerts_df_view is not None:
        st.caption(
            f"Khớp Flow ID với df: {len(alerts_df_view)} |"
            # f"JSONL gốc: {len(alerts_df)}"
        )
    if alerts_df_view is not None:
        st.dataframe(
            alerts_df_view,
            use_container_width=True,
            hide_index=True
        )