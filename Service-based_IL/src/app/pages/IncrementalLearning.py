# src/app/pages/IncrementalLearning.py
# STANDARDS LIBS
import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import time

# 3rd import

# Local import
from src.config.settings import settings
from src.config.incremental_config import incremental_settings, IncrementalSettings

# CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path("./src/config/.config_incremental")

def main():    
    st.title("🧠 Incremental Learning Configuration")

    # if 'il_config_fields' not in st.session_state:
    il_config_fields = IncrementalSettings.model_fields
            
    # Layout
    cfg_col, view_col = st.columns([1.5, 1])
    
    with cfg_col:
        st.markdown("## ⚙️ Incremental Config")
        
        with st.form("il_settings"):
            new_values = {}
            
            with st.container(border=True):
                for field_name, field_info in il_config_fields.items():
                    current_value = getattr(incremental_settings, field_name)
                    if isinstance(current_value, int) and field_name in ["MIN_INTERVAL_SEC"]:
                        new_val = st.number_input(field_name, value=current_value, step=1800, min_value= 600)
                    elif isinstance(current_value, int) and field_name in ["IL_FIXED_MEM_BUDGET"]:
                        new_val = st.number_input("Fixed Number of Samples In IL", key = field_name, value = current_value)
                    elif isinstance(current_value, Path):
                        new_val = st.text_input(
                            field_name,
                            value=str(current_value), #str(current_value.resolve() if isinstance(current_value, Path) else current_value),
                            help=field_info.description or ""
                        )
                    else:
                        new_val = current_value
                    new_values[field_name] = new_val
                    
            with st.container(border=True):
                st.markdown("### Current Label Samples")
                st.json(incremental_settings.IL_LABEL())

            with st.container(border=True):
                st.markdown("### 📦 Dataset")
                new_values["initial_train_ratio"] = st.slider(
                    "Initial Train Ratio",
                    0.1, 0.8, step=0.05,
                    key="initial_train_ratio",
                    value = getattr(incremental_settings, "initial_train_ratio")
                    # on_change=
                )
    
                new_values["increment_batch_size"] = st.number_input(
                    "Increment Batch Size",
                    min_value=100, max_value=50000, step=100,
                    key="increment_batch_size",
                    value= getattr(incremental_settings, "increment_batch_size")
                    # on_change=save_config_to_json
                )

            with st.container(border=True):
                st.markdown("### 🧠 Model (XGBoost)")
                new_values["learning_rate"] = st.slider(
                    "Learning Rate",
                    0.001, 0.2, step=0.001,
                    format="%.3f",
                    key="learning_rate",
                    value = getattr(incremental_settings, "learning_rate")
                )
                
                
                new_values["max_depth"] = st.slider(
                    "Max Depth",
                    3, 12,
                    key="max_depth",
                    value = getattr(incremental_settings, "max_depth")
                    # on_change=save_config_to_json
                )
                
                new_values["trees_per_step"] = st.slider(
                    "Trees per Step",
                    1, 100,
                    key="trees_per_step",
                    value= getattr(incremental_settings, "trees_per_step")
                )
                
                

            with st.container(border=True):
                st.markdown("### 🧬 Anti-Catastrophic Forgetting")
                new_values["replay_buffer_size"] = st.number_input(
                    "Replay Buffer Size",
                    min_value=0, max_value=100000, step=500,
                    key="replay_buffer_size",
                    value = getattr(incremental_settings, "replay_buffer_size")
                )
                
                new_values["replay_ratio"] = st.slider(
                    "Replay Ratio",
                    0.0, 1.0, step=0.05,
                    key="replay_ratio",
                    value = getattr(incremental_settings, "replay_ratio")
                )

            with st.container(border=True):
                st.markdown("### 🎮 Control")
                new_values["enable_training"] = st.checkbox(
                    "🚀 Enable Training",
                    key="enable_training",
                    value = getattr(incremental_settings, "enable_training")
                )

            # Nút lưu thủ công (dự phòng)
            if st.form_submit_button("💾 Lưu cấu hình ngay", type="primary"):
                lines = [f"{k}={v}" for k, v in new_values.items()]
                CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
                st.success("Đã lưu! Các thay đổi sẽ được IL job tự động áp dụng")
                
                time.sleep(2)
                st.rerun()

    with view_col:
        st.subheader("📋 Preview file src/config/.config_incremental hiện tại")

        if CONFIG_PATH.exists():
            current_content = CONFIG_PATH.read_text(encoding="utf-8")
            st.code(current_content, language="bash")
            st.caption(f"Last modified: {datetime.fromtimestamp(CONFIG_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("File `.config` chưa tồn tại. Nó sẽ được tạo khi bạn lưu lần đầu.")
            
        current_runtime = {}
        for field_name, field_info in il_config_fields.items():  # ← Lấy từ class
            current_runtime[field_name] = getattr(incremental_settings, field_name)
        
        st.json(current_runtime, expanded=True)
        
        st.markdown("---")
        st.caption("💡 Thay đổi ở đây chỉ có hiệu lực sau khi **restart app/ các service**. File `.config` có định dạng đơn giản key=value, dễ chỉnh tay bằng Notepad nếu cần.")
        # st.markdown("## 📊 Incremental Training Monitor")

        # col1, col2, col3 = st.columns(3)
        # col1.metric("Current Step", "12")
        # col2.metric("F1-Score", "0.89 ↑")
        # col3.metric("False Alarm Rate", "0.04 ↓")

        # # Demo chart
        # import numpy as np
        # steps = np.arange(1, 21)
        # st.line_chart({
        #     "F1-Score": 0.6 + steps * 0.015 + np.random.normal(0, 0.01, len(steps)),
        #     "Loss": [1/x for x in steps]
        # }, use_container_width=True)

        # st.info(f"🕐 Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
