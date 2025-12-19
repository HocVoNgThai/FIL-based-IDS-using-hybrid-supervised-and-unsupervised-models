# src/app/pages/IncrementalLearning.py
import streamlit as st
from datetime import datetime
from pathlib import Path
import json

# Local import
from src.config.settings import settings
from src.config.incremental_config import incremental_settings

CONFIG_PATH = Path("src/config/incremental_config.json")
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    st.title("🧠 Incremental Learning Configuration")

    # Đồng bộ session_state với settings (lần đầu hoặc khi reload)
    for field_name in incremental_settings.model_fields:
        session_key = field_name
        if session_key not in st.session_state:
            st.session_state[session_key] = getattr(incremental_settings, field_name)
    
    def save_config_to_json():
        """Lưu config hiện tại vào file JSON để giữ thói quen cũ"""
        config_dict = incremental_settings.model_dump()  # Pydantic v2
        config_dict["control"] = {"updated_at": datetime.now().isoformat()}
        
        CONFIG_PATH.write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False)
        )
        # st.success("✅ Đã lưu cấu hình!")
            
    # Layout
    cfg_col, view_col = st.columns([1.2, 2.8])

    with cfg_col:
        st.markdown("## ⚙️ Incremental Config")

        with st.container(border=True):
            st.markdown("### 📦 Dataset")
            st.slider(
                "Initial Train Ratio",
                0.1, 0.8, step=0.05,
                key="initial_train_ratio",
                on_change=save_config_to_json
            )
            st.number_input(
                "Increment Batch Size",
                min_value=100, max_value=50000, step=100,
                key="increment_batch_size",
                on_change=save_config_to_json
            )

        with st.container(border=True):
            st.markdown("### 🧠 Model (XGBoost)")
            st.slider(
                "Learning Rate",
                0.001, 0.2, step=0.001,
                format="%.3f",
                key="learning_rate",
                on_change=save_config_to_json
            )
            st.slider(
                "Max Depth",
                3, 12,
                key="max_depth",
                on_change=save_config_to_json
            )
            st.slider(
                "Trees per Step",
                1, 100,
                key="trees_per_step",
                on_change=save_config_to_json
            )

        with st.container(border=True):
            st.markdown("### 🧬 Anti-Catastrophic Forgetting")
            st.number_input(
                "Replay Buffer Size",
                min_value=0, max_value=100000, step=500,
                key="replay_buffer_size",
                on_change=save_config_to_json
            )
            st.slider(
                "Replay Ratio",
                0.0, 1.0, step=0.05,
                key="replay_ratio",
                on_change=save_config_to_json
            )

        with st.container(border=True):
            st.markdown("### 🎮 Control")
            st.checkbox(
                "🚀 Enable Training",
                key="enable_training",
                on_change=save_config_to_json
            )

        # Nút lưu thủ công (dự phòng)
        if st.button("💾 Lưu cấu hình ngay"):
            save_config_to_json()

    with view_col:
        st.markdown("## 📊 Incremental Training Monitor")

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Step", "12")
        col2.metric("F1-Score", "0.89 ↑")
        col3.metric("False Alarm Rate", "0.04 ↓")

        # Demo chart
        import numpy as np
        steps = np.arange(1, 21)
        st.line_chart({
            "F1-Score": 0.6 + steps * 0.015 + np.random.normal(0, 0.01, len(steps)),
            "Loss": [1/x for x in steps]
        }, use_container_width=True)

        st.info(f"🕐 Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
