# src/app/pages/Settings.
# STANDARD LIBS
import time

# 3rd libs
import streamlit as st
from datetime import datetime
from pathlib import Path

# Local import
from src.config.settings import settings, Settings

# Đường dẫn file .config (hoặc .env) để lưu thủ công khi thay đổi
CONFIG_FILE = Path("src/config/.config")

def main():
    st.title("⚙️ System Configuration")
    model_fields = Settings.model_fields

    col_left, col_right = st.columns([1.8, 1.2])
    with col_left:
        st.subheader("🔧 Chỉnh sửa cấu hình")
        st.caption(f"📂 Thư mục cha hiện tại: {Path.cwd()}")
        with st.form("settings_form"):
            new_values = {}

            for field_name, field_info in model_fields.items():  # ← Lấy từ class
                current_value = getattr(settings, field_name)

                if field_name in ["DATA_DIR", "ALERTS_DIR", "PKL_DIR", "MODEL_DIR", "NET_INTF"]:
                    # Đường dẫn → text_input với resolve()
                    new_val = st.text_input(
                        field_name,
                        value=str(current_value), #str(current_value.resolve() if isinstance(current_value, Path) else current_value),
                        help=field_info.description or ""
                    )
                elif isinstance(current_value, bool):
                    new_val = st.checkbox(field_name, value=current_value)
                elif isinstance(current_value, int):
                    new_val = st.number_input(field_name, value=current_value, step=1)
                elif isinstance(current_value, str):
                    new_val = st.text_input(field_name, value=current_value)
                else:
                    new_val = st.text_input(field_name, value=str(current_value))

                new_values[field_name] = new_val

            if st.form_submit_button("💾 Lưu cấu hình", type="primary"):
                # Ghi file .config
                lines = [f"{k}={v}" for k, v in new_values.items()]
                CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
                st.success("Đã lưu! Restart các services để áp dụng.")
                
                time.sleep(2)
                st.rerun()

    with col_right:
        st.subheader("📋 Preview file src/config/.config hiện tại")

        if CONFIG_FILE.exists():
            current_content = CONFIG_FILE.read_text(encoding="utf-8")
            st.code(current_content, language="bash")
            st.caption(f"Last modified: {datetime.fromtimestamp(CONFIG_FILE.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("File `.config` chưa tồn tại. Nó sẽ được tạo khi bạn lưu lần đầu.")

        st.subheader("ℹ️ Giá trị hiện tại trong runtime")
        current_runtime = {}
        for field_name, field_info in Settings.model_fields.items():  # ← Lấy từ class
            current_value = getattr(settings, field_name)

            if field_name in ["DATA_DIR", "ALERTS_DIR", "PKL_DIR", "MODEL_DIR", "NET_INTF"]:
                # Đường dẫn → text_input với resolve()
                current_runtime[field_name] = str(current_value)
            else:
                current_runtime[field_name] = current_value
                
        
        st.json(current_runtime, expanded=True)
        
        st.markdown("---")
        st.caption("💡 Thay đổi ở đây chỉ có hiệu lực sau khi **restart app/ các service**. File `.config` có định dạng đơn giản key=value, dễ chỉnh tay bằng Notepad nếu cần.")

    