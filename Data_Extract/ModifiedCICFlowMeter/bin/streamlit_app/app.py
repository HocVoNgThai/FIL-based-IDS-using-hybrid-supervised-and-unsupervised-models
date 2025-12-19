# ===== INIT LOAD LIBS =====
import sys, os
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu

## ===== QUAN TRỌNG =====
sys.path.append(str(Path.cwd()))


## ===== QUAN TRỌNG: Loại bỏ khoảng trống mặc định của Streamlit =====
# Đoạn CSS này sẽ loại bỏ padding và margin ở đầu trang và đầu cột
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Container của các column sẽ full height */
    /*.stApp {
        background: #f0f2f6;
    }*/
    
    /* Cách hiện đại nhất từ Streamlit 1.30+ */
    div[data-testid="column"] {
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    /* Nếu muốn nội dung bên trong column cũng stretch */
    div[data-testid="column"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* 1. Loại bỏ padding trên cùng của Main Content và Sidebar */
    .block-container {
        padding-top: 1rem; /* Giữ lại một chút padding để không bị dính vào mép trình duyệt */
        padding-bottom: 0rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* 2. Loại bỏ khoảng trống trên cùng của cột Menu */
    /* Target div chứa nội dung cột đầu tiên (Menu) */
    div[data-testid="column"] div:first-child {
        padding-top: 0px !important;
    }
    /* 3. Loại bỏ padding của option_menu container */
    .st-emotion-cache-1kyxisp { /* Class CSS có thể thay đổi, nhưng thường là option menu wrapper */
        margin-top: -10px !important; /* Đẩy menu lên cao thêm một chút */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Cấu hình Ban đầu ---
# Thiết lập session state cho trạng thái mở rộng/thu gọn của menu
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True # Mặc định là mở rộng (hiện chữ)

# ====== COMPONENTS =====
import time # Dùng để mô phỏng tải trang

# --- Cấu hình Trang ---
st.set_page_config(
    page_title="Menu Tùy Chỉnh",
    layout="wide"
)

# Dữ liệu Menu (Key là tên hiển thị, Icon là tên Bootstrap)
menu_data = {
    "Messenger": "chat-fill",
    "Discord": "controller",
    "Translate": "translate",
    "Drive": "cloud-upload",
    "Youtube": "youtube",
    "Facebook": "facebook",
    "Khóa": "key-fill",
}

# --- CSS Tùy chỉnh (Quan trọng: Ẩn chữ và thu hẹp cột) ---
sidebar_width = 100 if st.session_state.sidebar_expanded else 10 # Rộng 250px nếu mở, 70px nếu thu gọn
main_padding_left = sidebar_width + 10 # Đẩy nội dung chính sang phải


# --- Hàm xử lý nhấp chuột (Toggle) ---
def toggle_sidebar():
    st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
    st.rerun()

sidebar, mainview = st.columns([1, 12], vertical_alignment="top") if st.session_state.sidebar_expanded == True else st.columns([1, 15], vertical_alignment="top") 

with sidebar:
    # with st.container:
    #     if st.button(
    #             ":arrow_left:" if st.session_state.sidebar_expanded else ":arrow_right:",
    #             key="toggle_btn",
    #             help="Nhấn để ẩn/hiện tên Menu",
    #         ):
    #         toggle_sidebar()
    
    tile = sidebar.container(height=100, border=False)
    with tile:
        if st.button(
                ":arrow_left:" if st.session_state.sidebar_expanded else ":arrow_right:",
                key="toggle_btn",
                help="Nhấn để thu hẹp/phóng to tên Menu",
                width="stretch",
            ):
            toggle_sidebar()
        
    selected_option = option_menu(
        menu_title=None,  
        options=list(menu_data.keys()),
        icons= list(menu_data.values()),
        default_index=0,
        orientation="vertical", 
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa", "width": "100%"},
            "icon": {"color": "#0477be", "font-size": "15px"},
            # CSS được áp dụng qua class nav-link để ẩn chữ
            "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#c9c9c9"},
        }
    )

    # Lưu ID của navigation (đôi khi cần cho CSS chính xác)
    # Cần phải tìm ID sau khi render, nhưng tạm thời bỏ qua để đơn giản hóa.
    # st.session_state.nav_id = ...
    
# --- 2. Không Gian Chính (Main Content) ---
# Thêm hiệu ứng tải (tùy chọn)
with mainview:
    st.markdown("---")
    placeholder = st.empty()
    with placeholder.container():
        st.info(f"Đang tải nội dung cho: **{selected_option}**...")
        time.sleep(0.5)

    placeholder.empty()

    # Hiển thị nội dung chi tiết dựa trên lựa chọn
    st.title(f"Ứng dụng: {selected_option}")
    st.header(f"Chi tiết Chức năng: {selected_option}")

    if selected_option == "Drive":
        st.success("Bạn đang ở trang quản lý file Drive. Menu được tạo bằng `streamlit-option-menu`.")
    elif selected_option == "Youtube":
        st.video("https://www.youtube.com/watch?v=kYn6w0w4FvM")
    else:
        st.write(f"Nội dung cho {selected_option}...")

    st.markdown("---")
    st.markdown(f"**Lưu ý:** Thư viện này tự động xử lý trạng thái và icon, giúp bạn có một menu sidebar tùy chỉnh mà không cần CSS phức tạp.")
