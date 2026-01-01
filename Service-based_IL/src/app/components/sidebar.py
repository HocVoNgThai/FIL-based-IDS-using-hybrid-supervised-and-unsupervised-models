# components/sidebar.py

# standard libs


# 3rd libs
from streamlit_option_menu import option_menu
import streamlit as st

# #
# def render_sidebar(sidebar):
#     menu_data = {
#         "Alerts": "bell-fill",
#         "Data": "database",
#         "Resource Monitor": "gear",
#         "IL Results": "database",
#         "IL Config": "gear", #"journal-text"
#         "Settings": "gear"
        
#         # Thêm các trang khác ở đây
#     }

#     with sidebar:
#         # Nút toggle thu gọn/mở rộng (tùy chọn)
#         if st.button(
#             "☰" if st.session_state.get("sidebar_expanded", True) else "▶",
#             key="toggle_sidebar",
#             help="Thu gọn / Mở rộng menu"
#         ):
#             st.session_state.sidebar_expanded = not st.session_state.get("sidebar_expanded", True)
#             st.rerun()

#         # Điều chỉnh độ rộng sidebar dựa trên trạng thái
#         if not st.session_state.get("sidebar_expanded", True):
#             st.markdown(
#                 """
#                 <style>
#                     section[data-testid="stSidebar"] {
#                         width: 70px !important;
#                     }
#                     .st-emotion-cache-1kyxisp {  /* option_menu container */
#                         padding: 0 !important;
#                     }
#                     .st-emotion-cache-1kyxisp .nav-link span {
#                         display: none !important;
#                     }
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )

#         selected = option_menu(
#             menu_title=None,
#             options=list(menu_data.keys()),
#             icons=list(menu_data.values()),
#             default_index=0,
#             orientation="vertical",
#             styles={
#                 "container": {"padding": "0!important", "background-color": "#fafafa"},
#                 "icon": {"color": "#0477be", "font-size": "18px"},
#                 "nav-link": {
#                     "font-size": "14px",
#                     "text-align": "center" if not st.session_state.get("sidebar_expanded", True) else "left",
#                     "margin": "0px",
#                     "--hover-color": "#eee"
#                 },
#                 "nav-link-selected": {"background-color": "#c9c9c9"},
#             }
#         )

#     return selected

def render_sidebar(sidebar):
    # 1. Chỉ giữ lại danh sách tên các trang
    menu_options = [
        "Alerts", 
        "Data", 
        "Resource Monitor", 
        "IL Results", 
        "IL Config", 
        "Settings"
    ]

    with sidebar:
        # Nút toggle (giữ nguyên logic của bạn)
        if st.button(
            "☰" if st.session_state.get("sidebar_expanded", True) else "▶",
            key="toggle_sidebar"
        ):
            st.session_state.sidebar_expanded = not st.session_state.get("sidebar_expanded", True)
            st.rerun()

        # 2. Cập nhật CSS: Nếu thu gọn thì ẩn luôn chữ, chỉ hiện một thanh màu mảnh
        if not st.session_state.get("sidebar_expanded", True):
            st.markdown(
                """
                <style>
                    section[data-testid="stSidebar"] { width: 50px !important; }
                    .nav-link span { display: none !important; }
                </style>
                """,
                unsafe_allow_html=True
            )

        # 3. Gọi option_menu và bỏ tham số icons
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=None,                # <--- QUAN TRỌNG: Đặt là None để bỏ icon
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "5px 0px",       # Thêm chút khoảng cách giữa các dòng
                    "padding": "10px",         # Tăng padding để dễ nhấn khi không có icon
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {
                    "background-color": "#c9c9c9",
                    "font-weight": "bold"      # Làm đậm chữ khi được chọn để dễ nhận diện
                },
            }
        )

    return selected