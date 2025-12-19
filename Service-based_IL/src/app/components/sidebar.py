# components/sidebar.py

# standard libs


# 3rd libs
from streamlit_option_menu import option_menu
import streamlit as st

#
def render_sidebar(sidebar):
    menu_data = {
        "Alerts": "bell-fill",
        "IL Config": "gear", #"journal-text"
        "Data": "database",
        "Settings": "gear",
        # Thêm các trang khác ở đây
    }

    with sidebar:
        # Nút toggle thu gọn/mở rộng (tùy chọn)
        if st.button(
            "☰" if st.session_state.get("sidebar_expanded", True) else "▶",
            key="toggle_sidebar",
            help="Thu gọn / Mở rộng menu"
        ):
            st.session_state.sidebar_expanded = not st.session_state.get("sidebar_expanded", True)
            st.rerun()

        # Điều chỉnh độ rộng sidebar dựa trên trạng thái
        if not st.session_state.get("sidebar_expanded", True):
            st.markdown(
                """
                <style>
                    section[data-testid="stSidebar"] {
                        width: 70px !important;
                    }
                    .st-emotion-cache-1kyxisp {  /* option_menu container */
                        padding: 0 !important;
                    }
                    .st-emotion-cache-1kyxisp .nav-link span {
                        display: none !important;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

        selected = option_menu(
            menu_title=None,
            options=list(menu_data.keys()),
            icons=list(menu_data.values()),
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#0477be", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "center" if not st.session_state.get("sidebar_expanded", True) else "left",
                    "margin": "0px",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {"background-color": "#c9c9c9"},
            }
        )

    return selected