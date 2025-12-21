# app.py
#standard
import os
import sys

# 3 rd
import streamlit as st
import importlib

# Local Import 
from src.app.components.sidebar import render_sidebar
from src.app.pages import *

# Cấu hình trang
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# Loại bỏ padding mặc định
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== SESSION STATE SIDEBAR =====
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True

def toggle_sidebar():
    st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
    st.rerun()
    
sidebar, mainview = st.columns([1, 12], vertical_alignment="top") if st.session_state.sidebar_expanded == True else st.columns([1, 15], vertical_alignment="top") 
selected_page = render_sidebar(sidebar)

# ===== MAIN VIEW =====
page_mapping = {
        "Alerts": "Alerts",
        "Data" : "Data",
        "IL Config" : "IncrementalLearning",
        "Settings" : "Settings"
    }

# with mainview:
#     # Load và chạy trang tương ứng
#     if selected_page in page_mapping:
#         module_name = page_mapping[selected_page]
#         module = importlib.import_module(module_name)
        
#         # Gọi hàm main của trang (theo chuẩn)
#         if hasattr(module, "main"):
#             module.main()
#         else:
#             st.error(f"Trang {selected_page} chưa có hàm main()")
#     else:
#         st.warning("Chọn một mục từ menu bên trái.")
        
with mainview:
    # Load và chạy trang tương ứng
    if selected_page in page_mapping:
        module_name = page_mapping[selected_page]
        module = importlib.import_module("src.app.pages."+module_name)
        
        # Gọi hàm main của trang (theo chuẩn)
        if hasattr(module, "main"):
            module.main()
        else:
            st.error(f"Trang {selected_page} chưa có hàm main()")
    else:
        st.warning("Chọn một mục từ menu bên trái.")