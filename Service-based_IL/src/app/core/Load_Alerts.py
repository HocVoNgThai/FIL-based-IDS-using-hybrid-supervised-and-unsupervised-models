# src.app.core.Load_Alerts

# standard libs
import os, sys
from pathlib import Path
from datetime import datetime
import time
import gc
import json

# 3rd Libs
import pandas as pd
import streamlit as st

#
COLS_TO_DROP = ["Flow ID", "Timestamp"]

# ===================
# CLASS 
# ===================
class Load_Data:
    def __init__(self, dir_in, auto_refresh, last_update_time, refresh_interval =60, file = None):
        # TIME SEQ
        self.refresh_interval = refresh_interval
        self.auto_refresh = auto_refresh
        self.last_update_time = last_update_time
        self.enable_reload_immediate = True
        
        # INIT DF
        self.df = None
        
        # INIT DIR
        self.dir = self.get_full_path(dir_in)
        self.filePath = Path(self.dir) / f"{datetime.now().date()}.jsonl"
        if file is not None:
            self.filePath = self.dir / file
        
        if not Path(self.filePath).exists:
            return None
    
    def check_dir(self, dir_in):
        dir_in = Path(dir_in)
        if dir_in.exists():
            return True
        
        if dir_in.is_dir():
            return False
        
        return False
    
    def get_full_path(self, dir_in) -> None:
        if self.check_dir(dir_in):
            return Path(dir_in)
        
        dir_in = Path.cwd() / dir_in
        if not Path.exists(dir_in):
            print("[Load_Alerts] - Log Dir not found !")
            return None
        
        return dir_in
            
    def load_alerts(self, limit = 1000) -> pd.DataFrame:
        current_time = time.time()
        
        if not Path.exists(self.filePath):
            return pd.DataFrame(None)
        
        # Quyết định có cần reload không
        should_reload = True
        if not self.auto_refresh:
            should_reload= False
        if current_time - self.last_update_time < self.refresh_interval:
            should_reload= False
        
        if self.enable_reload_immediate:
            should_reload = True
            self.enable_reload_immediate = False

        print(f"[LOAD DATA] {should_reload}, delta time: {current_time - self.last_update_time}, lastuptime: {self.last_update_time}")
        if should_reload:
            try:
                rows = []
                with open(self.filePath, "r") as f:
                    for line in f:
                        rows.append(json.loads(line))
                        
                                
                self.df = pd.DataFrame(rows[-limit:])
                self.last_update_time=current_time
                del rows
                gc.collect()
                return self.df
                
            except Exception as e:
                st.error(f"Lỗi đọc file alerts (jsonl, json): {e}")
                return self.df if self.df is not None else pd.DataFrame()
            
        return self.df
        
        
    