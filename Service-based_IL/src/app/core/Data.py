# pages/Data.py
import os, sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st


class Data_Labeling:
    def __init__(self, dir_in):
        self.dir = Path(dir_in)
        if not self.check_dir():
            return None
        
        self.df = None
        
    def check_dir(self, dir: str = None):
        if dir:
            if not Path.exists(dir):
                return False
            else:
                return True
        
        if not Path.exists(self.dir):
            return False
        return True
    
    def load_data(self):
        if self.dir.suffix ==".parquet" :
            self.df = pd.read_parquet(self.dir)
        elif self.dir.suffix == ".csv":
            self.df = pd.read_csv(self.dir)
        else:
            self.df = None
            
        return self.df
    
    def save_data(self, df: pd.DataFrame, dir: str = None):
        if df.empty:
            return None
        try:
            if dir:
                self.dir = Path.parent/dir
            
            df.to_parquet(self.dir_in)
                
        except Exception as e:
            print(f"[DATA] Cannot Save new data to {self.dir.name}")
        
        return self.dir.name
        
    
    