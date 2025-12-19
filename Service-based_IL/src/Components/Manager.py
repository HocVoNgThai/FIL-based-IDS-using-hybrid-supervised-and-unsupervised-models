# STANDARD LIBS
from pathlib import Path

# 3RD LIBS
import joblib

# MINE
from src.config.settings import settings
from src.Components.Models import *
class Manager:
    def __init__(self, dir_in):
        self.dir_in = None
        if self.check_dir(dir_in):
            self.dir_in = dir_in
        else:
            self.dir_in = self.get_full_path(dir_in)
            if self.dir_in is None:
                print("[ERROR] Manager.py - Dir không tồn tại!")
                exit(0)
    
    def check_dir(self, dir_in):
        dir_in = Path(dir_in)
        if dir_in.exists():
            return True
        
        if dir_in.is_dir():
            return False
        
        return False
    
    def get_full_path(self, dir_in) -> None:
        dir_in = Path.cwd() / dir_in
        if Path.exists(dir_in):
            print("[DEBUG] Manager.py - Model path is: ", dir_in)
            return dir_in
        
        return None
    
    def load_models(self, models: list):
        dir_in = Path(self.dir_in)
        # xgb.load_model(dir_in/xgb.model_name)
        # ocsvm.load_model(dir_in/ocsvm.model_name)
        # ae.load_model(dir_in/ae.model_name)
        for model in models:
            model.load_model(dir_in/model.model_name)
        print("[MANAGER] Models Loaded Successfully")
            
        
