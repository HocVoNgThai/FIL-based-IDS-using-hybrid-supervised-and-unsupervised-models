# src/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


# XEM SRC LÀ PACKAGE - khi chạy là chạy src/main.py - ví dụ v
class Settings(BaseSettings):
    DATA_DIR: Path = Path("./flows_parquet")
    ALERTS_DIR: Path = Path("./app_logs")
    PKL_DIR: Path = Path("./src/pkl")
    MODEL_DIR: Path = Path("./src/models")
    
    #ARGS CICEXTRACT
    NET_INTF: str = "eth0"
    FTO: int = "1000" # 1000000
    ATO: int = "120000000"
    
    REFRESH_INTERVAL: int = 60
    AUTO_REFRESH: bool = False
    DEBUG: bool = False

    # JAVA
    APP_NAME: str= "CICFlowMeter"
    MAIN_CLASS: str= "cic.cs.unb.ca.ifm.RT_cmd"
    MEM_OPTS_MAX: str = "-Xmx10g" # Max 10 GB
    MEM_OPTS_MIN: str = "-Xms512m" # Min512m
    
    
    class Config:
        env_file = "src/config/.config"          # Tự động đọc file .config
        env_file_encoding = "utf-8"
        case_sensitive = False     # Không phân biệt HOA/thường

# Khởi tạo một lần duy nhất
settings = Settings()