# src/config/incremental_settings.py
# standard libs
import json

# 3rd libs
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal

class IncrementalSettings(BaseSettings):
    # Dataset
    initial_train_ratio: float = Field(
        default=0.5,
        ge=0.1, le=0.8,
        description="Tỷ lệ dữ liệu dùng để train ban đầu"
    )
    increment_batch_size: int = Field(
        default=500,
        ge=100, le=50000, multiple_of=100,
        description="Kích thước batch cho mỗi bước incremental"
    )

    # Model (XGBoost)
    learning_rate: float = Field(
        default=0.05,
        ge=0.001, le=0.2,
        description="Learning rate cho XGBoost"
    )
    max_depth: int = Field(
        default=6,
        ge=3, le=12,
        description="Độ sâu tối đa của cây"
    )
    trees_per_step: int = Field(
        default=20,
        ge=1, le=100,
        description="Số cây thêm vào mỗi bước incremental"
    )

    # Anti-forgetting
    replay_buffer_size: int = Field(
        default=5000,
        ge=0, le=100000, multiple_of=500,
        description="Kích thước buffer replay để chống catastrophic forgetting"
    )
    replay_ratio: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description="Tỷ lệ replay trong mỗi batch"
    )

    # Control
    enable_training: bool = Field(
        default=False,
        description="Bật/tắt training incremental"
    )

    class Config:
        env_file = "incremental_learning.json"
        env_prefix = "IL_"  # Nếu muốn dùng .env: INC_INITIAL_TRAIN_RATIO=0.4
        case_sensitive = False

# Singleton instance
incremental_settings = IncrementalSettings()