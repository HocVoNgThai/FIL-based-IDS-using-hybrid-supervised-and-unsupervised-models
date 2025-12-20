# scripts/run_ablation_study.py
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    SessionDataLoader, SessionManager, 
    evaluate_final_pipeline, calculate_weighted_metrics, # <--- Import hàm mới
    plot_pipeline_evolution_comparison, get_label_name
)

# Cấu hình
BASE_DATA_DIR = "merge1.4_3-4-5/Scenario-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "sessions/global_scaler.joblib"
SAVE_DIR = "results/test"

def run_evolution_analysis():
    print("🚀 STARTING PIPELINE EVOLUTION ANALYSIS (With Corrected Pre-IL Metrics)")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Biến lưu trữ dữ liệu để vẽ biểu đồ tổng hợp 5 cột mốc
    evolution_data = {} 
    
    loader = SessionDataLoader()
    loader.load_scaler(GLOBAL_SCALER_PATH)
    mgr = SessionManager()
    
    # ========================== SCENARIO 0 ==========================
    print("\n--- Processing SCENARIO 0 ---")
    # Load Model Case 0
    xgb0 = OpenSetXGBoost(0.7); ae0 = AETrainer(81, 32); ocsvm0 = IncrementalOCSVM(nu=0.15)
    mgr.load_models(0, {'ae.pt': ae0, 'ocsvm.pkl': ocsvm0, 'xgb.pkl': xgb0})
    pipe0 = SequentialHybridPipeline(xgb0, ae0, ocsvm0)
    
    # Load Data Test Case 0
    X0, y0 = loader.load_data_raw(f"{BASE_DATA_DIR}/test_session0.parquet")
    X0_s = loader.apply_scaling(X0, fit=False)
    
    # Predict
    pred0 = pipe0.predict(X0_s)
    
    # Calculate Metrics (Không cần map vì chưa có nhãn lạ)
    metrics0 = calculate_weighted_metrics(y0, pred0, map_new_to_unknown=None)
    evolution_data['Scenario 0'] = metrics0
    
    # Vẽ chi tiết (Bảng & CM)
    evaluate_final_pipeline(y0, pred0, "Scenario0_Final", f"{SAVE_DIR}/Scenario0")

    # ========================== SCENARIO 1 ==========================
    print("\n--- Processing SCENARIO 1 ---")
    
    # 1. Pre-IL (Dùng Model Case 0 test trên Data Train Case 1 chứa Reconn)
    # Reconn (3) -> Map thành UNKNOWN để tính điểm cho đúng
    X1_tr, y1_tr = loader.load_data_raw(f"{BASE_DATA_DIR}/train_session1.parquet")
    X1_tr_s = loader.apply_scaling(X1_tr, fit=False)
    pred1_pre = pipe0.predict(X1_tr_s) # Vẫn dùng pipe0
    
    # Tính điểm: Ép nhãn 3 (Reconn) thành UNKNOWN trong Ground Truth
    metrics1_pre = calculate_weighted_metrics(y1_tr, pred1_pre, map_new_to_unknown=[3])
    evolution_data['Scenario 1 (Pre-IL)'] = metrics1_pre
    
    # Vẽ chi tiết (Sẽ hiện bảng có cột UNKNOWN thay vì Reconn)
    evaluate_final_pipeline(y1_tr, pred1_pre, "Scenario1_PreIL", f"{SAVE_DIR}/Scenario1_Pre", map_new_to_unknown=[3])
    
    # 2. Post-IL (Dùng Model Case 1 test trên Data Test Case 1)
    xgb1 = OpenSetXGBoost(0.7); ae1 = AETrainer(81, 32); ocsvm1 = IncrementalOCSVM(nu=0.15)
    mgr.load_models(1, {'ae.pt': ae1, 'ocsvm.pkl': ocsvm1, 'xgb.pkl': xgb1})
    pipe1 = SequentialHybridPipeline(xgb1, ae1, ocsvm1)
    
    X1_te, y1_te = loader.load_data_raw(f"{BASE_DATA_DIR}/test_session1.parquet")
    X1_te_s = loader.apply_scaling(X1_te, fit=False)
    pred1_post = pipe1.predict(X1_te_s)
    
    # Tính điểm: Không map (để hiện Reconn)
    metrics1_post = calculate_weighted_metrics(y1_te, pred1_post, map_new_to_unknown=None)
    evolution_data['Scenario 1 (Post-IL)'] = metrics1_post
    
    evaluate_final_pipeline(y1_te, pred1_post, "Scenario1_PostIL", f"{SAVE_DIR}/Scenario1_Post", map_new_to_unknown=None)

    # ========================== SCENARIO 2 ==========================
    print("\n--- Processing SCENARIO 2 ---")
    
    # 1. Pre-IL (Dùng Model Case 1 test trên Data Train Case 2 chứa MITM/DNS)
    # MITM(4), DNS(5) -> Map thành UNKNOWN
    X2_tr, y2_tr = loader.load_data_raw(f"{BASE_DATA_DIR}/train_session2.parquet")
    X2_tr_s = loader.apply_scaling(X2_tr, fit=False)
    pred2_pre = pipe1.predict(X2_tr_s) # Vẫn dùng pipe1
    
    metrics2_pre = calculate_weighted_metrics(y2_tr, pred2_pre, map_new_to_unknown=[4, 5])
    evolution_data['Scenario 2 (Pre-IL)'] = metrics2_pre
    
    evaluate_final_pipeline(y2_tr, pred2_pre, "Scenario2_PreIL", f"{SAVE_DIR}/Scenario2_Pre", map_new_to_unknown=[4, 5])
    
    # 2. Post-IL (Dùng Model Case 2 test trên Data Test Case 2)
    xgb2 = OpenSetXGBoost(0.7); ae2 = AETrainer(81, 32); ocsvm2 = IncrementalOCSVM(nu=0.15)
    mgr.load_models(2, {'ae.pt': ae2, 'ocsvm.pkl': ocsvm2, 'xgb.pkl': xgb2})
    pipe2 = SequentialHybridPipeline(xgb2, ae2, ocsvm2)
    
    X2_te, y2_te = loader.load_data_raw(f"{BASE_DATA_DIR}/test_session2.parquet")
    X2_te_s = loader.apply_scaling(X2_te, fit=False)
    pred2_post = pipe2.predict(X2_te_s)
    
    metrics2_post = calculate_weighted_metrics(y2_te, pred2_post, map_new_to_unknown=None)
    evolution_data['Scenario 2 (Post-IL)'] = metrics2_post
    
    evaluate_final_pipeline(y2_te, pred2_post, "Scenario2_PostIL", f"{SAVE_DIR}/Scenario2_Post", map_new_to_unknown=None)

    # ========================== FINAL PLOT ==========================
    print("\n📊 Generating Evolution Comparison Chart...")
    plot_pipeline_evolution_comparison(evolution_data, f"{SAVE_DIR}/pipeline_evolution_comparison.png")
    
    print("\n✅ All Done! Check 'results/test/pipeline_evolution_comparison.png'")

if __name__ == "__main__":
    run_evolution_analysis()