# scripts/run_ablation_study.py
import os
import sys
import numpy as np
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import SessionDataLoader, SessionManager, plot_scenarios_comparison, get_label_name, plot_cm, plot_ablation_evolution

# Cấu hình đường dẫn
BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "sessions/global_scaler.joblib"
SAVE_DIR = "results/ablation_study"

def run_ablation_full():
    print("🚀 STARTING FULL ABLATION STUDY (All Cases)")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Biến lưu trữ kết quả toàn cục: { 'Case 0': {'XGB Only': rep, ...}, ... }
    ablation_history = {}
    
    # Danh sách các case cần chạy
    cases = [0, 1, 2]
    
    for case_id in cases:
        print(f"\n{'='*40}\n PROCESSING CASE {case_id}\n{'='*40}")
        case_name = f"Case {case_id}"
        ablation_history[case_name] = {}
        
        # 1. Load Data
        test_file = f"test_session{case_id}.parquet"
        print(f"Loading data: {test_file}...")
        test_path = os.path.join(BASE_DATA_DIR, test_file)
        
        loader = SessionDataLoader()
        loader.load_scaler(GLOBAL_SCALER_PATH)
        X_test, y_test = loader.load_data_raw(test_path)
        X_test = loader.apply_scaling(X_test, fit=False)
        y_str_test = [get_label_name(y) for y in y_test]
        
        # 2. Load Models
        print(f"Loading models for Case {case_id}...")
        mgr = SessionManager()
        
        # Khởi tạo vỏ model
        ae = AETrainer(81, 32)
        ocsvm = IncrementalOCSVM(nu=0.15)
        xgb = OpenSetXGBoost(0.7)
        
        # Load trọng số
        try:
            mgr.load_models(case_id, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
        except Exception as e:
            print(f"Error loading models for Case {case_id}: {e}")
            print("Skipping this case...")
            continue

        # 3. Define Scenarios
        scenarios = {
            "XGB Only":       SequentialHybridPipeline(xgb=xgb, ae=None, ocsvm=None),
            "XGB + AE":       SequentialHybridPipeline(xgb=xgb, ae=ae,   ocsvm=None),
            "XGB + OCSVM":    SequentialHybridPipeline(xgb=xgb, ae=None, ocsvm=ocsvm),
            "Full Pipeline":  SequentialHybridPipeline(xgb=xgb, ae=ae,   ocsvm=ocsvm)
        }
        
        # 4. Run Scenarios
        for sc_name, pipe in scenarios.items():
            print(f"   >>> Running: {sc_name}")
            preds = pipe.predict(X_test)
            
            # Save CM
            safe_name = f"{case_name}_{sc_name}".replace(" ", "_")
            plot_cm(y_str_test, preds, f"CM {sc_name} - {case_name}", f"{SAVE_DIR}/cm_{safe_name}.png")
            
            # Store Metrics
            rep = classification_report(y_str_test, preds, output_dict=True, zero_division=0)
            ablation_history[case_name][sc_name] = rep
            
        # Vẽ biểu đồ so sánh cột cho từng Case (Optional)
        plot_scenarios_comparison(ablation_history[case_name], f"{SAVE_DIR}/bar_chart_{case_name}.png")

    # 5. Vẽ biểu đồ đường tổng hợp qua các Case
    print("\n📊 Generating Global Evolution Charts...")
    plot_ablation_evolution(ablation_history, SAVE_DIR)
    
    print("\n✅ Full Ablation Study Completed!")

if __name__ == "__main__":
    run_ablation_full()