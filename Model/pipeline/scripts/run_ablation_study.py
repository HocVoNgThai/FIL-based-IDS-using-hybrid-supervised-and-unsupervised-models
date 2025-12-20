# scripts/run_ablation_study.py
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix lỗi hiển thị
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    SessionDataLoader, SessionManager, 
    plot_scenarios_comparison, get_label_name, plot_cm, 
    plot_ablation_evolution, plot_unknown_detection_comparison, 
)

# Cấu hình đường dẫn
BASE_DATA_DIR = "merge1.4_3-4-5/Scenario-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "sessions/global_scaler.joblib"
SAVE_DIR = "results/ablation_study"

def run_ablation_full():
    print("🚀 STARTING FULL ABLATION STUDY (With Unknown Detection Phase)")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Biến lưu trữ kết quả toàn cục
    ablation_history = {}
    
    # Biến lưu kết quả Unknown Detection: { 'XGB Only': {'Scenario 1': 0.5, 'Scenario 2': 0.6}, ... }
    unknown_detection_results = {
        "XGB Only": {}, "XGB + AE": {}, "XGB + OCSVM": {}, "Full Pipeline": {}
    }
    
    Scenarios = [0, 1, 2]
    
    loader = SessionDataLoader()
    loader.load_scaler(GLOBAL_SCALER_PATH)
    mgr = SessionManager()
    
    for Scenario_id in Scenarios:
        print(f"\n{'='*40}\n PROCESSING Scenario {Scenario_id}\n{'='*40}")
        Scenario_name = f"Scenario {Scenario_id}"
        ablation_history[Scenario_name] = {}
        
        # ==============================================================================
        # PHẦN 1: UNKNOWN DETECTION TEST (Chỉ chạy cho Scenario 1 và 2)
        # Mục tiêu: Dùng model cũ để bắt các nhãn mới trong tập Train hiện tại
        # ==============================================================================
        if Scenario_id > 0:
            print(f"   [Phase 1] Testing Unknown Detection Capability (Pre-IL)...")
            
            # 1.1 Xác định nhãn mới cần bắt
            target_unknown_labels = [3] if Scenario_id == 1 else [4, 5]
            
            # 1.2 Load Data Train (Chứa nhãn mới)
            train_path = os.path.join(BASE_DATA_DIR, f"train_session{Scenario_id}.parquet")
            X_train_raw, y_train_raw = loader.load_data_raw(train_path)
            X_train = loader.apply_scaling(X_train_raw, fit=False)
            
            # 1.3 Load Models CŨ (Scenario trước đó)
            prev_Scenario = Scenario_id - 1
            print(f"      -> Loading models from Scenario {prev_Scenario}...")
            ae_old = AETrainer(81, 32)
            ocsvm_old = IncrementalOCSVM(nu=0.15)
            xgb_old = OpenSetXGBoost(0.7)
            mgr.load_models(prev_Scenario, {'ae.pt': ae_old, 'ocsvm.pkl': ocsvm_old, 'xgb.pkl': xgb_old})
            
            # 1.4 Định nghĩa 4 kịch bản với model CŨ
            scenarios_old = {
                "XGB Only":       SequentialHybridPipeline(xgb=xgb_old, ae=None, ocsvm=None),
                "XGB + AE":       SequentialHybridPipeline(xgb=xgb_old, ae=ae_old, ocsvm=None),
                "XGB + OCSVM":    SequentialHybridPipeline(xgb=xgb_old, ae=None, ocsvm=ocsvm_old),
                "Full Pipeline":  SequentialHybridPipeline(xgb=xgb_old, ae=ae_old, ocsvm=ocsvm_old)
            }
            
            # 1.5 Chạy test và tính % Unknown Detection
            for sc_name, pipe in scenarios_old.items():
                preds = pipe.predict(X_train)
                
                # Logic tính toán:
                # - Mẫu thực sự là mới: y_train_raw nằm trong target_unknown_labels
                # - Dự đoán đúng là lạ: preds == "UNKNOWN"
                
                is_truly_new = np.isin(y_train_raw, target_unknown_labels)
                total_new_samples = np.sum(is_truly_new)
                
                if total_new_samples > 0:
                    is_predicted_unknown = (np.array(preds) == "UNKNOWN")
                    # TP: Thực sự mới VÀ đoán là Unknown
                    detected_count = np.sum(is_truly_new & is_predicted_unknown)
                    detection_rate = detected_count / total_new_samples
                else:
                    detection_rate = 0.0
                
                unknown_detection_results[sc_name][f"Scenario {Scenario_id}"] = detection_rate
                print(f"      -> {sc_name}: Detected {detection_rate:.2%} of new attacks as UNKNOWN")

        # ==============================================================================
        # PHẦN 2: STANDARD EVALUATION (Post-IL)
        # Mục tiêu: Đánh giá hiệu năng phân loại sau khi đã học xong Scenario hiện tại
        # ==============================================================================
        print(f"   [Phase 2] Standard Evaluation (Post-IL)...")
        
        # 2.1 Load Data Test
        test_path = os.path.join(BASE_DATA_DIR, f"test_session{Scenario_id}.parquet")
        X_test_raw, y_test_raw = loader.load_data_raw(test_path)
        X_test = loader.apply_scaling(X_test_raw, fit=False)
        y_str_test = [get_label_name(y) for y in y_test_raw]
        
        # 2.2 Load Models HIỆN TẠI (Scenario này)
        print(f"      -> Loading models for Scenario {Scenario_id}...")
        ae = AETrainer(81, 32)
        ocsvm = IncrementalOCSVM(nu=0.15)
        xgb = OpenSetXGBoost(0.7)
        try:
            mgr.load_models(Scenario_id, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
        except Exception as e:
            print(f"      [!] Error loading models: {e}. Skipping eval for this Scenario.")
            continue

        # 2.3 Định nghĩa 4 kịch bản với model HIỆN TẠI
        scenarios = {
            "XGB Only":       SequentialHybridPipeline(xgb=xgb, ae=None, ocsvm=None),
            "XGB + AE":       SequentialHybridPipeline(xgb=xgb, ae=ae,   ocsvm=None),
            "XGB + OCSVM":    SequentialHybridPipeline(xgb=xgb, ae=None, ocsvm=ocsvm),
            "Full Pipeline":  SequentialHybridPipeline(xgb=xgb, ae=ae,   ocsvm=ocsvm)
        }
        
        # 2.4 Chạy đánh giá
        for sc_name, pipe in scenarios.items():
            print(f"      >>> Running Eval: {sc_name}")
            preds = pipe.predict(X_test)
            
            # Save CM
            safe_name = f"{Scenario_name}_{sc_name}".replace(" ", "_")
            plot_cm(y_str_test, preds, f"CM {sc_name} - {Scenario_name}", f"{SAVE_DIR}/cm_{safe_name}.png")
            
            # Store Metrics
            rep = classification_report(y_str_test, preds, output_dict=True, zero_division=0)
            ablation_history[Scenario_name][sc_name] = rep
            
        # Vẽ biểu đồ cột so sánh Precision/Recall/F1 cho Scenario hiện tại
        plot_scenarios_comparison(ablation_history[Scenario_name], f"{SAVE_DIR}/bar_chart_{Scenario_name}.png", Scenario_name)

    # ==============================================================================
    # PHẦN 3: VẼ BIỂU ĐỒ TỔNG HỢP
    # ==============================================================================
    print("\n📊 Generating Summary Charts...")
    
    # 1. Biểu đồ đường (Line Chart) diễn biến F1/Pre/Rec qua các Scenario
    plot_ablation_evolution(ablation_history, SAVE_DIR)
    
    # 2. [NEW] Biểu đồ cột (Grouped Bar) so sánh khả năng phát hiện Unknown ở Scenario 1 & 2
    # Hàm này phải có trong src/utils.py (đã cung cấp ở bước trước)
    try:
        plot_unknown_detection_comparison(unknown_detection_results, f"{SAVE_DIR}/unknown_detection_comparison.png")
    except Exception as e:
        print(f"   [!] Could not plot unknown comparison: {e}")
    
    print("\n✅ Ablation Study Completed! Results saved in 'results/ablation_study'")

if __name__ == "__main__":
    run_ablation_full()