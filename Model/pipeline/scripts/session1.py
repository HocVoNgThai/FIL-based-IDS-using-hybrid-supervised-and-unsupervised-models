# scripts/session1.py
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
# Import đầy đủ các hàm từ utils đã cập nhật
from src.utils import (
    SessionDataLoader, SessionManager, evaluate_final_pipeline, 
    evaluate_unsupervised_detailed, evaluate_supervised_with_unknown, 
    evaluate_supervised_model, evaluate_gray_zone, plot_comparison_chart, 
    get_label_name, calculate_unknown_metrics
)

def load_replay_buffer(path):
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X, y = loader.load_session_data(path, scaler_fit=False)
    # Lấy mẫu ngẫu nhiên 30k dòng để làm Replay Buffer
    if len(X) > 30000: 
        idx = np.random.choice(len(X), 30000, replace=False)
        return X[idx], y[idx]
    return X, y

def session1_workflow():
    print("=== SESSION 1: Reconn (3) DETECTION & IL ==="); save_dir = "results/session1"
    
    # 1. Setup & Load Data
    mgr = SessionManager()
    mgr.advance_to_session_1("Code_XuLi/session1_train.parquet", "Code_XuLi/session1_test.parquet")
    
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X_train, y_train = loader.load_session_data("Code_XuLi/session1_train.parquet")
    X_test, y_test = loader.load_session_data("Code_XuLi/session1_test.parquet")
    
    # 2. Load Models (Tham số tối ưu: Dim=32, nu=0.1, Conf=0.7)
    ae = AETrainer(81, 32)
    ocsvm = IncrementalOCSVM(nu=0.1)
    xgb = OpenSetXGBoost(confidence_threshold=0.7)
    
    mgr.load_models(0, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    metrics_data = {'XGBoost': {}, 'AE': {}, 'OCSVM': {}, 'Pipeline': {}}
    
    # --- PHASE 1: Detection (Pre-IL) ---
    print("\n--- Phase 1: Detection (Pre-IL) ---")
    # Chạy pipeline trên tập train (chứa Unknown Label 3)
    preds, details = pipeline.predict(X_train, return_details=True)
    
    # A. Tính toán khả năng phát hiện Unknown
    calculate_unknown_metrics(y_train, preds, unknown_label=3, save_dir=save_dir, session_name="PreIL")
    
    # B. Đánh giá XGBoost khi gặp Unknown (dùng threshold lọc)
    xgb_pre, xgb_conf = pipeline.xgb.predict_with_confidence(X_train)
    metrics_data['XGBoost']['Pre'] = evaluate_supervised_with_unknown(
        y_train, xgb_pre, xgb_conf, threshold=0.7, 
        session_name="Sess1_PreIL", save_dir=save_dir, target_unknown_label=3
    )
    
    # C. Phân tích Vùng Xám (Gray Zone Analysis)
    evaluate_gray_zone(
        y_train, xgb_pre, xgb_conf, details['ae_pred'], details['ocsvm_pred'], 
        conf_min=0.7, conf_max=0.99, session_name="Sess1_PreIL", save_dir=save_dir
    )
    
    # D. Đánh giá Unsupervised (Overall)
    ae_f1, oc_f1 = evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Sess1_PreIL", save_dir, return_f1=True)
    metrics_data['AE']['Pre'] = ae_f1; metrics_data['OCSVM']['Pre'] = oc_f1
    
    # E. Pipeline Score
    y_str_train = [get_label_name(y) if y!=3 else "UNKNOWN" for y in y_train]
    metrics_data['Pipeline']['Pre'] = f1_score(y_str_train, preds, average='weighted')
    
    # --- PHASE 2: IL ---
    print("\n--- Phase 2: IL ---")
    # Load dữ liệu cũ để XGBoost không quên bài
    X_old, y_old = load_replay_buffer("Code_XuLi/session0_train.parquet")
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    
    # --- PHASE 3: Eval (Post-IL) ---
    print("\n--- Phase 3: Evaluation (Post-IL) ---")
    
    # 1. Test trên tập hiện tại (S1) - Label 3 giờ đã là Known
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    metrics_data['Pipeline']['Post'] = evaluate_final_pipeline(y_test, final_preds, "Sess1_PostIL", save_dir, return_f1=True)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    metrics_data['XGBoost']['Post'] = evaluate_supervised_model(y_test, xgb_post, "Sess1_PostIL", save_dir, "XGBoost", return_f1=True)
    
    ae_f1_p, oc_f1_p = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Sess1_PostIL", save_dir, return_f1=True)
    metrics_data['AE']['Post'] = ae_f1_p; metrics_data['OCSVM']['Post'] = oc_f1_p
    
    plot_comparison_chart(metrics_data, "Session 1", f"{save_dir}/comparison_chart_sess1.png")
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 1)
    
    # 2. CHECK FORGETTING (Test ngược lại trên Session 0 Testset)
    print("\n>> Checking Forgetting on Session 0...")
    loader0 = SessionDataLoader(); loader0.load_scaler('sessions/session0/scaler.joblib')
    X_s0, y_s0 = loader0.load_session_data("Code_XuLi/session0_test.parquet", scaler_fit=False)
    
    # Tính accuracy trên S1 (hiện tại)
    y_str_s1 = [get_label_name(y) for y in y_test]
    acc_s1 = accuracy_score(y_str_s1, final_preds)
    
    # Tính accuracy trên S0 (quá khứ)
    preds_s0 = pipeline.predict(X_s0)
    y_str_s0 = [get_label_name(y) for y in y_s0]
    acc_s0 = accuracy_score(y_str_s0, preds_s0)
    
    print(f"   Acc on S1 (New): {acc_s1:.4f}")
    print(f"   Acc on S0 (Old): {acc_s0:.4f}")

    # Trả về dict accuracy để run_complete_workflow tính IL Metrics
    return pipeline, {'S0': acc_s0, 'S1': acc_s1}