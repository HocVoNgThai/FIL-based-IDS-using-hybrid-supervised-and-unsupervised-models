# scripts/session1.py
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    SessionDataLoader, SessionManager, evaluate_final_pipeline, 
    evaluate_unsupervised_detailed, evaluate_supervised_with_unknown, 
    evaluate_supervised_model, evaluate_gray_zone, plot_comparison_chart, 
    get_label_name, calculate_unknown_metrics
)

# [UPDATE] Đường dẫn mới
BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"

def load_replay_buffer(path):
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X, y = loader.load_session_data(path, scaler_fit=False)
    if len(X) > 30000: idx = np.random.choice(len(X), 30000, replace=False); return X[idx], y[idx]
    return X, y

def session1_workflow():
    print("=== SESSION 1: Reconn (3) DETECTION & IL ==="); save_dir = "results/session1"
    
    # [UPDATE] Cập nhật đường dẫn file
    s1_train = os.path.join(BASE_DATA_DIR, "train_session1.parquet")
    s1_test = os.path.join(BASE_DATA_DIR, "test_session1.parquet")
    s0_train = os.path.join(BASE_DATA_DIR, "train_session0.parquet") # Cho replay buffer
    s0_test = os.path.join(BASE_DATA_DIR, "test_session0.parquet")   # Cho test forgetting
    
    mgr = SessionManager(); mgr.advance_to_session_1(s1_train, s1_test)
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X_train, y_train = loader.load_session_data(s1_train)
    X_test, y_test = loader.load_session_data(s1_test)
    
    ae = AETrainer(81, 12); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(0, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    metrics_data = {'XGBoost': {}, 'AE': {}, 'OCSVM': {}, 'Pipeline': {}}
    
    # --- PHASE 1: Detection ---
    print("\n--- Phase 1: Detection (Pre-IL) ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    
    calculate_unknown_metrics(y_train, preds, unknown_label=3, save_dir=save_dir, session_name="PreIL")
    
    xgb_pre, xgb_conf = pipeline.xgb.predict_with_confidence(X_train)
    metrics_data['XGBoost']['Pre'] = evaluate_supervised_with_unknown(
        y_train, xgb_pre, xgb_conf, threshold=0.7, 
        session_name="Sess1_PreIL", save_dir=save_dir, target_unknown=3
    )
    
    evaluate_gray_zone(y_train, xgb_pre, xgb_conf, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.99, "Sess1_PreIL", save_dir)
    
    ae_f1, oc_f1 = evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Sess1_PreIL", save_dir, return_f1=True)
    metrics_data['AE']['Pre'] = ae_f1; metrics_data['OCSVM']['Pre'] = oc_f1
    
    y_str_train = [get_label_name(y) if y!=3 else "UNKNOWN" for y in y_train]
    metrics_data['Pipeline']['Pre'] = f1_score(y_str_train, preds, average='weighted')
    
    # --- PHASE 2: IL ---
    print("\n--- Phase 2: IL ---")
    # [UPDATE] Dùng đường dẫn s0_train
    X_old, y_old = load_replay_buffer(s0_train)
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    
    # --- PHASE 3: Eval ---
    print("\n--- Phase 3: Eval (Post-IL) ---")
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    metrics_data['Pipeline']['Post'] = evaluate_final_pipeline(y_test, final_preds, "Sess1_PostIL", save_dir, return_f1=True)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    metrics_data['XGBoost']['Post'] = evaluate_supervised_model(y_test, xgb_post, "Sess1_PostIL", save_dir, "XGBoost", return_f1=True)
    
    ae_f1_p, oc_f1_p = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Sess1_PostIL", save_dir, return_f1=True)
    metrics_data['AE']['Post'] = ae_f1_p; metrics_data['OCSVM']['Post'] = oc_f1_p
    
    plot_comparison_chart(metrics_data, "Session 1", f"{save_dir}/comparison_chart_sess1.png")
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 1)
    
    # Tính Backward Transfer (BWT)
    print("\n>> Checking Forgetting on Session 0...")
    loader0 = SessionDataLoader(); loader0.load_scaler('sessions/session0/scaler.joblib')
    X0, y0 = loader0.load_session_data(s0_test)
    
    acc0 = accuracy_score([get_label_name(y) for y in y0], pipeline.predict(X0))
    acc1 = accuracy_score([get_label_name(y) for y in y_test], final_preds)
    
    print(f"   Acc on S1 (New): {acc1:.4f}")
    print(f"   Acc on S0 (Old): {acc0:.4f}")
    
    return pipeline, {'S0': acc0, 'S1': acc1}

if __name__ == "__main__": session1_workflow()