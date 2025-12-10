# scripts/session2.py
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
# [FIX] Thêm calculate_unknown_metrics
from src.utils import (
    SessionDataLoader, SessionManager, evaluate_final_pipeline, 
    evaluate_unsupervised_detailed, evaluate_supervised_model, 
    evaluate_supervised_with_unknown, evaluate_gray_zone, 
    plot_comparison_chart, get_label_name, calculate_unknown_metrics
)

BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "sessions/global_scaler.joblib"

def load_replay_buffer(path):
    loader = SessionDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    X_raw, y = loader.load_data_raw(path)
    X = loader.apply_scaling(X_raw, fit=False)
    if len(X) > 30000: idx = np.random.choice(len(X), 30000, replace=False); return X[idx], y[idx]
    return X, y

def session2_workflow():
    print("=== CASE 2: MITM (4) & DNS Spoofing (5) DETECTION & IL ==="); save_dir = "results/session2"
    
    s2_train = os.path.join(BASE_DATA_DIR, "train_session2.parquet")
    s2_test = os.path.join(BASE_DATA_DIR, "test_session2.parquet")
    s1_train = os.path.join(BASE_DATA_DIR, "train_session1.parquet")
    s1_test = os.path.join(BASE_DATA_DIR, "test_session1.parquet")
    s0_test = os.path.join(BASE_DATA_DIR, "test_session0.parquet")
    
    mgr = SessionManager(); mgr.advance_to_case_2(s2_train, s2_test)
    
    loader = SessionDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    X_train = loader.apply_scaling(loader.load_data_raw(s2_train)[0], fit=False)
    y_train = loader.load_data_raw(s2_train)[1]
    X_test = loader.apply_scaling(loader.load_data_raw(s2_test)[0], fit=False)
    y_test = loader.load_data_raw(s2_test)[1]
    
    ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(1, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    metrics_data = {'XGBoost': {}, 'AE': {}, 'OCSVM': {}, 'Pipeline': {}}
    
    print("\n--- Phase 1: Detection ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    
    calculate_unknown_metrics(y_train, preds, [4, 5], save_dir, "PreIL")
    
    xgb_pre, conf_pre = pipeline.xgb.predict_with_confidence(X_train)
    
    metrics_data['XGBoost']['Pre'] = evaluate_supervised_with_unknown(
        y_train, xgb_pre, conf_pre, 
        atk_thres=0.7, ben_thres=0.7, 
        session_name="Case2_PreIL", save_dir=save_dir, target_unknown=[4, 5]
    )
    
    evaluate_gray_zone(y_train, xgb_pre, conf_pre, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.95, "Case2_PreIL", save_dir)
    
    ae_f1, oc_f1 = evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Case2_PreIL", save_dir, return_f1=True)
    metrics_data['AE']['Pre'] = ae_f1; metrics_data['OCSVM']['Pre'] = oc_f1
    
    y_str_train = [get_label_name(y) if y not in [4, 5] else "UNKNOWN" for y in y_train]
    metrics_data['Pipeline']['Pre'] = f1_score(y_str_train, preds, average='weighted')
    
    # --- PHASE 2: IL ---
    print("\n--- Phase 2: IL ---")
    X_old, y_old = load_replay_buffer(s1_train)
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    
    # --- PHASE 3: Eval ---
    print("\n--- Phase 3: Eval ---")
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    metrics_data['Pipeline']['Post'] = evaluate_final_pipeline(y_test, final_preds, "Case2_PostIL", save_dir, return_f1=True)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    metrics_data['XGBoost']['Post'] = evaluate_supervised_model(y_test, xgb_post, "Case2_PostIL", save_dir, "XGBoost", return_f1=True)
    
    ae_f1_p, oc_f1_p = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Case2_PostIL", save_dir, return_f1=True)
    metrics_data['AE']['Post'] = ae_f1_p; metrics_data['OCSVM']['Post'] = oc_f1_p
    
    plot_comparison_chart(metrics_data, "Case 2", f"{save_dir}/comparison_chart_sess2.png")
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 2)
    
    y_s2 = [get_label_name(y) for y in y_test]; acc2 = accuracy_score(y_s2, final_preds)
    
    loader1 = SessionDataLoader(); loader1.load_scaler(GLOBAL_SCALER_PATH)
    X1 = loader1.apply_scaling(loader1.load_data_raw(s1_test)[0], fit=False)
    y1 = loader1.load_data_raw(s1_test)[1]
    acc1 = accuracy_score([get_label_name(y) for y in y1], pipeline.predict(X1))
    
    loader0 = SessionDataLoader(); loader0.load_scaler(GLOBAL_SCALER_PATH)
    X0 = loader0.apply_scaling(loader0.load_data_raw(s0_test)[0], fit=False)
    y0 = loader0.load_data_raw(s0_test)[1]
    acc0 = accuracy_score([get_label_name(y) for y in y0], pipeline.predict(X0))
    
    print(f"   Acc S2: {acc2:.4f} | Acc S1: {acc1:.4f} | Acc S0: {acc0:.4f}")
    
    return pipeline, {'S0': acc0, 'S1': acc1, 'S2': acc2}

if __name__ == "__main__": session2_workflow()