# scripts/session2.py
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    SessionDataLoader, SessionManager, evaluate_final_pipeline, 
    evaluate_unsupervised_detailed, evaluate_supervised_model, 
    evaluate_supervised_with_unknown, evaluate_gray_zone,
    plot_comparison_chart, get_label_name, plot_unknown_breakdown, 
    calculate_unknown_metrics
)

def load_replay_buffer(path):
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X, y = loader.load_session_data(path, scaler_fit=False)
    if len(X) > 30000: idx = np.random.choice(len(X), 30000, replace=False); return X[idx], y[idx]
    return X, y

def session2_workflow():
    print("=== SESSION 2: Vuln Scan (4) DETECTION & IL ==="); save_dir = "results/session2"
    mgr = SessionManager(); mgr.advance_to_session_2("Code_XuLi/session2_train.parquet", "Code_XuLi/session2_test.parquet")
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X_train, y_train = loader.load_session_data("Code_XuLi/session2_train.parquet")
    X_test, y_test = loader.load_session_data("Code_XuLi/session2_test.parquet")
    
    ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.1); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(1, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    metrics_data = {'XGBoost': {}, 'AE': {}, 'OCSVM': {}, 'Pipeline': {}}
    
    # --- PHASE 1: Detection ---
    print("\n--- Phase 1: Detection ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    
    calculate_unknown_metrics(y_train, preds, unknown_label=4, save_dir=save_dir, session_name="PreIL")
    
    xgb_pre, xgb_conf = pipeline.xgb.predict_with_confidence(X_train)
    metrics_data['XGBoost']['Pre'] = evaluate_supervised_with_unknown(
        y_train, xgb_pre, xgb_conf, 0.7, "Sess2_PreIL", save_dir, target_unknown_label=4
    )
    
    evaluate_gray_zone(y_train, xgb_pre, xgb_conf, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.99, "Sess2_PreIL", save_dir)
    
    ae_f1, oc_f1 = evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Sess2_PreIL", save_dir, return_f1=True)
    metrics_data['AE']['Pre'] = ae_f1; metrics_data['OCSVM']['Pre'] = oc_f1
    
    y_str_train = [get_label_name(y) if y!=4 else "UNKNOWN" for y in y_train]
    metrics_data['Pipeline']['Pre'] = f1_score(y_str_train, preds, average='weighted')
    
    # --- PHASE 2: IL ---
    print("\n--- Phase 2: IL ---")
    # Replay buffer từ session trước đó (S1)
    X_old, y_old = load_replay_buffer("Code_XuLi/session1_train.parquet")
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    
    # --- PHASE 3: Eval ---
    print("\n--- Phase 3: Eval (Post-IL) ---")
    # 1. Test S2 (Current)
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    metrics_data['Pipeline']['Post'] = evaluate_final_pipeline(y_test, final_preds, "Sess2_PostIL", save_dir, return_f1=True)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    metrics_data['XGBoost']['Post'] = evaluate_supervised_model(y_test, xgb_post, "Sess2_PostIL", save_dir, "XGBoost", return_f1=True)
    
    ae_f1_p, oc_f1_p = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Sess2_PostIL", save_dir, return_f1=True)
    metrics_data['AE']['Post'] = ae_f1_p; metrics_data['OCSVM']['Post'] = oc_f1_p
    
    plot_comparison_chart(metrics_data, "Session 2", f"{save_dir}/comparison_chart_sess2.png")
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 2)
    
    # 2. CHECK FORGETTING (Test ngược trên S1 và S0)
    y_str_s2 = [get_label_name(y) for y in y_test]
    acc_s2 = accuracy_score(y_str_s2, final_preds)
    
    # Test S1
    print("\n>> Checking Forgetting on Session 1...")
    loader1 = SessionDataLoader(); loader1.load_scaler('sessions/session0/scaler.joblib')
    X_s1, y_s1 = loader1.load_session_data("Code_XuLi/session1_test.parquet")
    preds_s1 = pipeline.predict(X_s1)
    acc_s1 = accuracy_score([get_label_name(y) for y in y_s1], preds_s1)
    
    # Test S0
    print(">> Checking Forgetting on Session 0...")
    loader0 = SessionDataLoader(); loader0.load_scaler('sessions/session0/scaler.joblib')
    X_s0, y_s0 = loader0.load_session_data("Code_XuLi/session0_test.parquet")
    preds_s0 = pipeline.predict(X_s0)
    acc_s0 = accuracy_score([get_label_name(y) for y in y_s0], preds_s0)
    
    print(f"   Acc S2: {acc_s2:.4f} | Acc S1: {acc_s1:.4f} | Acc S0: {acc_s0:.4f}")
    
    return pipeline, {'S0': acc_s0, 'S1': acc_s1, 'S2': acc_s2}