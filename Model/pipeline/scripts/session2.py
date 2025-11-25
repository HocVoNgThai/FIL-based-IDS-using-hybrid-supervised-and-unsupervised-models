# scripts/session2.py
import os, sys, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import SessionDataLoader, SessionManager, evaluate_final_pipeline, evaluate_unsupervised_detailed, plot_metric_comparison, plot_unknown_breakdown

def load_replay_buffer(path, sample_size=30000):
    loader = SessionDataLoader()
    loader.load_scaler('sessions/session0/scaler.joblib')
    X, y = loader.load_session_data(path, scaler_fit=False)
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        return X[idx], y[idx]
    return X, y

def analyze_and_plot_unknowns(y_true, preds, details, unknown_label, save_dir):
    y_true = np.array(y_true); preds = np.array(preds); reasons = details['pred_reason']
    
    is_true = (y_true == unknown_label)
    total_true = np.sum(is_true)
    
    is_pred = (preds == "UNKNOWN_ATTACK")
    total_pred = np.sum(is_pred)
    
    tp = np.sum(is_true & is_pred)
    tp_low_conf = np.sum(is_true & is_pred & (reasons == 2))
    tp_rejected = np.sum(is_true & is_pred & (reasons == 3))
    
    print(f"\n[UNKNOWN ANALYSIS] Target Label: {unknown_label}")
    print(f" - Actual: {total_true} | Predicted: {total_pred} | Correct: {tp}")
    print(f"   > Via Low Confidence: {tp_low_conf}")
    print(f"   > Via Unsup Rejection: {tp_rejected}")
    
    plot_unknown_breakdown(total_true, total_pred, tp, f"Label {unknown_label} (Session 2)", f"{save_dir}/unknown_breakdown_sess2.png")

def session2_workflow():
    print("=== SESSION 2 ==="); save_dir = "results/session2"
    mgr = SessionManager()
    mgr.advance_to_session_2("Code_XuLi/session2_train.parquet", "Code_XuLi/session2_test.parquet")
    loader = SessionDataLoader(); loader.load_scaler('sessions/session0/scaler.joblib')
    X_train, y_train = loader.load_session_data("Code_XuLi/session2_train.parquet")
    X_test, y_test = loader.load_session_data("Code_XuLi/session2_test.parquet")
    
    # Load
    ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.1); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(1, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    # --- PHASE 1: Detection ---
    print("\n--- Phase 1: Detection (Pre-IL) ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    
    analyze_and_plot_unknowns(y_train, preds, details, unknown_label=4, save_dir=save_dir)
    ae_pre_rep, oc_pre_rep = evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Session 2 Pre-IL", save_dir, return_dict=True)
    
    # --- PHASE 2: IL ---
    print("\n--- Phase 2: Incremental Learning ---")
    X_old, y_old = load_replay_buffer("Code_XuLi/session1_train.parquet")
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    
    # --- PHASE 3: Eval ---
    print("\n--- Phase 3: Evaluation (Post-IL) ---")
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    evaluate_final_pipeline(y_test, final_preds, "Session 2 Post-IL", save_dir)
    
    ae_post_rep, oc_post_rep = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Session 2 Post-IL", save_dir, return_dict=True)
    
    # VẼ BIỂU ĐỒ SO SÁNH
    plot_metric_comparison(ae_pre_rep, ae_post_rep, "Autoencoder", "Session 2", f"{save_dir}/ae_comparison_sess2.png")
    plot_metric_comparison(oc_pre_rep, oc_post_rep, "OCSVM", "Session 2", f"{save_dir}/ocsvm_comparison_sess2.png")
    
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 2)
    return pipeline

if __name__ == "__main__": session2_workflow()