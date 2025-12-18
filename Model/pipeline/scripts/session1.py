# scripts/session1.py
import os, sys, numpy as np
from sklearn.metrics import f1_score, accuracy_score 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import *

BASE_DATA_DIR = "merge1.4_3-4-5/Scenario-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "sessions/global_scaler.joblib"

def load_replay_buffer(path):
    loader = SessionDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    X, y = loader.load_data_raw(path); X = loader.apply_scaling(X, fit=False)
    if len(X) > 30000: idx = np.random.choice(len(X), 30000, replace=False); return X[idx], y[idx]
    return X, y

def session1_workflow():
    print("=== Scenario 1: Reconn (3) DETECTION & IL ==="); save_dir = "results/session1"
    s1_train = os.path.join(BASE_DATA_DIR, "train_session1.parquet")
    s1_test = os.path.join(BASE_DATA_DIR, "test_session1.parquet")
    s0_train = os.path.join(BASE_DATA_DIR, "train_session0.parquet") 
    s0_test = os.path.join(BASE_DATA_DIR, "test_session0.parquet")
    mgr = SessionManager(); mgr.advance_to_Scenario_1(s1_train, s1_test)
    loader = SessionDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    
    X_train = loader.apply_scaling(loader.load_data_raw(s1_train)[0], fit=False)
    y_train = loader.load_data_raw(s1_train)[1]
    X_test = loader.apply_scaling(loader.load_data_raw(s1_test)[0], fit=False)
    y_test = loader.load_data_raw(s1_test)[1]
    
    ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(0, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)
    results = {'metrics': {}, 'unknown_stats': {}}
    
    print("\n--- Phase 1: Detection (Pre-IL) ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    evaluate_final_pipeline(y_train, preds, "Scenario1_PreIL", save_dir)
    results['unknown_stats']['Pre'] = calculate_unknown_metrics(y_train, preds, unknown_label=3, save_dir=save_dir, session_name="Scenario1_PreIL")
    
    xgb_pre, xgb_conf = pipeline.xgb.predict_with_confidence(X_train)
    evaluate_supervised_with_unknown(y_train, xgb_pre, xgb_conf, atk_thres=0.7, ben_thres=0.7, session_name="Scenario1_PreIL", save_dir=save_dir, target_unknown=3)
    evaluate_gray_zone(y_train, xgb_pre, xgb_conf, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.9, "Scenario1_PreIL", save_dir)
    evaluate_unsupervised_detailed(y_train, details['ae_pred'], details['ocsvm_pred'], "Scenario1_PreIL", save_dir, return_f1=True)
    
    print("\n--- Phase 2: IL ---")
    X_old, y_old = load_replay_buffer(s0_train)
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    #pipeline.incremental_learning(X_train, y_train)
    print("\n--- Phase 3: Eval (Post-IL) ---")
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    results['metrics']['Pipeline'] = evaluate_final_pipeline(y_test, final_preds, "Scenario1_PostIL", save_dir)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    results['metrics']['XGBoost'] = evaluate_supervised_model(y_test, xgb_post, "Scenario1_PostIL", save_dir, "XGBoost")
    
    ae_rep, oc_rep = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Scenario1_PostIL", save_dir)
    results['metrics']['AE'] = ae_rep
    results['metrics']['OCSVM'] = oc_rep
    
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 1)
    
    loader0 = SessionDataLoader(); loader0.load_scaler(GLOBAL_SCALER_PATH)
    X0 = loader0.apply_scaling(loader0.load_data_raw(s0_test)[0], fit=False); y0 = loader0.load_data_raw(s0_test)[1]
    acc0 = accuracy_score([get_label_name(y) for y in y0], pipeline.predict(X0))
    acc1 = accuracy_score([get_label_name(y) for y in y_test], final_preds)
    results['acc_s0'] = acc0; results['acc_s1'] = acc1
    return pipeline, results

if __name__ == "__main__": session1_workflow()