# scripts/Scenario2.py
import os, sys, numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import *

BASE_DATA_DIR = "merge1.4_3-4-5/Scenario-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "Scenarios/global_scaler.joblib"

def load_replay_buffer(path):
    loader = ScenarioDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    X, y = loader.load_data_raw(path); X = loader.apply_scaling(X, fit=False)
    if len(X) > 30000: idx = np.random.choice(len(X), 30000, replace=False); return X[idx], y[idx]
    return X, y

def Scenario2_workflow():
    print("=== Scenario 2: MITM (4) & DNS Spoofing (5) DETECTION & IL ==="); save_dir = "results/Scenario2"
    s2_train = os.path.join(BASE_DATA_DIR, "train_Scenario2.parquet")
    s2_test = os.path.join(BASE_DATA_DIR, "test_Scenario2.parquet")
    s1_train = os.path.join(BASE_DATA_DIR, "train_Scenario1.parquet")
    s1_test = os.path.join(BASE_DATA_DIR, "test_Scenario1.parquet")
    s0_test = os.path.join(BASE_DATA_DIR, "test_Scenario0.parquet")
    mgr = ScenarioManager(); mgr.advance_to_Scenario_2(s2_train, s2_test)
    loader = ScenarioDataLoader(); loader.load_scaler(GLOBAL_SCALER_PATH)
    
    X_train = loader.apply_scaling(loader.load_data_raw(s2_train)[0], fit=False)
    y_train = loader.load_data_raw(s2_train)[1]
    X_test = loader.apply_scaling(loader.load_data_raw(s2_test)[0], fit=False)
    y_test = loader.load_data_raw(s2_test)[1]
    
    ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.7)
    mgr.load_models(1, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    pipeline = SequentialHybridPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)
    results = {'metrics': {}, 'unknown_stats': {}}
    
    print("\n--- Phase 1: Detection ---")
    preds, details = pipeline.predict(X_train, return_details=True)
    map_unknown = [4, 5]
    evaluate_final_pipeline(y_train, preds, "Scenario2_PreIL", save_dir, map_new_to_unknown=map_unknown)
    results['unknown_stats']['Pre'] = calculate_unknown_metrics(y_train, preds, [4, 5], save_dir, "Scenario1_PreIL")
    
    xgb_pre, conf_pre = pipeline.xgb.predict_with_confidence(X_train)
    evaluate_supervised_with_unknown(y_train, xgb_pre, conf_pre, atk_thres=0.7, ben_thres=0.7, Scenario_name="Scenario2_PreIL", save_dir=save_dir, target_unknown=[4, 5])
    evaluate_gray_zone(y_train, xgb_pre, conf_pre, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.90, "Scenario2_PreIL", save_dir)
    
    print("\n--- Phase 2: IL ---")
    X_old, y_old = load_replay_buffer(s1_train)
    pipeline.incremental_learning(X_train, y_train, X_old, y_old)
    #pipeline.incremental_learning(X_train, y_train)
    
    print("\n--- Phase 3: Eval ---")
    final_preds, details_test = pipeline.predict(X_test, return_details=True)
    results['metrics']['Pipeline'] = evaluate_final_pipeline(y_test, final_preds, "Scenario2_PostIL", save_dir)
    
    xgb_post, _ = pipeline.xgb.predict_with_confidence(X_test)
    results['metrics']['XGBoost'] = evaluate_supervised_model(y_test, xgb_post, "Scenario2_PostIL", save_dir, "XGBoost")
    
    ae_rep, oc_rep = evaluate_unsupervised_detailed(y_test, details_test['ae_pred'], details_test['ocsvm_pred'], "Scenario2_PostIL", save_dir)
    results['metrics']['AE'] = ae_rep
    results['metrics']['OCSVM'] = oc_rep
    
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 2)
    
    loader1 = ScenarioDataLoader(); loader1.load_scaler(GLOBAL_SCALER_PATH)
    X1 = loader1.apply_scaling(loader1.load_data_raw(s1_test)[0], fit=False); y1 = loader1.load_data_raw(s1_test)[1]
    acc1 = accuracy_score([get_label_name(y) for y in y1], pipeline.predict(X1))
    loader0 = ScenarioDataLoader(); loader0.load_scaler(GLOBAL_SCALER_PATH)
    X0 = loader0.apply_scaling(loader0.load_data_raw(s0_test)[0], fit=False); y0 = loader0.load_data_raw(s0_test)[1]
    acc0 = accuracy_score([get_label_name(y) for y in y0], pipeline.predict(X0))
    
    y_s2 = [get_label_name(y) for y in y_test]; acc2 = accuracy_score(y_s2, final_preds)
    results['acc_s0'] = acc0; results['acc_s1'] = acc1; results['acc_s2'] = acc2
    return pipeline, results

if __name__ == "__main__": Scenario2_workflow()