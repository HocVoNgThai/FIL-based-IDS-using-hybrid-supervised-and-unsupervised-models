# scripts/Scenario0.py
import os, sys, numpy as np
from sklearn.metrics import accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import *

BASE_DATA_DIR = "merge1.4_3-4-5/Scenario-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "Scenarios/global_scaler.joblib"

def Scenario0_initial_training():
    print("=== Scenario 0 (INITIAL TRAINING) ===")
    train_path = os.path.join(BASE_DATA_DIR, "train_Scenario0.parquet")
    test_path = os.path.join(BASE_DATA_DIR, "test_Scenario0.parquet")
    mgr = ScenarioManager(); mgr.initialize_Scenario_0(train_path, test_path)
    loader = ScenarioDataLoader()
    
    X_train_raw, y_train = loader.load_data_raw(train_path)
    X_train = loader.apply_scaling(X_train_raw, fit=True)
    loader.save_scaler(GLOBAL_SCALER_PATH)
    
    X_test = loader.apply_scaling(loader.load_data_raw(test_path)[0], fit=False)
    y_test = loader.load_data_raw(test_path)[1]
    
    ae = AETrainer(81, 32); ae.train_on_known_data(X_train[y_train==0], epochs=100)
    ocsvm = IncrementalOCSVM(nu=0.15); ocsvm.train(X_train[y_train==0])
    xgb = OpenSetXGBoost(0.7); xgb.train(X_train, y_train, is_incremental=False)
    pipeline = SequentialHybridPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)
    
    print("\n--- EVAL Scenario 0 ---")
    metrics = {}
    
    xgb_pred, _ = xgb.predict_with_confidence(X_test)
    metrics['XGBoost'] = evaluate_supervised_model(y_test, xgb_pred, "Scenario 0", "results", "XGBoost")
    
    final_preds, details = pipeline.predict(X_test, return_details=True)
    metrics['Pipeline'] = evaluate_final_pipeline(y_test, final_preds, "Scenario 0", "results")
    
    ae_rep, oc_rep = evaluate_unsupervised_detailed(y_test, details['ae_pred'], details['ocsvm_pred'], "Scenario 0", "results")
    metrics['AE'] = ae_rep
    metrics['OCSVM'] = oc_rep
    
    mgr.save_models({'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}, 0)
    acc = accuracy_score([get_label_name(y) for y in y_test], final_preds)
    return {'ae.pt': ae}, loader, pipeline, acc, metrics

if __name__ == "__main__": Scenario0_initial_training()