# scripts/session0.py
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import SessionDataLoader, SessionManager, evaluate_supervised_model, evaluate_final_pipeline, evaluate_unsupervised_detailed

def session0_initial_training():
    print("=== SESSION 0 ===")
    mgr = SessionManager()
    mgr.initialize_session_0("Code_XuLi/session0_train.parquet", "Code_XuLi/session0_test.parquet")
    loader = SessionDataLoader()
    X_train, y_train = loader.load_session_data("Code_XuLi/session0_train.parquet", scaler_fit=True)
    X_test, y_test = loader.load_session_data("Code_XuLi/session0_test.parquet")
    
    # Unsupervised (Benign Only)
    X_benign = X_train[y_train == 0]
    ae = AETrainer(81, 32) # Deep AE
    ae.train_on_known_data(X_benign, epochs=200)
    
    ocsvm = IncrementalOCSVM(nu=0.15) # nu=0.1 để chặt hơn
    ocsvm.train(X_benign)
    
    # Supervised
    xgb = OpenSetXGBoost(confidence_threshold=0.7)
    xgb.train(X_train, y_train, is_incremental=False)
    
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    print("\n--- EVAL SESSION 0 ---")
    final_preds, details = pipeline.predict(X_test, return_details=True)
    evaluate_final_pipeline(y_test, final_preds, "Session 0", "results")
    evaluate_unsupervised_detailed(y_test, details['ae_pred'], details['ocsvm_pred'], "Session 0", "results")
    
    models = {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}
    mgr.save_models(models, 0)
    loader.save_scaler('sessions/session0/scaler.joblib')
    return models, loader, pipeline

if __name__ == "__main__": session0_initial_training()