# scripts/session0.py
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score # Thêm import này
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    SessionDataLoader, SessionManager, 
    evaluate_supervised_model, evaluate_final_pipeline, evaluate_unsupervised_detailed,
    get_label_name # <--- QUAN TRỌNG: Thêm import này
)

def session0_initial_training():
    print("=== SESSION 0 ===")
    mgr = SessionManager()
    mgr.initialize_session_0("Code_XuLi/session0_train.parquet", "Code_XuLi/session0_test.parquet")
    
    loader = SessionDataLoader()
    X_train, y_train = loader.load_session_data("Code_XuLi/session0_train.parquet", scaler_fit=True)
    X_test, y_test = loader.load_session_data("Code_XuLi/session0_test.parquet")
    
    # 1. Unsupervised (Benign Only)
    X_benign = X_train[y_train == 0]
    # AE: Dim 32, Epochs 40
    ae = AETrainer(81, 32)
    ae.train_on_known_data(X_benign, epochs=40)
    
    # OCSVM: nu=0.15 (Khắt khe hơn)
    ocsvm = IncrementalOCSVM(nu=0.15)
    ocsvm.train(X_benign)
    
    # 2. Supervised (XGBoost Standard)
    xgb = OpenSetXGBoost(confidence_threshold=0.7)
    xgb.train(X_train, y_train, is_incremental=False)
    
    pipeline = SequentialHybridPipeline(ae, ocsvm, xgb)
    
    # 3. Evaluation
    print("\n--- EVAL SESSION 0 ---")
    # Đánh giá XGBoost riêng
    xgb_pred, _ = xgb.predict_with_confidence(X_test)
    evaluate_supervised_model(y_test, xgb_pred, "Session 0", "results", model_name="XGBoost")

    # Đánh giá Pipeline
    final_preds, details = pipeline.predict(X_test, return_details=True)
    evaluate_final_pipeline(y_test, final_preds, "Session 0", "results")
    evaluate_unsupervised_detailed(y_test, details['ae_pred'], details['ocsvm_pred'], "Session 0", "results")
    
    # Tính Accuracy để trả về cho IL Metrics (Sửa lỗi tại đây)
    y_str_test = [get_label_name(y) for y in y_test]
    acc_s0 = accuracy_score(y_str_test, final_preds)
    
    # Save
    models = {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}
    mgr.save_models(models, 0)
    loader.save_scaler('sessions/session0/scaler.joblib')
    
    # Trả về 4 giá trị để khớp với run_complete_workflow
    return models, loader, pipeline, acc_s0

if __name__ == "__main__":
    session0_initial_training()