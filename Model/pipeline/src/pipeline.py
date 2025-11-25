# src/pipeline.py
import numpy as np

class SequentialHybridPipeline:
    def __init__(self, ae, ocsvm, xgb):
        self.ae = ae
        self.ocsvm = ocsvm
        self.xgb = xgb
        
    def predict(self, X, return_details=False):
        print(f"Pipeline processing {len(X)} samples (Weak Consensus + Reason Tracking)...")
        
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        CONF_MIN = 0.70
        CONF_HIGH = 0.99 
        
        ae_errors = self.ae.get_reconstruction_errors(X)
        ae_pred = (ae_errors <= self.ae.known_threshold).astype(int) 
        
        ocsvm_scores = self.ocsvm.decision_function(X)
        ocsvm_pred = (ocsvm_scores > 0).astype(int)
        
        final_preds = []
        pred_reasons = [] # 0:Known, 1:HighBenign, 2:LowConf, 3:Rejected, 4:Recovered
        
        stats = {"low_conf": 0, "gray_zone_rejected": 0, "high_conf": 0, "recovered": 0}
        
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            # CASE 1: Low Confidence -> UNKNOWN
            if conf < CONF_MIN:
                final_preds.append("UNKNOWN_ATTACK")
                pred_reasons.append(2) # Reason: Low Conf
                stats["low_conf"] += 1
                continue
            
            # CASE 2: KNOWN ATTACK -> ACCEPT
            if p_val != 0:
                final_preds.append(f"KNOWN_ATTACK_{p_val}")
                pred_reasons.append(0) # Reason: Known Attack
                continue
                
            # CASE 3: BENIGN VERIFICATION
            if conf >= CONF_HIGH:
                final_preds.append("BENIGN")
                pred_reasons.append(1) # Reason: High Conf Benign
                stats["high_conf"] += 1
            else:
                # Vùng xám: Weak Consensus
                # Nếu 1 trong 2 bảo Normal -> Benign
                if ae_pred[i] == 1 or ocsvm_pred[i] == 1:
                    final_preds.append("BENIGN")
                    pred_reasons.append(4) # Reason: Recovered
                    stats["recovered"] += 1
                else:
                    # Cả 2 bảo Abnormal -> Unknown
                    final_preds.append("UNKNOWN_ATTACK")
                    pred_reasons.append(3) # Reason: Rejected by Unsup
                    stats["gray_zone_rejected"] += 1
        
        print(f"   [STATS] Low Conf: {stats['low_conf']} | Passed High: {stats['high_conf']}")
        print(f"   [STATS] Gray Zone -> Recovered: {stats['recovered']} | Rejected: {stats['gray_zone_rejected']}")
        
        details = {
            'ae_pred': ae_pred, 
            'ocsvm_pred': ocsvm_pred,
            'pred_reason': np.array(pred_reasons) # Trả về lý do
        }
        return (final_preds, details) if return_details else final_preds

    def incremental_learning(self, X_new, y_new, X_old, y_old):
        print("=== INCREMENTAL LEARNING ===")
        X_new_benign = X_new[y_new == 0]
        
        if len(X_new_benign) > 0:
            print(f"   [UNSUP] Updating AE & OCSVM on {len(X_new_benign)} NEW benign samples...")
            self.ae.train_on_known_data(X_new_benign, epochs=10, verbose=False)
            self.ocsvm.partial_fit(X_new_benign)
        
        print(f"   [SUPERVISED] Updating XGBoost...")
        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)
        print("=== DONE IL ===\n")