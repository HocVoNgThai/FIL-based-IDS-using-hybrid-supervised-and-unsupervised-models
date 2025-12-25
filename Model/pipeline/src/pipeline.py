# src/pipeline.py
import numpy as np

class SequentialHybridPipeline:
    def __init__(self, xgb, ae=None, ocsvm=None):
        self.xgb = xgb; self.ae = ae; self.ocsvm = ocsvm
        self.label_map = {0:"BENIGN", 1:"DDoS", 2:"DoS", 3:"Reconn", 4:"MITM", 5:"DNS Spoofing"}
        
    def predict(self, X, return_details=False):
        modes = ["XGB"]; 
        if self.ae: modes.append("AE")
        if self.ocsvm: modes.append("OCSVM")
        print(f"Pipeline [{' + '.join(modes)}] processing {len(X)} samples...")
        
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        CONF_REJECT = 0.7  
        CONF_HIGH = 0.9   
        
        ae_is_normal = self.ae.is_normal(X) if self.ae else None
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0) if self.ocsvm else None
        
        final_preds = []
        stats = {"low_unk":0, "atk_acc":0, "high_ben":0, "gray_pass":0, "gray_fail":0}
        
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            if conf < CONF_REJECT:
                final_preds.append("UNKNOWN")
                stats["low_unk"] += 1
                continue

            if p_val != 0:
                final_preds.append(self.label_map.get(p_val, "UNKNOWN"))
                stats["atk_acc"] += 1
            else:

                if conf >= CONF_HIGH:
                    final_preds.append("BENIGN")
                    stats["high_ben"] += 1
                else:

                    is_safe = True 
                    
                    if self.ae and self.ocsvm:

                        if ae_is_normal[i] and ocsvm_is_normal[i]: 
                            is_safe = True
                        else:
                            is_safe = False
                    elif self.ae:
                        is_safe = ae_is_normal[i]
                    elif self.ocsvm:
                        is_safe = ocsvm_is_normal[i]

                    if is_safe:
                        final_preds.append("BENIGN")
                        stats["gray_pass"] += 1
                    else:
                        final_preds.append("UNKNOWN")
                        stats["gray_fail"] += 1
        
        print(f"   Stats: HighBen: {stats['high_ben']} | GrayPass: {stats['gray_pass']} | GrayFail(Unk): {stats['gray_fail']}")
        
        details = {
            'ae_pred': ae_is_normal.astype(int) if ae_is_normal is not None else None,
            'ocsvm_pred': ocsvm_is_normal.astype(int) if ocsvm_is_normal is not None else None
        }
        return (final_preds, details) if return_details else final_preds

    def incremental_learning(self, X_new, y_new, X_old, y_old):
        print("=== INCREMENTAL LEARNING ===")
        X_benign = X_new[y_new == 0]
        if len(X_benign) > 0:
            if self.ae: 
                print(f"   -> Fine-tuning AE on {len(X_benign)} samples (50 epochs)...")
                self.ae.train_on_known_data(X_benign, epochs=50, verbose=False)
            if self.ocsvm: 
                self.ocsvm.partial_fit(X_benign)
        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)