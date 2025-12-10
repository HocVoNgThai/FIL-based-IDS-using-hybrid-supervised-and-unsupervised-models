# src/pipeline.py
import numpy as np

class SequentialHybridPipeline:
    def __init__(self, ae, ocsvm, xgb):
        self.ae = ae; self.ocsvm = ocsvm; self.xgb = xgb
        self.label_map = {0:"BENIGN", 1:"DDoS", 2:"DoS", 3:"Reconn", 4:"MITM", 5:"DNS Spoofing"}
        
    def predict(self, X, return_details=False):
        print(f"Pipeline processing {len(X)} samples...")
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        CONF_REJECT = 0.70
        
        CONF_HIGH = 0.90
        
        ae_is_normal = self.ae.is_normal(X)
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0)
        
        final_preds = []
        stats = {"low_unk":0, "atk_acc":0, "high_ben":0, "gray_rec":0, "gray_rej":0}
        
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
                    if ae_is_normal[i] and ocsvm_is_normal[i]:
                        final_preds.append("BENIGN")
                        stats["gray_rec"] += 1
                    else:
                        final_preds.append("UNKNOWN")
                        stats["gray_rej"] += 1
        
        print(f"   [STATS] HighBen: {stats['high_ben']} | AtkAcc: {stats['atk_acc']} | LowConf(Unknown): {stats['low_unk']}")
        print(f"           GrayRec (Benign): {stats['gray_rec']} | GrayRej (Unknown): {stats['gray_rej']}")
        
        details = {'ae_pred': ae_is_normal.astype(int), 'ocsvm_pred': ocsvm_is_normal.astype(int)}
        return (final_preds, details) if return_details else final_preds

    def incremental_learning(self, X_new, y_new, X_old, y_old):
        print("=== INCREMENTAL LEARNING ===")
        X_benign = X_new[y_new == 0]
        if len(X_benign) > 0:
            self.ae.train_on_known_data(X_benign, 100, verbose=False)
            self.ocsvm.partial_fit(X_benign)

        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)