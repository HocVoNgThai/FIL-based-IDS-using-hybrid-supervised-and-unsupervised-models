# src/pipeline.py
import numpy as np

class SequentialHybridPipeline:
    def __init__(self, ae, ocsvm, xgb):
        self.ae = ae; self.ocsvm = ocsvm; self.xgb = xgb
        self.label_map = {0:"BENIGN", 1:"DDoS", 2:"DoS", 3:"Reconn", 4:"Vuln Scan", 5:"Merlin"}
        
    def predict(self, X, return_details=False):
        print(f"Pipeline processing {len(X)} samples...")
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        # [QUAN TRỌNG] Quay lại ngưỡng 0.70
        # Nếu XGBoost không chắc chắn > 70%, ta coi là UNKNOWN ngay lập tức.
        # Điều này sẽ khôi phục lại khả năng phát hiện Unknown (52% -> 52%)
        CONF_MIN_UNKNOWN = 0.70 
        
        # Chỉ tin là Benign nếu cực kỳ chắc chắn
        CONF_HIGH_BENIGN = 0.95 
        
        ae_pred = (self.ae.get_reconstruction_errors(X) <= self.ae.known_threshold).astype(int)
        ocsvm_pred = (self.ocsvm.decision_function(X) > 0).astype(int)
        
        final_preds = []
        stats = {"low_unk":0, "high_benign":0, "gray_rec":0, "gray_rej":0}
        
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            # CASE 1: XGBoost dự đoán là ATTACK (DDoS, DoS...)
            if p_val != 0:
                if conf >= CONF_MIN_UNKNOWN:
                    # Đủ tự tin -> Chấp nhận nhãn
                    final_preds.append(self.label_map.get(p_val, "UNKNOWN"))
                else:
                    # Không đủ tự tin -> UNKNOWN (Để bắt Reconn/Vuln Scan)
                    final_preds.append("UNKNOWN")
                    stats["low_unk"] += 1

            # CASE 2: XGBoost dự đoán là BENIGN
            else:
                if conf >= CONF_HIGH_BENIGN:
                    final_preds.append("BENIGN")
                    stats["high_benign"] += 1
                else:
                    # Gray Zone (Benign < 0.98)
                    # Phải vượt qua cả AE và OCSVM mới được là Benign
                    if ae_pred[i] == 1 and ocsvm_pred[i] == 1:
                        final_preds.append("BENIGN")
                        stats["gray_rec"] += 1
                    else:
                        final_preds.append("UNKNOWN")
                        stats["gray_rej"] += 1
        
        print(f"   [STATS] HighBen: {stats['high_benign']} | LowConf (Unknown): {stats['low_unk']}")
        print(f"           GrayRec (Benign): {stats['gray_rec']} | GrayRej (Unknown): {stats['gray_rej']}")
        
        details = {'ae_pred': ae_pred, 'ocsvm_pred': ocsvm_pred}
        return (final_preds, details) if return_details else final_preds

    def incremental_learning(self, X_new, y_new, X_old, y_old):
        print("=== INCREMENTAL LEARNING ===")
        X_benign = X_new[y_new == 0]
        if len(X_benign) > 0:
            # Retrain DAE với noise
            self.ae.train_on_known_data(X_benign, 5, verbose=False)
            self.ocsvm.partial_fit(X_benign)
        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)