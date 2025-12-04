# src/pipeline.py
import numpy as np

class SequentialHybridPipeline:
    def __init__(self, ae, ocsvm, xgb):
        self.ae = ae; self.ocsvm = ocsvm; self.xgb = xgb
        self.label_map = {0:"BENIGN", 1:"DDoS", 2:"DoS", 3:"Reconn", 4:"Vuln Scan", 5:"Merlin"}
        
    def predict(self, X, return_details=False):
        print(f"Pipeline processing {len(X)} samples...")
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        # --- CẤU HÌNH QUAN TRỌNG ---
        # Ngưỡng từ chối cứng: Dưới mức này là UNKNOWN ngay lập tức.
        # 0.8 là mức an toàn để bảo vệ các Unknown mà XGBoost đã "đánh hơi" thấy (thường ~0.5-0.7)
        CONF_REJECT = 0.75 
        
        # Ngưỡng tin tưởng tuyệt đối
        CONF_HIGH = 0.95
        
        ae_is_normal = self.ae.is_normal(X)
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0)
        
        final_preds = []
        stats = {"low_unk":0, "atk_acc":0, "high_ben":0, "gray_rec":0, "gray_rej":0}
        
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            # --- LUỒNG XỬ LÝ "BÀN TAY SẮT" ---
            
            # BƯỚC 1: Kiểm tra độ tự tin của XGBoost trước
            if conf < CONF_REJECT:
                # Nếu XGBoost không chắc chắn (bất kể nó đoán là Benign hay Attack)
                # -> Gán ngay là UNKNOWN. Không cho OCSVM/AE can thiệp.
                final_preds.append("UNKNOWN")
                stats["low_unk"] += 1
                continue # Dừng xử lý sample này, chuyển sang sample tiếp theo
            
            # BƯỚC 2: Nếu đủ độ tự tin (>= 0.8)
            
            if p_val != 0: # XGBoost bảo là Attack
                # Vì conf >= 0.8, ta tin nó luôn
                final_preds.append(self.label_map.get(p_val, "UNKNOWN"))
                stats["atk_acc"] += 1
                
            else: # XGBoost bảo là Benign
                if conf >= CONF_HIGH:
                    # Rất tự tin -> Benign
                    final_preds.append("BENIGN")
                    stats["high_ben"] += 1
                else:
                    # Gray Zone (0.8 <= Conf < 0.99): Vùng này mới cần OCSVM/AE
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
            # Retrain AE nhẹ
            self.ae.train_on_known_data(X_benign, 200, verbose=False)
            # OCSVM partial_fit
            self.ocsvm.partial_fit(X_benign)
        
        # XGBoost retrain
        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)