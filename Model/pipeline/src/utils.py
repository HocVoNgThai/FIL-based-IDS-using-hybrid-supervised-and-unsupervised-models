# src/utils.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ... (Giữ nguyên phần SessionDataLoader và SessionManager cũ) ...
# ==================== DATA LOADER ====================
class SessionDataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def load_session_data(self, file_path, label_col='Label', binary_label_col='Binary Label', scaler_fit=False):
        if not os.path.exists(file_path): raise ValueError(f"File {file_path} does not exist")
        df = pd.read_parquet(file_path)
        y = None
        cols_drop = []
        if label_col in df.columns:
            cols_drop.append(label_col); y = df[label_col].astype(int)
        if binary_label_col in df.columns: cols_drop.append(binary_label_col)
        X_df = df.drop(columns=cols_drop)
        X_filled = X_df.fillna(0).values
        if scaler_fit:
            print("Fitting StandardScaler...")
            X = self.scaler.fit_transform(X_filled)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                if os.path.exists('sessions/session0/scaler.joblib'): self.load_scaler('sessions/session0/scaler.joblib')
                X = self.scaler.transform(X_filled)
            else: X = self.scaler.transform(X_filled)
        return X, y
    
    def save_scaler(self, p): os.makedirs(os.path.dirname(p), exist_ok=True); joblib.dump(self.scaler, p)
    def load_scaler(self, p): self.scaler = joblib.load(p); self.is_fitted = True

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self, base_dir='sessions'):
        self.base_dir = base_dir
        self.current_session = 0
        self.session_info = {}
        os.makedirs(base_dir, exist_ok=True)
        for i in range(3): os.makedirs(f"{base_dir}/session{i}", exist_ok=True)
        self.load_session_info()
    
    def initialize_session_0(self, train, test):
        self.current_session = 0; self.session_info[0] = {'train': train, 'test': test, 'labels': [0, 1, 2]}
        self.save_session_info()
    def advance_to_session_1(self, train, test):
        self.current_session = 1; self.session_info[1] = {'train': train, 'test': test, 'labels': [0, 1, 2, 3]}
        self.save_session_info()
    def advance_to_session_2(self, train, test):
        self.current_session = 2; self.session_info[2] = {'train': train, 'test': test, 'labels': [0, 1, 2, 3, 4]}
        self.save_session_info()
    def save_models(self, models, sess_id):
        p = f"{self.base_dir}/session{sess_id}/models"
        os.makedirs(p, exist_ok=True)
        for k, v in models.items():
            if hasattr(v, 'save_model'): v.save_model(f"{p}/{k}")
            else: joblib.dump(v, f"{p}/{k}.joblib")
    def load_models(self, sess_id, models):
        p = f"{self.base_dir}/session{sess_id}/models"
        for k, v in models.items():
            path = f"{p}/{k}"
            if hasattr(v, 'load_model'): v.load_model(path)
            elif os.path.exists(f"{path}.joblib"): models[k] = joblib.load(f"{path}.joblib")
        return models
    def save_session_info(self): joblib.dump({'curr': self.current_session, 'info': self.session_info}, f"{self.base_dir}/session_info.joblib")
    def load_session_info(self):
        if os.path.exists(f"{self.base_dir}/session_info.joblib"):
            d = joblib.load(f"{self.base_dir}/session_info.joblib")
            self.current_session = d['curr']; self.session_info = d['info']

# ==================== NEW VISUALIZATION FUNCTIONS ====================

def plot_cm(y_true, y_pred, title, save_path, labels=None):
    if labels is None: unique_labels = sorted(list(set(y_true) | set(y_pred)))
    else: unique_labels = labels
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(title)
    plt.ylabel('True'); plt.xlabel('Pred')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()

def plot_metric_comparison(pre_report, post_report, model_name, session_name, save_path):
    """Vẽ biểu đồ so sánh Metrics trước và sau IL cho AE/OCSVM (Target: Abnormal Detection)"""
    metrics = ['precision', 'recall', 'f1-score']
    
    # Lấy metrics của class 'Abnormal' (class 0 trong binary mapping)
    # Vì target_names=['Abnormal', 'Normal'], nên key trong dict là 'Abnormal'
    pre_vals = [pre_report['Abnormal'][m] for m in metrics]
    post_vals = [post_report['Abnormal'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, pre_vals, width, label='Pre-IL (Detection)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, post_vals, width, label='Post-IL (Evaluation)', color='#2ecc71')
    
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} Performance Change - {session_name} (Target: Abnormal)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()

def plot_unknown_breakdown(total_actual, total_predicted, true_detected, label_name, save_path):
    """Vẽ biểu đồ trực quan hóa khả năng phát hiện Unknown"""
    categories = ['Actual Unknowns', 'Predicted Unknowns', 'Correctly Detected (TP)']
    values = [total_actual, total_predicted, true_detected]
    colors = ['#95a5a6', '#f1c40f', '#27ae60'] # Xám (Gốc), Vàng (Dự đoán), Xanh (Đúng)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Tính Recall và Precision để hiển thị
    recall = true_detected / total_actual if total_actual > 0 else 0
    prec = true_detected / total_predicted if total_predicted > 0 else 0
    
    plt.title(f"Unknown Detection Breakdown - {label_name}\nRecall: {recall:.2%} | Precision: {prec:.2%}")
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()

# ==================== EVALUATION WRAPPERS ====================
def evaluate_supervised_model(y_true, y_pred, sess, save_dir):
    print(f"\n--- [METRICS] XGBoost Only - {sess} ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    plot_cm(y_true, y_pred, f"CM XGBoost - {sess}", f"{save_dir}/cm_xgb_{sess}.png")

def evaluate_final_pipeline(y_true, y_pred, sess, save_dir):
    print(f"\n--- [METRICS] Final Pipeline - {sess} ---")
    y_str = [("BENIGN" if y==0 else f"KNOWN_ATTACK_{y}") for y in y_true]
    print(classification_report(y_str, y_pred, digits=4, zero_division=0))
    plot_cm(y_str, y_pred, f"CM Pipeline - {sess}", f"{save_dir}/cm_pipe_{sess}.png")

def evaluate_unsupervised_detailed(y_true, ae_pred, ocsvm_pred, sess, save_dir, return_dict=False):
    # 1=Normal, 0=Abnormal
    y_bin = (y_true == 0).astype(int) 
    
    print("\n>>> Autoencoder (Target: Detect Abnormal):")
    rep_ae = classification_report(y_bin, ae_pred, target_names=['Abnormal', 'Normal'], digits=4, output_dict=True)
    print(classification_report(y_bin, ae_pred, target_names=['Abnormal', 'Normal'], digits=4))
    plot_cm(y_bin, ae_pred, f"CM AE - {sess}", f"{save_dir}/cm_ae_{sess}.png", labels=[0, 1])
            
    print("\n>>> OCSVM (Target: Detect Abnormal):")
    rep_oc = classification_report(y_bin, ocsvm_pred, target_names=['Abnormal', 'Normal'], digits=4, output_dict=True)
    print(classification_report(y_bin, ocsvm_pred, target_names=['Abnormal', 'Normal'], digits=4))
    plot_cm(y_bin, ocsvm_pred, f"CM OCSVM - {sess}", f"{save_dir}/cm_ocsvm_{sess}.png", labels=[0, 1])
    
    if return_dict: return rep_ae, rep_oc