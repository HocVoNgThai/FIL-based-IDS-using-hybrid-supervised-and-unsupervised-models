# src/utils.py
import os
import time
import threading
import psutil
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings

# Thử import GPUtil để theo dõi GPU
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================== 1. RESOURCE TRACKER (CPU/RAM/GPU) ====================
class ResourceTracker:
    def __init__(self):
        self.tracking = False
        self.stats = {'cpu': [], 'ram': [], 'gpu': []}
        self.thread = None
        self.start_time = 0

    def _monitor(self):
        while self.tracking:
            self.stats['cpu'].append(psutil.cpu_percent())
            self.stats['ram'].append(psutil.virtual_memory().percent)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus: self.stats['gpu'].append(gpus[0].load * 100)
                    else: self.stats['gpu'].append(0)
                except: self.stats['gpu'].append(0)
            time.sleep(0.5) # Sample mỗi 0.5s

    def start(self):
        self.tracking = True
        self.stats = {'cpu': [], 'ram': [], 'gpu': []}
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        self.start_time = time.time()

    def stop(self):
        self.tracking = False
        if self.thread: self.thread.join()
        duration = time.time() - self.start_time
        
        avg_cpu = np.mean(self.stats['cpu']) if self.stats['cpu'] else 0
        max_ram = np.max(self.stats['ram']) if self.stats['ram'] else 0
        avg_gpu = np.mean(self.stats['gpu']) if self.stats['gpu'] else 0
        
        return {
            'duration': duration,
            'avg_cpu': avg_cpu,
            'max_ram': max_ram,
            'avg_gpu': avg_gpu
        }

def plot_resource_usage(log_data, save_path):
    """Vẽ biểu đồ tài nguyên tiêu thụ"""
    sessions = list(log_data.keys())
    times = [log_data[s]['duration'] for s in sessions]
    cpus = [log_data[s]['avg_cpu'] for s in sessions]
    gpus = [log_data[s]['avg_gpu'] for s in sessions]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar chart cho thời gian
    ax1.bar(sessions, times, color='skyblue', alpha=0.6, label='Time (s)')
    ax1.set_ylabel('Duration (seconds)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Line chart cho CPU/GPU
    ax2 = ax1.twinx()
    ax2.plot(sessions, cpus, color='red', marker='o', linewidth=2, label='Avg CPU %')
    ax2.plot(sessions, gpus, color='green', marker='s', linewidth=2, label='Avg GPU %')
    ax2.set_ylabel('Usage (%)', color='black')
    ax2.set_ylim(0, 100)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Resource Consumption & Execution Time per Session')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

# ==================== 2. IL METRICS (BWT/Forgetting) ====================
class ILMetrics:
    def __init__(self):
        # R[i][j] = Accuracy của model sau khi học xong session i, test trên dữ liệu session j
        self.R = {} 

    def record(self, train_session, test_session, accuracy):
        if train_session not in self.R: self.R[train_session] = {}
        self.R[train_session][test_session] = accuracy

    def calculate_bwt(self, current_session):
        """
        Backward Transfer (BWT): Đo mức độ quên kiến thức cũ.
        BWT < 0 nghĩa là có Catastrophic Forgetting.
        """
        if current_session == 0: return 0
        bwt_sum = 0
        for i in range(current_session):
            acc_now = self.R[current_session][i]
            acc_orig = self.R[i][i] # Acc lúc mới học xong task i
            bwt_sum += (acc_now - acc_orig)
        return bwt_sum / current_session

    def get_matrix_plot_data(self):
        # Convert dict to matrix for heatmap
        sessions = sorted(self.R.keys())
        n = len(sessions)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j <= i: # Chỉ điền tam giác dưới (đã học rồi mới test được)
                    matrix[i, j] = self.R[i].get(j, 0)
        return matrix, sessions

def plot_il_matrix(il_metrics, save_path):
    matrix, sessions = il_metrics.get_matrix_plot_data()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu', 
                xticklabels=[f'Test S{s}' for s in sessions], 
                yticklabels=[f'Train S{s}' for s in sessions])
    plt.title('Incremental Learning Accuracy Matrix\n(Lower Triangle should be high)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

# ==================== 3. VISUALIZATION FUNCTIONS (ĐÃ CÓ + MỚI) ====================

def get_label_name(y_val):
    try: val = int(y_val)
    except: return str(y_val)
    mapping = {0: "BENIGN", 1: "DDoS", 2: "DoS", 3: "Reconn", 4: "Vuln Scan"}
    return mapping.get(val, "UNKNOWN")

def plot_unknown_binary_cm(y_true, preds, unknown_label, save_path):
    """CM cho khả năng phát hiện Unknown (Target vs Others)"""
    y_true = np.array(y_true); preds = np.array(preds)
    # 1 = Unknown (Target), 0 = Others (Benign/Known)
    y_bin_true = (y_true == unknown_label).astype(int)
    y_bin_pred = (preds == "UNKNOWN").astype(int)
    
    cm = confusion_matrix(y_bin_true, y_bin_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Oranges', 
                xticklabels=['Others', 'Unknown'], yticklabels=['Others', 'Unknown'], vmin=0, vmax=1)
    plt.title(f"Unknown Detection (Phase 1)\nTarget: {get_label_name(unknown_label)}")
    plt.ylabel('True Class'); plt.xlabel('Predicted Class')
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_gray_zone_analysis(y_true, xgb_pred, xgb_conf, ae_pred, ocsvm_pred, conf_min, conf_max, session_name, save_dir):
    """Vẽ CM cho AE và OCSVM chỉ trên các mẫu thuộc vùng Grey Zone"""
    print(f"\n--- [ANALYSIS] Gray Zone Analysis ({conf_min} <= Conf < {conf_max}) ---")
    
    # Lấy index các mẫu mà XGBoost đoán là BENIGN nhưng độ tin cậy thấp (Gray Zone)
    mask = (xgb_pred == 0) & (xgb_conf >= conf_min) & (xgb_conf < conf_max)
    
    if np.sum(mask) == 0:
        print("No samples in Gray Zone.")
        return

    y_gray = y_true[mask]
    ae_gray = ae_pred[mask]
    ocsvm_gray = ocsvm_pred[mask]
    
    # Binary: 0 (Benign) -> 1 (Normal), Others (Attack) -> 0 (Abnormal)
    # Để khớp với output của AE/OCSVM (1=Inlier, 0=Outlier)
    y_bin_gray = (y_gray == 0).astype(int) 
    
    # Vẽ CM cho AE
    cm_ae = confusion_matrix(y_bin_gray, ae_gray, labels=[0, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_ae_norm = cm_ae.astype('float') / cm_ae.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(np.nan_to_num(cm_ae_norm), annot=True, fmt='.1%', cmap='Reds', 
                xticklabels=['Abn', 'Nor'], yticklabels=['Attack', 'Benign'], vmin=0, vmax=1)
    plt.title(f"AE Performance on Gray Zone\n(Session {session_name})")
    
    # Vẽ CM cho OCSVM
    cm_oc = confusion_matrix(y_bin_gray, ocsvm_gray, labels=[0, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_oc_norm = cm_oc.astype('float') / cm_oc.sum(axis=1)[:, np.newaxis]
        
    plt.subplot(1, 2, 2)
    sns.heatmap(np.nan_to_num(cm_oc_norm), annot=True, fmt='.1%', cmap='Greens', 
                xticklabels=['Abn', 'Nor'], yticklabels=['Attack', 'Benign'], vmin=0, vmax=1)
    plt.title(f"OCSVM Performance on Gray Zone\n(Session {session_name})")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_dir), exist_ok=True); plt.savefig(f"{save_dir}/gray_zone_analysis_{session_name}.png"); plt.close()

# ... (Giữ nguyên SessionDataLoader, plot_cm, plot_metrics_bar, plot_comparison_chart) ...
# (Các hàm evaluate_supervised_with_unknown, evaluate_final_pipeline... giữ nguyên từ version trước)

# Copy lại SessionDataLoader từ code cũ
# ...
# Copy lại SessionManager từ code cũ
# ...
# Copy lại evaluate_... từ code cũ

# ==================== 3. DATA LOADER & SESSION MANAGER ====================
class SessionDataLoader:
    def __init__(self):
        self.scaler = StandardScaler(); self.feature_names = None; self.is_fitted = False
    def load_session_data(self, file_path, label_col='Label', binary_label_col='Binary Label', scaler_fit=False):
        if not os.path.exists(file_path): raise ValueError(f"File {file_path} not found")
        df = pd.read_parquet(file_path)
        y = None; cols = []
        if label_col in df.columns: cols.append(label_col); y = df[label_col].astype(int)
        if binary_label_col in df.columns: cols.append(binary_label_col)
        X = df.drop(columns=cols).fillna(0).values
        if scaler_fit: print("Fitting StandardScaler..."); X = self.scaler.fit_transform(X); self.is_fitted = True
        else: 
            if not self.is_fitted: self.load_scaler('sessions/session0/scaler.joblib')
            X = self.scaler.transform(X)
        return X, y
    def save_scaler(self, p): os.makedirs(os.path.dirname(p), exist_ok=True); joblib.dump(self.scaler, p)
    def load_scaler(self, p): self.scaler = joblib.load(p); self.is_fitted = True

class SessionManager:
    def __init__(self, base_dir='sessions'):
        self.base_dir = base_dir; self.current_session = 0; self.session_info = {}
        os.makedirs(base_dir, exist_ok=True)
        for i in range(3): os.makedirs(f"{base_dir}/session{i}", exist_ok=True)
        self.load_session_info()
    def initialize_session_0(self, tr, te): self.current_session=0; self.session_info[0]={'tr':tr, 'te':te, 'lbs':[0,1,2]}; self.save_session_info()
    def advance_to_session_1(self, tr, te): self.current_session=1; self.session_info[1]={'tr':tr, 'te':te, 'lbs':[0,1,2,3]}; self.save_session_info()
    def advance_to_session_2(self, tr, te): self.current_session=2; self.session_info[2]={'tr':tr, 'te':te, 'lbs':[0,1,2,3,4]}; self.save_session_info()
    def save_models(self, models, sid):
        p = f"{self.base_dir}/session{sid}/models"; os.makedirs(p, exist_ok=True)
        for k, v in models.items():
            if hasattr(v, 'save_model'): v.save_model(f"{p}/{k}")
            else: joblib.dump(v, f"{p}/{k}.joblib")
    def load_models(self, sid, models):
        p = f"{self.base_dir}/session{sid}/models"
        for k, v in models.items():
            if hasattr(v, 'load_model'): v.load_model(f"{p}/{k}")
            elif os.path.exists(f"{p}/{k}.joblib"): models[k] = joblib.load(f"{p}/{k}.joblib")
        return models
    def save_session_info(self): joblib.dump({'c': self.current_session, 'i': self.session_info}, f"{self.base_dir}/session_info.joblib")
    def load_session_info(self): 
        if os.path.exists(f"{self.base_dir}/session_info.joblib"): d=joblib.load(f"{self.base_dir}/session_info.joblib"); self.current_session=d['c']; self.session_info=d['i']

# ==================== 4. VISUALIZATION FUNCTIONS ====================
def get_label_name(y_val):
    try: val = int(y_val)
    except: return str(y_val)
    mapping = {0: "BENIGN", 1: "DDoS", 2: "DoS", 3: "Reconn", 4: "Vuln Scan"}
    return mapping.get(val, "UNKNOWN")

def plot_cm(y_true, y_pred, title, save_path, labels=None):
    if labels is None: unique_labels = sorted(list(set(y_true) | set(y_pred)))
    else: unique_labels = labels
    ordered_labels = []
    if "BENIGN" in unique_labels: ordered_labels.append("BENIGN")
    others = [l for l in unique_labels if l not in ["BENIGN", "UNKNOWN"]]
    ordered_labels.extend(sorted(others))
    if "UNKNOWN" in unique_labels: ordered_labels.append("UNKNOWN")
    
    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=ordered_labels, yticklabels=ordered_labels, vmin=0, vmax=1)
    plt.title(f"{title} (%)"); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_binary_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', xticklabels=['Abnormal', 'Normal'], yticklabels=['Abnormal', 'Normal'], vmin=0, vmax=1)
    plt.title(f"{title} (%)"); plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_metrics_bar(report_dict, title, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    values = [report_dict['weighted avg'][m] for m in metrics]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    for bar in bars:
        height = bar.get_height() # === FIX LỖI: Định nghĩa biến height
        plt.text(bar.get_x()+bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    plt.title(title); plt.ylim(0, 1.1); plt.ylabel('Score'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_unknown_breakdown(total_actual, total_predicted, true_detected, label_name, save_path):
    cats = ['Actual Unknowns', 'Predicted Unknowns', 'Correctly Detected (TP)']
    vals = [total_actual, total_predicted, true_detected]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(cats, vals, color=['#95a5a6', '#f1c40f', '#27ae60'])
    for bar in bars:
        height = bar.get_height() # === Định nghĩa biến height
        plt.text(bar.get_x()+bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
    rec = true_detected/total_actual if total_actual>0 else 0
    prec = true_detected/total_predicted if total_predicted>0 else 0
    plt.title(f"Unknown Detection - {label_name}\nRecall: {rec:.2%} | Precision: {prec:.2%}")
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_unknown_binary_cm(y_true, preds, unknown_label, save_path):
    y_true = np.array(y_true); preds = np.array(preds)
    y_bin_true = (y_true == unknown_label).astype(int)
    y_bin_pred = (preds == "UNKNOWN").astype(int)
    labels = ['Others', 'Unknown']
    cm = confusion_matrix(y_bin_true, y_bin_pred)
    with np.errstate(divide='ignore', invalid='ignore'): cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Oranges', xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.title(f"Detection Rate: {get_label_name(unknown_label)}")
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

def plot_comparison_chart(metrics_data, session_name, save_path):
    models = list(metrics_data.keys())
    pre_scores = [metrics_data[m].get('Pre', 0) for m in models]
    post_scores = [metrics_data[m].get('Post', 0) for m in models]
    x = np.arange(len(models)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pre_scores, width, label='Pre-IL', color='#95a5a6')
    rects2 = ax.bar(x + width/2, post_scores, width, label='Post-IL', color='#27ae60')
    ax.set_ylabel('Weighted F1-Score'); ax.set_title(f'Model Improvement - {session_name}')
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0, 1.1); ax.legend()
    for bar in rects1:
         height = bar.get_height()
         ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
    for bar in rects2:
         height = bar.get_height()
         ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path); plt.close()

# ==================== EVALUATION WRAPPERS ====================
def calculate_unknown_metrics(y_true, preds, unknown_label, save_dir, session_name):
    y_true = np.array(y_true); preds = np.array(preds)
    is_actual = (y_true == unknown_label); total_actual = np.sum(is_actual)
    is_pred = (preds == "UNKNOWN"); total_pred = np.sum(is_pred)
    tp = np.sum(is_actual & is_pred)
    print(f"\n   [METRICS] Label {unknown_label} ({get_label_name(unknown_label)}): Found {tp}/{total_actual} (Recall: {tp/total_actual if total_actual>0 else 0:.4f})")
    print(f"   [METRICS] Precision: {tp/total_pred if total_pred>0 else 0:.4f}")
    plot_unknown_binary_cm(y_true, preds, unknown_label, f"{save_dir}/unknown_cm_{session_name}.png")

def evaluate_supervised_with_unknown(y_true, y_pred, y_conf, threshold, session_name, save_dir, model_name="XGBoost", target_unknown_label=None):
    print(f"\n--- [METRICS] {model_name} w/ Unknown Thres ({threshold}) - {session_name} ---")
    y_str_true = []
    for y in y_true:
        if target_unknown_label is not None and y == target_unknown_label: y_str_true.append("UNKNOWN")
        else: y_str_true.append(get_label_name(y))
    y_str_pred = []
    for p, c in zip(y_pred, y_conf):
        if c < threshold: y_str_pred.append("UNKNOWN")
        else: y_str_pred.append(get_label_name(p))
            
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} (Unknown) - {session_name}", f"{save_dir}/cm_{model_name}_unknown_{session_name}.png")
    rep = classification_report(y_str_true, y_str_pred, output_dict=True, zero_division=0)
    plot_metrics_bar(rep, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name}_unknown_{session_name}.png")
    return rep['weighted avg']['f1-score']

def evaluate_supervised_model(y_true, y_pred, session_name, save_dir, model_name="Supervised", return_f1=False):
    print(f"\n--- [METRICS] {model_name} - {session_name} ---")
    y_str = [get_label_name(y) for y in y_true]
    y_pred_str = [get_label_name(y) for y in y_pred]
    print(classification_report(y_str, y_pred_str, digits=4, zero_division=0))
    plot_cm(y_str, y_pred_str, f"CM {model_name} - {session_name}", f"{save_dir}/cm_{model_name.lower().replace(' ', '_')}_{session_name.replace(' ', '_')}.png")
    report = classification_report(y_str, y_pred_str, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(report, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name.lower().replace(' ', '_')}_{session_name.replace(' ', '_')}.png")
    if return_f1: return report['weighted avg']['f1-score']

def evaluate_final_pipeline(y_true, y_pred, sess, save_dir, return_f1=False):
    print(f"\n--- [METRICS] Final Pipeline - {sess} ---")
    y_str = [get_label_name(y) for y in y_true]
    print(classification_report(y_str, y_pred, digits=4, zero_division=0))
    plot_cm(y_str, y_pred, f"CM Pipeline - {sess}", f"{save_dir}/cm_pipe_{sess}.png")
    rep = classification_report(y_str, y_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(rep, f"Metrics Pipeline - {sess}", f"{save_dir}/metrics_pipe_{sess}.png")
    if return_f1: return rep['weighted avg']['f1-score']

def evaluate_unsupervised_detailed(y_true, ae_pred, ocsvm_pred, sess, save_dir, return_f1=False):
    y_bin = (y_true == 0).astype(int) 
    print("\n>>> Autoencoder:"); print(classification_report(y_bin, ae_pred, target_names=['Abnormal', 'Normal'], digits=4))
    plot_binary_cm(y_bin, ae_pred, f"CM AE - {sess}", f"{save_dir}/cm_ae_{sess}.png")
    ae_f1 = f1_score(y_bin, ae_pred, average='weighted')
            
    print("\n>>> OCSVM:"); print(classification_report(y_bin, ocsvm_pred, target_names=['Abnormal', 'Normal'], digits=4))
    plot_binary_cm(y_bin, ocsvm_pred, f"CM OCSVM - {sess}", f"{save_dir}/cm_ocsvm_{sess}.png")
    oc_f1 = f1_score(y_bin, ocsvm_pred, average='weighted')
    if return_f1: return ae_f1, oc_f1

def evaluate_gray_zone(y_true, xgb_pred, xgb_conf, ae_pred, ocsvm_pred, conf_min, conf_max, session_name, save_dir):
    print(f"\n--- [ANALYSIS] Gray Zone ({conf_min} <= Conf < {conf_max}) ---")
    gray_indices = [i for i in range(len(y_true)) if xgb_pred[i] == 0 and xgb_conf[i] >= conf_min and xgb_conf[i] < conf_max]
    if len(gray_indices) == 0: print("No samples in Gray Zone."); return
    y_gray = y_true[gray_indices]; ae_gray = ae_pred[gray_indices]; ocsvm_gray = ocsvm_pred[gray_indices]
    y_bin = (y_gray == 0).astype(int)
    print(">>> AE Gray Zone:"); print(classification_report(y_bin, ae_gray, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, ae_gray, f"CM AE Gray - {session_name}", f"{save_dir}/cm_ae_gray_{session_name}.png")
    print(">>> OCSVM Gray Zone:"); print(classification_report(y_bin, ocsvm_gray, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, ocsvm_gray, f"CM OCSVM Gray - {session_name}", f"{save_dir}/cm_ocsvm_gray_{session_name}.png")