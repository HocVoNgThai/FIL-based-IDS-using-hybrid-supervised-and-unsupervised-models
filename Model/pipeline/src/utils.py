# src/utils.py
import os
import time
import psutil
import threading
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings

# Thử import GPUtil để theo dõi GPU, nếu không có thì bỏ qua
try: 
    import GPUtil
except ImportError: 
    GPUtil = None

warnings.filterwarnings('ignore')

# ==================== 1. RESOURCE TRACKER ====================
class ResourceTracker:
    def __init__(self):
        self.tracking = False
        self.stats = {'cpu': [], 'ram': [], 'gpu': []}
        self.thread = None

    def _monitor(self):
        while self.tracking:
            self.stats['cpu'].append(psutil.cpu_percent())
            self.stats['ram'].append(psutil.virtual_memory().percent)
            try:
                if GPUtil and GPUtil.getGPUs(): 
                    self.stats['gpu'].append(GPUtil.getGPUs()[0].load * 100)
                else: 
                    self.stats['gpu'].append(0)
            except: 
                self.stats['gpu'].append(0)
            time.sleep(0.5)

    def start(self):
        self.tracking = True
        self.stats = {'cpu': [], 'ram': [], 'gpu': []}
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        self.start_time = time.time()

    def stop(self):
        self.tracking = False
        if self.thread: 
            self.thread.join()
        duration = time.time() - self.start_time
        
        return {
            'duration': duration, 
            'avg_cpu': np.mean(self.stats['cpu']) if self.stats['cpu'] else 0, 
            'max_ram': np.max(self.stats['ram']) if self.stats['ram'] else 0, 
            'avg_gpu': np.mean(self.stats['gpu']) if self.stats['gpu'] else 0
        }

def plot_resource_usage(log, save_path):
    cases = list(log.keys())
    times = [log[s]['duration'] for s in cases]
    cpus = [log[s]['avg_cpu'] for s in cases]
    rams = [log[s]['max_ram'] for s in cases]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. Vẽ biểu đồ cột (Time)
    ax1.bar(cases, times, color='skyblue', alpha=0.5, label='Time (s)')
    ax1.set_ylabel('Time (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 2. Vẽ biểu đồ đường (CPU & RAM) trên trục phụ
    ax2 = ax1.twinx()
    ax2.plot(cases, cpus, 'r-o', label='CPU %', linewidth=2)   # Đỏ = CPU
    ax2.plot(cases, rams, 'g-x', label='RAM %', linewidth=2)   # Xanh lá = RAM
    
    ax2.set_ylabel('Usage %', color='black')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 3. Gộp Legend lại cho gọn
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    plt.title('Resource Usage Performance')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ==================== 2. IL METRICS ====================
class ILMetrics:
    def __init__(self): 
        self.R = {}

    def record(self, tr_sess, te_sess, acc):
        if tr_sess not in self.R: 
            self.R[tr_sess] = {}
        self.R[tr_sess][te_sess] = acc

    def calculate_bwt(self, curr):
        if curr == 0: return 0.0
        return np.mean([self.R[curr][i] - self.R[i][i] for i in range(curr)])

def plot_il_metrics(il, save_path):
    cases = sorted(il.R.keys())
    n = len(cases)
    if n == 0: return
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n): 
            matrix[i, j] = il.R[i].get(j, 0)
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu', 
                xticklabels=[f'Test C{s}' for s in cases], 
                yticklabels=[f'Train C{s}' for s in cases])
    plt.title('IL Accuracy Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ==================== 3. DATA & SESSION ====================
class SessionDataLoader:
    def __init__(self): 
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_data_raw(self, path, label_col='Label', binary_label_col='Binary Label'):
        if not os.path.exists(path): 
            raise ValueError(f"{path} not found")
        
        df = pd.read_parquet(path)
        y = df[label_col].astype(int) if label_col in df else None
        
        cols = [c for c in [label_col, binary_label_col] if c in df]
        X = df.drop(columns=cols).fillna(0).values
        return X, y

    def apply_scaling(self, X, fit=False):
        if fit:
            print("Fitting Global Scaler...")
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            return X_scaled
        else:
            if not self.is_fitted:
                try: 
                    self.load_scaler('sessions/global_scaler.joblib')
                except: 
                    pass
            if not self.is_fitted: 
                # Fallback nếu không load được (tránh crash, nhưng nên cảnh báo)
                print("Warning: Scaler not fitted/loaded! Using raw data.")
                return X
            return self.scaler.transform(X)

    def save_scaler(self, p): 
        os.makedirs(os.path.dirname(p), exist_ok=True)
        joblib.dump(self.scaler, p)
        print(f"Scaler saved to {p}")

    def load_scaler(self, p): 
        self.scaler = joblib.load(p)
        self.is_fitted = True
        print(f"Loaded Scaler from {p}")

class SessionManager:
    def __init__(self, base='sessions'):
        self.base = base
        self.curr = 0
        self.info = {}
        os.makedirs(base, exist_ok=True)
        for i in range(3): 
            os.makedirs(f"{base}/case{i}", exist_ok=True)

    def initialize_case_0(self, tr, te): self.curr=0; self.info[0]={'tr':tr,'te':te}; self.save()
    def advance_to_case_1(self, tr, te): self.curr=1; self.info[1]={'tr':tr,'te':te}; self.save()
    def advance_to_case_2(self, tr, te): self.curr=2; self.info[2]={'tr':tr,'te':te}; self.save()

    def save_models(self, models, sid):
        p = f"{self.base}/case{sid}/models"
        os.makedirs(p, exist_ok=True)
        for k, v in models.items():
            if hasattr(v, 'save_model'): 
                v.save_model(f"{p}/{k}")
            else: 
                joblib.dump(v, f"{p}/{k}.joblib")

    def load_models(self, sid, models):
        p = f"{self.base}/case{sid}/models"
        for k, v in models.items():
            if hasattr(v, 'load_model'): 
                v.load_model(f"{p}/{k}")
            elif os.path.exists(f"{p}/{k}.joblib"): 
                models[k] = joblib.load(f"{p}/{k}.joblib")
        return models

    def save(self): 
        joblib.dump({'c': self.curr, 'i': self.info}, f"{self.base}/info.joblib")

# ==================== 4. VISUALIZATION & EVALUATION ====================
def get_label_name(y):
    try: 
        val = int(y)
    except: 
        return str(y)
    return {
        0: "BENIGN", 
        1: "DDoS", 
        2: "DoS", 
        3: "Reconn", 
        4: "MITM", 
        5: "DNS Spoofing"
    }.get(val, "UNKNOWN")

def plot_cm(y_true, y_pred, title, save_path):
    labels = sorted(list(set(y_true) | set(y_pred)))
    ordered = [l for l in labels if l != "BENIGN" and l != "UNKNOWN"]
    if "BENIGN" in labels: ordered.insert(0, "BENIGN")
    if "UNKNOWN" in labels: ordered.append("UNKNOWN")
    
    cm = confusion_matrix(y_true, y_pred, labels=ordered)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=ordered, yticklabels=ordered, vmin=0, vmax=1)
    plt.title(f"{title} (%)")
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_binary_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', 
                xticklabels=['Abnormal', 'Normal'], 
                yticklabels=['Abnormal', 'Normal'], vmin=0, vmax=1)
    plt.title(f"{title} (%)")
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_unknown_binary_cm(y_true, preds, unknown_label, save_path):
    if isinstance(unknown_label, list):
        y_bin_true = np.isin(y_true, unknown_label).astype(int)
        label_text = f"List {unknown_label}"
    else:
        y_bin_true = (y_true == unknown_label).astype(int)
        label_text = get_label_name(unknown_label)
        
    y_bin_pred = (preds == "UNKNOWN").astype(int)
    cm = confusion_matrix(y_bin_true, y_bin_pred)
    
    print(f"\n{'-'*40}\n  UNKNOWN DETECTION TABLE: {label_text}\n{'-'*40}")
    try:
        print(f"True OTHER | {cm[0,0]:<6} | {cm[0,1]:<6} (Pred UNK)")
        print(f"True UNK   | {cm[1,0]:<6} | {cm[1,1]:<6} (Pred UNK)")
        tn, fp, fn, tp = cm.ravel()
        print(f"Recall: {tp/(tp+fn):.2%}")
    except: 
        pass

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Oranges', 
                xticklabels=['Others', 'Pred Unknown'], 
                yticklabels=['True Others', 'True Unknown'], vmin=0, vmax=1)
    plt.title(f"Detection: {label_text}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_bar(report_dict, title, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    try:
        values = [report_dict['weighted avg'][m] for m in metrics]
    except KeyError:
        values = [0, 0, 0]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
        
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_comparison_chart(metrics_data, session_name, save_path):
    models = list(metrics_data.keys())
    pre = [metrics_data[m].get('Pre', 0) for m in models]
    post = [metrics_data[m].get('Post', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pre, width, label='Pre-IL', color='#95a5a6')
    rects2 = ax.bar(x + width/2, post, width, label='Post-IL', color='#27ae60')
    
    ax.set_ylabel('Weighted F1')
    ax.set_title(f'Improvement - {session_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.2f')
    ax.bar_label(rects2, fmt='%.2f')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# [QUAN TRỌNG] Hàm wrapper này đã được thêm vào đầy đủ
def calculate_unknown_metrics(y_true, preds, unknown_label, save_dir, session_name):
    # Wrapper gọi hàm vẽ và tính toán metrics cho unknown
    y_true = np.array(y_true)
    preds = np.array(preds)
    
    if isinstance(unknown_label, list):
        label_text = f"List {unknown_label}"
    else:
        label_text = get_label_name(unknown_label)
        
    print(f"\n[METRICS CHECK] Checking detection for: {label_text}")
    plot_unknown_binary_cm(y_true, preds, unknown_label, f"{save_dir}/unknown_cm_{session_name}.png")

def evaluate_supervised_with_unknown(y_true, y_pred, y_conf, atk_thres=0.7, ben_thres=0.7, session_name="", save_dir="", model_name="XGBoost", target_unknown=None):
    print(f"\n--- [METRICS] {model_name} Dual-Thres (Atk>={atk_thres}, Ben>={ben_thres}) - {session_name} ---")
    
    def map_label(y):
        if isinstance(target_unknown, list):
            return "UNKNOWN" if y in target_unknown else get_label_name(y)
        elif target_unknown is not None:
            return "UNKNOWN" if y == target_unknown else get_label_name(y)
        else:
            return get_label_name(y)

    y_str_true = [map_label(y) for y in y_true]
    y_str_pred = []
    
    stats = {"unk_atk": 0, "unk_ben": 0}
    
    for p, c in zip(y_pred, y_conf):
        p_val = int(p)
        if p_val != 0:
            if c < atk_thres: 
                y_str_pred.append("UNKNOWN")
                stats["unk_atk"] += 1
            else:
                y_str_pred.append(get_label_name(p_val))
        else:
            if c < ben_thres: 
                y_str_pred.append("UNKNOWN")
                stats["unk_ben"] += 1
            else:
                y_str_pred.append("BENIGN")

    print(f"   -> XGBoost flagged UNKNOWN: {stats['unk_atk']} (Low Conf Atk) + {stats['unk_ben']} (Low Conf Ben)")

    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} (Unknown) - {session_name}", f"{save_dir}/cm_{model_name}_unknown_{session_name}.png")
    
    rep = classification_report(y_str_true, y_str_pred, output_dict=True, zero_division=0)
    plot_metrics_bar(rep, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name}_unknown_{session_name}.png")
    
    return rep['weighted avg']['f1-score']

def evaluate_supervised_model(y_true, y_pred, session_name, save_dir, model_name="Supervised", return_f1=False):
    print(f"\n--- [METRICS] {model_name} - {session_name} ---")
    
    y_str_true = [get_label_name(y) for y in y_true]
    y_str_pred = [get_label_name(y) for y in y_pred]
    
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} - {session_name}", f"{save_dir}/cm_{model_name.lower().replace(' ', '_')}_{session_name}.png")
    
    report = classification_report(y_str_true, y_str_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(report, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name.lower().replace(' ', '_')}_{session_name}.png")
    
    if return_f1: 
        return report['weighted avg']['f1-score']

def evaluate_final_pipeline(y_true, y_pred, sess, save_dir, return_f1=False):
    print(f"\n--- [METRICS] Final Pipeline - {sess} ---")
    
    y_str = [get_label_name(y) for y in y_true]
    print(classification_report(y_str, y_pred, digits=4, zero_division=0))
    
    plot_cm(y_str, y_pred, f"CM Pipeline - {sess}", f"{save_dir}/cm_pipe_{sess}.png")
    
    rep = classification_report(y_str, y_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(rep, f"Metrics Pipeline - {sess}", f"{save_dir}/metrics_pipe_{sess}.png")
    
    if return_f1: 
        return rep['weighted avg']['f1-score']

def evaluate_unsupervised_detailed(y_true, ae_pred, ocsvm_pred, sess, save_dir, return_f1=False):
    y_bin = (y_true == 0).astype(int) 
    
    print("\n>>> AE (Target: 0=Abn, 1=Nor):")
    print(classification_report(y_bin, ae_pred, target_names=['Abn', 'Nor'], digits=4))
    plot_binary_cm(y_bin, ae_pred, f"AE - {sess}", f"{save_dir}/cm_ae_{sess}.png")
    ae_f1 = f1_score(y_bin, ae_pred, average='weighted')
    
    print("\n>>> OCSVM (Target: 0=Abn, 1=Nor):")
    print(classification_report(y_bin, ocsvm_pred, target_names=['Abn', 'Nor'], digits=4))
    plot_binary_cm(y_bin, ocsvm_pred, f"OCSVM - {sess}", f"{save_dir}/cm_ocsvm_{sess}.png")
    oc_f1 = f1_score(y_bin, ocsvm_pred, average='weighted')
    
    if return_f1: 
        return ae_f1, oc_f1

def evaluate_gray_zone(y_true, xgb_pred, xgb_conf, ae_pred, ocsvm_pred, c_min, c_max, sess, save_dir):
    print(f"\n--- [ANALYSIS] Gray Zone ({c_min} <= Conf < {c_max}) ---")
    
    mask = (xgb_pred == 0) & (xgb_conf >= c_min) & (xgb_conf < c_max)
    
    if np.sum(mask) == 0: 
        print("No samples in Gray Zone.")
        return
        
    y_g = y_true[mask]
    ae_g = ae_pred[mask]
    oc_g = ocsvm_pred[mask]
    
    y_bin = (y_g == 0).astype(int)
    
    print(">>> AE Gray:")
    print(classification_report(y_bin, ae_g, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, ae_g, f"AE Gray - {sess}", f"{save_dir}/cm_ae_gray_{sess}.png")
    
    print(">>> OCSVM Gray:")
    print(classification_report(y_bin, oc_g, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, oc_g, f"OCSVM Gray - {sess}", f"{save_dir}/cm_ocsvm_gray_{sess}.png")