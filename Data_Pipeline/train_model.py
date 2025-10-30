import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler # <<< Quay lại StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import warnings

    
def train_model(DEVICE, model, train_loader, val_loader, epochs, patience, learning_rate, weight_decay, model_save_path, loss_fn):
    model.to(DEVICE)
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8) # Increased patience

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Bắt đầu huấn luyện model '{model.__class__.__name__}' với loss '{criterion.__class__.__name__}'...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for (data,) in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            reconstructed = output[0] if isinstance(output, tuple) else output
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * data.size(0)

        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(DEVICE)
                output = model(data)
                reconstructed = output[0] if isinstance(output, tuple) else output
                loss = criterion(reconstructed, data)
                val_loss += loss.item() * data.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping được kích hoạt tại epoch {epoch+1}.")
            break

    end_time = time.time()
    best_loss_str = f"{best_val_loss:.6f}" if best_val_loss != float('inf') else "N/A"
    print(f"Huấn luyện hoàn tất trong {end_time - start_time:.2f} giây. Best Val Loss: {best_loss_str}. Model tốt nhất đã lưu vào '{model_save_path}'./n")
    try:
        model.load_state_dict(torch.load(model_save_path))
    except FileNotFoundError: print(f"Warning: Could not load {model_save_path}.")
    except Exception as e: print(f"Warning: Error loading {model_save_path}: {e}.")
    return model

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        unique_preds=np.unique(y_pred); unique_true=np.unique(y_true); tn,fp,fn,tp=0,0,0,0
        if len(unique_true)==1:
            if unique_true[0]==0: tn=len(y_true);
            else: tp=len(y_true);
        elif len(unique_preds)==1:
             if unique_preds[0]==0: tn=np.sum(y_true==0); fn=np.sum(y_true==1);
             else: fp=np.sum(y_true==0); tp=np.sum(y_true==1);
        print("Warning: CM calculation issue.")
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    fnr = fn/(fn+tp) if (fn+tp)>0 else 0.0
    return accuracy, precision, recall, f1, fpr, fnr, (tn, fp, fn, tp)


def plot_evaluation(y_true, scores, y_pred, threshold, title):
    print(f"\n--- Đang vẽ biểu đồ cho: {title} ---")
    auc = 0.5
    try:
        y_true=np.asarray(y_true).astype(int); scores=np.asarray(scores); y_pred=np.asarray(y_pred).astype(int);
        if not np.issubdtype(scores.dtype, np.number): raise TypeError("scores non-numeric")
        scores=np.nan_to_num(scores, nan=np.nanmedian(scores), posinf=np.nanmax(scores[np.isfinite(scores)]), neginf=np.nanmin(scores[np.isfinite(scores)]))
        if not np.all(np.isfinite(scores)): raise ValueError("Non-finite scores remain")
        unique_labels_true = np.unique(y_true)
        if len(unique_labels_true)<2: print(f"Warning: Only one class ({unique_labels_true}). Cannot calc AUC.")
        else: auc=roc_auc_score(y_true, scores)
    except ValueError as e: print(f"Warning: Cannot calc AUC. Error: {e}")
    except TypeError as e: print(f"Error calc AUC: {e}")

    accuracy, precision, recall, f1, fpr, fnr, (tn, fp, fn, tp) = calculate_metrics(y_true, y_pred)
    print(f"AUC: {auc:.4f} | F1-Score: {f1:.4f}"); print(f"Precision: {precision:.4f} | Recall (TPR): {recall:.4f}"); print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}"); print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    plt.figure(figsize=(18, 6)); plt.suptitle(title, fontsize=16); plt.subplot(1, 3, 1)
    unique_labels_plot = np.unique(y_true)
    if len(unique_labels_plot) > 1:
        if 0 in unique_labels_plot: sns.histplot(scores[y_true==0], color='blue', label='Normal Scores', kde=True, bins=50, stat="density")
        if 1 in unique_labels_plot: sns.histplot(scores[y_true==1], color='red', label='Abnormal Scores', kde=True, bins=50, stat="density")
    elif len(unique_labels_plot) == 1: label_text = 'Normal' if unique_labels_plot[0] == 0 else 'Abnormal'; sns.histplot(scores, color='purple', label=f'{label_text} Scores Only', kde=True, bins=50, stat="density")
    else: plt.text(0.5, 0.5, 'No data', ha='center')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})'); plt.title('Phân phối Điểm/Lỗi Bất thường'); plt.legend(); plt.subplot(1, 3, 2)
    labels_cm = sorted(np.unique(y_true))
    if not np.all(np.isin(np.unique(y_pred), labels_cm)): print(f"Warning: Predictions contain labels not in y_true.")
    cm = confusion_matrix(y_true, y_pred, labels=labels_cm); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'{l}' for l in labels_cm], yticklabels=[f'{l}' for l in labels_cm]); plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.subplot(1, 3, 3)
    if auc > 0.5 and len(unique_labels_plot) > 1 :
        fpr_roc, tpr_roc, _ = roc_curve(y_true, scores); plt.plot(fpr_roc, tpr_roc, label=f'AUC = {auc:.4f}')
        if (tn + fp) > 0 and (tp + fn) > 0: plt.scatter(fpr, recall, marker='o', color='red', zorder=5, s=100, label=f'Operating Point\n(FPR={fpr:.2f}, TPR={recall:.2f})')
    else: plt.text(0.5, 0.5, 'Cannot draw ROC Curve', ha='center')
    plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True); plt.tight_layout(rect=[0, 0.03, 1, 0.93]); plt.savefig(f"evaluation_{title.replace(' ', '_')}.png"); plt.show()
