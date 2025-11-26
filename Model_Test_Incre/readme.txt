tôi đang thử nghiệm incremental learning với sgdocsvm (mô hình unsupervised). Ý tưởng của tôi là ban đầu sẽ train mô hình này với 3 nhãn 0,1,2 , lần thứ 2 sẽ tăng thêm nhãn 3, lần 3 là nhãn 4.



Lần đầu chỉ train, lần thứ 2 sẽ test với bộ dữ liệu đã thêm nhãn 3, rồi mới partial fit train_2, tương tự cho lần thứ 3 với bộ dữ liệu có thêm nhãn thứ 4.



Nhưng khi chạy thử pipeline kết quả nhận biết outlier cho các nhãn mới thêm vào rất tệ 

dir_in = [f"C:/Users/hoang/Documents/Dataset_KLTN/ciciot2023_extracted/merge-processed/Incremental_1.3/session{i}.parquet" for i in range(0, 3)]
dir_in_train = [f"C:/Users/hoang/Documents/Dataset_KLTN/ciciot2023_extracted/merge-processed/Incremental_1.3/session{i}_train.parquet" for i in range(0, 3)]
dir_in_test = [f"C:/Users/hoang/Documents/Dataset_KLTN/ciciot2023_extracted/merge-processed/Incremental_1.3/session{i}_test.parquet" for i in range(1, 3)]
gc.collect()

# ====== INITIALIZE =====
# ------ OCSVM ------
NU_PARAM = 0.1

# ====== FUNCTION =====
dtypes = {}    
with open('features.json') as json_file:
    data = json.load(json_file)
    for key, type in data.items():
        if type == "int8":
            dtypes[key]= np.int8
        elif type == "float32":
            dtypes[key] = np.float32
    
    json_file.close()

print(dtypes)

def astype(df):
    for key, type in df.dtypes.items():
        # print(f"Key: {key} \t {type}")
        if type == "int8":
            df[key] = df[key].astype(np.int8)
        elif type == "float32":
            df[key] = df[key].astype(np.float32)
            
    return df

# ============================================================
# ==== INCREMENTAL PIPELINE ====
def incremental_pipeline(train_files, test_files):
    results = []
    clf = SGDOneClassSVM(nu=NU_PARAM,
                         learning_rate='optimal',
                         max_iter=100,
                         tol=1e-3)

    # assert len(train_files) == len(test_files), "Train/Test mismatch!"

    # TURN 0
    df_train = pd.read_parquet(train_files[0])
    df_train = astype(df_train)
    X_train = df_train.drop(["Label", "Binary Label"], axis=1)

    clf.fit(X_train)
    
    del df_train, X_train
    gc.collect()
    
    for step in range(len(test_files)):
        print(f"\n========== Incremental Step {step + 1} ==========")

        # --- LOAD TRAIN & TEST ---
        df_train = pd.read_parquet(train_files[step + 1])
        df_test  = pd.read_parquet(test_files[step])
        df_train = astype(df_train)
        df_test  = astype(df_test)

        print(f"[TRAIN] {df_train['Binary Label'].value_counts().to_dict()}")
        print(f"[TEST]  {df_test['Binary Label'].value_counts().to_dict()}")

        X_train = df_train.drop(["Label", "Binary Label"], axis=1)
        X_test  = df_test.drop(["Label", "Binary Label"], axis=1)
        y_test  = df_test["Binary Label"]

        # --- TEST ---
        y_score = -clf.decision_function(X_test)
        threshold = np.percentile(y_score, 95)
        y_pred = (y_score > threshold).astype(int)

        # Label thật: Binary Label (0 = normal, 1 = attack)
        base_classes = [0]
        y_true = np.isin(y_test, base_classes, invert=True).astype(int)

        # --- METRICS ---
        auc = roc_auc_score(y_true, y_score)
        f1  = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred)

        metrics = {
            "step": step,
            "AUC": auc,
            "F1": f1,
            "Precision": pre,
            "Recall": rec,
            "Threshold": threshold,
            "ConfusionMatrix": cm
        }
        print(metrics)
        results.append(metrics)
        
        clf.partial_fit(X_train)    

        gc.collect()

    return results

# ===== RUN =====
if __name__ == '__main__':
    print(input)
    
    results = incremental_pipeline(dir_in_train, dir_in_test)
    # Convert sang DataFrame (trừ CM)
    
    # Convert metrics (exclude CM) sang DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "ConfusionMatrix"} for r in results])

    # ============================================================
    # ==== PLOTTING ====

    # --- Biểu đồ AUC ---
    plt.figure(figsize=(7,4))
    plt.plot(results_df["step"], results_df["AUC"], marker="o", color="blue", linewidth=2)
    plt.title("Incremental AUC over Steps")
    plt.xlabel("Incremental Step")
    plt.ylabel("AUC Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # --- Biểu đồ Threshold ---
    plt.figure(figsize=(7,4))
    plt.plot(results_df["step"], results_df["Threshold"], marker="s", color="orange", linewidth=2)
    plt.title("Decision Threshold (95th percentile) per Step")
    plt.xlabel("Incremental Step")
    plt.ylabel("Threshold Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()