import pandas as pd
import glob
import os

def create_training_and_evaluation_sets(
    folder_path: str,
    training_output_path: str = 'unsupervised_training_dataset.csv',
    evaluation_output_path: str = 'evaluation_dataset.csv'
):

    try:
        # --- Bước 1: Đọc và gộp tất cả dữ liệu từ thư mục Parquet ---
        search_pattern = os.path.join(folder_path, '*.parquet')
        parquet_files = glob.glob(search_pattern)

        if not parquet_files:
            print(f"Lỗi: Không tìm thấy file .parquet nào trong '{folder_path}'.")
            return

        print(f"Tìm thấy {len(parquet_files)} file parquet. Bắt đầu đọc và gộp...")
        all_dfs = [pd.read_parquet(f) for f in parquet_files]
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Đã gộp thành công. Tổng số mẫu: {len(full_df)}")

        # --- Bước 2: Tách dữ liệu thành các mẫu bình thường và bất thường ---
        normal_df = full_df[full_df['label'] == 0]
        anomaly_df = full_df[full_df['label'] != 0]

        print(f"Số mẫu bình thường (label 0): {len(normal_df)}")
        print(f"Số mẫu bất thường (label != 0): {len(anomaly_df)}")

        # --- Bước 3: Kiểm tra xem có đủ dữ liệu không ---
        if len(normal_df) < 200000:
            print(f"Lỗi: Không đủ mẫu bình thường. Cần ít nhất 200,000 nhưng chỉ có {len(normal_df)}.")
            return
        if len(anomaly_df) < 100000:
            print(f"Lỗi: Không đủ mẫu bất thường. Cần ít nhất 100,000 nhưng chỉ có {len(anomaly_df)}.")
            return

        # --- Bước 4: Tạo bộ dữ liệu HUẤN LUYỆN ---
        print("\nBắt đầu tạo bộ dữ liệu huấn luyện...")
        # Lấy 100k mẫu bình thường đầu tiên
        training_samples = normal_df.sample(n=100000, random_state=42)
        
        # Loại bỏ các mẫu đã lấy để chúng không được dùng cho bộ đánh giá
        remaining_normal_df = normal_df.drop(training_samples.index)

        # Bỏ cột label
        training_set_final = training_samples.drop(columns=['label'])

        # Lưu file
        training_set_final.to_csv(training_output_path, index=False)
        print(f"Đã lưu thành công {len(training_set_final)} mẫu vào '{training_output_path}'")

        # --- Bước 5: Tạo bộ dữ liệu ĐÁNH GIÁ ---
        print("\nBắt đầu tạo bộ dữ liệu đánh giá...")
        # Lấy 100k mẫu bình thường từ phần còn lại và gán label = 1
        eval_normal_samples = remaining_normal_df.sample(n=100000, random_state=42)
        eval_normal_samples['label'] = 1

        # Lấy 100k mẫu bất thường và gán label = -1
        eval_anomaly_samples = anomaly_df.sample(n=100000, random_state=42)
        eval_anomaly_samples['label'] = -1
        
        # Gộp và xáo trộn
        evaluation_set = pd.concat([eval_normal_samples, eval_anomaly_samples])
        shuffled_evaluation_set = evaluation_set.sample(frac=1, random_state=42).reset_index(drop=True)

        # Lưu file
        shuffled_evaluation_set.to_csv(evaluation_output_path, index=False)
        print(f"Đã lưu thành công {len(shuffled_evaluation_set)} mẫu đã xáo trộn vào '{evaluation_output_path}'")
        
        print("\nHoàn tất!")

    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")


# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN NÀY
    input_folder = 'scaled_output_parquet' 

    # Tên file đầu ra (có thể giữ nguyên)
    training_file = 'unsupervised_training_dataset.csv'
    evaluation_file = 'evaluation_dataset.csv'

    # Gọi hàm để thực hiện
    create_training_and_evaluation_sets(input_folder, training_file, evaluation_file)