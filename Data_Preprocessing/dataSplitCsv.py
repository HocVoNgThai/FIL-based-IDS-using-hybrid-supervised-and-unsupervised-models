import pandas as pd
import glob
import os

def create_training_and_evaluation_sets(
    folder_path: str,
    training_output_path: str = 'unsupervised_training_dataset.csv',
    evaluation_output_path: str = 'evaluation_dataset.csv'
):
    """
    Tạo bộ dữ liệu huấn luyện (chỉ mẫu normal) và đánh giá (normal + anomaly)
    từ các file Parquet theo logic mới.

    Args:
        folder_path (str): Thư mục chứa các file Parquet.
        training_output_path (str): Tên file CSV cho dữ liệu huấn luyện.
        evaluation_output_path (str): Tên file CSV cho dữ liệu đánh giá.
    """
    try:
        # --- Bước 1: Đọc và gộp tất cả dữ liệu từ thư mục Parquet ---
        
        input_file = "C:/Users/hoang/Documents/Dataset_KLTN/scaled_output_parquet"
        
        # print(f"Tìm thấy {len(parquet_files)} file parquet. Bắt đầu đọc và gộp...")
        # all_dfs = [pd.read_parquet(f) for f in parquet_files]
        # full_df = pd.concat(all_dfs, ignore_index=True)
        # print(f"Đã gộp thành công. Tổng số mẫu: {len(full_df)}")
        full_df = pd.read_parquet(input_file)
        # full_df = pd.read_csv(input_file)

        # --- Bước 2: Tách dữ liệu thành các mẫu bình thường và bất thường ---
        normal_df = full_df[full_df['Label'] == 0]
        # "Nhãn bất kì" ở đây được hiểu là các nhãn không phải 0
        anomaly_df = full_df[full_df['Label'] != 0]

        print(f"Số mẫu bình thường (label 0): {len(normal_df)}")
        print(f"Số mẫu bất thường (label != 0): {len(anomaly_df)}")

        # --- Bước 3: Kiểm tra xem có đủ dữ liệu không (theo logic mới) ---
        if len(normal_df) < 10000:
            print(f"Lỗi: Không đủ mẫu bình thường. Cần ít nhất 100 nhưng chỉ có {len(normal_df)}.")
            return
        if len(anomaly_df) < 100000:
            print(f"Lỗi: Không đủ mẫu bất thường. Cần ít nhất 100 nhưng chỉ có {len(anomaly_df)}.")
            return

        # --- Bước 4: Tạo bộ dữ liệu ĐÁNH GIÁ (Evaluation Set) ---
        print("\nBắt đầu tạo bộ dữ liệu đánh giá...")
        
        # Lấy 100 mẫu bình thường (label 0), giữ nguyên nhãn
        eval_normal_samples = normal_df.sample(n=10000, random_state=42)
        # eval_normal_samples['label'] = 0 (Không cần, vì nó đã là 0)

        # Lấy 100 mẫu bất thường và gán nhãn = 1
        eval_anomaly_samples = anomaly_df.sample(n=10000, random_state=42) #, replace= True
        eval_anomaly_samples['Label'] = 1
        
        # Gộp và xáo trộn
        evaluation_set = pd.concat([eval_normal_samples, eval_anomaly_samples])
        shuffled_evaluation_set = evaluation_set.sample(frac=1, random_state=42).reset_index(drop=True)

        # Lưu file
        shuffled_evaluation_set.to_csv(evaluation_output_path, index=False)
        print(f"Đã lưu thành công {len(shuffled_evaluation_set)} mẫu đã xáo trộn vào '{evaluation_output_path}'")

        # --- Bước 5: Tạo bộ dữ liệu HUẤN LUYỆN (Training Set) ---
        print("\nBắt đầu tạo bộ dữ liệu huấn luyện...")
        
        # Lấy TẤT CẢ các mẫu bình thường CÒN LẠI
        # (loại bỏ những mẫu đã dùng cho bộ đánh giá)
        remaining_normal_df = normal_df.drop(eval_normal_samples.index)

        if len(remaining_normal_df) == 0:
            print("Cảnh báo: Không còn mẫu bình thường nào cho bộ huấn luyện sau khi trích mẫu đánh giá.")
        
        # Bỏ cột label
        training_set_final = remaining_normal_df.drop(columns=['Label'])

        # Lưu file
        training_set_final.to_csv(training_output_path, index=False)
        print(f"Đã lưu thành công {len(training_set_final)} mẫu (normal) vào '{training_output_path}'")
        
        print("\nHoàn tất!")

    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")


# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN NÀY
    input_folder = 'C:/Users/hoang/Documents/Dataset_KLTN' 

    # Tên file đầu ra (có thể giữ nguyên)
    training_file = input_folder+ '/training_dataset.csv'
    evaluation_file = input_folder+ '/test_dataset.csv'

    # Gọi hàm để thực hiện
    create_training_and_evaluation_sets(input_folder, training_file, evaluation_file)