import pandas as pd
import glob
import os

def process_parquet_folder_to_csv(folder_path: str, output_csv_path: str, label_column: str = 'label'):
    """
    Đọc tất cả các file Parquet trong một thư mục, lọc mẫu bình thường (nhãn 0),
    loại bỏ cột nhãn, gộp tất cả dữ liệu lại và lưu ra một file CSV duy nhất.

    Args:
        folder_path (str): Đường dẫn đến thư mục chứa các file .parquet.
        output_csv_path (str): Đường dẫn để lưu file .csv kết quả.
        label_column (str): Tên của cột chứa nhãn. Mặc định là 'label'.
    """
    try:
        # 1. Tìm tất cả các file .parquet trong thư mục chỉ định
        # Sử dụng os.path.join để đảm bảo tương thích với mọi hệ điều hành
        search_pattern = os.path.join(folder_path, '*.parquet')
        parquet_files = glob.glob(search_pattern)

        if not parquet_files:
            print(f"Lỗi: Không tìm thấy file .parquet nào trong thư mục '{folder_path}'.")
            return

        print(f"Tìm thấy {len(parquet_files)} file parquet. Bắt đầu xử lý...")

        # 2. Xử lý từng file và lưu vào một danh sách
        list_of_processed_dfs = []
        for file_path in parquet_files:
            print(f"  - Đang xử lý file: {os.path.basename(file_path)}")
            
            # Đọc file parquet
            df = pd.read_parquet(file_path, engine='pyarrow')

            # Kiểm tra cột nhãn
            if label_column not in df.columns:
                print(f"    Cảnh báo: Bỏ qua file vì không có cột '{label_column}'.")
                continue

            # Lọc các mẫu có nhãn là 0
            normal_samples_df = df[df[label_column] == 0].copy()

            # Loại bỏ cột nhãn
            unsupervised_df = normal_samples_df.drop(columns=[label_column])
            
            # Thêm DataFrame đã xử lý vào danh sách
            list_of_processed_dfs.append(unsupervised_df)

        if not list_of_processed_dfs:
            print("Không có dữ liệu nào được xử lý sau khi lọc. Dừng lại.")
            return

        # 3. Gộp tất cả các DataFrame trong danh sách thành một
        print("\nĐang gộp tất cả dữ liệu đã xử lý...")
        final_df = pd.concat(list_of_processed_dfs, ignore_index=True)
        print("Gộp dữ liệu thành công.")
        print(f"Kích thước dataset cuối cùng: {final_df.shape[0]} mẫu, {final_df.shape[1]} thuộc tính.")

        # 4. Lưu DataFrame cuối cùng ra file CSV
        print(f"Đang lưu kết quả vào file: {output_csv_path}...")
        final_df.to_csv(output_csv_path, index=False)
        print("Hoàn tất! Đã lưu file thành công.")

    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # 1. Thay thế 'duong_dan_den_folder_parquet' bằng đường dẫn thực tế đến thư mục của bạn
    input_folder = 'scaled_output_parquet'
    
    # 2. Đặt tên cho file CSV đầu ra
    output_file = 'unsupervised_dataset.csv'

    # Gọi hàm để thực hiện toàn bộ quá trình
    process_parquet_folder_to_csv(input_folder, output_file)