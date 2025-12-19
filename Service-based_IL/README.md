# TEMP
🟢 CPU sử dụng (thực tế đo)

Một Python daemon kiểu:

while True:
    time.sleep(1)


→ CPU usage: 0.0%
(top hiển thị 0.0 hoặc 0.1%)

Kể cả worker queue block:

job = queue.get()  # chờ tới khi có job


→ CPU bằng 0 (thread blocking).

🟢 Khi nào Python tốn CPU?

Chỉ khi đang chạy job học incremental:

AutoEncoder training vài epoch

OCSVM partial_fit

XGBoost incremental training

Nhưng vì bạn chạy 1 job/ngày, và job chạy tuần tự (không song song), thì thời gian tiêu CPU rất ngắn.

🟢 Độ ổn định của Python daemon?

✔ Không rò rỉ bộ nhớ nếu bạn không tạo list khổng lồ mỗi vòng lặp
✔ Không bị treo CPU do sleep hoặc queue block
✔ Không gây nghẽn network
✔ Không tạo thread vô hạn

Bạn chạy bằng systemd nên:

crash → auto restart

memory quá lớn → systemd cắt

log theo dõi dễ

chạy nền 24/7 không cần screen/tmux

🟢 Nếu so sánh với Golang, Rust?
Ngôn ngữ	Idle RAM	Idle CPU	Độ ổn định dài hạn
Golang	~5 MB	0%	Rất cao
Rust	~1–3 MB	0%	Rất cao
Python	~60 MB	0%	Cao


# Illustration
        +------------------------+
        |   Incremental Daemon  |
        +------------------------+
            | enqueue mỗi 24h (JOB_INTERVAL)
            v
+-------------------------------------+
|             JOB QUEUE              |
+-------------------------------------+
     | job1 | job2 | job3 | ...
     v
Worker: chạy tuần tự → AE/OCSVM/XGB → save model



# How to use this archirtecture as a service

- In Service-based_IL run these following command on your Linux System
> sudo chmod +x install.sh
> sudo ./install.sh


# CÁC CÔNG PHÁP FIX LỖI/BẾ TẮC
### PIP REQUIREMENTS
Có nha — và đây là câu hỏi rất “đúng người đúng bệnh” 😄
Cái format kiểu:

keras @ file:///home/conda/feedstock_root/build_artifacts/keras_1747722781675/work


là đặc sản của conda list --export / pip freeze khi package được cài từ build path nội bộ của conda. Mang sang máy khác là… toang.

Dưới đây là những cách chuẩn – gọn – an toàn, tuỳ mục đích dùng của bạn.

✅ CÁCH 1 (KHUYẾN NGHỊ): pip list --format=freeze ✅

Trong conda env:

pip list --format=freeze > requirements.txt


✅ Kết quả:

keras==3.1.1
numpy==1.26.4
pandas==2.1.3


✅ Không có @ file://
✅ Dùng được ở máy khác
✅ Tương thích pip install -r requirements.txt

👉 Đây là cách nên dùng nhất nếu bạn deploy bằng pip

❌ KHÔNG DÙNG: pip freeze (trong conda)
pip freeze > requirements.txt


❌ Dễ sinh:

keras @ file:///home/conda/feedstock_root/...


Vì:

pip thấy package được conda build

nên trỏ về local build path (chỉ tồn tại ở máy bạn)

✅ CÁCH 2: dùng conda env export nhưng LOẠI pip path rác
1️⃣ Export YAML chuẩn
conda env export > environment.yml

2️⃣ Khi dùng cho máy khác
conda env create -f environment.yml


✅ Tốt cho conda-to-conda
❌ Không phù hợp nếu bạn muốn requirements.txt

✅ CÁCH 3: Convert conda → pip chuẩn (lọc path)
conda list --export \
| grep -v "@" \
| grep -v "file:" \
| cut -d'=' -f1 \
| sed 's/_/-/g' \
> requirements.txt


Nhưng ⚠️:

Không có version

Chỉ nên dùng khi:

bạn chấp nhận version mới

dùng cho dev/debug

✅ CÁCH 4 (VIP – chuẩn nhất cho ML): split conda + pip
1️⃣ Conda package (nặng, system)
conda list --export > conda.txt

2️⃣ Pip-only package
pip list --format=freeze > requirements.txt


Deploy:

conda install --file conda.txt
pip install -r requirements.txt


✅ Rất ổn cho:

CUDA

pytorch / tensorflow

scipy

🎯 Cách mình khuyên dùng cho bạn (context ML + IDS):

👉 Dùng cách 1

pip list --format=freeze > requirements.txt


và đảm bảo:

pip install --upgrade pip
pip install -r requirements.txt

⚠️ Checklist nhanh tránh bị dính @ file://

✅ Cài package bằng pip install

❌ Tránh conda install keras nếu định export requirements

✅ Prefer pip list --format=freeze

❌ Không dùng pip freeze trong conda


### SERVICE
while read p; do
  echo "Installing $p"
  pip install "$p" || echo "❌ Failed: $p"
done < requirements.txt

