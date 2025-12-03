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
