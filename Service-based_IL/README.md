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
Dùng luôn pigar
> pigar generate

### SERVICE
while read p; do
  echo "Installing $p"
  pip install "$p" || echo "❌ Failed: $p"
done < requirements.txt


Việc sử dụng git rebase thay vì git merge khi cập nhật code từ server giúp lịch sử commit của bạn luôn là một đường thẳng, sạch sẽ và dễ theo dõi hơn.Dưới đây là quy trình chuẩn để thực hiện việc này mà không gây xung đột (conflict) lung tung hay làm hỏng repo:Quy trình 4 bước chuẩn "vàng"Giả sử bạn đang làm việc trên branch feature-abc.1. Commit công việc hiện tạiTrước khi kéo code mới về, hãy đảm bảo bạn đã commit mọi thay đổi đang làm dở.Bashgit add .
git commit -m "Tính năng đang làm: mô tả ngắn gọn"
Lưu ý: Nếu chưa muốn commit vì code chưa xong, bạn có thể dùng git stash để tạm cất đi.2. Cập nhật code mới nhất từ serverThay vì dùng git pull (thường sẽ tự động tạo một merge commit), hãy dùng flag --rebase:Bashgit pull --rebase origin main
(Thay main bằng tên branch chính của dự án bạn, ví dụ develop hoặc master).3. Giải quyết xung đột (Nếu có)Nếu có file bị trùng lặp chỉnh sửa, Git sẽ dừng lại và báo lỗi. Bạn cần:Mở file bị lỗi lên và sửa lại cho đúng.Sau khi sửa xong, gõ: git add <tên_file_đã_sửa>.Tiếp tục quá trình rebase bằng lệnh:Bashgit rebase --continue
Lưu ý: Tuyệt đối không dùng git commit trong lúc này.4. Push code lên serverSau khi rebase xong, branch của bạn bây giờ đã nằm "đè" lên trên những commit mới nhất của main.Nếu bạn chưa bao giờ push branch này lên: Chỉ cần git push origin feature-abc.Nếu bạn đã từng push branch này lên trước đó: Bạn buộc phải dùng lệnh "ép" vì lịch sử đã thay đổi:Bashgit push --force-with-lease
(Dùng --force-with-lease an toàn hơn --force vì nó sẽ không đè nếu có ai đó khác cũng vừa push vào branch của bạn).Tại sao nên dùng cách này?Đặc điểmGit Merge (Mặc định)Git RebaseLịch sửChằng chịt, có nhiều nhánh rẽLà một đường thẳng tắpMerge CommitTạo ra commit rác kiểu "Merge branch..."Không tạo commit rácTính an toànRất an toàn, dễ hiểuCần hiểu rõ mình đang làm gìMột quy tắc "sống còn" khi dùng RebaseChỉ rebase trên những branch cá nhân (feature branch của riêng bạn). Tuyệt đối không bao giờ rebase trên các branch chung như main hay develop vì nó sẽ làm thay đổi lịch sử của tất cả mọi người trong team, gây ra thảm họa về quản lý code.Bạn có đang gặp phải thông báo lỗi cụ thể nào khi thực hiện các bước trên không? Tôi có thể giải thích chi tiết lỗi đó cho bạn.



# Công pháp pip
pip list --format=freeze > requirements.txt


# Nếu chạy service lỗi tại ký tự ^M$ của windows 
sed -i 's/\r$//' install_service.sh
> Sợ thì tạo backup
sed -i.bak 's/\r$//' install_service.sh
