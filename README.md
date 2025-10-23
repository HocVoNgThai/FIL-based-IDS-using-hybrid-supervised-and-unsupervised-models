# FIL-based-IDS-using-hybrid-supervised-and-unsupervised-models

## Mục tiêu
Mục tiêu của đề tài hướng tới xây dựng hệ thống Phát hiện xâm nhập mạng dựa trên mô hình
học máy nhằm tăng cường khả năng bảo mật cho hệ thống mạng và phát hiện nhanh chóng các bất
thường, các tấn công zero-day, sử dụng phương pháp Học máy liên kết tiệm tiến (FIL) kết hợp mô
hình học không giám sát và có giám sát vào trong quá trình huấn luyện. Hệ thống có khả năng phát
hiện các loại tấn công mạng nhằm vào thiết bị IoT với độ trễ thấp, cung cấp giao diện theo dõi và
quản lý lưu lượng mạng của các thiết bị này.
- **Phát triển Hệ thống phát hiện xâm nhập:** Phát triển công cụ phát hiện các tấn công mạng
dựa trên kiến trúc đã triển khai và mô hình đã huấn luyện, cung cấp giao diện theo dõi và quản lý lưu
lượng mạng của thiết bị.
- **Xây dựng kiến trúc học liên kết tiệm tiến:** Xây dựng và triển khai thực tế thành công mô hình
học liên kết tiệm tiến với ba máy con và một máy chủ. Tích hợp các biện pháp bảo vệ để đảm bảo an
toàn dữ liệu trong quá trình trao đổi giữa các máy.
- **Kết hợp học không giám sát và có giám sát:** Kết mô hình học không giám sát và có giám sát
để giảm các trường hợp dương tính và âm tính giả, cải thiện kết quả đầu ra của toàn kiến trúc, hệ
thống.
- **Đánh giá hiệu suất Hệ thống phát hiện xâm nhập:** Đảm bảo thời gian kiểm tra và phản hồi
của hệ thống nhanh chóng. Kết quả của hệ thống được hiển thị trực quan và có độ chính xác cao.



## Chạy Data Extracting với CICFlowMeter:

Tham khảo tại source: ```https://github.com/ahlashkari/CICFlowMeter```

Để compile ra file .jar:
```
pip install maven

#Build:
mvn package  # or using mvn clean package
```

Nếu có lỗi, hãy cài đặt jnetpcap và chạy lại lệnh trên:
```
mvn install:install-file -Dfile=./jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar
```

Sau khi đã có file jar, có thể dùng GUI để convert .pcap sang .csv, chọn tab `Offline` tại `Network`:

<p align = "center"> 

<img width="729" height="751" src="https://github.com/HocVoNgThai/FIL-based-IDS-using-hybrid-supervised-and-unsupervised-models/blob/main/image.png">
</p>

