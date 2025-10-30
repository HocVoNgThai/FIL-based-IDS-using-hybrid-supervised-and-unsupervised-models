import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

base_url = "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/PCAP/"

# Lấy nội dung HTML từ URL
response = requests.get(base_url)
soup = BeautifulSoup(response.content, "html.parser")

# Tải toàn bộ các liên kết trong thư mục
for link in soup.find_all("a"):
    href = link.get("href")
    if href.endswith("/"):  # Kiểm tra xem có phải thư mục không
        subdir_url = urljoin(base_url, href)
        subdir_response = requests.get(subdir_url)
        with open(href.rstrip("/"), "wb") as f:
            f.write(subdir_response.content)
            print(f"Đã tải: {href.rstrip('/')}")