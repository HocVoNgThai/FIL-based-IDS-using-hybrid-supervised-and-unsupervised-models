#!/bin/bash
set -e

# ===== [1/6] ======

echo -e "\033[0;32m[1/5]\033[0m Installing required apt packages..."

sudo apt update

packages=("python3" "python3-pip" "python3-venv" "jq")

for package in "${packages[@]}"; do
    
    if ! dpkg-query -l "$package" | grep -E "^ii"; then
    # if ! dpkg -l "$package";  then # | grep -E '^ii';
    # Cài đặt các gói nếu chúng chưa tồn tại
    # dpkg -l dùng để liệt kê tất cả các gói đã cài đặt.
    # grep -E '^ii' lọc ra các dòng bắt đầu bằng "ii" (đã cài đặt).
    # dpkg-query -W -f='${Status}\n' "$package" sẽ trả về trạng thái của gói cụ thể.
    # grep -q "ii" sẽ kiểm tra xem kết quả có chứa ii (đã cài đặt) hay không.

    # sudo apt update
    sudo apt install -y "$package"
    # echo -e "hello ${package}"
    fi
done

# ===== 2/6 ======
echo -e "==========================================="
echo -e "\033[0;32m[2/5]\033[0m Creating project structure..."

if [ ! -d "/opt/incremental_ids" ]; then
    # Nếu thư mục không tồn tại, tạo thư mục
    sudo mkdir -p /opt/incremental_ids
    # sudo mkdir -p /opt/incremental_ids/{data,models,logs}
fi

if [ ! -d "/opt/incremental_ids/logs" ]; then
    # Nếu thư mục không tồn tại, tạo thư mục
    sudo mkdir -p /opt/incremental_ids/logs
    # sudo mkdir -p /opt/incremental_ids/{data,models,logs}
fi

sudo cp *.py /opt/incremental_ids/
# sudo cp pipeline.py /opt/incremental_ids/
# sudo cp run_daily.py /opt/incremental_ids/
# sudo cp extract_flows.sh /opt/incremental_ids/
# sudo chmod +x /opt/incremental_ids/extract_flows.sh

# ====== [3/6] Create venv ... =====
echo -e "==========================================="
echo -e "\033[0;32m[3/5]\033[0m Creating venv ..."

python3 -m venv /opt/incremental_ids/venv
source /opt/incremental_ids/venv/bin/activate
pip install -r requirements.txt


# ==============[4/5] CREATE SYSTEMD SERVICE================
echo -e "==========================================="
echo -e "\033[0;32m[4/5]\033[0m Creating systemd service..."

sudo tee /etc/systemd/system/incremental_ids.service >/dev/null <<EOF

[Unit]
Description=Incremental IDS Daemon
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/incremental_ids

ExecStart=/opt/incremental_ids/venv/bin/python -u /opt/incremental_ids/daemon.py

StandardOutput=append:/opt/incremental_ids/logs/run.log
StandardError=append:/opt/incremental_ids/logs/error.log
Restart=always

[Install]
WantedBy=multi-user.target

EOF

# [Unit]
# Description=Daily Incremental IDS Pipeline
# After=network-online.target

# [Service]
# Type=oneshot - simple
# User=root
# ExecStart=/bin/bash /opt/incremental_ids/extract_flows.sh
# ExecStart=/usr/bin/python3 /opt/incremental_ids/run_daily.py
# StandardOutput=append:/opt/incremental_ids/logs/run.log
# StandardError=append:/opt/incremental_ids/logs/error.log

# ===== START SERVICE =====
echo -e "==========================================="
echo -e "\033[0;32m[5/5]\033[0m Reloading & Starting systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable incremental_ids.service
sudo systemctl start incremental_ids.service


# ===== NOTIFICATIONS =====
echo -e "\033[0;32m[DONE]\033[0m Installation complete."
echo -e "Check logs using: journalctl -u incremental_ids.service -f"
