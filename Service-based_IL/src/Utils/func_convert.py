# ===== HELPER =====
import hashlib  
import ipaddress
import json
import numpy as np
import pandas as pd

# MINMAX_COLS = ['Flow Duration', 'Fwd IAT Total', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Fwd Seg Size Min']

# STANDARD_COLS = ['Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


##### FUNC ############

dtypes = {}    
with open('src/features.json') as json_file:
    data = json.load(json_file)
    for key, type in data.items():
        if type == "int8":
            dtypes[key]= np.int8
        elif type == "float32":
            dtypes[key] = np.float64
        else:
            dtypes[key]= str
                
    json_file.close()
        
def ip_to_float(ip):
    try:
        ip = ".".join(map(str, [(ip >> i) & 0xFF for i in [24, 16, 8, 0]]))
        # temp = df[ip].str.split('.', expand=True).fillna(0).astype(np.float64)
        ret = float(int(ipaddress.IPv4Address(ip)))
        # ret = (temp[0] * 16777216) + (temp[1] * 65536) + (temp[2] * 256) + temp[3]
        
        if ret < 256:
            return ret/1e5
        
        return ret/1e9
    except:
        return 0.0  # for invalid or empty IPs

def ip_to_float_vectorized(series):
    # Ép kiểu về float64 để tránh lỗi tràn số khi tính toán
    vals = series.astype(np.float64)
    
    # Logic của bạn: nếu < 256 thì / 1e5, ngược lại / 1e9
    return np.where(vals < 255*256+255, vals / 1e5, vals / 1e9)

# Bucket port
def bucket_port(port):
    if port < 1024:
        return 0  # Well-known
    elif port < 49152:
        return 1  # Registered
    else:
        return 2  # Dynamic/private

def bucket_port_vectorized(series):
    """
    Ví dụ về phân nhóm Port (Vectorized).
    Thay thế hàm bucket_port(port) cũ của bạn.
    """
    vals = series.astype(np.int16)
    
    # Giả sử logic bucket_port của bạn là:
    # Port < 1024: System, 1024-49151: Registered, > 49151: Dynamic
    # Bạn có thể thay đổi các ngưỡng (thresholds) này theo logic thực tế của bạn
    conditions = [
        (vals < 1024),
        (vals >= 1024) & (vals < 49152),
        (vals >= 49152)
    ]
    choices = [0, 1, 2] # Các giá trị đại diện cho mỗi bucket
    
    return np.select(conditions, choices, default=0)

def sum_of_squares(partition):
    return pd.Series([(partition ** 2).sum()])

def string_to_float(s):
    if pd.notna(s):
        return int(hashlib.sha256(str(s).encode('utf-8')).hexdigest(), 16) % 10**8 / 1e8
    return 0


def down_ratio(f):
    return f/(f+1e-9)


    
def astype(df): 
    for key, type in data.items():
        if type == "int8":
            df[key] = df[key].astype(np.int8)
        elif type == "float32":
            df[key] = df[key].astype(np.float32)
        else:
            df[key] = df[key].astype(str)
            
    return df

def round_decimal(df, min_max_cols, standard_cols, minmax_decimal_bin, standard_decimal_bin):
    with open('src/features.json') as json_file:
        data = json.load(json_file)
        for key, type in data.items():
            if key in min_max_cols:
                df[key] = df[key].round(minmax_decimal_bin)
            elif key in standard_cols:
                df[key] = df[key].round(standard_decimal_bin)
    
        json_file.close()
    return df

def astype_il(df): 
    for key in df.columns:
        if key not in ["Label"]:
            df[key] = df[key].astype(dtypes[key])
            
    return df
                
def pre_astype(df):
    
    return df
##### FUNC ##############
