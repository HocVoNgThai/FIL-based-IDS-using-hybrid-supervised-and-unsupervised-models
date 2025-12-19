import gc
import pandas as pd
from pathlib import Path
from datetime import datetime

# ===== ALGO LIBS =====
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# ===== HELPER
import joblib


# ======== VARIABLES / FUNC IMPORT=========
# from FlowZmqServer import CURR_DIR
from src.config.settings import settings
from src.Utils.func_convert import *

# ======== CONSTANCE =========
PKL_PATH = settings.PKL_DIR
STANDARD_SCALER_PATH = PKL_PATH/"standardScaler.pkl"
MINMAX_SCALER_PATH = PKL_PATH/"minmaxScaler.pkl"
MINMAX_COLS = ['Flow Duration', 'Fwd IAT Total', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Fwd Seg Size Min']

STANDARD_COLS = ['Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

COLS_TO_DROP = ["Flow ID", "Timestamp"]

SAMPLE_COLS_TO_REMOVE = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Bulk Rate Avg']
DO_NOT_TOUCH_COLS = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol' ]

DECIMAL_BIN = 6

class FlowFlushTransformer:
    def __init__(
        self,
        minmax_scaler_path, standard_scaler_path, minmax_cols, standard_cols, 
        header =None, label_mapping=None, 
        decimal_bin=6, out_dir="flows_parquet", file_prefix="batch"
    ):
        # Load old scaler
        self.minmaxScaler = MinMaxScaler()
        self.minmaxScaler = joblib.load(minmax_scaler_path)
        self.standardScaler = StandardScaler()
        self.standardScaler = joblib.load(standard_scaler_path)

        # 
        self.minmax_cols = minmax_cols
        self.standard_cols = standard_cols
                
        self.header = header or []
        self.label_mapping = label_mapping or {}
        
        self.decimal_bin = decimal_bin
        
        # Tồn tại hay không cũng không sao, tạo hết
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # state batch
        self.file_prefix = file_prefix
        self.curr_folder = f"{datetime.now().date()}"
        self.curr_index = 0
        
        if (not Path.exists(self.out_dir/self.curr_folder)):
            Path.mkdir(self.out_dir/self.curr_folder, parents=True, exist_ok=True)
            
    def flush(self, rows: list | pd.DataFrame, header = None):
        # ---- Build DataFrame ----
        if isinstance(rows, list):
            if self.header.len() <1:
                print("[ERROR] There is no header!")
                return None
            
            df = pd.DataFrame(rows, columns= self.header)
        else:
            df = rows.copy()

        del rows

        # XỬ LÍ DATA
        df, indexdf = self.preProcessingData(df)
        df = astype(df)
        df = pd.concat([indexdf, df], axis= 1)
                
        # LƯU
        temp = f"{datetime.now().date()}"
        
        if temp != self.curr_folder:
            self.curr_folder = temp
            self.curr_index = 0
            temp = self.out_dir / self.curr_folder
            Path.mkdir(temp, parents=True, exist_ok=True)
            
        fname = (
            self.out_dir / self.curr_folder / f"{self.file_prefix}_{self.curr_index}.parquet"
        )
        
        self.curr_index +=1
        
        df.to_parquet(
            fname,
            # engine="pyarrow",
            # compression="snappy"
        )
        
        del df, indexdf
        gc.collect()
        
        return fname
    
    def detect(self, rows: list | pd.DataFrame, header = None):
        # ---- Build DataFrame ----
        if isinstance(rows, list):
            if self.header.len() <1:
                print("[ERROR] There is no header!")
                return None
            
            df = pd.DataFrame(rows, columns= self.header)
        else:
            df = rows.copy()

        del rows

        # XỬ LÍ DATA
        df, indexdf = self.preProcessingData(df)
        df = astype(df)
        gc.collect()        

        self.curr_index +=1
        
        retX = df.drop(columns=["Label"]).values # ,"Binary Label"
        rety = df["Label"]
        
        del df
        gc.collect()
        
        return indexdf, retX, rety
    
    def preProcessingData(self, df):
        # df[self.standard_cols] = df[self.standard_cols].apply(pd.to_numeric)
        # df[self.standard_cols] = df[self.standard_cols].apply(
        #     lambda col: np.where(col <= -1, 0, col),
        #     axis=0
        # )
        
        # DROP NA DUP COLS
        df = df.replace([np.inf, -np.inf, "inf", "-inf", "Infinity", "-Infinity", r'[N|n][a|A][N|n]', "(empty)"], 0)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.fillna(0)
        
        indexdf = df[COLS_TO_DROP]
        df = df.drop(columns=COLS_TO_DROP)

        # Bỏ các giá trị âm
        numeric_cols = self.loadNumericCols()
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        for col in SAMPLE_COLS_TO_REMOVE:
            df = df[df[col]>=0]
            
        
        df["Src Port"] = df["Src Port"].apply(bucket_port)
        df["Dst Port"] = df["Dst Port"].apply(bucket_port)
        
        df['Src IP'] = df['Src IP'].apply(ip_to_float)
        df['Dst IP'] = df['Dst IP'].apply(ip_to_float)
        
        df[self.standard_cols] = np.log1p(df[self.standard_cols])

        # scale
        df[self.standard_cols] = self.standardScaler.transform(df[self.standard_cols])
        df[self.minmax_cols] = self.minmaxScaler.transform(df[self.minmax_cols])

        # round
        df[self.minmax_cols] = df[self.minmax_cols].round(self.decimal_bin)
        df[self.standard_cols] = df[self.standard_cols].round(self.decimal_bin)
        
        gc.collect()
        return df, indexdf
    
    def loadNumericCols(self):
        cols = []
        for col in self.minmax_cols:
            cols.append(col)
        for col in self.standard_cols:
            cols.append(col)
            
        for col in SAMPLE_COLS_TO_REMOVE:
            check = True
            for tempCol in cols:
                if tempCol == col:
                    check= False
                    break
            if check == True:
                cols.append(col)
                
        cols.append("Src Port")
        cols.append("Dst Port")
        
        gc.collect()
        
        return cols