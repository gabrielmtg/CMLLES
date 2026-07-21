import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

TRAIN_PATH = "../../../datasets/LSTM/CMAPSSData/train_FD001.txt"
MODEL_DIR  = "model"
SEQ_LEN    = 30
MAX_RUL    = 125
SEED       = 42
FEATURES   = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
COLS       = ["id","cycle","set1","set2","set3"] + [f"s{i}" for i in range(1, 22)]


class _CalibReader(CalibrationDataReader):
    def __init__(self, X, n=50):
        self.data = [{"input": X[i:i+1]} for i in range(min(n, len(X)))]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d


scaler = StandardScaler()
df = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLS)
rul = df.groupby("id")["cycle"].max().reset_index()
rul.columns = ["id", "max"]
df = df.merge(rul, on="id", how="left")
df["RUL"] = (df["max"] - df["cycle"]).clip(upper=MAX_RUL) / MAX_RUL
df.drop("max", axis=1, inplace=True)
df[FEATURES] = scaler.fit_transform(df[FEATURES])
X_seqs = []
for eid in df["id"].unique():
    ed = df[df["id"] == eid]
    feats = ed[FEATURES].values
    for i in range(len(ed) - SEQ_LEN):
        X_seqs.append(feats[i:i + SEQ_LEN])
X_train = np.array(X_seqs, dtype=np.float32)

for hidden in [32, 64, 128]:
    f32 = os.path.join(MODEL_DIR, f"lstm_h{hidden}_f32.onnx")
    i8  = os.path.join(MODEL_DIR, f"lstm_h{hidden}_int8.onnx")
    if not os.path.exists(f32):
        print(f"SKIP {f32} (not found)")
        continue
    quantize_static(f32, i8, _CalibReader(X_train),
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
    print(f"Quantized {i8}")
