import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

DATASET_DIRS = [
    "../../../../datasets/MLP/IAES-dataset/2-cores",
    "../../../../datasets/MLP/IAES-dataset/3-cores",
]
MODEL_DIR        = "model"
WINDOW           = 10
SEED             = 42
MAX_TRAIN_SAMPLES = 50000


class _CalibReader(CalibrationDataReader):
    def __init__(self, X, n=100):
        self.data = [{"input": X[i:i+1]} for i in range(min(n, len(X)))]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d


def load_iaes(dirs, window=10):
    dfs = []
    for d in dirs:
        for fpath in glob.glob(os.path.join(d, "*.csv")):
            df = pd.read_csv(fpath)
            eps = 1e-9
            df["sig_ipc"]    = df["INSTRUCTIONS"] / (df["CPU_CYCLES"] + eps)
            df["sig_mpki"]   = (df["CACHE_MISSES"] * 1000.0) / (df["INSTRUCTIONS"] + eps)
            df["sig_l2p"]    = df["L2_CACHE_ACCESS"] / (df["CPU_CYCLES"] + eps)
            df["sig_branch"] = df["BRANCH_MISSES"] / (df["INSTRUCTIONS"] + eps)
            feat_cols = []
            for sig in ["sig_ipc", "sig_mpki", "sig_l2p", "sig_branch"]:
                df[f"{sig}_mean"]  = df[sig].rolling(window=window).mean()
                df[f"{sig}_std"]   = df[sig].rolling(window=window).std(ddof=1)
                df[f"{sig}_delta"] = df[sig].diff(periods=window)
                feat_cols.extend([f"{sig}_mean", f"{sig}_std", f"{sig}_delta"])
            df = df.dropna(subset=feat_cols).copy()
            df["target"] = df["LABEL"].isin([1, 2, 3]).astype(np.float32)
            dfs.append(df[feat_cols + ["target"]])
    combined = pd.concat(dfs, ignore_index=True)
    return combined[feat_cols].values.astype(np.float32), combined["target"].values.astype(np.float32)


X, y = load_iaes(DATASET_DIRS, window=WINDOW)
scaler = StandardScaler()
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train = scaler.fit_transform(X_train).astype(np.float32)
if len(X_train) > MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
    X_train = X_train[idx]

for size in ["small", "medium", "large"]:
    for act in ["relu", "sigmoid"]:
        f32 = os.path.join(MODEL_DIR, f"iaes_{size}_{act}_f32.onnx")
        i8  = os.path.join(MODEL_DIR, f"iaes_{size}_{act}_int8.onnx")
        if not os.path.exists(f32):
            print(f"SKIP {f32} (not found)")
            continue
        quantize_static(f32, i8, _CalibReader(X_train),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
        print(f"Quantized {i8}")
