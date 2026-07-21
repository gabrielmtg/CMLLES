import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ai_edge_quantizer as aq

DATASET_DIRS = [
    "../../../../datasets/MLP/IAES-dataset/2-cores",
    "../../../../datasets/MLP/IAES-dataset/3-cores",
]
MODEL_DIR        = "model"
WINDOW           = 10
SEED             = 42
MAX_TRAIN_SAMPLES = 50000


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
        f32 = os.path.join(MODEL_DIR, f"iaes_{size}_{act}_f32.tflite")
        i8  = os.path.join(MODEL_DIR, f"iaes_{size}_{act}_int8.tflite")
        if not os.path.exists(f32):
            print(f"SKIP {f32} (not found)")
            continue
        quantizer = aq.Quantizer(bytearray(open(f32, "rb").read()))
        quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
        calib = {"serving_default": [{"args_0": X_train[i:i+1]} for i in range(min(100, len(X_train)))]}
        i8_dir, i8_name = os.path.split(i8)
        quantizer.quantize(quantizer.calibrate(calib)).save(i8_dir, i8_name.replace(".tflite", ""), overwrite=True)
        print(f"Quantized {i8}")
