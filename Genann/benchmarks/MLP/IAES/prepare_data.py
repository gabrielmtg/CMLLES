import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_DIRS = [
    "../../../../datasets/MLP/IAES-dataset/2-cores",
    "../../../../datasets/MLP/IAES-dataset/3-cores",
]
MODEL_DIR         = "model"
WINDOW            = 10
SEED              = 42
MAX_TRAIN_SAMPLES = 50000
MAX_TEST_SAMPLES  = 1000

os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)


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
            df["target"] = df["LABEL"].isin([1, 2, 3]).astype(np.float64)
            dfs.append(df[feat_cols + ["target"]])
    combined = pd.concat(dfs, ignore_index=True)
    return combined[feat_cols].values.astype(np.float64), combined["target"].values.astype(np.float64)


X, y = load_iaes(DATASET_DIRS, window=WINDOW)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train = scaler.fit_transform(X_train)
y_test  = y_test[:MAX_TEST_SAMPLES]
X_test  = scaler.transform(X_test[:MAX_TEST_SAMPLES])

if len(X_train) > MAX_TRAIN_SAMPLES:
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]

train_path = os.path.join(MODEL_DIR, "iaes_genann_train.data")
with open(train_path, "w") as f:
    f.write(f"{len(X_train)} 12 1\n")
    for xi, yi in zip(X_train, y_train):
        f.write(" ".join(f"{v:.10f}" for v in xi) + "\n")
        f.write(f"{yi:.10f}\n")

test_path = os.path.join(MODEL_DIR, "test_data.txt")
with open(test_path, "w") as f:
    f.write(f"{len(X_test)} 12 1\n")
    for xi, yi in zip(X_test, y_test):
        f.write(" ".join(f"{v:.10f}" for v in xi) + f" {int(yi)}\n")

print(f"Train: {len(X_train)} samples, Test saved: {len(X_test)} samples")
print(f"Saved: {train_path}, {test_path}")
