import os
import json
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

DATASET_DIRS = [
    "../../../../datasets/MLP/IAES-dataset/2-cores",
    "../../../../datasets/MLP/IAES-dataset/3-cores",
]
MODEL_DIR       = "model"
WINDOW          = 10
EPOCHS          = 200
LR              = 0.001
SEED            = 42
BATCH_SIZE        = 2048
MAX_TRAIN_SAMPLES = 50000
MAX_TEST_SAMPLES  = 1000

IAES_CONFIGS = {
    "small":  [12, 32, 64, 1],
    "medium": [12, 64, 128, 64, 1],
    "large":  [12, 128, 256, 128, 64, 1],
}
ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class _CalibReader(CalibrationDataReader):
    def __init__(self, X):
        self.data = [{"input": X[i:i+1]} for i in range(min(len(X), 100))]
        self.idx = 0
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


def build_mlp(dims, act_cls):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def save_test_data(path, X, y):
    n, nf = X.shape
    with open(path, "w") as f:
        f.write(f"{n} {nf} 1\n")
        for xi, yi in zip(X, y):
            f.write(" ".join(f"{v:.6f}" for v in xi) + f" {int(yi)}\n")


os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = load_iaes(DATASET_DIRS, window=WINDOW)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train = scaler.fit_transform(X_train).astype(np.float32)
y_test  = y_test[:MAX_TEST_SAMPLES]
X_test  = scaler.transform(X_test[:MAX_TEST_SAMPLES]).astype(np.float32)

if len(X_train) > MAX_TRAIN_SAMPLES:
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]

save_test_data(os.path.join(MODEL_DIR, "test_data.txt"), X_test, y_test)
print(f"Test split saved: {len(y_test)} samples")

metrics = {}

dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train).unsqueeze(1),
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for size, dims in IAES_CONFIGS.items():
    for act_name, act_cls in ACTS.items():
        key   = f"iaes_{size}_{act_name}"
        model = build_mlp(dims, act_cls)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        t0 = time.time()
        final_loss = 0.0
        for _ in range(EPOCHS):
            for Xb, yb in loader:
                optimizer.zero_grad()
                out  = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
        elapsed = time.time() - t0

        metrics[key] = {
            "training_time_s": round(elapsed, 3),
            "epochs": EPOCHS,
            "final_train_loss": round(final_loss, 6),
        }
        print(f"{key}: loss={final_loss:.4f} time={elapsed:.1f}s")

        dummy = (torch.randn(1, dims[0]),)
        path  = os.path.join(MODEL_DIR, f"iaes_{size}_{act_name}_f32.onnx")
        torch.onnx.export(
            model.eval(), dummy[0], path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        print(f"  Exported {path}")
        int8_path = path.replace("_f32.onnx", "_int8.onnx")
        quantize_static(path, int8_path, _CalibReader(X_train),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
        print(f"  Quantized {int8_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
