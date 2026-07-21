import os
import json
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pnnx

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ncnn/build/tools/quantize")
_NCNN2TABLE = os.path.join(_TOOLS, "ncnn2table")
_NCNN2INT8  = os.path.join(_TOOLS, "ncnn2int8")

TRAIN_PATH = "../../../datasets/LSTM/CMAPSSData/train_FD001.txt"
TEST_PATH  = "../../../datasets/LSTM/CMAPSSData/test_FD001.txt"
RUL_PATH   = "../../../datasets/LSTM/CMAPSSData/RUL_FD001.txt"
MODEL_DIR  = "model"
SEQ_LEN    = 30
MAX_RUL    = 125
EPOCHS     = 100
BATCH_SIZE = 256
LR         = 1e-3
PATIENCE   = 10
SEED       = 42

FEATURES = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
INPUT_SIZE = len(FEATURES)
HIDDEN_SIZES = [32, 64, 128]
COLS = ["id","cycle","set1","set2","set3"] + [f"s{i}" for i in range(1, 22)]

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_train_data(path, scaler):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    rul = df.groupby("id")["cycle"].max().reset_index()
    rul.columns = ["id", "max"]
    df = df.merge(rul, on="id", how="left")
    df["RUL"] = (df["max"] - df["cycle"]).clip(upper=MAX_RUL) / MAX_RUL
    df.drop("max", axis=1, inplace=True)
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    X, y = [], []
    for eid in df["id"].unique():
        ed = df[df["id"] == eid]
        feats = ed[FEATURES].values
        rul_v = ed["RUL"].values
        for i in range(len(ed) - SEQ_LEN):
            X.append(feats[i:i + SEQ_LEN])
            y.append(rul_v[i + SEQ_LEN - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_test_data(test_path, rul_path, scaler):
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    df[FEATURES] = scaler.transform(df[FEATURES])
    true_rul = [float(v.strip()) for v in open(rul_path)]
    X = []
    for eid in df["id"].unique():
        ed = df[df["id"] == eid]
        feats = ed[FEATURES].values
        if len(feats) >= SEQ_LEN:
            X.append(feats[-SEQ_LEN:])
        else:
            pad = np.zeros((SEQ_LEN - len(feats), INPUT_SIZE), dtype=np.float32)
            X.append(np.vstack([pad, feats]))
    y = [min(r, MAX_RUL) / MAX_RUL for r in true_rul]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def save_test_data(path, X, y):
    n, sl, nf = X.shape
    with open(path, "w") as f:
        f.write(f"{n} {sl * nf} 1\n")
        for xi, yi in zip(X, y):
            row = xi.flatten()
            f.write(" ".join(f"{v:.6f}" for v in row) + f" {yi:.6f}\n")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


scaler = StandardScaler()
X_train, y_train = load_train_data(TRAIN_PATH, scaler)
X_test,  y_test  = load_test_data(TEST_PATH, RUL_PATH, scaler)
print(f"Train sequences: {len(X_train)}, Test engines: {len(X_test)}")

save_test_data(os.path.join(MODEL_DIR, "test_data.txt"), X_test, y_test)
print(f"Test data saved: {len(X_test)} samples")

metrics = {}

for hidden in HIDDEN_SIZES:
    key = f"lstm_h{hidden}"
    model = LSTMModel(INPUT_SIZE, hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    val_sz = int(0.2 * len(ds))
    trn_sz = len(ds) - val_sz
    trn_ds, val_ds = random_split(ds, [trn_sz, val_sz], generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    no_impro = 0
    best_state = None

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in trn_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= val_sz
        if val_loss < best_val:
            best_val = val_loss
            no_impro = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_impro += 1
        if no_impro >= PATIENCE:
            break
    elapsed = time.time() - t0

    model.load_state_dict(best_state)

    metrics[key] = {
        "training_time_s": round(elapsed, 3),
        "best_val_loss": round(best_val, 6),
    }
    print(f"{key}: best_val_loss={best_val:.6f} time={elapsed:.1f}s")

    dummy = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    pt_path = os.path.join(MODEL_DIR, f"lstm_h{hidden}_f32.pt")
    torch.jit.save(torch.jit.trace(model, dummy), pt_path)
    ncnn_param = pt_path.replace(".pt", ".ncnn.param")
    ncnn_bin   = pt_path.replace(".pt", ".ncnn.bin")
    try:
        pnnx.export(model, pt_path, dummy)
    except SyntaxError:
        if not (os.path.exists(ncnn_param) and os.path.exists(ncnn_bin)):
            raise
    print(f"  Exported {ncnn_param}")
    if os.path.exists(_NCNN2TABLE):
        table_path = ncnn_param.replace("_f32.ncnn.param", "_int8.table")
        int8_param = ncnn_param.replace("_f32.ncnn.param", "_int8.ncnn.param")
        int8_bin   = ncnn_bin.replace("_f32.ncnn.bin", "_int8.ncnn.bin")
        subprocess.run([_NCNN2TABLE, ncnn_param, ncnn_bin, table_path, "method=aciq"], check=True)
        subprocess.run([_NCNN2INT8, ncnn_param, ncnn_bin, int8_param, int8_bin, table_path], check=True)
        print(f"  Quantized {int8_param}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
