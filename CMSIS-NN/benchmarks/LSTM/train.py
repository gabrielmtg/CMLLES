import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

INPUT_SCALE    = 1.0 / 128.0
HIDDEN_SCALE   = 1.0 / 128.0
INTER_SCALE    = math.pow(2, -12)
CELL_SCALE_PWR = -15

FEATURES = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
INPUT_SIZE   = len(FEATURES)
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


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, scale):
    return np.clip(np.round(b / scale), -(2**31), 2**31 - 1).astype(np.int32)


def quantize_scale(scale):
    significand, exp = math.frexp(scale)
    q31 = int(round(significand * (1 << 31)))
    return q31, exp


def array_to_c(name, arr, dtype):
    lines = [f"static const {dtype} {name}[] = {{"]
    flat = arr.flatten()
    row = "    "
    for i, v in enumerate(flat):
        row += f"{int(v)}, "
        if (i + 1) % 12 == 0:
            lines.append(row.rstrip())
            row = "    "
    if row.strip():
        lines.append(row.rstrip())
    lines.append("};")
    return "\n".join(lines)


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

GATE_KEYS    = ["forget", "input", "cell", "output"]
GATE_PREFIX  = {"forget": "I2F", "input": "I2I", "cell": "I2C", "output": "I2O"}
GATE_RPREFIX = {"forget": "R2F", "input": "R2I", "cell": "R2C", "output": "R2O"}

metrics = {}

for hidden in HIDDEN_SIZES:
    key   = f"lstm_h{hidden}"
    model = LSTMModel(INPUT_SIZE, hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    val_sz = int(0.2 * len(ds))
    trn_sz = len(ds) - val_sz
    trn_ds, val_ds = random_split(ds, [trn_sz, val_sz],
                                  generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val   = float("inf")
    no_impro   = 0
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
            best_val = val_loss; no_impro = 0
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

    sd = model.state_dict()
    wih = sd["lstm.weight_ih_l0"].numpy()
    whh = sd["lstm.weight_hh_l0"].numpy()
    bih = sd["lstm.bias_ih_l0"].numpy()
    bhh = sd["lstm.bias_hh_l0"].numpy()
    wfc_arr = sd["fc.weight"].numpy()
    bfc_arr = sd["fc.bias"].numpy()

    gate_slices = {
        "input":  slice(0, hidden),
        "forget": slice(hidden, 2*hidden),
        "cell":   slice(2*hidden, 3*hidden),
        "output": slice(3*hidden, 4*hidden),
    }

    gate_data = {}
    for g in GATE_KEYS:
        sl = gate_slices[g]
        W = wih[sl]
        R = whh[sl]
        b = bih[sl] + bhh[sl]
        W_q, w_scale = quantize_weights(W)
        R_q, r_scale = quantize_weights(R)
        b_q = quantize_bias(b, INPUT_SCALE * w_scale)
        eff_in  = b_q.copy()
        eff_hid = np.zeros(hidden, dtype=np.int32)
        i2g_eff = INPUT_SCALE * w_scale / INTER_SCALE
        r2g_eff = HIDDEN_SCALE * r_scale / INTER_SCALE
        i_mult, i_shift = quantize_scale(i2g_eff)
        r_mult, r_shift = quantize_scale(r2g_eff)
        gate_data[g] = {
            "W_q": W_q, "R_q": R_q,
            "eff_in": eff_in, "eff_hid": eff_hid,
            "i_mult": i_mult, "i_shift": i_shift,
            "r_mult": r_mult, "r_shift": r_shift,
        }

    ftc_scale = math.pow(2, -15)
    itc_scale = math.pow(2, -15)
    ftc_mult, ftc_shift = quantize_scale(ftc_scale)
    itc_mult, itc_shift = quantize_scale(itc_scale)

    eff_hidden_scale = math.pow(2, -15) / HIDDEN_SCALE * math.pow(2, -15)
    out_mult, out_shift = quantize_scale(eff_hidden_scale)

    fc_Wq, fc_ws = quantize_weights(wfc_arr)
    fc_bq = quantize_bias(bfc_arr, HIDDEN_SCALE * fc_ws)
    fc_mult, fc_shift = quantize_scale(HIDDEN_SCALE * fc_ws / (1.0 / 127.0))

    weights_path = os.path.join(MODEL_DIR, f"lstm_h{hidden}_weights.h")
    params_path  = os.path.join(MODEL_DIR, f"lstm_h{hidden}_params.h")

    with open(weights_path, "w") as f:
        guard = f"LSTM_H{hidden}_WEIGHTS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
        for g in GATE_KEYS:
            d = gate_data[g]
            f.write(array_to_c(f"{g}_input_weights",  d["W_q"],    "int8_t")  + "\n\n")
            f.write(array_to_c(f"{g}_hidden_weights", d["R_q"],    "int8_t")  + "\n\n")
            f.write(array_to_c(f"{g}_eff_bias_input",  d["eff_in"],  "int32_t") + "\n\n")
            f.write(array_to_c(f"{g}_eff_bias_hidden", d["eff_hid"], "int32_t") + "\n\n")
        f.write(array_to_c("fc_weights", fc_Wq, "int8_t")  + "\n\n")
        f.write(array_to_c("fc_bias",    fc_bq, "int32_t") + "\n\n")
        f.write("#endif\n")

    with open(params_path, "w") as f:
        guard = f"LSTM_H{hidden}_PARAMS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define LSTM_INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define LSTM_HIDDEN_SIZE {hidden}\n")
        f.write(f"#define LSTM_SEQ_LEN {SEQ_LEN}\n")
        f.write(f"#define LSTM_BATCH_SIZE 1\n")
        f.write(f"#define LSTM_CELL_SCALE_POWER {CELL_SCALE_PWR}\n")
        f.write(f"#define LSTM_CELL_CLIP 32767\n")
        f.write(f"#define LSTM_INPUT_OFFSET 0\n")
        f.write(f"#define LSTM_OUTPUT_OFFSET 0\n")
        f.write(f"#define LSTM_INPUT_SCALE_F32 {INPUT_SCALE:.10f}f\n")
        f.write(f"#define LSTM_FORGET_TO_CELL_MULT {ftc_mult}\n")
        f.write(f"#define LSTM_FORGET_TO_CELL_SHIFT {ftc_shift}\n")
        f.write(f"#define LSTM_INPUT_TO_CELL_MULT {itc_mult}\n")
        f.write(f"#define LSTM_INPUT_TO_CELL_SHIFT {itc_shift}\n")
        f.write(f"#define LSTM_OUTPUT_MULT {out_mult}\n")
        f.write(f"#define LSTM_OUTPUT_SHIFT {out_shift}\n")
        f.write(f"#define LSTM_FC_MULT {fc_mult}\n")
        f.write(f"#define LSTM_FC_SHIFT {fc_shift}\n")
        for g in GATE_KEYS:
            d = gate_data[g]
            ip = GATE_PREFIX[g]
            rp = GATE_RPREFIX[g]
            f.write(f"#define LSTM_{ip}_MULT {d['i_mult']}\n")
            f.write(f"#define LSTM_{ip}_SHIFT {d['i_shift']}\n")
            f.write(f"#define LSTM_{rp}_MULT {d['r_mult']}\n")
            f.write(f"#define LSTM_{rp}_SHIFT {d['r_shift']}\n")
        f.write("#endif\n")

    print(f"  Generated {weights_path}, {params_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
