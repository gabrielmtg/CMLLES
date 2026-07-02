import os
import json
import glob
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_DIRS = [
    "../../../../datasets/MLP/IAES-dataset/2-cores",
    "../../../../datasets/MLP/IAES-dataset/3-cores",
]
MODEL_DIR        = "model"
WINDOW           = 10
EPOCHS           = 200
LR               = 0.001
SEED             = 42
INPUT_SCALE      = 1.0 / 127.0
BATCH_SIZE        = 2048
MAX_TRAIN_SAMPLES = 50000
MAX_TEST_SAMPLES  = 1000

IAES_CONFIGS = {
    "small":  [12, 32, 64, 1],
    "medium": [12, 64, 128, 64, 1],
    "large":  [12, 128, 256, 128, 64, 1],
}
ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
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
    return nn.Sequential(*layers)


def save_test_data(path, X, y):
    n, nf = X.shape
    with open(path, "w") as f:
        f.write(f"{n} {nf} 1\n")
        for xi, yi in zip(X, y):
            f.write(" ".join(f"{v:.6f}" for v in xi) + f" {int(yi)}\n")


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, in_scale, w_scale):
    bias_scale = in_scale * w_scale
    if bias_scale == 0:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / bias_scale), -(2**31), 2**31 - 1).astype(np.int32)


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
        key     = f"iaes_{size}_{act_name}"
        variant = f"iaes_{size}_{act_name}"
        model   = build_mlp(dims, act_cls)
        criterion = nn.BCEWithLogitsLoss()
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

        n_fc = len(dims) - 1
        fc_layers = [(m.weight.detach().numpy(), m.bias.detach().numpy())
                     for m in model if isinstance(m, nn.Linear)]
        weights, biases, w_scales = [], [], []
        in_scale = INPUT_SCALE
        for W_f, b_f in fc_layers:
            Wq, ws = quantize_weights(W_f)
            bq = quantize_bias(b_f, in_scale, ws)
            weights.append(Wq)
            biases.append(bq)
            w_scales.append(ws)
            in_scale = ws

        weights_path = os.path.join(MODEL_DIR, f"{variant}_weights.h")
        params_path  = os.path.join(MODEL_DIR, f"{variant}_params.h")

        with open(weights_path, "w") as f:
            guard = variant.upper() + "_WEIGHTS_H"
            f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
            for i, (Wq, bq) in enumerate(zip(weights, biases)):
                f.write(array_to_c(f"fc{i}_weights", Wq, "int8_t") + "\n\n")
                f.write(array_to_c(f"fc{i}_bias", bq, "int32_t") + "\n\n")
            f.write("static const int8_t * const FC_WEIGHTS[] = {")
            f.write(", ".join(f"fc{i}_weights" for i in range(n_fc)))
            f.write("};\n")
            f.write("static const int32_t * const FC_BIAS[] = {")
            f.write(", ".join(f"fc{i}_bias" for i in range(n_fc)))
            f.write("};\n\n#endif\n")

        with open(params_path, "w") as f:
            guard = variant.upper() + "_PARAMS_H"
            f.write(f"#ifndef {guard}\n#define {guard}\n\n")
            f.write(f"#define NUM_FC_LAYERS {n_fc}\n")
            f.write(f"#define INPUT_SCALE_F32 {INPUT_SCALE:.10f}f\n")
            f.write("static const int LAYER_SIZES[] = {")
            f.write(", ".join(str(d) for d in dims))
            f.write("};\n")
            mults, shifts = [], []
            for ws in w_scales:
                m = int(round(ws * (1 << 20)))
                mults.append(m)
                shifts.append(-20)
            f.write("static const int32_t FC_MULTIPLIERS[] = {")
            f.write(", ".join(str(m) for m in mults))
            f.write("};\n")
            f.write("static const int FC_SHIFTS[] = {")
            f.write(", ".join(str(s) for s in shifts))
            f.write("};\n\n#endif\n")

        print(f"  Generated {weights_path}, {params_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
