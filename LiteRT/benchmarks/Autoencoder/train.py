import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import litert_torch as ai_edge_torch
import ai_edge_quantizer as aq

TRAIN_PATH = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
TEST_PATH  = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
MODEL_DIR  = "model"
EPOCHS     = 50
BATCH_SIZE = 256
LR         = 0.001
SEED       = 42

COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CAT_COLS = ["protocol_type", "service", "flag"]

AE_CONFIGS = {
    "small":  [122, 64, 32, 16],
    "medium": [122, 96, 48, 24],
    "large":  [122, 128, 64, 32],
}


def load_data(train_path, test_path):
    df_tr = pd.read_csv(train_path, names=COL_NAMES).drop(["difficulty"], axis=1)
    df_te = pd.read_csv(test_path,  names=COL_NAMES).drop(["difficulty"], axis=1)
    df_tr["is_train"] = 1
    df_te["is_train"] = 0
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=CAT_COLS, dtype=float)
    df_all["target"] = df_all["label"].apply(lambda x: 0.0 if x == "normal" else 1.0)
    df_all.drop(["label"], axis=1, inplace=True)
    df_tr = df_all[df_all["is_train"] == 1].drop(["is_train"], axis=1).copy()
    df_te = df_all[df_all["is_train"] == 0].drop(["is_train"], axis=1).copy()
    feat_cols = [c for c in df_tr.columns if c != "target"]
    for col in feat_cols:
        mn, mx = df_tr[col].min(), df_tr[col].max()
        rng = mx - mn if mx != mn else 1e-9
        df_tr[col] = (df_tr[col] - mn) / rng
        df_te[col] = ((df_te[col] - mn) / rng).clip(0, 1)
    return df_tr, df_te, feat_cols


def build_autoencoder(dims):
    enc = list(dims)
    dec = list(reversed(dims[:-1]))
    all_dims = enc + dec
    layers = []
    for i in range(len(all_dims) - 1):
        layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        if i < len(all_dims) - 2:
            layers.append(nn.LeakyReLU(0.1))
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

df_train, df_test, feat_cols = load_data(TRAIN_PATH, TEST_PATH)
input_dim = len(feat_cols)
print(f"Input dim: {input_dim} features, train: {len(df_train)}, test: {len(df_test)}")

test_data_path = os.path.join(MODEL_DIR, "test_data.txt")
with open(test_data_path, "w") as f:
    f.write(f"{len(df_test)} {input_dim} 1\n")
with open(test_data_path, "a") as f:
    for start in range(0, len(df_test), 100000):
        arr = df_test[feat_cols + ["target"]].iloc[start:start + 100000].values
        np.savetxt(f, arr, fmt="%.6f", delimiter=" ")
print(f"Test data saved: {len(df_test)} samples")

df_normal = df_train[df_train["target"] == 0.0]
X_normal = torch.tensor(df_normal[feat_cols].values, dtype=torch.float32)

metrics = {}

for size, dims in AE_CONFIGS.items():
    key = f"ae_{size}"
    model = build_autoencoder(dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(X_normal, X_normal)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    final_loss = 0.0
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        recon = model(X_normal)
        mse_train = ((recon - X_normal) ** 2).mean(dim=1).numpy()
    threshold = float(np.mean(mse_train) + 2 * np.std(mse_train))

    metrics[key] = {
        "training_time_s": round(elapsed, 3),
        "epochs": EPOCHS,
        "final_train_loss": round(final_loss, 6),
        "threshold": round(threshold, 8),
    }
    print(f"{key}: loss={final_loss:.5f} threshold={threshold:.5f} time={elapsed:.1f}s")

    with open(os.path.join(MODEL_DIR, f"ae_{size}_threshold.txt"), "w") as f:
        f.write(f"{threshold:.8f}\n")

    dummy = (torch.randn(1, input_dim),)
    path = os.path.join(MODEL_DIR, f"ae_{size}_f32.tflite")
    edge_model = ai_edge_torch.convert(model.eval(), dummy)
    edge_model.export(path)
    print(f"  Exported {path}")
    int8_path = path.replace("_f32.tflite", "_int8.tflite")
    f32_buf = bytearray(open(path, "rb").read())
    quantizer = aq.Quantizer(f32_buf)
    quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
    X_calib = X_normal.numpy()
    calib_data = {"serving_default": [{"args_0": X_calib[i:i+1]} for i in range(min(len(X_calib), 100))]}
    calib_res = quantizer.calibrate(calib_data)
    int8_dir, int8_name = os.path.split(int8_path)
    quantizer.quantize(calib_res).save(int8_dir, int8_name.replace(".tflite", ""), overwrite=True)
    print(f"  Quantized {int8_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
