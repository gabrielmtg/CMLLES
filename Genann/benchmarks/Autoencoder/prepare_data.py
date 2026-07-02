import os
import numpy as np
import pandas as pd

TRAIN_PATH = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
TEST_PATH  = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
MODEL_DIR  = "model"
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

os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)

df_tr = pd.read_csv(TRAIN_PATH, names=COL_NAMES).drop(["difficulty"], axis=1)
df_te = pd.read_csv(TEST_PATH,  names=COL_NAMES).drop(["difficulty"], axis=1)
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

input_dim = len(feat_cols)
df_normal = df_tr[df_tr["target"] == 0.0]
X_normal  = df_normal[feat_cols].values.astype(np.float64)

with open(os.path.join(MODEL_DIR, "nsl_ae_train.data"), "w") as f:
    f.write(f"{len(X_normal)} {input_dim} {input_dim}\n")
    for xi in X_normal:
        line = " ".join(f"{v:.10f}" for v in xi)
        f.write(line + "\n")
        f.write(line + "\n")

X_test = df_te[feat_cols].values.astype(np.float64)
y_test = df_te["target"].values.astype(np.float64)
with open(os.path.join(MODEL_DIR, "test_data.txt"), "w") as f:
    f.write(f"{len(X_test)} {input_dim} 1\n")
    for xi, yi in zip(X_test, y_test):
        f.write(" ".join(f"{v:.10f}" for v in xi) + f" {int(yi)}\n")

print(f"Train (normal): {len(X_normal)} samples, dim={input_dim}")
print(f"Test: {len(X_test)} samples")
