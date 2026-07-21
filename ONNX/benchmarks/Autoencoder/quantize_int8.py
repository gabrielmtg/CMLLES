import os
import numpy as np
import pandas as pd
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

TRAIN_PATH = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
TEST_PATH  = "../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
MODEL_DIR  = "model"

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


df_tr = pd.read_csv(TRAIN_PATH, names=COL_NAMES).drop(["difficulty"], axis=1)
df_te = pd.read_csv(TEST_PATH,  names=COL_NAMES).drop(["difficulty"], axis=1)
df_tr["is_train"] = 1
df_te["is_train"] = 0
df_all = pd.concat([df_tr, df_te], ignore_index=True)
df_all = pd.get_dummies(df_all, columns=CAT_COLS, dtype=float)
df_all["target"] = df_all["label"].apply(lambda x: 0.0 if x == "normal" else 1.0)
df_all.drop(["label"], axis=1, inplace=True)
df_tr2 = df_all[df_all["is_train"] == 1].drop(["is_train"], axis=1).copy()
feat_cols = [c for c in df_tr2.columns if c != "target"]
for col in feat_cols:
    mn, mx = df_tr2[col].min(), df_tr2[col].max()
    rng = mx - mn if mx != mn else 1e-9
    df_tr2[col] = (df_tr2[col] - mn) / rng
df_normal = df_tr2[df_tr2["target"] == 0.0]
X_calib = df_normal[feat_cols].values.astype(np.float32)

for size in ["small", "medium", "large"]:
    f32 = os.path.join(MODEL_DIR, f"ae_{size}_f32.onnx")
    i8  = os.path.join(MODEL_DIR, f"ae_{size}_int8.onnx")
    if not os.path.exists(f32):
        print(f"SKIP {f32} (not found)")
        continue
    quantize_static(f32, i8, _CalibReader(X_calib),
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
    print(f"Quantized {i8}")
