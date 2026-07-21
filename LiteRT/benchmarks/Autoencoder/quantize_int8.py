import os
import numpy as np
import pandas as pd
import ai_edge_quantizer as aq

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
    r = mx - mn if mx != mn else 1e-9
    df_tr2[col] = (df_tr2[col] - mn) / r
df_normal = df_tr2[df_tr2["target"] == 0.0]
X_calib = df_normal[feat_cols].values.astype(np.float32)

for size in ["small", "medium", "large"]:
    f32 = os.path.join(MODEL_DIR, f"ae_{size}_f32.tflite")
    i8  = os.path.join(MODEL_DIR, f"ae_{size}_int8.tflite")
    if not os.path.exists(f32):
        print(f"SKIP {f32} (not found)")
        continue
    quantizer = aq.Quantizer(bytearray(open(f32, "rb").read()))
    quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
    calib = {"serving_default": [{"args_0": X_calib[i:i+1]} for i in range(min(100, len(X_calib)))]}
    i8_dir, i8_name = os.path.split(i8)
    quantizer.quantize(quantizer.calibrate(calib)).save(i8_dir, i8_name.replace(".tflite", ""), overwrite=True)
    print(f"Quantized {i8}")
