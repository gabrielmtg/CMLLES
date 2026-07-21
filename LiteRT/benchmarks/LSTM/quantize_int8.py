import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ai_edge_quantizer as aq

TRAIN_PATH = "../../../datasets/LSTM/CMAPSSData/train_FD001.txt"
MODEL_DIR  = "model"
SEQ_LEN    = 30
MAX_RUL    = 125
FEATURES   = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
COLS       = ["id","cycle","set1","set2","set3"] + [f"s{i}" for i in range(1, 22)]

scaler = StandardScaler()
df = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLS)
rul = df.groupby("id")["cycle"].max().reset_index()
rul.columns = ["id", "max"]
df = df.merge(rul, on="id", how="left")
df["RUL"] = (df["max"] - df["cycle"]).clip(upper=MAX_RUL) / MAX_RUL
df.drop("max", axis=1, inplace=True)
df[FEATURES] = scaler.fit_transform(df[FEATURES])
X_seqs = []
for eid in df["id"].unique():
    ed = df[df["id"] == eid]
    feats = ed[FEATURES].values
    for i in range(len(ed) - SEQ_LEN):
        X_seqs.append(feats[i:i + SEQ_LEN])
X_train = np.array(X_seqs, dtype=np.float32)

for hidden in [32, 64, 128]:
    f32 = os.path.join(MODEL_DIR, f"lstm_h{hidden}_f32.tflite")
    i8  = os.path.join(MODEL_DIR, f"lstm_h{hidden}_int8.tflite")
    if not os.path.exists(f32):
        print(f"SKIP {f32} (not found)")
        continue
    quantizer = aq.Quantizer(bytearray(open(f32, "rb").read()))
    quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
    calib = {"serving_default": [{"args_0": X_train[i:i+1]} for i in range(min(50, len(X_train)))]}
    i8_dir, i8_name = os.path.split(i8)
    quantizer.quantize(quantizer.calibrate(calib)).save(i8_dir, i8_name.replace(".tflite", ""), overwrite=True)
    print(f"Quantized {i8}")
