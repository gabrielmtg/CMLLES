import os
import json
import time
import struct
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pyvww

IMAGES_DIR = "../../../datasets/CNN-2D/coco/images/all2017"
TRAIN_ANN  = "../../../datasets/CNN-2D/coco/annotations/instances_train.json"
VAL_ANN    = "../../../datasets/CNN-2D/coco/annotations/instances_minival.json"
MODEL_DIR  = "model"
EPOCHS     = 5
BATCH      = 64
LR         = 0.001
SEED       = 42
N_TEST     = 100
INPUT_SCALE = 1.0 / 127.0

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, 2)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class IntermediateConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, 2)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class MobileNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 6 * 6, 2)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


CNN_CONFIGS = {
    "simple": {
        "model": SimpleConvNet,
        "img_size": 32,
        "conv_layers": [
            {"in_c": 3,   "out_c": 16,  "k": 3, "s": 2, "p": 1, "in_h": 32, "in_w": 32},
            {"in_c": 16,  "out_c": 32,  "k": 3, "s": 2, "p": 1, "in_h": 16, "in_w": 16},
            {"in_c": 32,  "out_c": 64,  "k": 3, "s": 2, "p": 1, "in_h": 8,  "in_w": 8},
        ],
        "pool_h": 4, "pool_w": 4, "final_c": 64,
    },
    "intermediate": {
        "model": IntermediateConvNet,
        "img_size": 32,
        "conv_layers": [
            {"in_c": 3,   "out_c": 32,  "k": 3, "s": 2, "p": 1, "in_h": 32, "in_w": 32},
            {"in_c": 32,  "out_c": 64,  "k": 3, "s": 2, "p": 1, "in_h": 16, "in_w": 16},
            {"in_c": 64,  "out_c": 128, "k": 3, "s": 2, "p": 1, "in_h": 8,  "in_w": 8},
        ],
        "pool_h": 4, "pool_w": 4, "final_c": 128,
    },
    "mobilenet": {
        "model": MobileNetLike,
        "img_size": 96,
        "conv_layers": [
            {"in_c": 3,   "out_c": 32,  "k": 3, "s": 2, "p": 1, "in_h": 96, "in_w": 96},
            {"in_c": 32,  "out_c": 64,  "k": 3, "s": 2, "p": 1, "in_h": 48, "in_w": 48},
            {"in_c": 64,  "out_c": 128, "k": 3, "s": 2, "p": 1, "in_h": 24, "in_w": 24},
            {"in_c": 128, "out_c": 128, "k": 3, "s": 2, "p": 1, "in_h": 12, "in_w": 12},
        ],
        "pool_h": 6, "pool_w": 6, "final_c": 128,
    },
}


def make_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def save_test_images(path, imgs):
    n, c, h, w = imgs.shape
    with open(path, "wb") as f:
        f.write(struct.pack("iiii", n, c, h, w))
        imgs.numpy().astype(np.float32).tofile(f)


def save_test_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(labels)))
        np.array(labels, dtype=np.int32).tofile(f)


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


val_ds32 = pyvww.pytorch.VisualWakeWordsClassification(root=IMAGES_DIR, annFile=VAL_ANN, transform=make_transform(32))
val_ds96 = pyvww.pytorch.VisualWakeWordsClassification(root=IMAGES_DIR, annFile=VAL_ANN, transform=make_transform(96))

imgs32, imgs96, labs = [], [], []
for i in range(N_TEST):
    img32, lbl = val_ds32[i]
    img96, _   = val_ds96[i]
    imgs32.append(img32); imgs96.append(img96); labs.append(int(lbl))

save_test_images(os.path.join(MODEL_DIR, "test_images_32.bin"), torch.stack(imgs32))
save_test_images(os.path.join(MODEL_DIR, "test_images_96.bin"), torch.stack(imgs96))
save_test_labels(os.path.join(MODEL_DIR, "test_labels.bin"), labs)
print(f"Test data saved: {N_TEST} samples")

metrics = {}

for cfg_name, cfg in CNN_CONFIGS.items():
    key      = f"cnn_{cfg_name}"
    img_size = cfg["img_size"]
    model    = cfg["model"]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_ds  = pyvww.pytorch.VisualWakeWordsClassification(
        root=IMAGES_DIR, annFile=TRAIN_ANN, transform=make_transform(img_size))
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)

    t0 = time.time()
    final_loss = 0.0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)
        print(f"  {key} epoch {epoch+1}/{EPOCHS}: loss={final_loss:.4f}")
    elapsed = time.time() - t0

    metrics[key] = {
        "training_time_s": round(elapsed, 3),
        "epochs": EPOCHS,
        "final_train_loss": round(final_loss, 6),
    }
    print(f"{key}: loss={final_loss:.4f} time={elapsed:.1f}s")

    model_cpu = model.to("cpu").eval()
    conv_layers = [(m.weight.detach().numpy(), m.bias.detach().numpy())
                   for m in model_cpu.features if isinstance(m, nn.Conv2d)]
    fc_W = model_cpu.fc.weight.detach().numpy()
    fc_b = model_cpu.fc.bias.detach().numpy()

    conv_weights_q, conv_biases_q, conv_w_scales = [], [], []
    in_scale = INPUT_SCALE
    for W_f, b_f in conv_layers:
        W_hwio = W_f.transpose(0, 2, 3, 1)
        Wq, ws = quantize_weights(W_hwio)
        bq = quantize_bias(b_f, in_scale, ws)
        conv_weights_q.append(Wq)
        conv_biases_q.append(bq)
        conv_w_scales.append(ws)
        in_scale = ws

    fc_Wq, fc_ws = quantize_weights(fc_W)
    fc_bq = quantize_bias(fc_b, in_scale, fc_ws)
    fc_in  = cfg["pool_h"] * cfg["pool_w"] * cfg["final_c"]
    fc_out = 2

    variant      = f"cnn_{cfg_name}"
    weights_path = os.path.join(MODEL_DIR, f"{variant}_weights.h")
    params_path  = os.path.join(MODEL_DIR, f"{variant}_params.h")

    with open(weights_path, "w") as f:
        guard = f"CNN_{cfg_name.upper()}_WEIGHTS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
        for i, (Wq, bq) in enumerate(zip(conv_weights_q, conv_biases_q)):
            f.write(array_to_c(f"conv{i}_weights", Wq, "int8_t") + "\n\n")
            f.write(array_to_c(f"conv{i}_bias",    bq, "int32_t") + "\n\n")
        f.write(array_to_c("fc_weights", fc_Wq, "int8_t") + "\n\n")
        f.write(array_to_c("fc_bias",    fc_bq, "int32_t") + "\n\n")
        f.write("#endif\n")

    with open(params_path, "w") as f:
        guard = f"CNN_{cfg_name.upper()}_PARAMS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        in_c0, in_h0, in_w0 = 3, img_size, img_size
        f.write(f"#define IN_CHANNELS {in_c0}\n")
        f.write(f"#define IN_HEIGHT   {in_h0}\n")
        f.write(f"#define IN_WIDTH    {in_w0}\n")
        f.write(f"#define FC_IN_SIZE  {fc_in}\n")
        f.write(f"#define FC_OUT_SIZE {fc_out}\n")
        f.write(f"#define INPUT_SCALE_F32 {INPUT_SCALE:.10f}f\n")
        n_conv = len(conv_w_scales)
        f.write(f"#define NUM_CONV_LAYERS {n_conv}\n\n")
        mults = [int(round(s * (1 << 20))) for s in conv_w_scales]
        f.write("static const int32_t CONV_MULTIPLIERS[] = {")
        f.write(", ".join(str(m) for m in mults))
        f.write("};\n")
        f.write("static const int CONV_SHIFTS[] = {")
        f.write(", ".join("-20" for _ in mults))
        f.write("};\n")
        f.write(f"#define FC_MULTIPLIER {int(round(fc_ws * (1 << 20)))}\n")
        f.write("#define FC_SHIFT -20\n\n")
        f.write(f"#define NUM_LAYER_INFO {n_conv}\n")
        f.write("typedef struct { int type; int in_c, out_c, k, s, p, in_h, in_w, out_h, out_w; } LayerInfo;\n")
        f.write("static const LayerInfo LAYER_INFO[] = {\n")
        for li in cfg["conv_layers"]:
            out_h = (li["in_h"] + 2*li["p"] - li["k"]) // li["s"] + 1
            out_w = (li["in_w"] + 2*li["p"] - li["k"]) // li["s"] + 1
            f.write(f"    {{0, {li['in_c']}, {li['out_c']}, {li['k']}, {li['s']}, {li['p']}, {li['in_h']}, {li['in_w']}, {out_h}, {out_w}}},\n")
        f.write("};\n\n#endif\n")

    print(f"  Generated {weights_path}, {params_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
