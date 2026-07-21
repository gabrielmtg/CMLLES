import os
import json
import time
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pyvww
import litert_torch as ai_edge_torch
import ai_edge_quantizer as aq

IMAGES_DIR = "../../../datasets/CNN-2D/coco/images/all2017"
TRAIN_ANN  = "../../../datasets/CNN-2D/coco/annotations/instances_train.json"
VAL_ANN    = "../../../datasets/CNN-2D/coco/annotations/instances_minival.json"
MODEL_DIR  = "model"
EPOCHS     = 5
BATCH      = 64
LR         = 0.001
SEED       = 42
N_TEST     = 100

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)


class IntermediateCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)


class MobileNetLike(nn.Module):
    @staticmethod
    def _dw_sep(in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c), nn.ReLU(),
            nn.Conv2d(in_c, out_c, 1), nn.ReLU(),
        )
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            self._dw_sep(32, 64),
            self._dw_sep(64, 128, stride=2),
            self._dw_sep(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)


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

CONFIGS = {
    "simple":       (SimpleCNN(),       32),
    "intermediate": (IntermediateCNN(), 32),
    "mobilenet":    (MobileNetLike(),   96),
}

metrics = {}

for cfg_name, (model, img_size) in CONFIGS.items():
    key = f"cnn_{cfg_name}"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_ds = pyvww.pytorch.VisualWakeWordsClassification(root=IMAGES_DIR, annFile=TRAIN_ANN, transform=make_transform(img_size))
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)

    calib_imgs = []
    for imgs, _ in DataLoader(train_ds, batch_size=1, shuffle=False):
        calib_imgs.append(imgs.cpu().numpy())
        if len(calib_imgs) >= 50:
            break
    calib_np = np.concatenate(calib_imgs, axis=0)

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
    dummy = (torch.randn(1, 3, img_size, img_size),)
    path = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_f32.tflite")
    edge_model = ai_edge_torch.convert(model_cpu, dummy)
    edge_model.export(path)
    print(f"  Exported {path}")
    int8_path = path.replace("_f32.tflite", "_int8.tflite")
    f32_buf = bytearray(open(path, "rb").read())
    quantizer = aq.Quantizer(f32_buf)
    quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
    calib_data = {"serving_default": [{"args_0": calib_np[i:i+1]} for i in range(len(calib_np))]}
    calib_res = quantizer.calibrate(calib_data)
    int8_dir, int8_name = os.path.split(int8_path)
    quantizer.quantize(calib_res).save(int8_dir, int8_name.replace(".tflite", ""), overwrite=True)
    print(f"  Quantized {int8_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
