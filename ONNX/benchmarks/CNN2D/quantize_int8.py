import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pyvww
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

IMAGES_DIR = "../../../datasets/CNN-2D/coco/images/all2017"
TRAIN_ANN  = "../../../datasets/CNN-2D/coco/annotations/instances_train.json"
MODEL_DIR  = "model"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CONFIGS = {
    "simple":       32,
    "intermediate": 32,
    "mobilenet":    96,
}


class _CalibReader(CalibrationDataReader):
    def __init__(self, X):
        self.data = [{"input": X[i:i+1]} for i in range(len(X))]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d


for cfg_name, img_size in CONFIGS.items():
    f32 = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_f32.onnx")
    i8  = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_int8.onnx")
    if not os.path.exists(f32):
        print(f"SKIP {f32} (not found)")
        continue

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    train_ds = pyvww.pytorch.VisualWakeWordsClassification(
        root=IMAGES_DIR, annFile=TRAIN_ANN, transform=transform)
    calib_imgs = []
    for imgs, _ in DataLoader(train_ds, batch_size=1, shuffle=False):
        calib_imgs.append(imgs.numpy())
        if len(calib_imgs) >= 50:
            break
    calib_np = np.concatenate(calib_imgs, axis=0)

    quantize_static(f32, i8, _CalibReader(calib_np),
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
    print(f"Quantized {i8}")
