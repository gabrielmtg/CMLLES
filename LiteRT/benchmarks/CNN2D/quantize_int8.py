import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pyvww
import ai_edge_quantizer as aq

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

for cfg_name, img_size in CONFIGS.items():
    f32 = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_f32.tflite")
    i8  = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_int8.tflite")
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

    quantizer = aq.Quantizer(bytearray(open(f32, "rb").read()))
    quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
    calib = {"serving_default": [{"args_0": calib_np[i:i+1]} for i in range(len(calib_np))]}
    i8_dir, i8_name = os.path.split(i8)
    quantizer.quantize(quantizer.calibrate(calib)).save(i8_dir, i8_name.replace(".tflite", ""), overwrite=True)
    print(f"Quantized {i8}")
