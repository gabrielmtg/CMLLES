import torch
from torchvision import transforms
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = "/home/gabriel/bolsa/artigo-SBESC/CMLLES/datasets/CNN-2D/coco/images/train2017/000000000036.jpg"
img = Image.open(img_path).convert('RGB')
tensor = transform(img)

tensor.numpy().astype(np.float32).tofile("imagem_teste.bin")
print("Tensor exportado para imagem_teste.bin com sucesso!")
