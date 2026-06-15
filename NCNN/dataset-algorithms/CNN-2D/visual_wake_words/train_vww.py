import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import pyvww 
import pnnx

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
    root="../../../../datasets/CNN-2D/coco/images/all2017", 
    annFile="../../../../datasets/CNN-2D/coco/annotations/instances_train.json",
    transform=transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, 2) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
EPOCHS = 5
for epoch in range(EPOCHS):
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

model.eval()
model.to("cpu")
dummy_input = torch.randn(1, 3, 96, 96)

pnnx.export(model, "model/vww_96_model.pt", dummy_input)
