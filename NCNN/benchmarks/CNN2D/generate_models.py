import os
import torch
import torch.nn as nn
import pnnx

SEED = 42
torch.manual_seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class IntermediateCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class MobileNetLike(nn.Module):
    def __init__(self):
        super().__init__()

        def dw_sep(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin), nn.ReLU(),
                nn.Conv2d(cin, cout, 1), nn.ReLU(),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            dw_sep(32, 64),
            dw_sep(64, 128, stride=2),
            dw_sep(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


configs = {
    "simple":       (SimpleCNN(),       (1, 1, 32, 32)),
    "intermediate": (IntermediateCNN(), (1, 1, 32, 32)),
    "mobilenet":    (MobileNetLike(),   (1, 3, 96, 96)),
}

for name, (model, shape) in configs.items():
    model.eval()
    dummy = torch.randn(*shape)
    pt_path = os.path.join(MODEL_DIR, f"cnn_{name}_f32.pt")
    torch.jit.save(torch.jit.trace(model, dummy), pt_path)
    pnnx.export(model, pt_path, dummy)
    param_src = pt_path.replace(".pt", ".ncnn.param")
    if os.path.exists(param_src):
        print(f"Exported {param_src}")
    else:
        print(f"WARNING: pnnx did not produce {param_src}")
