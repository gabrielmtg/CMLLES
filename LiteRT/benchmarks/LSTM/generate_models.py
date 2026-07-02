import os
import torch
import torch.nn as nn
import litert_torch

SEED = 42
torch.manual_seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_SIZE = 10
OUTPUT_SIZE = 1
HIDDEN_SIZES = [32, 64, 128]
SEQ_LEN = 50


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


for hidden in HIDDEN_SIZES:
    model = LSTMModel(INPUT_SIZE, hidden, OUTPUT_SIZE).eval()
    dummy = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    path = os.path.join(MODEL_DIR, f"lstm_h{hidden}_f32.tflite")
    edge_model = litert_torch.convert(model, (dummy,))
    edge_model.export(path)
    print(f"Exported {path} ({os.path.getsize(path)} bytes)")
