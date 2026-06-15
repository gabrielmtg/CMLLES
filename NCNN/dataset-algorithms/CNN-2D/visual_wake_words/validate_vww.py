import numpy as np
import ncnn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pyvww
import time

PARAM_PATH = "model/vww_96_model.ncnn.param"
BIN_PATH = "model/vww_96_model.ncnn.bin"
VAL_IMAGES_ROOT = "../../../../datasets/CNN-2D/coco/images/all2017"
VAL_ANN_FILE    = "../../../../datasets/CNN-2D/coco/annotations/instances_val.json"
BATCH_SIZE      = 1

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    val_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=VAL_IMAGES_ROOT,
        annFile=VAL_ANN_FILE,
        transform=transform
    )
    val_loader = __import__('torch').utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    net = ncnn.Net()
    net.load_param(PARAM_PATH)
    net.load_model(BIN_PATH)

    all_preds = []
    all_labels = []
    total_inference_time = 0.0

    for images, labels in tqdm(val_loader, desc="Validando", unit="batch"):
        images_np = images.numpy()

        t0 = time.perf_counter()
        ex = net.create_extractor()
        in_mat = ncnn.Mat(images_np[0])
        ex.input("in0", in_mat)
        _, out_mat = ex.extract("out0")
        outputs = np.array(out_mat).reshape(1, -1)
        t1 = time.perf_counter()
        total_inference_time += (t1 - t0)

        preds = np.argmax(outputs, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total_samples = len(all_labels)

    correct = (all_preds == all_labels).sum()
    accuracy = correct / total_samples

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

if __name__ == "__main__":
    main()
