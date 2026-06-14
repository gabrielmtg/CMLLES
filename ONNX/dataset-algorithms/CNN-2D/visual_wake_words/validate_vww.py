import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pyvww
import time


ONNX_MODEL_PATH = "model/vww_96_model.onnx"
VAL_IMAGES_ROOT = "../../../../datasets/CNN-2D/coco/images/all2017"
VAL_ANN_FILE    = "../../../../datasets/CNN-2D/coco/annotations/instances_val.json"
BATCH_SIZE      = 64


transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    print("Carregando dataset de validação...")
    val_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=VAL_IMAGES_ROOT,
        annFile=VAL_ANN_FILE,
        transform=transform
    )
    val_loader = __import__('torch').utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    print(f"Total de amostras de validação: {len(val_dataset)}")

    print(f"Carregando modelo ONNX: {ONNX_MODEL_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    active_provider = session.get_providers()[0]
    print(f"Provider ativo: {active_provider}")
    print(f"Input: '{input_name}'  |  Output: '{output_name}'")

    all_preds = []
    all_labels = []
    total_inference_time = 0.0

    print("\nIniciando validação...")
    for images, labels in tqdm(val_loader, desc="Validando", unit="batch"):
        images_np = images.numpy()

        t0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: images_np})[0]
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

    avg_latency_ms = (total_inference_time / total_samples) * 1000
    throughput = total_samples / total_inference_time

    print("\n" + "=" * 55)
    print("       RESULTADOS DA VALIDAÇÃO — Visual Wake Words")
    print("=" * 55)
    print(f"  Modelo:           {ONNX_MODEL_PATH}")
    print(f"  Provider:         {active_provider}")
    print(f"  Total amostras:   {total_samples}")
    print("-" * 55)
    print(f"  Acurácia:         {accuracy:.4f}  ({correct}/{total_samples})")
    print(f"  Precisão (P=1):   {precision:.4f}")
    print(f"  Recall (P=1):     {recall:.4f}")
    print(f"  F1-Score (P=1):   {f1:.4f}")
    print("-" * 55)
    print(f"  Latência média:   {avg_latency_ms:.3f} ms/amostra")
    print(f"  Throughput:       {throughput:.1f} amostras/s")
    print("-" * 55)
    print("  Matriz de Confusão:")
    print(f"                    Pred=No Person   Pred=Person")
    print(f"  Real=No Person    {tn:>10}       {fp:>10}")
    print(f"  Real=Person       {fn:>10}       {tp:>10}")
    print("=" * 55)

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0   = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    print(f"\n  Por classe:")
    print(f"    No Person  — Precisão: {prec_0:.4f}  Recall: {rec_0:.4f}  F1: {f1_0:.4f}")
    print(f"    Person     — Precisão: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print()


if __name__ == "__main__":
    main()
