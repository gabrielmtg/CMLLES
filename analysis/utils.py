from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

FRAMEWORK_ORDER  = ["FANN", "Genann", "NCNN", "LiteRT", "ONNX", "CMSIS-NN"]
INT8_FRAMEWORKS  = ["CMSIS-NN", "LiteRT", "ONNX"]
PRECISION_ORDER  = ["f32", "int8"]

COLORS = {
    "FANN":     "#0072B2",
    "Genann":   "#E69F00",
    "NCNN":     "#009E73",
    "LiteRT":   "#CC79A7",
    "ONNX":     "#D55E00",
    "CMSIS-NN": "#56B4E9",
}

HATCHES = {
    "FANN":     "",
    "Genann":   "",
    "NCNN":     "",
    "LiteRT":   "",
    "ONNX":     "",
    "CMSIS-NN": "",
}

_FW_PREFIX = {
    "fann":    "FANN",
    "genann":  "Genann",
    "ncnn":    "NCNN",
    "litert":  "LiteRT",
    "onnx":    "ONNX",
    "cmsisnn": "CMSIS-NN",
}

SIZE_ORDER   = ["small", "medium", "large"]
HIDDEN_ORDER = ["h32", "h64", "h128"]
CNN_ORDER    = ["simple", "intermediate", "mobilenet"]

SIZE_LABEL   = {"small": "S", "medium": "M", "large": "L"}
HIDDEN_LABEL = {"h32": "H=32", "h64": "H=64", "h128": "H=128"}
CNN_LABEL    = {"simple": "Simple", "intermediate": "Interm.", "mobilenet": "MobileNet"}


def _fw_from_name(name):
    n = name.lower()
    for prefix, fw in _FW_PREFIX.items():
        if n.startswith(prefix):
            return fw
    return None


def _algo_from_bench(name):
    n = name.lower()
    for prefix in _FW_PREFIX:
        pfx = prefix + "_"
        if n.startswith(pfx):
            rest = n[len(pfx):]
            if rest.startswith("bench_"):
                rest = rest[6:]
            if rest.startswith("mlp"):
                return "MLP-Iris"
            if rest.startswith("iaes"):
                return "MLP-IAES"
            if rest.startswith("ae"):
                return "Autoencoder"
            if rest.startswith("cnn2d"):
                return "CNN2D"
            if rest.startswith("lstm"):
                return "LSTM"
    return None


def load_latencies():
    lat_dir = RESULTS_DIR / "latencies_rpi"
    rows = []
    for csv_path in sorted(lat_dir.rglob("*.csv")):
        parts = csv_path.parts
        try:
            idx = next(i for i, p in enumerate(parts) if p == "latencies_rpi")
        except StopIteration:
            continue
        fw = parts[idx + 1]
        try:
            bidx = next(i for i, p in enumerate(parts) if p == "benchmarks")
            algo_dir = parts[bidx + 1]
            if algo_dir == "MLP" and bidx + 2 < len(parts):
                algo = "MLP-" + parts[bidx + 2]
            else:
                algo = algo_dir
        except (StopIteration, IndexError):
            continue
        try:
            df = pd.read_csv(csv_path)
            stem = csv_path.stem
            if "_int8" in stem:
                prec = "int8"
            elif "_f32" in stem:
                prec = "f32"
            else:
                prec = "int8" if fw == "CMSIS-NN" else "f32"
            df["framework"] = fw
            df["algorithm"] = algo
            df["precision"] = prec
            rows.append(df)
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_perf():
    df = pd.read_csv(RESULTS_DIR / "perf_metrics_all.csv")
    df["framework"] = df["benchmark"].apply(_fw_from_name)
    df["algorithm"] = df["benchmark"].apply(_algo_from_bench)

    def _variant(name):
        n = name.lower()
        for prefix in _FW_PREFIX:
            pfx = prefix + "_"
            if n.startswith(pfx):
                rest = n[len(pfx):]
                for tag in ("mlp_", "iaes_", "ae_", "cnn2d_", "lstm_"):
                    if rest.startswith(tag):
                        return rest[len(tag):]
        return name

    df["variant"] = df["benchmark"].apply(_variant)
    return df.dropna(subset=["framework", "algorithm"])


def load_binary():
    df = pd.read_csv(RESULTS_DIR / "binary_metrics.csv")
    df["framework"] = df["executable"].apply(_fw_from_name)
    df["algorithm"] = df["executable"].apply(_algo_from_bench)
    return df.dropna(subset=["framework", "algorithm"])


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#BBBBBB",
        "axes.labelcolor": "#1A2F55",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "text.color": "#1A2F55",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DDDDDD",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
