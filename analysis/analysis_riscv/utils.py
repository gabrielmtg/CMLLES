from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# CMSIS-NN is excluded from the RISC-V scenario (ARM-only SIMD intrinsics,
# not portable). ONNX is best-effort (built from source) - it will simply be
# absent from the plots if that benchmark run didn't have it available.
FRAMEWORK_ORDER = ["FANN", "Genann", "NCNN", "LiteRT", "ONNX"]
INT8_FRAMEWORKS = ["LiteRT", "ONNX"]
PRECISION_ORDER = ["f32", "int8"]

COLORS = {
    "FANN":    "#0072B2",
    "Genann":  "#E69F00",
    "NCNN":    "#009E73",
    "LiteRT":  "#CC79A7",
    "ONNX":    "#D55E00",
}

HATCHES = {
    "FANN":    "",
    "Genann":  "",
    "NCNN":    "",
    "LiteRT":  "",
    "ONNX":    "",
}

_FW_PREFIX = {
    "fann":   "FANN",
    "genann": "Genann",
    "ncnn":   "NCNN",
    "litert": "LiteRT",
    "onnx":   "ONNX",
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
    lat_dir = RESULTS_DIR / "latencies_riscv"
    rows = []
    for csv_path in sorted(lat_dir.rglob("*.csv")):
        parts = csv_path.parts
        try:
            idx = next(i for i, p in enumerate(parts) if p == "latencies_riscv")
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
                prec = "f32"
            df["framework"] = fw
            df["algorithm"] = algo
            df["precision"] = prec
            rows.append(df)
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_ALGO_TAG = {
    "Autoencoder": "ae_",
    "LSTM":        "lstm_",
    "CNN2D":       "cnn_",
    "MLP-IAES":    "iaes_",
    "MLP-Iris":    "mlp_",
}


def load_perf():
    # perf_metrics_all_riscv.csv comes from wrapping benchmarks in an external
    # `perf stat` call - the RISC-V image has no `perf` binary, so that file is
    # always header-only. The real PMU data is collected in-process (raw
    # perf_event_open syscalls in the benchmark C code, see benchmark_iaes.c)
    # and written per-inference into latencies_riscv/*/latencies_*.csv, so we
    # aggregate those directly instead.
    lat_dir = RESULTS_DIR / "latencies_riscv"
    rows = []
    for csv_path in sorted(lat_dir.rglob("*.csv")):
        parts = csv_path.parts
        try:
            idx = next(i for i, p in enumerate(parts) if p == "latencies_riscv")
        except StopIteration:
            continue
        fw = parts[idx + 1]
        try:
            bidx = next(i for i, p in enumerate(parts) if p == "benchmarks")
            algo_dir = parts[bidx + 1]
            algo = "MLP-" + parts[bidx + 2] if algo_dir == "MLP" and bidx + 2 < len(parts) else algo_dir
        except (StopIteration, IndexError):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "cycles" not in df.columns:
            continue
        df = df[df["run"] != 1]
        if df.empty:
            continue

        stem = csv_path.stem[len("latencies_"):] if csv_path.stem.startswith("latencies_") else csv_path.stem
        prec = "int8" if stem.endswith("_int8") else "f32"
        tag = _ALGO_TAG.get(algo, "")
        variant = stem
        if variant.startswith(tag):
            variant = variant[len(tag):]
        for suffix in ("_int8", "_f32"):
            if variant.endswith(suffix):
                variant = variant[: -len(suffix)]
                break

        cyc_sum = df["cycles"].sum()
        ins_sum = df["instructions"].sum()
        l1l_sum = df["l1_loads"].sum()
        l1m_sum = df["l1_misses"].sum()
        brm_sum = df["branch_misses"].sum()

        rows.append({
            "benchmark":        f"{fw.lower()}_{stem}",
            "framework":        fw,
            "algorithm":        algo,
            "variant":          variant,
            "precision":        prec,
            "cycles":           cyc_sum,
            "instructions":     ins_sum,
            "ipc":              ins_sum / cyc_sum if cyc_sum else float("nan"),
            "l1_miss_rate_pct": 100.0 * l1m_sum / l1l_sum if l1l_sum else float("nan"),
            "branch_misses":    brm_sum,
        })

    return pd.DataFrame(rows)


def load_binary():
    df = pd.read_csv(RESULTS_DIR / "binary_metrics_riscv.csv")
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

    png_path = path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved {png_path}")
