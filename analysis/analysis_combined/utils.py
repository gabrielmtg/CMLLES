from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# CMSIS-NN is ARM-only (Cortex-M/A SIMD intrinsics), so it never appears on
# the RISC-V side - it's kept in FRAMEWORK_ORDER so RPi bars still render,
# it just has no RISC-V counterpart (same NaN/missing handling as the rest
# of this codebase).
FRAMEWORK_ORDER = ["FANN", "Genann", "NCNN", "LiteRT", "ONNX", "CMSIS-NN"]
INT8_FRAMEWORKS_BY_PLATFORM = {
    "RPi":    ["CMSIS-NN", "LiteRT", "ONNX"],
    "RISC-V": ["LiteRT", "ONNX"],
}
PRECISION_ORDER = ["f32", "int8"]

PLATFORM_ORDER = ["RPi", "RISC-V"]
PLATFORM_LABEL = {"RPi": "Raspberry Pi 4", "RISC-V": "VisionFive 2"}
PLATFORM_HATCH = {"RPi": "", "RISC-V": "///"}

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


def _load_latencies_rpi():
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
            algo = "MLP-" + parts[bidx + 2] if algo_dir == "MLP" and bidx + 2 < len(parts) else algo_dir
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
            df["platform"] = "RPi"
            rows.append(df)
        except Exception:
            continue
    return rows


def _load_latencies_riscv():
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
            stem = csv_path.stem
            prec = "int8" if "_int8" in stem else "f32"
            df["framework"] = fw
            df["algorithm"] = algo
            df["precision"] = prec
            df["platform"] = "RISC-V"
            rows.append(df)
        except Exception:
            continue
    return rows


def load_latencies():
    rows = _load_latencies_rpi() + _load_latencies_riscv()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _variant_from_bench(name):
    n = name.lower()
    for prefix in _FW_PREFIX:
        pfx = prefix + "_"
        if n.startswith(pfx):
            rest = n[len(pfx):]
            for tag in ("mlp_", "iaes_", "ae_", "cnn2d_", "lstm_"):
                if rest.startswith(tag):
                    return rest[len(tag):]
    return name


def _precision_from_bench(name):
    n = name.lower()
    return "int8" if "int8" in n else "f32"


def _load_perf_rpi():
    df = pd.read_csv(RESULTS_DIR / "perf_metrics_all.csv")
    df["framework"] = df["benchmark"].apply(_fw_from_name)
    df["algorithm"] = df["benchmark"].apply(_algo_from_bench)
    df["variant"] = df["benchmark"].apply(_variant_from_bench)
    df["precision"] = df["benchmark"].apply(_precision_from_bench)
    df["platform"] = "RPi"
    df = df.dropna(subset=["framework", "algorithm"])
    return df[["framework", "algorithm", "variant", "precision", "platform", "ipc", "l1_miss_rate_pct", "branch_misses"]]


_ALGO_TAG = {
    "Autoencoder": "ae_",
    "LSTM":        "lstm_",
    "CNN2D":       "cnn_",
    "MLP-IAES":    "iaes_",
    "MLP-Iris":    "mlp_",
}


def _load_perf_riscv():
    # RISC-V rootfs has no `perf` binary, so perf_metrics_all_riscv.csv is
    # always header-only. PMU data instead comes from raw perf_event_open
    # syscalls baked into each benchmark, written per-inference into
    # latencies_riscv/*/latencies_*.csv - aggregate those directly.
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
        tag = _ALGO_TAG.get(algo, "")
        variant = stem
        if variant.startswith(tag):
            variant = variant[len(tag):]
        precision = "f32"
        for suffix in ("_int8", "_f32"):
            if variant.endswith(suffix):
                precision = suffix[1:]
                variant = variant[: -len(suffix)]
                break

        cyc_sum = df["cycles"].sum()
        ins_sum = df["instructions"].sum()
        l1l_sum = df["l1_loads"].sum()
        l1m_sum = df["l1_misses"].sum()
        brm_sum = df["branch_misses"].sum()

        rows.append({
            "framework":        fw,
            "algorithm":        algo,
            "variant":          variant,
            "precision":        precision,
            "platform":         "RISC-V",
            "ipc":              ins_sum / cyc_sum if cyc_sum else float("nan"),
            "l1_miss_rate_pct": 100.0 * l1m_sum / l1l_sum if l1l_sum else float("nan"),
            "branch_misses":    brm_sum,
        })

    return pd.DataFrame(rows)


def load_perf():
    rpi = _load_perf_rpi()
    riscv = _load_perf_riscv()
    frames = [f for f in (rpi, riscv) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_binary_platform(csv_name, platform):
    df = pd.read_csv(RESULTS_DIR / csv_name)
    df["framework"] = df["executable"].apply(_fw_from_name)
    df["algorithm"] = df["executable"].apply(_algo_from_bench)
    df["platform"] = platform
    return df.dropna(subset=["framework", "algorithm"])


def load_binary():
    rpi = _load_binary_platform("binary_metrics.csv", "RPi")
    riscv = _load_binary_platform("binary_metrics_riscv.csv", "RISC-V")
    frames = [f for f in (rpi, riscv) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        "font.size": 11,
        "font.weight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DDDDDD",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def platform_legend_handles():
    """Two-patch legend naming the platforms via their hatch (RPi plain,
    RISC-V hatched) - shared by the overlay figures instead of each one
    explaining the wide/narrow-bar convention on its own."""
    return [
        mpatches.Patch(facecolor="white", edgecolor="#444444",
                        hatch=PLATFORM_HATCH[p], label=PLATFORM_LABEL[p])
        for p in PLATFORM_ORDER
    ]


def overlay_bar(ax, pos, val_rpi, val_riscv, color, size=0.7, horizontal=False,
                 err_rpi=None, err_riscv=None):
    """Bar-in-bar overlay: the larger of the two platform values is drawn as
    a wide, translucent bar; the smaller as a narrow, solid bar on top of
    it, both centered at `pos`. Each platform keeps its own hatch
    (PLATFORM_HATCH) regardless of which one ends up large/small, so hatch
    still means "RISC-V" consistently across the figure - only alpha/width
    encode which value is bigger. err_rpi/err_riscv, if given, are drawn as
    an error bar (standard deviation) on top of each platform's bar."""
    vals = {"RPi": val_rpi, "RISC-V": val_riscv}
    errs = {"RPi": err_rpi, "RISC-V": err_riscv}
    present = {p: v for p, v in vals.items() if v is not None and not np.isnan(v)}
    if not present:
        return
    order = sorted(present, key=lambda p: present[p], reverse=True)
    for i, platform in enumerate(order):
        v = present[platform]
        e = errs[platform]
        if e is not None and np.isnan(e):
            e = None
        w = size if i == 0 else size * 0.5
        kwargs = dict(
            color=color, alpha=(0.35 if i == 0 else 1.0),
            hatch=PLATFORM_HATCH[platform],
            edgecolor=("none" if i == 0 else "white"),
            linewidth=(0 if i == 0 else 0.5),
            zorder=2 + i,
        )
        err_kwargs = dict(ecolor="#333333", capsize=2, error_kw=dict(elinewidth=0.8)) if e is not None else {}
        if horizontal:
            ax.barh(pos, v, height=w, xerr=e, **err_kwargs, **kwargs)
        else:
            ax.bar(pos, v, width=w, yerr=e, **err_kwargs, **kwargs)


def boldify(fig):
    """Force every text element to bold - rcParams font.weight covers text
    created after setup_style() but not manually-placed ax.text() bar
    annotations, so those need to be swept explicitly too."""
    for ax in fig.get_axes():
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
        for t in texts:
            t.set_fontweight("bold")
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_title() is not None:
                leg.get_title().set_fontweight("bold")
            for t in leg.get_texts():
                t.set_fontweight("bold")
    for t in fig.texts:
        t.set_fontweight("bold")
    for leg in getattr(fig, "legends", []):
        for t in leg.get_texts():
            t.set_fontweight("bold")


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    boldify(fig)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")

    png_path = path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved {png_path}")
