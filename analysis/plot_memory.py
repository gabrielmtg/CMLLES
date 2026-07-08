import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_binary, setup_style, save,
    FRAMEWORK_ORDER, COLORS, HATCHES,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP (Iris)",
    "MLP-IAES":   "MLP (IAES)",
    "Autoencoder": "Autoencoder",
    "CNN2D":       "CNN-2D",
    "LSTM":        "LSTM",
}


def plot_total_size(df):
    rows = []
    for algo in ALGO_ORDER:
        for fw in FRAMEWORK_ORDER:
            sub = df[(df["algorithm"] == algo) & (df["framework"] == fw)]
            if sub.empty:
                continue
            rows.append({
                "algorithm": algo,
                "framework": fw,
                "text_kb":  sub["text_bytes"].mean() / 1024,
                "bss_kb":   sub["bss_bytes"].mean()  / 1024,
                "total_kb": sub["total_bytes"].mean() / 1024,
            })
    data = pd.DataFrame(rows)
    if data.empty:
        return

    present = [a for a in ALGO_ORDER if a in data["algorithm"].values]
    n_algos = len(present)
    n_fw    = len(FRAMEWORK_ORDER)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, fw in enumerate(FRAMEWORK_ORDER):
        sub = data[data["framework"] == fw]
        text_vals = [
            sub[sub["algorithm"] == a]["text_kb"].values[0]
            if a in sub["algorithm"].values else np.nan
            for a in present
        ]
        bss_vals = [
            sub[sub["algorithm"] == a]["bss_kb"].values[0]
            if a in sub["algorithm"].values else np.nan
            for a in present
        ]
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, text_vals, width,
               color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5, label=fw)
        ax.bar(x_pos + offset, bss_vals, width,
               bottom=text_vals,
               color=COLORS[fw], alpha=0.4, hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present], rotation=15, ha="right")
    ax.set_ylabel("Size (KB)")
    ax.set_title("Static memory footprint of executables\n(solid = .text, transparent = .bss)")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in data["framework"].values]
    ax.legend(handles=handles, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "memory_footprint.pdf")
    plt.close(fig)


def plot_text_only(df):
    rows = []
    for algo in ALGO_ORDER:
        for fw in FRAMEWORK_ORDER:
            sub = df[(df["algorithm"] == algo) & (df["framework"] == fw)]
            if sub.empty:
                continue
            rows.append({
                "algorithm": algo,
                "framework": fw,
                "text_kb":  sub["text_bytes"].mean() / 1024,
            })
    data = pd.DataFrame(rows)
    if data.empty:
        return

    present = [a for a in ALGO_ORDER if a in data["algorithm"].values]
    n_algos = len(present)
    n_fw    = len(FRAMEWORK_ORDER)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, fw in enumerate(FRAMEWORK_ORDER):
        sub  = data[data["framework"] == fw]
        vals = [
            sub[sub["algorithm"] == a]["text_kb"].values[0]
            if a in sub["algorithm"].values else np.nan
            for a in present
        ]
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width, label=fw,
               color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present], rotation=15, ha="right")
    ax.set_ylabel("Code size (.text, KB)")
    ax.set_title("Compiled code size per framework and algorithm")
    ax.set_yscale("log")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in data["framework"].values]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "memory_text_size.pdf")
    plt.close(fig)


def main():
    df = load_binary()
    if df.empty:
        print("binary_metrics.csv not found or empty")
        return

    plot_total_size(df)
    plot_text_only(df)


if __name__ == "__main__":
    main()
