import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_latencies, setup_style, save,
    FRAMEWORK_ORDER, COLORS, HATCHES, PLATFORM_ORDER, PLATFORM_LABEL,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP\n(Iris)",
    "MLP-IAES":   "MLP\n(IAES)",
    "Autoencoder": "AE",
    "CNN2D":       "CNN",
    "LSTM":        "LSTM",
}


def _platform_means(df, platform):
    rows = []
    sub_platform = df[df["platform"] == platform]
    for algo in ALGO_ORDER:
        sub = sub_platform[sub_platform["algorithm"] == algo]
        if sub.empty:
            continue
        for fw in FRAMEWORK_ORDER:
            fwsub = sub[sub["framework"] == fw]
            if fwsub.empty:
                continue
            rows.append({"algorithm": algo, "framework": fw, "mean": fwsub["latency_us"].mean()})
    return pd.DataFrame(rows)


def plot_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)

    fw_present = set()
    for ax, platform in zip(axes, PLATFORM_ORDER):
        ov = _platform_means(df, platform)
        ax.set_title(PLATFORM_LABEL[platform])
        ax.set_yscale("log")
        ax.set_xlabel("Algorithm")
        if ov.empty:
            continue

        present_algos = [a for a in ALGO_ORDER if a in ov["algorithm"].values]
        n_algos = len(present_algos)
        n_fw    = len(FRAMEWORK_ORDER)
        width   = 0.8 / n_fw
        x_pos   = np.arange(n_algos)

        for i, fw in enumerate(FRAMEWORK_ORDER):
            sub = ov[ov["framework"] == fw]
            if sub.empty:
                continue
            fw_present.add(fw)
            means = [sub[sub["algorithm"] == a]["mean"].values[0]
                     if a in sub["algorithm"].values else np.nan
                     for a in present_algos]
            offset = (i - n_fw / 2 + 0.5) * width
            ax.bar(x_pos + offset, means, width, color=COLORS[fw], hatch=HATCHES[fw],
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])

    axes[0].set_ylabel("Average inference latency (µs, log scale)")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in fw_present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Inference latency comparison across platforms and algorithms")
    fig.tight_layout()
    save(fig, "latency_overview.pdf")
    plt.close(fig)


def main():
    df = load_latencies()
    if df.empty:
        print("No latency CSV found in results/latencies_rpi/ or results/latencies_riscv/")
        return
    plot_overview(df)


if __name__ == "__main__":
    main()
