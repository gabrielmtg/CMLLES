import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_latencies, setup_style, save,
    COLORS, PLATFORM_ORDER, PLATFORM_LABEL, INT8_FRAMEWORKS_BY_PLATFORM,
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


def _mean_lat(df, algo, fw, prec):
    sub = df[(df["algorithm"] == algo) & (df["framework"] == fw) & (df["precision"] == prec)]
    if sub.empty:
        return np.nan
    return sub["latency_us"].mean()


def plot_int8_speedup(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)

    fw_seen = []
    for ax, platform in zip(axes, PLATFORM_ORDER):
        sub_platform = df[df["platform"] == platform]
        # CMSIS-NN is INT8-only in this benchmark, so it has no F32
        # baseline and is excluded from the speedup ratio.
        fw_list = [f for f in INT8_FRAMEWORKS_BY_PLATFORM[platform]
                   if f != "CMSIS-NN" and f in sub_platform["framework"].values]
        present_algos = [a for a in ALGO_ORDER if a in sub_platform["algorithm"].values]

        ax.set_title(PLATFORM_LABEL[platform])
        ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Algorithm")

        if fw_list and present_algos:
            n_algos = len(present_algos)
            n_fw    = len(fw_list)
            width   = 0.8 / n_fw
            x_pos   = np.arange(n_algos)

            for i, fw in enumerate(fw_list):
                fw_seen.append(fw) if fw not in fw_seen else None
                speedups = []
                for algo in present_algos:
                    m_f32  = _mean_lat(sub_platform, algo, fw, "f32")
                    m_int8 = _mean_lat(sub_platform, algo, fw, "int8")
                    speedups.append(m_f32 / m_int8 if (not np.isnan(m_f32) and not np.isnan(m_int8) and m_int8 > 0) else np.nan)
                offset = (i - n_fw / 2 + 0.5) * width
                ax.bar(x_pos + offset, speedups, width, color=COLORS[fw], edgecolor="white", linewidth=0.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])

    axes[0].set_ylabel("Speedup (F32 latency / INT8 latency)")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("INT8 speedup over F32 per framework and platform")
    fig.tight_layout()
    save(fig, "int8_speedup.pdf")
    plt.close(fig)


def main():
    df = load_latencies()
    if df.empty:
        print("No latency CSV found in results/latencies_rpi/ or results/latencies_riscv/")
        return
    plot_int8_speedup(df)


if __name__ == "__main__":
    main()
