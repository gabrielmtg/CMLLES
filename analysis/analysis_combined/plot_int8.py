import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_latencies, setup_style, save, overlay_bar, platform_legend_handles,
    COLORS, FRAMEWORK_ORDER, PLATFORM_ORDER, INT8_FRAMEWORKS_BY_PLATFORM,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP (Iris)",
    "MLP-IAES":   "MLP (IAES)",
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)

    fw_seen = []
    for ax, platform in zip(axes, PLATFORM_ORDER):
        sub_platform = df[df["platform"] == platform]
        # CMSIS-NN is INT8-only in this benchmark, so it has no F32
        # baseline and is excluded from the speedup ratio.
        fw_list = [f for f in INT8_FRAMEWORKS_BY_PLATFORM[platform]
                   if f != "CMSIS-NN" and f in sub_platform["framework"].values]
        present_algos = [a for a in ALGO_ORDER if a in sub_platform["algorithm"].values]

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
    fig.tight_layout()
    save(fig, "int8_speedup.pdf")
    plt.close(fig)


def plot_int8_speedup_overlay(df):
    """Same content as plot_int8_speedup, but instead of separate
    per-platform subplots, both platforms are overlaid at the same x
    position (see utils.overlay_bar). Only frameworks common to both
    platforms (LiteRT, ONNX) have a value on both sides to overlay."""
    fw_list = [f for f in INT8_FRAMEWORKS_BY_PLATFORM["RPi"]
               if f != "CMSIS-NN" and f in INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]]
    present_algos = [a for a in ALGO_ORDER if a in df["algorithm"].values]
    if not fw_list or not present_algos:
        return

    n_algos = len(present_algos)
    n_fw    = len(fw_list)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")

    for i, fw in enumerate(fw_list):
        offset = (i - n_fw / 2 + 0.5) * width
        for xi, algo in enumerate(present_algos):
            speedups = {}
            for platform in PLATFORM_ORDER:
                sub = df[(df["platform"] == platform) & (df["framework"] == fw) & (df["algorithm"] == algo)]
                m_f32  = sub[sub["precision"] == "f32"]["latency_us"].mean()
                m_int8 = sub[sub["precision"] == "int8"]["latency_us"].mean()
                speedups[platform] = m_f32 / m_int8 if (not np.isnan(m_f32) and not np.isnan(m_int8) and m_int8 > 0) else np.nan
            overlay_bar(ax, x_pos[xi] + offset, speedups["RPi"], speedups["RISC-V"], COLORS[fw], size=width * 0.95)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    ax.set_xlabel("Algorithm")
    
    ax.set_ylim(bottom=0, top=3)
    fig.supylabel("Speedup (F32 / INT8)", fontsize=10)

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list]
    leg1 = ax.legend(handles=fw_handles, loc="upper left", fontsize=15, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=platform_legend_handles(), loc="upper right", fontsize=15, framealpha=0.9)

    fig.tight_layout()
    save(fig, "int8_speedup_overlay.pdf")
    plt.close(fig)


def plot_latency_speedup_overlay(df):
    present_algos = [a for a in ALGO_ORDER if a in df["algorithm"].values]
    if not present_algos:
        return
    x_pos = np.arange(len(present_algos))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))

    n_fw = len(FRAMEWORK_ORDER)
    width = 0.92 / n_fw
    fw_present = set()
    for i, fw in enumerate(FRAMEWORK_ORDER):
        offset = (i - n_fw / 2 + 0.5) * width
        for xi, algo in enumerate(present_algos):
            sub = df[(df["algorithm"] == algo) & (df["framework"] == fw)]
            if sub.empty:
                continue
            fw_present.add(fw)
            rpi_sub = sub[sub["platform"] == "RPi"]["latency_us"]
            riscv_sub = sub[sub["platform"] == "RISC-V"]["latency_us"]
            overlay_bar(axes[0], x_pos[xi] + offset, rpi_sub.mean(), riscv_sub.mean(), COLORS[fw],
                        size=width * 0.95, err_rpi=rpi_sub.std(), err_riscv=riscv_sub.std())

    axes[0].set_yscale("log")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    axes[0].set_xlabel("Algorithm")
    axes[0].set_ylabel("Avg. inference latency (µs, log scale)")
    axes[0].set_title("(a)", loc="left", fontsize=14)
    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in FRAMEWORK_ORDER if f in fw_present]
    axes[0].legend(handles=fw_handles, loc="upper left", ncol=3, fontsize=8, framealpha=0.9)

    fw_list = [f for f in INT8_FRAMEWORKS_BY_PLATFORM["RPi"]
               if f != "CMSIS-NN" and f in INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]]
    n_fw2 = len(fw_list)
    width2 = 0.8 / n_fw2
    axes[1].axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    for i, fw in enumerate(fw_list):
        offset = (i - n_fw2 / 2 + 0.5) * width2
        for xi, algo in enumerate(present_algos):
            speedups = {}
            for platform in PLATFORM_ORDER:
                sub = df[(df["platform"] == platform) & (df["framework"] == fw) & (df["algorithm"] == algo)]
                m_f32  = sub[sub["precision"] == "f32"]["latency_us"].mean()
                m_int8 = sub[sub["precision"] == "int8"]["latency_us"].mean()
                speedups[platform] = m_f32 / m_int8 if (not np.isnan(m_f32) and not np.isnan(m_int8) and m_int8 > 0) else np.nan
            overlay_bar(axes[1], x_pos[xi] + offset, speedups["RPi"], speedups["RISC-V"], COLORS[fw], size=width2 * 0.95)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    axes[1].set_xlabel("Algorithm")
    axes[1].set_ylabel("Speedup (F32 / INT8)")
    axes[1].set_title("(b)", loc="left", fontsize=14)
    fw_handles2 = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list]
    leg1 = axes[1].legend(handles=fw_handles2, loc="upper left", fontsize=8, framealpha=0.9)
    axes[1].add_artist(leg1)
    axes[1].legend(handles=platform_legend_handles(), loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save(fig, "latency_speedup_overlay.pdf")
    plt.close(fig)


def main():
    df = load_latencies()
    if df.empty:
        print("No latency CSV found in results/latencies_rpi/ or results/latencies_riscv/")
        return
    plot_int8_speedup(df)
    plot_int8_speedup_overlay(df)
    plot_latency_speedup_overlay(df)


if __name__ == "__main__":
    main()
