import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_binary, setup_style, save, FRAMEWORK_ORDER, COLORS

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP (Iris)",
    "MLP-IAES":   "MLP (IAES)",
    "Autoencoder": "Autoencoder",
    "CNN2D":       "CNN-2D",
    "LSTM":        "LSTM",
}

HEADER_COLOR = "#1A2F55"


def _render_table(out, name):
    if out.empty:
        return
    n_rows, n_cols = out.shape
    fig_w = max(6.0, 1.35 * n_cols)
    fig_h = 0.34 * (n_rows + 1) + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=out.values, colLabels=list(out.columns),
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    tbl.auto_set_column_width(col=list(range(n_cols)))

    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F5F5F5" if row % 2 == 0 else "white")

    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


def bar_footprint_summary(df):
    """Compact one-bar-per-framework overview: total static footprint (mean
    across algorithms, since .data/.bss barely vary by algorithm within a
    framework - see memory_footprint_detail for the per-algorithm numbers),
    log-scale because NCNN sits ~1-2 orders of magnitude above the rest."""
    sub = df[df["algorithm"].isin(ALGO_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)]
    if sub.empty:
        return

    agg = sub.groupby("framework", as_index=False)[
        ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]
    ].mean()
    agg["framework"] = pd.Categorical(agg["framework"], categories=FRAMEWORK_ORDER, ordered=True)
    agg = agg.sort_values("total_bytes")
    for col in ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]:
        agg[col] = agg[col] / 1024

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(agg) + 0.8))
    y_pos = np.arange(len(agg))

    ax.barh(y_pos, agg["total_bytes"], height=0.6,
            color=[COLORS[f] for f in agg["framework"]], edgecolor="white")
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg["framework"])
    ax.set_xlabel("Static memory footprint, mean across algorithms (KB, log scale)")
    ax.set_title("Executable memory footprint by framework")

    for i, row in enumerate(agg.itertuples()):
        label = (f"{row.total_bytes:,.0f} KB  "
                  f"(.text {row.text_bytes:,.0f} / .data {row.data_bytes:,.0f} / .bss {row.bss_bytes:,.0f})")
        ax.text(row.total_bytes * 1.08, i, label, va="center", fontsize=7.5, color=HEADER_COLOR)

    ax.set_xlim(right=agg["total_bytes"].max() * 12)
    fig.tight_layout()
    save(fig, "memory_footprint.pdf")
    plt.close(fig)


def table_footprint_detail(df):
    """Per-algorithm breakdown, kept as supplementary detail behind the
    compact bar_footprint_summary chart above."""
    sub = df[df["algorithm"].isin(ALGO_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)]
    if sub.empty:
        return

    agg = sub.groupby(["algorithm", "framework"], as_index=False)[
        ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]
    ].mean()

    agg["algorithm"] = pd.Categorical(agg["algorithm"], categories=ALGO_ORDER, ordered=True)
    agg["framework"] = pd.Categorical(agg["framework"], categories=FRAMEWORK_ORDER, ordered=True)
    agg = agg.sort_values(["algorithm", "framework"])
    agg["algorithm"] = agg["algorithm"].map(ALGO_LABEL)

    for col in ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]:
        agg[col] = (agg[col] / 1024).map("{:.1f}".format)

    out = agg.rename(columns={
        "algorithm":   "Algorithm",
        "framework":   "Framework",
        "text_bytes":  ".text (KB)",
        "data_bytes":  ".data (KB)",
        "bss_bytes":   ".bss (KB)",
        "total_bytes": "Total (KB)",
    })
    _render_table(out, "memory_footprint_detail.pdf")


def table_text_size(df):
    sub = df[df["algorithm"].isin(ALGO_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)].copy()
    if sub.empty:
        return
    sub["text_kb"] = sub["text_bytes"] / 1024

    pivot = sub.pivot_table(index="framework", columns="algorithm", values="text_kb", aggfunc="mean")
    pivot = pivot.reindex(index=FRAMEWORK_ORDER, columns=ALGO_ORDER).dropna(how="all")
    pivot = pivot.rename(columns=ALGO_LABEL).map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
    pivot.columns.name = None
    pivot.index.name = "Framework"

    out = pivot.reset_index()
    _render_table(out, "memory_text_size.pdf")


def main():
    df = load_binary()
    if df.empty:
        print("binary_metrics.csv not found or empty")
        return

    bar_footprint_summary(df)
    table_footprint_detail(df)
    table_text_size(df)


if __name__ == "__main__":
    main()
