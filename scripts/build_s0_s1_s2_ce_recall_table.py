#!/usr/bin/env python3
"""Build S0/S1/S2 per-disease CheXpert CE-R table and heatmap."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_ORDER = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

SHORT_LABELS = [
    "Atelect.",
    "Cardiom.",
    "Consol.",
    "Edema",
    "Enl. Cardiom.",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Eff.",
    "Pleural Other",
    "Pneumonia",
    "Pneumoth.",
    "Support Dev.",
]

EXPERIMENTS = ["S0", "S1", "S2"]


def build_table(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[(df["experiment"].isin(EXPERIMENTS)) & (df["label_type"] == "disease")].copy()
    if df.empty:
        raise ValueError(f"No S0/S1/S2 disease rows found in {input_csv}")

    pivot = df.pivot(index="label", columns="experiment", values="Recall")
    missing_labels = [label for label in LABEL_ORDER if label not in pivot.index]
    missing_exps = [exp for exp in EXPERIMENTS if exp not in pivot.columns]
    if missing_labels or missing_exps:
        raise ValueError(f"Missing labels={missing_labels}, missing experiments={missing_exps}")

    pivot = pivot.loc[LABEL_ORDER, EXPERIMENTS]
    out = pd.DataFrame(
        {
            "Disease": LABEL_ORDER,
            "S0 CE-R": pivot["S0"].to_numpy(),
            "S1 CE-R": pivot["S1"].to_numpy(),
            "S2 CE-R": pivot["S2"].to_numpy(),
        }
    )
    out["delta_S2_minus_S1"] = out["S2 CE-R"] - out["S1 CE-R"]
    return out


def write_markdown(table: pd.DataFrame, output_md: Path) -> None:
    headers = ["Disease", "S0 CE-R", "S1 CE-R", "S2 CE-R", "delta_S2_minus_S1"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Disease"]),
                    f"{row['S0 CE-R']:.4f}",
                    f"{row['S1 CE-R']:.4f}",
                    f"{row['S2 CE-R']:.4f}",
                    f"{row['delta_S2_minus_S1']:+.4f}",
                ]
            )
            + " |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmap(table: pd.DataFrame, output_prefix: Path) -> None:
    values = table[["S0 CE-R", "S1 CE-R", "S2 CE-R", "delta_S2_minus_S1"]].to_numpy(dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 6.8),
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.12},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    recall_values = values[:, :3]
    delta_values = values[:, 3:]

    im0 = axes[0].imshow(recall_values, cmap="YlGnBu", vmin=0.0, vmax=0.70, aspect="auto")
    axes[0].set_title("Per-disease CE-R", fontsize=13, pad=10)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(["S0", "S1", "S2"], fontsize=11)
    axes[0].set_yticks(np.arange(len(SHORT_LABELS)))
    axes[0].set_yticklabels(SHORT_LABELS, fontsize=9.5)
    axes[0].tick_params(length=0)

    for i in range(recall_values.shape[0]):
        for j in range(recall_values.shape[1]):
            text_color = "white" if recall_values[i, j] >= 0.45 else "black"
            axes[0].text(j, i, f"{recall_values[i, j]:.3f}", ha="center", va="center", fontsize=8.5, color=text_color)

    vmax_delta = max(abs(float(delta_values.min())), abs(float(delta_values.max())), 0.001)
    im1 = axes[1].imshow(delta_values, cmap="RdBu_r", vmin=-vmax_delta, vmax=vmax_delta, aspect="auto")
    axes[1].set_title("S2-S1", fontsize=13, pad=10)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["Delta"], fontsize=11)
    axes[1].set_yticks(np.arange(len(SHORT_LABELS)))
    axes[1].set_yticklabels([])
    axes[1].tick_params(length=0)

    for i in range(delta_values.shape[0]):
        val = delta_values[i, 0]
        axes[1].text(0, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.5, color="black")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, ax.get_xlim()[1] + 0.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SHORT_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    cbar0.set_label("CE-R", fontsize=10)
    cbar0.ax.tick_params(labelsize=9)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.14, pad=0.05)
    cbar1.set_label("Delta", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)

    fig.suptitle("CheXpert per-disease recall on MIMIC-CXR", fontsize=15, y=1.02)

    fig.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    input_csv = Path("paper/per_disease_chexpert.csv")
    output_csv = Path("paper/s0_s1_s2_per_disease_ce_recall.csv")
    output_md = Path("paper/s0_s1_s2_per_disease_ce_recall.md")
    heatmap_prefix = Path("paper/s0_s1_s2_per_disease_ce_recall_heatmap")

    table = build_table(input_csv)
    table.to_csv(output_csv, index=False)
    write_markdown(table, output_md)
    plot_heatmap(table, heatmap_prefix)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    print(f"Wrote {heatmap_prefix}.pdf")
    print(f"Wrote {heatmap_prefix}.svg")
    print(f"Wrote {heatmap_prefix}.png")


if __name__ == "__main__":
    main()
