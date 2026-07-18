#!/usr/bin/env python3
"""Build per-disease CheXpert clinical efficacy tables for paper supplements."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metric.clinical import _configure_f1chexbert_runtime

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

EXPERIMENTS = [
    {
        "name": "Baseline",
        "row_id": "B_potential2_0424",
        "report_csv": "experiments/mimic_ablation_results/runs/missing_clinical_sweep_20260608_160130/baseline_potential2_full_eval/mimic_cxr_reports.csv",
    },
    {"name": "S0", "row_id": "S0"},
    {"name": "S1", "row_id": "S1"},
    {"name": "S2", "row_id": "S2"},
    {"name": "S2+CCD g0.05", "row_id": "S2_CCD_g0p05"},
    {"name": "S2+CCD g0.10", "row_id": "S2_CCD_g0p10"},
    {"name": "S2+CCD g0.15", "row_id": "S2_CCD_g0p15"},
    {"name": "S2+CCD g0.20", "row_id": "S2_CCD_g0p20"},
    {"name": "S2+CCD g0.25", "row_id": "S2_CCD_val_selected_g0p25"},
    {"name": "S2+CCD g0.30", "row_id": "S2_CCD_g0p30"},
]


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_summary(path):
    rows = read_csv_rows(path)
    return {row["row_id"]: row for row in rows}


def find_checkpoint(run_dir):
    run_dir = Path(run_dir)
    best_test = run_dir / "model_best.pth"
    best_val = run_dir / "model_best_val.pth"
    if best_test.exists():
        return best_test
    if best_val.exists():
        return best_val
    raise FileNotFoundError(f"No model_best.pth/model_best_val.pth in {run_dir}")


def dump_reports(row, row_id, out_dir, config_path, cuda, seed, batch_size, beam_size, max_seq_length, force):
    report_csv = out_dir / row_id / "mimic_cxr_reports.csv"
    if report_csv.exists() and report_csv.stat().st_size > 0 and not force:
        return report_csv

    out_dir = out_dir / row_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint(row["run_dir"])
    args = [
        sys.executable,
        "eval_dump_reports.py",
        "--c",
        config_path,
        "--model",
        row.get("model_path") or "vlci",
        "--task_name",
        row_id,
        "--result_dir",
        str(out_dir),
        "--load_model_path",
        str(checkpoint),
        "--cuda",
        str(cuda),
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--beam_size",
        str(beam_size),
        "--sample_method",
        "beam_search",
        "--max_seq_length",
        str(max_seq_length),
        "--suppress_UNK",
        "1",
        "--split",
        "test",
        "--use_anatomical_adapter",
        str(int(float(row.get("A") or 0))),
        "--use_memory_prompt",
        str(int(float(row.get("B") or 0))),
        "--use_counterfactual_intervention",
        str(int(float(row.get("C") or 0))),
    ]
    print("Dumping reports:", " ".join(args), flush=True)
    subprocess.run(args, check=True)
    if not report_csv.exists():
        raise FileNotFoundError(f"Expected report CSV was not created: {report_csv}")
    return report_csv


def resolve_report_csv(exp, summary_by_id, report_work_dir, args):
    explicit = exp.get("report_csv")
    if explicit and Path(explicit).exists():
        return Path(explicit), "explicit_report_csv"

    row_id = exp["row_id"]
    row = summary_by_id.get(row_id)
    if row is None:
        raise KeyError(f"Missing row_id in summary: {row_id}")

    run_dir = Path(row.get("run_dir", ""))
    candidate = run_dir / "mimic_cxr_reports.csv"
    if candidate.exists() and candidate.stat().st_size > 0 and not args.force_dump:
        return candidate, "run_dir_report_csv"

    if row_id.startswith("S2_CCD"):
        raise FileNotFoundError(
            f"CCD row {row_id} has no report CSV at {candidate}. "
            "CCD reports should already exist because they depend on decoding settings."
        )

    report_csv = dump_reports(
        row=row,
        row_id=row_id,
        out_dir=report_work_dir,
        config_path=args.config,
        cuda=args.cuda,
        seed=args.seed,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        max_seq_length=args.max_seq_length,
        force=args.force_dump,
    )
    return report_csv, "dumped_from_main_row_checkpoint"


def load_report_pairs(path):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    gt_col = next((c for c in df.columns if c in ["reference_report", "truth", "gt"]), None)
    pred_col = next((c for c in df.columns if c in ["generated_report", "pred", "res", "gen"]), None)
    if not gt_col or not pred_col:
        raise ValueError(f"Missing report columns in {path}: {df.columns.tolist()}")
    refs = df[gt_col].fillna("").astype(str).tolist()
    hyps = df[pred_col].fillna("").astype(str).tolist()
    if len(refs) != len(hyps):
        raise ValueError(f"Mismatched refs/hyps length in {path}")
    if not refs:
        raise ValueError(f"No test report pairs in {path}")
    return refs, hyps


def label_reports(evaluator, reports):
    labels = []
    for idx, report in enumerate(reports, 1):
        labels.append(evaluator.get_label(report.strip(), mode="rrg"))
        if idx % 250 == 0 or idx == len(reports):
            print(f"  labeled {idx}/{len(reports)} reports", flush=True)
    arr = np.asarray(labels, dtype=np.int64)
    if arr.shape[1] != len(CHEXPERT_LABELS):
        raise ValueError(f"Expected {len(CHEXPERT_LABELS)} labels, got {arr.shape[1]}")
    return arr


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def per_label_metrics(refs_binary, hyps_binary):
    n_samples = int(refs_binary.shape[0])
    rows = []
    for label_idx, label in enumerate(CHEXPERT_LABELS):
        y_true = refs_binary[:, label_idx]
        y_pred = hyps_binary[:, label_idx]
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision, recall, f1 = prf(tp, fp, fn)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
        rows.append(
            {
                "label": label,
                "label_type": "disease",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "support": int(y_true.sum()),
            }
        )

    disease_rows = rows[:]
    rows.append(
        {
            "label": "macro_avg",
            "label_type": "macro_avg",
            "TP": "",
            "FP": "",
            "FN": "",
            "Precision": float(np.mean([r["Precision"] for r in disease_rows])),
            "TN": "",
            "Accuracy": float(np.mean([r["Accuracy"] for r in disease_rows])),
            "Recall": float(np.mean([r["Recall"] for r in disease_rows])),
            "F1": float(np.mean([r["F1"] for r in disease_rows])),
            "support": int(sum(r["support"] for r in disease_rows)),
        }
    )

    tp_sum = int(sum(r["TP"] for r in disease_rows))
    fp_sum = int(sum(r["FP"] for r in disease_rows))
    fn_sum = int(sum(r["FN"] for r in disease_rows))
    precision, recall, f1 = prf(tp_sum, fp_sum, fn_sum)
    rows.append(
        {
            "label": "micro_avg",
            "label_type": "micro_avg",
            "TP": tp_sum,
            "FP": fp_sum,
            "FN": fn_sum,
            "Precision": precision,
            "TN": int(sum(r["TN"] for r in disease_rows)),
            "Accuracy": float((tp_sum + int(sum(r["TN"] for r in disease_rows))) / (n_samples * len(CHEXPERT_LABELS))) if n_samples > 0 else 0.0,
            "Recall": recall,
            "F1": f1,
            "support": int(sum(r["support"] for r in disease_rows)),
        }
    )
    return rows


def validate_micro_against_class_report(refs_binary, hyps_binary, metric_rows, experiment):
    if classification_report is None:
        raise RuntimeError("scikit-learn is required to validate micro averages against F1CheXbert classification_report.")
    report = classification_report(
        refs_binary,
        hyps_binary,
        target_names=CHEXPERT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    if "micro avg" not in report:
        raise RuntimeError(f"classification_report missing micro avg for {experiment}")
    micro = next(row for row in metric_rows if row["label"] == "micro_avg")
    for ours_key, report_key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
        ours = float(micro[ours_key])
        expected = float(report["micro avg"][report_key])
        if abs(ours - expected) > 1e-8:
            raise RuntimeError(
                f"Micro {ours_key} mismatch for {experiment}: manual={ours}, classification_report={expected}"
            )


def build_delta(per_disease_df):
    s1 = per_disease_df[
        (per_disease_df["experiment"] == "S1") & (per_disease_df["label_type"] == "disease")
    ].set_index("label")
    s2 = per_disease_df[
        (per_disease_df["experiment"] == "S2") & (per_disease_df["label_type"] == "disease")
    ].set_index("label")
    rows = []
    for label in CHEXPERT_LABELS:
        rows.append(
            {
                "label": label,
                "S1_Recall": s1.loc[label, "Recall"],
                "S2_Recall": s2.loc[label, "Recall"],
                "delta_recall": s2.loc[label, "Recall"] - s1.loc[label, "Recall"],
                "S1_F1": s1.loc[label, "F1"],
                "S2_F1": s2.loc[label, "F1"],
                "delta_f1": s2.loc[label, "F1"] - s1.loc[label, "F1"],
                "S1_support": int(s1.loc[label, "support"]),
                "S2_support": int(s2.loc[label, "support"]),
            }
        )
    delta = pd.DataFrame(rows)
    delta["recall_change_rank"] = delta["delta_recall"].rank(method="min", ascending=False).astype(int)
    delta["f1_change_rank"] = delta["delta_f1"].rank(method="min", ascending=False).astype(int)
    return delta


def plot_heatmap(per_disease_df, output_pdf, output_svg):
    disease = per_disease_df[per_disease_df["label_type"] == "disease"].copy()
    exp_order = [exp["name"] for exp in EXPERIMENTS]
    recall = disease.pivot(index="label", columns="experiment", values="Recall").loc[CHEXPERT_LABELS, exp_order]
    f1 = disease.pivot(index="label", columns="experiment", values="F1").loc[CHEXPERT_LABELS, exp_order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, data, title in zip(axes, [recall, f1], ["Recall", "F1"]):
        im = ax.imshow(data.values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(exp_order)))
        ax.set_xticklabels(exp_order, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(CHEXPERT_LABELS)))
        ax.set_yticklabels(CHEXPERT_LABELS)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-disease CheXpert clinical efficacy on MIMIC-CXR test set", fontsize=14)
    fig.savefig(output_pdf)
    fig.savefig(output_svg)
    plt.close(fig)


def format_metric(value):
    return f"{float(value):.4f}"


def write_summary_md(path, per_disease_df, delta_df, sources):
    avg = per_disease_df[per_disease_df["label"].isin(["macro_avg", "micro_avg"])].copy()
    best_recall = delta_df.sort_values("delta_recall", ascending=False).iloc[0]
    worst_recall = delta_df.sort_values("delta_recall", ascending=True).iloc[0]
    best_f1 = delta_df.sort_values("delta_f1", ascending=False).iloc[0]
    worst_f1 = delta_df.sort_values("delta_f1", ascending=True).iloc[0]

    lines = [
        "# Per-disease CheXpert Clinical Efficacy",
        "",
        "This supplement uses the same F1CheXbert labeler and `mode=\"rrg\"` label mapping as the main CE_P/CE_R/CE_F1 computation. No bootstrap or paired test was performed, so changes are described as observed differences only.",
        "",
        "## Inputs",
        "",
        "| experiment | source row | report csv | source type |",
        "| --- | --- | --- | --- |",
    ]
    for source in sources:
        lines.append(
            f"| {source['experiment']} | {source['row_id']} | `{source['report_csv']}` | {source['source_type']} |"
        )

    lines.extend(
        [
            "",
            "## Macro and Micro Averages",
            "",
            "| experiment | average | Precision | Recall | F1 | support |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in avg.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['label']} | {format_metric(row['Precision'])} | {format_metric(row['Recall'])} | {format_metric(row['F1'])} | {int(row['support'])} |"
        )

    lines.extend(
        [
            "",
            "## S2 vs S1 Observed Changes",
            "",
            f"- Largest Recall increase: {best_recall['label']} ({best_recall['delta_recall']:+.4f})",
            f"- Largest Recall decrease: {worst_recall['label']} ({worst_recall['delta_recall']:+.4f})",
            f"- Largest F1 increase: {best_f1['label']} ({best_f1['delta_f1']:+.4f})",
            f"- Largest F1 decrease: {worst_f1['label']} ({worst_f1['delta_f1']:+.4f})",
            "",
            "## Output Files",
            "",
            "- `paper/per_disease_chexpert.csv`",
            "- `paper/per_disease_delta_s2_vs_s1.csv`",
            "- `paper/per_disease_recall_f1.pdf`",
            "- `paper/per_disease_recall_f1.svg`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(output_dir, per_disease_df, delta_df):
    expected_rows = len(EXPERIMENTS) * (len(CHEXPERT_LABELS) + 2)
    if len(per_disease_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in per-disease table, got {len(per_disease_df)}")
    if len(delta_df) != len(CHEXPERT_LABELS):
        raise ValueError(f"Expected {len(CHEXPERT_LABELS)} rows in delta table, got {len(delta_df)}")
    for name in [
        "per_disease_chexpert.csv",
        "per_disease_delta_s2_vs_s1.csv",
        "per_disease_recall_f1.pdf",
        "per_disease_recall_f1.svg",
        "per_disease_summary.md",
    ]:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty output: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="experiments/mimic_ablation_results/summaries/experiment_summary_20260527/all_ablation_summary.csv")
    parser.add_argument("--output-dir", default="paper")
    parser.add_argument("--work-dir", default="paper/per_disease_work")
    parser.add_argument("--config", default="config/mimic_cxr/vlci.json")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=80)
    parser.add_argument("--force-dump", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    report_work_dir = work_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_work_dir.mkdir(parents=True, exist_ok=True)

    _configure_f1chexbert_runtime()
    try:
        from f1chexbert import F1CheXbert
    except ImportError as exc:
        raise SystemExit(
            "f1chexbert is not installed in the active Python environment. "
            "Please run this script in the same conda environment used for eval_clinical.py."
        ) from exc

    summary_by_id = load_summary(args.summary_csv)
    evaluator = F1CheXbert(device=f"cuda:{args.cuda}" if args.cuda >= 0 else "cpu")
    if list(evaluator.target_names) != CHEXPERT_LABELS:
        raise ValueError(f"Unexpected F1CheXbert target label order: {evaluator.target_names}")

    all_metric_rows = []
    sources = []
    for exp in EXPERIMENTS:
        report_csv, source_type = resolve_report_csv(exp, summary_by_id, report_work_dir, args)
        print(f"Evaluating {exp['name']} from {report_csv}", flush=True)
        refs, hyps = load_report_pairs(report_csv)
        print(f"  labeling references for {exp['name']}", flush=True)
        refs_binary = label_reports(evaluator, refs)
        print(f"  labeling predictions for {exp['name']}", flush=True)
        hyps_binary = label_reports(evaluator, hyps)
        metric_rows = per_label_metrics(refs_binary, hyps_binary)
        validate_micro_against_class_report(refs_binary, hyps_binary, metric_rows, exp["name"])
        for metric_row in metric_rows:
            metric_row.update(
                {
                    "experiment": exp["name"],
                    "row_id": exp["row_id"],
                    "report_csv": str(report_csv),
                    "source_type": source_type,
                }
            )
            all_metric_rows.append(metric_row)
        sources.append(
            {
                "experiment": exp["name"],
                "row_id": exp["row_id"],
                "report_csv": str(report_csv),
                "source_type": source_type,
            }
        )

    per_disease_df = pd.DataFrame(all_metric_rows)
    column_order = [
        "experiment",
        "row_id",
        "label",
        "label_type",
        "TP",
        "FP",
        "FN",
        "TN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "support",
        "report_csv",
        "source_type",
    ]
    per_disease_df = per_disease_df[column_order]
    delta_df = build_delta(per_disease_df)

    per_disease_path = output_dir / "per_disease_chexpert.csv"
    delta_path = output_dir / "per_disease_delta_s2_vs_s1.csv"
    pdf_path = output_dir / "per_disease_recall_f1.pdf"
    svg_path = output_dir / "per_disease_recall_f1.svg"
    summary_path = output_dir / "per_disease_summary.md"

    per_disease_df.to_csv(per_disease_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    plot_heatmap(per_disease_df, pdf_path, svg_path)
    write_summary_md(summary_path, per_disease_df, delta_df, sources)
    validate_outputs(output_dir, per_disease_df, delta_df)

    print(f"Wrote {per_disease_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
