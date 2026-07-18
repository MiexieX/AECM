#!/usr/bin/env python3
"""Merge missing clinical metrics from supplemental evaluations.

The original ablation summary is kept read-only. This script writes a dated
copy with only empty clinical_CE_P/R/F1 fields filled for the target rows.
"""

import argparse
import csv
from pathlib import Path


TARGET_ROW_IDS = ["C1", "C2", "C3", "E4", "E5", "A6", "B6", "AB6", "BC6", "ABC6"]
SUMMARY_CE_COLUMNS = {
    "clinical_CE_P": "test_CE_P",
    "clinical_CE_R": "test_CE_R",
    "clinical_CE_F1": "test_CE_F1",
}
CHECK_METRICS = ["BLEU_4", "METEOR", "ROUGE_L", "CIDEr"]


def read_csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_row(path):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0]


def is_blank(value):
    return value is None or str(value).strip() == ""


def as_float(value):
    if is_blank(value):
        return None
    return float(value)


def load_eval_sources(eval_summary):
    sources = {}
    for row in read_csv_rows(eval_summary):
        if row.get("kind") != "ablation":
            continue
        row_id = row.get("row_id", "")
        if row_id not in TARGET_ROW_IDS:
            continue
        sources[row_id] = row
    return sources


def metric_warnings(original_row, clinical_row, tolerance):
    warnings = []
    for metric in CHECK_METRICS:
        old_value = as_float(original_row.get(f"test_{metric}", ""))
        new_value = as_float(clinical_row.get(f"test_{metric}", ""))
        if old_value is None or new_value is None:
            continue
        diff = abs(old_value - new_value)
        if diff > tolerance:
            warnings.append(
                {
                    "row_id": original_row.get("row_id", ""),
                    "metric": metric,
                    "original": f"{old_value:.12g}",
                    "new_eval": f"{new_value:.12g}",
                    "abs_diff": f"{diff:.12g}",
                }
            )
    return warnings


def write_markdown_summary(path, audit_rows, warning_rows, output_csv, baseline_csv):
    filled = [r for r in audit_rows if r["status"] == "filled"]
    skipped_existing = [r for r in audit_rows if r["status"] == "kept_existing"]
    missing = [r for r in audit_rows if r["status"].startswith("missing")]

    lines = [
        "# Missing Clinical Metric Merge",
        "",
        f"- Supplemented summary: `{output_csv}`",
        f"- Baseline metrics: `{baseline_csv}`" if baseline_csv else "- Baseline metrics: not provided",
        f"- Filled rows: {len(filled)}",
        f"- Kept existing rows: {len(skipped_existing)}",
        f"- Missing/problem rows: {len(missing)}",
        f"- Metric mismatch warnings: {len(warning_rows)}",
        "",
        "## Audit",
        "",
        "| row_id | status | CE_P | CE_R | CE_F1 | source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in audit_rows:
        lines.append(
            "| {row_id} | {status} | {clinical_CE_P} | {clinical_CE_R} | "
            "{clinical_CE_F1} | `{source}` |".format(**row)
        )

    if warning_rows:
        lines.extend(["", "## Metric Warnings", ""])
        lines.append("| row_id | metric | original | new_eval | abs_diff |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in warning_rows:
            lines.append(
                "| {row_id} | {metric} | {original} | {new_eval} | {abs_diff} |".format(**row)
            )

    Path(path).write_text("\n".join(lines) + "\n")


def write_baseline_tables(output_dir, baseline_clinical):
    if not baseline_clinical:
        return "", ""

    baseline_path = Path(baseline_clinical)
    if not baseline_path.exists():
        return "", ""

    row = first_row(baseline_path)
    output_csv = output_dir / "baseline_potential2_full_eval_metrics_20260608.csv"
    write_csv_rows(output_csv, [row], list(row.keys()))

    output_md = output_dir / "baseline_potential2_full_eval_metrics_20260608.md"
    preferred = [
        "test_BLEU_1",
        "test_BLEU_2",
        "test_BLEU_3",
        "test_BLEU_4",
        "test_METEOR",
        "test_ROUGE_L",
        "test_CIDEr",
        "test_CE_P",
        "test_CE_R",
        "test_CE_F1",
    ]
    lines = [
        "# Baseline Potential2 Full Test Evaluation",
        "",
        f"- Source: `{baseline_path}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in preferred:
        if key in row:
            lines.append(f"| {key} | {row[key]} |")
    output_md.write_text("\n".join(lines) + "\n")
    return str(output_csv), str(output_md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-summary", required=True)
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-clinical", default="")
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = read_csv_rows(args.original_summary)
    if not original_rows:
        raise ValueError(f"Original summary is empty: {args.original_summary}")
    fieldnames = list(original_rows[0].keys())
    row_by_id = {row["row_id"]: row for row in original_rows}
    sources = load_eval_sources(args.eval_summary)

    audit_rows = []
    warning_rows = []

    for row_id in TARGET_ROW_IDS:
        original_row = row_by_id.get(row_id)
        if original_row is None:
            audit_rows.append(
                {
                    "row_id": row_id,
                    "status": "missing_original_row",
                    "clinical_CE_P": "",
                    "clinical_CE_R": "",
                    "clinical_CE_F1": "",
                    "source": "",
                }
            )
            continue

        source = sources.get(row_id)
        if source is None:
            audit_rows.append(
                {
                    "row_id": row_id,
                    "status": "missing_eval_summary_row",
                    "clinical_CE_P": original_row.get("clinical_CE_P", ""),
                    "clinical_CE_R": original_row.get("clinical_CE_R", ""),
                    "clinical_CE_F1": original_row.get("clinical_CE_F1", ""),
                    "source": "",
                }
            )
            continue

        clinical_csv = source.get("clinical_csv", "")
        if not clinical_csv or not Path(clinical_csv).exists():
            audit_rows.append(
                {
                    "row_id": row_id,
                    "status": "missing_clinical_csv",
                    "clinical_CE_P": original_row.get("clinical_CE_P", ""),
                    "clinical_CE_R": original_row.get("clinical_CE_R", ""),
                    "clinical_CE_F1": original_row.get("clinical_CE_F1", ""),
                    "source": clinical_csv,
                }
            )
            continue

        clinical_row = first_row(clinical_csv)
        already_had_all = all(not is_blank(original_row.get(col, "")) for col in SUMMARY_CE_COLUMNS)
        for summary_col, clinical_col in SUMMARY_CE_COLUMNS.items():
            if is_blank(original_row.get(summary_col, "")):
                original_row[summary_col] = clinical_row.get(clinical_col, "")

        status = "kept_existing" if already_had_all else "filled"
        audit_rows.append(
            {
                "row_id": row_id,
                "status": status,
                "clinical_CE_P": original_row.get("clinical_CE_P", ""),
                "clinical_CE_R": original_row.get("clinical_CE_R", ""),
                "clinical_CE_F1": original_row.get("clinical_CE_F1", ""),
                "source": clinical_csv,
            }
        )
        warning_rows.extend(metric_warnings(original_row, clinical_row, args.metric_tolerance))

    output_csv = output_dir / "all_ablation_summary_with_missing_clinical_20260608.csv"
    output_md = output_dir / "all_ablation_summary_with_missing_clinical_20260608.md"
    audit_csv = output_dir / "missing_metric_update_audit_20260608.csv"
    warning_csv = output_dir / "missing_metric_bleu_warning_20260608.csv"

    write_csv_rows(output_csv, original_rows, fieldnames)
    write_csv_rows(audit_csv, audit_rows, ["row_id", "status", "clinical_CE_P", "clinical_CE_R", "clinical_CE_F1", "source"])
    write_csv_rows(warning_csv, warning_rows, ["row_id", "metric", "original", "new_eval", "abs_diff"])

    baseline_csv, _baseline_md = write_baseline_tables(output_dir, args.baseline_clinical)
    write_markdown_summary(output_md, audit_rows, warning_rows, output_csv, baseline_csv)

    missing = [row for row in audit_rows if row["status"].startswith("missing")]
    print(f"Wrote supplemented summary: {output_csv}")
    print(f"Wrote audit CSV: {audit_csv}")
    print(f"Wrote warning CSV: {warning_csv}")
    if baseline_csv:
        print(f"Wrote baseline metric table: {baseline_csv}")
    if missing:
        print("Rows still missing clinical metrics:")
        for row in missing:
            print(f"  {row['row_id']}: {row['status']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
