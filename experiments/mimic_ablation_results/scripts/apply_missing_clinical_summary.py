#!/usr/bin/env python3
"""Apply supplemented clinical metrics back to the canonical MIMIC table."""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


DISPLAY_COLUMNS = [
    ("实验", "row_id"),
    ("类别", "group"),
    ("模块", "modules"),
    ("轮次", "epochs"),
    ("Seed", "seed"),
    ("Test BLEU_4", "test_BLEU_4"),
    ("Test CIDEr", "test_CIDEr"),
    ("CE_F1", "clinical_CE_F1"),
    ("中文总结", "chinese_summary"),
]


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_row(path):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0]


def metric_from_clinical(row, clinical_key, default=""):
    return row.get(clinical_key, default)


def ensure_baseline_row(rows, fieldnames, baseline_clinical, args):
    if not baseline_clinical:
        return None
    baseline_path = Path(baseline_clinical)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline clinical CSV: {baseline_path}")

    metrics = first_row(baseline_path)
    row_id = args.baseline_row_id
    existing = next((row for row in rows if row.get("row_id") == row_id), None)
    if existing is None:
        existing = {key: "" for key in fieldnames}
        rows.append(existing)

    values = {
        "row_id": row_id,
        "group": "baseline_ladder",
        "status": "supplementary",
        "experiment": "baseline potential2 full test evaluation",
        "model_path": args.model_name,
        "protocol": "test-only evaluation",
        "start_checkpoint": args.baseline_checkpoint,
        "A": "0",
        "B": "0",
        "C": "0",
        "modules": "None",
        "epochs": "",
        "seed": args.seed,
        "test_epoch": "",
        "test_BLEU_1": metric_from_clinical(metrics, "test_BLEU_1"),
        "test_BLEU_2": metric_from_clinical(metrics, "test_BLEU_2"),
        "test_BLEU_3": metric_from_clinical(metrics, "test_BLEU_3"),
        "test_BLEU_4": metric_from_clinical(metrics, "test_BLEU_4"),
        "test_METEOR": metric_from_clinical(metrics, "test_METEOR"),
        "test_ROUGE_L": metric_from_clinical(metrics, "test_ROUGE_L"),
        "test_CIDEr": metric_from_clinical(metrics, "test_CIDEr"),
        "clinical_CE_P": metric_from_clinical(metrics, "test_CE_P"),
        "clinical_CE_R": metric_from_clinical(metrics, "test_CE_R"),
        "clinical_CE_F1": metric_from_clinical(metrics, "test_CE_F1"),
        "chinese_summary": "强基线 baseline_potential2 的完整测试集复评结果，补充 BLEU/CIDEr 与临床 CE 指标，用作无模块强基线参考。",
        "run_dir": args.baseline_result_dir,
        "csv_path": str(baseline_path),
        "notes": "supplemented full clinical evaluation for baseline_potential2",
    }
    for key, value in values.items():
        if key in existing:
            existing[key] = value
    return row_id


def backup_if_needed(path, backup_dir):
    path = Path(path)
    if not path.exists():
        return ""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / path.name
    shutil.copy2(path, dst)
    return str(dst)


def format_cell(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|")


def write_markdown(path, rows):
    lines = [
        "# All Ablation Summary",
        "",
        "当前版本只保留测试集指标；验证集相关字段不进入主表。",
        "",
        "| " + " | ".join(label for label, _ in DISPLAY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_cell(row.get(key, "")) for _, key in DISPLAY_COLUMNS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 表中 Test 指标统一来自测试集结果。",
            "- `clinical_CE_F1` 为空表示该行暂未做离线临床指标评估。",
            "- `legacy_dirty_ablation` 只保留历史记录，不建议作为主论文结论。",
            "- `ccd_validation_selected` 是先用验证集选择 gamma 后再汇报测试集的 CCD 结果。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(path, rows, fieldnames, baseline_row_id, backups):
    targets = ["C1", "C2", "C3", "E4", "E5", "A6", "B6", "AB6", "BC6", "ABC6"]
    if baseline_row_id:
        targets.append(baseline_row_id)
    by_id = {row.get("row_id"): row for row in rows}
    audit_rows = []
    for row_id in targets:
        row = by_id.get(row_id, {})
        audit_rows.append(
            {
                "row_id": row_id,
                "clinical_CE_P": row.get("clinical_CE_P", ""),
                "clinical_CE_R": row.get("clinical_CE_R", ""),
                "clinical_CE_F1": row.get("clinical_CE_F1", ""),
                "status": "present" if row else "missing",
            }
        )
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_id", "clinical_CE_P", "clinical_CE_R", "clinical_CE_F1", "status"],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    backup_note = audit_path.with_suffix(".md")
    backup_lines = [
        "# Canonical Summary Update Audit",
        "",
        f"- Timestamp: `{datetime.now().strftime('%F %T')}`",
        f"- CSV backup: `{backups.get('csv', '')}`",
        f"- Markdown backup: `{backups.get('md', '')}`",
        f"- Columns: `{', '.join(fieldnames)}`",
        f"- Audit CSV: `{audit_path}`",
    ]
    backup_note.write_text("\n".join(backup_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemented-summary", required=True)
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--target-md", required=True)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--baseline-clinical", default="")
    parser.add_argument("--baseline-row-id", default="B_potential2_0424")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--baseline-result-dir", default="")
    parser.add_argument("--model-name", default="vlci")
    parser.add_argument("--seed", default="10200")
    args = parser.parse_args()

    rows = read_rows(args.supplemented_summary)
    if not rows:
        raise ValueError(f"Supplemented summary is empty: {args.supplemented_summary}")
    fieldnames = list(rows[0].keys())

    baseline_row_id = ensure_baseline_row(rows, fieldnames, args.baseline_clinical, args)

    backups = {
        "csv": backup_if_needed(args.target_summary, args.backup_dir),
        "md": backup_if_needed(args.target_md, args.backup_dir),
    }
    write_rows(args.target_summary, rows, fieldnames)
    write_markdown(args.target_md, rows)
    write_audit(args.audit_out, rows, fieldnames, baseline_row_id, backups)

    print(f"Updated canonical CSV: {args.target_summary}")
    print(f"Updated canonical markdown: {args.target_md}")
    print(f"Backups written under: {args.backup_dir}")
    print(f"Audit written to: {args.audit_out}")


if __name__ == "__main__":
    main()
