#!/usr/bin/env python3
"""Collect IU-Xray ablation metrics into a compact CSV/Markdown table."""

import argparse
import csv
from pathlib import Path


FIELDS = [
    "row_id",
    "group",
    "experiment",
    "model_path",
    "protocol",
    "start_checkpoint",
    "modules",
    "epochs",
    "seed",
    "test_BLEU_1",
    "test_BLEU_2",
    "test_BLEU_3",
    "test_BLEU_4",
    "test_METEOR",
    "test_ROUGE_L",
    "test_CIDEr",
    "clinical_CE_P",
    "clinical_CE_R",
    "clinical_CE_F1",
    "chinese_summary",
    "run_dir",
    "csv_path",
    "notes",
]

METRIC_MAP = {
    "test_BLEU_1": "test_BLEU_1",
    "test_BLEU_2": "test_BLEU_2",
    "test_BLEU_3": "test_BLEU_3",
    "test_BLEU_4": "test_BLEU_4",
    "test_METEOR": "test_METEOR",
    "test_ROUGE_L": "test_ROUGE_L",
    "test_CIDEr": "test_CIDEr",
    "clinical_CE_P": "test_CE_P",
    "clinical_CE_R": "test_CE_R",
    "clinical_CE_F1": "test_CE_F1",
}

ROW_META = {
    "IU_B0_author_original_test": {
        "group": "author_baseline",
        "experiment": "IU author original checkpoint test",
        "model_path": "vlci_original",
        "protocol": "test-only reproduction",
        "modules": "original VLCI",
        "epochs": "",
        "chinese_summary": "作者 IU-Xray checkpoint 复现基线，使用真正原始 VLCI 模型入口进行 test-only 评估。",
    },
    "IU_B1_original_continue": {
        "group": "original_continue",
        "experiment": "IU original VLCI continued training",
        "model_path": "vlci_original",
        "protocol": "same-budget original continuation",
        "modules": "original VLCI",
        "epochs": "10",
        "chinese_summary": "原始 VLCI 从作者 IU checkpoint 继续训练，用来排除只是多训练带来提升的解释。",
    },
    "IU_S1_AB_continue": {
        "group": "clean_ablation",
        "experiment": "IU clean A+B continued training",
        "model_path": "vlci_ccli_clean",
        "protocol": "A+B without CCLI",
        "modules": "AB",
        "epochs": "10",
        "chinese_summary": "在 IU 小数据集上加入解剖适配与动态医学提示，关闭 CCLI，验证 A+B 的贡献。",
    },
    "IU_S2_ABC_continue": {
        "group": "clean_ablation",
        "experiment": "IU clean A+B+C continued training",
        "model_path": "vlci_ccli_clean",
        "protocol": "A+B+C with CCLI",
        "modules": "ABC",
        "epochs": "10",
        "chinese_summary": "从 IU_S1_AB_continue 的 best-val checkpoint 继续训练并加入 CCLI，验证 C 模块的进一步贡献。",
    },
    "IU_CCD_selected": {
        "group": "ccd_validation_selected",
        "experiment": "IU CCD selected by validation",
        "model_path": "vlci_ccli_clean",
        "protocol": "validation-selected CCD decoding",
        "modules": "ABC+CCD",
        "epochs": "",
        "chinese_summary": "CCD 推理阶段补充实验：先用验证集选择 gamma，再汇报对应测试集结果，不属于训练模块消融。",
    },
}


def read_first_row(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            return row
    return {}


def fmt(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def metric_values(clinical_csv):
    raw = read_first_row(clinical_csv)
    return {dst: fmt(raw.get(src, "")) for dst, src in METRIC_MAP.items()}


def collect_rows(run_root, default_seed):
    rows = []
    summary_csv = run_root / "summary.csv"
    if not summary_csv.exists():
        return rows

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            row_id = item.get("row_id", "")
            clinical_csv = Path(item.get("clinical_csv", ""))
            if not row_id or not clinical_csv.exists():
                continue

            meta = ROW_META.get(row_id, {})
            row = {key: "" for key in FIELDS}
            row.update(meta)
            row["row_id"] = row_id
            row["seed"] = item.get("seed", default_seed)
            row["start_checkpoint"] = item.get("checkpoint", "")
            row["run_dir"] = item.get("result_dir", "")
            row["csv_path"] = str(clinical_csv)
            row["notes"] = item.get("notes", "")
            if item.get("epochs"):
                row["epochs"] = item["epochs"]
            if item.get("selected_gamma"):
                row["notes"] = f"selected_gamma={item['selected_gamma']}; {row['notes']}".strip("; ")
            row.update(metric_values(clinical_csv))
            rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def write_md(path, rows):
    headers = ["实验", "类别", "模块", "轮次", "Seed", "BLEU_4", "CIDEr", "CE_F1", "中文说明"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# IU-Xray Ablation Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [
                row.get("row_id", ""),
                row.get("group", ""),
                row.get("modules", ""),
                row.get("epochs", ""),
                row.get("seed", ""),
                row.get("test_BLEU_4", ""),
                row.get("test_CIDEr", ""),
                row.get("clinical_CE_F1", ""),
                row.get("chinese_summary", ""),
            ]
            f.write("| " + " | ".join(str(v).replace("|", "/") for v in values) + " |\n")
        f.write("\n## Notes\n\n")
        f.write("- IU-Xray experiments use the official IU checkpoint as initialization.\n")
        f.write("- CCD rows are inference-time decoding supplements, not training ablations.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", default="experiments/iu_ablation_results/summaries")
    parser.add_argument("--seed", default="10200")
    args = parser.parse_args()

    rows = collect_rows(Path(args.run_root), args.seed)
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "iu_ablation_summary.csv"
    md_path = out_dir / "iu_ablation_summary.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    print(f"Updated IU summary CSV: {csv_path}")
    print(f"Updated IU summary MD : {md_path}")


if __name__ == "__main__":
    main()
