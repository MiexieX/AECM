#!/usr/bin/env python3
"""Collect IU-Xray from-pretrain ablation metrics."""

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
    "IU_P0_original_from_pretrain": {
        "group": "from_pretrain_original",
        "experiment": "IU vlci base from RadCARE pretrain",
        "model_path": "vlci",
        "protocol": "fine-tune from nano pretrain",
        "modules": "vlci no A/B/C",
        "epochs": "10",
        "chinese_summary": "使用当前 vlci 入口并关闭 A/B/C，从 nano_current_checkpoint 预训练权重开始复现 IU fine-tuning。",
    },
    "IU_P1_AB_from_pretrain": {
        "group": "from_pretrain_clean_ablation",
        "experiment": "IU A+B from RadCARE pretrain",
        "model_path": "vlci",
        "protocol": "A+B fine-tune from nano pretrain",
        "modules": "AB",
        "epochs": "10",
        "chinese_summary": "使用当前 vlci 入口，从同一个预训练起点加入 A+B，关闭 CCLI，验证新增结构模块在 IU 上的贡献。",
    },
    "IU_P2_ABC_from_pretrain": {
        "group": "from_pretrain_clean_ablation",
        "experiment": "IU A+B+C from P1 checkpoint",
        "model_path": "vlci",
        "protocol": "A+B+C fine-tune from P1",
        "modules": "ABC",
        "epochs": "10",
        "chinese_summary": "使用当前 vlci 入口，从 IU_P1_AB_from_pretrain 的 best-val checkpoint 继续并加入 CCLI，验证 C 模块增益。",
    },
    "IU_P2_CCD_selected": {
        "group": "from_pretrain_ccd",
        "experiment": "IU P2 CCD selected by validation",
        "model_path": "vlci",
        "protocol": "validation-selected CCD decoding",
        "modules": "ABC+CCD",
        "epochs": "",
        "chinese_summary": "基于 from-pretrain P2 模型做 CCD 解码，先验证集选 gamma，再汇报测试集。",
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


def collect_rows(run_root, default_seed):
    summary_csv = run_root / "summary.csv"
    if not summary_csv.exists():
        return []

    rows = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            row_id = item.get("row_id", "")
            clinical_csv = Path(item.get("clinical_csv", ""))
            if not row_id or not clinical_csv.exists():
                continue
            raw = read_first_row(clinical_csv)
            row = {key: "" for key in FIELDS}
            row.update(ROW_META.get(row_id, {}))
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
            for dst, src in METRIC_MAP.items():
                row[dst] = fmt(raw.get(src, ""))
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
        f.write("# IU-Xray From-Pretrain Ablation Summary\n\n")
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
        f.write("- All rows start from the IU-compatible 3-layer `nano_current_checkpoint.pth` path, except staged P2/CCD continuations.\n")
        f.write("- CCD rows are inference-time decoding supplements.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", default="experiments/iu_ablation_results/summaries")
    parser.add_argument("--seed", default="10200")
    args = parser.parse_args()

    rows = collect_rows(Path(args.run_root), args.seed)
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "iu_from_pretrain_summary.csv"
    md_path = out_dir / "iu_from_pretrain_summary.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    print(f"Updated IU from-pretrain summary CSV: {csv_path}")
    print(f"Updated IU from-pretrain summary MD : {md_path}")


if __name__ == "__main__":
    main()
