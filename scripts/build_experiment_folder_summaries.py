#!/usr/bin/env python3
"""Build folder-level result summary tables for MIMIC-CXR and IU-Xray experiments."""

from __future__ import annotations

import csv
from pathlib import Path


MIMIC_CANONICAL = Path(
    "experiments/mimic_ablation_results/summaries/experiment_summary_20260527/"
    "all_ablation_summary.csv"
)
IU_FROM_CHECKPOINT = Path("experiments/iu_ablation_results/summaries/iu_ablation_summary.csv")
IU_FROM_PRETRAIN = Path("experiments/iu_ablation_results/summaries/iu_from_pretrain_summary.csv")
IU_RECIPE_SWEEP = Path(
    "experiments/iu_ablation_results/runs/iu_recipe_sweep_20260604_104426/"
    "recipe_sweep_summary.csv"
)
IU_FOCUSED_SWEEP = Path(
    "experiments/iu_ablation_results/runs/iu_focused_20epoch_sweep_20260605_090822/"
    "recipe_sweep_summary.csv"
)
IU_ZERO_SHOT_S0_S2 = Path(
    "experiments/iu_ablation_results/runs/mimic_zero_shot_on_iu_20260612_171533/"
    "summary.csv"
)
IU_ZERO_SHOT_B0_S2 = Path(
    "experiments/iu_ablation_results/runs/mimic_b0_s2_zero_shot_on_iu_20260613_b0_s2/"
    "summary.csv"
)

MIMIC_OUT_DIR = Path(
    "experiments/mimic_ablation_results/final_results/"
    "experiment_folder_summary_20260613"
)
IU_OUT_DIR = Path(
    "experiments/iu_ablation_results/final_results/"
    "experiment_folder_summary_20260613"
)

BASE_COLUMNS = [
    "row_id",
    "group",
    "status",
    "experiment",
    "model_path",
    "protocol",
    "start_checkpoint",
    "A",
    "B",
    "C",
    "modules",
    "epochs",
    "seed",
    "test_epoch",
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
    "summary_scope",
    "source_table",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalize_row(row) for row in rows)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {col: str(row.get(col, "") or "") for col in BASE_COLUMNS}


def metric(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return ""


def with_source(rows: list[dict[str, str]], scope: str, source: Path) -> list[dict[str, str]]:
    out = []
    for row in rows:
        item = dict(row)
        item["summary_scope"] = scope
        item["source_table"] = str(source)
        out.append(item)
    return out


def convert_recipe_sweep(rows: list[dict[str, str]], scope: str, source: Path) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        recipe_id = row.get("recipe_id", "")
        row_id = row.get("row_id", "")
        item = {
            "row_id": f"{scope}:{recipe_id}:{row_id}",
            "group": scope,
            "status": "supplementary_sweep",
            "experiment": row_id,
            "model_path": "",
            "protocol": "IU from-pretrain hyperparameter sweep",
            "start_checkpoint": row.get("checkpoint", ""),
            "modules": infer_modules(row_id),
            "epochs": row.get("epochs", ""),
            "seed": row.get("seed", ""),
            "test_BLEU_4": row.get("test_BLEU_4", ""),
            "test_CIDEr": row.get("test_CIDEr", ""),
            "clinical_CE_F1": row.get("clinical_CE_F1", ""),
            "chinese_summary": "IU-Xray from-pretrain 超参数搜索记录。",
            "run_dir": row.get("run_root", ""),
            "csv_path": "",
            "notes": "; ".join(
                part
                for part in [
                    f"recipe_id={recipe_id}",
                    f"lr={row.get('lr', '')}",
                    f"batch_size={row.get('batch_size', '')}",
                    row.get("notes", ""),
                ]
                if part
            ),
            "summary_scope": scope,
            "source_table": str(source),
        }
        converted.append(item)
    return converted


def convert_zero_shot_old(rows: list[dict[str, str]], scope: str, source: Path) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        experiment = row.get("experiment", "")
        if experiment.startswith("delta_"):
            continue
        item = {
            "row_id": experiment,
            "group": scope,
            "status": "supplementary_zero_shot",
            "experiment": experiment,
            "model_path": "vlci_ccli_clean",
            "protocol": "MIMIC checkpoint zero-shot inference on IU-Xray test",
            "modules": infer_modules(experiment),
            "test_BLEU_4": row.get("test_BLEU_4", ""),
            "clinical_CE_P": row.get("test_CE_P", ""),
            "clinical_CE_R": row.get("test_CE_R", ""),
            "clinical_CE_F1": row.get("test_CE_F1", ""),
            "chinese_summary": "MIMIC 训练好的模型直接在 IU-Xray test set 上推理，不训练、不微调。",
            "run_dir": str(Path(row.get("report_csv", "")).parent) if row.get("report_csv") else "",
            "csv_path": row.get("clinical_csv", ""),
            "notes": f"report_csv={row.get('report_csv', '')}",
            "summary_scope": scope,
            "source_table": str(source),
        }
        converted.append(item)
    return converted


def convert_zero_shot_b0_s2(rows: list[dict[str, str]], scope: str, source: Path) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        model = row.get("Model", "")
        row_id = (
            model.replace(" ", "_")
            .replace("->", "to")
            .replace("/", "_")
            .replace("__", "_")
        )
        item = {
            "row_id": row_id,
            "group": scope,
            "status": "supplementary_zero_shot",
            "experiment": model,
            "model_path": "vlci_ccli_clean",
            "protocol": "MIMIC checkpoint zero-shot inference on IU-Xray test",
            "modules": infer_modules(model),
            "test_BLEU_4": row.get("BLEU-4", ""),
            "clinical_CE_P": row.get("CE-P", ""),
            "clinical_CE_R": row.get("CE-R", ""),
            "clinical_CE_F1": row.get("CE-F1", ""),
            "chinese_summary": "MIMIC 训练好的 Baseline/S2 模型直接在 IU-Xray test set 上推理。",
            "run_dir": str(Path(row.get("report_csv", "")).parent) if row.get("report_csv") else "",
            "csv_path": row.get("clinical_csv", ""),
            "notes": row.get("Description", ""),
            "summary_scope": scope,
            "source_table": str(source),
        }
        converted.append(item)
    return converted


def infer_modules(text: str) -> str:
    lower = text.lower()
    if "ccd" in lower:
        return "ABC+CCD"
    if "abc" in lower or "s2" in lower or "ccli" in lower:
        return "ABC"
    if "ab" in lower or "s0" in lower or "s1" in lower:
        return "AB"
    if "baseline" in lower or "author" in lower or "original" in lower or "p0" in lower:
        return "original VLCI"
    return ""


def write_markdown(rows: list[dict[str, str]], path: Path, title: str) -> None:
    display_cols = [
        "row_id",
        "group",
        "status",
        "modules",
        "epochs",
        "seed",
        "test_BLEU_4",
        "test_CIDEr",
        "clinical_CE_P",
        "clinical_CE_R",
        "clinical_CE_F1",
        "summary_scope",
    ]
    lines = [
        f"# {title}",
        "",
        f"- Rows: {len(rows)}",
        "- CSV keeps the all_ablation_summary-style columns plus source tracking.",
        "- Original experiment summary files are not modified.",
        "",
        "| " + " | ".join(display_cols) + " |",
        "| " + " | ".join(["---"] * len(display_cols)) + " |",
    ]
    for row in rows:
        normalized = normalize_row(row)
        lines.append("| " + " | ".join(md_cell(normalized[col]) for col in display_cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_mimic() -> list[dict[str, str]]:
    rows = with_source(
        read_csv(MIMIC_CANONICAL),
        "mimic_canonical_all_ablation",
        MIMIC_CANONICAL,
    )
    if not rows:
        raise FileNotFoundError(f"No rows loaded from {MIMIC_CANONICAL}")
    return rows


def build_iu() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(with_source(read_csv(IU_FROM_CHECKPOINT), "iu_canonical_from_checkpoint", IU_FROM_CHECKPOINT))
    rows.extend(with_source(read_csv(IU_FROM_PRETRAIN), "iu_canonical_from_pretrain", IU_FROM_PRETRAIN))
    rows.extend(convert_recipe_sweep(read_csv(IU_RECIPE_SWEEP), "iu_recipe_sweep_10epoch", IU_RECIPE_SWEEP))
    rows.extend(convert_recipe_sweep(read_csv(IU_FOCUSED_SWEEP), "iu_focused_sweep_20epoch", IU_FOCUSED_SWEEP))
    rows.extend(convert_zero_shot_old(read_csv(IU_ZERO_SHOT_S0_S2), "mimic_to_iu_zero_shot_s0_s2", IU_ZERO_SHOT_S0_S2))
    rows.extend(convert_zero_shot_b0_s2(read_csv(IU_ZERO_SHOT_B0_S2), "mimic_to_iu_zero_shot_b0_s2", IU_ZERO_SHOT_B0_S2))
    if not rows:
        raise FileNotFoundError("No IU rows were loaded")
    return rows


def main() -> None:
    mimic_rows = build_mimic()
    iu_rows = build_iu()

    mimic_csv = MIMIC_OUT_DIR / "mimic_results_summary.csv"
    mimic_md = MIMIC_OUT_DIR / "mimic_results_summary.md"
    iu_csv = IU_OUT_DIR / "iu_results_summary.csv"
    iu_md = IU_OUT_DIR / "iu_results_summary.md"

    write_csv(mimic_rows, mimic_csv)
    write_markdown(mimic_rows, mimic_md, "MIMIC-CXR Experiment Folder Result Summary")
    write_csv(iu_rows, iu_csv)
    write_markdown(iu_rows, iu_md, "IU-Xray Experiment Folder Result Summary")

    print(f"Wrote {mimic_csv} ({len(mimic_rows)} rows)")
    print(f"Wrote {mimic_md}")
    print(f"Wrote {iu_csv} ({len(iu_rows)} rows)")
    print(f"Wrote {iu_md}")


if __name__ == "__main__":
    main()
