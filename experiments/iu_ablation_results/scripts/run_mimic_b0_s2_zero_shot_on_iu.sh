#!/bin/bash
set -euo pipefail

# Zero-shot cross-dataset evaluation:
# MIMIC-CXR trained B0/S2 checkpoints -> IU-Xray test set.
# This script does not train or fine-tune models.

ROOT="/home/sjx/work/project/CMCRL-main"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-experiments/iu_ablation_results/runs/mimic_b0_s2_zero_shot_on_iu_${RUN_STAMP}}"
LOG_FILE="${OUT_ROOT}/run.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
SUMMARY_MD="${OUT_ROOT}/summary.md"

CONFIG="${CONFIG:-config/iu_xray/vlci_mimic_zeroshot.json}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-10200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
SUPPRESS_UNK="${SUPPRESS_UNK:-1}"

B0_CKPT="${B0_CKPT:-experiments/mimic_ablation_results/runs/all_supplementary_20260527_183614/strong_baseline/B0_original_clean_6epoch_seed10200/model_best_val.pth}"
S2_CKPT="${S2_CKPT:-experiments/mimic_ablation_results/final_results/cmcrl_s0_s1_s2_ccd_20260612/weights/S2/model_best_val.pth}"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo ">>> MIMIC B0/S2 -> IU-Xray zero-shot evaluation"
echo ">>> started_at=$(date)"
echo ">>> out_root=${OUT_ROOT}"
echo ">>> config=${CONFIG}"
echo ">>> cuda=${CUDA_ID} seed=${SEED} batch_size=${BATCH_SIZE} beam_size=${BEAM_SIZE} max_seq_length=${MAX_SEQ_LENGTH}"
echo ">>> b0_ckpt=${B0_CKPT}"
echo ">>> s2_ckpt=${S2_CKPT}"
echo "============================================================"

if [ ! -s "$B0_CKPT" ]; then
  echo "ERROR: missing B0 checkpoint: $B0_CKPT" >&2
  exit 1
fi
if [ ! -s "$S2_CKPT" ]; then
  echo "ERROR: missing S2 checkpoint: $S2_CKPT" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local ckpt="$2"
  local model="$3"
  local use_ccli="$4"
  local save_dir="${OUT_ROOT}/${name}"
  local report_csv="${save_dir}/iu_xray_reports.csv"

  mkdir -p "$save_dir"

  echo
  echo "============================================================"
  echo ">>> Running ${name}"
  echo ">>> model=${model}"
  echo ">>> checkpoint=${ckpt}"
  echo ">>> use_counterfactual_intervention=${use_ccli}"
  echo "============================================================"

  python eval_dump_reports.py \
    --c "$CONFIG" \
    --model "$model" \
    --task_name "$name" \
    --result_dir "$save_dir" \
    --load_model_path "$ckpt" \
    --cuda "$CUDA_ID" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --beam_size "$BEAM_SIZE" \
    --sample_method "beam_search" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --suppress_UNK "$SUPPRESS_UNK" \
    --split "test" \
    --use_anatomical_adapter 1 \
    --use_memory_prompt 1 \
    --use_counterfactual_intervention "$use_ccli" \
    --ccd_gamma 0.0

  python eval_clinical.py \
    --c "$CONFIG" \
    --task_name "$name" \
    --result_dir "$save_dir" \
    --report_csv "$report_csv" \
    --report_split "test" \
    --cuda "$CUDA_ID" \
    --seed "$SEED"
}

# B0 is the strong original VLCI baseline. Its checkpoint is compatible with
# the currently registered vlci_original model entry; CCLI is disabled.
run_one "MIMIC_B0_baseline_on_IU_zero_shot" "$B0_CKPT" "vlci_original" 0

# S2 is the CCLI-enabled Stage-II model; CCLI inference is enabled and CCD is off.
run_one "MIMIC_S2_CCLI_on_IU_zero_shot" "$S2_CKPT" "vlci" 1

python - <<PY
import csv
from pathlib import Path

out_root = Path("$OUT_ROOT")
rows = []
for name, label in [
    ("MIMIC_B0_baseline_on_IU_zero_shot", "MIMIC Baseline -> IU"),
    ("MIMIC_S2_CCLI_on_IU_zero_shot", "MIMIC S2 -> IU"),
]:
    clinical_csv = out_root / name / "iu_xray_clinical.csv"
    if not clinical_csv.exists():
        raise FileNotFoundError(clinical_csv)
    row = next(csv.DictReader(open(clinical_csv)))
    rows.append({
        "Model": label,
        "BLEU-4": row.get("test_BLEU_4", ""),
        "CE-P": row.get("test_CE_P", ""),
        "CE-R": row.get("test_CE_R", ""),
        "CE-F1": row.get("test_CE_F1", ""),
        "Description": "zero-shot",
        "report_csv": str(out_root / name / "iu_xray_reports.csv"),
        "clinical_csv": str(clinical_csv),
    })

summary_csv = out_root / "summary.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "BLEU-4",
            "CE-P",
            "CE-R",
            "CE-F1",
            "Description",
            "report_csv",
            "clinical_csv",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

summary_md = out_root / "summary.md"
with open(summary_md, "w") as f:
    f.write("# MIMIC to IU-Xray Zero-shot Results\n\n")
    f.write("| Model | BLEU-4 | CE-P | CE-R | CE-F1 | Description |\n")
    f.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
    for row in rows:
        f.write(
            f"| {row['Model']} | {float(row['BLEU-4']):.4f} | "
            f"{float(row['CE-P']):.4f} | {float(row['CE-R']):.4f} | "
            f"{float(row['CE-F1']):.4f} | {row['Description']} |\n"
        )

print(f"Wrote {summary_csv}")
print(f"Wrote {summary_md}")
PY

echo
echo "============================================================"
echo ">>> Finished at $(date)"
echo ">>> log_file=${LOG_FILE}"
echo ">>> summary_csv=${SUMMARY_CSV}"
echo ">>> summary_md=${SUMMARY_MD}"
echo "============================================================"
