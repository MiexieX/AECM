#!/bin/bash
set -euo pipefail

# Zero-shot cross-dataset evaluation:
# MIMIC-CXR trained checkpoints -> IU-Xray test set.
# This script does not train or fine-tune models.

ROOT="/home/sjx/work/project/CMCRL-main"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-experiments/iu_ablation_results/runs/mimic_zero_shot_on_iu_${RUN_STAMP}}"
LOG_FILE="${OUT_ROOT}/run.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
CONFIG="${CONFIG:-config/iu_xray/vlci_mimic_zeroshot.json}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-10200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
SUPPRESS_UNK="${SUPPRESS_UNK:-1}"

S0_CKPT="${S0_CKPT:-experiments/mimic_ablation_results/final_results/cmcrl_s0_s1_s2_ccd_20260612/weights/S0/model_best_val.pth}"
S2_CKPT="${S2_CKPT:-experiments/mimic_ablation_results/final_results/cmcrl_s0_s1_s2_ccd_20260612/weights/S2/model_best_val.pth}"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf "============================================================\n"
printf ">>> MIMIC -> IU-Xray zero-shot evaluation\n"
printf ">>> started_at=%s\n" "$(date)"
printf ">>> out_root=%s\n" "$OUT_ROOT"
printf ">>> config=%s\n" "$CONFIG"
printf ">>> cuda=%s seed=%s batch_size=%s beam_size=%s max_seq_length=%s\n" "$CUDA_ID" "$SEED" "$BATCH_SIZE" "$BEAM_SIZE" "$MAX_SEQ_LENGTH"
printf ">>> s0_ckpt=%s\n" "$S0_CKPT"
printf ">>> s2_ckpt=%s\n" "$S2_CKPT"
printf "============================================================\n"

if [ ! -s "$S0_CKPT" ]; then
  echo "ERROR: missing S0 checkpoint: $S0_CKPT" >&2
  exit 1
fi
if [ ! -s "$S2_CKPT" ]; then
  echo "ERROR: missing S2 checkpoint: $S2_CKPT" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local ckpt="$2"
  local use_ccli="$3"
  local save_dir="${OUT_ROOT}/${name}"
  local report_csv="${save_dir}/iu_xray_reports.csv"

  mkdir -p "$save_dir"
  printf "\n============================================================\n"
  printf ">>> Running %s\n" "$name"
  printf ">>> checkpoint=%s\n" "$ckpt"
  printf ">>> use_counterfactual_intervention=%s\n" "$use_ccli"
  printf "============================================================\n"

  python eval_dump_reports.py \
    --c "$CONFIG" \
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

run_one "S0_mimic_stage1_AB_on_iu_test" "$S0_CKPT" 0
run_one "S2_mimic_CCLI_on_iu_test" "$S2_CKPT" 1

python - <<PY
import csv
from pathlib import Path

out_root = Path("$OUT_ROOT")
rows = []
for name in ["S0_mimic_stage1_AB_on_iu_test", "S2_mimic_CCLI_on_iu_test"]:
    clinical_csv = out_root / name / "iu_xray_clinical.csv"
    if not clinical_csv.exists():
        raise FileNotFoundError(clinical_csv)
    row = next(csv.DictReader(open(clinical_csv)))
    rows.append({
        "experiment": name,
        "test_BLEU_4": row.get("test_BLEU_4", ""),
        "test_CE_P": row.get("test_CE_P", ""),
        "test_CE_R": row.get("test_CE_R", ""),
        "test_CE_F1": row.get("test_CE_F1", ""),
        "report_csv": str(out_root / name / "iu_xray_reports.csv"),
        "clinical_csv": str(clinical_csv),
    })

s1 = rows[0]
s2 = rows[1]
try:
    delta_bleu4 = float(s2["test_BLEU_4"]) - float(s1["test_BLEU_4"])
    delta_cef1 = float(s2["test_CE_F1"]) - float(s1["test_CE_F1"])
    rows.append({
        "experiment": "delta_S2_minus_S0",
        "test_BLEU_4": f"{delta_bleu4:.10f}",
        "test_CE_P": "",
        "test_CE_R": "",
        "test_CE_F1": f"{delta_cef1:.10f}",
        "report_csv": "",
        "clinical_csv": "",
    })
except Exception:
    pass

summary_csv = out_root / "summary.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["experiment", "test_BLEU_4", "test_CE_P", "test_CE_R", "test_CE_F1", "report_csv", "clinical_csv"])
    writer.writeheader()
    writer.writerows(rows)

summary_md = out_root / "summary.md"
with open(summary_md, "w") as f:
    f.write("# MIMIC to IU-Xray Zero-shot Evaluation\\n\\n")
    f.write("No IU-Xray training or fine-tuning is performed. MIMIC-trained S0/S2 checkpoints are evaluated directly on IU-Xray test set.\\n\\n")
    f.write("| experiment | BLEU-4 | CE-P | CE-R | CE-F1 |\\n")
    f.write("| --- | ---: | ---: | ---: | ---: |\\n")
    for row in rows:
        f.write(f"| {row['experiment']} | {row['test_BLEU_4']} | {row['test_CE_P']} | {row['test_CE_R']} | {row['test_CE_F1']} |\\n")

print(f"Wrote {summary_csv}")
print(f"Wrote {summary_md}")
PY

printf "\n============================================================\n"
printf ">>> Finished at %s\n" "$(date)"
printf ">>> log_file=%s\n" "$LOG_FILE"
printf ">>> summary_csv=%s\n" "$SUMMARY_CSV"
printf "============================================================\n"
