#!/bin/bash

set -e
set -u

if [ -f current_5seed_supp_root.txt ]; then
  RESULT_ROOT="${RESULT_ROOT:-$(cat current_5seed_supp_root.txt)}"
else
  RESULT_ROOT="${RESULT_ROOT:-}"
fi

if [ -z "$RESULT_ROOT" ]; then
  echo "RESULT_ROOT is empty. Set RESULT_ROOT or run from a workspace with current_5seed_supp_root.txt."
  exit 1
fi

CUDA="${CUDA:-0}"
TRAIN_SUMMARY="${RESULT_ROOT}/new_multiseed_stage2/summary.csv"
OUT_ROOT="${RESULT_ROOT}/multiseed_clinical_eval_best_test_ckpt"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
SCORED_CSV="${OUT_ROOT}/summary_with_scores.csv"
LOG_FILE="${OUT_ROOT}/run.log"

mkdir -p "$OUT_ROOT"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [ ! -f "$TRAIN_SUMMARY" ]; then
  echo "Missing training summary: $TRAIN_SUMMARY"
  exit 1
fi

echo "seed,row_id,stage,checkpoint_type,run_dir,checkpoint,report_csv,clinical_csv,result_dir" > "$SUMMARY_CSV"

eval_report_csv_test() {
  local row_id="$1"
  local report_csv="$2"
  local out_dir="$3"
  local seed_value="$4"

  mkdir -p "$out_dir"
  python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "${row_id}_test" \
    --result_dir "$out_dir" \
    --report_csv "$report_csv" \
    --report_split "test" \
    --cuda "$CUDA" \
    --seed "$seed_value"
}

echo "============================================================"
echo ">>> Evaluating model_best.pth checkpoints for 5-seed stage-wise run"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> train_summary=${TRAIN_SUMMARY}"
echo ">>> out_root=${OUT_ROOT}"
echo "============================================================"

tail -n +2 "$TRAIN_SUMMARY" | while IFS=, read -r seed stage run_dir init_ckpt csv_path model_best_val model_best notes; do
  if [ -z "${seed:-}" ] || [ -z "${stage:-}" ]; then
    continue
  fi

  checkpoint="$model_best"
  if [ ! -f "$checkpoint" ]; then
    echo ">>> Missing model_best checkpoint, skip: seed=${seed} stage=${stage} checkpoint=${checkpoint}"
    continue
  fi

  row_id="MS${seed}_${stage}"
  result_dir="${OUT_ROOT}/${row_id}"
  report_csv="${result_dir}/mimic_cxr_reports.csv"
  clinical_csv="${result_dir}/test_metrics/mimic_cxr_clinical.csv"

  echo ""
  echo "============================================================"
  echo ">>> Best-test checkpoint eval ${row_id}"
  echo ">>> checkpoint=${checkpoint}"
  echo ">>> result_dir=${result_dir}"
  echo "============================================================"

  if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "$clinical_csv" ]; then
    echo ">>> Existing clinical CSV found, skip eval: ${clinical_csv}"
  else
    python eval_dump_reports.py \
      --c config/mimic_cxr/vlci.json \
      --model vlci_ccli_clean \
      --task_name "$row_id" \
      --result_dir "$result_dir" \
      --load_model_path "$checkpoint" \
      --cuda "$CUDA" \
      --seed "$seed" \
      --batch_size 16 \
      --beam_size 3 \
      --sample_method "beam_search" \
      --max_seq_length 80 \
      --suppress_UNK 1 \
      --split "test"

    eval_report_csv_test "$row_id" "$report_csv" "${result_dir}/test_metrics" "$seed"
  fi

  echo "${seed},${row_id},${stage},model_best,${run_dir},${checkpoint},${report_csv},${clinical_csv},${result_dir}" >> "$SUMMARY_CSV"
done

python - "$SUMMARY_CSV" "$SCORED_CSV" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []

with summary.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        clinical_csv = Path(row["clinical_csv"])
        metrics = {}
        if clinical_csv.exists():
            with clinical_csv.open("r", newline="", encoding="utf-8") as mf:
                metrics = next(csv.DictReader(mf))
        merged = dict(row)
        merged.update(metrics)
        rows.append(merged)

if rows:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    out.write_text("", encoding="utf-8")

print(out)
PY

echo ""
echo "============================================================"
echo ">>> Finished model_best.pth checkpoint evaluation"
echo ">>> summary=${SUMMARY_CSV}"
echo ">>> scored_summary=${SCORED_CSV}"
echo "============================================================"
