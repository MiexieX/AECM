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
TRAIN_TEXT_CSV="${RESULT_ROOT}/new_multiseed_stage2/training_log_best_test_scores.csv"
TRAIN_SUMMARY="${RESULT_ROOT}/new_multiseed_stage2/summary.csv"
OUT_ROOT="${RESULT_ROOT}/multiseed_best_test_clinical_matched"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
SCORED_CSV="${OUT_ROOT}/summary_with_clinical.csv"
FINAL_CSV="${RESULT_ROOT}/multiseed_final_tables/best_test_text_plus_clinical_scores.csv"
LOG_FILE="${OUT_ROOT}/run.log"

mkdir -p "$OUT_ROOT" "$(dirname "$FINAL_CSV")"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [ ! -f "$TRAIN_SUMMARY" ]; then
  echo "Missing training summary: $TRAIN_SUMMARY"
  exit 1
fi

if [ ! -f "$TRAIN_TEXT_CSV" ]; then
  echo "Missing training text score CSV: $TRAIN_TEXT_CSV"
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
echo ">>> Evaluating clinical metrics for model_best.pth checkpoints"
echo ">>> Uses matched train-time inference flags"
echo ">>> result_root=${RESULT_ROOT}"
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

  use_ccli=0
  if [ "$stage" = "S2" ]; then
    use_ccli=1
  fi

  row_id="MS${seed}_${stage}"
  result_dir="${OUT_ROOT}/${row_id}"
  report_csv="${result_dir}/mimic_cxr_reports.csv"
  clinical_csv="${result_dir}/test_metrics/mimic_cxr_clinical.csv"

  echo ""
  echo "============================================================"
  echo ">>> Matched best-test clinical eval ${row_id}"
  echo ">>> checkpoint=${checkpoint}"
  echo ">>> use_counterfactual_intervention=${use_ccli}"
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
      --suppress_UNK 0 \
      --split "test" \
      --use_anatomical_adapter 1 \
      --use_memory_prompt 1 \
      --use_counterfactual_intervention "$use_ccli" \
      --use_confidence_gated_pathology_prompt 1 \
      --pathology_gate_threshold 0.50 \
      --pathology_gate_temperature 0.12 \
      --use_normal_aware_suppression 1 \
      --normal_suppression_threshold 0.45 \
      --normal_suppression_temperature 0.10 \
      --ccli_anomaly_threshold 0.35 \
      --ccli_anomaly_temperature 0.10 \
      --ccli_max_alpha 0.25

    eval_report_csv_test "$row_id" "$report_csv" "${result_dir}/test_metrics" "$seed"
  fi

  echo "${seed},${row_id},${stage},model_best,${run_dir},${checkpoint},${report_csv},${clinical_csv},${result_dir}" >> "$SUMMARY_CSV"
done

python - "$SUMMARY_CSV" "$SCORED_CSV" "$TRAIN_TEXT_CSV" "$FINAL_CSV" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
scored_out = Path(sys.argv[2])
train_text_csv = Path(sys.argv[3])
final_out = Path(sys.argv[4])

clinical_rows = []
clinical_by_key = {}
with summary.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        clinical_csv = Path(row["clinical_csv"])
        metrics = {}
        if clinical_csv.exists():
            with clinical_csv.open("r", newline="", encoding="utf-8") as mf:
                metrics = next(csv.DictReader(mf))
        merged = dict(row)
        merged.update(metrics)
        clinical_rows.append(merged)
        clinical_by_key[(row["seed"], row["stage"])] = metrics

if clinical_rows:
    with scored_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clinical_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clinical_rows)
else:
    scored_out.write_text("", encoding="utf-8")

final_rows = []
with train_text_csv.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        metrics = clinical_by_key.get((row["seed"], row["stage"]), {})
        out = {
            "seed": row["seed"],
            "stage": row["stage"],
            "checkpoint": row["checkpoint"],
            "text_score_source": "training_log_best_model_from_test",
            "clinical_score_source": "matched_eval_dump_reports_model_best",
            "epoch": row.get("epoch", ""),
            "test_BLEU_1": row.get("test_BLEU_1", ""),
            "test_BLEU_2": row.get("test_BLEU_2", ""),
            "test_BLEU_3": row.get("test_BLEU_3", ""),
            "test_BLEU_4": row.get("test_BLEU_4", ""),
            "test_METEOR": row.get("test_METEOR", ""),
            "test_ROUGE_L": row.get("test_ROUGE_L", ""),
            "test_CIDEr": row.get("test_CIDEr", ""),
            "test_CE_P": metrics.get("test_CE_P", ""),
            "test_CE_R": metrics.get("test_CE_R", ""),
            "test_CE_F1": metrics.get("test_CE_F1", ""),
        }
        final_rows.append(out)

with final_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
    writer.writeheader()
    writer.writerows(final_rows)

print(scored_out)
print(final_out)
PY

echo ""
echo "============================================================"
echo ">>> Finished matched best-test clinical evaluation"
echo ">>> clinical_summary=${SCORED_CSV}"
echo ">>> final_table=${FINAL_CSV}"
echo "============================================================"
