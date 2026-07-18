#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
SWEEP_ROOT="${SWEEP_ROOT:-experiments/iu_ablation_results/runs/iu_recipe_sweep_${EXP_TIME}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-experiments/iu_ablation_results/summaries/recipe_sweep_${EXP_TIME}}"
LOG_FILE="${SWEEP_ROOT}/sweep.log"
STATUS_FILE="${SWEEP_ROOT}/status.txt"
MASTER_SUMMARY="${SWEEP_ROOT}/recipe_sweep_summary.csv"

BASE_SCRIPT="${BASE_SCRIPT:-experiments/iu_ablation_results/scripts/run_iu_from_pretrain_ablation.sh}"
CONFIG="${CONFIG:-config/iu_xray/vlci.json}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/home/sjx/work/project/CMCRL-main/pretrained/nano_current_checkpoint.pth}"
CUDA="${CUDA:-0}"
EPOCHS_LIST="${EPOCHS_LIST:-10}"
LR_LIST="${LR_LIST:-1e-5 7e-6 5e-6 3e-6 2e-6}"
SEED_LIST="${SEED_LIST:-10200 20200 30200}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-16}"
RUN_P2_ON_EACH="${RUN_P2_ON_EACH:-0}"
RUN_CCD_ON_EACH="${RUN_CCD_ON_EACH:-0}"
TARGET_BLEU4="${TARGET_BLEU4:-0.189340}"

mkdir -p "$SWEEP_ROOT" "$SUMMARY_OUT_DIR"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_RECIPE_SWEEP_LOGGED:-}" ]; then
  export CMCRL_RECIPE_SWEEP_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "recipe_id,seed,epochs,lr,batch_size,run_root,row_id,test_BLEU_4,test_CIDEr,clinical_CE_F1,checkpoint,notes" > "$MASTER_SUMMARY"

write_status() {
  cat > "$STATUS_FILE" <<EOF
timestamp=$(date +"%F %T")
current_recipe=$1
message=$2
sweep_root=${SWEEP_ROOT}
summary_out_dir=${SUMMARY_OUT_DIR}
config=${CONFIG}
pretrain_ckpt=${PRETRAIN_CKPT}
cuda=${CUDA}
epochs_list=${EPOCHS_LIST}
lr_list=${LR_LIST}
seed_list=${SEED_LIST}
batch_size_list=${BATCH_SIZE_LIST}
target_bleu4=${TARGET_BLEU4}
EOF
}

append_recipe_summary() {
  local recipe_id="$1"
  local seed="$2"
  local epochs="$3"
  local lr="$4"
  local batch_size="$5"
  local run_root="$6"

  python - "$MASTER_SUMMARY" "$recipe_id" "$seed" "$epochs" "$lr" "$batch_size" "$run_root" <<'PY'
import csv
import sys
from pathlib import Path

master, recipe_id, seed, epochs, lr, batch_size, run_root = sys.argv[1:]
run_root = Path(run_root)
summary = run_root / "summary.csv"
if not summary.exists():
    raise SystemExit(f"Missing run summary: {summary}")

with summary.open("r", newline="", encoding="utf-8") as f:
    for item in csv.DictReader(f):
        row_id = item.get("row_id", "")
        if row_id not in {"IU_P0_original_from_pretrain", "IU_P1_AB_from_pretrain", "IU_P2_ABC_from_pretrain", "IU_P2_CCD_selected"}:
            continue
        clinical_csv = Path(item.get("clinical_csv", ""))
        metrics = {}
        if clinical_csv.exists():
            with clinical_csv.open("r", newline="", encoding="utf-8") as cf:
                metrics = next(csv.DictReader(cf), {})
        out = {
            "recipe_id": recipe_id,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "run_root": str(run_root),
            "row_id": row_id,
            "test_BLEU_4": metrics.get("test_BLEU_4", ""),
            "test_CIDEr": metrics.get("test_CIDEr", ""),
            "clinical_CE_F1": metrics.get("test_CE_F1", ""),
            "checkpoint": item.get("checkpoint", ""),
            "notes": item.get("notes", ""),
        }
        with open(master, "a", newline="", encoding="utf-8") as mf:
            writer = csv.DictWriter(mf, fieldnames=list(out.keys()))
            writer.writerow(out)
PY
}

rank_summary() {
  python - "$MASTER_SUMMARY" "$SWEEP_ROOT/recipe_sweep_ranked.csv" "$TARGET_BLEU4" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
target = float(sys.argv[3])
with src.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def fnum(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return -1.0

rows.sort(key=lambda r: (fnum(r, "test_BLEU_4"), fnum(r, "clinical_CE_F1"), fnum(r, "test_CIDEr")), reverse=True)
fields = list(rows[0].keys()) + ["bleu4_gap_to_target"] if rows else []
with dst.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        row = dict(row)
        score = fnum(row, "test_BLEU_4")
        row["bleu4_gap_to_target"] = f"{score - target:.6f}" if score >= 0 else ""
        writer.writerow(row)

print(f"Updated ranked summary: {dst}")
if rows:
    best = rows[0]
    print("Best row so far:")
    print(f"  row_id={best.get('row_id')}")
    print(f"  recipe_id={best.get('recipe_id')}")
    print(f"  BLEU_4={best.get('test_BLEU_4')}")
    print(f"  CE_F1={best.get('clinical_CE_F1')}")
    print(f"  gap_to_target={fnum(best, 'test_BLEU_4') - target:.6f}")
PY
}

echo "============================================================"
echo ">>> IU-Xray recipe sweep started at $(date)"
echo ">>> sweep_root=${SWEEP_ROOT}"
echo ">>> target_bleu4=${TARGET_BLEU4}"
echo "============================================================"

write_status "INIT" "starting"

for epochs in ${EPOCHS_LIST}; do
  for lr in ${LR_LIST}; do
    for seed in ${SEED_LIST}; do
      for batch_size in ${BATCH_SIZE_LIST}; do
        lr_tag=${lr//./p}
        lr_tag=${lr_tag//-e/m}
        recipe_id="ep${epochs}_lr${lr_tag}_bs${batch_size}_seed${seed}"
        run_root="${SWEEP_ROOT}/${recipe_id}"

        write_status "$recipe_id" "running"
        echo ""
        echo "------------------------------------------------------------"
        echo ">>> Running recipe: ${recipe_id}"
        echo ">>> run_root=${run_root}"
        echo "------------------------------------------------------------"

        RESULT_ROOT="$run_root" \
        SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR}/${recipe_id}" \
        CONFIG="$CONFIG" \
        PRETRAIN_CKPT="$PRETRAIN_CKPT" \
        CUDA="$CUDA" \
        SEED="$seed" \
        EPOCHS="$epochs" \
        LR="$lr" \
        BATCH_SIZE="$batch_size" \
        RUN_P0=1 \
        RUN_P1=1 \
        RUN_P2="$RUN_P2_ON_EACH" \
        RUN_CCD="$RUN_CCD_ON_EACH" \
        bash "$BASE_SCRIPT"

        append_recipe_summary "$recipe_id" "$seed" "$epochs" "$lr" "$batch_size" "$run_root"
        rank_summary
      done
    done
  done
done

write_status "DONE" "all_completed"

echo ""
echo "============================================================"
echo ">>> IU-Xray recipe sweep finished at $(date)"
echo ">>> master_summary=${MASTER_SUMMARY}"
echo ">>> ranked_summary=${SWEEP_ROOT}/recipe_sweep_ranked.csv"
echo "============================================================"
