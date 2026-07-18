# Ablation Plan

## Goal

Validate the three retained innovation modules with a combination-style ablation matrix.

## Three Retained Modules

### A. Anatomy-Aware Visual Alignment

Module:

- `AnatomicalAdapter`

### B. Dynamic Medical Prompting

Modules:

- `AnatomyPathologyMemoryPrompt`
- `confidence-gated pathology prompt`
- `normal-aware suppression`

### C. Counterfactual Causal Logit Intervention

Modules:

- `CounterfactualNormalizer`
- `CounterfactualLogitIntervention`

## Important Dependency Rule

`C` depends on `B` in the current implementation, because CCLI needs pathology
logits from the memory prompt path.

This means:

- `baseline + C` is **invalid**
- `baseline + AC` is **invalid**

The valid matrix is:

1. `baseline`
2. `baseline + A`
3. `baseline + B`
4. `baseline + AB`
5. `baseline + BC`
6. `baseline + ABC`

## Baseline

Reference only unless rerun is needed.

- Directory:
  - `results/mimic_cxr/train/baseline_potential2_20260424_113736`
- Role:
  - reproduced baseline around `BLEU_4 ≈ 0.118`

## Valid Combination Runs

### E1. baseline + A

Enable only:

- `AnatomicalAdapter`

Disable:

- `memory_prompt`
- `counterfactual_intervention`

Script:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_e1_baseline_plus_a.sh`

Question:

- Does anatomy-aware region alignment alone improve over the plain baseline?

### E2. baseline + B

Enable:

- `memory_prompt`
- `confidence-gated pathology prompt`
- `normal-aware suppression`

Disable:

- `AnatomicalAdapter`
- `counterfactual_intervention`

Script:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_e2_baseline_plus_b.sh`

Question:

- Can dynamic medical prompting alone improve over the plain baseline without anatomy routing?

### E3. baseline + AB

Enable:

- `AnatomicalAdapter`
- `memory_prompt`
- `confidence-gated pathology prompt`
- `normal-aware suppression`

Disable:

- `counterfactual_intervention`

Script:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_e3_baseline_plus_ab.sh`

Question:

- Does the full Stage-I framework improve over `A` and `B` individually?

### E4. baseline + BC

Enable:

- `memory_prompt`
- `confidence-gated pathology prompt`
- `normal-aware suppression`
- `counterfactual_intervention`

Disable:

- `AnatomicalAdapter`

Use the clean model path:

- `vlci_ccli_clean`

Script:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_e4_baseline_plus_bc.sh`

Question:

- If anatomy alignment is removed, can prompt-driven pathology logits still support useful CCLI gains?

### E5. baseline + ABC

Enable:

- `AnatomicalAdapter`
- `memory_prompt`
- `confidence-gated pathology prompt`
- `normal-aware suppression`
- `counterfactual_intervention`

Use the clean model path:

- `vlci_ccli_clean`

Script:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_e5_baseline_plus_abc.sh`

Question:

- Does the full clean pipeline reproduce the known strongest result around `BLEU_4 ≈ 0.1214`?

## Start Checkpoints

### Fair comparison for E1, E2, E3

These three must all start from the same reproduced baseline checkpoint:

- `results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth`

### Fair comparison for E4 and E5

These two include CCLI and should also start from the same baseline checkpoint
to isolate the contribution of `A` inside the full stack:

- `results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth`

This is a stricter module-combination ablation than the historical two-stage
protocol. You can later report the historical Stage-II result separately.

## Main Comparison Table

Recommended rows:

1. `baseline`
2. `baseline + A`
3. `baseline + B`
4. `baseline + AB`
5. `baseline + BC`
6. `baseline + ABC`

Recommended metrics:

- `test_BLEU_4`
- `test_CIDEr`
- `test_ROUGE_L`
- `test_CE_F1` if evaluated offline

## Result Layout

All new runs go under:

- `results/mimic_cxr/ablations_20260515`

Recommended layout:

- `results/mimic_cxr/ablations_20260515/E1_baseline_plus_A/...`
- `results/mimic_cxr/ablations_20260515/E2_baseline_plus_B/...`
- `results/mimic_cxr/ablations_20260515/E3_baseline_plus_AB/...`
- `results/mimic_cxr/ablations_20260515/E4_baseline_plus_BC/...`
- `results/mimic_cxr/ablations_20260515/E5_baseline_plus_ABC/...`

Each run writes:

- `<run_dir>/run.log`
- `<run_dir>/mimic_cxr.csv`

## Suggested Run Order

1. `E1 baseline + A`
2. `E2 baseline + B`
3. `E3 baseline + AB`
4. `E4 baseline + BC`
5. `E5 baseline + ABC`

## Batch Runner

You can run the whole valid matrix sequentially with:

- `experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_all_valid_combinations.sh`

Recommended launch:

```bash
nohup bash experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_all_valid_combinations.sh &
```

Optional overrides:

```bash
SEED=10200 EPOCHS=3 nohup bash experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_all_valid_combinations.sh &
```

You can also skip selected steps:

```bash
RUN_E4=0 RUN_E5=0 nohup bash experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts/run_all_valid_combinations.sh &
```

Progress tracking:

- master log:
  - `results/mimic_cxr/ablations_20260515/batches/<batch_name>/run.log`
- current step status:
  - `results/mimic_cxr/ablations_20260515/batches/<batch_name>/status.txt`
- compact summary:
  - `results/mimic_cxr/ablations_20260515/batches/<batch_name>/summary.txt`

To monitor progress:

```bash
tail -f results/mimic_cxr/ablations_20260515/batches/<batch_name>/run.log
cat results/mimic_cxr/ablations_20260515/batches/<batch_name>/status.txt
```

## Summary Template

```text
baseline                  BLEU_4 = ...
baseline + A              BLEU_4 = ...
baseline + B              BLEU_4 = ...
baseline + AB             BLEU_4 = ...
baseline + BC             BLEU_4 = ...
baseline + ABC            BLEU_4 = ...
```
