# Experiment Tracking

## Naming Rule

Use explicit task prefixes so the directory itself tells you the method family:

- `distill_prompt_*` : Stage-I gated prompt
- `ccli_stage2_*` : Stage-II original CCLI
- `region_ccli_stage2_*` : Stage-II region-aware counterfactual normalizer
- `explicit_cpmi_stage2_*` : Stage-II CCLI + Explicit CPMI
- `ccd_g*` : inference-only CCD decoding sweep

## What To Keep

For every experiment, keep three things:

1. the launch script name
2. the result directory
3. the final summary CSV

Recommended quick note format:

```text
script: mimic_train_ccli_stage2.sh
result: results/mimic_cxr/train/ccli_stage2_bleu_0.1214
key metric: test_BLEU_4 = 0.1214
note: current best CCLI-only
```

## Auto Summary

After several runs, rebuild the global summary tables with:

```bash
python scripts/collect_experiment_summary.py
```

This generates:

- `results/mimic_cxr/experiment_summary_train.csv`
- `results/mimic_cxr/experiment_summary_eval.csv`

## Recommended Comparison Protocol

For training experiments:

1. compare `results/mimic_cxr/experiment_summary_train.csv`
2. sort by `test_BLEU_4`
3. inspect the paired `val_BLEU_4`, `test_CIDEr`, and `best_model_from`

For test-time decoding experiments:

1. compare `results/mimic_cxr/experiment_summary_eval.csv`
2. sort by `test_BLEU_4` or `test_CE_F1`
3. keep a note whether the run is `val-selected` or `test-selected`
|
python scripts/collect_experiment_summary.py
