# Clean Ablation Supplements

This workspace reruns the polluted-risk ablation rows with the clean
`vlci_ccli_clean` model path.

The original E4/E5 rows already used `vlci_ccli_clean`; the main rows that need
clean replacement are:

- `baseline + A`
- `baseline + B`
- `baseline + AB`

Result root:

- `results/mimic_cxr/clean_ablations_20260518`

Batch run:

```bash
nohup bash experiments/mimic_ablation_results/legacy_scripts/clean_ablations_20260518/scripts/run_all_clean_ablations.sh &
```

Optional multi-seed run:

```bash
SEEDS="10200 20200 30200" nohup bash experiments/mimic_ablation_results/legacy_scripts/clean_ablations_20260518/scripts/run_all_clean_ablations.sh &
```

