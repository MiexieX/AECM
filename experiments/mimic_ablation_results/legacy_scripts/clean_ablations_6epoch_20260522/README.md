# Clean 6-Epoch Ablation Matrix

Purpose:

- test whether the 3-epoch strict ablation matrix under-trained the innovation
  modules
- rerun the valid clean combinations with `EPOCHS=6`
- use only the clean `vlci_ccli_clean` model path
- start all rows from the same reproduced baseline checkpoint

Rows:

- `A`
- `B`
- `AB`
- `BC`
- `ABC`

Invalid rows remain excluded:

- `C`
- `AC`

Run:

```bash
nohup bash experiments/mimic_ablation_results/legacy_scripts/clean_ablations_6epoch_20260522/scripts/run_all_clean_6epoch.sh &
```

Optional:

```bash
SEED=10200 EPOCHS=6 nohup bash experiments/mimic_ablation_results/legacy_scripts/clean_ablations_6epoch_20260522/scripts/run_all_clean_6epoch.sh &
```

Result root:

- `results/mimic_cxr/clean_ablations_6epoch_20260522`

