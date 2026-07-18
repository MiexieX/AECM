# Stage-II Protocol Supplement

This workspace is for the protocol-level experiments that explain why the
historical clean CCLI result differs from the strict joint `ABC` ablation.

## Main Question

Does CCLI improve because of the counterfactual intervention itself, or because
the Stage-I checkpoint is simply trained for more steps?

## Runs

### S0. Clean Stage-I AB

Script:

- `scripts/run_s0_clean_stage1_ab.sh`

Purpose:

- optional clean re-run of Stage-I AB from the reproduced baseline checkpoint
- useful for multi-seed stability when the older `vlci` path is considered
  polluted

### S1. Stage-I -> continue without CCLI

Script:

- `scripts/run_s1_stage1_continue_no_ccli.sh`

Purpose:

- matched Stage-II control
- same Stage-I checkpoint, epoch count, optimizer settings, prompt losses, and
  distillation settings as clean Stage-II, but `use_counterfactual_intervention=0`
- uses the clean `vlci_ccli_clean` model path with CCLI disabled, not the older
  polluted `vlci` path

### S2. Stage-I -> CCLI

Script:

- `scripts/run_s2_stage1_plus_ccli.sh`

Purpose:

- Stage-II CCLI reproduction under the supplement result root
- multi-seed reproducibility line for the final system

### Batch Runner

Run all key experiments sequentially:

```bash
nohup bash experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_all_stage2_protocol.sh &
```

Useful overrides:

```bash
SEEDS="10200 20200 30200" EPOCHS=3 RUN_CLINICAL=1 RUN_S0=1 \
  nohup bash experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_all_stage2_protocol.sh &
```

Fully clean chained protocol:

```bash
RUN_S0=1 USE_S0_AS_STAGE1=1 SEEDS="10200" \
  nohup bash experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_all_stage2_protocol.sh &
```

Skip selected steps:

```bash
RUN_S1=0 RUN_CLINICAL=0 nohup bash experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_all_stage2_protocol.sh &
```

Result root:

- `results/mimic_cxr/stage2_protocol_20260518`
