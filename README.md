# SCM-CC: Structured Clinical Memory and Bounded Counterfactual Correction for Radiology Report Generation

This repository contains the implementation of **SCM-CC**, a stage-wise framework for chest X-ray report generation. SCM-CC is designed for the clinically important setting where frequent normal descriptions can obscure localized or rare abnormal findings.

The method separates evidence construction from output correction:

1. **Stage I: clinical evidence routing.** An anatomy-aware adapter routes visual patch features through latent chest-region prototypes. Structured clinical memory then produces compact regional, pathology, and reporting-style prompts. Confidence gating and normal-aware suppression reduce unsupported pathology prompting.
2. **Stage II: bounded counterfactual correction.** Counterfactual Causal Logit Intervention (CCLI) compares factual decoding logits with logits under a soft normal-reference context. The correction is bounded and activated only when pathology confidence supports it.
3. **Matched continuation control.** S1 (no CCLI) and S2 (with CCLI) start from the *same* Stage-I checkpoint and use the same optimization setup. This isolates the effect of logit correction from ordinary additional fine-tuning.

The repository is a clean code release: datasets, pretrained checkpoints, run outputs, paper-writing materials, and ablation records are intentionally not included.

## Method at a Glance

```text
Chest X-ray
    |
    +--> Stage I: anatomy prototypes -> structured clinical memory -> factual decoder context
    |                                                         |
    |                                                         +--> S0 checkpoint
    |
    +--> Stage II matched continuations from the same S0 checkpoint
            S1: no-CCLI continuation (control)
            S2: CCLI continuation (SCM-CC)
                    |
                    +--> optional clinical contrastive decoding (CCD)
```

The anatomical prototypes are latent soft anchors, not segmentation masks. CCLI is a controlled model-level factual-versus-normal logit contrast; it does not claim patient-level causal identification.

## Repository Layout

```text
config/                         MIMIC-CXR and IU-Xray configurations
data/                           Small public helper assets and data downloader
metric/                         BLEU, METEOR, ROUGE-L, CIDEr, and clinical metrics
models/                         VLCI backbone and SCM-CC model variants
modules/                        Anatomy adapter, memory prompt, and CCLI modules
trainer/                        Pretraining and fine-tuning trainers
utils/                          Data loading, tokenization, optimization utilities
main.py                         Training and validation entry point
eval_dump_reports.py            Generate reports and lexical metrics
eval_clinical.py                CE-P / CE-R / CE-F1 evaluation with F1CheXbert
mimic_train_gated_prompt_stage1.sh
mimic_train_ccli_stage2_clean.sh
mimic_test.sh
mimic_test_clinical.sh
iu_train.sh
iu_test.sh
```

## Environment

```bash
conda env create -f requirements.yaml
conda activate mrg
```

The clinical evaluation uses [F1CheXbert](https://github.com/rajpurkarlab/chexbert). Install it in the same environment and ensure its model assets are available before running `eval_clinical.py`.

```bash
pip install f1chexbert
```

## Data and Checkpoints

This release does not redistribute MIMIC-CXR, IU-Xray images, annotations, or model checkpoints.

- Obtain MIMIC-CXR through its official PhysioNet access process and comply with its data-use agreement.
- Obtain IU-Xray from the source used by the underlying VLCI/R2Gen setup.
- Place data under `data/mimic_cxr/` and `data/iu_xray/`, then update `image_dir` and `ann_path` in the corresponding JSON file under `config/`.
- Supply your own baseline and Stage-I checkpoints. Checkpoints are deliberately excluded by `.gitignore`.

For the Stage-I recipe, the following local supervision files are expected:

```text
data/mimic_cxr_pseudo_labels.json
data/mimic_cxr_template_labels.json
```

They can be produced from the local MIMIC-CXR training annotations with the data-preparation utilities in `scripts/`, or replaced with compatible files from your preprocessing pipeline.

## Reproduction Protocol

The paper's central MIMIC-CXR comparison is a stage-wise matched-continuation experiment:

| Run | Initialization | CCLI | Purpose |
| --- | --- | --- | --- |
| Base | pretrained VLCI checkpoint | no | initialization baseline |
| S0 | Base checkpoint | no | Stage-I clinical evidence routing |
| S1 | the selected S0 checkpoint | no | Stage-II matched continuation control |
| S2 | the same selected S0 checkpoint as S1 | yes | SCM-CC final model |

For a fair S1/S2 comparison, use the same random seed, epochs, batch size, learning rates, decoder settings, and the exact same `S0/model_best_val.pth` path for both continuations.

### 1. Stage I: Train Clinical Evidence Routing (S0)

Set the path to a pretrained MIMIC-CXR VLCI checkpoint and run:

```bash
BASELINE_CKPT=/absolute/path/to/baseline_model_best.pth \
bash mimic_train_gated_prompt_stage1.sh
```

The included Stage-I recipe uses innovation-only fine-tuning, batch size 16, Adam, encoder/decoder learning rate `1e-6`, weight decay `5e-5`, maximum report length 80, and a three-epoch continuation. The selected validation checkpoint is the S0 starting point for both Stage-II branches.

### 2. Stage II Control (S1): Continue Without CCLI

```bash
STAGE1_CKPT=/absolute/path/to/S0/model_best_val.pth \
USE_COUNTERFACTUAL_INTERVENTION=0 \
TASK_PREFIX=s1_no_ccli \
bash mimic_train_ccli_stage2_clean.sh
```

### 3. Stage II SCM-CC (S2): Continue With CCLI

```bash
STAGE1_CKPT=/absolute/path/to/S0/model_best_val.pth \
USE_COUNTERFACTUAL_INTERVENTION=1 \
TASK_PREFIX=s2_ccli \
bash mimic_train_ccli_stage2_clean.sh
```

The two commands above must use the same `STAGE1_CKPT`. The Stage-II recipe defaults to an anomaly threshold of `0.35`, anomaly temperature `0.10`, maximum CCLI strength `0.25`, and hidden dimension `16`.

### 4. Generate Test Reports and Lexical Scores

```bash
BEST_MODEL=/absolute/path/to/S2/model_best_val.pth \
TASK_NAME=s2_test \
bash mimic_test.sh
```

This uses beam search with beam size 3 and reports BLEU-1/2/3/4, METEOR, ROUGE-L, and CIDEr.

### 5. Clinical Evaluation and CCD

For the standard S2 operating point, first generate reports with `mimic_test.sh`, then evaluate the generated CSV:

```bash
python eval_clinical.py \
  --c config/mimic_cxr/vlci.json \
  --task_name s2_test \
  --result_dir results/mimic_cxr/test/s2_test \
  --report_csv results/mimic_cxr/test/s2_test/mimic_cxr_reports.csv \
  --cuda 0 --seed 10200
```

For the optional clinical contrastive decoding (CCD) operating point:

```bash
BEST_MODEL=/absolute/path/to/S2/model_best_val.pth \
CCD_GAMMA=0.1 \
TASK_NAME=s2_ccd_test \
bash mimic_test_clinical.sh
```

CCD is an inference-time operating-point analysis. Select `CCD_GAMMA` on the validation set only, then evaluate the selected value once on the test set. It should not be tuned on the test set.

### IU-Xray

After updating the paths in `config/iu_xray/vlci.json`, the base training and evaluation entry points are:

```bash
bash iu_train.sh
BEST_MODEL=/absolute/path/to/model_best_val.pth bash iu_test.sh
```

The default IU-Xray configuration uses batch size 16, beam size 3, maximum sequence length 60, Adam, and StepLR. The transfer setting in the paper uses lightweight innovation-only adaptation from an author-provided checkpoint rather than training the whole backbone from scratch.

## Evaluation

Lexical metrics are computed from generated reports with BLEU, METEOR, ROUGE-L, and CIDEr. Clinical metrics are micro-averaged CE-P, CE-R, and CE-F1 from the 14-observation F1CheXbert/CheXbert-style report labeling protocol implemented in `metric/clinical.py`.

Clinical-label metrics are automatic consistency checks, not radiologist assessment. Report metrics, model checkpoints, and operating-point choices should always be selected on the validation set before a final test evaluation.

## Reproducibility Notes

- Keep S1 and S2 matched: same S0 checkpoint, seed, number of epochs, batch size, optimizer, learning rate, and decoding configuration.
- Store generated reports together with the exact checkpoint path and command used for evaluation.
- The training scripts select `model_best_val.pth` by validation BLEU-4 unless you deliberately change the monitor setting.
- Absolute dataset and checkpoint paths in the example configs are machine-specific. Replace them before running.
- The repository omits large binary checkpoints and protected clinical data by design.

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@article{wang2026scmcc,
  title   = {SCM-CC: Structured Clinical Memory and Bounded Counterfactual Correction for Radiology Report Generation},
  author  = {Wang, Dandan and Sun, Jiaxiang and Guo, Wenping and Zhao, Shuchang and Zhang, Shiqing},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Acknowledgments

This codebase builds on the original CMCRL/VLCI implementation and the R2Gen ecosystem. Please follow the licenses and dataset access requirements of all upstream projects and data providers.
