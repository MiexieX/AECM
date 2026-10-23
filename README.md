# AECM: Anatomy-Aware Evidence and Clinical Memory for Radiology Report Generation

Official implementation of **AECM**, the framework described in the IEEE Journal of
Biomedical and Health Informatics manuscript *"AECM: Anatomy-Aware Evidence and
Clinical Memory for Radiology Report Generation"*.

AECM targets a central failure mode of chest X-ray report generation: normal
expressions are frequent in report corpora, whereas abnormal findings may be rare,
small, or weakly expressed. A generator can therefore produce fluent,
dataset-plausible text that omits the evidence present in the current image.
AECM addresses this by organizing image patches into an anatomy-sensitive evidence
path before decoding and by admitting clinical cues only when the routed image
evidence supports them.

> **Repository name.** This repository is named **AECM** after the paper. It was
> previously named `scm_cc`; some historical file and class names retain the old
> identifier. The code path used by the AECM paper is the Stage-I **AER + CGCM**
> configuration described below. The earlier Stage-II counterfactual branch is
> retained for provenance but is not part of the paper; see
> [Legacy Stage-II code](#legacy-stage-ii-code-not-in-the-paper).

## Method at a Glance

AECM couples two modules:

1. **Anatomy-aware Evidence Routing (AER).** Patch tokens cross-attend to a fixed
   set of learnable chest-region prototypes. The module returns anatomy-adapted
   visual tokens `H_v` and attention weights `A_att`. No segmentation masks are
   used: the prototypes are latent soft anchors, not supervised localization maps.
   `A_att` is used only to compute the regional activation vector `rho`; it is not
   concatenated to the decoder.

2. **Confidence-Gated Clinical Memory (CGCM).** The routed evidence is read
   through region, pathology, and template memories. The module emits three compact
   prompt tokens: a regional prompt, a pathology prompt, and a template prompt. A
   sigmoid confidence gate and normal-aware suppression attenuate weakly supported
   pathology cues before decoding.

The two modules are ordered by design: memory cues are admitted only after routing
has localized the supporting evidence. At inference, AECM adds no module beyond the
three prompt tokens. The full model contains 104.47M parameters, of which 1.09M
(1.05%) are introduced by AER and CGCM over the 103.37M backbone.

```text
Chest X-ray
    |
    v
VLCI visual encoder  ->  patch tokens Z
    |
    v
AER:  Z x 11 chest-region prototypes  ->  H_v, A_att
    |
    v
CGCM:
    region prompt     P_reg  = rho * M_reg,       rho = mean(A_att)
    pathology prompt  P_path = confidence-gated read of M_path
    template prompt   P_temp = normalized read of M_temp
    |
    v
concat(H_v, beta_reg * P_reg, beta_path * P_path, beta_temp * P_temp)
    |
    v
Transformer decoder  ->  radiology report
```

The region prototypes cover the heart, left and right lung, mediastinum, hila,
pleura, diaphragm, ribs, spine, trachea, and aortic arch (11 regions). The
pathology memory has 14 CheXpert-style slots, and the template memory has 5 slots
corresponding to normal descriptions, negated findings, positive findings, device
statements, and comparisons.

### Key equations

```text
rho      = mean over patches and heads of A_att
P_reg    = rho M_reg
h_bar    = LN(mean(H_v))
s        = sigmoid(W_path h_bar)                 # pathology confidence
q        = sigmoid(W_temp h_bar)                 # template activation
g_k      = sigmoid((s_k - tau_p) / T_p)          # pathology gate
r_abn    = sigmoid((max_k s_k - tau_n) / T_n)    # normal-aware suppression
P_path   = (1/K) * sum_k g_k r_abn M_path,k
P_temp   = sum_j q_j M_temp,j / (sum_j q_j + eps)
H_tilde  = concat(H_v, beta_reg P_reg, beta_path P_path, beta_temp P_temp)
```

## Contributions

- **AER** preserves weak, region-specific abnormalities by organizing patch
  evidence through learnable chest-region prototypes, without segmentation masks.
- **CGCM** reads region, pathology, and template memories from the routed evidence
  and admits only visually supported abnormal cues to the decoder.
- On MIMIC-CXR and IU-Xray, AECM obtains the highest reported BLEU-1–BLEU-4 among
  the compared supervised methods. Under one controlled protocol on identical test
  studies, it improves significantly over the reproduced initialization on BLEU-4
  and clinical-efficacy F1.

## Paper-to-Code Map

| Paper component | Implementation |
| --- | --- |
| Anatomy-aware Evidence Routing (AER) | `modules/anatomical_adapter.py` (`AnatomicalAdapter`) |
| Confidence-Gated Clinical Memory (CGCM) | `modules/memory_prompt.py` (`AnatomyPathologyMemoryPrompt`) |
| AECM model path | `models/vlci_ccli_clean.py` (`VLCICCLIClean`) with `use_counterfactual_intervention=0` |
| VLCI backbone | `models/baseline.py`, `modules/modules4vlci.py`, `modules/modules4transformer.py`, `modules/modules4vlp.py` |
| AECM training entry | `mimic_train_gated_prompt_stage1.sh` |
| Report generation | `eval_dump_reports.py` |
| Lexical and clinical metrics | `metric/metrics.py`, `metric/clinical.py` |

File and class names such as `anatomical_adapter`, `memory_prompt`, and
`vlci_ccli_clean` are historical. They implement AER, CGCM, and the combined AECM
path, respectively.

## Repository Layout

```text
config/
  mimic_cxr/vlci.json                  MIMIC-CXR configuration
  iu_xray/vlci.json                    IU-Xray configuration
data/
  datadownloader.py                    Optional IU-Xray/MIMIC-CXR download helper
metric/                                BLEU, METEOR, ROUGE-L, CIDEr, CE metrics
models/
  baseline.py                          VLCI backbone base class
  vlci.py                              Model entry used by the configs
  vlci_ccli_clean.py                   AER + CGCM path; legacy CCLI branch off
  vlci_original.py                     Backbone reference implementation
modules/
  anatomical_adapter.py                AER
  memory_prompt.py                     CGCM
  modules4vlci.py                      Causal encoder/decoder components
  modules4transformer.py               Transformer layers
  modules4vlp.py                       Visual/text embeddings and masks
  beam_search.py                       Beam-search decoding
trainer/
  BaseTrainer.py                       Optimization, validation, checkpointing
  FinetuneTrainer.py                   Report-generation fine-tuning and losses
utils/                                 Data loading, tokenization, optimization
scripts/                               Analysis and table-building helpers
main.py                                Training and inference entry point
build_2d_memory.py                     Builds pathology pseudo labels + memory bank
build_template_labels.py               Builds report-template pseudo labels
eval_dump_reports.py                   Dumps generated/reference reports to CSV
eval_clinical.py                       Lexical + clinical evaluation
mimic_train_gated_prompt_stage1.sh     AECM training entry point
mimic_test.sh                          Legacy test wrapper (see Evaluation)
iu_train.sh                            Generic IU-Xray fine-tuning entry point
iu_test.sh                             IU-Xray test entry point
requirements.yaml                      Conda environment (Python 3.7, PyTorch 1.13.1)
```

The repository does not redistribute datasets, pretrained checkpoints, run outputs,
or paper-writing materials. Large files and local result directories are excluded
by `.gitignore`.

## Environment

```bash
conda env create -f requirements.yaml
conda activate mrg
```

The environment pins Python 3.7, PyTorch 1.13.1, CUDA 11.7, and torchvision 0.14.1.

Clinical-efficacy metrics use [F1CheXbert](https://github.com/rajpurkarlab/chexbert):

```bash
pip install f1chexbert
```

`metric/clinical.py` will use cached `bert-base-uncased` and `chexbert.pth`
artifacts when they are available and switch to offline mode in that case. Set
`CHEXBERT_CACHE_DIR` and `HUGGINGFACE_HUB_CACHE` if the default cache locations are
not used. If F1CheXbert is unavailable, the clinical metrics return `0.0`; lexical
metrics are unaffected.

## Data

### MIMIC-CXR

The paper uses the processed CMCRL/VLCI annotation:

| Split | Single-image report entries |
| --- | --- |
| Train | 270,790 |
| Validation | 2,130 |
| Test | 3,858 |

Obtain MIMIC-CXR through the official PhysioNet credentialed-access process and
comply with its data-use agreement. Place images and `annotation.json` under
`data/mimic_cxr/`, then update `data.image_dir` and `data.ann_path` in
`config/mimic_cxr/vlci.json`. The paths in the released configs are placeholders
from the development machine and must be replaced.

Preprocessing follows the paper:

- training images: resize to 256, random crop to 224 x 224, horizontal flip,
  ImageNet normalization;
- validation and test images: resize to 224 x 224, ImageNet normalization;
- reports: lower-cased, line breaks/repeated spaces or periods/numbered-list
  markers/special punctuation removed;
- vocabulary: minimum frequency 10 for MIMIC-CXR;
- maximum sequence length: 80 tokens.

### IU-Xray

Obtain IU-Xray from the source used by the VLCI/R2Gen setup, place it under
`data/iu_xray/`, and update `config/iu_xray/vlci.json`. The default IU-Xray
configuration uses a minimum frequency of 3 and a maximum sequence length of 60.

### Auxiliary label files

AECM uses two train-report-derived supervision files:

```text
data/mimic_cxr_pseudo_labels.json     # 14 CheXpert-style pathology labels
data/mimic_cxr_template_labels.json   # 5 report-template labels
```

They can be rebuilt from the training annotation:

```bash
python build_2d_memory.py \
    --ann_path data/mimic_cxr/annotation.json \
    --output_memory data/anatomy_pathology_polarity_memory.json \
    --output_labels data/mimic_cxr_pseudo_labels.json

python build_template_labels.py \
    --ann_path data/mimic_cxr/annotation.json \
    --output data/mimic_cxr_template_labels.json
```

The resulting `anatomy_pathology_polarity_memory.json` is a diagnostic artifact;
AECM itself uses learnable region/pathology/template memories. Training consumes
only the two label files. The pathology loss is applied only to samples with
available pseudo-labels.

## Checkpoints

Checkpoints are not redistributed. AECM is initialized from a locally reproduced
CMCRL/VLCI checkpoint and expects a checkpoint dictionary with a `state_dict` entry.

The paper uses two reference points:

| Reference | Meaning |
| --- | --- |
| Released CMCRL weights | External checkpoint re-evaluated on the paper's test split (0.113 BLEU-4) |
| Reproduced initialization | Released weights continued for six epochs under the original objective (0.118 BLEU-4, 0.377 CE-F1) |

AECM is trained from the reproduced initialization. If you start from the released
CMCRL weights instead, first reproduce the initialization with the original
objective; otherwise the comparison is not the one reported in the paper.

Training writes two checkpoints:

- `model_best_val.pth` — selected by maximum validation BLEU-4. **Use this
  checkpoint for all paper numbers.**
- `model_best.pth` — selected by test BLEU-4 for diagnostic analysis only.

The paper selects checkpoints by validation BLEU-4 and does not use test results
for model selection.

## Training AECM on MIMIC-CXR

### Paper settings

| Setting | Value |
| --- | --- |
| Initialization | Reproduced CMCRL/VLCI checkpoint |
| Epochs | 6 |
| Batch size | 16 |
| Data-loading workers | 12 |
| Optimizer | Adam with AMSGrad |
| Learning rate, encoder/decoder | 1e-6 |
| Learning rate, AER/CGCM | 5e-6 (implemented as 5x the base LR) |
| Weight decay | 5e-5 |
| LR scheduler | StepLR, step 50, gamma 0.1 (no decay within six epochs) |
| Maximum report length | 80 |
| Vocabulary minimum frequency | 10 |
| Region prototypes `M` | 11 |
| Pathology slots `K` | 14 |
| Template slots `J` | 5 |
| Loss weights `lambda_p`, `lambda_t`, `lambda_d` | 0.001, 0.001, 0.30 |
| Distillation temperature `T_d` | 2.0 |
| Pathology gate `tau_p`, `T_p` | 0.50, 0.12 |
| Normal suppression `tau_n`, `T_n` | 0.45, 0.10 |
| Parameters | 103.37M backbone / 104.47M AECM; AER + CGCM add 1.09M (1.05%) |
| Seed | 10200 |
| Hardware | 1 x NVIDIA RTX 5880 Ada (49 GB), approximately 17 h active training |

### Objective

```text
L = L_RRG + lambda_p L_path + lambda_t L_temp + lambda_d L_distill
```

- `L_RRG`: masked token-level negative log-likelihood.
- `L_path`: binary cross-entropy over 14 pathology labels, evaluated only on
  samples with available pseudo-labels.
- `L_temp`: binary cross-entropy over 5 report-template labels.
- `L_distill`: masked token-level KL divergence from a frozen teacher created
  before training, with AER and CGCM disabled.

### Command

```bash
BASELINE_CKPT=/absolute/path/to/reproduced_cmcrl_vlci_model_best.pth \
bash mimic_train_gated_prompt_stage1.sh
```

The wrapper sets innovation-only fine-tuning, enables AER and CGCM, keeps the
counterfactual branch off, and applies the auxiliary-loss and gate settings above.
It writes logs and checkpoints under `results/mimic_cxr/train/<task_name>/`.

> **Epoch count.** The released wrapper runs a 3-epoch continuation (the shorter
> run that produced the 0.1202 BLEU-4 checkpoint mentioned in the script). The
> paper reports the six-epoch protocol. Set `--epochs 6` in the wrapper, or pass
> `--epochs 6` to `main.py`, to match the paper setting.

The wrapper validates four times per epoch (`--val_intervals 4`) so that
`model_best_val.pth` can be selected within the short continuation.

## Evaluation

### Generate reports

For AECM, keep the legacy counterfactual branch disabled during generation:

```bash
BEST_MODEL=/absolute/path/to/model_best_val.pth
SAVE_DIR=results/mimic_cxr/test/aecm_test

python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --task_name aecm_test \
    --result_dir "$SAVE_DIR" \
    --load_model_path "$BEST_MODEL" \
    --cuda 0 \
    --seed 10200 \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method beam_search \
    --max_seq_length 80 \
    --suppress_UNK 1 \
    --split test \
    --use_counterfactual_intervention 0
```

`eval_dump_reports.py` writes `mimic_cxr_reports.csv` with paired reference and
generated reports.

> **Legacy wrapper caveat.** `mimic_test.sh` predates the AECM release and does
> not pass `--use_counterfactual_intervention 0`. Do not use it unchanged for
> AECM evaluation; add that flag to its `eval_dump_reports.py` call, or use the
> direct command above.

### Lexical and clinical metrics

```bash
python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name aecm_test \
    --result_dir "$SAVE_DIR" \
    --report_csv "$SAVE_DIR/mimic_cxr_reports.csv" \
    --cuda 0 \
    --seed 10200
```

This computes BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, and micro-averaged CE-P,
CE-R, and CE-F1. Clinical scores are produced by F1CheXbert/CheXpert-style report
labeling and are automatic consistency measures, not radiologist-validated
diagnostic accuracy.

Decoding follows the paper: beam size 3, trigram blocking, and an 80-token limit.

### Per-disease analysis

`scripts/build_per_disease_chexpert.py` computes per-disease CheXpert agreement
and clinical-efficacy changes from a collected experiment summary. The paper's
per-disease F1/recall analysis is reported in Fig. 4, and label-agreement accuracy
is reported in Fig. 5.

## Reported Results

All MIMIC-CXR rows are evaluated on the identical 3,858-study test set with the
same reference reports, preprocessing, and metric implementation, using
validation-selected checkpoints.

### Table I: Comparison with supervised RRG systems on MIMIC-CXR

| Method | B-1 | B-2 | B-3 | B-4 | R-L | METEOR | CE-P | CE-R | CE-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2Gen | 0.353 | 0.218 | 0.145 | 0.103 | 0.277 | 0.142 | 0.333 | 0.273 | 0.276 |
| R2GenCMN | 0.356 | 0.219 | 0.147 | 0.105 | 0.278 | 0.141 | 0.334 | 0.275 | 0.278 |
| CMCL | 0.334 | 0.217 | 0.140 | 0.097 | 0.281 | 0.133 | – | – | – |
| PPKED | 0.360 | 0.224 | 0.149 | 0.106 | 0.284 | 0.149 | – | – | – |
| CA | 0.350 | 0.219 | 0.152 | 0.109 | 0.283 | 0.151 | 0.352 | 0.298 | 0.303 |
| AlignTransformer | 0.378 | 0.235 | 0.156 | 0.112 | 0.283 | 0.158 | – | – | – |
| M2TR | 0.378 | 0.232 | 0.154 | 0.107 | 0.272 | 0.145 | 0.240 | 0.428 | 0.308 |
| MGSK | 0.363 | 0.228 | 0.156 | 0.115 | 0.284 | – | 0.458 | 0.348 | 0.371 |
| RAMT | 0.362 | 0.229 | 0.157 | 0.113 | 0.284 | 0.153 | 0.380 | 0.342 | 0.335 |
| MMTN | 0.379 | 0.238 | 0.159 | 0.116 | 0.283 | 0.161 | – | – | – |
| DCL | – | – | – | 0.109 | 0.284 | 0.150 | 0.471 | 0.352 | 0.373 |
| CAMANet | 0.374 | 0.230 | 0.155 | 0.112 | 0.279 | 0.145 | 0.483 | 0.323 | 0.387 |
| CMCRL | 0.400 | 0.245 | 0.165 | 0.119 | 0.280 | 0.150 | 0.489 | 0.340 | 0.401 |
| **AECM** | **0.406** | **0.251** | **0.169** | **0.122** | 0.283 | 0.154 | 0.463 | 0.342 | 0.393 |

AECM obtains the highest B-1–B-4 among the compared supervised systems. CMCRL
remains highest on CE-P and CE-F1; MMTN remains highest on METEOR. Values other
than AECM are quoted from the original papers. The AECM row is obtained by us on
the paper's test split with the shared label state. Clinical-efficacy values from
the literature use a CheXpert-style label extractor; residual differences from our
pipeline are confined to preprocessing and extractor implementation.

### Table II: Comparison with supervised RRG systems on IU-Xray

| Method | B-1 | B-2 | B-3 | B-4 | R-L | METEOR |
| --- | --- | --- | --- | --- | --- | --- |
| R2Gen | 0.470 | 0.304 | 0.219 | 0.165 | 0.371 | 0.187 |
| PPKED | 0.483 | 0.315 | 0.224 | 0.168 | 0.376 | 0.190 |
| CA | 0.492 | 0.314 | 0.222 | 0.169 | 0.381 | 0.193 |
| AlignTransformer | 0.484 | 0.313 | 0.225 | 0.173 | 0.379 | 0.204 |
| MGSK | 0.496 | 0.327 | 0.238 | 0.178 | 0.381 | – |
| RAMT | 0.482 | 0.310 | 0.221 | 0.165 | 0.377 | 0.195 |
| MMTN | 0.486 | 0.321 | 0.232 | 0.175 | 0.375 | – |
| DCL | – | – | – | 0.163 | 0.383 | 0.193 |
| CMCRL | 0.505 | 0.334 | 0.245 | 0.190 | 0.394 | 0.210 |
| **AECM** | **0.508** | **0.338** | **0.250** | **0.195** | **0.401** | 0.206 |

AECM improves B-1–B-4 and ROUGE-L over the reported CMCRL values, while CMCRL
remains higher on METEOR (0.210 versus 0.206). Clinical-efficacy scores are
omitted because the quoted predictions do not support a common label-based
evaluation.

### Table III: Controlled comparison on MIMIC-CXR

| Role | Model | B-1 | B-2 | B-3 | B-4 | METEOR | R-L | CIDEr | CE-P | CE-R | CE-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| External checkpoint | Released CMCRL weights | 0.385 | 0.235 | 0.158 | 0.113 | 0.144 | 0.276 | 0.174 | 0.433 | 0.271 | 0.334 |
| Reproduced initialization | CMCRL/VLCI reproduction | 0.399 | 0.244 | 0.164 | 0.118 | 0.152 | 0.281 | 0.187 | 0.461 | 0.319 | 0.377 |
| **AECM (ours)** | **AER + CGCM** | **0.406** | **0.251** | **0.169** | **0.122** | **0.154** | **0.283** | **0.194** | **0.463** | **0.342** | **0.393** |

The released CMCRL checkpoint yields 0.113 BLEU-4 on our test split; the original
publication reports 0.119. The gap is consistent with the released weights being
an intermediate checkpoint rather than the one behind the reported value, so the
two CMCRL rows are not expected to coincide.

### Table IV: Paired uncertainty analysis on MIMIC-CXR

Differences are computed on the identical 3,858-study test set over 10,000 paired
bootstrap resamples. Intervals are 95% confidence intervals; positive values favor
the second method; p-values are Holm-adjusted within each metric.

| Comparison | Delta B-4 | Delta CIDEr | Delta CE-F1 |
| --- | --- | --- | --- |
| Initialization -> AECM | +0.0034 [+0.0011, +0.0054]** | +0.0074 [-0.0033, +0.0184] n.s. | +0.0155 [+0.0088, +0.0221]*** |
| + AER -> AECM | +0.0020 [+0.0006, +0.0034]* | +0.0060 [-0.0018, +0.0142] n.s. | +0.0040 [-0.0010, +0.0090] n.s. |
| + CGCM -> AECM | +0.0027 [+0.0003, +0.0046]* | +0.0108 [+0.0001, +0.0218] n.s. | +0.0173 [+0.0108, +0.0237]*** |
| External CMCRL -> AECM | +0.0085 [+0.0057, +0.0109]*** | +0.0200 [+0.0058, +0.0339]** | +0.0591 [+0.0500, +0.0680]*** |

\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001; n.s., not significant after Holm
adjustment. AECM improves significantly over the external CMCRL checkpoint on
BLEU-4, CIDEr, and CE-F1, with a 7.9% relative BLEU-4 gain (0.113 to 0.122).

### Table V: Component ablation on MIMIC-CXR

| Configuration | AER | CGCM | B-1 | B-2 | B-3 | B-4 | METEOR | CIDEr | CE-P | CE-R | CE-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | – | – | 0.399 | 0.244 | 0.164 | 0.118 | 0.152 | 0.187 | 0.461 | 0.319 | 0.377 |
| + AER | yes | – | 0.404 | 0.248 | 0.167 | 0.120 | 0.153 | 0.188 | 0.463 | 0.335 | 0.389 |
| + CGCM | – | yes | 0.400 | 0.245 | 0.165 | 0.119 | 0.153 | 0.183 | 0.457 | 0.318 | 0.375 |
| **AECM** | **yes** | **yes** | **0.406** | **0.251** | **0.169** | **0.122** | **0.154** | **0.194** | **0.463** | **0.342** | **0.393** |

All rows start from the same reproduced CMCRL/VLCI checkpoint and are trained for
six epochs with the same optimizer, learning-rate schedule, and data pipeline.
AER alone supplies the primary gain. CGCM alone does not improve over the base
configuration, which is consistent with the ordering of the modules: a global
pathology score can nominate a finding without indicating where in the image the
finding is supported, so the memory prompts become useful once routing has
localized the evidence.

### Per-disease clinical efficacy

On the MIMIC-CXR test set, nine of thirteen CheXpert categories improve in F1 and
ten improve in recall over the reproduced baseline. The largest recall gains are on
pleural effusion (+0.060), atelectasis (+0.059), and cardiomegaly (+0.032).
Pneumonia and lung opacity show small F1 decreases (-0.020 and -0.006). On the
abnormal-finding subset (excluding No Finding), F1 increases from 0.395 to 0.411
and recall from 0.311 to 0.335.

## IU-Xray Transfer Protocol

The paper uses a lightweight frozen-backbone protocol on IU-Xray:

1. Start from the released VLCI checkpoint already fine-tuned on IU-Xray.
2. Update only AER and CGCM.
3. Train for 50 epochs with a `5e-7` learning rate for the trainable modules.
4. Keep the remaining hyperparameters unchanged.

The repository's `iu_train.sh` is a generic full fine-tuning entry point; it does
not configure the frozen-backbone transfer setting by itself. To reproduce the
paper protocol, use `main.py` with `--train_only_innovations 1` and set the base
learning rate so that the innovation group receives `5e-7` (this code applies a 5x
multiplier to AER/CGCM parameters in `BaseTrainer.py`). Use the same auxiliary-loss
and gate settings as MIMIC-CXR. The IU-Xray run uses `StepLR` with step 50 and
decay 0.1.

For evaluation:

```bash
BEST_MODEL=/absolute/path/to/iu_xray_aecm_model_best_val.pth \
bash iu_test.sh
```

Update `iu_test.sh` so that `BEST_MODEL` points to the validation-selected
checkpoint and the report CSV path matches the task name. As with MIMIC-CXR, keep
`--use_counterfactual_intervention 0` when the legacy branch is present in the
checkpoint path.

## Reproducibility Notes

- **Match the initialization.** AECM's controlled gain is measured on top of the
  reproduced CMCRL/VLCI checkpoint, which has already absorbed the additional
  training budget. Training from the released CMCRL checkpoint directly is a
  different experiment.
- **Use validation-selected checkpoints.** Select `model_best_val.pth` by maximum
  validation BLEU-4; do not select on the test split.
- **Keep the test set fixed.** All reported MIMIC-CXR comparisons use the same
  3,858 studies, the same reference reports, and one metric implementation.
- **Keep the counterfactual branch off.** AECM uses only AER and CGCM. The
  `use_counterfactual_intervention` default inherited from the historical model
  path is `1`, so set it explicitly to `0` for both training and generation.
- **Do not tune on the test set.** Decoding settings and operating points should be
  selected on validation data and evaluated once on the test set.
- **Clinical metrics are automatic.** CE-P, CE-R, and CE-F1 depend on the label
  extractor and indicate clinical consistency, not radiologist-validated
  diagnostic accuracy.

## Legacy Stage-II Code (Not in the Paper)

This repository previously hosted a Stage-II line of work called SCM-CC, which
added Counterfactual Causal Logit Intervention (CCLI) and optional clinical
contrastive decoding (CCD). Those components are not part of AECM and are not
reported in the AECM paper. They are retained only for provenance:

- `mimic_train_ccli_stage2.sh`
- `mimic_train_ccli_stage2_clean.sh`
- `mimic_test_clinical.sh`
- `mimic_test_ccd_sweep.sh`
- `modules/counterfactual_logit_intervention.py`
- CCLI-related branches inside `models/vlci_ccli_clean.py`

For AECM, ignore these files and keep `use_counterfactual_intervention=0` and
`ccd_gamma=0`. The AECM results do not depend on Stage-II training.

## Citation

If you use this repository, please cite the AECM manuscript:

```bibtex
@article{wang2026aecm,
  title   = {AECM: Anatomy-Aware Evidence and Clinical Memory for Radiology Report Generation},
  author  = {Wang, Dandan and Sun, Jiaxiang and Xu, Honghui and Chen, Hangwei and
             Luo, Jian and Guo, Wenping and Zhao, Shuchang and Zhang, Shiqing},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Manuscript}
}
```

## Acknowledgments

This codebase builds on the CMCRL/VLCI implementation and the R2Gen ecosystem.
Please follow the licenses and dataset access requirements of all upstream
projects and data providers. MIMIC-CXR and IU-Xray are publicly available, but
MIMIC-CXR requires credentialed access under its data-use agreement.
