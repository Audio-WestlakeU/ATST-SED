# ATSTSEDv2

ATSTSEDv2 is the final reproducible package for the two-stage semi-supervised sound event detection system described in the paper draft ATST-SEDv2: Semi-Supervised SED with Embedding-Level Contrastive Loss and Conditional Mixup.

This folder collects the exact training code path, configs, scripts, and checkpoints needed to reproduce the final model family without depending on the rest of the historical experiment repository.

## What Is Included

- Stage 1 training code and script for the base checkpoint used by ATSTSEDv2.
- Stage 2 training code for the final two-views one-forward variant.
- Final evaluation script and pure-Python PSDS validation entry.
- Model, trainer, dataloader, loss, scheduler, and evaluation modules required by the final pipeline.
- The ATST pretrained checkpoint used by the model.
- The Stage 1 checkpoint used to initialize Stage 2.
- A final Stage 2 checkpoint for direct evaluation.

## Folder Layout

- trainer: Lightning trainers, CLI utilities, augmentation utilities, decoding utilities.
- model: ATSTSED sparse model and required submodules.
- data_loader: DESED datamodule, datasets, transforms, and IO helpers.
- loss: supervised, unsupervised, and contrastive objectives.
- scheduler: learning-rate and weight schedulers.
- evaluation: metric computation, PSDS helpers, and SEBB-related utilities.
- configs: training configs copied from the final experiment path.
- scripts: runnable Stage 1, Stage 2, lr-sweep, and test scripts using local relative paths.
- checkpoints: bundled Stage 1 and final Stage 2 checkpoints.
- pretrained_ckpts: bundled ATST-Frame pretrained checkpoint.

## Paper Summary

ATSTSEDv2 keeps the ATST-SED two-stage fine-tuning pipeline and redesigns only Stage 2. The main change is to combine decision-level pseudo-label learning with embedding-level contrastive learning in one objective. To make the pseudo-label loss and contrastive loss compatible, the method introduces conditional mixup:

- Vanilla mixup branch: when the interpolation ratio is near 0.5, both sources are treated as semantically active.
- Perturbation mixup branch: when the primary source dominates, the added sample is treated as perturbation.

The resulting system uses Mean Teacher style pseudo labels together with BYOL-style embedding alignment during Stage 2 fine-tuning. According to the paper draft, the final system reaches 0.607 PSDS1 and 0.818 PSDS2 on DESED.

## Environment

Recommended environment:

- Linux
- CUDA-enabled PyTorch
- Python 3.10 or 3.11

Install dependencies with:

```bash
cd ATSTSEDv2
pip install -r requirements.txt
```

If a package name differs on your platform, keep the code unchanged and install the equivalent wheel manually.

## Dataset Preparation

This package does not duplicate the DESED audio itself. The dataset paths are still defined in configs/dataset/desed+id.yaml.

Before running training or evaluation, update the following entries in configs/dataset/desed+id.yaml to match your machine:

- strong_folder
- train_synth_folder
- train_weak_folder
- unlabeled_id_folder
- unlabeled_od_folder
- test_folder
- eval_folder
- val_unlabeled_od_folder
- all duration and TSV paths if they differ locally

The TSV metadata files referenced under configs/dataset/desed_tsv are already included in this package.

## Reproduce The Final Model

### 1. Train Stage 1

This produces the base checkpoint used to initialize Stage 2:

```bash
cd ATSTSEDv2
bash scripts/stage_1.sh 0,1
```

The bundled reference Stage 1 checkpoint is:

```text
checkpoints/stage1_epoch199.ckpt
```

### 2. Train Stage 2 Final Model

The direct final recipe bundled from the current final training path is:

```bash
cd ATSTSEDv2
bash scripts/stage_2viewsonefoward.sh 0,1
```

The Stage 2 script uses the local Stage 1 checkpoint automatically.

If you want the lr scheduling sweep variant associated with the bundled final evaluation checkpoint, run:

```bash
cd ATSTSEDv2
bash scripts/stage_2viewsonefoward_lr_sweep.sh 0,1 0.5 0.999
```

### 3. Evaluate The Bundled Final Checkpoint

Quick evaluation through the Lightning test entry:

```bash
cd ATSTSEDv2
bash scripts/test.sh 0,1
```

Pure-Python validation on the copied final checkpoint:

```bash
cd ATSTSEDv2
python inference/validate_ckpt_psds.py
```

The bundled final Stage 2 checkpoint is:

```text
checkpoints/stage2_final_epoch479_metric0.5821.ckpt
```

## Important Notes

- The model code expects pretrained_ckpts/atstframe_as2M.ckpt to exist inside this folder. It is already bundled.
- Stage 2 training initializes from the bundled Stage 1 checkpoint by default.
- For strict reproducibility, keep batch size, GPU count, and config overrides consistent with the provided scripts.
- The current package is intended to be self-contained at the code level. External reproducibility still depends on matching DESED data layout, CUDA stack, and Python package versions.

## Recommended Entry Points

- Stage 1 training: scripts/stage_1.sh
- Final Stage 2 training: scripts/stage_2viewsonefoward.sh
- Stage 2 lr sweep: scripts/stage_2viewsonefoward_lr_sweep.sh
- Final checkpoint evaluation: scripts/test.sh
- Pure-Python PSDS validation: inference/validate_ckpt_psds.py