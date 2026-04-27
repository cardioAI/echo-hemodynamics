# Echo-Hemodynamics

Non-invasive prediction of hemodynamic parameters from multi-view echocardiography using a Vision Transformer with multistage attention.

## Overview

Right heart catheterization (RHC) remains the gold standard for hemodynamic assessment but is invasive, costly, and carries procedural risk. This project trains a deep learning model to predict nine hemodynamic parameters directly from standard echocardiographic views, potentially enabling non-invasive hemodynamic screening.

The model uses a pre-trained ViT-Base backbone (`google/vit-base-patch16-224`) with temporal and cross-view fusion attention modules, trained with progressive unfreezing and discriminative learning rates.

## Hemodynamic Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| RAP | Right atrial pressure | mmHg |
| SPAP | Systolic pulmonary artery pressure | mmHg |
| dPAP | Diastolic pulmonary artery pressure | mmHg |
| meanPAP | Mean pulmonary artery pressure | mmHg |
| PCWP | Pulmonary capillary wedge pressure | mmHg |
| CO | Cardiac output | L/min |
| CI | Cardiac index | L/min/m2 |
| SVRI | Systemic vascular resistance index | dyn.s/cm5.m2 |
| PVR | Pulmonary vascular resistance | Wood units |

## Echocardiographic Views

The model integrates four standard views per patient:

- **FC** -- Apical four-chamber (A4C)
- **TC** -- Apical two-chamber (A2C)
- **SA** -- Parasternal short-axis (PSAX)
- **LA** -- Parasternal long-axis (PLAX)

Each view is represented as 32 ECG-gated frames at 224x224 resolution.

## Architecture

```
Input: 4 views x 32 frames x 224x224
          |
    ViT-Base (per frame)        -- spatial feature extraction
          |
    Temporal attention          -- frame aggregation with learnable gating
          |
    Fusion attention            -- cross-view integration with learnable gating
          |
    9 regression heads          -- parameter-specific prediction
          |
Output: 9 hemodynamic values
```

Both temporal and fusion stages use a gated residual design: `alpha * attention + (1-alpha) * mean_pool`, where alpha is learned.

## Dataset

The complete study cohort comprised **308 total participants** from Johns Hopkins Hospital with paired invasive RHC measurements, divided into:

- **Cohort I** (n = 235, patients 1-235) — model training and 5-fold cross-validation
- **Cohort II** (n = 73, patients 236-308) — independent testing

Input tensors are pre-processed DICOM files with automated artifact removal (text, ECG traces, labels) and fan-shaped region extraction.

## Project Structure

```
echo_hemodynamics/
├── models/                   # ViT-based model definition
│   ├── heads.py              # ParameterHeadWithResidual
│   ├── temporal_attention.py # Gated temporal aggregation
│   ├── progressive_model.py  # ProgressiveCardioAI
│   ├── explainability.py     # Attention rollout + Integrated Gradients
│   └── factory.py            # create_model, create_progressive_optimizer
│
├── data/                     # Dataset + preprocessing
│   ├── preprocessing.py      # DICOM-to-tensor with fan-region extraction
│   ├── batch_conversion.py   # Batch DICOM converters
│   ├── dataset.py            # CardioAIDataset
│   ├── splits.py             # parse_train_indices, PH-balanced splits
│   └── loaders.py            # DataLoader factory + winsorize/correlation helpers
│
├── training/                 # Trainer + cross-validation
│   ├── losses.py             # ProgressiveMSELoss
│   ├── trainer.py            # ProgressiveTrainer
│   ├── checkpoints.py        # find_latest_trained_model
│   └── cross_validation.py   # 5-fold CV orchestration
│
├── ablation/                 # Attention ablation study
│   ├── variants.py           # ProgressiveAblationVariant
│   ├── factory.py            # six-variant factory
│   ├── trainer.py            # Lightweight ablation trainer
│   └── study.py              # run_ablation_study
│
├── visualization/            # Saliency / attention overlays
│   ├── apex_mask.py          # Transducer apex suppression
│   ├── frame_selection.py    # Top-N frame picking
│   ├── attention_rollout.py  # Temporal rollout curves
│   ├── integrated_gradients.py  # IG overlay rendering
│   └── visualizer.py         # ProgressiveAttentionVisualizer
│
├── analysis/                 # Validation + test figure generation
│   ├── metrics.py            # calculate_correlation, palette, cutoffs
│   ├── inference.py          # generate_model_predictions, denormalize
│   ├── correlation_plots.py  # Bar plot + heatmap variants
│   ├── scatter_plots.py      # Plain + PH-stratified scatter
│   ├── bland_altman.py       # Plain + PH-stratified Bland-Altman
│   ├── roc_auc.py            # Simple + dual ROC
│   ├── confusion_matrix.py   # Median-split confusion matrices
│   ├── embeddings.py         # UMAP / t-SNE + KMeans / DBSCAN
│   ├── heteroscedasticity.py # Quartile residual SD + Breusch-Pagan
│   └── excel_report.py       # Multi-sheet Excel writer
│
├── figures/                  # Aggregate figure utilities
│   ├── styling.py            # rcParams, save_figure helpers
│   ├── training_curves.py    # Loss / correlation / stage / LR plots
│   └── ablation_plots.py     # Five ablation comparison plots
│
├── utils/                    # Singleton + palette + output management
│   ├── palette.py            # Color helpers
│   ├── output.py             # Output directory management
│   └── singleton.py          # CardioAIUtils singleton (back-compat alias ColorManager)
│
└── runners/                  # CLI entry points (one per pipeline stage)
    ├── train.py              # python -m echo_hemodynamics.runners.train
    ├── ablation.py           # python -m echo_hemodynamics.runners.ablation
    ├── visualize.py          # python -m echo_hemodynamics.runners.visualize
    ├── validate.py           # python -m echo_hemodynamics.runners.validate
    └── test.py               # python -m echo_hemodynamics.runners.test

main.py                       # Pipeline orchestrator (calls each runner)
requirements.txt
pyproject.toml                # Editable install (pip install -e .)
README.md
LICENSE
```

## Usage

### Installation

```bash
pip install -e .
# or:
pip install -r requirements.txt
```

### Full pipeline

```bash
# Full pipeline (training + ablation + visualization + validation + test)
python main.py --epochs 100 --batch_size 16 --stages 12

# Training only
python main.py --epochs 100 --training-only

# Skip training, use existing weights
python main.py --skip-train
```

### Individual stages

```bash
# 5-fold CV training on Cohort I
python -m echo_hemodynamics.runners.train

# Attention ablation study (uses the same epoch count as main training)
python -m echo_hemodynamics.runners.ablation

# Attention visualizations (rollout curves + IG overlays)
python -m echo_hemodynamics.runners.visualize

# Internal validation on Cohort I (n=235)
python -m echo_hemodynamics.runners.validate

# Independent test on Cohort II (n=73)
python -m echo_hemodynamics.runners.test
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs per fold |
| `--batch_size` | 16 | Batch size |
| `--stages` | 12 | ViT layers to progressively unfreeze (0-12) |
| `--num-folds` | 5 | Cross-validation folds |
| `--training-frames` | 32 | Temporal frames per view |
| `--skip-train` | false | Load existing weights instead of training |
| `--training-only` | false | Run only the training step |

### Environment variables

The runners read configuration from a shared environment-variable contract (set by `main.py`). Example:

```bash
set CARDIOAI_EPOCHS=100
set CARDIOAI_BATCH_SIZE=16
set CARDIOAI_TRAINING_FRAMES=32
set CARDIOAI_NUM_FOLDS=5
set CARDIOAI_OUTPUT_DIR=E:\results_cardioAI\EchoCath_cardioAI\YYYYMMDD_HHMMSS\train_cardioAI
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m echo_hemodynamics.runners.train
```

## Outputs

Results are saved to a timestamped directory with one subfolder per stage:

```
{timestamp}/
  train_cardioAI/                       # Per-fold models, loss curves, CV summary (Cohort I)
  attention_ablation_cardioAI/          # Ablation comparison figures (same epochs as main)
  attention_visualizations_cardioAI/    # Temporal rollout curves + IG heatmaps
  validation_cardioAI/                  # Cohort I: correlation, scatter, Bland-Altman, ROC, UMAP
  test_cardioAI/                        # Cohort II: independent test evaluation
  logs/                                 # Pipeline summary + per-stage logs
```

Figures are saved in EPS, PNG (300 DPI), and TIFF (300 DPI) formats.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
