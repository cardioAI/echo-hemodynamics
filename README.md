# Echo-Hemodynamics

Non-invasive prediction of hemodynamic parameters from multi-view echocardiography using a Vision Transformer with multistage attention.

## Overview

Right heart catheterization (RHC) remains the gold standard for hemodynamic assessment but is invasive, costly, and carries procedural risk. This project trains a deep learning model to predict nine hemodynamic parameters directly from standard echocardiographic views, potentially enabling non-invasive hemodynamic screening.

The model uses a pre-trained ViT-Base backbone (google/vit-base-patch16-224) with temporal and cross-view fusion attention modules, trained with progressive unfreezing and discriminative learning rates.

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

- 308 patients from Johns Hopkins Hospital with paired invasive RHC measurements
- Training/CV: patients 1-235 (5-fold cross-validation)
- Independent test: patients 236-308

Input tensors are pre-processed DICOM files with automated artifact removal (text, ECG traces, labels) and fan-shaped region extraction.

## Project Structure

```
model_cardioAI.py                   # ViT model with temporal/fusion attention
train_cardioAI.py                   # 5-fold CV training with progressive unfreezing
dataset_cardioAI.py                 # Data loading + DICOM-to-tensor conversion
attention_ablation_cardioAI.py      # Ablation study (Aim 1)
attention_visualizations_cardioAI.py # Attention rollout + Integrated Gradients
validation_cardioAI.py              # Validation analysis and figures
test_cardioAI.py                    # Independent test evaluation
utils_cardioAI.py                   # Color palette and figure utilities
main_cardioAI.py                    # Pipeline orchestrator
```

## Usage

### Requirements

```
torch>=2.0
transformers
numpy
pandas
scikit-learn
scipy
scikit-image
opencv-python
matplotlib
seaborn
openpyxl
Pillow
```

### Training

```bash
# Full pipeline (training + ablation + visualization + evaluation)
python main_cardioAI.py --epochs 100 --batch_size 16 --stages 12

# Training only
python main_cardioAI.py --epochs 100 --training-only

# Skip training, use existing weights
python main_cardioAI.py --skip-train
```

### Individual Scripts

```bash
# Train with 5-fold CV
python train_cardioAI.py

# Run ablation study
python attention_ablation_cardioAI.py

# Generate attention visualizations
python attention_visualizations_cardioAI.py

# Evaluate on independent test set
python test_cardioAI.py
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs per fold |
| `--batch_size` | 16 | Batch size |
| `--stages` | 12 | ViT layers to progressively unfreeze (0-12) |
| `--num-folds` | 5 | Cross-validation folds |
| `--training-frames` | 32 | Temporal frames per view |
| `--skip-train` | false | Load existing weights instead of training |
| `--training-only` | false | Run only training step |

### Environment

```bash
# Recommended for large model GPU memory allocation
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Outputs

Results are saved to a timestamped directory with per-script subfolders:

```
{timestamp}/
  train_cardioAI/           # Per-fold models, loss curves, CV summary
  attention_ablation_cardioAI/   # Ablation comparison figures
  attention_visualizations_cardioAI/  # Temporal curves + IG heatmaps
  validation_cardioAI/      # Correlation, scatter, Bland-Altman, ROC, UMAP
  test_cardioAI/            # Independent test evaluation
```

Figures are saved in EPS, PNG (300 DPI), and TIFF (300 DPI) formats.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
