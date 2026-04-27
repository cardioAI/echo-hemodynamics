"""CLI entrypoint for training: ``python -m echo_hemodynamics.runners.train``."""

import os
from pathlib import Path

import torch

from ..training.cross_validation import run_cross_validation


def main():
    epochs = int(os.environ.get("CARDIOAI_EPOCHS", 100))
    stage_epochs = int(os.environ.get("CARDIOAI_STAGE_EPOCHS", 50))
    batch_size = int(os.environ.get("CARDIOAI_BATCH_SIZE", 16))
    training_frames = int(os.environ.get("CARDIOAI_TRAINING_FRAMES", 32))
    stages = int(os.environ.get("CARDIOAI_STAGES", 1))
    ablation_attentions = os.environ.get("CARDIOAI_ABLATION_ATTENTIONS", "temporal,fusion")
    num_folds = int(os.environ.get("CARDIOAI_NUM_FOLDS", 5))
    train_size = int(os.environ.get("CARDIOAI_TRAIN_SIZE", 235))

    print("Progressive training configuration:")
    print(f"  Total epochs: {epochs}")
    print(f"  Stage epochs: {stage_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training frames: {training_frames}")
    print(f"  ViT stages to unfreeze: {stages} (0=all frozen, 12=all unfrozen)")
    print(f"  Attention modules: {ablation_attentions}")
    print(f"  Num folds: {num_folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tensor_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    excel_file = Path("All.xlsx")
    output_dir = Path(os.environ.get("CARDIOAI_OUTPUT_DIR", "."))

    run_cross_validation(
        tensor_dir=tensor_dir,
        excel_file=excel_file,
        output_dir=output_dir,
        epochs=epochs,
        stage_epochs=stage_epochs,
        batch_size=batch_size,
        training_frames=training_frames,
        stages=stages,
        ablation_attentions=ablation_attentions,
        num_folds=num_folds,
        device=device,
        train_size=train_size,
    )

    print("\nProgressive training completed successfully!")
    print(f"Training outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
