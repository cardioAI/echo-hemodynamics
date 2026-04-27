"""ProgressiveTrainer: progressive ViT unfreezing + per-task correlation tracking."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..models import create_progressive_optimizer
from .losses import ProgressiveMSELoss


class ProgressiveTrainer:
    """Multi-stage trainer with progressive ViT unfreezing and discriminative LRs."""

    def __init__(self, model, train_loader, val_loader, device="cuda",
                 stage_epochs=50, task_lr=3e-4, vit_lr=3e-5, stages=0,
                 total_epochs=100, test_loader=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.stage_epochs = stage_epochs
        self.task_lr = task_lr
        self.vit_lr = vit_lr
        self.stages = stages
        self.total_epochs = total_epochs

        self.loss_fn = ProgressiveMSELoss(device)

        self.current_stage = 0
        self.total_stages = max(1, self.stages + 1) if self.stages > 0 else 1
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_correlations = None
        self.best_test_correlations = None

        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_correlations": [],
            "per_task_val_corr": [],
            "test_loss": [],
            "test_correlations": [],
            "per_task_test_corr": [],
            "stage_info": [],
            "learning_rates": [],
        }

        self.optimizer = self._create_stage_optimizer()
        self.scheduler = None

        self.output_dir = Path(os.environ.get("CARDIOAI_OUTPUT_DIR", "."))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_stage_optimizer(self):
        return create_progressive_optimizer(self.model, task_lr=self.task_lr, vit_lr=self.vit_lr)

    def _advance_stage(self):
        self.current_stage += 1

        if self.stages > 0 and self.current_stage <= self.stages:
            block_idx = 12 - self.current_stage
            print(f"\n{'=' * 60}")
            print(f"ADVANCING TO STAGE {self.current_stage}")
            print(f"Unfreezing ViT transformer block {block_idx} (from last, max {self.stages} blocks)")
            print(f"{'=' * 60}")

            self.model.unfreeze_vit_block(block_idx)
            self.optimizer = self._create_stage_optimizer()

            stage_info = {
                "stage": self.current_stage,
                "unfrozen_block": block_idx,
                "epoch": self.current_epoch,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
            self.training_history["stage_info"].append(stage_info)
        elif self.stages == 0:
            print(f"\n{'=' * 60}")
            print(f"STAGE {self.current_stage}: ViT BLOCKS REMAIN FROZEN (stages=0)")
            print(f"{'=' * 60}")
            stage_info = {
                "stage": self.current_stage,
                "unfrozen_block": None,
                "epoch": self.current_epoch,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
            self.training_history["stage_info"].append(stage_info)
        else:
            print(f"All requested ViT blocks unfrozen ({self.stages}/{self.stages}).")

    def normalize_targets(self, targets):
        return self.model.normalize_targets(targets)

    def compute_correlations(self, predictions, targets):
        correlations = []
        pred_denorm = self.model.denormalize_predictions(predictions)
        targ_denorm = targets

        for i in range(predictions.shape[1]):
            pred_i = pred_denorm[:, i].detach().cpu().numpy()
            targ_i = targ_denorm[:, i].detach().cpu().numpy()

            if np.std(pred_i) > 1e-8 and np.std(targ_i) > 1e-8:
                corr = np.corrcoef(pred_i, targ_i)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            correlations.append(abs(corr))

        return correlations

    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_targets = []

        for batch_idx, (views, targets, patient_ids) in enumerate(self.train_loader):
            views = [view.to(self.device) for view in views]
            targets = targets.to(self.device)

            targets_norm = self.normalize_targets(targets)

            self.optimizer.zero_grad()

            try:
                predictions = self.model(views, return_aux=False)

                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print(f"Warning: NaN/Inf in predictions at batch {batch_idx}")
                    continue

                loss = self.loss_fn(predictions, targets_norm)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            if all_predictions:
                all_pred = torch.cat(all_predictions, dim=0)
                all_targ = torch.cat(all_targets, dim=0)
                train_correlations = self.compute_correlations(all_pred, all_targ)
            else:
                train_correlations = [0.0] * 9
        else:
            avg_loss = float("inf")
            train_correlations = [0.0] * 9

        return avg_loss, train_correlations

    def validate_on_loader(self, loader, loader_name="Validation"):
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (views, targets, patient_ids) in enumerate(loader):
                views = [view.to(self.device) for view in views]
                targets = targets.to(self.device)
                targets_norm = self.normalize_targets(targets)

                try:
                    predictions = self.model(views, return_aux=False)
                    loss = self.loss_fn(predictions, targets_norm)
                    val_losses.append(loss.item())
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                except Exception as e:
                    print(f"{loader_name} error in batch {batch_idx}: {e}")
                    continue

        if val_losses:
            avg_val_loss = np.mean(val_losses)
            if all_predictions:
                all_pred = torch.cat(all_predictions, dim=0)
                all_targ = torch.cat(all_targets, dim=0)
                val_correlations = self.compute_correlations(all_pred, all_targ)
            else:
                val_correlations = [0.0] * 9
        else:
            avg_val_loss = float("inf")
            val_correlations = [0.0] * 9

        return avg_val_loss, val_correlations

    def validate_epoch(self):
        return self.validate_on_loader(self.val_loader, "Internal Validation")

    def validate_test(self):
        if self.test_loader is None:
            return None, None
        return self.validate_on_loader(self.test_loader, "Test")

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            "epoch": self.current_epoch,
            "stage": self.current_stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None,
            "best_val_loss": self.best_val_loss,
            "best_correlations": self.best_correlations,
            "training_history": self.training_history,
        }

        try:
            torch.save(checkpoint, self.output_dir / "latest_checkpoint.pth")
        except Exception as e:
            print(f"Warning: Could not save latest checkpoint: {e}")

        if is_best:
            try:
                torch.save(checkpoint, self.output_dir / "best_checkpoint.pth")
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
            except Exception as e:
                print(f"Warning: Could not save best checkpoint: {e}")

    def check_correlation_threshold(self, correlations, threshold=0.6):
        return all(corr >= threshold for corr in correlations)

    def train(self, total_epochs):
        print(f"\n{'=' * 80}")
        print("STARTING PROGRESSIVE TRAINING")
        print(f"{'=' * 80}")
        print(
            f"Epochs: {total_epochs}, Stage epochs: {self.stage_epochs}, "
            f"Stages: {self.total_stages}"
        )
        print(f"LR: task={self.task_lr}, vit={self.vit_lr}")
        print(f"{'=' * 80}")

        stage_start_epoch = 0

        for epoch in range(total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            if epoch > 0 and (epoch - stage_start_epoch) >= self.stage_epochs:
                if self.current_stage < self.total_stages - 1:
                    self._advance_stage()
                    stage_start_epoch = epoch

            train_loss, train_corr = self.train_epoch()
            val_loss, val_corr = self.validate_epoch()

            if self.test_loader:
                test_loss, test_corr = self.validate_test()
            else:
                test_loss, test_corr = None, None

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_correlations"].append(np.mean(val_corr))
            self.training_history["per_task_val_corr"].append(val_corr)

            if test_loss is not None:
                self.training_history["test_loss"].append(test_loss)
                self.training_history["test_correlations"].append(np.mean(test_corr))
                self.training_history["per_task_test_corr"].append(test_corr)

            lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.training_history["learning_rates"].append(lrs)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_correlations = val_corr.copy()
                if test_corr is not None:
                    self.best_test_correlations = test_corr.copy()

            self.save_checkpoint(is_best)

            epoch_time = time.time() - epoch_start_time
            avg_corr = np.mean(val_corr)
            above_threshold = sum(1 for c in val_corr if c >= 0.6)

            print(
                f"Epoch {epoch + 1:3d}/{total_epochs} | Stage {self.current_stage:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Int Val Loss: {val_loss:.4f} | "
                f"Int Val Corr: {avg_corr:.3f} ({above_threshold}/9 >=0.6)"
            )

            if test_corr is not None:
                test_avg_corr = np.mean(test_corr)
                test_above_threshold = sum(1 for c in test_corr if c >= 0.6)
                print(
                    f"{'':>18} Test Loss: {test_loss:.4f} | "
                    f"Test Corr: {test_avg_corr:.3f} ({test_above_threshold}/9 >=0.6) | "
                    f"Time: {epoch_time:.1f}s"
                )
            else:
                print(f"{'':>18} Time: {epoch_time:.1f}s")

            if self.check_correlation_threshold(val_corr, 0.6):
                print(f"    All val correlations above 0.6 at epoch {epoch + 1}")

        self.save_training_results()
        print(f"\n{'=' * 60}")
        print("PROGRESSIVE TRAINING COMPLETED")
        print(f"{'=' * 60}")

        param_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        print("\nCross-Validation Fold Results:")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        if self.best_correlations:
            for param, corr in zip(param_names, self.best_correlations):
                status = "[PASS]" if corr >= 0.6 else "[FAIL]"
                print(f"  {param:8s}: {corr:.3f} {status}")

        if self.best_test_correlations:
            print("\nTest Results (Cohort II, n=73):")
            for param, corr in zip(param_names, self.best_test_correlations):
                status = "[PASS]" if corr >= 0.6 else "[FAIL]"
                print(f"  {param:8s}: {corr:.3f} {status}")

    def save_training_results(self):
        from ..figures.training_curves import generate_training_figures

        results = {
            "training_config": {
                "stage_epochs": self.stage_epochs,
                "task_lr": self.task_lr,
                "vit_lr": self.vit_lr,
                "total_stages": self.total_stages,
                "total_epochs": self.current_epoch + 1,
            },
            "final_metrics": {
                "best_val_loss": self.best_val_loss,
                "best_correlations": self.best_correlations,
                "best_test_correlations": self.best_test_correlations,
                "final_stage": self.current_stage,
            },
            "training_history": self.training_history,
        }

        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

        print(f"Training results saved to: {self.output_dir}")
        generate_training_figures(self.training_history, self.output_dir)
