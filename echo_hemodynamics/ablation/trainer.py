"""Lightweight trainer for ablation variants (single LR, no progressive unfreezing)."""

import numpy as np
import torch

from ..training.losses import ProgressiveMSELoss


class ProgressiveAblationTrainer:
    """AdamW trainer for ablation variants — fewer epochs, no progressive unfreezing."""

    def __init__(self, model, train_loader, val_loader, device="cuda", epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

        self.loss_fn = ProgressiveMSELoss(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_correlations": [],
        }

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

        for batch_idx, (views, targets, _) in enumerate(self.train_loader):
            views = [view.to(self.device) for view in views]
            targets = targets.to(self.device)
            targets_norm = self.model.normalize_targets(targets)

            self.optimizer.zero_grad()
            try:
                predictions = self.model(views, return_aux=False)
                loss = self.loss_fn(predictions, targets_norm)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        return np.mean(epoch_losses) if epoch_losses else float("inf")

    def validate_epoch(self):
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for _, (views, targets, _) in enumerate(self.val_loader):
                views = [view.to(self.device) for view in views]
                targets = targets.to(self.device)
                targets_norm = self.model.normalize_targets(targets)

                try:
                    predictions = self.model(views, return_aux=False)
                    loss = self.loss_fn(predictions, targets_norm)
                    val_losses.append(loss.item())
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                except Exception:
                    continue

        if val_losses and all_predictions:
            avg_val_loss = np.mean(val_losses)
            all_pred = torch.cat(all_predictions, dim=0)
            all_targ = torch.cat(all_targets, dim=0)
            val_correlations = self.compute_correlations(all_pred, all_targ)
        else:
            avg_val_loss = float("inf")
            val_correlations = [0.0] * 9

        return avg_val_loss, val_correlations

    def train(self):
        print(f"Training ablation variant for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_corr = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_correlations"].append(np.mean(val_corr))

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                avg_corr = np.mean(val_corr)
                print(
                    f"  Epoch {epoch + 1:2d}/{self.epochs}: "
                    f"Loss {train_loss:.4f}/{val_loss:.4f}, Corr {avg_corr:.3f}"
                )

        return self.history
