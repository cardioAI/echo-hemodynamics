"""Mixin providing attention rollout and Integrated Gradients explainability."""

import torch


class ExplainabilityMixin:
    """Adds attention_rollout and get_integrated_gradients to ProgressiveCardioAI.

    Expects the host class to expose:
      - self.use_temporal_attention, self.temporal_attention
      - self.extract_spatial_features(view_flat)
      - self.forward(views, return_aux=False)
      - self.view_names (list[str])
    """

    def attention_rollout(self, views, target_param_idx=0, head_fusion="mean"):
        if not self.use_temporal_attention or self.temporal_attention is None:
            print("Warning: No temporal attention enabled for rollout analysis")
            return {}

        self.eval()
        with torch.no_grad():
            batch_size = views[0].shape[0]
            rollout_scores = {}

            for view_idx, view in enumerate(views):
                if len(view.shape) == 4:
                    frames = view.shape[1]
                    height, width = view.shape[2], view.shape[3]
                elif len(view.shape) == 3:
                    frames = view.shape[0]
                    height, width = view.shape[1], view.shape[2]
                    view = view.unsqueeze(0)
                else:
                    continue

                view_name = ["FC", "TC", "SA", "LA"][view_idx] if view_idx < 4 else f"View_{view_idx}"

                view_flat = view.reshape(batch_size * frames, 1, height, width)
                frame_features = self.extract_spatial_features(view_flat)
                frame_features = frame_features.reshape(batch_size, frames, -1)

                _, _, frame_weights = self.temporal_attention(frame_features)
                rollout_scores[view_name] = frame_weights.cpu().numpy()

            return rollout_scores

    def get_integrated_gradients(self, views, target_param_idx=0, n_steps=50):
        """Integrated Gradients (Sundararajan et al., ICML 2017) for input attribution."""
        self.eval()
        device = views[0].device
        baseline_views = [torch.zeros_like(view) for view in views]
        ig_results = {}

        for view_idx, (view, baseline) in enumerate(zip(views, baseline_views)):
            view_name = self.view_names[view_idx] if view_idx < len(self.view_names) else f"View_{view_idx}"

            if len(view.shape) == 4:
                batch_size = view.shape[0]
            elif len(view.shape) == 3:
                view = view.unsqueeze(0)
                baseline = baseline.unsqueeze(0)
                batch_size = 1
            else:
                continue

            alphas = torch.linspace(0, 1, n_steps).to(device)
            accumulated_gradients = torch.zeros_like(view)

            for alpha in alphas:
                interpolated = baseline + alpha * (view - baseline)
                interpolated.requires_grad = True

                views_copy = [
                    baseline_views[i].clone() if i != view_idx else interpolated
                    for i in range(len(views))
                ]

                predictions = self.forward(views_copy, return_aux=False)
                target_output = predictions[:, target_param_idx]
                loss = target_output.sum()
                loss.backward(retain_graph=False)

                if interpolated.grad is not None:
                    accumulated_gradients += interpolated.grad.detach()
                interpolated.grad = None

            avg_gradients = accumulated_gradients / n_steps
            integrated_gradients = (view - baseline) * avg_gradients
            ig_results[view_name] = integrated_gradients.detach()

        return ig_results
