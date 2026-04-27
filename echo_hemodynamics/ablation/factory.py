"""Factory functions for the six ablation variants."""

from ..models import create_model
from .variants import ProgressiveAblationVariant


def create_fresh_model_for_variant(base_model, norm_params):
    """Create a fresh model with ImageNet-only ViT pre-training for fair variant training."""
    fresh_model = create_model(
        num_outputs=9,
        num_frames=base_model.num_frames,
        num_views=4,
        dropout_rate=0.15,
    )
    fresh_model.set_winsorized_normalization(norm_params)
    return fresh_model


def create_ablation_variants(base_model, norm_params):
    """Six variants: full_model + no_temporal, no_fusion, no_attention, temporal_only, fusion_only."""
    fresh_base_no_temporal = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_no_fusion = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_no_attention = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_temporal_only = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_fusion_only = create_fresh_model_for_variant(base_model, norm_params)

    return {
        "full_model": base_model,
        "no_temporal": ProgressiveAblationVariant(
            fresh_base_no_temporal, spatial_attention=False,
            temporal_attention=False, fusion_attention=True,
        ),
        "no_fusion": ProgressiveAblationVariant(
            fresh_base_no_fusion, spatial_attention=False,
            temporal_attention=True, fusion_attention=False,
        ),
        "no_attention": ProgressiveAblationVariant(
            fresh_base_no_attention, spatial_attention=False,
            temporal_attention=False, fusion_attention=False,
        ),
        "temporal_only": ProgressiveAblationVariant(
            fresh_base_temporal_only, spatial_attention=False,
            temporal_attention=True, fusion_attention=False,
        ),
        "fusion_only": ProgressiveAblationVariant(
            fresh_base_fusion_only, spatial_attention=False,
            temporal_attention=False, fusion_attention=True,
        ),
    }
