"""Figure styling and saving (wrappers around the CardioAIUtils singleton)."""

from ..utils.singleton import cardio_utils


def setup_matplotlib_style():
    """Apply the project-wide matplotlib rcParams (300 DPI, no grids, etc.)."""
    cardio_utils._setup_matplotlib_style()


def create_figure(figsize=(8, 6), clear_previous=True):
    return cardio_utils.create_figure(figsize=figsize, clear_previous=clear_previous)


def style_axis(ax, title=None, xlabel=None, ylabel=None,
               remove_top_right_spines=False, legend=True):
    return cardio_utils.style_axis(
        ax, title=title, xlabel=xlabel, ylabel=ylabel,
        remove_top_right_spines=remove_top_right_spines, legend=legend,
    )


def save_figure(fig, filename, subdir="figures", bbox_inches="tight", pad_inches=0.15):
    """Save figure as EPS, PNG, and TIFF (300 DPI). Returns list of paths written."""
    return cardio_utils.save_figure(
        fig, filename, subdir=subdir,
        bbox_inches=bbox_inches, pad_inches=pad_inches,
    )
