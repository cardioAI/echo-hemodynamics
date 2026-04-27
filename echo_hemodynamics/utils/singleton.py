"""CardioAIUtils singleton: shared color palette, figure styling, and output management.

The class is preserved intact for backward compatibility; lightweight wrappers in
``palette.py``, ``output.py``, and ``figures/styling.py`` expose its functionality
as free functions.
"""

import io
import sys
import warnings as _warnings
from datetime import datetime
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class CardioAIUtils:
    """Utility class for project styling and output management."""

    def __init__(self, palette_path="./palette.jpeg", output_base_dir=None):
        self.palette_path = Path(palette_path)
        self.output_base_dir = output_base_dir
        self.colors = None
        self.current_timestamp = None
        self.current_output_dir = None
        self.subdirs = {}

        self._extract_palette_colors()
        self._setup_matplotlib_style()
        self.validate_no_white_colors()

    def _extract_palette_colors(self):
        try:
            img = Image.open(self.palette_path)
            img_array = np.array(img)

            height, width = img_array.shape[:2]
            block_width = width // 10

            extracted_colors = []
            for i in range(10):
                x_start = i * block_width + block_width // 4
                x_end = (i + 1) * block_width - block_width // 4
                y_start = height // 4
                y_end = height * 3 // 4

                block_pixels = img_array[y_start:y_end, x_start:x_end, :3]
                median_color = np.median(block_pixels.reshape(-1, 3), axis=0)
                extracted_colors.append(median_color / 255.0)

            additional_colors = [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.15, 0.15, 0.15],
                [0.3, 0.3, 0.3],
                [0.45, 0.45, 0.45],
                [0.6, 0.6, 0.6],
                [0.75, 0.75, 0.75],
                [0.9, 0.9, 0.9],
            ]

            self.colors = np.vstack([extracted_colors, additional_colors])
            print(f"Extracted {len(self.colors)} colors from palette (10 from palette + 8 grayscale)")
        except Exception as e:
            print(f"Warning: Could not extract colors from palette: {e}")
            self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def _setup_matplotlib_style(self):
        plt.rcParams.update({
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        })

    def get_color_palette(self, n_colors=None):
        if n_colors is None:
            safe_colors = []
            for color in self.colors:
                if len(color) >= 3 and not (color[0] > 0.85 and color[1] > 0.85 and color[2] > 0.85):
                    safe_colors.append(color)
            return np.array(safe_colors)
        else:
            safe_colors = self.get_color_palette()
            if len(safe_colors) == 0:
                return np.array([[0.0, 0.0, 0.0]])
            indices = np.linspace(0, len(safe_colors) - 1, n_colors, dtype=int)
            return safe_colors[indices]

    def get_color(self, index):
        safe_colors = self.get_color_palette()
        return safe_colors[index % len(safe_colors)]

    def get_safe_line_colors(self, n_colors=10):
        return self.get_color_palette(n_colors)

    def get_figure_colors(self, n_colors, figure_type="regular"):
        if figure_type == "heatmap":
            return self.get_color_palette(n_colors)
        return self.get_dark_colors(n_colors)

    def get_dark_colors(self, n_colors=None):
        dark_colors = []
        for color in self.colors:
            if len(color) >= 3:
                r, g, b = color[:3]
                brightness = (r + g + b) / 3
                if brightness < 0.6:
                    dark_colors.append(color)

        if not dark_colors:
            dark_colors = [[0.2, 0.2, 0.2], [0.1, 0.2, 0.4], [0.4, 0.1, 0.1], [0.1, 0.4, 0.1]]

        if n_colors is None:
            return np.array(dark_colors)
        indices = np.linspace(0, len(dark_colors) - 1, n_colors, dtype=int)
        return np.array(dark_colors)[indices]

    def validate_no_white_colors(self):
        safe_colors = self.get_color_palette()
        for i, color in enumerate(safe_colors):
            if len(color) >= 3:
                r, g, b = color[:3]
                if r > 0.9 and g > 0.9 and b > 0.9:
                    print(f"Warning: Color {i} might be too light: RGB({r:.3f}, {g:.3f}, {b:.3f})")

    def plot_lines_safe(self, ax, x_data, y_data_list, labels=None, linestyles=None, linewidths=None):
        safe_colors = self.get_safe_line_colors(len(y_data_list))
        for i, y_data in enumerate(y_data_list):
            color = safe_colors[i]
            label = labels[i] if labels else None
            linestyle = linestyles[i] if linestyles else "-"
            linewidth = linewidths[i] if linewidths else 2
            ax.plot(x_data, y_data, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
        return ax

    def setup_output_directory(self, timestamp=None, base_dir=None):
        if base_dir is None:
            base_dir = self.output_base_dir or r"E:\results_cardioAI\EchoCath_cardioAI"
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_timestamp = timestamp
        self.current_output_dir = Path(base_dir) / timestamp
        self.subdirs = {"logs": self.current_output_dir / "logs"}
        self.subdirs["logs"].mkdir(parents=True, exist_ok=True)

        return self.current_output_dir

    def get_output_path(self, subdir_key, filename):
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")
        if subdir_key in self.subdirs:
            return self.subdirs[subdir_key] / filename
        return self.current_output_dir / filename

    def save_figure(self, fig, filename, subdir="figures", bbox_inches="tight", pad_inches=0.15):
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")

        base_name = Path(filename).stem
        output_dir = self.subdirs.get(subdir, self.current_output_dir)

        try:
            fig.tight_layout(pad=2.0)
        except Exception:
            pass

        formats = [("eps", "eps"), ("png", "png"), ("tiff", "tiff")]
        saved_files = []

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning)
            _warnings.filterwarnings("ignore", message=".*transparency.*")
            _warnings.filterwarnings("ignore", message=".*PostScript.*")

            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                for fmt, ext in formats:
                    filepath = output_dir / f"{base_name}.{ext}"
                    try:
                        if fmt == "eps":
                            fig.savefig(
                                filepath, format=fmt, dpi=300,
                                bbox_inches=bbox_inches, pad_inches=pad_inches,
                                facecolor="white", edgecolor="none", transparent=False,
                            )
                        else:
                            fig.savefig(
                                filepath, format=fmt, dpi=300,
                                bbox_inches=bbox_inches, pad_inches=pad_inches,
                                facecolor="white", edgecolor="none",
                            )
                        saved_files.append(str(filepath))
                    except Exception as e:
                        print(f"Warning: Could not save {filepath}: {e}", file=old_stderr)
            finally:
                sys.stderr = old_stderr

        return saved_files

    def get_heatmap_colormap(self, name="blue_cyan_yellow"):
        if name == "blue_cyan_yellow":
            if self.colors is not None and len(self.colors) >= 10:
                palette_colors = self.colors[:10]
                palette_10_colors = palette_colors[::-1]
                return mcolors.LinearSegmentedColormap.from_list("palette_10_colors", palette_10_colors)
            fallback_colors = [
                [0.12, 0.27, 0.44], [0.22, 0.40, 0.58], [0.32, 0.56, 0.67], [0.45, 0.74, 0.84],
                [0.67, 0.86, 0.88], [0.99, 0.90, 0.70], [0.99, 0.82, 0.43], [0.97, 0.67, 0.35],
                [0.94, 0.54, 0.27], [0.91, 0.38, 0.33],
            ]
            return mcolors.LinearSegmentedColormap.from_list("blue_cyan_yellow_palette", fallback_colors)
        elif name == "blue_gray_orange":
            fallback_colors = [
                [0.1, 0.3, 0.6], [0.3, 0.5, 0.8], [0.5, 0.5, 0.5], [0.8, 0.5, 0.2], [0.9, 0.4, 0.1]
            ]
            return mcolors.LinearSegmentedColormap.from_list("blue_gray_orange", fallback_colors)
        return plt.cm.viridis

    def create_figure(self, figsize=(8, 6), clear_previous=True):
        if clear_previous:
            plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        color_cycle = self.get_color_palette(10)
        ax.set_prop_cycle("color", color_cycle)
        return fig, ax

    def style_axis(self, ax, title=None, xlabel=None, ylabel=None,
                   remove_top_right_spines=False, legend=True):
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, labelpad=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=12)
        if remove_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(False)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(
                frameon=True, fancybox=False, shadow=False,
                framealpha=0.9, edgecolor="black", linewidth=0.5,
                bbox_to_anchor=(1.02, 1), loc="upper left",
            )

    def save_data_to_output(self, data, filename, subdir_key, format="json"):
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")

        output_path = self.get_output_path(subdir_key, filename)

        if format == "json":
            import json
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)
        elif format == "numpy":
            np.save(output_path, data)

        print(f"Saved {filename} to {output_path}")
        return str(output_path)

    def get_parameter_color_map(self):
        parameter_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        colors = self.get_color_palette(len(parameter_names))
        return dict(zip(parameter_names, colors))

    def get_view_color_map(self):
        view_names = ["FC", "TC", "SA", "LA"]
        dark_colors = self.get_dark_colors(4)
        view_colors = [
            self.colors[5],
            dark_colors[0],
            self.colors[7],
            self.colors[9],
        ]
        return dict(zip(view_names, view_colors))


cardio_utils = CardioAIUtils()


def setup_cardio_output(timestamp=None):
    return cardio_utils.setup_output_directory(timestamp)


def save_cardio_figure(fig, filename, subdir="figures"):
    return cardio_utils.save_figure(fig, filename, subdir)


def get_cardio_colors(n_colors=None):
    return cardio_utils.get_color_palette(n_colors)


def get_cardio_heatmap_cmap(name="blue_cyan_yellow"):
    return cardio_utils.get_heatmap_colormap(name)


def create_cardio_figure(figsize=(8, 6)):
    return cardio_utils.create_figure(figsize)


ColorManager = CardioAIUtils
