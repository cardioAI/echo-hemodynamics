#!/usr/bin/env python3
"""
Utils for CardioAI Project
Provides consistent color palette, figure formatting, and output management
All figures saved in EPS, PNG (300 DPI), and TIFF (300 DPI) formats
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from datetime import datetime
import seaborn as sns


class CardioAIUtils:
    """Utility class for CardioAI project styling and output management"""
    
    def __init__(self, palette_path="./palette.jpeg", output_base_dir=None):
        self.palette_path = Path(palette_path)
        self.output_base_dir = output_base_dir
        self.colors = None
        self.current_timestamp = None
        self.current_output_dir = None
        
        # Initialize color palette
        self._extract_palette_colors()
        self._setup_matplotlib_style()
        
        # Validate no white colors in line palette
        self.validate_no_white_colors()
    
    def _extract_palette_colors(self):
        """Extract 10 colors from palette.jpeg directly"""
        try:
            # Load palette image
            img = Image.open(self.palette_path)
            img_array = np.array(img)

            # The palette has 10 color blocks arranged horizontally
            # Extract the dominant color from each block
            height, width = img_array.shape[:2]
            block_width = width // 10

            extracted_colors = []
            for i in range(10):
                # Extract center region of each color block
                x_start = i * block_width + block_width // 4
                x_end = (i + 1) * block_width - block_width // 4
                y_start = height // 4
                y_end = height * 3 // 4

                # Get pixels from center of block
                block_pixels = img_array[y_start:y_end, x_start:x_end, :3]

                # Use median color to avoid edge effects
                median_color = np.median(block_pixels.reshape(-1, 3), axis=0)
                extracted_colors.append(median_color / 255.0)

            # Add black, white, and gray shades
            additional_colors = [
                [0.0, 0.0, 0.0],  # Black
                [1.0, 1.0, 1.0],  # White (for backgrounds)
                [0.15, 0.15, 0.15],  # Very dark gray
                [0.3, 0.3, 0.3],   # Dark gray
                [0.45, 0.45, 0.45], # Medium-dark gray
                [0.6, 0.6, 0.6],   # Medium gray
                [0.75, 0.75, 0.75], # Light gray
                [0.9, 0.9, 0.9],   # Very light gray
            ]

            # Combine: 10 palette colors + 8 grayscale = 18 colors total
            self.colors = np.vstack([extracted_colors, additional_colors])

            print(f"Extracted {len(self.colors)} colors from palette (10 from palette + 8 grayscale)")

        except Exception as e:
            print(f"Warning: Could not extract colors from palette: {e}")
            # Fallback to default colors
            self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def _setup_matplotlib_style(self):
        """Setup consistent matplotlib style"""
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.grid': False,  # No grids
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 0.8,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        })
    
    def get_color_palette(self, n_colors=None):
        """Get color palette for plotting (NO WHITE for lines)"""
        if n_colors is None:
            # Return all colors excluding white and very light colors that could appear white
            safe_colors = []
            for color in self.colors:
                # Skip colors that are too light (might appear white-ish)
                if len(color) >= 3 and not (color[0] > 0.85 and color[1] > 0.85 and color[2] > 0.85):
                    safe_colors.append(color)
            return np.array(safe_colors)
        else:
            # Return evenly spaced colors from safe palette
            safe_colors = self.get_color_palette()
            if len(safe_colors) == 0:
                return np.array([[0.0, 0.0, 0.0]])  # Fallback to black
            indices = np.linspace(0, len(safe_colors)-1, n_colors, dtype=int)
            return safe_colors[indices]
    
    def get_color(self, index):
        """Get a specific color by index (NO WHITE for lines)"""
        safe_colors = self.get_color_palette()  # Excludes white and light colors
        return safe_colors[index % len(safe_colors)]
    
    def get_safe_line_colors(self, n_colors=10):
        """Get safe colors for lines (guaranteed no white)"""
        return self.get_color_palette(n_colors)
    
    def get_figure_colors(self, n_colors, figure_type='regular'):
        """Get colors appropriate for different figure types
        
        Args:
            n_colors: Number of colors needed
            figure_type: 'regular' (dark colors only) or 'heatmap' (light and dark)
        """
        if figure_type == 'heatmap':
            return self.get_color_palette(n_colors)  # Allow light and dark
        else:
            return self.get_dark_colors(n_colors)  # Only dark colors
    
    def get_dark_colors(self, n_colors=None):
        """Get specifically dark colors from palette for better visibility"""
        dark_colors = []
        for color in self.colors:
            # Select colors that are dark (RGB values < 0.6 for better visibility)
            if len(color) >= 3:
                r, g, b = color[:3]
                # Consider a color dark if all RGB components are below 0.6
                brightness = (r + g + b) / 3
                if brightness < 0.6:
                    dark_colors.append(color)
        
        if not dark_colors:
            # Fallback to very dark colors if none found
            dark_colors = [[0.2, 0.2, 0.2], [0.1, 0.2, 0.4], [0.4, 0.1, 0.1], [0.1, 0.4, 0.1]]
            
        if n_colors is None:
            return np.array(dark_colors)
        else:
            indices = np.linspace(0, len(dark_colors)-1, n_colors, dtype=int)
            return np.array(dark_colors)[indices]
    
    def validate_no_white_colors(self):
        """Validate that no white colors are in the line palette"""
        safe_colors = self.get_color_palette()
        for i, color in enumerate(safe_colors):
            if len(color) >= 3:
                r, g, b = color[:3]
                if r > 0.9 and g > 0.9 and b > 0.9:
                    print(f"Warning: Color {i} might be too light: RGB({r:.3f}, {g:.3f}, {b:.3f})")
    
    def plot_lines_safe(self, ax, x_data, y_data_list, labels=None, linestyles=None, linewidths=None):
        """Plot multiple lines with safe colors (no white)"""
        safe_colors = self.get_safe_line_colors(len(y_data_list))
        
        for i, y_data in enumerate(y_data_list):
            color = safe_colors[i]
            label = labels[i] if labels else None
            linestyle = linestyles[i] if linestyles else '-'
            linewidth = linewidths[i] if linewidths else 2
            
            ax.plot(x_data, y_data, color=color, label=label, 
                   linestyle=linestyle, linewidth=linewidth)
        
        return ax
    
    def setup_output_directory(self, timestamp=None, base_dir=None):
        """Setup organized output directory structure"""
        if base_dir is None:
            base_dir = self.output_base_dir or r"E:\results_cardioAI\EchoCath_cardioAI"
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_timestamp = timestamp
        self.current_output_dir = Path(base_dir) / timestamp
        
        # Define subdirectory paths but don't create them automatically
        # Each script now creates its own dedicated subfolder structure
        self.subdirs = {
            'logs': self.current_output_dir / "logs"
        }
        
        # Only create the logs directory - scripts create their own subfolders
        self.subdirs['logs'].mkdir(parents=True, exist_ok=True)
        
        return self.current_output_dir
    
    def get_output_path(self, subdir_key, filename):
        """Get full output path for a file in a specific subdirectory"""
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")
        
        if subdir_key in self.subdirs:
            return self.subdirs[subdir_key] / filename
        else:
            return self.current_output_dir / filename
    
    def save_figure(self, fig, filename, subdir='figures', bbox_inches='tight', pad_inches=0.15):
        """Save figure in EPS, PNG (300 DPI), and TIFF (300 DPI) formats

        Enhanced to prevent overlapping of scale bars, text, inset boxes, and labels
        EPS format is saved with transparency disabled to avoid PostScript backend warnings
        """
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")

        # Get base filename without extension
        base_name = Path(filename).stem
        output_dir = self.subdirs.get(subdir, self.current_output_dir)

        # Apply tight layout with extra padding to prevent overlaps
        try:
            fig.tight_layout(pad=2.0)  # Extra padding for text/label spacing
        except:
            pass  # In case tight_layout fails for any reason

        # Save in three formats
        formats = [
            ('eps', 'eps'),
            ('png', 'png'),
            ('tiff', 'tiff')
        ]

        saved_files = []

        # Suppress ALL matplotlib backend warnings (including EPS transparency)
        import warnings
        import sys
        import io

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*transparency.*")
            warnings.filterwarnings("ignore", message=".*PostScript.*")

            # Also redirect stderr to suppress C++ backend warnings
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                for fmt, ext in formats:
                    filepath = output_dir / f"{base_name}.{ext}"
                    try:
                        # For EPS format, explicitly disable transparency to avoid warnings
                        if fmt == 'eps':
                            fig.savefig(
                                filepath,
                                format=fmt,
                                dpi=300,
                                bbox_inches=bbox_inches,
                                pad_inches=pad_inches,
                                facecolor='white',  # White background for figures
                                edgecolor='none',
                                transparent=False  # Explicitly disable transparency for EPS
                            )
                        else:
                            # For PNG and TIFF, allow transparency
                            fig.savefig(
                                filepath,
                                format=fmt,
                                dpi=300,
                                bbox_inches=bbox_inches,
                                pad_inches=pad_inches,
                                facecolor='white',  # White background for figures
                                edgecolor='none'
                            )
                        saved_files.append(str(filepath))
                    except Exception as e:
                        print(f"Warning: Could not save {filepath}: {e}", file=old_stderr)
            finally:
                # Restore stderr
                sys.stderr = old_stderr

        return saved_files
    
    def get_heatmap_colormap(self, name='blue_cyan_yellow'):
        """Get custom colormap for heatmaps using 10 colors from palette.jpeg"""
        if name == 'blue_cyan_yellow':
            # Use the 10 colors from palette.jpeg in order from low to high attention
            # Low attention: Blue colors (navy → dark → medium → sky blue)
            # Medium attention: Light colors (cyan → cream → yellow)
            # High attention: Red/warm colors (peach → orange → coral red)

            if self.colors is not None and len(self.colors) >= 10:
                # Extract first 10 colors from palette (already in correct order)
                palette_colors = self.colors[:10]
                # Reverse order: index 9→0 becomes 0→9 (low to high attention)
                palette_10_colors = palette_colors[::-1]  # Navy to Coral Red
                return mcolors.LinearSegmentedColormap.from_list('palette_10_colors', palette_10_colors)
            else:
                # Fallback if palette not loaded
                fallback_colors = [
                    [0.12, 0.27, 0.44],  # Navy Blue (low)
                    [0.22, 0.40, 0.58],  # Dark Blue
                    [0.32, 0.56, 0.67],  # Medium Blue
                    [0.45, 0.74, 0.84],  # Sky Blue
                    [0.67, 0.86, 0.88],  # Light Cyan
                    [0.99, 0.90, 0.70],  # Pale Yellow/Cream
                    [0.99, 0.82, 0.43],  # Yellow
                    [0.97, 0.67, 0.35],  # Light Orange/Peach
                    [0.94, 0.54, 0.27],  # Orange
                    [0.91, 0.38, 0.33],  # Coral/Salmon Red (high)
                ]
                return mcolors.LinearSegmentedColormap.from_list('blue_cyan_yellow_palette', fallback_colors)
        elif name == 'blue_gray_orange':
            # Keep old scheme as fallback
            fallback_colors = [[0.1, 0.3, 0.6], [0.3, 0.5, 0.8], [0.5, 0.5, 0.5], [0.8, 0.5, 0.2], [0.9, 0.4, 0.1]]
            return mcolors.LinearSegmentedColormap.from_list('blue_gray_orange', fallback_colors)
        else:
            # Fallback to a non-white colormap
            return plt.cm.viridis
    
    def create_figure(self, figsize=(8, 6), clear_previous=True):
        """Create a new figure with consistent styling"""
        if clear_previous:
            plt.close('all')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Remove grid
        ax.grid(False)
        
        # Set color cycle to our palette (NO WHITE)
        color_cycle = self.get_color_palette(10)  # Use first 10 non-white colors for cycling
        ax.set_prop_cycle('color', color_cycle)
        
        return fig, ax
    
    def style_axis(self, ax, title=None, xlabel=None, ylabel=None, 
                   remove_top_right_spines=False, legend=True):
        """Apply consistent axis styling with enhanced spacing to prevent overlaps"""
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)  # Increased padding
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, labelpad=12)  # Increased padding
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=12)  # Increased padding
        
        # Remove top and right spines if requested
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Style tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # No grid
        ax.grid(False)
        
        # Legend styling with better spacing
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, fancybox=False, shadow=False, 
                     framealpha=0.9, edgecolor='black', linewidth=0.5,
                     bbox_to_anchor=(1.02, 1), loc='upper left')  # Position outside plot area
    
    def save_data_to_output(self, data, filename, subdir_key, format='json'):
        """Save data to organized output directory"""
        if self.current_output_dir is None:
            raise ValueError("Output directory not set. Call setup_output_directory() first.")
        
        output_path = self.get_output_path(subdir_key, filename)
        
        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)
        elif format == 'numpy':
            np.save(output_path, data)
        
        print(f"Saved {filename} to {output_path}")
        return str(output_path)
    
    def get_parameter_color_map(self):
        """Get consistent color mapping for the 9 hemodynamic parameters"""
        parameter_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        colors = self.get_color_palette(len(parameter_names))
        return dict(zip(parameter_names, colors))
    
    def get_view_color_map(self):
        """Get consistent color mapping for the 4 cardiac views"""
        view_names = ['FC', 'TC', 'SA', 'LA']
        # Get dark colors for better visibility, especially for TC
        dark_colors = self.get_dark_colors(4)

        # Use distinct colors with dark color for TC for better visibility
        # Note: Palette has 18 colors total (10 from palette + 8 grayscale, indices 0-17)
        view_colors = [
            self.colors[5],        # Teal for FC
            dark_colors[0],        # Dark color for TC (better visibility)
            self.colors[7],        # Color for SA (changed from invalid index 25)
            self.colors[9]         # Color for LA (changed from invalid index 35)
        ]
        return dict(zip(view_names, view_colors))


# Global instance for easy access
cardio_utils = CardioAIUtils()


def setup_cardio_output(timestamp=None):
    """Convenience function to setup output directory"""
    return cardio_utils.setup_output_directory(timestamp)


def save_cardio_figure(fig, filename, subdir='figures'):
    """Convenience function to save figure in all formats"""
    return cardio_utils.save_figure(fig, filename, subdir)


def get_cardio_colors(n_colors=None):
    """Convenience function to get color palette"""
    return cardio_utils.get_color_palette(n_colors)


def get_cardio_heatmap_cmap(name='blue_cyan_yellow'):
    """Convenience function to get heatmap colormap"""
    return cardio_utils.get_heatmap_colormap(name)


def create_cardio_figure(figsize=(8, 6)):
    """Convenience function to create styled figure"""
    return cardio_utils.create_figure(figsize)


# Alias for compatibility
ColorManager = CardioAIUtils


if __name__ == "__main__":
    # Test the utils module
    print("Testing CardioAI Utils...")
    
    # Test color extraction
    utils = CardioAIUtils()
    print(f"Extracted {len(utils.colors)} colors")
    
    # Test output directory setup
    output_dir = utils.setup_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Test figure creation and saving
    fig, ax = utils.create_figure()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color=utils.get_color(0))
    utils.style_axis(ax, title="Test Figure", xlabel="X", ylabel="Y")
    saved_files = utils.save_figure(fig, "test_figure")
    print(f"Saved test figure: {saved_files}")
    
    plt.close(fig)
    print("CardioAI Utils test completed successfully!")