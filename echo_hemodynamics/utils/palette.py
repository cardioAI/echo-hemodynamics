from .singleton import cardio_utils


def extract_palette_colors():
    return cardio_utils.colors


def get_color_palette(n_colors=None):
    return cardio_utils.get_color_palette(n_colors)


def get_color(index):
    return cardio_utils.get_color(index)


def get_dark_colors(n_colors=None):
    return cardio_utils.get_dark_colors(n_colors)


def get_heatmap_colormap(name="blue_cyan_yellow"):
    return cardio_utils.get_heatmap_colormap(name)


def get_parameter_color_map():
    return cardio_utils.get_parameter_color_map()


def get_view_color_map():
    return cardio_utils.get_view_color_map()
