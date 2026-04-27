"""Output directory + data-saving helpers (wrappers around CardioAIUtils singleton)."""

from .singleton import cardio_utils


def setup_output_directory(timestamp=None, base_dir=None):
    return cardio_utils.setup_output_directory(timestamp, base_dir)


def get_output_path(subdir_key, filename):
    return cardio_utils.get_output_path(subdir_key, filename)


def save_data_to_output(data, filename, subdir_key, format="json"):
    return cardio_utils.save_data_to_output(data, filename, subdir_key, format)
