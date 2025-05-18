import argparse
from pathlib import Path


def validate_file_extension(file_path: str, expected_extension: str) -> Path:
    path = Path(file_path)
    if not path.name.lower().endswith(expected_extension.lower()):
        raise argparse.ArgumentTypeError(
            f"File {file_path} does not have required extension \
            '{expected_extension}'!"
        )
    return path
