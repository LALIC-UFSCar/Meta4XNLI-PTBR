from pathlib import Path

import yaml


def parse_yaml(file_path: Path) -> dict:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    return data
