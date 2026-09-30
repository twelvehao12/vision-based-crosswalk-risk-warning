from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
