from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_SECTIONS = {
    "source",
    "model",
    "tracking",
    "storage",
    "rules",
}


def load_config(config_path) -> dict[str, Any]:
    """
    Load and perform basic structural validation of a YAML
    application configuration file.

    Args:
        config_path:
            Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError:
            When the configuration file does not exist.

        ValueError:
            When the YAML content is empty, is not a mapping,
            or is missing required top-level sections.
    """

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            "Configuration file is empty."
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    missing_sections = (
        REQUIRED_TOP_LEVEL_SECTIONS
        - set(config.keys())
    )

    if missing_sections:
        missing = ", ".join(
            sorted(missing_sections)
        )

        raise ValueError(
            "Configuration is missing required "
            f"sections: {missing}"
        )

    return config