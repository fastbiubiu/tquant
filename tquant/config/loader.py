"""
Configuration loading utilities for tquant.
Handles loading, saving, and managing configuration from multiple sources.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .schema import Config

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get the tquant configuration directory."""
    config_dir = Path.home() / ".tquant"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return get_config_dir() / "config.json"


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or environment variables.

    Priority order:
    1. Environment variables (TQUANT_* prefix)
    2. Configuration file (JSON)
    3. Default configuration

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.

    Raises:
        ValueError: If configuration is invalid.
    """
    path = config_path or get_config_path()

    try:
        if path.exists():
            logger.info(f"Loading configuration from {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convert to snake_case for Pydantic
            data = _convert_keys_to_snake_case(data)
            return Config(**data)
        else:
            logger.info("No configuration file found, using environment variables and defaults")
            return Config()
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        logger.info("Using environment variables and defaults")
        return Config()


def save_config(config: Config, config_path: Optional[Path] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.

    Raises:
        IOError: If unable to write configuration file.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert to camelCase for JSON storage
        data = config.model_dump()
        data = _convert_keys_to_camel_case(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {path}")
    except IOError as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def _convert_keys_to_snake_case(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {_camel_to_snake(k): _convert_keys_to_snake_case(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys_to_snake_case(item) for item in data]
    return data


def _convert_keys_to_camel_case(data: Any) -> Any:
    """Convert snake_case keys to camelCase for JSON storage."""
    if isinstance(data, dict):
        return {_snake_to_camel(k): _convert_keys_to_camel_case(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys_to_camel_case(item) for item in data]
    return data


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to config file.

    Returns:
        Configuration object.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = load_config(config_path)

    return _config_instance


def reload_config(config_path: Optional[Path] = None) -> Config:
    """
    Reload the global configuration instance.

    Args:
        config_path: Optional path to config file.

    Returns:
        Reloaded configuration object.
    """
    global _config_instance

    logger.info("Reloading configuration...")
    _config_instance = load_config(config_path)
    logger.info("Configuration reloaded")

    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None
