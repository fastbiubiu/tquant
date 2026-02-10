"""
Configuration module for tquant.
Provides configuration loading, validation, and management functionality.

This module uses Pydantic BaseSettings for type-safe configuration management.
"""

from .loader import (
    get_config,
    load_config,
    save_config,
    get_config_dir,
    get_config_path,
    reload_config,
    reset_config,
)
# Pydantic-based configuration system
from .schema import Config

__all__ = [
    "Config",
    "get_config",
    "load_config",
    "save_config",
    "get_config_dir",
    "get_config_path",
    "reload_config",
    "reset_config",
]