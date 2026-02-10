"""
CLI commands for tquant.
Provides command-line interface for managing the trading system.
"""

import json
import logging
from typing import Optional

import typer

from tquant.config import get_config, save_config, load_config, get_config_path

app = typer.Typer(help="Tquant - Quantitative Trading System")

logger = logging.getLogger(__name__)


@app.command()
def config_show(
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Specific config key to show"),
) -> None:
    """Show current configuration."""
    try:
        config = get_config()

        if key:
            # Show specific key
            parts = key.split(".")
            value = config.model_dump()
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    typer.echo(f"Key not found: {key}", err=True)
                    raise typer.Exit(1)
            typer.echo(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            # Show all config
            typer.echo(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def config_path() -> None:
    """Show configuration file path."""
    path = get_config_path()
    typer.echo(f"Configuration file: {path}")
    typer.echo(f"Exists: {path.exists()}")


@app.command()
def config_init() -> None:
    """Initialize configuration file."""
    try:
        path = get_config_path()

        if path.exists():
            typer.echo(f"Configuration file already exists: {path}")
            if typer.confirm("Overwrite?"):
                config = load_config()
                save_config(config, path)
                typer.echo("Configuration file updated.")
            else:
                typer.echo("Cancelled.")
        else:
            config = load_config()
            save_config(config, path)
            typer.echo(f"Configuration file created: {path}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def config_validate() -> None:
    """Validate configuration."""
    try:
        config = get_config()
        typer.echo("✓ Configuration is valid")

        # Show summary
        typer.echo("\nConfiguration Summary:")
        typer.echo(f"  Trading symbols: {len(config.trading.symbols)}")
        typer.echo(f"  Initial balance: {config.trading.account.initial_balance}")
        typer.echo(f"  LLM models: {sum(1 for m in [config.llm.gpt4o, config.llm.gpt4o_mini] if m)}")
    except Exception as e:
        typer.echo(f"✗ Configuration is invalid: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    try:
        from importlib.metadata import version
        tquant_version = version("tquant")
        typer.echo(f"tquant version: {tquant_version}")
    except Exception:
        typer.echo("tquant version: unknown")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Tquant - Quantitative Trading System with LLM-powered decision making."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app()
