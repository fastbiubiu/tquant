"""
Pydantic-based configuration schema for tquant.
Provides type-safe configuration management with validation.
"""

from datetime import date
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingSymbols(BaseModel):
    """Trading symbols configuration."""
    symbols: List[str] = Field(default_factory=list, description="List of trading symbols")


class AccountConfig(BaseModel):
    """Account configuration."""
    initial_balance: float = Field(gt=0, default=1000000.0, description="Initial account balance")
    max_position_ratio: float = Field(ge=0, le=1, default=1.0, description="Maximum position ratio")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_loss_ratio: float = Field(ge=0, le=1, default=0.1, description="Maximum daily loss ratio")
    stop_loss_ratio: float = Field(ge=0, le=1, default=0.03, description="Stop loss ratio per trade")
    take_profit_ratio: float = Field(ge=0, le=1, default=0.06, description="Take profit ratio per trade")


class TradingConfig(BaseModel):
    """Trading configuration."""
    symbols: List[str] = Field(default_factory=list, description="Trading symbols")
    account: AccountConfig = Field(default_factory=AccountConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)


class LLMModelConfig(BaseModel):
    """LLM model configuration."""
    model: str = Field(description="Model name")
    api_key: str = Field(description="API key")
    temperature: float = Field(ge=0, le=2, description="Temperature parameter")
    max_tokens: int = Field(gt=0, description="Maximum tokens")
    timeout: int = Field(gt=0, description="Request timeout in seconds")


class LLMConfig(BaseModel):
    """LLM configuration."""
    gpt4o: Optional[LLMModelConfig] = Field(default=None, description="GPT-4o configuration")
    gpt4o_mini: Optional[LLMModelConfig] = Field(default=None, description="GPT-4o-mini configuration")


class MAIndicatorConfig(BaseModel):
    """Moving Average indicator configuration."""
    periods: List[int] = Field(default_factory=list, description="MA periods")


class MACDIndicatorConfig(BaseModel):
    """MACD indicator configuration."""
    fast_period: int = Field(gt=0, description="Fast period")
    slow_period: int = Field(gt=0, description="Slow period")
    signal_period: int = Field(gt=0, description="Signal period")


class RSIIndicatorConfig(BaseModel):
    """RSI indicator configuration."""
    period: int = Field(gt=0, description="RSI period")
    overbought: float = Field(ge=0, le=100, description="Overbought threshold")
    oversold: float = Field(ge=0, le=100, description="Oversold threshold")


class BollingerBandsConfig(BaseModel):
    """Bollinger Bands indicator configuration."""
    period: int = Field(gt=0, description="Period")
    std_dev: float = Field(gt=0, description="Standard deviation multiplier")


class IndicatorsConfig(BaseModel):
    """Technical indicators configuration."""
    ma: Optional[MAIndicatorConfig] = Field(default=None, description="MA configuration")
    macd: Optional[MACDIndicatorConfig] = Field(default=None, description="MACD configuration")
    rsi: Optional[RSIIndicatorConfig] = Field(default=None, description="RSI configuration")
    bollinger_bands: Optional[BollingerBandsConfig] = Field(default=None, description="Bollinger Bands configuration")


class TQSDKConfig(BaseModel):
    """TQSDK configuration."""
    auth: str = Field(default="", description="Authentication credentials")
    backtest: bool = Field(default=False, description="Enable backtest mode")
    demo: bool = Field(default=True, description="Enable demo mode")
    log_level: str = Field(default="INFO", description="Log level")


class BacktestConfig(BaseModel):
    """Backtest runtime configuration."""
    start_date: date = Field(description="Backtest start date (YYYY-MM-DD)")
    end_date: date = Field(description="Backtest end date (YYYY-MM-DD)")
    initial_balance: float = Field(gt=0, default=1_000_000.0, description="Initial balance for backtest")
    commission: float = Field(ge=0, default=0.0003, description="Commission rate")
    slippage: float = Field(ge=0, default=0.001, description="Slippage rate")
    refresh_interval: int = Field(gt=0, default=60, description="Backtest refresh interval in seconds")
    enable_short: bool = Field(default=True, description="Allow short selling in backtest")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: Optional[str] = Field(default=None, description="Log file path")


class SystemConfig(BaseModel):
    """System configuration."""
    refresh_interval: int = Field(default=60, gt=0, description="Refresh interval in seconds")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    account: str = Field(default="", description="tqsdk account")
    password: str = Field(default="", description="tqsdk password")


class Config(BaseSettings):
    """Root configuration for tquant."""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    tqsdk: TQSDKConfig = Field(default_factory=TQSDKConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    backtest: Optional[BacktestConfig] = Field(default=None, description="Backtest configuration")

    model_config = SettingsConfigDict(
        env_prefix="TQUANT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        json_file=None,
        yaml_file=None,
    )

    @property
    def config_dir(self) -> Path:
        """Get configuration directory."""
        return Path.home() / ".tquant"

    @property
    def config_file(self) -> Path:
        """Get configuration file path."""
        return self.config_dir / "config.json"

    @field_validator("trading")
    @classmethod
    def validate_trading(cls, v: TradingConfig) -> TradingConfig:
        """Validate trading configuration."""
        if not v.symbols:
            raise ValueError("At least one trading symbol must be specified")
        return v

    @field_validator("llm")
    @classmethod
    def validate_llm(cls, v: LLMConfig) -> LLMConfig:
        """Validate LLM configuration."""
        if not v.gpt4o and not v.gpt4o_mini:
            raise ValueError("At least one LLM model must be configured")
        return v
