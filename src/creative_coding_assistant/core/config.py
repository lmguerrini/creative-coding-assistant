"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = Field(default="Creative Coding Assistant")
    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    artifact_dir: Path = Field(default=Path("data/artifacts"))
    default_domain: str = Field(default="three_js")
    default_mode: str = Field(default="generate")

    model_config = SettingsConfigDict(
        env_prefix="CCA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()


def load_settings() -> Settings:
    """Load settings without creating external resources."""

    return Settings()
