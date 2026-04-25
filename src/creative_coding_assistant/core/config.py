"""Application settings."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationProviderName(StrEnum):
    OPENAI = "openai"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = Field(default="Creative Coding Assistant")
    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    artifact_dir: Path = Field(default=Path("data/artifacts"))
    default_domain: str = Field(default="three_js")
    default_mode: str = Field(default="generate")
    default_generation_provider: GenerationProviderName = Field(
        default=GenerationProviderName.OPENAI
    )
    openai_model: str = Field(default="gpt-5-mini", min_length=1)
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "CCA_OPENAI_API_KEY"),
    )

    model_config = SettingsConfigDict(
        env_prefix="CCA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("openai_model")
    @classmethod
    def normalize_openai_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_embedding_model")
    @classmethod
    def normalize_openai_embedding_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_openai_api_key(
        cls,
        value: SecretStr | str | None,
    ) -> str | None:
        if value is None:
            return None

        raw_value = (
            value.get_secret_value()
            if isinstance(value, SecretStr)
            else str(value)
        ).strip()
        return raw_value or None

    @property
    def has_openai_api_key(self) -> bool:
        return self.get_openai_api_key() is not None

    @property
    def has_openai_embedding_config(self) -> bool:
        return self.has_openai_api_key and bool(self.openai_embedding_model)

    def get_openai_api_key(self) -> str | None:
        if self.openai_api_key is None:
            return None
        secret_value = self.openai_api_key.get_secret_value().strip()
        return secret_value or None


def load_settings() -> Settings:
    """Load settings without creating external resources."""

    return Settings()
