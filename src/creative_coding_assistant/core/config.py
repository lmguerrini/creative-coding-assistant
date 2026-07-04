"""Application settings."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationProviderName(StrEnum):
    OPENAI = "openai"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = Field(default="Creative Coding Assistant")
    environment: str = Field(
        default="local",
        validation_alias=AliasChoices(
            "CCA_ENVIRONMENT",
            "CCA_API_ENVIRONMENT",
            "CCA_RUNTIME_ENVIRONMENT",
        ),
    )
    log_level: str = Field(default="INFO")
    log_format: Literal["text", "json"] = Field(default="text")
    cors_allowed_origins: tuple[str, ...] = Field(
        default=("*",),
        validation_alias=AliasChoices(
            "CCA_CORS_ALLOWED_ORIGINS",
            "CCA_CORS_ALLOW_ORIGINS",
            "CCA_ALLOWED_ORIGINS",
        ),
    )
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    artifact_dir: Path = Field(default=Path("data/artifacts"))
    workspace_session_db_path: Path = Field(
        default=Path("data/workspace_sessions.sqlite3"),
        validation_alias=AliasChoices(
            "CCA_WORKSPACE_SESSION_DB_PATH",
            "CCA_WORKSPACE_DB_PATH",
        ),
    )
    eval_data_path: Path = Field(default=Path("data/eval/live_sessions.jsonl"))
    eval_ragas_results_path: Path = Field(default=Path("data/eval/ragas_results.jsonl"))
    eval_ragas_model: str = Field(default="gpt-4o-mini", min_length=1)
    eval_ragas_timeout_seconds: int = Field(default=180, ge=1)
    eval_ragas_max_retries: int = Field(default=2, ge=0)
    eval_ragas_max_workers: int = Field(default=2, ge=1)
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
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
            "CCA_LANGSMITH_TRACING",
        ),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "CCA_LANGSMITH_API_KEY",
        ),
        exclude=True,
    )
    langsmith_project: str = Field(
        default="creative-coding-assistant",
        min_length=1,
        validation_alias=AliasChoices(
            "LANGSMITH_PROJECT",
            "LANGCHAIN_PROJECT",
            "CCA_LANGSMITH_PROJECT",
        ),
    )
    langsmith_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGSMITH_ENDPOINT",
            "LANGCHAIN_ENDPOINT",
            "CCA_LANGSMITH_ENDPOINT",
        ),
    )
    langsmith_timeout_ms: int = Field(default=1500, ge=100)
    langsmith_sampling_rate: float = Field(default=1.0, ge=0, le=1)

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

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        return normalized or "local"

    @field_validator("log_format")
    @classmethod
    def normalize_log_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"text", "json"}:
            raise ValueError("CCA_LOG_FORMAT must be text or json.")
        return normalized

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def normalize_cors_allowed_origins(
        cls,
        value: str | tuple[str, ...] | list[str] | None,
    ) -> tuple[str, ...]:
        if value is None:
            return ("*",)
        if isinstance(value, str):
            origins = tuple(origin.strip() for origin in value.split(","))
        else:
            origins = tuple(str(origin).strip() for origin in value)
        normalized = tuple(origin for origin in origins if origin)
        return normalized or ("*",)

    @field_validator("openai_model")
    @classmethod
    def normalize_openai_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_embedding_model")
    @classmethod
    def normalize_openai_embedding_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("eval_ragas_model")
    @classmethod
    def normalize_eval_ragas_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("langsmith_project", mode="before")
    @classmethod
    def normalize_langsmith_project(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("langsmith_endpoint", mode="before")
    @classmethod
    def normalize_langsmith_endpoint(cls, value: object | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("openai_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def normalize_api_key(
        cls,
        value: SecretStr | str | None,
    ) -> str | None:
        if value is None:
            return None

        raw_value = (
            value.get_secret_value() if isinstance(value, SecretStr) else str(value)
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

    @property
    def has_langsmith_api_key(self) -> bool:
        return self.get_langsmith_api_key() is not None

    @property
    def structured_logging(self) -> bool:
        return self.log_format == "json"

    def get_langsmith_api_key(self) -> str | None:
        if self.langsmith_api_key is None:
            return None
        secret_value = self.langsmith_api_key.get_secret_value().strip()
        return secret_value or None


def load_settings() -> Settings:
    """Load settings without creating external resources."""

    return Settings()
