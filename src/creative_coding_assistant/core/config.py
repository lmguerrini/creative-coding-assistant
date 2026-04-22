"""Application settings for the bootstrap foundation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings that avoid side effects during import."""

    app_name: str = "Creative Coding Assistant"
    environment: str = "local"
    log_level: str = "INFO"
    chroma_persist_dir: Path = Path("data/chroma")
    artifact_dir: Path = Path("data/artifacts")
    default_domain: str = "three_js"
    default_mode: str = "generate"


def load_settings() -> Settings:
    """Load settings from environment variables with conservative defaults."""

    return Settings(
        app_name=os.getenv("CCA_APP_NAME", Settings.app_name),
        environment=os.getenv("CCA_ENVIRONMENT", Settings.environment),
        log_level=os.getenv("CCA_LOG_LEVEL", Settings.log_level).upper(),
        chroma_persist_dir=Path(
            os.getenv("CCA_CHROMA_PERSIST_DIR", str(Settings.chroma_persist_dir))
        ),
        artifact_dir=Path(os.getenv("CCA_ARTIFACT_DIR", str(Settings.artifact_dir))),
        default_domain=os.getenv("CCA_DEFAULT_DOMAIN", Settings.default_domain),
        default_mode=os.getenv("CCA_DEFAULT_MODE", Settings.default_mode),
    )
