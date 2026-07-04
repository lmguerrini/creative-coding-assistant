"""Production WSGI entrypoint for Gunicorn or compatible WSGI servers."""

from __future__ import annotations

from creative_coding_assistant.api.dev_server import create_backend_dev_app
from creative_coding_assistant.core.config import load_settings
from creative_coding_assistant.core.logging import configure_logging

_settings = load_settings()
configure_logging(_settings.log_level, structured=_settings.structured_logging)

application = create_backend_dev_app(settings_factory=lambda: _settings)
