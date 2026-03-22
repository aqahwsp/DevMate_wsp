"""LangSmith integration helpers."""

from __future__ import annotations

import logging
import os

from devmate.config import AppConfig, is_config_secret_set

LOGGER = logging.getLogger(__name__)


def configure_langsmith(config: AppConfig) -> None:
    """Set LangSmith-related environment variables from config."""

    langsmith = config.langsmith
    if not langsmith.enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        LOGGER.info("LangSmith tracing is disabled in config")
        return

    api_key = langsmith.langsmith_api_key or langsmith.langchain_api_key
    if is_config_secret_set(api_key):
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
    else:
        LOGGER.warning(
            "LangSmith is enabled but no valid API key is configured; "
            "tracing will be unavailable"
        )

    if langsmith.langchain_tracing_v2:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    os.environ["LANGSMITH_PROJECT"] = langsmith.project
    os.environ["LANGCHAIN_PROJECT"] = langsmith.project
    os.environ["LANGSMITH_ENDPOINT"] = langsmith.endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = langsmith.endpoint
