"""Factories for chat model initialization."""

from __future__ import annotations

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI


def resolve_provider(model_cfg) -> str:
    """Infer the chat-model provider from configuration."""

    provider = getattr(model_cfg, "provider", "openai")
    if provider != "auto":
        return provider

    text = (
        f"{getattr(model_cfg, 'ai_base_url', '')} "
        f"{getattr(model_cfg, 'model_name', '')}"
    ).lower()
    if "deepseek" in text:
        return "deepseek"
    return "openai"


def build_chat_model(model_cfg):
    """Build the configured chat model."""

    provider = resolve_provider(model_cfg)
    common_kwargs = {
        "model": model_cfg.model_name,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "timeout": model_cfg.timeout_seconds,
    }

    if provider == "openai":
        return ChatOpenAI(
            api_key=model_cfg.api_key,
            base_url=model_cfg.ai_base_url or None,
            **common_kwargs,
        )

    if provider == "deepseek":
        if ChatDeepSeek is None:
            raise RuntimeError(
                "配置了 provider=deepseek，但未安装 langchain-deepseek"
            )

        kwargs = {
            "model": model_cfg.model_name,
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
            "timeout": model_cfg.timeout_seconds,
        }
        if getattr(model_cfg, "api_key", None):
            kwargs["api_key"] = model_cfg.api_key
        if getattr(model_cfg, "ai_base_url", None):
            kwargs["base_url"] = model_cfg.ai_base_url

        return ChatDeepSeek(**kwargs)

    raise ValueError(f"Unsupported model provider: {provider}")
