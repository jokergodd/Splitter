from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory


@dataclass(slots=True)
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str


def load_deepseek_config() -> DeepSeekConfig:
    try:
        from runtime.settings import load_settings

        settings = load_settings()
        return DeepSeekConfig(
            api_key=settings.runtime.deepseek_api_key,
            base_url=settings.runtime.deepseek_base_url,
            model=settings.runtime.deepseek_model,
        )
    except Exception:
        load_dotenv(dotenv_path=Path.cwd() / ".env")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model = os.getenv("DEEPSEEK_MODEL")

        missing = [
            name
            for name, value in (
                ("DEEPSEEK_API_KEY", api_key),
                ("DEEPSEEK_BASE_URL", base_url),
                ("DEEPSEEK_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required DeepSeek config: {', '.join(missing)}")

        return DeepSeekConfig(api_key=api_key, base_url=base_url, model=model)


def build_deepseek_llm(config: DeepSeekConfig) -> ChatDeepSeek:
    return ChatDeepSeek(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )


def build_deepseek_openai_client(config: DeepSeekConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


def build_deepseek_async_openai_client(config: DeepSeekConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


def build_ragas_eval_llm(config: DeepSeekConfig):
    client = build_deepseek_async_openai_client(config)
    return llm_factory(config.model, client=client)
