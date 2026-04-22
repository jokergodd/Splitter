from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path

import pytest

from rag_demo.llm import (
    DeepSeekConfig,
    build_deepseek_async_openai_client,
    build_deepseek_openai_client,
    build_deepseek_llm,
    build_ragas_eval_llm,
    load_deepseek_config,
)


def test_deepseek_config_is_a_dataclass():
    config = DeepSeekConfig(
        api_key="test-key",
        base_url="https://api.example.com",
        model="deepseek-chat",
    )

    assert is_dataclass(config)
    assert config.api_key == "test-key"
    assert config.base_url == "https://api.example.com"
    assert config.model == "deepseek-chat"


@pytest.mark.parametrize(
    "missing_key",
    ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"],
)
def test_load_deepseek_config_raises_when_a_required_value_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_key: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    env_lines = {
        "DEEPSEEK_API_KEY": "file-key",
        "DEEPSEEK_BASE_URL": "https://file.example.com",
        "DEEPSEEK_MODEL": "file-model",
    }
    env_lines.pop(missing_key)
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in env_lines.items()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=missing_key):
        load_deepseek_config()


def test_load_deepseek_config_reads_from_env_and_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=file-key\n"
        "DEEPSEEK_BASE_URL=https://file.example.com\n"
        "DEEPSEEK_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    config = load_deepseek_config()

    assert config == DeepSeekConfig(
        api_key="file-key",
        base_url="https://file.example.com",
        model="file-model",
    )


def test_load_deepseek_config_reads_from_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")

    config = load_deepseek_config()

    assert config == DeepSeekConfig(
        api_key="env-key",
        base_url="https://env.example.com",
        model="env-model",
    )


def test_build_deepseek_llm_passes_config_through_to_chatdeepseek(monkeypatch):
    captured: dict[str, str] = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("rag_demo.llm.ChatDeepSeek", FakeChatDeepSeek)

    llm = build_deepseek_llm(
        DeepSeekConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            model="deepseek-chat",
        )
    )

    assert isinstance(llm, FakeChatDeepSeek)
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://api.example.com",
        "model": "deepseek-chat",
    }


def test_build_deepseek_openai_client_passes_config_through(monkeypatch):
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("rag_demo.llm.OpenAI", FakeOpenAI)

    client = build_deepseek_openai_client(
        DeepSeekConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            model="deepseek-chat",
        )
    )

    assert isinstance(client, FakeOpenAI)
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://api.example.com",
    }


def test_build_ragas_eval_llm_uses_openai_compatible_client_and_model(monkeypatch):
    fake_client = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "rag_demo.llm.build_deepseek_async_openai_client",
        lambda config: fake_client,
    )
    monkeypatch.setattr(
        "rag_demo.llm.llm_factory",
        lambda model, *, client: captured.update({"model": model, "client": client})
        or "ragas-llm",
    )

    llm = build_ragas_eval_llm(
        DeepSeekConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            model="deepseek-chat",
        )
    )

    assert llm == "ragas-llm"
    assert captured == {
        "model": "deepseek-chat",
        "client": fake_client,
    }


def test_build_deepseek_async_openai_client_passes_config_through(monkeypatch):
    captured: dict[str, str] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("rag_demo.llm.AsyncOpenAI", FakeAsyncOpenAI)

    client = build_deepseek_async_openai_client(
        DeepSeekConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            model="deepseek-chat",
        )
    )

    assert isinstance(client, FakeAsyncOpenAI)
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://api.example.com",
    }
