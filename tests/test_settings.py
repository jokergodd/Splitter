from __future__ import annotations

from pathlib import Path

import pytest

import runtime.settings as runtime_settings


def _set_required_deepseek_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")


def test_load_settings_uses_defaults_with_required_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _set_required_deepseek_env(monkeypatch)

    settings = runtime_settings.load_settings()

    assert settings.app.app_name == runtime_settings.DEFAULT_APP_NAME
    assert settings.app.app_version == runtime_settings.DEFAULT_APP_VERSION
    assert settings.app.api_host == runtime_settings.DEFAULT_API_HOST
    assert settings.app.api_port == runtime_settings.DEFAULT_API_PORT
    assert settings.app.log_level == runtime_settings.DEFAULT_LOG_LEVEL
    assert settings.app.task_max_workers == runtime_settings.DEFAULT_TASK_MAX_WORKERS
    assert settings.runtime.deepseek_api_key == "test-key"
    assert settings.runtime.deepseek_base_url == "https://api.example.com"
    assert settings.runtime.deepseek_model == "deepseek-chat"


def test_load_settings_reads_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _set_required_deepseek_env(monkeypatch)
    monkeypatch.setenv("APP_NAME", "Custom API")
    monkeypatch.setenv("APP_VERSION", "2.0.0")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("TASK_MAX_WORKERS", "8")
    monkeypatch.setenv("DENSE_MODEL_NAME", "dense-x")
    monkeypatch.setenv("SPARSE_MODEL_NAME", "sparse-y")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "reranker-z")

    settings = runtime_settings.load_settings()

    assert settings.app.app_name == "Custom API"
    assert settings.app.app_version == "2.0.0"
    assert settings.app.api_host == "127.0.0.1"
    assert settings.app.api_port == 9000
    assert settings.app.log_level == "DEBUG"
    assert settings.app.task_max_workers == 8
    assert settings.runtime.dense_model_name == "dense-x"
    assert settings.runtime.sparse_model_name == "sparse-y"
    assert settings.runtime.reranker_model_name == "reranker-z"


@pytest.mark.parametrize(
    "missing_key",
    ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"],
)
def test_load_settings_requires_deepseek_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_key: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    values = {
        "DEEPSEEK_API_KEY": "test-key",
        "DEEPSEEK_BASE_URL": "https://api.example.com",
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    values.pop(missing_key)
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match=missing_key):
        runtime_settings.load_settings()
