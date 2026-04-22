from __future__ import annotations

from runtime.settings import AppSettings, RuntimeSettings, Settings

import serve


def test_serve_main_uses_settings_for_uvicorn(monkeypatch):
    captured: dict[str, object] = {}
    settings = Settings(
        app=AppSettings(api_host="127.0.0.1", api_port=9000, log_level="DEBUG"),
        runtime=RuntimeSettings(
            deepseek_api_key="key",
            deepseek_base_url="https://api.example.com",
            deepseek_model="deepseek-chat",
        ),
    )

    monkeypatch.setattr(serve, "get_settings", lambda: settings)
    monkeypatch.setattr(
        serve.uvicorn,
        "run",
        lambda app, *, host, port, log_level: captured.update(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        ),
    )

    assert serve.main() == 0
    assert captured == {
        "app": "api.app:app",
        "host": "127.0.0.1",
        "port": 9000,
        "log_level": "debug",
    }
