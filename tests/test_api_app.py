from __future__ import annotations

from api.app import create_app
from runtime.settings import AppSettings, RuntimeSettings, Settings


def test_create_app_uses_settings_title_and_version():
    settings = Settings(
        app=AppSettings(app_name="Custom Splitter", app_version="9.9.9"),
        runtime=RuntimeSettings(
            deepseek_api_key="key",
            deepseek_base_url="https://api.example.com",
            deepseek_model="deepseek-chat",
        ),
    )

    app = create_app(settings=settings)

    assert app.title == "Custom Splitter"
    assert app.version == "9.9.9"
    assert app.state.settings is settings
