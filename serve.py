from __future__ import annotations

import uvicorn

from runtime.settings import get_settings


def main() -> int:
    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.app.api_host,
        port=settings.app.api_port,
        log_level=settings.app.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
