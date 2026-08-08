"""Business card OCR Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from .database import Database


def create_app(test_config: dict | None = None) -> Flask:
    package_root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(package_root / "templates"),
        static_folder=str(package_root / "static"),
        static_url_path="/static",
    )
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=str(Path(app.instance_path) / "business_cards.db"),
        SCAN_DIR=str(Path(app.instance_path) / "scans"),
        MAX_CONTENT_LENGTH=15 * 1024 * 1024,
    )

    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["SCAN_DIR"]).mkdir(parents=True, exist_ok=True)

    database = Database(app.config["DATABASE"])
    database.initialize()
    app.extensions["database"] = database

    from .routes import web

    app.register_blueprint(web)
    return app


__all__ = ["create_app"]
