from __future__ import annotations
import logging
import mimetypes
from threading import Thread
from flask import Flask, send_from_directory
from .routes.api import api_bp
from .boot import autoload_initial_dataset

log = logging.getLogger(__name__)

def create_app():
    # Make sure .js is served as application/javascript (Safari cares)
    mimetypes.add_type("application/javascript", ".js")

    app = Flask(
        __name__,
        static_folder="static",       # app/static
        static_url_path="/static",    # served at /static/...
        template_folder="templates",  # optional; fine to keep
    )

    # Helpful in dev to avoid stale cached JS/CSS
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSON_AS_ASCII"] = False

    # Kick off dataset load in a background thread (non-blocking)
    def _bg_load():
        try:
            status = autoload_initial_dataset()
            app.logger.info("Dataset autoload status: %s", status)
        except Exception as e:
            app.logger.error("Dataset autoload failed: %s", e)

    Thread(target=_bg_load, daemon=True).start()

    # API routes
    app.register_blueprint(api_bp)

    # Serve the SPA
    @app.get("/")
    def root():
        return send_from_directory(app.static_folder, "index.html")

    # (Optional) SPA fallback for client-side routes:
    # @app.get("/<path:path>")
    # def spa_fallback(path):
    #     return send_from_directory(app.static_folder, "index.html")

    return app
