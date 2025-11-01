# app/__init__.py

from flask import Flask, send_from_directory
from threading import Thread
import logging

from .boot import autoload_initial_dataset
from .state import set_dataset_info

def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="/")
    app.logger.setLevel(logging.INFO)

    # Background dataset loader
    def _bg_load():
        try:
            set_dataset_info(
                source="remote_csv",
                ok=False,
                loading=True,
                label="Downloading dataset… please wait",
            )
            result = autoload_initial_dataset()
            result["loading"] = False
            set_dataset_info(**result)
            app.logger.info("Dataset autoload result: %s", result)
        except Exception as e:
            set_dataset_info(
                source="error",
                ok=False,
                loading=False,
                label=f"Dataset load failed: {e}",
            )
            app.logger.exception("Dataset autoload failed: %s", e)

    Thread(target=_bg_load, daemon=True).start()

    # Register routes
    from .routes.api import api_bp
    app.register_blueprint(api_bp)

    @app.get("/")
    def root():
        return send_from_directory(app.static_folder, "index.html")

    return app
