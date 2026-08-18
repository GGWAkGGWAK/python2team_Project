"""Development entry point."""

import os

from cardocr import create_app

app = create_app()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=debug,
        use_reloader=False,
        threaded=True,
    )
