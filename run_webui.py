import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src_m.web import create_app
from src_m.web.config import get_web_config


def main():
    config = get_web_config()

    host = config.host
    port = config.port
    debug = config.debug

    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        elif arg == "--debug":
            debug = True

    print(f"PPC9 WebUI Server starting on http://{host}:{port}")
    print(f"Debug mode: {debug}")

    app = create_app("development" if debug else "production")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()