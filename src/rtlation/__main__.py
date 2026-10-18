"""Start: python -m rtlation"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path

from .config import load_config
from .cuda import add_cuda_dll_dirs


def main() -> None:
    parser = argparse.ArgumentParser(prog="rtlation", description="RTlation - Real Time / Translation")
    parser.add_argument("--config", type=Path, help="Pfad zur config.yaml (Standard: aktueller Ordner)")
    parser.add_argument("--host", help="überschreibt server.host")
    parser.add_argument("--port", type=int, help="überschreibt server.port")
    parser.add_argument("--open", action="store_true", help="Oberfläche im Browser öffnen")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    # Diese Bibliotheken melden sonst jede einzelne Anfrage.
    for noisy in ("httpx", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(args.config)
    host = args.host or config.server.host
    port = args.port or config.server.port

    # Vor dem Import von faster-whisper, damit die GPU-Bibliotheken gefunden werden.
    add_cuda_dll_dirs(config.root)

    import uvicorn

    from .server import create_app

    if args.open:
        shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        threading.Timer(1.5, webbrowser.open, [f"http://{shown_host}:{port}"]).start()

    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
