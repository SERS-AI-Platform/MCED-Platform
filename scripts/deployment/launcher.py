#!/usr/bin/env python3
"""
SERS Clinical Webapp - Desktop Launcher

PyInstaller로 .exe 빌드 시 사용되는 런처.
서버 시작 → 브라우저 자동 열기 → 종료 시 정리
"""

import os
import sys
import time
import socket
import webbrowser
import threading
from pathlib import Path

# Determine bundle location (PyInstaller _MEIPASS or source dir)
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parents[2]
    APP_DIR = BUNDLE_DIR

# Ensure project root on path
sys.path.insert(0, str(BUNDLE_DIR))

HOST = "127.0.0.1"
PORT = 8080
URL = f"http://{HOST}:{PORT}"


def find_free_port(start: int = 8080) -> int:
    """Find a free port starting from `start`."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    return start


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll the server until it responds or timeout."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def open_browser_when_ready(url: str):
    """Open browser after server starts."""
    if wait_for_server(url):
        print(f"\n  Server ready! Opening {url} ...\n")
        webbrowser.open(url)
    else:
        print(f"\n  WARNING: Server did not respond in time. Open manually: {url}\n")


def print_banner():
    print("=" * 60)
    print("  SERS 암 선별검사 시스템 (Clinical Webapp)")
    print("  SOLUM Healthcare")
    print("=" * 60)
    print(f"  서버 주소: {URL}")
    print("  창을 닫으면 서버가 종료됩니다.")
    print("=" * 60)
    print()


def main():
    global PORT, URL

    print_banner()

    # Find free port
    PORT = find_free_port(8080)
    URL = f"http://{HOST}:{PORT}"

    # Set working directory to app bundle (so DB and templates are found)
    if getattr(sys, "frozen", False):
        os.chdir(APP_DIR)

    # Schedule browser open
    threading.Thread(target=open_browser_when_ready, args=(URL,), daemon=True).start()

    # Start uvicorn server (blocks until Ctrl+C / window close)
    try:
        import uvicorn
        from scripts.deployment.sers_clinical_webapp import app

        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n\n  Server stopped by user.")
    except Exception as e:
        print(f"\n\n  ERROR: {e}")
        input("\n  Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
