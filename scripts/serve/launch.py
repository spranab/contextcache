#!/usr/bin/env python3
"""One-command launcher for the ContextCache server + web UI.

Usage:
  python scripts/serve/launch.py
  python scripts/serve/launch.py --port 8080
  python scripts/serve/launch.py --no-browser
  python scripts/serve/launch.py --preload-tools scripts/serve/sample_tools.json
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVE_SCRIPT = Path(__file__).resolve().parent / "serve_context_cache.py"


def check_dependencies():
    """Check that required packages are installed."""
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
    }
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"  {name} ({module})")

    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(m)
        print(f"\nInstall with: pip install {' '.join(m.split('(')[1].rstrip(')') for m in missing)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Launch ContextCache server with web UI",
    )
    parser.add_argument("--port", type=int, default=8421, help="Server port (default: 8421)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "context_cache_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--preload-tools", type=Path, default=None,
        help="Path to a tools JSON file to preload on startup",
    )
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on startup")
    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check config exists
    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    print()
    print("=" * 50)
    print("  ContextCache Server")
    print("=" * 50)
    print(f"  Config:  {args.config}")
    print(f"  URL:     {url}")
    if args.preload_tools:
        print(f"  Preload: {args.preload_tools}")
    print()
    print("  The model will load in the background (~30s).")
    print("  The web UI will show a loading screen until ready.")
    print("=" * 50)
    print()

    # Open browser
    if not args.no_browser:
        webbrowser.open(url)

    # Build command
    cmd = [
        sys.executable, str(SERVE_SCRIPT),
        "--host", args.host,
        "--port", str(args.port),
        "--config", str(args.config),
    ]
    if args.preload_tools:
        cmd.extend(["--preload-tools", str(args.preload_tools)])

    # Run server (blocks until Ctrl+C)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
