#!/usr/bin/env python3
"""One-command launcher for the ContextCache server + web UI.

Usage:
  python scripts/serve/launch.py                # Live mode (requires GPU)
  python scripts/serve/launch.py --demo         # Demo mode (no GPU needed)
  python scripts/serve/launch.py --port 8080
  python scripts/serve/launch.py --no-browser
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVE_SCRIPT = Path(__file__).resolve().parent / "serve_context_cache.py"


def check_dependencies(demo=False):
    """Check that required packages are installed."""
    required = {"fastapi": "FastAPI", "uvicorn": "Uvicorn", "pydantic": "Pydantic"}
    if not demo:
        required.update({"torch": "PyTorch", "transformers": "Transformers"})

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
    parser = argparse.ArgumentParser(description="Launch ContextCache server with web UI")
    parser.add_argument("--port", type=int, default=8421, help="Server port (default: 8421)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "context_cache_config.yaml")
    parser.add_argument("--preload-tools", type=Path, default=None)
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on startup")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no GPU, simulated responses)")
    args = parser.parse_args()

    if not check_dependencies(demo=args.demo):
        sys.exit(1)

    if not args.demo and not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    mode = "DEMO" if args.demo else "LIVE"
    url = f"http://localhost:{args.port}"
    print()
    print("=" * 50)
    print(f"  ContextCache Server [{mode}]")
    print("=" * 50)
    if not args.demo:
        print(f"  Config:  {args.config}")
    print(f"  URL:     {url}")
    if args.demo:
        print(f"  No GPU needed - using pre-recorded responses")
    else:
        print(f"  The model will load in the background (~30s).")
    print("=" * 50)
    print()

    if not args.no_browser:
        webbrowser.open(url)

    cmd = [sys.executable, str(SERVE_SCRIPT), "--host", args.host, "--port", str(args.port)]
    if args.demo:
        cmd.append("--demo")
    else:
        cmd.extend(["--config", str(args.config)])
    if args.preload_tools:
        cmd.extend(["--preload-tools", str(args.preload_tools)])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
