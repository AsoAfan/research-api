"""Expose the spine-clarity app via a Cloudflare quick tunnel.

Tunnels only the Vite dev server (default :8080). Vite's `server.proxy` forwards
backend paths (/health, /cases, /analyze) to FastAPI on :8000, so a single
public `https://<random>.trycloudflare.com` URL serves the whole app.

Usage:
    python tunnel.py               # tunnel localhost:8080
    python tunnel.py --port 8080   # override port
    python tunnel.py --bin PATH    # use a specific cloudflared binary
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
TRYCF_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def find_or_install_cloudflared(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            sys.exit(f"--bin path does not exist: {explicit}")
        return explicit

    found = shutil.which("cloudflared")
    if found:
        return found

    cache = Path(__file__).resolve().parent / "bin" / "cloudflared"
    if cache.is_file():
        return str(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloudflared not found — downloading to {cache} ...", flush=True)
    tmp = cache.with_suffix(".part")
    with urllib.request.urlopen(CLOUDFLARED_URL) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp.rename(cache)
    print("downloaded.", flush=True)
    return str(cache)


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def print_url_banner(url: str) -> None:
    line = "─" * (len(url) + 4)
    print(f"\n┌{line}┐", flush=True)
    print(f"│  {url}  │", flush=True)
    print(f"└{line}┘\n", flush=True)
    print("Share this URL with remote users. Ctrl+C to stop.\n", flush=True)


def stream_output(proc: subprocess.Popen, on_url) -> None:
    seen = False
    assert proc.stderr is not None
    for raw in proc.stderr:
        line = raw.rstrip()
        if not seen:
            m = TRYCF_RE.search(line)
            if m:
                seen = True
                on_url(m.group(0))
        print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080, help="local port to tunnel (default: 8080)")
    ap.add_argument("--bin", dest="binary", default=None, help="path to a cloudflared binary")
    args = ap.parse_args()

    binary = find_or_install_cloudflared(args.binary)

    if not port_is_open(args.port):
        print(
            f"⚠  Nothing is listening on localhost:{args.port}.\n"
            f"   Start the frontend first:\n"
            f"     cd spine-clarity-view && npm run dev\n"
            f"   And the backend:\n"
            f"     uvicorn api.main:app --port 8000\n"
            f"   Then re-run this script.\n",
            file=sys.stderr,
        )
        return 1

    cmd = [binary, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{args.port}"]
    print("launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def shutdown(*_a):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))

    reader = threading.Thread(target=stream_output, args=(proc, print_url_banner), daemon=True)
    reader.start()

    try:
        rc = proc.wait()
    finally:
        shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
