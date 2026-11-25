"""Serve the viewer on this machine: the page (``static/``) and its scene (``scene.json``, gzipped).

    python -m lumen.viewer tank.gdml --open                          # any GDML, its detector volume guessed
    python -m lumen.viewer tank.gdml --recipe my_recipes.py:TANK     # with a recipe of your own
    python -m lumen.viewer lumen_wc:sk                               # a detector package's detector
    python -m lumen.viewer tank.gdml --out tank.view.json            # write the scene; --scene serves it again

The server binds 127.0.0.1 by default, before it builds the scene, so a port in use is reported at once. It serves the
page only under a random token in its address (``http://127.0.0.1:8765/<token>/``): another user of the machine,
who cannot read the address, gets nothing.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import hmac
import json
import re
import secrets
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
PORT = 8765  # the page's default port on this machine
GZIP_LEVEL = 6  # the scene's compression: most of level 9's size at a fraction of its time
TOKEN_BYTES = 16  # the token's random bytes (secrets.token_urlsafe): 22 characters of the address
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def encode(payload) -> bytes:
    """Return the scene as JSON bytes (NaN-free: a non-finite number would not parse in the browser)."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False, default=_default).encode()


def _default(x):
    import numpy as np

    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON: {type(x).__name__}")


def _same(a: str, b: str) -> bool:
    """Return whether two strings are equal, compared in constant time (one of them holds the token)."""
    return hmac.compare_digest(a.encode(), b.encode())


def handler(scene_gz: bytes, token: str):
    """Return a request handler serving the page and the gzipped scene under ``/<token>/``, and nothing else."""
    root = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- the http.server method name
            path = self.path.split("?", 1)[0]
            if _same(path, root[:-1]):  # the page's relative addresses need the final slash
                return self._redirect(root)
            if not _same(path[: len(root)], root):
                self.send_error(HTTPStatus.FORBIDDEN, "open the address the viewer printed: it holds the page's token")
                return None
            path = path[len(root) :] or "index.html"
            if path == "scene.json":
                sys.stderr.write(f"[lumen.viewer] scene sent to {self.client_address[0]}\n")
                return self._send(scene_gz, "application/json", gz=True)
            f = STATIC / path
            if f.parent != STATIC or not f.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
            return self._send(f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        def _redirect(self, location):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, body, ctype, gz=False):
            if gz and "gzip" not in self.headers.get("Accept-Encoding", ""):
                body, gz = gzip.decompress(body), False
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return None

    return Handler


def bind(host="127.0.0.1", port=PORT) -> ThreadingHTTPServer:
    """Return a server listening on ``host:port`` (port 0: a free one), which answers once ``serve`` gives it the page.

    Bound before the scene is built, a port in use is reported at once, not after minutes of building.
    """
    return ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)


def serve(httpd: ThreadingHTTPServer, scene_json: bytes, token: str, open_browser=False):
    """Serve the page and a scene (JSON bytes) on a bound server (``bind``), under ``/<token>/``, until interrupted."""
    httpd.RequestHandlerClass = handler(gzip.compress(scene_json, GZIP_LEVEL), token)
    host, port = httpd.server_address[:2]
    url = f"http://{host}:{port}/{token}/"
    sys.stderr.write(f"[lumen.viewer] {url}  (Ctrl-C stops)\n")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def _token(text: str) -> str:
    """Return a token given on the command line: letters, digits, '-' and '_' (it is a path of the address)."""
    if not re.fullmatch(r"[\w-]+", text, re.ASCII):
        raise argparse.ArgumentTypeError(f"{text!r}: letters, digits, '-' and '_' only")
    return text


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser (``python -m lumen.viewer --help``)."""
    ap = argparse.ArgumentParser(
        prog="python -m lumen.viewer",
        description="A browser viewer of a GDML file as "
        "lumen reads it: its geometry, and lumen's reading of every material and surface.",
    )
    ap.add_argument("source", nargs="?", help="a GDML file, or a detector package's detector (package:name)")
    ap.add_argument("--recipe", help="a recipe of your own for a GDML file: FILE.py:NAME")
    ap.add_argument("--data", help="a package detector's data directory (default LUMEN_DATA)")
    ap.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="the light's wavelength range (nm), shaded on every table",
    )
    ap.add_argument("--up", choices=("x", "y", "z"), default="z", help="the axis drawn vertical first (default z)")
    ap.add_argument("--max-instances", type=int, default=None, help="the placed copies drawn at most")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT, help=f"the port (default {PORT}; 0: a free one)")
    ap.add_argument("--token", type=_token, help="the token in the page's address (default: a new random one)")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    ap.add_argument("--out", help="write the scene (JSON) to this file instead of serving it")
    ap.add_argument("--scene", help="serve a scene written by --out")
    return ap


def main(argv=None) -> int:
    """Run the viewer (see the module docstring) on the command line ``argv``; return the exit status."""
    a = parser().parse_args(argv)
    if not a.scene and not a.source:
        sys.stderr.write("lumen.viewer: give a GDML file or a package's detector (or --scene)\n")
        return 2
    write = bool(a.out) and not a.scene
    httpd = None
    if not write:
        try:
            httpd = bind(a.host, a.port)
        except OSError as e:
            hint = " (pick another with --port)" if e.errno == errno.EADDRINUSE else ""
            sys.stderr.write(f"lumen.viewer: cannot listen on {a.host}:{a.port}: {e.strerror or e}{hint}\n")
            return 2
    try:
        data = Path(a.scene).read_bytes() if a.scene else _scene(a)
        if write:
            Path(a.out).write_bytes(data)
            sys.stderr.write(f"[lumen.viewer] wrote {a.out} ({len(data) / 2**20:.1f} MiB)\n")
        else:
            serve(httpd, data, a.token or secrets.token_urlsafe(TOKEN_BYTES), a.open)
    finally:
        if httpd is not None:
            httpd.server_close()
    return 0


def _scene(a) -> bytes:
    """Return the scene of the command line's source (JSON bytes), with what failed and how long it took."""
    from lumen.viewer.scene import MAX_INSTANCES, build

    sys.stderr.write(f"[lumen.viewer] reading {a.source} as lumen reads it ...\n")
    view = build(a.source, a.recipe, data=a.data, max_instances=a.max_instances or MAX_INSTANCES, up=a.up,
                 wavelength_range=a.range)  # fmt: skip
    for stage, text in view.errors:
        sys.stderr.write(f"[lumen.viewer] {stage}: {text}\n")
    meta = view.payload.get("meta", {})
    sys.stderr.write(f"[lumen.viewer] {meta.get('counts', {}).get('instances', 0):,} placed volumes, "
                     f"built in {meta.get('seconds', '?')} s\n")  # fmt: skip
    return encode(view.payload)
