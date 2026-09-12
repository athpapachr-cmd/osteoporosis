"""Loopback-only transport for the shared Knee-OA product projection.

Run from a repository checkout: python clinic_utilities/physio_referral_product/prototype/server.py
This HTTP server is synthetic/local only and is never mounted in production.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clinic_utilities.physio_referral_product.knee_oa_projection import (
    PACKAGE,
    PRODUCT,
    E,
    T,
    C,
    ENGINE,
    LANG,
    SCOPE,
    HIDDEN,
    CATEGORIES,
    FLAGS,
    SAFETY_LABELS,
    MIXED_COPY,
    load_yaml,
    render,
    check,
    short_text,
    clean_request,
    suggestion_draft,
    project,
    bootstrap,
)

HERE = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    server_version = "PhysioPrototype"

    def log_message(self, *_args):
        pass

    def allowed_host(self) -> bool:
        host = f"127.0.0.1:{self.server.server_port}"
        return self.headers.get("Host") == host and self.client_address[0] == "127.0.0.1"

    def reply(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, code: int, data: dict) -> None:
        self.reply(code, json.dumps(data, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if not self.allowed_host():
            return self.json_reply(403, {"error": "local_access_only"})
        if self.path == "/api/bootstrap":
            return self.json_reply(200, bootstrap())
        files = {
            "/": ("index.html", "text/html"),
            "/app.js": ("app.js", "text/javascript"),
            "/qualifiers.js": ("qualifiers.js", "text/javascript"),
            "/styles.css": ("styles.css", "text/css"),
            "/qualifiers.css": ("qualifiers.css", "text/css"),
        }
        if self.path not in files:
            return self.json_reply(404, {"error": "not_found"})
        name, mime = files[self.path]
        data = (HERE / name).read_bytes()
        if name == "qualifiers.js":
            data += b"\n" + (HERE / "more_v3.js").read_bytes()
        elif name == "qualifiers.css":
            data += b"\n" + (HERE / "more_v3.css").read_bytes()
        self.reply(200, data, mime + "; charset=utf-8")

    def do_POST(self):
        expected_origin = f"http://127.0.0.1:{self.server.server_port}"
        if (not self.allowed_host() or self.headers.get("Origin") not in {None, expected_origin}
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
                or self.headers.get("X-Physio-Prototype") != "1"):
            return self.json_reply(403, {"error": "local_access_only"})
        if self.path != "/api/project" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.json_reply(400, {"error": "invalid_prototype_request"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            check(0 < length <= 32768)
            result = project(json.loads(self.rfile.read(length)))
            self.json_reply(200, result)
        except (ValueError, TypeError, KeyError, AssertionError, UnicodeError):
            self.json_reply(400, {"error": "invalid_or_stale_prototype_request"})
        except Exception:
            self.json_reply(500, {"error": "prototype_unavailable"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic local prototype; never enter real patient data.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Synthetic Knee-OA prototype: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
