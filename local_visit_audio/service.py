from __future__ import annotations

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, RLock, Timer
from typing import Callable
from urllib import request
import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import time

from .pipeline import (
    LocalAudioError,
    LocalAudioPipeline,
    MAX_AUDIO_BYTES,
    MODEL_NAME,
    MODEL_SHA256,
    PipelineConfig,
    _digest,
    normalize_ollama_digest,
    private_request_dir,
    verify_environment,
)


HOST = "127.0.0.1"
PORT = 8765
ORIGIN = f"http://{HOST}:{PORT}"
PRODUCTION_ORIGIN = "https://ortho-reception-backend.onrender.com"
PRODUCTION_CAPTURE = PRODUCTION_ORIGIN + "/static/cockpit/visit-capture/index.html"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22}$")


class DedicatedOllama:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.control_lock = RLock()

    def start(self) -> None:
        with self.control_lock:
            self._start_unlocked()

    def _start_unlocked(self) -> None:
        binary = Path("/opt/homebrew/Cellar/ollama/0.40.11/bin/ollama")
        receipt = self.config.support_dir / "model_receipt_v1.json"
        try:
            proof = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LocalAudioError("classification_failed") from exc
        if set(proof) != {"model", "source_gguf_sha256", "ollama_manifest_digest"}:
            raise LocalAudioError("classification_failed")
        if proof["model"] != MODEL_NAME or proof["source_gguf_sha256"] != MODEL_SHA256:
            raise LocalAudioError("classification_failed")
        try:
            proof_digest = normalize_ollama_digest(proof["ollama_manifest_digest"])
        except ValueError:
            raise LocalAudioError("classification_failed")
        source_gguf = self.config.support_dir / "models" / "llama-krikri-8b-instruct-q4_k_m.gguf"
        if not source_gguf.is_file() or _digest(source_gguf) != MODEL_SHA256:
            raise LocalAudioError("classification_failed")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise LocalAudioError("classification_failed")
        version = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        if version.returncode != 0 or "0.40.11" not in (version.stdout + version.stderr):
            raise LocalAudioError("classification_failed")
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex((HOST, 11435)) == 0:
                raise LocalAudioError("classification_failed")
        env = {
            "HOME": str(Path.home()),
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            "OLLAMA_HOST": "127.0.0.1:11435",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_DEBUG": "0",
            "OLLAMA_MODELS": str(self.config.support_dir / "ollama-models"),
        }
        self.process = subprocess.Popen(
            [str(binary), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise LocalAudioError("classification_failed")
                try:
                    with request.urlopen(self.config.ollama_url + "/api/version", timeout=1) as response:
                        body = json.loads(response.read(4096))
                    if body.get("version") != "0.40.11":
                        raise LocalAudioError("classification_failed")
                    with request.urlopen(self.config.ollama_url + "/api/tags", timeout=2) as response:
                        tags = json.loads(response.read(1024 * 1024))
                    installed = next(
                        (item for item in tags.get("models", []) if item.get("name") == MODEL_NAME),
                        None,
                    )
                    try:
                        installed_digest = normalize_ollama_digest(
                            installed.get("digest") if installed is not None else None
                        )
                    except ValueError:
                        raise LocalAudioError("classification_failed")
                    if installed_digest != proof_digest:
                        raise LocalAudioError("classification_failed")
                    return
                except LocalAudioError:
                    raise
                except Exception:
                    time.sleep(0.25)
            raise LocalAudioError("classification_failed")
        except Exception:
            self._stop_unlocked()
            raise

    def stop(self) -> None:
        with self.control_lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


class CompanionState:
    def __init__(
        self,
        pipeline: LocalAudioPipeline,
        *,
        cancel_hook: Callable[[], None] | None = None,
        recover_hook: Callable[[], None] | None = None,
        processing_timeout: float = 600.0,
    ) -> None:
        self.pipeline = pipeline
        self.sessions: dict[str, str] = {}
        self.job_lock = Lock()
        self.active_lock = Lock()
        self.active_job: tuple[str, str, str, Event, Event, Timer] | None = None
        self.cancel_hook = cancel_hook
        self.recover_hook = recover_hook
        self.processing_timeout = processing_timeout
        self.launcher_js = (Path(__file__).resolve().parent / "launcher.js").read_bytes()

    def new_session(self) -> tuple[str, str]:
        session_id = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        with self.active_lock:
            self.sessions[session_id] = csrf
            active_browser_session = self.active_job[0] if self.active_job else None
            while len(self.sessions) > 8:
                stale = next(
                    (candidate for candidate in self.sessions if candidate != active_browser_session),
                    None,
                )
                if stale is None:
                    break
                self.sessions.pop(stale, None)
        return session_id, csrf

    def session_csrf(self, session_id: str) -> str | None:
        with self.active_lock:
            return self.sessions.get(session_id)

    def begin_job(
        self, browser_session_id: str, local_session_id: str, request_id: str
    ) -> tuple[Event, Event]:
        cancel_event = Event()
        timeout_event = Event()
        timer = Timer(
            self.processing_timeout,
            self.timeout_job,
            args=(cancel_event, timeout_event),
        )
        timer.daemon = True
        with self.active_lock:
            if self.active_job is not None:
                raise LocalAudioError("invalid_request")
            self.active_job = (
                browser_session_id, local_session_id, request_id,
                cancel_event, timeout_event, timer,
            )
        timer.start()
        return cancel_event, timeout_event

    def _stop_active_backend(self) -> None:
        if self.cancel_hook is not None:
            try:
                self.cancel_hook()
            except Exception:
                pass

    def timeout_job(self, cancel_event: Event, timeout_event: Event) -> None:
        with self.active_lock:
            if self.active_job is None or self.active_job[3] is not cancel_event:
                return
            timeout_event.set()
            cancel_event.set()
        self._stop_active_backend()

    def cancel_job(
        self, browser_session_id: str, local_session_id: str, request_id: str
    ) -> bool:
        with self.active_lock:
            if self.active_job is None or self.active_job[:3] != (
                browser_session_id, local_session_id, request_id
            ):
                return False
            cancel_event = self.active_job[3]
            cancel_event.set()
        self._stop_active_backend()
        return True

    def finish_job(self, cancel_event: Event) -> tuple[bool, bool]:
        with self.active_lock:
            cancelled = cancel_event.is_set()
            timed_out = False
            if self.active_job is not None and self.active_job[3] is cancel_event:
                timed_out = self.active_job[4].is_set()
                self.active_job[5].cancel()
                self.sessions.pop(self.active_job[0], None)
                self.active_job = None
        return cancelled, timed_out


class CompanionHandler(BaseHTTPRequestHandler):
    server_version = "OrthoCockpitLocalAudio/1"
    protocol_version = "HTTP/1.1"

    @property
    def companion(self) -> CompanionState:
        return self.server.companion  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _host_ok(self) -> bool:
        return self.headers.get("Host") == f"{HOST}:{PORT}"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, request_id: str = "") -> None:
        if self.command == "POST":
            self.close_connection = True
        body = json.dumps({"code": code, "request_id": request_id}, separators=(",", ":")).encode()
        self._send(status, body, "application/json")

    def do_GET(self) -> None:
        if not self._host_ok():
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        if self.path == "/launcher.js":
            self._send(HTTPStatus.OK, self.companion.launcher_js, "text/javascript; charset=utf-8")
            return
        if self.path != "/":
            self._error(HTTPStatus.NOT_FOUND, "invalid_request")
            return
        session_id, csrf = self.companion.new_session()
        html = f"""<!doctype html>
<html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta name="csrf-token" content="{csrf}"><title>Τοπική καταγραφή επίσκεψης</title>
<style>body{{font:16px -apple-system;margin:3rem;max-width:760px;color:#153246}}button{{padding:.8rem 1rem;border:0;border-radius:.7rem;background:#0b6fb8;color:white;font-weight:700}}#status{{margin:1rem 0;padding:1rem;background:#eef6fb;border-radius:.8rem}}</style></head>
<body><h1>Τοπική καταγραφή επίσκεψης</h1><p>Ο ήχος και η μεταγραφή παραμένουν σε αυτόν τον Mac.</p>
<button id="openCockpit" type="button">Άνοιγμα ασφαλούς Visit Capture</button><p id="status" role="status">Έτοιμο.</p>
<script src="/launcher.js"></script></body></html>""".encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Set-Cookie", f"local_audio_session={session_id}; HttpOnly; SameSite=Strict; Path=/")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(html)

    def _session(self) -> tuple[str, str] | None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("local_audio_session")
        if morsel is None:
            return None
        session_id = morsel.value
        csrf = self.companion.session_csrf(session_id)
        if csrf is None:
            return None
        return session_id, csrf

    def do_POST(self) -> None:
        request_id = self.headers.get("X-Request-Id", "")
        if self.path not in {"/api/process", "/api/cancel"} or not self._host_ok():
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", request_id)
            return
        session = self._session()
        if (
            session is None
            or self.headers.get("Origin") != ORIGIN
            or self.headers.get("Sec-Fetch-Site") != "same-origin"
            or self.headers.get("X-CSRF-Token") != session[1]
        ):
            self._error(HTTPStatus.FORBIDDEN, "invalid_request", request_id)
            return
        local_session_id = self.headers.get("X-Local-Session-Id", "")
        if not ID_PATTERN.fullmatch(request_id) or not ID_PATTERN.fullmatch(local_session_id):
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request", request_id)
            return
        if self.path == "/api/cancel":
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", request_id)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", request_id)
                return
            if length != 0:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", request_id)
                return
            if not self.companion.cancel_job(session[0], local_session_id, request_id):
                self._error(HTTPStatus.CONFLICT, "invalid_request", request_id)
                return
            body = json.dumps(
                {"code": "cancelling", "request_id": request_id}, separators=(",", ":")
            ).encode()
            self._send(HTTPStatus.ACCEPTED, body, "application/json")
            return
        if self.headers.get("Content-Type") != "audio/webm;codecs=opus":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "invalid_request", request_id)
            return
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", request_id)
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
            duration_ms = int(self.headers.get("X-Duration-Ms", ""))
        except ValueError:
            self._error(HTTPStatus.LENGTH_REQUIRED, "invalid_request", request_id)
            return
        if not 1 <= length <= MAX_AUDIO_BYTES or not 1 <= duration_ms <= 1_800_000:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "audio_limit", request_id)
            return
        if not self.companion.job_lock.acquire(blocking=False):
            self._error(HTTPStatus.CONFLICT, "invalid_request", request_id)
            return

        cancel_event: Event | None = None
        timeout_event: Event | None = None
        temporary = None
        request_dir = None
        result: dict[str, object] | None = None
        error: LocalAudioError | None = None
        try:
            cancel_event, timeout_event = self.companion.begin_job(
                session[0], local_session_id, request_id
            )
            temporary = private_request_dir(self.companion.pipeline.config.support_dir / "tmp")
            request_dir = Path(temporary.name)
            encoded = request_dir / "audio.webm"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(encoded, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                self.connection.settimeout(15)
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise LocalAudioError("invalid_request")
                    handle.write(chunk)
                    remaining -= len(chunk)
            result = self.companion.pipeline.process(
                encoded_path=encoded,
                request_dir=request_dir,
                local_session_id=local_session_id,
                request_id=request_id,
                duration_ms=duration_ms,
                cancelled=cancel_event.is_set,
            )
        except socket.timeout:
            error = LocalAudioError("timeout")
        except LocalAudioError as exc:
            error = exc
        except Exception:
            error = LocalAudioError("internal_failure")
        finally:
            if temporary is not None:
                temporary.cleanup()
                if request_dir is not None and request_dir.exists():
                    result = None
                    error = LocalAudioError("cleanup_failed")
            cancelled = False
            timed_out = False
            if cancel_event is not None:
                cancelled, timed_out = self.companion.finish_job(cancel_event)
            if cancelled:
                result = None
                error = LocalAudioError("timeout" if timed_out else "cancelled")
            if cancelled and self.companion.recover_hook is not None:
                try:
                    self.companion.recover_hook()
                except Exception:
                    result = None
                    error = LocalAudioError("classification_failed")
            self.companion.job_lock.release()

        if error is not None or result is None:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, (error or LocalAudioError("internal_failure")).code, request_id)
            return
        body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")


class CompanionServer(ThreadingHTTPServer):
    # SO_REUSEADDR permits immediate restart after TIME_WAIT; a live listener
    # on the frozen port still makes bind() fail closed.
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, companion: CompanionState) -> None:
        super().__init__((HOST, PORT), CompanionHandler)
        self.companion = companion


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description="Loopback-only Visit Capture audio companion")
    parser.add_argument("--check", action="store_true", help="verify exact local dependencies and exit")
    args = parser.parse_args()
    config = PipelineConfig.default()
    verify_environment(config)
    if args.check:
        print("Priority 4 local audio prerequisites verified; no patient data processed.")
        return
    ollama = DedicatedOllama(config)
    ollama.start()
    server = CompanionServer(CompanionState(
        LocalAudioPipeline(config), cancel_hook=ollama.stop, recover_hook=ollama.start
    ))
    try:
        print(f"Local Visit Capture launcher: {ORIGIN}")
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        ollama.stop()
        shutil.rmtree(config.support_dir / "tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
