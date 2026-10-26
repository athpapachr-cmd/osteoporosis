from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib import request
import json
import os
import shutil
import socket
import subprocess
import time

from .pipeline import (
    MODEL_NAME,
    MODEL_SHA256,
    WHISPER_COMMIT,
    WHISPER_MODEL_SHA256,
    PipelineConfig,
    normalize_ollama_digest,
    verify_decoder,
    verify_filevault,
)


KRIKRI_URL = (
    "https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF/resolve/"
    "57c29d45cc89f4c60779f5094b3e5e5cb3556f4e/llama-krikri-8b-instruct-q4_k_m.gguf"
)
WHISPER_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
)


def download_verified(url: str, target: Path, expected: str) -> None:
    if target.is_file():
        existing = sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                existing.update(chunk)
        if existing.hexdigest() == expected:
            return
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = target.with_suffix(target.suffix + ".part")
    digest = sha256()
    with request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file digest mismatch")
    os.chmod(partial, 0o600)
    partial.replace(target)


def build_whisper(config: PipelineConfig) -> None:
    if not config.whisper_repo.exists():
        subprocess.run(
            ["/usr/bin/git", "clone", "https://github.com/ggml-org/whisper.cpp", str(config.whisper_repo)],
            check=True,
        )
    subprocess.run(["/usr/bin/git", "-C", str(config.whisper_repo), "fetch", "--depth=1", "origin", WHISPER_COMMIT], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(config.whisper_repo), "checkout", "--detach", WHISPER_COMMIT], check=True)
    subprocess.run(["/opt/homebrew/bin/cmake", "-S", str(config.whisper_repo), "-B", str(config.whisper_repo / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["/opt/homebrew/bin/cmake", "--build", str(config.whisper_repo / "build"), "-j", "--config", "Release"], check=True)


def create_ollama_model(config: PipelineConfig, gguf: Path) -> None:
    binary = Path("/opt/homebrew/Cellar/ollama/0.40.11/bin/ollama")
    if not binary.is_file():
        raise RuntimeError("Exact Ollama 0.40.11 is required")
    models = config.support_dir / "ollama-models"
    models.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {
        "HOME": str(Path.home()),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
        "OLLAMA_HOST": "127.0.0.1:11435",
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_DEBUG": "0",
        "OLLAMA_MODELS": str(models),
    }
    model_file = config.support_dir / "Modelfile.priority4-v1"
    model_file.write_text(
        f"FROM {gguf}\nPARAMETER temperature 0\nPARAMETER num_ctx 16384\n",
        encoding="utf-8",
    )
    os.chmod(model_file, 0o600)
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", 11435)) == 0:
            raise RuntimeError("Dedicated Ollama port 11435 is already in use")
    process = subprocess.Popen(
        [str(binary), "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=env, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Dedicated Ollama stopped during setup")
            try:
                with request.urlopen(config.ollama_url + "/api/version", timeout=1) as response:
                    version = json.loads(response.read(4096))
                if version.get("version") != "0.40.11":
                    raise RuntimeError("Unexpected dedicated Ollama version")
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("Dedicated Ollama did not start")
        subprocess.run([str(binary), "create", MODEL_NAME, "-f", str(model_file)], env=env, check=True)
        payload = json.dumps({
            "model": MODEL_NAME,
            "prompt": "x",
            "stream": False,
            "format": {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
            "keep_alive": 0,
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 16},
        }).encode()
        req = request.Request(config.ollama_url + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read(1024 * 1024))
        if int(result.get("prompt_eval_count", 999999)) > 1024:
            raise RuntimeError("Ollama template overhead self-test failed")
        with request.urlopen(config.ollama_url + "/api/tags", timeout=2) as response:
            tags = json.loads(response.read(1024 * 1024))
        installed = next(
            (item for item in tags.get("models", []) if item.get("name") == MODEL_NAME),
            None,
        )
        if not installed:
            raise RuntimeError("Pinned Ollama model alias was not installed")
        try:
            manifest_digest = normalize_ollama_digest(installed.get("digest"))
        except ValueError as exc:
            raise RuntimeError("Pinned Ollama model alias returned an invalid digest") from exc
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    receipt = config.support_dir / "model_receipt_v1.json"
    receipt.write_text(json.dumps({
        "model": MODEL_NAME,
        "source_gguf_sha256": MODEL_SHA256,
        "ollama_manifest_digest": manifest_digest,
    }, sort_keys=True), encoding="utf-8")
    os.chmod(receipt, 0o600)


def main() -> None:
    config = PipelineConfig.default()
    verify_filevault()
    verify_decoder(config)
    if shutil.disk_usage(config.support_dir.parent).free < 12 * 1024 * 1024 * 1024:
        raise RuntimeError("At least 12 GiB free disk space is required during setup")
    config.support_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config.support_dir, 0o700)
    build_whisper(config)
    download_verified(WHISPER_URL, config.whisper_model, WHISPER_MODEL_SHA256)
    gguf = config.support_dir / "models" / "llama-krikri-8b-instruct-q4_k_m.gguf"
    download_verified(KRIKRI_URL, gguf, MODEL_SHA256)
    create_ollama_model(config, gguf)
    print("Priority 4 local audio installed and verified. No patient data was used.")


if __name__ == "__main__":
    main()
