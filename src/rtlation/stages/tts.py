"""Sprachausgabe: steuert einen Worker-Prozess mit dem Sprachmodell.

Das Protokoll steht in workers/base.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..languages import Language
from ..voices import shipped_voice, voices_dir

log = logging.getLogger(__name__)

_LOAD_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class EngineSpec:
    label: str
    worker: str               # Ordner unter workers/ und Name der Python-Umgebung
    model_dir: str            # relativ zum Projektordner
    needs_reference: bool     # ohne Referenzaufnahme keine Stimme
    options: dict = field(default_factory=dict)   # feste Einstellungen dieser Betriebsart


# Mehrere Einträge können sich einen Worker teilen; dann wechselt nur die
# Betriebsart und das Modell bleibt geladen.
ENGINES: dict[str, EngineSpec] = {
    "chatterbox": EngineSpec(
        "Chatterbox · schnell", "chatterbox", "models/tts/chatterbox", needs_reference=False,
        options={"stream": True},
    ),
    "chatterbox-full": EngineSpec(
        "Chatterbox · ganzer Satz", "chatterbox", "models/tts/chatterbox", needs_reference=False,
        options={"stream": False},
    ),
    "qwen3": EngineSpec("Qwen3-TTS", "qwen3", "models/tts/qwen3-tts-0.6b-base", needs_reference=True),
}


class TtsError(RuntimeError):
    pass


def _worker_python(config: Config, worker: str) -> Path:
    venv = config.root / ".tools" / "venvs" / worker
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def is_installed(config: Config, engine: str) -> bool:
    spec = ENGINES.get(engine)
    return bool(spec) and _worker_python(config, spec.worker).is_file() and config.resolve(spec.model_dir).is_dir()


def engine_options(config: Config, engine: str) -> dict:
    """Einstellungen aus config.yaml, überlagert von denen der Betriebsart."""
    return {**config.tts.options, **ENGINES[engine].options}


def resolve_voice(config: Config, engine: str, name: str) -> tuple[Path | None, str | None]:
    """Stimme -> (Referenzaufnahme, Abschrift). Leerer Name = Standardstimme."""
    if not name:
        if not ENGINES[engine].needs_reference:
            return None, None
        wav = shipped_voice(config)
    else:
        wav = voices_dir(config) / f"{name}.wav"
    if not wav.is_file():
        raise TtsError(f"Die Stimme „{name or 'Standard'}“ fehlt ({wav}).")
    transcript = wav.with_suffix(".txt")
    text = transcript.read_text(encoding="utf-8").strip() if transcript.is_file() else None
    return wav, text or None


class TtsWorker:
    """Ein laufender Worker-Prozess. Aufträge werden nacheinander bearbeitet."""

    def __init__(self, config: Config, engine: str):
        if engine not in ENGINES:
            raise TtsError(f"Unbekanntes Sprachmodell: {engine}")
        self.name = ENGINES[engine].worker
        self._model_dir = config.resolve(ENGINES[engine].model_dir)
        self._config = config
        self._process: subprocess.Popen | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    async def start(self) -> None:
        """Startet den Prozess und wartet, bis das Modell geladen ist."""
        config = self._config
        token = secrets.token_hex(16)
        connected: asyncio.Future = asyncio.get_running_loop().create_future()

        async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                header, _ = await _read_frame(reader)
            except (asyncio.IncompleteReadError, ValueError):
                header = {}
            if header.get("type") == "ready" and header.get("token") == token and not connected.done():
                connected.set_result((reader, writer))
            else:
                writer.close()

        server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        env = {
            **os.environ,
            "RTLATION_WORKER_TOKEN": token,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            # Was die Modelle beim ersten Lauf nachladen, soll im Projektordner landen.
            "HF_HOME": str(config.root / "models" / "hf-cache"),
            "PKUSEG_HOME": str(config.root / "models" / "pkuseg"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Fortschrittsbalken und Hinweise pro Satz würden die Konsole fluten.
            "TQDM_DISABLE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
        }
        command = [
            str(_worker_python(config, self.name)),
            str(config.root / "workers" / self.name / "worker.py"),
            "--connect", str(port),
            "--model-dir", str(self._model_dir),
            "--device", config.tts.device,
        ]
        try:
            self._process = subprocess.Popen(command, cwd=config.root, env=env)
            deadline = asyncio.get_running_loop().time() + _LOAD_TIMEOUT_S
            while not connected.done():
                if self._process.poll() is not None:
                    raise TtsError(f"Der Prozess wurde beim Laden beendet (Code {self._process.returncode}).")
                if asyncio.get_running_loop().time() > deadline:
                    raise TtsError("Das Modell hat sich nicht rechtzeitig gemeldet.")
                await asyncio.sleep(0.2)
            self._reader, self._writer = connected.result()
        except BaseException:
            await self.close()
            raise
        finally:
            server.close()

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()

    async def synthesize(self, text: str, language: Language, voice: Path | None = None,
                         voice_text: str | None = None,
                         options: dict | None = None) -> AsyncIterator[tuple[bytes, int]]:
        """Liefert (PCM 16 Bit mono, Abtastrate) Stück für Stück."""
        request = {
            "text": text,
            "language": language.whisper,
            "language_name": language.name,
            "voice": str(voice) if voice else None,
            "voice_text": voice_text,
            "options": options if options is not None else self._config.tts.options,
        }
        async for chunk in self._request(request):
            yield chunk

    async def prepare(self, voice: Path | None, voice_text: str | None = None) -> None:
        """Lädt eine Stimme vor, damit der erste Satz damit nicht warten muss."""
        request = {"prepare": True, "voice": str(voice) if voice else None, "voice_text": voice_text}
        async for _ in self._request(request):
            pass

    async def _request(self, request: dict) -> AsyncIterator[tuple[bytes, int]]:
        async with self._lock:
            if self._writer is None or self._reader is None or not self.alive:
                raise TtsError("Die Sprachausgabe läuft nicht.")
            self._next_id += 1
            request_id = request["id"] = self._next_id
            try:
                self._writer.write((json.dumps(request) + "\n").encode("utf-8"))
                await self._writer.drain()
                while True:
                    header, payload = await _read_frame(self._reader)
                    if header.get("id") != request_id:
                        continue   # Rest eines abgebrochenen früheren Auftrags
                    kind = header.get("type")
                    if kind == "audio":
                        yield payload, header["sample_rate"]
                    elif kind == "done":
                        return
                    elif kind == "error":
                        raise TtsError(header.get("message", "Unbekannter Fehler"))
            except (asyncio.IncompleteReadError, ConnectionError) as exc:
                raise TtsError("Die Verbindung zur Sprachausgabe ist abgebrochen.") from exc


async def _read_frame(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    size = int.from_bytes(await reader.readexactly(4), "big")
    header = json.loads(await reader.readexactly(size))
    payload = await reader.readexactly(header["bytes"]) if header.get("bytes") else b""
    return header, payload
