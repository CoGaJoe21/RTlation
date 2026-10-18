"""Web-Server: liefert die Oberfläche aus und nimmt Audio per WebSocket an."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, gpu
from .config import Config
from .engine import Pipeline, Session
from .languages import LANGUAGES
from .stages.stt import available_models, model_label
from .stages.translate import STYLES
from .stages.tts import ENGINES, is_installed
from .voices import delete_voice, list_voices, move_project_voices, save_voice

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
_background: set[asyncio.Task] = set()   # hält laufende Hintergrundaufgaben am Leben
_MAX_VOICE_UPLOAD = 20 * 1024 * 1024


def create_app(config: Config) -> FastAPI:
    move_project_voices(config)
    pipeline = Pipeline(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await pipeline.start()
        yield
        await pipeline.close()

    app = FastAPI(title="RTlation", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Die Oberfläche soll nach einem Update sofort aktuell sein.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/info")
    async def info() -> dict:
        return {
            "version": __version__,
            "languages": [{"code": lang.code, "label": lang.label} for lang in LANGUAGES.values()],
            "defaults": {
                "source": config.source_language,
                "target": config.target_language,
                "min_silence_ms": config.vad.min_silence_ms,
                "style": config.translate.style if config.translate.style in STYLES else "literal",
            },
            "styles": [{"id": key, "label": label} for key, (label, _) in STYLES.items()],
            "status": pipeline.status,
            "stt": {
                "model": pipeline.stt_model,
                "models": [{"id": name, "label": model_label(name)} for name in available_models(config)],
            },
            "tts": {
                "enabled": config.tts.enabled,
                "engine": pipeline.tts_engine,
                "voice": config.tts.voice,
                "expression": float(config.tts.options.get("exaggeration", 0.5)),
                "voices": list_voices(config),
                "engines": [
                    {"id": name, "label": spec.label, "installed": is_installed(config, name)}
                    for name, spec in ENGINES.items()
                ],
            },
        }

    @app.get("/api/gpu")
    async def gpu_memory() -> dict:
        """Grafikspeicher: Bedarf der aktuellen Auswahl und tatsächliche Belegung."""
        speaking = pipeline.status["tts"]["state"] in ("loading", "ready")
        worker = ENGINES[pipeline.tts_engine].worker if speaking and pipeline.tts_engine in ENGINES else None
        return {
            "needed": gpu.needed(pipeline.stt_model, config.translate.model, worker),
            "usage": await asyncio.to_thread(gpu.usage),
        }

    @app.post("/api/voices/{name}")
    async def add_voice(name: str, request: Request) -> dict:
        """Speichert eine Aufnahme (WAV im Rumpf der Anfrage) als Stimme."""
        data = await request.body()
        if len(data) > _MAX_VOICE_UPLOAD:
            raise HTTPException(413, "Die Aufnahme ist zu gross.")
        try:
            path, seconds = save_voice(config, name, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Die Abschrift hilft Modellen, die Stimme genauer zu treffen.
        text = ""
        try:
            from faster_whisper.audio import decode_audio

            audio = await asyncio.to_thread(decode_audio, str(path), sampling_rate=16000)
            text = (await pipeline.transcribe(audio, None)).text
            if text:
                path.with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        except Exception:
            log.exception("Abschrift der Stimme %s fehlgeschlagen", path.stem)
        return {"name": path.stem, "seconds": round(seconds, 1), "text": text, "voices": list_voices(config)}

    @app.delete("/api/voices/{name}")
    async def remove_voice(name: str) -> dict:
        try:
            delete_voice(config, name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"voices": list_voices(config)}

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        """Zum Server: PCM binär (16 Bit, mono, 16 kHz) und JSON-Steuerbefehle.
        Zum Browser: JSON-Ereignisse; auf jedes "audio"-Ereignis folgt dessen PCM binär."""
        await ws.accept()
        session = Session(pipeline)

        async def send_events() -> None:
            while True:
                event = await session.events.get()
                if isinstance(event, bytes):
                    await ws.send_bytes(event)
                else:
                    await ws.send_json(event)

        sender = asyncio.create_task(send_events())
        try:
            await ws.send_json({
                "type": "status", "status": pipeline.status,
                "tts_engine": pipeline.tts_engine, "stt_model": pipeline.stt_model,
            })
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    session.feed(message["bytes"])
                elif message.get("text") is not None:
                    _control(pipeline, session, message["text"])
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await session.close()

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _control(pipeline: Pipeline, session: Session, text: str) -> None:
    try:
        command = json.loads(text)
        kind = command.get("type")
        if kind == "config":
            session.configure(
                source=command.get("source"),
                target=command.get("target"),
                min_silence_ms=command.get("min_silence_ms"),
                speak=command.get("speak"),
                voice=command.get("voice"),
                style=command.get("style"),
                expression=command.get("expression"),
                auto_expression=command.get("auto_expression"),
            )
        elif kind == "stop":
            session.stop()
        elif kind == "stt_model":
            model = command.get("model")
            if model not in available_models(pipeline.config):
                raise ValueError(f"Unbekanntes Whisper-Modell: {model}")
            task = asyncio.create_task(pipeline.set_stt_model(model))
            _background.add(task)
            task.add_done_callback(_background.discard)
        elif kind == "tts_engine":
            engine = command.get("engine")
            if engine not in ENGINES:
                raise ValueError(f"Unbekanntes Sprachmodell: {engine}")
            # Das Laden dauert; die Oberfläche erfährt den Fortschritt über "status".
            task = asyncio.create_task(pipeline.set_tts_engine(engine))
            _background.add(task)
            task.add_done_callback(_background.discard)
    except (ValueError, AttributeError, TypeError) as exc:
        session.events.put_nowait({"type": "error", "message": f"Ungültiger Befehl: {exc}"})
