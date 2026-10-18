"""Die Pipeline: PCM-Audio rein, Ereignisse (Text, Übersetzung, Sprache) raus.

Die Engine kennt weder Mikrofon noch Netzwerk. Lokal und im Server-Betrieb
hängt dieselbe Session hinter einer WebSocket-Verbindung.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from .config import Config
from .languages import AUTO, LANGUAGES, Language, by_whisper_code
from .prosody import MAX_EXPRESSION, MIN_EXPRESSION, ExpressionTracker
from .stages.stt import Transcriber, Transcript, available_models, model_label
from .stages.translate import STYLES, TranslationError, Translator
from .stages.tts import ENGINES, TtsError, TtsWorker, engine_options, is_installed, resolve_voice
from .stages.vad import Segment, Segmenter, SileroVad, SpeechDiscarded, SpeechEnd, SpeechStart
from .voices import voice_path

log = logging.getLogger(__name__)

_PARTIAL_MIN_AUDIO_MS = 600    # ab hier lohnt sich ein Zwischenstand
_PARTIAL_INTERVAL_S = 0.8
_TRANSLATOR_RECHECK_S = 10.0


class Pipeline:
    """Die schweren, zwischen allen Verbindungen geteilten Bausteine."""

    def __init__(self, config: Config):
        self.config = config
        self.vad = SileroVad(config.resolve(config.vad.model))
        self.translator = Translator(config.translate)
        self.tts: TtsWorker | None = None
        self.tts_engine = config.tts.engine
        self.status = {
            "stt": {"state": "loading", "detail": "Modell wird geladen"},
            "translate": {"state": "loading", "detail": "Verbindung zu Ollama"},
            "tts": {"state": "off", "detail": "Sprachausgabe ist ausgeschaltet"},
        }
        self._transcriber: Transcriber | None = None
        self.stt_model = Path(config.stt.model).name
        self._stt_switch = asyncio.Lock()
        # Ein einzelner Thread: Die GPU rechnet einen Auftrag nach dem anderen.
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._stt_loaded = asyncio.Event()
        self._listeners: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._tts_task: asyncio.Task | None = None
        self._tts_switch = asyncio.Lock()
        self._voices_in_use: set[str] = set()      # werden nach jedem Modellstart vorab geladen
        self._background: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._load_transcriber()),
            asyncio.create_task(self._watch_translator()),
        ]
        if self.config.tts.enabled:
            self._tts_task = asyncio.create_task(self._run_tts(self.tts_engine))

    async def close(self) -> None:
        await self._stop_tts()
        tasks = [*self._tasks, *self._background]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.translator.aclose()
        self._gpu.shutdown(wait=False, cancel_futures=True)

    def subscribe(self, queue: asyncio.Queue) -> None:
        self._listeners.add(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._listeners.discard(queue)

    async def transcribe(self, audio: np.ndarray, language: str | None, *, fast: bool = False) -> Transcript:
        await self._stt_loaded.wait()
        if self._transcriber is None:
            raise RuntimeError(self.status["stt"]["detail"])
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._gpu, partial(self._transcriber.transcribe, audio, language, fast=fast)
        )

    async def speak(self, text: str, language: Language, voice: str = "",
                    expression: float | None = None) -> AsyncIterator[tuple[bytes, int]]:
        """Liefert die Sprachausgabe als (PCM 16 Bit mono, Abtastrate), Stück für Stück.

        voice: Name einer eigenen Aufnahme; leer = Standardstimme.
        expression: wie lebhaft gesprochen wird (0.5 = neutral); None = Einstellung aus config.yaml.
        """
        worker, engine = self.tts, self.tts_engine
        if worker is None:
            raise TtsError("Die Sprachausgabe ist nicht bereit.")
        path, transcript = resolve_voice(self.config, engine, voice)
        options = engine_options(self.config, engine)
        if expression is not None:
            options["exaggeration"] = expression
        async for chunk in worker.synthesize(text, language, path, transcript, options):
            yield chunk

    def prepare_voice(self, voice: str) -> None:
        """Lädt eine Stimme im Hintergrund vor (das dauert beim ersten Mal einige Sekunden)."""
        self._voices_in_use.add(voice)
        task = asyncio.create_task(self._prepare_voice(voice))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _prepare_voice(self, voice: str) -> None:
        worker, engine = self.tts, self.tts_engine
        if worker is None:
            return   # passiert dann, sobald das Modell bereit ist
        try:
            await worker.prepare(*resolve_voice(self.config, engine, voice))
        except TtsError as exc:
            log.warning("Stimme %r konnte nicht vorab geladen werden: %s", voice or "Standard", exc)

    async def set_tts_engine(self, engine: str) -> None:
        """Wechselt das Sprachmodell bzw. dessen Betriebsart."""
        if engine not in ENGINES:
            raise ValueError(f"Unbekanntes Sprachmodell: {engine}")
        async with self._tts_switch:
            if engine == self.tts_engine and self._tts_task and not self._tts_task.done():
                return
            if self.tts is not None and self.tts.name == ENGINES[engine].worker:
                # Gleiches Modell, andere Betriebsart: nichts neu laden.
                self.tts_engine = engine
                self.status["tts"] = {"state": "ready", "detail": f"{ENGINES[engine].label} auf der Grafikkarte"}
                self._broadcast()
                return
            await self._stop_tts()
            self.tts_engine = engine
            self._tts_task = asyncio.create_task(self._run_tts(engine))

    def _set_status(self, key: str, state: str, detail: str) -> None:
        new = {"state": state, "detail": detail}
        if self.status[key] == new:
            return
        self.status[key] = new
        self._broadcast()

    def _broadcast(self) -> None:
        for queue in self._listeners:
            queue.put_nowait({
                "type": "status", "status": self.status,
                "tts_engine": self.tts_engine, "stt_model": self.stt_model,
            })

    async def set_stt_model(self, name: str) -> None:
        """Wechselt das Whisper-Modell. Sätze, die währenddessen ankommen, warten."""
        if name not in available_models(self.config):
            raise ValueError(f"Unbekanntes Whisper-Modell: {name}")
        async with self._stt_switch:
            if name == self.stt_model and self._transcriber is not None:
                return
            self._stt_loaded.clear()
            self.stt_model = name
            self._set_status("stt", "loading", f"{model_label(name)} wird geladen")
            await self._load_transcriber()

    def _replace_transcriber(self, model_dir: Path) -> Transcriber:
        # Läuft im GPU-Thread: erst das alte Modell freigeben, dann das neue laden,
        # damit nicht beide gleichzeitig im Grafikspeicher liegen.
        self._transcriber = None
        gc.collect()
        return Transcriber(self.config, model_dir)

    async def _load_transcriber(self) -> None:
        loop = asyncio.get_running_loop()
        name = self.stt_model
        try:
            model_dir = available_models(self.config)[name]
            self._transcriber = await loop.run_in_executor(self._gpu, self._replace_transcriber, model_dir)
        except Exception as exc:
            log.exception("Spracherkennung konnte nicht geladen werden")
            self._set_status("stt", "error", f"Laden fehlgeschlagen: {exc}")
        else:
            where = "Grafikkarte" if self._transcriber.device == "cuda" else "CPU (langsam)"
            self._set_status("stt", "ready", f"{name} auf {where}")
            log.info("Spracherkennung bereit: %s auf %s", name, self._transcriber.device)
        finally:
            self._stt_loaded.set()

    async def _watch_translator(self) -> None:
        """Hält den Status der Übersetzung aktuell und lädt das Modell vor."""
        model = self.config.translate.model
        warm = False
        while True:
            problem = await self.translator.check()
            if problem:
                warm = False
                self._set_status("translate", "error", problem)
            elif not warm:
                self._set_status("translate", "loading", f"{model} wird geladen")
                try:
                    await self.translator.translate("Hallo", LANGUAGES["de"], LANGUAGES["en"])
                except TranslationError as exc:
                    self._set_status("translate", "error", str(exc))
                else:
                    warm = True
                    self._set_status("translate", "ready", f"{model} über Ollama")
                    log.info("Übersetzung bereit: %s", model)
            await asyncio.sleep(_TRANSLATOR_RECHECK_S)

    async def _stop_tts(self) -> None:
        task, self._tts_task = self._tts_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run_tts(self, engine: str) -> None:
        """Startet den Worker, hält ihn bereit und meldet, wenn er ausfällt."""
        label = ENGINES[engine].label if engine in ENGINES else engine
        if not is_installed(self.config, engine):
            self._set_status("tts", "error", f"{label} ist nicht eingerichtet (setup.bat -Tts {engine})")
            return
        self._set_status("tts", "loading", f"{label} wird geladen")
        worker = TtsWorker(self.config, engine)
        try:
            started = time.perf_counter()
            await worker.start()
            # Der erste Satz ist langsam; er soll nicht den Nutzer treffen.
            voice, voice_text = resolve_voice(self.config, engine, "")
            warm_up = worker.synthesize(
                "Hello, this is a short test.", LANGUAGES["en"], voice, voice_text,
                engine_options(self.config, engine),
            )
            async for _ in warm_up:
                pass
            self.tts = worker
            self._set_status("tts", "ready", f"{label} auf der Grafikkarte")
            for name in sorted(self._voices_in_use):
                await self._prepare_voice(name)
            log.info("Sprachausgabe bereit: %s (%.0f s)", label, time.perf_counter() - started)
            while worker.alive:
                await asyncio.sleep(1.0)
            self._set_status("tts", "error", f"{label} wurde unerwartet beendet")
        except TtsError as exc:
            log.error("Sprachausgabe %s: %s", label, exc)
            self._set_status("tts", "error", f"{label}: {exc}")
        finally:
            self.tts = None
            await worker.close()


class Session:
    """Eine Verbindung: eigener Audiostrom, eigene Sprachwahl."""

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        config = pipeline.config
        # dict = Ereignis (JSON), bytes = Audio zum direkt davor gesendeten "audio"-Ereignis
        self.events: asyncio.Queue[dict | bytes] = asyncio.Queue()
        self.source = config.source_language
        self.target = config.target_language
        self.speak = True
        self.style = config.translate.style if config.translate.style in STYLES else "literal"
        # Ausdruck der Sprachausgabe: Grundwert, und ob er sich pro Satz nach
        # der Energie des Gesprochenen richtet.
        self.expression = float(config.tts.options.get("exaggeration", 0.5))
        self.auto_expression = True
        self._expression_tracker = ExpressionTracker()
        self.voice = config.tts.voice   # Name einer Aufnahme in voices/, leer = Standardstimme
        self._voice_requested = False   # ob das Vorab-Laden der Stimme schon angestossen ist
        self._segmenter = Segmenter(pipeline.vad, config.vad)
        self._stt_queue: asyncio.Queue[Segment] = asyncio.Queue()
        self._mt_queue: asyncio.Queue[tuple[Segment, Transcript]] = asyncio.Queue()
        self._tts_queue: asyncio.Queue[tuple[Segment, str, Language, float]] = asyncio.Queue()
        self._finals_open = 0        # Satzteile, deren Erkennung aussteht oder läuft
        self._transcribed_upto = 0   # höchste ID mit abgeschlossener Erkennung
        self._partial_task: asyncio.Task | None = None
        self._last_partial = 0.0
        self._synthesizing = False   # gerade entsteht Sprachausgabe für einen Satz
        self._workers = [
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._mt_worker()),
            asyncio.create_task(self._tts_worker()),
        ]
        pipeline.subscribe(self.events)

    async def close(self) -> None:
        self._pipeline.unsubscribe(self.events)
        tasks = [*self._workers]
        if self._partial_task:
            tasks.append(self._partial_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def configure(self, *, source: str | None = None, target: str | None = None,
                  min_silence_ms: int | None = None, speak: bool | None = None,
                  voice: str | None = None, style: str | None = None,
                  expression: float | None = None, auto_expression: bool | None = None) -> None:
        if source is not None:
            if source != AUTO and source not in LANGUAGES:
                raise ValueError(f"Unbekannte Ausgangssprache: {source}")
            self.source = source
        if target is not None:
            if target not in LANGUAGES:
                raise ValueError(f"Unbekannte Zielsprache: {target}")
            self.target = target
        if min_silence_ms is not None:
            self._segmenter.min_silence_ms = min(2000, max(200, int(min_silence_ms)))
        if speak is not None:
            self.speak = bool(speak)
        if expression is not None:
            self.expression = min(MAX_EXPRESSION, max(MIN_EXPRESSION, float(expression)))
        if auto_expression is not None:
            self.auto_expression = bool(auto_expression)
        if style is not None:
            if style not in STYLES:
                raise ValueError(f"Unbekannter Übersetzungsstil: {style}")
            self.style = style
        if voice is not None:
            if voice and not voice_path(self._pipeline.config, voice).is_file():
                raise ValueError(f"Die Stimme „{voice}“ gibt es nicht.")
            if voice != self.voice or not self._voice_requested:
                self._voice_requested = True
                self._pipeline.prepare_voice(voice)
            self.voice = voice

    def feed(self, data: bytes) -> None:
        """data: PCM, 16 Bit, mono, 16 kHz."""
        pcm = np.frombuffer(data[:len(data) - len(data) % 2], dtype=np.int16)
        self._handle(self._segmenter.feed(pcm))
        self._maybe_start_partial()

    def stop(self) -> None:
        """Aufnahme beendet: einen offenen Satzteil noch verarbeiten."""
        self._handle(self._segmenter.flush())

    def _emit(self, **event) -> None:
        self.events.put_nowait(event)

    def _handle(self, events) -> None:
        for event in events:
            if isinstance(event, SpeechStart):
                self._emit(type="speech_start", id=event.id)
            elif isinstance(event, SpeechEnd):
                self._finals_open += 1
                self._stt_queue.put_nowait(event.segment)
                self._emit(type="speech_end", id=event.segment.id)
            elif isinstance(event, SpeechDiscarded):
                self._emit(type="discard", id=event.id)

    def _whisper_language(self) -> str | None:
        return None if self.source == AUTO else LANGUAGES[self.source].whisper

    def _maybe_start_partial(self) -> None:
        """Zwischenstand des laufenden Satzteils, solange die GPU sonst frei ist."""
        segmenter = self._segmenter
        if not segmenter.active or self._finals_open:
            return
        # Die Spracherkennung belegt die Grafikkarte stossweise; das bringt die
        # stückweise Sprachausgabe ins Stocken. Der Ton hat Vorrang vor der Vorschau.
        if self._synthesizing:
            return
        if self._partial_task and not self._partial_task.done():
            return
        now = time.monotonic()
        if (segmenter.current_duration_ms < _PARTIAL_MIN_AUDIO_MS
                or now - self._last_partial < _PARTIAL_INTERVAL_S):
            return
        self._last_partial = now
        self._partial_task = asyncio.create_task(
            self._partial(segmenter.current_id, segmenter.current_audio())
        )

    async def _partial(self, segment_id: int, audio: np.ndarray) -> None:
        try:
            result = await self._pipeline.transcribe(audio, self._whisper_language(), fast=True)
        except Exception:
            return
        if result.text and segment_id > self._transcribed_upto:
            self._emit(type="partial", id=segment_id, text=result.text)

    async def _stt_worker(self) -> None:
        while True:
            segment = await self._stt_queue.get()
            started = time.perf_counter()
            try:
                result = await self._pipeline.transcribe(segment.audio, self._whisper_language())
            except Exception as exc:
                log.exception("Spracherkennung fehlgeschlagen")
                self._emit(type="error", message=f"Spracherkennung fehlgeschlagen: {exc}")
                result = None
            finally:
                self._finals_open -= 1
                self._transcribed_upto = segment.id

            if result is None or not result.text:
                self._emit(type="discard", id=segment.id)
                continue
            self._emit(
                type="transcript", id=segment.id, text=result.text, language=result.language,
                audio_ms=segment.duration_ms,
                stt_ms=round((time.perf_counter() - started) * 1000),
            )
            self._mt_queue.put_nowait((segment, result))

    async def _mt_worker(self) -> None:
        while True:
            segment, result = await self._mt_queue.get()
            source = by_whisper_code(result.language) if self.source == AUTO else LANGUAGES[self.source]
            target = LANGUAGES[self.target]
            started = time.perf_counter()
            try:
                if source is None:
                    raise TranslationError(
                        f"Die erkannte Sprache „{result.language}“ wird nicht unterstützt."
                    )
                if source.code == target.code:
                    text = result.text
                else:
                    pieces = []
                    async for piece in self._pipeline.translator.stream(result.text, source, target, self.style):
                        pieces.append(piece)
                        self._emit(type="translation_delta", id=segment.id, delta=piece)
                    text = "".join(pieces).strip()
            except TranslationError as exc:
                self._emit(type="error", id=segment.id, message=str(exc))
                continue
            will_speak = bool(self.speak and text and self._pipeline.tts is not None)
            self._emit(
                type="translation", id=segment.id, text=text, target=target.code,
                mt_ms=round((time.perf_counter() - started) * 1000),
                # Vom Ende des Sprechens bis zur fertigen Übersetzung.
                latency_ms=round((time.monotonic() - segment.speech_ended_at) * 1000),
                speaking=will_speak,
            )
            if will_speak:
                expression = self.expression
                if self.auto_expression:
                    expression = self._expression_tracker.expression(expression, segment.audio, len(result.text))
                self._tts_queue.put_nowait((segment, text, target, expression))

    async def _tts_worker(self) -> None:
        while True:
            segment, text, target, expression = await self._tts_queue.get()
            started = time.perf_counter()
            samples = 0
            rate = 0
            self._synthesizing = True
            try:
                async for pcm, rate in self._pipeline.speak(text, target, self.voice, expression):
                    header = {"type": "audio", "id": segment.id, "sample_rate": rate}
                    if samples == 0:
                        # Vom Ende des Sprechens bis zum ersten hörbaren Ton.
                        header["latency_ms"] = round((time.monotonic() - segment.speech_ended_at) * 1000)
                    self.events.put_nowait(header)
                    self.events.put_nowait(pcm)
                    samples += len(pcm) // 2
            except TtsError as exc:
                self._emit(type="error", id=segment.id, message=f"Sprachausgabe: {exc}")
                continue
            finally:
                self._synthesizing = False
            self._emit(
                type="audio_end", id=segment.id, expression=round(expression, 2),
                tts_ms=round((time.perf_counter() - started) * 1000),
                audio_ms=round(samples * 1000 / rate) if rate else 0,
            )
