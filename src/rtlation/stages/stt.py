"""Spracherkennung mit faster-whisper."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config

log = logging.getLogger(__name__)

# Whisper erfindet auf Rauschen gern Abspann-Texte aus seinen Trainingsdaten.
_HALLUCINATION = re.compile(
    r"^\s*("
    r"Untertitel(ung)?\s+(der|des|von|im Auftrag|aufgrund)\b.*"
    r"|Copyright WDR.*"
    r"|SWR \d{4}"
    r"|Subtitles by the Amara\.org community"
    r"|Thanks for watching[.!]?"
    r")\s*$",
    re.IGNORECASE,
)


_LABELS = {
    "large-v3-turbo": "Turbo (schnell)",
    "large-v3": "Large (grösser)",
}


def available_models(config: Config) -> dict[str, Path]:
    """Eingerichtete Whisper-Modelle: Name -> Ordner. Das Modell aus config.yaml steht vorn."""
    configured = config.resolve(config.stt.model)
    models = {configured.name: configured}
    for folder in sorted(config.resolve("models/whisper").glob("*")):
        if (folder / "model.bin").is_file():
            models.setdefault(folder.name, folder)
    return models


def model_label(name: str) -> str:
    return _LABELS.get(name, name)


@dataclass
class Transcript:
    text: str
    language: str   # von Whisper verwendeter bzw. erkannter Sprachcode


class Transcriber:
    def __init__(self, config: Config, model_dir: Path | None = None):
        from faster_whisper import WhisperModel

        cfg = config.stt
        local = model_dir or config.resolve(cfg.model)
        # Ein vorhandener Ordner wird direkt geladen; sonst gilt der Wert als
        # Modellname und faster-whisper lädt ihn nach models/whisper.
        model = str(local) if local.is_dir() else cfg.model
        download_root = str(config.resolve("models/whisper"))
        self._beam_size = cfg.beam_size

        try:
            self._model = WhisperModel(
                model, device=cfg.device, compute_type=cfg.compute_type, download_root=download_root
            )
            self.device = cfg.device
            # cuDNN wird erst beim ersten Rechnen geladen - hier fällt ein
            # fehlendes GPU-Setup auf, und der erste echte Satz ist schnell.
            self.transcribe(np.zeros(16000, dtype=np.float32), "de", fast=True)
        except Exception as exc:
            if cfg.device == "cpu":
                raise
            log.warning("GPU nicht nutzbar (%s) - Spracherkennung läuft auf der CPU.", exc)
            self._model = WhisperModel(
                model, device="cpu", compute_type="int8", download_root=download_root
            )
            self.device = "cpu"

    def transcribe(self, audio: np.ndarray, language: str | None, *, fast: bool = False) -> Transcript:
        """audio: float32, mono, 16 kHz. fast=True für Zwischenstände."""
        segments, info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1 if fast else self._beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        text = " ".join(part for s in segments if (part := s.text.strip()))
        if _HALLUCINATION.match(text):
            text = ""
        return Transcript(text, info.language)
