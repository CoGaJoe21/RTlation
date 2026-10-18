"""Einstellungen aus config.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class SttConfig:
    model: str = "models/whisper/large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5


@dataclass
class VadConfig:
    model: str = "models/vad/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 250
    speech_pad_ms: int = 200
    soft_max_segment_s: float = 8.0
    max_segment_s: float = 14.0


@dataclass
class TranslateConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "translategemma:4b"
    keep_alive: str = "30m"
    style: str = "literal"


@dataclass
class TtsConfig:
    enabled: bool = True
    engine: str = "chatterbox"
    voice: str = ""
    device: str = "cuda"
    options: dict = field(default_factory=dict)


@dataclass
class Config:
    root: Path = field(default_factory=Path.cwd)
    source_language: str = "de"
    target_language: str = "en"
    voices_dir: str = ""
    server: ServerConfig = field(default_factory=ServerConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    def resolve(self, path: str) -> Path:
        """Pfad aus der Konfiguration relativ zum Projektordner auflösen."""
        p = Path(path)
        return p if p.is_absolute() else self.root / p


def _fill(target: Any, values: dict[str, Any], where: str) -> None:
    known = {f.name: f for f in dataclasses.fields(target)}
    for key, value in values.items():
        if key not in known or key == "root":
            raise ValueError(f"Unbekannte Einstellung in config.yaml: {where}{key}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config.yaml: {where}{key} muss ein Abschnitt sein")
            _fill(current, value, f"{where}{key}.")
        elif isinstance(current, dict):
            setattr(target, key, dict(value or {}))
        else:
            setattr(target, key, type(current)(value))


def load_config(path: Path | None = None) -> Config:
    """Lädt config.yaml; ohne Datei gelten die Standardwerte."""
    path = (path or Path.cwd() / "config.yaml").resolve()
    config = Config(root=path.parent)
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _fill(config, data, "")
    return config
