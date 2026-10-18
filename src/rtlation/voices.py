"""Stimmen: Referenzaufnahmen im Ordner voices/.

Eine Stimme ist eine kurze WAV-Datei, optional mit einer Abschrift daneben
(<name>.txt). Die Sprachmodelle ahmen die Stimme aus dieser Aufnahme nach.

Eigene Aufnahmen liegen im Benutzerprofil und nicht im Projektordner, damit
sie beim Weitergeben des Ordners nicht mitwandern. Im Projekt liegt nur die
mitgelieferte Aufnahme voices/default.wav.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

from .config import Config

# "default" ist die mitgelieferte Aufnahme für Modelle, die immer eine brauchen.
_RESERVED = {"default", "con", "prn", "aux", "nul"}
_NAME = re.compile(r"^\w[\w \-]{0,39}$")

MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
_FRAME_S = 0.02
_KEEP_AROUND_S = 0.15     # so viel Stille bleibt vor und nach der Sprache stehen


def voices_dir(config: Config) -> Path:
    """Ordner für eigene Stimmen (config.yaml: voices_dir, sonst im Benutzerprofil)."""
    if config.voices_dir:
        return config.resolve(config.voices_dir)
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "RTlation" / "voices"
    data_home = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data_home) / "rtlation" / "voices"


def shipped_voice(config: Config) -> Path:
    """Mitgelieferte Aufnahme für Modelle, die immer eine Referenz brauchen."""
    return config.root / "voices" / "default.wav"


def move_project_voices(config: Config) -> list[str]:
    """Verschiebt eigene Aufnahmen, die noch im Projektordner liegen, ins Benutzerprofil."""
    source, target = config.root / "voices", voices_dir(config)
    if source.resolve() == target.resolve():
        return []
    moved = []
    for wav in source.glob("*.wav"):
        if wav.stem.lower() in _RESERVED or (target / wav.name).exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wav), target / wav.name)
        if wav.with_suffix(".txt").is_file():
            shutil.move(str(wav.with_suffix(".txt")), target / f"{wav.stem}.txt")
        moved.append(wav.stem)
    if moved:
        logging.getLogger(__name__).info("Stimmen nach %s verschoben: %s", target, ", ".join(moved))
    return moved


def check_name(name: str) -> str:
    name = name.strip()
    if not _NAME.match(name) or name.lower() in _RESERVED:
        raise ValueError("Der Name darf nur Buchstaben, Ziffern, Leerzeichen, - und _ enthalten (höchstens 40 Zeichen).")
    return name


def voice_path(config: Config, name: str) -> Path:
    return voices_dir(config) / f"{check_name(name)}.wav"


def list_voices(config: Config) -> list[dict]:
    voices = []
    for path in sorted(voices_dir(config).glob("*.wav"), key=lambda p: p.stem.lower()):
        if path.stem.lower() in _RESERVED:
            continue
        try:
            with wave.open(str(path), "rb") as wav:
                seconds = wav.getnframes() / wav.getframerate()
        except (wave.Error, EOFError, OSError):
            continue
        voices.append({"name": path.stem, "seconds": round(seconds, 1)})
    return voices


def delete_voice(config: Config, name: str) -> None:
    path = voice_path(config, name)
    if not path.is_file():
        raise ValueError(f"Die Stimme „{name}“ gibt es nicht.")
    path.unlink()
    path.with_suffix(".txt").unlink(missing_ok=True)


def save_voice(config: Config, name: str, data: bytes) -> tuple[Path, float]:
    """Prüft und speichert eine Aufnahme (WAV, 16 Bit). Gibt (Datei, Dauer) zurück.

    Stille am Anfang und Ende wird abgeschnitten und die Lautstärke
    angeglichen, damit das Modell möglichst viel Stimme zu hören bekommt.
    """
    path = voice_path(config, name)
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("Die Aufnahme muss 16 Bit haben.")
            rate, channels = wav.getframerate(), wav.getnchannels()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Die Aufnahme ist keine gültige WAV-Datei.") from exc
    if not 16000 <= rate <= 48000:
        raise ValueError("Die Aufnahme muss eine Abtastrate zwischen 16 und 48 kHz haben.")

    audio = samples.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio[:len(audio) - len(audio) % channels].reshape(-1, channels).mean(axis=1)

    audio = _trim_silence(audio, rate)[:int(MAX_SECONDS * rate)]
    seconds = len(audio) / rate
    if seconds < MIN_SECONDS:
        raise ValueError(f"Die Aufnahme enthält nur {seconds:.1f} s Sprache; nötig sind mindestens {MIN_SECONDS:.0f} s.")
    peak = float(np.abs(audio).max())
    if peak > 0.02:
        audio = audio * (0.9 / peak)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    path.with_suffix(".txt").unlink(missing_ok=True)   # gehörte zur alten Aufnahme
    return path, seconds


def _trim_silence(audio: np.ndarray, rate: int) -> np.ndarray:
    frame = int(_FRAME_S * rate)
    count = len(audio) // frame
    if count < 3:
        return audio
    loudness = np.sqrt((audio[:count * frame].reshape(count, frame) ** 2).mean(axis=1))
    threshold = max(0.004, 0.08 * float(np.percentile(loudness, 95)))
    voiced = np.flatnonzero(loudness > threshold)
    if len(voiced) == 0:
        return audio[:0]
    keep = int(_KEEP_AROUND_S * rate)
    start = max(0, voiced[0] * frame - keep)
    end = min(len(audio), (voiced[-1] + 1) * frame + keep)
    return audio[start:end]
