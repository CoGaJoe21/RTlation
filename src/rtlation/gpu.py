"""Grafikspeicher: Bedarf der gewählten Modelle und aktuelle Belegung."""

from __future__ import annotations

import subprocess

# Gemessen auf einer RTX 4070 Ti SUPER (Differenz vor/nach dem Laden, in MB).
# Auf anderen Karten weicht der Wert etwas ab; unbekannte Modelle fehlen.
_NEEDED_MB = {
    "stt": {"large-v3-turbo": 2350, "large-v3": 3900},
    "translate": {"translategemma:4b": 3750},
    "tts": {"chatterbox": 3900, "qwen3": 2700},
}


def needed(stt_model: str, translate_model: str, tts_worker: str | None) -> dict:
    """Bedarf der aktuellen Auswahl. tts_worker: None, wenn keine Sprachausgabe läuft."""
    parts = [
        ("Spracherkennung", _NEEDED_MB["stt"].get(stt_model)),
        ("Übersetzung", _NEEDED_MB["translate"].get(translate_model)),
    ]
    if tts_worker is not None:
        parts.append(("Sprachausgabe", _NEEDED_MB["tts"].get(tts_worker)))
    return {
        "parts": [{"label": label, "mb": mb} for label, mb in parts],
        "total_mb": sum(mb for _, mb in parts if mb is not None),
        "complete": all(mb is not None for _, mb in parts),   # False: ein Modell ist nicht vermessen
    }


def usage() -> dict | None:
    """Belegter und gesamter Speicher der ersten Grafikkarte; None ohne nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        used, total = (int(value) for value in out.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"used_mb": used, "total_mb": total}
