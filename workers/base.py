"""Gemeinsamer Rahmen für die Sprachausgabe-Prozesse.

Jedes Sprachmodell läuft in einem eigenen Prozess mit eigener Python-Umgebung,
weil sich die Abhängigkeiten der Modelle gegenseitig ausschliessen. Der
Hauptprozess öffnet einen lokalen Port und startet den Worker; dieser meldet
sich dort, sobald sein Modell geladen ist.

Protokoll (TCP, nur 127.0.0.1):

  Hauptprozess -> Worker: eine JSON-Zeile pro Auftrag
      {"id": 1, "text": "...", "language": "en", "language_name": "English",
       "voice": "pfad.wav" | null, "voice_text": "..." | null, "options": {}}
      {"id": 2, "prepare": true, "voice": "pfad.wav", "voice_text": "..." | null}
          lädt nur die Stimme vor; Antwort ist ein einzelnes "done"

  Worker -> Hauptprozess: Rahmen aus [4 Byte Länge][JSON-Kopf][Nutzdaten]
      {"type": "ready", "token": "..."}
      {"type": "audio", "id": 1, "sample_rate": 24000, "bytes": N}   + N Bytes PCM (16 Bit, mono)
      {"type": "done",  "id": 1}
      {"type": "error", "id": 1, "message": "..."}

Diese Datei darf nur die Standardbibliothek und numpy verwenden.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])\s+|(?<=[。！？])")
_MIN_SENTENCE_CHARS = 25


class Engine(Protocol):
    def synthesize(self, text: str, *, language: str, language_name: str, voice: str | None,
                   voice_text: str | None, options: dict) -> Iterator[tuple[np.ndarray, int]]:
        """Liefert den Satz als ein oder mehrere Stücke (Audio float32 mono, Abtastrate)."""


def split_sentences(text: str) -> list[str]:
    """Teilt an Satzenden, damit der erste Satz schon läuft, während der
    nächste noch berechnet wird. Sehr kurze Stücke bleiben am Nachbarn."""
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < _MIN_SENTENCE_CHARS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and len(merged[-1]) < _MIN_SENTENCE_CHARS:
        merged[-2:] = [f"{merged[-2]} {merged[-1]}"]
    return merged or [text]


def _send(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    if payload:
        header["bytes"] = len(payload)
    head = json.dumps(header).encode("utf-8")
    sock.sendall(len(head).to_bytes(4, "big") + head + payload)


def _to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def serve(load: Callable[[argparse.Namespace], Engine]) -> None:
    """Lädt das Modell, meldet sich beim Hauptprozess und arbeitet Aufträge ab."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", type=int, required=True, help="Port des Hauptprozesses")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    engine = load(args)

    sock = socket.create_connection(("127.0.0.1", args.connect))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _send(sock, {"type": "ready", "token": os.environ.get("RTLATION_WORKER_TOKEN", "")})

    # Endet, sobald der Hauptprozess die Verbindung schliesst.
    for line in sock.makefile("r", encoding="utf-8"):
        request = json.loads(line)
        request_id = request["id"]
        try:
            if request.get("prepare"):
                # Stimme vorab laden, damit der erste Satz damit nicht warten muss.
                if hasattr(engine, "prepare"):
                    engine.prepare(voice=request.get("voice"), voice_text=request.get("voice_text"))
                _send(sock, {"type": "done", "id": request_id})
                continue
            for sentence in split_sentences(request["text"]):
                pieces = engine.synthesize(
                    sentence,
                    language=request["language"],
                    language_name=request["language_name"],
                    voice=request.get("voice"),
                    voice_text=request.get("voice_text"),
                    options=request.get("options") or {},
                )
                for audio, sample_rate in pieces:
                    _send(sock, {"type": "audio", "id": request_id, "sample_rate": int(sample_rate)},
                          _to_pcm16(audio))
            _send(sock, {"type": "done", "id": request_id})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send(sock, {"type": "error", "id": request_id, "message": f"{type(exc).__name__}: {exc}"})
