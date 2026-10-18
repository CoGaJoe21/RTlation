"""Spielt eine WAV-Datei gegen einen laufenden RTlation-Server und zeigt, was zurückkommt.

    python tests/stream_sample.py [--target es] [--voice name] [--save ordner] [--no-speak] [datei.wav]

Die Datei muss 16 kHz, mono, 16 Bit sein. Das Audio wird im Echtzeit-Takt
gesendet, damit die gemessenen Verzögerungen stimmen. Mit --save landet die
Sprachausgabe jedes Satzes als WAV im angegebenen Ordner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

BLOCK = 1024
SAMPLE_RATE = 16000
DEFAULT_SAMPLE = Path(__file__).parent / "assets" / "de_sample.wav"


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--source", default="de")
    parser.add_argument("--target", default="en")
    parser.add_argument("--no-speak", action="store_true", help="nur Text, keine Sprachausgabe")
    parser.add_argument("--voice", default="", help="Name einer Stimme aus voices/ (leer = Standardstimme)")
    parser.add_argument("--save", type=Path, help="Ordner für die Sprachausgabe als WAV")
    args = parser.parse_args()

    with wave.open(str(args.wav), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            print("Die Datei muss 16 kHz, mono, 16 Bit sein.", file=sys.stderr)
            return 2
        pcm = wav.readframes(wav.getnframes()) + bytes(SAMPLE_RATE * 2)  # 1 s Stille am Ende

    translations = []
    heard = []                               # Verzögerung bis zum ersten Ton je Satz
    audio: dict[int, bytearray] = {}
    rates: dict[int, int] = {}
    # Wiedergabe nachstellen: Wann wäre der bisher angekommene Ton zu Ende?
    loop = asyncio.get_running_loop()
    played_until = 0.0
    playing = None          # Satz, dessen Ton zuletzt ankam
    gaps = []               # Aussetzer mitten in einem Satz, in ms
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "config", "source": args.source, "target": args.target, "speak": not args.no_speak,
            "voice": args.voice,
        }))

        async def send() -> None:
            step = BLOCK * 2
            for offset in range(0, len(pcm), step):
                await ws.send(pcm[offset:offset + step])
                await asyncio.sleep(BLOCK / SAMPLE_RATE)
            await ws.send(json.dumps({"type": "stop"}))

        sender = asyncio.create_task(send())
        waiting: set[int] = set()   # Sätze, auf deren Text oder Ton noch gewartet wird
        header = None
        quiet = 0.0                 # Sekunden ohne Ereignis, seit alles gesendet ist
        while not (sender.done() and not waiting and quiet >= 1.0):
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                if sender.done():
                    quiet += 0.5
                if quiet > 60:
                    print("Zeitüberschreitung: keine Antwort mehr vom Server.", file=sys.stderr)
                    return 1
                continue
            quiet = 0.0
            if isinstance(message, bytes):
                audio.setdefault(header["id"], bytearray()).extend(message)
                rates[header["id"]] = header["sample_rate"]
                now = loop.time()
                if playing == header["id"] and now > played_until:
                    gaps.append(round((now - played_until) * 1000))
                playing = header["id"]
                played_until = max(played_until, now) + len(message) / 2 / header["sample_rate"]
                continue
            event = json.loads(message)
            kind = event["type"]
            if kind == "speech_start":
                waiting.add(event["id"])
            elif kind == "transcript":
                print(f"[{event['id']}] {event['text']}   (Erkennung {event['stt_ms']} ms)")
            elif kind == "translation":
                translations.append(event)
                if not event.get("speaking"):
                    waiting.discard(event["id"])
                print(f"[{event['id']}] → {event['text']}   "
                      f"(Übersetzung {event['mt_ms']} ms, Text nach {event['latency_ms']} ms)")
            elif kind == "audio":
                header = event
                if "latency_ms" in event:
                    heard.append(event["latency_ms"])
                    print(f"[{event['id']}] ♪ erster Ton {event['latency_ms']} ms nach Sprechende")
            elif kind == "audio_end":
                waiting.discard(event["id"])
                print(f"[{event['id']}] ♪ {event['audio_ms']} ms Sprache in {event['tts_ms']} ms erzeugt, "
                      f"Ausdruck {event.get('expression', 0.5):.2f}")
            elif kind == "discard":
                waiting.discard(event["id"])
            elif kind == "error":
                print(f"FEHLER: {event['message']}", file=sys.stderr)
                return 1

    if not translations:
        print("Es kam keine Übersetzung zurück.", file=sys.stderr)
        return 1
    text_avg = sum(t["latency_ms"] for t in translations) / len(translations)
    print(f"\n{len(translations)} Sätze. Text im Schnitt {text_avg:.0f} ms nach Sprechende", end="")
    print(f", erster Ton nach {sum(heard) / len(heard):.0f} ms." if heard else ".")
    if heard:
        print(f"Aussetzer mitten im Satz: {len(gaps)}" + (f" (längster {max(gaps)} ms)" if gaps else ""))
    if args.save:
        for segment_id, data in audio.items():
            write_wav(args.save / f"{args.target}_{segment_id}.wav", bytes(data), rates[segment_id])
        print(f"Sprachausgabe gespeichert in {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
