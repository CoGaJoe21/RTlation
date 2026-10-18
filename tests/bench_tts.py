"""Misst ein Sprachmodell: Ladezeit, Rechenzeit pro Satz und Echtzeitfaktor.

    python tests/bench_tts.py chatterbox [--voice name] [--out ordner] [--check]

Zur Wahl stehen die Einträge aus rtlation.stages.tts.ENGINES, also auch die
Betriebsarten desselben Modells (chatterbox = stückweise, chatterbox-full =
ganzer Satz am Stück).

Der Server darf dabei nicht laufen (sonst ist das Modell doppelt im
Grafikspeicher). Mit --check wird jede Ausgabe mit Whisper zurück in Text
verwandelt, damit sichtbar ist, ob das Gesprochene zum Text passt.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import wave
from pathlib import Path

from rtlation.config import load_config
from rtlation.cuda import add_cuda_dll_dirs
from rtlation.languages import LANGUAGES
from rtlation.stages.tts import ENGINES, TtsWorker, engine_options, resolve_voice

SENTENCES = {
    "en": [
        "Good day, I am currently testing a new program.",
        "It is supposed to translate my voice into another language in real time.",
        "Tomorrow I'm taking the train to Zurich, where I'll meet an old friend.",
    ],
    "es": [
        "Buenos días, estoy probando un nuevo programa.",
        "El objetivo es que mi voz se traduzca a otro idioma en tiempo real.",
        "Mañana viajaré en tren a Zúrich y allí me reuniré con un viejo amigo.",
    ],
    "zh-Hans": [
        "您好，我现在正在测试一个新程序。",
        "它的目标是实时将我的声音翻译成另一种语言。",
        "明天我将乘坐火车前往苏黎世，在那里与一位老朋友见面。",
    ],
}


def gpu_memory_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=sorted(ENGINES), help="Sprachmodell bzw. Betriebsart")
    parser.add_argument("--voice", default=None, help="Name einer Aufnahme in voices/ (ohne .wav)")
    parser.add_argument("--languages", default="en,es,zh-Hans")
    parser.add_argument("--out", type=Path, default=Path("tests/out"))
    parser.add_argument("--check", action="store_true", help="Ausgabe mit Whisper gegenprüfen")
    args = parser.parse_args()

    config = load_config()
    if args.voice is not None:
        config.tts.voice = args.voice
    voice, voice_text = resolve_voice(config, args.engine, config.tts.voice)
    options = engine_options(config, args.engine)

    before = gpu_memory_mb()
    worker = TtsWorker(config, args.engine)
    started = time.perf_counter()
    await worker.start()
    print(f"{ENGINES[args.engine].label}: geladen in {time.perf_counter() - started:.1f} s")

    results = []   # (Sprache, Text, Rechenzeit s, Audiodauer s, Datei)
    firsts = []    # Zeit bis zum ersten Ton je Satz
    try:
        started = time.perf_counter()
        async for _ in worker.synthesize("Hello, this is a short test.", LANGUAGES["en"], voice, voice_text, options):
            pass
        print(f"erster Satz (Aufwärmen): {time.perf_counter() - started:.1f} s")
        after = gpu_memory_mb()
        if before is not None and after is not None:
            print(f"Grafikspeicher: +{after - before} MB")

        for code in args.languages.split(","):
            language = LANGUAGES[code]
            for index, text in enumerate(SENTENCES[code], start=1):
                started = time.perf_counter()
                pcm, rate, first, pieces = bytearray(), 0, 0.0, 0
                async for chunk, rate in worker.synthesize(text, language, voice, voice_text, options):
                    first = first or time.perf_counter() - started
                    pieces += 1
                    pcm.extend(chunk)
                seconds = time.perf_counter() - started
                duration = len(pcm) / 2 / rate
                path = args.out / f"{args.engine}_{code}_{index}.wav"
                write_wav(path, bytes(pcm), rate)
                results.append((code, text, seconds, duration, path))
                firsts.append(first)
                print(f"  {code:8} erster Ton nach {first:4.2f} s, fertig nach {seconds:5.2f} s für {duration:5.2f} s Sprache "
                      f"(Faktor {seconds / duration:.2f}, {pieces} Stück{'e' if pieces != 1 else ''})")
    finally:
        await worker.close()

    total_time = sum(r[2] for r in results)
    total_audio = sum(r[3] for r in results)
    print(f"\nGesamt: {total_time:.1f} s Rechenzeit für {total_audio:.1f} s Sprache, "
          f"Echtzeitfaktor {total_time / total_audio:.2f} (unter 1 = schneller als gesprochen), "
          f"erster Ton im Schnitt nach {sum(firsts) / len(firsts):.2f} s")

    if args.check:
        add_cuda_dll_dirs(config.root)
        from faster_whisper.audio import decode_audio

        from rtlation.stages.stt import Transcriber

        transcriber = Transcriber(config)
        print("\nGegenprobe mit Whisper:")
        for code, text, _, _, path in results:
            heard = transcriber.transcribe(decode_audio(str(path), sampling_rate=16000), LANGUAGES[code].whisper)
            print(f"  {code:8} soll: {text}\n  {'':8} ist:  {heard.text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
