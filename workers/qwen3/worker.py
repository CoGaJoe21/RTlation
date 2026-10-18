"""Sprachausgabe mit Qwen3-TTS (läuft in .tools/venvs/qwen3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base import serve  # noqa: E402


class Qwen3:
    def __init__(self, args):
        import torch
        from qwen_tts import Qwen3TTSModel

        device = "cuda:0" if args.device == "cuda" else args.device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._model = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map=device, dtype=dtype)
        self._prompts: dict[tuple[str, float], object] = {}   # Referenzaufnahme -> vorbereitete Stimme

    def prepare(self, *, voice, voice_text, **_):
        if voice:
            self._prompt(voice, voice_text)

    def synthesize(self, text, *, language_name, voice, voice_text, **_):
        if not voice:
            raise ValueError("Qwen3-TTS braucht eine Referenzaufnahme der Stimme.")
        wavs, sample_rate = self._model.generate_voice_clone(
            text=text, language=language_name, voice_clone_prompt=self._prompt(voice, voice_text)
        )
        yield wavs[0], sample_rate

    def _prompt(self, voice, voice_text):
        key = (voice, os.path.getmtime(voice))   # neu aufgenommene Stimme = neuer Eintrag
        if key not in self._prompts:
            # Mit Abschrift der Aufnahme klont das Modell genauer; ohne sie
            # nimmt es nur die Klangfarbe.
            self._prompts[key] = self._model.create_voice_clone_prompt(
                ref_audio=voice, ref_text=voice_text or None, x_vector_only_mode=not voice_text
            )
        return self._prompts[key]


if __name__ == "__main__":
    serve(Qwen3)
