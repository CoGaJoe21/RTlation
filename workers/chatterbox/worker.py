"""Sprachausgabe mit Chatterbox Multilingual (läuft in .tools/venvs/chatterbox).

Chatterbox erzeugt zuerst Sprach-Tokens (25 pro Sekunde) und wandelt sie dann
in Ton um. Die Bibliothek macht das für den ganzen Satz am Stück. Hier wird
der Ton stückweise ausgegeben, während die Tokens noch entstehen, damit der
Anfang des Satzes früher zu hören ist. Das Zusammensetzen der Stücke folgt
dem Verfahren von CosyVoice 2, von dem der Decoder abstammt.
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base import serve  # noqa: E402

# Einstellungen, die aus config.yaml (tts.options) durchgereicht werden dürfen.
_OPTIONS = ("exaggeration", "cfg_weight", "temperature")

# Grösse der Stücke in Tokens (25 Tokens = 1 s Ton). Das erste Stück muss so
# viel Ton liefern, dass er reicht, bis das zweite fertig ist. Danach richtet
# sich die Grösse nach dem Vorrat: Jedes Stück kostet den Decoder gleich viel
# Zeit, egal wie lang es ist, also sind wenige grosse Stücke günstiger.
_FIRST_HOP = 30
_MAX_HOP = 32
_TOKEN_AUDIO_S = 0.04    # ein Token ergibt 40 ms Ton
_SAFETY_S = 0.2          # so viel Ton soll beim Eintreffen des nächsten Stücks noch übrig sein
# Der Decoder braucht gut 30 ms pro Schritt. Sechs statt zehn Schritte verändern
# das Ergebnis deutlich weniger als der Zufall zwischen zwei normalen
# Durchläufen (gemessen), sparen aber über 100 ms pro Stück.
_STREAM_STEPS = 6
_MEL_CACHE = 8           # Mel-Frames, die sich zwei aufeinanderfolgende Stücke teilen
_SAMPLES_PER_FRAME = 480
_OVERLAP = _MEL_CACHE * _SAMPLES_PER_FRAME   # überblendeter Bereich in Samples (160 ms)


class _Aborted(Exception):
    pass


class Chatterbox:
    def __init__(self, args):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._torch = torch
        self._model = ChatterboxMultilingualTTS.from_local(args.model_dir, args.device, t3_model="v3")
        self._builtin = self._model.conds          # mitgelieferte Standardstimme
        self._voices: dict[tuple[str, float], object] = {}
        # Festes Rauschen als Startpunkt des Decoders: Nur so ergibt derselbe
        # Satzanfang bei jedem Stück wieder dasselbe Ergebnis.
        self._noise = torch.randn(1, 80, 50 * 120, device=self._model.device)
        # Hann-Fenster: steigende und fallende Hälfte ergänzen sich an jeder Stelle zu 1.
        self._window = torch.hann_window(2 * _OVERLAP, periodic=True, device=self._model.device)
        # Gemessene Geschwindigkeiten dieses Rechners (gleitende Mittelwerte);
        # daraus ergibt sich, wie gross das nächste Stück sein darf.
        self._token_s = 0.022
        self._decode_s = 0.30
        self.last_tokens: list[int] = []           # für Tests

    def synthesize(self, text, *, language, voice, options, **_):
        self._select_voice(voice)
        settings = {key: float(options[key]) for key in _OPTIONS if key in options}
        if options.get("stream", True):
            yield from self._stream(text, language, settings)
        else:
            wav = self._model.generate(text, language_id=language, **settings)
            yield wav.squeeze(0).numpy(), self._model.sr

    def prepare(self, *, voice, **_):
        self._select_voice(voice)

    def _select_voice(self, voice: str | None) -> None:
        if not voice:
            self._model.conds = self._builtin
            return
        key = (voice, os.path.getmtime(voice))   # neu aufgenommene Stimme = neuer Eintrag
        if key not in self._voices:
            self._model.prepare_conditionals(voice)
            self._voices[key] = self._model.conds
        self._model.conds = self._voices[key]

    # ------------------------------------------------------ stückweise Ausgabe

    def _stream(self, text, language, settings):
        torch = self._torch
        from chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE

        model = self._model
        text_tokens = self._text_tokens(text, language, settings.get("exaggeration", 0.5))
        conds = model.conds

        # Die Bibliothek gibt Tokens erst am Satzende heraus. Jeder neue Token
        # läuft aber durch speech_emb - dort wird mitgelesen.
        produced: queue.Queue = queue.Queue()
        abort = threading.Event()
        failure: list[BaseException] = []
        # Tokens erzeugen und Ton berechnen bremsen sich gegenseitig stark aus,
        # wenn sie gleichzeitig laufen (beides hängt an Python). Deshalb wartet
        # die Token-Erzeugung, solange ein Stück in Ton umgewandelt wird.
        may_generate = threading.Event()
        may_generate.set()

        stamps: list[float] = []

        def on_token(_module, inputs):
            produced.put(int(inputs[0].reshape(-1)[0]))
            stamps.append(time.perf_counter())
            may_generate.wait()
            if abort.is_set():
                raise _Aborted

        def generate_tokens():
            hook = model.t3.speech_emb.register_forward_pre_hook(on_token)
            try:
                model.t3.inference(
                    t3_cond=conds.t3,
                    text_tokens=text_tokens,
                    # Grosszügige Obergrenze, damit ein entgleister Satz nicht 40 s lang redet.
                    max_new_tokens=min(1000, 75 + 12 * len(text)),
                    temperature=settings.get("temperature", 0.8),
                    cfg_weight=settings.get("cfg_weight", 0.5),
                    repetition_penalty=1.2,
                    min_p=0.05,
                    top_p=1.0,
                )
            except _Aborted:
                pass
            except BaseException as exc:
                failure.append(exc)
            finally:
                hook.remove()
                produced.put(None)

        thread = threading.Thread(target=generate_tokens, name="t3", daemon=True)
        thread.start()

        lookahead = model.s3gen.flow.pre_lookahead_len
        tokens: list[int] = []
        offset = 0          # so viele Tokens sind schon als Ton ausgegeben
        cache = None
        tail = np.zeros(0, dtype=np.float32)   # Ende des schon ausgegebenen Tons
        finished = False
        hop = _FIRST_HOP
        played_until = 0.0  # bis zu diesem Zeitpunkt reicht der schon ausgegebene Ton
        try:
            while not finished:
                item = produced.get()
                while True:
                    if item is None:
                        finished = True
                    elif item < SPEECH_VOCAB_SIZE:   # Start- und Endmarken überspringen
                        tokens.append(item)
                    try:
                        item = produced.get_nowait()
                    except queue.Empty:
                        break
                if failure:
                    raise failure[0]

                if finished:
                    # Der letzte Token vor dem Satzende klingt nach Rauschen
                    # (die Bibliothek schneidet ihn ebenfalls ab).
                    final = tokens[:-1] if len(tokens) > 1 else tokens
                    if len(final) > offset:
                        audio, tail, cache = self._decode(final, offset, cache, tail, finalize=True)
                        yield audio, model.sr
                else:
                    ready = len(tokens) - offset - lookahead
                    if ready >= hop:
                        prefix = tokens[:offset + ready + lookahead]
                        may_generate.clear()
                        try:
                            started = time.perf_counter()
                            audio, tail, cache = self._decode(prefix, offset, cache, tail, finalize=False)
                            now = time.perf_counter()
                        finally:
                            may_generate.set()
                        offset += ready
                        self._decode_s += 0.5 * (now - started - self._decode_s)
                        # Wie viele Tokens können entstehen und umgewandelt werden,
                        # bevor der Vorrat an Ton aufgebraucht ist?
                        played_until = max(played_until, now) + len(audio) / model.sr
                        spare = played_until - now - self._decode_s - _SAFETY_S
                        hop = max(self._smallest_hop(), min(_MAX_HOP, int(spare / self._token_s)))
                        yield audio, model.sr
            self.last_tokens = tokens
            # Tempo der Token-Erzeugung; Pausen während des Decoders zählen nicht mit.
            gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b - a < 0.1)
            if len(gaps) >= 10:
                self._token_s += 0.5 * (gaps[len(gaps) // 2] - self._token_s)
        finally:
            abort.set()
            may_generate.set()
            thread.join()

    def _smallest_hop(self) -> int:
        """Kleinste sinnvolle Stückgrösse. Darunter kostet ein Stück mehr Zeit,
        als es Ton liefert - der Vorrat würde mit jedem Stück kleiner."""
        gain_per_token = max(_TOKEN_AUDIO_S - self._token_s, 0.002)
        return min(_MAX_HOP, math.ceil(1.3 * self._decode_s / gain_per_token))

    def _text_tokens(self, text, language, exaggeration):
        """Vorbereitung wie in ChatterboxMultilingualTTS.generate()."""
        torch = self._torch
        import torch.nn.functional as F
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.mtl_tts import punc_norm

        model = self._model
        current = model.conds.t3
        if float(exaggeration) != float(current.emotion_adv[0, 0, 0].item()):
            model.conds.t3 = T3Cond(
                speaker_emb=current.speaker_emb,
                cond_prompt_speech_tokens=current.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)

        tokens = model.tokenizer.text_to_tokens(punc_norm(text), language_id=language).to(model.device)
        tokens = torch.cat([tokens, tokens], dim=0)   # zweite Zeile für die Classifier-Free Guidance
        tokens = F.pad(tokens, (1, 0), value=model.t3.hp.start_text_token)
        return F.pad(tokens, (0, 1), value=model.t3.hp.stop_text_token)

    def _decode(self, tokens, offset, cache, before, *, finalize):
        """Wandelt den Satzanfang bis hierher in Ton um.

        cache = (Mel, Anregungssignal, Ton) vom Ende des vorigen Stücks. Der
        Vocoder rechnet diese 160 ms noch einmal mit, und die beiden Fassungen
        werden überblendet - so gibt es an der Nahtstelle kein Knacken.
        before = das Ende des zuvor ausgegebenen Tons (ohne Wasserzeichen).

        Gibt zurück: (neuer Ton, dessen Ende als nächstes "before", neuer cache).
        """
        torch = self._torch
        model = self._model
        s3gen = model.s3gen
        with torch.inference_mode():
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)
            with self._fixed_noise():
                mel = s3gen.flow_inference(
                    token_tensor, ref_dict=model.conds.gen, finalize=finalize, n_cfm_timesteps=_STREAM_STEPS
                )
            mel = mel[:, :, offset * 2:].to(dtype=s3gen.dtype)

            if cache is None:
                wav, source = s3gen.hift_inference(mel, None)
                wav[:, :len(s3gen.trim_fade)] *= s3gen.trim_fade
            else:
                cached_mel, cached_source, cached_wav = cache
                mel = torch.cat([cached_mel, mel], dim=2)
                wav, source = s3gen.hift_inference(mel, cached_source)
                wav[:, :_OVERLAP] = wav[:, :_OVERLAP] * self._window[:_OVERLAP] + cached_wav * self._window[_OVERLAP:]

            cache = None if finalize else (mel[:, :, -_MEL_CACHE:], source[:, :, -_OVERLAP:], wav[:, -_OVERLAP:].clone())
            raw = wav.squeeze(0).float().cpu().numpy()

        # Das Ende bleibt für die Überblendung mit dem nächsten Stück zurück.
        emit, held_back = (raw, raw[:0]) if finalize else (raw[:-_OVERLAP], raw[-_OVERLAP:])
        # Das Wasserzeichen der Bibliothek bekommt den Ton davor und danach mit,
        # sonst entstehen an den Stückgrenzen hörbare Sprünge (gemessen).
        marked = model.watermarker.apply_watermark(
            np.concatenate([before, emit, held_back]), sample_rate=model.sr
        )
        return marked[len(before):len(before) + len(emit)], emit[-_OVERLAP:], cache

    @contextmanager
    def _fixed_noise(self):
        """Lässt den Decoder mit dem festen Rauschen statt mit neuem Zufall starten."""
        torch = self._torch
        cfm = self._model.s3gen.flow.decoder

        def forward(mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False):
            z = self._noise[:, :, :mu.size(2)].to(dtype=mu.dtype)
            # Bei einem Zwischenstück kürzt die Bibliothek mu um die Vorausschau,
            # die Maske aber nicht.
            mask = mask[..., :mu.size(2)]
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if cfm.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
            return cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

        cfm.forward = forward
        try:
            yield
        finally:
            del cfm.forward   # wieder die Methode der Klasse


if __name__ == "__main__":
    serve(Chatterbox)
