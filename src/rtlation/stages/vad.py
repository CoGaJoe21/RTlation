"""Pausenerkennung: zerlegt den Audiostrom an Sprechpausen in Satzteile."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import VadConfig

SAMPLE_RATE = 16000
FRAME = 512                       # Samples pro VAD-Schritt (32 ms)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE
_CONTEXT = 64                     # Samples, die Silero aus dem vorigen Schritt mitbekommt


class SileroVad:
    """Silero VAD als ONNX-Modell. Der Zustand liegt beim Aufrufer,
    damit sich mehrere Verbindungen ein geladenes Modell teilen können."""

    def __init__(self, model_path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    @staticmethod
    def new_state() -> dict[str, np.ndarray]:
        return {
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "context": np.zeros(_CONTEXT, dtype=np.float32),
        }

    def probability(self, frame: np.ndarray, state: dict[str, np.ndarray]) -> float:
        """Sprachwahrscheinlichkeit für einen Frame (float32, FRAME Samples)."""
        x = np.concatenate([state["context"], frame])[np.newaxis, :]
        out, new_state = self._session.run(
            None, {"input": x, "state": state["state"], "sr": self._sr}
        )
        state["state"] = new_state
        state["context"] = frame[-_CONTEXT:]
        return float(out[0, 0])


@dataclass
class Segment:
    id: int
    audio: np.ndarray        # float32, mono, 16 kHz
    speech_ended_at: float   # time.monotonic(), als der Sprecher aufgehört hat

    @property
    def duration_ms(self) -> int:
        return int(len(self.audio) * 1000 / SAMPLE_RATE)


@dataclass
class SpeechStart:
    id: int


@dataclass
class SpeechEnd:
    segment: Segment


@dataclass
class SpeechDiscarded:
    id: int


Event = SpeechStart | SpeechEnd | SpeechDiscarded


class Segmenter:
    """Nimmt PCM-Daten in beliebig grossen Stücken an und meldet Satzteile."""

    def __init__(self, vad: SileroVad, config: VadConfig):
        self._vad = vad
        self._cfg = config
        self.min_silence_ms = config.min_silence_ms
        self._state = vad.new_state()
        self._pending = np.empty(0, dtype=np.float32)
        pad_frames = max(1, round(config.speech_pad_ms / FRAME_MS))
        self._pad_frames = pad_frames
        self._preroll: deque[np.ndarray] = deque(maxlen=pad_frames)
        self._frames: list[np.ndarray] = []
        self._active = False
        self._id = 0
        self._speech_frames = 0
        self._silence_frames = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def current_id(self) -> int:
        return self._id

    @property
    def current_duration_ms(self) -> float:
        return len(self._frames) * FRAME_MS if self._active else 0.0

    def current_audio(self) -> np.ndarray:
        """Kopie des bisher gesprochenen, noch offenen Satzteils."""
        return np.concatenate(self._frames) if self._frames else np.empty(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> list[Event]:
        """pcm: int16, mono, 16 kHz."""
        samples = pcm.astype(np.float32) / 32768.0
        data = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        events: list[Event] = []
        offset = 0
        while offset + FRAME <= len(data):
            self._step(data[offset:offset + FRAME], events)
            offset += FRAME
        self._pending = data[offset:].copy()
        return events

    def flush(self) -> list[Event]:
        """Schliesst einen offenen Satzteil sofort ab (z.B. beim Stoppen)."""
        events: list[Event] = []
        if self._active:
            self._close(events)
        self._pending = np.empty(0, dtype=np.float32)
        self._state = self._vad.new_state()
        return events

    def _step(self, frame: np.ndarray, events: list[Event]) -> None:
        prob = self._vad.probability(frame, self._state)
        threshold = self._cfg.threshold

        if not self._active:
            if prob >= threshold:
                self._active = True
                self._id += 1
                self._frames = [*self._preroll, frame]
                self._preroll.clear()
                self._speech_frames = 1
                self._silence_frames = 0
                events.append(SpeechStart(self._id))
            else:
                self._preroll.append(frame)
            return

        self._frames.append(frame)
        if prob >= threshold:
            self._speech_frames += 1
            self._silence_frames = 0
        elif prob < threshold - 0.15:
            self._silence_frames += 1

        duration_ms = len(self._frames) * FRAME_MS
        # Nach langem Reden am Stück reicht schon eine kurze Pause zum Schneiden.
        needed_ms = self.min_silence_ms
        if duration_ms >= self._cfg.soft_max_segment_s * 1000:
            needed_ms = min(needed_ms, 200)
        if (
            self._silence_frames * FRAME_MS >= needed_ms
            or duration_ms >= self._cfg.max_segment_s * 1000
        ):
            self._close(events)

    def _close(self, events: list[Event]) -> None:
        silence = self._silence_frames
        speech_ended_at = time.monotonic() - silence * FRAME_MS / 1000
        drop = max(0, silence - self._pad_frames)
        frames = self._frames[:len(self._frames) - drop] if drop else self._frames

        if self._speech_frames * FRAME_MS >= self._cfg.min_speech_ms:
            audio = np.concatenate(frames)
            events.append(SpeechEnd(Segment(self._id, audio, speech_ended_at)))
        else:
            events.append(SpeechDiscarded(self._id))

        self._active = False
        self._frames = []
        self._speech_frames = 0
        self._silence_frames = 0
