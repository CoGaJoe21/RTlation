"""Pausenerkennung auf der Test-Aufnahme (drei Sätze mit Pausen dazwischen)."""

import wave
from pathlib import Path

import numpy as np
import pytest

from rtlation.config import VadConfig
from rtlation.stages.vad import SAMPLE_RATE, Segmenter, SileroVad, SpeechEnd, SpeechStart

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "vad" / "silero_vad.onnx"
SAMPLE = ROOT / "tests" / "assets" / "de_sample.wav"

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason="VAD-Modell fehlt (setup.ps1 ausführen)")


def read_sample() -> np.ndarray:
    with wave.open(str(SAMPLE), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def run(pcm: np.ndarray, chunk: int = 1024) -> list:
    segmenter = Segmenter(SileroVad(MODEL), VadConfig())
    events = []
    for start in range(0, len(pcm), chunk):
        events += segmenter.feed(pcm[start:start + chunk])
    return events + segmenter.flush()


def test_three_sentences_become_three_segments():
    events = run(read_sample())
    starts = [e for e in events if isinstance(e, SpeechStart)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert len(starts) == len(ends) == 3
    assert [e.segment.id for e in ends] == [1, 2, 3]
    for end in ends:
        assert 1500 < end.segment.duration_ms < 8000


def test_chunk_size_does_not_change_the_result():
    pcm = read_sample()
    lengths = lambda events: [len(e.segment.audio) for e in events if isinstance(e, SpeechEnd)]
    assert lengths(run(pcm, chunk=1024)) == lengths(run(pcm, chunk=333))


def test_silence_produces_nothing():
    assert run(np.zeros(SAMPLE_RATE * 3, dtype=np.int16)) == []


def test_long_speech_is_cut_at_the_hard_limit():
    config = VadConfig(soft_max_segment_s=1.0, max_segment_s=2.0)
    segmenter = Segmenter(SileroVad(MODEL), config)
    # Der erste Satz dauert länger als zwei Sekunden und hat keine Pause.
    events = segmenter.feed(read_sample()[: SAMPLE_RATE * 4])
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert ends and ends[0].segment.duration_ms <= 2000 + 32
