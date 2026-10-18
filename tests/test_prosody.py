"""Messung der Sprech-Energie (rtlation.prosody)."""

import numpy as np

from rtlation.prosody import MAX_EXPRESSION, MIN_EXPRESSION, SAMPLE_RATE, ExpressionTracker, measure


def voice(seconds: float, level: float, pitch_swing: float) -> np.ndarray:
    """Künstliche Stimme: Grundton 140 Hz mit Obertönen, Tonhöhe schwankt um pitch_swing Halbtöne."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    frequency = 140 * 2 ** (pitch_swing * np.sin(2 * np.pi * 0.7 * t) / 12)
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (level * signal / np.abs(signal).max()).astype(np.float32)


def test_measures_loudness_pitch_movement_and_rate():
    calm = measure(voice(3, 0.1, 0.5), characters=30)
    lively = measure(voice(3, 0.4, 4.0), characters=60)
    assert lively.loudness_db - calm.loudness_db > 10
    assert lively.pitch_spread > 3 * calm.pitch_spread
    assert lively.rate == 2 * calm.rate == 20


def test_silence_and_snippets_give_no_measurement():
    assert measure(np.zeros(SAMPLE_RATE, dtype=np.float32), 10) is None
    assert measure(voice(0.05, 0.3, 1), 3) is None


def test_lively_sentences_raise_and_calm_ones_lower_the_expression():
    tracker = ExpressionTracker()
    normal = voice(3, 0.2, 2.0)
    assert tracker.expression(0.5, normal, 45) == 0.5          # erster Satz ist der Massstab
    assert abs(tracker.expression(0.5, normal, 45) - 0.5) < 0.02
    assert tracker.expression(0.5, voice(3, 0.5, 5.0), 60) > 0.65
    assert tracker.expression(0.5, voice(3, 0.06, 0.3), 30) < 0.4


def test_expression_stays_in_range():
    tracker = ExpressionTracker()
    tracker.expression(0.9, voice(3, 0.05, 0.2), 20)
    assert tracker.expression(0.95, voice(3, 0.9, 8.0), 90) <= MAX_EXPRESSION
    tracker = ExpressionTracker()
    tracker.expression(0.3, voice(3, 0.9, 8.0), 90)
    assert tracker.expression(0.3, voice(3, 0.02, 0.1), 15) >= MIN_EXPRESSION
