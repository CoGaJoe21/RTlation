"""Wie lebhaft wurde ein Satz gesprochen?

Die Übersetzung trägt nur Wörter weiter. Damit die Sprachausgabe nicht immer
gleich ruhig klingt, wird aus jedem gesprochenen Satz grob die Energie
gemessen (Lautstärke, Bewegung der Tonhöhe, Tempo) und mit dem verglichen,
was für diesen Sprecher bisher üblich war. Feine Gefühle erfasst das nicht.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
_FRAME = 640          # 40 ms
_HOP = 320            # 20 ms
_FFT = 2048
_MIN_LAG = SAMPLE_RATE // 350   # höchste gesuchte Stimmlage: 350 Hz
_MAX_LAG = SAMPLE_RATE // 70    # tiefste: 70 Hz

# Gewicht der drei Merkmale und wie stark sie den Ausdruck verschieben.
_WEIGHTS = (0.5, 0.3, 0.2)      # Lautstärke, Tonhöhe, Tempo
_STRENGTH = 0.2                 # Ausdruck je Standardabweichung
# Übliche Streuung, solange noch zu wenig vom Sprecher bekannt ist.
_DEFAULT_SPREAD = (4.0, 1.0, 0.2)   # dB, Halbtöne, Anteil am Tempo
_LEARN = 0.2                    # wie schnell sich "üblich" anpasst
MIN_EXPRESSION, MAX_EXPRESSION = 0.25, 1.0


@dataclass
class Features:
    loudness_db: float      # mittlere Lautstärke der stimmhaften Abschnitte
    pitch_spread: float     # Streuung der Tonhöhe in Halbtönen
    rate: float             # Zeichen pro Sekunde


def measure(audio: np.ndarray, characters: int) -> Features | None:
    """audio: float32, mono, 16 kHz. None, wenn zu wenig Stimme enthalten ist."""
    if len(audio) < _FRAME * 4:
        return None
    count = 1 + (len(audio) - _FRAME) // _HOP
    index = np.arange(_FRAME)[None, :] + _HOP * np.arange(count)[:, None]
    frames = audio[index] * np.hanning(_FRAME)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms > max(0.003, 0.15 * float(np.percentile(rms, 95)))
    if loud.sum() < 5:
        return None

    # Tonhöhe über die Autokorrelation: Bei stimmhaften Lauten wiederholt sich
    # das Signal im Abstand einer Stimmbandschwingung.
    spectrum = np.abs(np.fft.rfft(frames[loud], n=_FFT)) ** 2
    correlation = np.fft.irfft(spectrum)[:, :_MAX_LAG + 1]
    strength = correlation[:, _MIN_LAG:] / np.maximum(correlation[:, :1], 1e-12)
    lag = strength.argmax(axis=1) + _MIN_LAG
    voiced = strength.max(axis=1) > 0.45
    pitch_spread = 0.0
    if voiced.sum() >= 5:
        semitones = 12 * np.log2(SAMPLE_RATE / lag[voiced])
        low, high = np.percentile(semitones, [10, 90])   # Ausreisser (Oktavfehler) abschneiden
        pitch_spread = float(np.std(np.clip(semitones, low, high)))

    return Features(
        loudness_db=float(20 * np.log10(rms[loud].mean())),
        pitch_spread=pitch_spread,
        rate=characters / (len(audio) / SAMPLE_RATE),
    )


class ExpressionTracker:
    """Lernt, was für einen Sprecher üblich ist, und bewertet jeden Satz dagegen."""

    def __init__(self) -> None:
        self._mean: list[float] | None = None
        self._spread = list(_DEFAULT_SPREAD)
        self._seen = 0

    def expression(self, base: float, audio: np.ndarray, characters: int) -> float:
        """Ausdruck für diesen Satz: base, nach oben oder unten verschoben."""
        features = measure(audio, characters)
        if features is None:
            return base
        values = [features.loudness_db, features.pitch_spread, features.rate]
        if self._mean is None:
            self._mean = values
            self._seen = 1
            return base    # der erste Satz ist der Massstab

        score = 0.0
        for weight, value, mean, spread in zip(_WEIGHTS, values, self._mean, self._spread):
            score += weight * float(np.clip((value - mean) / spread, -2.0, 2.0))

        # "Üblich" nachführen; die Streuung erst, wenn ein paar Sätze da sind.
        for i, value in enumerate(values):
            deviation = abs(value - self._mean[i])
            self._mean[i] += _LEARN * (value - self._mean[i])
            if self._seen >= 3:
                floor = _DEFAULT_SPREAD[i] * 0.5
                self._spread[i] = max(floor, self._spread[i] + _LEARN * (1.25 * deviation - self._spread[i]))
        self._spread[2] = max(self._spread[2], 0.1 * self._mean[2])   # Tempo relativ zum üblichen
        self._seen += 1
        return float(np.clip(base + _STRENGTH * score, MIN_EXPRESSION, MAX_EXPRESSION))
