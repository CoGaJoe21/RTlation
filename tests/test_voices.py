"""Speichern und Prüfen von Stimmaufnahmen (rtlation.voices)."""

import io
import wave

import numpy as np
import pytest

from rtlation.config import Config
from rtlation.voices import check_name, delete_voice, list_voices, move_project_voices, save_voice, voices_dir

RATE = 24000


def wav_bytes(audio: np.ndarray, rate: int = RATE, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def tone(seconds: float, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


@pytest.fixture
def config(tmp_path):
    return Config(root=tmp_path, voices_dir="eigene")


def test_silence_is_trimmed_and_level_raised(config):
    recording = np.concatenate([silence(2), tone(6), silence(3)])
    path, seconds = save_voice(config, "Meine Stimme", wav_bytes(recording))
    assert path == config.root / "eigene" / "Meine Stimme.wav"
    assert 6.0 <= seconds <= 6.5          # 6 s Ton plus etwas Rand
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels()) == (RATE, 1)
        saved = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert abs(np.abs(saved).max() / 32767 - 0.9) < 0.01
    assert list_voices(config) == [{"name": "Meine Stimme", "seconds": round(seconds, 1)}]


def test_stereo_is_mixed_down(config):
    stereo = np.repeat(tone(5), 2)        # beide Kanäle gleich
    _, seconds = save_voice(config, "stereo", wav_bytes(stereo, channels=2))
    assert 4.9 <= seconds <= 5.4


def test_too_little_speech_is_rejected(config):
    with pytest.raises(ValueError, match="mindestens"):
        save_voice(config, "kurz", wav_bytes(np.concatenate([silence(4), tone(1), silence(4)])))
    assert list_voices(config) == []


def test_long_recordings_are_cut(config):
    _, seconds = save_voice(config, "lang", wav_bytes(tone(45)))
    assert seconds == pytest.approx(30.0, abs=0.01)


def test_garbage_is_rejected(config):
    with pytest.raises(ValueError, match="WAV"):
        save_voice(config, "kaputt", b"kein wav")


@pytest.mark.parametrize("name", ["", "default", "../oben", "a/b", "a\\b", "x" * 41, "punkt.wav", "NUL"])
def test_unsafe_names_are_rejected(name):
    with pytest.raises(ValueError):
        check_name(name)


def test_delete_removes_recording_and_transcript(config):
    path, _ = save_voice(config, "weg", wav_bytes(tone(5)))
    path.with_suffix(".txt").write_text("Abschrift", encoding="utf-8")
    delete_voice(config, "weg")
    assert not path.exists() and not path.with_suffix(".txt").exists()
    with pytest.raises(ValueError):
        delete_voice(config, "weg")


def test_default_is_never_listed(config):
    save_voice(config, "meine", wav_bytes(tone(5)))
    (config.root / "eigene" / "default.wav").write_bytes(wav_bytes(tone(5)))
    assert [v["name"] for v in list_voices(config)] == ["meine"]


def test_voices_left_in_the_project_are_moved_out(config):
    project = config.root / "voices"
    project.mkdir()
    for name in ("default", "Alt"):
        (project / f"{name}.wav").write_bytes(wav_bytes(tone(5)))
    (project / "Alt.txt").write_text("Abschrift", encoding="utf-8")

    assert move_project_voices(config) == ["Alt"]
    assert sorted(p.name for p in project.iterdir()) == ["default.wav"]
    assert sorted(p.name for p in (config.root / "eigene").iterdir()) == ["Alt.txt", "Alt.wav"]


def test_without_setting_voices_live_in_the_user_profile(tmp_path):
    assert tmp_path not in voices_dir(Config(root=tmp_path)).parents
