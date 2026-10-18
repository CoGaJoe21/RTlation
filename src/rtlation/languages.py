"""Unterstützte Sprachen.

Die Liste ist bewusst auf Sprachen beschränkt, die sowohl das
Übersetzungsmodell (TranslateGemma) als auch das später geplante
Sprachausgabe-Modell (Chatterbox Multilingual) beherrschen.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str      # Code für das Übersetzungsmodell
    name: str      # englischer Name für den Übersetzungs-Prompt
    label: str     # Anzeige in der Oberfläche
    whisper: str   # Code für die Spracherkennung


_ALL = [
    Language("de", "German", "Deutsch", "de"),
    Language("en", "English", "Englisch", "en"),
    Language("es", "Spanish", "Spanisch", "es"),
    Language("zh-Hans", "Chinese", "Chinesisch (vereinfacht)", "zh"),
    Language("zh-Hant", "Chinese", "Chinesisch (traditionell)", "zh"),
    Language("fr", "French", "Französisch", "fr"),
    Language("it", "Italian", "Italienisch", "it"),
    Language("pt", "Portuguese", "Portugiesisch", "pt"),
    Language("nl", "Dutch", "Niederländisch", "nl"),
    Language("pl", "Polish", "Polnisch", "pl"),
    Language("ru", "Russian", "Russisch", "ru"),
    Language("ja", "Japanese", "Japanisch", "ja"),
    Language("ko", "Korean", "Koreanisch", "ko"),
    Language("tr", "Turkish", "Türkisch", "tr"),
    Language("ar", "Arabic", "Arabisch", "ar"),
    Language("hi", "Hindi", "Hindi", "hi"),
    Language("sv", "Swedish", "Schwedisch", "sv"),
    Language("da", "Danish", "Dänisch", "da"),
    Language("no", "Norwegian", "Norwegisch", "no"),
    Language("fi", "Finnish", "Finnisch", "fi"),
    Language("el", "Greek", "Griechisch", "el"),
    Language("he", "Hebrew", "Hebräisch", "he"),
    Language("ms", "Malay", "Malaiisch", "ms"),
    Language("sw", "Swahili", "Suaheli", "sw"),
]

LANGUAGES: dict[str, Language] = {lang.code: lang for lang in _ALL}
AUTO = "auto"


def by_whisper_code(code: str) -> Language | None:
    """Sprache zu einem von Whisper erkannten Code (erste passende)."""
    return next((lang for lang in _ALL if lang.whisper == code), None)
