"""Übersetzung mit TranslateGemma über Ollama."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ..config import TranslateConfig
from ..languages import Language

# Vom Modell erwartetes Format; die zwei Leerzeilen vor dem Text gehören dazu.
_PROMPT = (
    "You are a professional {src} ({src_code}) to {tgt} ({tgt_code}) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original {src} text "
    "while adhering to {tgt} grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the {tgt} translation, without any additional explanations or commentary. "
    "{style}Please translate the following {src} text into {tgt}:\n\n\n{text}"
)

# Zusatzanweisungen zum Stil. Das Modell ist auf das feste Format oben
# trainiert und folgt Zusätzen nur behutsam: Es glättet Füllwörter und
# Versprecher, schreibt aber kaum etwas dazu.
STYLES: dict[str, tuple[str, str]] = {
    "literal": ("Wörtlich", ""),
    "clear": (
        "Deutlicher",
        "Fix slips of the tongue and incomplete phrasing so that the result is one clear, "
        "complete, natural sentence. ",
    ),
    "polite": (
        "Ausführlicher",
        "Phrase the result as a slightly longer, complete and polite spoken sentence, "
        "without adding new facts. ",
    ),
}


class TranslationError(RuntimeError):
    pass


class Translator:
    def __init__(self, config: TranslateConfig):
        self._cfg = config
        self._client = httpx.AsyncClient(
            base_url=config.ollama_url, timeout=httpx.Timeout(60.0, connect=3.0)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def check(self) -> str | None:
        """None, wenn Ollama läuft und das Modell vorhanden ist; sonst der Grund."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return f"Ollama ist unter {self._cfg.ollama_url} nicht erreichbar."
        names = {m.get("name", "") for m in response.json().get("models", [])}
        wanted = self._cfg.model
        if wanted not in names and f"{wanted}:latest" not in names:
            return f"Das Modell {wanted} fehlt in Ollama (ollama pull {wanted})."
        return None

    async def stream(self, text: str, source: Language, target: Language,
                     style: str = "literal") -> AsyncIterator[str]:
        """Liefert die Übersetzung stückweise, so wie das Modell sie erzeugt."""
        prompt = _PROMPT.format(
            src=source.name, src_code=source.code, tgt=target.name, tgt_code=target.code, text=text,
            style=STYLES.get(style, STYLES["literal"])[1],
        )
        payload = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "keep_alive": self._cfg.keep_alive,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 512},
        }
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise TranslationError(f"Ollama meldet {response.status_code}: {body[:200]}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise TranslationError(str(data["error"]))
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise TranslationError(f"Ollama nicht erreichbar: {exc}") from exc

    async def translate(self, text: str, source: Language, target: Language) -> str:
        return "".join([piece async for piece in self.stream(text, source, target)]).strip()
