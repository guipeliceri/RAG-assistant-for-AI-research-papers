"""LLM providers behind one small interface, so the provider can be swapped.

Groq, OpenRouter and Gemini all expose an OpenAI-compatible chat endpoint, so
one client covers the three.
"""

import os
from dataclasses import dataclass
from typing import Protocol

import requests


class LLMError(RuntimeError):
    """The provider failed: network, quota, or a malformed answer."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    default_model: str
    key_url: str


# Default models are free-tier at the time of writing; free model lists change
# often, so the app lets the user override them.
PROVIDERS = {
    "groq": Provider(
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        "openai/gpt-oss-20b",
        "https://console.groq.com/keys",
    ),
    "openrouter": Provider(
        "OpenRouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "openai/gpt-oss-20b:free",
        "https://openrouter.ai/settings/keys",
    ),
    "gemini": Provider(
        "Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "gemini-2.5-flash",
        "https://aistudio.google.com/apikey",
    ),
}
DEFAULT_MODEL = PROVIDERS["groq"].default_model


class OpenAICompatibleLLM:
    def __init__(self, provider: Provider, api_key: str, model: str | None = None, timeout: float = 60):
        self.provider = provider
        # Pasted keys often carry a stray space or newline, which the provider rejects with 401.
        self._api_key = api_key.strip()
        self.model = model or provider.default_model
        self.timeout = timeout

    def __repr__(self) -> str:
        # Keep the key out of tracebacks and debug output.
        return f"{type(self).__name__}(provider={self.provider.name!r}, model={self.model!r})"

    def complete(self, system: str, user: str) -> str:
        try:
            response = requests.post(
                self.provider.url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"{self.provider.name} request failed: {type(exc).__name__}") from None
        if response.status_code == 429:
            raise LLMError(f"{self.provider.name} quota exceeded (HTTP 429)")
        if response.status_code in (401, 403):
            raise LLMError(f"{self.provider.name} rejected the API key (HTTP {response.status_code})")
        if not response.ok:
            # Never include the request (it carries the key) in the error.
            raise LLMError(f"{self.provider.name} HTTP {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError, AttributeError):
            raise LLMError(f"{self.provider.name} returned a malformed response") from None


class GroqLLM(OpenAICompatibleLLM):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 60):
        super().__init__(PROVIDERS["groq"], api_key, model, timeout)


def from_env() -> LLM | None:
    """The job's own provider, configured from environment variables."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return GroqLLM(key, model=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
