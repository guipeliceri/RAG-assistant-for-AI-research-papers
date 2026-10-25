"""LLM providers behind one small interface, so the provider can be swapped."""

import os
from typing import Protocol

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"


class LLMError(RuntimeError):
    """The provider failed: network, quota, or a malformed answer."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class GroqLLM:
    """Groq's OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
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
            raise LLMError(f"request failed: {type(exc).__name__}") from None
        if response.status_code == 429:
            raise LLMError("quota exceeded (HTTP 429)")
        if not response.ok:
            # Never include the request (it carries the key) in the error.
            raise LLMError(f"HTTP {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError):
            raise LLMError("malformed response") from None


def from_env() -> LLM | None:
    """The job's own provider, configured from environment variables."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return GroqLLM(key, model=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
