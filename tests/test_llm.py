import pytest
import requests

from rag_papers import llm


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload

    def json(self):
        return self._payload


def test_groq_returns_content(monkeypatch):
    payload = {"choices": [{"message": {"content": " hi "}}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, payload))
    assert llm.GroqLLM("key").complete("s", "u") == "hi"


@pytest.mark.parametrize("status, message", [(429, "quota"), (500, "HTTP 500")])
def test_groq_errors_never_leak_the_key(monkeypatch, status, message):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status, {}))
    with pytest.raises(llm.LLMError) as err:
        llm.GroqLLM("secret-key").complete("s", "u")
    assert message in str(err.value)
    assert "secret-key" not in str(err.value)


def test_from_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.from_env() is None
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    assert llm.from_env().model == "some-model"
