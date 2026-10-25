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


def test_repr_hides_key():
    assert "secret-key" not in repr(llm.GroqLLM("secret-key"))


@pytest.mark.parametrize("provider", list(llm.PROVIDERS))
def test_every_provider_parses_openai_format(monkeypatch, provider):
    payload = {"choices": [{"message": {"content": "ok"}}]}
    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        return FakeResponse(200, payload)

    monkeypatch.setattr(requests, "post", fake_post)
    client = llm.OpenAICompatibleLLM(llm.PROVIDERS[provider], "k")
    assert client.complete("s", "u") == "ok"
    assert seen["url"] == llm.PROVIDERS[provider].url
