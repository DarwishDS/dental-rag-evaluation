import sys
from types import SimpleNamespace

import pytest

from dental_rag.generation import LLMGenerationError, generate, request_completion


@pytest.mark.parametrize(
    "payload,abstained",
    [
        ('{"answer":"Supported answer", "cited_ids":["gum-causes"], "abstained":false}', False),
        ('{"answer":"Unsupported", "cited_ids":["invented"], "abstained":false}', True),
        ('{"answer":"Unsupported", "cited_ids":[], "abstained":false}', True),
        ('{"answer":"Do not know", "cited_ids":[], "abstained":true}', True),
        ('{"answer":"", "cited_ids":["gum-causes"], "abstained":false}', True),
        ('{"answer":"Text", "cited_ids":["gum-causes"], "abstained":"false"}', True),
        ("not json", True),
    ],
)
def test_cloud_output_validation(pipeline, monkeypatch, payload, abstained):
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
    )

    monkeypatch.setattr(
        "dental_rag.generation.request_completion", lambda *args: (response, "test-model")
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = pipeline.settings.model_copy(update={"generator": "openai", "min_smoke_score": 0})
    hits = pipeline.retriever.search("Who removes tartar?", "dense")
    answer = generate("Who removes tartar?", hits, settings)
    assert answer.abstained == abstained
    assert answer.usage["input_tokens"] == 12
    if abstained:
        assert answer.cited_ids == []


def test_extract_uses_top_ranked_passage(pipeline):
    hits = pipeline.retriever.search("How does saliva protect teeth?", "hybrid_rerank")
    settings = pipeline.settings.model_copy(update={"min_smoke_score": 0})
    answer = generate("How does saliva protect teeth?", hits, settings)
    assert answer.answer == hits[0].chunk.text
    assert answer.cited_ids == [hits[0].chunk.id]


def completion(payload=None):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=payload
                    or '{"answer":"A dental professional removes tartar.",'
                    '"cited_ids":["gum-causes"],"abstained":false}'
                )
            )
        ],
    )


@pytest.mark.parametrize(
    "provider,key_name,url,model",
    [
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
        (
            "gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-2.5-flash",
        ),
        ("openai", "OPENAI_API_KEY", None, "gpt-4.1-mini"),
    ],
)
def test_provider_endpoint_and_model(settings, monkeypatch, provider, key_name, url, model):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured["request"] = kwargs
            return completion()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    monkeypatch.setenv(key_name, "unit-test-key")
    _, actual_model = request_completion(provider, "Who removes tartar?", [], settings)
    assert actual_model == model
    assert captured.get("base_url") == url
    assert captured["api_key"] == "unit-test-key"
    assert captured["request"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("failure", ["provider_error", "invalid_response"])
def test_groq_gemini_fallback(pipeline, monkeypatch, failure):
    providers = []

    def request(provider, *_):
        providers.append(provider)
        if provider == "groq":
            if failure == "provider_error":
                raise RuntimeError("sensitive provider exception")
            return completion("not json"), "groq-model"
        return completion(), "gemini-model"

    monkeypatch.setattr("dental_rag.generation.request_completion", request)
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    settings = pipeline.settings.model_copy(update={"generator": "groq", "min_smoke_score": 0})
    hits = pipeline.retriever.search("Who removes tartar?", "dense")
    answer = generate("Who removes tartar?", hits, settings)
    assert providers == ["groq", "gemini"]
    assert answer.generator == "gemini"
    assert answer.model == "gemini-model"
    assert answer.fallback_used
    assert answer.fallback_reason == "primary_" + failure
    assert "sensitive" not in str(answer)


def test_placeholder_skips_fallback_and_provider_errors_are_sanitized(pipeline, monkeypatch):
    providers = []

    def request(provider, *_):
        providers.append(provider)
        raise RuntimeError("private key in provider error")

    monkeypatch.setattr("dental_rag.generation.request_completion", request)
    monkeypatch.setenv("GEMINI_API_KEY", "replace_with_your_gemini_api_key")
    settings = pipeline.settings.model_copy(update={"generator": "groq", "min_smoke_score": 0})
    hits = pipeline.retriever.search("Who removes tartar?", "dense")
    with pytest.raises(LLMGenerationError) as caught:
        generate("Who removes tartar?", hits, settings)
    assert "private key" not in str(caught.value)
    assert providers == ["groq"]


def test_valid_abstention_never_triggers_fallback(pipeline, monkeypatch):
    providers = []

    def request(provider, *_):
        providers.append(provider)
        return completion('{"answer":"No evidence", "cited_ids":[], "abstained":true}'), "test"

    monkeypatch.setattr("dental_rag.generation.request_completion", request)
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    settings = pipeline.settings.model_copy(update={"generator": "groq", "min_smoke_score": 0})
    hits = pipeline.retriever.search("Who removes tartar?", "dense")
    answer = generate("Who removes tartar?", hits, settings)
    assert answer.abstained
    assert providers == ["groq"]


def test_fallback_can_be_disabled(pipeline, monkeypatch):
    providers = []

    def request(provider, *_):
        providers.append(provider)
        raise RuntimeError("provider failed")

    monkeypatch.setattr("dental_rag.generation.request_completion", request)
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    settings = pipeline.settings.model_copy(
        update={"generator": "groq", "min_smoke_score": 0, "gemini_fallback_enabled": False}
    )
    hits = pipeline.retriever.search("Who removes tartar?", "dense")
    with pytest.raises(LLMGenerationError):
        generate("Who removes tartar?", hits, settings)
    assert providers == ["groq"]
