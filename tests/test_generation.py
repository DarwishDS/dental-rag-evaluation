import sys
from types import SimpleNamespace

import pytest

from dental_rag.generation import generate


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

    def create(**_):
        return response

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: fake_client))
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
