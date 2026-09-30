from pathlib import Path

import pytest

from dental_rag.config import Settings
from dental_rag.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        backend="smoke",
        generator="extractive",
        data_dir=ROOT / "data",
        cache_dir=tmp_path / "cache",
        trace_path=tmp_path / "traces.jsonl",
        langfuse_enabled=False,
    )


@pytest.fixture
def pipeline(settings):
    instance = Pipeline(settings)
    yield instance
    instance.close()
