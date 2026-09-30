from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DENTAL_", env_file=".env", extra="ignore")
    backend: Literal["semantic", "smoke"] = "semantic"
    generator: Literal["extractive", "openai"] = "extractive"
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    trace_path: Path = Path("artifacts/traces.jsonl")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    openai_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    top_k: int = 3
    candidate_k: int = 12
    rrf_constant: int = 60
    # Fixed before test evaluation. Calibrate on dev only for a larger corpus.
    min_dense_score: float = 0.55
    min_smoke_score: float = 0.15
    langfuse_enabled: bool = False
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
