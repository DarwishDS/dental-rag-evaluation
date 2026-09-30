from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import Pipeline

STATIC = Path(__file__).parent / "static"


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=3, ge=1, le=10)


def create_app(settings: Settings | None = None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = Pipeline(settings)
        try:
            yield
        finally:
            app.state.pipeline.close()

    app = FastAPI(title="Dental RAG Evaluation", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "backend": settings.backend,
            "generator": settings.generator,
            "chunks": len(app.state.pipeline.chunks),
            "corpus_sha256": app.state.pipeline.corpus_sha256,
        }

    @app.get("/api/sources")
    def sources():
        return [
            {"id": c.id, "title": c.title, "section": c.section, "url": c.url}
            for c in app.state.pipeline.chunks
        ]

    @app.post("/api/query")
    def query(body: Query):
        try:
            return app.state.pipeline.query(body.question, body.strategy, body.top_k)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            # Provider errors may contain credentials or request details; keep them off the UI.
            raise HTTPException(
                503, "Inference failed. Check local configuration and model access."
            ) from exc

    return app
