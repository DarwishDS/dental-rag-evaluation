import json
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone

from .config import Settings
from .data import fingerprint, load_chunks
from .generation import NOTICE, generate
from .retrieval import Retriever


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        path = self.settings.data_dir / "corpus.jsonl"
        self.chunks = load_chunks(path)
        self.corpus_sha256 = fingerprint(path)
        self.retriever = Retriever(self.chunks, self.settings)
        self.lock = threading.Lock()
        self.langfuse = None
        if self.settings.langfuse_enabled:
            from langfuse import get_client

            self.langfuse = get_client()

    def query(self, question: str, strategy: str = "hybrid_rerank", top_k=None):
        question = question.strip()
        if not 3 <= len(question) <= 2000:
            raise ValueError("Question must contain 3 to 2000 characters")
        # Serialize inference and trace writes to avoid racing local ONNX/client resources.
        with self.lock:
            return self._query(question, strategy, top_k)

    def _query(self, question, strategy, top_k):
        trace_id = uuid.uuid4().hex
        context = (
            self.langfuse.start_as_current_observation(
                name="dental-rag", input={"question": question, "strategy": strategy}
            )
            if self.langfuse
            else nullcontext()
        )
        with context as span:
            started = time.perf_counter()
            hits = self.retriever.search(question, strategy, top_k)
            retrieval_ms = (time.perf_counter() - started) * 1000
            answer = generate(question, hits, self.settings)
            result = {
                "trace_id": trace_id,
                "question": question,
                "strategy": strategy,
                "backend": self.settings.backend,
                **asdict(answer),
                "notice": NOTICE,
                "hits": [
                    {**asdict(h.chunk), "score": h.score, "dense_score": h.dense_score}
                    for h in hits
                ],
                "retrieval_ms": retrieval_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
                "corpus_sha256": self.corpus_sha256,
            }
            self.settings.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **result})
                    + "\n"
                )
            if span:
                span.update(output=result, metadata={"local_trace_id": trace_id})
            return result

    def close(self):
        self.retriever.close()
        if self.langfuse:
            self.langfuse.flush()
