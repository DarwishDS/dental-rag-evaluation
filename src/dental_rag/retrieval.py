import hashlib
import re
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import Settings
from .data import Chunk

STOP = set(
    "a an the is are was be to of and or in on for with how what why can do does i my me it its have has should which from that this when".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def reciprocal_rank_fusion(
    rankings: list[list[int]], constant: int = 60
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))


class SmokeEmbedder:
    """Deterministic lexical hashing for tests; NEVER report this as semantic retrieval."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), 512), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokens(text):
                idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % 512
                vectors[i, idx] += 1
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        self.chunks, self.settings = chunks, settings
        self.bm25 = BM25Okapi([tokens(c.search_text) for c in chunks])
        self.reranker = None
        if settings.backend == "semantic":
            from fastembed import TextEmbedding

            self.embedder = TextEmbedding(
                settings.embedding_model, cache_dir=str(settings.cache_dir), threads=2
            )
        else:
            self.embedder = SmokeEmbedder()
        vectors = self.encode([c.search_text for c in chunks], query=False)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "dental",
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        self.client.upsert(
            "dental",
            points=[models.PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)],
        )

    def encode(self, texts, query=False):
        if self.settings.backend == "smoke":
            return self.embedder.encode(texts)
        method = self.embedder.query_embed if query else self.embedder.passage_embed
        return np.asarray(list(method(texts)), dtype=np.float32)

    def warmup(self, strategy):
        if strategy == "hybrid_rerank" and self.settings.backend == "semantic":
            if self.reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self.reranker = TextCrossEncoder(
                    self.settings.reranker_model, cache_dir=str(self.settings.cache_dir), threads=2
                )
            list(self.reranker.rerank("warmup", [self.chunks[0].search_text]))
        self.search("How does fluoride protect teeth?", strategy)

    def search(self, question: str, strategy: str = "hybrid_rerank", top_k: int | None = None):
        if strategy not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unknown retrieval strategy")
        k = top_k or self.settings.top_k
        candidate_k = min(len(self.chunks), max(k, self.settings.candidate_k))
        vector = self.encode([question], query=True)[0]
        points = self.client.query_points(
            "dental", query=vector.tolist(), limit=len(self.chunks)
        ).points
        dense = {int(p.id): float(p.score) for p in points}
        dense_order = [int(p.id) for p in points[:candidate_k]]
        lexical = self.bm25.get_scores(tokens(question))
        lexical_order = [int(i) for i in np.argsort(-lexical, kind="stable") if lexical[i] > 0][
            :candidate_k
        ]
        if strategy == "dense":
            ranked = [(i, dense[i]) for i in dense_order]
        elif strategy == "bm25":
            ranked = [(i, float(lexical[i])) for i in lexical_order]
        else:
            ranked = reciprocal_rank_fusion(
                [dense_order, lexical_order], self.settings.rrf_constant
            )[:candidate_k]
        if strategy == "hybrid_rerank" and ranked:
            if self.settings.backend == "semantic":
                if self.reranker is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self.reranker = TextCrossEncoder(
                        self.settings.reranker_model,
                        cache_dir=str(self.settings.cache_dir),
                        threads=2,
                    )
                scores = list(
                    self.reranker.rerank(question, [self.chunks[i].search_text for i, _ in ranked])
                )
            else:
                # Smoke approximation is lexical overlap, explicitly labelled in all reports.
                q = set(tokens(question))
                scores = [
                    len(q & set(tokens(self.chunks[i].text))) / max(len(q), 1) for i, _ in ranked
                ]
            ranked = sorted(
                [(i, float(s)) for (i, _), s in zip(ranked, scores)],
                key=lambda row: (-row[1], row[0]),
            )
        return [Hit(self.chunks[i], score, dense[i]) for i, score in ranked[:k]]

    def close(self):
        self.client.close()
