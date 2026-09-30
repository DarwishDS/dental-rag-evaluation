# Architecture

```mermaid
flowchart LR
    C[Versioned source passages] --> E[FastEmbed BGE embeddings]
    E --> Q[Qdrant local cosine search]
    C --> B[BM25 index]
    U[Question] --> Q
    U --> B
    Q --> R[Reciprocal rank fusion]
    B --> R
    R --> X[ONNX cross-encoder shortlist reranking]
    X --> G[Evidence gate and generator]
    G --> A[Cited answer or abstention]
    G --> T[JSONL trace / optional Langfuse]
    D[Versioned dev/test questions] --> V[Evaluation runner]
    A --> V
    V --> M[Per-case results, aggregate metrics, paired intervals]
```

The CLI, FastAPI server, and experiment runner share one `Pipeline`. There is no second
implementation hidden in the web UI. The vector collection is rebuilt at startup;
model files are cached locally. Qdrant local mode removes the need for a separate database
for this tiny corpus. It is not a distributed serving configuration. The pipeline serializes
queries to protect inference resources and local trace writes.

The API returns passages and strategy-specific scores so a reviewer can inspect failures.
The browser renders text through `textContent` and only links to the corpus source host.
Cloud services are optional and must be configured explicitly. Secrets are read from
environment variables, never bundled into the frontend or recorded in the report.
