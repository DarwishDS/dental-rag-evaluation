# Evaluation methodology

This is a small engineering benchmark, not a clinical validation study.

## Corpus and labels

The seed corpus contains 24 original, short summaries derived from six public NIDCR
educational pages. `data/sources.json` records source URLs and the date they were checked.
No patient records, copyrighted images, journal articles, or government logos are included.
The corpus is deliberately versioned: changing a source or passage changes its SHA-256.
The summaries are not exhaustive and must not be used as treatment guidelines.

There are 12 development questions and 30 test questions (24 answerable and 6 unsupported).
Each answerable question has an explicit relevant passage ID and short reference answer.
Questions are manually authored alongside the seed summaries by the project builder;
they have **not** been independently labelled or reviewed by a dental professional.
Questions are disjoint across splits, but topics and supporting passages overlap. This is a
question-held-out split, not a source-held-out or blinded external benchmark. The labels
identify a preferred passage; other passages may also contain partially relevant evidence.
This can penalize semantically reasonable results and is a limitation of the current gold set.

The ingestion path reads only `corpus.jsonl`. Neither reference answers nor evaluation
questions are indexed. Do not tune on test results. After inspecting failures, create a
new version and reserve a fresh test set before making a new improvement claim.

## Retrieval experiment

All strategies share the same corpus, embeddings, query, top-k (3), candidate budget (12),
and generator. BGE-small-en-v1.5 provides real dense embeddings through FastEmbed/ONNX.
An in-memory Qdrant collection runs exact cosine search. BM25 runs over normalized tokens.
Hybrid search combines dense and BM25 rankings with reciprocal rank fusion (constant 60).
The cross-encoder scores the fused shortlist using ms-marco-MiniLM-L-6-v2.

`smoke` uses lexical feature hashing and an overlap reranker solely for no-network tests.
It is not a semantic experiment and must never be cited as a model-quality result.

The runner warms every strategy before collecting per-request latency. Startup, ingestion,
model download, and model loading are excluded. One run is an illustrative measurement,
not a rigorous serving benchmark: repeat on a fixed machine for latency claims. Strategies
are executed in a fixed order and may share caches. No concurrency or load testing is implied.

## Metrics and denominators

- Hit@1: top passage is labelled relevant.
- Recall@k: unique relevant passages retrieved / all labelled relevant passages.
- MRR@k: inverse rank of the first relevant result, or zero.
- nDCG@k: binary relevance, logarithmic rank discount, normalized to the ideal ranking.
- Retrieval metrics include only the 24 answerable test questions. With a single gold
  passage per question, recall@k equals hit@k.
- Abstention correctness: unsupported question refused or supported question answered.
  All selected questions enter this denominator. A keyword guard contributes to this
  metric; it is not a demonstration of learned uncertainty or medical safety.
- Citation validity: cited IDs exist in the retrieved context; only outputs with citations
  enter the denominator. This is **not** entailment or factual correctness.
- Evidence hit in citations: at least one cited passage matches a gold ID; answerable cases only.
- DeepEval faithfulness and answer relevancy: optional LLM-judge measurements on generated,
  non-abstained answers to answerable questions, with scored counts reported separately.
  These metrics are null when not run. Abstentions and unsupported cases are excluded;
  compare coverage before interpreting mean judge scores.
- Cost: unknown unless explicit input/output prices are supplied; token usage is recorded
  for OpenAI generation. Estimated generation cost excludes judge calls, tracing, storage,
  electricity, and model download. Extractive generation makes no paid model requests.

Paired bootstrap intervals resample question-level strategy deltas 2,000 times with seed 42.
The small, manually authored set and correlated topics limit statistical generalization.
An interval spanning zero is insufficient evidence for an improvement claim. Even an
interval excluding zero is a finding about this seed set, not clinical effectiveness.

## Generation and tracing

The default generator selects one of the retrieved passages verbatim. It produces a
cited educational excerpt, not a conversational LLM answer. It also checks a dense-score
threshold and a short keyword refusal guard. Thresholds are fixed in config before the
initial test run and are not calibrated clinical confidence scores. The guard is incomplete
and may reject legitimate educational questions. It does not establish medical safety.

The optional Groq, Gemini, and OpenAI generators use retrieved evidence and return structured JSON.
Unknown or absent citation IDs cause abstention. Valid IDs do not guarantee that every
generated claim is supported; evaluate and review the output.

Groq can fall back to Gemini on a provider failure or invalid response when a real Gemini
key is configured. A valid abstention does not trigger fallback. Responses record the actual
provider, model, and whether fallback was used. Placeholders are never sent as credentials.
The initial checked-in benchmark remains an extractive run; live generation is a separate
validation and does not supply faithfulness scores for the original report.

Every successful query writes a local JSONL trace including question, answer, passages,
latency, backend, and corpus hash. Local traces may contain sensitive questions and are
ignored by Git. Do not submit patient-identifying information. Langfuse is explicitly
opt-in and transmits the same data to the configured instance. This local demo has no
authentication or rate limiting and should not be exposed publicly as a clinical service.

## Extending the corpus responsibly

Add attributed, permission-compatible summaries as JSONL records with unique stable IDs,
source IDs, title, section, URL, and text. Keep passages below the model token limit and
split long material into self-contained sections. Preserve provenance and record review
dates. Use `dental-rag validate`, reserve new test questions, and benchmark each corpus version.
For a stronger resume study, use hundreds of independently reviewed passages, multiple
gold passages when appropriate, realistic multi-source questions, and external test labels.
