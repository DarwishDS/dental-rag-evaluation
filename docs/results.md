# Initial semantic experiment — 30 September 2026

This report is based on a real local ONNX inference run, with the extractive generator.
There are 24 answerable and six unsupported test questions. No API keys or paid model
requests were used. See `reports/semantic-test/` for the complete records and configuration.

| Strategy | Hit@1 | Recall@3 | MRR@3 | Median latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Dense | 87.5% | 95.8% | 0.910 | 14.3 ms | 20.8 ms |
| BM25 | 83.3% | 87.5% | 0.854 | 13.5 ms | 17.2 ms |
| Hybrid RRF | 87.5% | 91.7% | 0.889 | 12.8 ms | 17.8 ms |
| Hybrid + cross-encoder | 95.8% | 95.8% | 0.958 | 237.5 ms | 252.4 ms |

Reranking places the labelled passage first on 23/24 questions, versus 21/24 for dense.
The observed Hit@1 delta is **+8.3 percentage points**. Its paired bootstrap 95% interval is
**[0.0, +20.8] percentage points** and includes zero. It is a promising seed-set observation,
not statistically established improvement. Recall@3 is unchanged, so this run shows a
ranking difference rather than improved candidate coverage. Hybrid fusion alone did not
improve Hit@1 and reduced recall@3 on this set.

The reranked pipeline has roughly 17 times the median latency of dense in this run.
These are sequential localhost measurements after warmup, with no concurrency or
repeat-run performance study. BM25 also computes the dense embedding because the shared
pipeline uses it for evidence gating; its timing is not that of a standalone BM25 service.

All strategies refuse all six unsupported questions. One supported question is refused
by the evidence threshold, giving abstention correctness of 29/30. The unsupported cases
are covered by explicit keyword rules; they are not a robustness benchmark. Citation ID
validity is 100% for cited answers, which establishes only that the IDs exist in context.
It does not establish factual support or clinical usefulness.

## Failures worth discussing in an interview

The per-case report lists exact gold and retrieved IDs. Some failures arise from competing
passages that share vocabulary or describe related conditions. Exact-ID gold labels can
mark an otherwise useful passage as wrong. The dense threshold also causes a supported
question to be refused. Neither issue is hidden by an aggregate score.

The current corpus and labels were created together. The test split holds out question
wordings, not sources, and the topics overlap with dev. A stronger experiment needs more
realistic questions and independent relevance judgments. The current numbers should not
be used to claim general medical knowledge, clinical reliability, or LLM faithfulness.

## What remains unmeasured

DeepEval faithfulness and answer relevancy are null because this run did not generate or
judge LLM answers. Live OpenAI generation, hosted Langfuse tracing, Docker execution, and
GitHub Actions are separate validations. They must not be inferred from the local benchmark.
Optional generation output validation is covered by mock-based tests, which do not test a
live provider. Local JSONL tracing and the FastAPI endpoints are exercised by tests.

## Next experiment

1. Expand the attributed corpus to at least 100–200 passages, preserving source dates and hashes.
2. Have a dental professional or independent annotator review the summaries and label
   multiple acceptable supporting passages where needed.
3. Add multi-source, unfamiliar-topic, and adversarial unsupported questions; reserve a
   fresh test set before changing retrieval parameters.
4. Calibrate evidence thresholds and candidate counts on development questions only.
5. Run optional LLM generation and DeepEval judges with fixed model IDs, recording
   response coverage, claim support, token usage, and failures alongside aggregate scores.
6. Repeat latency measurements and compare the quality benefit against reranking cost.

An honest quantitative resume description is: “Observed Hit@1 of 87.5% for dense retrieval
and 95.8% for hybrid + reranking on a 24-question source-backed seed benchmark; documented
latency tradeoffs and paired confidence intervals.” Add broader improvement claims only
after independent evaluation supports them.
