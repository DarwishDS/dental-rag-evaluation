# Evaluation: test

Backend: **semantic**; generator: **extractive**; cases: 30.

| Strategy | Hit@1 | Recall@3 | MRR@3 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| dense | 0.875 | 0.958 | 0.910 | 14.3 | 20.8 |
| bm25 | 0.833 | 0.875 | 0.854 | 13.5 | 17.2 |
| hybrid | 0.875 | 0.917 | 0.889 | 12.8 | 17.8 |
| hybrid_rerank | 0.958 | 0.958 | 0.958 | 237.5 | 252.4 |

Retrieval metrics exclude unanswerable cases. Latency excludes model loading and warmup. CI is paired bootstrap over questions, not over passages.

Faithfulness and answer relevancy are null unless DeepEval judges an OpenAI run. Citation validity checks IDs only; it does not establish claim-level support.

## Failures

- `test-decay-symptoms` / `dense`: top hit `decay-treatment`; gold ['decay-symptoms']; abstained=False.
- `test-decay-symptoms` / `bm25`: top hit `hygiene-flossing`; gold ['decay-symptoms']; abstained=False.
- `test-decay-symptoms` / `hybrid`: top hit `decay-treatment`; gold ['decay-symptoms']; abstained=False.
- `test-decay-symptoms` / `hybrid_rerank`: top hit `decay-early`; gold ['decay-symptoms']; abstained=False.
- `test-decay-treatment` / `bm25`: top hit `hygiene-routine`; gold ['decay-treatment']; abstained=False.
- `test-decay-treatment` / `hybrid`: top hit `hygiene-flossing`; gold ['decay-treatment']; abstained=False.
- `test-gum-causes` / `dense`: top hit `decay-treatment`; gold ['gum-causes']; abstained=False.
- `test-gum-causes` / `bm25`: top hit `decay-mechanism`; gold ['gum-causes']; abstained=False.
- `test-hygiene-brushing` / `dense`: top hit `hygiene-flossing`; gold ['hygiene-brushing']; abstained=False.
- `test-hygiene-brushing` / `bm25`: top hit `cancer-exam`; gold ['hygiene-brushing']; abstained=False.
- `test-hygiene-brushing` / `hybrid`: top hit `hygiene-routine`; gold ['hygiene-brushing']; abstained=False.
- `test-diabetes-treatment` / `dense`: top hit `diabetes-treatment`; gold ['diabetes-treatment']; abstained=True.
- `test-diabetes-treatment` / `bm25`: top hit `diabetes-treatment`; gold ['diabetes-treatment']; abstained=True.
- `test-diabetes-treatment` / `hybrid`: top hit `diabetes-treatment`; gold ['diabetes-treatment']; abstained=True.
- `test-diabetes-treatment` / `hybrid_rerank`: top hit `diabetes-treatment`; gold ['diabetes-treatment']; abstained=True.
