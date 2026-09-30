import importlib.metadata
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import fingerprint, load_cases
from .pipeline import Pipeline

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def retrieval_metrics(retrieved: list[str], relevant: list[str]) -> dict:
    gold = set(relevant)
    if not gold:
        return {}
    positions = [i for i, cid in enumerate(retrieved, 1) if cid in gold]
    dcg = sum(1 / math.log2(i + 1) for i in positions)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), len(retrieved)) + 1))
    return {
        "hit_at_1": float(bool(retrieved and retrieved[0] in gold)),
        "hit_at_k": float(bool(positions)),
        "recall_at_k": len(set(retrieved) & gold) / len(gold),
        "mrr_at_k": 1 / min(positions) if positions else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def paired_interval(before, after, seed=42, samples=2000):
    if len(before) != len(after) or not before:
        raise ValueError("Paired nonempty samples required")
    deltas = np.asarray(after) - np.asarray(before)
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, len(deltas), size=(samples, len(deltas)))].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def judge_answer(question, answer, reference, hits, model):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=reference,
        retrieval_context=[h["text"] for h in hits],
    )
    result = {}
    for label, metric in [
        ("faithfulness", FaithfulnessMetric(model=model, async_mode=False)),
        ("answer_relevancy", AnswerRelevancyMetric(model=model, async_mode=False)),
    ]:
        metric.measure(case)
        result[label] = {"score": metric.score, "reason": metric.reason}
    return result


def evaluate(
    pipeline: Pipeline,
    split="test",
    out: Path = Path("artifacts/evaluation"),
    judge=False,
    progress=print,
):
    if judge and pipeline.settings.generator != "openai":
        raise ValueError(
            "Use --generator openai with --judge; extractive scores are not LLM evidence"
        )
    path = pipeline.settings.data_dir / "eval.jsonl"
    cases = [c for c in load_cases(path, pipeline.chunks) if c["split"] == split]
    if not cases:
        raise ValueError("No cases in selected split")
    # Explicitly exclude loading and first inference from measured request latency.
    for strategy in STRATEGIES:
        progress(f"Warming {strategy} ({pipeline.settings.backend})")
        pipeline.retriever.warmup(strategy)
    rows = []
    for case in cases:
        progress(f"Evaluating {case['id']}")
        for strategy in STRATEGIES:
            result = pipeline.query(case["question"], strategy)
            ids = [h["id"] for h in result["hits"]]
            metrics = retrieval_metrics(ids, case["relevant_ids"])
            valid_ids = {h["id"] for h in result["hits"]}
            metrics["citation_validity"] = (
                float(set(result["cited_ids"]) <= valid_ids) if result["cited_ids"] else None
            )
            metrics["abstention_correct"] = float(result["abstained"] != case["answerable"])
            metrics["evidence_hit_in_citations"] = (
                float(bool(set(result["cited_ids"]) & set(case["relevant_ids"])))
                if case["answerable"]
                else None
            )
            record = {
                "case_id": case["id"],
                "category": case["category"],
                "answerable": case["answerable"],
                "reference_answer": case["reference_answer"],
                "relevant_ids": case["relevant_ids"],
                "metrics": metrics,
                **result,
            }
            if judge and case["answerable"] and not result["abstained"]:
                record["judge"] = judge_answer(
                    case["question"],
                    result["answer"],
                    case["reference_answer"],
                    result["hits"],
                    pipeline.settings.judge_model,
                )
            rows.append(record)
    summary = {}
    for strategy in STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]
        means = {}
        for key in sorted({k for r in subset for k in r["metrics"]}):
            values = [r["metrics"][key] for r in subset if r["metrics"].get(key) is not None]
            means[key] = statistics.mean(values) if values else None
        latency = [r["total_ms"] for r in subset]
        summary[strategy] = {
            **means,
            "latency_p50_ms": float(np.median(latency)),
            "latency_p95_ms": float(np.quantile(latency, 0.95)),
            "answerable_cases": sum(r["answerable"] for r in subset),
            "unanswerable_cases": sum(not r["answerable"] for r in subset),
        }
        for metric in ["faithfulness", "answer_relevancy"]:
            values = [r["judge"][metric]["score"] for r in subset if "judge" in r]
            summary[strategy][metric] = statistics.mean(values) if values else None
            summary[strategy][metric + "_scored_count"] = len(values)
        costs = [r["usage"].get("estimated_usd") for r in subset]
        summary[strategy]["estimated_generation_usd"] = (
            sum(costs) if all(c is not None for c in costs) else None
        )
    pairs = {}
    for strategy in STRATEGIES[1:]:
        for metric in ["hit_at_1", "mrr_at_k", "recall_at_k"]:
            before = [
                r["metrics"][metric] for r in rows if r["strategy"] == "dense" and r["answerable"]
            ]
            after = [
                r["metrics"][metric] for r in rows if r["strategy"] == strategy and r["answerable"]
            ]
            pairs[f"{strategy}_vs_dense/{metric}"] = paired_interval(before, after)
    versions = {}
    for package in ["fastembed", "qdrant-client", "rank-bm25", "numpy", "deepeval"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "case_count": len(cases),
        "backend": pipeline.settings.backend,
        "generator": pipeline.settings.generator,
        "judge_enabled": judge,
        "corpus_sha256": pipeline.corpus_sha256,
        "eval_sha256": fingerprint(path),
        "configuration": pipeline.settings.model_dump(mode="json"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
        },
        "summary": summary,
        "paired_comparisons": pairs,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lines = [
        f"# Evaluation: {split}",
        "",
        f"Backend: **{pipeline.settings.backend}**; "
        f"generator: **{pipeline.settings.generator}**; cases: {len(cases)}.",
        "",
        "| Strategy | Hit@1 | Recall@3 | MRR@3 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, row in summary.items():
        lines.append(
            f"| {strategy} | {row['hit_at_1']:.3f} | {row['recall_at_k']:.3f} | "
            f"{row['mrr_at_k']:.3f} | {row['latency_p50_ms']:.1f} | "
            f"{row['latency_p95_ms']:.1f} |"
        )
    lines += [
        "",
        "Retrieval metrics exclude unanswerable cases. Latency excludes model loading "
        "and warmup. CI is paired bootstrap over questions, not over passages.",
        "",
        "Faithfulness and answer relevancy are null unless DeepEval judges an OpenAI run. "
        "Citation validity checks IDs only; it does not establish claim-level support.",
        "",
        "## Failures",
        "",
    ]
    for row in rows:
        if row["answerable"] and (row["metrics"]["hit_at_1"] == 0 or row["abstained"]):
            lines.append(
                f"- `{row['case_id']}` / `{row['strategy']}`: "
                f"top hit `{row['hits'][0]['id'] if row['hits'] else 'none'}`; "
                f"gold {row['relevant_ids']}; abstained={row['abstained']}."
            )
        elif not row["answerable"] and not row["abstained"]:
            lines.append(
                f"- `{row['case_id']}` / `{row['strategy']}`: unsupported question answered."
            )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
