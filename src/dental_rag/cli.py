import argparse
import json
from pathlib import Path

from .config import Settings
from .data import load_cases, load_chunks
from .evaluation import STRATEGIES, evaluate
from .pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Source-backed dental RAG experiments")
    parser.add_argument("--backend", choices=["semantic", "smoke"], default=None)
    parser.add_argument(
        "--generator", choices=["extractive", "openai", "groq", "gemini"], default=None
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=STRATEGIES, default="hybrid_rerank")
    experiment = sub.add_parser("evaluate")
    experiment.add_argument("--split", choices=["dev", "test"], default="test")
    experiment.add_argument("--out", type=Path, default=Path("artifacts/evaluation"))
    experiment.add_argument("--judge", action="store_true")
    sub.add_parser("validate")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings(
        **{k: v for k, v in {"backend": args.backend, "generator": args.generator}.items() if v}
    )
    if args.command == "validate":
        chunks = load_chunks(settings.data_dir / "corpus.jsonl")
        cases = load_cases(settings.data_dir / "eval.jsonl", chunks)
        print(f"Valid: {len(chunks)} chunks, {len(cases)} evaluation cases")
        return
    if args.command == "serve":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    pipeline = Pipeline(settings)
    try:
        if args.command == "ask":
            print(json.dumps(pipeline.query(args.question, args.strategy), indent=2))
        elif args.command == "evaluate":
            report = evaluate(pipeline, args.split, args.out, args.judge)
            print(json.dumps(report["summary"], indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
