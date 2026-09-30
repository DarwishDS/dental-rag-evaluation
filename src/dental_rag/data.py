import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    title: str
    section: str
    url: str
    text: str

    @property
    def search_text(self):
        return f"{self.title}. {self.section}. {self.text}"


def load_chunks(path: Path) -> list[Chunk]:
    rows = [
        Chunk(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("Corpus must contain nonempty, unique chunk IDs")
    return rows


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path, chunks: list[Chunk]) -> list[dict]:
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = {c.id for c in chunks}
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation IDs must be unique and nonempty")
    for case in cases:
        if case["split"] not in {"dev", "test"}:
            raise ValueError("Invalid split")
        if bool(case["relevant_ids"]) != case["answerable"]:
            raise ValueError("Answerability and gold labels disagree")
        if not set(case["relevant_ids"]) <= ids:
            raise ValueError("Evaluation references missing corpus chunks")
    return cases
