"""Check staged files and normalize final newlines before the release commit."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = (
        subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    sensitive = re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    for name in filter(None, files):
        path = ROOT / name
        if any(part in {".venv", ".cache", ".env", "artifacts"} for part in path.parts):
            raise ValueError(f"Local-only file staged: {name}")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpected large file staged: {name}")
        if path.suffix == ".png":
            continue
        content = path.read_text(encoding="utf-8")
        if sensitive.search(content):
            raise ValueError(f"Possible secret in {name}; value withheld")
        normalized = content.rstrip() + "\n"
        if normalized != content:
            path.write_text(normalized, encoding="utf-8", newline="\n")
    print("Release file checks passed")


if __name__ == "__main__":
    main()
