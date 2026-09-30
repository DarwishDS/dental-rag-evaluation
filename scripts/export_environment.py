"""Record dependency versions and downloaded model artifact hashes without private paths."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    lines = []
    for line in frozen.splitlines():
        if line.startswith(("-e", "#", "dental-rag-evaluation")):
            continue
        if line.lower().startswith(("pywin32==", "win32-setctime==")):
            line += '; sys_platform == "win32"'
        lines.append(line)
    (ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    models = []
    for path in sorted((ROOT / ".cache").rglob("*.onnx")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        models.append({"artifact": path.relative_to(ROOT / ".cache").as_posix(),
                       "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    destination = ROOT / "reports" / "semantic-test" / "model-artifacts.json"
    destination.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(lines)} dependencies and {len(models)} ONNX artifacts")


if __name__ == "__main__":
    main()
