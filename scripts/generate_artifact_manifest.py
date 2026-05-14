from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["artifacts", "configs", "docs", "outputs", "reports", "slides", "src", "tests"]
OUTPUT = ROOT / "artifacts" / "artifact_manifest.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def main() -> None:
    lines = [
        "# Artifact Manifest",
        "",
        "Use this file as a submission audit trail. Regenerate with:",
        "",
        "```bash",
        "python scripts/generate_artifact_manifest.py",
        "```",
        "",
        "| Path | Size bytes | SHA256 prefix |",
        "|---|---:|---|",
    ]
    for directory in INCLUDE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.resolve() == OUTPUT.resolve():
                continue
            if path.is_file() and path.name != ".gitkeep":
                rel = path.relative_to(ROOT)
                lines.append(f"| `{rel}` | {path.stat().st_size} | `{sha256(path)}` |")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
