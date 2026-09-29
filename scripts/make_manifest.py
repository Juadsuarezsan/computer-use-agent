"""Write ``data/MANIFEST.txt`` with the SHA-256 of every dataset artefact.

Covers the hand-written eval set, the adversarial set, the fixture web apps
(the "environment" the tasks run on) and any downloaded raw benchmark files.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import sha256_of  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "MANIFEST.txt"
TRACKED = [
    REPO_ROOT / "data" / "eval" / "tasks.json",
    REPO_ROOT / "data" / "eval" / "adversarial.json",
    *sorted((REPO_ROOT / "sandbox" / "webapps").glob("*.html")),
]


def build_manifest(paths: list[Path], raw_dir: Path = REPO_ROOT / "data" / "raw") -> str:
    """Render the manifest text for the given files (plus raw downloads if present)."""
    lines = [
        "# data/MANIFEST.txt — SHA-256 of every dataset artefact (regenerate: python scripts/make_manifest.py)",
        f"# generated: {date.today().isoformat()}",
        "# eval set: hand-written by the author (ground_truth_source=manual); no LLM-generated data.",
        "",
    ]
    all_paths = list(paths)
    if raw_dir.exists():
        all_paths.extend(sorted(p for p in raw_dir.iterdir() if p.is_file()))
    for path in all_paths:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"{sha256_of(path)}  {rel.as_posix()}  ({path.stat().st_size} bytes)")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the manifest."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(build_manifest(TRACKED), encoding="utf-8")
    sys.stdout.write(f"wrote {MANIFEST}\n")


if __name__ == "__main__":
    main()
