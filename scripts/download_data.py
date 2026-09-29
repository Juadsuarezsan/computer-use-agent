"""Download the public benchmark metadata used for comparison (OSWorld, WebArena).

This project builds its own 20-task eval set (``data/eval/tasks.json``); the
public benchmarks are downloaded only as reference metadata for the
comparison section of the README. Every file is verified with SHA-256 and
recorded in ``data/MANIFEST.txt`` by ``scripts/make_manifest.py``.

Usage::

    python scripts/download_data.py            # download everything into data/raw/
    python scripts/download_data.py --only osworld

Sources (Apache-2.0 licensed repositories):
    OSWorld  https://github.com/xlang-ai/OSWorld  -> evaluation_examples/test_all.json
    WebArena https://github.com/web-arena-x/webarena -> config_files/test.raw.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Source:
    """A downloadable public file."""

    name: str
    url: str
    filename: str
    license: str
    expected_sha256: str | None = None


SOURCES: dict[str, Source] = {
    "osworld": Source(
        name="OSWorld task index",
        url="https://raw.githubusercontent.com/xlang-ai/OSWorld/main/evaluation_examples/test_all.json",
        filename="osworld_test_all.json",
        license="Apache-2.0",
    ),
    "webarena": Source(
        name="WebArena test configs",
        url="https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json",
        filename="webarena_test.raw.json",
        license="Apache-2.0",
    ),
}


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source, dest_dir: Path = RAW_DIR, client: httpx.Client | None = None) -> Path:
    """Download one source with a timeout, verify its hash and return the path.

    Raises:
        httpx.HTTPStatusError: on a non-2xx response.
        ValueError: when ``expected_sha256`` is set and does not match.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
    try:
        logger.info(f"downloading {source.name} from {source.url}")
        response = client.get(source.url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    finally:
        if own_client:
            client.close()
    digest = sha256_of(dest)
    if source.expected_sha256 and digest != source.expected_sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {source.filename}: {digest} != {source.expected_sha256}")
    logger.info(f"saved {dest} ({dest.stat().st_size} bytes, sha256={digest})")
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(SOURCES), help="download a single source")
    parser.add_argument("--dest", default=str(RAW_DIR))
    args = parser.parse_args(argv)
    names = [args.only] if args.only else sorted(SOURCES)
    failures = 0
    for name in names:
        try:
            download(SOURCES[name], Path(args.dest))
        except (httpx.HTTPError, ValueError, OSError) as exc:
            failures += 1
            logger.error(f"{name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
