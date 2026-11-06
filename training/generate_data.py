"""Record the fine-tuning dataset from the scripted oracle (no API, no hand labels).

Usage::

    python -m training.generate_data                       # 25 variants per task -> data/local_model/
    python -m training.generate_data --variants 40 --out data/local_model_v2
    python -m training.generate_data --heldout-apps legacy.html   # also hold out a whole app

Needs the Playwright Chromium sandbox (``python -m playwright install chromium``
and ``CHROMIUM_PATH`` pointing to it, or empty to use Playwright's own build).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from src.config import REPO_ROOT, get_settings
from src.eval.tasks import load_tasks
from src.local_model.recording import DEFAULT_HISTORY, generate_dataset
from src.local_model.variants import DEFAULT_HELDOUT, DEFAULT_VARIANTS_PER_TASK
from src.observability import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "local_model"))
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS_PER_TASK)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--heldout", default=",".join(DEFAULT_HELDOUT), help="task ids reserved for test")
    parser.add_argument("--heldout-apps", default="", help="fixture apps reserved for test (comma-separated)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    settings = get_settings()
    tasks = load_tasks(settings.tasks_file)
    heldout = tuple(t.strip() for t in args.heldout.split(",") if t.strip())
    heldout_apps = tuple(a.strip() for a in args.heldout_apps.split(",") if a.strip())
    report = asyncio.run(
        generate_dataset(
            tasks,
            Path(args.out),
            variants_per_task=args.variants,
            seed=args.seed,
            heldout=heldout,
            heldout_apps=heldout_apps,
            val_fraction=args.val_fraction,
            history_len=args.history,
            settings=settings,
        )
    )
    logger.info(
        f"trajectories={report.trajectories} pairs={report.pairs} "
        f"visual_only_fallbacks={report.visual_only_fallbacks} discarded_runs={len(report.discarded)}"
    )


if __name__ == "__main__":
    main()
