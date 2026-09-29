"""Regenerate ``eval/runs/*.json`` and ``eval/RESULTS.md``.

Usage::

    python -m eval.run                # stub baseline + ablations on the Playwright sandbox
    python -m eval.run --vm fake      # harness smoke run without Chromium
    python -m eval.run --with-claude  # adds the Claude rows (needs ANTHROPIC_API_KEY)

Rows that need an API key are never invented: without the key they stay
``pendiente (requiere ANTHROPIC_API_KEY)``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings
from src.eval.runner import (
    RESULTS_MD,
    RUNS_DIR,
    SuiteConfig,
    render_results_md,
    run_safety_ablation,
    run_suite,
    save_run,
)
from src.observability import configure_logging


def _configs(vm: str, with_claude: bool) -> list[SuiteConfig]:
    configs = [
        SuiteConfig(
            name=f"stub-{vm}",
            label=f"Baseline 1 — StubReasoner (scripted DOM oracle, no LLM) + heuristic verifier on {vm}",
            reasoner="stub",
            vm=vm,
            verifier=True,
            note=(
                "Upper bound of the harness: the plan is scripted against DOM selectors, so this row "
                "validates the sandbox, executor, safety layer and checkers, not model capability."
            ),
        ),
        SuiteConfig(
            name=f"stub-{vm}-no-verifier",
            label=f"Baseline 2 / ablation — StubReasoner without verifier on {vm}",
            reasoner="stub",
            vm=vm,
            verifier=False,
            note="Same scripted plans with the per-step verifier disabled (ablation of the verifier node).",
        ),
    ]
    if with_claude:
        configs.append(
            SuiteConfig(
                name=f"claude-{vm}",
                label="Claude Computer Use agent (full system)",
                reasoner="claude",
                vm=vm,
                verifier=True,
            )
        )
        configs.append(
            SuiteConfig(
                name=f"claude-{vm}-no-verifier",
                label="Ablation — Claude agent without verifier",
                reasoner="claude",
                vm=vm,
                verifier=False,
            )
        )
    return configs


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.with_claude and not settings.llm_enabled:
        raise SystemExit("--with-claude requires ANTHROPIC_API_KEY")
    today = date.today()
    files: list[str] = []
    suites: list[dict[str, Any]] = []
    for config in _configs(args.vm, args.with_claude):
        report = await run_suite(config, settings=settings)
        files.append(save_run(report, runs_dir=RUNS_DIR, day=today).name)
        suites.append(report)
    safety = None
    if not args.skip_safety:
        safety = await run_safety_ablation(settings=settings, vm_kind=args.vm)
        files.append(save_run(safety, runs_dir=RUNS_DIR, day=today).name)
    existing = sorted(p.name for p in RUNS_DIR.glob("*.json"))
    markdown = render_results_md(
        suites, safety=safety, run_files=sorted(set(files) | set(existing)), model=settings.anthropic_model
    )
    Path(RESULTS_MD).write_text(markdown, encoding="utf-8")
    logger.info(f"wrote {RESULTS_MD}")
    summary = {s["name"]: s["aggregate"]["overall"]["success_rate"] for s in suites}
    logger.info(f"success rates (sin LLM salvo filas claude-*): {json.dumps(summary)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vm", choices=["playwright", "fake"], default="playwright")
    parser.add_argument("--with-claude", action="store_true", help="also run the Claude rows (needs a key)")
    parser.add_argument("--skip-safety", action="store_true", help="skip the safety ablation")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
