"""Record the demo traces: 20 tasks run by the offline StubReasoner on the Playwright sandbox.

Writes ``demo/predictions.json`` (one trace per task: action, reasoning, safety
verdict, verifier verdict and a downscaled JPEG screenshot per step) plus the
screenshots under ``demo/screenshots/``. The traces are explicitly labelled as
produced by the deterministic stub (no LLM), so the static demo never presents
them as Claude runs.

Usage::

    python scripts/record_demo_traces.py [--width 512]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.orchestrator import run_task  # noqa: E402
from src.agent.playwright_vm import PlaywrightVM  # noqa: E402
from src.agent.reasoner import StubReasoner  # noqa: E402
from src.agent.verifier import HeuristicVerifier  # noqa: E402
from src.api.schemas import Action  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.eval.runner import AdversarialReasoner, load_adversarial_actions  # noqa: E402
from src.eval.tasks import load_tasks  # noqa: E402
from src.observability import configure_logging  # noqa: E402
from src.safety.blocklist import check_action  # noqa: E402

DEMO_DIR = REPO_ROOT / "demo"
SHOTS_DIR = DEMO_DIR / "screenshots"


def downscale(png_path: str, dest: Path, width: int) -> None:
    """Save a JPEG thumbnail of a PNG screenshot."""
    with Image.open(png_path) as im:
        ratio = width / im.width
        small = im.convert("RGB").resize((width, int(im.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=62, optimize=True)
        dest.write_bytes(buf.getvalue())


async def record(width: int) -> dict[str, Any]:
    """Run every task and collect traces."""
    settings = get_settings()
    tasks = load_tasks(settings.tasks_file)
    if SHOTS_DIR.exists():
        shutil.rmtree(SHOTS_DIR)
    SHOTS_DIR.mkdir(parents=True)
    vm = PlaywrightVM(
        chromium_path=settings.chromium_path,
        webapps_dir=settings.webapps_dir,
        screenshot_dir=str(REPO_ROOT / "data" / "screenshots" / "demo"),
    )
    traces: list[dict[str, Any]] = []
    try:
        for spec in tasks:
            logger.info(f"recording {spec.id}")
            response = await run_task(
                task=spec.task,
                task_spec=spec.as_spec(),
                vm=vm,
                reasoner=StubReasoner(),
                verifier=HeuristicVerifier(),
                settings=settings,
            )
            final = await vm.screenshot()
            steps = []
            for st in response.steps:
                shot = SHOTS_DIR / f"{spec.id}_{st.step:02d}.jpg"
                if st.screenshot_path:
                    downscale(st.screenshot_path, shot, width)
                verdict = check_action(st.action)
                steps.append(
                    {
                        "step": st.step,
                        "screenshot": f"screenshots/{shot.name}",
                        "action": st.action.model_dump(exclude_none=True),
                        "action_summary": st.action.summary(),
                        "reasoning": st.reasoning,
                        "execution_result": st.execution_result,
                        "safety": {
                            "allowed": verdict.allowed and not st.safety_blocked,
                            "rule": verdict.rule,
                            "reason": st.safety_reason or verdict.reason,
                            "requires_human": verdict.requires_human,
                        },
                        "verifier": {"ok": st.verifier_ok, "note": st.verifier_note},
                        "latency_ms": st.latency_ms,
                    }
                )
            final_shot = SHOTS_DIR / f"{spec.id}_final.jpg"
            downscale(final.screenshot_path, final_shot, width)
            traces.append(
                {
                    "id": spec.id,
                    "category": spec.category,
                    "task": spec.task,
                    "app": spec.app,
                    "notes": spec.notes,
                    "success": response.success,
                    "ground_truth_checked": response.ground_truth_checked,
                    "checker": spec.checker,
                    "total_steps": response.total_steps,
                    "latency_ms": response.latency_ms,
                    "verifier_failures": response.verifier_failures,
                    "final_output": response.final_output,
                    "final_screenshot": f"screenshots/{final_shot.name}",
                    "steps": steps,
                }
            )
        safety_demo = []
        for case in load_adversarial_actions()[:6]:
            action = Action.model_validate(case["action"])
            response = await run_task(
                task="adversarial probe",
                max_steps=2,
                task_spec={"id": "adv", "app": "forms.html", "start": "#support", "plan": []},
                reasoner=AdversarialReasoner([action]),
                vm=vm,
                settings=settings,
            )
            st = response.steps[0]
            safety_demo.append(
                {
                    "id": case["id"],
                    "action_summary": action.summary(),
                    "blocked": st.safety_blocked,
                    "reason": st.safety_reason,
                    "expected_rule": case["expected_rule"],
                }
            )
    finally:
        await vm.close()
    return {
        "project": "computer-use-agent",
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": (
            "StubReasoner (scripted DOM-oracle plan, deterministic, NO LLM) on the Playwright/Chromium "
            "sandbox. These traces show the harness, sandbox and safety layer; they are not Claude runs."
        ),
        "llm_used": False,
        "display": {"width": settings.display_width, "height": settings.display_height},
        "thumbnail_width": width,
        "n_tasks": len(traces),
        "success_count": sum(1 for t in traces if t["success"]),
        "traces": traces,
        "safety_demo": safety_demo,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    configure_logging("INFO")
    payload = asyncio.run(record(args.width))
    out = DEMO_DIR / "predictions.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in SHOTS_DIR.glob("*.jpg"))
    logger.info(f"wrote {out} with {payload['n_tasks']} traces; screenshots {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
