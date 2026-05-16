"""Computer-Use eval — runs each task through the agent on a fake VM."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics

from src.agent.orchestrator import run_task
from src.eval.tasks import TASKS


async def run_eval(max_steps: int = 10) -> dict:
    results = []
    for t in TASKS:
        out = await run_task(task=t["task"], max_steps=max_steps)
        results.append({
            "id": t["id"],
            "category": t["category"],
            "task": t["task"][:60],
            "success": out.success,
            "steps": out.total_steps,
            "blocked": out.blocked_actions,
            "latency_ms": out.latency_ms,
        })

    n = len(results)
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    return {
        "n": n,
        "success_rate": sum(r["success"] for r in results) / n,
        "avg_steps": statistics.mean(r["steps"] for r in results),
        "avg_latency_ms": statistics.mean(r["latency_ms"] for r in results),
        "blocked_total": sum(r["blocked"] for r in results),
        "by_category": {
            cat: {
                "n": len(items),
                "success_rate": sum(r["success"] for r in items) / len(items),
                "avg_steps": statistics.mean(r["steps"] for r in items),
            }
            for cat, items in by_cat.items()
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    report = asyncio.run(run_eval(max_steps=args.max_steps))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Tasks: {report['n']}  success: {report['success_rate']:.1%}")
        for cat, info in report["by_category"].items():
            print(f"  {cat:<18} {info['success_rate']:.1%}  avg_steps={info['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
