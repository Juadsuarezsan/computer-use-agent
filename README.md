# Project 05 — Computer Use Agent

> Autonomous agent that operates a virtualized Ubuntu desktop to automate workflows without API access. Reads screenshots, decides actions, executes clicks/keyboard. Anthropic's Computer Use frontier applied to back-office automation.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5%20%2B%20computer__use-7c5cff)]()
[![Bench](https://img.shields.io/badge/bench-OSWorld%20%2B%20WebArena-22d3ee)]()

**Industrial use case:** RPA for legacy systems without APIs, back-office automation in banking/insurance.

## What this project does

Receives a task in natural language ("extract Q3 sales from the legacy system and save to CSV"). Operates a Ubuntu VM by: capturing screenshot → Claude with `computer_use` tool decides next action → xdotool executes it → loop until task complete or max steps.

## Architecture

```
Task: "Extract Q3 sales from legacy system"
   │
   ▼
[Planner] Claude → initial plan
   │
   ▼
LOOP DE ACCIÓN:
   ├──► [Screenshot capture] scrot on Xvfb
   ├──► [Vision + Reasoning] Claude with computer_use tool
   │      → click(x, y) | type(text) | key(combo) | scroll
   ├──► [Action Executor] xdotool
   ├──► [Verifier] Claude — "did the step complete? unexpected dialog?"
   ├──► [State Updater] LangGraph state with screenshot history
   └──► continue until task_complete or max_steps
   │
   ▼
[Final Validator] Claude → does output satisfy original task?
   │
   ▼
[Audit Log] PostgreSQL
```

## Roadmap to v1.0.0

1. [ ] Dockerized Ubuntu 22.04 + Xvfb + VNC environment
2. [ ] xdotool wrappers for click/type/key/scroll
3. [ ] Screenshot capture with scrot
4. [ ] LangGraph loop with screenshot history in state
5. [ ] Action executor with safety filters (no rm -rf, no sudo)
6. [ ] 20 custom evaluation tasks with ground truth success criteria
7. [ ] OSWorld benchmark comparison (subset)
8. [ ] Next.js demo with live VNC viewer
9. [ ] 5 recorded MP4 videos of task executions
10. [ ] Safety layer documented (blocked action list)

## Stack

| Layer | Technology |
|---|---|
| Computer Use | Anthropic Claude Sonnet 4.5 with `computer_20250124` tool |
| VM | Docker container — Ubuntu 22.04 + Xvfb + VNC |
| Screenshots | scrot or gnome-screenshot |
| Input automation | xdotool |
| Orchestration | LangGraph with screenshot history in state |
| Storage | PostgreSQL for action logs, filesystem for screenshots |
| Frontend | Next.js with live VNC viewer |
| Observability | LangSmith with action-by-action traces |

## Definition of Done — project-specific

- [ ] Reproducible VM via `docker compose up`
- [ ] 20 custom tasks defined with ground truth
- [ ] Success rate reported by category (form filling, data extraction, web nav, multi-step)
- [ ] Demo lets user pick a task and watch the agent execute in streaming
- [ ] 5 recorded videos (MP4 in repo or YouTube unlisted) of real executions
- [ ] Safety layer documented (blocked actions list)

Plus the 12 universal DoD blocks.

## License

MIT.
