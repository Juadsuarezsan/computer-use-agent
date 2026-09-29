# LinkedIn post (draft)

I built a Computer Use agent that operates a screen the way a back-office analyst does: look at
a screenshot, decide one click or keystroke, execute, look again.

What I think matters more than the loop itself:

1. Ground truth that is not the agent's opinion. Each of the 20 evaluation tasks (forms, legacy
   table extraction, dashboard navigation, multi-step workflows) has a deterministic checker on
   the application DOM. Typing "Mallory" instead of "Alice" fails. Reaching Home through the
   sidebar when the task said "use the breadcrumb" fails.

2. A safety layer you can audit. Every action passes a blocklist (rm -rf, sudo, curl | sh,
   writes to /dev/sda, force push, external URLs, session-killing key combos) and a
   human-in-the-loop policy for payments, mass deletion and external e-mail. Measured: 16/16
   adversarial actions blocked with the layer on, 0/16 without.

3. No invented numbers. The offline rows in the results table come from a scripted baseline and
   are labelled "no LLM". The Claude rows say "pending, requires API key" until a real run
   fills them.

Stack: Anthropic computer_20250124 tool (claude-sonnet-4-5-20250929, pinned), LangGraph with
five explicit nodes, Playwright/Chromium sandbox (real screenshots and pixel-coordinate input),
Xvfb + xdotool desktop image for legacy apps, FastAPI, PostgreSQL/SQLite audit log, 155 tests
at 97 % coverage, mypy --strict.

Repo: https://github.com/Juadsuarezsan/computer-use-agent
Demo (recorded traces, step by step): https://juadsuarezsan.github.io/computer-use-agent/demo/

#AIEngineering #ComputerUse #LangGraph #Anthropic #RPA #Python
