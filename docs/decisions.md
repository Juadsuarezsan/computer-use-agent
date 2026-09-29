# Technical decisions — Computer Use Agent

Each entry: the choice, the alternative rejected, and the concrete reason.

## 1. LangGraph with five explicit nodes instead of a plain `while` loop

*Rejected:* a single Python loop (`while not done: screenshot(); reason(); execute()`).
*Why LangGraph:* the loop has four decision points that must be individually observable and
swappable — verification, reasoning, safety, execution. Making each a node gives per-node
input/output logging with the trace id (DoD block 5), lets the eval harness ablate a node
(`verifier=None`) without `if` soup, and gives LangSmith tracing for free. The cost is one
gotcha we hit: node names must not collide with state keys (`step`), which is now covered by a
test.

## 2. Playwright + headless Chromium as the reproducible sandbox, Xvfb/xdotool as the production path

*Rejected as the only option:* an Ubuntu + Xvfb + xdotool container.
*Why both:* a Docker desktop is the right target for legacy native apps, but it cannot run in
CI without privileged containers and it has no ground truth (a screenshot cannot tell you a
form was submitted). The Playwright sandbox runs anywhere Chromium runs, gives **real
screenshots and real input events at pixel coordinates** (the same interface the model sees),
and exposes the DOM for deterministic checkers. `XdoToolVM` keeps the desktop path
(double/right click, scroll, key names) behind the same `VM` protocol, and the sandbox image
ships `xdotool`, `scrot`, Xvfb, noVNC, Firefox and LibreOffice.

## 3. `xdotool` + `scrot` over `pyautogui` for the desktop driver

*Rejected:* `pyautogui`.
*Why:* `pyautogui` needs an X display in the API process and drags in `Xlib` bindings; the
executor should be a thin subprocess wrapper that can run in a different container from the API
and be mocked with an injected `runner`. `xdotool` also maps 1:1 to the key syntax Anthropic's
`computer` tool emits (`ctrl+a`, `Return`), so no translation layer is needed on the desktop
path (the Playwright path has an explicit map).

## 4. Deterministic blocklist as the first safety layer, LLM judge only as a second opinion

*Rejected:* asking the model "is this action safe?" as the primary guard.
*Why:* a guard must be cheap, auditable and impossible to talk out of. Substring and regex rules
run in microseconds, are unit-tested per family (17 dangerous / 7 benign cases), and produce a
rule id for the audit log. The measured ablation (16/16 blocked vs 0/16 without) is only
possible because the rules are deterministic. LLM-based review is reserved for what regexes
cannot see (visual social engineering) and is gated behind the API key.

## 5. `computer_20250124` rather than `computer_20241022`

*Rejected:* the 2024-10-22 tool version.
*Why:* the 2025-01-24 version adds `scroll` with direction/amount, `hold_key`, `wait` with a
duration, `triple_click` and `left_mouse_down/up`, which map onto richer `Action` types and
reduce the number of steps for scroll-heavy pages. The beta header is chosen per tool version
in `BETA_FLAGS`, so switching back is a config change (`COMPUTER_USE_TOOL_VERSION`).

## 6. Ground truth as DOM checkers, never as screenshot comparison or LLM judgement

*Rejected:* comparing the final screenshot with a golden image; using the LLM judge as the
success signal.
*Why:* screenshots are brittle to a 1-px layout change, and an LLM judge is exactly the kind of
"metric" the DoD forbids as sole evidence. Each task has a hand-written checker that reads
attributes the app sets on success (`#contact-result[data-email]`) or parses the final answer
(`answer_number`, `answer_set`). Negative tests prove the checkers are not vacuous (typing
"Mallory" instead of "Alice" fails; reaching Home via the sidebar instead of the breadcrumb
fails).

## 7. A scripted DOM-oracle stub as the offline baseline, labelled as such

*Rejected:* a keyword stub that always succeeds on a fake VM (the previous state of the repo);
inventing numbers for the Claude rows.
*Why:* the harness needs a run that exercises every real component without a key. The stub
resolves selectors from the live DOM, so its 20/20 proves the sandbox, executor, checkers,
audit and safety layer work end to end. It cannot measure model capability, and every table
that includes it says "fallback determinista, sin LLM"; the Claude rows are
`pendiente (requiere ANTHROPIC_API_KEY)`.

## 8. One `LLMClient` with `tenacity` instead of the SDK's built-in retries

*Rejected:* `AsyncAnthropic(max_retries=3)`.
*Why:* one retry policy for the reasoner, verifier and judge, visible in logs, with the same
exception classes we test against (`RateLimitError`, `APIConnectionError`,
`InternalServerError`, `APITimeoutError`) and a hard cap so a wall-clock budget can be reasoned
about (`timeout × attempts`). The SDK retries are disabled (`max_retries=0`) to avoid double
retrying.

## 9. Image pruning in the conversation instead of server-side compaction

*Rejected:* sending every screenshot of the run.
*Why:* a 1024×768 PNG costs on the order of 1,000+ input tokens; a 30-step run would resend
30 images per step. Keeping the last 3 images (`max_images`) and replacing older ones with a
text marker keeps cost roughly linear in the number of steps. The knob is measured in
`docs/performance.md`.

## 10. SQLite fallback for the audit log

*Rejected:* PostgreSQL-only.
*Why:* the audit log must exist in every run, including tests and the offline eval, otherwise
the "every action is logged" claim is only true in production. The same DDL runs on both
backends; PostgreSQL uses a `psycopg_pool` and a logged fallback instead of a silent
`except Exception`.
