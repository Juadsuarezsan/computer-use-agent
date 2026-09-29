# Performance — where the time goes

## Measured (offline, no LLM) — from `eval/runs/2026-09-29-stub-playwright.json`

| Metric | Value |
|---|---|
| Mean latency per task (20 tasks) | 0.65 s |
| p95 latency per task | 1.12 s |
| Mean steps per task | 6.5 (min 2, max 12) |
| Latency per step (≈ latency / steps) | ≈ 100 ms |

Per step the harness does: `page.screenshot()` (≈ 40–60 ms at 1024×768 PNG), the element-map
`page.evaluate` (≈ 5 ms), the action (`mouse.click` + 30 ms settle, `keyboard.type` at 5 ms per
character), the heuristic verifier (SHA-256 over the PNG, < 1 ms) and the SQLite/JSON
bookkeeping. Typing dominates the form-filling category (9.2 steps, 1.04 s): "The export button
is disabled" is 30 characters × 5 ms. Data extraction is the fastest (2.6 steps, 0.26 s).

The verifier ablation shows the heuristic verifier is free (0.65 s vs 0.66 s without it).

## Expected with the model (the real bottleneck)

With Claude in the loop the harness cost (~100 ms/step) is noise. Every step is one Messages
request with 1–4 images: latency is dominated by **vision input tokens** and generation. Public
numbers for Sonnet-class models put a 1024×768 screenshot around 1,100–1,600 tokens and a
computer-use turn at 3–8 s. Expect 15-step tasks to take 60–120 s end to end; p95 will be set
by retries on 429/5xx (`tenacity`, up to 3 attempts with 1–20 s backoff).

Levers, in order of expected impact (to be measured when the key is available — the harness
already records `latency_ms`, `tokens_in`, `tokens_out` and `cost_usd` per step):

1. **Screenshot size.** Downscale to 800×600 or 1024×768→768×576 before sending; Anthropic's
   tool takes `display_width_px/height_px`, and coordinates are scaled back by the executor.
   Roughly halves image tokens.
2. **Image pruning.** `max_images=3` (default) vs the full history: bounds context growth so the
   per-step cost stays flat instead of growing linearly with the step index.
3. **Identical-screenshot cache.** `Observation.digest()` already exists; if the digest did not
   change after a `wait`/`screenshot` action, skip the model call and re-use the previous
   decision context (only safe when the verifier agrees nothing changed).
4. **Prompt caching** of the system prompt and tool definition (fixed prefix on every request).
5. **Verifier only when needed.** Call the LLM verifier only when the heuristic verifier says
   "no visible change" — the ablation row is designed to quantify this.

## Harness overheads worth knowing

* Chromium launch is ~400–600 ms; the eval reuses one browser for all tasks and `reset()` only
  navigates (`about:blank` then the app) to guarantee fresh JS state.
* Screenshots are written to disk on every step (needed for the audit trail and the demo); on a
  slow disk this becomes visible — set `SCREENSHOT_DIR` to tmpfs for benchmarking.
* The audit log writes synchronously in the request path (SQLite: < 2 ms per run).

## How to reproduce the numbers

```bash
python -m eval.run              # writes eval/runs/*.json and eval/RESULTS.md
python -m src.eval.runner --vm playwright --json | jq '.aggregate.overall.latency_ms'
```
