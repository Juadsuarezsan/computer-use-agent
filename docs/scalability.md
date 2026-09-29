# Scalability — Computer Use Agent

Assumptions used below (from `eval/RESULTS.md` and the pricing table in `src/observability.py`):

* a task takes 6.5 steps on the scripted baseline; a model-driven run is expected to take more
  (OSWorld/WebArena style agents typically need 10–30 steps) — we plan with **15 steps/task**;
* each step sends one 1024×768 screenshot (≈1,100–1,600 input tokens for Sonnet-class models)
  plus the pruned history (3 images) and returns ≈100 output tokens;
* pinned price: $3 / M input, $15 / M output tokens (`claude-sonnet-4-5-20250929`).

## Cost per task (model, not measured yet — needs the key)

Per step ≈ 4 images × 1,400 + 800 text ≈ 6,400 input tokens and 100 output tokens →
≈ $0.019 + $0.0015 ≈ **$0.02 per step**, **≈ $0.30 per 15-step task**. A verifier call adds two
images (≈ 2,800 tokens, ≈ $0.009) per step, so "with verifier" roughly doubles the bill, which
is why the ablation row exists. Prompt caching of the system prompt and tool definition would
shave the fixed part of every request.

## 1,000 monthly users

Assume 1,000 users × 20 tasks/month = 20,000 tasks. Tokens: 20k × 15 × 6.4k ≈ 1.9 B input,
30 M output → ≈ $5,800 + $450 ≈ **$6.3k/month in model spend** without the verifier, ~$9k
with it. Infrastructure: one sandbox instance per concurrent task; with 15 steps × ~5 s of model
latency ≈ 75 s per task and a peak of 5 % of daily tasks in the busiest hour, that is
≈ 35 concurrent sandboxes. Chromium sandboxes cost ~300 MB RAM each (≈ 12 GB), Xvfb desktops
~1 GB each. Two 16-vCPU/64 GB nodes suffice; PostgreSQL (audit) stays tiny (≈ 300 k step rows
per month, < 1 GB).

## 100× tasks

* **Sandbox pool.** `PlaywrightVM` is one browser per task and is created per request. At
  scale it becomes a pool of warm browser contexts (Playwright `browser.new_context()` is ~50 ms
  vs ~500 ms for a launch), sized by a queue; the `VM` protocol does not change.
* **Queue instead of synchronous HTTP.** `POST /api/run-task` blocks for the whole run. At
  100× the endpoint should enqueue (Redis/RabbitMQ), return a `trace_id`, and stream progress
  (SSE) from the audit log; the demo already consumes trace JSON, so the front end needs no new
  format.
* **Screenshots.** Move PNGs from local disk to object storage (S3) keyed by
  `trace_id/step`; the audit row already stores the path.
* **Cost controls.** Per-user step budgets, image downscaling (see `docs/performance.md`),
  caching of identical screenshots (same digest → reuse the previous model decision only when the
  verifier says nothing changed), and prompt caching for the fixed prefix.
* **Rate limits.** slowapi per-IP limits protect the API; at 100× the limit moves to the queue
  (per-tenant concurrency) and the LLM client needs a global token-bucket to stay under the
  Anthropic rate limit.

## 10,000× (platform scale)

Shard by tenant: each tenant gets its own sandbox pool and audit database; the API becomes
stateless workers behind a load balancer. The safety layer and the checkers are pure functions
and scale horizontally. The remaining bottleneck is model latency and cost, so the next levers
are a cheaper model for "easy" steps (routing by verifier confidence) and batching the final
validator through the Batch API.

## What does not scale in the current code (honest list)

* One browser per task, launched per request — fine for the eval, not for a service.
* The audit log writes synchronously from the request path.
* Screenshots on local disk; `data/screenshots/` grows without rotation.
* No prompt caching, no screenshot downscaling by default.
* HITL is a flag; there is no operator queue/UI.
