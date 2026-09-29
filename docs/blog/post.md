# Building a Computer Use agent you can actually audit

*Draft for Dev.to / Medium. Code: https://github.com/Juadsuarezsan/computer-use-agent*

Every company I have worked with has at least one system nobody is allowed to touch. It is a
supplier portal from 2009, an ERP that only runs inside a Citrix session, an internal admin
console whose vendor went out of business. There is no API. There is a person who opens it every
morning, reads numbers off a table and types them somewhere else.

"Computer use" models promise to replace that person with an agent that looks at the screen and
acts. The demos are impressive. What the demos do not show is the part that decides whether you
can put this in front of a bank's back office: how do you know the agent did the task, how do you
stop it from doing something catastrophic, and how do you report results without lying to
yourself?

This post is about the second half. I built a Computer Use agent around Anthropic's `computer`
tool and LangGraph, and spent most of the effort on three things: **ground truth that is not the
agent's own opinion**, **a safety layer that can be audited**, and **an evaluation report that
refuses to invent numbers**.

## The loop, briefly

The agent is a five-node LangGraph graph:

```
observe → verify → reason → safety → execute → (loop until task_complete or max_steps)
```

* **observe** takes a screenshot of the sandbox. With the Playwright driver it also extracts a
  map of visible interactive elements (selector, role, label, centre coordinates) and the values
  of elements marked `data-field`.
* **verify** compares the screen before and after the previous action. Offline it is a hash of
  the screenshot plus the extracted fields; with a key it is Claude looking at both images and
  answering `OK:` or `FAIL:`. A failure is fed back to the reasoner as text in the next tool
  result, and a task that fails a step but still succeeds counts as *recovered*.
* **reason** asks the model for exactly one `computer` tool call and parses it into a typed
  `Action` (`click`, `double_click`, `right_click`, `mouse_move`, `type`, `key`, `scroll`,
  `wait`, `screenshot`, `task_complete`). Unsupported actions such as `left_click_drag` become a
  screenshot plus an error message the model sees on the next turn.
* **safety** runs the blocklist and the human-in-the-loop policy.
* **execute** performs the action with real input events at pixel coordinates.

Why a graph rather than a `while` loop? Because each of those four decisions needs to be
observable and swappable on its own. Every node logs its input and output state under the task's
`trace_id`; the eval harness disables the verifier by passing `verifier=None` instead of adding
`if` branches; and LangSmith tracing comes for free through the environment variables. There was
one lesson: LangGraph refuses a node whose name equals a state key. The original prototype
called the node `step` while the state had a `step` counter and crashed at build time. There is
now a test for that.

## A sandbox that runs in CI

The stack the spec asks for is an Ubuntu container with Xvfb, `scrot` and `xdotool`, viewed over
noVNC. I wrote that image (`docker/sandbox.Dockerfile`) and the driver (`XdoToolVM`), but I
could not validate it in the development environment (no Docker daemon), and even where Docker
is available, a desktop screenshot cannot tell you whether a form was submitted.

So the reproducible sandbox is different: a headless Chromium driven by Playwright over four
single-file web apps I wrote for the purpose, loaded from `file://` URLs with networking
disabled:

* a **forms portal** (contact, registration, profile with pre-filled fields, a five-question
  survey, a support request);
* a **legacy ERP** with a Windows-95 look: a Q3 sales spreadsheet, a contact list, invoices
  that open a detail panel, a file browser with a distractor folder, and a Help → About dialog;
* an **admin dashboard** behind a demo/demo login with users, detail pages, breadcrumbs, search
  and settings;
* a **back-office console** with orders that export to CSV, a file list, an e-mail composer with
  an attachment picker, an upload form, a downloads pane (download → unzip → open readme → copy a
  passage), a calendar and a notes pad with a virtual clipboard.

The driver clicks at coordinates with `page.mouse.click(x, y)`, types with `page.keyboard`, and
translates xdotool key syntax (`ctrl+a`, `Return`) into Playwright's. The screenshot is a real
1024×768 PNG, exactly what the model receives. This runs on any machine with Chromium, including
GitHub Actions, in about 100 ms per step.

## Ground truth is a checker on the DOM

Each of the 20 tasks (five per category: form filling, data extraction, web navigation,
multi-step workflow) carries a hand-written checker. The apps set attributes when something
really happens — `#contact-result[data-email="a@b.com"]` only exists after a valid submit — and
the checkers read them:

```json
{"type": "all", "checkers": [
  {"type": "dom_attr_equals", "selector": "#contact-result", "attr": "data-name", "value": "Alice"},
  {"type": "dom_attr_equals", "selector": "#contact-result", "attr": "data-email", "value": "a@b.com"}
]}
```

Extraction tasks check the agent's final answer instead: `answer_number` parses the first number
(thousands separators allowed) and compares with a tolerance; `answer_set` extracts every e-mail
or file name with a regex and requires the set to match exactly, so a missing or invented value
fails. Navigation tasks look at `location.hash` and at a `data-last-nav` attribute the app writes
with the id of the control that triggered the navigation. That last detail matters: "use the
breadcrumb to go back to Home" must fail when the agent uses the sidebar, and it does — there is
a test for it, along with one that types "Mallory" instead of "Alice" and expects a failure.
Checkers that cannot fail are not ground truth.

## The safety layer is boring on purpose

The temptation is to ask the model whether an action is safe. I did not, at least not as the
first line. The primary guard is a deterministic function that runs in microseconds, returns the
id of the rule that fired, and is tested per rule family:

* **literal substrings** typed into the screen: `rm -rf /`, `sudo `, `dd if=/dev/zero`,
  `> /dev/sda`, `mkfs.`, fork bombs, `curl | sh`, `wget | bash`, `cat /etc/shadow`;
* **regexes** for the things substrings miss: recursive `rm` with flags in any order, a download
  piped into any shell or interpreter (`curl … | sudo bash`, `wget … | python3`), writes to any
  block device, recursive `chmod`/`chown` on `/`, `kill -9 -1`, SQL `DROP`/`TRUNCATE`/`DELETE`
  without `WHERE`, `git push --force`;
* **key combos** that log out, switch TTY or open a terminal;
* a **URL allowlist**: anything with `://` must be `file://`, `about:` or an allowed host;
* a **human-in-the-loop policy**: payments, mass deletion, e-mail to external domains and
  financial credentials are flagged `requires_human`. Without an operator attached they are
  blocked; with one, executed after approval.

Because the guard is deterministic I could *measure* it. `data/eval/adversarial.json` holds 16
dangerous actions. The safety ablation runs each through the real loop on the sandbox, once with
the layer and once without: 16 blocked / 0 executed versus 0 blocked / 16 executed. The
"executed" ones only typed text into a web form — that is the other half of the design: the
agent can only act inside a browser page with no shell, no file system and no egress, so a
blocklist miss cannot escalate.

Benign look-alikes are tested too: "delete from the list the item number 3", "curl is a command
line tool" and `git push origin feature/login` all pass.

## The model client: timeouts, retries, cost

All LLM traffic goes through one `LLMClient` wrapping `AsyncAnthropic` with an explicit timeout
and retries owned by `tenacity` (exponential backoff on 429, 5xx, connection and timeout errors;
the SDK's own retries are disabled so nothing retries twice). The reasoner keeps the
conversation — `tool_use` blocks from the assistant, `tool_result` blocks with the new screenshot
from the user — and prunes images older than the last three so the context does not grow with
every step. Usage is converted to USD from a pricing table keyed by the pinned model id
(`claude-sonnet-4-5-20250929`); an unknown model yields zero and a warning, never a guess.

The tool definition is the one Anthropic documents for `computer_20250124`:

```python
tools=[{"type": "computer_20250124", "name": "computer",
        "display_width_px": 1024, "display_height_px": 768, "display_number": 1}],
betas=["computer-use-2025-01-24"]
```

Everything is tested against a mocked client: the click that becomes an `Action`, the
end-of-turn that becomes `task_complete`, the parallel tool calls of which only the first is
executed, the pruning, the retry that succeeds on the third attempt, the 400 that is not retried.
The real API is never called in the test suite.

## Results that say what they are

The DoD for this portfolio has a rule I have come to like: numbers in the README must come from
a run file in the repository, and any run made without the LLM must be labelled as a fallback,
never presented as the system's result.

`python -m eval.run` runs three suites and writes `eval/runs/<date>-<name>.json`, then renders
`eval/RESULTS.md` from those files only. The rows I can measure without a key use the
`StubReasoner`: it follows a scripted plan whose targets are CSS selectors, resolved against the
element map of the current screenshot into pixel clicks. It is a DOM oracle, and the table says
so in the first paragraph. It passes 20/20 in 6.5 steps and 0.65 s on average, which proves
exactly one thing: the sandbox, the executor, the checkers, the audit log and the safety layer
work end to end. The four Claude cells per category are `pendiente (requiere ANTHROPIC_API_KEY)`.
So are the recovery rate on real failures, the with/without-verifier ablation on real failures,
and the LLM-as-judge final validator, whose five criteria live as numbered code in
`src/eval/judge.py` rather than as "rate this from 1 to 10".

I find this more useful than a table of plausible-looking percentages. When the key arrives,
`python -m eval.run --with-claude` fills the rows and the error-analysis generator lists the ten
worst cases with their last three actions.

## What the audit log knows

Every run produces a `TaskResponse` with a `trace_id`, total latency, input and output tokens,
cost, the number of blocked actions and verifier failures, and one `Step` per iteration
(action, screenshot path, reasoning, execution result, safety verdict and rule, verifier note,
per-step latency and cost). The audit log stores it in SQLite by default and PostgreSQL through a
`psycopg` pool when `DATABASE_URL` is set; if PostgreSQL is unreachable the fallback is logged as
an error, not swallowed. `GET /api/audit/recent` returns the last hundred runs for a dashboard.

## What I would tell a reviewer

* The measured numbers are harness numbers. Model capability on these tasks is unknown until a
  keyed run.
* The sandbox apps are deterministic fixtures. They are representative of the UI patterns
  (forms, tables, dialogs, breadcrumbs, dependent steps) but simpler than OSWorld desktops.
* The safety layer checks each `type` action independently; a command split across several
  keystroke batches could evade the regexes. Reconstructing the keystroke buffer is next.
* The Docker path (sandbox image, compose stack, PostgreSQL) is written and mock-tested, not
  validated against a daemon.

## Numbers about the code

155 tests, 97 % line coverage on `src/`, `mypy --strict` clean, `ruff` and `black` clean,
`gitleaks` clean, CI running all of it plus the sandbox tests on a Playwright-installed Chromium.
Twenty tasks, sixteen adversarial probes, four fixture apps, ten decisions documented with the
alternative that lost.

If you work with systems that have no API and you are thinking about agents, start with the
checkers and the blocklist. The loop is the easy part.
