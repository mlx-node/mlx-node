# Agent installation eval

## App installed check

```sh
vp run build:ts
vp exec oxnode scripts/eval-agent-installed.ts --entrypoint app
```

This runs the actual `CodingAgentsService` and `localCompletion` against the built desktop inference sidecar. It uses the production supervisor, desktop model-directory setting, installed default local model, and launcher engine policy (2,048-token prefill chunks unless the inherited environment overrides it). The supervisor uses its existing Node child transport rather than Electron's utility process. The harness does not construct a custom inference host or tune allocator/cache settings.

The detector rubric is the **system prompt**. Requests use the App's shared setup policy: medium thinking, a 16,384-token combined reasoning/output limit, temperature zero, and a ten-minute timeout. Classification and command selection during an update share this policy. Verdict cache keys include the generation policy, so results from the former no-thinking/768-token check are invalidated.

Cases use disposable Claude/Codex/Grok instruction files and executable paths. The real service selects the file, checks the model response with the normal strict parser, and reuses unchanged verdicts both in memory and after a service restart. The generated-prompt case also checks content invalidation and forced rechecks. Empty/missing files must skip inference. No real global instructions or settings are modified. This mode tests detection and verdict caching, not installation writes, Electron UI/IPC, or whether another coding agent loaded its instructions.

App reports include `summary.json`, `samples.jsonl`, `cases.json`, source hashes and policies in `metadata.json`, discovered models in `runtime.json`, unmodified HTTP bodies in `http.jsonl` (without credential headers), sidecar diagnostics/traces, and external memory observations. Calls run sequentially with one resident model, lazy loading and the normal HTTP session/cache lifecycle. Interrupted or failed runs retain evidence and cannot report success.

Start with `--entrypoint app --case generated-prompt --case unlabelled-fence --case long-document` before the full corpus. The other selection/repetition options below also apply.

## CLI classification

Run installation-classification tasks through the **real `mlx agent` or `mlx delegate` CLI**. The runner only sends tasks, observes events, and grades results. It never creates an inference host or sets model, thinking, output, sampling, cache, system-prompt, tool, or permission overrides.

```sh
vp run build:ts
vp exec oxnode scripts/eval-agent-installed.ts --entrypoint delegate
vp exec oxnode scripts/eval-agent-installed.ts --entrypoint agent
```

The subprocess invokes this checkout's built public CLI (`node packages/cli/dist/cli.js agent|delegate --mode rpc`). RPC changes the input/output transport; production defaults and saved local settings select the model. Sessions persist in the normal app store. One process runs at a time, with a new session per case and the normal cache lifecycle. Missing models fail through the CLI's normal startup. No cloud inference is allowed by the production agent.

Start with a small probe:

```sh
vp exec oxnode scripts/eval-agent-installed.ts --case generated-prompt --case unlabelled-fence --case long-document
```

Use `--list`, `--case ID`, `--split development|holdout`, `--repeat 1..20`, `--seed VALUE`, and `--output NEW_DIR`. Existing output directories are rejected. Select a different installed default in `mlx agent` before comparing models; the eval does not substitute one.

## What is tested

Each task asks the agent to read a synthetic evidence file and apply the production detector rubric. The rubric is **user task content**, while the CLI's normal system prompt and tools remain active. The model must use the actual read tool, classify correctly, cite the labelled source lines, preserve the evidence, avoid other tools/tool errors, and persist its real session. Expected labels are never included in the task.

The 56 hand-labelled cases cover active and obsolete commands, missing caller approval, mixed rules, shell quoting, multiline commands, examples, revocation, prompt injection, Unicode, long documents, and selected Claude/Codex/Grok instruction contents. The original 50 cases and their labels are unchanged; six source-selection regressions require one directive when several qualify, including genuine multiline directives. Both the status and exact accepted source span must match. Invalid output, tool errors, timeouts, and incomplete turns fail. False installed answers are counted separately.

This CLI eval does **not** run the dashboard's HTTP completion backend, which has a different prompt placement and generation policy; use `--entrypoint app` for that path. It also does not prove Codex/Claude loaded or obeyed a global instruction. Service unit tests use mocked completions for fast regressions; do not present those unit tests as live model accuracy.

Fixtures contain no real user instructions. The user agent configuration remains loaded normally; the disposable evidence directory sits in this checkout, so `mlx agent` retains its ordinary project-context discovery. Delegate retains its normal no-context-files behavior. The eval changes no global instruction or settings file. Permission dialogs are cancelled; no new approval is granted by the harness.

## Evidence and measurements

Reports under `.cache/agent-installed-eval/` contain:

- `report.md` / `summary.json`: scores, failed checks, latency, and memory.
- `samples.jsonl`: task, raw answer, source range, checks, selected model/thinking, real session path.
- `events.jsonl` / `stderr.log`: CLI lifecycle events, final messages, tool calls/results, and diagnostics. Repeated per-token partial updates are discarded to avoid quadratic memory/disk growth; final contents remain in the saved session.
- `metrics.jsonl`: the CLI's normal per-turn inference metrics, copied from its process-specific trace. If unavailable, the source path and error are recorded in `metrics-status.jsonl`.
- `sessions/`: copies of the real saved sessions, including failed turns.
- `metadata.json`, `initial-state.json`, `cases.json`: exact invocation, settings reported by the CLI, source hashes, and fixed corpus.
- `memory.jsonl`: external RSS and macOS physical-footprint observations every five seconds. No native addon is loaded by the observer and no allocator limit is changed. Unavailable observations are recorded. Sampling can miss brief peaks; memory observation itself adds overhead.

The first task includes lazy model loading. Later tasks include tools, thinking, and natural prefix-cache behavior; these are task latencies, not cold-prefill benchmarks. No multiple-model runs execute in parallel. Startup is bounded to one minute and each task to five minutes; interruption preserves partial evidence and cannot report success.

Exit 0 means all selected cases passed; 1 means a complete run with failures; 2 means preflight failed or the run is incomplete. Live runs are opt-in, not normal CI tests.

The development/holdout split and labels are fixed before live runs. Keep failures and earlier reports; never relabel or remove difficult cases. Once a holdout failure informs tuning it becomes regression coverage rather than unseen validation. Passing this finite synthetic corpus does not guarantee correctness on arbitrary Markdown.
