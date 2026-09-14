# Recorded installation evaluations — 2026-09-14

## App detector: one directive per source range

The detector now assesses the complete file before choosing exactly one source directive. For `needs-update`, it selects the first active obsolete invocation; for `installed`, the first active current invocation. A range spans multiple lines only when that single directive spans multiple lines. A mixed-current/obsolete example reinforces the selection rule. The parser and strict grader are unchanged; prompt hashing invalidates earlier cached verdicts automatically.

The final App eval passed **56/56 strict checks**: all original 50 cases with unchanged contents/labels, plus six source-selection regressions covering separated or adjacent directives, current/obsolete mixtures, revoked rules, and genuine multiline commands. The previously failing `two-obsolete-routes` returned `{"status":"needs-update","startLine":1,"endLine":1}`. Zero false installed answers, invalid source spans, JSON errors, or incomplete responses occurred.

| Check                                    |              Result |
| ---------------------------------------- | ------------------: |
| Strict verdict + source + service checks |               56/56 |
| Original cases / added regressions       |         50/50 / 6/6 |
| Memory and disk verdict-cache reuse      |          56/56 each |
| Content invalidation / forced recheck    |     Passed / passed |
| Model calls                              |                  55 |
| Model-call latency p50 / p95             |     12.1 s / 26.2 s |
| Observed peak physical footprint / RSS   | 47.8 GiB / 22.2 GiB |

This uses the same production App detector, local completion client and desktop sidecar as the initial run below, with the installed default Qwen3.8-27B MXFP4 model. All 55 HTTP calls confirmed medium thinking, the 16,384-token limit, nonzero reasoning tokens and 2,048-token prefill chunks. Calls used 12,135 reasoning tokens in total, with a maximum of 512 combined output tokens per response. The run was sequential, used fresh case state without conversation continuation, and stopped its sidecar on completion. No inference settings or allocator limits were changed.

An initial targeted probe of the original failure and six new cases, repeated twice, passed 14/14 before final full-suite validation. Raw reports are retained in `.cache/agent-installed-eval/app-single-source-probe/` and `app-single-source-full/`, with per-run prompt/source hashes. The earlier failed run remains below. These reused synthetic cases are regression evidence, not an unseen production-accuracy estimate. The App eval tests detection/file/cache behavior through the Node sidecar transport, not Electron UI/IPC, installation writes, or live coding-agent instruction loading.

## Initial App detector: medium thinking, 16,384 output tokens

Actual `CodingAgentsService` → `localCompletion` → built desktop inference sidecar, supervised through the production Node fallback transport. Installed default `qwen3.8-27b-mxfp4-mlx`, Apple M5 Max with 128 GiB. The App policy uses medium thinking, a 16,384-token combined reasoning/output limit, temperature zero, and a ten-minute timeout. No allocator/cache overrides. The detector prompt and fixed labels were unchanged from the CLI evaluation below.

| Check                                    |              Result |
| ---------------------------------------- | ------------------: |
| Strict verdict + source + service checks |               49/50 |
| Installation verdict alone               |               50/50 |
| False installed / runtime or JSON errors |               0 / 0 |
| Memory and disk verdict-cache reuse      |          50/50 each |
| Content invalidation / forced recheck    |     Passed / passed |
| Model calls                              |                  49 |
| Model-call latency p50 / p95             |     12.4 s / 25.5 s |
| Observed peak physical footprint / RSS   | 45.5 GiB / 22.1 GiB |

The one strict failure is `two-obsolete-routes`. Both lines contain separate obsolete directives. The model correctly returned `needs-update`, but selected lines 1–2 instead of the smallest complete directive, line 1 or line 2. The original oracle rejects that broader span; this historical result remains 49/50. The subsequent prompt fix and rerun are reported separately above; the case and its accepted ranges were not relaxed.

All 49 HTTP requests completed with medium thinking, the 16,384-token cap, nonzero reasoning tokens, and the App's 2,048-token prefill chunks confirmed in usage. They generated 10,326 reasoning tokens in total; the largest response used 661 combined output tokens. Three empty/missing fixtures skipped inference, and the generated-prompt case added two calls for content invalidation and forced rechecking. Unchanged memory/disk cache checks made no additional model calls.

The preceding three-case probe passed 3/3, including the formerly troublesome unlabelled fenced example and long document. An initial startup-only attempt ran zero cases because the parent oxnode loader leaked into the Node sidecar under its native-addon override. The Node transport now starts built JavaScript without inherited `execArgv`; a real-process regression test covers that boundary. The Electron utility-process transport is unchanged.

Runs were sequential and the eval sidecar was stopped afterward. The App measurements include lazy loading, normal HTTP session/cache behavior, temporary-file/cache operations, and external memory observation. They are not a matched performance comparison with the CLI runs. Only the detector/file/cache path is exercised: no Electron UI/IPC, real global instructions, installation writes, or live coding-agent instruction loading. The 50 synthetic cases include reused holdout-labelled fixtures and are regression coverage, not an unseen accuracy estimate.

Local raw reports: `.cache/agent-installed-eval/app-medium-16k-probe/` and `app-medium-16k-full/`, including exact fixtures, prompts/answers, unmodified HTTP bodies without credentials, source hashes, sidecar traces, timing and memory. The startup-only failure is retained in `app-medium-probe/`. Reproduce with `vp exec oxnode scripts/eval-agent-installed.ts --entrypoint app` after `vp run build:ts`.

## Earlier CLI runs

Apple M5 Max, 128 GiB memory. Installed default `qwen3.8-27b-mxfp4-mlx`, saved thinking level `high`. Actual built CLI entry points in RPC mode; no model, inference, cache, system-prompt, tool, permission, or environment overrides. Runs were sequential.

| Entry point                  | Strict passes | Task p50 | Task p95 | Observed peak physical footprint |
| ---------------------------- | ------------: | -------: | -------: | -------------------------------: |
| mlx delegate — full corpus   |         50/50 |   17.4 s |   38.8 s |                         32.6 GiB |
| mlx agent — three-case probe |           2/3 |   35.0 s |   51.1 s |                         34.0 GiB |

The delegate run completed all 50 fixed cases (30 development, 20 holdout-labelled), with zero false installed answers and zero runtime errors. Its normal metrics trace contains 100 inference turns and 106,399 reused prompt tokens. All cases used the read tool, preserved evidence, and saved normal sessions.

The agent probe covered `generated-prompt`, `unlabelled-fence`, and `old-path`. The fenced case returned the correct not-installed verdict but prefixed the JSON with explanatory prose. Strict parsing rejected it; it remains a failed case. The eval does not strip arbitrary prose to turn that failure into a pass.

An earlier real-CLI probe scored 1/3 and exposed a runtime bug: after a new session, the resident ChatSession rejected a changed cacheOwnerId. The fix rotates and disposes the old ChatSession on the serialized host while retaining model weights. The repeated-session probe then passed 3/3 before the full delegate run. Focused validation passed 147 tests, TypeScript build, and type-aware lint.

These numbers include tools, thinking, lazy loading in the first task, and natural prefix caching. They are not cold-prefill or model-comparison benchmarks. Memory was observed externally; RSS and physical footprint differ, and brief peaks may be missed. The former custom HTTP run was interrupted after excessive memory and is not included in these CLI scores.

The rubric is user task content under each CLI profile’s normal system prompt. The dashboard HTTP detector uses a different completion policy; these results do not establish its live accuracy or prove that Codex/Claude loaded their global instructions. File selection and verdict-cache behavior have separate service unit tests. The finite synthetic corpus, including the reused holdout-labelled cases, is regression evidence rather than an unseen production-accuracy estimate.

Local raw reports: `.cache/agent-installed-eval/cli-delegate-probe/` (initial failure), `cli-delegate-fixed-probe/`, `cli-delegate-full/`, and `cli-agent-probe/`. Each retains invocation/source hashes, exact tasks and answers, saved sessions, events, metrics, and memory observations. See [README.md](README.md) to reproduce.
