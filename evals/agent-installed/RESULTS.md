# Recorded CLI evaluation — 2026-09-14

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
