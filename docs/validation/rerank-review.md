# Jina and BGE increment review

Review date: 2026-10-21. Reviewed branch: `feat/rerank`, through
`8b47712e6714aa133e35a2cb510149c535947d3d`, plus the corrective changes described
below. Fetching upstream found `origin/main` at `1e21e9c5`; the common ancestor
remains `6ccf98fe8ed2a27a5883a1468ccaea24892b3aad`. The reviewed increment is all
23 branch commits after that ancestor (108 files before these corrections).
The three newer upstream dependency commits were not merged.

Corrective commits: `30d8062b` (checkpoint contracts), `0bd4d3cb` (chat selection
after eviction), `2913abcc` (SDK ownership/startup limits), and `cd71e899`
(validation task/report lifecycle).

The primary review covered SDK, service admission, host residency, CLI, contracts
and validation evidence. A fresh, read-only pair reviewer independently examined
native backends, conversion, runtime ownership, evidence gates and host alias
behavior. A separate implementation worker applied native fixes; that worker was
not the independent reviewer. The pair reviewer inspected the actual corrective
diff again, including issues discovered during confirmation. Neither code review
nor a passing test suite establishes an absolute absence of defects.

## Commit coverage

Each row identifies the contract examined in the commit and its final state.
This is a joint review ledger, not a claim that each reviewer read every file.

| Commit     | Review focus and conclusion                                                                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `08e01992` | Jina profiles, attention, packed weights, token/block planning, conversion. Found incomplete checkpoint and converter validation; corrected below.                                                           |
| `f3736fd2` | SDK cancellation, native-free catalog, family discovery and chat/training exclusion. Fixed mutable request-option lifetime handling.                                                                         |
| `7cc7a907` | HTTP validation, writer/reader permits, atomic cold admission transfer, FIFO identity, shutdown, logging and capability residency. Found and corrected chat alias selection after eviction and timer bounds. |
| `1644ae80` | CLI dispatch, input normalization, download attribution and serve configuration. Model-specific inference remains below CLI; startup now also validates service limits before host mutations.                |
| `1fc3baee` | Jina reference cases, frozen thresholds, numerical and coexistence evidence. Fixed direct-native task ownership and stale output reports in legacy scripts.                                                  |
| `63a0f9b1` | Shared native worker/task cancellation and joined disposal. Lifecycle-only abstraction retained; inference mathematics remain in each backend.                                                               |
| `433548fe` | XLM-R pair encoding, pad masks/positions, post-norm encoder, classifier, stable sigmoid, batching and packed weights. Rejected malformed module overrides; no scoring change.                                |
| `3ec47e81` | BGE recipe exclusions, dtype/provenance and independent output. Added source dtype/compute-metadata checks before generic conversion casts.                                                                  |
| `284c5359` | Family descriptors, discriminated metadata, exact capability rejection, family policies and CLI detection. Added the native BGE integer-range constraint to startup validation.                              |
| `5de8c9b5` | SDK native task retention through completion. Correct in SDK; extended the same ownership rule to the older Jina scripts.                                                                                    |
| `d9515101` | Independent Transformers/MLX references, fixed revisions, native fixtures, quality/performance/surface tooling. Thresholds and sigmoid-only public contract retained.                                        |
| `ef5a9bfb` | Support matrix, family policy migration, scoring and licensing documentation. No unearned overall acceptance claim added.                                                                                    |
| `343d3790` | Full-source regression identity and duplicate query-text recovery. Sample-index identity preserved.                                                                                                          |
| `a1b1a26f` | Unsupported BGE quantization on dense modules. Extended validation to empty/scalar module declarations and aligned Jina declarations.                                                                        |
| `cf0a0b18` | Early throughput gate and incomplete-report state. Failed precheck remains a merge blocker; no threshold relaxation.                                                                                         |
| `42187886` | Verified FP32 baseline, complete rankings, dataset identity and independent metric recomputation. Negative contract tests retained.                                                                          |
| `778f5537` | Frozen script, Node and execution-environment identity. Historical quality artifacts remain bound to their original binary and scripts.                                                                      |
| `b8dbaa65` | Native-free dependency paths and real SDK/HTTP/CLI harness loading. No production family fallback added.                                                                                                     |
| `3d2d4dd9` | Forward-allocation cancellation and joined active/queued shutdown. Allocation observation is not described as proof of an executing GPU kernel.                                                              |
| `329c03c6` | Preserved regression attempts and active-resource baseline equality. Initial failure and later passing repeat remain separate evidence.                                                                      |
| `cbef5076` | Failed BGE comparison invalidates stale passing output. Applied equivalent fresh-run behavior to Jina scripts.                                                                                               |
| `413f3829` | Intermediate numerical/resource evidence and explicit unresolved performance gates. Checked against harness behavior and later evidence.                                                                     |
| `8b47712e` | Final fixed-subset quality results and acceptance limitations. Passing nDCG gates do not imply full MTEB acceptance or close the failed performance gates.                                                   |

## Findings and corrections

1. **P2 — incomplete Jina tensor contract during conversion.** Missing, incorrectly
   shaped, bias-bearing or unexpected tensors could survive conversion even when
   the result could not load. Loading and the conversion recipe now share
   `Weights::validate_config`; source storage is checked before dtype casts.
   Unused causal-LM vocabulary-head tensors remain explicitly excluded.
2. **P2 — Jina index/header ambiguity.** Canonicalization previously lost original
   tensor-to-shard identity and collapsed aliases. Loading now verifies exact raw
   index membership, rejects omitted/misplaced/duplicate tensors and canonical
   alias collisions, then normalizes names. Standalone projector files outside
   the main index remain valid.
3. **P2 — ignored quantization declarations.** Dense Jina modules could declare
   unsupported formats or unknown modules; both backends ignored some empty or
   scalar module overrides. Declarations now fail independently of whether a
   tensor currently has packed sidecars. Unrelated nested quantization metadata
   continues to follow the existing shared parser contract.
4. **P2 — BGE conversion could conceal invalid source dtype.** Generic casting
   could turn an invalid integer weight into an apparently valid floating weight
   before recipe validation. BGE validates source tensors and compute metadata
   before casts. New fixtures cover integer weights, positions and boolean bias.
5. **P2 — chat alias regression after cross-capability eviction.** In `single`
   mode, `chat A → chat B → ASR/rerank → unknown chat alias` selected A after
   clearing the resident. Selection is now separate from resident ownership;
   aliases reload B without retaining its native allocation. A direct before/after
   reproduction returned A before the fix and B afterward.
6. **P2 — SDK request mutation could change cancellation ownership.** Reusing a
   mutable request object while awaiting inference changed the signal used for
   error reporting and listener cleanup. Signal and returned-document preference
   are now captured before awaiting. Tests cover original and replacement aborts
   and removal of the original listener.
7. **P2 — Jina validation task owners could be collected while awaiting.** Legacy
   direct-native scripts now retain each task through completion using the same
   `finally` ownership pattern as the SDK/BGE tools. They still await native work.
8. **P2 — failed Jina reruns retained stale successful reports.** Fresh comparison
   and benchmark runs now invalidate the requested output before reading inputs
   or loading models. Comparison first rejects an output that is its reference.
   These scripts have no resume semantics.
9. **P3 — configuration bounds failed too late.** BGE's seven policy limits now
   enforce the backend's positive signed-32-bit range during native-free startup
   validation. HTTP deadlines exceeding Node's timer range are rejected instead
   of becoming 1 ms timers. Host service validation precedes discovery and engine
   policy mutations. Context-dependent policy constraints remain native-owned.

The original binary reproduced all 16 initial Jina negative cases. Additional
BGE probes reproduced four malformed-override cases and an I32 conversion case;
the BOOL/I16 probes were already rejected by the old lower-level reader. No
baseline passing behavior was described as a newly fixed failure.

## Architecture review

- Keep the shared runtime limited to task ownership, serialization, cancellation,
  synchronization and disposal. Jina listwise attention/fusion and XLM-R global
  bidirectional pair encoding have different semantics and remain independent.
- Share the strict checkpoint metadata boundary: `rerank_checkpoint` owns unique
  JSON keys and safetensors header validation. Adapters retain raw index mapping,
  canonical names, projector handling, permitted tensors and shape contracts.
  This removes duplicated safety logic without imposing BGE's uniform dense
  dtype rule on Jina's valid BF16 backbone/F32 projector.
- Reuse each backend's tensor validator for loading and conversion. It inspects
  storage metadata and logical shapes; it does not instantiate random modules,
  materialize ignored tensors or dequantize the full packed embedding table.
- Separate user selection from resource residency. Keep the host coordinator
  responsible for cross-capability eviction under the existing exclusive permit;
  capability adapters retain their own loaders and registries. The generic
  admission/FIFO machinery remains reused rather than replaced.
- Keep tokenization, truncation and scoring below SDK/server/CLI. Structural
  architecture names, tensor mappings, trained prompts and protocol token counts
  are model contracts, not directory-name heuristics. The production implementation
  still has no checkpoint repository-name dispatch or Hindsight dependency.
- Do not combine QKV, defer all layer evaluation or alter the chat transformer on
  the strength of a static performance hypothesis. Such changes must first show
  a native profile benefit and preserve numerical/cancellation boundaries. This
  review makes no new throughput claim and does not relax acceptance thresholds.

## Verification

Results for the corrective implementation on the same Apple M4 / 32 GiB machine:

| Check                                                      | Result                                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| TypeScript build, repository lint, touched-file formatting | Pass; lint has no remaining warnings                                                |
| Complete pure-TS lane, one worker                          | 2,126 passed in 108 files                                                           |
| Native release build and both Metal packaging smoke checks | Pass                                                                                |
| Jina/BGE native fixtures, including new failure cases      | 64 passed                                                                           |
| Complete addon lane, one worker                            | 1,801 passed, 57 skipped; 108 passed / 23 skipped files                             |
| Rust fmt; workspace/all-target clippy with `-D warnings`   | Pass                                                                                |
| Complete Rust core unit suite, one thread                  | **3,996 passed, 1 failed, 161 ignored**; failure described below                    |
| Eight Jina checkpoint feature/score goldens                | Pass against unchanged references                                                   |
| Six BGE checkpoint logit/stage/score goldens               | Pass; FP32, FP16, BF16, affine, MXFP4 and mixed weights, frozen five-case input set |
| Jina direct-native compare under forced V8 GC              | Pass; v3.5 MXFP4, all six cases, 1,083 GC cycles                                    |
| Real BGE SDK/HTTP/CLI surface comparison                   | Pass                                                                                |
| Evidence-contract and failed-report lifecycle tests        | 14 passed                                                                           |

The addon suite emitted nonfatal process-listener and circular-module warnings;
the raw log preserves them. They were not suppressed or described as lint errors.
The full Rust failure was
`models::qwen3_5::attention::tests::segmented_block_crossover_is_calibrated_and_routes`:
its calibration took **133.280 ms**, exceeding the existing **100 ms** bound.
An isolated rerun took **147.897 ms** and also failed. The threshold, test and C++
calibration implementation were not modified. Static independent review found no
rerank initialization/resource path exercised by that isolated test; the cause is
still undetermined. Historical passing runs do not substitute for a pass in this
audit, and this report does not claim that the complete Rust suite passed.

Raw logs, the alias before/after reproduction, new GC/surface outputs and the
sequential runner are in `.cache/rerank-review/`. The compact
[review evidence](rerank-review-results.json) hashes the corrective source files,
rebuilt addon, Metal libraries and test logs. No simultaneous GPU test or build
ran during the numerical checks. The long-input BGE, fixed-subset quality and
performance reports from the previous implementation remain historical evidence
bound to their original artifact; those expensive matrices were not rerun or
silently relabeled as results for the new binary.

## Remaining acceptance limits

The pre-existing FP16 batching gate remains failed: approximately `+0.085%` and
`−1.008%` throughput changes for the required 125/200-document workloads, against
a `≥20%` target. The rest of that performance matrix is incomplete. The first
Jina regression attempt failed, and the unchanged-source repeat passed; its cause
has not been established. These performance limits remain explicit merge blockers
in [BGE validation](bge-rerank.md). The current Rust calibration failure above is
an additional unresolved verification result in this audit. This audit does not mark the branch ready to merge,
promise zero risk, merge upstream, publish a release or alter the user's existing
uncommitted changes to `docs/validation/rerank.md`.
