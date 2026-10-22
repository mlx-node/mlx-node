# Rerank validation — Apple Silicon

Implementation branch: `feat/rerank`, based on `6ccf98fe8ed2a27a5883a1468ccaea24892b3aad`.
Machine: Mac16,12, 32 GiB unified memory, macOS 27.0.1, Node 24.22.0.
Reference: Python MLX 0.32.4 / MLX-LM 0.31.3; exact repository revisions and thresholds are in [manifest.json](../../scripts/rerank/manifest.json).

## Numerical gates

All six version/format combinations passed the frozen same-checkpoint score gate across six cases. Two additional community layout probes also passed. These compare native segmented execution against full-sequence Python MLX on the **same weights**, not quantized output against BF16.

| Checkpoint                                                | Maximum score error | Worst case mean error | Separated-pair inversions |
| --------------------------------------------------------- | ------------------: | --------------------: | ------------------------: |
| v3 BF16                                                   |            0.002469 |              0.000795 |                         0 |
| v3.5 BF16                                                 |            0.002816 |              0.000339 |                         0 |
| v3 affine 4-bit                                           |            0.002360 |              0.000423 |                         0 |
| v3 MXFP4                                                  |            0.002734 |              0.000384 |                         0 |
| v3.5 affine 4-bit                                         |            0.002041 |              0.000695 |                         0 |
| v3.5 MXFP4                                                |            0.001613 |              0.000373 |                         0 |
| Community v3 MXFP4, quantized merged projector            |         0.000000462 |           0.000000169 |                         0 |
| Community v3.5 affine, dense embedding/separate projector |            0.001704 |              0.000462 |                         0 |

Limits remain max ≤ 0.01, per-case mean ≤ 0.001, and no inversions when the reference score separation exceeds 0.02. Exact tokens, marker positions, block counts and token totals matched. Projected vectors passed a separate relative-L2 diagnostic (< 0.03); block fusion weights differed by at most 0.005. The final score limits were never relaxed.

Coverage includes English/Chinese, reserved markers, empty/duplicate candidates, sliding windows, weighted fusion, 125/126 boundaries, sharded checkpoints and separate/merged/quantized projectors. Hermetic native fixtures use different marker IDs and model dimensions and compare full prefill with chunks smaller/larger than the sliding window. Fault tests cover missing/incorrect weights, conflicting profile metadata, zero projector norms, conversion round trips, truncation, original indices, cancellation and resource disposal.

## Performance and allocations

These measurements use a release build with the reranker cache-budget registration. Model loads create fresh instances with a potentially warm filesystem cache. Ordinary desktop applications remained open, but no compiler or second GPU validation process ran during this final benchmark series. These are functional hardware measurements, not isolated benchmark claims or latency guarantees.

Short passages contain approximately 50 input tokens each, plus prompt/query overhead.

| Checkpoint | Load (s) | 50 docs (s) | 125 docs (s) | 200 docs (s) | Peak active (GiB) | Cancel-to-stop (ms) |
| ---------- | -------: | ----------: | -----------: | -----------: | ----------------: | ------------------: |
| v3         |     0.21 |        3.77 |        14.15 |        20.72 |              2.06 |               13.15 |
| v35        |     0.22 |        3.81 |        10.60 |        16.41 |              1.65 |               10.97 |
| v3-affine  |     0.15 |        3.86 |        14.45 |        20.81 |              1.31 |               18.62 |
| v3-mxfp4   |     0.15 |        3.99 |        15.13 |        20.25 |              1.29 |               14.83 |
| v35-affine |     0.15 |        4.00 |        11.08 |        19.03 |              0.86 |                9.87 |
| v35-mxfp4  |     0.16 |        3.73 |        11.64 |        21.10 |              0.84 |                1.36 |

Long passages use approximately 270 input tokens each on v3.5 MXFP4, with the default 32,768-token block budget and 512-token prefill chunks:

| Candidates | Prompt tokens | Blocks | Latency (s) | Peak active (GiB) | Cached after request (GiB) |
| ---------- | ------------: | -----: | ----------: | ----------------: | -------------------------: |
| 50         |         14438 |      1 |       19.15 |              1.13 |                      10.50 |
| 125        |         36051 |      2 |       89.89 |              2.04 |                      16.66 |
| 200        |         57501 |      2 |      163.02 |              2.04 |                      16.71 |

The long-input test initially exposed missing registration with the existing process-wide MLX cache-budget coordinator. The final implementation registers deterministic packed weight bytes through a model-owned guard, then unregisters on disposal. The table above was rerun after that fix. The retained allocator cache is substantial and distinct from live model resources; existing cache/headroom configuration and the host idle sweeper remain applicable.

Packed weights remained packed: steady active model allocations were approximately 337 MB for affine and 318 MB for MXFP4, compared with 1.19 GB for BF16. Six repeated requests in every benchmark retained identical active byte counts. Explicit disposal returned active allocations to zero in all six formats and the long-input run. Long-input cancellation, requested after one second, stopped in 17.14 ms. These times measure the tested safe cancellation boundary, not a universal upper bound. Cached bytes can remain after disposal.

### Chat co-residency

The real HTTP host ran Qwen3-0.6B BF16 with v3.5 MXFP4. Chat used deterministic eight-token responses; every response matched across the three controls. Each warm control has three samples:

| Control                  | Median chat latency (ms) |
| ------------------------ | -----------------------: |
| Chat alone               |                   126.17 |
| Chat with idle reranker  |                   126.12 |
| Simultaneous chat/rerank |                  1270.65 |

Resident active allocations rose from 1.19 GB (chat) to 1.51 GB (both models); concurrent peak active allocations were 1.84 GB. Simultaneous rerank requests took 4.08–4.63 seconds. The idle control showed no measurable median change in this small sample; concurrent inference clearly competed for GPU resources. This is a within-build control, not a statistical before/after benchmark against `main` or a claim of zero overhead.

The complete rows, pinned sources and allocator counters are in [rerank-results.json](rerank-results.json). Peak active memory excludes allocator cache, private paged-KV Metal pools, driver allocations and other processes; it must not be interpreted as total machine memory use.

## Regression checks

- Release native build, packaging smoke checks, `tsc -b`, repository lint, `cargo fmt --all -- --check` and workspace/all-target clippy with `-D warnings` passed.
- Pure TypeScript lane: 2,112 tests passed across 107 files.
- Addon lane: 1,743 tests passed; 57 tests skipped across the suite's conditional/fixture gates. The new family required updating the agent's explicit non-generative family expectation; the complete lane passed after that update.
- Rerank native/SDK/HTTP/host suite: 33 tests passed. This includes lifecycle faults and native-free imports; these counts overlap the two full TypeScript lanes.
- Rust core unit suite: 3,983 passed, 160 ignored. After the final signed-zero comparator change, the affected Jina module was rerun: 8 passed, 1 golden test ignored by default. That golden test was explicitly executed separately on all eight checkpoints. Exact ties, including `+0`/`-0`, retain original index order.
- Workspace Rust integration: 89 test binaries, 278 passed, 153 ignored. Other crate unit tests: 224 passed, 8 ignored. Workspace doc tests: 5 passed, 9 ignored.
- Real Qwen3 flat and paged mid-prefill cancellation: both passed; the next paged request reported zero cached tokens, proving cancelled partial state was not reused.
- Real CLI inference passed JSON-only stdout, original index/document and effective block-policy metadata checks; stderr was empty.
- All eight pinned checkpoints passed scores and intermediate-feature comparisons after the allocator-budget fix. Final scores were rerun after the sorting-only change; the inference path measured in the performance tables is unchanged.

Full local logs are retained under `.cache/rerank-validation/logs/` (ignored by Git); commands and reference setup are in [the validation procedure](../../scripts/rerank/README.md).

The pre-change baseline recorded a clean `main` at the stated commit and a passing TypeScript build. This report does not claim a complete archived pre-change GPU performance baseline. Skipped/ignored tests are not counted as passing model validations; no full real-weight ASR/TTS matrix was run for this rerank change.

## Review follow-up — 2026-10-21

Independent pair review found three issues, now corrected: `loadSession()` rejects rerank families before native loading; a known model evicted during cold-load switching returns retryable `503/model_unavailable` (unknown models still return 404); reserved-marker sanitization removes nested or newly joined markers in linear time and reapplies the invariant after truncation decoding. Ordinary text and returned original documents are preserved.

Five added regression cases failed on the reviewed implementation before the fixes. After rebuilding the release addon, the focused rerank/loader/service suite passed all 66 tests. The broader host/service/loader selection passed 200 tests; these counts overlap. Jina Rust unit tests passed 10 tests, with the optional golden test excluded from that ordinary run. Its exact-token, marker-position and intermediate-feature checks were then explicitly rerun and passed on v3 BF16 and v3.5 MXFP4.

All eight pinned checkpoints passed the six-case final-score comparison again (48 cases), using unchanged reference files and frozen thresholds. Maximum score error was 0.00281585, worst per-case mean error was 0.00079451, with no separated-pair inversions. Real HTTP tests on v3 BF16 and v3.5 MXFP4 returned 200 for nested markers and matched clean-input scores, token counts and truncation metadata; marker-only queries returned 400. Original document text remained intact, and `loadSession()` rejected both real checkpoints before loading.

Release build/packaging, TypeScript build, lint, formatting, workspace/all-target clippy with `-D warnings`, and `git diff --check` passed. The independent reviewer rechecked the patches without making edits and found no further blocking issues. This follow-up reran affected regression and numerical checks, not the full historical suite or performance benchmarks above. Follow-up score rows are recorded under `reviewFollowup` in [rerank-results.json](rerank-results.json); local logs use the `mlx-rerank-review-` prefix under `.cache/rerank-validation/logs/`.

## Compatibility and operational scope

Chat's transformer fast path and ordinary Qwen3 quantization rejection are unchanged. Native-free imports, architecture-based discovery, training-family exclusion, strict rerank model resolution, HTTP limits/auth/CORS, cold-load admission transfer, cancelled waiters, native cancellation, disconnect, failed load retry, shutdown/load races and capability/single residency are covered by regression tests.

The SDK and server retain original candidate order when constructing blocks. Changing token/document block budgets can change scores; this is intentional ranking configuration and is reported in response metadata. MLX allocator caching and simultaneous GPU use can increase memory/latency. An HTTP deadline requests cancellation and waits for native completion before releasing its lease.

Single-mode capability switching removes chat bindings and drains chat sessions through the existing lifecycle. Physical chat model allocations can remain until JavaScript GC releases the model handle; immediate native chat deallocation is not established by this change. Rerank and audio expose explicit disposal. Four-capability residency tests validate admission and logical bindings; only chat + rerank co-residency uses real weights in this report.

The feature remains unmerged and unpublished. Official Jina model licensing remains CC BY-NC 4.0; the runtime license does not grant commercial rights to those model weights. Arbitrary community Q4 formats, fine-tuning, embedding APIs and Hindsight-specific changes are outside this release.
