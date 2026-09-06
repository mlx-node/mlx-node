# Inference architecture research and execution

Date: 2026-09-05. Audience: mlx-node maintainers. Baseline: `100a03ad`.
Worktree: `.worktrees/inference-control-2026-09-05`.

The task is to reduce recurring model-integration code, explain speculative
decoding with paged KV and concurrent requests, and improve measured inference
control-flow costs on Apple unified memory. Preserve bounded resident KV pools,
SSD cold persistence, exact committed frontiers, request isolation, cancellation,
and per-family tensor/cache ownership. No new dependencies. Refactor size and API compatibility are not constraints;
preserve inference correctness and request/cache ownership.

Sources: supplied local checkouts pinned after remote refresh; primary vLLM,
Apple Metal, MLX and reference-project documentation; local tests and alternating
performance measurements. Distinguish source facts, proposals and measurements.

1. **Complete — discovery and follow-up.** Audit current registration,
   scheduler, sampling, speculative transactions and SSD lifecycle. Compare vLLM
   MTP/DFlash/DSpark and MTPLX/oMLX/mlx-vlm. Resolve hardware/API constraints and
   record source provenance and remaining gaps.
2. **Complete — synthesis.** Produce the architecture comparison and choose changes
   justified by the audit. Record deferred changes with explicit prerequisites.
3. **Complete — structural implementation.** Replace per-family command envelopes and adapters with a shared generic type
   and default barrier dispatch;
   reduce avoidable GPU completion waits in shared inference paths. Keep larger
   speculative scheduling or Metal backend migrations behind demonstrated need.
4. **Complete — verify the structural revision and update PR #138.** Native build, Rust/TS checks,
   sampler and real-model parity gates, paired performance measurements, source
   links and final diff verified. Results and limitations are recorded in
   [validation.md](validation.md). The selected changes are ready for review;
   the larger staged migrations remain proposed work.

5. **Complete — full inference transfer audit.** After the structural revision is
   validated and pushed, inspect prefill, decode, speculative verification and SSD
   restore across Rust, C++ and MLX. Inventory CPU/GPU copies, host readbacks and
   synchronization separately; measure and remove confirmed avoidable overhead.
   The [audit](transfer-audit.md) records five implemented fixes, required
   boundaries, fallback costs and source/measurement limits. Prefix reuse,
   concurrent inference, sampled drafting and cache publication were validated.

Success requires a reviewable implementation plus cited research and a concrete
next-stage plan. A microbenchmark is evidence about the measured operation only;
no model throughput claim follows without a real-model comparison.

## Follow-up: verify and optimize every remaining opportunity

Requested 2026-09-06 against PR #138 at `bbc3157c`. All existing PR checks passed
before this follow-up. Preserve that revision as the performance baseline. The
user authorizes substantial refactoring and changes to implementation-level RNG
mapping; sampling distributions, cache ownership and emitted-token correctness
remain requirements. Sources are current code, the supplied reference checkouts,
official MLX/Metal/vLLM material, and isolated local measurements.

6. **Complete — verify remaining costs and implementation seams.** GPU dense
   draft sampling, grouped sampled acceptance, direct paged prefill, scheduled
   speculation, adaptive DSpark verification, bounded SSD transfers, and ordinary
   decode CPU/GPU timeline. Record any refuted premise or unsupported API path.
7. **Implemented and measured — sampling and paged-prefill improvements.** Validate
   numerical/sampling behavior, rejection frontiers and cache-hit parity, then
   compare operations and complete model turns against the preserved baseline.
   GPU draft chains and grouped acceptance are implemented. Short Qwen3 suffixes
   use direct paged attention at up to 16 query tokens; larger chunks retain
   gather plus SDPA, based on the context/width crossover benchmark.
8. **Partially implemented — speculative scheduling and adaptive budgets.** Establish
   resumable state, per-owner isolation and variable accepted spans before
   admitting speculation to the shared scheduler. Verify cancellation, rejection,
   owner recycling, allocation failures and real concurrent inference.
   Gemma fixed-depth DSpark and Muse fixed-depth DFlash share the scheduler and
   a trait with default verification transactions. Gemma real-model owner
   content/order isolation passed; concurrency throughput improved with observed
   batch-shape numerical differences. Muse model/scheduler tests pass and its
   real-model throughput did not show a consistent win, so scheduled DFlash is
   opt-in. Sampled multi-owner tests exposed and fixed residual/bonus global RNG
   coupling; prefill and cycle draws now use each scheduled owner's RNG. MTP draft token chains stay on
   device, and verifier IDs now remain authoritative host slices until their
   single embedding upload. Recurrent MTP scheduling still needs a resumable
   phase split and owner-specific tape replay; Qwen DFlash2 remains flat-only.
   Greedy whole-turn DSpark confidence/cost budgeting is implemented; adaptive
   scheduled speculation remains gated until its batch cost policy is validated.
9. **Implemented and measured — SSD transfers and ordinary decode.** Preserve
   bounded staging, completion-gated publication and resident-to-SSD storage;
   select backend changes from measured timelines and cold-cache workloads.
   Restore reads are bounded and overlap batched uploads; completion proofs gate
   publication. Upload-stage benchmarks improve at multi-block sizes. A valid
   Metal trace of the actual inference worker captured 21,732 compute intervals.
   Ordinary decode async scheduling showed no reliable gain, so Qwen retains
   synchronous paged completion. A guard drains pending forced-token work before
   error cleanup, and memory sizing counts each staging allowance once.
10. **In progress — PR handoff after local verification.** Review every opportunity
    against evidence, run relevant native/TS/real-model gates, record paired
    performance and limitations, then update the authorized PR and check CI.

Discovery decisions:

- All references were refreshed without switching their checkouts. vLLM now has
  CPU-primary and filesystem/object-store secondary tiers. Our SSD tier remains
  appropriate; unified memory does not eliminate GPU completion synchronization.
- The shared greedy scheduler path already evaluates one argmax graph. Mixed
  sampling still evaluates each request separately. Preserve sampler construction
  order and per-request parameters, combine evaluation roots, then read results.
- Deterministic speculative penalties depend on the known draft prefix until the
  first mismatch. Prepare those independent argmax graphs before one evaluation;
  retain sequential stochastic acceptance and its RNG consumption.
- Remove `FromChatCmd`, `HybridSchedulerCommand` and their adapter macro. Use
  `ModelCommand<FamilyCommand>` with shared construction, extraction and dispatch.
  Give ordinary chat barriers a trait default; retain only owner-state overrides.
  Keep the native export macro as the concrete binding layer.
- Scheduled speculation requires token-span results, per-owner draft/tape state,
  ragged verify and resumable drivers. A lane flag alone is unsafe. Metal 4 and
  Metal IO require allocator/storage integration and measured benefit; existing
  MLX already owns events, resource hazards and residency sets.
