# Inference architecture research and execution

Date: 2026-09-05. Audience: mlx-node maintainers. Baseline: `100a03ad`.
Worktree: `.worktrees/inference-control-2026-09-05`.

The task is to reduce recurring model-integration code, explain speculative
decoding with paged KV and concurrent requests, and improve measured inference
control-flow costs on Apple unified memory. Preserve bounded resident KV pools,
SSD cold persistence, exact committed frontiers, request isolation, cancellation,
and per-family tensor/cache ownership. No new dependencies or public API breaks.

Sources: supplied local checkouts pinned after remote refresh; primary vLLM,
Apple Metal, MLX and reference-project documentation; local tests and alternating
performance measurements. Distinguish source facts, proposals and measurements.

1. **Complete — discovery and follow-up.** Audit current registration,
   scheduler, sampling, speculative transactions and SSD lifecycle. Compare vLLM
   MTP/DFlash/DSpark and MTPLX/oMLX/mlx-vlm. Resolve hardware/API constraints and
   record source provenance and remaining gaps.
2. **Complete — synthesis.** Produce the architecture comparison and choose changes
   justified by the audit. Record deferred changes with explicit prerequisites.
3. **Complete — implementation.** Consolidate repeated native model plumbing;
   reduce avoidable GPU completion waits in shared inference paths. Keep larger
   speculative scheduling or Metal backend migrations behind demonstrated need.
4. **Complete — verification and delivery.** Native build, Rust/TS checks,
   sampler and real-model parity gates, paired performance measurements, source
   links and final diff verified. Results and limitations are recorded in
   [validation.md](validation.md). The selected changes are ready for review;
   the larger staged migrations remain proposed work.

Success requires a reviewable implementation plus cited research and a concrete
next-stage plan. A microbenchmark is evidence about the measured operation only;
no model throughput claim follows without a real-model comparison.

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
- Consolidate repeated `FromChatCmd`/`HybridSchedulerCommand` implementations and
  scheduler telemetry NAPI forwarding. Keep typed family-specific commands and
  generated declarations compatible.
- Scheduled speculation requires token-span results, per-owner draft/tape state,
  ragged verify and resumable drivers. A lane flag alone is unsafe. Metal 4 and
  Metal IO require allocator/storage integration and measured benefit; existing
  MLX already owns events, resource hazards and residency sets.
