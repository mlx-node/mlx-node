# Speculative decoding × paged KV: what we align with vLLM, and what we do not

vLLM is the reference for how speculative decoding interacts with a paged KV cache. This document records
which of its designs we adopted, which we deliberately did not, and what is still ahead.

The rule we hold ourselves to is **identical laws, our own mechanics**. vLLM is a throughput-first datacenter
server on discrete HBM; mlx-node is a latency-first local runtime on unified memory with one model thread per
model. Several vLLM designs exist to solve problems those constraints create. Where the constraint applies to
us, copying is right. Where it does not, copying costs performance.

Reference: `vllm` at `bd8865a299`. Our side is cited by symbol, not line number — line numbers rot.

---

## Part 1 — Adopted laws

These are vLLM's rules, and they are now ours. They are enforced in code, not just documented.

| Law | vLLM | Where it lives here |
|---|---|---|
| Rejection is cursor arithmetic: roll the counter back, overwrite the slots in place. Never erase, never free. | `scheduler.py` decrements `num_computed_tokens` by the rejected count and does nothing else | `PagedKVCacheAdapter::rollback_last_tokens{,_for}`, `Gemma4KVCacheCoordinator::rollback_last_tokens_all` |
| Maintain at the **committed** frontier, never the optimistic write cursor. No prune, checkpoint, sidecar, or registration between a verify write and its commit. | `kv_cache_manager.py` consumes committed lengths | Invariant I9. `prune_sliding_window_for_committed`, `settle_grouped_kv_step_at`, `MusePagedSettle::Committed`, and the L-SETTLE law on `SpecPagedCache` |
| Never persist unverified tokens; cap every commit at the request's real token count. | prefix-cache commit capped at `request.num_tokens` | Invariant I3. The registration length guard on the adapter; the `paged_gdn_state_dirty` refuse-to-persist latch |
| Reserve lookahead slots before the cycle; exhaustion degrades to AR, never to a mid-verify error. | `allocate_slots(..., num_lookahead_tokens=K)` | Invariant I1. `reserve_rows{,_for}`, `reserve_rows_all`, `MtpStepper::reserve_cycle_lookahead`, per **cycle** |
| Greedy acceptance is `target_argmax[k] == draft[k]`, and the boundary token is the target's argmax. | `rejection_sampler.py` | `engine/dspark_turn.rs` greedy fast path; `run_mtp_turn`'s accept loop |
| Prefix reuse is longest-common-prefix over full blocks. | `get_computed_blocks` → `find_longest_cache_hit` | `find_cached_prefix_for_prepare` → `BlockAllocator::find_longest_cache_hit` (**paged lane only** — see Part 3) |

---

## Part 2 — Deliberate divergences

Labelled `X*` here to avoid collision with the Stage-D ladder in Part 3. Where an entry corresponds to a
numbered non-goal in the Stage-D plan, the mapping is: X1=N4, X2=N3, X3=N2, X5=N6, X6=N5, X7=N9.

Each entry states what vLLM does, what we do, why, and **what would reopen it**. A divergence without a
reopen trigger is dogma, not engineering.

### X1. Recurrent (GDN/Mamba) state: tape replay, not checkpoint blocks

**vLLM:** stores Mamba state in checkpoint blocks and selects among them on rejection.
**Us:** replay the layer tape from a snapshot to the accepted depth.

**Why:** two measurements on unified memory point the other way. Restore beats prefill by 3.3–21.4×, and
oversizing the paged pool costs ~10× on long-context decode through residency thrash. Reshaping the pool to
hold recurrent state walks into that cliff. Tape replay is AR-exact, `O(depth ≤ 4)`, and runs on-thread.

vLLM has not solved this either — it avoided it. Its Mamba align-mode assumes draft models carry no Mamba
layers.

**Reopens if:** a GDN-bearing *drafter* ever appears, or the pool residency cliff is eliminated.

### X2. Rollback direction: state chases the transcript

**vLLM:** history is derived from a counter, so rejection is a decrement and the history follows.
**Us:** the emitted transcript is truth, and every state kind is rewound to match it.

**Why:** we emit tokens *inside* the cycle as they are accepted, for streaming latency. vLLM emits at step
boundaries. Once a token has gone to the user, no counter decrement takes it back. A literal port of vLLM's
direction would be a correctness bug here, not an improvement.

**Reopens if:** we ever stop emitting mid-cycle — which would be a product regression in interactive latency.

### X3. Stop discipline: clamp before commit

**vLLM:** nothing to align to. It never emits mid-cycle, so the question does not arise.
**Us:** `run_dspark_turn` clamps the accepted run at the stop token *before* committing, so the stop token's
slot is never written and the reconcile surplus is structurally zero.

**Reopens if:** X2 reopens. They are the same decision seen from two sides.

### X4. Two engine loops, permanently

**vLLM:** proposers specialize; verify and accept live in the runner.
**Us:** `run_dspark_turn` (clamp-before-commit) and `run_mtp_turn` (emit-then-rewind) are never merged.

**Why:** opposite stop disciplines and different trait surfaces — hidden states cross `MtpStepper` and never
`DsparkStepper`. vLLM splits at the same joint; we simply draw the line one level lower. Merging them would
mean one loop with a mode flag threading two incompatible commit orders.

**Reopens if:** the two stop disciplines ever converge, which requires X2/X3 to reopen first.

### X5. Per-group metadata capture and block-table swapping

**vLLM:** captures per-group attention metadata and swaps block tables per KV-cache group.
**Us:** nothing to port — we get this by construction. Our verify runs *inside* the family's grouped layer
loop, which already routes per-layer groups through the coordinator.

**Reopens if:** a verify path is ever written outside a family's grouped loop.

### X6. Multi-module speculative scheduling machinery

**vLLM:** `_reserve_prefill_lookahead`, `scheduler_block_size >= K`, `extra_retained_tokens`.
**Us:** none of it. That machinery exists because vLLM's drafters write pool KV during chunked prefill. No
mlx-node drafter writes target-visible pool blocks, so these have no referent.

**Reopens if:** any drafter starts writing pool KV during prefill — i.e. as part of C3 below.

### X7. Fused slot-mapping kernels

**vLLM:** Triton kernels for slot math.
**Us:** slot math is Rust bookkeeping. There is no Triton on Metal, and the arithmetic is not the bottleneck.

**Reopens if:** profiling ever shows slot bookkeeping on the critical path.

### X8. Test contract: strict byte equality, not thresholded matching

**vLLM:** `assert_request_outputs_match` allows bounded mismatches, because the `T=1` decode kernel and the
`T=1+L` verify kernel differ by ~1 ULP in bf16.
**Us:** speculative output must byte-match the AR baseline exactly.

**Why:** determinism is a feature for a local runtime, and the strict oracle has caught real bookkeeping bugs
that a thresholded one would have absorbed. The cost is that fixtures must be screened for near-ties — and
that screening must cover the model's **reasoning block**, not only the visible answer, since a tie inside the
thought spends the shared token budget and truncates the answer. See `bf16-tie-screening`.

Screening is a real cost, paid per fixture. On the 35B-A3B MoE checkpoint an unscreened free-form prompt
diverges between paged MTP and paged AR — and equally between *flat* MTP and *flat* AR, and even between paged
AR and flat AR with no speculation involved — while every screened fixture is byte-identical across all three.
A `D*` gate written on an unscreened prompt therefore measures kernel rounding, and reads as a correctness bug.

**Reopens if:** screening cost ever exceeds the bugs it catches.

### X9. `preserve_thinking` is opt-in, not on

All four reference stacks (transformers, mlx-lm, mlx-vlm, vLLM) render Gemma4's template as shipped and drop
prior assistant reasoning. So do we, **by default** — a stateless render is byte-identical to theirs. A
stateful `ChatSession` opts in, because it owns the transcript and the KV cache and would otherwise lose all
cross-turn reuse. vLLM ships a vendored Gemma4 template with the same gate shape and lets clients set the flag;
we scope it to the one caller that benefits.

---

## Part 3 — Still worth aligning (the ladder)

Everything here is a gap we intend to close. Ordered.

| Stage | What | Gate |
|---|---|---|
| ~~**D1**~~ | **Landed.** gemma4 DSpark verifies against the paged pools through `SpecPagedCache`; `SpecTurnEpilogue` makes L-EPILOGUE executable for the driver that takes one — `run_paged_dspark_turn`, where an abandoned epilogue is counted and debug-asserts. It is opt-in, not a seal: `finish_paged_turn` stays reachable directly, and qwen3.5 DENSE paged MTP's forked epilogue (`paged_turn_sync_core`) never calls it at all, so that fork is still live and unobserved. The flat lane narrowed to the assistant drafter, whose Q-only attention reads flat `Gemma4LayerCache` K/V directly — that is D4. | **Met.** 2.3×–4.5× vs paged AR, same binary, `gemma4_dspark` e2e on Gemma-4-12B-IT + `dspark_gemma4_12b_block7`: 4.5× on `decode_wrap` (constrained count, sub-window prompt, the window wraps mid-DECODE — 90.2 vs 20.1 tok/s), 3.8× on `prefill_wrap` (the >1100-token prompt, the window wraps during PREFILL — 74.1 vs 19.5), 2.3× on `multi_cycle` (free-form, 200 tok — 45.9 vs 20.3). T=0 byte parity on both sliding-wrap legs |
| ~~**D2**~~ | **Landed.** qwen3.5 MoE paged MTP. The MoE speculative plan publishes `supports_paged_attention`, and the generic paged driver runs the family's speculative core in place of the autoregressive loop (`PagedBackend::admit_paged_speculative_decode` + `run_paged_speculative_decode`), so both paged decoders share one epilogue. History is CYCLE history — dense's committed-history mode is gated on a prompt-hidden seed no MoE prefill can produce, so the flag and its inert seed were deleted rather than left as an unsatisfiable option. | **Met.** 1.28× at depth 2 and 1.23× at depth 1 vs paged AR (35B-A3B MXFP8-MTP, 400-token decode, release, alternating A/B in one binary); 1.15×/1.14× at depth 3/4. T=0 three-way parity paged-MTP == paged-AR == flat-MTP over screened fixtures |
| **D3** | muse DFlash on paged. Needs `DecoderPlan::Speculative`, settle-as-parameter (landed in D0), and an admission cap at `min(prefix_hit, context.logical_len())`. | > paged AR |
| **D4** | gemma4 assistant Q-only over target KV — vLLM's `kv_sharing_target_layer_name` shape, zero drafter KV. | Pool-kernel Q-only, or a clean 4K/16K/32K A/B. **NO-GO acceptable** |
| **LCP-flat** | Longest-common-prefix reuse on the **flat** lane for pure-attention families. Today it returns 0 or everything; vLLM always keeps the common prefix. Sanctioned in `engine/cache.rs` for families without recurrent state. | Surfaced by the gemma4 continuation bug. D1 made it moot for gemma4 speculation; the flat lane now serves only the assistant drafter |
| **B1** | `dense_gdn_consumed_tokens` audit stored in the checkpoint. | — |
| ~~**B3**~~ | **Landed.** `engine::spec_owner::SpecOwner` addresses the paged adapter by the sequence a turn claimed at entry, so `DenseMtpStepper` no longer moves it out of the model and restores it in `Drop`. Was a soft prereq of D3 and a hard one of C2. | — |
| **C2** | Scheduled-lane speculation — drafts as ordinary scheduled tokens instead of a barrier turn. **The largest remaining structural gap.** | LOOM Stage-1 + B3. Measure first: the depth-1 MTP ceiling is 1.1–1.15×, so this may stay default-off |
| **C3** | Drafter KV resident in the pool (full I8): DSpark private KV, DFlash retained context, MoE drafter cache. EAGLE tail-drop ships in the **same PR**, never before. | C2 paying off first |

**Sequencing note.** C2 gates C3, and C3 is the trigger that would reopen X6. So the ladder is not merely a
list: the divergences in Part 2 that carry reopen triggers are mostly waiting on C2/C3, while X1, X2, X3, X4
and X7 are permanent unless the hardware or the product changes underneath them.
