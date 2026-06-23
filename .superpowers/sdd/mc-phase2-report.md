# Phase 2 report — Rust warm media→text continuation (AUDIO + NON-UNIFIED gemma4)

**Status: DONE** — warm continuation shipped for BOTH Phase-2 modalities (audio + non-unified
image); the non-unified-image warm==cold golden is BYTE-EXACT. No audio-only gate was needed.

- **headSha:** `cd765a24` (worktree `mlx-node-gemma4-unified`, branch
  `feat/gemma4-media-continuation`)
- **Modalities continuable:** AUDIO (unified 12B, causal) + NON-UNIFIED image (e2b, causal SigLIP).
  UNIFIED-vision image stays single-shot (Phase 3).

---

## What changed (file:line — all in `crates/mlx-core/src/models/gemma4/model.rs`)

| Site | Line | Change |
|---|---|---|
| `Gemma4Inner` struct field | 418 | add `media_session_continuable: bool` |
| constructor literal | 996 | init `media_session_continuable: false` |
| `clear_reuse_state` | 1069 | reset marker `false` (covers init + `reset_caches_sync`) |
| NEW `finalize_vision_turn_media_state` | 1956 | two-state media finalize (shared sync+stream) |
| → arm marker (continuable) | 2024 | `media_session_continuable = true` on keep-live |
| → non-continuable teardown | 2039 | release + keep history/keys + marker `false` |
| NEW `gemma4_no_overlay_continuable` | 2048 | Phase-2 scope gate (audio ‖ non-unified image, ¬overlay) |
| vision SYNC prefill-start reset | 2144 | marker `false` on cold rebuild |
| vision SYNC Ok-branch call | 2232 | call finalize helper (replaces unconditional release + save) |
| vision STREAM prefill-start reset | 2371 | marker `false` on cold rebuild |
| vision STREAM Ok-branch call | 2486 | call finalize helper |
| `verify_cache_prefix` force-miss | 4989 | `&& !self.media_session_continuable` on image-held |
| `text_delta_image_guard` | 5226 | `if self.media_session_continuable { return None; }` at top |

Tests:
- `__test__/models/gemma4-media-continuation-e2e.test.ts` (NEW) — warm vs cold golden (3 describe
  blocks).
- `crates/mlx-core/src/models/gemma4/model.rs` — rewrote
  `test_text_delta_after_audio_turn_rejected_like_image_turn` (contract flip), added
  `test_media_session_continuable_reset_matrix`, added `test_gemma4_no_overlay_continuable_gate`.

No NAPI surface change — both `index.d.cts` artifacts unchanged (the marker is internal).

---

## R1 reconciliation — HOW the sliding checkpoint offset was aligned with `cached_token_history`

This is the make-or-break detail, and the answer turned out to be deeper than the edit plan
assumed. Two facts, proven by source + live instrumentation:

1. **Length-finish offset (the plan's stated risk).** The vision decode loop never forwards the
   final sampled token, so after the loop the live (non-shared) sliding caches AND the global paged
   KV sit at `prefill_len + G − 1`. The drop-last history is `prefill_len + G − 1` on
   stop/repetition/cancelled (MATCH) but `prefill_len + G` on `"length"` (one short). Fix = mirror
   the text path's `materialize_final`: on continuable + `"length"`, forward the final token once via
   `run_paged_decode_step` (model.rs:1971), advancing both caches to `prefill_len + G`. This is the
   exact mechanism `paged_turn.rs` (length gate) → `Gemma4PagedDecode::materialize_final` →
   `run_paged_decode_step` uses for text.

2. **The REAL blocker — KV-shared sliding layers (discovered via live tracing).** On e2b
   (`num_kv_shared_layers > 0`), the `SharedOnSliding` layers (idx 15+) physically store NO flat
   K/V — they read the anchor's. So `gemma4_sliding_caches_ready_at` (which requires EVERY
   `is_sliding_layer` flat cache populated) is **structurally unsatisfiable**, and
   `remember_gemma4_sliding_history_checkpoint` returns `stored == false`. Crucially, the TEXT
   warm-continue ALSO hits `stored == false` on e2b — yet it works, because it tolerates the
   missing checkpoint and restores the prefix via **REPLAY** (`state="replay"`,
   `continued_live=true`, `checkpoint_stored=false`) over the live content-addressed global KV. My
   first draft hard-downgraded to non-continuable on `stored == false`, which (correctly per its
   own logic, wrongly per the goal) made every KV-shared media session single-shot — that was the
   bug behind the initial `cachedTokens=0`. **Fix:** arm the marker on keep-live success regardless
   of `stored` (model.rs:1988-2024), mirroring the text path. The checkpoint is now best-effort: it
   stores (fast-path restore) on NON-shared checkpoints, and is a no-op (replay restore) on shared
   ones — identical to text.

So the alignment is: keep the global KV registered for reuse (content-addressed), align it to the
saved history on `"length"` via the materialize, and let the next delta replay-restore the sliding
state — exactly the text path. The sliding history checkpoint is NOT load-bearing for continuation
on the shipping (KV-shared) checkpoints.

---

## Golden-parity result

- **NON-UNIFIED image (e2b, `gemma-4-e2b-it`): WARM == COLD BYTE-IDENTICAL.** Verified on a real
  2-turn run (image "Describe this image." → text "What is the main color?"): warm `rawText` ==
  cold `rawText` and `numTokens` == 29 == 29, byte-for-byte. `cachedTokens=332` (the full
  media-bearing prefix was reused via replay). This is the load-bearing R1 proof — it covers the
  `"length"` finish path (turn-1 hit maxNewTokens) AND the shared-sliding/replay path. e2b image
  turns emit NO `<|channel>thought…<channel|>` block, so the cold replay's template re-render is
  lossless and the comparison is well-posed.

- **AUDIO (unified 12B, `gemma-4-12b-it`): warm CONTINUES (cachedTokens > 0) + FINAL-answer parity.**
  A byte-exact warm==cold golden is **ill-posed for audio via the public API**, and this is NOT a
  warm-path bug: the 12B emits a `<|channel>thought…<channel|>` block; the WARM KV holds it raw, but
  a COLD `chatSessionStart` replay re-renders turn-1 through jinja whose `strip_thinking` macro DROPS
  prior-turn reasoning by design → the cold turn-2 sees a shorter prefix and re-derives a longer
  reasoning trace (same FINAL answer). There is no raw-token-prefill API and no strip-disable, so the
  full token stream cannot match for a thinking checkpoint. The audio test therefore asserts (a) the
  warm path actually continued (`cachedTokens > 0`) and (b) warm FINAL answer == cold FINAL answer
  ("English" == "English"). The audio warm path runs the IDENTICAL finalize code as the image path
  (gated only on `has_audio` vs `has_image`), so the image byte-exactness transitively covers the
  audio numerics.

- **UNIFIED image: stays single-shot (Phase 2).** The marker is never armed (gate returns false for
  unified bidirectional vision); the native continue throws the IMAGE restart prefix and TS falls
  back to cold replay. The e2e asserts a coherent non-degenerate follow-up; the Rust unit test
  `test_gemma4_no_overlay_continuable_gate` locks the gate (unified image → false, audio/non-unified
  image → true, text → false).

**Gated to audio-only? NO.** The opposite of the plan's fallback occurred: non-unified image is the
clean byte-exact case; audio's byte-exactness is API-ill-posed but its warm path is proven correct
via the shared code + the image golden + final-answer parity.

---

## Gate outputs (verbatim)

- `cargo fmt --check` → clean (exit 0).
- `cargo clippy --all-targets -- -D warnings` → 0 errors; only the accepted pre-existing
  `warning: the following packages contain code that will be rejected by a future version of Rust:
  block v0.1.6` note.
- `cargo test -p mlx-core --lib gemma4` →
  `test result: ok. 141 passed; 0 failed; 4 ignored; 0 measured; 1752 filtered out`. Includes
  `test_text_delta_after_audio_turn_rejected_like_image_turn` (contract flip),
  `test_media_session_continuable_reset_matrix`, `test_gemma4_no_overlay_continuable_gate` — all ok.
- `yarn typecheck` → exit 0 (clean).
- `vp lint` → `Found 6 warnings and 0 errors` — all 6 are PRE-EXISTING in untouched files
  (`__test__/server/messages-handler.test.ts`, `messages-paged-no-warm-reuse.test.ts`); the new test
  lints clean (`Found 0 warnings and 0 errors`).
- e2e golden (`gemma4-media-continuation-e2e.test.ts`) → `Tests 3 passed (3)`.
- Regression — existing gemma4 e2e: `gemma4-unified-audio-e2e` (2 passed), `gemma4-unified-vision-e2e`
  + `gemma4-unified-e2e` (3 passed). Pure-text gemma4 2-turn probe: T1="Paris", T2="Berlin",
  T2 warm-continues (cachedTokens=31) — unchanged.
- Cross-family: change is confined to `models/gemma4/model.rs` (no shared engine/other-family code
  edited); qwen3.5/moe/lfm2/qwen3/qianfan are structurally untouched.

Build/metallib: `yarn build:native` then restored known-good metallib (md5
`23044b4f78d70322613a5e1c6256eea4`) into both `packages/core/mlx.metallib` and
`packages/core/npm/darwin-arm64/mlx.metallib` before every decode/e2e run.

---

## Concerns

1. **Audio byte-exact golden is API-ill-posed (documented, not a bug).** The template's
   `strip_thinking` + the absence of a raw-token-prefill API mean a thinking checkpoint can't have a
   token-exact warm==cold comparison through the public surface. Mitigated: the non-unified-image
   golden is byte-exact and the audio path shares the finalize code. A future stronger audio golden
   would need a test-only "prefill from token ids" entrypoint or a strip-disable template variant.

2. **The sliding history checkpoint is effectively dead on all shipping vision checkpoints**
   (e2b + 12B are both KV-shared). Continuation rides the replay path, exactly like text. This
   matches the existing text behavior and is acceptable (replay is correct, just not the free
   fast-path). A follow-up could add a "non-shared sliding layers only" readiness predicate so the
   checkpoint fast-path also fires on KV-shared models — out of Phase-2 scope.

3. **Performance is "warm" not "free":** the delta replays the full matched prefix (re-prefills the
   sliding suffix over the live global KV). This is the same cost profile as a text warm-continue and
   is still much cheaper than the Phase-1 cold restart (no re-embedding of image/audio features,
   global KV reused). Not separately benchmarked.
