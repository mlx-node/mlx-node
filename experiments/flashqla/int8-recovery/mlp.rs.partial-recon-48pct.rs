use crate::array::MxArray;
 "summary": "Root-cause the worktree default-path generation garbage via parallel diff analysis",
* Qwen3.5 int8 W8A8 prefill A/B harness — single-arm measurement primitive.
use crate::nn::Linear;
* Stage 4 of the int8 W8A8 prefill GEMM integration: measures the REALIZED
   "map_summary": "transformer/mlp.rs is NOT on all four garbage paths. Routing: qwen3.5-dense, qwen3(old, via TransformerBlock), and lfm2-dense DO use the shared transformer::MLP; but gemma4 uses its OWN GemmaMLP (crates/mlx-core/src/models/gemma4/mlp.rs, calling mlx_geglu) and qwen3.5_moe's expert layers use SparseMoeBlock/SwitchGLU — neither touches transformer/mlp.rs. Since gemma4 garbages WITHOUT ever executing mlp.rs, mlp.rs cannot be the common root cause; per the key logical constraint the suspect is eliminated. All four families run the EAGER Rust forward at decode (gemma4 and qwen3 have no compiled C++ path; the mlx_qwen35*.cpp compiled paths are qwen3.5/MoE-specific and opt-in). The int8 work is fully gated (MLX_INT8_PREFILL unset + M<256 => every int8 branch returns Ok(None) and fields are None), and mlx-sys/lib.rs adds only unused extern decls while mlx_na_int8.cpp only runs when called — so NO int8 code executes on the default decode/short-prefill path. The SINGLE most-shared compute-adjacent change across all four families is prewarm_checkpoint_pages in persistence_common.rs (a read-only CPU page-cache prewarm wired into every loader), which is already ruled out and cannot alter logits. CONCLUSION: the four-family common corruption surface is NOT in any of the uncommitted Rust diffs reviewed here (mlp.rs / int8_gemm.rs / gated_delta_net.rs / lib.rs / persistence*). The remaining shared candidate is the mlx-sys C++/metallib build itself — i.e. whether adding mlx_na_int8.cpp + the new metal .inc shaders changed the compiled native addon/metallib that ALL families link against (e.g. a rebuilt metallib, an altered mlx_common.h include, or a build-graph side effect) — OR a defect already present on the research-flashqla base relative to 493dcf9c that is not in this file set. Next: diff the built .node/metallib and the mlx-sys build graph (build.rs / cc inputs) and bisect base vs HEAD, rather than re-examining mlp.rs.",
   "analysis_suspect_levels": [
/// Default minimum `M = batch * seq_len` at which the int8 W8A8 prefill GEMM is
/// worth taking. Below this (notably `M == 1` decode), per-token activation
/// quant overhead dominates and int8 regresses vs bf16, so we fall through to
* caller via the `MLX_INT8_PREFILL` env var, read at LOAD TIME on the Rust
* side (`MLP::finalize_gate_up` quantizes the gate/up/down weights to int8
* ONLY when the flag is truthy at load). A single loaded model therefore
/// Returns `true` when `MLX_INT8_PREFILL` is set to a truthy value (non-empty,
/// not "0"/"false"). The int8 W8A8 prefill path is OFF by default — the bf16
* (`examples/qwen35-int8-prefill-pair.py`) does exactly that.
fn int8_prefill_enabled() -> bool {
* Toggle polarity (note: OPPOSITE of the lfm2 DISABLE-style toggles):
*   treatment = MLX_INT8_PREFILL=1     (int8 MLP path ON)
*   baseline  = MLX_INT8_PREFILL unset (bf16 fused path, unchanged default)
           !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
* Metrics come from the native `reportPerformance` path (measured AFTER
* model load, so load variance does not pollute them).
       "file": "gated_delta_net.rs (+moe/mod)",
       "suspect": "none",
*   [MLX_INT8_PREFILL=1] PATH=/usr/bin:$PATH oxnode \
/// The `M` threshold (`batch * seq_len`) at or above which the int8 path is
/// taken. Reads `MLX_INT8_PREFILL_MIN_M`, falling back to the default.
*     --mode ttft --prompt-tokens 1024 --max-new 4 --reps 4 --warmup 1
     "common_corruption_surface": "The mlx-sys native addon itself — the freshly-built `mlx_ffi` static lib (`packages/core/mlx-core.darwin-arm64.node`, built Jun 9 00:35, NEWER than all source) plus the MLX core GEMM/attention/eval primitives and the runtime-`fast::metal_kernel`-JIT'd Metal kernels that EVERY family dispatches at decode/short-prefill. This is the only code locus all four garbage families (qwen3.5-dense, qwen3-old, gemma4 via its own GemmaMLP/mlx_geglu, lfm2/MoE) share once you EXCLUDE the per-family Rust forwards: verified directly that gemma4 uses its own GemmaMLP (crates/mlx-core/src/models/gemma4/mlp.rs, calls mlx_geglu) and never imports transformer::MLP, so it garbages WITHOUT executing mlp.rs, gated_delta_net.rs, or any int8 code. The ONLY changeset perturbation that touches this shared native layer is adding the new translation unit crates/mlx-sys/src/mlx_na_int8.cpp + 3 na_int8_*.metal.inc fragments to the build.rs `cc` glob (read_dir of src/*.cpp at build.rs:411) — which forces a full mlx_ffi recompile/relink — together with the 9 unused `extern \"C-unwind\"` decls in mlx-sys/src/lib.rs. All of these are dead-at-runtime when MLX_INT8_PREFILL is unset, so if they are the cause it is a BUILD/LINK/JIT-registration side effect on the rebuilt native addon, not a source control-flow path. The equally-likely alternative is a defect already present in the BASE build (HEAD == 493dcf9c, the committed prewarm commit) independent of the uncommitted diffs.",
* Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
       .and_then(|v| v.trim().parse::<i64>().ok())
         "hypothesis": "The four-family garbage is NOT in any uncommitted Rust SOURCE diff. It lives in the rebuilt native mlx-sys addon (mlx_ffi static lib + its MLX GEMM/attention/eval primitives + runtime-JIT'd Metal kernels) that all four families link and dispatch against. The trigger is adding mlx_na_int8.cpp + na_int8_*.metal.inc to the build.rs cc-glob (build.rs:411 read_dir) which forces a full mlx_ffi recompile/relink; the resulting addon corrupts a shared GPU path. This is a build/link/JIT side effect, not a runtime control-flow path (those are all gated off).",
         "file_and_location": "crates/mlx-sys/src/mlx_na_int8.cpp (new TU auto-added at crates/mlx-sys/build.rs:411 cc-glob) + crates/mlx-sys/src/metal/na_int8_{gemm,quant,rescale}.metal.inc (runtime fast::metal_kernel JIT) → links/JITs into packages/core/mlx-core.darwin-arm64.node",
import { parseArgs } from 'node:util';
         "why": "Enforced by the Map constraint + my direct verification: gemma4 garbages WITHOUT executing mlp.rs/gdn/int8 (it uses its own GemmaMLP→mlx_geglu; confirmed no transformer::MLP import). Every uncommitted SOURCE path is gated/inert/read-only (verified finalize_gate_up unconditionally None-clears then gates on int8_prefill_enabled()==false-by-default; forward()'s bf16 path byte-identical; build.rs+mlx_common.h unchanged; new cpp is anonymous-namespace, dead unless called). So the only shared thing left that ALL four execute is the native addon. The addon is freshly built (newer than all source), so it is current — meaning a build/link/JIT perturbation from the new TU+shaders is the remaining shared candidate. Held at medium (not high) because no mechanism by which a separate anonymous-namespace TU + uncompiled-into-metallib .inc fragments corrupt a shared kernel has been PROVEN — it must be confirmed by the bisection, and the base-defect alternative is comparably likely."
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
       {
         "hypothesis": "The bug is already present on BASE (HEAD == 493dcf9c, the committed cold-mmap prewarm commit) independent of all uncommitted work — i.e. a defect committed in 493dcf9c or earlier (e.g. the prewarm/load refactor altering weight finalize/materialize ordering, or a pre-existing main regression) that corrupts logits for every family.",
         "file_and_location": "Committed at HEAD 493dcf9c — candidate surface: crates/mlx-core/src/models/qwen3_5/persistence_common.rs / per-family persistence.rs load+materialize ordering (the committed half of the prewarm work) and anything 493dcf9c touched in the shared load/eval path",
         "confidence": "medium",
         "why": "HEAD is exactly 493dcf9c (the prewarm commit) — there is NO separate 'main' to fall back to in this worktree; BASE already contains the committed prewarm. The task's own evidence (stash-isolating the prewarm gave BYTE-IDENTICAL garbage) proves the uncommitted prewarm is not the cause, but it does NOT exonerate the COMMITTED 493dcf9c changes. If a clean rebuild at 493dcf9c with ALL uncommitted work stashed still garbages, the bug is on BASE, not in the uncommitted diffs. This is co-equal with H1 until the bisection separates them."
pub struct MLP {
   model: { type: 'string', default: DEFAULT_MODEL },
         "hypothesis": "[DOWN-RANKED — subset only, possible STACKED secondary bug, NOT the common cause] An unconditional defect in transformer/mlp.rs corrupts the shared bf16 SwiGLU path used by qwen3.5-dense, qwen3-old (via TransformerBlock), and lfm2-dense.",
         "file_and_location": "crates/mlx-core/src/transformer/mlp.rs finalize_gate_up()/forward() (+472)",
   /// E39: pre-stacked `[w_gate; w_up]` then transposed to `[hidden, 2*intermediate]`.
         "why": "ELIMINATED as the COMMON cause because gemma4 (own GemmaMLP→mlx_geglu) and qwen3.5_moe experts (SwitchGLU) garbage WITHOUT executing mlp.rs. Directly verified the bf16 path is byte-identical to base and the new try_forward_int8 call returns Ok(None) when MLX_INT8_PREFILL is unset (default). Listed only as a possible stacked secondary bug to confirm AFTER the common cause; cannot explain gemma4/qwen3."
   /// `forward()` uses `mlx_swiglu_mlp_forward_stacked` (one matmul instead of two
   'emit-text': { type: 'boolean', default: false },
         "hypothesis": "[DOWN-RANKED — subset/gated, NOT common] int8 quant/gemm or qkvz/GDN int8 fires on the default path and corrupts logits.",
         "file_and_location": "crates/mlx-core/src/models/qwen3_5/int8_gemm.rs, gated_delta_net.rs try_forward_qkvz_int8, mlx_na_int8.cpp",
   /// hoist the per-forward transpose to load time.
         "why": "Triple-gated and fail-closed (verified): int8_prefill_enabled()/qkvz default false on Err(_); fields None at finalize unless flag truthy; per-call env re-read; M>=256 threshold excludes M=1 decode and M~40 prefill; C++ ops gate on gen>=17 && K%16==0 and return false→bf16. Also qkvz/GDN is qwen3.5-only and cannot explain gemma4/qwen3. Cannot be the common cause."
   /// Stage 3 (NA int8 W8A8 prefill): opaque int8 quant of the fused
   /// `[gate; up]` weight `[2*intermediate, hidden]` (rows = output channels =
   /// `N`, matching `quantize_weight_int8`'s `[N, K]` contract). Populated by
   /// `finalize_gate_up()` ONLY when `MLX_INT8_PREFILL` is truthy. `None` keeps
const warmup = Number.parseInt(values.warmup!, 10);
         "step": "NO-REBUILD determinism + int8-off proof. On the CURRENT binary, run each garbage prompt twice and confirm byte-identical garbage (rules out a sampler/nondeterminism red herring; problem already says T=0 AND T=0.7 both garbage). Then re-run the SAME binary with MLX_INT8_PREFILL='' and MLX_INT8_PREFILL_QKVZ='' explicitly unset AND with MLX_NO_COMPILE=1 plus MLX_EVAL_ALL_CACHES=1. Because these env vars are read PER-CALL, if the int8 path were the cause, forcing it off would change output; if MLX_NO_COMPILE/MLX_EVAL_ALL_CACHES change nothing, the corruption is below the compile/cache layer (in the linked native kernels), not in the compiled-graph cache.",
         "stash_or_revert": "none (no file changes; env-var toggles only on the existing packages/core/mlx-core.darwin-arm64.node)",
// Neutral prose, ~16 tokens/sentence, so `copies = ceil(promptTokens/16)`
         "expected_if_hypothesis_true": "Garbage is byte-identical across both runs and UNCHANGED by MLX_INT8_PREFILL/_QKVZ toggles and by MLX_NO_COMPILE=1/MLX_EVAL_ALL_CACHES=1 — confirming int8 is truly inert and the corruption is in the shared native/linked layer, consistent with H1 (and not refuting H2).",
         "expected_if_false": "If toggling MLX_INT8_PREFILL=0 vs the default removes the garbage, the int8 path was somehow firing (contradicts the gating analysis) → jump to the int8 hypotheses. If MLX_NO_COMPILE=1 fixes it, the corruption is in the compiled-graph cache layer instead."
 'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
   /// Per-output-channel f32 scale `[hidden]` for `down_w_i8`.
         "step": "RUST-ONLY revert of the int8 CALLERS (keep all C++/shaders so no cpp cachebust). Stash ONLY the Rust files that call into int8, reverting them to 493dcf9c: this removes the unconditional try_forward_int8/try_forward_qkvz_int8 calls and the int8_gemm import while leaving mlx_na_int8.cpp + the extern decls in place (declared-but-uncalled externs are harmless and do NOT require recompiling mlx-sys). Rebuild with `yarn build:native` (rust-only; mlx-sys TUs unchanged → no full cpp recompile). Re-run ALL FOUR family prompts.",
         "stash_or_revert": "git checkout 493dcf9c -- crates/mlx-core/src/transformer/mlp.rs crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs crates/mlx-core/src/models/qwen3_5/mod.rs crates/mlx-core/src/models/qwen3_5_moe/decoder_layer.rs ; and temporarily move the untracked crates/mlx-core/src/models/qwen3_5/int8_gemm.rs aside (mv to /tmp) since mod.rs no longer declares it. Leave crates/mlx-sys/src/lib.rs, mlx_na_int8.cpp, and the .metal.inc files IN PLACE.",
         "rebuild_cost": "rust-only",
         "expected_if_hypothesis_true": "If H1 (build/link/JIT side effect of the C++/shaders) is the cause: garbage PERSISTS for all four families, because the native mlx-sys addon (cpp+shaders) is untouched by this rust-only revert. This localizes the corruption to the native layer and away from the Rust int8 callers.",
         "expected_if_false": "If garbage DISAPPEARS for all four families after only reverting the Rust callers, then an unconditional Rust int8-caller side effect (not the C++) is the cause despite the gating analysis — re-audit try_forward_int8/qkvz for a side effect on the default path. If it disappears for only the qwen3.5/lfm2 subset but gemma4/qwen3 still garbage, mlp.rs is a stacked secondary and the gemma4/qwen3 common cause is elsewhere (→ next step)."
 const f = xs.filter((x) => Number.isFinite(x));
 if (f.length === 0) return Number.NaN;
         "step": "FULL-CPP-CACHEBUST revert to a clean 493dcf9c build (separates H1 from H2). Stash/remove ALL uncommitted work so the tree equals committed 493dcf9c, force the cc-crate cache bust, and rebuild from scratch. Re-run ALL FOUR family prompts on this pristine-BASE binary.",
         "stash_or_revert": "git stash --include-untracked (stashes all tracked diffs AND the untracked mlx_na_int8.cpp + na_int8_*.metal.inc + int8_gemm.rs + example scripts) so src/ no longer contains the new .cpp; then `rm -rf target/release/build/mlx-sys-*` to bust the cc-crate cache and force a full mlx-sys recompile; then `yarn build:native`.",
   pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
         "expected_if_hypothesis_true": "If H1 (the new TU/shaders perturbing the native build) is the cause: garbage DISAPPEARS — clean 493dcf9c with the new cpp/shaders gone produces coherent text for all four families. This CONFIRMS the corruption is introduced by the native-build perturbation in the uncommitted C++/shader work, NOT by BASE.",
         "expected_if_false": "If garbage PERSISTS on the pristine 493dcf9c build (all uncommitted work stashed, full clean rebuild), the bug is ALREADY ON BASE (H2) — it is in the committed 493dcf9c prewarm/load refactor or earlier, independent of the uncommitted int8 work. This is the test that PROVES main-not-uncommitted: a clean BASE rebuild that still garbages is dispositive."
       let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
       let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;
     "notes": "WORKTREE CONFIRMED: pwd = /Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla; all reads/diffs were against this worktree (verified main-checkout path /Users/.../mlx-node/crates/mlx-core/src/transformer/mlp.rs returns 'File does not exist' from here). CRITICAL CONTEXT THE FINDINGS DID NOT STATE: HEAD == 493dcf9c (the committed cold-mmap prewarm commit) — i.e. BASE in this worktree IS the prewarm commit, and there is no separate clean 'main' below it to diff against. The garbage comes from the UNCOMMITTED working tree relative to that committed HEAD. DIRECTLY VERIFIED (seeing-is-believing): (1) gemma4 uses its own GemmaMLP→mlx_geglu and does NOT import transformer::MLP — this is the linchpin proving mlp.rs/gdn/int8 cannot be the COMMON cause; (2) finalize_gate_up (mlp.rs:105-169) builds bf16 weights byte-identically then unconditionally None-clears the 4 int8 fields and only populates under `if int8_prefill_enabled()` which is false-by-default; forward() bf16 path (273-293) byte-identical, new line 269 is a no-op when off; (3) build.rs (UNCHANGED per git status) globs ALL src/*.cpp into mlx_ffi at line 411 (so the new TU is auto-compiled but as a separate anonymous-namespace TU), and the paged_attn.metallib is built from a HARDCODED 3-file list (build.rs:42-45), NOT from *.metal.inc — so the new shaders are never compiled into the shared metallib; (4) mlx_common.h UNCHANGED; (5) all per-family persistence diffs are pure prewarm-call + import (read-only); qwen3.5/persistence.rs net -83 is the prewarm fn being MOVED into persistence_common.rs (refactor, not behavior); (6) the built addon (Jun 9 00:35) is NEWER than every source file → NOT a stale artifact, the binary matches current source. STRONGEST EVIDENCE FOR points_to_main=true: the task's own stash-isolation already gave byte-identical garbage, and every uncommitted SOURCE path is gated/inert; the only thing that could carry corruption to all four families is the rebuilt native layer or a committed-BASE defect — and the step-3 clean-493dcf9c rebuild is the decisive discriminator. METHODOLOGY REMINDER consistent with project memory: drive ONE GPU/binary sequentially, warm + interleave if comparing tok/s (not relevant here since this is a correctness/garbage check, not perf), and prefer rust-only stashes (step 2) before paying the full mlx-sys cpp recompile (step 3). The 6 measurement-only int8 FFI ops (mlx_int8_gemm_pretransposed*, etc.) in lib.rs are pure extern decls and add no codegen risk."
for (const [k, v] of Object.entries(process.env)) {
 if (k.startsWith('MLX_INT8_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE') {
   relevantToggles[k] = v ?? '';
           down_proj,
           gate_up_proj_wt: None,
           down_proj_wt: None,
const loaded = await loadModel(modelPath);
           gate_up_s_w: None,
           down_w_i8: None,
           down_s_w: None,
): Promise<{ ttftMs: number; prefillTps: number; decodeTps: number; text: string; promptTok: number }> {
 // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
 // reuseCache:false ensures no cross-turn cache reuse leaks into prefill.
 const session = new ChatSession(loaded as unknown as SessionCapableModel, {
   /// `down_proj^T` weight once after all three projection weights are loaded.
   /// Forward will then use `mlx_swiglu_mlp_forward_stacked`, which does ONE
   /// (x @ wgu_t) matmul instead of two separate (x @ w_gate.T) + (x @ w_up.T)
   config: { maxNewTokens: maxNew, temperature: 0, reportPerformance: true, reuseCache: false },
   /// graph nodes vanish.
 const p = res.performance;
   /// Safe to call repeatedly (idempotent — overwrites). Callers from the
   /// persistence layer should invoke it once after the gate/up/down weights
   prefillTps: p?.prefillTokensPerSecond ?? Number.NaN,
   decodeTps: p?.decodeTokensPerSecond ?? Number.NaN,
       let w_gate = self.gate_proj.get_weight();
   promptTok: res.promptTokens ?? Number.NaN,
       let w_down = self.down_proj.get_weight();
       // gate, up: [intermediate, hidden] → stacked: [2*intermediate, hidden] →
       // transpose to [hidden, 2*intermediate] for the matmul x @ wgu_t.
for (let i = 0; i < warmup; i++) await oneTurn(`warmup-${i} `);
       let wgu_t = stacked.transpose(Some(&[1, 0]))?;
const ttftMs: number[] = [];
       // down: [hidden, intermediate] → [intermediate, hidden]
       let wd_t = w_down.transpose(Some(&[1, 0]))?;
let firstText = '';
       self.gate_up_proj_wt = Some(wgu_t);
       self.down_proj_wt = Some(wd_t);

       // Stage 3 (NA int8 W8A8 prefill, opt-in via MLX_INT8_PREFILL): quantize
 // ttft: unique nonce per rep → cold prefill (miss any content-addressed
 // prefix cache, including cross-process) so we measure real prefill cost.
 // decode: decodeTps is cache-independent; keep prompt FIXED for determinism.
       // channels so its `s_w[N]` scale broadcasts onto the GEMM accumulator
       // `acc[M, N]`. The UN-transposed `stacked` is `[2*intermediate, hidden]`
       // = `[N=2*intermediate, K=hidden]`, and `w_down` is
       // `[hidden, intermediate]` = `[N=hidden, K=intermediate]` — both already
       // in `[N, K]`, so we pass them straight in (the `*_wt` transposed forms
       // are for the bf16 matmul ONLY and must NOT be quantized here).
   firstText = t.text;
       // Stage 4b: `quantize_weight_int8` now ALSO hoists the per-forward
       // transpose — it returns the opaque int8 weight already in the `[K, N]`
       // kernel layout, so `try_forward_int8` does zero weight reshaping. The
       // stored orientation is opaque to Rust; we still pass the `[N, K]` input.
       //
       // SCOPE: int8 prefill is DENSE qwen3_5 ONLY. qwen3_5_moe routes its MLP
       // through the per-expert SwitchMLP / gather_qmm persistence path and never
       // calls this `finalize_gate_up()`, so `MLX_INT8_PREFILL` is a silent bf16
       // no-op on MoE (intended — not wired for MoE; documented, not fixed).
 promptTokensActual: promptTokActual,
       // MEMORY: with the flag ON, BOTH the bf16 stacked/transposed weight
       // (`gate_up_proj_wt` / `down_proj_wt`, above) AND the int8 weight stay
       // resident — the bf16 form is the per-call `Err`-fallback target in
       // `try_forward_int8`. Opt-in trades extra weight memory for prefill speed.
       //
       // VALIDATED REGIME: opt-in, greedy (T=0), bf16 models, prefill M>=256.
       // Sampling / MTP / long-context (>8k) are NOT yet validation-gated, which
       // is why this path is deliberately default-OFF.
 medPrefillTps: median(prefillTps),
 medDecodeTps: median(decodeTps),
 ...(emitText ? { textHash: hasher.digest('hex'), firstText: firstText.slice(0, 400) } : {}),
       self.down_s_w = None;
       if int8_prefill_enabled() {
           // Fail-soft: if quant fails (e.g. unsupported shape), leave the
           // fields None so forward() stays on the unchanged bf16 path.
           if let Ok((gu_i8, gu_s)) = int8_gemm::quantize_weight_int8(&stacked) {
               gu_i8.eval();
               gu_s.eval();
               self.gate_up_w_i8 = Some(gu_i8);
               self.gate_up_s_w = Some(gu_s);
           }
           if let Ok((d_i8, d_s)) = int8_gemm::quantize_weight_int8(&w_down) {
               d_i8.eval();
               d_s.eval();
               self.down_w_i8 = Some(d_i8);
               self.down_s_w = Some(d_s);
           }
       }
       Ok(())
   }

   /// Stage 3 NA int8 W8A8 prefill MLP path.
   ///
   /// Returns:
   ///   * `Ok(Some(out))` — the int8 path ran and produced the MLP output.
   ///   * `Ok(None)`      — not eligible (flag off / no int8 weights / `M`
   ///     below threshold / a fail-soft int8-op `Err`); caller must use bf16.
   ///   * `Err`           — only a genuine non-int8 error (reshape/shape).
   ///
   /// Pipeline (mirrors `mlx_swiglu_mlp_forward_stacked`):
   ///   `x[B,T,hidden]` → `[M, hidden]`
   ///   → `gate_up = int8_w8a8(x, gate_up_w_i8, gate_up_s_w)` `[M, 2*inter]`
   ///   → split → `swiglu = silu(gate) * up` (bf16, no f32 promotion)
   ///   → `out = int8_w8a8(swiglu, down_w_i8, down_s_w)` `[M, hidden]`
   ///   → reshape back to the original leading dims.
   ///
   /// The int8 op narrows its result to bf16 internally, so the residual add in
   /// the caller is not promoted to f32.
   fn try_forward_int8(&self, x: &MxArray) -> Result<Option<MxArray>> {
       // Gate 1: flag + quantized weights present.
       let (Some(gu_i8), Some(gu_s), Some(d_i8), Some(d_s)) = (
           &self.gate_up_w_i8,
           &self.gate_up_s_w,
           &self.down_w_i8,
           &self.down_s_w,
       ) else {
           return Ok(None);
       };
       if !int8_prefill_enabled() {
           return Ok(None);
       }

       // Gate 2: M = product of leading dims (everything but the last). For
       // `[B, T, hidden]`, M = B*T; for already-2D `[M, hidden]`, M = M.
       let shape = x.shape()?;
       let dims: &[i64] = &shape;
       if dims.len() < 2 {
           return Ok(None);
       }
       let hidden = dims[dims.len() - 1];
       let m: i64 = dims[..dims.len() - 1].iter().product();
       // M == 1 (decode) and small prefill regress vs bf16 → fall through.
       if m < int8_prefill_min_m() {
           return Ok(None);
       }

       // Reshape to 2D [M, hidden] for the int8 GEMM.
       let x2d = x.reshape(&[m, hidden])?;

       // gate_up: int8 W8A8. On Err (e.g. gen<17 / K%16!=0) fall back to bf16.
       let gate_up = match int8_gemm::int8_w8a8_matmul(&x2d, gu_i8, gu_s) {
           Ok(v) => v,
           Err(_) => return Ok(None),
       };
       // gate_up: [M, 2*intermediate] → split halves.
       let two_inter = gate_up.shape_at(1)?;
       let intermediate = two_inter / 2;
       let gate = gate_up.slice(&[0, 0], &[m, intermediate])?;
       let up = gate_up.slice(&[0, intermediate], &[m, two_inter])?;

       // swiglu = silu(gate) * up. `Activations::silu` preserves bf16 dtype, so
       // `gated` stays bf16 (no f32 promotion).
       let gate_act = Activations::silu(&gate)?;
       let gated = gate_act.mul(&up)?;

       // down: int8 W8A8 → [M, hidden]. On Err fall back to bf16.
       let out2d = match int8_gemm::int8_w8a8_matmul(&gated, d_i8, d_s) {
           Ok(v) => v,
           Err(_) => return Ok(None),
       };

       // Optional debug breadcrumb so a smoke test can confirm the int8 path
       // actually fired (gated so it never pollutes the default path).
       if std::env::var("MLX_INT8_PREFILL_DEBUG").is_ok() {
           eprintln!("[int8-prefill] fired: M={m} hidden={hidden} two_inter={two_inter}");
       }

       // Reshape [M, hidden] back to the original leading dims (mirror bf16).
       let mut out_shape: Vec<i64> = dims[..dims.len() - 1].to_vec();
       out_shape.push(hidden);
       let out = out2d.reshape(&out_shape)?;
       Ok(Some(out))
   }

   /// Forward pass: down(silu(gate(x)) * up(x))
   ///
   /// Uses fused C++ implementation for maximum performance (1 FFI call vs 8).
   ///
   /// # Arguments
   /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
   ///
   /// # Returns
   /// Output tensor, shape: (batch, seq_len, hidden_size)
   pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
       // Stage 3 (NA int8 W8A8 prefill, opt-in): route the fused gate_up and
       // down matmuls through the int8 GEMM when enabled, eligible, and the
       // weights are quantized. Any failure inside falls through to the bf16
       // path below (returns Ok(None)). OFF by default ⇒ zero change.
       if let Some(out) = self.try_forward_int8(x)? {
           return Ok(out);
       }

       // E39: fast path — pre-stacked + pre-transposed weights.
       // Env-toggle MLX_DISABLE_E39_STACKED_MLP=1 reverts to the legacy
       // two-matmul path for A/B testing.
       if let (Some(wgu_t), Some(wd_t)) = (&self.gate_up_proj_wt, &self.down_proj_wt)
           && std::env::var("MLX_DISABLE_E39_STACKED_MLP").is_err()
       {
           let handle = unsafe {
               sys::mlx_swiglu_mlp_forward_stacked(x.handle.0, wgu_t.handle.0, wd_t.handle.0)
           };
           return MxArray::from_handle(handle, "swiglu_mlp_forward_stacked");
       }

       // Legacy path: two matmuls with per-call transposes.
       let w_gate = self.gate_proj.get_weight();
       let w_up = self.up_proj.get_weight();
       let w_down = self.down_proj.get_weight();

       let handle = unsafe {
           sys::mlx_swiglu_mlp_forward(x.handle.0, w_gate.handle.0, w_up.handle.0, w_down.handle.0)
       };
       MxArray::from_handle(handle, "swiglu_mlp_forward")
   }

   /// Forward pass with cached intermediates for backward pass
   ///
   /// Returns: [output, gate, up, gate_act, gated]
   /// - output: final output
   /// - gate: gate_proj(x)
   /// - up: up_proj(x)
   /// - gate_act: silu(gate)
   /// - gated: gate_act * up
   #[cfg(test)]
   pub fn forward_with_cache(&self, x: &MxArray) -> Result<Vec<MxArray>> {
       // Compute gate and up projections
       let gate = self.gate_proj.forward(x)?;
       let up = self.up_proj.forward(x)?;

       // Apply SiLU activation to gate
       let gate_act = Activations::silu(&gate)?;

       // Element-wise multiplication
       let gated = gate_act.mul(&up)?;

       // Down projection
       let output = self.down_proj.forward(&gated)?;

       Ok(vec![output, gate, up, gate_act, gated])
   }

   // Weight setters for loading pretrained models

   pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
       self.gate_proj.set_weight(weight)?;
       // Invalidate the E39 stacked cache — caller must call finalize_gate_up().
       self.gate_up_proj_wt = None;
       // Stage 3: also invalidate the int8 quant of the fused gate_up weight.
       self.gate_up_w_i8 = None;
       self.gate_up_s_w = None;
       Ok(())
   }

   pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
       self.up_proj.set_weight(weight)?;
       self.gate_up_proj_wt = None;
       self.gate_up_w_i8 = None;
       self.gate_up_s_w = None;
       Ok(())
   }

   pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
       self.down_proj.set_weight(weight)?;
       self.down_proj_wt = None;
       self.down_w_i8 = None;
       self.down_s_w = None;
       Ok(())
   }

   // Mutable projection accessors.
   //
   // Expose the underlying `Linear`s so a persistence layer can drive
   // affine-quantized loads (`Linear::load_quantized`) or plain bf16 loads
   // uniformly without this shared module needing to know about each model's
   // quantization scheme. Each accessor invalidates the E39 stacked-MLP cache
   // (`gate_up_proj_wt` / `down_proj_wt`), because a caller obtaining a `&mut
   // Linear` may replace the weight; a stale stacked cache would otherwise be
   // served by `forward()`. The caller must re-run `finalize_gate_up()` if it
   // wants the stacked fast path after mutating a projection. This mirrors the
   // invalidation already done by `set_{gate,up,down}_proj_weight`.

   pub fn gate_proj_mut(&mut self) -> &mut Linear {
       self.gate_up_proj_wt = None;
       self.gate_up_w_i8 = None;
       self.gate_up_s_w = None;
       &mut self.gate_proj
   }

   pub fn up_proj_mut(&mut self) -> &mut Linear {
       self.gate_up_proj_wt = None;
       self.gate_up_w_i8 = None;
       self.gate_up_s_w = None;
       &mut self.up_proj
   }

   pub fn down_proj_mut(&mut self) -> &mut Linear {
       self.down_proj_wt = None;
       self.down_w_i8 = None;
       self.down_s_w = None;
       &mut self.down_proj
   }

   // Weight getters for backward pass

   pub fn get_gate_proj_weight(&self) -> MxArray {
       self.gate_proj.get_weight()
   }

   pub fn get_up_proj_weight(&self) -> MxArray {
       self.up_proj.get_weight()
   }

   pub fn get_down_proj_weight(&self) -> MxArray {
       self.down_proj.get_weight()
   }
}

impl Clone for MLP {
   fn clone(&self) -> Self {
       Self {
           gate_proj: self.gate_proj.clone(),
           up_proj: self.up_proj.clone(),
           down_proj: self.down_proj.clone(),
           gate_up_proj_wt: self.gate_up_proj_wt.clone(),
           down_proj_wt: self.down_proj_wt.clone(),
           gate_up_w_i8: self.gate_up_w_i8.clone(),
           gate_up_s_w: self.gate_up_s_w.clone(),
           down_w_i8: self.down_w_i8.clone(),
           down_s_w: self.down_s_w.clone(),
       }
   }
}

// =================== NA int8 W8A8 prefill: forward-path wiring ===================
// Review N1: ONE integration test that exercises the WIRED forward control flow
// (`finalize_gate_up` + `forward` -> `try_forward_int8`) end-to-end WITHOUT a model
// file, using small synthetic bf16 weights. Lives in `mlp.rs` (not the sibling
// `mlp_test.rs` / `int8_gemm.rs`) so it can assert the PRIVATE int8 fields are
// invalidated after a weight setter — the true (re-quantized / None) check.
//
// It mutates `MLX_INT8_PREFILL`, so it serializes on a private lock and restores
// the var via an RAII guard. Run serially is also safe:
//   cargo test -p mlx-core --lib mlp::int8_forward_wiring -- --test-threads=1
#[cfg(test)]
mod int8_forward_wiring {
   use super::*;
   use crate::array::DType;
   use std::sync::Mutex;

   // Serializes any test in this module that toggles MLX_INT8_PREFILL so a
   // concurrent test never observes another's setting.
   static ENV_LOCK: Mutex<()> = Mutex::new(());

   /// RAII guard: restores `MLX_INT8_PREFILL` on drop (even on panic).
   struct EnvGuard {
       prev: Option<String>,
   }
   impl EnvGuard {
       fn set(value: &str) -> Self {
           let prev = std::env::var("MLX_INT8_PREFILL").ok();
           // SAFETY: holders serialize on ENV_LOCK; no concurrent env access.
           unsafe {
               std::env::set_var("MLX_INT8_PREFILL", value);
           }
           Self { prev }
       }
   }
   impl Drop for EnvGuard {
       fn drop(&mut self) {
           // SAFETY: see EnvGuard::set.
           unsafe {
               match self.prev.take() {
                   Some(v) => std::env::set_var("MLX_INT8_PREFILL", v),
                   None => std::env::remove_var("MLX_INT8_PREFILL"),
               }
           }
       }
   }

   fn gpu_gen() -> i32 {
       unsafe { sys::mlx_gpu_architecture_gen() }
   }

   /// Deterministic LCG int in [lo,hi] (avoids the reserved `gen` ident).
   fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
       *state = state
           .wrapping_mul(6364136223846793005)
           .wrapping_add(1442695040888963407);
       let span = (hi - lo + 1) as u64;
       lo + ((*state >> 33) % span) as i32
   }

   /// Deterministic bf16 array of the given shape with small magnitudes
   /// (~[-0.2,0.2]) so the per-token quant absmax/round/clip paths are hit.
   fn rand_bf16_shape(state: &mut u64, shape: &[i64]) -> MxArray {
       let n: i64 = shape.iter().product();
       let mut f = vec![0f32; n as usize];
       for v in f.iter_mut() {
           *v = next_int(state, -200, 200) as f32 / 1000.0;
       }
       MxArray::from_float32(&f, shape)
           .unwrap()
           .astype(DType::BFloat16)
           .unwrap()
   }

   /// 2D `[rows, cols]` convenience (weights).
   fn rand_bf16(state: &mut u64, rows: i64, cols: i64) -> MxArray {
       rand_bf16_shape(state, &[rows, cols])
   }

   /// Build an MLP with deterministic bf16 weights. Uses the SAME shape class
   /// as `int8_gemm::tests::s2_w8a8_cosine_parity` (hidden=2560, intermediate=
   /// 2560 — K=2560 %16==0 for BOTH the gate_up (K=hidden) and down (K=inter)
   /// GEMMs, and N >= the 128x64 MPP tile) so the NA int8 kernels get realistic,
   /// well-supported extents (tiny hidden like 64 trips an MPP matmul2d edge and
   /// aborts via a foreign C++ exception). The forward cost at M=256 is trivial.
   const HIDDEN: u32 = 2560;
   const INTER: u32 = 2560;
   fn build_mlp(state: &mut u64) -> MLP {
       let hidden = HIDDEN;
       let inter = INTER;
       let mut mlp = MLP::new(hidden, inter).unwrap();
       // gate/up: [inter,hidden]; down: [hidden,inter].
       let w_gate = rand_bf16(state, inter as i64, hidden as i64);
       let w_up = rand_bf16(state, inter as i64, hidden as i64);
       let w_down = rand_bf16(state, hidden as i64, inter as i64);
       mlp.set_gate_proj_weight(&w_gate).unwrap();
       mlp.set_up_proj_weight(&w_up).unwrap();
       mlp.set_down_proj_weight(&w_down).unwrap();
       mlp
   }

   /// Per-row min cosine of two [M,N] f32 buffers.
   fn min_row_cosine(a: &[f32], b: &[f32], m: usize, n: usize) -> f64 {
       let mut min_cos = f64::INFINITY;
       for mi in 0..m {
           let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
           for ni in 0..n {
               let x = a[mi * n + ni] as f64;
               let y = b[mi * n + ni] as f64;
               dot += x * y;
               na += x * x;
               nb += y * y;
           }
           let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
           min_cos = min_cos.min(dot / denom);
       }
       min_cos
   }

   // This test validates the forward CONTROL FLOW (gating / threshold fall-
   // through / setter invalidation), NOT the W8A8 numeric quality:
   //   (a) M>=256: the int8 path FIRES — produces a finite, correctly-shaped
   //       bf16 output that DIFFERS from the bf16 forward (proving the int8
   //       branch actually ran rather than silently falling through). The
   //       per-row cosine vs bf16 is COMPUTED and printed as a diagnostic; the
   //       numeric-accuracy gate (cosine >= 0.999) lives in the dedicated
   //       `int8_gemm::tests::{s2,qkvz}_w8a8_cosine_parity` tests, so it is not
   //       re-asserted here (a wiring test must not double as the numeric gate).
   //   (b) M<256 (M=1): the int8 path FALLS THROUGH — byte-identical to bf16.
   //   (c) a weight setter INVALIDATES the int8 fields (set to None).
   #[test]
   fn int8_forward_wiring() {
       let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
       if gpu_gen() < 17 {
           eprintln!(
               "[int8-wire] SKIP: gpu gen {} < 17 (NA needs M5+)",
               gpu_gen()
           );
           return;
       }
       let hidden: i64 = HIDDEN as i64;
       const SEED: u64 = 0x1278_a1be_0000_0001; // fixed seed for reproducibility
       let mut state: u64 = SEED;

       // --- Reference bf16 outputs: build + finalize with the flag OFF so the
       // int8 fields stay None and `forward()` uses the unchanged bf16 path. ---
       {
           // Ensure OFF for the reference build/finalize.
           let _g = EnvGuard::set("0");
           let mut ref_mlp = build_mlp(&mut state);
           ref_mlp.finalize_gate_up().unwrap();
           assert!(
               ref_mlp.gate_up_w_i8.is_none() && ref_mlp.down_w_i8.is_none(),
               "flag OFF must leave int8 weights None"
           );
           // 3D inputs [B=1, T, hidden]: MLP::forward's stacked bf16 path slices
           // a 3D gate_up, so the input must be 3D. M = B*T drives the int8 gate.
           // x256: T=256 (M=256, at/above the 256 threshold → int8 fires);
           // x1:   T=1   (M=1, below threshold → bf16 fall-through).
           let x256 = rand_bf16_shape(&mut state, &[1, 256, hidden]);
           let x1 = rand_bf16_shape(&mut state, &[1, 1, hidden]);
           let y_ref_256 = ref_mlp.forward(&x256).unwrap();
           let y_ref_1 = ref_mlp.forward(&x1).unwrap();
           y_ref_256.eval();
           y_ref_1.eval();
           let ref256 = y_ref_256
               .astype(DType::Float32)
               .unwrap()
               .to_float32()
               .unwrap();
           let ref1 = y_ref_1
               .astype(DType::Float32)
               .unwrap()
               .to_float32()
               .unwrap();

           // --- int8 build/finalize with the flag ON. SAME weights + inputs
           // (reseed the weight LCG to the build seed used above). ---
           let _g_on = EnvGuard::set("1");
           // Rebuild from a fresh, identical seed so weights match the reference.
           let mut state2: u64 = SEED;
           let mut mlp = build_mlp(&mut state2);
           mlp.finalize_gate_up().unwrap();
           // Flag ON + supported shape -> int8 weights MUST be populated.
           assert!(
               mlp.gate_up_w_i8.is_some()
                   && mlp.gate_up_s_w.is_some()
                   && mlp.down_w_i8.is_some()
                   && mlp.down_s_w.is_some(),
               "flag ON must populate int8 weights at finalize"
           );

           // (a) M=256: the int8 path FIRES. Assert WIRING invariants only:
           // correct shape + dtype, all-finite output, and that it DIFFERS from
           // the bf16 forward (so we know the int8 branch ran, not a silent bf16
           // fall-through). Cosine vs bf16 is printed as a diagnostic; the
           // numeric-quality gate is s2/qkvz_w8a8_cosine_parity (NOT here).
           let y_256 = mlp.forward(&x256).unwrap();
           y_256.eval();
           assert_eq!(
               y_256.dtype().unwrap(),
               DType::BFloat16,
               "int8 out must be bf16"
           );
           assert_eq!(y_256.ndim().unwrap(), 3, "int8 out must be [B,T,hidden]");
           assert_eq!(y_256.shape_at(0).unwrap(), 1, "int8 out B must be 1");
           assert_eq!(y_256.shape_at(1).unwrap(), 256, "int8 out T must be M=256");
           assert_eq!(
               y_256.shape_at(2).unwrap(),
               hidden,
               "int8 out hidden dim must match"
           );
           let got256 = y_256.astype(DType::Float32).unwrap().to_float32().unwrap();
           assert_eq!(got256.len(), ref256.len());
           assert!(
               got256.iter().all(|v| v.is_finite()),
               "int8 path produced non-finite output"
           );
           let differs = got256
               .iter()
               .zip(ref256.iter())
               .any(|(a, b)| a.to_bits() != b.to_bits());
           assert!(
               differs,
               "int8 path output is byte-identical to bf16 — int8 branch did not fire at M=256"
           );
           let min_cos = min_row_cosine(&got256, &ref256, 256, hidden as usize);
           eprintln!(
               "[int8-wire] (a) M=256 int8 path FIRED (shape/dtype/finite ok, \
                differs from bf16); diagnostic min_row_cos = {min_cos:.6} \
                (numeric gate: s2/qkvz_w8a8_cosine_parity)"
           );

           // (b) M=1: int8 path falls through to bf16 -> byte-identical.
           let y_1 = mlp.forward(&x1).unwrap();
           y_1.eval();
           let got1 = y_1.astype(DType::Float32).unwrap().to_float32().unwrap();
           assert_eq!(got1.len(), ref1.len());
           let mut bad1 = 0usize;
           for i in 0..got1.len() {
               if got1[i].to_bits() != ref1[i].to_bits() {
                   bad1 += 1;
               }
           }
           eprintln!("[int8-wire] (b) M=1 byte-diffs vs bf16 = {bad1}");
           assert_eq!(bad1, 0, "M<256 must be byte-identical to bf16 forward");

           // (c) a weight setter invalidates the int8 fields. w_new is the
           // gate_proj shape [out=inter, in=hidden]; down takes its transpose
           // [out=hidden, in=inter] (both square here).
           let w_new = rand_bf16(&mut state2, INTER as i64, hidden);
           mlp.set_gate_proj_weight(&w_new).unwrap();
           assert!(
               mlp.gate_up_w_i8.is_none() && mlp.gate_up_s_w.is_none(),
               "set_gate_proj_weight must invalidate gate_up int8 fields"
           );
           mlp.set_down_proj_weight(&w_new.transpose(Some(&[1, 0])).unwrap())
               .unwrap();
           assert!(
               mlp.down_w_i8.is_none() && mlp.down_s_w.is_none(),
               "set_down_proj_weight must invalidate down int8 fields"
           );
           eprintln!("[int8-wire] (c) weight setters invalidated int8 fields");
       }
   }
}

// <<<RECOVERY-GAP line 680>>>
// <<<RECOVERY-GAP line 681>>>
// <<<RECOVERY-GAP line 682>>>
// <<<RECOVERY-GAP line 683>>>
// <<<RECOVERY-GAP line 684>>>
// <<<RECOVERY-GAP line 685>>>
// <<<RECOVERY-GAP line 686>>>
// <<<RECOVERY-GAP line 687>>>
// <<<RECOVERY-GAP line 688>>>
// <<<RECOVERY-GAP line 689>>>
// <<<RECOVERY-GAP line 690>>>
// <<<RECOVERY-GAP line 691>>>
// <<<RECOVERY-GAP line 692>>>
// <<<RECOVERY-GAP line 693>>>
// <<<RECOVERY-GAP line 694>>>
// <<<RECOVERY-GAP line 695>>>
// <<<RECOVERY-GAP line 696>>>
// <<<RECOVERY-GAP line 697>>>
// <<<RECOVERY-GAP line 698>>>
// <<<RECOVERY-GAP line 699>>>
// <<<RECOVERY-GAP line 700>>>
// <<<RECOVERY-GAP line 701>>>
// <<<RECOVERY-GAP line 702>>>
// <<<RECOVERY-GAP line 703>>>
// <<<RECOVERY-GAP line 704>>>
// <<<RECOVERY-GAP line 705>>>
// <<<RECOVERY-GAP line 706>>>
// <<<RECOVERY-GAP line 707>>>
// <<<RECOVERY-GAP line 708>>>
// <<<RECOVERY-GAP line 709>>>
// <<<RECOVERY-GAP line 710>>>
// <<<RECOVERY-GAP line 711>>>
// <<<RECOVERY-GAP line 712>>>
// <<<RECOVERY-GAP line 713>>>
// <<<RECOVERY-GAP line 714>>>
// <<<RECOVERY-GAP line 715>>>
// <<<RECOVERY-GAP line 716>>>
// <<<RECOVERY-GAP line 717>>>
// <<<RECOVERY-GAP line 718>>>
// <<<RECOVERY-GAP line 719>>>
// <<<RECOVERY-GAP line 720>>>
// <<<RECOVERY-GAP line 721>>>
// <<<RECOVERY-GAP line 722>>>
// <<<RECOVERY-GAP line 723>>>
// <<<RECOVERY-GAP line 724>>>
// <<<RECOVERY-GAP line 725>>>
// <<<RECOVERY-GAP line 726>>>
// <<<RECOVERY-GAP line 727>>>
// <<<RECOVERY-GAP line 728>>>
// <<<RECOVERY-GAP line 729>>>
// <<<RECOVERY-GAP line 730>>>
// <<<RECOVERY-GAP line 731>>>
// <<<RECOVERY-GAP line 732>>>
// <<<RECOVERY-GAP line 733>>>
// <<<RECOVERY-GAP line 734>>>
// <<<RECOVERY-GAP line 735>>>
// <<<RECOVERY-GAP line 736>>>
// <<<RECOVERY-GAP line 737>>>
// <<<RECOVERY-GAP line 738>>>
// <<<RECOVERY-GAP line 739>>>
// <<<RECOVERY-GAP line 740>>>
// <<<RECOVERY-GAP line 741>>>
// <<<RECOVERY-GAP line 742>>>
// <<<RECOVERY-GAP line 743>>>
// <<<RECOVERY-GAP line 744>>>
// <<<RECOVERY-GAP line 745>>>
// <<<RECOVERY-GAP line 746>>>
// <<<RECOVERY-GAP line 747>>>
// <<<RECOVERY-GAP line 748>>>
// <<<RECOVERY-GAP line 749>>>
// <<<RECOVERY-GAP line 750>>>
// <<<RECOVERY-GAP line 751>>>
// <<<RECOVERY-GAP line 752>>>
// <<<RECOVERY-GAP line 753>>>
// <<<RECOVERY-GAP line 754>>>
// <<<RECOVERY-GAP line 755>>>
// <<<RECOVERY-GAP line 756>>>
// <<<RECOVERY-GAP line 757>>>
// <<<RECOVERY-GAP line 758>>>
// <<<RECOVERY-GAP line 759>>>
// <<<RECOVERY-GAP line 760>>>
// <<<RECOVERY-GAP line 761>>>
// <<<RECOVERY-GAP line 762>>>
// <<<RECOVERY-GAP line 763>>>
// <<<RECOVERY-GAP line 764>>>
// <<<RECOVERY-GAP line 765>>>
// <<<RECOVERY-GAP line 766>>>
// <<<RECOVERY-GAP line 767>>>
// <<<RECOVERY-GAP line 768>>>
// <<<RECOVERY-GAP line 769>>>
// <<<RECOVERY-GAP line 770>>>
// <<<RECOVERY-GAP line 771>>>
// <<<RECOVERY-GAP line 772>>>
// <<<RECOVERY-GAP line 773>>>
// <<<RECOVERY-GAP line 774>>>
// <<<RECOVERY-GAP line 775>>>
// <<<RECOVERY-GAP line 776>>>
// <<<RECOVERY-GAP line 777>>>
// <<<RECOVERY-GAP line 778>>>
// <<<RECOVERY-GAP line 779>>>
// <<<RECOVERY-GAP line 780>>>
// <<<RECOVERY-GAP line 781>>>
// <<<RECOVERY-GAP line 782>>>
// <<<RECOVERY-GAP line 783>>>
// <<<RECOVERY-GAP line 784>>>
// <<<RECOVERY-GAP line 785>>>
// <<<RECOVERY-GAP line 786>>>
// <<<RECOVERY-GAP line 787>>>
// <<<RECOVERY-GAP line 788>>>
// <<<RECOVERY-GAP line 789>>>
// <<<RECOVERY-GAP line 790>>>
// <<<RECOVERY-GAP line 791>>>
// <<<RECOVERY-GAP line 792>>>
// <<<RECOVERY-GAP line 793>>>
// <<<RECOVERY-GAP line 794>>>
// <<<RECOVERY-GAP line 795>>>
// <<<RECOVERY-GAP line 796>>>
// <<<RECOVERY-GAP line 797>>>
// <<<RECOVERY-GAP line 798>>>
// <<<RECOVERY-GAP line 799>>>
// <<<RECOVERY-GAP line 800>>>
// <<<RECOVERY-GAP line 801>>>
// <<<RECOVERY-GAP line 802>>>
// <<<RECOVERY-GAP line 803>>>
// <<<RECOVERY-GAP line 804>>>
// <<<RECOVERY-GAP line 805>>>
// <<<RECOVERY-GAP line 806>>>
// <<<RECOVERY-GAP line 807>>>
// <<<RECOVERY-GAP line 808>>>
// <<<RECOVERY-GAP line 809>>>
// <<<RECOVERY-GAP line 810>>>
// <<<RECOVERY-GAP line 811>>>
// <<<RECOVERY-GAP line 812>>>
// <<<RECOVERY-GAP line 813>>>
// <<<RECOVERY-GAP line 814>>>
// <<<RECOVERY-GAP line 815>>>
// <<<RECOVERY-GAP line 816>>>
// <<<RECOVERY-GAP line 817>>>
// <<<RECOVERY-GAP line 818>>>
// <<<RECOVERY-GAP line 819>>>
// <<<RECOVERY-GAP line 820>>>
// <<<RECOVERY-GAP line 821>>>
// <<<RECOVERY-GAP line 822>>>
// <<<RECOVERY-GAP line 823>>>
// <<<RECOVERY-GAP line 824>>>
// <<<RECOVERY-GAP line 825>>>
// <<<RECOVERY-GAP line 826>>>
// <<<RECOVERY-GAP line 827>>>
// <<<RECOVERY-GAP line 828>>>
// <<<RECOVERY-GAP line 829>>>
// <<<RECOVERY-GAP line 830>>>
// <<<RECOVERY-GAP line 831>>>
// <<<RECOVERY-GAP line 832>>>
// <<<RECOVERY-GAP line 833>>>
// <<<RECOVERY-GAP line 834>>>
// <<<RECOVERY-GAP line 835>>>
// <<<RECOVERY-GAP line 836>>>
// <<<RECOVERY-GAP line 837>>>
// <<<RECOVERY-GAP line 838>>>
// <<<RECOVERY-GAP line 839>>>
// <<<RECOVERY-GAP line 840>>>
// <<<RECOVERY-GAP line 841>>>
// <<<RECOVERY-GAP line 842>>>
// <<<RECOVERY-GAP line 843>>>
// <<<RECOVERY-GAP line 844>>>
// <<<RECOVERY-GAP line 845>>>
// <<<RECOVERY-GAP line 846>>>
// <<<RECOVERY-GAP line 847>>>
// <<<RECOVERY-GAP line 848>>>
// <<<RECOVERY-GAP line 849>>>
// <<<RECOVERY-GAP line 850>>>
// <<<RECOVERY-GAP line 851>>>
// <<<RECOVERY-GAP line 852>>>
// <<<RECOVERY-GAP line 853>>>
// <<<RECOVERY-GAP line 854>>>
// <<<RECOVERY-GAP line 855>>>
// <<<RECOVERY-GAP line 856>>>
// <<<RECOVERY-GAP line 857>>>
// <<<RECOVERY-GAP line 858>>>
// <<<RECOVERY-GAP line 859>>>
// <<<RECOVERY-GAP line 860>>>
// <<<RECOVERY-GAP line 861>>>
// <<<RECOVERY-GAP line 862>>>
// <<<RECOVERY-GAP line 863>>>
// <<<RECOVERY-GAP line 864>>>
// <<<RECOVERY-GAP line 865>>>
// <<<RECOVERY-GAP line 866>>>
// <<<RECOVERY-GAP line 867>>>
// <<<RECOVERY-GAP line 868>>>
// <<<RECOVERY-GAP line 869>>>
// <<<RECOVERY-GAP line 870>>>
// <<<RECOVERY-GAP line 871>>>
// <<<RECOVERY-GAP line 872>>>
// <<<RECOVERY-GAP line 873>>>
// <<<RECOVERY-GAP line 874>>>
// <<<RECOVERY-GAP line 875>>>
// <<<RECOVERY-GAP line 876>>>
// <<<RECOVERY-GAP line 877>>>
// <<<RECOVERY-GAP line 878>>>
// <<<RECOVERY-GAP line 879>>>
// <<<RECOVERY-GAP line 880>>>
// <<<RECOVERY-GAP line 881>>>
// <<<RECOVERY-GAP line 882>>>
// <<<RECOVERY-GAP line 883>>>
// <<<RECOVERY-GAP line 884>>>
// <<<RECOVERY-GAP line 885>>>
// <<<RECOVERY-GAP line 886>>>
// <<<RECOVERY-GAP line 887>>>
// <<<RECOVERY-GAP line 888>>>
// <<<RECOVERY-GAP line 889>>>
// <<<RECOVERY-GAP line 890>>>
// <<<RECOVERY-GAP line 891>>>
// <<<RECOVERY-GAP line 892>>>
// <<<RECOVERY-GAP line 893>>>
// <<<RECOVERY-GAP line 894>>>
// <<<RECOVERY-GAP line 895>>>
// <<<RECOVERY-GAP line 896>>>
// <<<RECOVERY-GAP line 897>>>
// <<<RECOVERY-GAP line 898>>>
// <<<RECOVERY-GAP line 899>>>
// <<<RECOVERY-GAP line 900>>>
// <<<RECOVERY-GAP line 901>>>
// <<<RECOVERY-GAP line 902>>>
// <<<RECOVERY-GAP line 903>>>
// <<<RECOVERY-GAP line 904>>>
// <<<RECOVERY-GAP line 905>>>
// <<<RECOVERY-GAP line 906>>>
// <<<RECOVERY-GAP line 907>>>
// <<<RECOVERY-GAP line 908>>>
// <<<RECOVERY-GAP line 909>>>
// <<<RECOVERY-GAP line 910>>>
// <<<RECOVERY-GAP line 911>>>
// <<<RECOVERY-GAP line 912>>>
// <<<RECOVERY-GAP line 913>>>
// <<<RECOVERY-GAP line 914>>>
// <<<RECOVERY-GAP line 915>>>
// <<<RECOVERY-GAP line 916>>>
// <<<RECOVERY-GAP line 917>>>
// <<<RECOVERY-GAP line 918>>>
// <<<RECOVERY-GAP line 919>>>
// <<<RECOVERY-GAP line 920>>>
// <<<RECOVERY-GAP line 921>>>
// <<<RECOVERY-GAP line 922>>>
// <<<RECOVERY-GAP line 923>>>
// <<<RECOVERY-GAP line 924>>>
// <<<RECOVERY-GAP line 925>>>
// <<<RECOVERY-GAP line 926>>>
// <<<RECOVERY-GAP line 927>>>
// <<<RECOVERY-GAP line 928>>>
// <<<RECOVERY-GAP line 929>>>
// <<<RECOVERY-GAP line 930>>>
// <<<RECOVERY-GAP line 931>>>
// <<<RECOVERY-GAP line 932>>>
// <<<RECOVERY-GAP line 933>>>
// <<<RECOVERY-GAP line 934>>>
// <<<RECOVERY-GAP line 935>>>
// <<<RECOVERY-GAP line 936>>>
// <<<RECOVERY-GAP line 937>>>
// <<<RECOVERY-GAP line 938>>>
// <<<RECOVERY-GAP line 939>>>
// <<<RECOVERY-GAP line 940>>>
// <<<RECOVERY-GAP line 941>>>
// <<<RECOVERY-GAP line 942>>>
// <<<RECOVERY-GAP line 943>>>
// <<<RECOVERY-GAP line 944>>>
// <<<RECOVERY-GAP line 945>>>
// <<<RECOVERY-GAP line 946>>>
// <<<RECOVERY-GAP line 947>>>
// <<<RECOVERY-GAP line 948>>>
// <<<RECOVERY-GAP line 949>>>
// <<<RECOVERY-GAP line 950>>>
// <<<RECOVERY-GAP line 951>>>
// <<<RECOVERY-GAP line 952>>>
// <<<RECOVERY-GAP line 953>>>
// <<<RECOVERY-GAP line 954>>>
// <<<RECOVERY-GAP line 955>>>
// <<<RECOVERY-GAP line 956>>>
// <<<RECOVERY-GAP line 957>>>
// <<<RECOVERY-GAP line 958>>>
// <<<RECOVERY-GAP line 959>>>
// <<<RECOVERY-GAP line 960>>>
// <<<RECOVERY-GAP line 961>>>
// <<<RECOVERY-GAP line 962>>>
// <<<RECOVERY-GAP line 963>>>
// <<<RECOVERY-GAP line 964>>>
// <<<RECOVERY-GAP line 965>>>
// <<<RECOVERY-GAP line 966>>>
// <<<RECOVERY-GAP line 967>>>
// <<<RECOVERY-GAP line 968>>>
// <<<RECOVERY-GAP line 969>>>
// <<<RECOVERY-GAP line 970>>>
// <<<RECOVERY-GAP line 971>>>
// <<<RECOVERY-GAP line 972>>>
// <<<RECOVERY-GAP line 973>>>
// <<<RECOVERY-GAP line 974>>>
// <<<RECOVERY-GAP line 975>>>
// <<<RECOVERY-GAP line 976>>>
// <<<RECOVERY-GAP line 977>>>
// <<<RECOVERY-GAP line 978>>>
// <<<RECOVERY-GAP line 979>>>
// <<<RECOVERY-GAP line 980>>>
// <<<RECOVERY-GAP line 981>>>
// <<<RECOVERY-GAP line 982>>>
// <<<RECOVERY-GAP line 983>>>
// <<<RECOVERY-GAP line 984>>>
// <<<RECOVERY-GAP line 985>>>
// <<<RECOVERY-GAP line 986>>>
// <<<RECOVERY-GAP line 987>>>
// <<<RECOVERY-GAP line 988>>>
// <<<RECOVERY-GAP line 989>>>
// <<<RECOVERY-GAP line 990>>>
// <<<RECOVERY-GAP line 991>>>
// <<<RECOVERY-GAP line 992>>>
// <<<RECOVERY-GAP line 993>>>
// <<<RECOVERY-GAP line 994>>>
// <<<RECOVERY-GAP line 995>>>
// <<<RECOVERY-GAP line 996>>>
// <<<RECOVERY-GAP line 997>>>
// <<<RECOVERY-GAP line 998>>>
// <<<RECOVERY-GAP line 999>>>
// <<<RECOVERY-GAP line 1000>>>
// <<<RECOVERY-GAP line 1001>>>
// <<<RECOVERY-GAP line 1002>>>
// <<<RECOVERY-GAP line 1003>>>
// <<<RECOVERY-GAP line 1004>>>
// <<<RECOVERY-GAP line 1005>>>
// <<<RECOVERY-GAP line 1006>>>
// <<<RECOVERY-GAP line 1007>>>
// <<<RECOVERY-GAP line 1008>>>
// <<<RECOVERY-GAP line 1009>>>
// <<<RECOVERY-GAP line 1010>>>
// <<<RECOVERY-GAP line 1011>>>
// <<<RECOVERY-GAP line 1012>>>
// <<<RECOVERY-GAP line 1013>>>
// <<<RECOVERY-GAP line 1014>>>
// <<<RECOVERY-GAP line 1015>>>
// <<<RECOVERY-GAP line 1016>>>
// <<<RECOVERY-GAP line 1017>>>
// <<<RECOVERY-GAP line 1018>>>
// <<<RECOVERY-GAP line 1019>>>
// <<<RECOVERY-GAP line 1020>>>
// <<<RECOVERY-GAP line 1021>>>
// <<<RECOVERY-GAP line 1022>>>
// <<<RECOVERY-GAP line 1023>>>
// <<<RECOVERY-GAP line 1024>>>
// <<<RECOVERY-GAP line 1025>>>
// <<<RECOVERY-GAP line 1026>>>
// <<<RECOVERY-GAP line 1027>>>
// <<<RECOVERY-GAP line 1028>>>
// <<<RECOVERY-GAP line 1029>>>
// <<<RECOVERY-GAP line 1030>>>
// <<<RECOVERY-GAP line 1031>>>
// <<<RECOVERY-GAP line 1032>>>
// <<<RECOVERY-GAP line 1033>>>
// <<<RECOVERY-GAP line 1034>>>
// <<<RECOVERY-GAP line 1035>>>
// <<<RECOVERY-GAP line 1036>>>
// <<<RECOVERY-GAP line 1037>>>
// <<<RECOVERY-GAP line 1038>>>
// <<<RECOVERY-GAP line 1039>>>
// <<<RECOVERY-GAP line 1040>>>
// <<<RECOVERY-GAP line 1041>>>
// <<<RECOVERY-GAP line 1042>>>
// <<<RECOVERY-GAP line 1043>>>
// <<<RECOVERY-GAP line 1044>>>
// <<<RECOVERY-GAP line 1045>>>
// <<<RECOVERY-GAP line 1046>>>
// <<<RECOVERY-GAP line 1047>>>
// <<<RECOVERY-GAP line 1048>>>
// <<<RECOVERY-GAP line 1049>>>
// <<<RECOVERY-GAP line 1050>>>
// <<<RECOVERY-GAP line 1051>>>
// <<<RECOVERY-GAP line 1052>>>
// <<<RECOVERY-GAP line 1053>>>
// <<<RECOVERY-GAP line 1054>>>
// <<<RECOVERY-GAP line 1055>>>
// <<<RECOVERY-GAP line 1056>>>
// <<<RECOVERY-GAP line 1057>>>
// <<<RECOVERY-GAP line 1058>>>
// <<<RECOVERY-GAP line 1059>>>
// <<<RECOVERY-GAP line 1060>>>
// <<<RECOVERY-GAP line 1061>>>
// <<<RECOVERY-GAP line 1062>>>
// <<<RECOVERY-GAP line 1063>>>
// <<<RECOVERY-GAP line 1064>>>
// <<<RECOVERY-GAP line 1065>>>
// <<<RECOVERY-GAP line 1066>>>
// <<<RECOVERY-GAP line 1067>>>
// <<<RECOVERY-GAP line 1068>>>
// <<<RECOVERY-GAP line 1069>>>
// <<<RECOVERY-GAP line 1070>>>
// <<<RECOVERY-GAP line 1071>>>
// <<<RECOVERY-GAP line 1072>>>
// <<<RECOVERY-GAP line 1073>>>
// <<<RECOVERY-GAP line 1074>>>
// <<<RECOVERY-GAP line 1075>>>
// <<<RECOVERY-GAP line 1076>>>
// <<<RECOVERY-GAP line 1077>>>
// <<<RECOVERY-GAP line 1078>>>
// <<<RECOVERY-GAP line 1079>>>
// <<<RECOVERY-GAP line 1080>>>
// <<<RECOVERY-GAP line 1081>>>
// <<<RECOVERY-GAP line 1082>>>
// <<<RECOVERY-GAP line 1083>>>
// <<<RECOVERY-GAP line 1084>>>
// <<<RECOVERY-GAP line 1085>>>
// <<<RECOVERY-GAP line 1086>>>
// <<<RECOVERY-GAP line 1087>>>
// <<<RECOVERY-GAP line 1088>>>
// <<<RECOVERY-GAP line 1089>>>
// <<<RECOVERY-GAP line 1090>>>
// <<<RECOVERY-GAP line 1091>>>
// <<<RECOVERY-GAP line 1092>>>
// <<<RECOVERY-GAP line 1093>>>
// <<<RECOVERY-GAP line 1094>>>
// <<<RECOVERY-GAP line 1095>>>
// <<<RECOVERY-GAP line 1096>>>
// <<<RECOVERY-GAP line 1097>>>
// <<<RECOVERY-GAP line 1098>>>
// <<<RECOVERY-GAP line 1099>>>
// <<<RECOVERY-GAP line 1100>>>
// <<<RECOVERY-GAP line 1101>>>
// <<<RECOVERY-GAP line 1102>>>
// <<<RECOVERY-GAP line 1103>>>
// <<<RECOVERY-GAP line 1104>>>
// <<<RECOVERY-GAP line 1105>>>
// <<<RECOVERY-GAP line 1106>>>
// <<<RECOVERY-GAP line 1107>>>
// <<<RECOVERY-GAP line 1108>>>
// <<<RECOVERY-GAP line 1109>>>
// <<<RECOVERY-GAP line 1110>>>
// <<<RECOVERY-GAP line 1111>>>
// <<<RECOVERY-GAP line 1112>>>
// <<<RECOVERY-GAP line 1113>>>
// <<<RECOVERY-GAP line 1114>>>
// <<<RECOVERY-GAP line 1115>>>
// <<<RECOVERY-GAP line 1116>>>
// <<<RECOVERY-GAP line 1117>>>
// <<<RECOVERY-GAP line 1118>>>
// <<<RECOVERY-GAP line 1119>>>
// <<<RECOVERY-GAP line 1120>>>
// <<<RECOVERY-GAP line 1121>>>
// <<<RECOVERY-GAP line 1122>>>
// <<<RECOVERY-GAP line 1123>>>
// <<<RECOVERY-GAP line 1124>>>
// <<<RECOVERY-GAP line 1125>>>
// <<<RECOVERY-GAP line 1126>>>
// <<<RECOVERY-GAP line 1127>>>
// <<<RECOVERY-GAP line 1128>>>
// <<<RECOVERY-GAP line 1129>>>
// <<<RECOVERY-GAP line 1130>>>
// <<<RECOVERY-GAP line 1131>>>
// <<<RECOVERY-GAP line 1132>>>
// <<<RECOVERY-GAP line 1133>>>
// <<<RECOVERY-GAP line 1134>>>
// <<<RECOVERY-GAP line 1135>>>
// <<<RECOVERY-GAP line 1136>>>
// <<<RECOVERY-GAP line 1137>>>
// <<<RECOVERY-GAP line 1138>>>
// <<<RECOVERY-GAP line 1139>>>
// <<<RECOVERY-GAP line 1140>>>
// <<<RECOVERY-GAP line 1141>>>
// <<<RECOVERY-GAP line 1142>>>
// <<<RECOVERY-GAP line 1143>>>
// <<<RECOVERY-GAP line 1144>>>
// <<<RECOVERY-GAP line 1145>>>
// <<<RECOVERY-GAP line 1146>>>
// <<<RECOVERY-GAP line 1147>>>
// <<<RECOVERY-GAP line 1148>>>
// <<<RECOVERY-GAP line 1149>>>
// <<<RECOVERY-GAP line 1150>>>
// <<<RECOVERY-GAP line 1151>>>
// <<<RECOVERY-GAP line 1152>>>
// <<<RECOVERY-GAP line 1153>>>
// <<<RECOVERY-GAP line 1154>>>
// <<<RECOVERY-GAP line 1155>>>
// <<<RECOVERY-GAP line 1156>>>
// <<<RECOVERY-GAP line 1157>>>
// <<<RECOVERY-GAP line 1158>>>
// <<<RECOVERY-GAP line 1159>>>
// <<<RECOVERY-GAP line 1160>>>
// <<<RECOVERY-GAP line 1161>>>
// <<<RECOVERY-GAP line 1162>>>
// <<<RECOVERY-GAP line 1163>>>
// <<<RECOVERY-GAP line 1164>>>
// <<<RECOVERY-GAP line 1165>>>
// <<<RECOVERY-GAP line 1166>>>
// <<<RECOVERY-GAP line 1167>>>
// <<<RECOVERY-GAP line 1168>>>
// <<<RECOVERY-GAP line 1169>>>
// <<<RECOVERY-GAP line 1170>>>
// <<<RECOVERY-GAP line 1171>>>
// <<<RECOVERY-GAP line 1172>>>
// <<<RECOVERY-GAP line 1173>>>
// <<<RECOVERY-GAP line 1174>>>
// <<<RECOVERY-GAP line 1175>>>
// <<<RECOVERY-GAP line 1176>>>
// <<<RECOVERY-GAP line 1177>>>
// <<<RECOVERY-GAP line 1178>>>
// <<<RECOVERY-GAP line 1179>>>
// <<<RECOVERY-GAP line 1180>>>
// <<<RECOVERY-GAP line 1181>>>
// <<<RECOVERY-GAP line 1182>>>
// <<<RECOVERY-GAP line 1183>>>
// <<<RECOVERY-GAP line 1184>>>
// <<<RECOVERY-GAP line 1185>>>
// <<<RECOVERY-GAP line 1186>>>
// <<<RECOVERY-GAP line 1187>>>
// <<<RECOVERY-GAP line 1188>>>
// <<<RECOVERY-GAP line 1189>>>
// <<<RECOVERY-GAP line 1190>>>
// <<<RECOVERY-GAP line 1191>>>
// <<<RECOVERY-GAP line 1192>>>
// <<<RECOVERY-GAP line 1193>>>
// <<<RECOVERY-GAP line 1194>>>
// <<<RECOVERY-GAP line 1195>>>
// <<<RECOVERY-GAP line 1196>>>
// <<<RECOVERY-GAP line 1197>>>
// <<<RECOVERY-GAP line 1198>>>
// <<<RECOVERY-GAP line 1199>>>
// <<<RECOVERY-GAP line 1200>>>
// <<<RECOVERY-GAP line 1201>>>
// <<<RECOVERY-GAP line 1202>>>
// <<<RECOVERY-GAP line 1203>>>
// <<<RECOVERY-GAP line 1204>>>
// <<<RECOVERY-GAP line 1205>>>
// <<<RECOVERY-GAP line 1206>>>
// <<<RECOVERY-GAP line 1207>>>
// <<<RECOVERY-GAP line 1208>>>
// <<<RECOVERY-GAP line 1209>>>
// <<<RECOVERY-GAP line 1210>>>
// <<<RECOVERY-GAP line 1211>>>
// <<<RECOVERY-GAP line 1212>>>
// <<<RECOVERY-GAP line 1213>>>
// <<<RECOVERY-GAP line 1214>>>
// <<<RECOVERY-GAP line 1215>>>
// <<<RECOVERY-GAP line 1216>>>
// <<<RECOVERY-GAP line 1217>>>
// <<<RECOVERY-GAP line 1218>>>
// <<<RECOVERY-GAP line 1219>>>
// <<<RECOVERY-GAP line 1220>>>
// <<<RECOVERY-GAP line 1221>>>
// <<<RECOVERY-GAP line 1222>>>
// <<<RECOVERY-GAP line 1223>>>
// <<<RECOVERY-GAP line 1224>>>
// <<<RECOVERY-GAP line 1225>>>
// <<<RECOVERY-GAP line 1226>>>
// <<<RECOVERY-GAP line 1227>>>
// <<<RECOVERY-GAP line 1228>>>
// <<<RECOVERY-GAP line 1229>>>
// <<<RECOVERY-GAP line 1230>>>
// <<<RECOVERY-GAP line 1231>>>
// <<<RECOVERY-GAP line 1232>>>
// <<<RECOVERY-GAP line 1233>>>
// <<<RECOVERY-GAP line 1234>>>
// <<<RECOVERY-GAP line 1235>>>
// <<<RECOVERY-GAP line 1236>>>
// <<<RECOVERY-GAP line 1237>>>
// <<<RECOVERY-GAP line 1238>>>
// <<<RECOVERY-GAP line 1239>>>
// <<<RECOVERY-GAP line 1240>>>
// <<<RECOVERY-GAP line 1241>>>
// <<<RECOVERY-GAP line 1242>>>
// <<<RECOVERY-GAP line 1243>>>
// <<<RECOVERY-GAP line 1244>>>
// <<<RECOVERY-GAP line 1245>>>
// <<<RECOVERY-GAP line 1246>>>
// <<<RECOVERY-GAP line 1247>>>
// <<<RECOVERY-GAP line 1248>>>
// <<<RECOVERY-GAP line 1249>>>
// <<<RECOVERY-GAP line 1250>>>
// <<<RECOVERY-GAP line 1251>>>
// <<<RECOVERY-GAP line 1252>>>
// <<<RECOVERY-GAP line 1253>>>
// <<<RECOVERY-GAP line 1254>>>
// <<<RECOVERY-GAP line 1255>>>
// <<<RECOVERY-GAP line 1256>>>
// <<<RECOVERY-GAP line 1257>>>
// <<<RECOVERY-GAP line 1258>>>
// <<<RECOVERY-GAP line 1259>>>
// <<<RECOVERY-GAP line 1260>>>
// <<<RECOVERY-GAP line 1261>>>
// <<<RECOVERY-GAP line 1262>>>
// <<<RECOVERY-GAP line 1263>>>
// <<<RECOVERY-GAP line 1264>>>
// <<<RECOVERY-GAP line 1265>>>
// <<<RECOVERY-GAP line 1266>>>
// <<<RECOVERY-GAP line 1267>>>
// <<<RECOVERY-GAP line 1268>>>
// <<<RECOVERY-GAP line 1269>>>
// <<<RECOVERY-GAP line 1270>>>
// <<<RECOVERY-GAP line 1271>>>
// <<<RECOVERY-GAP line 1272>>>
// <<<RECOVERY-GAP line 1273>>>
// <<<RECOVERY-GAP line 1274>>>
// <<<RECOVERY-GAP line 1275>>>
// <<<RECOVERY-GAP line 1276>>>
// <<<RECOVERY-GAP line 1277>>>
// <<<RECOVERY-GAP line 1278>>>
// <<<RECOVERY-GAP line 1279>>>
// <<<RECOVERY-GAP line 1280>>>
// <<<RECOVERY-GAP line 1281>>>
// <<<RECOVERY-GAP line 1282>>>
// <<<RECOVERY-GAP line 1283>>>
// <<<RECOVERY-GAP line 1284>>>
// <<<RECOVERY-GAP line 1285>>>
// <<<RECOVERY-GAP line 1286>>>
// <<<RECOVERY-GAP line 1287>>>
// <<<RECOVERY-GAP line 1288>>>
// <<<RECOVERY-GAP line 1289>>>
// <<<RECOVERY-GAP line 1290>>>
// <<<RECOVERY-GAP line 1291>>>
// <<<RECOVERY-GAP line 1292>>>
// <<<RECOVERY-GAP line 1293>>>
// <<<RECOVERY-GAP line 1294>>>
// <<<RECOVERY-GAP line 1295>>>
// <<<RECOVERY-GAP line 1296>>>
// <<<RECOVERY-GAP line 1297>>>
// <<<RECOVERY-GAP line 1298>>>
// <<<RECOVERY-GAP line 1299>>>
// <<<RECOVERY-GAP line 1300>>>
// <<<RECOVERY-GAP line 1301>>>
// <<<RECOVERY-GAP line 1302>>>
// <<<RECOVERY-GAP line 1303>>>
// <<<RECOVERY-GAP line 1304>>>
// <<<RECOVERY-GAP line 1305>>>
// <<<RECOVERY-GAP line 1306>>>
// <<<RECOVERY-GAP line 1307>>>
// <<<RECOVERY-GAP line 1308>>>
// <<<RECOVERY-GAP line 1309>>>
// <<<RECOVERY-GAP line 1310>>>
// <<<RECOVERY-GAP line 1311>>>
// <<<RECOVERY-GAP line 1312>>>
// <<<RECOVERY-GAP line 1313>>>
// <<<RECOVERY-GAP line 1314>>>
// <<<RECOVERY-GAP line 1315>>>
// <<<RECOVERY-GAP line 1316>>>
// <<<RECOVERY-GAP line 1317>>>
// <<<RECOVERY-GAP line 1318>>>
// <<<RECOVERY-GAP line 1319>>>
// <<<RECOVERY-GAP line 1320>>>
// <<<RECOVERY-GAP line 1321>>>
// <<<RECOVERY-GAP line 1322>>>
// <<<RECOVERY-GAP line 1323>>>
// <<<RECOVERY-GAP line 1324>>>
// <<<RECOVERY-GAP line 1325>>>
// <<<RECOVERY-GAP line 1326>>>
// <<<RECOVERY-GAP line 1327>>>
// <<<RECOVERY-GAP line 1328>>>
// <<<RECOVERY-GAP line 1329>>>
// <<<RECOVERY-GAP line 1330>>>
// <<<RECOVERY-GAP line 1331>>>
// <<<RECOVERY-GAP line 1332>>>
// <<<RECOVERY-GAP line 1333>>>
// <<<RECOVERY-GAP line 1334>>>
// <<<RECOVERY-GAP line 1335>>>
// <<<RECOVERY-GAP line 1336>>>
// <<<RECOVERY-GAP line 1337>>>
// <<<RECOVERY-GAP line 1338>>>
// <<<RECOVERY-GAP line 1339>>>
// <<<RECOVERY-GAP line 1340>>>
// <<<RECOVERY-GAP line 1341>>>
// <<<RECOVERY-GAP line 1342>>>
// <<<RECOVERY-GAP line 1343>>>
// <<<RECOVERY-GAP line 1344>>>
// <<<RECOVERY-GAP line 1345>>>
// <<<RECOVERY-GAP line 1346>>>
// <<<RECOVERY-GAP line 1347>>>
// <<<RECOVERY-GAP line 1348>>>
// <<<RECOVERY-GAP line 1349>>>
// <<<RECOVERY-GAP line 1350>>>
// <<<RECOVERY-GAP line 1351>>>
// <<<RECOVERY-GAP line 1352>>>
// <<<RECOVERY-GAP line 1353>>>
// <<<RECOVERY-GAP line 1354>>>
// <<<RECOVERY-GAP line 1355>>>
// <<<RECOVERY-GAP line 1356>>>
// <<<RECOVERY-GAP line 1357>>>
// <<<RECOVERY-GAP line 1358>>>
// <<<RECOVERY-GAP line 1359>>>
// <<<RECOVERY-GAP line 1360>>>
// <<<RECOVERY-GAP line 1361>>>
// <<<RECOVERY-GAP line 1362>>>
// <<<RECOVERY-GAP line 1363>>>
// <<<RECOVERY-GAP line 1364>>>
// <<<RECOVERY-GAP line 1365>>>
// <<<RECOVERY-GAP line 1366>>>
// <<<RECOVERY-GAP line 1367>>>
// <<<RECOVERY-GAP line 1368>>>
// <<<RECOVERY-GAP line 1369>>>
// <<<RECOVERY-GAP line 1370>>>
// <<<RECOVERY-GAP line 1371>>>
// <<<RECOVERY-GAP line 1372>>>
// <<<RECOVERY-GAP line 1373>>>
// <<<RECOVERY-GAP line 1374>>>
// <<<RECOVERY-GAP line 1375>>>
// <<<RECOVERY-GAP line 1376>>>
// <<<RECOVERY-GAP line 1377>>>
// <<<RECOVERY-GAP line 1378>>>
// <<<RECOVERY-GAP line 1379>>>
// <<<RECOVERY-GAP line 1380>>>
// <<<RECOVERY-GAP line 1381>>>
// <<<RECOVERY-GAP line 1382>>>
// <<<RECOVERY-GAP line 1383>>>
// <<<RECOVERY-GAP line 1384>>>
// <<<RECOVERY-GAP line 1385>>>
// <<<RECOVERY-GAP line 1386>>>
// <<<RECOVERY-GAP line 1387>>>
// <<<RECOVERY-GAP line 1388>>>
// <<<RECOVERY-GAP line 1389>>>
// <<<RECOVERY-GAP line 1390>>>
// <<<RECOVERY-GAP line 1391>>>
// <<<RECOVERY-GAP line 1392>>>
// <<<RECOVERY-GAP line 1393>>>
// <<<RECOVERY-GAP line 1394>>>
// <<<RECOVERY-GAP line 1395>>>
// <<<RECOVERY-GAP line 1396>>>
// <<<RECOVERY-GAP line 1397>>>
// <<<RECOVERY-GAP line 1398>>>
// <<<RECOVERY-GAP line 1399>>>
// <<<RECOVERY-GAP line 1400>>>
// <<<RECOVERY-GAP line 1401>>>
// <<<RECOVERY-GAP line 1402>>>
// <<<RECOVERY-GAP line 1403>>>
// <<<RECOVERY-GAP line 1404>>>
// <<<RECOVERY-GAP line 1405>>>
// <<<RECOVERY-GAP line 1406>>>
// <<<RECOVERY-GAP line 1407>>>
// <<<RECOVERY-GAP line 1408>>>
// <<<RECOVERY-GAP line 1409>>>
// <<<RECOVERY-GAP line 1410>>>
// <<<RECOVERY-GAP line 1411>>>
// <<<RECOVERY-GAP line 1412>>>
// <<<RECOVERY-GAP line 1413>>>
// <<<RECOVERY-GAP line 1414>>>
// <<<RECOVERY-GAP line 1415>>>
// <<<RECOVERY-GAP line 1416>>>
// <<<RECOVERY-GAP line 1417>>>
// <<<RECOVERY-GAP line 1418>>>
// <<<RECOVERY-GAP line 1419>>>
// <<<RECOVERY-GAP line 1420>>>
// <<<RECOVERY-GAP line 1421>>>
// <<<RECOVERY-GAP line 1422>>>
// <<<RECOVERY-GAP line 1423>>>
// <<<RECOVERY-GAP line 1424>>>
// <<<RECOVERY-GAP line 1425>>>
// <<<RECOVERY-GAP line 1426>>>
// <<<RECOVERY-GAP line 1427>>>
// <<<RECOVERY-GAP line 1428>>>
// <<<RECOVERY-GAP line 1429>>>
// <<<RECOVERY-GAP line 1430>>>
// <<<RECOVERY-GAP line 1431>>>
// <<<RECOVERY-GAP line 1432>>>
// <<<RECOVERY-GAP line 1433>>>
// <<<RECOVERY-GAP line 1434>>>
// <<<RECOVERY-GAP line 1435>>>
// <<<RECOVERY-GAP line 1436>>>
// <<<RECOVERY-GAP line 1437>>>
// <<<RECOVERY-GAP line 1438>>>
// <<<RECOVERY-GAP line 1439>>>
// <<<RECOVERY-GAP line 1440>>>
// <<<RECOVERY-GAP line 1441>>>
// <<<RECOVERY-GAP line 1442>>>
// <<<RECOVERY-GAP line 1443>>>
// <<<RECOVERY-GAP line 1444>>>
// <<<RECOVERY-GAP line 1445>>>
// <<<RECOVERY-GAP line 1446>>>
// <<<RECOVERY-GAP line 1447>>>
// <<<RECOVERY-GAP line 1448>>>
// <<<RECOVERY-GAP line 1449>>>
// <<<RECOVERY-GAP line 1450>>>
// <<<RECOVERY-GAP line 1451>>>
// <<<RECOVERY-GAP line 1452>>>
// <<<RECOVERY-GAP line 1453>>>
// <<<RECOVERY-GAP line 1454>>>
// <<<RECOVERY-GAP line 1455>>>
// <<<RECOVERY-GAP line 1456>>>
// <<<RECOVERY-GAP line 1457>>>
// <<<RECOVERY-GAP line 1458>>>
// <<<RECOVERY-GAP line 1459>>>
// <<<RECOVERY-GAP line 1460>>>
// <<<RECOVERY-GAP line 1461>>>
// <<<RECOVERY-GAP line 1462>>>
// <<<RECOVERY-GAP line 1463>>>
// <<<RECOVERY-GAP line 1464>>>
// <<<RECOVERY-GAP line 1465>>>
// <<<RECOVERY-GAP line 1466>>>
// <<<RECOVERY-GAP line 1467>>>
// <<<RECOVERY-GAP line 1468>>>
// <<<RECOVERY-GAP line 1469>>>
// <<<RECOVERY-GAP line 1470>>>
// <<<RECOVERY-GAP line 1471>>>
// <<<RECOVERY-GAP line 1472>>>
// <<<RECOVERY-GAP line 1473>>>
// <<<RECOVERY-GAP line 1474>>>
// <<<RECOVERY-GAP line 1475>>>
// <<<RECOVERY-GAP line 1476>>>
// <<<RECOVERY-GAP line 1477>>>
// <<<RECOVERY-GAP line 1478>>>
// <<<RECOVERY-GAP line 1479>>>
// <<<RECOVERY-GAP line 1480>>>
// <<<RECOVERY-GAP line 1481>>>
// <<<RECOVERY-GAP line 1482>>>
// <<<RECOVERY-GAP line 1483>>>
// <<<RECOVERY-GAP line 1484>>>
// <<<RECOVERY-GAP line 1485>>>
// <<<RECOVERY-GAP line 1486>>>
// <<<RECOVERY-GAP line 1487>>>
// <<<RECOVERY-GAP line 1488>>>
// <<<RECOVERY-GAP line 1489>>>
// <<<RECOVERY-GAP line 1490>>>
// <<<RECOVERY-GAP line 1491>>>
               // WATCHDOG / cold-mmap pre-warm.
               //
               // `apply_weights_inner` runs the per-layer `finalize_in_proj` /
               // `finalize_gate_up` precompute — a `concatenate` + `transpose` +
               // `eval` whose GPU kernel reads the gate/up/qkv/z weights DIRECTLY
               // from the lazy mmap-backed checkpoint. On a slow / cold mmap
               // checkpoint. On a slow / cold mmap source (e.g. a model served
               // from a USB SSD) the GPU command buffer stalls on the page fault
               // can exceed the macOS GPU command-buffer watchdog (~5 s) → an
               // uncatchable `kIOGPUCommandBufferCallbackErrorTimeout` abort
               // `kIOGPUCommandBufferCallbackErrorTimeout` abort mid-load. The
               // CPU device + stream has direct access to the mmap'd pages and is
               // Fix: materialize every weight ON THE CPU STREAM first. The CPU
               // path; see `CpuConvertGuard` in `convert.rs`). The guard restores
               // the prior GPU device + stream on drop, before inference runs.
               // `CpuConvertGuard` in `convert.rs`); a slow read just takes
                   let _cpu_guard = crate::convert::CpuConvertGuard::enter_cpu();
               // precompute below runs entirely in the warm regime — no page
               // faults inside a command buffer — so it stays fast (the eager
               // GPU transposes are ~free) AND safe. This replaces the old
               // post-`apply` materialize: warming up front is what makes
               // `apply` itself watchdog-safe, and re-materializing afterwards
                       quant_group_size,
                       top_level_mode,
                   let _cpu_guard = crate::convert::CpuConvertGuard::enter_cpu();
                   let arrays: Vec<&MxArray> = params.values().collect();
                   crate::inference_trace::write(format_args!(
               crate::inference_trace::write(format_args!("[MLX_TRACE] FQ stage apply_done"));
                       arrays.len()
               // Register weights with C++. The dense compiled graph now
               // dispatches via the registry-aware `linear_proj` helper,
               // which keys off the per-projection quant-info entries this
               // call populates alongside the weights themselves. With
               // Apply weights (GPU finalize precompute reads now-resident pages).
               // explicitly, MXFP4 / MXFP8 / NVFP4 / affine checkpoints
               // all take the compiled decode path — no more Rust
               // forward-path fallback bypass.
               register_weights_with_cpp(
                   quant_bits,
                   quant_group_size,
                   top_level_mode,
                   &per_layer_quant,
                   quant_bits,
               crate::inference_trace::write(format_args!("[MLX_TRACE] FQ stage apply_done"));
               );
               crate::inference_trace::write(format_args!("[MLX_TRACE] FQ stage register_done"));
               // dispatches via the registry-aware `linear_proj` helper,
               // which keys off the per-projection quant-info entries this
               // exposure as the finalize precompute above: a 2 GB embedding /
               // lm_head leaf eval over a cold USB-mmap source is run on the CPU
               // stream (watchdog-immune) rather than risking a GPU stall.
               // all take the compiled decode path — no more Rust
                   let _cpu_guard = crate::convert::CpuConvertGuard::enter_cpu();
                   let arrays: Vec<&MxArray> = params.values().collect();
                   crate::inference_trace::write(format_args!(
                       "[MLX_TRACE] FQ stage before_materialize arrays={}",
                       arrays.len()
                   &per_layer_quant,
                   crate::array::memory::materialize_weights(&arrays)?;
               }

               // Set tokenizer
