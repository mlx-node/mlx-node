// Jacobian-lens OFFLINE fit job (Task T1.2) — native-only (`#[cfg(feature = "full")]`).
//
// This is the estimator side of the J-lens: it FITS the per-boundary Jacobian
// matrices `J_l` that T1.1's `lensReadout(useJacobian=true)` consumes. It is a
// faithful MLX/Rust port of the reference estimator
// (`anthropics/jacobian-lens` `jlens/fitting.py::jacobian_for_prompt`), NOT a
// new algorithm.
//
// ## The estimator (ported verbatim)
//
//   lens_l(h) = unembed( final_norm( J_l · h ) )
//
// For each output dimension `i` we inject a one-hot cotangent at EVERY valid
// target position at once and backprop. Causality makes the gradient at source
// position `p` equal `Σ_{p' ≥ p} ∂h_final[p'] / ∂h_l[p]`; we take the mean over
// source positions `p`, then (across prompts) an `n_prompts`-weighted mean.
// Skip the first `skip_first` (=16) source positions (attention sinks) and the
// final position (no next-token target).
//
// ## The MLX analog of the PyTorch forward hooks (design D4)
//
// Instead of hooks we insert 24 zero-valued probe tensors `z_l`, one added to
// the residual at each non-final boundary `h_l` (`0..=23`), in a probe-inserting
// forward ([`functional::qwen3_5_probe_forward_hidden_states`]). The probes are
// the `value_and_grad` differentiation inputs; because `z_l` is added to `h_l`,
// `∂h_final/∂z_l == ∂h_final/∂h_l == J_l`. ONE backward yields the contribution
// for ALL 24 boundaries. The 1024 output basis rows are batched through the
// batch dim (`dim_batch`, default 8) ⇒ `ceil(hidden/dim_batch)` backwards per
// prompt. Probes are added OUTSIDE any `checkpoint_apply` wrapper; the forward
// STOPS pre-final-norm (target = `h_24`).
//
// ## Orientation (must agree with T1.1)
//
// `J_l[i, j] = ∂h_final,i / ∂h_l,j`, shape `[hidden, hidden]`, saved as the
// safetensors tensor `J.{l}` so T1.1's `loadLensPackFromFile` reads it verbatim.
// T1.1 applies `x = h · Jᵀ` (row-vector convention) — the readout transposes,
// so the two conventions agree (reference `transport` is likewise `h @ J.T`).
// `J.24` is the identity by definition and is NEVER stored.
//
// ## Numerics (design D5)
//
// Params stay bf16; each grad contribution is cast to f32 BEFORE accumulating;
// running mean; a NaN/Inf guard skips a poisoned prompt. `heavy_cleanup()` runs
// between prompts (and periodically between backward passes) to release the
// compiled-graph cache — the fit stays OFF every ChatSession/paged-cache path
// (whose vjp throws).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::{DType, MxArray, heavy_cleanup};
use crate::autograd;
use crate::models::qwen3_5::Qwen3_5Config;
use crate::utils::functional;
use crate::utils::safetensors::{SafeTensorsFile, save_safetensors};

/// Default output basis dims computed per backward pass (design D4).
const DEFAULT_DIM_BATCH: usize = 8;
/// Default leading source positions to exclude (reference `SKIP_FIRST_N_POSITIONS`).
const DEFAULT_SKIP_FIRST: usize = 16;
/// Default per-prompt token truncation (reference `max_seq_len`; the paper uses
/// equal-length 128-token prompts).
const DEFAULT_MAX_SEQ_LEN: usize = 128;
/// Random directions probed (in addition to the gradient direction) by the
/// finite-difference gate, so it is an all-direction gradcheck rather than a
/// single projection along the implementation's own gradient.
const N_RAND_DIRS: usize = 3;

/// Options for [`fit_jacobian_lens`] / the `fitJacobianLens` NAPI method.
#[napi(object)]
pub struct JlensFitOptions {
    /// One entry per prompt: pre-tokenized input ids (u32). Each is truncated to
    /// `maxSeqLen`. Prompts too short to leave any valid position are skipped.
    pub prompt_ids: Vec<Vec<u32>>,
    /// Output basis dims per backward pass (higher = fewer passes, more memory).
    /// Default 8. Total backward FLOPs are `dimBatch`-independent.
    pub dim_batch: Option<u32>,
    /// Leading source positions to exclude. Default 16.
    pub skip_first: Option<u32>,
    /// Truncate each prompt to this many tokens. Default 128.
    pub max_seq_len: Option<u32>,
    /// fp32 master checkpoint path (atomic write-temp-then-rename; resumable).
    pub checkpoint_path: String,
    /// If set, export a `J.{layer}` safetensors pack loadable by T1.1's
    /// `loadLensPackFromFile` (the running mean = `jacobian_sum / n_done`).
    pub pack_path: Option<String>,
    /// Gradient checkpointing (probes stay outside the wrapper either way).
    /// Default false (simplest correct path for the pilot).
    pub use_checkpointing: Option<bool>,
    /// Resume from `checkpointPath` if it exists. Default true.
    pub resume: Option<bool>,
}

/// Result of [`fit_jacobian_lens`].
#[napi(object)]
pub struct JlensFitResult {
    /// Total prompts accumulated into the checkpoint (incl. any resumed).
    pub n_done: u32,
    /// Prompts actually fit THIS invocation (excludes resumed).
    pub prompts_this_run: u32,
    /// Prompts skipped (too short / NaN-guarded) this invocation.
    pub prompts_skipped: u32,
    /// Wall-clock seconds per prompt fit THIS run — prices the T1.3 corpus run.
    pub sec_per_prompt: f64,
    pub peak_memory_mb: f64,
    pub active_memory_mb: f64,
    /// Valid source positions of the last fit prompt.
    pub n_valid_last: i32,
    /// Diagnostic: `max_l ||mean J_l||_F / sqrt(hidden)` (reference's convergence
    /// flag; typically O(1) for a healthy fit).
    pub max_j_norm_over_sqrt_d: f64,
    pub checkpoint_path: String,
    pub pack_path: Option<String>,
}

/// Result of [`verify_finite_diff`] — THE pilot GO/NO-GO gate. Per requested
/// boundary: the autograd directional derivative vs the numerical
/// central-difference, probed along the gradient direction AND several random
/// directions (see `dir_check_max_err`), plus a causality and basis-batch check.
#[napi(object)]
pub struct JlensFiniteDiffResult {
    pub boundaries: Vec<u32>,
    pub epsilon: f64,
    /// Autograd directional derivative `⟨∂L/∂z_l, δ⟩`.
    pub predicted: Vec<f64>,
    /// Numerical `(L(z_l=+εδ) − L(z_l=−εδ)) / (2ε)`.
    pub numerical: Vec<f64>,
    /// `|predicted − numerical| / max(|numerical|, 1e-6)` along the gradient
    /// direction (max SNR).
    pub rel_error: Vec<f64>,
    /// Per boundary: `max` over `1 + N_RAND_DIRS` probe directions (gradient +
    /// random) of `|predicted_d − numerical_d| / ‖g_l‖`. Normalizing by `‖g_l‖`
    /// keeps this well-conditioned for ALL directions, so a VJP error orthogonal
    /// to the gradient still surfaces. This is the rigorous all-direction check.
    pub dir_check_max_err: Vec<f64>,
    /// Directions probed per boundary (`1 + N_RAND_DIRS`).
    pub dir_check_n: i32,
    /// `true` iff boundary `l`'s downstream block is GatedDeltaNet (linear).
    pub is_gdn: Vec<bool>,
    /// Causality probe split position `p` (a mid-sequence position). A probe
    /// perturbation injected ONLY at position `p` must not change `h_final`
    /// before `p` (the Σ_{t'≥t} causal assumption the autograd gate is blind to).
    pub causality_split_pos: i32,
    /// Per boundary: `||Δh_final[:, <p, :]||_F` under a single-position-`p`
    /// probe perturbation. Must be ~0 (causal — earlier positions unchanged).
    pub causality_leak_before: Vec<f64>,
    /// Per boundary: `||Δh_final[:, ≥p, :]||_F`. Must be > 0 (the perturbation
    /// actually did something — proves the leak check has teeth).
    pub causality_signal_after: Vec<f64>,
    /// Basis-batch consistency (the #1 MLX value_and_grad mis-port risk): the
    /// boundary and output dim at which one `J_l` row was computed both solo
    /// (`bv=1`) and inside a `bv=8` batch.
    pub batch_consistency_boundary: u32,
    pub batch_consistency_out_dim: i32,
    /// `max_j |J_solo[i,j] − J_batched[i,j]|`. Must be ~0 (a broadcast/shared
    /// probe would sum the basis rows and corrupt this).
    pub batch_consistency_max_abs_diff: f64,
    /// `||J_solo[i] − J_batched[i]||_F / ||J_solo[i]||_F`.
    pub batch_consistency_rel: f64,
}

// ---------------------------------------------------------------------------
// Per-prompt estimator (port of `jacobian_for_prompt`)
// ---------------------------------------------------------------------------

/// Compute the per-boundary Jacobian estimator for one prompt.
///
/// Returns `(jacobians, n_valid, peak_mb, active_mb)` where `jacobians[l]` is the
/// `[hidden, hidden]` fp32 `J_l = ∂h_final/∂h_l` for boundary `l`
/// (`0..=num_layers-1`), `n_valid` is the number of valid source positions, and
/// `peak_mb`/`active_mb` are the per-pass memory high-water marks.
fn jacobian_for_prompt(
    config: &Qwen3_5Config,
    params: &HashMap<String, MxArray>,
    prompt_ids: &[i32],
    dim_batch: usize,
    skip_first: usize,
    use_checkpointing: bool,
) -> Result<(Vec<MxArray>, i64, f64, f64)> {
    let hidden = config.hidden_size as usize;
    let num_layers = config.num_layers as usize;
    let seq_len = prompt_ids.len();

    // Valid target/source positions: [skip_first, seq_len - 1). The final
    // position has no next-token target; the first `skip_first` are sinks.
    if seq_len < skip_first + 2 {
        return Err(Error::from_reason(format!(
            "prompt too short: seq_len={} needs > {} tokens",
            seq_len,
            skip_first + 1
        )));
    }
    let valid_start = skip_first as i64;
    let valid_end = (seq_len - 1) as i64; // exclusive
    let n_valid = valid_end - valid_start;

    let bv = dim_batch;
    // Replicate the prompt `bv` times along the batch axis so each backward
    // computes `bv` output-basis rows at once (same FLOPs, amortized dispatch).
    let mut ids_rep: Vec<i32> = Vec::with_capacity(bv * seq_len);
    for _ in 0..bv {
        ids_rep.extend_from_slice(prompt_ids);
    }
    let input_ids = MxArray::from_int32(&ids_rep, &[bv as i64, seq_len as i64])?;

    // Probe dtype matches the residual dtype (params' embedding dtype).
    let probe_dtype = params
        .get("embedding.weight")
        .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?
        .dtype()?;
    let probe_shape = [bv as i64, seq_len as i64, hidden as i64];

    // 24 zero probes reused across passes — the Jacobian is evaluated at z=0.
    let probes: Vec<MxArray> = (0..num_layers)
        .map(|_| MxArray::zeros(&probe_shape, Some(probe_dtype)))
        .collect::<Result<Vec<_>>>()?;

    // Per-boundary row chunks (each `[n_dims, hidden]` fp32), concatenated at end.
    let mut chunks: Vec<Vec<MxArray>> = (0..num_layers).map(|_| Vec::new()).collect();

    // High-water marks sampled after pass 0's forward+backward — a representative
    // per-pass peak, captured BEFORE any periodic `heavy_cleanup()` resets the
    // MLX peak counter.
    let mut pass_peak_mb = 0.0f64;
    let mut pass_active_mb = 0.0f64;

    let n_passes = hidden.div_ceil(bv);
    for pass in 0..n_passes {
        let dim_start = pass * bv;
        let n_dims = std::cmp::min(bv, hidden - dim_start);

        // Cotangent selector C[b, dim_start+b] = 1 (b < n_dims) — the one-hot
        // output basis for this pass, broadcast over valid target positions.
        let mut c_host = vec![0f32; bv * hidden];
        for (b, row) in c_host.chunks_exact_mut(hidden).enumerate().take(n_dims) {
            row[dim_start + b] = 1.0;
        }
        let c = MxArray::from_float32(&c_host, &[bv as i64, hidden as i64])?;
        let c_bcast = c.expand_dims(1)?; // [bv, 1, hidden]

        let probe_refs: Vec<&MxArray> = probes.iter().collect();

        // loss = Σ_b Σ_{t valid} h_final[b, t, dim_start+b]  (a scalar). Its
        // grad w.r.t. z_l is exactly the reference's `grad_outputs=cotangent`
        // VJP, evaluated at z=0.
        let config_c = config.clone();
        let params_c = params.clone();
        let input_ids_c = input_ids.clone();
        let c_c = c_bcast.clone();
        let ckpt = Rc::new(RefCell::new(autograd::CheckpointContexts::new()));
        let ckpt_in = ckpt.clone();
        let loss_fn = move |p: &[MxArray]| -> Result<MxArray> {
            let h_final = functional::qwen3_5_probe_forward_hidden_states(
                &config_c,
                &params_c,
                &input_ids_c,
                p,
                use_checkpointing,
                &mut ckpt_in.borrow_mut(),
            )?; // [bv, T, hidden]
            let h_valid = h_final.slice_axis(1, valid_start, valid_end)?; // [bv, n_valid, hidden]
            let prod = h_valid.astype(DType::Float32)?.mul(&c_c)?;
            prod.sum(None, None) // scalar
        };

        let (loss, grads) = autograd::value_and_grad(probe_refs, loss_fn)?;
        loss.eval();
        for g in &grads {
            g.eval();
        }
        ckpt.borrow_mut().reclaim();

        if pass == 0 {
            use crate::array::memory::{get_active_memory, get_peak_memory};
            pass_peak_mb = get_peak_memory() / 1e6;
            pass_active_mb = get_active_memory() / 1e6;
        }

        // Row b of J_l at output dim (dim_start+b) = mean over valid source
        // positions of grad_l (cast to f32 BEFORE the mean/accumulate).
        for (l, chunk) in chunks.iter_mut().enumerate() {
            let g_valid = grads[l].slice_axis(1, valid_start, valid_end)?; // [bv, n_valid, hidden]
            let rows = g_valid
                .astype(DType::Float32)?
                .mean(Some(&[1]), Some(false))?; // [bv, hidden]
            let rows = if n_dims == bv {
                rows
            } else {
                rows.slice(&[0, 0], &[n_dims as i64, hidden as i64])? // [n_dims, hidden]
            };
            rows.eval();
            chunk.push(rows);
        }
        drop(grads);

        // Release the compiled-graph cache periodically so peak stays bounded
        // across the ~128 backward passes.
        if pass % 16 == 15 {
            heavy_cleanup();
        }
    }

    // Concatenate the ordered row chunks per boundary -> [hidden, hidden] fp32.
    let mut jacobians = Vec::with_capacity(num_layers);
    for chunk in &chunks {
        let refs: Vec<&MxArray> = chunk.iter().collect();
        let j = MxArray::concatenate_many(refs, Some(0))?;
        j.eval();
        jacobians.push(j);
    }

    Ok((jacobians, n_valid, pass_peak_mb, pass_active_mb))
}

// ---------------------------------------------------------------------------
// fp32 master checkpoint (atomic, resumable) + f16/f32 pack export
// ---------------------------------------------------------------------------

struct FitState {
    /// Running SUM of per-prompt `J_l` (fp32 `[hidden, hidden]`), one per boundary.
    jacobian_sum: Vec<MxArray>,
    /// Prompts accumulated so far.
    n_done: i64,
    /// Prompt-list index to resume from (tracked separately so a skipped
    /// too-short prompt is not re-processed on resume).
    next_idx: i64,
}

fn zero_state(num_layers: usize, hidden: usize) -> Result<FitState> {
    let jacobian_sum = (0..num_layers)
        .map(|_| MxArray::zeros(&[hidden as i64, hidden as i64], Some(DType::Float32)))
        .collect::<Result<Vec<_>>>()?;
    Ok(FitState {
        jacobian_sum,
        n_done: 0,
        next_idx: 0,
    })
}

/// Constant fingerprint of the corpus a checkpoint was fitted on. Persisted in
/// the checkpoint and revalidated on resume so the SAME `checkpointPath` can't
/// be silently reused with a DIFFERENT prompt corpus or `maxSeqLen`. Without
/// this, resuming skips every prompt with `idx < next_idx` and exports a
/// stale/mixed running mean as a fresh J pack — a corruption path, not just bad
/// UX (the driver skips by index and `export_pack` runs whenever `n_done > 0`).
///
/// NOTE: this guards ONLY the resume-into-the-same-checkpoint path. T1.3's
/// future cross-shard merge deliberately combines DIFFERENT disjoint corpora by
/// summing `Jsum.{l}` + `n_done` — a separate code path that must NOT be gated
/// by this per-checkpoint check.
#[derive(Clone, Copy)]
struct CorpusFingerprint {
    /// Per-prompt truncation length the corpus was hashed under.
    max_seq_len: usize,
    /// Number of prompts in the corpus (list length).
    prompt_count: usize,
    /// FNV-1a/64 over the TRUNCATED token-id sequences (see [`corpus_fingerprint`]).
    hash: u64,
}

/// Deterministic, order- and content-sensitive fingerprint of the corpus being
/// fit: FNV-1a/64 over, per prompt in list order, the truncated (to
/// `max_seq_len`) token-id sequence hashed as `(u32 length, then each u32 id)`.
/// Cheap (a single pass over the ids) and matches EXACTLY on a legitimate resume
/// of the identical corpus, so the resume round-trip is preserved.
fn corpus_fingerprint(prompt_ids: &[Vec<u32>], max_seq_len: usize) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    #[inline]
    fn mix(h: &mut u64, v: u32) {
        for b in v.to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(FNV_PRIME);
        }
    }
    let mut h = FNV_OFFSET;
    for ids in prompt_ids {
        let take = ids.len().min(max_seq_len);
        mix(&mut h, take as u32);
        for &id in &ids[..take] {
            mix(&mut h, id);
        }
    }
    h
}

/// Load a fit checkpoint written by [`save_checkpoint`]. Returns `Ok(None)` if
/// the file does not exist. Validates `skip_first` / `hidden` AND the corpus
/// fingerprint (`max_seq_len` / `prompt_count` / `corpus_hash`) against the run,
/// so a mismatched resume errors BEFORE any fit work rather than corrupting the
/// running mean.
fn load_checkpoint(
    path: &str,
    num_layers: usize,
    hidden: usize,
    skip_first: usize,
    fp: &CorpusFingerprint,
) -> Result<Option<FitState>> {
    if !std::path::Path::new(path).exists() {
        return Ok(None);
    }
    let file = SafeTensorsFile::load(path)?;
    let tensors = file.load_tensors(path)?;

    let read_meta = |key: &str| -> Result<i64> {
        let t = tensors
            .get(key)
            .ok_or_else(|| Error::from_reason(format!("checkpoint {path} missing '{key}'")))?;
        Ok(t.item_at_float32(0)? as i64)
    };
    let ckpt_hidden = read_meta("meta.hidden")?;
    let ckpt_skip = read_meta("meta.skip_first")?;
    if ckpt_hidden != hidden as i64 {
        return Err(Error::from_reason(format!(
            "checkpoint {path} was fitted with hidden={ckpt_hidden}, not {hidden}; \
             pass resume=false to discard it"
        )));
    }
    if ckpt_skip != skip_first as i64 {
        return Err(Error::from_reason(format!(
            "checkpoint {path} was fitted with skip_first={ckpt_skip}, not {skip_first}; \
             pass resume=false to discard it"
        )));
    }

    // Corpus fingerprint: reject a resume against a DIFFERENT corpus or
    // maxSeqLen BEFORE loading Jsum / doing any fit work. Resuming would
    // otherwise skip prompts by index and export a stale/mixed J pack.
    let ckpt_max_seq = read_meta("meta.max_seq_len")?;
    if ckpt_max_seq != fp.max_seq_len as i64 {
        return Err(Error::from_reason(format!(
            "checkpoint {path} was fitted with maxSeqLen={ckpt_max_seq}, not {}; \
             pass resume=false to discard it",
            fp.max_seq_len
        )));
    }
    let ckpt_prompt_count = read_meta("meta.prompt_count")?;
    if ckpt_prompt_count != fp.prompt_count as i64 {
        return Err(Error::from_reason(format!(
            "checkpoint {path} was fitted on {ckpt_prompt_count} prompts, not {}; \
             it belongs to a different corpus — pass resume=false to discard it",
            fp.prompt_count
        )));
    }
    // 64-bit hash stored LOSSLESSLY as a U32 [2] (hi, lo) tensor — read exactly.
    let hash_t = tensors.get("meta.corpus_hash").ok_or_else(|| {
        Error::from_reason(format!("checkpoint {path} missing 'meta.corpus_hash'"))
    })?;
    let ckpt_hash =
        ((hash_t.item_at_uint32(0)? as u64) << 32) | (hash_t.item_at_uint32(1)? as u64);
    if ckpt_hash != fp.hash {
        return Err(Error::from_reason(format!(
            "checkpoint {path} was fitted with a different prompt corpus (or maxSeqLen); \
             pass resume=false to discard it"
        )));
    }

    let mut jacobian_sum = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let key = format!("Jsum.{l}");
        let t = tensors
            .get(&key)
            .ok_or_else(|| Error::from_reason(format!("checkpoint {path} missing '{key}'")))?
            .astype(DType::Float32)?;
        t.eval();
        jacobian_sum.push(t);
    }
    Ok(Some(FitState {
        jacobian_sum,
        n_done: read_meta("meta.n_done")?,
        next_idx: read_meta("meta.next_idx")?,
    }))
}

/// Atomically write the fit checkpoint: `Jsum.{l}` fp32 + `meta.*` scalars +
/// the corpus fingerprint, written to `{path}.tmp.{pid}` then renamed so a crash
/// never leaves a half-written checkpoint.
fn save_checkpoint(
    path: &str,
    state: &FitState,
    skip_first: usize,
    hidden: usize,
    fp: &CorpusFingerprint,
) -> Result<()> {
    let mut tensors: HashMap<String, MxArray> = HashMap::new();
    for (l, j) in state.jacobian_sum.iter().enumerate() {
        tensors.insert(format!("Jsum.{l}"), j.clone());
    }
    // Small integer scalars: f32 [1] is exact for all of these (skip_first,
    // hidden, maxSeqLen, prompt_count are all << 2^24).
    let meta_i = |v: i64| MxArray::from_float32(&[v as f32], &[1]);
    tensors.insert("meta.n_done".into(), meta_i(state.n_done)?);
    tensors.insert("meta.next_idx".into(), meta_i(state.next_idx)?);
    tensors.insert("meta.skip_first".into(), meta_i(skip_first as i64)?);
    tensors.insert("meta.hidden".into(), meta_i(hidden as i64)?);
    tensors.insert("meta.max_seq_len".into(), meta_i(fp.max_seq_len as i64)?);
    tensors.insert("meta.prompt_count".into(), meta_i(fp.prompt_count as i64)?);
    // The 64-bit corpus hash MUST be stored losslessly: an f32 [1] rounds any
    // value above 2^24 and would break the resume round-trip. Split into a U32
    // [2] (hi, lo) tensor — U32 round-trips exactly through save/load_safetensors.
    let hash_hi = (fp.hash >> 32) as u32;
    let hash_lo = (fp.hash & 0xffff_ffff) as u32;
    tensors.insert(
        "meta.corpus_hash".into(),
        MxArray::from_uint32(&[hash_hi, hash_lo], &[2])?,
    );

    let tmp = format!("{path}.tmp.{}", std::process::id());
    save_safetensors(&tmp, &mut tensors, None)?;
    std::fs::rename(&tmp, path).map_err(|e| {
        Error::from_reason(format!("checkpoint rename {tmp} -> {path} failed: {e}"))
    })?;
    Ok(())
}

/// Export the running-mean pack `J.{l} = jacobian_sum[l] / n_done` as
/// safetensors loadable by T1.1's `loadLensPackFromFile`.
///
/// BOUNDARY CONVENTION (must agree with T1.1's `forward_capture_hidden` +
/// readout, verified against model.rs): the readout applies `pack[l]` to the
/// residual captured at boundary `l`, where boundary 0 = embedding output and
/// boundary `l` (1..=23) = output of block `l-1` (= input to block `l`), and
/// boundary 24 = the final hidden state. `jacobian_sum[l] = ∂h_24/∂(boundary l)`
/// (the probe `probes[l]` is added at boundary `l`), so key `J.{l}` is already
/// boundary-aligned — NO off-by-one.
///
/// We emit EXACTLY `J.1..J.23` (23 matrices): boundary 0 (the raw embedding
/// output) is left UNFITTED to match the reference (`fitting.py` starts at the
/// first block's output), and `J.24` (final boundary = identity) is NEVER
/// written (the loader rejects it).
///
/// CALLER CONTRACT: since the pack omits `J.0`, a `lensReadout(useJacobian=true)`
/// MUST pass `layers` within the fitted set `1..=24` — the readout's DEFAULT
/// `layers` is all 25 boundaries, which would request `J.0` and error. This
/// matches the reference `JacobianLens.apply`, whose default `layers` is the
/// fitted `source_layers`, not every boundary.
fn export_pack(path: &str, state: &FitState) -> Result<()> {
    if state.n_done == 0 {
        return Err(Error::from_reason(
            "export_pack: n_done == 0 (no prompts fit)",
        ));
    }
    let mut tensors: HashMap<String, MxArray> = HashMap::new();
    // Skip boundary 0 (embedding output, unfitted). `jacobian_sum.len()` is
    // `num_layers` (24), so `1..len` == boundaries 1..=23.
    for l in 1..state.jacobian_sum.len() {
        let mean = state.jacobian_sum[l].div_scalar(state.n_done as f64)?;
        mean.eval();
        tensors.insert(format!("J.{l}"), mean);
    }
    let tmp = format!("{path}.tmp.{}", std::process::id());
    save_safetensors(&tmp, &mut tensors, None)?;
    std::fs::rename(&tmp, path)
        .map_err(|e| Error::from_reason(format!("pack rename {tmp} -> {path} failed: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Fit driver (port of `fit`)
// ---------------------------------------------------------------------------

/// Fit `J_l` over the prompts in `opts`, accumulating into a resumable fp32
/// master checkpoint and (optionally) exporting a loadable pack. `params` are
/// the frozen bf16 model weights (from `get_parameters_sync`); the fit
/// differentiates only w.r.t. the injected probes.
pub(crate) fn fit_jacobian_lens(
    config: &Qwen3_5Config,
    params: &HashMap<String, MxArray>,
    opts: JlensFitOptions,
) -> Result<JlensFitResult> {
    use crate::array::memory::reset_peak_memory;

    reset_peak_memory();

    let hidden = config.hidden_size as usize;
    let num_layers = config.num_layers as usize;
    let dim_batch = opts
        .dim_batch
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_DIM_BATCH);
    let skip_first = opts
        .skip_first
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_SKIP_FIRST);
    let max_seq_len = opts
        .max_seq_len
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MAX_SEQ_LEN);
    let use_checkpointing = opts.use_checkpointing.unwrap_or(false);
    let resume = opts.resume.unwrap_or(true);

    if dim_batch == 0 {
        return Err(Error::from_reason("fitJacobianLens: dimBatch must be >= 1"));
    }

    // Corpus fingerprint for the whole run — persisted in every checkpoint write
    // and revalidated on resume so this `checkpointPath` can't be reused with a
    // different corpus / maxSeqLen (which would skip prompts by index and export
    // a stale/mixed J pack). Computed once (constant across the fit).
    let fp = CorpusFingerprint {
        max_seq_len,
        prompt_count: opts.prompt_ids.len(),
        hash: corpus_fingerprint(&opts.prompt_ids, max_seq_len),
    };

    // Resume or start fresh.
    let mut state = if resume {
        match load_checkpoint(&opts.checkpoint_path, num_layers, hidden, skip_first, &fp)? {
            Some(s) => s,
            None => zero_state(num_layers, hidden)?,
        }
    } else {
        zero_state(num_layers, hidden)?
    };

    let mut prompts_this_run: u32 = 0;
    let mut prompts_skipped: u32 = 0;
    let mut n_valid_last: i64 = 0;
    let mut elapsed_secs: f64 = 0.0;
    // High-water marks sampled per prompt BEFORE `heavy_cleanup()` (which resets
    // the MLX peak counter), so the reported peak reflects the real fit cost.
    let mut peak_mb: f64 = 0.0;
    let mut active_mb: f64 = 0.0;

    for (idx, ids) in opts.prompt_ids.iter().enumerate() {
        if (idx as i64) < state.next_idx {
            continue; // already accumulated in a prior run
        }
        let take = ids.len().min(max_seq_len);
        let ids_i32: Vec<i32> = ids[..take].iter().map(|&x| x as i32).collect();

        // Too-short prompt (no valid source position) is a LEGITIMATE skip, not
        // an error. Detect it up front so that any Err from `jacobian_for_prompt`
        // below is a REAL failure (missing params, shape/graph error, a broken
        // autograd path) and PROPAGATES — the pilot must never silently skip a
        // broken fit and export a stale running mean (fail-open).
        if ids_i32.len() < skip_first + 2 {
            tracing::warn!(
                "jlens fit: skipping prompt {idx}: too short (len {} <= skip_first+1={})",
                ids_i32.len(),
                skip_first + 1
            );
            prompts_skipped += 1;
            state.next_idx = idx as i64 + 1;
            heavy_cleanup();
            // Persist the advanced cursor so a crash doesn't reprocess this skip.
            save_checkpoint(&opts.checkpoint_path, &state, skip_first, hidden, &fp)?;
            continue;
        }

        let start = Instant::now();
        // Real errors propagate (`?`) — do NOT swallow them as a skip.
        let (per_j, n_valid, prompt_peak_mb, prompt_active_mb) = jacobian_for_prompt(
            config,
            params,
            &ids_i32,
            dim_batch,
            skip_first,
            use_checkpointing,
        )?;

        // NaN/Inf guard — a poisoned prompt must not corrupt the running mean.
        let mut invalid = false;
        for j in &per_j {
            if j.has_nan_or_inf()? {
                invalid = true;
                break;
            }
        }
        if invalid {
            tracing::warn!("jlens fit: prompt {idx} produced NaN/Inf J — skipping");
            prompts_skipped += 1;
            state.next_idx = idx as i64 + 1;
            heavy_cleanup();
            // Persist the advanced cursor so a crash doesn't reprocess this
            // (expensive) NaN prompt on resume.
            save_checkpoint(&opts.checkpoint_path, &state, skip_first, hidden, &fp)?;
            continue;
        }

        // Accumulate the running SUM (fp32).
        for (l, j) in per_j.iter().enumerate() {
            let s = state.jacobian_sum[l].add(j)?;
            s.eval();
            state.jacobian_sum[l] = s;
        }
        state.n_done += 1;
        state.next_idx = idx as i64 + 1;
        prompts_this_run += 1;
        n_valid_last = n_valid;
        elapsed_secs += start.elapsed().as_secs_f64();

        // High-water marks captured mid-`jacobian_for_prompt` (before its
        // periodic cleanup zeroes the MLX peak counter).
        peak_mb = peak_mb.max(prompt_peak_mb);
        active_mb = active_mb.max(prompt_active_mb);

        // Between prompts: release the compiled-graph cache, then persist.
        heavy_cleanup();
        save_checkpoint(&opts.checkpoint_path, &state, skip_first, hidden, &fp)?;
    }

    // Final checkpoint + optional pack export.
    save_checkpoint(&opts.checkpoint_path, &state, skip_first, hidden, &fp)?;
    if let Some(pack_path) = &opts.pack_path {
        if state.n_done > 0 {
            export_pack(pack_path, &state)?;
        }
    }

    // Diagnostic: max_l ||mean J_l||_F / sqrt(hidden), over the EXPORTED
    // boundaries 1..=23 (skip boundary 0, the unfitted embedding output).
    let sqrt_d = (hidden as f64).sqrt();
    let mut max_norm = 0.0f64;
    if state.n_done > 0 {
        for l in 1..state.jacobian_sum.len() {
            let mean = state.jacobian_sum[l].div_scalar(state.n_done as f64)?;
            let frob = mean.square()?.sum(None, None)?.sqrt()?;
            let v = frob.item_at_float32(0)? as f64 / sqrt_d;
            if v > max_norm {
                max_norm = v;
            }
        }
    }

    heavy_cleanup();

    let sec_per_prompt = if prompts_this_run > 0 {
        elapsed_secs / prompts_this_run as f64
    } else {
        0.0
    };

    Ok(JlensFitResult {
        n_done: state.n_done as u32,
        prompts_this_run,
        prompts_skipped,
        sec_per_prompt,
        peak_memory_mb: peak_mb,
        active_memory_mb: active_mb,
        n_valid_last: n_valid_last as i32,
        max_j_norm_over_sqrt_d: max_norm,
        checkpoint_path: opts.checkpoint_path,
        pack_path: opts.pack_path,
    })
}

// ---------------------------------------------------------------------------
// Finite-difference gate (THE pilot GO/NO-GO signal)
// ---------------------------------------------------------------------------

/// Run the probe forward (all f32, no checkpointing) with the given probes and
/// return the evaluated `h_final` `[1, T, hidden]`. Used by the causality check.
fn probe_forward_eval(
    config: &Qwen3_5Config,
    params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    probes: &[MxArray],
) -> Result<MxArray> {
    let mut ckpt = autograd::CheckpointContexts::new();
    let h = functional::qwen3_5_probe_forward_hidden_states(
        config, params, input_ids, probes, false, &mut ckpt,
    )?;
    h.eval();
    Ok(h)
}

/// Run the probe forward with ONLY boundary `target_l` perturbed (`= pert`, all
/// other probes zero) and return the evaluated `h_final`. Used by the numerical
/// directional derivative (differenced in h-space to avoid f32 cancellation).
fn forward_single_probe(
    config: &Qwen3_5Config,
    params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    probe_shape: &[i64],
    num_layers: usize,
    target_l: usize,
    pert: &MxArray,
) -> Result<MxArray> {
    let probes: Vec<MxArray> = (0..num_layers)
        .map(|i| {
            if i == target_l {
                Ok(pert.clone())
            } else {
                MxArray::zeros(probe_shape, Some(DType::Float32))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    probe_forward_eval(config, params, input_ids, &probes)
}

/// Compute the `[bv, hidden]` per-batch `J_l` rows for one backward pass in f32:
/// the prompt is replicated `bv` times, batch element `b` carries a one-hot
/// cotangent at output dim `dim_start+b` over the valid target positions, and
/// the returned row `b` is the mean over valid SOURCE positions of the grad
/// w.r.t. `probes[l]`. This is the same estimator as [`jacobian_for_prompt`],
/// used by the basis-batch consistency check (row `i` at `bv=1` must equal the
/// same row inside a `bv=8` batch).
#[allow(clippy::too_many_arguments)]
fn estimator_rows_f32(
    config: &Qwen3_5Config,
    params: &HashMap<String, MxArray>,
    prompt_ids: &[i32],
    num_layers: usize,
    hidden: usize,
    l: usize,
    bv: usize,
    dim_start: usize,
    valid_start: i64,
    valid_end: i64,
) -> Result<MxArray> {
    let seq_len = prompt_ids.len();
    let mut ids_rep: Vec<i32> = Vec::with_capacity(bv * seq_len);
    for _ in 0..bv {
        ids_rep.extend_from_slice(prompt_ids);
    }
    let input_ids = MxArray::from_int32(&ids_rep, &[bv as i64, seq_len as i64])?;
    let probe_shape = [bv as i64, seq_len as i64, hidden as i64];

    let n_dims = bv.min(hidden - dim_start);
    let mut c_host = vec![0f32; bv * hidden];
    for (b, row) in c_host.chunks_exact_mut(hidden).enumerate().take(n_dims) {
        row[dim_start + b] = 1.0;
    }
    let c = MxArray::from_float32(&c_host, &[bv as i64, hidden as i64])?;
    let c_bcast = c.expand_dims(1)?;

    let probes: Vec<MxArray> = (0..num_layers)
        .map(|_| MxArray::zeros(&probe_shape, Some(DType::Float32)))
        .collect::<Result<Vec<_>>>()?;
    let probe_refs: Vec<&MxArray> = probes.iter().collect();
    let config_c = config.clone();
    let params_c = params.clone();
    let ids_c = input_ids.clone();
    let c_c = c_bcast.clone();
    let ckpt = Rc::new(RefCell::new(autograd::CheckpointContexts::new()));
    let ckpt_in = ckpt.clone();
    let (_loss, grads) = autograd::value_and_grad(probe_refs, move |p| {
        let h = functional::qwen3_5_probe_forward_hidden_states(
            &config_c,
            &params_c,
            &ids_c,
            p,
            false,
            &mut ckpt_in.borrow_mut(),
        )?;
        let h_valid = h.slice_axis(1, valid_start, valid_end)?;
        h_valid.astype(DType::Float32)?.mul(&c_c)?.sum(None, None)
    })?;
    for g in &grads {
        g.eval();
    }
    ckpt.borrow_mut().reclaim();

    let g_valid = grads[l].slice_axis(1, valid_start, valid_end)?;
    let rows = g_valid
        .astype(DType::Float32)?
        .mean(Some(&[1]), Some(false))?; // [bv, hidden]
    rows.eval();
    Ok(rows)
}

/// Finite-difference check that MLX autograd flows correctly through BOTH mixer
/// types. Runs in **f32** (params cast up) so the check isolates autograd
/// correctness from bf16 rounding — the standard gradcheck discipline; the fit
/// itself runs the identical op graph in bf16.
///
/// For each requested boundary `l` it compares the autograd directional
/// derivative `⟨∂L/∂z_l, δ⟩` against the numerical central difference
/// `Σ (h(+εδ) − h(−εδ)) · c / 2ε`, where `L(z) = Σ h_final(z) · c` for a fixed
/// cotangent `c = randn/(|h_baseline|+1)`. It probes `1 + N_RAND_DIRS`
/// directions per boundary — `δ = g_l/‖g_l‖` (max SNR, reported) plus random
/// unit directions — with the error normalized by `‖g_l‖` (`dir_check_max_err`)
/// so it catches a VJP error in ANY direction, not just along the gradient. A
/// broken GDN / attention VJP blows the error up (or NaNs); agreement to a few
/// digits over all directions is the proof the Jacobian is real. Also checks
/// causality (single-position perturbation) and basis-batch independence.
pub(crate) fn verify_finite_diff(
    config: &Qwen3_5Config,
    params_bf16: &HashMap<String, MxArray>,
    prompt_ids: &[i32],
    boundaries: &[usize],
    epsilon: f64,
) -> Result<JlensFiniteDiffResult> {
    let hidden = config.hidden_size as usize;
    let num_layers = config.num_layers as usize;
    let seq_len = prompt_ids.len();
    if seq_len < 2 {
        return Err(Error::from_reason(
            "verify_finite_diff: prompt needs >= 2 tokens",
        ));
    }
    // Guard bad inputs BEFORE any model work: an empty `boundaries` would panic
    // at `boundaries[0]` (basis-batch check) and a non-positive/non-finite
    // `epsilon` makes the `/ (2ε)` central difference inf/nan (or silently wrong
    // for ε<0). Return clean NAPI errors instead of a panic across the boundary.
    if boundaries.is_empty() {
        return Err(Error::from_reason(
            "verify_finite_diff: boundaries must be non-empty",
        ));
    }
    if !(epsilon.is_finite() && epsilon > 0.0) {
        return Err(Error::from_reason(
            "verify_finite_diff: epsilon must be finite and > 0",
        ));
    }
    for &l in boundaries {
        if l >= num_layers {
            return Err(Error::from_reason(format!(
                "verify_finite_diff: boundary {l} out of range (0..={})",
                num_layers - 1
            )));
        }
    }

    // f32 params for a clean gradcheck.
    let mut params: HashMap<String, MxArray> = HashMap::with_capacity(params_bf16.len());
    for (k, v) in params_bf16 {
        let f = v.astype(DType::Float32)?;
        f.eval();
        params.insert(k.clone(), f);
    }

    let input_ids = MxArray::from_int32(prompt_ids, &[1, seq_len as i64])?;
    let probe_shape = [1i64, seq_len as i64, hidden as i64];

    // Baseline h_final at z=0 — conditions the cotangent and roots the causality
    // check.
    let zero_probes: Vec<MxArray> = (0..num_layers)
        .map(|_| MxArray::zeros(&probe_shape, Some(DType::Float32)))
        .collect::<Result<Vec<_>>>()?;
    let h_baseline = probe_forward_eval(config, &params, &input_ids, &zero_probes)?;

    // Fixed cotangent `c = randn / (|h_baseline| + 1)`. Qwen's residual stream
    // has MASSIVE activation outliers (values in the hundreds); a flat cotangent
    // makes `L = Σ h·c` enormous and both `L(+εδ)−L(−εδ)` and the outlier terms
    // of `Σ Δh·c` vanish into f32 roundoff. Downweighting by `1/(|h|+1)` keeps
    // every element's contribution O(1) so the gate is well-conditioned. It is a
    // FIXED cotangent (h_baseline is a constant), so both sides use the same `c`
    // — a broken VJP still fails; only the numerics are stabilized.
    let c_rand = MxArray::random_normal(&probe_shape, 0.0, 1.0, Some(DType::Float32))?;
    let denom = h_baseline.abs()?.add_scalar(1.0)?;
    let cotangent = c_rand.div(&denom)?;
    cotangent.eval();

    // Autograd: grads of L(z) = Σ h_final · c w.r.t. all 24 probes, at z=0.
    let probes: Vec<MxArray> = (0..num_layers)
        .map(|_| MxArray::zeros(&probe_shape, Some(DType::Float32)))
        .collect::<Result<Vec<_>>>()?;
    let probe_refs: Vec<&MxArray> = probes.iter().collect();
    let config_c = config.clone();
    let params_c = params.clone();
    let ids_c = input_ids.clone();
    let cot_c = cotangent.clone();
    let ckpt = Rc::new(RefCell::new(autograd::CheckpointContexts::new()));
    let ckpt_in = ckpt.clone();
    let (_loss, grads) = autograd::value_and_grad(probe_refs, move |p| {
        let h = functional::qwen3_5_probe_forward_hidden_states(
            &config_c,
            &params_c,
            &ids_c,
            p,
            false,
            &mut ckpt_in.borrow_mut(),
        )?;
        h.mul(&cot_c)?.sum(None, None)
    })?;
    for g in &grads {
        g.eval();
    }
    ckpt.borrow_mut().reclaim();

    let mut out_b = Vec::new();
    let mut out_pred = Vec::new();
    let mut out_num = Vec::new();
    let mut out_rel = Vec::new();
    let mut out_maxerr = Vec::new();
    let mut out_gdn = Vec::new();
    let mut out_leak = Vec::new();
    let mut out_signal = Vec::new();

    let split_pos = (seq_len / 2).max(1).min(seq_len - 1); // 0 < p < T
    // Position mask [1, T, 1]: 1 at position `split_pos`, else 0.
    let mut mask_host = vec![0f32; seq_len];
    mask_host[split_pos] = 1.0;
    let pos_mask = MxArray::from_float32(&mask_host, &[1, seq_len as i64, 1])?;

    for &l in boundaries {
        // MULTI-DIRECTION gradcheck. We probe several directions δ and compare
        // the autograd directional derivative `⟨g_l, δ⟩` against the numerical
        // one, with the error normalized by `‖g_l‖` (NOT by `|numerical|`). That
        // normalization stays well-conditioned even for directions near-
        // orthogonal to g_l (where `|numerical|→0` would make a relative error
        // explode in f32), so this is an ALL-DIRECTION check: a VJP error
        // ORTHOGONAL to the implementation's own gradient still surfaces here.
        // Direction 0 is `g_l/‖g_l‖` (max SNR, reported as predicted/numerical);
        // the rest are random unit directions.
        let g_l = &grads[l];
        let g_norm = g_l.square()?.sum(None, None)?.sqrt()?;
        let g_norm_v = (g_norm.item_at_float32(0)? as f64).max(1e-12);

        let grad_dir = g_l.div(&g_norm)?;
        grad_dir.eval();
        let mut dirs: Vec<MxArray> = Vec::with_capacity(1 + N_RAND_DIRS);
        dirs.push(grad_dir);
        for _ in 0..N_RAND_DIRS {
            let r = MxArray::random_normal(&probe_shape, 0.0, 1.0, Some(DType::Float32))?;
            let rn = r.square()?.sum(None, None)?.sqrt()?;
            let ru = r.div(&rn)?;
            ru.eval();
            dirs.push(ru);
        }

        let mut predicted_v = 0.0f64;
        let mut numerical_v = 0.0f64;
        let mut max_err_norm = 0.0f64;
        for (di, d) in dirs.iter().enumerate() {
            // Autograd directional derivative ⟨g_l, δ⟩.
            let pred = grads[l].mul(d)?.sum(None, None)?.item_at_float32(0)? as f64;
            // Numerical central difference along δ, differenced IN h-SPACE first:
            // `Σ (h(+εδ) − h(−εδ)) · c / 2ε`. Computing `Δh = h₊ − h₋` elementwise
            // (each ~O(ε), well below the outlier magnitude) BEFORE the
            // contraction avoids the catastrophic cancellation of two huge losses.
            let pert_p = d.mul_scalar(epsilon)?;
            let pert_m = d.mul_scalar(-epsilon)?;
            let h_p = forward_single_probe(
                config,
                &params,
                &input_ids,
                &probe_shape,
                num_layers,
                l,
                &pert_p,
            )?;
            let h_m = forward_single_probe(
                config,
                &params,
                &input_ids,
                &probe_shape,
                num_layers,
                l,
                &pert_m,
            )?;
            let num = h_p
                .sub(&h_m)?
                .mul(&cotangent)?
                .sum(None, None)?
                .item_at_float32(0)? as f64
                / (2.0 * epsilon);
            max_err_norm = max_err_norm.max((pred - num).abs() / g_norm_v);
            if di == 0 {
                predicted_v = pred; // gradient direction — reported
                numerical_v = num;
            }
        }
        let rel = (predicted_v - numerical_v).abs() / numerical_v.abs().max(1e-6);
        let d_unit = &dirs[0]; // gradient direction, reused by the causality check

        // Causality check: perturb probe `l` at ONLY position `split_pos`
        // (`δ_unit · pos_mask · ε`), run forward, and measure how much h_final
        // changes before vs at/after `split_pos`. Before: must be ~0 (earlier
        // positions cannot depend on a later input under causal masking + the
        // GDN recurrence). After: must be > 0 (the perturbation is real).
        let single_pos_pert = d_unit.mul(&pos_mask)?.mul_scalar(epsilon)?;
        let probes_c: Vec<MxArray> = (0..num_layers)
            .map(|i| {
                if i == l {
                    Ok(single_pos_pert.clone())
                } else {
                    MxArray::zeros(&probe_shape, Some(DType::Float32))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let h_pert = probe_forward_eval(config, &params, &input_ids, &probes_c)?;
        let diff = h_pert.sub(&h_baseline)?;
        let before = diff.slice_axis(1, 0, split_pos as i64)?; // [1, p, hidden]
        let after = diff.slice_axis(1, split_pos as i64, seq_len as i64)?; // [1, T-p, hidden]
        let leak = before
            .square()?
            .sum(None, None)?
            .sqrt()?
            .item_at_float32(0)? as f64;
        let signal = after
            .square()?
            .sum(None, None)?
            .sqrt()?
            .item_at_float32(0)? as f64;

        out_b.push(l as u32);
        out_pred.push(predicted_v);
        out_num.push(numerical_v);
        out_rel.push(rel);
        out_maxerr.push(max_err_norm);
        out_gdn.push(config.is_linear_layer(l));
        out_leak.push(leak);
        out_signal.push(signal);
    }

    // ---- Basis-batch consistency (the #1 MLX value_and_grad mis-port risk) ----
    // One `J_l` row must be identical whether computed solo (`bv=1`) or as one
    // element of a `bv=8` batch. A broadcast/shared probe would make
    // value_and_grad SUM the basis rows and corrupt every row; a distinct
    // per-batch-element probe (what we use) keeps them independent.
    let bc_boundary = boundaries[0];
    let bc_out_dim = hidden / 2;
    let valid_start = (DEFAULT_SKIP_FIRST.min(seq_len.saturating_sub(2))) as i64;
    let valid_end = (seq_len - 1) as i64;

    // Solo: bv=1, the whole (single-row) window sits at `bc_out_dim`.
    let row_solo = estimator_rows_f32(
        config,
        &params,
        prompt_ids,
        num_layers,
        hidden,
        bc_boundary,
        1,
        bc_out_dim,
        valid_start,
        valid_end,
    )?; // [1, hidden]
    // Batched: bv=8, `bc_out_dim` sits at batch element `bc_out_dim - dim_start`.
    let bc_bv = 8usize.min(hidden - (bc_out_dim / 8) * 8);
    let bc_dim_start = (bc_out_dim / bc_bv) * bc_bv;
    let bc_b_target = bc_out_dim - bc_dim_start;
    let rows_batched = estimator_rows_f32(
        config,
        &params,
        prompt_ids,
        num_layers,
        hidden,
        bc_boundary,
        bc_bv,
        bc_dim_start,
        valid_start,
        valid_end,
    )?; // [bv, hidden]

    let row_solo_1 = row_solo.slice(&[0, 0], &[1, hidden as i64])?; // [1, hidden]
    let row_batched_1 = rows_batched.slice(
        &[bc_b_target as i64, 0],
        &[bc_b_target as i64 + 1, hidden as i64],
    )?; // [1, hidden]
    let bc_diff = row_solo_1.sub(&row_batched_1)?;
    let bc_max_abs = bc_diff.abs()?.max(None, None)?.item_at_float32(0)? as f64;
    let solo_frob = row_solo_1
        .square()?
        .sum(None, None)?
        .sqrt()?
        .item_at_float32(0)? as f64;
    let diff_frob = bc_diff
        .square()?
        .sum(None, None)?
        .sqrt()?
        .item_at_float32(0)? as f64;
    let bc_rel = diff_frob / solo_frob.max(1e-12);

    heavy_cleanup();

    Ok(JlensFiniteDiffResult {
        boundaries: out_b,
        epsilon,
        predicted: out_pred,
        numerical: out_num,
        rel_error: out_rel,
        dir_check_max_err: out_maxerr,
        dir_check_n: (1 + N_RAND_DIRS) as i32,
        is_gdn: out_gdn,
        causality_split_pos: split_pos as i32,
        causality_leak_before: out_leak,
        causality_signal_after: out_signal,
        batch_consistency_boundary: bc_boundary as u32,
        batch_consistency_out_dim: bc_out_dim as i32,
        batch_consistency_max_abs_diff: bc_max_abs,
        batch_consistency_rel: bc_rel,
    })
}
