//! Native CUDA speculative decoding for Inkling-2-Mini:
//! - **DFlash2 (`incoai/Inkling-2-Mini-DFlash2`)** with adaptive MoE-cost-aware verify tree
//!   (`M <= 16`, canonical buckets `[4, 6, 8, 12, 16]`) and optional Suffix-Automaton (SAM) lookup
//!   (`MLX_INKLING2_LOOKUP=1`; defaults off because 128-expert MoE verify cost favors DFlash2-only trees)
//! - **Native DSpark (`dspark-*.gguf` v1/v2 or `mtp.safetensors`)** on flat `Inkling2CudaCache`
//! - Unified [`Inkling2SpecStepper`] + [`FlatDsparkBackend`] implementation routing flat CUDA
//!   speculative turns through [`crate::engine::dspark_turn::run_flat_dspark_turn`].

#[cfg(test)]
use std::fs;
use std::path::Path;
use std::time::Instant;

use mlx_sys as sys;
use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::engine::backend::{
    DsparkProposal, DsparkStepper, DsparkTreeGreedyResult, DsparkVerifyOutput,
};
use crate::engine::dspark_turn::{FlatDsparkBackend, FlatDsparkPrefix};
use crate::engine::params::ChatParams;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{
    DFlash2ContextCache, DFlash2Model, DFlash2TargetShape, DFlash2TargetView, SelectorPath,
    load_dflash2, read_dflash2_config,
};
use crate::models::qwen3_5::verify_tree::cuda::{
    TreeLookupSource, finalize_tree_greedy_walk, prepare_tree_lookup, select_tree_gpu,
};
use crate::models::qwen3_5::verify_tree::{
    DraftLattice, LookupAgreement, LookupIndex, MAX_TREE_NODES, TreeCostModel, TreeDeviceViews,
    VerifyTree, build_verify_tree, build_verify_tree_cost_aware,
};
use crate::nn::Embedding;
use crate::sampling::is_greedy_temperature;
use crate::stream::{Stream, StreamContext};

use super::attn_res::apply_sublayer_rms_norm;
use super::config::Inkling2Config;
use super::cuda_forward::{CudaLmHeadMode, CudaTreeInputs};
use super::dspark::{AdaptiveDraftDepth, Inkling2DsparkContext};
use super::dspark_decode::Inkling2DsparkStepper;
use super::model::Inkling2Inner;
use super::short_conv::Inkling2ShortConvTape;

/// Default Inkling-2-Mini MoE verify-tree cost model (`base=36.0 ms`, `slope=1.41 ms/node`,
/// fitted to live RTX Spark N1X 9400 MHz Wave 4 `T in {4, 8, 12, 16}` sweeps;
/// `min_nodes=4`, `bucketed=true` -> canonical CUDA-Graph-friendly buckets `[4, 6, 8, 12, 16]`).
pub(crate) const INKLING2_TREE_COST_DEFAULT: TreeCostModel = TreeCostModel {
    base_ms: 36.0,
    slope_ms: 1.41,
    min_nodes: 4,
    bucketed: true,
};

/// Default SAM prompt-lookup policy for Inkling-2-Mini CUDA DFlash2 verify trees.
///
/// Defaults off (`false`): on 128-expert top-8 MoE (`slope = 1.41 ms/node`), Qwen3.8-calibrated
/// `LookupAcceptance::PRIOR` entries push extra speculative SAM nodes that inflate `mean_T` and
/// reserve probability mass away from DFlash2 branches (live sweeps showed `LOOKUP=0` beating
/// `LOOKUP=1` by `+6.4..+7.0 tok/s` on `serve1_dc5d`/`splash2k_wave4` and `+14%` on `code2k`).
/// Set `MLX_INKLING2_LOOKUP=1` to opt in (e.g. verbatim-copy workloads or A/B sweeps).
pub(crate) const INKLING2_LOOKUP_DEFAULT: bool = false;

/// Maximum number of draft tokens verified in one flat-CUDA DSpark cycle (`1` anchor + `15`
/// drafts = `16` total rows, matching the `M <= 16` fast CUDA decode/verify kernels).
pub(crate) const CUDA_DSPARK_MAX_DRAFT: usize = 15;

/// Clamp a DSpark draft count (`block_size - 1`) to the flat-CUDA fast-verify budget (`<= 15`).
#[inline]
pub(crate) fn clamp_cuda_dspark_max_draft(num_draft_tokens: usize) -> usize {
    num_draft_tokens.min(CUDA_DSPARK_MAX_DRAFT)
}

/// Prompt length threshold above which SAM lookup requires a longer suffix match to avoid
/// spurious short syntax matches (`fn `, `let `) stealing tree slots from DFlash2 on large files.
const INKLING2_LONG_PROMPT_TOKENS: usize = 512;
const INKLING2_LONG_PROMPT_MIN_MATCH: usize = 5;

/// Resolve the minimum suffix-automaton match length for Inkling-2-Mini verify trees.
///
/// Honors `MLX_INKLING2_LOOKUP_MIN_MATCH` (clamped to `>= LookupIndex::MIN_MATCH`) when set;
/// otherwise defaults to `5` for long prompts (`>= 512` tokens) and `LookupIndex::MIN_MATCH` (`3`)
/// for short prompts.
pub(crate) fn resolve_inkling2_lookup_min_match(
    prompt_len: usize,
    env_override: Option<&str>,
) -> usize {
    if let Some(raw) = env_override
        && let Ok(parsed) = raw.trim().parse::<usize>()
    {
        return parsed.max(LookupIndex::MIN_MATCH);
    }
    if prompt_len >= INKLING2_LONG_PROMPT_TOKENS {
        INKLING2_LONG_PROMPT_MIN_MATCH
    } else {
        LookupIndex::MIN_MATCH
    }
}

pub(crate) fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let trimmed = v.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                false
            } else if trimmed.eq_ignore_ascii_case("1")
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("on")
                || trimmed.eq_ignore_ascii_case("yes")
            {
                true
            } else {
                default
            }
        }
        Err(_) => default,
    }
}

pub(crate) fn env_inkling2_tree_cost_model() -> TreeCostModel {
    let mut model = INKLING2_TREE_COST_DEFAULT;
    if let Ok(v) = std::env::var("MLX_INKLING2_TREE_COST_BASE")
        && let Ok(parsed) = v.trim().parse::<f64>()
        && parsed.is_finite()
        && parsed >= 0.0
    {
        model.base_ms = parsed;
    }
    if let Ok(v) = std::env::var("MLX_INKLING2_TREE_COST_SLOPE")
        && let Ok(parsed) = v.trim().parse::<f64>()
        && parsed.is_finite()
        && parsed > 0.0
    {
        model.slope_ms = parsed;
    }
    if let Ok(v) = std::env::var("MLX_INKLING2_TREE_MIN_NODES")
        && let Ok(parsed) = v.trim().parse::<usize>()
    {
        model.min_nodes = parsed.clamp(1, MAX_TREE_NODES);
    }
    model.bucketed = env_flag_default("MLX_INKLING2_TREE_BUCKETED", model.bucketed);
    model
}

pub(crate) fn env_inkling2_tree_fixed_max_nodes() -> Option<usize> {
    std::env::var("MLX_INKLING2_TREE_MAX_NODES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, MAX_TREE_NODES))
}

/// True when `path` is a directory containing a DFlash2 `config.json` declaring
/// `"DFlash2DraftModel"` in `architectures`.
#[cfg(test)]
pub(crate) fn is_dflash2_draft_dir(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let cfg_path = path.join("config.json");
    let Ok(raw) = fs::read_to_string(&cfg_path) else {
        return false;
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    val.get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|arr| {
            arr.iter()
                .any(|entry| entry.as_str() == Some("DFlash2DraftModel"))
        })
}

/// Build the [`DFlash2TargetShape`] contract for an Inkling-2-Mini target checkpoint.
#[allow(dead_code)]
pub(crate) fn inkling2_dflash2_target_shape(config: &Inkling2Config) -> DFlash2TargetShape {
    DFlash2TargetShape {
        hidden_size: config.hidden_size,
        vocab_size: config.effective_padded_vocab_size(),
        valid_vocab_size: config.effective_unpadded_vocab_size(),
        num_layers: config.num_layer_visits(),
        final_logit_softcapping: (config.final_logit_softcapping > 0.0)
            .then_some(config.final_logit_softcapping as f32),
        reads_layer_inputs: true,
    }
}

/// Load and validate a DFlash2 draft directory (e.g. `incoai/Inkling-2-Mini-DFlash2`)
/// against `config`, configuring the Inkling-2-Mini execution profile and compiled CUDA draft.
#[allow(dead_code)]
pub(crate) fn load_inkling2_dflash2(dir: &Path, config: &Inkling2Config) -> Result<DFlash2Model> {
    let refused = |reason: String| {
        Error::from_reason(format!(
            "Inkling2 load: DFlash2 draft at {} does not match this model: {reason}",
            dir.display()
        ))
    };
    let target_shape = inkling2_dflash2_target_shape(config);
    let parsed_cfg = read_dflash2_config(dir).map_err(|e| refused(e.reason))?;
    parsed_cfg
        .validate_target(target_shape)
        .map_err(|e| refused(e.reason))?;
    let (mut model, _) = load_dflash2(dir).map_err(|e| refused(e.reason))?;
    model
        .validate_target(target_shape)
        .map_err(|e| refused(e.reason))?;
    model.configure_inkling2_execution(target_shape);
    if crate::accelerator::cuda_ops_available() {
        model.configure_dense_execution();
    }
    Ok(model)
}

/// Single-launch FP32-accumulated RMSNorm returning BF16 via `mlx_rmsnorm_precise_f32_cuda`,
/// falling back to `apply_sublayer_rms_norm` when input is non-BF16 (e.g. GGUF F16) or off CUDA.
#[inline]
fn precise_embed_norm_bf16(x: &MxArray, weight: &MxArray, eps: f32) -> Result<MxArray> {
    if crate::accelerator::cuda_ops_available()
        && x.dtype()? == DType::BFloat16
        && weight.dtype()? == DType::Float32
    {
        let handle =
            unsafe { sys::mlx_rmsnorm_precise_f32_cuda(x.as_raw_ptr(), weight.as_raw_ptr(), eps) };
        if !handle.is_null() {
            return MxArray::from_handle(handle, "inkling2_dflash2_embed_norm_precise_f32");
        }
    }
    apply_sublayer_rms_norm(x, weight, eps, DType::BFloat16)
}

/// Target view exposing Inkling-2-Mini's normalized embeddings (`embed_tokens` -> `embed_norm`)
/// and shared `lm_head` (`LinearProj`, both GGUF Q8_0 and SafeTensors BF16) to `DFlash2Model`.
pub(crate) struct Inkling2DFlash2TargetView<'a> {
    embed_tokens: &'a Embedding,
    embed_norm: &'a MxArray,
    lm_head: &'a LinearProj,
    rms_norm_eps: f32,
    hidden_size: usize,
    head_limit: Option<usize>,
}

impl<'a> Inkling2DFlash2TargetView<'a> {
    pub(crate) fn new(inner: &'a Inkling2Inner, head_limit: Option<usize>) -> Self {
        Self {
            embed_tokens: &inner.embed_tokens,
            embed_norm: &inner.embed_norm,
            lm_head: &inner.lm_head,
            rms_norm_eps: inner.config.rms_norm_eps as f32,
            hidden_size: inner.config.hidden_size,
            head_limit,
        }
    }
}

impl DFlash2TargetView for Inkling2DFlash2TargetView<'_> {
    fn embed(&mut self, block: &MxArray, _ids: &[i32]) -> Result<MxArray> {
        if let Some(normed) = super::cuda_forward::try_embed_rmsnorm_cuda(
            self.embed_tokens,
            block,
            self.embed_norm,
            self.rms_norm_eps,
        )? {
            return Ok(normed);
        }
        let rows = block.shape_at(1)?;
        let d = self.hidden_size as i64;
        let raw_emb = self.embed_tokens.forward(block)?; // [1, rows, D]
        let raw_2d = raw_emb.reshape(&[rows, d])?;
        let normed_2d = precise_embed_norm_bf16(&raw_2d, self.embed_norm, self.rms_norm_eps)?;
        normed_2d.reshape(&[1, rows, d])
    }

    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray> {
        self.lm_head.forward_logits(hidden)
    }

    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)> {
        // Inkling-2-Mini applies `embed_norm` after `embed_tokens`, so always use the
        // pre-embedded compiled draft path (`target_lm_head_proj`) instead of raw `Embedding`.
        None
    }

    fn target_lm_head_proj(&self) -> Option<&LinearProj> {
        Some(self.lm_head)
    }

    fn proposal_head_limit(&self) -> Option<usize> {
        self.head_limit
    }
}

/// Asynchronous CUDA eval skipping already-settled arrays (`is_available() == true`).
fn cuda_async_eval(arrays: &[&MxArray]) {
    if arrays.is_empty() {
        return;
    }
    let mut handles: Vec<*mut sys::mlx_array> = arrays
        .iter()
        .map(|a| a.as_raw_ptr())
        .filter(|&ptr| !unsafe { sys::mlx_array_is_available(ptr) })
        .collect();
    if handles.is_empty() {
        return;
    }
    unsafe {
        sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
    }
}

/// Collect unevaluated pre-conv row buffers from `tape` so verify evaluates output + taps + tape
/// in ONE forward pass without MLX discarding intermediates and recomputing on commit.
fn collect_pending_tape_rows<'a>(tape: &'a Inkling2ShortConvTape, out: &mut Vec<&'a MxArray>) {
    for v in &tape.visits {
        if let Some(ref s) = v.k
            && !s.rows.is_available()
        {
            out.push(&s.rows);
        }
        if let Some(ref s) = v.v
            && !s.rows.is_available()
        {
            out.push(&s.rows);
        }
        if let Some(ref s) = v.attn_post
            && !s.rows.is_available()
        {
            out.push(&s.rows);
        }
        if let Some(ref s) = v.mlp_post
            && !s.rows.is_available()
        {
            out.push(&s.rows);
        }
    }
}

/// Convert a `[T]` Int32 or Uint32 argmax array to `Vec<u32>`.
fn argmax_to_u32_vec(arr: &MxArray) -> Result<Vec<u32>> {
    if arr.dtype()? == DType::Uint32 {
        Ok(arr.to_uint32()?.to_vec())
    } else {
        Ok(arr.to_int32()?.iter().map(|&v| v as u32).collect())
    }
}

/// Ensure a 1D argmax array has dtype `Int32` for `accept_dspark_greedy_argmax`.
fn ensure_argmax_i32(arr: MxArray) -> Result<MxArray> {
    if arr.dtype()? == DType::Int32 {
        Ok(arr)
    } else {
        arr.astype(DType::Int32)
    }
}

/// Convert 2D `[T, D]` taps into 3D `[1, keep, D]` accepted rows for `DFlash2Model::fuse_context`.
fn extract_dflash2_tap_rows(
    taps: &[MxArray],
    accepted_nodes: &[usize],
    is_contiguous: bool,
    hidden_size: i64,
) -> Result<Vec<MxArray>> {
    let keep = accepted_nodes.len();
    if keep == 0 {
        return Err(Error::from_reason(
            "Inkling2 DFlash2 commit: empty accepted_nodes",
        ));
    }
    let keep_i64 = keep as i64;
    if is_contiguous {
        let mut out = Vec::with_capacity(taps.len());
        for tap in taps {
            let rows = tap.shape_at(0)?;
            if rows < keep_i64 {
                return Err(Error::from_reason(format!(
                    "Inkling2 DFlash2 commit: tap has {rows} rows < keep={keep}"
                )));
            }
            let sliced = if rows == keep_i64 {
                tap.clone()
            } else {
                tap.slice(&[0, 0], &[keep_i64, hidden_size])?
            };
            out.push(sliced.reshape(&[1, keep_i64, hidden_size])?);
        }
        return Ok(out);
    }

    let mut indices = Vec::with_capacity(keep);
    for &n in accepted_nodes {
        indices.push(i32::try_from(n).map_err(|_| {
            Error::from_reason(format!(
                "Inkling2 DFlash2 commit: accepted node {n} exceeds i32::MAX"
            ))
        })?);
    }
    let idx_arr = MxArray::from_int32(&indices, &[keep_i64])?;
    let mut out = Vec::with_capacity(taps.len());
    for tap in taps {
        let rows = tap.shape_at(0)? as usize;
        for &n in accepted_nodes {
            if n >= rows {
                return Err(Error::from_reason(format!(
                    "Inkling2 DFlash2 commit: accepted node {n} >= tapped rows {rows}"
                )));
            }
        }
        out.push(
            tap.take(&idx_arr, 0)?
                .reshape(&[1, keep_i64, hidden_size])?,
        );
    }
    Ok(out)
}

enum TreeLookupBacking {
    Lattice(DraftLattice),
    Agreements(Vec<LookupAgreement>),
}

enum PendingDFlash2Verify {
    Tree {
        position: usize,
        tree_tokens: Vec<u32>,
        taps: Vec<MxArray>,
        tape: Inkling2ShortConvTape,
    },
    Linear {
        position: usize,
        tokens: Vec<u32>,
        taps: Vec<MxArray>,
        tape: Inkling2ShortConvTape,
    },
}

/// CUDA DFlash2 speculative stepper for Inkling-2-Mini with MoE-cost-aware adaptive verify tree
/// (`M <= 16`) + SAM prompt lookup and linear fallback (for sampled turns / `MLX_INKLING2_TREE=0`).
pub(crate) struct Inkling2CudaDFlash2Stepper<'a> {
    inner: &'a mut Inkling2Inner,
    ctx: DFlash2ContextCache,
    tap_visits: Vec<usize>,
    tree_enabled: bool,
    lookup_enabled: bool,
    gpu_tree_enabled: bool,
    tree_fixed_max_nodes: Option<usize>,
    tree_cost_model: TreeCostModel,
    lookup_index: LookupIndex,
    lookup_min_match: usize,
    pending: Option<PendingDFlash2Verify>,
    tree_cycles: usize,
    tree_nodes_sum: usize,
    tree_accepted_sum: usize,
    tree_noncontig_commits: usize,
    tree_lookup_accepted: usize,
    tree_propose_us: u64,
    tree_verify_us: u64,
}

impl Drop for Inkling2CudaDFlash2Stepper<'_> {
    fn drop(&mut self) {
        if let Some(ref cache) = self.inner.cuda_cache {
            let _ = cache.eval_pending_state();
        }
        let _ = self.ctx.eval();
    }
}

impl<'a> Inkling2CudaDFlash2Stepper<'a> {
    pub(crate) fn new(inner: &'a mut Inkling2Inner, ctx: DFlash2ContextCache) -> Result<Self> {
        let tap_visits = inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded DFlash2 draft model"))?
            .config
            .target_layers
            .clone();
        let tree_enabled = inner.use_cuda_fast() && env_flag_default("MLX_INKLING2_TREE", true);
        let lookup_enabled = env_flag_default("MLX_INKLING2_LOOKUP", INKLING2_LOOKUP_DEFAULT);
        let gpu_tree_enabled = env_flag_default("MLX_INKLING2_GPU_TREE", false);
        let tree_fixed_max_nodes = env_inkling2_tree_fixed_max_nodes();
        let tree_cost_model = env_inkling2_tree_cost_model();
        let prompt_len = usize::try_from(ctx.logical_len()).unwrap_or(0);
        let env_min_match = std::env::var("MLX_INKLING2_LOOKUP_MIN_MATCH").ok();
        let lookup_min_match =
            resolve_inkling2_lookup_min_match(prompt_len, env_min_match.as_deref());

        Ok(Self {
            inner,
            ctx,
            tap_visits,
            tree_enabled,
            lookup_enabled,
            gpu_tree_enabled,
            tree_fixed_max_nodes,
            tree_cost_model,
            lookup_index: LookupIndex::new(),
            lookup_min_match,
            pending: None,
            tree_cycles: 0,
            tree_nodes_sum: 0,
            tree_accepted_sum: 0,
            tree_noncontig_commits: 0,
            tree_lookup_accepted: 0,
            tree_propose_us: 0,
            tree_verify_us: 0,
        })
    }

    fn print_tree_stats(&self) {
        if self.tree_cycles > 0
            && (std::env::var("MLX_INKLING2_TREE_STATS").is_ok()
                || std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok())
        {
            let c = self.tree_cycles as f64;
            eprintln!(
                "[inkling2-tree-stats] cycles={} mean_T={:.2} mean_acc={:.2} noncontig={} lookup={} lookup_acc={} min_match={} wide_vocab={} propose_ms={:.2} verify_ms={:.2} cost=({:.1}+{:.2}*T) fixed={:?}",
                self.tree_cycles,
                self.tree_nodes_sum as f64 / c,
                self.tree_accepted_sum as f64 / c,
                self.tree_noncontig_commits,
                self.lookup_enabled,
                self.tree_lookup_accepted,
                self.lookup_min_match,
                self.ctx.wide_vocab(),
                (self.tree_propose_us as f64 / c) / 1000.0,
                (self.tree_verify_us as f64 / c) / 1000.0,
                self.tree_cost_model.base_ms,
                self.tree_cost_model.slope_ms,
                self.tree_fixed_max_nodes,
            );
        }
    }

    fn ensure_clean(&self, op: &str) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 {op}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn verify_linear_inner(&mut self, verify_ids: &[u32], greedy_argmax: bool) -> Result<MxArray> {
        self.ensure_clean("verify")?;
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Inkling2 CUDA DFlash2 verify_ids must not be empty",
            ));
        }
        let mut cache = self.inner.cuda_cache.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 verify: cuda_cache missing")
        })?;
        let position = cache.pos_offset;
        let ids_i32: Vec<i32> = verify_ids.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[ids_i32.len() as i64])?;
        let lm_mode = if greedy_argmax {
            CudaLmHeadMode::GreedyArgmax
        } else {
            CudaLmHeadMode::LogitsF32All
        };
        let fwd_res = self.inner.forward_cuda_ids(
            &ids_arr,
            None,
            &mut cache,
            Some(&self.tap_visits),
            None,
            true,
            lm_mode,
        );
        let out = match fwd_res {
            Ok(out) => out,
            Err(err) => {
                self.inner.cuda_cache = Some(cache);
                return Err(err);
            }
        };
        let raw = out
            .logits_or_tokens
            .ok_or_else(|| Error::from_reason("Inkling2 CUDA DFlash2 verify produced no output"))?;
        let result = if greedy_argmax {
            ensure_argmax_i32(raw)?
        } else {
            let vocab = raw.shape_at(1)?;
            raw.reshape(&[1, verify_ids.len() as i64, vocab])?
        };
        let tape = out.tape.ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 verify missing ShortConvTape")
        })?;
        let taps = out.taps;

        // Materialize output + taps + tape rows + in-place KV aliases in ONE GPU pass so commit
        // never triggers a duplicate forward pass or races in-place KV writes.
        let mut to_eval: Vec<&MxArray> = Vec::with_capacity(1 + taps.len() + 48 * 8);
        to_eval.push(&result);
        for tap in &taps {
            if !tap.is_available() {
                to_eval.push(tap);
            }
        }
        collect_pending_tape_rows(&tape, &mut to_eval);
        cache.collect_pending_state_arrays(&mut to_eval);
        let eval_res =
            MxArray::eval_arrays_with_context(&to_eval, "inkling2::cuda_dflash2_verify_linear");
        self.inner.cuda_cache = Some(cache);
        eval_res?;

        self.pending = Some(PendingDFlash2Verify::Linear {
            position,
            tokens: verify_ids.to_vec(),
            taps,
            tape,
        });
        Ok(result)
    }
}

impl DsparkStepper for Inkling2CudaDFlash2Stepper<'_> {
    fn supports_tree_verify(&self) -> bool {
        self.tree_enabled
    }

    fn observe_emitted_tokens(&mut self, tokens: &[u32]) {
        if !tokens.is_empty() {
            self.ctx.observe_tokens(tokens);
            if self.tree_enabled && self.lookup_enabled {
                self.lookup_index.append(tokens);
            }
        }
    }

    fn verify_tree_greedy(
        &mut self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<DsparkTreeGreedyResult> {
        if !self.tree_enabled {
            return Err(Error::from_reason(
                "Inkling2 CUDA DFlash2 tree verification is not enabled",
            ));
        }
        self.ensure_clean("verify_tree_greedy")?;

        let position = self
            .inner
            .cuda_cache
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 verify_tree_greedy: cuda_cache missing")
            })?
            .pos_offset;
        if usize::try_from(self.ctx.logical_len()).ok() != Some(position) {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 tree: draft context ends at {}, target at {position}",
                self.ctx.logical_len()
            )));
        }

        let t_propose = Instant::now();
        let (tree, views, lookup, lookup_backing) = {
            let draft = self.inner.dflash2.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 draft model disappeared")
            })?;
            let draft_slots = draft.config.block_size;
            let mut view = Inkling2DFlash2TargetView::new(self.inner, None);
            let (candidates, scores) = draft.propose_tree_lattice_arrays_embedded(
                &mut view,
                &mut self.ctx,
                anchor,
                draft_slots,
            )?;

            let mut lookup =
                prepare_tree_lookup(&self.lookup_index, self.lookup_enabled, max_depth);
            if !self.lookup_enabled || lookup.match_len < self.lookup_min_match {
                lookup.match_len = 0;
                lookup.lookup_len = 0;
                lookup.proposals.clear();
            }
            let max_cap = max_nodes.clamp(1, MAX_TREE_NODES);

            let (tree, views, backing) = if let Some(fixed) = self.tree_fixed_max_nodes
                && self.gpu_tree_enabled
                && crate::accelerator::cuda_ops_available()
            {
                let node_budget = max_cap.min(fixed);
                if let Some(sel) = select_tree_gpu(
                    &candidates,
                    &scores,
                    &lookup,
                    &self.inner.cuda_lookup_acceptance,
                    &self.inner.cuda_verify_budget,
                    anchor,
                    node_budget,
                )? {
                    MxArray::eval_arrays_with_context(
                        &[&sel.tree_tokens, &sel.parent, &sel.node_meta],
                        "inkling2::gpu_tree_select",
                    )?;
                    let raw_tokens = sel.tree_tokens.to_uint32()?.to_vec();
                    let raw_parent = sel.parent.to_int32()?.to_vec();
                    let raw_meta = sel.node_meta.to_float32()?.to_vec();
                    let (tree, agreements, _) =
                        VerifyTree::from_gpu_readback(&raw_tokens, &raw_parent, &raw_meta)?;
                    let views = if tree.len() == node_budget {
                        TreeDeviceViews {
                            depths_i32: sel.depths,
                            mask_u32: sel.ancestors_mask,
                            parent_i32: sel.parent,
                            conv_taps_i32: sel.conv_taps,
                            tree_len: node_budget,
                        }
                    } else {
                        tree.to_device_views()?
                    };
                    (tree, views, TreeLookupBacking::Agreements(agreements))
                } else {
                    let lattice = draft.lattice_from_arrays(&candidates, &scores, draft_slots)?;
                    let tree = build_verify_tree(
                        anchor,
                        &lattice,
                        &lookup.proposals,
                        lookup.match_len,
                        node_budget,
                        Some(lookup.eff_max_depth),
                        false,
                        &self.inner.cuda_verify_budget,
                        &self.inner.cuda_lookup_acceptance,
                    );
                    let views = tree.to_device_views()?;
                    (tree, views, TreeLookupBacking::Lattice(lattice))
                }
            } else {
                let lattice = draft.lattice_from_arrays(&candidates, &scores, draft_slots)?;
                let tree = if let Some(fixed) = self.tree_fixed_max_nodes {
                    build_verify_tree(
                        anchor,
                        &lattice,
                        &lookup.proposals,
                        lookup.match_len,
                        max_cap.min(fixed),
                        Some(lookup.eff_max_depth),
                        false,
                        &self.inner.cuda_verify_budget,
                        &self.inner.cuda_lookup_acceptance,
                    )
                } else {
                    build_verify_tree_cost_aware(
                        anchor,
                        &lattice,
                        &lookup.proposals,
                        lookup.match_len,
                        max_cap,
                        Some(lookup.eff_max_depth),
                        false,
                        &self.inner.cuda_verify_budget,
                        &self.inner.cuda_lookup_acceptance,
                        &self.tree_cost_model,
                    )
                };
                let views = tree.to_device_views()?;
                (tree, views, TreeLookupBacking::Lattice(lattice))
            };
            (tree, views, lookup, backing)
        };
        let propose_us = t_propose.elapsed().as_micros() as u64;

        let t_verify = Instant::now();
        let tree_ids_i32: Vec<i32> = tree.tokens.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&tree_ids_i32, &[tree_ids_i32.len() as i64])?;
        let tree_inputs = CudaTreeInputs {
            depths_i32: &views.depths_i32,
            ancestors_mask_u32: &views.mask_u32,
            conv_taps_i32: &views.conv_taps_i32,
        };

        let mut cache = self.inner.cuda_cache.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 verify_tree_greedy: cuda_cache missing")
        })?;
        let fwd_res = self.inner.forward_cuda_ids(
            &ids_arr,
            None,
            &mut cache,
            Some(&self.tap_visits),
            Some(tree_inputs),
            true,
            CudaLmHeadMode::GreedyArgmax,
        );
        let out = match fwd_res {
            Ok(out) => out,
            Err(err) => {
                self.inner.cuda_cache = Some(cache);
                return Err(err);
            }
        };
        let argmax = out.logits_or_tokens.ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 tree verify produced no argmax")
        })?;
        let tape = out.tape.ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 tree verify missing ShortConvTape")
        })?;
        let taps = out.taps;

        // ONE single eval for argmax + taps + ShortConv tape rows + in-place KV aliases.
        let mut to_eval: Vec<&MxArray> = Vec::with_capacity(1 + taps.len() + 48 * 8);
        to_eval.push(&argmax);
        for tap in &taps {
            if !tap.is_available() {
                to_eval.push(tap);
            }
        }
        collect_pending_tape_rows(&tape, &mut to_eval);
        cache.collect_pending_state_arrays(&mut to_eval);
        let eval_res = MxArray::eval_arrays_with_context(&to_eval, "inkling2::tree_greedy_accept");
        self.inner.cuda_cache = Some(cache);
        eval_res?;
        let verify_us = t_verify.elapsed().as_micros() as u64;

        let target_argmax = argmax_to_u32_vec(&argmax)?;
        if target_argmax.len() != tree.len() {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 tree argmax length {} != tree length {}",
                target_argmax.len(),
                tree.len()
            )));
        }
        let lookup_source = match &lookup_backing {
            TreeLookupBacking::Lattice(lattice) => TreeLookupSource::Lattice(lattice),
            TreeLookupBacking::Agreements(agreements) => TreeLookupSource::Agreements(agreements),
        };
        let (walk, lookup_accepted) = finalize_tree_greedy_walk(
            &tree,
            &target_argmax,
            &lookup,
            lookup_source,
            &mut self.inner.cuda_verify_budget,
            &mut self.inner.cuda_lookup_acceptance,
        );
        self.tree_lookup_accepted += lookup_accepted;
        for &tok in &walk.emitted_tokens {
            self.ctx.observe_token(tok);
        }

        let draft_count = tree.depth.iter().copied().max().unwrap_or(0).max(0) as usize;
        let accepted_draft_count = walk.accepted_nodes.len().saturating_sub(1);
        let tree_nodes = tree.len();
        self.tree_cycles += 1;
        self.tree_nodes_sum += tree_nodes;
        self.tree_accepted_sum += walk.emitted_tokens.len();
        self.tree_propose_us += propose_us;
        self.tree_verify_us += verify_us;

        self.pending = Some(PendingDFlash2Verify::Tree {
            position,
            tree_tokens: tree.tokens,
            taps,
            tape,
        });
        Ok(DsparkTreeGreedyResult {
            accepted_nodes: walk.accepted_nodes,
            emitted_tokens: walk.emitted_tokens,
            is_contiguous_prefix: walk.is_contiguous_prefix,
            draft_count,
            tree_nodes,
            accepted_draft_count,
        })
    }

    fn commit_tree(&mut self, accepted_nodes: &[usize], is_contiguous_prefix: bool) -> Result<()> {
        let pending = self.pending.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 commit_tree without pending verify")
        })?;
        let PendingDFlash2Verify::Tree {
            position,
            tree_tokens,
            taps,
            tape,
        } = pending
        else {
            return Err(Error::from_reason(
                "Inkling2 CUDA DFlash2 commit_tree called on non-tree verify",
            ));
        };
        let keep = accepted_nodes.len();
        let total = tree_tokens.len();
        if keep == 0 || keep > total || accepted_nodes[0] != 0 {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 commit_tree invalid accepted_nodes {accepted_nodes:?} for T={total}"
            )));
        }
        let mut nodes_u32 = Vec::with_capacity(keep);
        let mut accepted_tokens = Vec::with_capacity(keep);
        for &idx in accepted_nodes {
            if idx >= total {
                return Err(Error::from_reason(format!(
                    "Inkling2 CUDA DFlash2 commit_tree node {idx} >= tree size {total}"
                )));
            }
            nodes_u32.push(idx as u32);
            accepted_tokens.push(tree_tokens[idx]);
        }
        if !is_contiguous_prefix {
            self.tree_noncontig_commits += 1;
        }

        let hidden_size = self.inner.config.hidden_size as i64;
        let taps_3d =
            extract_dflash2_tap_rows(&taps, accepted_nodes, is_contiguous_prefix, hidden_size)?;
        drop(taps);

        {
            let cache = self.inner.cuda_cache.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 commit_tree: cuda_cache missing")
            })?;
            cache.commit_tree_with_tape(&nodes_u32, is_contiguous_prefix, &tape)?;
        }

        let draft =
            self.inner.dflash2.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 draft model disappeared")
            })?;
        let fused = draft.fuse_context(&taps_3d)?;
        self.ctx
            .append(draft, &fused, position as i32, &accepted_tokens)?;
        Ok(())
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        self.ensure_clean("propose")?;
        if max_len == 0 {
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }
        let position = self
            .inner
            .cuda_cache
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 CUDA DFlash2 propose: cuda_cache missing"))?
            .pos_offset;
        if usize::try_from(self.ctx.logical_len()).ok() != Some(position) {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 propose: draft context ends at {}, target at {position}",
                self.ctx.logical_len()
            )));
        }

        let draft =
            self.inner.dflash2.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 draft model disappeared")
            })?;
        let draft_slots = draft.config.block_size;
        let want = max_len.min(draft_slots);
        let head_limit = (want < draft_slots).then_some(want);
        let temperature = params
            .sampling_config
            .and_then(|c| c.temperature)
            .unwrap_or(1.0);
        let greedy = is_greedy_temperature(temperature);
        let device_path = greedy
            && params.repetition_penalty == 1.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0;

        let mut view = Inkling2DFlash2TargetView::new(self.inner, head_limit);
        let (path, mut rows) = draft.propose_with(
            &mut view,
            &mut self.ctx,
            anchor_id,
            draft_slots,
            temperature,
            device_path,
            None,
            rng,
        )?;
        let ids: Vec<i32> = match path {
            SelectorPath::Device(ids) => {
                if ids.dtype()? == DType::Uint32 {
                    ids.to_uint32()?.iter().map(|&v| v as i32).collect()
                } else {
                    ids.to_int32()?.to_vec()
                }
            }
            SelectorPath::Host(ids) => ids,
            SelectorPath::Sampled { ids, rows: drawn } => {
                rows = drawn.host()?;
                if ids.dtype()? == DType::Uint32 {
                    ids.to_uint32()?.iter().map(|&v| v as i32).collect()
                } else {
                    ids.to_int32()?.to_vec()
                }
            }
        };
        let mut proposal = DsparkProposal {
            draft_ids: ids,
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists: Vec::new(),
            draft_sparse_dists: if greedy { Vec::new() } else { rows },
            keep_probabilities: None,
        };
        proposal.truncate(want);
        Ok(proposal)
    }

    fn supports_fused_greedy_verify(&self) -> bool {
        true
    }

    fn verify_greedy(&mut self, verify_ids: &[u32]) -> Result<MxArray> {
        self.verify_linear_inner(verify_ids, true)
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        let logits = self.verify_linear_inner(verify_ids, false)?;
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        let pending = self.pending.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DFlash2 commit without pending verify")
        })?;
        let PendingDFlash2Verify::Linear {
            position,
            tokens,
            taps,
            tape,
        } = pending
        else {
            return Err(Error::from_reason(
                "Inkling2 CUDA DFlash2 linear commit called on pending tree verify",
            ));
        };
        if keep == 0 || keep > total_written || total_written != tokens.len() {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DFlash2 invalid commit keep={keep}, total_written={total_written}, verify_len={}",
                tokens.len()
            )));
        }

        let accepted_nodes: Vec<usize> = (0..keep).collect();
        let hidden_size = self.inner.config.hidden_size as i64;
        let taps_3d = extract_dflash2_tap_rows(&taps, &accepted_nodes, true, hidden_size)?;
        drop(taps);

        {
            let cache = self.inner.cuda_cache.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 commit: cuda_cache missing")
            })?;
            cache.commit_linear(keep);
            tape.commit_prefix(keep, &mut cache.conv_state)?;
        }

        let draft =
            self.inner.dflash2.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DFlash2 draft model disappeared")
            })?;
        let fused = draft.fuse_context(&taps_3d)?;
        self.ctx
            .append(draft, &fused, position as i32, &tokens[..keep])?;
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(256);
        if let Some(ref cache) = self.inner.cuda_cache {
            cache.collect_pending_state_arrays(&mut arrays);
        }
        self.ctx.collect_arrays(&mut arrays);
        arrays.push(token);
        cuda_async_eval(&arrays);
    }

    fn finish(mut self) -> Result<()> {
        self.ensure_clean("finish")?;
        self.print_tree_stats();
        if let Some(ref cache) = self.inner.cuda_cache {
            cache.eval_pending_state()?;
        }
        self.ctx.eval()?;
        let empty = DFlash2ContextCache::new(
            &self
                .inner
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Inkling2 DFlash2 disappeared"))?
                .config,
        );
        let ctx = std::mem::replace(&mut self.ctx, empty);
        self.inner.cuda_dflash2_ctx = Some(ctx);
        Ok(())
    }
}

/// Native CUDA DSpark stepper driving `Inkling2DsparkModel` (`dspark-*.gguf` v1/v2 or
/// `mtp.safetensors`) over flat `Inkling2CudaCache` + `Inkling2ShortConvTape`.
pub(crate) struct Inkling2CudaDsparkStepper<'a> {
    inner: &'a mut Inkling2Inner,
    ctx: Inkling2DsparkContext,
    adaptive_depth: AdaptiveDraftDepth,
    tap_visits: Vec<usize>,
    ar_fallback: bool,
    open_written: Option<usize>,
    open_tape: Option<Inkling2ShortConvTape>,
    open_taps: Option<Vec<MxArray>>,
    ar_probe_pending: bool,
    ar_probe_taps: Option<Vec<MxArray>>,
}

impl Drop for Inkling2CudaDsparkStepper<'_> {
    fn drop(&mut self) {
        if let Some(ref cache) = self.inner.cuda_cache {
            let _ = cache.eval_pending_state();
        }
        let _ = self.ctx.eval_state();
    }
}

impl<'a> Inkling2CudaDsparkStepper<'a> {
    pub(crate) fn new(
        inner: &'a mut Inkling2Inner,
        ctx: Inkling2DsparkContext,
        max_drafts: usize,
    ) -> Result<Self> {
        let tap_visits = inner
            .dspark
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded DSpark draft model"))?
            .config
            .target_layer_ids
            .clone();
        let capped_max_drafts = clamp_cuda_dspark_max_draft(max_drafts);
        Ok(Self {
            inner,
            ctx,
            adaptive_depth: AdaptiveDraftDepth::new(capped_max_drafts),
            tap_visits,
            ar_fallback: false,
            open_written: None,
            open_tape: None,
            open_taps: None,
            ar_probe_pending: false,
            ar_probe_taps: None,
        })
    }

    fn ensure_clean(&self, op: &str) -> Result<()> {
        if self.open_written.is_some()
            || self.open_tape.is_some()
            || self.open_taps.is_some()
            || self.ar_probe_pending
        {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DSpark {op}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn verify_ids_inner(
        &mut self,
        ids_arr: &MxArray,
        len: usize,
        greedy_argmax: bool,
    ) -> Result<MxArray> {
        self.ensure_clean("verify")?;
        if len == 0 {
            return Err(Error::from_reason(
                "Inkling2 CUDA DSpark verify_ids must not be empty",
            ));
        }

        let mut cache =
            self.inner.cuda_cache.take().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DSpark verify: cuda_cache missing")
            })?;

        // Fast AR fallback path after break-even decision: skip tape & taps cleanly.
        if self.ar_fallback && len == 1 {
            let lm_mode = if greedy_argmax {
                CudaLmHeadMode::GreedyArgmax
            } else {
                CudaLmHeadMode::LogitsF32Last
            };
            let fwd_res = self
                .inner
                .forward_cuda_ids(ids_arr, None, &mut cache, None, None, false, lm_mode);
            let out = match fwd_res {
                Ok(out) => out,
                Err(err) => {
                    self.inner.cuda_cache = Some(cache);
                    return Err(err);
                }
            };
            let raw = out.logits_or_tokens.ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DSpark AR fallback produced no output")
            })?;
            let result = if greedy_argmax {
                ensure_argmax_i32(raw)?
            } else {
                let vocab = raw.shape_at(1)?;
                raw.reshape(&[1, 1, vocab])?
            };
            let mut to_eval: Vec<&MxArray> = Vec::with_capacity(48 * 8 + 1);
            to_eval.push(&result);
            cache.collect_pending_state_arrays(&mut to_eval);
            let eval_res =
                MxArray::eval_arrays_with_context(&to_eval, "inkling2::cuda_dspark_ar_fallback");
            self.inner.cuda_cache = Some(cache);
            eval_res?;
            return Ok(result);
        }

        let tap_indices: &[usize] = if self.ar_fallback {
            &[]
        } else {
            &self.tap_visits
        };
        let lm_mode = if greedy_argmax {
            CudaLmHeadMode::GreedyArgmax
        } else {
            CudaLmHeadMode::LogitsF32All
        };
        let fwd_res = self.inner.forward_cuda_ids(
            ids_arr,
            None,
            &mut cache,
            Some(tap_indices),
            None,
            true,
            lm_mode,
        );
        let out = match fwd_res {
            Ok(out) => out,
            Err(err) => {
                self.inner.cuda_cache = Some(cache);
                return Err(err);
            }
        };
        let raw = out
            .logits_or_tokens
            .ok_or_else(|| Error::from_reason("Inkling2 CUDA DSpark verify produced no output"))?;
        let result = if greedy_argmax {
            ensure_argmax_i32(raw)?
        } else {
            let vocab = raw.shape_at(1)?;
            raw.reshape(&[1, len as i64, vocab])?
        };
        let tape = out.tape.ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DSpark verify missing ShortConvTape")
        })?;
        let taps = out.taps;

        let mut to_eval: Vec<&MxArray> = Vec::with_capacity(1 + taps.len() + 48 * 8);
        to_eval.push(&result);
        for tap in &taps {
            if !tap.is_available() {
                to_eval.push(tap);
            }
        }
        collect_pending_tape_rows(&tape, &mut to_eval);
        cache.collect_pending_state_arrays(&mut to_eval);
        let eval_res = MxArray::eval_arrays_with_context(&to_eval, "inkling2::cuda_dspark_verify");
        self.inner.cuda_cache = Some(cache);
        eval_res?;

        self.open_written = Some(len);
        self.open_tape = Some(tape);
        self.open_taps = if self.ar_fallback { None } else { Some(taps) };
        Ok(result)
    }
}

impl DsparkStepper for Inkling2CudaDsparkStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        true
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        self.ensure_clean("enter_ar_fallback")?;
        self.ar_fallback = true;
        Ok(())
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        if let Some(ref cache) = self.inner.cuda_cache {
            cache.eval_pending_state()?;
        }
        self.ctx.eval_state()
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        self.ensure_clean("propose")?;
        if self.ar_fallback || max_len == 0 {
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }
        let capped_len = clamp_cuda_dspark_max_draft(max_len);
        let effective_len = if params.mtp_adaptive_depth {
            self.adaptive_depth.current_depth(capped_len)
        } else {
            capped_len
        };
        let draft = self
            .inner
            .dspark
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 DSpark draft model disappeared"))?;
        self.ctx.propose_block(
            draft,
            &self.inner.embed_tokens,
            &self.inner.embed_norm,
            &self.inner.lm_head,
            anchor_id,
            effective_len,
            params,
            rng,
        )
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("verify_ar_probe")?;
        let mut cache = self.inner.cuda_cache.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DSpark verify_ar_probe: cuda_cache missing")
        })?;
        let ids_arr = MxArray::from_int32(&[anchor_id as i32], &[1])?;
        let fwd_res = self.inner.forward_cuda_ids(
            &ids_arr,
            None,
            &mut cache,
            Some(&self.tap_visits),
            None,
            false,
            CudaLmHeadMode::LogitsF32Last,
        );
        let out = match fwd_res {
            Ok(out) => out,
            Err(err) => {
                self.inner.cuda_cache = Some(cache);
                return Err(err);
            }
        };
        let logits_2d = out.logits_or_tokens.ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DSpark AR probe produced no logits")
        })?;
        let vocab = logits_2d.shape_at(1)?;
        let logits = logits_2d.reshape(&[1, 1, vocab])?;
        let taps = out.taps;

        let mut to_eval: Vec<&MxArray> = Vec::with_capacity(1 + taps.len() + 48 * 8);
        to_eval.push(&logits);
        for tap in &taps {
            if !tap.is_available() {
                to_eval.push(tap);
            }
        }
        cache.collect_pending_state_arrays(&mut to_eval);
        let eval_res =
            MxArray::eval_arrays_with_context(&to_eval, "inkling2::cuda_dspark_ar_probe");
        self.inner.cuda_cache = Some(cache);
        eval_res?;

        self.ar_probe_pending = true;
        self.ar_probe_taps = Some(taps);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        if !self.ar_probe_pending {
            return Err(Error::from_reason(
                "Inkling2 CUDA DSpark commit_ar_probe without open AR probe",
            ));
        }
        self.ar_probe_pending = false;
        let taps = self.ar_probe_taps.take();
        if !self.ar_fallback
            && let Some(taps) = taps
            && let Some(ref draft) = self.inner.dspark
        {
            self.ctx.append_taps(draft, &taps, 1)?;
        }
        Ok(())
    }

    fn supports_fused_greedy_verify(&self) -> bool {
        true
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        let ids_i32: Vec<i32> = verify_ids.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[ids_i32.len() as i64])?;
        let logits = self.verify_ids_inner(&ids_arr, verify_ids.len(), false)?;
        Ok(DsparkVerifyOutput { logits })
    }

    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        let len = verify_ids.size()? as usize;
        let logits = self.verify_ids_inner(verify_ids, len, false)?;
        Ok(DsparkVerifyOutput { logits })
    }

    fn verify_greedy(&mut self, verify_ids: &[u32]) -> Result<MxArray> {
        let ids_i32: Vec<i32> = verify_ids.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[ids_i32.len() as i64])?;
        self.verify_ids_inner(&ids_arr, verify_ids.len(), true)
    }

    fn verify_device_greedy(&mut self, verify_ids: &MxArray) -> Result<MxArray> {
        let len = verify_ids.size()? as usize;
        self.verify_ids_inner(verify_ids, len, true)
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if self.ar_fallback && self.open_written.is_none() {
            return Ok(());
        }
        let written = self
            .open_written
            .take()
            .ok_or_else(|| Error::from_reason("Inkling2 CUDA DSpark commit without open verify"))?;
        let tape = self.open_tape.take().ok_or_else(|| {
            Error::from_reason("Inkling2 CUDA DSpark commit without ShortConvTape")
        })?;
        let taps = self.open_taps.take();

        if keep == 0 || keep > total_written || total_written != written {
            return Err(Error::from_reason(format!(
                "Inkling2 CUDA DSpark invalid commit keep={keep}, total_written={total_written}, written={written}"
            )));
        }

        {
            let cache = self.inner.cuda_cache.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 CUDA DSpark commit: cuda_cache missing")
            })?;
            cache.commit_linear(keep);
            tape.commit_prefix(keep, &mut cache.conv_state)?;
        }

        if !self.ar_fallback
            && let Some(taps) = taps
            && let Some(ref draft) = self.inner.dspark
        {
            self.ctx.append_taps(draft, &taps, keep)?;
        }
        let drafted = total_written.saturating_sub(1);
        if !self.ar_fallback && drafted > 0 {
            self.adaptive_depth.observe(drafted, keep.saturating_sub(1));
        }
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(256);
        if let Some(ref cache) = self.inner.cuda_cache {
            cache.collect_pending_state_arrays(&mut arrays);
        }
        self.ctx.collect_arrays(&mut arrays);
        arrays.push(token);
        cuda_async_eval(&arrays);
    }

    fn finish(mut self) -> Result<()> {
        self.ensure_clean("finish")?;
        if let Some(ref cache) = self.inner.cuda_cache {
            cache.eval_pending_state()?;
        }
        if !self.ar_fallback {
            self.ctx.eval_state()?;
            let n_layers = self
                .inner
                .dspark
                .as_ref()
                .map_or(3, |d| d.config.num_hidden_layers);
            let ctx = std::mem::replace(&mut self.ctx, Inkling2DsparkContext::new(n_layers));
            self.inner.cuda_dspark_ctx = Some(ctx);
        } else {
            self.inner.cuda_dspark_ctx = None;
        }
        Ok(())
    }
}

/// Unified Inkling-2-Mini speculative stepper across Metal paged DSpark, CUDA DFlash2+Tree+SAM,
/// and CUDA native DSpark.
pub(crate) enum Inkling2SpecStepper<'a> {
    PagedDspark(Box<Inkling2DsparkStepper<'a>>),
    CudaDFlash2(Box<Inkling2CudaDFlash2Stepper<'a>>),
    CudaDspark(Box<Inkling2CudaDsparkStepper<'a>>),
}

impl DsparkStepper for Inkling2SpecStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        match self {
            Self::PagedDspark(s) => s.supports_adaptive_ar_fallback(),
            Self::CudaDFlash2(s) => s.supports_adaptive_ar_fallback(),
            Self::CudaDspark(s) => s.supports_adaptive_ar_fallback(),
        }
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.enter_ar_fallback(),
            Self::CudaDFlash2(s) => s.enter_ar_fallback(),
            Self::CudaDspark(s) => s.enter_ar_fallback(),
        }
    }

    fn device_sampled_rows(&self) -> bool {
        match self {
            Self::PagedDspark(s) => s.device_sampled_rows(),
            Self::CudaDFlash2(s) => s.device_sampled_rows(),
            Self::CudaDspark(s) => s.device_sampled_rows(),
        }
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.materialize_adaptive_state(),
            Self::CudaDFlash2(s) => s.materialize_adaptive_state(),
            Self::CudaDspark(s) => s.materialize_adaptive_state(),
        }
    }

    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        match self {
            Self::PagedDspark(s) => s.reserve_cycle_lookahead(rows),
            Self::CudaDFlash2(s) => s.reserve_cycle_lookahead(rows),
            Self::CudaDspark(s) => s.reserve_cycle_lookahead(rows),
        }
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        match self {
            Self::PagedDspark(s) => s.propose(anchor_id, max_len, params, rng),
            Self::CudaDFlash2(s) => s.propose(anchor_id, max_len, params, rng),
            Self::CudaDspark(s) => s.propose(anchor_id, max_len, params, rng),
        }
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        match self {
            Self::PagedDspark(s) => s.verify_ar_probe(anchor_id),
            Self::CudaDFlash2(s) => s.verify_ar_probe(anchor_id),
            Self::CudaDspark(s) => s.verify_ar_probe(anchor_id),
        }
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.commit_ar_probe(),
            Self::CudaDFlash2(s) => s.commit_ar_probe(),
            Self::CudaDspark(s) => s.commit_ar_probe(),
        }
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        match self {
            Self::PagedDspark(s) => s.verify(verify_ids),
            Self::CudaDFlash2(s) => s.verify(verify_ids),
            Self::CudaDspark(s) => s.verify(verify_ids),
        }
    }

    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        match self {
            Self::PagedDspark(s) => s.verify_device(verify_ids),
            Self::CudaDFlash2(s) => s.verify_device(verify_ids),
            Self::CudaDspark(s) => s.verify_device(verify_ids),
        }
    }

    fn supports_fused_greedy_verify(&self) -> bool {
        match self {
            Self::PagedDspark(s) => s.supports_fused_greedy_verify(),
            Self::CudaDFlash2(s) => s.supports_fused_greedy_verify(),
            Self::CudaDspark(s) => s.supports_fused_greedy_verify(),
        }
    }

    fn verify_greedy(&mut self, verify_ids: &[u32]) -> Result<MxArray> {
        match self {
            Self::PagedDspark(s) => s.verify_greedy(verify_ids),
            Self::CudaDFlash2(s) => s.verify_greedy(verify_ids),
            Self::CudaDspark(s) => s.verify_greedy(verify_ids),
        }
    }

    fn verify_device_greedy(&mut self, verify_ids: &MxArray) -> Result<MxArray> {
        match self {
            Self::PagedDspark(s) => s.verify_device_greedy(verify_ids),
            Self::CudaDFlash2(s) => s.verify_device_greedy(verify_ids),
            Self::CudaDspark(s) => s.verify_device_greedy(verify_ids),
        }
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.commit(keep, total_written),
            Self::CudaDFlash2(s) => s.commit(keep, total_written),
            Self::CudaDspark(s) => s.commit(keep, total_written),
        }
    }

    fn commit_with_provenance(
        &mut self,
        keep: usize,
        total_written: usize,
        verified_ids: &[u32],
    ) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.commit_with_provenance(keep, total_written, verified_ids),
            Self::CudaDFlash2(s) => s.commit_with_provenance(keep, total_written, verified_ids),
            Self::CudaDspark(s) => s.commit_with_provenance(keep, total_written, verified_ids),
        }
    }

    fn finish(self) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.finish(),
            Self::CudaDFlash2(s) => s.finish(),
            Self::CudaDspark(s) => s.finish(),
        }
    }

    fn eval_boundary(&self, token: &MxArray) {
        match self {
            Self::PagedDspark(s) => s.eval_boundary(token),
            Self::CudaDFlash2(s) => s.eval_boundary(token),
            Self::CudaDspark(s) => s.eval_boundary(token),
        }
    }

    fn supports_tree_verify(&self) -> bool {
        match self {
            Self::PagedDspark(s) => s.supports_tree_verify(),
            Self::CudaDFlash2(s) => s.supports_tree_verify(),
            Self::CudaDspark(s) => s.supports_tree_verify(),
        }
    }

    fn observe_emitted_tokens(&mut self, tokens: &[u32]) {
        match self {
            Self::PagedDspark(s) => s.observe_emitted_tokens(tokens),
            Self::CudaDFlash2(s) => s.observe_emitted_tokens(tokens),
            Self::CudaDspark(s) => s.observe_emitted_tokens(tokens),
        }
    }

    fn verify_tree_greedy(
        &mut self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<DsparkTreeGreedyResult> {
        match self {
            Self::PagedDspark(s) => s.verify_tree_greedy(anchor, max_nodes, max_depth),
            Self::CudaDFlash2(s) => s.verify_tree_greedy(anchor, max_nodes, max_depth),
            Self::CudaDspark(s) => s.verify_tree_greedy(anchor, max_nodes, max_depth),
        }
    }

    fn commit_tree(&mut self, accepted_nodes: &[usize], is_contiguous_prefix: bool) -> Result<()> {
        match self {
            Self::PagedDspark(s) => s.commit_tree(accepted_nodes, is_contiguous_prefix),
            Self::CudaDFlash2(s) => s.commit_tree(accepted_nodes, is_contiguous_prefix),
            Self::CudaDspark(s) => s.commit_tree(accepted_nodes, is_contiguous_prefix),
        }
    }
}

impl Inkling2Inner {
    fn prepare_flat_dspark_turn_with<F>(
        &mut self,
        tokens: &[u32],
        params: &ChatParams,
        is_delta: bool,
        alloc_cache: F,
    ) -> Result<FlatDsparkPrefix>
    where
        F: FnOnce(&Self) -> Result<super::cuda_kv::Inkling2CudaCache>,
    {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Inkling2 CUDA speculative turn: empty prompt tokens",
            ));
        }
        let block_size = if let Some(ref dflash2) = self.dflash2 {
            dflash2.config.block_size
        } else if let Some(ref dspark) = self.dspark {
            clamp_cuda_dspark_max_draft(dspark.config.num_draft_tokens())
        } else {
            return Err(Error::from_reason(
                "Inkling2 CUDA speculative turn: no DFlash2 or DSpark draft model loaded",
            ));
        };

        let prior_cached_len = if is_delta {
            self.cached_token_history.len()
        } else {
            0
        };

        // Admit prefix reuse ONLY when both `cuda_cache` and the active draft context match the
        // exact cached token prefix (so a previous turn never leaves stale draft KV behind).
        let cached_len = self.cached_token_history.len();
        let can_reuse = params.reuse_cache
            && self.cached_image_key.is_none()
            && cached_len > 0
            && cached_len < tokens.len()
            && tokens[..cached_len] == self.cached_token_history[..]
            && self
                .cuda_cache
                .as_ref()
                .is_some_and(|c| c.pos_offset == cached_len)
            && if self.dflash2.is_some() {
                self.cuda_dflash2_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.token_history() == &tokens[..cached_len])
            } else {
                self.cuda_dspark_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.committed_len as usize == cached_len)
            };

        if can_reuse {
            return Ok(FlatDsparkPrefix {
                prefill_tokens: tokens[cached_len..].to_vec(),
                cached_prefix_len: cached_len,
                prior_cached_len,
                block_size,
            });
        }

        // Cold reset: drop any existing KV/ShortConv cache and clear session/draft state BEFORE
        // allocating the replacement GPU cache so we never hold 2x KV buffers simultaneously and
        // never leave a stale `cuda_cache` behind if replacement allocation fails.
        self.flat_cache = None;
        self.cuda_cache = None;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
            ctx.reset();
        }
        self.cuda_dspark_ctx = None;

        self.cuda_cache = Some(alloc_cache(self)?);
        if let Some(ref draft) = self.dflash2 {
            if self.cuda_dflash2_ctx.is_none() {
                self.cuda_dflash2_ctx = Some(DFlash2ContextCache::new(&draft.config));
            }
            self.cuda_dspark_ctx = None;
        } else if let Some(ref draft) = self.dspark {
            self.cuda_dspark_ctx = Some(Inkling2DsparkContext::new(draft.config.num_hidden_layers));
            self.cuda_dflash2_ctx = None;
        }

        Ok(FlatDsparkPrefix {
            prefill_tokens: tokens.to_vec(),
            cached_prefix_len: 0,
            prior_cached_len,
            block_size,
        })
    }
}

impl FlatDsparkBackend for Inkling2Inner {
    const OVERWRITE_THINKING_ENABLED: bool = false;

    fn prepare_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        params: &ChatParams,
        is_delta: bool,
    ) -> Result<FlatDsparkPrefix> {
        self.prepare_flat_dspark_turn_with(tokens, params, is_delta, Self::new_cuda_cache)
    }

    fn flat_prefill_with_draft_state(
        &mut self,
        prefix: &FlatDsparkPrefix,
        stream: Stream,
    ) -> Result<MxArray> {
        let _ctx = StreamContext::new(stream);
        if self.is_turn_cancelled() {
            self.cuda_cache = None;
            return Err(Error::from_reason("prefill cancelled"));
        }
        let (tap_visits, tap_tail_limit): (Vec<usize>, Option<usize>) =
            if let Some(ref draft) = self.dflash2 {
                (
                    draft.config.target_layers.clone(),
                    Some(draft.config.sliding_window.saturating_sub(1)),
                )
            } else if let Some(ref draft) = self.dspark {
                (draft.config.target_layer_ids.clone(), None)
            } else {
                return Err(Error::from_reason(
                    "Inkling2 CUDA speculative prefill: no draft model loaded",
                ));
            };

        let mut cache = match self.cuda_cache.take() {
            Some(c) => c,
            None => self.new_cuda_cache()?,
        };
        let (logits_2d, taps) = self.prefill_cuda_tokens(
            &prefix.prefill_tokens,
            None,
            &mut cache,
            Some(&tap_visits),
            tap_tail_limit,
        )?;
        if self.is_turn_cancelled() {
            return Err(Error::from_reason("prefill cancelled"));
        }

        if let Some(ref draft) = self.dflash2 {
            let mut ctx = self
                .cuda_dflash2_ctx
                .take()
                .unwrap_or_else(|| DFlash2ContextCache::new(&draft.config));
            // Scan all prompt tokens up front so adaptive narrow/wide draft
            // head selection is established before the first decode proposal.
            ctx.observe_tokens(&prefix.prefill_tokens);
            let n = prefix.prefill_tokens.len();
            let ring_rows = draft.config.sliding_window.saturating_sub(1);
            let skip = n.saturating_sub(ring_rows);
            if skip > 0 {
                ctx.record_only(&prefix.prefill_tokens[..skip])?;
            }
            let keep = n - skip;
            if keep > 0 {
                let d = self.config.hidden_size as i64;
                let mut taps_3d = Vec::with_capacity(taps.len());
                for tap in &taps {
                    let rows = tap.shape_at(0)? as usize;
                    if rows != keep {
                        return Err(Error::from_reason(format!(
                            "Inkling2 CUDA DFlash2 prefill tap row count mismatch: got {rows}, expected {keep}"
                        )));
                    }
                    taps_3d.push(tap.reshape(&[1, keep as i64, d])?);
                }
                let fused = draft.fuse_context(&taps_3d)?;
                let base = (prefix.cached_prefix_len + skip) as i32;
                ctx.append(draft, &fused, base, &prefix.prefill_tokens[skip..])?;
            }
            ctx.eval()?;
            self.cuda_dflash2_ctx = Some(ctx);
        } else if let Some(ref draft) = self.dspark {
            let mut ctx = self
                .cuda_dspark_ctx
                .take()
                .unwrap_or_else(|| Inkling2DsparkContext::new(draft.config.num_hidden_layers));
            ctx.append_taps(draft, &taps, prefix.prefill_tokens.len())?;
            ctx.eval_state()?;
            self.cuda_dspark_ctx = Some(ctx);
        }

        let logits = logits_2d.squeeze(Some(&[0]))?;
        self.cuda_cache = Some(cache);
        Ok(logits)
    }

    fn eval_after_flat_prefill(&self) -> Result<()> {
        if let Some(ref cache) = self.cuda_cache {
            cache.eval_pending_state()?;
        }
        if let Some(ref ctx) = self.cuda_dflash2_ctx {
            ctx.eval()?;
        }
        if let Some(ref ctx) = self.cuda_dspark_ctx {
            ctx.eval_state()?;
        }
        Ok(())
    }

    fn fail_closed_flat_dspark(&mut self, error: Error) -> Error {
        self.cuda_cache = None;
        self.flat_cache = None;
        if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
            ctx.reset();
        }
        self.cuda_dspark_ctx = None;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        error
    }

    fn finish_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        generated_tokens: &[u32],
        _finish_reason: &str,
        last_in_cache: bool,
        _is_delta: bool,
        reuse_cache: bool,
        _stream: Stream,
    ) -> Result<()> {
        if let Some(ref cache) = self.cuda_cache {
            cache.eval_pending_state()?;
        }
        if let Some(ref ctx) = self.cuda_dflash2_ctx {
            ctx.eval()?;
        }
        if let Some(ref ctx) = self.cuda_dspark_ctx {
            ctx.eval_state()?;
        }

        if reuse_cache {
            let saved_generated = if !last_in_cache && !generated_tokens.is_empty() {
                &generated_tokens[..generated_tokens.len() - 1]
            } else {
                generated_tokens
            };
            let mut history = Vec::with_capacity(tokens.len() + saved_generated.len());
            history.extend_from_slice(tokens);
            history.extend_from_slice(saved_generated);

            let cache_ok = self
                .cuda_cache
                .as_ref()
                .is_some_and(|c| c.pos_offset == history.len());
            let draft_ok = if self.dflash2.is_some() {
                self.cuda_dflash2_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.token_history() == history.as_slice())
            } else if self.dspark.is_some() {
                self.cuda_dspark_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.committed_len as usize == history.len())
            } else {
                false
            };

            if cache_ok {
                self.cached_token_history = history;
                self.cached_image_key = None;
                if !draft_ok {
                    if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
                        ctx.reset();
                    }
                    self.cuda_dspark_ctx = None;
                }
            } else {
                self.cuda_cache = None;
                if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
                    ctx.reset();
                }
                self.cuda_dspark_ctx = None;
                self.cached_token_history.clear();
                self.cached_image_key = None;
            }
        } else {
            self.cuda_cache = None;
            if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
                ctx.reset();
            }
            self.cuda_dspark_ctx = None;
            self.cached_token_history.clear();
            self.cached_image_key = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling2_tree_cost_model_default_values() {
        const {
            assert!(INKLING2_TREE_COST_DEFAULT.bucketed);
            assert!(!INKLING2_LOOKUP_DEFAULT);
        }
        assert!((INKLING2_TREE_COST_DEFAULT.base_ms - 36.0).abs() < 1e-6);
        assert!((INKLING2_TREE_COST_DEFAULT.slope_ms - 1.41).abs() < 1e-6);
        assert_eq!(INKLING2_TREE_COST_DEFAULT.min_nodes, 4);
    }

    #[test]
    fn lookup_min_match_guards_long_prompts_and_honors_override() {
        assert_eq!(resolve_inkling2_lookup_min_match(0, None), 3);
        assert_eq!(resolve_inkling2_lookup_min_match(511, None), 3);
        assert_eq!(resolve_inkling2_lookup_min_match(512, None), 5);
        assert_eq!(resolve_inkling2_lookup_min_match(2048, None), 5);
        assert_eq!(resolve_inkling2_lookup_min_match(2048, Some("3")), 3);
        assert_eq!(resolve_inkling2_lookup_min_match(64, Some("6")), 6);
        assert_eq!(resolve_inkling2_lookup_min_match(2048, Some("1")), 3);
        assert_eq!(resolve_inkling2_lookup_min_match(2048, Some("bad")), 5);
    }

    #[test]
    fn extract_dflash2_tap_rows_contiguous_and_non_contiguous() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tap = MxArray::from_float32(&data, &[4, 4])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let contig =
            extract_dflash2_tap_rows(std::slice::from_ref(&tap), &[0, 1, 2], true, 4).unwrap();
        assert_eq!(contig.len(), 1);
        assert_eq!(contig[0].shape().unwrap().as_ref(), [1, 3, 4]);
        let vals = contig[0]
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(
            vals.as_ref(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );

        let noncontig = extract_dflash2_tap_rows(&[tap], &[0, 2, 3], false, 4).unwrap();
        assert_eq!(noncontig.len(), 1);
        assert_eq!(noncontig[0].shape().unwrap().as_ref(), [1, 3, 4]);
        let vals2 = noncontig[0]
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(
            vals2.as_ref(),
            &[
                0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
            ]
        );
    }

    #[test]
    fn v2_dspark_gguf_metadata_is_recognized() {
        use crate::utils::gguf::{GgufMetaValue, is_inkling2_dspark_gguf};
        use std::collections::HashMap;

        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("dflash".to_string()),
        );
        meta.insert(
            "tokenizer.ggml.pre".to_string(),
            GgufMetaValue::String("inkling2".to_string()),
        );
        assert!(!is_inkling2_dspark_gguf(&meta));

        meta.insert(
            "dflash.has_markov_context".to_string(),
            GgufMetaValue::Bool(true),
        );
        assert!(is_inkling2_dspark_gguf(&meta));
    }

    #[test]
    fn dflash2_draft_dir_detection() {
        let dir = std::env::temp_dir().join(format!(
            "inkling2_dflash2_detect_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(!is_dflash2_draft_dir(&dir));
        fs::write(
            dir.join("config.json"),
            r#"{"architectures":["DFlash2DraftModel"]}"#,
        )
        .unwrap();
        assert!(is_dflash2_draft_dir(&dir));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cuda_dspark_depth_cap_clamps_block_size_gt_16() {
        use super::super::dspark::Inkling2DsparkDraftConfig;

        let make_cfg = |block_size: usize| Inkling2DsparkDraftConfig {
            hidden_size: 2304,
            intermediate_size: 4608,
            num_hidden_layers: 3,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 201_216,
            unpadded_vocab_size: 201_141,
            block_size,
            target_layer_ids: vec![5, 23, 41],
            markov_rank: 256,
            sliding_window: 1024,
            sliding_window_pattern: vec![true, true, false],
            final_logit_softcapping: 30.0,
            max_position_embeddings: 32_768,
        };

        assert_eq!(
            clamp_cuda_dspark_max_draft(make_cfg(8).num_draft_tokens()),
            7
        );
        assert_eq!(
            clamp_cuda_dspark_max_draft(make_cfg(16).num_draft_tokens()),
            15
        );
        // block_size = 18 -> 17 draft tokens -> clamped to 15 (1 anchor + 15 drafts = 16 rows).
        let capped = clamp_cuda_dspark_max_draft(make_cfg(18).num_draft_tokens());
        assert_eq!(capped, CUDA_DSPARK_MAX_DRAFT);
        assert_eq!(capped, 15);
        let depth = AdaptiveDraftDepth::new(capped);
        assert!(depth.current_depth(17) <= CUDA_DSPARK_MAX_DRAFT);
    }

    #[test]
    fn inkling2_dflash2_contract_allows_final_norm_tap() {
        let raw = serde_json::json!({
            "model_type": "inkling2",
            "hidden_size": 2304,
            "num_hidden_layers": 48,
            "num_physical_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "final_logit_softcapping": 30.0,
            "n_layers_per_attn_res_block": 6,
            "loop_config": {
                "core_start": 6,
                "core_end": 18,
                "num_passes": 3,
                "gate_rank": 64,
                "is_varloop": false
            }
        })
        .to_string();
        let cfg = Inkling2Config::from_json_str(&raw, true).unwrap();
        let shape = inkling2_dflash2_target_shape(&cfg);
        assert!(shape.reads_layer_inputs);
        assert_eq!(shape.readable_layers(), 49);
        assert_eq!(shape.num_layers, 48);
        assert_eq!(shape.vocab_size, 201024);
        assert_eq!(shape.valid_vocab_size, 200058);
        assert_eq!(shape.final_logit_softcapping, Some(30.0));
    }

    fn make_test_dspark_inner() -> Inkling2Inner {
        use std::collections::HashMap;
        use std::sync::Arc;

        use mlx_paged_attn::CacheRecency;

        use crate::disk_tier::DiskTierOptions;
        use crate::models::inkling2::cuda_kv::Inkling2CudaKvFormat;
        use crate::models::inkling2::dspark::{
            Inkling2DsparkDraftConfig, Inkling2DsparkModel, Inkling2MarkovHead,
        };
        use crate::models::inkling2::model::{default_decode_share, next_inkling2_model_id};
        use crate::models::inkling2::scheduled_dspark::Inkling2ScheduledDsparkStates;
        use crate::nn::{Linear, RoPE};
        use crate::transformer::Activations;

        let hidden = 16usize;
        let vocab = 32usize;
        let cfg_raw = serde_json::json!({
            "model_type": "inkling2",
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_physical_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": vocab,
            "sliding_window": 16,
            "n_layers_per_attn_res_block": 1
        })
        .to_string();
        let config = Inkling2Config::from_json_str(&cfg_raw, true).unwrap();

        let embed_w =
            MxArray::zeros(&[vocab as i64, hidden as i64], Some(DType::BFloat16)).unwrap();
        let embed_tokens = Embedding::from_weight(&embed_w).unwrap();
        let embed_norm = MxArray::ones(&[hidden as i64], Some(DType::Float32)).unwrap();
        let output_res_query = MxArray::zeros(&[1, hidden as i64], Some(DType::Float32)).unwrap();
        let output_res_query_t = output_res_query.transpose(None).unwrap();
        let output_norm = MxArray::ones(&[hidden as i64], Some(DType::Float32)).unwrap();
        let lm_head = LinearProj::Standard(Linear::from_weights(&embed_w, None).unwrap());

        let rank = 8usize;
        let dspark_cfg = Inkling2DsparkDraftConfig {
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: vocab,
            unpadded_vocab_size: vocab,
            block_size: 8,
            target_layer_ids: vec![0],
            markov_rank: rank,
            sliding_window: 16,
            sliding_window_pattern: vec![false],
            final_logit_softcapping: 30.0,
            max_position_embeddings: 64,
        };
        let dspark = Inkling2DsparkModel {
            config: dspark_cfg,
            layers: Vec::new(),
            output_norm: MxArray::ones(&[hidden as i64], Some(DType::Float32)).unwrap(),
            mask_embd: MxArray::zeros(&[1, hidden as i64], Some(DType::BFloat16)).unwrap(),
            fuse_alpha: vec![vec![1.0], vec![1.0]],
            markov_head: Inkling2MarkovHead {
                markov_rank: rank,
                dflare_norm: MxArray::ones(&[hidden as i64], Some(DType::Float32)).unwrap(),
                markov_w1: MxArray::zeros(&[vocab as i64, rank as i64], Some(DType::BFloat16))
                    .unwrap(),
                rnn_bias_f32: MxArray::zeros(&[1, (3 * rank) as i64], Some(DType::Float32))
                    .unwrap(),
                rnn_proj_rec_w_t: MxArray::zeros(
                    &[(2 * rank) as i64, (3 * rank) as i64],
                    Some(DType::BFloat16),
                )
                .unwrap(),
                rnn_proj_ctx_w_t: MxArray::zeros(
                    &[(2 * hidden) as i64, (3 * rank) as i64],
                    Some(DType::BFloat16),
                )
                .unwrap(),
                fold_weight_q8: None,
                fold_weight_t: None,
                fold_bias: MxArray::zeros(&[1, vocab as i64], Some(DType::BFloat16)).unwrap(),
                one_scalar: MxArray::scalar_float(1.0).unwrap(),
            },
            rope: RoPE::new(8, Some(false), Some(10_000.0), Some(1.0)),
        };

        Inkling2Inner {
            config,
            embed_tokens,
            embed_norm,
            layers: Vec::new(),
            output_res_query,
            output_res_query_t,
            output_norm,
            lm_head,
            lm_head_bf16_unpadded: None,
            vision_encoder: None,
            tokenizer: None,
            dspark: Some(dspark),
            dflash2: None,
            dflash2_identity: None,
            dspark_turn_contexts: HashMap::new(),
            cuda_dflash2_ctx: None,
            cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
            cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
            cuda_dspark_ctx: None,
            scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
            paged: None,
            flat_cache: None,
            cuda_cache: None,
            cuda_kv_format: Inkling2CudaKvFormat::Bf16,
            cached_token_history: Vec::new(),
            cached_image_key: Some(42),
            turn_cancel: None,
            model_id: next_inkling2_model_id(),
            memory_plan: None,
            memory_registration: None,
            wired: None,
            cache_recency: Arc::new(CacheRecency::new()),
            decode_share: default_decode_share(),
            max_context: None,
            activations: Activations::Auto,
            disk_options: DiskTierOptions::default(),
            prefill_runs: HashMap::new(),
        }
    }

    #[test]
    fn finish_flat_dspark_turn_preserves_session_after_ar_fallback_and_cold_replays_next_spec_turn()
    {
        use crate::engine::backend::ChatBackend;

        let mut inner = make_test_dspark_inner();

        // Simulate `Inkling2CudaDsparkStepper::finish` after adaptive AR fallback:
        // `cuda_cache` matches full `history` (`cache_ok == true`), while `cuda_dspark_ctx`
        // was cleared to `None` (`draft_ok == false`).
        let prompt = [1u32, 2, 3, 4];
        let generated = [5u32, 6, 7];
        let expected_history = vec![1u32, 2, 3, 4, 5, 6, 7];
        let mut cache = inner.new_cuda_cache().unwrap();
        cache.commit_linear(expected_history.len());
        inner.cuda_cache = Some(cache);
        inner.cuda_dspark_ctx = None;

        inner
            .finish_flat_dspark_turn(
                &prompt,
                &generated,
                "stop",
                true,
                false,
                true,
                Stream::generation(),
            )
            .unwrap();

        // Live session and target KV cache must remain intact; draft context stays cleared.
        assert!(inner.cuda_cache.is_some());
        assert!(inner.has_live_session());
        assert_eq!(inner.cached_token_history, expected_history);
        assert_eq!(inner.cached_image_key, None);
        assert!(inner.cuda_dspark_ctx.is_none());

        // A follow-up non-speculative continuation can reuse `cuda_cache` directly.
        let next_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(
            inner.verify_cache_prefix(&next_tokens, true),
            expected_history.len()
        );

        // A follow-up speculative continuation detects the missing draft context and cleanly
        // cold-replays the full prompt (`cached_prefix_len == 0`) instead of reusing stale state.
        let mut params =
            crate::engine::extract_chat_params(&crate::engine::types::ChatConfig::default());
        params.reuse_cache = true;
        let prefix = inner
            .prepare_flat_dspark_turn(&next_tokens, &params, true)
            .unwrap();
        assert_eq!(prefix.cached_prefix_len, 0);
        assert_eq!(prefix.prefill_tokens, next_tokens);
        assert_eq!(prefix.prior_cached_len, expected_history.len());
        assert_eq!(inner.cuda_cache.as_ref().unwrap().pos_offset, 0);
        assert_eq!(inner.cuda_dspark_ctx.as_ref().unwrap().committed_len, 0);

        // When `cuda_cache` does NOT match `history` (`cache_ok == false`), fail closed.
        inner
            .finish_flat_dspark_turn(
                &prompt,
                &generated,
                "stop",
                true,
                false,
                true,
                Stream::generation(),
            )
            .unwrap();
        assert!(inner.cuda_cache.is_none());
        assert!(!inner.has_live_session());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cuda_dspark_ctx.is_none());
    }

    #[test]
    fn prepare_flat_dspark_cold_reset_drops_old_cache_before_alloc_and_fails_closed_on_err() {
        use crate::engine::backend::ChatBackend;

        let mut inner = make_test_dspark_inner();

        // Populate a live post-AR-fallback session (`cuda_cache = Some(...)`, `cuda_dspark_ctx = None`).
        let history = vec![1u32, 2, 3, 4, 5];
        let mut live_cache = inner.new_cuda_cache().unwrap();
        live_cache.commit_linear(history.len());
        inner.cuda_cache = Some(live_cache);
        inner.cached_token_history = history.clone();
        inner.cached_image_key = Some(99);
        inner.cuda_dspark_ctx = None;
        assert!(inner.has_live_session());

        let mut params =
            crate::engine::extract_chat_params(&crate::engine::types::ChatConfig::default());
        params.reuse_cache = true;
        let next_tokens = vec![1u32, 2, 3, 4, 5, 6, 7];

        // Inject an allocation failure during cold reset: the old `cuda_cache` and session state
        // MUST already be released before the replacement allocator runs!
        let mut allocator_called = false;
        let err = inner
            .prepare_flat_dspark_turn_with(&next_tokens, &params, true, |state| {
                allocator_called = true;
                assert!(
                    state.cuda_cache.is_none(),
                    "old cuda_cache must be dropped before allocating replacement"
                );
                assert!(state.flat_cache.is_none());
                assert!(state.cached_token_history.is_empty());
                assert!(state.cached_image_key.is_none());
                assert!(state.cuda_dspark_ctx.is_none());
                Err(Error::from_reason("injected OOM"))
            })
            .err()
            .expect("injected allocation failure must propagate");
        assert!(allocator_called);
        assert!(err.to_string().contains("injected OOM"));
        assert!(inner.cuda_cache.is_none());
        assert!(inner.flat_cache.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cuda_dspark_ctx.is_none());
        assert!(!inner.has_live_session());

        // Also verify cold reset when a prior `cuda_dspark_ctx` was populated (`reuse_cache = false`):
        // stale draft context must be cleared before replacement allocation runs.
        let mut live_cache2 = inner.new_cuda_cache().unwrap();
        live_cache2.commit_linear(history.len());
        inner.cuda_cache = Some(live_cache2);
        inner.cached_token_history = history;
        let mut stale_ctx = Inkling2DsparkContext::new(1);
        stale_ctx.committed_len = 5;
        inner.cuda_dspark_ctx = Some(stale_ctx);
        params.reuse_cache = false;

        let err2 = inner
            .prepare_flat_dspark_turn_with(&next_tokens, &params, false, |state| {
                assert!(state.cuda_cache.is_none());
                assert!(state.cuda_dspark_ctx.is_none());
                Err(Error::from_reason("injected OOM 2"))
            })
            .err()
            .expect("second injected allocation failure must propagate");
        assert!(err2.to_string().contains("injected OOM 2"));
        assert!(inner.cuda_cache.is_none());
        assert!(inner.cuda_dspark_ctx.is_none());
        assert!(!inner.has_live_session());
    }

    #[test]
    fn cuda_prefill_honors_turn_cancel_and_fails_closed() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        use crate::engine::backend::ChatBackend;
        use crate::models::inkling2::cuda_forward::cuda_fast_config_supported;
        use crate::nn::Linear;

        let mut inner = make_test_dspark_inner();
        let cancel = Arc::new(AtomicBool::new(true));
        inner.set_turn_cancel_flag(Some(Arc::clone(&cancel)));

        // 1. Direct `prefill_cuda_tokens` immediately aborts without advancing `cache.pos_offset`.
        let mut cache = inner.new_cuda_cache().unwrap();
        assert_eq!(cache.pos_offset, 0);
        let err = inner
            .prefill_cuda_tokens(&[1, 2, 3], None, &mut cache, None, None)
            .err()
            .expect("pre-cancelled prefill_cuda_tokens must fail");
        assert!(
            err.to_string().contains("prefill cancelled"),
            "unexpected error: {err}"
        );
        assert_eq!(cache.pos_offset, 0);

        // 2. `prefill_cuda_flat` drops `cuda_cache` (fail-closed) when cancelled.
        inner.cuda_cache = Some(cache);
        let err = inner
            .prefill_cuda_flat(&[1, 2, 3])
            .err()
            .expect("pre-cancelled prefill_cuda_flat must fail");
        assert!(
            err.to_string().contains("prefill cancelled"),
            "unexpected error: {err}"
        );
        assert!(inner.cuda_cache.is_none());

        // 3. `flat_prefill_with_draft_state` drops `cuda_cache` (fail-closed) when cancelled.
        inner.cuda_cache = Some(inner.new_cuda_cache().unwrap());
        let prefix = FlatDsparkPrefix {
            prefill_tokens: vec![1, 2, 3],
            cached_prefix_len: 0,
            prior_cached_len: 0,
            block_size: 8,
        };
        let err = inner
            .flat_prefill_with_draft_state(&prefix, Stream::generation())
            .err()
            .expect("pre-cancelled flat_prefill_with_draft_state must fail");
        assert!(
            err.to_string().contains("prefill cancelled"),
            "unexpected error: {err}"
        );
        assert!(inner.cuda_cache.is_none());

        // 4. Mid-prefill error from `prefill_cuda_tokens` (after `.take()`) also fails closed.
        inner.set_turn_cancel_flag(None);
        inner.cuda_cache = Some(inner.new_cuda_cache().unwrap());
        assert!(inner.prefill_cuda_flat(&[]).is_err());
        assert!(inner.cuda_cache.is_none());

        inner.cuda_cache = Some(inner.new_cuda_cache().unwrap());
        let empty_prefix = FlatDsparkPrefix {
            prefill_tokens: Vec::new(),
            cached_prefix_len: 0,
            prior_cached_len: 0,
            block_size: 8,
        };
        assert!(
            inner
                .flat_prefill_with_draft_state(&empty_prefix, Stream::generation())
                .is_err()
        );
        assert!(inner.cuda_cache.is_none());

        // 5. Exercise `ChatBackend::prefill` directly through the live CUDA fast path when
        //    supported on this host (`hidden_size = 2304`, 1-layer minimal config).
        let fast_cfg_raw = serde_json::json!({
            "model_type": "inkling2",
            "hidden_size": 2304,
            "num_hidden_layers": 1,
            "num_physical_layers": 1,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 64,
            "sliding_window": 1024,
            "n_layers_per_attn_res_block": 1
        })
        .to_string();
        let fast_cfg = Inkling2Config::from_json_str(&fast_cfg_raw, true).unwrap();
        assert!(cuda_fast_config_supported(&fast_cfg));

        let embed_w = MxArray::zeros(&[64, 2304], Some(DType::BFloat16)).unwrap();
        inner.config = fast_cfg;
        inner.embed_tokens = crate::nn::Embedding::from_weight(&embed_w).unwrap();
        inner.embed_norm = MxArray::ones(&[2304], Some(DType::Float32)).unwrap();
        inner.output_res_query = MxArray::zeros(&[1, 2304], Some(DType::Float32)).unwrap();
        inner.output_res_query_t = inner.output_res_query.transpose(None).unwrap();
        inner.output_norm = MxArray::ones(&[2304], Some(DType::Float32)).unwrap();
        inner.lm_head = crate::models::quantized_linear::LinearProj::Standard(
            Linear::from_weights(&embed_w, None).unwrap(),
        );

        if inner.use_cuda_fast() {
            inner.cuda_cache = Some(inner.new_cuda_cache().unwrap());
            inner.set_turn_cancel_flag(Some(Arc::clone(&cancel)));
            let err = ChatBackend::prefill(&mut inner, &[1, 2, 3], Stream::generation())
                .err()
                .expect("pre-cancelled ChatBackend::prefill must fail on CUDA fast path");
            assert!(
                err.to_string().contains("prefill cancelled"),
                "unexpected error: {err}"
            );
            assert!(inner.cuda_cache.is_none());

            // Also confirm non-cancel `prefill_cuda_tokens` error leaves `cuda_cache == None`.
            inner.set_turn_cancel_flag(None);
            inner.cuda_cache = Some(inner.new_cuda_cache().unwrap());
            assert!(ChatBackend::prefill(&mut inner, &[], Stream::generation()).is_err());
            assert!(inner.cuda_cache.is_none());
        }
    }
}
