//! Inkling-2-Mini DFlash2 target adapter, context wrapper, and companion discovery.
//!
//! Bridges Inkling-2-Mini's `embed_tokens + embed_norm`, shared `lm_head`
//! (`Sym-Q8_0` / `BF16`, unpadded `200058` columns, raw un-softcapped logits),
//! and 48-visit pre-layer/final-norm taps (`[5, 23, 41, 48]`) into the shared
//! [`DFlash2Model`] drafter.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::engine::backend::DsparkProposal;
use crate::engine::params::ChatParams;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{
    DFlash2BatchProposalLane, DFlash2Config, DFlash2ContextCache, DFlash2Model, DFlash2TargetShape,
    DFlash2TargetView, SelectorPath, read_dflash2_config,
};
use crate::models::qwen3_5::verify_tree::{
    DRAFT_SLOTS, DraftLattice, LookupAcceptance, LookupIndex, MAX_TREE_NODES, TreeCostModel,
    VerifyBudget, VerifyTree, build_verify_tree, build_verify_tree_cost_aware,
    cuda::{TreeLookupQuery, prepare_tree_lookup},
};
use crate::nn::Embedding;
use crate::sampling::is_greedy_temperature;
use crate::transformer::paged_kv_cache_adapter::SeqId;

use super::attn_res::apply_sublayer_rms_norm;
use super::dspark::Inkling2DsparkContext;
use super::kernels::{Inkling2TreeRow, encode_inkling2_tree_rows};
use super::model::Inkling2Inner;

/// Calibrated Inkling-2-Mini MoE verify-tree step-cost curve for M5 Max:
/// `min_nodes = 8` matches Splash's `max(passed, kBlockRows)` floor (never verify
/// fewer than the full 8-row DFlash2 block unless `remaining_tokens < 8` or the KV
/// pool is tight), with `Cost(T) = 14.0 + 1.2 * T` for `8..=16` (`Cost(8) = 23.6 ms`,
/// `Cost(16) = 33.2 ms`, unbucketed since Metal kernels support every `T in 1..=16`).
pub(crate) const INKLING2_TREE_COST_DEFAULT: TreeCostModel = TreeCostModel {
    base_ms: 14.0,
    slope_ms: 1.2,
    min_nodes: 8,
    bucketed: false,
};

/// Default minimum SAM suffix match length required to trigger `M = 8..16` Verify Tree
/// in [`Inkling2VerifyTreeMode::Auto`] (`MLX_INKLING2_TREE_SAM_MIN_LEN`).
pub(crate) const INKLING2_TREE_SAM_MIN_LEN_DEFAULT: usize = 4;

/// Inkling-2-Mini DFlash2 logistic acceptance prior (higher multi-slot acceptance
/// than Qwen3.8's `VerifyBudget::PRIOR`, with faster 16-cycle MAP refit).
pub(crate) const INKLING2_BUDGET_SLOPE: f64 = 0.7;
pub(crate) const INKLING2_BUDGET_INTERCEPTS: [f64; DRAFT_SLOTS] =
    [0.0, -0.15, -0.35, -0.45, -0.50, -0.55, -0.30];
pub(crate) const INKLING2_BUDGET_REFIT_CYCLES: usize = 16;

/// Verify-tree routing mode controlled by `MLX_INKLING2_VERIFY_TREE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Inkling2VerifyTreeMode {
    /// Adaptive zero-sync hybrid default: run zero-sync overlapped 1D DFlash2 chain
    /// (`M=8`, `p ~ 0.5 ms`) unless SAM prompt lookup finds a suffix match of length
    /// `>= sam_trigger_len` (default 4) with continuation tokens and `remaining >= 4`.
    Auto,
    /// Always run Verify Tree + SAM (`M = 8..16`) on every greedy cycle.
    Always,
    /// Always run 1D DFlash2 chain (`M = 8`).
    Off,
}

impl Inkling2VerifyTreeMode {
    /// Parse `MLX_INKLING2_VERIFY_TREE` value (`auto|1|true|on|hybrid`, `always|force|all`,
    /// `0|false|off`; unset defaults to [`Self::Auto`]).
    pub(crate) fn parse(raw: Option<&str>) -> Self {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Self::Auto;
        };
        if trimmed.eq_ignore_ascii_case("0")
            || trimmed.eq_ignore_ascii_case("false")
            || trimmed.eq_ignore_ascii_case("off")
        {
            Self::Off
        } else if trimmed.eq_ignore_ascii_case("always")
            || trimmed.eq_ignore_ascii_case("force")
            || trimmed.eq_ignore_ascii_case("all")
        {
            Self::Always
        } else {
            Self::Auto
        }
    }
}

/// Runtime controls for Inkling-2-Mini DFlash2 verify-tree + prompt-lookup decoding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inkling2VerifyTreePolicy {
    /// Routing mode (`MLX_INKLING2_VERIFY_TREE=auto|always|off`, default `Auto`).
    pub(crate) mode: Inkling2VerifyTreeMode,
    /// Suffix-automaton prompt lookup switch (`MLX_INKLING2_LOOKUP=0|1`, default `true`).
    pub(crate) lookup_enabled: bool,
    /// Minimum SAM suffix match length that triggers Verify Tree in [`Inkling2VerifyTreeMode::Auto`]
    /// (`MLX_INKLING2_TREE_SAM_MIN_LEN`, default `4`, clamped `3..=64`).
    pub(crate) sam_trigger_len: usize,
    /// Upper bound on tree rows (`MLX_INKLING2_MAX_TREE_ROWS`, default `16`, clamped `1..=16`).
    pub(crate) max_tree_rows: usize,
    /// `None` (`MLX_INKLING2_VERIFY_ROWS=auto`, default) uses cost-aware sizing via
    /// [`Self::cost_model`]; `Some(n)` (`all` or `<N>`) builds a fixed `min(n, max_tree_rows)` tree.
    pub(crate) fixed_rows: Option<usize>,
    /// Linear MoE step-cost curve (`MLX_INKLING2_TREE_COST_BASE` / `MLX_INKLING2_TREE_COST_SLOPE`).
    pub(crate) cost_model: TreeCostModel,
    /// Emit per-cycle tree diagnostics when `MLX_INKLING2_TREE_STATS=1`.
    pub(crate) log_stats: bool,
}

impl Default for Inkling2VerifyTreePolicy {
    fn default() -> Self {
        Self {
            mode: Inkling2VerifyTreeMode::Auto,
            lookup_enabled: true,
            sam_trigger_len: INKLING2_TREE_SAM_MIN_LEN_DEFAULT,
            max_tree_rows: MAX_TREE_NODES,
            fixed_rows: None,
            cost_model: INKLING2_TREE_COST_DEFAULT,
            log_stats: false,
        }
    }
}

impl Inkling2VerifyTreePolicy {
    /// Parse Inkling-2-Mini verify-tree policy from environment variables.
    pub(crate) fn from_env() -> Self {
        let mode_raw = std::env::var("MLX_INKLING2_VERIFY_TREE").ok();
        let mode = Inkling2VerifyTreeMode::parse(mode_raw.as_deref());
        let lookup_enabled = !matches!(
            std::env::var("MLX_INKLING2_LOOKUP")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0" | "false" | "FALSE" | "off" | "OFF")
        );
        let sam_trigger_len = std::env::var("MLX_INKLING2_TREE_SAM_MIN_LEN")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .map_or(INKLING2_TREE_SAM_MIN_LEN_DEFAULT, |n| {
                n.clamp(LookupIndex::MIN_MATCH, LookupIndex::MAX_MATCH)
            });
        let max_tree_rows = std::env::var("MLX_INKLING2_MAX_TREE_ROWS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .map_or(MAX_TREE_NODES, |n| n.clamp(1, MAX_TREE_NODES));

        let fixed_rows = std::env::var("MLX_INKLING2_VERIFY_ROWS")
            .ok()
            .and_then(|raw| {
                let s = raw.trim();
                if s.is_empty() || s.eq_ignore_ascii_case("auto") {
                    None
                } else if s.eq_ignore_ascii_case("all") {
                    Some(max_tree_rows)
                } else {
                    s.parse::<usize>().ok().map(|n| n.clamp(1, max_tree_rows))
                }
            });

        let mut cost_model = INKLING2_TREE_COST_DEFAULT;
        cost_model.min_nodes = cost_model.min_nodes.min(max_tree_rows);
        if let Some(min_nodes) = std::env::var("MLX_INKLING2_TREE_MIN_NODES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            cost_model.min_nodes = min_nodes.clamp(1, max_tree_rows);
        }
        if let Some(base) = std::env::var("MLX_INKLING2_TREE_COST_BASE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            cost_model.base_ms = base;
        }
        if let Some(slope) = std::env::var("MLX_INKLING2_TREE_COST_SLOPE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
        {
            cost_model.slope_ms = slope;
        }
        let log_stats = matches!(
            std::env::var("MLX_INKLING2_TREE_STATS")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );

        Self {
            mode,
            lookup_enabled,
            sam_trigger_len,
            max_tree_rows,
            fixed_rows,
            cost_model,
            log_stats,
        }
    }

    /// True when the stepper should advertise tree-verify support (so `run_dspark_turn`
    /// feeds prompt & emitted tokens into SAM lookup on every cycle).
    #[inline]
    pub(crate) fn tree_path_active(&self) -> bool {
        self.mode != Inkling2VerifyTreeMode::Off
    }

    /// O(1) per-cycle routing decision (`< 1 µs`, pure CPU, non-mutating):
    /// returns `true` when this cycle should run Verify Tree + SAM (`M = 8..16`),
    /// or `false` when it should stay on the zero-sync overlapped 1D chain (`M = 8`).
    #[inline]
    pub(crate) fn should_use_tree_cycle(&self, lookup_match_len: usize, max_depth: usize) -> bool {
        match self.mode {
            Inkling2VerifyTreeMode::Off => false,
            Inkling2VerifyTreeMode::Always => true,
            Inkling2VerifyTreeMode::Auto => {
                // `max_depth >= 3` corresponds to `remaining >= 4` tokens.
                self.lookup_enabled && max_depth >= 3 && lookup_match_len >= self.sam_trigger_len
            }
        }
    }

    /// Build a DFS-preorder [`VerifyTree`] for one greedy Inkling-2-Mini DFlash2 cycle,
    /// combining the Markov `lattice`, suffix-automaton `lookup`, calibrated `verify_budget`,
    /// and either cost-aware MoE sizing (`fixed_rows == None`) or a fixed node cap.
    pub(crate) fn build_tree(
        &self,
        anchor: u32,
        lattice: &DraftLattice,
        lookup: &TreeLookupQuery,
        max_nodes: usize,
        verify_budget: &VerifyBudget,
        lookup_acc: &LookupAcceptance,
    ) -> VerifyTree {
        let cap = max_nodes.min(self.max_tree_rows).clamp(1, MAX_TREE_NODES);
        match self.fixed_rows {
            Some(fixed) => build_verify_tree(
                anchor,
                lattice,
                &lookup.proposals,
                lookup.match_len,
                cap.min(fixed).max(1),
                Some(lookup.eff_max_depth),
                false,
                verify_budget,
                lookup_acc,
            ),
            None => build_verify_tree_cost_aware(
                anchor,
                lattice,
                &lookup.proposals,
                lookup.match_len,
                cap,
                Some(lookup.eff_max_depth),
                false,
                verify_budget,
                lookup_acc,
                &self.cost_model,
            ),
        }
    }
}

/// Returns `true` iff `tree` is a strict linear chain `[0 -> 1 -> ... -> M-1]`
/// where `parent[0] == -1`, `depth[0] == 0`, and every child `i > 0` has
/// `parent[i] == i - 1` and `depth[i] == i`. Any branching tree returns `false`.
pub(crate) fn is_strict_linear_chain(tree: &VerifyTree) -> bool {
    !tree.is_empty()
        && tree.parent[0] == -1
        && tree.depth[0] == 0
        && (1..tree.len()).all(|i| tree.parent[i] == (i as i32) - 1 && tree.depth[i] == i as i32)
}

/// Convert a DFS-preorder [`VerifyTree`] into Track 2's packed `[T, 2]` `UInt32`
/// tree descriptor plus per-node `depths` (`Vec<u16>`).
///
/// Validates all topological and ancestor-mask invariants required by
/// `inkling2_paged_attention` and the 4 tree ShortConv kernels:
/// - `1 <= T <= MAX_TREE_NODES` (`16`)
/// - Root `0`: `parent == u16::MAX`, `depth == 0`, `ancestors == 1`
/// - Every child `i > 0`: `0 <= parent[i] < i`, `depth[i] == depth[parent[i]] + 1`,
///   and `ancestors[i]` contains both self bit `1 << i` and `ancestors[parent[i]]`
pub(crate) fn verify_tree_to_inkling2_rows(tree: &VerifyTree) -> Result<(MxArray, Vec<u16>)> {
    let t = tree.len();
    if t == 0 || t > MAX_TREE_NODES {
        return Err(Error::from_reason(format!(
            "Inkling2 verify tree length {t} outside 1..={MAX_TREE_NODES}"
        )));
    }
    let mut rows = Vec::with_capacity(t);
    let mut depths = Vec::with_capacity(t);
    for i in 0..t {
        let parent_i32 = tree.parent[i];
        let depth_i32 = tree.depth[i];
        let mask = tree.ancestors_mask[i];
        let parent = if i == 0 {
            if parent_i32 != -1 || depth_i32 != 0 || mask != 1 {
                return Err(Error::from_reason(format!(
                    "Inkling2 verify tree root invariant violated: parent={parent_i32} depth={depth_i32} mask={mask:#x}"
                )));
            }
            u16::MAX
        } else {
            if parent_i32 < 0 || (parent_i32 as usize) >= i {
                return Err(Error::from_reason(format!(
                    "Inkling2 verify tree non-topological parent[{i}]={parent_i32}"
                )));
            }
            let parent_idx = parent_i32 as usize;
            let expected_depth = tree.depth[parent_idx] + 1;
            if depth_i32 != expected_depth || depth_i32 > MAX_TREE_NODES as i32 {
                return Err(Error::from_reason(format!(
                    "Inkling2 verify tree invalid depth[{i}]={depth_i32} (expected {expected_depth})"
                )));
            }
            let parent_mask = tree.ancestors_mask[parent_idx];
            if (mask & (1u32 << i)) == 0 || (mask & parent_mask) != parent_mask {
                return Err(Error::from_reason(format!(
                    "Inkling2 verify tree ancestors[{i}]={mask:#x} inconsistent with self/parent {parent_mask:#x}"
                )));
            }
            parent_i32 as u16
        };
        let depth = depth_i32 as u16;
        rows.push(Inkling2TreeRow {
            ancestors: mask,
            parent,
            depth,
        });
        depths.push(depth);
    }
    let encoded = encode_inkling2_tree_rows(&rows)?;
    Ok((encoded, depths))
}

/// Inkling-2-Mini target view for [`DFlash2Model`]:
/// - [`Self::embed`] runs `embed_tokens` followed by `embed_norm` in BF16 (matching
///   Splash `Inkling2LoopTarget::addDraftEmbedding`).
/// - [`Self::logits`] projects ONLY the `L` proposal rows (`project_proposals_only() == true`)
///   through the target's shared `Sym-Q8_0` or `BF16` `lm_head` across the active
///   unpadded vocabulary `[0..200058)` with **no** softcap (`softcap = 0.0`, since
///   Inkling-2-Mini DFlash2 applies `final_logit_softcapping = 30.0` in the selector
///   after edge correction).
pub(crate) struct Inkling2TargetView<'a> {
    pub(crate) embed_tokens: &'a Embedding,
    pub(crate) embed_norm: &'a MxArray,
    pub(crate) lm_head: &'a LinearProj,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) lm_head_bf16_unpadded: Option<&'a MxArray>,
    pub(crate) rms_norm_eps: f32,
    pub(crate) unpadded_vocab_size: usize,
}

impl DFlash2TargetView for Inkling2TargetView<'_> {
    fn embed(&mut self, block: &MxArray, _ids: &[i32]) -> Result<MxArray> {
        let raw = self.embed_tokens.forward(block)?;
        let batch = raw.shape_at(0)?;
        let rows = raw.shape_at(1)?;
        let hidden = raw.shape_at(2)?;
        let total_rows = batch * rows;
        let flat = raw.reshape(&[total_rows, hidden])?;
        let normed =
            apply_sublayer_rms_norm(&flat, self.embed_norm, self.rms_norm_eps, DType::BFloat16)?;
        normed.reshape(&[batch, rows, hidden])
    }

    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray> {
        let batch = hidden.shape_at(0)?;
        let rows = hidden.shape_at(1)?;
        let dim = hidden.shape_at(2)?;
        let total_rows = batch * rows;
        let x2d_raw = hidden.reshape(&[total_rows, dim])?;
        let x2d = if x2d_raw.dtype()? == DType::BFloat16 {
            x2d_raw
        } else {
            x2d_raw.astype(DType::BFloat16)?
        };

        #[cfg(target_os = "macos")]
        if let LinearProj::Quantized(ql) = self.lm_head
            && ql.is_symmetric_q8()
            && unsafe { mlx_sys::mlx_metal_is_available() }
            && (1..=32).contains(&total_rows)
            && dim > 0
            && dim % 32 == 0
        {
            let w = ql.get_weight();
            let sc = ql.get_scales();
            let vocab_cols = w.shape_at(0)? as usize;
            let out_cols = self.unpadded_vocab_size.min(vocab_cols);
            let logits_2d = super::kernels::inkling2_lm_head_q8_bf16(
                &x2d,
                w,
                sc,
                out_cols,
                0.0,
                DType::Float32,
            )?;
            return logits_2d.reshape(&[batch, rows, out_cols as i64]);
        }

        #[cfg(target_os = "macos")]
        if let LinearProj::Standard(lin) = self.lm_head
            && lin.get_bias().is_none()
            && unsafe { mlx_sys::mlx_metal_is_available() }
            && (1..=32).contains(&total_rows)
            && dim > 0
            && dim % 4 == 0
        {
            let w = lin.get_weight();
            if w.dtype()? == DType::BFloat16 {
                let vocab_cols = w.shape_at(0)? as usize;
                let out_cols = self.unpadded_vocab_size.min(vocab_cols);
                let w_unpadded_owned;
                let w_unpadded = if let Some(cached) = self.lm_head_bf16_unpadded
                    && cached.shape_at(0)? as usize == out_cols
                {
                    cached
                } else if out_cols < vocab_cols {
                    w_unpadded_owned = w.slice_axis(0, 0, out_cols as i64)?;
                    &w_unpadded_owned
                } else {
                    &w
                };
                let logits_2d = super::kernels::inkling2_bf16_linear(&x2d, w_unpadded)?;
                return logits_2d.reshape(&[batch, rows, out_cols as i64]);
            }
        }

        let raw_2d = super::attention::forward_linear_proj_fast(self.lm_head, &x2d)?;
        let vocab_cols = raw_2d.shape_at(1)? as usize;
        let out_cols = self.unpadded_vocab_size.min(vocab_cols);
        let sliced = if out_cols < vocab_cols {
            raw_2d.slice_axis(1, 0, out_cols as i64)?
        } else {
            raw_2d
        };
        sliced.reshape(&[batch, rows, out_cols as i64])
    }

    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)> {
        None
    }

    fn project_proposals_only(&self) -> bool {
        true
    }
}

/// Per-sequence DFlash2 context wrapper for Inkling-2-Mini, holding the sliding
/// draft KV cache plus per-sequence suffix-automaton prompt lookup and online
/// verify-tree calibration state.
pub(crate) struct Inkling2Dflash2Context {
    pub(crate) cache: DFlash2ContextCache,
    pub(crate) lookup: LookupIndex,
    pub(crate) lookup_acc: LookupAcceptance,
    pub(crate) verify_budget: VerifyBudget,
}

/// One lane participating in [`Inkling2Dflash2Context::propose_blocks_batched`].
pub(crate) struct BatchedDflash2Lane<'a> {
    pub(crate) ctx: &'a mut Inkling2Dflash2Context,
    pub(crate) anchor_id: u32,
    pub(crate) max_drafts: usize,
    pub(crate) params: &'a ChatParams,
    pub(crate) confidence: bool,
}

impl Inkling2Dflash2Context {
    pub(crate) fn new(config: &DFlash2Config) -> Self {
        Self {
            cache: DFlash2ContextCache::new(config),
            lookup: LookupIndex::new(),
            lookup_acc: LookupAcceptance::new(),
            verify_budget: VerifyBudget::with_prior(
                INKLING2_BUDGET_SLOPE,
                INKLING2_BUDGET_INTERCEPTS,
                INKLING2_BUDGET_REFIT_CYCLES,
            ),
        }
    }

    /// Reset the suffix-automaton index at the start of a whole-turn decode
    /// (before `run_dspark_turn` feeds the full conversation prompt history),
    /// while preserving calibrated `lookup_acc` and `verify_budget` across turns.
    pub(crate) fn reset_lookup_for_turn(&mut self) {
        self.lookup = LookupIndex::new();
    }

    /// Feed prompt or newly emitted tokens into the suffix-automaton lookup index.
    pub(crate) fn observe_tokens(&mut self, tokens: &[u32]) {
        if !tokens.is_empty() {
            self.lookup.append(tokens);
        }
    }

    /// Non-mutating O(1) check returning the current suffix-automaton match length
    /// (`0` when no usable continuation tokens follow the earlier occurrence).
    #[inline]
    pub(crate) fn peek_lookup_match_len(&self) -> usize {
        let info = self.lookup.match_info();
        if info.match_len >= LookupIndex::MIN_MATCH && !self.lookup.proposal_slice(15).is_empty() {
            info.match_len
        } else {
            0
        }
    }

    /// Snapshot suffix-automaton prompt-lookup proposals for one verify-tree cycle.
    pub(crate) fn prepare_lookup(&self, lookup_enabled: bool, max_depth: usize) -> TreeLookupQuery {
        prepare_tree_lookup(&self.lookup, lookup_enabled, max_depth)
    }

    pub(crate) fn committed_len(&self) -> i32 {
        self.cache.logical_len()
    }

    /// Total resident device + host bytes held by this DFlash2 context:
    /// sliding draft KV ring buffers + host `token_history` capacity + SAM `LookupIndex` heap.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.cache
            .buffer_bytes()
            .saturating_add(self.cache.host_bytes())
            .saturating_add(self.lookup.heap_bytes())
    }

    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.resident_bytes()
    }

    /// MLX GPU buffer bytes held by the sliding draft KV ring (freed from MLX active memory on drop).
    pub(crate) fn mlx_buffer_bytes(&self) -> u64 {
        self.cache.buffer_bytes()
    }

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.cache.collect_arrays(out);
    }

    pub(crate) fn eval_state(&self) -> Result<()> {
        self.cache.eval()
    }

    /// Convert Inkling-2-Mini's 2D `[T, hidden]` target taps into `[1, keep, hidden]` BF16,
    /// fuse via `draft.fuse_context`, and append to the sliding DFlash2 KV ring.
    ///
    /// Returns the `[1, keep, D]` fused context tensor for live 4-bit [`DraftContextWindow`]
    /// capture without running a second `fc` + `hidden_norm` pass.
    pub(crate) fn append_taps(
        &mut self,
        draft: &DFlash2Model,
        taps: &[MxArray],
        token_ids: &[u32],
    ) -> Result<Option<MxArray>> {
        let keep = token_ids.len();
        if keep == 0 {
            return Ok(None);
        }
        let expected = draft.config.target_layers.len();
        if taps.len() != expected {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 append_taps expected {expected} taps, got {}",
                taps.len()
            )));
        }
        let total_rows = taps[0].shape_at(0)? as usize;
        if keep > total_rows {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 append_taps keep={keep} exceeds tap rows {total_rows}"
            )));
        }
        let base = self.cache.logical_len();

        // Keep FC and merged K/V on the M=8 Metal NAX route whenever the verify
        // block has 8 rows (same optimization as Qwen3.8 DFlash2).
        if total_rows == 8 && draft.config.block_size == 7 && keep <= 8 {
            let mut taps_3d = Vec::with_capacity(taps.len());
            for tap in taps {
                let bf16 = if tap.dtype()? == DType::BFloat16 {
                    tap.clone()
                } else {
                    tap.astype(DType::BFloat16)?
                };
                taps_3d.push(bf16.expand_dims(0)?);
            }
            let fused = draft.fuse_context(&taps_3d)?;
            self.cache
                .append_full_width_prefix(draft, &fused, base, token_ids)?;
            let fused_kept = if keep < 8 {
                fused.slice_axis(1, 0, keep as i64)?
            } else {
                fused
            };
            return Ok(Some(fused_kept));
        }

        let mut taps_3d = Vec::with_capacity(taps.len());
        for tap in taps {
            let sliced = if keep < total_rows {
                tap.slice_axis(0, 0, keep as i64)?
            } else {
                tap.clone()
            };
            let bf16 = if sliced.dtype()? == DType::BFloat16 {
                sliced
            } else {
                sliced.astype(DType::BFloat16)?
            };
            taps_3d.push(bf16.expand_dims(0)?);
        }
        let fused = draft.fuse_context(&taps_3d)?;
        self.cache.append(draft, &fused, base, token_ids)?;
        Ok(Some(fused))
    }

    /// Batched counterpart to [`Self::append_taps`] across `N` lanes: stacks all
    /// healthy `keep > 0` lanes' kept tap rows along axis 0 (`[1, sum(keep_i), H]`),
    /// runs ONE `draft.fuse_context` and ONE batched K/V projection via
    /// [`DFlash2ContextCache::append_batched`], and returns one `Result<Option<MxArray>>`
    /// per lane (`Ok(None)` when `keep_i == 0`, `Ok(Some(fused_i))` for PR #408 live
    /// `DraftContextWindow` capture) so a fault on one lane NEVER aborts healthy peers.
    pub(crate) fn append_taps_batched(
        draft: &DFlash2Model,
        items: &mut [(&mut Inkling2Dflash2Context, &[MxArray], &[u32])],
    ) -> Vec<Result<Option<MxArray>>> {
        let mut results: Vec<Option<Result<Option<MxArray>>>> =
            (0..items.len()).map(|_| None).collect();
        let expected = draft.config.target_layers.len();
        let hidden_size = draft.config.hidden_size;

        // 1. Pre-validate every lane independently so a malformed lane gets its own `Err`
        //    without aborting healthy peers in the wave.
        let mut valid_indices = Vec::with_capacity(items.len());
        for (idx, (ctx, taps, token_ids)) in items.iter().enumerate() {
            if token_ids.is_empty() {
                results[idx] = Some(Ok(None));
                continue;
            }
            let keep = token_ids.len();
            let check = (|| -> Result<()> {
                if expected == 0 {
                    return Err(Error::from_reason(
                        "Inkling2 DFlash2 append_taps_batched: empty target_layers",
                    ));
                }
                if taps.len() != expected {
                    return Err(Error::from_reason(format!(
                        "Inkling2 DFlash2 append_taps_batched expected {expected} taps, got {}",
                        taps.len()
                    )));
                }
                for tap in *taps {
                    if tap.ndim()? != 2 {
                        return Err(Error::from_reason(format!(
                            "Inkling2 DFlash2 append_taps_batched expected 2D tap, got {}D",
                            tap.ndim()?
                        )));
                    }
                    let total_rows = tap.shape_at(0)? as usize;
                    if keep > total_rows {
                        return Err(Error::from_reason(format!(
                            "Inkling2 DFlash2 append_taps_batched keep={keep} exceeds tap rows {total_rows}"
                        )));
                    }
                    let cols = tap.shape_at(1)? as usize;
                    if cols != hidden_size {
                        return Err(Error::from_reason(format!(
                            "Inkling2 DFlash2 append_taps_batched tap hidden {cols} != {hidden_size}"
                        )));
                    }
                }
                draft.validate_context_for_append(&ctx.cache, keep)?;
                Ok(())
            })();
            match check {
                Ok(()) => valid_indices.push(idx),
                Err(err) => results[idx] = Some(Err(err)),
            }
        }

        let finish = |results: Vec<Option<Result<Option<MxArray>>>>| {
            results
                .into_iter()
                .map(|slot| {
                    slot.unwrap_or_else(|| {
                        Err(Error::from_reason(
                            "Inkling2 DFlash2 append_taps_batched missing slot",
                        ))
                    })
                })
                .collect::<Vec<_>>()
        };

        if valid_indices.is_empty() {
            return finish(results);
        }
        if valid_indices.len() == 1 {
            let idx = valid_indices[0];
            let (ctx, taps, token_ids) = &mut items[idx];
            results[idx] = Some(ctx.append_taps(draft, taps, token_ids));
            return finish(results);
        }

        // 2. Slice and BF16-cast each healthy lane's taps individually (isolating any per-lane
        //    slice/cast error before concatenating).
        let mut prepared_lanes: Vec<(usize, Vec<MxArray>)> =
            Vec::with_capacity(valid_indices.len());
        for &idx in &valid_indices {
            let (_, taps, token_ids) = &items[idx];
            let keep = token_ids.len() as i64;
            let lane_slices = (|| -> Result<Vec<MxArray>> {
                let mut slices = Vec::with_capacity(expected);
                for tap in *taps {
                    let rows = tap.shape_at(0)?;
                    let sliced = if keep < rows {
                        tap.slice_axis(0, 0, keep)?
                    } else {
                        tap.clone()
                    };
                    let bf16 = if sliced.dtype()? == DType::BFloat16 {
                        sliced
                    } else {
                        sliced.astype(DType::BFloat16)?
                    };
                    slices.push(bf16);
                }
                Ok(slices)
            })();
            match lane_slices {
                Ok(slices) => prepared_lanes.push((idx, slices)),
                Err(err) => results[idx] = Some(Err(err)),
            }
        }

        if prepared_lanes.is_empty() {
            return finish(results);
        }
        if prepared_lanes.len() == 1 {
            let idx = prepared_lanes[0].0;
            let (ctx, taps, token_ids) = &mut items[idx];
            results[idx] = Some(ctx.append_taps(draft, taps, token_ids));
            return finish(results);
        }

        // 3. Stack & run ONE `draft.fuse_context`; on shared error, fall back per-lane so
        //    healthy peers still commit their draft caches.
        let fused_all_res = (|| -> Result<MxArray> {
            let mut stacked_taps = Vec::with_capacity(expected);
            for tap_idx in 0..expected {
                let refs = prepared_lanes
                    .iter()
                    .map(|(_, slices)| &slices[tap_idx])
                    .collect::<Vec<_>>();
                stacked_taps.push(MxArray::concatenate_many(refs, Some(0))?.expand_dims(0)?);
            }
            draft.fuse_context(&stacked_taps)
        })();
        let fused_all = match fused_all_res {
            Ok(fused) => fused,
            Err(_) => {
                for (idx, _) in prepared_lanes {
                    let (ctx, taps, token_ids) = &mut items[idx];
                    results[idx] = Some(ctx.append_taps(draft, taps, token_ids));
                }
                return finish(results);
            }
        };

        // 4. Run ONE batched KV projection + ring append. Snapshot every lane's `logical_len`
        //    and per-layer ring `end` cursors beforehand so if `append_batched` fails:
        //    - earlier lanes that already advanced `logical_len` stay `Ok(Some(fused_i))`,
        //    - any partially mutated lane (ring `end` moved without advancing `logical_len`) or
        //      the first unadvanced lane after a succeeded lane gets `Err(batch_err)` and is NEVER
        //      retried (caller discards its draft owner),
        //    - untouched lanes (either after the failing lane, or when `project_context_batched`
        //      failed before any ring write) commit independently via `cache.append`.
        let mut active_mask = vec![false; items.len()];
        let mut before_lens = Vec::with_capacity(prepared_lanes.len());
        let mut before_ends = Vec::with_capacity(prepared_lanes.len());
        let mut active_keeps = Vec::with_capacity(prepared_lanes.len());
        for &(idx, _) in &prepared_lanes {
            active_mask[idx] = true;
            before_lens.push(items[idx].0.cache.logical_len());
            before_ends.push(DFlash2Model::context_window_ends(&items[idx].0.cache));
            active_keeps.push(items[idx].2.len());
        }

        let batch_res = {
            let mut cache_lanes: Vec<(&mut DFlash2ContextCache, &[u32])> = items
                .iter_mut()
                .enumerate()
                .filter(|(idx, _)| active_mask[*idx])
                .map(|(_, (ctx, _, token_ids))| (&mut ctx.cache, *token_ids))
                .collect();
            DFlash2ContextCache::append_batched(draft, &fused_all, &mut cache_lanes)
        };

        match batch_res {
            Ok(()) => {
                let mut off = 0i64;
                for (&(idx, _), &keep) in prepared_lanes.iter().zip(&active_keeps) {
                    let keep_i64 = keep as i64;
                    results[idx] = Some(fused_all.slice_axis(1, off, off + keep_i64).map(Some));
                    off += keep_i64;
                }
            }
            Err(batch_err) => {
                let mut off = 0i64;
                let mut any_advanced = false;
                let mut seen_failed = false;
                for (((&(idx, _), &before_len), before_end), &keep) in prepared_lanes
                    .iter()
                    .zip(&before_lens)
                    .zip(&before_ends)
                    .zip(&active_keeps)
                {
                    let keep_i64 = keep as i64;
                    let (ctx, _, token_ids) = &mut items[idx];
                    if ctx.cache.logical_len() == before_len + keep as i32 {
                        any_advanced = true;
                        results[idx] = Some(fused_all.slice_axis(1, off, off + keep_i64).map(Some));
                    } else if DFlash2Model::context_window_ends(&ctx.cache) != *before_end
                        || (any_advanced && !seen_failed)
                    {
                        seen_failed = true;
                        results[idx] = Some(Err(Error::from_reason(batch_err.reason.clone())));
                    } else {
                        let lane_res = (|| -> Result<Option<MxArray>> {
                            let fused_i = fused_all.slice_axis(1, off, off + keep_i64)?;
                            ctx.cache.append(draft, &fused_i, before_len, token_ids)?;
                            Ok(Some(fused_i))
                        })();
                        if lane_res.is_err() {
                            seen_failed = true;
                        }
                        results[idx] = Some(lane_res);
                    }
                    off += keep_i64;
                }
            }
        }

        finish(results)
    }

    /// Tree-verify counterpart to [`Self::append_taps`]: reuses contiguous prefix slicing
    /// when `accepted_rows == 0..keep`, otherwise gathers `accepted_rows` via `take(axis=0)`
    /// before fusing and appending to the draft KV ring.
    pub(crate) fn append_taps_path(
        &mut self,
        draft: &DFlash2Model,
        taps: &[MxArray],
        accepted_rows: &[usize],
        accepted_tokens: &[u32],
    ) -> Result<Option<MxArray>> {
        if accepted_rows.len() != accepted_tokens.len() {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 append_taps_path row count {} != token count {}",
                accepted_rows.len(),
                accepted_tokens.len()
            )));
        }
        if accepted_rows.is_empty() {
            return Ok(None);
        }
        let is_contiguous_prefix = accepted_rows
            .iter()
            .enumerate()
            .all(|(index, &row)| row == index);
        if is_contiguous_prefix {
            return self.append_taps(draft, taps, accepted_tokens);
        }
        let expected = draft.config.target_layers.len();
        if taps.len() != expected {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 append_taps_path expected {expected} taps, got {}",
                taps.len()
            )));
        }
        let total_rows = taps[0].shape_at(0)? as usize;
        let indices: Vec<i32> = accepted_rows
            .iter()
            .map(|&row| {
                if row < total_rows {
                    Ok(row as i32)
                } else {
                    Err(Error::from_reason(format!(
                        "Inkling2 DFlash2 append_taps_path node {row} >= tap rows {total_rows}"
                    )))
                }
            })
            .collect::<Result<_>>()?;
        let idx = MxArray::from_int32(&indices, &[indices.len() as i64])?;
        let mut taps_3d = Vec::with_capacity(taps.len());
        for tap in taps {
            let gathered = tap.take(&idx, 0)?;
            let bf16 = if gathered.dtype()? == DType::BFloat16 {
                gathered
            } else {
                gathered.astype(DType::BFloat16)?
            };
            taps_3d.push(bf16.expand_dims(0)?);
        }
        let base = self.cache.logical_len();
        let fused = draft.fuse_context(&taps_3d)?;
        self.cache.append(draft, &fused, base, accepted_tokens)?;
        Ok(Some(fused))
    }

    /// Draft a greedy chain of up to `max_len` tokens, keeping the draft block at full
    /// `draft.config.block_size` (`7`, 8 total rows) so the compiled Metal draft body
    /// never retraces, then truncating if `max_len < block_size`.
    pub(crate) fn propose_greedy_chain(
        &mut self,
        draft: &DFlash2Model,
        view: &mut Inkling2TargetView<'_>,
        anchor: u32,
        max_len: usize,
        device_path: bool,
        with_confidence: bool,
    ) -> Result<DsparkProposal> {
        if max_len == 0 {
            return Ok(empty_proposal());
        }
        let full_len = draft.config.block_size;
        let cap = max_len.min(full_len);
        let mut conf_arr: Option<MxArray> = None;
        let mut dummy_rng = rand::rng();
        let (path, _) = draft.propose_with(
            view,
            &mut self.cache,
            anchor,
            full_len,
            0.0,
            device_path,
            with_confidence.then_some(&mut conf_arr),
            &mut dummy_rng,
        )?;
        let keep_probabilities = match conf_arr {
            Some(arr) => {
                let mut probs = arr.to_float32()?.to_vec();
                probs.truncate(cap);
                Some(probs)
            }
            None => None,
        };
        let mut proposal = match path {
            SelectorPath::Device(ids) => DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: Some(ids),
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities,
            },
            SelectorPath::Host(ids) => DsparkProposal {
                draft_ids: ids,
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities,
            },
            SelectorPath::Sampled { .. } => {
                return Err(Error::from_reason(
                    "Inkling2 DFlash2 greedy proposal unexpectedly returned a sampled path",
                ));
            }
        };
        truncate_proposal(&mut proposal, cap)?;
        Ok(proposal)
    }

    /// Draft a temperature-sampled chain of up to `max_len` tokens with sparse
    /// top-16 proposal distributions (`vocab_size == unpadded_vocab_size == 200058`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propose_sampled_chain(
        &mut self,
        draft: &DFlash2Model,
        view: &mut Inkling2TargetView<'_>,
        anchor: u32,
        max_len: usize,
        temperature: f64,
        device_path: bool,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        if max_len == 0 {
            return Ok(empty_proposal());
        }
        let full_len = draft.config.block_size;
        let cap = max_len.min(full_len);
        let (path, draft_sparse_dists) = draft.propose_with(
            view,
            &mut self.cache,
            anchor,
            full_len,
            temperature,
            device_path,
            None,
            rng,
        )?;
        let mut proposal = match path {
            SelectorPath::Sampled { ids, rows } => DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: Some(ids),
                device_draft_sparse: Some(rows),
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            },
            SelectorPath::Host(ids) => DsparkProposal {
                draft_ids: ids,
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists,
                keep_probabilities: None,
            },
            SelectorPath::Device(_) => {
                return Err(Error::from_reason(
                    "Inkling2 DFlash2 sampled proposal unexpectedly returned a greedy device path",
                ));
            }
        };
        truncate_proposal(&mut proposal, cap)?;
        Ok(proposal)
    }

    /// Unified greedy/sampled proposal entry point honoring `ChatParams` penalties
    /// and sampling temperature.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propose_block(
        &mut self,
        draft: &DFlash2Model,
        view: &mut Inkling2TargetView<'_>,
        anchor: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
        with_confidence: bool,
    ) -> Result<DsparkProposal> {
        if max_len == 0 {
            return Ok(empty_proposal());
        }
        let temperature = params
            .sampling_config
            .unwrap_or_default()
            .temperature
            .unwrap_or(1.0);
        // Offer the device path only when acceptance consumes it without a
        // mid-cycle host read (penalized turns need host ids for their history).
        let device_path = params.repetition_penalty == 1.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0;
        if is_greedy_temperature(temperature) {
            self.propose_greedy_chain(draft, view, anchor, max_len, device_path, with_confidence)
        } else {
            self.propose_sampled_chain(draft, view, anchor, max_len, temperature, device_path, rng)
        }
    }

    /// Batched multi-lane DFlash2 proposal over `2..=4` active lanes: runs ONE
    /// `[N, block_size + 1]` embed, ONE batched 5-layer draft pass, ONE
    /// `[N, block_size, valid_vocab]` `lm_head` matmul, and per-lane candidate
    /// selection + truncation to each lane's `max_drafts.min(block_size)`,
    /// returning one `Result<DsparkProposal>` per lane so a single-lane selector
    /// or conversion failure never aborts healthy peers.
    ///
    /// Keeps `<= 1` active lane on [`Self::propose_block`] so single-lane turns
    /// stay on their compiled solo Metal graph.
    pub(crate) fn propose_blocks_batched(
        draft: &DFlash2Model,
        view: &mut Inkling2TargetView<'_>,
        lanes: &mut [BatchedDflash2Lane<'_>],
        rngs: &mut [&mut dyn rand::Rng],
    ) -> Vec<Result<DsparkProposal>> {
        if lanes.len() != rngs.len() {
            let reason = format!(
                "Inkling2Dflash2Context::propose_blocks_batched: lanes ({}) != rngs ({})",
                lanes.len(),
                rngs.len()
            );
            return (0..lanes.len())
                .map(|_| Err(Error::from_reason(reason.clone())))
                .collect();
        }
        let active_count = lanes.iter().filter(|lane| lane.max_drafts > 0).count();
        if active_count == 0 {
            return (0..lanes.len()).map(|_| Ok(empty_proposal())).collect();
        }
        if active_count == 1 {
            let mut out = Vec::with_capacity(lanes.len());
            for (lane, rng) in lanes.iter_mut().zip(rngs.iter_mut()) {
                if lane.max_drafts == 0 {
                    out.push(Ok(empty_proposal()));
                } else {
                    out.push(lane.ctx.propose_block(
                        draft,
                        view,
                        lane.anchor_id,
                        lane.max_drafts,
                        lane.params,
                        *rng,
                        lane.confidence,
                    ));
                }
            }
            return out;
        }

        let full_len = draft.config.block_size;
        let mut active_indices = Vec::with_capacity(active_count);
        let mut active_caps = Vec::with_capacity(active_count);
        let mut active_greedy = Vec::with_capacity(active_count);
        let mut dflash_lanes = Vec::with_capacity(active_count);
        let mut active_rngs: Vec<&mut dyn rand::Rng> = Vec::with_capacity(active_count);
        for (idx, (lane, rng)) in lanes.iter_mut().zip(rngs.iter_mut()).enumerate() {
            if lane.max_drafts == 0 {
                continue;
            }
            let temperature = lane
                .params
                .sampling_config
                .unwrap_or_default()
                .temperature
                .unwrap_or(1.0);
            let greedy = is_greedy_temperature(temperature);
            let device_path = lane.params.repetition_penalty == 1.0
                && lane.params.presence_penalty == 0.0
                && lane.params.frequency_penalty == 0.0;
            active_indices.push(idx);
            active_caps.push(lane.max_drafts.min(full_len));
            active_greedy.push(greedy);
            dflash_lanes.push(DFlash2BatchProposalLane {
                context: &mut lane.ctx.cache,
                anchor: lane.anchor_id,
                temperature,
                device_path,
                want_confidence: greedy && lane.confidence,
            });
            active_rngs.push(&mut **rng);
        }

        let raw_outputs =
            draft.propose_with_batched(view, &mut dflash_lanes, &mut active_rngs, full_len);

        // Best-effort single-sync eval for all healthy confidence tensors; if the batch eval
        // fails on one broken lane, each lane's individual `to_float32()` below isolates it.
        let conf_refs: Vec<&MxArray> = raw_outputs
            .iter()
            .filter_map(|res| res.as_ref().ok().and_then(|(_, _, conf)| conf.as_ref()))
            .collect();
        if !conf_refs.is_empty() {
            let _ = MxArray::eval_arrays(&conf_refs);
        }

        let mut out: Vec<Result<DsparkProposal>> =
            (0..lanes.len()).map(|_| Ok(empty_proposal())).collect();
        for (((orig_idx, cap), greedy), raw_res) in active_indices
            .into_iter()
            .zip(active_caps)
            .zip(active_greedy)
            .zip(raw_outputs)
        {
            out[orig_idx] = raw_res.and_then(|(path, draft_sparse_dists, conf_arr)| {
                let keep_probabilities = match conf_arr {
                    Some(arr) => {
                        let mut probs = arr.to_float32()?.to_vec();
                        probs.truncate(cap);
                        Some(probs)
                    }
                    None => None,
                };
                let mut proposal = if greedy {
                    match path {
                        SelectorPath::Device(ids) => DsparkProposal {
                            draft_ids: Vec::new(),
                            device_draft_ids: Some(ids),
                            device_draft_sparse: None,
                            draft_dists: Vec::new(),
                            draft_sparse_dists: Vec::new(),
                            keep_probabilities,
                        },
                        SelectorPath::Host(ids) => DsparkProposal {
                            draft_ids: ids,
                            device_draft_ids: None,
                            device_draft_sparse: None,
                            draft_dists: Vec::new(),
                            draft_sparse_dists: Vec::new(),
                            keep_probabilities,
                        },
                        SelectorPath::Sampled { .. } => {
                            return Err(Error::from_reason(
                                "Inkling2 DFlash2 greedy proposal unexpectedly returned a sampled path",
                            ));
                        }
                    }
                } else {
                    match path {
                        SelectorPath::Sampled { ids, rows } => DsparkProposal {
                            draft_ids: Vec::new(),
                            device_draft_ids: Some(ids),
                            device_draft_sparse: Some(rows),
                            draft_dists: Vec::new(),
                            draft_sparse_dists: Vec::new(),
                            keep_probabilities: None,
                        },
                        SelectorPath::Host(ids) => DsparkProposal {
                            draft_ids: ids,
                            device_draft_ids: None,
                            device_draft_sparse: None,
                            draft_dists: Vec::new(),
                            draft_sparse_dists,
                            keep_probabilities: None,
                        },
                        SelectorPath::Device(_) => {
                            return Err(Error::from_reason(
                                "Inkling2 DFlash2 sampled proposal unexpectedly returned a greedy device path",
                            ));
                        }
                    }
                };
                truncate_proposal(&mut proposal, cap)?;
                Ok(proposal)
            });
        }
        out
    }

    /// Draft unevaluated `(candidates [1, L, K] Int32, scores [L, K, K] Float32)`
    /// transition lattice arrays for verify-tree construction.
    pub(crate) fn propose_tree_lattice(
        &mut self,
        draft: &DFlash2Model,
        view: &mut Inkling2TargetView<'_>,
        anchor: u32,
        max_len: usize,
    ) -> Result<(MxArray, MxArray)> {
        draft.propose_tree_lattice_arrays_embedded(view, &mut self.cache, anchor, max_len)
    }
}

/// Unified per-sequence draft context owned by whole-turn & scheduled Inkling-2-Mini turns.
pub(crate) enum Inkling2DraftContext {
    Dspark(Inkling2DsparkContext),
    Dflash2(Box<Inkling2Dflash2Context>),
}

impl Inkling2DraftContext {
    pub(crate) fn committed_len(&self) -> i32 {
        match self {
            Self::Dspark(ctx) => ctx.committed_len,
            Self::Dflash2(ctx) => ctx.committed_len(),
        }
    }

    pub(crate) fn buffer_bytes(&self) -> u64 {
        match self {
            Self::Dspark(ctx) => ctx.buffer_bytes(),
            Self::Dflash2(ctx) => ctx.buffer_bytes(),
        }
    }

    pub(crate) fn mlx_buffer_bytes(&self) -> u64 {
        match self {
            Self::Dspark(ctx) => ctx.buffer_bytes(),
            Self::Dflash2(ctx) => ctx.mlx_buffer_bytes(),
        }
    }

    pub(crate) fn eval_state(&self) -> Result<()> {
        match self {
            Self::Dspark(ctx) => ctx.eval_state(),
            Self::Dflash2(ctx) => ctx.eval_state(),
        }
    }

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match self {
            Self::Dspark(ctx) => ctx.collect_arrays(out),
            Self::Dflash2(ctx) => ctx.collect_arrays(out),
        }
    }
}

pub(super) fn empty_proposal() -> DsparkProposal {
    DsparkProposal {
        draft_ids: Vec::new(),
        device_draft_ids: None,
        device_draft_sparse: None,
        draft_dists: Vec::new(),
        draft_sparse_dists: Vec::new(),
        keep_probabilities: None,
    }
}

fn truncate_proposal(proposal: &mut DsparkProposal, cap: usize) -> Result<()> {
    if let Some(ref ids) = proposal.device_draft_ids {
        let len = ids.shape_at(0)? as usize;
        if cap < len {
            proposal.device_draft_ids = Some(ids.slice_axis(0, 0, cap as i64)?);
        }
    }
    if let Some(ref mut sparse) = proposal.device_draft_sparse {
        let len = sparse.ids.shape_at(0)? as usize;
        if cap < len {
            sparse.ids = sparse.ids.slice_axis(0, 0, cap as i64)?;
            sparse.probs = sparse.probs.slice_axis(0, 0, cap as i64)?;
        }
    }
    if proposal.draft_ids.len() > cap {
        proposal.draft_ids.truncate(cap);
    }
    if proposal.draft_sparse_dists.len() > cap {
        proposal.draft_sparse_dists.truncate(cap);
    }
    if let Some(ref mut probs) = proposal.keep_probabilities
        && probs.len() > cap
    {
        probs.truncate(cap);
    }
    Ok(())
}

/// True when `path` is a DFlash2 companion directory holding a valid `config.json`
/// declaring `"DFlash2DraftModel"` in `architectures` plus `model.safetensors`.
pub(crate) fn is_dflash2_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("model.safetensors").is_file() {
        return false;
    }
    let config_path = path.join("config.json");
    let Ok(raw) = fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    parsed
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|archs| {
            archs
                .iter()
                .any(|entry| entry.as_str() == Some("DFlash2DraftModel"))
        })
}

/// Auto-discover compatible Inkling-2-Mini DFlash2 companion directories in priority order:
/// 1. Colocated `<modelDir>/draft/`
/// 2. Sibling `<parent(modelDir)>/inkling-2-mini-dflash2/` (case-insensitive)
/// 3. Any other sibling DFlash2 directory whose config matches `target_shape`
pub(crate) fn discover_inkling2_dflash2_candidates(
    source_dir: &Path,
    asset_dir: &Path,
    target_shape: DFlash2TargetShape,
) -> Vec<PathBuf> {
    let mut base_dirs = Vec::new();
    let mut seen_base = HashSet::new();
    for dir in [source_dir, asset_dir] {
        let owned = dir.to_path_buf();
        if seen_base.insert(owned.clone()) {
            base_dirs.push(owned);
        }
        if let Ok(canon) = fs::canonicalize(dir)
            && seen_base.insert(canon.clone())
        {
            base_dirs.push(canon);
        }
    }

    let mut out = Vec::new();
    let mut seen_out = HashSet::new();
    let mut push_if_valid = |cand: PathBuf| {
        if !is_dflash2_dir(&cand) {
            return;
        }
        let Ok(cfg) = read_dflash2_config(&cand) else {
            return;
        };
        if cfg.validate_target(target_shape).is_err() {
            return;
        }
        let key = fs::canonicalize(&cand).unwrap_or_else(|_| cand.clone());
        if seen_out.insert(key) {
            out.push(cand);
        }
    };

    // 1. Explicit colocated `<modelDir>/draft/`
    for dir in &base_dirs {
        push_if_valid(dir.join("draft"));
    }

    // 2. Named companion `inkling-2-mini-dflash2` inside `base_dirs` (standalone GGUF
    // residing directly in a shared models store) and `parent(base_dirs)` (repo subdir),
    // followed by any other shape-matching sibling DFlash2 directory.
    let mut search_roots = Vec::new();
    let mut seen_roots = HashSet::new();
    for dir in &base_dirs {
        if seen_roots.insert(dir.clone()) {
            search_roots.push(dir.clone());
        }
        if let Some(parent) = dir.parent() {
            let owned = parent.to_path_buf();
            if seen_roots.insert(owned.clone()) {
                search_roots.push(owned);
            }
        }
    }
    for root in &search_roots {
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        let mut named = Vec::new();
        let mut others = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let lower = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if lower == "inkling-2-mini-dflash2" {
                named.push(path);
            } else if lower != "draft" {
                others.push(path);
            }
        }
        named.sort();
        others.sort();
        for cand in named.into_iter().chain(others) {
            push_if_valid(cand);
        }
    }

    out
}

impl Inkling2Inner {
    /// Construct an [`Inkling2TargetView`] borrowing this backbone's embedding,
    /// `embed_norm`, and shared `lm_head`.
    pub(crate) fn target_view(&self) -> Inkling2TargetView<'_> {
        Inkling2TargetView {
            embed_tokens: &self.embed_tokens,
            embed_norm: &self.embed_norm,
            lm_head: &self.lm_head,
            lm_head_bf16_unpadded: self.lm_head_bf16_unpadded.as_ref(),
            rms_norm_eps: self.config.rms_norm_eps as f32,
            unpadded_vocab_size: self.config.effective_unpadded_vocab_size(),
        }
    }

    /// Chunked whole-turn paged prefill that seeds a DFlash2 context cache with
    /// target hidden taps (`[5, 23, 41, 48]`), splits at rolling checkpoints,
    /// extra capture boundaries, and prompt junction via
    /// `prefill_sub_boundaries_with_extra`, captures live 4-bit
    /// [`DraftContextWindow`] snapshots at every block boundary, and publishes
    /// rolling/junction checkpoints into [`Inkling2StateStore`].
    pub(crate) fn paged_prefill_dflash2_with_taps(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        prefix: &super::paged_forward::Inkling2PrefixState,
        ctx: &mut Inkling2Dflash2Context,
    ) -> Result<MxArray> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Inkling2 DFlash2 prefill: empty tokens"));
        }
        let (chunk_limit, bs) = {
            let rt = self.paged.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 prefill: paged runtime missing")
            })?;
            let existing = rt.coordinator.request_token_count_all(seq_id).unwrap_or(0);
            let added = u32::try_from(tokens.len())
                .map_err(|_| Error::from_reason("Inkling2 DFlash2 prompt length overflows u32"))?;
            let final_total = existing.checked_add(added).ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 prompt total length overflows u32")
            })?;
            if !rt.coordinator.can_fit_logical_tokens(final_total) {
                let capacity = rt.coordinator.max_capacity_tokens();
                return Err(Error::from_reason(format!(
                    "context_length_exceeded: requested {final_total} tokens, \
                     paged cache capacity is {capacity} tokens"
                )));
            }
            (rt.prefill_chunk_size, rt.coordinator.block_size().max(1))
        };
        let (tap_visits, sliding_window) = {
            let draft = self.dflash2.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 prefill: DFlash2 draft model missing")
            })?;
            (
                draft.config.target_layers.clone(),
                draft.config.sliding_window,
            )
        };

        let abs_base = prefix.effective_cached_prefix_len;
        let full_tokens = &prefix.full_tokens;
        let full_prompt_len = full_tokens.len();
        let cache_salt = prefix.cache_salt;
        let reuse_cache = prefix.reuse_cache;

        // Collect all capture/end targets where a complete `sliding_window`-long
        // draft window or live draft KV ring is required: planned 4096 rolling
        // checkpoints, installed extra capture boundaries, and prompt junction
        // (only when `reuse_cache` permits shared StateStore publication) plus
        // the final prompt end.
        let extra_boundaries = if reuse_cache {
            self.paged
                .as_ref()
                .and_then(|rt| rt.seq_extra_capture_boundaries.get(&seq_id))
                .cloned()
        } else {
            None
        };
        let mut targets: Vec<usize> = Vec::new();
        if reuse_cache
            && let (Ok(prompt_u32), Ok(base_u32)) =
                (u32::try_from(full_prompt_len), u32::try_from(abs_base))
        {
            for cp in super::state_disk::planned_rolling_checkpoints(prompt_u32, base_u32, bs) {
                let cp_us = cp as usize;
                if cp_us > abs_base && cp_us < full_prompt_len {
                    targets.push(cp_us);
                }
            }
            let junction = ((prompt_u32.saturating_sub(1) / bs) * bs) as usize;
            if junction > abs_base && junction < full_prompt_len {
                targets.push(junction);
            }
            if let Some(ref extra) = extra_boundaries {
                for &b in extra {
                    let b_us = b as usize;
                    if b_us > abs_base && b_us < full_prompt_len {
                        targets.push(b_us);
                    }
                }
            }
        }
        targets.push(full_prompt_len);
        targets.sort_unstable();
        targets.dedup();

        let total_len = tokens.len();
        let mut offset = 0usize;
        let mut final_logits = None;

        while offset < total_len {
            let chunk_end = (offset + chunk_limit).min(total_len);
            let abs_chunk_start = abs_base + offset;
            let abs_chunk_end = abs_base + chunk_end;
            let subs = if reuse_cache {
                super::paged_forward::prefill_sub_boundaries_with_extra(
                    abs_chunk_start,
                    abs_chunk_end,
                    full_prompt_len,
                    bs,
                    extra_boundaries.as_ref(),
                )
            } else {
                vec![abs_chunk_end]
            };
            let mut abs_sub_start = abs_chunk_start;

            for abs_sub_end in subs {
                if self
                    .turn_cancel
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    return Err(Error::from_reason("prefill cancelled"));
                }
                let sub_start = abs_sub_start - abs_base;
                let sub_end = abs_sub_end - abs_base;
                let sub_chunk = &tokens[sub_start..sub_end];
                let is_final_chunk = sub_end == total_len;
                // Tap whenever `[abs_sub_start..abs_sub_end)` overlaps the
                // `[target - sliding_window .. target)` window of ANY planned rolling
                // checkpoint, prompt junction, or final prompt end.
                let needs_taps = targets.iter().any(|&target| {
                    abs_sub_end > target.saturating_sub(sliding_window) && abs_sub_start < target
                });

                let first_pos = {
                    let rt = self.paged.as_mut().ok_or_else(|| {
                        Error::from_reason("Inkling2 DFlash2 prefill: paged runtime missing")
                    })?;
                    rt.ensure_conv_state(seq_id, &self.config)?;
                    rt.coordinator
                        .request_token_count_all(seq_id)
                        .map_err(Error::from_reason)?
                };
                self.record_tokens_all_exclusive(seq_id, sub_chunk)
                    .map_err(Error::from_reason)?;

                let tap_arg: &[usize] = if needs_taps { &tap_visits } else { &[] };
                let (logits_opt, taps) = self.forward_paged_single_chunk_with_taps(
                    seq_id,
                    sub_chunk,
                    None,
                    first_pos,
                    is_final_chunk,
                    tap_arg,
                )?;

                let _ = self.settle_paged_step(seq_id)?;
                if needs_taps {
                    let fused = {
                        let draft = self.dflash2.as_ref().ok_or_else(|| {
                            Error::from_reason(
                                "Inkling2 DFlash2 prefill: DFlash2 draft model disappeared",
                            )
                        })?;
                        ctx.append_taps(draft, &taps, sub_chunk)?
                    };
                    if reuse_cache && let Some(fused) = fused {
                        self.record_fused_to_draft_window_best_effort(seq_id, first_pos, &fused);
                    }
                } else {
                    ctx.cache.record_only(sub_chunk)?;
                }
                if let Ok(pos_after) = u32::try_from(first_pos as usize + sub_chunk.len()) {
                    if let Some(rt) = self.paged.as_mut() {
                        rt.record_block_boundary_conv(seq_id, pos_after);
                    }
                    if reuse_cache && pos_after as usize <= full_tokens.len() {
                        self.maybe_capture_rolling_checkpoint_best_effort(
                            seq_id,
                            full_tokens,
                            pos_after as usize,
                            cache_salt,
                            "DFlash2 prefill",
                        );
                    }
                }

                if is_final_chunk {
                    final_logits = logits_opt;
                } else {
                    crate::array::clear_cache();
                }
                abs_sub_start = abs_sub_end;
            }
            offset = chunk_end;
        }

        final_logits
            .ok_or_else(|| Error::from_reason("Inkling2 DFlash2 prefill produced no final logits"))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::dflash2::tiny_dflash2_model_with_block;
    use crate::nn::Linear;
    use crate::sampling::SamplingConfig;

    fn det_bf16(shape: &[i64], seed: u64, scale: f32) -> MxArray {
        let len = shape.iter().product::<i64>() as usize;
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut vals = Vec::with_capacity(len);
        for _ in 0..len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32) - 1.0;
            vals.push(u * scale);
        }
        MxArray::from_float32(&vals, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn assert_close(a: &MxArray, b: &MxArray, tol: f32, label: &str) {
        assert_eq!(
            a.shape().unwrap().as_ref(),
            b.shape().unwrap().as_ref(),
            "{label}: shape"
        );
        let av = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bv = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        let max_diff = av
            .iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= tol,
            "{label}: max_diff={max_diff} exceeds tol={tol}"
        );
    }

    #[test]
    fn inkling2_target_view_batch_embed_and_logits_match_solo() {
        let hidden = 64i64;
        let padded_vocab = 64i64;
        let unpadded_vocab = 50usize;
        let embed_w = det_bf16(&[padded_vocab, hidden], 11, 0.4);
        let embed_tokens = Embedding::from_weight(&embed_w).unwrap();
        let embed_norm = det_bf16(&[hidden], 12, 0.5)
            .add_scalar(1.0)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let lm_head_w = det_bf16(&[padded_vocab, hidden], 13, 0.3);
        let lm_head = LinearProj::Standard(Linear::from_weights(&lm_head_w, None).unwrap());
        let lm_head_unpadded = lm_head_w.slice_axis(0, 0, unpadded_vocab as i64).unwrap();
        let mut view = Inkling2TargetView {
            embed_tokens: &embed_tokens,
            embed_norm: &embed_norm,
            lm_head: &lm_head,
            lm_head_bf16_unpadded: Some(&lm_head_unpadded),
            rms_norm_eps: 1e-6,
            unpadded_vocab_size: unpadded_vocab,
        };

        for batch in [1usize, 2, 3, 4] {
            let rows = 7usize;
            let ids: Vec<i32> = (0..batch * rows)
                .map(|i| ((i * 7 + 3) % unpadded_vocab) as i32)
                .collect();
            let block = MxArray::from_int32(&ids, &[batch as i64, rows as i64]).unwrap();
            let emb_b = view.embed(&block, &ids).unwrap();
            assert_eq!(
                emb_b.shape().unwrap().as_ref(),
                &[batch as i64, rows as i64, hidden]
            );

            let logits_b = view.logits(&emb_b).unwrap();
            assert_eq!(
                logits_b.shape().unwrap().as_ref(),
                &[batch as i64, rows as i64, unpadded_vocab as i64]
            );

            for b in 0..batch {
                let ids_i = &ids[b * rows..(b + 1) * rows];
                let block_i = MxArray::from_int32(ids_i, &[1, rows as i64]).unwrap();
                let emb_solo = view.embed(&block_i, ids_i).unwrap();
                let logits_solo = view.logits(&emb_solo).unwrap();
                assert_close(
                    &emb_b.slice_axis(0, b as i64, b as i64 + 1).unwrap(),
                    &emb_solo,
                    1e-3,
                    &format!("embed B={batch} slice={b}"),
                );
                assert_close(
                    &logits_b.slice_axis(0, b as i64, b as i64 + 1).unwrap(),
                    &logits_solo,
                    2e-2,
                    &format!("logits B={batch} slice={b}"),
                );
            }
        }
    }

    #[test]
    fn inkling2_dflash2_batched_propose_and_append_taps_match_sequential() {
        let target_cfg = Qwen3_5Config {
            vocab_size: 48,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 32,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 1.0,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            rope_scaling: None,
        };

        let draft = tiny_dflash2_model_with_block(&target_cfg, 7).unwrap();
        let h = target_cfg.hidden_size as i64;
        let padded_vocab = 64i64;
        let unpadded_vocab = target_cfg.vocab_size as usize;

        let embed_w = det_bf16(&[padded_vocab, h], 21, 0.35);
        let embed_tokens = Embedding::from_weight(&embed_w).unwrap();
        let embed_norm = MxArray::ones(&[h], Some(DType::BFloat16)).unwrap();
        let lm_head_w = det_bf16(&[padded_vocab, h], 22, 0.3);
        let lm_head = LinearProj::Standard(Linear::from_weights(&lm_head_w, None).unwrap());
        let lm_head_unpadded = lm_head_w.slice_axis(0, 0, unpadded_vocab as i64).unwrap();
        let mut view = Inkling2TargetView {
            embed_tokens: &embed_tokens,
            embed_norm: &embed_norm,
            lm_head: &lm_head,
            lm_head_bf16_unpadded: Some(&lm_head_unpadded),
            rms_norm_eps: 1e-6,
            unpadded_vocab_size: unpadded_vocab,
        };

        let greedy_params = ChatParams {
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
            max_new_tokens: 32,
            repetition_penalty: 1.0,
            repetition_context_size: 0,
            presence_penalty: 0.0,
            presence_context_size: 0,
            frequency_penalty: 0.0,
            frequency_context_size: 0,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            sampling_config: Some(SamplingConfig {
                temperature: Some(0.0),
                ..SamplingConfig::default()
            }),
            report_performance: false,
            reuse_cache: true,
            include_reasoning: false,
            extra_eos_ids: Vec::new(),
            enable_mtp: true,
            mtp_depth: 7,
            mtp_adaptive_depth: false,
        };

        for (n_lanes, init_lens, keeps, max_drafts_list, anchors) in [
            (
                2usize,
                vec![2usize, 6],
                vec![3usize, 5],
                vec![7usize, 4],
                vec![5u32, 17],
            ),
            (
                3usize,
                vec![0usize, 4, 9],
                vec![2usize, 6, 3],
                vec![7usize, 5, 3],
                vec![3u32, 19, 41],
            ),
        ] {
            let mut seq_ctxs: Vec<_> = (0..n_lanes)
                .map(|_| Inkling2Dflash2Context::new(&draft.config))
                .collect();
            let mut bat_ctxs: Vec<_> = (0..n_lanes)
                .map(|_| Inkling2Dflash2Context::new(&draft.config))
                .collect();

            for (i, &init_len) in init_lens.iter().enumerate() {
                if init_len > 0 {
                    let taps = vec![
                        det_bf16(&[init_len as i64, h], 500 + i as u64 * 10, 0.3),
                        det_bf16(&[init_len as i64, h], 501 + i as u64 * 10, 0.3),
                    ];
                    let toks: Vec<u32> = (0..init_len)
                        .map(|t| ((i * 7 + t + 1) % unpadded_vocab) as u32)
                        .collect();
                    seq_ctxs[i].append_taps(&draft, &taps, &toks).unwrap();
                    bat_ctxs[i].append_taps(&draft, &taps, &toks).unwrap();
                }
            }

            // 1. Batched propose vs sequential propose with different max_drafts + confidence=true.
            let mut seq_props = Vec::with_capacity(n_lanes);
            for i in 0..n_lanes {
                let mut rng = rand::rng();
                let mut prop = seq_ctxs[i]
                    .propose_block(
                        &draft,
                        &mut view,
                        anchors[i],
                        max_drafts_list[i],
                        &greedy_params,
                        &mut rng,
                        true,
                    )
                    .unwrap();
                prop.materialize_draft_ids().unwrap();
                seq_props.push(prop);
            }

            let mut rngs_owned: Vec<_> = (0..n_lanes).map(|_| rand::rng()).collect();
            let mut rng_refs: Vec<&mut dyn rand::Rng> = rngs_owned
                .iter_mut()
                .map(|r| r as &mut dyn rand::Rng)
                .collect();
            let mut bat_lanes: Vec<BatchedDflash2Lane<'_>> = bat_ctxs
                .iter_mut()
                .enumerate()
                .map(|(i, ctx)| BatchedDflash2Lane {
                    ctx,
                    anchor_id: anchors[i],
                    max_drafts: max_drafts_list[i],
                    params: &greedy_params,
                    confidence: true,
                })
                .collect();
            let mut bat_props: Vec<DsparkProposal> =
                Inkling2Dflash2Context::propose_blocks_batched(
                    &draft,
                    &mut view,
                    &mut bat_lanes,
                    &mut rng_refs,
                )
                .into_iter()
                .map(Result::unwrap)
                .collect();
            for (i, bp) in bat_props.iter_mut().enumerate() {
                bp.materialize_draft_ids().unwrap();
                assert_eq!(bp.draft_ids.len(), max_drafts_list[i]);
                assert_eq!(
                    bp.draft_ids, seq_props[i].draft_ids,
                    "Inkling2 N={n_lanes} lane={i} draft_ids mismatch"
                );
                let sp = seq_props[i].keep_probabilities.as_ref().unwrap();
                let kp = bp.keep_probabilities.as_ref().unwrap();
                assert_eq!(kp.len(), max_drafts_list[i]);
                for (a, b) in sp.iter().zip(kp.iter()) {
                    assert!((a - b).abs() <= 2e-2, "confidence diff: {a} vs {b}");
                }
            }

            // 2. Batched tap append vs sequential tap append (including fused_kept slices!).
            let mut lane_taps = Vec::with_capacity(n_lanes);
            let mut lane_toks = Vec::with_capacity(n_lanes);
            let mut seq_fused = Vec::with_capacity(n_lanes);
            for (i, &keep) in keeps.iter().enumerate() {
                let taps = vec![
                    det_bf16(&[8, h], 700 + i as u64 * 10, 0.3),
                    det_bf16(&[8, h], 701 + i as u64 * 10, 0.3),
                ];
                let toks: Vec<u32> = (0..keep)
                    .map(|t| ((i * 9 + t + 2) % unpadded_vocab) as u32)
                    .collect();
                let fused = seq_ctxs[i]
                    .append_taps(&draft, &taps, &toks)
                    .unwrap()
                    .unwrap();
                seq_fused.push(fused);
                lane_taps.push(taps);
                lane_toks.push(toks);
            }
            let bat_fused: Vec<Option<MxArray>> = {
                let mut items: Vec<(&mut Inkling2Dflash2Context, &[MxArray], &[u32])> = bat_ctxs
                    .iter_mut()
                    .zip(lane_taps.iter())
                    .zip(lane_toks.iter())
                    .map(|((ctx, taps), toks)| (ctx, taps.as_slice(), toks.as_slice()))
                    .collect();
                Inkling2Dflash2Context::append_taps_batched(&draft, &mut items)
                    .into_iter()
                    .map(Result::unwrap)
                    .collect()
            };

            for i in 0..n_lanes {
                assert_eq!(bat_ctxs[i].committed_len(), seq_ctxs[i].committed_len());
                assert_eq!(
                    bat_ctxs[i].cache.token_history(),
                    seq_ctxs[i].cache.token_history()
                );
                assert_close(
                    bat_fused[i].as_ref().unwrap(),
                    &seq_fused[i],
                    2e-2,
                    &format!("Inkling2 N={n_lanes} lane={i} fused_kept"),
                );
                for (l, (sv, bv)) in seq_ctxs[i]
                    .cache
                    .window_views()
                    .into_iter()
                    .zip(bat_ctxs[i].cache.window_views())
                    .enumerate()
                {
                    let (sk, sv_arr) = sv.unwrap();
                    let (bk, bv_arr) = bv.unwrap();
                    assert_close(&sk, &bk, 2e-2, &format!("N={n_lanes} lane={i} layer={l} K"));
                    assert_close(
                        &sv_arr,
                        &bv_arr,
                        2e-2,
                        &format!("N={n_lanes} lane={i} layer={l} V"),
                    );
                }
            }
        }
    }

    #[test]
    fn inkling2_dflash2_batched_mid_append_and_selector_fault_isolation() {
        let target_cfg = Qwen3_5Config {
            vocab_size: 48,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 32,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 1.0,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            rope_scaling: None,
        };

        let draft = tiny_dflash2_model_with_block(&target_cfg, 7).unwrap();
        let h = target_cfg.hidden_size as i64;
        let padded_vocab = 64i64;
        let unpadded_vocab = target_cfg.vocab_size as usize;
        let n_taps = draft.config.target_layers.len();

        let embed_w = det_bf16(&[padded_vocab, h], 31, 0.35);
        let embed_tokens = Embedding::from_weight(&embed_w).unwrap();
        let embed_norm = MxArray::ones(&[h], Some(DType::BFloat16)).unwrap();
        let lm_head_w = det_bf16(&[padded_vocab, h], 32, 0.3);
        let lm_head = LinearProj::Standard(Linear::from_weights(&lm_head_w, None).unwrap());
        let lm_head_unpadded = lm_head_w.slice_axis(0, 0, unpadded_vocab as i64).unwrap();
        let mut view = Inkling2TargetView {
            embed_tokens: &embed_tokens,
            embed_norm: &embed_norm,
            lm_head: &lm_head,
            lm_head_bf16_unpadded: Some(&lm_head_unpadded),
            rms_norm_eps: 1e-6,
            unpadded_vocab_size: unpadded_vocab,
        };

        let greedy_params = ChatParams {
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
            max_new_tokens: 32,
            repetition_penalty: 1.0,
            repetition_context_size: 0,
            presence_penalty: 0.0,
            presence_context_size: 0,
            frequency_penalty: 0.0,
            frequency_context_size: 0,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            sampling_config: Some(SamplingConfig {
                temperature: Some(0.0),
                ..SamplingConfig::default()
            }),
            report_performance: false,
            reuse_cache: true,
            include_reasoning: false,
            extra_eos_ids: Vec::new(),
            enable_mtp: true,
            mtp_depth: 7,
            mtp_adaptive_depth: false,
        };
        let mut host_sampled_params = greedy_params.clone();
        host_sampled_params.repetition_penalty = 1.1;
        host_sampled_params.sampling_config = Some(SamplingConfig {
            temperature: Some(0.8),
            ..SamplingConfig::default()
        });

        // 1. Mid-`append_batched` ring-store failure on lane 1 of 3: lane 0 already committed,
        //    lane 1 fails on its corrupted layer buffer, lane 2 still commits independently and
        //    matches sequential K/V + fused slices.
        let mut seq_ctxs: Vec<_> = (0..3)
            .map(|_| Inkling2Dflash2Context::new(&draft.config))
            .collect();
        let mut bat_ctxs: Vec<_> = (0..3)
            .map(|_| Inkling2Dflash2Context::new(&draft.config))
            .collect();
        for i in 0..3 {
            let init_len = 3 + i;
            let taps: Vec<MxArray> = (0..n_taps)
                .map(|t| det_bf16(&[init_len as i64, h], 900 + i as u64 * 10 + t as u64, 0.25))
                .collect();
            let toks: Vec<u32> = (0..init_len)
                .map(|k| ((i * 7 + k + 1) % unpadded_vocab) as u32)
                .collect();
            seq_ctxs[i].append_taps(&draft, &taps, &toks).unwrap();
            bat_ctxs[i].append_taps(&draft, &taps, &toks).unwrap();
        }
        // Corrupt the last layer's ring buffer on lane 1 so `store_rows` fails mid-batch.
        let last_layer = draft.config.num_hidden_layers.saturating_sub(1);
        DFlash2Model::corrupt_window_buffer_for_test(&mut bat_ctxs[1].cache, last_layer);

        let keeps = [3usize, 4, 2];
        let mut lane_taps = Vec::with_capacity(3);
        let mut lane_toks = Vec::with_capacity(3);
        let mut seq_fused = Vec::with_capacity(3);
        for (i, &keep) in keeps.iter().enumerate() {
            let taps: Vec<MxArray> = (0..n_taps)
                .map(|t| det_bf16(&[8, h], 1100 + i as u64 * 10 + t as u64, 0.25))
                .collect();
            let toks: Vec<u32> = (0..keep)
                .map(|k| ((i * 5 + k + 2) % unpadded_vocab) as u32)
                .collect();
            let fused = seq_ctxs[i]
                .append_taps(&draft, &taps, &toks)
                .unwrap()
                .unwrap();
            seq_fused.push(fused);
            lane_taps.push(taps);
            lane_toks.push(toks);
        }
        let bat_results = {
            let mut items: Vec<(&mut Inkling2Dflash2Context, &[MxArray], &[u32])> = bat_ctxs
                .iter_mut()
                .zip(lane_taps.iter())
                .zip(lane_toks.iter())
                .map(|((ctx, taps), toks)| (ctx, taps.as_slice(), toks.as_slice()))
                .collect();
            Inkling2Dflash2Context::append_taps_batched(&draft, &mut items)
        };
        assert!(
            bat_results[0].is_ok(),
            "lane 0 must commit: {:?}",
            bat_results[0].as_ref().err()
        );
        assert!(
            bat_results[1].is_err(),
            "corrupted lane 1 must return per-lane Err"
        );
        assert!(
            bat_results[2].is_ok(),
            "lane 2 must recover and commit: {:?}",
            bat_results[2].as_ref().err()
        );
        for i in [0usize, 2] {
            assert_eq!(bat_ctxs[i].committed_len(), seq_ctxs[i].committed_len());
            assert_eq!(
                bat_ctxs[i].cache.token_history(),
                seq_ctxs[i].cache.token_history()
            );
            let fused = bat_results[i].as_ref().unwrap().as_ref().unwrap();
            assert_close(
                fused,
                &seq_fused[i],
                2e-2,
                &format!("healthy lane {i} fused"),
            );
            for (l, (sv, bv)) in seq_ctxs[i]
                .cache
                .window_views()
                .into_iter()
                .zip(bat_ctxs[i].cache.window_views())
                .enumerate()
            {
                let (sk, sv_arr) = sv.unwrap();
                let (bk, bv_arr) = bv.unwrap();
                assert_close(&sk, &bk, 2e-2, &format!("healthy lane {i} layer {l} K"));
                assert_close(
                    &sv_arr,
                    &bv_arr,
                    2e-2,
                    &format!("healthy lane {i} layer {l} V"),
                );
            }
        }

        // 2. Per-lane selector fault during `propose_blocks_batched`: poison lane 0 with NaN
        //    context + host sampling (`temperature=0.8, repetition_penalty=1.1`), while healthy
        //    lane 2 stays bit-identical to solo `propose_block`.
        let nan_vals = vec![f32::NAN; 2 * h as usize];
        let nan_tap = MxArray::from_float32(&nan_vals, &[2, h])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let nan_taps: Vec<MxArray> = (0..n_taps).map(|_| nan_tap.clone()).collect();
        let mut bad_ctx = Inkling2Dflash2Context::new(&draft.config);
        bad_ctx.append_taps(&draft, &nan_taps, &[1, 2]).unwrap();

        let mut solo_rng = rand::rng();
        let mut expected_healthy = seq_ctxs[2]
            .propose_block(
                &draft,
                &mut view,
                11,
                5,
                &greedy_params,
                &mut solo_rng,
                true,
            )
            .unwrap();
        expected_healthy.materialize_draft_ids().unwrap();

        let mut rngs_owned: Vec<_> = (0..2).map(|_| rand::rng()).collect();
        let mut rng_refs: Vec<&mut dyn rand::Rng> = rngs_owned
            .iter_mut()
            .map(|r| r as &mut dyn rand::Rng)
            .collect();
        let mut prop_lanes = vec![
            BatchedDflash2Lane {
                ctx: &mut bad_ctx,
                anchor_id: 3,
                max_drafts: 7,
                params: &host_sampled_params,
                confidence: false,
            },
            BatchedDflash2Lane {
                ctx: &mut bat_ctxs[2],
                anchor_id: 11,
                max_drafts: 5,
                params: &greedy_params,
                confidence: true,
            },
        ];
        let mut prop_iter = Inkling2Dflash2Context::propose_blocks_batched(
            &draft,
            &mut view,
            &mut prop_lanes,
            &mut rng_refs,
        )
        .into_iter();
        assert!(
            prop_iter.next().unwrap().is_err(),
            "NaN-poisoned host-sampled lane must fail selection cleanly"
        );
        let mut healthy_prop = prop_iter
            .next()
            .unwrap()
            .expect("healthy peer must succeed despite sibling selector error");
        healthy_prop.materialize_draft_ids().unwrap();
        assert_eq!(healthy_prop.draft_ids, expected_healthy.draft_ids);
    }

    /// WS2: Exclusive DFlash2 `paged_prefill_dflash2_with_taps` must include installed
    /// `seq_extra_capture_boundaries` in both `targets` (so `needs_taps` activates over
    /// `[B - sliding_window .. B)`) and `prefill_sub_boundaries_with_extra`, capturing a
    /// `StateClass::Ordinary` prompt head (`is_head = true`) with a complete `PackedDraftWindow`
    /// at `B` across both small chunks (`chunk_limit = 32`, skipping `[0..32)` taps) and large
    /// chunks (`chunk_limit = 128`, splitting `[0..128)` at `B = 96`).
    #[test]
    #[cfg(target_os = "macos")]
    fn paged_prefill_dflash2_with_taps_captures_complete_draft_window_at_extra_breakpoint() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let prompt: Vec<u32> = (0u32..256u32).map(|i| 10 + (i % 70)).collect();
        let breakpoint_b = 96u32; // [B - sliding_window .. B) = [32..96), disjoint from junction [160..224)

        for (idx, chunk_limit) in [32usize, 128usize].into_iter().enumerate() {
            let mut inner =
                super::super::scheduler::tests::build_synthetic_dflash2_inner_with_window(
                    512, None, 64, 2,
                );
            inner.paged.as_mut().expect("paged").prefill_chunk_size = chunk_limit;
            let seq_id = 1u32;
            let salt = 0xDF20u64 + idx as u64;

            inner.set_seq_prefix_breakpoints_inner(seq_id, prompt.len() as u32, &[breakpoint_b]);
            let prefix = inner
                .prepare_prefix_sync(seq_id, &prompt, true, salt)
                .expect("prepare cold prefix");
            assert_eq!(prefix.effective_cached_prefix_len, 0);

            let mut ctx = Inkling2Dflash2Context::new(&inner.dflash2.as_ref().unwrap().config);
            let _logits = inner
                .paged_prefill_dflash2_with_taps(seq_id, &prompt, &prefix, &mut ctx)
                .expect("paged_prefill_dflash2_with_taps");

            let rt = inner.paged.as_ref().expect("paged");
            let entry_b = rt
                .state_store
                .store
                .entries()
                .find(|e| e.depth() == breakpoint_b as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "chunk_limit={chunk_limit}: must capture extra breakpoint B={breakpoint_b}"
                    )
                });
            assert!(
                !entry_b.checkpoint(),
                "chunk_limit={chunk_limit}: extra breakpoint must publish as StateClass::Ordinary"
            );
            assert!(
                rt.state_store.is_prompt_head(entry_b.id()),
                "chunk_limit={chunk_limit}: extra breakpoint must be marked in prompt_heads"
            );
            let win_b = entry_b
                .ram()
                .expect("ram anchor at B")
                .draft_window
                .as_ref()
                .expect("complete PackedDraftWindow must be captured at B");
            assert_eq!(win_b.slots, 64);
            assert!(
                win_b.is_complete_for(breakpoint_b),
                "chunk_limit={chunk_limit}: PackedDraftWindow at B={breakpoint_b} must be complete \
                 (begin={}, end={}, slots={})",
                win_b.begin,
                win_b.end,
                win_b.slots
            );

            // Junction 224 must also be captured with a complete PackedDraftWindow (non-head).
            let entry_j = rt
                .state_store
                .store
                .entries()
                .find(|e| e.depth() == 224)
                .expect("junction 224 must also be captured");
            assert!(!rt.state_store.is_prompt_head(entry_j.id()));
            assert!(
                entry_j
                    .ram()
                    .and_then(|r| r.draft_window.as_ref())
                    .is_some_and(|w| w.is_complete_for(224))
            );
        }

        // Control: without installing extra breakpoint `96`, `paged_prefill_dflash2_with_taps`
        // captures ONLY junction `224` and does NOT capture `96`.
        let mut ctrl = super::super::scheduler::tests::build_synthetic_dflash2_inner_with_window(
            512, None, 64, 2,
        );
        ctrl.paged.as_mut().expect("paged").prefill_chunk_size = 32;
        let prefix = ctrl
            .prepare_prefix_sync(2, &prompt, true, 0xDF99)
            .expect("prepare control");
        let mut ctx = Inkling2Dflash2Context::new(&ctrl.dflash2.as_ref().unwrap().config);
        let _ = ctrl
            .paged_prefill_dflash2_with_taps(2, &prompt, &prefix, &mut ctx)
            .expect("control prefill");
        let depths: Vec<usize> = ctrl
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .store
            .entries()
            .map(|e| e.depth())
            .collect();
        assert_eq!(
            depths,
            vec![224],
            "control run without extra breakpoint must capture only junction 224"
        );
    }
}
