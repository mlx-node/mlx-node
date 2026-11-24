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
    DFlash2Config, DFlash2ContextCache, DFlash2Model, DFlash2TargetShape, DFlash2TargetView,
    SelectorPath, read_dflash2_config,
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
    pub(crate) lm_head_bf16_unpadded: Option<&'a MxArray>,
    pub(crate) rms_norm_eps: f32,
    pub(crate) unpadded_vocab_size: usize,
}

impl DFlash2TargetView for Inkling2TargetView<'_> {
    fn embed(&mut self, block: &MxArray, _ids: &[i32]) -> Result<MxArray> {
        let raw = self.embed_tokens.forward(block)?;
        let rows = raw.shape_at(1)?;
        let hidden = raw.shape_at(2)?;
        let flat = raw.reshape(&[rows, hidden])?;
        let normed =
            apply_sublayer_rms_norm(&flat, self.embed_norm, self.rms_norm_eps, DType::BFloat16)?;
        normed.reshape(&[1, rows, hidden])
    }

    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray> {
        let rows = hidden.shape_at(1)?;
        let dim = hidden.shape_at(2)?;
        let x2d_raw = hidden.reshape(&[rows, dim])?;
        let x2d = if x2d_raw.dtype()? == DType::BFloat16 {
            x2d_raw
        } else {
            x2d_raw.astype(DType::BFloat16)?
        };

        #[cfg(target_os = "macos")]
        if let LinearProj::Quantized(ql) = self.lm_head
            && ql.is_symmetric_q8()
            && unsafe { mlx_sys::mlx_metal_is_available() }
            && (1..=32).contains(&rows)
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
            return logits_2d.reshape(&[1, rows, out_cols as i64]);
        }

        #[cfg(target_os = "macos")]
        if let LinearProj::Standard(lin) = self.lm_head
            && lin.get_bias().is_none()
            && unsafe { mlx_sys::mlx_metal_is_available() }
            && (1..=32).contains(&rows)
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
                return logits_2d.reshape(&[1, rows, out_cols as i64]);
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
        sliced.reshape(&[1, rows, out_cols as i64])
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

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.cache.collect_arrays(out);
    }

    pub(crate) fn eval_state(&self) -> Result<()> {
        self.cache.eval()
    }

    /// Convert Inkling-2-Mini's 2D `[T, hidden]` target taps into `[1, keep, hidden]` BF16,
    /// fuse via `draft.fuse_context`, and append to the sliding DFlash2 KV ring.
    pub(crate) fn append_taps(
        &mut self,
        draft: &DFlash2Model,
        taps: &[MxArray],
        token_ids: &[u32],
    ) -> Result<()> {
        let keep = token_ids.len();
        if keep == 0 {
            return Ok(());
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
            return self
                .cache
                .append_full_width_prefix(draft, &fused, base, token_ids);
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
        self.cache.append(draft, &fused, base, token_ids)
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
    ) -> Result<()> {
        if accepted_rows.len() != accepted_tokens.len() {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 append_taps_path row count {} != token count {}",
                accepted_rows.len(),
                accepted_tokens.len()
            )));
        }
        if accepted_rows.is_empty() {
            return Ok(());
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
        self.cache.append(draft, &fused, base, accepted_tokens)
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

fn empty_proposal() -> DsparkProposal {
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
    /// target hidden taps (`[5, 23, 41, 48]`) on the final sliding window tail.
    pub(crate) fn paged_prefill_dflash2_with_taps(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        ctx: &mut Inkling2Dflash2Context,
    ) -> Result<MxArray> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Inkling2 DFlash2 prefill: empty tokens"));
        }
        let chunk_limit = {
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
            rt.prefill_chunk_size
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

        let total_len = tokens.len();
        let mut offset = 0usize;
        let mut final_logits = None;

        while offset < total_len {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let end = (offset + chunk_limit).min(total_len);
            let chunk = &tokens[offset..end];
            let is_final_chunk = end == total_len;
            // Rows strictly older than `sliding_window` behind `total_len` are
            // evicted unread before the first draft cycle: skip tap fusion for them.
            let needs_taps = total_len.saturating_sub(end) < sliding_window;

            let first_pos = {
                let rt = self.paged.as_mut().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 prefill: paged runtime missing")
                })?;
                rt.ensure_conv_state(seq_id, &self.config)?;
                rt.coordinator
                    .request_token_count_all(seq_id)
                    .map_err(Error::from_reason)?
            };
            self.record_tokens_all_exclusive(seq_id, chunk)
                .map_err(Error::from_reason)?;

            let tap_arg: &[usize] = if needs_taps { &tap_visits } else { &[] };
            let (logits_opt, taps) = self.forward_paged_single_chunk_with_taps(
                seq_id,
                chunk,
                None,
                first_pos,
                is_final_chunk,
                tap_arg,
            )?;

            let _ = self.settle_paged_step(seq_id)?;
            if needs_taps {
                let draft = self.dflash2.as_ref().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 prefill: DFlash2 draft model disappeared")
                })?;
                ctx.append_taps(draft, &taps, chunk)?;
            } else {
                ctx.cache.record_only(chunk)?;
            }
            if let Some(rt) = self.paged.as_mut()
                && let Ok(pos_after) = u32::try_from(first_pos as usize + chunk.len())
            {
                rt.record_block_boundary_conv(seq_id, pos_after);
            }

            if is_final_chunk {
                final_logits = logits_opt;
            } else {
                crate::array::clear_cache();
            }
            offset = end;
        }

        final_logits
            .ok_or_else(|| Error::from_reason("Inkling2 DFlash2 prefill produced no final logits"))
    }
}
