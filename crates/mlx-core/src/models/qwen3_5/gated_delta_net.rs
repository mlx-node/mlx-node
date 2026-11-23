use crate::array::{DType, MxArray};
use crate::nn::{Activations, Conv1d, Linear, RMSNormGated, rms_norm_scaled, rms_norm_unscaled};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::{
    GdnCarry, GdnCompleteOut, GdnKernelTape, GdnPrefill, GdnPrologue, GdnTail,
    gated_delta_fused_complete_deferred, gated_delta_fused_complete_with_prepared,
    gated_delta_fused_prefill, gated_delta_split_at, gated_delta_update, gated_delta_update_fused,
    gated_delta_update_fused_deferred, gated_delta_update_tree_deferred,
    gated_delta_update_tree_with_tape, gated_delta_update_with_tape, prefill_fused_enabled,
};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_MODE, LinearProj, MixerResidual, PreparedSg8, QuantizedLinear,
};

/// Chunk-relative rows at which a prefill forward captures every GDN layer's
/// state: strictly increasing, each in `1..=T`. Only the recurrence dispatch
/// is split at these rows; projections, conv and taps still run over the
/// whole chunk, so the chunk's outputs and caches keep their bits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCapturePlan<'a> {
    pub rows: &'a [usize],
}

/// One GDN layer's state after chunk rows `[0, c)`: the conv history
/// `[1, K-1, W]` and the f32 recurrent state `[1, Hv, Dv, Dk]`. Both are
/// fresh buffers: no cache slot, state blob or later chunk can write them.
#[derive(Clone)]
pub(crate) struct GdnLayerCapture {
    #[cfg_attr(not(test), expect(dead_code))]
    pub conv: MxArray,
    #[cfg_attr(not(test), expect(dead_code))]
    pub rec: MxArray,
}

#[cfg(test)]
thread_local! {
    // Recurrence routes the GDN forwards took on this thread, in order.
    static GDN_ROUTES: std::cell::RefCell<Vec<GdnRoute>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnRoute {
    FusedPrefill,
    Complete,
    FusedTail,
    Chain,
}

#[cfg(test)]
fn note_route(route: GdnRoute) {
    GDN_ROUTES.with(|routes| routes.borrow_mut().push(route));
}

#[cfg(test)]
pub(crate) fn take_gdn_routes() -> Vec<GdnRoute> {
    GDN_ROUTES.with(|routes| routes.take())
}

/// One lane of [`GatedDeltaNet::forward_wave`]: the lane's own cell (the
/// verify layout of `GdnStateBlobs::verify_cell`: slots 0/1 its live cell,
/// 2/3 its other cell, 4..9 its pending tape) and its tape sink.
pub(crate) struct GdnWaveLane<'a> {
    pub cell: &'a mut ArraysCache,
    pub tape: &'a mut Option<GdnLayerTape>,
}

/// Rows `[start, end)` of a `[B, T, ...]` window.
fn window_rows(x: &MxArray, start: usize, end: usize) -> Result<MxArray> {
    x.slice_axis(1, start as i64, end as i64)
}

/// Collects each split row's carried recurrent state.
fn captured_states(captured: Vec<GdnCarry>) -> Result<Vec<MxArray>> {
    captured
        .into_iter()
        .map(|carry| {
            carry
                .state
                .ok_or_else(|| Error::from_reason("GDN capture: a split row carried no state"))
        })
        .collect()
}

/// Per-GDN-layer tape recorded during the eager MTP verify forward.
///
/// Holds everything the eager MTP rollback replay needs to reconstruct the
/// AR-exact carried GDN state for a layer:
///   * `kernel` — the `(q, k, v, g, beta)` window handed to the per-step
///     recurrence kernel (used to replay the recurrent state at T=1).
///   * `qkv` — the post-mask, pre-conv `[B, T, conv_dim]` activation (used to
///     rebuild the conv state by slicing the accepted prefix).
///   * `conv_kernel_dim` — depthwise conv kernel size; `keep = conv_kernel_dim
///     - 1` is the conv-state window length.
///
/// On the deferred complete route (the DFlash2 verify) `qkv` is the kernel's
/// own row-contiguous raw-input output, not a view of the projection.
///
/// All array fields are lazy `MxArray` clones (no eval, no copy) so recording
/// stays inside the fused lazy MLX graph.
#[derive(Clone)]
pub(crate) struct GdnLayerTape {
    pub kernel: GdnKernelTape,
    pub qkv: MxArray,
    pub conv_kernel_dim: i32,
}

/// Rebuild the convolution history from the pre-verify snapshot and accepted rows.
fn replay_conv_state(
    qkv: &MxArray,
    snapshot: Option<&MxArray>,
    keep: i64,
    accepted_steps: usize,
) -> Result<MxArray> {
    let accepted = i64::try_from(accepted_steps)
        .map_err(|_| Error::from_reason("GDN convolution replay prefix is too large"))?;
    if keep <= 0 || accepted > qkv.shape_at(1)? {
        return Err(Error::from_reason("invalid GDN convolution replay window"));
    }
    if accepted >= keep {
        return qkv.slice_axis(1, accepted - keep, accepted);
    }
    if accepted == 0 {
        return match snapshot {
            Some(state) => Ok(state.clone()),
            None => MxArray::zeros(
                &[qkv.shape_at(0)?, keep, qkv.shape_at(2)?],
                Some(qkv.dtype()?),
            ),
        };
    }
    let prefix = qkv.slice_axis(1, 0, accepted)?;
    let tail_keep = keep - accepted;
    let prior_tail = match snapshot {
        Some(state) => {
            let state_len = state.shape_at(1)?;
            state.slice_axis(1, state_len - tail_keep, state_len)?
        }
        None => MxArray::zeros(
            &[qkv.shape_at(0)?, tail_keep, qkv.shape_at(2)?],
            Some(qkv.dtype()?),
        )?,
    };
    MxArray::concatenate(&prior_tail, &prefix, 1)
}

impl GdnLayerTape {
    pub(crate) fn stack_rows(rows: &[Self]) -> Result<Self> {
        let first = rows
            .first()
            .ok_or_else(|| Error::from_reason("empty GDN tape batch"))?;
        let width = first.kernel.window_len()?;
        for row in rows {
            if row.qkv.shape_at(0)? != 1
                || row.kernel.window_len()? != width
                || row.conv_kernel_dim != first.conv_kernel_dim
            {
                return Err(Error::from_reason("incompatible GDN owner tapes"));
            }
        }
        let combine = |get: fn(&Self) -> &MxArray| {
            MxArray::concatenate_many(rows.iter().map(get).collect(), Some(0))
        };
        Ok(Self {
            kernel: GdnKernelTape {
                q: combine(|row| &row.kernel.q)?,
                k: combine(|row| &row.kernel.k)?,
                v: combine(|row| &row.kernel.v)?,
                g: combine(|row| &row.kernel.g)?,
                beta: combine(|row| &row.kernel.beta)?,
                tiled_heads: first.kernel.tiled_heads,
            },
            qkv: combine(|row| &row.qkv)?,
            conv_kernel_dim: first.conv_kernel_dim,
        })
    }

    /// Keep one independent owner's full time window for unequal acceptance
    /// replay. Slicing the batch dimension retains GPU storage and graph roots.
    pub(crate) fn row(&self, row: usize, batch: usize) -> Result<Self> {
        if row >= batch || self.qkv.shape_at(0)? != batch as i64 {
            return Err(Error::from_reason("GDN tape owner row is out of range"));
        }
        let select = |array: &MxArray| -> Result<MxArray> {
            if array.shape_at(0)? != batch as i64 {
                return Err(Error::from_reason("GDN tape batch dimensions disagree"));
            }
            array.slice_axis(0, row as i64, row as i64 + 1)
        };
        Ok(Self {
            kernel: GdnKernelTape {
                q: select(&self.kernel.q)?,
                k: select(&self.kernel.k)?,
                v: select(&self.kernel.v)?,
                g: select(&self.kernel.g)?,
                beta: select(&self.kernel.beta)?,
                tiled_heads: self.kernel.tiled_heads,
            },
            qkv: select(&self.qkv)?,
            conv_kernel_dim: self.conv_kernel_dim,
        })
    }

    /// Replay the accepted prefix into the pre-verify snapshot caches.
    ///
    /// `accepted_steps = accepted_drafts + 1`. Rebuilds BOTH the recurrent
    /// state (per-step T=1 kernel replay from the f32 `snapshot_recurrent`,
    /// = AR decode bit-for-bit) and the conv state (slice the
    /// accepted prefix of the recorded `qkv` onto `snapshot_conv`), then writes
    /// them into the live cache slots (slot 0 = conv_state, slot 1 =
    /// recurrent_state).
    ///
    /// On full accept this is idempotent with what the verify forward already
    /// set the conv state to; on partial accept it correctly trims to the
    /// accepted prefix. Stays inside the lazy graph (no eval).
    pub(crate) fn replay_into(
        &self,
        cache: &mut ArraysCache,
        snapshot_conv: Option<&MxArray>,
        snapshot_recurrent: Option<&MxArray>,
        accepted_steps: usize,
    ) -> Result<()> {
        // --- Recurrent state ---------------------------------------------
        // Start from the pre-verify (f32) recurrent state. If the snapshot
        // had no recurrent state (cold cache — should not happen at decode
        // time), zero-init to the recorded shapes, f32 like the kernel's own
        // zero-state default.
        let start_state = match snapshot_recurrent {
            Some(s) => s.clone(),
            None => {
                let batch = self.kernel.v.shape_at(0)?;
                let num_v_heads = self.kernel.v.shape_at(2)?;
                let v_dim = self.kernel.v.shape_at(3)?;
                let k_dim = self.kernel.q.shape_at(3)?;
                MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(DType::Float32))?
            }
        };
        let new_recurrent = self
            .kernel
            .replay_recurrent_state(&start_state, accepted_steps)?;
        cache.set(1, new_recurrent)?;

        // --- Conv state --------------------------------------------------
        self.replay_conv_into(cache, snapshot_conv, accepted_steps)
    }

    /// Rebuild only `cache[0]` (`conv_state`) to the accepted prefix. Used by
    /// CUDA deferred-GDN commit where `cache[1]` already holds `S_folded`.
    pub(crate) fn replay_conv_into(
        &self,
        cache: &mut ArraysCache,
        snapshot_conv: Option<&MxArray>,
        accepted_steps: usize,
    ) -> Result<()> {
        let keep = (self.conv_kernel_dim - 1) as i64;
        if keep > 0 {
            cache.set(
                0,
                replay_conv_state(&self.qkv, snapshot_conv, keep, accepted_steps)?,
            )?;
        }
        Ok(())
    }

    /// Rebuild `cache[0]` (`conv_state`) along an accepted speculative tree
    /// path (`accepted_nodes`, ordered ancestor-to-descendant).
    ///
    /// When `is_contiguous_prefix` is true, delegates to [`Self::replay_conv_into`].
    /// Otherwise gathers the trailing `min(keep_conv, accepted_nodes.len())`
    /// rows from `self.qkv` (reusing the caller's `accepted_i32` when provided)
    /// and prepends any needed tail from `snapshot_conv`.
    pub(crate) fn replay_conv_tree_into(
        &self,
        cache: &mut ArraysCache,
        snapshot_conv: Option<&MxArray>,
        accepted_nodes: &[usize],
        is_contiguous_prefix: bool,
        accepted_i32: Option<&MxArray>,
    ) -> Result<()> {
        let keep_conv = (self.conv_kernel_dim - 1) as i64;
        if keep_conv <= 0 {
            return Ok(());
        }
        let window_len = self.qkv.shape_at(1)? as usize;
        for &node in accepted_nodes {
            if node >= window_len {
                return Err(Error::from_reason(format!(
                    "GdnLayerTape::replay_conv_tree_into: node {node} >= window {window_len}"
                )));
            }
        }
        if is_contiguous_prefix || accepted_nodes.is_empty() {
            return self.replay_conv_into(cache, snapshot_conv, accepted_nodes.len());
        }
        let n = accepted_nodes.len() as i64;
        let new_conv = if n >= keep_conv {
            // Only gather the final `keep_conv` nodes along the accepted branch.
            if let Some(idx_arr) = accepted_i32 {
                let tail_idx = idx_arr.slice_axis(0, n - keep_conv, n)?;
                self.qkv.take(&tail_idx, 1)?
            } else {
                let start = accepted_nodes.len() - keep_conv as usize;
                let tail_nodes: Vec<i32> = accepted_nodes[start..]
                    .iter()
                    .map(|&node| node as i32)
                    .collect();
                let tail_idx = MxArray::from_int32(&tail_nodes, &[keep_conv])?;
                self.qkv.take(&tail_idx, 1)?
            }
        } else {
            let owned_idx;
            let idx_ref = match accepted_i32 {
                Some(arr) => arr,
                None => {
                    let nodes: Vec<i32> = accepted_nodes.iter().map(|&node| node as i32).collect();
                    owned_idx = MxArray::from_int32(&nodes, &[n])?;
                    &owned_idx
                }
            };
            let gathered = self.qkv.take(idx_ref, 1)?;
            let tail_keep = keep_conv - n;
            let prior_tail = match snapshot_conv {
                Some(state) => {
                    let state_len = state.shape_at(1)?;
                    state.slice_axis(1, state_len - tail_keep, state_len)?
                }
                None => MxArray::zeros(
                    &[self.qkv.shape_at(0)?, tail_keep, self.qkv.shape_at(2)?],
                    Some(self.qkv.dtype()?),
                )?,
            };
            MxArray::concatenate(&prior_tail, &gathered, 1)?
        };
        cache.set(0, new_conv)?;
        Ok(())
    }
}

/// Only the target's small row-major pair: these modes use identical sg8
/// per-column arithmetic at N32 and N64; never tile the merged 64 rows.
fn merge_split_ba_m8(b: &LinearProj, a: &LinearProj) -> Result<Option<LinearProj>> {
    let eligible = |projection: &LinearProj| -> Result<bool> {
        let LinearProj::Quantized(q) = projection else {
            return Ok(false);
        };
        Ok(
            matches!((q.mode(), q.bits()), ("q4k", 4) | ("q5k", 5) | ("q6k", 6))
                && q.get_weight().shape_at(0)? == 32
                && q.get_weight().shape_at(1)? * 32 / i64::from(q.bits()) == 2048,
        )
    };
    if !eligible(b)? || !eligible(a)? {
        return Ok(None);
    }
    b.concat_rows(a)
}

fn project_split_ba_m8(
    x: &MxArray,
    merged: Option<&LinearProj>,
) -> Result<Option<(MxArray, MxArray)>> {
    let Some(merged) = merged else {
        return Ok(None);
    };
    if x.ndim()? != 3
        || x.dtype()? != DType::BFloat16
        || x.shape_at(2)? != 2048
        || x.size()? != 8 * 2048
        || !unsafe { sys::mlx_metal_is_available() }
        || unsafe { sys::mlx_default_device() } != 1
    {
        return Ok(None);
    }
    let output = merged.forward(x)?;
    let parts = output.split_sections(&[32], -1)?;
    Ok(Some((parts[0].clone(), parts[1].clone())))
}

/// GatedDeltaNet: Linear attention module using gated delta recurrence.
///
/// This replaces standard attention in most layers of Qwen3.5.
/// Uses depthwise convolution + state-space recurrence instead of softmax attention.
pub struct GatedDeltaNet {
    // Projections
    in_proj_qkvz: LinearProj, // hidden → key_dim*2 + value_dim*2 (q,k,v,z combined)
    in_proj_ba: LinearProj,   // hidden → num_v_heads * 2 (b and a combined)
    // Dynamic GGUFs can assign different packed modes to each half. In that
    // case the source projections stay split and each uses its own native
    // quantized matmul; no dense weight materialization is needed.
    split_in_proj_qkv_z: Option<(LinearProj, LinearProj)>,
    split_in_proj_b_a: Option<(LinearProj, LinearProj)>,
    // N32+32 row-major K-quants use the same sg8 tier at M8 after merging.
    // Wider prefill keeps the individual views because split-K follows N.
    split_in_proj_b_a_m8: Option<LinearProj>,
    conv1d: Conv1d,       // depthwise conv, groups = conv_dim
    norm: RMSNormGated,   // per-head norm: weight dim = value_head_dim
    out_proj: LinearProj, // value_dim → hidden

    // Learnable parameters
    dt_bias: MxArray, // [num_v_heads]
    a_log: MxArray,   // [num_v_heads]

    // Dimensions
    num_k_heads: i32,
    num_v_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel_dim: i32,
    tiled_gguf_layout: bool,
    /// Pre-stacked `[w_qkvz; w_ba]` transposed to `[hidden, qkvz_dim + ba_dim]`.
    /// Populated by `finalize_after_load()`. When present
    /// (and non-quantized), `forward()` does ONE matmul + two slices instead of
    /// two separate matmuls.
    in_proj_qkvz_ba_t: Option<MxArray>,
    /// Packed row-merged `[w_qkvz; w_ba]` quantized projection — the
    /// quantized counterpart of `in_proj_qkvz_ba_t`. `in_proj_qkvz` and
    /// `in_proj_ba` are replaced by row-slice views of it.
    in_proj_qkvz_ba_q: Option<LinearProj>,
    /// F32 `[conv_dim, 4]` view of the depthwise conv filter for the fused
    /// `mlx_qwen4_window_conv` kernel (history prepend + conv + SiLU +
    /// next-state in one dispatch). `None` when the loaded weight layout is
    /// not the flat `[conv_dim, 4]` tap-major order the kernel reads.
    conv1d_w4_f32: Option<MxArray>,
    /// Constant `[key_head_dim]` norm weights folding the q/k `inv_scale`
    /// factors into `fast_rms_norm` (one dispatch instead of norm + mul).
    qk_norm_w_q: MxArray,
    qk_norm_w_k: MxArray,
    /// F32 `[num_v_heads]` `-exp(a_log)` for the fused `mlx_qwen4_gdn_prepare`
    /// kernel, whose decay output is `exp(softplus(a + dt) * scale)`. Built in
    /// `set_a_log`; `None` until the weight loads or when the build fails.
    gdn_scale_f32: Option<MxArray>,
    /// F32 copy of `dt_bias` for `gdn_prepare`, which requires `float*` —
    /// checkpoints that store it bf16 would otherwise always miss the kernel.
    dt_bias_f32: Option<MxArray>,
}

/// What the prompt-window prep produced: either the gated output and new
/// state from the fused-prefill dispatch, or the PRE-NORM conv views for the
/// shared chain to finish (the fused kernel declined after the conv already
/// ran — the conv work is not redone).
enum PrefillPrep {
    Fused {
        gated: MxArray,
        state: MxArray,
        captured: Vec<MxArray>,
    },
    Raw {
        q: MxArray,
        k: MxArray,
        v: MxArray,
    },
}

/// The complete fused core's output for the layer: the gated rows, the new
/// state and conv history, and the per-split-row states. A split run has no
/// kernel tape and no prepared SG8 operands.
struct CompleteRun {
    out: MxArray,
    prepared_sg8: Option<PreparedSg8>,
    state: MxArray,
    history: MxArray,
    tape: Option<GdnKernelTape>,
    raw: Option<MxArray>,
    captured: Vec<MxArray>,
}

impl GatedDeltaNet {
    pub(crate) fn clear_cuda_dense_prefill(&self) {
        self.in_proj_qkvz.clear_cuda_dense_prefill();
        self.in_proj_ba.clear_cuda_dense_prefill();
        self.out_proj.clear_cuda_dense_prefill();
        if let Some((qkv, z)) = &self.split_in_proj_qkv_z {
            qkv.clear_cuda_dense_prefill();
            z.clear_cuda_dense_prefill();
        }
        if let Some((b, a)) = &self.split_in_proj_b_a {
            b.clear_cuda_dense_prefill();
            a.clear_cuda_dense_prefill();
        }
        if let Some(merged) = &self.split_in_proj_b_a_m8 {
            merged.clear_cuda_dense_prefill();
        }
        if let Some(merged) = &self.in_proj_qkvz_ba_q {
            merged.clear_cuda_dense_prefill();
        }
    }

    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_k_heads = config.linear_num_key_heads;
        let num_v_heads = config.linear_num_value_heads;
        let key_head_dim = config.linear_key_head_dim;
        let value_head_dim = config.linear_value_head_dim;
        let conv_kernel_dim = config.linear_conv_kernel_dim;

        let key_dim = num_k_heads * key_head_dim;
        let value_dim = num_v_heads * value_head_dim;
        // conv_dim = q + k + v channels (NOT key_dim + value_dim)
        let conv_dim = key_dim * 2 + value_dim;

        // Combined projection for q, k, v, z
        // Output: key_dim (q) + key_dim (k) + value_dim (v) + value_dim (z)
        let in_proj_qkvz = Linear::new(
            hidden_size as u32,
            (key_dim * 2 + value_dim * 2) as u32,
            Some(false),
        )?;

        // Combined projection for b and a
        let in_proj_ba = Linear::new(hidden_size as u32, (num_v_heads * 2) as u32, Some(false))?;

        // Depthwise conv1d: groups = conv_dim (each channel has its own filter)
        let conv1d = Conv1d::new(
            conv_dim as u32, // in_channels
            conv_dim as u32, // out_channels
            conv_kernel_dim as u32,
            Some(1),               // stride
            Some(0),               // padding (no padding, we prepend conv_state manually)
            Some(1),               // dilation
            Some(conv_dim as u32), // groups = depthwise
            Some(false),           // no bias
        )?;

        // Norm operates per-head: weight dim = value_head_dim (NOT value_dim)
        let norm = RMSNormGated::new(value_head_dim as u32, Some(config.rms_norm_eps))?;
        let out_proj = Linear::new(value_dim as u32, hidden_size as u32, Some(false))?;

        // Learnable parameters
        let dt_bias = MxArray::ones(&[num_v_heads as i64], None)?;
        let a_log = MxArray::zeros(&[num_v_heads as i64], None)?; // Will be loaded from weights

        // Constant per-channel norm weights folding the q/k head-dim scale
        // factors into `fast_rms_norm` — bf16 to match the family compute
        // dtype (mul_scalar also rounds the scalar to bf16, so the folded
        // constant keeps identical rounding; the fused op just drops the
        // intermediate round between norm and scale).
        let inv_scale = (key_head_dim as f64).powf(-0.5);
        let qk_norm_w_q = MxArray::full(
            &[key_head_dim as i64],
            Either::A(inv_scale * inv_scale),
            Some(crate::array::DType::BFloat16),
        )?;
        let qk_norm_w_k = MxArray::full(
            &[key_head_dim as i64],
            Either::A(inv_scale),
            Some(crate::array::DType::BFloat16),
        )?;

        Ok(Self {
            in_proj_qkvz: LinearProj::Standard(in_proj_qkvz),
            in_proj_ba: LinearProj::Standard(in_proj_ba),
            split_in_proj_qkv_z: None,
            split_in_proj_b_a: None,
            split_in_proj_b_a_m8: None,
            conv1d,
            norm,
            out_proj: LinearProj::Standard(out_proj),
            dt_bias,
            a_log,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_dim,
            tiled_gguf_layout: config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled"),
            in_proj_qkvz_ba_t: None,
            in_proj_qkvz_ba_q: None,
            conv1d_w4_f32: None,
            qk_norm_w_q,
            qk_norm_w_k,
            gdn_scale_f32: None,
            dt_bias_f32: None,
        })
    }

    /// Runs once after every weight of the layer is loaded. Safe to call
    /// repeatedly (idempotent).
    ///
    /// Evaluates the constants the setters derive from the weights: the loader
    /// evaluates only checkpoint tensors, and a compiled verify traced while
    /// one of these is lazy bakes its producer ops into a tape that is
    /// replayed, recomputing them, for the whole process.
    ///
    /// Precomputes the stacked `[qkvz; ba]` input projection, so forward uses
    /// one matmul plus two axis-2 slices instead of two separate matmuls.
    ///
    /// Dense path: stacks the transposed weights into `in_proj_qkvz_ba_t`.
    /// Quantized path: row-merges the packed weights into `in_proj_qkvz_ba_q`
    /// when both formats are mergeable (`LinearProj::concat_rows`), and swaps
    /// `in_proj_qkvz`/`in_proj_ba` for zero-copy row-slice views so getters
    /// keep working and no duplicate storage is retained. Incompatible pairs
    /// (mixed modes, split in_proj variants, special layouts) keep the
    /// unfused two-matmul path.
    ///
    /// A Tiled64 K-quant `in_proj_qkvz` (`QuantizedLinear::tile_kquant_layout`)
    /// only merges with a tiled `in_proj_ba`, whose `2 * num_v_heads` rows (96
    /// on Qwen3.8) are not whole tiles: it is zero-padded to the next tile
    /// (128) and tiled here first. The padded rows decode to exactly-zero
    /// output columns that `forward` drops (`split_ba_padded`), so the merge
    /// stays one dispatch per layer with the numerics of the unpadded pair.
    pub fn finalize_after_load(&mut self) -> Result<()> {
        let mut derived = vec![&self.a_log, &self.qk_norm_w_q, &self.qk_norm_w_k];
        derived.extend(
            [&self.gdn_scale_f32, &self.dt_bias_f32, &self.conv1d_w4_f32]
                .into_iter()
                .flatten(),
        );
        MxArray::eval_arrays(&derived)?;
        self.split_in_proj_b_a_m8 = None;
        if let Some((b, a)) = self.split_in_proj_b_a.as_ref()
            && !std::env::var("MLX_GDN_BA_FUSION").is_ok_and(|v| v == "0")
            && let Some(merged) = merge_split_ba_m8(b, a)?
        {
            let b_key = b.amax_key().map(str::to_owned);
            let a_key = a.amax_key().map(str::to_owned);
            // Retain separate projections as views, so the resident bytes do
            // not grow and every non-M8 forward keeps its original kernel.
            self.split_in_proj_b_a = Some((
                merged.slice_rows(0, 32)?.with_amax_key(b_key),
                merged.slice_rows(32, 64)?.with_amax_key(a_key),
            ));
            self.split_in_proj_b_a_m8 = Some(merged);
        }
        if self.split_in_proj_qkv_z.is_some() || self.split_in_proj_b_a.is_some() {
            self.in_proj_qkvz_ba_t = None;
            self.in_proj_qkvz_ba_q = None;
            return Ok(());
        }
        match (&self.in_proj_qkvz, &self.in_proj_ba) {
            (LinearProj::Standard(_), LinearProj::Standard(_)) => {}
            (LinearProj::Quantized(_), LinearProj::Quantized(_)) => {
                if self.in_proj_qkvz_ba_q.is_none() && self.ba_pads_to_tiled_qkvz() {
                    self.in_proj_ba.tile_kquant_layout_padded()?;
                }
                if self.in_proj_qkvz_ba_q.is_none()
                    && let Some(merged) = self.in_proj_qkvz.concat_rows(&self.in_proj_ba)?
                {
                    let qkvz_rows = self.in_proj_qkvz.packed_out_features()?;
                    let ba_rows = self.in_proj_ba.packed_out_features()?;
                    // Preserve calibration keys for individual projection views.
                    let qkvz_key = self.in_proj_qkvz.amax_key().map(str::to_owned);
                    let ba_key = self.in_proj_ba.amax_key().map(str::to_owned);
                    self.in_proj_qkvz = merged.slice_rows(0, qkvz_rows)?.with_amax_key(qkvz_key);
                    self.in_proj_ba = merged
                        .slice_rows(qkvz_rows, qkvz_rows + ba_rows)?
                        .with_amax_key(ba_key);
                    self.in_proj_qkvz_ba_q = Some(merged);
                }
                return Ok(());
            }
            _ => return Ok(()),
        }
        let w_qkvz = self.in_proj_qkvz.get_weight(); // [qkvz_dim, hidden]
        let w_ba = self.in_proj_ba.get_weight(); // [ba_dim, hidden]
        let stacked = MxArray::concatenate(&w_qkvz, &w_ba, 0)?; // [qkvz_dim+ba_dim, hidden]
        let stacked_t = stacked.transpose(Some(&[1, 0]))?; // [hidden, qkvz_dim+ba_dim]
        stacked_t.eval();
        self.in_proj_qkvz_ba_t = Some(stacked_t);
        Ok(())
    }

    /// Whether `in_proj_ba` should be zero-padded to whole tiles and tiled so
    /// it can merge with `in_proj_qkvz`: both quantized in the same contract
    /// (the same K-quant mode, or MLX affine at the (bits, group) the tiled
    /// qkvz reads through: `affine` 4/64 pads into `a4g64@t64`), qkvz already
    /// Tiled64, ba still row-major.
    fn ba_pads_to_tiled_qkvz(&self) -> bool {
        use crate::models::quant_dispatch::{kquant_affine_mode_params, split_kquant_layout};
        match (&self.in_proj_qkvz, &self.in_proj_ba) {
            (LinearProj::Quantized(qkvz), LinearProj::Quantized(ba)) => {
                let base = split_kquant_layout(qkvz.mode()).0;
                let ba_contract = if ba.mode() == DEFAULT_QUANT_MODE {
                    kquant_affine_mode_params(ba.bits(), ba.group_size())
                        .map_or(ba.mode(), |kq| kq.mode_str)
                } else {
                    ba.mode()
                };
                qkvz.is_kquant_tiled() && !ba.is_kquant_tiled() && base == ba_contract
            }
            _ => false,
        }
    }

    /// `[B, T, >= 2 * num_v_heads]` -> `(b, a)`, each `[B, T, num_v_heads]`.
    /// Columns past `2 * num_v_heads` are the exactly-zero outputs of the
    /// Tiled64 row padding of `in_proj_ba` (see `finalize_after_load`) and
    /// are dropped; an unpadded projection keeps the plain two-way split.
    fn split_ba_padded(&self, ba: &MxArray) -> Result<(MxArray, MxArray)> {
        let nv = self.num_v_heads as i64;
        let parts = if ba.shape_at(2)? == 2 * nv {
            ba.split_sections(&[nv], 2)?
        } else {
            ba.split_sections(&[nv, 2 * nv], 2)?
        };
        Ok((parts[0].clone(), parts[1].clone()))
    }

    /// Forward pass for GatedDeltaNet.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, hidden_size]
    /// * `mask` - Optional mask [B, T]
    /// * `cache` - Optional ArraysCache with 2 slots: [conv_state, recurrent_state]
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_with_tape(x, mask, cache, use_kernel, None)
    }

    /// Tape-recording variant of [`GatedDeltaNet::forward`].
    ///
    /// When `tape_sink` is `Some`, records the post-mask pre-conv `qkv` plus
    /// the per-step kernel inputs into a [`GdnLayerTape`] for the eager MTP
    /// rollback replay. When `None`, behavior is byte-identical to
    /// [`GatedDeltaNet::forward`]. All recording is by lazy `.clone()` (no
    /// eval), so it stays inside the fused MLX graph.
    pub(crate) fn is_cuda_deferred_eligible(&self) -> bool {
        self.key_head_dim == 128
            && self.value_head_dim == 128
            && self.num_k_heads > 0
            && self.num_v_heads > 0
            && self.num_v_heads % self.num_k_heads == 0
            && self.norm.weight().dtype().ok() == Some(DType::BFloat16)
    }

    pub(crate) fn forward_with_tape(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<GdnLayerTape>>,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x, mask, None, cache, use_kernel, tape_sink, None, None, None, None,
        )
    }

    /// CUDA deferred-fold variant of [`Self::forward_with_tape`].
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn forward_with_tape_deferred(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x,
            mask,
            None,
            cache,
            use_kernel,
            tape_sink,
            None,
            None,
            cuda_deferred,
            None,
        )
    }

    /// Tree-verification + CUDA deferred-fold variant of [`Self::forward_with_tape`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_tape_deferred_tree(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
        tree: Option<&super::verify_tree::TreeDeviceViews>,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x,
            mask,
            None,
            cache,
            use_kernel,
            tape_sink,
            None,
            None,
            cuda_deferred,
            tree,
        )
    }

    /// Batched forward pass supporting right-padded sequences with explicit
    /// per-row valid lengths so the cached `conv_state` captures each row's
    /// true last `conv_kernel_dim - 1` pre-conv tokens rather than trailing
    /// zero-padded positions.
    ///
    /// * `x` - Input `[B, T, hidden_size]`
    /// * `mask` - Optional boolean mask `[B, T]` (`true` for valid tokens,
    ///   `false` for right-padded positions)
    /// * `valid_lengths` - Optional per-row valid token counts of length `B`
    ///   (`0 <= lengths[b] <= T`). When `Some` and `cache` is `Some`, row `b`'s
    ///   cached `conv_state` is taken from `conv_input[b, lengths[b] .. lengths[b] + (K - 1), :]`.
    /// * `cache` - Optional `ArraysCache` (`[conv_state, recurrent_state]`)
    #[allow(dead_code)]
    pub(crate) fn forward_batched_masked(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        valid_lengths: Option<&[i32]>,
        cache: Option<&mut ArraysCache>,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x,
            mask,
            valid_lengths,
            cache,
            true,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// The unmasked, untaped prefill forward whose output projection adds
    /// `residual` in its GEMM store where the shape takes the epilogue GEMM
    /// ([`Self::out_proj_store_route`]); residual-inclusive output, the bits
    /// of `residual + forward(x)`.
    pub(crate) fn forward_store_residual(
        &self,
        x: &MxArray,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        residual: &MxArray,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x,
            None,
            None,
            cache,
            use_kernel,
            None,
            Some(MixerResidual::Store(residual)),
            None,
            None,
            None,
        )
    }

    /// The taped forward of a DFlash2 verify's rows whose output projection
    /// adds `residual` in its planes `_r` store
    /// ([`Self::out_proj_verify_route`]); residual-inclusive output, same
    /// cache and tape as [`Self::forward_with_tape`].
    pub(crate) fn forward_verify_residual_with_tape(
        &self,
        x: &MxArray,
        cache: Option<&mut ArraysCache>,
        tape_sink: Option<&mut Option<GdnLayerTape>>,
        residual: &MxArray,
    ) -> Result<MxArray> {
        self.forward_with_tape_impl(
            x,
            None,
            None,
            cache,
            true,
            tape_sink,
            Some(MixerResidual::Verify(residual)),
            None,
            None,
            None,
        )
    }

    /// Whether a DFlash2 verify's `rows` rows reach the output projection's
    /// planes.
    pub(crate) fn out_proj_verify_route(&self, rows: i64) -> bool {
        self.out_proj.verify_residual_route(rows)
    }

    /// Whether the output projection of `rows` rows takes the store-epilogue
    /// GEMM.
    pub(crate) fn out_proj_store_route(&self, rows: i64) -> Result<bool> {
        self.out_proj
            .store_epilogue_route(rows, self.value_dim as i64)
    }

    /// Unmasked batch-1 prefill forward (no tape) that also captures the
    /// layer state at every row of `plan`. Output and cache are bit-identical
    /// to the uncaptured forward: each route runs the entry the whole window
    /// takes on the split segments, carrying the f32 state between them.
    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) fn forward_captured(
        &self,
        x: &MxArray,
        cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        residual: Option<MixerResidual<'_>>,
        plan: &GdnCapturePlan<'_>,
    ) -> Result<(MxArray, Vec<GdnLayerCapture>)> {
        let mut captured = Vec::with_capacity(plan.rows.len());
        let out = self.forward_with_tape_impl(
            x,
            None,
            None,
            cache,
            use_kernel,
            None,
            residual,
            Some((plan, &mut captured)),
            None,
            None,
        )?;
        Ok((out, captured))
    }

    /// The conv history after chunk row `c`: the raw conv-input rows
    /// `[c - (K-1), c)`, reaching into the incoming history when `c < K-1`.
    /// This is what the conv slot holds after a forward of rows `[0, c)`.
    fn conv_history_at(
        &self,
        qkv: &MxArray,
        incoming: Option<&MxArray>,
        c: usize,
    ) -> Result<MxArray> {
        let keep = (self.conv_kernel_dim - 1) as i64;
        let c = c as i64;
        let rows = if c >= keep {
            qkv.slice_axis(1, c - keep, c)?
        } else {
            let prior = match incoming {
                Some(history) => history.clone(),
                None => MxArray::zeros(&[1, keep, self.conv_dim as i64], Some(qkv.dtype()?))?,
            };
            MxArray::concatenate(&prior.slice_axis(1, c, keep)?, &qkv.slice_axis(1, 0, c)?, 1)?
        };
        rows.deep_copy()
    }

    /// Hands `capture` one [`GdnLayerCapture`] per plan row from the states
    /// the split recurrence carried at those rows.
    fn record_captures(
        &self,
        capture: Option<(&GdnCapturePlan<'_>, &mut Vec<GdnLayerCapture>)>,
        qkv: &MxArray,
        incoming: Option<&MxArray>,
        states: Vec<MxArray>,
    ) -> Result<()> {
        let Some((plan, out)) = capture else {
            return Ok(());
        };
        if states.len() != plan.rows.len() {
            return Err(Error::from_reason(format!(
                "GDN capture: {} states for {} rows",
                states.len(),
                plan.rows.len()
            )));
        }
        for (&c, state) in plan.rows.iter().zip(states) {
            out.push(GdnLayerCapture {
                conv: self.conv_history_at(qkv, incoming, c)?,
                rec: state.deep_copy()?,
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_tape_impl(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        valid_lengths: Option<&[i32]>,
        mut cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        mut tape_sink: Option<&mut Option<GdnLayerTape>>,
        residual: Option<MixerResidual<'_>>,
        capture: Option<(&GdnCapturePlan<'_>, &mut Vec<GdnLayerCapture>)>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
        tree: Option<&super::verify_tree::TreeDeviceViews>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        if let Some(tv) = tree {
            if batch != 1 {
                return Err(Error::from_reason(format!(
                    "GatedDeltaNet tree forward requires batch == 1, got {batch}"
                )));
            }
            if seq_len as usize != tv.tree_len {
                return Err(Error::from_reason(format!(
                    "GatedDeltaNet tree forward seq_len {seq_len} != tree_len {}",
                    tv.tree_len
                )));
            }
            if mask.is_some() || valid_lengths.is_some() {
                return Err(Error::from_reason(
                    "GatedDeltaNet tree forward does not support an explicit padding mask or valid_lengths",
                ));
            }
        }
        if let Some(lengths) = valid_lengths {
            if lengths.len() != batch as usize {
                return Err(Error::from_reason(format!(
                    "GatedDeltaNet valid_lengths length {} does not match batch {}",
                    lengths.len(),
                    batch
                )));
            }
            for (row, &len) in lengths.iter().enumerate() {
                if len < 0 || i64::from(len) > seq_len {
                    return Err(Error::from_reason(format!(
                        "GatedDeltaNet valid_lengths[{row}]={len} out of range 0..={seq_len}"
                    )));
                }
            }
        }
        let rows = capture
            .as_ref()
            .map(|(plan, _)| plan.rows)
            .filter(|rows| !rows.is_empty());
        if rows.is_some()
            && (batch != 1
                || mask.is_some()
                || valid_lengths.is_some()
                || tape_sink.is_some()
                || tree.is_some()
                || cache.as_deref().and_then(|c| c.get(4)).is_some())
        {
            return Err(Error::from_reason(
                "GDN state capture needs an unmasked batch-1 window with no tape or deferred state",
            ));
        }

        let (qkv, z, b, a) = self.in_projections(x)?;

        // Apply mask before conv to prevent masked values leaking through convolution
        let qkv = if let Some(m) = mask {
            // m: [B, T] → [B, T, 1] for broadcasting
            let m_3d = m.reshape(&[batch, seq_len, 1])?;
            // Use qkv's dtype to avoid f32 promotion for bf16/f16 models
            m_3d.where_(&qkv, &MxArray::zeros(&[1], Some(qkv.dtype()?))?)?
        } else {
            qkv
        };

        // Record the post-mask, pre-conv `qkv` for the eager MTP tape (lazy
        // clone, no eval). The conv-state rebuild on accept slices the accepted
        // prefix of this exact tensor.
        let tape_qkv = tape_sink.as_ref().map(|_| qkv.clone());

        // Handle conv_state: always prepend padding (zeros or cached state)
        let conv_state = if let Some(ref cache) = cache {
            cache.get(0).cloned()
        } else {
            None
        };
        let incoming_conv = rows.and(conv_state.clone());

        // Prompt-window fusion: one primitive for the whole GDN core of an
        // unmasked batch-1 `T >= 64` window — bit-identical to the
        // `prep_via_conv` -> `gated_delta_update` -> tail chain it replaces.
        // Decode/verify (`T < 64`) and the tape path never see it.
        let prefill_prep = if tape_sink.is_none()
            && cuda_deferred.is_none()
            && tree.is_none()
            && valid_lengths.is_none()
        {
            self.prefill_fused(
                &qkv,
                &z,
                &a,
                &b,
                conv_state.as_ref(),
                cache.as_deref_mut(),
                mask,
                use_kernel,
                batch,
                seq_len,
                rows,
            )?
        } else {
            None
        };
        if let Some(PrefillPrep::Fused {
            gated,
            state,
            captured,
        }) = prefill_prep
        {
            if let Some(cache) = cache {
                cache.set(1, state)?;
            }
            #[cfg(test)]
            note_route(GdnRoute::FusedPrefill);
            self.record_captures(capture, &qkv, incoming_conv.as_ref(), captured)?;
            return self.out_proj.forward_mixer_output(&gated, None, residual);
        }

        // Whole GDN core in ONE dispatch (prep + recurrence + gated norm +
        // z gate): the decode/verify geometry with compact tiled heads. Same
        // bits as the gdn_prepare -> fused step chain below, which stays the
        // fallback (as does the generic path after it).
        if cuda_deferred.is_none()
            && tree.is_none()
            && valid_lengths.is_none()
            && let Some(complete) = self.forward_complete(
                &qkv,
                &z,
                &a,
                &b,
                conv_state.as_ref(),
                cache.as_deref(),
                mask,
                use_kernel,
                batch,
                seq_len,
                true,
                rows,
            )?
        {
            if let Some(cache) = cache {
                cache.set(
                    0,
                    complete.history.reshape(&[
                        1,
                        (self.conv_kernel_dim - 1) as i64,
                        self.conv_dim as i64,
                    ])?,
                )?;
                cache.set(1, complete.state)?;
            }
            #[cfg(test)]
            note_route(GdnRoute::Complete);
            self.record_captures(capture, &qkv, incoming_conv.as_ref(), complete.captured)?;
            if let (Some(sink), Some(qkv), Some(kernel)) =
                (tape_sink.take(), complete.raw.or(tape_qkv), complete.tape)
            {
                *sink = Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                });
            }
            return self.out_proj.forward_mixer_output(
                &complete.out,
                complete.prepared_sg8.as_ref(),
                residual,
            );
        }

        // A deferred cache holds a state behind its logical frontier by the
        // pending rows; no other route folds them. Fail closed (Splash
        // `compiledGdn`).
        if cache.as_deref().and_then(|c| c.get(4)).is_some() {
            return Err(Error::from_reason("deferred GDN complete kernel declined"));
        }

        // Fully-fused prep: conv + SiLU + q|k|v split + q/k L2-norm + decay/beta
        // gating in ONE GPU dispatch (`kernels::gdn::prepare` or `prepare_tree`
        // — Metal or CUDA, hardcoded to this family's 10240-wide 16k/48v×128 geometry).
        let prepared = (|| -> Option<[MxArray; 6]> {
            let max_prepare_seq = if crate::accelerator::cuda_ops_available() {
                4097
            } else {
                64
            };
            if batch != 1
                || seq_len >= max_prepare_seq
                || valid_lengths.is_some()
                || self.conv_kernel_dim != 4
                || self.conv_dim != 10240
                || self.num_k_heads != 16
                || self.key_head_dim != 128
                || self.num_v_heads != 48
                || self.value_head_dim != 128
                || !use_kernel
            {
                return None;
            }
            let conv_w = self.conv1d_w4_f32.as_ref()?;
            let scale = self.gdn_scale_f32.as_ref()?;
            let dt = self.dt_bias_f32.as_ref()?;
            if qkv.dtype().ok()? != crate::array::DType::BFloat16
                || a.dtype().ok()? == crate::array::DType::Float16
                || b.dtype().ok()? == crate::array::DType::Float16
            {
                return None;
            }
            let history = match &conv_state {
                Some(s) => s.squeeze(Some(&[0])).ok()?,
                None => MxArray::zeros(
                    &[(self.conv_kernel_dim - 1) as i64, self.conv_dim as i64],
                    Some(crate::array::DType::BFloat16),
                )
                .ok()?,
            };
            // qwen3_5 semantics: mean-eps norm (rms_norm eps inside the
            // mean) and bf16 beta — bit-faithful to the conv fallback.
            if let Some(tv) = tree {
                crate::kernels::gdn::prepare_tree(
                    &qkv,
                    &a,
                    &b,
                    conv_w,
                    &history,
                    scale,
                    dt,
                    &tv.conv_taps_i32,
                    true,
                    true,
                )
                .ok()?
            } else {
                crate::kernels::gdn::prepare(&qkv, &a, &b, conv_w, &history, scale, dt, true, true)
                    .ok()?
            }
        })();

        let (q, k, v, precomputed) = if let Some([pq, pk, pv, decay, beta, history]) = prepared {
            if let Some(cache) = cache.as_deref_mut() {
                // Same [K-1, W] window tail the fused/fallback conv paths store.
                cache.set(
                    0,
                    history.reshape(&[
                        1,
                        (self.conv_kernel_dim - 1) as i64,
                        self.conv_dim as i64,
                    ])?,
                )?;
            }
            (pq, pk, pv, Some((decay, beta)))
        } else if let Some(tv) = tree {
            let (q, k, v) = self.prep_via_conv_tree(
                &qkv,
                batch,
                seq_len,
                cache.as_deref_mut(),
                conv_state,
                &tv.conv_taps_i32,
            )?;
            (q, k, v, None)
        } else {
            let (q, k, v) = match prefill_prep {
                // The conv already ran inside `prefill_fused`; only the q/k
                // norms are left of the prep stage.
                Some(PrefillPrep::Raw { q, k, v }) => {
                    let (q, k) = self.norm_qk(&q, &k)?;
                    (q, k, v)
                }
                _ => self.prep_via_conv(
                    &qkv,
                    batch,
                    seq_len,
                    valid_lengths,
                    cache.as_deref_mut(),
                    conv_state,
                    use_kernel,
                )?,
            };
            (q, k, v, None)
        };
        if self.tiled_gguf_layout
            && (self.num_k_heads <= 0 || self.num_v_heads % self.num_k_heads != 0)
        {
            return Err(Error::from_reason(format!(
                "Qwen3.5 tiled GGUF GDN layout requires Hv ({}) to be divisible by Hk ({})",
                self.num_v_heads, self.num_k_heads
            )));
        }

        // Run gated delta recurrence
        let recurrent_state = cache.as_deref().and_then(|c| c.get(1));
        let seq = seq_len as usize;

        // CUDA deferred-fold verify step (linear or tree): folds the prior
        // cycle's pending rows into `state_base` before stepping the current
        // window/tree, storing `S_folded` back into `cache[1]` and recording
        // this cycle's tape. Fails closed if the deferred kernel declines so no
        // non-deferred path can silently advance `cache[1]` past `S_folded`.
        if let Some(pending) = cuda_deferred {
            if rows.is_some() || mask.is_some() || !crate::accelerator::cuda_ops_available() {
                return Err(Error::from_reason(
                    "deferred CUDA GDN step requires unmasked CUDA forward without capture splits",
                ));
            }
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let stepped = if let Some(tv) = tree {
                gated_delta_update_tree_deferred(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    recurrent_state,
                    pending,
                    &tv.parent_i32,
                    mask,
                    use_kernel,
                    self.tiled_gguf_layout,
                    precomputed.as_ref().map(|(d, b)| (d, b)),
                    GdnTail {
                        z: &z,
                        norm_weight: self.norm.weight(),
                        eps: self.norm.eps(),
                        state_dst: None,
                    },
                    tape_sink.is_some().then_some(&mut kernel_sink),
                )?
            } else {
                gated_delta_update_fused_deferred(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    recurrent_state,
                    pending,
                    mask,
                    use_kernel,
                    self.tiled_gguf_layout,
                    precomputed.as_ref().map(|(d, b)| (d, b)),
                    GdnTail {
                        z: &z,
                        norm_weight: self.norm.weight(),
                        eps: self.norm.eps(),
                        state_dst: None,
                    },
                    tape_sink.is_some().then_some(&mut kernel_sink),
                )?
            };
            let Some((gated, committed_state)) = stepped else {
                return Err(Error::from_reason("deferred CUDA GDN step kernel declined"));
            };
            if let Some(cache) = cache {
                cache.set(1, committed_state)?;
            }
            #[cfg(test)]
            note_route(GdnRoute::FusedTail);
            if let (Some(sink), Some(kernel), Some(qkv)) = (tape_sink.take(), kernel_sink, tape_qkv)
            {
                *sink = Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                });
            }
            return self.out_proj.forward_mixer_output(&gated, None, residual);
        }

        // Non-deferred tree fallback (e.g. Metal/CPU or unit-test geometries).
        if let Some(tv) = tree {
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let (y, new_state) = gated_delta_update_tree_with_tape(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                recurrent_state,
                &tv.parent_i32,
                self.tiled_gguf_layout,
                precomputed.as_ref().map(|(d, b)| (d, b)),
                tape_sink.is_some().then_some(&mut kernel_sink),
            )?;
            if let Some(cache) = cache {
                cache.set(1, new_state)?;
            }
            #[cfg(test)]
            note_route(GdnRoute::Chain);
            if let (Some(sink), Some(kernel), Some(qkv)) = (tape_sink.take(), kernel_sink, tape_qkv)
            {
                *sink = Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                });
            }
            let z = z.reshape(&[
                batch,
                seq_len,
                self.num_v_heads as i64,
                self.value_head_dim as i64,
            ])?;
            let y_normed = self.norm.forward(&y, Some(&z))?;
            let y_flat = y_normed.reshape(&[batch, seq_len, self.value_dim as i64])?;
            return self.out_proj.forward_mixer_output(&y_flat, None, residual);
        }

        // Fused tail: recurrence + gated norm + z gate in one dispatch (the
        // production decode/verify geometry). Slot 2 of the cache, when a
        // caller provides it, is the buffer the new state is written into.
        {
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let fused = match rows {
                None => gated_delta_update_fused(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    recurrent_state,
                    mask,
                    use_kernel,
                    self.tiled_gguf_layout,
                    precomputed.as_ref().map(|(d, b)| (d, b)),
                    GdnTail {
                        z: &z,
                        norm_weight: self.norm.weight(),
                        eps: self.norm.eps(),
                        state_dst: cache.as_deref().and_then(|c| c.get(2)),
                    },
                    tape_sink.is_some().then_some(&mut kernel_sink),
                )?
                .map(|out| (out, Vec::new())),
                // The whole window would be off the fused tail's T <= 32
                // contract, so its segments must stay on the chain too.
                Some(_) if seq > 32 => None,
                Some(rows) => {
                    // The first segment meets every contract the whole window
                    // does except its length, so its decline is the window's.
                    let mut declined = false;
                    let split = gated_delta_split_at(
                        seq,
                        rows,
                        GdnCarry {
                            state: recurrent_state.cloned(),
                            history: None,
                        },
                        |s, e, carry| {
                            let r = |x: &MxArray| window_rows(x, s, e);
                            let gates = match &precomputed {
                                Some((d, b)) => Some((r(d)?, r(b)?)),
                                None => None,
                            };
                            let z = r(&z)?;
                            let out = gated_delta_update_fused(
                                &r(&q)?,
                                &r(&k)?,
                                &r(&v)?,
                                &r(&a)?,
                                &r(&b)?,
                                &self.a_log,
                                &self.dt_bias,
                                carry.state.as_ref(),
                                None,
                                use_kernel,
                                self.tiled_gguf_layout,
                                gates.as_ref().map(|(d, b)| (d, b)),
                                GdnTail {
                                    z: &z,
                                    norm_weight: self.norm.weight(),
                                    eps: self.norm.eps(),
                                    state_dst: (e == seq)
                                        .then(|| cache.as_deref().and_then(|c| c.get(2)))
                                        .flatten(),
                                },
                                None,
                            )?;
                            match out {
                                Some((gated, state)) => Ok((
                                    gated,
                                    GdnCarry {
                                        state: Some(state),
                                        history: None,
                                    },
                                )),
                                None => {
                                    declined = s == 0;
                                    Err(Error::from_reason(
                                        "fused GDN tail declined a capture segment",
                                    ))
                                }
                            }
                        },
                    );
                    match split {
                        Ok(split) => {
                            let state = split.carry.state.ok_or_else(|| {
                                Error::from_reason("GDN capture: the split carried no state")
                            })?;
                            Some(((split.y, state), captured_states(split.captured)?))
                        }
                        Err(_) if declined => None,
                        Err(error) => return Err(error),
                    }
                }
            };
            if let Some(((gated, new_state), captured)) = fused {
                if let Some(cache) = cache {
                    cache.set(1, new_state)?;
                }
                #[cfg(test)]
                note_route(GdnRoute::FusedTail);
                self.record_captures(capture, &qkv, incoming_conv.as_ref(), captured)?;
                if let (Some(sink), Some(kernel), Some(qkv)) =
                    (tape_sink.take(), kernel_sink, tape_qkv)
                {
                    *sink = Some(GdnLayerTape {
                        kernel,
                        qkv,
                        conv_kernel_dim: self.conv_kernel_dim,
                    });
                }
                return self.out_proj.forward_mixer_output(&gated, None, residual);
            }
        }

        let (y, new_state) = if let Some(rows) = rows {
            // `gated_delta_update` is this call with untiled heads.
            let split = gated_delta_split_at(
                seq,
                rows,
                GdnCarry {
                    state: recurrent_state.cloned(),
                    history: None,
                },
                |s, e, carry| {
                    let r = |x: &MxArray| window_rows(x, s, e);
                    let gates = match &precomputed {
                        Some((d, b)) => Some((r(d)?, r(b)?)),
                        None => None,
                    };
                    let (y, state) = gated_delta_update_with_tape(
                        &r(&q)?,
                        &r(&k)?,
                        &r(&v)?,
                        &r(&a)?,
                        &r(&b)?,
                        &self.a_log,
                        &self.dt_bias,
                        carry.state.as_ref(),
                        None,
                        use_kernel,
                        self.tiled_gguf_layout,
                        gates.as_ref().map(|(d, b)| (d, b)),
                        None,
                    )?;
                    Ok((
                        y,
                        GdnCarry {
                            state: Some(state),
                            history: None,
                        },
                    ))
                },
            )?;
            let state = split
                .carry
                .state
                .ok_or_else(|| Error::from_reason("GDN capture: the split carried no state"))?;
            self.record_captures(
                capture,
                &qkv,
                incoming_conv.as_ref(),
                captured_states(split.captured)?,
            )?;
            (split.y, state)
        } else if tape_sink.is_some() {
            // Record the per-step kernel inputs into a local sink, then fold
            // them (plus the recorded qkv) into the layer tape below.
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let result = gated_delta_update_with_tape(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                recurrent_state,
                mask,
                use_kernel,
                self.tiled_gguf_layout,
                precomputed.as_ref().map(|(d, b)| (d, b)),
                Some(&mut kernel_sink),
            )?;
            if let (Some(sink), Some(kernel), Some(qkv)) = (tape_sink.take(), kernel_sink, tape_qkv)
            {
                *sink = Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                });
            }
            result
        } else {
            let mut start_state_opt = None;
            if mask.is_none()
                && batch == 1
                && use_kernel
                && crate::accelerator::cuda_ops_available()
                && self.key_head_dim == 128
                && self.value_head_dim == 128
                && let Some((decay, beta)) = precomputed.as_ref()
            {
                let start_state = match recurrent_state {
                    Some(s) => s.clone(),
                    None => MxArray::zeros(
                        &[
                            1,
                            self.num_v_heads as i64,
                            self.value_head_dim as i64,
                            self.key_head_dim as i64,
                        ],
                        Some(crate::array::DType::Float32),
                    )?,
                };
                if let Some((y_flat, new_state)) = crate::kernels::gdn::prefill_chain_gated(
                    &q,
                    &k,
                    &v,
                    decay,
                    beta,
                    &start_state,
                    &z,
                    self.norm.weight(),
                    self.norm.eps(),
                    self.tiled_gguf_layout,
                )? {
                    if let Some(cache) = cache {
                        cache.set(1, new_state)?;
                    }
                    #[cfg(test)]
                    note_route(GdnRoute::Chain);
                    return self.out_proj.forward_mixer_output(&y_flat, None, residual);
                }
                start_state_opt = Some(start_state);
            }
            let cuda_chain = if let (Some(start_state), Some((decay, beta))) =
                (start_state_opt.as_ref(), precomputed.as_ref())
            {
                crate::kernels::gdn::prefill_chain(
                    &q,
                    &k,
                    &v,
                    decay,
                    beta,
                    start_state,
                    self.tiled_gguf_layout,
                )?
            } else {
                None
            };
            if let Some(pair) = cuda_chain {
                pair
            } else if self.tiled_gguf_layout {
                gated_delta_update_with_tape(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    recurrent_state,
                    mask,
                    use_kernel,
                    true,
                    precomputed.as_ref().map(|(d, b)| (d, b)),
                    None,
                )?
            } else {
                gated_delta_update(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    recurrent_state,
                    mask,
                    use_kernel,
                    precomputed.as_ref().map(|(d, b)| (d, b)),
                )?
            }
        };

        // Update recurrent state in cache
        if let Some(cache) = cache {
            cache.set(1, new_state)?;
        }
        #[cfg(test)]
        note_route(GdnRoute::Chain);

        // Reshape z to per-head format: [B, T, value_dim] → [B, T, Hv, Dv]
        let z = z.reshape(&[
            batch,
            seq_len,
            self.num_v_heads as i64,
            self.value_head_dim as i64,
        ])?;

        // Apply RMSNormGated on per-head tensors: [B, T, Hv, Dv]
        // Norm weight is [Dv], operates on last dimension
        let y_normed = self.norm.forward(&y, Some(&z))?;

        // Flatten heads: [B, T, Hv, Dv] → [B, T, value_dim]
        let y_flat = y_normed.reshape(&[batch, seq_len, self.value_dim as i64])?;

        // Output projection
        self.out_proj.forward_mixer_output(&y_flat, None, residual)
    }

    /// `x` `[B, T, H]` -> `(qkv, z, b, a)`, the layer's input projections.
    fn in_projections(&self, x: &MxArray) -> Result<(MxArray, MxArray, MxArray, MxArray)> {
        // When the stacked weight is available, do one matmul + one split.
        let qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) as i64;
        let stacked = if let Some(wqb_t) = &self.in_proj_qkvz_ba_t {
            let combined = x.matmul(wqb_t)?; // [B, T, qkvz_dim + ba_dim]
            let parts = combined.split_sections(&[qkvz_dim], 2)?;
            Some((parts[0].clone(), parts[1].clone()))
        } else if let Some(merged) = &self.in_proj_qkvz_ba_q {
            let combined = merged.forward(x)?; // [B, T, qkvz_dim + ba_dim]
            let parts = combined.split_sections(&[qkvz_dim], 2)?;
            Some((parts[0].clone(), parts[1].clone()))
        } else if self.split_in_proj_qkv_z.is_none() && self.split_in_proj_b_a.is_none() {
            crate::models::quantized_linear::mixed_projection_forward(
                x,
                &[&self.in_proj_qkvz, &self.in_proj_ba],
            )?
            .map(|outputs| (outputs[0].clone(), outputs[1].clone()))
        } else {
            None
        };

        let (qkv, z, joined_ba) = if let Some((qkv_proj, z_proj)) = &self.split_in_proj_qkv_z {
            let joined = if stacked.is_none() && self.split_in_proj_b_a.is_none() {
                crate::models::quantized_linear::mixed_projection_forward(
                    x,
                    &[qkv_proj, z_proj, &self.in_proj_ba],
                )?
            } else {
                None
            };
            match joined {
                Some(outputs) => (
                    outputs[0].clone(),
                    outputs[1].clone(),
                    Some(outputs[2].clone()),
                ),
                None => match crate::models::quantized_linear::mixed_projection_forward(
                    x,
                    &[qkv_proj, z_proj],
                )? {
                    Some(outputs) => (outputs[0].clone(), outputs[1].clone(), None),
                    None => (qkv_proj.forward(x)?, z_proj.forward(x)?, None),
                },
            }
        } else {
            let qkvz = if let Some((qkvz, _)) = &stacked {
                qkvz.clone()
            } else {
                self.in_proj_qkvz.forward(x)?
            };
            let qkvz_split = qkvz.split_sections(&[self.conv_dim as i64], 2)?;
            (qkvz_split[0].clone(), qkvz_split[1].clone(), None)
        };

        let (b, a) = if let Some((b_proj, a_proj)) = &self.split_in_proj_b_a {
            match project_split_ba_m8(x, self.split_in_proj_b_a_m8.as_ref())? {
                Some(pair) => pair,
                None => (b_proj.forward(x)?, a_proj.forward(x)?),
            }
        } else {
            let ba = if let Some(ba) = joined_ba {
                ba
            } else if let Some((_, ba)) = &stacked {
                ba.clone()
            } else {
                self.in_proj_ba.forward(x)?
            };
            self.split_ba_padded(&ba)?
        };
        Ok((qkv, z, b, a))
    }

    /// [`Self::in_projections`] of a wave `x` `[N, T, H]`: lane `i`'s rows
    /// of each output carry the bits `in_projections(x[i])` gives. Each
    /// projection runs once over the lanes (`forward_lanes`); the mixed
    /// dispatch of one lane keeps its standalone projections' bits, so its
    /// segments run as their own projections. The dense stacked weight and
    /// the merged M8 b|a pair (an 8-row contract) run per lane.
    pub(super) fn in_projections_lanes(
        &self,
        x: &MxArray,
    ) -> Result<(MxArray, MxArray, MxArray, MxArray)> {
        let lanes = x.shape_at(0)?;
        let per_lane = |f: &dyn Fn(&MxArray) -> Result<Vec<MxArray>>| -> Result<Vec<MxArray>> {
            let mut parts: Vec<Vec<MxArray>> = Vec::new();
            for lane in 0..lanes {
                let outs = f(&x.slice_axis(0, lane, lane + 1)?)?;
                parts.resize_with(outs.len(), Vec::new);
                for (part, out) in parts.iter_mut().zip(outs) {
                    part.push(out);
                }
            }
            parts
                .iter()
                .map(|part| MxArray::concatenate_many(part.iter().collect(), Some(0)))
                .collect()
        };
        if self.in_proj_qkvz_ba_t.is_some() {
            let parts = per_lane(&|x| {
                let (qkv, z, b, a) = self.in_projections(x)?;
                Ok(vec![qkv, z, b, a])
            })?;
            return Ok((
                parts[0].clone(),
                parts[1].clone(),
                parts[2].clone(),
                parts[3].clone(),
            ));
        }
        let qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) as i64;
        let merged = match &self.in_proj_qkvz_ba_q {
            Some(merged) => {
                let parts = merged.forward_lanes(x)?.split_sections(&[qkvz_dim], 2)?;
                Some((parts[0].clone(), parts[1].clone()))
            }
            None => None,
        };
        let (qkv, z) = if let Some((qkv_proj, z_proj)) = &self.split_in_proj_qkv_z {
            (qkv_proj.forward_lanes(x)?, z_proj.forward_lanes(x)?)
        } else {
            let qkvz = match &merged {
                Some((qkvz, _)) => qkvz.clone(),
                None => self.in_proj_qkvz.forward_lanes(x)?,
            };
            let parts = qkvz.split_sections(&[self.conv_dim as i64], 2)?;
            (parts[0].clone(), parts[1].clone())
        };
        let (b, a) = if let Some((b_proj, a_proj)) = &self.split_in_proj_b_a {
            if self.split_in_proj_b_a_m8.is_some() {
                let parts = per_lane(&|x| {
                    let (b, a) = match project_split_ba_m8(x, self.split_in_proj_b_a_m8.as_ref())? {
                        Some(pair) => pair,
                        None => (b_proj.forward(x)?, a_proj.forward(x)?),
                    };
                    Ok(vec![b, a])
                })?;
                (parts[0].clone(), parts[1].clone())
            } else {
                (b_proj.forward_lanes(x)?, a_proj.forward_lanes(x)?)
            }
        } else {
            let ba = match &merged {
                Some((_, ba)) => ba.clone(),
                None => self.in_proj_ba.forward_lanes(x)?,
            };
            self.split_ba_padded(&ba)?
        };
        Ok((qkv, z, b, a))
    }

    /// Why [`Self::forward_wave`] on `hidden`-wide lanes would not stream its
    /// in-projection weights once on this device: the complete kernel's
    /// static contract for a deferred cell fails, or the q|k|v(|z) projection
    /// misses the lane tier. `None` when it would.
    pub(crate) fn wave_unready(&self, hidden: i64) -> Result<Option<&'static str>> {
        let heads = self.tiled_gguf_layout
            || self.num_k_heads == self.num_v_heads
            || (self.num_k_heads > 0 && self.num_v_heads % self.num_k_heads == 0);
        if self.conv_kernel_dim != 4
            || self.key_head_dim != 128
            || self.value_head_dim != 128
            || !heads
            || self.conv1d_w4_f32.is_none()
            || self.gdn_scale_f32.is_none()
            || self.dt_bias_f32.is_none()
            || self.norm.weight().dtype()? != crate::array::DType::BFloat16
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(Some("the complete GDN kernel's contract"));
        }
        let tier = if self.in_proj_qkvz_ba_t.is_some() {
            false
        } else if let Some((qkv, z)) = &self.split_in_proj_qkv_z {
            qkv.lane_tier_holds(hidden)? && z.lane_tier_holds(hidden)?
        } else if let Some(merged) = &self.in_proj_qkvz_ba_q {
            merged.lane_tier_holds(hidden)?
        } else {
            self.in_proj_qkvz.lane_tier_holds(hidden)?
        };
        Ok((!tier).then_some("the GDN in-projection's lane tier"))
    }

    /// One verify wave of this layer: `x` `[N, T, H]`, lane `i`'s rows for
    /// `lanes[i]`. Each lane's output rows, cell slots and tape are the bits
    /// [`Self::forward_with_tape`] gives that lane alone with its cell and a
    /// tape sink, on the complete kernel. The projections run once over the
    /// stacked lanes; the core once per lane on the lane's own cell. A lane
    /// the complete kernel declines fails the wave and no lane's cell or
    /// tape changes. With `residual` (`[N, T, H]`) the output is the whole
    /// hidden state, the output projection adding it in its planes `_r`
    /// store over the lanes where [`Self::out_proj_verify_lanes_route`]
    /// holds.
    pub(crate) fn forward_wave(
        &self,
        x: &MxArray,
        lanes: &mut [GdnWaveLane<'_>],
        residual: Option<&MxArray>,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if lanes.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != lanes.len() as i64
            || shape[1] <= 0
        {
            return Err(Error::from_reason(format!(
                "GatedDeltaNet::forward_wave expects [N,T,H] for {} lanes, got {:?}",
                lanes.len(),
                shape.as_ref()
            )));
        }
        let (qkv, z, b, a) = self.in_projections_lanes(x)?;
        let cores = self.wave_core(&qkv, &z, &a, &b, lanes)?;
        if cores.iter().all(|(_, prepared)| prepared.is_none()) {
            let gated =
                MxArray::concatenate_many(cores.iter().map(|(out, _)| out).collect(), Some(0))?;
            return match residual {
                Some(residual) => self.out_proj.forward_verify_lanes_mixer(&gated, residual),
                None => self.out_proj.forward_lanes(&gated),
            };
        }
        let outs = cores
            .iter()
            .map(|(out, prepared)| self.out_proj.forward_prepared_sg8(out, prepared.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let out = MxArray::concatenate_many(outs.iter().collect(), Some(0))?;
        match residual {
            Some(residual) => residual.add(&out),
            None => Ok(out),
        }
    }

    /// Whether a verify wave of `lanes` lanes reaches the output
    /// projection's planes.
    pub(crate) fn out_proj_verify_lanes_route(&self, lanes: i64) -> bool {
        self.out_proj.verify_lanes_residual_route(lanes)
    }

    /// The per-lane core of [`Self::forward_wave`] over its projections
    /// (each `[N, T, ...]`): the complete kernel on lane `i`'s rows and cell,
    /// then every lane's cell slots 0/1 and tape. Returns each lane's gated
    /// rows `[1, T, value_dim]` and prepared SG8 operands.
    pub(super) fn wave_core(
        &self,
        qkv: &MxArray,
        z: &MxArray,
        a: &MxArray,
        b: &MxArray,
        lanes: &mut [GdnWaveLane<'_>],
    ) -> Result<Vec<(MxArray, Option<PreparedSg8>)>> {
        let seq_len = qkv.shape_at(1)?;
        if qkv.shape_at(0)? != lanes.len() as i64 {
            return Err(Error::from_reason(format!(
                "GDN wave: {} lanes of projections for {} cells",
                qkv.shape_at(0)?,
                lanes.len()
            )));
        }
        let mut runs = Vec::with_capacity(lanes.len());
        for (i, lane) in lanes.iter().enumerate() {
            let row = |x: &MxArray| x.slice_axis(0, i as i64, i as i64 + 1);
            let qkv = row(qkv)?;
            let run = {
                let cell: &ArraysCache = &*lane.cell;
                let conv_state = cell.get(0).cloned();
                self.forward_complete(
                    &qkv,
                    &row(z)?,
                    &row(a)?,
                    &row(b)?,
                    conv_state.as_ref(),
                    Some(cell),
                    None,
                    true,
                    1,
                    seq_len,
                    true,
                    None,
                )?
            };
            let run = run.ok_or_else(|| {
                Error::from_reason(format!("GDN wave: the complete kernel declined lane {i}"))
            })?;
            runs.push((run, qkv));
        }
        let mut outputs = Vec::with_capacity(runs.len());
        let mut tapes = Vec::with_capacity(runs.len());
        for (run, qkv) in runs {
            let tape = match (run.raw.or(Some(qkv)), run.tape) {
                (Some(qkv), Some(kernel)) => Some(GdnLayerTape {
                    kernel,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                }),
                _ => None,
            };
            tapes.push(tape);
            outputs.push((run.out, run.prepared_sg8, run.state, run.history));
        }
        let mut cores = Vec::with_capacity(outputs.len());
        for (i, ((out, prepared, state, history), tape)) in
            outputs.into_iter().zip(tapes).enumerate()
        {
            self.adopt_wave_run(&mut lanes[i], state, &history, tape)?;
            #[cfg(test)]
            note_route(GdnRoute::Complete);
            cores.push((out, prepared));
        }
        Ok(cores)
    }

    /// Lane `lane`'s cell slots 0/1 after its complete run, and its tape.
    fn adopt_wave_run(
        &self,
        lane: &mut GdnWaveLane<'_>,
        state: MxArray,
        history: &MxArray,
        tape: Option<GdnLayerTape>,
    ) -> Result<()> {
        lane.cell.set(
            0,
            history.reshape(&[1, (self.conv_kernel_dim - 1) as i64, self.conv_dim as i64])?,
        )?;
        lane.cell.set(1, state)?;
        if tape.is_some() {
            *lane.tape = tape;
        }
        Ok(())
    }

    /// The whole GDN core as one dispatch (`gated_delta_fused_complete`):
    /// `gdn_prepare`'s contract (batch 1, 4-tap conv, 128-wide heads, bf16
    /// activations, the f32 conv/scale/dt sidecars) plus the fused tail's
    /// (unmasked, T <= 16, bf16 gate and norm weight, f32 state) and compact
    /// tiled heads (`hv % Hk`; standard checkpoints repeat-expand q/k and
    /// stay on the chain below). A deferred cache (slots 4..9, the DFlash2
    /// verify) also takes grouped heads (`hv / (Hv / Hk)`), which only the
    /// deferred entry maps. `None` leaves the chain to run.
    #[allow(clippy::too_many_arguments)]
    fn forward_complete(
        &self,
        qkv: &MxArray,
        z: &MxArray,
        a: &MxArray,
        b: &MxArray,
        conv_state: Option<&MxArray>,
        cache: Option<&ArraysCache>,
        mask: Option<&MxArray>,
        use_kernel: bool,
        batch: i64,
        seq_len: i64,
        allow_native_prepared: bool,
        rows: Option<&[usize]>,
    ) -> Result<Option<CompleteRun>> {
        use crate::array::DType;
        let whole = |out: GdnCompleteOut| CompleteRun {
            out: out.out,
            prepared_sg8: out.prepared_sg8,
            state: out.state,
            history: out.history,
            tape: Some(out.tape),
            raw: out.raw,
            captured: Vec::new(),
        };
        let deferred = cache.is_some_and(|c| c.get(4).is_some());
        let grouped = deferred && self.num_k_heads > 0 && self.num_v_heads % self.num_k_heads == 0;
        if batch != 1
            || !(1..=16).contains(&seq_len)
            || mask.is_some()
            || !use_kernel
            || self.conv_kernel_dim != 4
            || self.key_head_dim != 128
            || self.value_head_dim != 128
            || !(self.tiled_gguf_layout || self.num_k_heads == self.num_v_heads || grouped)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(None);
        }
        let (Some(conv), Some(scale), Some(dt_bias)) = (
            self.conv1d_w4_f32.as_ref(),
            self.gdn_scale_f32.as_ref(),
            self.dt_bias_f32.as_ref(),
        ) else {
            return Ok(None);
        };
        let recurrent = cache.and_then(|c| c.get(1));
        if [qkv, z, a, b, self.norm.weight()]
            .iter()
            .any(|x| x.dtype().ok() != Some(DType::BFloat16))
            || recurrent.is_some_and(|s| s.dtype().ok() != Some(DType::Float32))
        {
            return Ok(None);
        }
        let history = match conv_state {
            Some(s) => s.squeeze(Some(&[0]))?,
            None => MxArray::zeros(
                &[(self.conv_kernel_dim - 1) as i64, self.conv_dim as i64],
                Some(DType::BFloat16),
            )?,
        };
        let zero_state;
        let state = match recurrent {
            Some(s) => s,
            None => {
                zero_state = MxArray::zeros(
                    &[
                        1,
                        self.num_v_heads as i64,
                        self.value_head_dim as i64,
                        self.key_head_dim as i64,
                    ],
                    Some(DType::Float32),
                )?;
                &zero_state
            }
        };
        if let Some(rows) = rows {
            return self.complete_split(qkv, z, a, b, &history, state, cache, seq_len, rows);
        }
        let prologue = GdnPrologue {
            qkv,
            a,
            b,
            conv,
            history: &history,
            scale,
            dt_bias,
            history_dst: cache.and_then(|c| c.get(3)),
        };
        let tail = GdnTail {
            z,
            norm_weight: self.norm.weight(),
            eps: self.norm.eps(),
            state_dst: cache.and_then(|c| c.get(2)),
        };
        // The source mixer packs its own Table16 operands. Avoid unused
        // native SG8 siblings when that residual-inclusive path is requested.
        let prepared = allow_native_prepared
            && seq_len == 8
            && self.num_v_heads == 32
            && self.out_proj.supports_prepared_sg8()
            && std::env::var("MLX_GDN_PREPARED_SG8").is_ok_and(|v| v == "1");
        if let Some(cache) = cache.filter(|c| c.get(4).is_some()) {
            let pending = [
                cache.get(4),
                cache.get(5),
                cache.get(6),
                cache.get(7),
                cache.get(8),
                cache.get(9),
            ];
            let [Some(p0), Some(p1), Some(p2), Some(p3), Some(p4), Some(p5)] = pending else {
                return Ok(None);
            };
            Ok(gated_delta_fused_complete_deferred(
                &prologue,
                state,
                &tail,
                prepared,
                [p0, p1, p2, p3, p4, p5],
                self.tiled_gguf_layout,
            )
            .map(whole))
        } else {
            Ok(
                gated_delta_fused_complete_with_prepared(&prologue, state, &tail, prepared)
                    .map(whole),
            )
        }
    }

    /// [`Self::forward_complete`] split at `rows`: the complete kernel on each
    /// segment, carrying the f32 state and the conv history. The state and
    /// history destinations (slots 2/3) receive only the window-final
    /// segment's. No prepared SG8 operands and no kernel tape: a capture
    /// forward records none.
    #[allow(clippy::too_many_arguments)]
    fn complete_split(
        &self,
        qkv: &MxArray,
        z: &MxArray,
        a: &MxArray,
        b: &MxArray,
        history: &MxArray,
        state: &MxArray,
        cache: Option<&ArraysCache>,
        seq_len: i64,
        rows: &[usize],
    ) -> Result<Option<CompleteRun>> {
        let (Some(conv), Some(scale), Some(dt_bias)) = (
            self.conv1d_w4_f32.as_ref(),
            self.gdn_scale_f32.as_ref(),
            self.dt_bias_f32.as_ref(),
        ) else {
            return Ok(None);
        };
        let seq = seq_len as usize;
        // The first segment meets every contract the whole window does, so
        // its decline is the window's: the chain runs.
        let mut declined = false;
        let split = gated_delta_split_at(
            seq,
            rows,
            GdnCarry {
                state: Some(state.clone()),
                history: Some(history.clone()),
            },
            |s, e, carry| {
                let r = |x: &MxArray| window_rows(x, s, e);
                let (qkv, a, b, z) = (r(qkv)?, r(a)?, r(b)?, r(z)?);
                let last = e == seq;
                let (Some(history), Some(state)) = (carry.history.as_ref(), carry.state.as_ref())
                else {
                    return Err(Error::from_reason("GDN capture: complete carry is empty"));
                };
                let out = gated_delta_fused_complete_with_prepared(
                    &GdnPrologue {
                        qkv: &qkv,
                        a: &a,
                        b: &b,
                        conv,
                        history,
                        scale,
                        dt_bias,
                        history_dst: last.then(|| cache.and_then(|c| c.get(3))).flatten(),
                    },
                    state,
                    &GdnTail {
                        z: &z,
                        norm_weight: self.norm.weight(),
                        eps: self.norm.eps(),
                        state_dst: last.then(|| cache.and_then(|c| c.get(2))).flatten(),
                    },
                    false,
                );
                match out {
                    Some(out) => Ok((
                        out.out,
                        GdnCarry {
                            state: Some(out.state),
                            history: Some(out.history),
                        },
                    )),
                    None => {
                        declined = s == 0;
                        Err(Error::from_reason(
                            "complete GDN kernel declined a capture segment",
                        ))
                    }
                }
            },
        );
        let split = match split {
            Ok(split) => split,
            Err(_) if declined => return Ok(None),
            Err(error) => return Err(error),
        };
        let (Some(state), Some(history)) = (split.carry.state, split.carry.history) else {
            return Err(Error::from_reason(
                "GDN capture: the split carried no state",
            ));
        };
        Ok(Some(CompleteRun {
            out: split.y,
            prepared_sg8: None,
            state,
            history,
            tape: None,
            raw: None,
            captured: captured_states(split.captured)?,
        }))
    }

    /// The conv stage of [`prep_via_conv`]: depthwise conv (fused
    /// `window_conv` when possible) → SiLU → q|k|v split → head reshape.
    /// Returns the PRE-NORM head tensors; the q/k RMS-norm+scale runs in
    /// `prep_via_conv` (or inside the fused-prefill kernel).
    #[allow(clippy::too_many_arguments)]
    fn prep_conv_qkv(
        &self,
        qkv: &MxArray,
        batch: i64,
        seq_len: i64,
        valid_lengths: Option<&[i32]>,
        mut cache: Option<&mut ArraysCache>,
        conv_state: Option<MxArray>,
        use_kernel: bool,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        // Fused path: one Metal dispatch covering history prepend + 4-tap
        // depthwise conv + SiLU + next-state emission (`mlx_qwen4_window_conv`
        // is width-generic: bf16 x [1,T,W], bf16 [3,W] history, f32 [W,4]
        // tap-major weight, batch == 1). Anything off-contract falls back to
        // the generic conv path below.
        let fused_conv = (|| -> Option<(MxArray, MxArray)> {
            if batch != 1
                || valid_lengths.is_some()
                || self.conv_kernel_dim != 4
                || !use_kernel
                || !crate::engine::persistence::compiled_forward_backend_available()
            {
                return None;
            }
            let w = self.conv1d_w4_f32.as_ref()?;
            let qkv_shape = qkv.shape().ok()?;
            if qkv.dtype().ok()? != crate::array::DType::BFloat16
                || qkv_shape.len() != 3
                || qkv_shape[0] != 1
                || qkv_shape[1] != seq_len
                || qkv_shape[2] != self.conv_dim as i64
            {
                return None;
            }
            let history = match &conv_state {
                Some(s) => s.squeeze(Some(&[0])).ok()?,
                None => MxArray::zeros(
                    &[(self.conv_kernel_dim - 1) as i64, self.conv_dim as i64],
                    Some(crate::array::DType::BFloat16),
                )
                .ok()?,
            };
            let mut out = std::ptr::null_mut();
            let mut next = std::ptr::null_mut();
            if unsafe {
                sys::mlx_qwen4_window_conv(
                    qkv.handle.0,
                    history.handle.0,
                    w.handle.0,
                    &mut out,
                    &mut next,
                )
            } {
                Some((
                    MxArray::from_handle(out, "window_conv:out").ok()?,
                    MxArray::from_handle(next, "window_conv:history").ok()?,
                ))
            } else {
                None
            }
        })();

        let conv_out = if let Some((fused_out, next_history)) = fused_conv {
            if let Some(cache) = cache.as_deref_mut() {
                // next_history is [K-1, W] — the raw window tail, identical to
                // the conv_input slice the fallback path stores.
                cache.set(
                    0,
                    next_history.reshape(&[
                        1,
                        (self.conv_kernel_dim - 1) as i64,
                        self.conv_dim as i64,
                    ])?,
                )?;
            }
            fused_out
        } else {
            let conv_input = match conv_state {
                Some(state) => {
                    // Prepend cached conv_state: [B, kernel-1, conv_dim]
                    MxArray::concatenate(&state, qkv, 1)?
                }
                None => {
                    // No cache: prepend zeros of size (kernel_size - 1)
                    // Use qkv's dtype to avoid f32 promotion for bf16/f16 models
                    let pad_len = (self.conv_kernel_dim - 1) as i64;
                    let zeros = MxArray::zeros(
                        &[batch, pad_len, self.conv_dim as i64],
                        Some(qkv.dtype()?),
                    )?;
                    MxArray::concatenate(&zeros, qkv, 1)?
                }
            };

            // Update conv_state in cache
            if let Some(cache) = cache {
                // Save last (kernel_size - 1) valid timesteps as new conv_state.
                // For right-padded batches with explicit `valid_lengths`, row b's
                // true last K-1 taps live at conv_input[b, lengths[b] .. lengths[b] + K - 1, :]
                // rather than at the trailing padded end of conv_input.
                let total_len = conv_input.shape_at(1)?;
                let keep = (self.conv_kernel_dim - 1) as i64;
                if let Some(lengths) = valid_lengths {
                    if keep > 0 && total_len >= keep {
                        let mut rows = Vec::with_capacity(lengths.len());
                        for (b, &len) in lengths.iter().enumerate() {
                            let start = i64::from(len);
                            let row = conv_input
                                .slice_axis(0, b as i64, b as i64 + 1)?
                                .slice_axis(1, start, start + keep)?;
                            rows.push(row);
                        }
                        let next_state = if rows.len() == 1 {
                            rows.remove(0)
                        } else {
                            let refs: Vec<&MxArray> = rows.iter().collect();
                            MxArray::concatenate_many(refs, Some(0))?
                        };
                        cache.set(0, next_state)?;
                    } else if total_len >= keep {
                        let parts = conv_input.split_sections(&[total_len - keep], 1)?;
                        cache.set(0, parts[1].clone())?;
                    }
                } else if total_len >= keep {
                    let parts = conv_input.split_sections(&[total_len - keep], 1)?;
                    cache.set(0, parts[1].clone())?;
                }
            }

            // Conv1d: [B, T_in, conv_dim] → [B, T_out, conv_dim]
            let conv_out = self.conv1d.forward(&conv_input)?;

            // Take last seq_len timesteps (conv may produce more than seq_len if conv_state was prepended)
            let conv_out_len = conv_out.shape_at(1)?;
            let conv_out = if conv_out_len > seq_len {
                conv_out
                    .split_sections(&[conv_out_len - seq_len], 1)?
                    .swap_remove(1)
            } else {
                conv_out
            };

            // Apply SiLU activation — silu(x) = x * sigmoid(x), fused through the
            // compiled sigmoid-mul kernel (one dispatch instead of Sigmoid+Multiply).
            Activations::sigmoid_mul_compiled(&conv_out, &conv_out)?
        };

        // Split into q, k, v
        let conv_split =
            conv_out.split_sections(&[self.key_dim as i64, (self.key_dim * 2) as i64], 2)?;
        let q_flat = &conv_split[0];
        let k_flat = &conv_split[1];
        let v_flat = &conv_split[2];

        // Reshape to head format
        // q, k: [B, T, key_dim] → [B, T, Hk, Dk]
        let q = q_flat.reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        let k = k_flat.reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        // v: [B, T, value_dim] → [B, T, Hv, Dv]
        let v = v_flat.reshape(&[
            batch,
            seq_len,
            self.num_v_heads as i64,
            self.value_head_dim as i64,
        ])?;
        Ok((q, k, v))
    }

    /// Apply RMS norm scaling to q and k (matching Python exactly):
    ///   inv_scale = head_k_dim^(-0.5)
    ///   q = (inv_scale^2) * rms_norm(q, None, 1e-6)
    ///   k = inv_scale * rms_norm(k, None, 1e-6)
    /// The scale factors live in constant bf16 vectors so the norm+scale
    /// is ONE fast_rms_norm dispatch per tensor (bf16 activations only —
    /// other dtypes keep the scalar path to avoid a promote).
    fn norm_qk(&self, q: &MxArray, k: &MxArray) -> Result<(MxArray, MxArray)> {
        if q.dtype()? == crate::array::DType::BFloat16
            && k.dtype()? == crate::array::DType::BFloat16
        {
            Ok((
                rms_norm_scaled(q, &self.qk_norm_w_q, 1e-6)?,
                rms_norm_scaled(k, &self.qk_norm_w_k, 1e-6)?,
            ))
        } else {
            let inv_scale = (self.key_head_dim as f64).powf(-0.5);
            Ok((
                rms_norm_unscaled(q, 1e-6)?.mul_scalar(inv_scale * inv_scale)?,
                rms_norm_unscaled(k, 1e-6)?.mul_scalar(inv_scale)?,
            ))
        }
    }

    /// Generic prep path: depthwise conv (fused `window_conv` when possible)
    /// → SiLU → q|k|v split → head reshape → q/k RMS-norm+scale. Runs whenever
    /// the fully-fused `gdn_prepare` kernel is off-contract (batch > 1,
    /// seq ≥ 64, non-10240 geometry) or declines.
    #[allow(clippy::too_many_arguments)]
    fn prep_via_conv(
        &self,
        qkv: &MxArray,
        batch: i64,
        seq_len: i64,
        valid_lengths: Option<&[i32]>,
        cache: Option<&mut ArraysCache>,
        conv_state: Option<MxArray>,
        use_kernel: bool,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let (q, k, v) = self.prep_conv_qkv(
            qkv,
            batch,
            seq_len,
            valid_lengths,
            cache,
            conv_state,
            use_kernel,
        )?;
        let (q, k) = self.norm_qk(&q, &k)?;
        Ok((q, k, v))
    }

    /// Tree-conv1d fallback when fused CUDA `prepare_tree` does not apply
    /// (Metal/CPU or non-10240 test geometries). Matches `mlx_gdn_prepare_tree_cuda`
    /// tap indexing (`src < 0 -> history[keep + src]`, `src >= 0 -> qkv[src]`).
    fn prep_via_conv_tree(
        &self,
        qkv: &MxArray,
        batch: i64,
        seq_len: i64,
        cache: Option<&mut ArraysCache>,
        conv_state: Option<MxArray>,
        conv_taps: &MxArray,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let kd = i64::from(self.conv_kernel_dim);
        let keep = kd - 1;
        let conv_dim = i64::from(self.conv_dim);
        let dtype = qkv.dtype()?;
        let history = match conv_state {
            Some(s) => s,
            None => MxArray::zeros(&[batch, keep, conv_dim], Some(dtype))?,
        };
        let combined = MxArray::concatenate(&history, qkv, 1)?;
        if let Some(cache) = cache {
            let total_len = combined.shape_at(1)?;
            if total_len >= keep {
                let tail = combined.slice_axis(1, total_len - keep, total_len)?;
                cache.set(0, tail)?;
            }
        }
        let shift = MxArray::from_int32(&[keep as i32], &[1])?;
        let shifted = conv_taps.add(&shift)?.reshape(&[seq_len * kd])?;
        let gathered = combined
            .take(&shifted, 1)?
            .reshape(&[batch, seq_len, kd, conv_dim])?;
        let w = self
            .conv1d
            .get_weight()
            .reshape(&[conv_dim, kd])?
            .transpose(Some(&[1, 0]))?
            .reshape(&[1, 1, kd, conv_dim])?
            .astype(dtype)?;
        let conv_raw = gathered
            .astype(DType::Float32)?
            .mul(&w.astype(DType::Float32)?)?
            .sum(Some(&[2]), Some(false))?
            .astype(dtype)?;
        let conv_out = Activations::sigmoid_mul_compiled(&conv_raw, &conv_raw)?;
        let conv_split =
            conv_out.split_sections(&[self.key_dim as i64, (self.key_dim * 2) as i64], 2)?;
        let q = conv_split[0].reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        let k = conv_split[1].reshape(&[
            batch,
            seq_len,
            self.num_k_heads as i64,
            self.key_head_dim as i64,
        ])?;
        let v = conv_split[2].reshape(&[
            batch,
            seq_len,
            self.num_v_heads as i64,
            self.value_head_dim as i64,
        ])?;
        let (q, k) = self.norm_qk(&q, &k)?;
        Ok((q, k, v))
    }

    /// Prompt-window fused path (`mlx_gdn_fused_prefill`): one primitive for
    /// the gating + q/k norm + chunked recurrence + gated norm + z gate of an
    /// unmasked batch-1 window, bit-identical to `prep_via_conv` +
    /// `gated_delta_update` + the layer tail. Gated to `seq_len >= 64`: below
    /// that `gdn_prepare` (17..=63) and the fused step (1..=32) keep their
    /// exact existing code paths and bits. `MLX_GDN_PREFILL_FUSED=0` disables
    /// it for A/B. `None` — or `Raw` — leaves the chain to run.
    #[allow(clippy::too_many_arguments)]
    fn prefill_fused(
        &self,
        qkv: &MxArray,
        z: &MxArray,
        a: &MxArray,
        b: &MxArray,
        conv_state: Option<&MxArray>,
        cache: Option<&mut ArraysCache>,
        mask: Option<&MxArray>,
        use_kernel: bool,
        batch: i64,
        seq_len: i64,
        rows: Option<&[usize]>,
    ) -> Result<Option<PrefillPrep>> {
        if batch != 1
            || seq_len < 64
            || mask.is_some()
            || !use_kernel
            || !prefill_fused_enabled()
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(None);
        }
        // A lazy clone of the carried state (no eval): the borrow of `cache`
        // must end before `prep_conv_qkv` takes it for the conv-history slot.
        let recurrent = cache.as_deref().and_then(|c| c.get(1).cloned());
        if [qkv, z, a, b, self.norm.weight()]
            .iter()
            .any(|x| x.dtype().ok() != Some(crate::array::DType::BFloat16))
        {
            return Ok(None);
        }
        if !super::gated_delta::prefill_scalar_dtypes(&self.a_log, &self.dt_bias)
            || recurrent
                .as_ref()
                .is_some_and(|s| s.dtype().ok() != Some(crate::array::DType::Float32))
        {
            return Ok(None);
        }
        if self.num_k_heads <= 0 || self.num_v_heads % self.num_k_heads != 0 {
            return Ok(None);
        }
        // Conv first: on a late kernel decline the prepared views feed the
        // shared chain instead of being recomputed.
        let (q_raw, k_raw, v) = self.prep_conv_qkv(
            qkv,
            batch,
            seq_len,
            None,
            cache,
            conv_state.cloned(),
            use_kernel,
        )?;
        if [q_raw.dtype().ok(), k_raw.dtype().ok(), v.dtype().ok()]
            .iter()
            .any(|d| *d != Some(crate::array::DType::BFloat16))
        {
            return Ok(Some(PrefillPrep::Raw {
                q: q_raw,
                k: k_raw,
                v,
            }));
        }
        let window = |s: usize, e: usize, state: Option<&MxArray>| {
            let whole = (s, e) == (0, seq_len as usize);
            let r = |x: &MxArray| {
                if whole {
                    Ok(x.clone())
                } else {
                    window_rows(x, s, e)
                }
            };
            let (a, b, z) = (r(a)?, r(b)?, r(z)?);
            gated_delta_fused_prefill(
                &r(&q_raw)?,
                &r(&k_raw)?,
                &r(&v)?,
                state,
                &GdnPrefill {
                    a: &a,
                    b: &b,
                    a_log: &self.a_log,
                    dt_bias: &self.dt_bias,
                    wq: &self.qk_norm_w_q,
                    wk: &self.qk_norm_w_k,
                    qk_eps: 1e-6,
                    tiled: self.tiled_gguf_layout,
                    tail: GdnTail {
                        z: &z,
                        norm_weight: self.norm.weight(),
                        eps: self.norm.eps(),
                        state_dst: None,
                    },
                },
            )
        };
        let fused = match rows {
            None => window(0, seq_len as usize, recurrent.as_ref())?
                .map(|(gated, state)| (gated, state, Vec::new())),
            Some(rows) => {
                // The first segment meets every contract the whole window
                // does, so its decline is the window's: the chain runs.
                let mut declined = false;
                let split = gated_delta_split_at(
                    seq_len as usize,
                    rows,
                    GdnCarry {
                        state: recurrent.clone(),
                        history: None,
                    },
                    |s, e, carry| match window(s, e, carry.state.as_ref())? {
                        Some((gated, state)) => Ok((
                            gated,
                            GdnCarry {
                                state: Some(state),
                                history: None,
                            },
                        )),
                        None => {
                            declined = s == 0;
                            Err(Error::from_reason(
                                "fused GDN prefill declined a capture segment",
                            ))
                        }
                    },
                );
                match split {
                    Ok(split) => {
                        let state = split.carry.state.ok_or_else(|| {
                            Error::from_reason("GDN capture: the split carried no state")
                        })?;
                        Some((split.y, state, captured_states(split.captured)?))
                    }
                    Err(_) if declined => None,
                    Err(error) => return Err(error),
                }
            }
        };
        Ok(Some(match fused {
            Some((gated, state, captured)) => PrefillPrep::Fused {
                gated,
                state,
                captured,
            },
            None => PrefillPrep::Raw {
                q: q_raw,
                k: k_raw,
                v,
            },
        }))
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_in_proj_qkvz_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz_ba_t = None; // invalidate stacked cache
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_qkv_z = None;
        self.in_proj_qkvz.set_weight(w, "in_proj_qkvz")
    }
    pub fn set_in_proj_ba_weight(&mut self, w: &MxArray) -> Result<()> {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_b_a = None;
        self.split_in_proj_b_a_m8 = None;
        self.in_proj_ba.set_weight(w, "in_proj_ba")
    }
    pub fn set_conv1d_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        let w = super::sidecar_to_compute_dtype(w, compute_dtype)?;
        // Prepare the fused window-conv weight from the SAME post-normalization
        // tensor the fallback conv reads: bf16→f32 widening is exact, so both
        // paths see identical values and toggling the kernel changes dispatch
        // only. The kernel wants f32 tap-major `[conv_dim, 4]`, which is the
        // flat order of the MLX `[conv_dim, K, 1]` / `[conv_dim, 1, K]` /
        // `[conv_dim, K]` conv layouts. Other layouts stay None → conv_general.
        self.conv1d_w4_f32 = None;
        if self.conv_kernel_dim == 4
            && let Ok(shape) = w.shape()
        {
            let cd = self.conv_dim as i64;
            let kd = self.conv_kernel_dim as i64;
            // Accept only layouts whose flat order is channel-major
            // `[conv_dim, K]` taps: [W,K], [W,K,1], [W,1,K].
            let tap_major = match shape.len() {
                2 => shape[0] == cd && shape[1] == kd,
                3 => {
                    shape[0] == cd
                        && ((shape[1] == kd && shape[2] == 1) || (shape[1] == 1 && shape[2] == kd))
                }
                _ => false,
            };
            if tap_major {
                self.conv1d_w4_f32 = w
                    .reshape(&[cd, kd])
                    .and_then(|v| v.astype(crate::array::DType::Float32))
                    .ok();
            }
        }
        self.conv1d.set_weight(&w)
    }
    /// Splash `compiledGdn` (fail closed): why the DFlash2 verify's deferred
    /// complete kernel would decline this layer. Checked when a draft is
    /// attached, so a turn never fails after its prefill. The gated-norm
    /// weight carries the compute dtype (`set_norm_weight`).
    pub(crate) fn deferred_verify_refusal(&self) -> Option<&'static str> {
        if self.conv_kernel_dim != 4 || self.key_head_dim != 128 || self.value_head_dim != 128 {
            return Some("4 conv taps and 128-wide key and value heads");
        }
        if self.num_k_heads <= 0 || self.num_v_heads % self.num_k_heads != 0 {
            return Some("value heads a multiple of key heads");
        }
        if self.conv1d_w4_f32.is_none()
            || self.gdn_scale_f32.is_none()
            || self.dt_bias_f32.is_none()
        {
            return Some("an f32 tap-major conv weight, scale and dt_bias");
        }
        if self.norm.weight().dtype().ok() != Some(DType::BFloat16) {
            return Some("bf16 activations");
        }
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return Some("Metal");
        }
        None
    }
    pub fn set_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        // norm.weight may be stored as f32 in checkpoints for precision;
        // cast to the model's compute dtype, not to dt_bias's dtype (bf16 or
        // f32 depending on the checkpoint).
        self.norm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }
    pub fn set_out_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.out_proj.set_weight(w, "out_proj")
    }
    pub fn set_dt_bias(&mut self, w: &MxArray) {
        self.dt_bias = w.clone();
        self.dt_bias_f32 = w.astype(crate::array::DType::Float32).ok();
    }
    pub fn set_a_log(&mut self, w: &MxArray) -> Result<()> {
        // Cast A_log to model dtype (bf16) to avoid f32→bf16 promotion overhead.
        // The precision difference is negligible for inference.
        self.a_log = w.astype(self.dt_bias.dtype()?)?;
        // `-exp(a_log)` in f32 for the fused gdn_prepare kernel — exp() of the
        // f32-widened STORED a_log matches what the gating kernels compute at
        // runtime. `None` on failure keeps the generic prep path.
        self.gdn_scale_f32 = self
            .a_log
            .astype(crate::array::DType::Float32)
            .and_then(|v| v.exp())
            .and_then(|v| v.mul_scalar(-1.0))
            .ok();
        Ok(())
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_in_proj_qkvz(&mut self, ql: QuantizedLinear) {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_qkv_z = None;
        self.in_proj_qkvz.set_quantized(ql);
    }
    pub fn set_quantized_in_proj_ba(&mut self, ql: QuantizedLinear) {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_b_a = None;
        self.split_in_proj_b_a_m8 = None;
        self.in_proj_ba.set_quantized(ql);
    }
    pub fn set_split_in_proj_qkv_z(&mut self, qkv: LinearProj, z: LinearProj) {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_qkv_z = Some((qkv, z));
    }
    pub fn set_split_in_proj_b_a(&mut self, b: LinearProj, a: LinearProj) {
        self.in_proj_qkvz_ba_t = None;
        self.in_proj_qkvz_ba_q = None;
        self.split_in_proj_b_a = Some((b, a));
        self.split_in_proj_b_a_m8 = None;
    }
    pub fn set_quantized_out_proj(&mut self, ql: QuantizedLinear) {
        self.out_proj.set_quantized(ql);
    }

    /// Whether any mode-aware projection in this GDN block is quantized.
    /// Convolution, norms, and recurrent parameters are always dense.
    pub fn is_quantized(&self) -> bool {
        self.in_proj_qkvz.is_quantized()
            || self.in_proj_ba.is_quantized()
            || self.split_in_proj_qkv_z.is_some()
            || self.split_in_proj_b_a.is_some()
            || self.out_proj.is_quantized()
    }

    #[cfg(test)]
    pub(crate) fn split_in_proj_quantized_sides(
        &self,
    ) -> (Option<(bool, bool)>, Option<(bool, bool)>) {
        let qkv_z = self
            .split_in_proj_qkv_z
            .as_ref()
            .map(|(qkv, z)| (qkv.is_quantized(), z.is_quantized()));
        let b_a = self
            .split_in_proj_b_a
            .as_ref()
            .map(|(b, a)| (b.is_quantized(), a.is_quantized()));
        (qkv_z, b_a)
    }

    /// The quantization modes of `in_proj_qkvz`, `in_proj_ba` and the merged
    /// `[qkvz; ba]` projection (`None` when unmerged or dense).
    #[cfg(test)]
    pub(crate) fn in_proj_modes(&self) -> (Option<String>, Option<String>, Option<String>) {
        let mode = |proj: &LinearProj| match proj {
            LinearProj::Quantized(ql) => Some(ql.mode().to_string()),
            LinearProj::Standard(_) => None,
        };
        (
            mode(&self.in_proj_qkvz),
            mode(&self.in_proj_ba),
            self.in_proj_qkvz_ba_q.as_ref().and_then(mode),
        )
    }

    /// The output projection, where a load attaches its planes.
    pub(crate) fn out_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.out_proj
    }

    pub(crate) fn out_proj(&self) -> &LinearProj {
        &self.out_proj
    }

    /// The quantization mode of `out_proj` (`None` when dense).
    #[cfg(test)]
    pub(crate) fn out_proj_mode(&self) -> Option<&str> {
        match &self.out_proj {
            LinearProj::Quantized(ql) => Some(ql.mode()),
            LinearProj::Standard(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn prism_hadamard_sites(&self) -> (bool, bool, bool, bool) {
        let qkv = self
            .split_in_proj_qkv_z
            .as_ref()
            .map(|(qkv, _)| qkv.has_hadamard())
            .unwrap_or_else(|| self.in_proj_qkvz.has_hadamard());
        let z = self
            .split_in_proj_qkv_z
            .as_ref()
            .map(|(_, z)| z.has_hadamard())
            .unwrap_or(false);
        let ba = self.in_proj_ba.has_hadamard()
            || self
                .split_in_proj_b_a
                .as_ref()
                .map(|(b, a)| b.has_hadamard() || a.has_hadamard())
                .unwrap_or(false);
        (qkv, z, self.out_proj.has_hadamard(), ba)
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_in_proj_qkvz_weight(&self) -> MxArray {
        self.in_proj_qkvz.get_weight()
    }
    pub fn get_in_proj_ba_weight(&self) -> MxArray {
        self.in_proj_ba.get_weight()
    }
    pub fn get_conv1d_weight(&self) -> MxArray {
        self.conv1d.get_weight()
    }
    pub fn get_norm_weight(&self) -> MxArray {
        self.norm.get_weight()
    }
    pub fn get_out_proj_weight(&self) -> MxArray {
        self.out_proj.get_weight()
    }
    pub fn get_dt_bias(&self) -> MxArray {
        self.dt_bias.clone()
    }
    pub fn get_a_log(&self) -> MxArray {
        self.a_log.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    #[test]
    fn conv_replay_preserves_every_prefix_and_batch() -> Result<()> {
        for batch in [1i64, 2] {
            for keep in [1i64, 3, 9] {
                for dtype in [DType::Float32, DType::BFloat16] {
                    // The retained five channels are strided views when the
                    // backing row has ten channels. This catches assumptions
                    // that a batched tape slice must be contiguous.
                    for width in [5i64, 10] {
                        let data: Vec<f32> =
                            (0..batch * 8 * width).map(|i| (i % 97) as f32).collect();
                        let qkv = MxArray::from_float32(&data, &[batch, 8, width])?
                            .astype(dtype)?
                            .slice_axis(2, 0, 5)?;
                        let history: Vec<f32> = (0..batch * keep * 5)
                            .map(|i| -1.0 - (i % 97) as f32)
                            .collect();
                        let snapshot =
                            MxArray::from_float32(&history, &[batch, keep, 5])?.astype(dtype)?;
                        for warm in [false, true] {
                            for accepted in 0usize..=8 {
                                let state = warm.then_some(&snapshot);
                                let result = replay_conv_state(&qkv, state, keep, accepted)?;
                                let mut expected = Vec::new();
                                for b in 0..batch {
                                    for position in accepted as i64..accepted as i64 + keep {
                                        for channel in 0..5 {
                                            let value = if position < keep {
                                                if warm {
                                                    history[((b * keep + position) * 5 + channel)
                                                        as usize]
                                                } else {
                                                    0.0
                                                }
                                            } else {
                                                data[((b * 8 + position - keep) * width + channel)
                                                    as usize]
                                            };
                                            expected.push(value);
                                        }
                                    }
                                }
                                assert_eq!(result.shape()?.as_ref(), &[batch, keep, 5]);
                                assert_eq!(result.dtype()?, dtype);
                                assert_eq!(
                                    result.astype(DType::Float32)?.to_float32()?.as_ref(),
                                    expected.as_slice()
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn rand_bf16(shape: &[i64]) -> MxArray {
        MxArray::random_normal(shape, 0.0, 0.3, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn max_abs_diff(a: &MxArray, b: &MxArray) -> f32 {
        let af = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bf = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        // NaN must never compare as "equal": f32::max drops NaN, so fold
        // manually — any non-finite diff surfaces as +inf and fails bounds.
        af.as_ref()
            .iter()
            .zip(bf.as_ref())
            .map(|(x, y)| {
                assert!(
                    x.is_finite() && y.is_finite(),
                    "non-finite element: {x} vs {y}"
                );
                (x - y).abs()
            })
            .fold(0.0f32, |acc, d| {
                if d.is_nan() {
                    f32::INFINITY
                } else {
                    acc.max(d)
                }
            })
    }

    /// Kernel geometry: 16k×128 + 48v×128 → conv_dim 10240, the only shape
    /// `mlx_qwen4_gdn_prepare` accepts.
    fn kernel_geometry_net() -> GatedDeltaNet {
        kernel_geometry_net_with(rand_bf16)
    }

    fn kernel_geometry_net_with(weight: impl Fn(&[i64]) -> MxArray) -> GatedDeltaNet {
        kernel_geometry_net_heads(48, weight)
    }

    /// Hk 16 and `hv` value heads of 128 (48: the 27B, 32: the 35B MoE).
    fn kernel_geometry_net_heads(hv: i64, weight: impl Fn(&[i64]) -> MxArray) -> GatedDeltaNet {
        let config = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: hv as i32,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };
        let mut net = GatedDeltaNet::new(&config).unwrap();
        // key_dim*2 + value_dim*2 rows out (27B: 2048*2 + 6144*2 = 16384).
        let values = hv * 128;
        net.set_in_proj_qkvz_weight(&weight(&[4096 + 2 * values, 64]))
            .unwrap();
        net.set_in_proj_ba_weight(&weight(&[2 * hv, 64])).unwrap();
        net.set_conv1d_weight(&weight(&[4096 + values, 1, 4]), DType::BFloat16)
            .unwrap();
        net.set_norm_weight(&weight(&[128]), DType::BFloat16)
            .unwrap();
        net.set_out_proj_weight(&weight(&[64, values])).unwrap();
        net.set_dt_bias(&weight(&[hv]));
        net.set_a_log(&weight(&[hv])).unwrap();
        net
    }

    #[test]
    fn gdn_right_pad_conv_state_parity() {
        fn check_geometry(head_dim: i32, lengths: [usize; 3]) {
            let hidden = 64i32;
            let cfg = Qwen3_5Config {
                rope_scaling: None,
                qwen35_gguf_gdn_layout: None,
                kv_format: None,
                vocab_size: 32,
                hidden_size: hidden,
                num_layers: 1,
                num_heads: 4,
                num_kv_heads: 2,
                intermediate_size: 128,
                rms_norm_eps: 1e-6,
                head_dim: 16,
                tie_word_embeddings: true,
                attention_bias: false,
                max_position_embeddings: 128,
                pad_token_id: 0,
                eos_token_id: 1,
                bos_token_id: 2,
                linear_num_value_heads: 4,
                linear_num_key_heads: 2,
                linear_key_head_dim: head_dim,
                linear_value_head_dim: head_dim,
                linear_conv_kernel_dim: 4,
                full_attention_interval: 4,
                partial_rotary_factor: 0.25,
                rope_theta: 10_000.0,
                paged_cache_memory_mb: None,
                paged_block_size: None,
                n_mtp_layers: 0,
            };
            let mut gdn = GatedDeltaNet::new(&cfg).unwrap();
            let det = |shape: &[i64], salt: f32, scale: f32| -> MxArray {
                let n: usize = shape.iter().map(|&d| d as usize).product();
                let vals: Vec<f32> = (0..n)
                    .map(|i| ((i as f32) * 0.4173 + salt).sin() * scale)
                    .collect();
                MxArray::from_float32(&vals, shape).unwrap()
            };
            let key_dim = cfg.linear_key_dim() as i64;
            let val_dim = cfg.linear_value_dim() as i64;
            let conv_dim = cfg.linear_conv_dim() as i64;
            let hv = cfg.linear_num_value_heads as i64;
            gdn.set_in_proj_qkvz_weight(&det(
                &[2 * key_dim + 2 * val_dim, hidden as i64],
                1.0,
                0.12,
            ))
            .unwrap();
            gdn.set_in_proj_ba_weight(&det(&[2 * hv, hidden as i64], 2.0, 0.12))
                .unwrap();
            gdn.set_conv1d_weight(&det(&[conv_dim, 4, 1], 3.0, 0.2), DType::Float32)
                .unwrap();
            gdn.set_out_proj_weight(&det(&[hidden as i64, val_dim], 4.0, 0.12))
                .unwrap();

            let hidden_usize = hidden as usize;
            let max_len = *lengths.iter().max().unwrap();

            // Build 3 unpadded sequences of lengths `lengths` + 2 continuation tokens each.
            let mut seqs: Vec<Vec<f32>> = Vec::with_capacity(3);
            let mut cont_steps: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
            for (b, &len) in lengths.iter().enumerate() {
                let seq: Vec<f32> = (0..(len * hidden_usize))
                    .map(|i| (((b * 1009 + i) as f32) * 0.041).sin() * 0.3)
                    .collect();
                seqs.push(seq);
                for (step, slot) in cont_steps.iter_mut().enumerate() {
                    let tok: Vec<f32> = (0..hidden_usize)
                        .map(|i| (((b * 733 + step * 197 + i) as f32) * 0.073).cos() * 0.3)
                        .collect();
                    slot.push(tok);
                }
            }

            // Reference: run each sequence independently at B=1 unpadded + 2 decode steps.
            let mut ref_prefill = Vec::with_capacity(3);
            let mut ref_prefill_conv = Vec::with_capacity(3);
            let mut ref_prefill_rec = Vec::with_capacity(3);
            let mut ref_decode: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
            for b in 0..3 {
                let mut cache = ArraysCache::new(2);
                let x_single =
                    MxArray::from_float32(&seqs[b], &[1, lengths[b] as i64, hidden as i64])
                        .unwrap();
                let y = gdn
                    .forward(&x_single, None, Some(&mut cache), true)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec();
                ref_prefill.push(y);
                ref_prefill_conv.push(cache.get(0).unwrap().to_float32().unwrap().to_vec());
                ref_prefill_rec.push(cache.get(1).unwrap().to_float32().unwrap().to_vec());
                for step in 0..2 {
                    let x_step =
                        MxArray::from_float32(&cont_steps[step][b], &[1, 1, hidden as i64])
                            .unwrap();
                    let y_step = gdn
                        .forward(&x_step, None, Some(&mut cache), true)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .to_vec();
                    ref_decode[step].push(y_step);
                }
            }

            // Batched right-padded prefill (B=3, T=max_len) with mask + valid_lengths.
            let mut padded_x = vec![0.0f32; 3 * max_len * hidden_usize];
            let mut mask_vals = vec![0i32; 3 * max_len];
            for (b, &len) in lengths.iter().enumerate() {
                for t in 0..max_len {
                    if t < len {
                        mask_vals[b * max_len + t] = 1;
                        let dst = (b * max_len + t) * hidden_usize;
                        let src = t * hidden_usize;
                        padded_x[dst..dst + hidden_usize]
                            .copy_from_slice(&seqs[b][src..src + hidden_usize]);
                    } else {
                        // Put non-zero garbage in padded positions to prove masking +
                        // valid_lengths completely isolate them from conv_state & state.
                        let dst = (b * max_len + t) * hidden_usize;
                        for v in &mut padded_x[dst..dst + hidden_usize] {
                            *v = 7.5;
                        }
                    }
                }
            }
            let x_batch =
                MxArray::from_float32(&padded_x, &[3, max_len as i64, hidden as i64]).unwrap();
            let mask_batch = MxArray::from_int32(&mask_vals, &[3, max_len as i64])
                .unwrap()
                .greater(&MxArray::scalar_int(0).unwrap())
                .unwrap();
            let valid_lengths: Vec<i32> = lengths.iter().map(|&l| l as i32).collect();

            let mut batch_cache = ArraysCache::new(2);
            let y_batch = gdn
                .forward_batched_masked(
                    &x_batch,
                    Some(&mask_batch),
                    Some(&valid_lengths),
                    Some(&mut batch_cache),
                )
                .unwrap()
                .to_float32()
                .unwrap();

            // Check every valid prefill position AND cached conv_state / recurrent_state match B=1.
            for (b, &len) in lengths.iter().enumerate() {
                let mut max_diff = 0.0f32;
                for t in 0..len {
                    let b_off = (b * max_len + t) * hidden_usize;
                    let r_off = t * hidden_usize;
                    for d in 0..hidden_usize {
                        max_diff = max_diff
                            .max((y_batch.as_ref()[b_off + d] - ref_prefill[b][r_off + d]).abs());
                    }
                }
                assert!(
                    max_diff < 1e-4,
                    "prefill mismatch (head_dim={head_dim}, row={b}, len={len}): max_diff={max_diff}"
                );

                let row_conv = batch_cache
                    .get(0)
                    .unwrap()
                    .slice_axis(0, b as i64, b as i64 + 1)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let conv_diff = row_conv
                    .as_ref()
                    .iter()
                    .zip(&ref_prefill_conv[b])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    conv_diff < 1e-5,
                    "conv_state mismatch (head_dim={head_dim}, row={b}, len={len}): max_diff={conv_diff}"
                );

                let row_rec = batch_cache
                    .get(1)
                    .unwrap()
                    .slice_axis(0, b as i64, b as i64 + 1)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let rec_diff = row_rec
                    .as_ref()
                    .iter()
                    .zip(&ref_prefill_rec[b])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    rec_diff < 1e-5,
                    "recurrent_state mismatch (head_dim={head_dim}, row={b}, len={len}): max_diff={rec_diff}"
                );
            }

            // Continuation decode steps (B=3, T=1) without mask.
            for step in 0..2 {
                let mut flat_step = Vec::with_capacity(3 * hidden_usize);
                for row in &cont_steps[step] {
                    flat_step.extend_from_slice(row);
                }
                let x_step = MxArray::from_float32(&flat_step, &[3, 1, hidden as i64]).unwrap();
                let y_step = gdn
                    .forward_batched_masked(&x_step, None, None, Some(&mut batch_cache))
                    .unwrap()
                    .to_float32()
                    .unwrap();

                for (b, ref_row) in ref_decode[step].iter().enumerate() {
                    let off = b * hidden_usize;
                    let max_diff = y_step.as_ref()[off..off + hidden_usize]
                        .iter()
                        .zip(ref_row)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_diff < 1e-4,
                        "decode step {step} mismatch (head_dim={head_dim}, row={b}): max_diff={max_diff}"
                    );
                }
            }
        }

        // Exact prompt geometry (head_dim=16 -> ops path) with lengths [5, 9, 3]
        check_geometry(16, [5, 9, 3]);
        // Metal masked kernel geometry (head_dim=32) including len < conv_k - 1 (len=2)
        check_geometry(32, [5, 9, 2]);
    }

    /// Probe the FFI directly so the parity test below can never pass
    /// vacuously when the Metal backend is absent.
    fn gdn_prepare_backend_probe() -> bool {
        let qkv = rand_bf16(&[1, 1, 10240]);
        let a = rand_bf16(&[1, 1, 48]);
        let b = rand_bf16(&[1, 1, 48]);
        let conv = MxArray::random_normal(&[10240, 4], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let history = rand_bf16(&[3, 10240]);
        let scale = MxArray::full(&[48], Either::A(-0.5), Some(DType::Float32)).unwrap();
        let dt = MxArray::full(&[48], Either::A(0.0), Some(DType::Float32)).unwrap();
        let mut outputs = [std::ptr::null_mut(); 6];
        let ok = unsafe {
            sys::mlx_qwen4_gdn_prepare(
                qkv.handle.0,
                a.handle.0,
                b.handle.0,
                conv.handle.0,
                history.handle.0,
                scale.handle.0,
                dt.handle.0,
                true,
                true,
                outputs.as_mut_ptr(),
            )
        };
        if ok {
            for p in outputs {
                drop(MxArray::from_handle(p, "gdn_prepare probe"));
            }
        }
        ok
    }

    /// The fused `gdn_prepare` path (conv + SiLU + q|k|v + q/k L2-norm +
    /// decay/beta in one dispatch) must reproduce `prep_via_conv` + fused
    /// gating within bf16 rounding, and leave the conv history bit-identical
    /// (it is the raw bf16 input window tail on both paths).
    #[test]
    fn fused_gdn_prepare_matches_conv_prep_path() -> Result<()> {
        if !gdn_prepare_backend_probe() {
            return Ok(());
        }
        let mut net = kernel_geometry_net();
        let x1 = rand_bf16(&[1, 3, 64]);
        let x2 = rand_bf16(&[1, 5, 64]);
        // Near-zero input exercises the norm's small-Σ regime where eps
        // placement (Σ+ε vs Σ+d·ε) actually diverges — the review case.
        let x3 = rand_bf16(&[1, 4, 64]).mul_scalar(1e-3)?;

        // Exercise the genuine missing-sidecar fallback without process-global
        // switches: only fused prepare needs this cached scale.
        let prepare_scale = net.gdn_scale_f32.take();
        assert!(
            prepare_scale.is_some(),
            "fixture must support fused prepare"
        );
        let mut cache_a = ArraysCache::new(2);
        let out_a1 = net.forward(&x1, None, Some(&mut cache_a), true)?;
        let out_a2 = net.forward(&x2, None, Some(&mut cache_a), true)?;
        let out_a3 = net.forward(&x3, None, Some(&mut cache_a), true)?;

        // Restore the same sidecar for the optimized path; all weights and
        // inputs remain shared with the independently constructed reference.
        net.gdn_scale_f32 = prepare_scale;
        let mut cache_b = ArraysCache::new(2);
        let out_b1 = net.forward(&x1, None, Some(&mut cache_b), true)?;
        let out_b2 = net.forward(&x2, None, Some(&mut cache_b), true)?;
        let out_b3 = net.forward(&x3, None, Some(&mut cache_b), true)?;

        for (step, (a, b)) in [(&out_a1, &out_b1), (&out_a2, &out_b2), (&out_a3, &out_b3)]
            .iter()
            .enumerate()
        {
            let diff = max_abs_diff(a, b);
            assert!(
                diff <= 0.1,
                "gdn_prepare vs conv-prep output diverged at step {step}: {diff}"
            );
        }
        let hdiff = max_abs_diff(cache_a.get(0).unwrap(), cache_b.get(0).unwrap());
        assert_eq!(
            hdiff, 0.0,
            "conv history must be bit-identical on both paths"
        );
        let sdiff = max_abs_diff(cache_a.get(1).unwrap(), cache_b.get(1).unwrap());
        assert!(
            sdiff <= 0.1,
            "gdn_prepare vs conv-prep recurrent state diverged: {sdiff}"
        );
        Ok(())
    }

    /// The loader evaluates the checkpoint tensors, not the constants the
    /// setters derive from them. A compiled step traced while one is still
    /// lazy bakes its producer ops into the tape and recomputes them on every
    /// replay, so the step graph must reach nothing that is input-free.
    #[test]
    fn compiled_gdn_step_recomputes_no_derived_constant() -> Result<()> {
        use crate::compiled_graph::{
            erase_compiled_graphs_matching, input_free_ops, invoke_compiled_graph,
        };
        if !gdn_prepare_backend_probe() {
            eprintln!(
                "SKIP compiled_gdn_step_recomputes_no_derived_constant: no Metal gdn_prepare"
            );
            return Ok(());
        }
        assert!(
            std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
            "compiled replay requires MLX_DISABLE_COMPILE to be unset"
        );
        let loaded = |shape: &[i64]| {
            let w = rand_bf16(shape);
            w.eval();
            w
        };
        let mut net = kernel_geometry_net_with(loaded);
        net.finalize_after_load()?;
        // A >= 64-token prefill takes the conv path, like a real prompt, so it
        // leaves the short-step constants untouched.
        let mut prefill = ArraysCache::new(2);
        net.forward(&loaded(&[1, 64, 64]), None, Some(&mut prefill), true)?
            .eval();
        let (Some(conv_state), Some(recurrent)) = (prefill.get(0), prefill.get(1)) else {
            return Err(Error::from_reason("prefill left no GDN state"));
        };
        conv_state.eval();
        recurrent.eval();
        let x = loaded(&[1, 8, 64]);
        let inputs = [&x, conv_state, recurrent];
        let fn_id = 0xDFC6_0000_0000_0001_u64;
        let mut builder = |graph_inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            let mut cache = ArraysCache::new(2);
            cache.set(0, graph_inputs[1].clone())?;
            cache.set(1, graph_inputs[2].clone())?;
            let y = net.forward(&graph_inputs[0], None, Some(&mut cache), true)?;
            let (Some(c), Some(r)) = (cache.get(0), cache.get(1)) else {
                return Err(Error::from_reason("step left no GDN state"));
            };
            Ok(vec![y, c.clone(), r.clone()])
        };
        for call in 0..2 {
            let out = invoke_compiled_graph(fn_id, &inputs, 3, true, &mut builder)?
                .ok_or_else(|| Error::from_reason("compiled GDN step failed"))?;
            let (count, names) = input_free_ops(&out.iter().collect::<Vec<_>>(), &inputs);
            assert_eq!(count, 0, "call {call} recomputes lazy constants: {names}");
        }
        erase_compiled_graphs_matching(u64::MAX, fn_id);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_prepare_raw(
        qkv: &MxArray,
        a: &MxArray,
        b: &MxArray,
        conv: &MxArray,
        history: &MxArray,
        scale: &MxArray,
        dt: &MxArray,
        mean_eps: bool,
        beta_input_dtype: bool,
    ) -> [MxArray; 6] {
        let mut outputs = [std::ptr::null_mut(); 6];
        let ok = unsafe {
            sys::mlx_qwen4_gdn_prepare(
                qkv.handle.0,
                a.handle.0,
                b.handle.0,
                conv.handle.0,
                history.handle.0,
                scale.handle.0,
                dt.handle.0,
                mean_eps,
                beta_input_dtype,
                outputs.as_mut_ptr(),
            )
        };
        assert!(ok, "gdn_prepare rejected valid inputs");
        outputs.map(|p| MxArray::from_handle(p, "gdn_prepare").unwrap())
    }

    fn raw_bits(x: &MxArray) -> Vec<u32> {
        x.eval();
        if x.dtype().unwrap() == DType::Float32 {
            x.to_float32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        } else {
            x.to_uint16_native()
                .unwrap()
                .into_iter()
                .map(u32::from)
                .collect()
        }
    }

    /// The prepare kernel reads qkv/a/b/history through their strides. Feeding
    /// it the production split views (qkvz and ba sections, a column slice of
    /// a wider history) must give the same bits as feeding row-contiguous
    /// copies of the same values.
    #[test]
    fn gdn_prepare_strided_inputs_bit_identical() -> Result<()> {
        if !gdn_prepare_backend_probe() {
            eprintln!("SKIP gdn_prepare_strided_inputs_bit_identical: no Metal gdn_prepare");
            return Ok(());
        }
        let conv = MxArray::random_normal(&[10240, 4], 0.0, 0.3, Some(DType::Float32))?;
        let scale = MxArray::random_normal(&[48], -0.5, 0.2, Some(DType::Float32))?;
        let dt = MxArray::random_normal(&[48], 0.0, 0.5, Some(DType::Float32))?;
        let mut cases = 0;
        for t in [1i64, 3, 8, 63] {
            let qkvz = rand_bf16(&[1, t, 16384]);
            let qkv = qkvz.split_sections(&[10240], 2)?[0].clone();
            let history = rand_bf16(&[3, 16384]).slice_axis(1, 0, 10240)?;
            for ba_dtype in [DType::BFloat16, DType::Float32] {
                let ba = MxArray::random_normal(&[1, t, 96], 0.0, 2.0, Some(DType::Float32))?
                    .astype(ba_dtype)?;
                let parts = ba.split_sections(&[48], 2)?;
                let (b, a) = (parts[0].clone(), parts[1].clone());
                let dense = [&qkv, &a, &b, &history].map(|x| {
                    let y = x.deep_copy().unwrap();
                    y.eval();
                    y
                });
                for mean_eps in [false, true] {
                    for beta_input_dtype in [false, true] {
                        let strided = gdn_prepare_raw(
                            &qkv,
                            &a,
                            &b,
                            &conv,
                            &history,
                            &scale,
                            &dt,
                            mean_eps,
                            beta_input_dtype,
                        );
                        let reference = gdn_prepare_raw(
                            &dense[0],
                            &dense[1],
                            &dense[2],
                            &conv,
                            &dense[3],
                            &scale,
                            &dt,
                            mean_eps,
                            beta_input_dtype,
                        );
                        for (i, (x, y)) in strided.iter().zip(reference.iter()).enumerate() {
                            assert_eq!(
                                raw_bits(x),
                                raw_bits(y),
                                "output {i} differs: T={t} ba={ba_dtype:?} mean_eps={mean_eps} beta_input_dtype={beta_input_dtype}"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 32);
        Ok(())
    }

    /// A random q4k `[n, k]` projection (uint8 (sc, m) scales, f16 (d, dmin));
    /// the same seed gives the same bytes.
    fn q4k(n: i64, k: i64, seed: u32) -> QuantizedLinear {
        let mut st = seed;
        let mut lcg = move || {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            st
        };
        let words: Vec<u32> = (0..n * k / 8).map(|_| lcg()).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|_| (lcg() % 48 + 1) as u8).collect();
        let half_scales = [0x2800u16, 0x2c00, 0x3000, 0x3200];
        let biases: Vec<u16> = (0..n * k / 128)
            .map(|_| half_scales[(lcg() as usize) % half_scales.len()])
            .collect();
        QuantizedLinear::new(
            MxArray::from_uint32(&words, &[n, k / 8]).unwrap(),
            MxArray::from_uint8(&scales, &[n, k / 16]).unwrap(),
            Some(MxArray::from_float16(&biases, &[n, k / 128]).unwrap()),
            None,
            32,
            4,
            "q4k".to_string(),
        )
    }

    #[test]
    fn split_ba_m8_merge_is_bitwise_and_other_widths_keep_originals() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } || unsafe { sys::mlx_default_device() } != 1 {
            return Ok(());
        }
        let b = LinearProj::Quantized(q4k(32, 2048, 71));
        let a = LinearProj::Quantized(q4k(32, 2048, 73));
        let merged = merge_split_ba_m8(&b, &a)?.expect("target pair merges");
        for m in [1, 7, 8, 33] {
            let values = (0..m * 2048)
                .map(|i| ((i as f32) * 0.031).sin() * 0.125)
                .collect::<Vec<_>>();
            let x = MxArray::from_float32(&values, &[1, m, 2048])?.astype(DType::BFloat16)?;
            let result = project_split_ba_m8(&x, Some(&merged))?;
            assert_eq!(result.is_some(), m == 8);
            let original_b = b.forward(&x)?;
            let original_a = a.forward(&x)?;
            let (actual_b, actual_a) = result.unwrap_or((
                merged.slice_rows(0, 32)?.forward(&x)?,
                merged.slice_rows(32, 64)?.forward(&x)?,
            ));
            MxArray::eval_arrays(&[&actual_b, &actual_a, &original_b, &original_a])?;
            assert_eq!(
                actual_b.to_uint16_native()?,
                original_b.to_uint16_native()?,
                "b M{m}"
            );
            assert_eq!(
                actual_a.to_uint16_native()?,
                original_a.to_uint16_native()?,
                "a M{m}"
            );
        }
        Ok(())
    }

    /// Kernel geometry on a 256-wide hidden (whole K-quant super-blocks) with
    /// q4k in_proj_qkvz (16384 rows) and in_proj_ba (96 rows); `tile_qkvz`
    /// repacks the qkvz into Tiled64 before it is installed, as the loader does.
    fn quantized_in_proj_net(tile_qkvz: bool) -> GatedDeltaNet {
        let hidden = 256i64;
        let config = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: hidden as i32,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 48,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };
        // Deterministic dense sidecars so both nets share every weight.
        let dense = |shape: &[i64], seed: u32| {
            let n: i64 = shape.iter().product();
            let mut st = seed;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((st >> 16) as i32 - 32_768) as f32 / 32_768.0 * 0.3
                })
                .collect();
            MxArray::from_float32(&values, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        };
        let mut net = GatedDeltaNet::new(&config).unwrap();
        let mut qkvz = q4k(16384, hidden, 31);
        if tile_qkvz {
            assert!(qkvz.tile_kquant_layout().unwrap());
        }
        net.set_quantized_in_proj_qkvz(qkvz);
        net.set_quantized_in_proj_ba(q4k(96, hidden, 32));
        net.set_conv1d_weight(&dense(&[10240, 1, 4], 33), DType::BFloat16)
            .unwrap();
        net.set_norm_weight(&dense(&[128], 34), DType::BFloat16)
            .unwrap();
        net.set_out_proj_weight(&dense(&[hidden, 6144], 35))
            .unwrap();
        net.set_dt_bias(&dense(&[48], 36));
        net.set_a_log(&dense(&[48], 37)).unwrap();
        net.finalize_after_load().unwrap();
        net
    }

    /// A Tiled64 `in_proj_qkvz` only merges with a tiled `in_proj_ba`, whose 96
    /// rows are not whole tiles: `finalize_after_load` must zero-pad it to 128
    /// rows, merge into one 16512-row projection, and `forward` must drop the
    /// padding so the block reproduces the all-row-major merge for the M = 1
    /// decode, the M = 8 verify and a 64-token prefill.
    #[test]
    fn tiled_qkvz_pads_and_merges_row_major_ba() -> Result<()> {
        let reference = quantized_in_proj_net(false);
        let tiled = quantized_in_proj_net(true);
        let merged_rm = reference
            .in_proj_qkvz_ba_q
            .as_ref()
            .expect("row-major pair merges");
        assert_eq!(merged_rm.packed_out_features()?, 16384 + 96);
        let merged_t = tiled
            .in_proj_qkvz_ba_q
            .as_ref()
            .expect("tiled qkvz + padded ba must merge into one projection");
        assert_eq!(merged_t.packed_out_features()?, 16384 + 128);
        assert_eq!(tiled.in_proj_ba.packed_out_features()?, 128);
        assert!(matches!(
            &tiled.in_proj_ba,
            LinearProj::Quantized(ql) if ql.is_kquant_tiled()
        ));
        assert!(matches!(
            merged_t,
            LinearProj::Quantized(ql) if ql.is_kquant_tiled()
        ));
        // SAFETY: nullary predicate that catches internally.
        let gpu = unsafe { sys::mlx_metal_is_available() };
        for t in [1i64, 8, 64] {
            let mut st = 40 + t as u32;
            let n = t * 256;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((st >> 16) as i32 - 32_768) as f32 / 32_768.0
                })
                .collect();
            let x = MxArray::from_float32(&values, &[1, t, 256])?.astype(DType::BFloat16)?;
            let out_ref = reference.forward(&x, None, None, true)?;
            let out = tiled.forward(&x, None, None, true)?;
            assert_eq!(out.shape()?.as_ref(), &[1, t, 256]);
            let diff = max_abs_diff(&out, &out_ref);
            // T = 64 takes the prefill matmul, the same kernel in both
            // layouts (and the CPU reference always is): bit-identical. The
            // GPU M = 1 / M = 8 routes change kernels with the layout, so the
            // in_proj differs at bf16 rounding and the block follows.
            if !gpu || t == 64 {
                assert_eq!(diff, 0.0, "T={t}: padded merge changed the block output");
            } else {
                let peak = out_ref
                    .astype(DType::Float32)?
                    .to_float32()?
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()));
                assert!(
                    diff <= 3e-2 * peak,
                    "T={t}: padded merge off by {diff} of peak {peak}"
                );
            }
        }
        Ok(())
    }

    /// bf16 values `k / 16`, `k` in `-2..=2`: every product and partial sum
    /// of a 64-wide dot product is exact in f32, so a projection of these
    /// gives the same bits at any GEMM M (row count) and tiling.
    fn exact_bf16(shape: &[i64], seed: u32) -> MxArray {
        let n: i64 = shape.iter().product();
        let mut st = seed;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((st >> 16) % 5) as f32 / 16.0 - 2.0 / 16.0
            })
            .collect();
        MxArray::from_float32(&values, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    /// The production geometry with f32 `dt_bias`/`a_log` (a GGUF load keeps
    /// them bf16: `gdn_bf16_gates_tests`), compact tiled heads or the repeat-expanded map. The
    /// input projections are exact (`exact_bf16`), so a forward of the
    /// first `c` rows of a window sees the same projected rows as the whole
    /// window: GEMM M cannot change the recurrence inputs.
    fn capture_net(tiled: bool) -> GatedDeltaNet {
        capture_net_heads(48, tiled)
    }

    fn capture_net_heads(hv: i64, tiled: bool) -> GatedDeltaNet {
        let mut net = kernel_geometry_net_heads(hv, rand_bf16);
        net.set_in_proj_qkvz_weight(&exact_bf16(&[4096 + 2 * hv * 128, 64], 11))
            .unwrap();
        net.set_in_proj_ba_weight(&exact_bf16(&[2 * hv, 64], 12))
            .unwrap();
        net.tiled_gguf_layout = tiled;
        net.set_dt_bias(&MxArray::random_normal(&[hv], 0.0, 0.3, Some(DType::Float32)).unwrap());
        net.set_a_log(&MxArray::random_normal(&[hv], 0.0, 0.3, Some(DType::Float32)).unwrap())
            .unwrap();
        net
    }

    fn copy_cache(cache: &ArraysCache) -> ArraysCache {
        let mut copy = ArraysCache::new(2);
        for slot in 0..2 {
            if let Some(x) = cache.get(slot) {
                copy.set(slot, x.clone()).unwrap();
            }
        }
        copy
    }

    fn aliases(a: &MxArray, b: &MxArray) -> bool {
        MxArray::eval_arrays(&[a, b]).unwrap();
        unsafe { sys::mlx_array_aliases(a.as_raw_ptr(), b.as_raw_ptr()) }
    }

    /// Capturing at rows of a prefill window keeps the layer's route and
    /// bits: the output and both cache slots match the uncaptured forward
    /// bit for bit on every prompt route (fused prefill, complete core,
    /// prepare + fused tail, prepare + chain), cold and with a carried
    /// state. Each capture owns its buffers (the window-end capture is not
    /// the live cache array), stays unchanged after a later chunk, and its
    /// conv history is the raw rows the cache holds after a forward of rows
    /// `[0, c)`; its state equals that forward's whenever the shorter window
    /// takes the same route (bit for bit, except to bf16 rounding on the chain
    /// route).
    #[test]
    fn capture_keeps_route_bits_and_owns_state() -> Result<()> {
        use crate::models::qwen3_5::gated_delta::tests::{
            fused_complete_launches, fused_prefill_launches, fused_step_launches,
        };
        if !gdn_prepare_backend_probe() {
            eprintln!("SKIP capture_keeps_route_bits_and_owns_state: no fused GDN kernels");
            return Ok(());
        }
        // Each fused route's entry declines when its pipeline cannot hold a
        // value head per threadgroup on this GPU; the forward then falls back.
        let host_runs = |route: GdnRoute| match route {
            GdnRoute::FusedPrefill => prefill_fused_enabled() && fused_prefill_launches(),
            GdnRoute::Complete => fused_complete_launches(),
            GdnRoute::FusedTail => fused_step_launches(),
            GdnRoute::Chain => true,
        };
        let cases: [(bool, usize, &[usize], GdnRoute); 7] = [
            (
                false,
                80,
                &[1, 2, 3, 16, 33, 64, 79, 80],
                GdnRoute::FusedPrefill,
            ),
            (true, 80, &[2, 31, 32, 63, 64, 80], GdnRoute::FusedPrefill),
            (true, 16, &[1, 2, 3, 8, 15, 16], GdnRoute::Complete),
            (false, 16, &[1, 2, 3, 8, 15, 16], GdnRoute::FusedTail),
            (true, 24, &[1, 3, 17, 24], GdnRoute::FusedTail),
            (false, 40, &[1, 3, 32, 39, 40], GdnRoute::Chain),
            (true, 63, &[1, 16, 32, 62, 63], GdnRoute::Chain),
        ];
        for (tiled, t, rows, intended) in cases {
            let net = capture_net(tiled);
            for warm in [false, true] {
                let tag = format!("tiled={tiled} T={t} warm={warm}");
                let mut base = ArraysCache::new(2);
                if warm {
                    net.forward(&exact_bf16(&[1, 5, 64], 13), None, Some(&mut base), true)?;
                    MxArray::eval_arrays(&[base.get(0).unwrap(), base.get(1).unwrap()])?;
                }
                let x = exact_bf16(&[1, t as i64, 64], 14 + t as u32);
                x.eval();
                take_gdn_routes();
                let mut plain = copy_cache(&base);
                let out_plain = net.forward(&x, None, Some(&mut plain), true)?;
                let host_route = take_gdn_routes();
                if host_runs(intended) {
                    assert_eq!(host_route, [intended], "{tag}: uncaptured route");
                } else {
                    assert_eq!(host_route.len(), 1, "{tag}: uncaptured route");
                    assert_ne!(host_route, [intended], "{tag}: declined route ran");
                }
                let route = host_route[0];
                let mut live = copy_cache(&base);
                let (out, captured) = net.forward_captured(
                    &x,
                    Some(&mut live),
                    true,
                    None,
                    &GdnCapturePlan { rows },
                )?;
                assert_eq!(take_gdn_routes(), [route], "{tag}: captured route");
                assert_eq!(raw_bits(&out), raw_bits(&out_plain), "{tag}: output");
                for slot in 0..2 {
                    assert_eq!(
                        raw_bits(live.get(slot).unwrap()),
                        raw_bits(plain.get(slot).unwrap()),
                        "{tag}: cache slot {slot}"
                    );
                }
                assert_eq!(captured.len(), rows.len(), "{tag}");
                let mut fingerprints = Vec::new();
                let mut same_route = 0;
                for (&c, capture) in rows.iter().zip(&captured) {
                    let tag = format!("{tag} c={c}");
                    assert!(
                        !aliases(&capture.rec, live.get(1).unwrap())
                            && !aliases(&capture.conv, live.get(0).unwrap()),
                        "{tag}: capture shares the live cache buffer"
                    );
                    let mut truncated = copy_cache(&base);
                    net.forward(
                        &x.slice_axis(1, 0, c as i64)?,
                        None,
                        Some(&mut truncated),
                        true,
                    )?;
                    let truncated_route = take_gdn_routes();
                    assert_eq!(
                        capture.conv.shape()?.as_ref(),
                        truncated.get(0).unwrap().shape()?.as_ref(),
                        "{tag}"
                    );
                    assert_eq!(
                        raw_bits(&capture.conv),
                        raw_bits(truncated.get(0).unwrap()),
                        "{tag}: conv history"
                    );
                    if truncated_route == [route] {
                        let reference = truncated.get(1).unwrap();
                        if route == GdnRoute::Chain {
                            // A chain capture inside a 64-row chunk comes from other
                            // chunk products than the shorter forward: measured 0.5%
                            // of the state range (about one bf16 step), not bitwise.
                            let f32s = |v: &MxArray| v.astype(DType::Float32)?.to_float32();
                            let (got, want) = (f32s(&capture.rec)?, f32s(reference)?);
                            let range = want.iter().fold(0f32, |m, y| m.max(y.abs()));
                            let diff = got
                                .iter()
                                .zip(want.iter())
                                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                            assert!(
                                diff <= range / 64.0,
                                "{tag}: state vs the truncated forward: {diff:e} > {range:e}/64"
                            );
                        } else {
                            assert_eq!(
                                raw_bits(&capture.rec),
                                raw_bits(reference),
                                "{tag}: state vs the truncated forward"
                            );
                        }
                        same_route += 1;
                    }
                    fingerprints.push((raw_bits(&capture.conv), raw_bits(&capture.rec)));
                }
                eprintln!(
                    "GDN_CAPTURE {tag} route={route:?}: state checked against {same_route}/{} truncated forwards",
                    rows.len()
                );
                assert!(same_route > 0, "{tag}: no truncated forward kept the route");
                net.forward(&rand_bf16(&[1, 70, 64]), None, Some(&mut live), true)?;
                MxArray::eval_arrays(&[live.get(0).unwrap(), live.get(1).unwrap()])?;
                take_gdn_routes();
                for ((&c, capture), (conv, rec)) in rows.iter().zip(&captured).zip(&fingerprints) {
                    assert_eq!(&raw_bits(&capture.conv), conv, "{tag} c={c}: conv moved");
                    assert_eq!(&raw_bits(&capture.rec), rec, "{tag} c={c}: state moved");
                }
            }
        }
        Ok(())
    }

    /// A capture plan is a prefill contract: masked or taped forwards and
    /// out-of-range or unordered rows are refused, and an empty plan is the
    /// plain forward.
    #[test]
    fn capture_refuses_off_contract_windows() -> Result<()> {
        let net = kernel_geometry_net();
        let x = rand_bf16(&[1, 8, 64]);
        let mut cache = ArraysCache::new(2);
        for rows in [&[0usize][..], &[9], &[4, 4], &[5, 2]] {
            assert!(
                net.forward_captured(&x, Some(&mut cache), true, None, &GdnCapturePlan { rows })
                    .is_err(),
                "rows {rows:?}"
            );
        }
        let mut taped = None;
        let mut captured = Vec::new();
        assert!(
            net.forward_with_tape_impl(
                &x,
                None,
                None,
                Some(&mut ArraysCache::new(2)),
                true,
                Some(&mut taped),
                None,
                Some((&GdnCapturePlan { rows: &[4] }, &mut captured)),
                None,
                None,
            )
            .is_err()
        );
        let mask = MxArray::ones(&[1, 8], Some(DType::Float32))?;
        assert!(
            net.forward_with_tape_impl(
                &x,
                Some(&mask),
                None,
                Some(&mut ArraysCache::new(2)),
                true,
                None,
                None,
                Some((&GdnCapturePlan { rows: &[4] }, &mut captured)),
                None,
                None,
            )
            .is_err()
        );
        let mut plain = ArraysCache::new(2);
        let mut empty = ArraysCache::new(2);
        let out = net.forward(&x, None, Some(&mut plain), true)?;
        let (got, none) = net.forward_captured(
            &x,
            Some(&mut empty),
            true,
            None,
            &GdnCapturePlan { rows: &[] },
        )?;
        assert!(none.is_empty());
        assert_eq!(raw_bits(&got), raw_bits(&out));
        assert_eq!(
            raw_bits(empty.get(1).unwrap()),
            raw_bits(plain.get(1).unwrap())
        );
        Ok(())
    }

    /// The DFlash2 verify's layer: a cache with the pending tape (slots
    /// 4..9) takes the deferred complete kernel for tiled and grouped heads,
    /// 27B (Hv 48) and 35B (Hv 32) geometry. With the zero tape and keep 0
    /// its cells keep the incoming bits; on the 27B its output equals the
    /// same net's ordinary route (grouped: `qwen4_gdn_prepare` and the fused
    /// tail) and folding its own tape at keep 8 lands on that route's
    /// window-final state and conv history, bit for bit. The 35B's ordinary
    /// grouped route is the MLX conv chain, whose q/k norm and decay round
    /// differently from the prepare arithmetic the kernel shares (up to one
    /// bf16 / f32 step): there the output and the fold match it only to that
    /// rounding. Mutation G2 (the gate refuses grouped heads) turns it red.
    #[test]
    fn deferred_verify_takes_the_complete_kernel_for_both_orders() -> Result<()> {
        if !gdn_prepare_backend_probe() {
            eprintln!("SKIP deferred_verify_takes_the_complete_kernel_for_both_orders");
            return Ok(());
        }
        for (hv, tiled) in [(48i64, false), (48, true), (32, false)] {
            let tag = format!("hv={hv} tiled={tiled}");
            let net = capture_net_heads(hv, tiled);
            let width = 4096 + hv * 128;
            let mut base = ArraysCache::new(2);
            net.forward(&exact_bf16(&[1, 5, 64], 13), None, Some(&mut base), true)?;
            MxArray::eval_arrays(&[base.get(0).unwrap(), base.get(1).unwrap()])?;
            let x = exact_bf16(&[1, 8, 64], 21);
            let mut plain = copy_cache(&base);
            let out_plain = net.forward(&x, None, Some(&mut plain), true)?;
            take_gdn_routes();
            let deferred = |pending: [MxArray; 6],
                            x: &MxArray|
             -> Result<(MxArray, ArraysCache, GdnLayerTape)> {
                let mut cache = ArraysCache::new(10);
                cache.set(0, base.get(0).unwrap().clone())?;
                cache.set(1, base.get(1).unwrap().clone())?;
                cache.set(2, MxArray::zeros(&[1, hv, 128, 128], Some(DType::Float32))?)?;
                cache.set(3, MxArray::zeros(&[3, width], Some(DType::BFloat16))?)?;
                for (slot, array) in pending.into_iter().enumerate() {
                    cache.set(4 + slot, array)?;
                }
                let mut tape = None;
                unsafe { sys::mlx_test_kquant_counting(true) };
                let run = net
                    .forward_with_tape(x, None, Some(&mut cache), true, Some(&mut tape))
                    .and_then(|out| {
                        MxArray::eval_arrays(&[
                            &out,
                            cache.get(0).unwrap(),
                            cache.get(1).unwrap(),
                        ])?;
                        Ok(out)
                    });
                let launches =
                    unsafe { sys::mlx_test_kquant_family_count(c"gdn_fused_deferred".as_ptr()) };
                unsafe { sys::mlx_test_kquant_counting(false) };
                let out = run?;
                assert_eq!(launches, 1, "{tag}: one deferred launch");
                assert_eq!(take_gdn_routes(), [GdnRoute::Complete], "{tag}: route");
                Ok((out, cache, tape.expect("verify tape")))
            };
            let zero = [
                MxArray::zeros(&[1, 1, 16, 128], Some(DType::BFloat16))?,
                MxArray::zeros(&[1, 1, hv, 128], Some(DType::BFloat16))?,
                MxArray::zeros(&[1, 1, hv], Some(DType::Float32))?,
                MxArray::zeros(&[1, 1, hv], Some(DType::BFloat16))?,
                MxArray::zeros(&[1, 1, width], Some(DType::BFloat16))?,
                MxArray::from_int32(&[0], &[1])?,
            ];
            let close = |got: &MxArray, want: &MxArray, what: &str| -> Result<()> {
                if hv == 48 {
                    assert_eq!(raw_bits(got), raw_bits(want), "{tag}: {what}");
                    return Ok(());
                }
                let f32s = |v: &MxArray| v.astype(DType::Float32)?.to_float32();
                let (got, want) = (f32s(got)?, f32s(want)?);
                let range = want.iter().fold(0f32, |m, y| m.max(y.abs()));
                let diff = got
                    .iter()
                    .zip(want.iter())
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                assert!(
                    diff <= range / 64.0,
                    "{tag}: {what}: {diff:e} > {range:e}/64"
                );
                Ok(())
            };
            let (out, cache, tape) = deferred(zero, &x)?;
            close(&out, &out_plain, "output")?;
            for slot in 0..2 {
                assert_eq!(
                    raw_bits(cache.get(slot).unwrap()),
                    raw_bits(base.get(slot).unwrap()),
                    "{tag}: keep 0 leaves slot {slot}"
                );
            }
            assert_eq!(
                tape.qkv.shape()?.as_ref(),
                [1, 8, width],
                "{tag}: raw inputs"
            );
            let pending = [
                tape.kernel.k.clone(),
                tape.kernel.v.clone(),
                tape.kernel.g.clone(),
                tape.kernel.beta.clone(),
                tape.qkv.clone(),
                MxArray::from_int32(&[8], &[1])?,
            ];
            let (_, folded, _) = deferred(pending, &exact_bf16(&[1, 8, 64], 22))?;
            for slot in 0..2 {
                close(
                    folded.get(slot).unwrap(),
                    plain.get(slot).unwrap(),
                    &format!("the fold of 8 rows vs the window-final slot {slot}"),
                )?;
            }
        }
        Ok(())
    }

    /// End-to-end Wave B2a CUDA tree GDN parity test:
    /// 1. Prime prefix state (5 tokens).
    /// 2. Run cycle 1 on a 6-node branching tree (`parent = [-1, 0, 1, 0, 3, 4]`)
    ///    using `forward_with_tape_deferred_tree` (`prepare_tree` + `tree_step_deferred`).
    /// 3. Commit non-contiguous branch `[0, 3, 4, 5]` via `commit_tree_dflash2_deferred`.
    /// 4. Run cycle 2 (a 3-node chain) folding cycle 1's `prev_indices = Some([0, 3, 4, 5])`
    ///    through `forward_with_tape_deferred_tree` (both `tree = Some` and `tree = None` paths).
    /// 5. Settle via `settle_flat_dflash2_deferred`.
    ///
    /// Every output row on `[0, 3, 4, 5]` + cycle 2 and the final settled
    /// `(conv_state, recurrent_state)` must match running the exact accepted
    /// branch sequentially as a 1D chain.
    #[test]
    fn tree_forward_deferred_commit_and_settle_match_branch_chain() -> Result<()> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(());
        }
        use crate::models::qwen3_5::layer_cache::{
            CudaPendingGdn, Qwen3_5LayerCache, commit_tree_dflash2_deferred,
            settle_flat_dflash2_deferred, snapshot_flat_dflash2,
        };
        use crate::models::qwen3_5::verify_tree::TreeDeviceViews;

        let net = capture_net_heads(48, false);
        let prefix = exact_bf16(&[1, 5, 64], 101);
        let mut ref_cache = ArraysCache::new(2);
        net.forward(&prefix, None, Some(&mut ref_cache), true)?;
        MxArray::eval_arrays(&[ref_cache.get(0).unwrap(), ref_cache.get(1).unwrap()])?;

        let mut tree_caches = vec![Qwen3_5LayerCache::Linear(copy_cache(&ref_cache))];
        let mut pending: Vec<Option<CudaPendingGdn>> = vec![None];

        // 6-node branching tree:
        //   0 (root) -> 1 -> 2
        //   0 (root) -> 3 -> 4 -> 5
        let x_tree = exact_bf16(&[1, 6, 64], 202);
        let tv1 = TreeDeviceViews {
            depths_i32: MxArray::from_int32(&[0, 1, 2, 1, 2, 3], &[6])?,
            mask_u32: MxArray::from_uint32(&[1, 3, 7, 9, 25, 57], &[6])?,
            parent_i32: MxArray::from_int32(&[-1, 0, 1, 0, 3, 4], &[6])?,
            conv_taps_i32: MxArray::from_int32(
                &[
                    -3, -2, -1, 0, // node 0
                    -2, -1, 0, 1, // node 1
                    -1, 0, 1, 2, // node 2
                    -2, -1, 0, 3, // node 3
                    -1, 0, 3, 4, // node 4
                    0, 3, 4, 5, // node 5
                ],
                &[6, 4],
            )?,
            tree_len: 6,
        };

        // Cycle 1: tree forward with deferred fold (pending = None).
        let snap1 = snapshot_flat_dflash2(&tree_caches)?;
        let mut tape1 = None;
        let out_tree1 = net.forward_with_tape_deferred_tree(
            &x_tree,
            None,
            tree_caches[0].as_arrays_cache_mut(),
            true,
            Some(&mut tape1),
            Some(pending[0].as_ref()),
            Some(&tv1),
        )?;
        pending[0] = None;

        // Commit non-contiguous accepted branch [0, 3, 4, 5].
        let accepted1 = [0usize, 3, 4, 5];
        let tapes1 = vec![tape1];
        commit_tree_dflash2_deferred(
            &mut tree_caches,
            &snap1,
            &tapes1,
            &accepted1,
            false,
            false,
            &mut pending,
            "tree_gdn_test_cycle1",
        )?;
        assert!(
            pending[0]
                .as_ref()
                .is_some_and(|p| p.keep == 4 && p.prev_indices.is_some())
        );

        // Cycle 2: 3-node chain with `tree = None` (verifying `fused_step_deferred`
        // handles non-contiguous `prev_indices` from cycle 1!).
        let x_chain2 = exact_bf16(&[1, 3, 64], 303);
        let snap2 = snapshot_flat_dflash2(&tree_caches)?;
        let mut tape2 = None;
        let out_tree2 = net.forward_with_tape_deferred_tree(
            &x_chain2,
            None,
            tree_caches[0].as_arrays_cache_mut(),
            true,
            Some(&mut tape2),
            Some(pending[0].as_ref()),
            None,
        )?;
        pending[0] = None;

        // Commit partial prefix [0, 1] of cycle 2, then settle!
        let tapes2 = vec![tape2];
        commit_tree_dflash2_deferred(
            &mut tree_caches,
            &snap2,
            &tapes2,
            &[0, 1],
            true,
            true,
            &mut pending,
            "tree_gdn_test_cycle2",
        )?;
        settle_flat_dflash2_deferred(&mut tree_caches, &mut pending)?;
        assert!(pending[0].is_none());

        // Reference: run accepted branch [0, 3, 4, 5] then [0, 1] of cycle 2 linearly.
        let branch_idx = MxArray::from_int32(&[0, 3, 4, 5], &[4])?;
        let x_branch1 = x_tree.take(&branch_idx, 1)?;
        let out_ref1 = net.forward(&x_branch1, None, Some(&mut ref_cache), true)?;
        let out_ref2 = net.forward(&x_chain2, None, Some(&mut copy_cache(&ref_cache)), true)?;
        let x_chain2_keep = x_chain2.slice_axis(1, 0, 2)?;
        net.forward(&x_chain2_keep, None, Some(&mut ref_cache), true)?;

        let out_tree1_branch = out_tree1.take(&branch_idx, 1)?;
        let max_diff = |a: &MxArray, b: &MxArray| -> Result<f32> {
            let af = a.astype(DType::Float32)?.to_float32()?;
            let bf = b.astype(DType::Float32)?.to_float32()?;
            Ok(af
                .iter()
                .zip(bf.iter())
                .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs())))
        };
        assert!(
            max_diff(&out_tree1_branch, &out_ref1)? <= 1e-2,
            "cycle 1 branch output mismatch"
        );
        assert!(
            max_diff(&out_tree2, &out_ref2)? <= 1e-2,
            "cycle 2 output after non-contiguous fold mismatch"
        );
        let settled_arrays = tree_caches[0].as_arrays_cache_mut().unwrap();
        assert_eq!(
            raw_bits(settled_arrays.get(0).unwrap()),
            raw_bits(ref_cache.get(0).unwrap()),
            "settled conv_state must match branch chain bit-for-bit"
        );
        assert!(
            max_diff(settled_arrays.get(1).unwrap(), ref_cache.get(1).unwrap())? <= 1e-4,
            "settled recurrent_state mismatch"
        );
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos"))]
mod wave_tests;

#[cfg(all(test, target_os = "macos"))]
mod join_tests;
