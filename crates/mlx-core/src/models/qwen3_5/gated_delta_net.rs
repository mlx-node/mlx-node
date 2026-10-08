use crate::array::{DType, MxArray};
use crate::nn::{Activations, Conv1d, Linear, RMSNormGated, rms_norm_scaled, rms_norm_unscaled};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::{
    GdnCompleteOut, GdnKernelTape, GdnPrologue, GdnTail, gated_delta_fused_complete,
    gated_delta_update, gated_delta_update_fused, gated_delta_update_with_tape,
};
use crate::models::quantized_linear::{DEFAULT_QUANT_MODE, LinearProj, QuantizedLinear};

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
///   * `final_state` / `final_conv` — the recurrent state `[1, Hv, Dv, Dk]`
///     and conv history `[K-1, W]` after the WHOLE window, when the verify
///     wrote them into the DFlash2 spare blob rows (compiled verify with
///     destinations; the layer records the history, the compiled unpack
///     the state). With the f32 carry the state equals the replay of every
///     window token, so a full accept adopts both instead of replaying.
///
/// All array fields are lazy `MxArray` clones (no eval, no copy) so recording
/// stays inside the fused lazy MLX graph.
#[derive(Clone)]
pub(crate) struct GdnLayerTape {
    pub kernel: GdnKernelTape,
    pub qkv: MxArray,
    pub conv_kernel_dim: i32,
    pub final_state: Option<MxArray>,
    pub final_conv: Option<MxArray>,
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
    let prefix = qkv.slice_axis(1, 0, accepted)?;
    let input = match snapshot {
        Some(state) => MxArray::concatenate(state, &prefix, 1)?,
        None => {
            let zeros = MxArray::zeros(
                &[qkv.shape_at(0)?, keep, qkv.shape_at(2)?],
                Some(qkv.dtype()?),
            )?;
            MxArray::concatenate(&zeros, &prefix, 1)?
        }
    };
    let total = input.shape_at(1)?;
    let parts = input.split_sections(&[total - keep], 1)?;
    Ok(parts[1].clone())
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
            },
            qkv: combine(|row| &row.qkv)?,
            conv_kernel_dim: first.conv_kernel_dim,
            final_state: None,
            final_conv: None,
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
            },
            qkv: select(&self.qkv)?,
            conv_kernel_dim: self.conv_kernel_dim,
            final_state: None,
            final_conv: None,
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
        let keep = (self.conv_kernel_dim - 1) as i64;
        if keep > 0 {
            cache.set(
                0,
                replay_conv_state(&self.qkv, snapshot_conv, keep, accepted_steps)?,
            )?;
        }
        Ok(())
    }
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

impl GatedDeltaNet {
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

    /// Depthwise conv kernel width — the caller needs it to rebuild
    /// [`GdnLayerTape`]s from compiled-graph outputs (`keep = kd - 1`).
    pub(crate) fn conv_kernel_dim(&self) -> i32 {
        self.conv_kernel_dim
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
    pub(crate) fn forward_with_tape(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        mut cache: Option<&mut ArraysCache>,
        use_kernel: bool,
        mut tape_sink: Option<&mut Option<GdnLayerTape>>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

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
        } else {
            None
        };

        let (qkv, z) = if let Some((qkv_proj, z_proj)) = &self.split_in_proj_qkv_z {
            (qkv_proj.forward(x)?, z_proj.forward(x)?)
        } else {
            let qkvz = if let Some((qkvz, _)) = &stacked {
                qkvz.clone()
            } else {
                self.in_proj_qkvz.forward(x)?
            };
            let qkvz_split = qkvz.split_sections(&[self.conv_dim as i64], 2)?;
            (qkvz_split[0].clone(), qkvz_split[1].clone())
        };

        let (b, a) = if let Some((b_proj, a_proj)) = &self.split_in_proj_b_a {
            (b_proj.forward(x)?, a_proj.forward(x)?)
        } else {
            let ba = if let Some((_, ba)) = &stacked {
                ba.clone()
            } else {
                self.in_proj_ba.forward(x)?
            };
            self.split_ba_padded(&ba)?
        };

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

        // Whole GDN core in ONE dispatch (prep + recurrence + gated norm +
        // z gate): the decode/verify geometry with compact tiled heads. Same
        // bits as the gdn_prepare -> fused step chain below, which stays the
        // fallback (as does the generic path after it).
        if let Some(complete) = self.forward_complete(
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
        ) {
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
            if let (Some(sink), Some(qkv)) = (tape_sink.take(), tape_qkv) {
                // The kernel's own history output (not the reshaped cache
                // view): a sibling of the gated output, so it is evaluated
                // with it and a full accept can adopt it in place.
                *sink = Some(GdnLayerTape {
                    kernel: complete.tape,
                    qkv,
                    conv_kernel_dim: self.conv_kernel_dim,
                    final_state: None,
                    final_conv: Some(complete.history),
                });
            }
            return self.out_proj.forward(&complete.out);
        }

        // Fully-fused prep: conv + SiLU + q|k|v split + q/k L2-norm + decay/beta
        // gating in ONE Metal dispatch (`mlx_qwen4_gdn_prepare` — hardcoded to
        // this family's 10240-wide 16k/48v×128 geometry). Keep its existing
        // decode/verify eligibility (seq_len < 64); decay and beta feed the
        // per-step recurrence directly, with Qwen3.5 rounding semantics below.
        let prepared = (|| -> Option<[MxArray; 6]> {
            if batch != 1
                || seq_len >= 64
                || self.conv_kernel_dim != 4
                || self.conv_dim != 10240
                || self.num_k_heads != 16
                || self.key_head_dim != 128
                || self.num_v_heads != 48
                || self.value_head_dim != 128
                || !use_kernel
                || !crate::engine::persistence::compiled_forward_backend_available()
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
            let mut outputs = [std::ptr::null_mut(); 6];
            if unsafe {
                // qwen3_5 semantics: mean-eps norm (rms_norm eps inside the
                // mean) and bf16 beta — bit-faithful to the conv fallback.
                sys::mlx_qwen4_gdn_prepare(
                    qkv.handle.0,
                    a.handle.0,
                    b.handle.0,
                    conv_w.handle.0,
                    history.handle.0,
                    scale.handle.0,
                    dt.handle.0,
                    true,
                    true,
                    outputs.as_mut_ptr(),
                )
            } {
                Some([
                    MxArray::from_handle(outputs[0], "gdn_prepare:q").ok()?,
                    MxArray::from_handle(outputs[1], "gdn_prepare:k").ok()?,
                    MxArray::from_handle(outputs[2], "gdn_prepare:v").ok()?,
                    MxArray::from_handle(outputs[3], "gdn_prepare:decay").ok()?,
                    MxArray::from_handle(outputs[4], "gdn_prepare:beta").ok()?,
                    MxArray::from_handle(outputs[5], "gdn_prepare:history").ok()?,
                ])
            } else {
                None
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
        } else {
            let (q, k, v) = self.prep_via_conv(
                &qkv,
                batch,
                seq_len,
                cache.as_deref_mut(),
                conv_state,
                use_kernel,
            )?;
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

        // Fused tail: recurrence + gated norm + z gate in one dispatch (the
        // production decode/verify geometry). Slot 2 of the cache, when a
        // caller provides it, is the buffer the new state is written into.
        {
            let mut kernel_sink: Option<GdnKernelTape> = None;
            let fused = gated_delta_update_fused(
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
            )?;
            if let Some((gated, new_state)) = fused {
                if let Some(cache) = cache {
                    cache.set(1, new_state)?;
                }
                if let (Some(sink), Some(kernel), Some(qkv)) =
                    (tape_sink.take(), kernel_sink, tape_qkv)
                {
                    *sink = Some(GdnLayerTape {
                        kernel,
                        qkv,
                        conv_kernel_dim: self.conv_kernel_dim,
                        final_state: None,
                        final_conv: None,
                    });
                }
                return self.out_proj.forward(&gated);
            }
        }

        let (y, new_state) = if tape_sink.is_some() {
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
                    final_state: None,
                    final_conv: None,
                });
            }
            result
        } else {
            if self.tiled_gguf_layout {
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
        self.out_proj.forward(&y_flat)
    }

    /// The whole GDN core as one dispatch (`gated_delta_fused_complete`):
    /// `gdn_prepare`'s contract (batch 1, 4-tap conv, 128-wide heads, bf16
    /// activations, the f32 conv/scale/dt sidecars) plus the fused tail's
    /// (unmasked, T <= 16, bf16 gate and norm weight, f32 state) and compact
    /// tiled heads (`hv % Hk`; standard checkpoints repeat-expand q/k and
    /// stay on the chain below). `None` leaves the chain to run.
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
    ) -> Option<GdnCompleteOut> {
        use crate::array::DType;
        if batch != 1
            || !(1..=16).contains(&seq_len)
            || mask.is_some()
            || !use_kernel
            || self.conv_kernel_dim != 4
            || self.key_head_dim != 128
            || self.value_head_dim != 128
            || !(self.tiled_gguf_layout || self.num_k_heads == self.num_v_heads)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return None;
        }
        let conv = self.conv1d_w4_f32.as_ref()?;
        let scale = self.gdn_scale_f32.as_ref()?;
        let dt_bias = self.dt_bias_f32.as_ref()?;
        let recurrent = cache.and_then(|c| c.get(1));
        if [qkv, z, a, b, self.norm.weight()]
            .iter()
            .any(|x| x.dtype().ok() != Some(DType::BFloat16))
            || recurrent.is_some_and(|s| s.dtype().ok() != Some(DType::Float32))
        {
            return None;
        }
        let history = match conv_state {
            Some(s) => s.squeeze(Some(&[0])).ok()?,
            None => MxArray::zeros(
                &[(self.conv_kernel_dim - 1) as i64, self.conv_dim as i64],
                Some(DType::BFloat16),
            )
            .ok()?,
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
                )
                .ok()?;
                &zero_state
            }
        };
        gated_delta_fused_complete(
            &GdnPrologue {
                qkv,
                a,
                b,
                conv,
                history: &history,
                scale,
                dt_bias,
                history_dst: cache.and_then(|c| c.get(3)),
            },
            state,
            &GdnTail {
                z,
                norm_weight: self.norm.weight(),
                eps: self.norm.eps(),
                state_dst: cache.and_then(|c| c.get(2)),
            },
        )
    }

    /// Generic prep path: depthwise conv (fused `window_conv` when possible)
    /// → SiLU → q|k|v split → head reshape → q/k RMS-norm+scale. Runs whenever
    /// the fully-fused `gdn_prepare` kernel is off-contract (batch > 1,
    /// seq ≥ 64, non-10240 geometry) or declines.
    fn prep_via_conv(
        &self,
        qkv: &MxArray,
        batch: i64,
        seq_len: i64,
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
                // Save last (kernel_size - 1) timesteps as new conv_state
                let total_len = conv_input.shape_at(1)?;
                let keep = (self.conv_kernel_dim - 1) as i64;
                if total_len >= keep {
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

        // Apply RMS norm scaling to q and k (matching Python exactly):
        //   inv_scale = head_k_dim^(-0.5)
        //   q = (inv_scale^2) * rms_norm(q, None, 1e-6)
        //   k = inv_scale * rms_norm(k, None, 1e-6)
        // The scale factors live in constant bf16 vectors so the norm+scale
        // is ONE fast_rms_norm dispatch per tensor (bf16 activations only —
        // other dtypes keep the scalar path to avoid a promote).
        let (q, k) = if q.dtype()? == crate::array::DType::BFloat16
            && k.dtype()? == crate::array::DType::BFloat16
        {
            (
                rms_norm_scaled(&q, &self.qk_norm_w_q, 1e-6)?,
                rms_norm_scaled(&k, &self.qk_norm_w_k, 1e-6)?,
            )
        } else {
            let inv_scale = (self.key_head_dim as f64).powf(-0.5);
            (
                rms_norm_unscaled(&q, 1e-6)?.mul_scalar(inv_scale * inv_scale)?,
                rms_norm_unscaled(&k, 1e-6)?.mul_scalar(inv_scale)?,
            )
        };
        Ok((q, k, v))
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
    pub fn set_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        // norm.weight may be stored as f32 in checkpoints for precision;
        // cast to the model's compute dtype — not to dt_bias's dtype
        // (GGUF keeps dt_bias f32 for the gating kernel's `float*` reads).
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
        let config = Qwen3_5Config {
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
            linear_num_value_heads: 48,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_cache_initial_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(true),
            persist_paged_cache: None,
            n_mtp_layers: 0,
        };
        let mut net = GatedDeltaNet::new(&config).unwrap();
        // key_dim*2 + value_dim*2 = 2048*2 + 6144*2 = 16384 rows out.
        net.set_in_proj_qkvz_weight(&weight(&[16384, 64])).unwrap();
        net.set_in_proj_ba_weight(&weight(&[96, 64])).unwrap();
        net.set_conv1d_weight(&weight(&[10240, 1, 4]), DType::BFloat16)
            .unwrap();
        net.set_norm_weight(&weight(&[128]), DType::BFloat16)
            .unwrap();
        net.set_out_proj_weight(&weight(&[64, 6144])).unwrap();
        net.set_dt_bias(&weight(&[48]));
        net.set_a_log(&weight(&[48])).unwrap();
        net
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

    /// Kernel geometry on a 256-wide hidden (whole K-quant super-blocks) with
    /// q4k in_proj_qkvz (16384 rows) and in_proj_ba (96 rows); `tile_qkvz`
    /// repacks the qkvz into Tiled64 before it is installed, as the loader does.
    fn quantized_in_proj_net(tile_qkvz: bool) -> GatedDeltaNet {
        let hidden = 256i64;
        let config = Qwen3_5Config {
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
            paged_cache_initial_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(true),
            persist_paged_cache: None,
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
}
