use super::*;

/// RAII guard that temporarily disables MLX CUDA Graph capture for large
/// prefill passes (`M > 16`) and restores the previous state on drop.
///
/// Delegates to the process-wide reference-counted
/// [`crate::accelerator::CudaGraphsGuard`] so concurrent prefill scopes across
/// models never race or prematurely re-enable CUDA Graphs.
pub(super) struct CudaGraphsGuard {
    _inner: Option<crate::accelerator::CudaGraphsGuard>,
}

impl CudaGraphsGuard {
    pub(super) fn disable_if(should_disable: bool) -> Self {
        Self {
            _inner: should_disable.then(crate::accelerator::CudaGraphsGuard::disable_for_prefill),
        }
    }
}

/// Pending CUDA deferred GDN recurrent-state update from the last committed
/// flat speculative verify cycle: the next verify folds `keep` rows of
/// `(k, v, g, beta)` into `cache.recurrent` in-register inside
/// `fused_step_deferred_with_gate` / `tree_step_deferred_with_gate` before
/// advancing its own verify rows.
#[derive(Clone)]
pub(in super::super) struct CudaPendingGdn {
    pub(in super::super) k: MxArray,
    pub(in super::super) v: MxArray,
    pub(in super::super) g: MxArray,
    pub(in super::super) beta: MxArray,
    pub(in super::super) keep: usize,
    /// `None` for contiguous prefix `0..keep`; `Some(int32[keep])` for
    /// non-contiguous accepted branches from a speculative tree.
    pub(in super::super) prev_indices: Option<MxArray>,
}

/// Per-layer GDN tape captured during a deferred flat verify pass so
/// `commit_deferred_verify` / `commit_tree_deferred` can roll back 2D
/// `[3, 10240]` `cache.conv` and arm `cuda_pending_gdn[i]` without ever
/// materializing intermediate speculative recurrent states.
#[derive(Clone)]
pub(in super::super) struct VerifyGdnLayerTape {
    pub(in super::super) qkv: MxArray,
    pub(in super::super) pre_conv: Option<MxArray>,
    pub(in super::super) k: MxArray,
    pub(in super::super) v: MxArray,
    pub(in super::super) g: MxArray,
    pub(in super::super) beta: MxArray,
}

/// Per-layer PLE tape captured during a tree verify pass so
/// `commit_tree_deferred` can rebuild exact `[9, 10240]` `cache.ple_conv`
/// along any accepted branch (contiguous or non-contiguous).
#[derive(Clone)]
pub(in super::super) struct VerifyPleLayerTape {
    pub(in super::super) pre_conv: Option<MxArray>,
    pub(in super::super) conv_in: MxArray,
}

/// Per-layer QSA full-attention tape captured during a tree verify pass so
/// `commit_tree_deferred` can trim or compact flat KV rows and append committed
/// 4-token indexer blocks.
#[derive(Clone)]
pub(in super::super) struct VerifyAttnLayerTape {
    pub(in super::super) pre_kv_len: usize,
    pub(in super::super) pre_index_tail: Vec<MxArray>,
    pub(in super::super) ik_rows: MxArray,
}

/// Lightweight per-layer rollback snapshot for a flat CUDA deferred verify
/// cycle (avoids building 4 full `DecoderState`s with 4×48 `LayerCache`s).
pub(in super::super) struct DeferredVerifyTicket {
    pub(in super::super) base_history_len: usize,
    pub(in super::super) rows: usize,
    pub(in super::super) gdn_tapes: Vec<Option<VerifyGdnLayerTape>>,
    /// Per-row snapshots of non-GDN mutable metadata (`ple_conv`, `flat_kv_rows`,
    /// `flat_index_blocks`, `index_tail`, `index_blocks`) indexed as `[row][layer]`.
    pub(in super::super) row_caches: Vec<Vec<LayerCache>>,
}

/// Lightweight per-layer commit ticket for a flat CUDA deferred tree verify
/// cycle (`T in 1..=16`).
pub(in super::super) struct TreeVerifyTicket {
    pub(in super::super) base_history_len: usize,
    pub(in super::super) tree_tokens: Vec<u32>,
    pub(in super::super) gdn_tapes: Vec<Option<VerifyGdnLayerTape>>,
    pub(in super::super) ple_tapes: Vec<Option<VerifyPleLayerTape>>,
    pub(in super::super) attn_tapes: Vec<Option<VerifyAttnLayerTape>>,
}

impl Decoder {
    pub(in super::super) fn clear_cuda_deferred_gdn(&mut self) {
        for slot in &mut self.cuda_pending_gdn {
            *slot = None;
        }
        self.cuda_verify_gdn_tape = None;
        self.cuda_verify_row_caches = None;
        self.cuda_verify_final_window = None;
        self.cuda_verify_final_normed = None;
    }

    /// Fold any remaining pending CUDA deferred GDN updates into
    /// `self.caches[i].recurrent` via `gdn::replay` and clear `cuda_pending_gdn`.
    /// Must be called on turn finish, `Drop`, or before any non-deferred step.
    pub(in super::super) fn settle_cuda_pending_gdn(&mut self) -> Result<()> {
        if !self.cuda_pending_gdn.iter().any(Option::is_some) {
            return Ok(());
        }
        let nh = self.config.linear_num_value_heads as i64;
        let kd = self.config.linear_key_head_dim as i64;
        let vd = self.config.linear_value_head_dim as i64;
        let tiled = self.gguf();
        let mut settled_states = Vec::new();
        for (i, slot) in self.cuda_pending_gdn.iter_mut().enumerate() {
            let Some(p) = slot.take() else {
                continue;
            };
            if p.keep == 0 {
                continue;
            }
            let start_state = match &self.caches[i].recurrent {
                Some(s) => s.clone(),
                None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
            };
            let (k, v, g, beta) = if let Some(ref idx) = p.prev_indices {
                (
                    p.k.take(idx, 1)?,
                    p.v.take(idx, 1)?,
                    p.g.take(idx, 1)?,
                    p.beta.take(idx, 1)?,
                )
            } else {
                (p.k, p.v, p.g, p.beta)
            };
            let settled =
                crate::kernels::gdn::replay(&k, &v, &g, &beta, &start_state, p.keep as i32, tiled)?
                    .ok_or_else(|| {
                        Error::from_reason("Qwen4 CUDA deferred GDN settle replay failed")
                    })?;
            self.caches[i].recurrent = Some(settled.clone());
            settled_states.push(settled);
        }
        if !settled_states.is_empty() {
            let refs: Vec<&MxArray> = settled_states.iter().collect();
            MxArray::eval_arrays_with_context(&refs, "qwen4::decoder::settle_gdn")?;
        }
        Ok(())
    }

    pub(in super::super) fn cuda_async_eval(arrays: &[&MxArray]) {
        if arrays.is_empty() {
            return;
        }
        let mut handles: Vec<*mut mlx_sys::mlx_array> = arrays
            .iter()
            .map(|a| a.as_raw_ptr())
            .filter(|&ptr| !unsafe { mlx_sys::mlx_array_is_available(ptr) })
            .collect();
        if handles.is_empty() {
            return;
        }
        unsafe {
            mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
        }
    }
    pub(in super::super) fn flush_cuda_ar_step(
        &self,
        next_token: &MxArray,
        logits: &MxArray,
        budget_forced: bool,
    ) {
        let mut completed: Vec<&MxArray> = self.pending_state_arrays();
        if let Some(taps) = &self.hidden_taps {
            completed.extend(taps.last_pass_arrays().iter());
        }
        if budget_forced {
            completed.push(logits);
        }
        completed.push(next_token);
        Self::cuda_async_eval(&completed);
    }
    pub(in super::super) fn sync_cuda_pending_states(&self) -> Result<()> {
        let mut completed: Vec<&MxArray> = self.pending_state_arrays();
        if let Some(taps) = &self.hidden_taps {
            completed.extend(taps.last_pass_arrays().iter());
        }
        if !completed.is_empty() {
            MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::end_decode")?;
        }
        Ok(())
    }
    pub(super) fn cuda_deferred_state(&self) -> bool {
        super::super::cuda_qwen4_fast_enabled() && self.weights.plan.direct && self.scope.is_empty()
    }

    /// Batched CUDA deferred GDN step for `verify_layer`: runs `gdn_prepare` ONCE
    /// across all `R` verify rows and `fused_step_deferred_with_gate` ONCE, folding
    /// any pending previous cycle's `prev_keep` accepted rows into `cache.recurrent`
    /// in-register and returning `(out [1, R, 6144], VerifyGdnLayerTape)` without
    /// materializing intermediate speculative recurrent states.
    pub(super) fn cuda_verify_gdn_deferred(
        &mut self,
        projections: &GdnProjections,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<Option<(MxArray, VerifyGdnLayerTape)>> {
        let c = self.config.clone();
        let (nh, kh, kd, vd) = (
            c.linear_num_value_heads as i64,
            c.linear_num_key_heads as i64,
            c.linear_key_head_dim as i64,
            c.linear_value_head_dim as i64,
        );
        if !super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || (kh, nh, kd, vd) != (16, 48, 128, 128)
            || c.linear_conv_kernel_dim != 4
            || !c.rms_norm_eps.is_finite()
            || c.rms_norm_eps <= 0.0
            || projections.qkv.dtype()? != DType::BFloat16
            || projections.z.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        if conv.dtype()? != DType::Float32 {
            return Ok(None);
        }
        let (scale, dt) = self.gdn_scale_dt(i, &p, &g)?;
        let norm_w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let pre_conv = cache.conv.clone();
        let Some([q, k, v, decay, beta]) = math::gdn_prepare(
            &projections.qkv,
            &projections.a,
            &projections.b,
            &conv,
            &mut cache.conv,
            &scale,
            &dt,
        )?
        else {
            cache.conv = pre_conv;
            return Ok(None);
        };
        let state_base = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let prev_pending = self.cuda_pending_gdn[i].take();
        let has_prev_indices = prev_pending
            .as_ref()
            .is_some_and(|p| p.keep > 0 && p.prev_indices.is_some());
        let step_result = if has_prev_indices {
            let rows = projections.qkv.shape_at(1)? as i32;
            let chain_parent: Vec<i32> = (0..rows).map(|r| r - 1).collect();
            let parent_arr = MxArray::from_int32(&chain_parent, &[rows as i64])?;
            let pending_ref = prev_pending.as_ref().filter(|p| p.keep > 0).map(|p| {
                (
                    &p.k,
                    &p.v,
                    &p.g,
                    &p.beta,
                    p.keep as i32,
                    p.prev_indices.as_ref(),
                )
            });
            crate::kernels::gdn::tree_step_deferred_with_gate(
                &q,
                &k,
                &v,
                &decay,
                &beta,
                &state_base,
                pending_ref,
                &parent_arr,
                &projections.z,
                &norm_w,
                c.rms_norm_eps as f32,
                self.gguf(),
                c.output_gate_type == "sigmoid",
            )?
        } else {
            let pending_ref = prev_pending
                .as_ref()
                .filter(|p| p.keep > 0)
                .map(|p| (&p.k, &p.v, &p.g, &p.beta, p.keep as i32));
            crate::kernels::gdn::fused_step_deferred_with_gate(
                &q,
                &k,
                &v,
                &decay,
                &beta,
                &state_base,
                pending_ref,
                &projections.z,
                &norm_w,
                c.rms_norm_eps as f32,
                self.gguf(),
                c.output_gate_type == "sigmoid",
            )?
        };
        let Some((out, next_state_base)) = step_result else {
            self.cuda_pending_gdn[i] = prev_pending;
            cache.conv = pre_conv;
            return Ok(None);
        };
        cache.recurrent = Some(next_state_base);
        Ok(Some((
            out,
            VerifyGdnLayerTape {
                qkv: projections.qkv.clone(),
                pre_conv,
                k,
                v,
                g: decay,
                beta,
            },
        )))
    }

    /// Tree-verify CUDA deferred GDN step: runs `gdn::prepare_tree` with
    /// `views.conv_taps_i32` and `gdn::tree_step_deferred_with_gate` with
    /// `views.parent_i32`, folding any pending previous cycle's accepted rows
    /// (contiguous or non-contiguous `prev_indices`) into `cache.recurrent`
    /// in-register without materializing intermediate speculative states.
    pub(super) fn cuda_verify_gdn_tree(
        &mut self,
        projections: &GdnProjections,
        i: usize,
        cache: &mut LayerCache,
        views: &crate::models::qwen3_5::verify_tree::TreeDeviceViews,
    ) -> Result<(MxArray, VerifyGdnLayerTape)> {
        let c = self.config.clone();
        let (nh, kh, kd, vd) = (
            c.linear_num_value_heads as i64,
            c.linear_num_key_heads as i64,
            c.linear_key_head_dim as i64,
            c.linear_value_head_dim as i64,
        );
        if !super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || (kh, nh, kd, vd) != (16, 48, 128, 128)
            || c.linear_conv_kernel_dim != 4
            || !c.rms_norm_eps.is_finite()
            || c.rms_norm_eps <= 0.0
            || projections.qkv.dtype()? != DType::BFloat16
            || projections.z.dtype()? != DType::BFloat16
        {
            return Err(Error::from_reason(
                "Qwen4 tree GDN verify requires flat CUDA BF16 (16/48/128/128, k=4)",
            ));
        }
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        let (scale, dt) = self.gdn_scale_dt(i, &p, &g)?;
        let norm_w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let pre_conv = cache.conv.clone();
        let old_conv = match &pre_conv {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[3, 10240], Some(DType::BFloat16))?,
        };
        let [q, k, v, decay, beta, next_conv] = crate::kernels::gdn::prepare_tree(
            &projections.qkv,
            &projections.a,
            &projections.b,
            &conv,
            &old_conv,
            &scale,
            &dt,
            &views.conv_taps_i32,
            false,
            false,
        )?
        .ok_or_else(|| Error::from_reason("Qwen4 CUDA gdn::prepare_tree failed"))?;
        let state_base = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let prev_pending = self.cuda_pending_gdn[i].take();
        let pending_ref = prev_pending.as_ref().filter(|p| p.keep > 0).map(|p| {
            (
                &p.k,
                &p.v,
                &p.g,
                &p.beta,
                p.keep as i32,
                p.prev_indices.as_ref(),
            )
        });
        let Some((out, next_state_base)) = crate::kernels::gdn::tree_step_deferred_with_gate(
            &q,
            &k,
            &v,
            &decay,
            &beta,
            &state_base,
            pending_ref,
            &views.parent_i32,
            &projections.z,
            &norm_w,
            c.rms_norm_eps as f32,
            self.gguf(),
            c.output_gate_type == "sigmoid",
        )?
        else {
            self.cuda_pending_gdn[i] = prev_pending;
            return Err(Error::from_reason(
                "Qwen4 CUDA gdn::tree_step_deferred_with_gate failed",
            ));
        };
        cache.conv = Some(next_conv);
        cache.recurrent = Some(next_state_base);
        Ok((
            out,
            VerifyGdnLayerTape {
                qkv: projections.qkv.clone(),
                pre_conv,
                k,
                v,
                g: decay,
                beta,
            },
        ))
    }
}
