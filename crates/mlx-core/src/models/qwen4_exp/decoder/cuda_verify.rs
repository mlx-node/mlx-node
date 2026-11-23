use super::*;

impl Decoder {
    /// Flat CUDA deferred-GDN verification pass: folds any pending previous
    /// cycle's accepted GDN rows in-register via `fused_step_deferred_with_gate`,
    /// avoids building `R` full `DecoderState`s, submits asynchronously, and
    /// returns lazy `[R, vocab]` logits + a lightweight [`DeferredVerifyTicket`]
    /// so `accept` evaluates logits + argmax in a single GPU sync.
    pub(in super::super) fn verify_chunk_deferred(
        &mut self,
        tokens: &[u32],
    ) -> Result<(MxArray, DeferredVerifyTicket)> {
        if tokens.is_empty() || tokens.len() > 4 {
            return Err(Error::from_reason("Qwen4 verification width must be 1-4"));
        }
        if self.paged.is_some() {
            return Err(Error::from_reason(
                "Qwen4 deferred CUDA verification requires flat KV",
            ));
        }
        self.rotary_tables.get_mut().clear();
        self.cuda_verify_final_window = None;
        self.cuda_verify_final_normed = None;
        let base_history_len = self.history.len();
        let layers = self.config.num_hidden_layers;
        let started = std::time::Instant::now();
        self.verification = Some(Vec::new());
        self.cuda_verify_gdn_tape = Some(vec![None; layers]);
        self.cuda_verify_row_caches = Some(vec![vec![LayerCache::default(); layers]; tokens.len()]);
        let pass = std::mem::replace(&mut self.decode_pass, true);
        let result = (|| {
            self.prefill_chunk_inner(tokens, &[], None, false)?;
            let window = self.cuda_verify_final_window.take().ok_or_else(|| {
                Error::from_reason("Qwen4 deferred verification lost its final window")
            })?;
            let normed = self.cuda_verify_final_normed.take();
            let (mixed, _) = self.hyper_rows(
                &window,
                "hyper_connection_mixer",
                "output_hc",
                false,
                normed.as_ref(),
            )?;
            let logits = self
                .linear_rows(&mixed, "lm_head.weight", "output.weight")?
                .reshape(&[tokens.len() as i64, -1])?;
            self.verification = None;
            let gdn_tapes = self.cuda_verify_gdn_tape.take().ok_or_else(|| {
                Error::from_reason("Qwen4 deferred verification lost its GDN tapes")
            })?;
            let row_caches = self.cuda_verify_row_caches.take().ok_or_else(|| {
                Error::from_reason("Qwen4 deferred verification lost its row caches")
            })?;
            // Submit all caches, GDN tapes, hidden taps, and logits in one
            // non-blocking CUDA async batch (logits LAST so `accept`'s argmax
            // eval drains the entire verify wave in a single GPU sync).
            let mut completed: Vec<&MxArray> = Vec::new();
            completed.extend(self.caches.iter().flat_map(LayerCache::arrays));
            for tape in gdn_tapes.iter().flatten() {
                completed.push(&tape.qkv);
                completed.push(&tape.k);
                completed.push(&tape.v);
                completed.push(&tape.g);
                completed.push(&tape.beta);
            }
            if let Some(taps) = &self.hidden_taps {
                completed.extend(taps.last_pass_arrays().iter());
            }
            completed.push(&logits);
            Self::cuda_async_eval(&completed);
            Ok((
                logits,
                DeferredVerifyTicket {
                    base_history_len,
                    rows: tokens.len(),
                    gdn_tapes,
                    row_caches,
                },
            ))
        })();
        self.decode_pass = pass;
        self.rotary_tables.get_mut().clear();
        match result {
            Ok(verified) => {
                self.observe_pass(tokens.len(), started.elapsed());
                Ok(verified)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    /// Commit `keep` of `total_written` rows from a [`DeferredVerifyTicket`]:
    /// rolls back `history`, non-GDN caches, and 2D `[3, 10240]` GDN conv
    /// state (when `keep < total_written`), and arms `cuda_pending_gdn[i]` so
    /// the next cycle folds the `keep` accepted GDN rows in-register.
    pub(in super::super) fn commit_deferred_verify(
        &mut self,
        mut ticket: DeferredVerifyTicket,
        keep: usize,
        total_written: usize,
    ) -> Result<()> {
        if keep == 0 || keep > total_written || total_written != ticket.rows {
            return Err(Error::from_reason(format!(
                "Qwen4 deferred verify commit keeps {keep} of {total_written} rows (ticket has {})",
                ticket.rows
            )));
        }
        self.history.truncate(ticket.base_history_len + keep);
        let kept_caches = ticket
            .row_caches
            .get(keep - 1)
            .ok_or_else(|| Error::from_reason("Qwen4 deferred verify ticket missing kept row"))?;
        let layers = self.config.num_hidden_layers;
        let conv_hist = (self.config.linear_conv_kernel_dim.saturating_sub(1)) as i64;
        for i in 0..layers {
            let saved = &kept_caches[i];
            let cache = &mut self.caches[i];
            if self.config.ple_layer_ids.contains(&(i + 1)) {
                cache.ple_conv = saved.ple_conv.clone();
            }
            if self.config.linear(i) {
                if let Some(tape) = ticket.gdn_tapes[i].take() {
                    if keep < total_written {
                        let qkv_2d = tape.qkv.squeeze(Some(&[0]))?;
                        let keep_i64 = keep as i64;
                        let new_conv = if keep_i64 >= conv_hist {
                            qkv_2d.slice_axis(0, keep_i64 - conv_hist, keep_i64)?
                        } else {
                            let old_conv = match tape.pre_conv {
                                Some(prev) => prev,
                                None => MxArray::zeros(
                                    &[conv_hist, qkv_2d.shape_at(1)?],
                                    Some(qkv_2d.dtype()?),
                                )?,
                            };
                            let kept_old = old_conv.slice_axis(0, keep_i64, conv_hist)?;
                            let kept_new = qkv_2d.slice_axis(0, 0, keep_i64)?;
                            MxArray::concatenate(&kept_old, &kept_new, 0)?
                        };
                        cache.conv = Some(new_conv);
                    }
                    self.cuda_pending_gdn[i] = Some(CudaPendingGdn {
                        k: tape.k,
                        v: tape.v,
                        g: tape.g,
                        beta: tape.beta,
                        keep,
                        prev_indices: None,
                    });
                } else {
                    cache.conv = saved.conv.clone();
                    cache.recurrent = saved.recurrent.clone();
                    self.cuda_pending_gdn[i] = None;
                }
            } else {
                cache.keys = saved.keys.clone();
                cache.values = saved.values.clone();
                cache.flat_kv_rows = saved.flat_kv_rows;
                cache.index_tail = saved.index_tail.clone();
                cache.index_keys = saved.index_keys.clone();
                cache.flat_index_blocks = saved.flat_index_blocks;
                cache.index_blocks = saved.index_blocks;
            }
        }
        Ok(())
    }

    /// Flat CUDA deferred-GDN tree verification pass (`T in 1..=16`):
    /// - Layer-1 PLE uses tree-aware n-gram hashing + dilated conv (`conv_taps`)
    /// - 36L GDN uses `gdn::prepare_tree` + `gdn::tree_step_deferred_with_gate`
    /// - 12L QSA FullAttn uses `qk_norm_rope_kv_store_tree` + `sdpa_tree_few` /
    ///   `qsa_sparse_sdpa_cuda(mask_mode = 1)`
    /// - 48L MoE + 97× mHC run natively for `M in 1..=16`
    ///
    /// Returns lazy `[T, vocab]` logits + [`TreeVerifyTicket`] submitted
    /// asynchronously in one CUDA wave so `argmax` drains in a single GPU sync.
    pub(in super::super) fn verify_tree_deferred(
        &mut self,
        tree: &crate::models::qwen3_5::verify_tree::VerifyTree,
        views: &crate::models::qwen3_5::verify_tree::TreeDeviceViews,
    ) -> Result<(MxArray, TreeVerifyTicket)> {
        let t = tree.len();
        if !(1..=crate::models::qwen3_5::verify_tree::MAX_TREE_NODES).contains(&t) {
            return Err(Error::from_reason(format!(
                "Qwen4 tree verification width {t} outside 1..={}",
                crate::models::qwen3_5::verify_tree::MAX_TREE_NODES
            )));
        }
        if views.tree_len != t {
            return Err(Error::from_reason(format!(
                "Qwen4 tree views len {} != tree len {t}",
                views.tree_len
            )));
        }
        if self.paged.is_some() || !super::super::cuda_qwen4_fast_enabled() {
            return Err(Error::from_reason(
                "Qwen4 deferred CUDA tree verification requires flat CUDA fast mode",
            ));
        }
        self.rotary_tables.get_mut().clear();
        let base_history_len = self.history.len();
        let layers = self.config.num_hidden_layers;
        let started = std::time::Instant::now();
        self.verification = Some(Vec::new());
        let pass = std::mem::replace(&mut self.decode_pass, true);
        let result = (|| {
            // Prefetch token embeddings and tree-aware PLE rows across 32 reader
            // threads BEFORE Layer 0 runs so SSD I/O overlaps with GPU compute.
            let emb_key = self.key("embed_tokens.weight", "token_embd.weight");
            let emb_prefetch: Vec<(String, usize)> = tree
                .tokens
                .iter()
                .map(|&tok| (emb_key.clone(), tok as usize))
                .collect();
            self.weights.prefetch_rows(&emb_prefetch)?;
            self.prefetch_tree_ple_rows(&tree.tokens, &tree.parent)?;
            let ple_taps_i32 = if self.config.ple_layer_ids.is_empty() {
                None
            } else {
                Some(tree.ple_conv_taps_i32()?)
            };

            // Layer-0 token embeddings -> 4 hyper-connection streams [1, T, 4H] BF16.
            let row_ids: Vec<usize> = tree.tokens.iter().map(|&tok| tok as usize).collect();
            let emb = self.weights.lookup_rows(&emb_key, &row_ids)?.reshape(&[
                1,
                t as i64,
                self.config.hidden_size as i64,
            ])?;
            let emb = if emb.dtype()? == DType::BFloat16 {
                emb
            } else {
                emb.astype(DType::BFloat16)?
            };
            let mut x = MxArray::tile(&emb, &[1, 1, self.config.hc_count as i32])?;
            let dtype = x.dtype()?;

            let mut gdn_tapes: Vec<Option<VerifyGdnLayerTape>> = vec![None; layers];
            let mut ple_tapes: Vec<Option<VerifyPleLayerTape>> = vec![None; layers];
            let mut attn_tapes: Vec<Option<VerifyAttnLayerTape>> = vec![None; layers];
            let mut attn_normed: Option<MxArray> = None;

            self.begin_taps();
            for i in 0..layers {
                self.check_cancelled()?;
                let has_ple = self.config.ple_layer_ids.contains(&(i + 1));
                if has_ple {
                    let ple_emb = self.ple_embeddings_tree(&tree.tokens, &tree.parent, i, dtype)?;
                    let (pk, pv) = self.ple_projections(&ple_emb, i)?;
                    let pre_ple_conv = self.caches[i].ple_conv.clone();
                    let taps = ple_taps_i32.as_ref().ok_or_else(|| {
                        Error::from_reason("Qwen4 tree PLE missing ple_conv_taps_i32")
                    })?;
                    let (out_x, conv_in) =
                        self.ple_mix_tree_cuda(&x, &pk, &pv, i, pre_ple_conv.as_ref(), taps)?;
                    ple_tapes[i] = Some(VerifyPleLayerTape {
                        pre_conv: pre_ple_conv,
                        conv_in,
                    });
                    x = out_x;
                    attn_normed = None;
                }

                let (mixed, gate) = self.hyper_rows(
                    &x,
                    &format!("layers.{i}.attn_hyper_connection"),
                    &format!("blk.{i}.hc_attn"),
                    true,
                    attn_normed.as_ref(),
                )?;
                let mut cache = std::mem::take(&mut self.caches[i]);
                let branch = if self.config.linear(i) {
                    let p = format!("layers.{i}.linear_attn");
                    let g = format!("blk.{i}");
                    let projections = self.gdn_projections(&mixed, &p, &g)?;
                    let (out, tape) =
                        self.cuda_verify_gdn_tree(&projections, i, &mut cache, views)?;
                    gdn_tapes[i] = Some(tape);
                    self.caches[i] = cache;
                    self.linear_rows(
                        &out,
                        &format!("{p}.out_proj.weight"),
                        &format!("{g}.ssm_out.weight"),
                    )?
                } else {
                    let projections = self.attention_projections(&mixed, i)?;
                    let (out, tape) = self.cuda_verify_attention_tree(
                        &mixed,
                        i,
                        &mut cache,
                        &projections,
                        tree,
                        views,
                    )?;
                    attn_tapes[i] = Some(tape);
                    self.caches[i] = cache;
                    self.linear_rows(
                        &out,
                        &format!("layers.{i}.self_attn.o_proj.weight"),
                        &format!("blk.{i}.attn_output.weight"),
                    )?
                };
                let gate = gate.ok_or_else(|| {
                    Error::from_reason("Qwen4 tree attention is missing its injection gate")
                })?;
                let (x_mid, mlp_normed) = self.inject_for_mlp(&x, &branch, &gate, i, true)?;
                let (mixed, gate) = self.hyper_rows(
                    &x_mid,
                    &format!("layers.{i}.mlp_hyper_connection"),
                    &format!("blk.{i}.hc_ffn"),
                    true,
                    mlp_normed.as_ref(),
                )?;
                let branch = self.moe_rows(&mixed, i)?;
                let gate = gate.ok_or_else(|| {
                    Error::from_reason("Qwen4 tree MLP is missing its injection gate")
                })?;
                let next_layer = i + 1;
                let (next_x, next_normed) = if next_layer < layers {
                    if !self.config.ple_layer_ids.contains(&(next_layer + 1)) {
                        self.inject_for_hyper(
                            &x_mid,
                            &branch,
                            &gate,
                            &format!("layers.{next_layer}.attn_hyper_connection.hc_norm.weight"),
                            &format!("blk.{next_layer}.hc_attn_norm.weight"),
                            true,
                        )?
                    } else {
                        (Self::inject(&x_mid, &branch, &gate)?, None)
                    }
                } else if self.scope.is_empty() {
                    self.inject_for_hyper(
                        &x_mid,
                        &branch,
                        &gate,
                        "hyper_connection_mixer.hc_norm.weight",
                        "output_hc_norm.weight",
                        true,
                    )?
                } else {
                    (Self::inject(&x_mid, &branch, &gate)?, None)
                };
                // Match linear `verify_chunk_deferred`: submit Layer 0 immediately so
                // GPU overlaps with Layer-1 PLE reads, Layer 3 to keep GPU saturated
                // while Layers 4..11 build, then every 12 layers (5 async waves).
                if i == 0 || i == 3 || ((i + 1).is_multiple_of(12) && i + 1 < layers) {
                    if let Some(ref normed) = next_normed {
                        self.submit(&[&next_x, normed], "qwen4::decoder::verify")?;
                    } else {
                        self.submit(&[&next_x], "qwen4::decoder::verify")?;
                    }
                }
                self.tap_layer(i, std::slice::from_ref(&next_x))?;
                x = next_x;
                attn_normed = next_normed;
            }
            self.finish_taps(base_history_len)?;

            let (mixed, _) = self.hyper_rows(
                &x,
                "hyper_connection_mixer",
                "output_hc",
                false,
                attn_normed.as_ref(),
            )?;
            let logits = self
                .linear_rows(&mixed, "lm_head.weight", "output.weight")?
                .reshape(&[t as i64, -1])?;
            self.verification = None;

            // Submit all caches, GDN/PLE/QSA tapes, hidden taps, and logits in
            // one non-blocking CUDA async batch (logits LAST so `argmax` drains
            // the entire tree verify wave in a single GPU sync).
            let mut completed: Vec<&MxArray> = Vec::new();
            completed.extend(self.caches.iter().flat_map(LayerCache::arrays));
            for tape in gdn_tapes.iter().flatten() {
                completed.push(&tape.qkv);
                completed.push(&tape.k);
                completed.push(&tape.v);
                completed.push(&tape.g);
                completed.push(&tape.beta);
            }
            for tape in ple_tapes.iter().flatten() {
                completed.push(&tape.conv_in);
            }
            for tape in attn_tapes.iter().flatten() {
                completed.push(&tape.ik_rows);
            }
            if let Some(taps) = &self.hidden_taps {
                completed.extend(taps.last_pass_arrays().iter());
            }
            completed.push(&logits);
            Self::cuda_async_eval(&completed);

            Ok((
                logits,
                TreeVerifyTicket {
                    base_history_len,
                    tree_tokens: tree.tokens.clone(),
                    gdn_tapes,
                    ple_tapes,
                    attn_tapes,
                },
            ))
        })();
        self.verification = None;
        self.decode_pass = pass;
        self.rotary_tables.get_mut().clear();
        match result {
            Ok(verified) => {
                self.observe_pass(t, started.elapsed());
                Ok(verified)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    /// Commit `accepted_nodes` (always starting with root `0`) from a
    /// [`TreeVerifyTicket`]:
    /// - Appends accepted tokens to `self.history`
    /// - Rebuilds Layer-1 PLE `[9, 10240]` `ple_conv` from `pre_conv` + accepted `conv_in`
    /// - Rebuilds 36L GDN `[3, 10240]` `conv` from `pre_conv` + accepted `qkv` and arms
    ///   `cuda_pending_gdn[i]` with shared `prev_indices` (`None` when contiguous!)
    /// - Trims (zero-copy) or compacts (`kv_compact_rows`) 12L QSA flat KV and updates
    ///   `index_keys` & `index_tail` along the accepted branch.
    pub(in super::super) fn commit_tree_deferred(
        &mut self,
        mut ticket: TreeVerifyTicket,
        accepted_nodes: &[usize],
        is_contiguous_prefix: bool,
        is_chain: bool,
    ) -> Result<()> {
        let keep = accepted_nodes.len();
        let total = ticket.tree_tokens.len();
        if keep == 0 || keep > total || accepted_nodes[0] != 0 {
            return Err(Error::from_reason(format!(
                "Qwen4 tree commit invalid accepted_nodes {accepted_nodes:?} for T={total}"
            )));
        }
        if self.history.len() != ticket.base_history_len {
            return Err(Error::from_reason(format!(
                "Qwen4 tree commit history len {} != base {}",
                self.history.len(),
                ticket.base_history_len
            )));
        }
        for &node in accepted_nodes {
            if node >= total {
                return Err(Error::from_reason(format!(
                    "Qwen4 tree commit node {node} >= T={total}"
                )));
            }
            self.history.push(ticket.tree_tokens[node]);
        }

        let accepted_i32 = if is_contiguous_prefix {
            None
        } else {
            let idx: Vec<i32> = accepted_nodes.iter().map(|&n| n as i32).collect();
            Some(MxArray::from_int32(&idx, &[keep as i64])?)
        };

        let layers = self.config.num_hidden_layers;
        let conv_hist = self.config.linear_conv_kernel_dim.saturating_sub(1);
        let conv_hist_i64 = conv_hist as i64;
        // Pre-build the shared tail index array for non-contiguous GDN conv rebuild
        // when `keep >= conv_hist` so all 36 GDN layers share one 12-byte tensor.
        let gdn_tail_i32 = if !is_contiguous_prefix && keep >= conv_hist && conv_hist > 0 {
            let tail_idx: Vec<i32> = accepted_nodes[keep - conv_hist..keep]
                .iter()
                .map(|&n| n as i32)
                .collect();
            Some(MxArray::from_int32(&tail_idx, &[conv_hist_i64])?)
        } else {
            None
        };
        let ple_keep = self.config.ple_conv_kernel_size.saturating_sub(1) * self.config.ngram_size;
        let ple_keep_i64 = ple_keep as i64;

        for i in 0..layers {
            if let Some(ple_tape) = ticket.ple_tapes[i].take() {
                let keep_i64 = keep as i64;
                let new_ple_conv = if keep >= ple_keep {
                    if is_contiguous_prefix {
                        ple_tape
                            .conv_in
                            .slice_axis(0, keep_i64 - ple_keep_i64, keep_i64)?
                    } else {
                        let tail_idx: Vec<i32> = accepted_nodes[keep - ple_keep..keep]
                            .iter()
                            .map(|&n| n as i32)
                            .collect();
                        let idx_arr = MxArray::from_int32(&tail_idx, &[ple_keep_i64])?;
                        ple_tape.conv_in.take(&idx_arr, 0)?
                    }
                } else {
                    let old_conv = match ple_tape.pre_conv {
                        Some(prev) => prev,
                        None => MxArray::zeros(
                            &[ple_keep_i64, ple_tape.conv_in.shape_at(1)?],
                            Some(ple_tape.conv_in.dtype()?),
                        )?,
                    };
                    let kept_old = old_conv.slice_axis(0, keep_i64, ple_keep_i64)?;
                    let kept_new = if is_contiguous_prefix {
                        ple_tape.conv_in.slice_axis(0, 0, keep_i64)?
                    } else {
                        ple_tape.conv_in.take(
                            accepted_i32.as_ref().ok_or_else(|| {
                                Error::from_reason("Qwen4 PLE tree commit missing accepted_i32")
                            })?,
                            0,
                        )?
                    };
                    MxArray::concatenate(&kept_old, &kept_new, 0)?
                };
                self.caches[i].ple_conv = Some(new_ple_conv);
            }

            if self.config.linear(i) {
                let tape = ticket.gdn_tapes[i].take().ok_or_else(|| {
                    Error::from_reason(format!("Qwen4 tree commit missing GDN tape at layer {i}"))
                })?;
                // `gdn::prepare_tree` already produced the exact `conv` state iff the
                // entire linear chain was accepted contiguously; otherwise rebuild from
                // `tape.pre_conv` + accepted rows of `tape.qkv`.
                if !(is_chain && is_contiguous_prefix && keep == total) {
                    let qkv_2d = tape.qkv.squeeze(Some(&[0]))?;
                    let keep_i64 = keep as i64;
                    let new_conv = if keep >= conv_hist {
                        if is_contiguous_prefix {
                            qkv_2d.slice_axis(0, keep_i64 - conv_hist_i64, keep_i64)?
                        } else {
                            qkv_2d.take(
                                gdn_tail_i32.as_ref().ok_or_else(|| {
                                    Error::from_reason("Qwen4 GDN tree commit missing tail_i32")
                                })?,
                                0,
                            )?
                        }
                    } else {
                        let old_conv = match tape.pre_conv {
                            Some(prev) => prev,
                            None => MxArray::zeros(
                                &[conv_hist_i64, qkv_2d.shape_at(1)?],
                                Some(qkv_2d.dtype()?),
                            )?,
                        };
                        let kept_old = old_conv.slice_axis(0, keep_i64, conv_hist_i64)?;
                        let kept_new = if is_contiguous_prefix {
                            qkv_2d.slice_axis(0, 0, keep_i64)?
                        } else {
                            qkv_2d.take(
                                accepted_i32.as_ref().ok_or_else(|| {
                                    Error::from_reason("Qwen4 GDN tree commit missing accepted_i32")
                                })?,
                                0,
                            )?
                        };
                        MxArray::concatenate(&kept_old, &kept_new, 0)?
                    };
                    self.caches[i].conv = Some(new_conv);
                }
                self.cuda_pending_gdn[i] = Some(CudaPendingGdn {
                    k: tape.k,
                    v: tape.v,
                    g: tape.g,
                    beta: tape.beta,
                    keep,
                    prev_indices: accepted_i32.clone(),
                });
            } else {
                let tape = ticket.attn_tapes[i].take().ok_or_else(|| {
                    Error::from_reason(format!("Qwen4 tree commit missing attn tape at layer {i}"))
                })?;
                let mut cache = std::mem::take(&mut self.caches[i]);
                let res = self.commit_attn_tree_layer(
                    i,
                    &mut cache,
                    tape,
                    keep,
                    is_contiguous_prefix,
                    accepted_i32.as_ref(),
                );
                self.caches[i] = cache;
                res?;
            }
        }
        Ok(())
    }
}
