//! Matrix batching for prompt windows and scheduled decode rows. Stateful
//! attention/PLE preserve each owner's order; stateless HC/MLP share weights.
use super::*;
use crate::models::qwen4_exp::runtime_flags;
use std::collections::BTreeMap;

impl Decoder {
    pub(super) fn resident_experts(
        &self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        if !self.gguf() || runtime_flags::is_zero(c"MLX_QWEN4_RESIDENT_GATHER") {
            return Ok(None);
        }
        let banks = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_exps.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = banks else {
            return Ok(None);
        };
        // The dense F32 reference preserves singleton GEMV association.
        if [&gate, &up, &down].iter().any(|w| w.scales.is_none()) {
            return Ok(None);
        }
        let tokens = x.shape()?[1];
        let top = self.config.num_experts_per_tok;
        let h = self.config.hidden_size;
        let banks = [gate, up, down];
        if top == 10
            && let Some(out) = math::routed_experts(x, selected, scores, &banks)?
        {
            return Ok(Some(out));
        }
        let token_rows = MxArray::arange(0.0, tokens as f64, None, Some(DType::Int32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top as i64])?
            .reshape(&[-1])?;
        let source = x.reshape(&[tokens, h as i64])?;
        if let Some((output, inverse)) = Self::indirect_expert_assignment_rows(
            &source,
            &token_rows,
            &selected.reshape(&[-1])?,
            &banks,
            self.config.num_experts,
            self.config.num_experts,
        )? {
            return math::combine_expert_rows(&output, scores, Some(&inverse), top).map(Some);
        }
        let input = source.take(&token_rows, 0)?;
        let (output, inverse) = Self::sorted_expert_assignment_rows(
            &input,
            &selected.reshape(&[-1])?,
            &banks,
            self.config.num_experts,
        )?;
        math::combine_expert_rows(&output, scores, inverse.as_ref(), top).map(Some)
    }

    #[cfg(test)]
    pub(in super::super) fn expert_assignment_rows(
        input: &MxArray,
        ids: &MxArray,
        banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
        experts: usize,
    ) -> Result<MxArray> {
        let (output, inverse) = Self::sorted_expert_assignment_rows(input, ids, banks, experts)?;
        match inverse {
            Some(inverse) => output.take(&inverse, 0),
            None => Ok(output),
        }
    }

    // The fused 32-row layout benefits dispersed routing; concentrated routes
    // retain 64-row tiles to amortize packed-weight reads. Partial banks already
    // know their active count. Full residency keeps device-only routing.
    fn indirect_expert_assignment_rows(
        input: &MxArray,
        token_rows: &MxArray,
        ids: &MxArray,
        banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
        experts: usize,
        active_experts: usize,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_INDIRECT")
            || ids.size()? < 256
            || (ids.size()? / active_experts.max(1) as u64 >= 32
                && runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_REFERENCE_INDIRECT"))
            || input.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let Some((order, inverse, selected)) = math::route_sort(ids, experts)? else {
            return Ok(None);
        };
        let token_rows = token_rows.astype(DType::Uint32)?.take(&order, 0)?;
        let Some(out) = math::prefill_indirect(input, &selected, &token_rows, banks, experts)?
        else {
            return Ok(None);
        };
        Ok(Some((
            out.reshape(&[ids.size()? as i64, input.shape()?[1]])?,
            inverse,
        )))
    }

    fn sorted_expert_assignment_rows(
        input: &MxArray,
        ids: &MxArray,
        banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
        experts: usize,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let rows = ids.size()? as i64;
        let sorted = rows > 80;
        let routing = if sorted {
            super::super::math::route_sort(ids, experts)?
        } else {
            None
        };
        let order = if let Some((order, _, _)) = &routing {
            Some(order.clone())
        } else if sorted {
            Some(ids.argsort(Some(0))?)
        } else {
            None
        };
        let x = if let Some(order) = &order {
            input.take(order, 0)?
        } else {
            input.clone()
        };
        let selected = if let Some((_, _, ids)) = &routing {
            ids.clone()
        } else if let Some(order) = &order {
            ids.take(order, 0)?
        } else {
            ids.clone()
        };
        let x = x.reshape(&[rows, 1, input.shape()?[1]])?;
        let tiles = if sorted && !runtime_flags::is_zero(c"MLX_QWEN4_EXPERT_TILES") {
            math::expert_tiles(&selected, experts)?
        } else {
            None
        };
        let project = |bank: &super::super::weights::Weight, x: &MxArray| -> Result<MxArray> {
            if let Some(tiles) = &tiles
                && let Some(out) = bank.tiled_expert_rows(x, &selected, tiles, experts)?
            {
                return Ok(out);
            }
            bank.expert_rows(x, &selected, experts, sorted)
        };
        let gate = project(&banks[0], &x)?;

        let up = project(&banks[1], &x)?;

        let hidden = Activations::swiglu_compiled(&gate, &up)?;

        let out = project(&banks[2], &hidden)?.reshape(&[rows, -1])?;

        let inverse = match order {
            Some(order) => Some(match routing {
                Some((_, inverse, _)) => inverse,
                None => order.argsort(Some(0))?,
            }),
            None => None,
        };
        Ok((out, inverse))
    }

    // Every assignment belongs to this bank, so token rows have a regular
    // layout and the GPU sort already supplies the complete inverse. Avoid
    // uploading host-built identity maps or composing a second inverse.
    pub(in super::super) fn single_slot_group_rows(
        x: &MxArray,
        selected: &MxArray,
        banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
        capacity: usize,
        active_experts: usize,
        top: usize,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let tokens = x.shape_at(1)?;
        let source = x.reshape(&[tokens, -1])?;
        let token_rows = MxArray::arange(0.0, tokens as f64, None, Some(DType::Uint32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top as i64])?
            .reshape(&[-1])?;
        if let Some((out, inverse)) = Self::indirect_expert_assignment_rows(
            &source,
            &token_rows,
            selected,
            banks,
            capacity,
            active_experts,
        )? {
            return Ok((out, Some(inverse)));
        }
        Self::sorted_expert_assignment_rows(
            &source.take(&token_rows, 0)?,
            selected,
            banks,
            capacity,
        )
    }

    fn slotted_experts(
        &mut self,
        x: &MxArray,
        ids: &[u32],
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        if !self.gguf() {
            return Ok(None);
        }
        let c = self.config.clone();
        let capacity = self
            .weights
            .expert_slot_capacity(c.num_experts, c.num_experts_per_tok)?;
        if capacity == 0 {
            return Ok(None);
        }
        // A small window that fits the slots keeps router order throughout.
        // Retain the reduced result as a reader of all three mutable banks.
        if x.shape()?[1] <= 8
            && c.num_experts_per_tok == 10
            && c.hidden_size == 2560
            && c.moe_intermediate_size == 640
            && x.dtype()? == DType::BFloat16
            && ids.len() == x.shape()?[1] as usize * c.num_experts_per_tok
            && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_EXPERTS")
            && ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
                <= capacity
            && let Some((banks, slots, _)) =
                self.weights
                    .expert_slots(i, c.num_experts, c.num_experts_per_tok, ids)?
        {
            let selected = MxArray::from_uint32(&slots, &[slots.len() as i64])?;
            if let Some(out) = math::routed_experts(x, &selected, scores, &banks)? {
                self.weights.finish_expert_slots(i, &out)?;
                return Ok(Some(out));
            }
        }
        // Leave the experts used nearest the prompt boundary in the slots.
        // Router order is restored below, independently of SSD load order.
        let linear_plan =
            x.shape_at(1)? > 8 && !runtime_flags::is_zero(c"MLX_QWEN4_LINEAR_ROUTE_PLAN");
        let mut last = BTreeMap::new();
        let linear_order = if linear_plan {
            Some(
                super::super::route_plan::expert_order(ids, c.num_experts)
                    .map_err(Error::from_reason)?,
            )
        } else {
            for (assignment, &expert) in ids.iter().enumerate() {
                last.insert(expert, assignment);
            }
            None
        };
        let active_experts = linear_order.as_ref().map_or(last.len(), Vec::len);
        if x.shape_at(1)? > 8 {
            self.weights.note_prefill_routes(
                i,
                linear_order
                    .clone()
                    .unwrap_or_else(|| last.keys().copied().collect()),
            );
        }
        if active_experts <= capacity
            && x.shape_at(1)? > 8
            && !runtime_flags::is_zero(c"MLX_QWEN4_SINGLE_SLOT_GROUP")
        {
            self.check_cancelled()?;
            let Some((banks, slots, count)) =
                self.weights
                    .expert_slots(i, c.num_experts, c.num_experts_per_tok, ids)?
            else {
                return Ok(None);
            };
            let selected = MxArray::from_uint32(&slots, &[slots.len() as i64])?;
            let (out, inverse) = Self::single_slot_group_rows(
                x,
                &selected,
                &banks,
                count,
                active_experts,
                c.num_experts_per_tok,
            )?;
            let reduced =
                math::combine_expert_rows(&out, scores, inverse.as_ref(), c.num_experts_per_tok)?;
            if self.verification.is_none()
                && !runtime_flags::is_zero(c"MLX_QWEN4_DEFER_EXPERT_REDUCTION")
                && !runtime_flags::is_zero(c"MLX_QWEN4_ASYNC_SUBMISSION")
            {
                // This one bank group has no intervening slot reuse. Keep
                // its reduction as the lease, so the large assignment output
                // is released after GPU execution. A later router/frontier
                // completes it; misses fence the retained lease explicitly.
                self.weights.defer_expert_reduction(i, &reduced)?;
            } else {
                self.weights.finish_expert_slots(i, &out)?;
                self.weights.complete_expert_window(i, &reduced)?;
            }
            return Ok(Some(reduced));
        }
        let groups = linear_order.unwrap_or_else(|| {
            let mut groups = last.keys().copied().collect::<Vec<_>>();
            groups.sort_by_key(|e| last[e]);
            groups
        });
        let mut planned_assignments = if linear_plan {
            Some(
                super::super::route_plan::assignment_groups(ids, &groups, c.num_experts, capacity)
                    .map_err(Error::from_reason)?,
            )
        } else {
            None
        };
        let mut values = Vec::new();
        let mut group_inverses = Vec::new();
        let mut inverse = vec![0i32; ids.len()];
        let mut output_row = 0;
        for (group, experts) in groups.chunks(capacity).enumerate() {
            self.check_cancelled()?;
            let assignments = if let Some(planned) = &mut planned_assignments {
                std::mem::take(&mut planned[group])
            } else {
                let wanted: std::collections::HashSet<_> = experts.iter().copied().collect();
                ids.iter()
                    .enumerate()
                    .filter(|(_, e)| wanted.contains(e))
                    .map(|(slot, _)| slot)
                    .collect::<Vec<_>>()
            };
            let requested: Vec<_> = assignments.iter().map(|&s| ids[s]).collect();
            let Some((banks, slots, count)) =
                self.weights
                    .expert_slots(i, c.num_experts, c.num_experts_per_tok, &requested)?
            else {
                return Ok(None);
            };
            let rows: Vec<_> = assignments
                .iter()
                .map(|&s| (s / c.num_experts_per_tok) as i32)
                .collect();
            let source = x.reshape(&[-1, c.hidden_size as i64])?;
            let token_rows = MxArray::from_int32(&rows, &[rows.len() as i64])?;
            let selected = MxArray::from_uint32(&slots, &[slots.len() as i64])?;
            let (out, local_inverse) = if let Some((out, inverse)) =
                Self::indirect_expert_assignment_rows(
                    &source,
                    &token_rows,
                    &selected,
                    &banks,
                    count,
                    experts.len(),
                )? {
                (out, Some(inverse))
            } else {
                let input = source.take(&token_rows, 0)?;
                Self::sorted_expert_assignment_rows(&input, &selected, &banks, count)?
            };
            self.weights.finish_expert_slots(i, &out)?;
            let local_inverse = match local_inverse {
                Some(inverse) => inverse.astype(DType::Int32)?,
                None => MxArray::arange(0.0, assignments.len() as f64, None, Some(DType::Int32))?,
            };
            group_inverses.push(local_inverse.add(&MxArray::from_int32(&[output_row], &[])?)?);
            for slot in assignments {
                inverse[slot] = output_row;
                output_row += 1;
            }
            values.push(out);
        }
        let out = MxArray::concatenate_many(values.iter().collect(), Some(0))?;
        let inverse = MxArray::concatenate_many(group_inverses.iter().collect(), Some(0))?
            .take(&MxArray::from_int32(&inverse, &[inverse.len() as i64])?, 0)?;
        let tokens = x.shape()?[1];
        let reduced =
            math::combine_expert_rows(&out, scores, Some(&inverse), c.num_experts_per_tok)?;
        if tokens > 8 {
            // A wide assignment tensor is tens of MiB per layer. Once its
            // reduction is complete it must not remain pinned by slot leases.
            self.weights.complete_expert_window(i, &reduced)?;
        }
        Ok(Some(reduced))
    }

    pub(super) fn selected_experts(
        &mut self,
        x: &MxArray,
        ids: &[u32],
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        if let Some(out) = self.slotted_experts(x, ids, scores, i)? {
            return Ok(Some(out));
        }
        use super::super::weights::{MAX_READ_BYTES, Weight};
        if !self.gguf() || runtime_flags::is_zero(c"MLX_QWEN4_BATCH_EXPERTS") {
            return Ok(None);
        }
        let (h, m) = (self.config.hidden_size, self.config.moe_intermediate_size);
        // Limit even the F32-equivalent live weight set before fetching it.
        let bytes = (h as u64)
            .checked_mul(m as u64)
            .and_then(|n| n.checked_mul(ids.len() as u64 * 12));
        if bytes.is_none_or(|n| n > MAX_READ_BYTES * 3) {
            return Ok(None);
        }
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for &e in ids {
            self.check_cancelled()?;
            let e = e as usize;
            let w = self
                .weights
                .read(&format!("blk.{i}.ffn_gate_exps.weight"), e * m, m)?;
            if w.scales.is_none() {
                return Ok(None);
            }
            gate.push(w);
            up.push(
                self.weights
                    .read(&format!("blk.{i}.ffn_up_exps.weight"), e * m, m)?,
            );
            down.push(
                self.weights
                    .read(&format!("blk.{i}.ffn_down_exps.weight"), e * h, h)?,
            );
        }
        let (Some(gate), Some(up), Some(down)) = (
            Weight::stack(&gate)?,
            Weight::stack(&up)?,
            Weight::stack(&down)?,
        ) else {
            return Ok(None);
        };
        let input = x.broadcast_to(&[ids.len() as i64, 1, h as i64])?;
        let hidden = math::swiglu(&gate.linear(&input)?, &up.linear(&input)?)?;
        let out = down
            .linear(&hidden)?
            .mul(&scores.reshape(&[ids.len() as i64, 1, 1])?)?
            .sum(Some(&[0]), Some(false))?
            .reshape(&[1, 1, h as i64])?;
        MxArray::eval_arrays_with_context(&[&out], "qwen4::batch::selected_experts")?;
        Ok(Some(out))
    }

    pub(super) fn attention_batch(
        &mut self,
        rows: &[MxArray],
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<Vec<MxArray>> {
        let x = MxArray::concatenate_many(rows.iter().collect(), Some(1))?;
        let out = self.attention_matrix(&x, tokens, i, cache)?;
        (0..rows.len())
            .map(|j| out.slice_axis(1, j as i64, j as i64 + 1))
            .collect()
    }

    pub(super) fn attention_matrix(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        Ok(self
            .attention_matrix_for_mlp(x, tokens, i, cache, false, None)?
            .0)
    }

    pub(super) fn attention_matrix_for_mlp(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
        incoming_norm: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let base = self.history.len();
        let x = if self.config.ple_layer_ids.contains(&(i + 1)) {
            x.add(&self.ple_window(x, tokens, i, cache)?)?
        } else {
            x.clone()
        };

        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            incoming_norm,
        )?;

        let branch = if self.config.linear(i) {
            self.gdn_batch(&mixed, i, cache)?
        } else {
            let projections = self.attention_projections(&mixed, i)?;
            let attention =
                if let Some(window) = self.attention_window(&mixed, i, cache, &projections)? {
                    window
                } else {
                    let mut outputs = Vec::with_capacity(tokens.len());
                    for (j, &token) in tokens.iter().enumerate() {
                        self.check_cancelled()?;
                        outputs.push(self.attention_projected(
                            &mixed.slice_axis(1, j as i64, j as i64 + 1)?,
                            i,
                            cache,
                            &projections.row(j)?,
                            false,
                        )?);
                        self.history.push(token);
                    }
                    self.history.truncate(base);
                    MxArray::concatenate_many(outputs.iter().collect(), Some(1))?
                };
            self.linear(
                &attention,
                &format!("layers.{i}.self_attn.o_proj.weight"),
                &format!("blk.{i}.attn_output.weight"),
            )?
        };

        let gate = gate.ok_or_else(|| {
            Error::from_reason("Qwen4 batched attention is missing its injection gate")
        })?;
        let (out, normed) = self.inject_for_mlp(&x, &branch, &gate, i, normalize)?;
        if !self.async_device_prefill() {
            self.submit(&[&out], "qwen4::batch::attention")?;
        }

        Ok((out, normed))
    }

    fn gdn_batch(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let t = x.shape()?[1];
        let (nh, kh, kd, vd) = (
            c.linear_num_value_heads as i64,
            c.linear_num_key_heads as i64,
            c.linear_key_head_dim as i64,
            c.linear_value_head_dim as i64,
        );
        let GdnProjections { qkv, z, a, b } = self.gdn_projections(x, &p, &g)?;

        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        let scale = self
            .dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?
            .astype(DType::Float32)?;
        let scale = if self.gguf() {
            scale
        } else {
            scale.exp()?.negative()?
        };
        let dt = self
            .dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?
            .astype(DType::Float32)?;
        let prepared = if self.gguf()
            && c.linear_conv_kernel_dim == 4
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_GDN_PREP")
        {
            math::gdn_prepare(&qkv, &a, &b, &conv, &mut cache.conv, &scale, &dt)?
        } else {
            None
        };
        let [q, k, v, decay, beta] = if let Some(values) = prepared {
            values
        } else {
            let qkv = math::conv_sequence(&qkv, &conv, &mut cache.conv, c.linear_conv_kernel_dim)?;
            let q = math::l2(&qkv.slice_axis(2, 0, kh * kd)?.reshape(&[1, t, kh, kd])?)?
                .mul_scalar((kd as f64).powf(-0.5))?;
            let k = math::l2(
                &qkv.slice_axis(2, kh * kd, 2 * kh * kd)?
                    .reshape(&[1, t, kh, kd])?,
            )?;
            let compact = x.dtype()? == DType::BFloat16
                && kd >= 32
                && kd % 32 == 0
                && crate::engine::persistence::compiled_forward_backend_available()
                && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_GDN")
                && runtime_flags::is_one(c"MLX_QWEN4_PREFILL_GDN_BF16");
            let storage = if compact {
                DType::BFloat16
            } else {
                DType::Float32
            };
            let v = qkv
                .slice_axis(2, 2 * kh * kd, 2 * kh * kd + nh * vd)?
                .reshape(&[1, t, nh, vd])?
                .astype(storage)?;
            let (q, k) = if self.gguf() {
                (q, k)
            } else {
                (
                    q.repeat((nh / kh) as i32, 2)?,
                    k.repeat((nh / kh) as i32, 2)?,
                )
            };
            let (decay, beta) = math::gdn_gates(&a, &b, &scale, &dt)?;
            [q.astype(storage)?, k.astype(storage)?, v, decay, beta]
        };

        let state = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let (out, state) = math::recurrent_sequence(&q, &k, &v, &decay, &beta, &state)?;

        self.submit_state(&[&state], "qwen4::batch::recurrent_state")?;
        cache.recurrent = Some(state);
        let norm = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let replay = if c.output_gate_type == "sigmoid"
            && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_EPILOGUE")
        {
            math::gdn_epilogue(&out, &z, &norm, c.rms_norm_eps)?
        } else {
            None
        };
        let out = if let Some(replay) = replay {
            replay.reshape(&[1, t, nh * vd])?
        } else {
            let out = math::norm(
                &out.astype(x.dtype()?)?,
                &norm,
                c.linear_value_head_dim,
                c.rms_norm_eps,
                false,
            )?
            .astype(DType::Float32)?;
            let z = z.astype(DType::Float32)?;
            let out = if c.output_gate_type == "sigmoid" {
                math::sigmoid_mul(&z, &out)?
            } else {
                math::swiglu(&z, &out)?
            };

            out.astype(x.dtype()?)?.reshape(&[1, t, nh * vd])?
        };

        let output = self.linear(
            &out,
            &format!("{p}.out_proj.weight"),
            &format!("{g}.ssm_out.weight"),
        )?;

        Ok(output)
    }

    pub(super) fn mlp_batch(&mut self, rows: &[MxArray], i: usize) -> Result<Vec<MxArray>> {
        let x = MxArray::concatenate_many(rows.iter().collect(), Some(1))?;
        let out = self.mlp_matrix(&x, i)?;
        (0..rows.len())
            .map(|j| out.slice_axis(1, j as i64, j as i64 + 1))
            .collect()
    }

    pub(super) fn mlp_matrix(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        Ok(self.mlp_matrix_for_attention(x, i, None, false)?.0)
    }

    pub(super) fn mlp_matrix_for_attention(
        &mut self,
        x: &MxArray,
        i: usize,
        normed: Option<&MxArray>,
        normalize_next: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let (mixed, gate) = self.hyper_with_norm(
            x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed,
        )?;

        let branch = self.moe_batch(&mixed, i)?;

        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 batched MLP is missing its injection gate"))?;
        let (out, next_norm) =
            self.inject_for_next_attention(x, &branch, &gate, i, normalize_next)?;
        if self.async_prefill || runtime_flags::is_one(c"MLX_QWEN4_PREFILL_DEFER_MLP") {
            // The next router readback (or final window output) fences this
            // graph. Slot-bank readers remain retained until their own fence.
            // Reference schedule: first two layers, then groups of three.
            if !self.async_device_prefill() || (i >= 1 && (i - 1).is_multiple_of(3)) {
                self.submit(&[&out], "qwen4::batch::mlp")?;
            }
        } else {
            MxArray::eval_arrays_with_context(&[&out], "qwen4::batch::mlp")?;
        }

        Ok((out, next_norm))
    }

    pub(super) fn moe_batch(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let c = self.config.clone();
        let tokens = x.shape()?[1] as usize;
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let logits = self.linear(
            x,
            &format!("{p}.gate.weight"),
            &format!("{g}.ffn_gate_inp.weight"),
        )?;
        let top = c.num_experts_per_tok;
        let (selected, scores) = if let Some(routes) = math::prefill_routes(&logits, top)? {
            routes
        } else {
            let probs = Activations::softmax_precise(&logits, Some(-1))?;
            let selected = probs.argpartition(-(top as i32), Some(-1))?.slice_axis(
                2,
                (c.num_experts - top) as i64,
                c.num_experts as i64,
            )?;
            let scores = probs.take_along_axis(&selected, -1)?;
            let scores = scores.div(&scores.sum(Some(&[-1]), Some(true))?)?;
            (selected, scores)
        };
        let scores = scores.reshape(&[1, (tokens * top) as i64, 1])?;

        if self.device_routes.is_some()
            && runtime_flags::is_one(c"MLX_QWEN4_PREFILL_SHARED_COMBINE")
        {
            let (shared, gate) = self.shared_expert_parts(x, i, None)?;
            let out =
                self.tentative_prefill_experts(x, &selected, &scores, i, Some((&shared, &gate)))?;

            return Ok(out);
        }
        let resident = if self.device_routes.is_some() {
            Some(self.tentative_prefill_experts(x, &selected, &scores, i, None)?)
        } else {
            self.resident_experts(x, &selected, &scores, i)?
        };
        let sum = if let Some(sum) = resident {
            sum
        } else {
            let ids = { selected.to_uint32()? };
            if let Some(out) = self.slotted_experts(x, &ids, &scores, i)? {
                out
            } else {
                let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                for (slot, &expert) in ids.iter().enumerate() {
                    groups.entry(expert).or_default().push(slot);
                }
                let mut weighted: Vec<Option<MxArray>> = vec![None; tokens * top];
                for (expert, slots) in groups {
                    self.check_cancelled()?;
                    let indices = MxArray::from_int32(
                        &slots.iter().map(|s| (s / top) as i32).collect::<Vec<_>>(),
                        &[slots.len() as i64],
                    )?;
                    let input = x.take(&indices, 1)?;
                    let e = expert as usize;
                    let m = c.moe_intermediate_size;
                    let (gate, up) = if self.gguf() {
                        (
                            self.weights
                                .read(&format!("{g}.ffn_gate_exps.weight"), e * m, m)?
                                .linear(&input)?,
                            self.weights
                                .read(&format!("{g}.ffn_up_exps.weight"), e * m, m)?
                                .linear(&input)?,
                        )
                    } else {
                        let key = self.key(&format!("{p}.experts.gate_up_proj"), "");
                        let both = self.weights.read(&key, e * 2 * m, 2 * m)?.linear(&input)?;
                        (
                            both.slice_axis(2, 0, m as i64)?,
                            both.slice_axis(2, m as i64, 2 * m as i64)?,
                        )
                    };
                    let hidden = math::swiglu(&gate, &up)?;
                    let key = self.key(
                        &format!("{p}.experts.down_proj"),
                        &format!("{g}.ffn_down_exps.weight"),
                    );
                    let out = self
                        .weights
                        .read(&key, e * c.hidden_size, c.hidden_size)?
                        .linear(&hidden)?;
                    // Retain at most this expert's three matrices in the pending graph.
                    MxArray::eval_arrays_with_context(&[&out], "qwen4::batch::expert")?;
                    for (j, slot) in slots.into_iter().enumerate() {
                        weighted[slot] = Some(
                            out.slice_axis(1, j as i64, j as i64 + 1)?
                                .mul(&scores.slice_axis(1, slot as i64, slot as i64 + 1)?)?,
                        );
                    }
                }
                // Restore each token's original router order before reduction. Sorting
                // expert loads must not reorder the floating-point accumulation.

                let weighted = weighted
                    .iter()
                    .map(|row| {
                        row.as_ref().ok_or_else(|| {
                            Error::from_reason("Qwen4 MoE did not populate every routed output row")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                MxArray::concatenate_many(weighted, Some(1))?
                    .reshape(&[1, tokens as i64, top as i64, c.hidden_size as i64])?
                    .sum(Some(&[2]), Some(false))?
            }
        };

        let (shared, gate) = self.shared_expert_parts(x, i, None)?;
        let output = sum
            .astype(x.dtype()?)?
            .add(&math::sigmoid_mul(&gate, &shared)?)?;

        Ok(output)
    }
}
