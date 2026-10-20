//! Tentative GPU routing for warm partial banks. A token is published only
//! after its complete route tape has been checked against the unchanged host
//! mappings. Any miss restores the recurrent/page frontier and replays using
//! the ordinary loader. This never predicts an expert or approximates routing.
use super::*;

impl Decoder {
    pub(super) fn deferred_state_submission(&self) -> bool {
        self.device_routes.is_some()
    }

    pub(super) fn deferred_forward_completion(&self) -> bool {
        self.deferred_state_submission() && self.verification.is_none()
    }

    fn pending_state_arrays(&self) -> Vec<&MxArray> {
        self.caches.iter().flat_map(LayerCache::arrays).collect()
    }

    pub(super) fn async_device_prefill(&self) -> bool {
        self.device_routes.is_some()
    }

    /// TrackP12Prefill.sortedMoE's device-only route/sort/indirect execution,
    /// adapted to immutable partial banks and our existing checked replay.
    pub(super) fn tentative_prefill_experts(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        layer: usize,
    ) -> Result<MxArray> {
        let tokens = x.shape_at(1)?;
        let top = self.config.num_experts_per_tok;
        let (banks, local, capacity) = self.weights.prefill_device_expert_slots(layer, selected)?;
        let (order, inverse, sorted) = math::route_sort(&local, capacity)?
            .ok_or_else(|| Error::from_reason("Qwen4 tentative prefill requires device sorting"))?;
        let token_rows = MxArray::arange(0., tokens as f64, None, Some(DType::Uint32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top as i64])?
            .reshape(&[-1])?
            .take(&order, 0)?;
        let output = math::prefill_indirect(
            &x.reshape(&[tokens, -1])?,
            &sorted,
            &token_rows,
            &banks,
            capacity,
        )?
        .ok_or_else(|| Error::from_reason("Qwen4 tentative prefill requires indirect experts"))?;
        let output = output.reshape(&[tokens * top as i64, self.config.hidden_size as i64])?;
        let reduced = math::combine_expert_rows(&output, scores, Some(&inverse), top)?;
        self.weights.finish_expert_slots(layer, &reduced)?;
        self.device_routes
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 tentative prefill lost its route tape"))?
            .push((layer, selected.reshape(&[-1])?));
        Ok(reduced)
    }

    pub(super) fn prefill_with_device_routes(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project: bool,
    ) -> Result<MxArray> {
        if !(256..=1024).contains(&tokens.len())
            || embeddings.is_some()
            || !self.gguf()
            || !self.batch_prefill
            || !self.window_carry
            || self.verification.is_some()
            || self.device_routes.is_some()
            || self.config.hidden_size != 2560
            || self.config.moe_intermediate_size != 640
            || self.config.num_experts_per_tok != 10
            || !crate::engine::persistence::compiled_forward_backend_available()
            || !unsafe { mlx_sys::mlx_metal_is_nax_available() }
            || !super::super::tf32_enabled()
            || !self
                .weights
                .prefill_device_slots_ready(self.config.num_hidden_layers)
        {
            return self.prefill_chunk_inner(tokens, embeddings, project);
        }
        self.weights.complete_device_readers()?;
        let base = self.snapshot();
        self.device_routes = Some(Vec::with_capacity(self.config.num_hidden_layers));
        let tentative = self.prefill_chunk_inner(tokens, embeddings, project);
        let routes = self.device_routes.take();
        let output = tentative?;
        let routes = routes
            .ok_or_else(|| Error::from_reason("Qwen4 tentative prefill lost its route tape"))?;
        if routes.len() != self.config.num_hidden_layers {
            return Err(Error::from_reason(
                "Qwen4 tentative prefill has an incomplete route tape",
            ));
        }
        let tape = MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))?;
        let mut completed = vec![&output, &tape];
        completed.extend(self.pending_state_arrays());
        MxArray::eval_arrays_with_context(&completed, "qwen4::prefill_device_routes::commit")?;
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        self.weights.complete_device_readers()?;
        let ids = tape.to_uint32()?;
        let layers: Vec<_> = routes.iter().map(|(layer, _)| *layer).collect();
        let hit = self.weights.commit_device_routes(
            &layers,
            &ids,
            tokens.len() * self.config.num_experts_per_tok,
        )?;
        if hit {
            return Ok(output);
        }
        drop(output);
        self.replay_device_prefill(base, tokens, embeddings, project)
    }

    pub(in super::super) fn replay_device_prefill(
        &mut self,
        base: DecoderState,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project: bool,
    ) -> Result<MxArray> {
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
            adapter
                .rollback_last_tokens(u32::try_from(tokens.len()).map_err(|_| {
                    Error::from_reason("Qwen4 prefill replay exceeds the page token range")
                })?)
                .map_err(Error::from_reason)?;
        }
        self.restore_state(base);
        self.prefill_chunk_inner(tokens, embeddings, project)
    }

    pub(super) fn decode_submit_span(&self) -> usize {
        if self.device_routes.is_some() {
            self.config.num_hidden_layers
        } else {
            3
        }
    }

    pub(super) fn decode_should_submit(&self, completed_layers: usize) -> bool {
        // Match the upstream overlap schedule: start the first two layers,
        // then submit groups of three while building the rest of this token.
        // Ordinary decoding retains its per-layer submission and short waits.
        self.device_routes.is_none()
            || (completed_layers >= 2 && (completed_layers - 2).is_multiple_of(3))
    }

    pub(super) fn tentative_shared_experts(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        layer: usize,
    ) -> Result<Option<MxArray>> {
        if self.device_routes.is_none() {
            return Ok(None);
        }
        let shared = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{layer}.ffn_{part}_shexp.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = shared else {
            return Ok(None);
        };
        let shared = [gate, up, down];
        let Some((banks, local)) = self.weights.device_expert_slots(layer, selected)? else {
            return Ok(None);
        };
        let shared_gate = self.linear(
            x,
            &format!("layers.{layer}.mlp.shared_expert_gate.weight"),
            &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        )?;
        let Some(out) =
            math::routed_shared_experts(x, &local, scores, &banks, &shared, &shared_gate)?
        else {
            return Ok(None);
        };
        self.weights.finish_expert_slots(layer, &out)?;
        self.device_routes
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 tentative decode lost its route tape"))?
            .push((layer, selected.reshape(&[-1])?));
        Ok(Some(out))
    }

    /// Every mapped expert has a bank row, so a verify pass reads no expert
    /// and its routes can wait for the pass end. Slot banks keep the host
    /// routes: a verify pass loads the experts it misses. The shapes are the
    /// row kernels': a layer whose rows fall back to the one-row MoE would
    /// put one-row routes on the R-row tape.
    pub(super) fn verify_routes_on_device(&self) -> bool {
        self.gguf()
            && self.weights.plan.mapped
            && self.config.hidden_size == 2560
            && self.config.moe_intermediate_size == 640
            && self.config.num_experts == 512
            && self.config.num_experts_per_tok == 10
            && self
                .weights
                .device_slots_ready(self.config.num_hidden_layers)
    }

    /// `tentative_shared_experts` over the `[1, R, H]` verify rows.
    pub(super) fn tentative_shared_expert_rows(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        layer: usize,
    ) -> Result<Option<MxArray>> {
        if self.device_routes.is_none() || x.dtype()? != DType::BFloat16 {
            return Ok(None);
        }
        let shared = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{layer}.ffn_{part}_shexp.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = shared else {
            return Ok(None);
        };
        let shared = [gate, up, down];
        let Some((banks, local)) = self.weights.device_expert_slots(layer, selected)? else {
            return Ok(None);
        };
        let shared_gate = self.linear_rows(
            x,
            &format!("layers.{layer}.mlp.shared_expert_gate.weight"),
            &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        )?;
        let Some(out) =
            math::rows::routed_shared_experts(x, &local, scores, &banks, &shared, &shared_gate)?
        else {
            return Ok(None);
        };
        self.weights.finish_expert_slots(layer, &out)?;
        self.device_routes
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 tentative verify lost its route tape"))?
            .push((layer, selected.reshape(&[-1])?));
        Ok(Some(out))
    }

    pub(super) fn tentative_experts(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        layer: usize,
    ) -> Result<Option<MxArray>> {
        if self.device_routes.is_none() {
            return Ok(None);
        }
        let Some((banks, local)) = self.weights.device_expert_slots(layer, selected)? else {
            return Ok(None);
        };
        let Some(out) = math::routed_experts(x, &local, scores, &banks)? else {
            return Ok(None);
        };
        self.weights.finish_expert_slots(layer, &out)?;
        self.device_routes
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 tentative decode lost its route tape"))?
            .push((layer, selected.reshape(&[-1])?));
        Ok(Some(out))
    }

    fn ordinary_token(&mut self, token: u32, project: bool) -> Result<MxArray> {
        if self.paged.is_some() {
            self.prefill_chunk_inner(&[token], None, project)
        } else {
            self.forward_token_inner(token, project)
        }
    }

    pub(super) fn forward_with_device_routes(
        &mut self,
        token: u32,
        project: bool,
    ) -> Result<MxArray> {
        if self.device_route_cooldown > 0 {
            self.device_route_cooldown -= 1;
            return self.ordinary_token(token, project);
        }
        if !self.gguf()
            || self.verification.is_some()
            || self.config.hidden_size != 2560
            || self.config.moe_intermediate_size != 640
            || self.config.num_experts_per_tok != 10
            || !self
                .weights
                .device_slots_ready(self.config.num_hidden_layers)
        {
            return self.ordinary_token(token, project);
        }
        let base = self.snapshot();
        self.device_routes = Some(Vec::with_capacity(self.config.num_hidden_layers));
        let tentative = self.ordinary_token(token, project);
        // Always leave tentative mode, including cancellation or a read error.
        let routes = self.device_routes.take();
        let out = tentative?;
        let routes = routes
            .ok_or_else(|| Error::from_reason("Qwen4 tentative decode lost its route tape"))?;
        if routes.is_empty() {
            return Ok(out);
        }
        let tape = MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))?;
        // The normal decoder already completes its final hidden/logits. Join
        // the tape explicitly, so one readback validates all layers at once.
        let mut completed = vec![&out, &tape];
        completed.extend(self.pending_state_arrays());
        MxArray::eval_arrays_with_context(&completed, "qwen4::device_routes::commit")?;
        // A deferred final layer can leave its page write lazy. Complete it
        // before validating routes, releasing readers, or publishing output.
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        self.weights.complete_device_readers()?;
        let ids = tape.to_uint32()?;
        let layers: Vec<_> = routes.iter().map(|(layer, _)| *layer).collect();
        let hit =
            self.weights
                .commit_device_routes(&layers, &ids, self.config.num_experts_per_tok)?;
        if hit {
            return Ok(out);
        }
        drop(out);
        self.device_route_cooldown = 16;
        self.replay_device_token(base, token, project)
    }

    pub(in super::super) fn replay_device_token(
        &mut self,
        base: DecoderState,
        token: u32,
        project: bool,
    ) -> Result<MxArray> {
        // Rejected rows are beyond the committed prefix. Complete their pool
        // writes before rewinding the cursor; the replay overwrites those same
        // rows. The adapter retains the physical blocks, as in MTP rejection.
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
            adapter
                .rollback_last_tokens(1)
                .map_err(Error::from_reason)?;
        }
        self.restore_state(base);
        self.ordinary_token(token, project)
    }
}
