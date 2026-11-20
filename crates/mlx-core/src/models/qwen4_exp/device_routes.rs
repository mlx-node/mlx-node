//! Tentative GPU routing for warm partial banks. A token is published only
//! after its complete route tape has been checked against the unchanged host
//! mappings. A miss keeps the layers before the first missed layer and runs
//! the rest again: on device routes after reading the missed experts, or on
//! the ordinary loader once the token re-ran a whole pass. This never
//! predicts an expert or approximates routing.
use super::*;

/// Device-route tokens the route guard weighs.
const GUARD_WINDOW: usize = 32;
/// The mean cost per token, in device layers, past which host routes are
/// cheaper than trying device routes (break-even ~78 on 48 layers).
const GUARD_LIMIT: u64 = 64;
/// Decode tokens a tripped guard sends to host routes.
const GUARD_HOST_TOKENS: usize = 16;
/// A layer run again on host routes costs about this many device layers.
const HOST_LAYER_COST: u64 = 3;

/// What route misses cost a request: each device-route token adds the
/// device layers it ran again plus [`HOST_LAYER_COST`] per layer it ran
/// again on host routes. When the last [`GUARD_WINDOW`] tokens add more
/// than [`GUARD_LIMIT`] each on average, the next [`GUARD_HOST_TOKENS`]
/// tokens run on host routes and the window starts empty. The request
/// keeps its guard across row switches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in super::super) struct RouteGuard {
    costs: [u64; GUARD_WINDOW],
    next: usize,
    sum: u64,
    host_tokens: usize,
}

impl RouteGuard {
    /// True when this decode token runs on host routes; counts it.
    pub(in super::super) fn host_token(&mut self) -> bool {
        if self.host_tokens == 0 {
            return false;
        }
        self.host_tokens -= 1;
        true
    }

    /// Adds the cost of a device-route token; true when the guard trips.
    pub(in super::super) fn record(&mut self, cost: u64) -> bool {
        self.sum = self.sum - self.costs[self.next] + cost;
        self.costs[self.next] = cost;
        self.next = (self.next + 1) % GUARD_WINDOW;
        if self.sum <= GUARD_WINDOW as u64 * GUARD_LIMIT {
            return false;
        }
        *self = Self {
            host_tokens: GUARD_HOST_TOKENS,
            ..Self::default()
        };
        true
    }
}

impl Decoder {
    pub(in super::super) fn deferred_state_submission(&self) -> bool {
        self.device_routes.is_some()
    }

    pub(super) fn deferred_forward_completion(&self) -> bool {
        self.deferred_state_submission() && self.verification.is_none()
    }

    pub(super) fn pending_state_arrays(&self) -> Vec<&MxArray> {
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

    pub(super) fn prefill_group_with_device_routes(
        &mut self,
        windows: &[&[u32]],
        next: &[u32],
        embeddings: Option<&MxArray>,
        project: bool,
    ) -> Result<MxArray> {
        if windows
            .iter()
            .any(|tokens| !(256..=1024).contains(&tokens.len()))
            || embeddings.is_some()
            || !self.gguf()
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
            return self.prefill_group_inner(windows, next, embeddings, project);
        }
        self.weights.complete_device_readers()?;
        // Only a one-window pass replays a miss; a group of more is an error.
        let base = (windows.len() == 1).then(|| self.snapshot());
        self.device_routes = Some(Vec::with_capacity(
            self.config.num_hidden_layers * windows.len(),
        ));
        let tentative = self.prefill_group_inner(windows, next, embeddings, project);
        let routes = self.device_routes.take();
        let output = tentative?;
        let Some(routes) = routes else {
            if windows.len() > 1 {
                // A trip between two sweeps committed the tape so far and
                // closed the transaction; the rest of the group ran plain.
                return Ok(output);
            }
            return Err(Error::from_reason(
                "Qwen4 tentative prefill lost its route tape",
            ));
        };
        if routes.len() != self.config.num_hidden_layers * windows.len() {
            return Err(Error::from_reason(
                "Qwen4 tentative prefill has an incomplete route tape",
            ));
        }
        let streams = std::mem::take(&mut self.group_streams);
        let mut outputs = vec![&output];
        outputs.extend(streams.iter());
        let hit = self.commit_tentative(&outputs, &routes);
        self.group_streams = streams;
        #[cfg(test)]
        self.tentative_commits.push(windows.len());
        if hit? {
            return Ok(output);
        }
        let Some(base) = base else {
            return Err(Error::from_reason(
                "Qwen4 prefill group routes lost their expert banks",
            ));
        };
        drop(output);
        self.replay_device_prefill(base, windows[0], embeddings, project)
    }

    /// Complete a tentative pass and check its tape: `outputs` (the logits
    /// and every residual stream), the tape and every state array evaluate
    /// together, then routes are checked against the unchanged host mapping.
    pub(super) fn commit_tentative(
        &mut self,
        outputs: &[&MxArray],
        routes: &[(usize, MxArray)],
    ) -> Result<bool> {
        let tape = MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))?;
        let mut completed = outputs.to_vec();
        completed.push(&tape);
        completed.extend(self.pending_state_arrays());
        MxArray::eval_arrays_with_context(&completed, "qwen4::prefill_device_routes::commit")?;
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        self.weights.complete_device_readers()?;
        self.note("qwen4::routes::commit");
        let ids = tape.to_uint32()?;
        let mut entries = Vec::with_capacity(routes.len());
        let mut offset = 0;
        for (layer, selected) in routes {
            let len = selected.shape_at(0)? as usize;
            let end = offset + len;
            let Some(entry) = ids.get(offset..end) else {
                return Err(Error::from_reason(
                    "Qwen4 device route tape has an invalid shape",
                ));
            };
            entries.push((*layer, entry));
            offset = end;
        }
        if offset != ids.len() {
            return Err(Error::from_reason(
                "Qwen4 device route tape has an invalid shape",
            ));
        }
        self.weights.commit_device_route_entries(&entries)
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
        self.weights
            .live
            .route_replays
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The tentative pass already requested the next window's rows.
        self.prefill_chunk_inner(tokens, &[], embeddings, project)
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
        #[cfg(test)]
        {
            self.fused_shared_layers += 1;
        }
        Ok(Some(out))
    }

    /// Full direct banks have every expert, so a verify pass reads no expert
    /// and its routes can wait for the pass end. Partial banks keep host
    /// routes: a verify pass loads the experts it misses. The shapes are the
    /// row kernels': a layer whose rows fall back to the one-row MoE would
    /// put one-row routes on the R-row tape.
    pub(super) fn verify_routes_on_device(&self) -> bool {
        self.gguf()
            && self.weights.plan.direct
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
        self.decode_token(token, &[], None, self.paged.is_some(), project)
    }

    pub(super) fn forward_with_device_routes(
        &mut self,
        token: u32,
        project: bool,
    ) -> Result<MxArray> {
        #[cfg(test)]
        if self.route_mode_pin == Some(RouteMode::Host) {
            return self.ordinary_token(token, project);
        }
        if self.route_guard.host_token() {
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
        self.weights
            .live
            .route_tries
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let base = self.snapshot();
        let result = self.device_token(&base, token, project);
        self.weights.end_preloads();
        let (out, cost) = result?;
        if self.route_guard.record(cost) {
            self.weights
                .live
                .route_guard_trips
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(out)
    }

    /// One decode token on device routes. A pass that names an expert with
    /// no slot keeps the layers before the first such layer m: the token
    /// reads the experts layer m routed (and, as hints, those of the later
    /// layers that missed in that pass) and runs layers m.. again on device
    /// routes, from the stream that pass fed layer m. A pass stops building
    /// layers once it reads such a miss from a finished route group. Once
    /// the token's device layers built twice would pass one whole pass,
    /// layers m.. run on host routes instead. Returns the token's output and
    /// its cost for the route guard.
    fn device_token(
        &mut self,
        base: &DecoderState,
        token: u32,
        project: bool,
    ) -> Result<(MxArray, u64)> {
        let layers = self.config.num_hidden_layers;
        let top = self.config.num_experts_per_tok;
        self.device_routes = Some(Vec::with_capacity(layers));
        let tentative = self.decode_pass(token, &[], None, self.paged.is_some(), project);
        let mut pass = self.check_device_pass(tentative)?;
        let mut rerun = 0;
        let mut first_pass = true;
        loop {
            let entries = pass.entries(top);
            let missed = self.weights.first_unmapped(&entries);
            let kept = entries
                .iter()
                .take_while(|(layer, _)| missed.is_none_or(|from| *layer < from))
                .count();
            if !self.weights.commit_device_route_entries(&entries[..kept])? {
                return Err(Error::from_reason(
                    "Qwen4 device route commit refused mapped routes",
                ));
            }
            let Some(from) = missed else {
                let out = pass.out.ok_or_else(|| {
                    Error::from_reason("Qwen4 device pass stopped on mapped routes")
                })?;
                return Ok((out, rerun as u64));
            };
            if std::mem::take(&mut first_pass) {
                self.weights
                    .live
                    .route_miss_tokens
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // The layers from..end this pass built run again.
            let built = pass.end.saturating_sub(from);
            let device = rerun + built <= layers
                && pass
                    .inputs
                    .as_ref()
                    .is_some_and(|inputs| from < inputs.len())
                && self.preload_routes(from, &entries, &pass.late)?;
            drop(entries);
            let CheckedPass { out, inputs, .. } = pass;
            drop(out);
            if !device {
                let host = (layers - from) as u64;
                let out = self.replay_from_layer(base, from, inputs, token, project)?;
                return Ok((out, rerun as u64 + HOST_LAYER_COST * host));
            }
            let inputs = inputs
                .ok_or_else(|| Error::from_reason("Qwen4 device replay lost its layer inputs"))?;
            let x = inputs
                .get(from)
                .cloned()
                .ok_or_else(|| Error::from_reason("Qwen4 device replay lost its layer inputs"))?;
            self.restart_from(base, from)?;
            rerun += built;
            #[cfg(test)]
            {
                self.device_rerun_peak = self.device_rerun_peak.max(rerun);
            }
            self.weights
                .live
                .replayed_layers
                .fetch_add(built as u64, std::sync::atomic::Ordering::Relaxed);
            self.layer_inputs = Some(inputs);
            self.device_routes = Some(Vec::with_capacity(layers - from));
            let tentative = self.decode_layers(from, x, token, project);
            pass = self.check_device_pass(tentative)?;
        }
    }

    /// At a group submit of a device pass: read, without waiting and oldest
    /// first, the pass's route entries from `polled` up to `upto` (the
    /// groups submitted before this one) until one is still in flight.
    /// True when a finished entry names an expert with no slot.
    pub(super) fn routes_missed(&mut self, upto: usize, polled: &mut usize) -> Result<bool> {
        let Some(routes) = &self.device_routes else {
            return Ok(false);
        };
        let mut missed = false;
        while *polled < upto
            && let Some((layer, entry)) = routes.get(*polled)
        {
            let Some(ids) = entry.ready_ids()? else {
                #[cfg(test)]
                self.unready_polls.set(self.unready_polls.get() + 1);
                break;
            };
            if self.weights.first_unmapped(&[(*layer, &ids)]).is_some() {
                missed = true;
                break;
            }
            *polled += 1;
        }
        #[cfg(test)]
        {
            self.early_stops += u64::from(missed);
        }
        Ok(missed)
    }

    /// The route entries a group submit reads, by the test pin: `older`
    /// (the live rule), or every entry submitted so far, as in flight or
    /// after the GPU completed them.
    #[cfg(test)]
    pub(super) fn pinned_poll(&self, older: usize, submitted: usize) -> usize {
        match self.after_group_submit {
            None | Some(GroupPoll::Never | GroupPoll::Older) => older,
            Some(GroupPoll::InFlight) => submitted,
            Some(GroupPoll::Ready) => {
                crate::array::synchronize();
                submitted
            }
        }
    }

    /// Leave tentative mode after a device pass, including on its error,
    /// then complete the pass with its route tape, page writes and readers
    /// and read the tape back. A stopped pass keeps its work in flight: it
    /// reads the finished entries of its tape, which hold its miss.
    fn check_device_pass(&mut self, tentative: Result<LayerPass>) -> Result<CheckedPass> {
        let routes = self.device_routes.take();
        let inputs = self.layer_inputs.take();
        let pass = tentative?;
        let routes = routes
            .ok_or_else(|| Error::from_reason("Qwen4 tentative decode lost its route tape"))?;
        let out = match pass {
            LayerPass::Done(out) => out,
            LayerPass::Stopped { at } => return self.stopped_pass(routes, at, inputs),
        };
        let end = self.config.num_hidden_layers;
        if routes.is_empty() {
            return Ok(CheckedPass {
                out: Some(out),
                end,
                layers: Vec::new(),
                ids: Vec::new(),
                late: Vec::new(),
                inputs,
            });
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
        let ids = tape.to_uint32()?.to_vec();
        let layers = routes.iter().map(|(layer, _)| *layer).collect();
        self.checked_pass(Some(out), end, layers, ids, inputs)
    }

    /// A device pass stopped before layer `at` on a miss its group reads
    /// found: its tape entries up to that miss, which the read found
    /// finished, read without waiting. The later entries stay unread for
    /// the preload, which reads them as hints once it has waited for the
    /// device. Queued layers complete when the preload or the restart wait
    /// for them.
    fn stopped_pass(
        &mut self,
        routes: Vec<(usize, MxArray)>,
        at: usize,
        inputs: Option<Vec<MxArray>>,
    ) -> Result<CheckedPass> {
        let (mut layers, mut ids) = (Vec::new(), Vec::new());
        let mut late = Vec::new();
        let mut routes = routes.into_iter();
        for (layer, entry) in routes.by_ref() {
            let Some(read) = entry.ready_ids()? else {
                late.push((layer, entry));
                break;
            };
            let missed = self.weights.first_unmapped(&[(layer, &read)]).is_some();
            layers.push(layer);
            ids.extend(read);
            if missed {
                break;
            }
        }
        late.extend(routes);
        let mut pass = self.checked_pass(None, at, layers, ids, inputs)?;
        pass.late = late;
        Ok(pass)
    }

    fn checked_pass(
        &mut self,
        out: Option<MxArray>,
        end: usize,
        layers: Vec<usize>,
        ids: Vec<u32>,
        inputs: Option<Vec<MxArray>>,
    ) -> Result<CheckedPass> {
        let top = self.config.num_experts_per_tok;
        if top == 0 || ids.len() != layers.len() * top {
            return Err(Error::from_reason(
                "Qwen4 device route tape has an invalid shape",
            ));
        }
        // A layer that fell back to host reads has no entry: a replay
        // starts at the missed entry's own layer, never at its position.
        if layers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::from_reason(
                "Qwen4 device route tape is out of layer order",
            ));
        }
        let pass = CheckedPass {
            out,
            end,
            layers,
            ids,
            late: Vec::new(),
            inputs,
        };
        #[cfg(test)]
        self.route_tapes.push(
            pass.entries(top)
                .iter()
                .map(|&(layer, ids)| (layer, ids.to_vec()))
                .collect(),
        );
        Ok(pass)
    }

    /// Before a device pass from layer `from`: read the experts `from`
    /// routed, and as hints those of each later layer whose routes missed
    /// the slots: its checked entries, then the `late` entries of a stopped
    /// pass that have finished once that read waited for the device. False
    /// when layer `from` cannot hold its routes in slots.
    fn preload_routes(
        &mut self,
        from: usize,
        entries: &[(usize, &[u32])],
        late: &[(usize, MxArray)],
    ) -> Result<bool> {
        let experts = self.config.num_experts;
        let top = self.config.num_experts_per_tok;
        #[cfg(test)]
        let mut loaded: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut later = Vec::new();
        for &(layer, ids) in entries.iter().filter(|(layer, _)| *layer >= from) {
            if layer != from {
                later.push((layer, ids.to_vec()));
                continue;
            }
            if !self
                .weights
                .preload_expert_slots(layer, experts, top, ids)?
            {
                return Ok(false);
            }
            #[cfg(test)]
            loaded.push((layer, ids.to_vec()));
        }
        if !late.is_empty() {
            // Entries of a stopped pass serve as hints only once complete;
            // a fill that read nothing waited for none of them.
            crate::array::synchronize();
        }
        for (layer, entry) in late {
            let Some(ids) = entry.ready_ids()? else {
                break;
            };
            later.push((*layer, ids));
        }
        for (layer, ids) in later {
            if self.weights.first_unmapped(&[(layer, &ids)]).is_none() {
                continue;
            }
            #[cfg(test)]
            let ids = if self.wrong_hints {
                (0..experts as u32)
                    .filter(|e| !ids.contains(e))
                    .take(top)
                    .collect()
            } else {
                ids
            };
            // A hint that does not fit costs only a later miss.
            let _held = self
                .weights
                .preload_expert_slots(layer, experts, top, &ids)?;
            #[cfg(test)]
            loaded.push((layer, ids));
        }
        #[cfg(test)]
        for (layer, ids) in loaded {
            let mapped = self.weights.first_unmapped(&[(layer, &ids)]).is_none();
            self.preload_checks.push((layer, mapped));
        }
        Ok(true)
    }

    /// Give layers `from..` the caches they held before the token began and
    /// `history` its length: a replay from `from` runs them again. The page
    /// cursor keeps the token: the replay writes its rows again at the same
    /// position, so the cursor must not roll back.
    fn restart_from(&mut self, base: &DecoderState, from: usize) -> Result<()> {
        let layers = self.config.num_hidden_layers;
        if base.caches.len() != layers || self.caches.len() != layers {
            return Err(Error::from_reason(
                "Qwen4 decode replay has a cache per layer mismatch",
            ));
        }
        if let Some(adapter) = &mut self.paged {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        self.caches[from..].clone_from_slice(&base.caches[from..]);
        self.history.truncate(base.history.len());
        self.layer_inputs = None;
        self.weights
            .live
            .route_replays
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Finish the decode token a tentative pass began, from layer `from`
    /// on host routes. Layers before `from` keep what the passes wrote;
    /// layers `from..` get `base`'s caches back and run again on
    /// `inputs[from]`, the stream the last pass fed layer `from`.
    pub(in super::super) fn replay_from_layer(
        &mut self,
        base: &DecoderState,
        from: usize,
        inputs: Option<Vec<MxArray>>,
        token: u32,
        project: bool,
    ) -> Result<MxArray> {
        let x = inputs
            .and_then(|mut inputs| (from < inputs.len()).then(|| inputs.swap_remove(from)))
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen4 decode replay holds no input for layer {from}"
                ))
            })?;
        self.restart_from(base, from)?;
        self.weights.live.host_tail_layers.fetch_add(
            (self.config.num_hidden_layers - from) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        match self.decode_layers(from, x, token, project)? {
            LayerPass::Done(out) => Ok(out),
            LayerPass::Stopped { at } => Err(Error::from_reason(format!(
                "Qwen4 decode replay stopped before layer {at}"
            ))),
        }
    }
}

/// A checked device pass: its output (`None` when it stopped), the layer it
/// stopped before (or the layer count), the layer and ids of each route
/// tape entry it read, the entries a stopped pass left unread, and the
/// layer inputs it kept.
struct CheckedPass {
    out: Option<MxArray>,
    end: usize,
    layers: Vec<usize>,
    ids: Vec<u32>,
    late: Vec<(usize, MxArray)>,
    inputs: Option<Vec<MxArray>>,
}

impl CheckedPass {
    fn entries(&self, top: usize) -> Vec<(usize, &[u32])> {
        self.layers
            .iter()
            .copied()
            .zip(self.ids.chunks_exact(top))
            .collect()
    }
}
