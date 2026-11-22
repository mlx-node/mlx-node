//! Matrix batching for prompt windows and scheduled decode rows. Stateful
//! attention/PLE preserve each owner's order; stateless HC/MLP share weights.
use super::super::math::ExpertKernel;
use super::super::weights::Weight;
use super::*;
use std::collections::BTreeMap;

type Banks = [std::sync::Arc<Weight>; 3];

/// One window's expert rows, built piece by piece on leased banks with the
/// window's kernel, and the inverse that puts them back in router order.
struct WindowRows {
    source: MxArray,
    kernel: ExpertKernel,
    top: usize,
    values: Vec<MxArray>,
    group_inverses: Vec<MxArray>,
    inverse: Vec<i32>,
    output_row: i32,
}

impl WindowRows {
    fn new(x: &MxArray, assignments: usize, kernel: ExpertKernel, top: usize) -> Result<Self> {
        Ok(Self {
            source: x.reshape(&[-1, x.shape_at(2)?])?,
            kernel,
            top,
            values: Vec::new(),
            group_inverses: Vec::new(),
            inverse: vec![0; assignments],
            output_row: 0,
        })
    }

    /// The rows of `assignments` (indices into the window's routes, not
    /// empty) whose experts sit at `slots` of `banks`.
    fn add(
        &mut self,
        assignments: &[usize],
        slots: &[u32],
        banks: &Banks,
        count: usize,
    ) -> Result<MxArray> {
        let rows: Vec<_> = assignments.iter().map(|&s| (s / self.top) as i32).collect();
        let token_rows = MxArray::from_int32(&rows, &[rows.len() as i64])?;
        let selected = MxArray::from_uint32(slots, &[slots.len() as i64])?;
        let (out, local_inverse) = Decoder::window_expert_rows(
            self.kernel,
            &self.source,
            &token_rows,
            &selected,
            banks,
            count,
        )?;
        let local_inverse = match local_inverse {
            Some(inverse) => inverse.astype(DType::Int32)?,
            None => MxArray::arange(0.0, assignments.len() as f64, None, Some(DType::Int32))?,
        };
        self.group_inverses
            .push(local_inverse.add(&MxArray::from_int32(&[self.output_row], &[])?)?);
        for &slot in assignments {
            self.inverse[slot] = self.output_row;
            self.output_row += 1;
        }
        self.values.push(out.clone());
        Ok(out)
    }

    fn finish(self) -> Result<(MxArray, MxArray)> {
        let out = MxArray::concatenate_many(self.values.iter().collect(), Some(0))?;
        let inverse = MxArray::concatenate_many(self.group_inverses.iter().collect(), Some(0))?
            .take(
                &MxArray::from_int32(&self.inverse, &[self.inverse.len() as i64])?,
                0,
            )?;
        Ok((out, inverse))
    }
}

struct LayerSlots<'a> {
    decoder: &'a mut Decoder,
    layer: usize,
}

impl GroupBanks for LayerSlots<'_> {
    fn lease(&mut self, experts: &[u32]) -> Result<Option<(Banks, Vec<u32>, usize)>> {
        self.decoder.check_cancelled()?;
        let (count, top) = (
            self.decoder.config.num_experts,
            self.decoder.config.num_experts_per_tok,
        );
        self.decoder
            .weights
            .expert_slots(self.layer, count, top, experts)
    }

    fn release(&mut self, output: &MxArray) -> Result<()> {
        self.decoder.weights.finish_expert_slots(self.layer, output)
    }
}

impl Decoder {
    /// Layer `i`'s resident expert banks, when the routed experts run on them.
    fn resident_banks(&self, i: usize) -> Option<Banks> {
        if !self.gguf() {
            return None;
        }
        let banks = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_exps.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = banks else {
            return None;
        };
        // The dense F32 reference preserves singleton GEMV association.
        if [&gate, &up, &down].iter().any(|w| w.scales.is_none()) {
            return None;
        }
        Some([gate, up, down])
    }

    /// Whether layer `i` of a group of `windows` runs its routed experts as
    /// one union lease: windows above the fused small-window size on a
    /// partial store's slots, where each window would lease on its own.
    pub(super) fn union_lease(&mut self, windows: &[&[u32]], i: usize) -> Result<bool> {
        let (experts, top) = (self.config.num_experts, self.config.num_experts_per_tok);
        Ok(windows.len() > 1
            && windows.iter().all(|w| w.len() > 8)
            && self.verification.is_none()
            && self.device_routes.is_none()
            && self.gguf()
            && !self.weights.plan.direct
            && self.resident_banks(i).is_none()
            && self.weights.expert_slot_capacity(experts, top)? > 0)
    }

    /// Resident shared-expert `[gate, up, down]` banks for layer `i` (GGUF).
    fn cuda_shared_banks(&self, i: usize) -> Option<Banks> {
        if !self.gguf() {
            return None;
        }
        let [Some(gate), Some(up), Some(down)] = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_shexp.weight"))
        }) else {
            return None;
        };
        Some([gate, up, down])
    }

    /// Fused CUDA routed + shared MoE for layer `i` on direct/resident banks:
    /// - `M in 1..=16`: `moe_routed_shared_few_cuda` (routed + shared in one primitive!)
    /// - `M >= 17`: `shared_expert_parts` + `moe_prefill_cuda` (routed + shared combine!)
    ///
    /// Returns `Ok(None)` when disabled or off-contract so callers fall back cleanly.
    pub(super) fn cuda_fused_moe(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        if !super::super::cuda_qwen4_fast_enabled() || !self.gguf() || !self.scope.is_empty() {
            return Ok(None);
        }
        let c = self.config.clone();
        if c.hidden_size != 2560
            || c.moe_intermediate_size != 640
            || c.num_experts_per_tok == 0
            || c.num_experts_per_tok > 10
            || x.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let (banks, local) = if let Some(banks) = self.resident_banks(i) {
            (banks, selected.reshape(&[-1])?)
        } else if self.weights.plan.direct {
            let Some((banks, local)) = self.weights.device_expert_slots(i, selected)? else {
                return Ok(None);
            };
            (banks, local)
        } else {
            return Ok(None);
        };
        let tokens = x.shape_at(1)?;
        if (1..=16).contains(&tokens) {
            if let Some(shared_banks) = self.cuda_shared_banks(i)
                && shared_banks
                    .iter()
                    .all(|w| w.mode == "affine" && w.bits == 8 && w.group == 32)
            {
                let shared_gate = self.linear_rows(
                    x,
                    &format!("layers.{i}.mlp.shared_expert_gate.weight"),
                    &format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                )?;
                if let Some(out) = math::cuda_moe_routed_shared_few(
                    x,
                    &local,
                    scores,
                    &banks,
                    Some((&shared_banks, &shared_gate)),
                )? {
                    return Ok(Some(out));
                }
            }
            if let Some(routed) = math::cuda_moe_routed_shared_few(x, &local, scores, &banks, None)?
            {
                return self.with_shared_expert(x, &routed, i).map(Some);
            }
            return Ok(None);
        }
        let (shared_out, shared_gate) = self.shared_expert_parts(x, i)?;
        if let Some(out) =
            math::cuda_moe_prefill(x, &local, scores, &banks, Some((&shared_out, &shared_gate)))?
        {
            return Ok(Some(out));
        }
        Ok(None)
    }

    pub(super) fn resident_experts(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        let (banks, ids, experts) = if let Some(banks) = self.resident_banks(i) {
            (banks, selected.reshape(&[-1])?, self.config.num_experts)
        } else if self.weights.plan.direct
            && !crate::engine::persistence::compiled_forward_backend_available()
        {
            // Non-Metal direct residency (Windows CUDA): wait for layer `i`'s
            // direct fill if still running and run routed experts directly on
            // the GPU-resident banks without host route readbacks.
            let Some((banks, local)) = self.weights.device_expert_slots(i, selected)? else {
                return Ok(None);
            };
            (banks, local, self.config.num_experts)
        } else {
            return Ok(None);
        };
        let tokens = x.shape()?[1];
        let top = self.config.num_experts_per_tok;
        let h = self.config.hidden_size;
        if top == 10
            && let Some(out) = math::routed_experts(x, &ids, scores, &banks)?
        {
            return Ok(Some(out));
        }
        if super::super::cuda_qwen4_fast_enabled() && top <= 10 && h == 2560 {
            let cuda_out = if (1..=16).contains(&tokens) {
                math::cuda_moe_routed_shared_few(x, &ids, scores, &banks, None)?
            } else {
                math::cuda_moe_prefill(x, &ids, scores, &banks, None)?
            };
            if let Some(out) = cuda_out {
                return Ok(Some(out));
            }
        }
        let token_rows = MxArray::arange(0.0, tokens as f64, None, Some(DType::Int32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top as i64])?
            .reshape(&[-1])?;
        let source = x.reshape(&[tokens, h as i64])?;
        let kernel = ExpertKernel::for_window(ids.size()? as usize, experts);
        let (output, inverse) =
            Self::window_expert_rows(kernel, &source, &token_rows, &ids, &banks, experts)?;
        math::combine_expert_rows(&output, scores, inverse.as_ref(), top).map(Some)
    }

    #[cfg(test)]
    pub(in super::super) fn expert_assignment_rows(
        input: &MxArray,
        ids: &MxArray,
        banks: &Banks,
        experts: usize,
    ) -> Result<MxArray> {
        let kernel = ExpertKernel::for_window(ids.size()? as usize, experts);
        let (output, inverse) =
            Self::sorted_expert_assignment_rows(input, ids, banks, experts, kernel)?;
        match inverse {
            Some(inverse) => output.take(&inverse, 0),
            None => Ok(output),
        }
    }

    /// The rows of one window, or of one slot group of it, through the
    /// window's kernel. `experts` is the row count of the bank in hand.
    fn window_expert_rows(
        kernel: ExpertKernel,
        source: &MxArray,
        token_rows: &MxArray,
        ids: &MxArray,
        banks: &Banks,
        experts: usize,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if kernel == ExpertKernel::Tiles
            && let Some((out, inverse)) =
                Self::indirect_expert_assignment_rows(source, token_rows, ids, banks, experts)?
        {
            return Ok((out, Some(inverse)));
        }
        Self::sorted_expert_assignment_rows(
            &source.take(token_rows, 0)?,
            ids,
            banks,
            experts,
            kernel,
        )
    }

    fn indirect_expert_assignment_rows(
        input: &MxArray,
        token_rows: &MxArray,
        ids: &MxArray,
        banks: &Banks,
        experts: usize,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if input.dtype()? != DType::BFloat16 {
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
        banks: &Banks,
        experts: usize,
        kernel: ExpertKernel,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let rows = ids.size()? as i64;
        let width = input.shape()?[1];
        let (x, selected, inverse, tiles) = if kernel == ExpertKernel::Rows {
            (input.clone(), ids.clone(), None, None)
        } else {
            let (order, inverse, selected) = match math::route_sort(ids, experts)? {
                Some(sorted) => sorted,
                None => {
                    let order = ids.argsort(Some(0))?;
                    (order.clone(), order.argsort(Some(0))?, ids.take(&order, 0)?)
                }
            };
            let tiles = if kernel == ExpertKernel::Tiles {
                math::expert_tiles(&selected, experts)?
            } else {
                None
            };
            (input.take(&order, 0)?, selected, Some(inverse), tiles)
        };
        let x = x.reshape(&[rows, 1, width])?;
        let project = |bank: &Weight, x: &MxArray| -> Result<MxArray> {
            if kernel == ExpertKernel::Rows {
                return bank.expert_rows(x, &selected, experts, false);
            }
            if let Some(tiles) = &tiles
                && let Some(out) = bank.tiled_expert_rows(x, &selected, tiles, experts)?
            {
                return Ok(out);
            }
            // MLX picks its matrix kernel from rows / bank experts, a bank
            // property; only a Tiles window the NAX kernels refused may let it.
            bank.gather_rows(x, &selected, experts, kernel == ExpertKernel::Tiles)
        };
        let gate = project(&banks[0], &x)?;

        let up = project(&banks[1], &x)?;

        let hidden = Activations::swiglu_compiled(&gate, &up)?;

        let out = project(&banks[2], &hidden)?.reshape(&[rows, -1])?;

        Ok((out, inverse))
    }

    // Every assignment belongs to this bank, so token rows have a regular
    // layout and the GPU sort already supplies the complete inverse. Avoid
    // uploading host-built identity maps or composing a second inverse.
    pub(in super::super) fn single_slot_group_rows(
        x: &MxArray,
        selected: &MxArray,
        banks: &Banks,
        capacity: usize,
        kernel: ExpertKernel,
        top: usize,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let tokens = x.shape_at(1)?;
        let source = x.reshape(&[tokens, -1])?;
        let token_rows = MxArray::arange(0.0, tokens as f64, None, Some(DType::Uint32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top as i64])?
            .reshape(&[-1])?;
        Self::window_expert_rows(kernel, &source, &token_rows, selected, banks, capacity)
    }

    /// A window over more experts than the slots hold: each group of
    /// `groups` (assignment indices into `ids`, any subset of the window's
    /// experts) runs on its leased bank with the window's kernel, whatever
    /// the group's size; an empty group takes no lease. Returns the stacked
    /// rows and the inverse that restores router order, or None once a lease
    /// is refused.
    pub(in super::super) fn grouped_expert_rows(
        x: &MxArray,
        ids: &[u32],
        groups: &[Vec<usize>],
        kernel: ExpertKernel,
        top: usize,
        banks: &mut impl GroupBanks,
    ) -> Result<Option<(MxArray, MxArray)>> {
        let mut rows = WindowRows::new(x, ids.len(), kernel, top)?;
        for assignments in groups {
            // The device sort refuses zero rows.
            if assignments.is_empty() {
                continue;
            }
            let requested: Vec<_> = assignments.iter().map(|&s| ids[s]).collect();
            let Some((group_banks, slots, count)) = banks.lease(&requested)? else {
                return Ok(None);
            };
            let out = rows.add(assignments, &slots, &group_banks, count)?;
            banks.release(&out)?;
        }
        rows.finish().map(Some)
    }

    /// Layer `i`'s routed sums for every window of a group (or one window),
    /// `(x, ids, scores)` each, over one lease plan: the union's experts
    /// that hold a slot run first, then the missing ones in last-use chunks
    /// of the capacity, each sub-group leased once for all windows, so a
    /// missing expert is read once per layer. The reads of each next
    /// sub-group run while the GPU computes the rows of the one before.
    /// Each window keeps its own kernel class and its rows return to router
    /// order before its sum. None when the store leases nothing for this
    /// layer.
    pub(super) fn union_expert_rows(
        &mut self,
        windows: &[(MxArray, Vec<u32>, MxArray)],
        i: usize,
    ) -> Result<Option<Vec<MxArray>>> {
        let (experts, top) = (self.config.num_experts, self.config.num_experts_per_tok);
        let group = windows.len() > 1;
        let capacity = self.weights.expert_slot_capacity(experts, top)?;
        let all: Vec<u32> = windows
            .iter()
            .flat_map(|(_, ids, _)| ids.iter().copied())
            .collect();
        let order =
            super::super::route_plan::expert_order(&all, experts).map_err(Error::from_reason)?;
        let resident = self.weights.experts_in_slots(i, &order);
        let mut held = vec![false; experts];
        for &e in &resident {
            held[e as usize] = true;
        }
        let missing: Vec<u32> = order
            .iter()
            .copied()
            .filter(|&e| !held[e as usize])
            .collect();
        let mut sub_groups: Vec<&[u32]> = Vec::new();
        if !resident.is_empty() {
            sub_groups.push(&resident);
        }
        sub_groups.extend(missing.chunks(capacity.max(1)));
        let mut group_of = vec![usize::MAX; experts];
        for (g, members) in sub_groups.iter().enumerate() {
            for &e in *members {
                group_of[e as usize] = g;
            }
        }
        let mut rows = Vec::with_capacity(windows.len());
        let mut pieces = Vec::with_capacity(windows.len());
        for (x, ids, _) in windows {
            let kernel = ExpertKernel::for_window(ids.len(), experts);
            if group {
                self.note(&format!("qwen4::union::{i}::{kernel:?}"));
            }
            let mut window_pieces = vec![Vec::new(); sub_groups.len()];
            for (assignment, &e) in ids.iter().enumerate() {
                window_pieces[group_of[e as usize]].push(assignment);
            }
            pieces.push(window_pieces);
            rows.push(WindowRows::new(x, ids.len(), kernel, top)?);
        }
        let mut slot_of = vec![0u32; experts];
        let mut ahead = false;
        for (g, members) in sub_groups.iter().enumerate() {
            self.check_cancelled()?;
            let Some((banks, slots, count)) =
                self.weights.expert_slots(i, experts, top, members)?
            else {
                return Ok(None);
            };
            if ahead {
                self.note(&format!("qwen4::fill::finish::{i}"));
            }
            for (&e, &slot) in members.iter().zip(&slots) {
                slot_of[e as usize] = slot;
            }
            #[cfg(test)]
            let watched = self
                .fill_bank_checks
                .is_some()
                .then(|| self.weights.slot_rows_bits(i, &slots));
            ahead = match sub_groups.get(g + 1) {
                Some(next) => self.weights.begin_expert_fill(i, experts, top, next)?,
                None => false,
            };
            if ahead {
                self.note(&format!("qwen4::fill::begin::{i}"));
            }
            let mut outputs = Vec::new();
            for ((_, ids, _), (window_rows, window_pieces)) in
                windows.iter().zip(rows.iter_mut().zip(&pieces))
            {
                let assignments = &window_pieces[g];
                // The device sort refuses zero rows.
                if assignments.is_empty() {
                    continue;
                }
                let slots: Vec<u32> = assignments
                    .iter()
                    .map(|&s| slot_of[ids[s] as usize])
                    .collect();
                let out = window_rows.add(assignments, &slots, &banks, count)?;
                self.weights.finish_expert_slots(i, &out)?;
                outputs.push(out);
            }
            if ahead {
                self.submit(
                    &outputs.iter().collect::<Vec<_>>(),
                    &format!("qwen4::fill::rows::{i}"),
                )?;
            }
            #[cfg(test)]
            if let (Some(before), Some(victims)) = (watched, self.weights.pending_victims())
                && ahead
            {
                let read = victims.iter().any(|&v| slots.contains(&(v as u32)));
                let unchanged = before == self.weights.slot_rows_bits(i, &slots);
                if let Some(checks) = self.fill_bank_checks.as_mut() {
                    checks.push((i, read, unchanged));
                }
            }
        }
        self.weights.touch_in_order(i, &order);
        #[cfg(test)]
        if group {
            self.union_routes.push((i, all));
            self.group_held.push(
                rows.iter()
                    .flat_map(|window| &window.values)
                    .map(|values| values.nbytes() as u64)
                    .sum(),
            );
        }
        let mut sums = Vec::with_capacity(windows.len());
        for ((_, _, scores), window_rows) in windows.iter().zip(rows) {
            let (out, inverse) = window_rows.finish()?;
            sums.push(math::combine_expert_rows(
                &out,
                scores,
                Some(&inverse),
                top,
            )?);
        }
        if let Some((last, earlier)) = sums.split_last() {
            for sum in earlier {
                self.weights.finish_expert_slots(i, sum)?;
            }
            self.weights.complete_expert_window(i, last)?;
        }
        Ok(Some(sums))
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
        let order = super::super::route_plan::expert_order(ids, c.num_experts)
            .map_err(Error::from_reason)?;
        let wide = x.shape_at(1)? > 8;
        // Windows of up to 8 rows keep the arithmetic of the fused kernel.
        if wide
            && self.verification.is_none()
            && !self.weights.plan.direct
            && self.weights.experts_in_slots(i, &order).len() < order.len()
        {
            let window = [(x.clone(), ids.to_vec(), scores.clone())];
            return Ok(self
                .union_expert_rows(&window, i)?
                .and_then(|mut sums| sums.pop()));
        }
        let kernel = ExpertKernel::for_window(ids.len(), c.num_experts);
        if order.len() <= capacity && wide {
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
                kernel,
                c.num_experts_per_tok,
            )?;
            let reduced =
                math::combine_expert_rows(&out, scores, inverse.as_ref(), c.num_experts_per_tok)?;
            if self.verification.is_none() {
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
        let groups =
            super::super::route_plan::assignment_groups(ids, &order, c.num_experts, capacity)
                .map_err(Error::from_reason)?;
        let mut slots = LayerSlots {
            decoder: self,
            layer: i,
        };
        let Some((out, inverse)) =
            Self::grouped_expert_rows(x, ids, &groups, kernel, c.num_experts_per_tok, &mut slots)?
        else {
            return Ok(None);
        };
        let reduced =
            math::combine_expert_rows(&out, scores, Some(&inverse), c.num_experts_per_tok)?;
        if wide {
            // A wide assignment tensor is tens of MiB per layer. Once its
            // reduction is complete it must not remain pinned by slot leases.
            self.weights.complete_expert_window(i, &reduced)?;
        }
        Ok(Some(reduced))
    }

    /// The MTP head's experts live in two resident banks (gate|up fused, down)
    /// built at load; route them on the device like the target's.
    pub(super) fn mtp_resident_experts(
        &self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
    ) -> Result<Option<MxArray>> {
        let prefix = format!("{}layers.0.mlp.experts", self.scope);
        let (Some(gate_up), Some(down)) = (
            self.weights
                .resident_bank(&format!("{prefix}.gate_up_proj")),
            self.weights.resident_bank(&format!("{prefix}.down_proj")),
        ) else {
            return Ok(None);
        };
        let c = &self.config;
        let tokens = x.shape_at(1)?;
        let (h, m, top) = (
            c.hidden_size as i64,
            c.moe_intermediate_size as i64,
            c.num_experts_per_tok as i64,
        );
        let rows = tokens * top;
        let token_rows = MxArray::arange(0.0, tokens as f64, None, Some(DType::Int32))?
            .reshape(&[tokens, 1])?
            .broadcast_to(&[tokens, top])?
            .reshape(&[-1])?;
        let input = x
            .reshape(&[tokens, h])?
            .take(&token_rows, 0)?
            .reshape(&[rows, 1, h])?;
        let ids = selected.reshape(&[-1])?;
        let both = gate_up.expert_rows(&input, &ids, c.num_experts, false)?;
        let hidden = math::swiglu(&both.slice_axis(2, 0, m)?, &both.slice_axis(2, m, 2 * m)?)?;
        let out = down
            .expert_rows(&hidden, &ids, c.num_experts, false)?
            .reshape(&[rows, h])?;
        math::combine_expert_rows(&out, scores, None, top as usize).map(Some)
    }

    #[cfg(test)]
    pub(in super::super) fn mtp_resident_experts_for_test(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
    ) -> Result<Option<MxArray>> {
        self.scope = "mtp.".into();
        let out = self.mtp_resident_experts(x, selected, scores);
        self.scope.clear();
        out
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
        if !self.gguf() {
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

    pub(super) fn attention_matrix_for_mlp_with_norm(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
        attn_normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let base = self.history.len();
        let has_ple = self.config.ple_layer_ids.contains(&(i + 1));
        let x = if has_ple {
            self.ple_apply(x, tokens, i, cache)?
        } else {
            x.clone()
        };

        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            if has_ple { None } else { attn_normed },
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
        if !self.async_device_prefill() && !self.cuda_deferred_state() {
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
        let (scale, dt) = self.gdn_scale_dt(i, &p, &g)?;
        let prepared = if (self.gguf() || super::super::cuda_qwen4_fast_enabled())
            && c.linear_conv_kernel_dim == 4
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && qkv.dtype()? == DType::BFloat16
            && conv.dtype()? == DType::Float32
        {
            math::gdn_prepare(&qkv, &a, &b, &conv, &mut cache.conv, &scale, &dt)?
        } else {
            None
        };
        let norm = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let state = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        if super::super::cuda_qwen4_fast_enabled()
            && let Some([ref q, ref k, ref v, ref decay, ref beta]) = prepared
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && z.dtype()? == DType::BFloat16
        {
            let tiled_heads = self.gguf();
            let sigmoid_gate = c.output_gate_type == "sigmoid";
            let cuda_out = if t <= 16 {
                crate::kernels::gdn::fused_step_with_gate(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    &state,
                    &z,
                    &norm,
                    c.rms_norm_eps as f32,
                    tiled_heads,
                    sigmoid_gate,
                    None,
                )?
            } else {
                crate::kernels::gdn::prefill_chain_gated_with_gate(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    &state,
                    &z,
                    &norm,
                    c.rms_norm_eps as f32,
                    tiled_heads,
                    sigmoid_gate,
                )?
            };
            if let Some((out, next_state)) = cuda_out {
                self.submit_state(&[&next_state], "qwen4::batch::recurrent_state")?;
                cache.recurrent = Some(next_state);
                return self.linear(
                    &out,
                    &format!("{p}.out_proj.weight"),
                    &format!("{g}.ssm_out.weight"),
                );
            }
        }
        let a_f32 = if a.dtype()? == DType::Float32 {
            a
        } else {
            a.astype(DType::Float32)?
        };
        let b_f32 = if b.dtype()? == DType::Float32 {
            b
        } else {
            b.astype(DType::Float32)?
        };
        let [q, k, v, decay, beta] = if let Some([q, k, v, decay, beta]) = prepared {
            if crate::engine::persistence::compiled_forward_backend_available() {
                [q, k, v, decay, beta]
            } else {
                let (q, k) = if self.gguf() {
                    (q, k)
                } else {
                    (
                        q.repeat((nh / kh) as i32, 2)?,
                        k.repeat((nh / kh) as i32, 2)?,
                    )
                };
                [
                    q.astype(DType::Float32)?,
                    k.astype(DType::Float32)?,
                    v.astype(DType::Float32)?,
                    decay,
                    beta,
                ]
            }
        } else {
            let qkv = math::conv_sequence(&qkv, &conv, &mut cache.conv, c.linear_conv_kernel_dim)?;
            let q = math::l2(&qkv.slice_axis(2, 0, kh * kd)?.reshape(&[1, t, kh, kd])?)?
                .mul_scalar((kd as f64).powf(-0.5))?;
            let k = math::l2(
                &qkv.slice_axis(2, kh * kd, 2 * kh * kd)?
                    .reshape(&[1, t, kh, kd])?,
            )?;
            let v = qkv
                .slice_axis(2, 2 * kh * kd, 2 * kh * kd + nh * vd)?
                .reshape(&[1, t, nh, vd])?
                .astype(DType::Float32)?;
            let (q, k) = if self.gguf() {
                (q, k)
            } else {
                (
                    q.repeat((nh / kh) as i32, 2)?,
                    k.repeat((nh / kh) as i32, 2)?,
                )
            };
            let (decay, beta) = math::gdn_gates(&a_f32, &b_f32, &scale, &dt)?;
            [
                q.astype(DType::Float32)?,
                k.astype(DType::Float32)?,
                v,
                decay,
                beta,
            ]
        };

        let (out, state) = math::recurrent_sequence(&q, &k, &v, &decay, &beta, &state)?;

        self.submit_state(&[&state], "qwen4::batch::recurrent_state")?;
        cache.recurrent = Some(state);
        let replay = if c.output_gate_type == "sigmoid" {
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
        self.mlp_matrix_for_attention(x, i, None)
    }

    pub(super) fn mlp_matrix_for_attention(
        &mut self,
        x: &MxArray,
        i: usize,
        normed: Option<&MxArray>,
    ) -> Result<MxArray> {
        Ok(self.mlp_matrix_for_attention_next(x, i, normed, false)?.0)
    }

    pub(super) fn mlp_matrix_for_attention_next(
        &mut self,
        x: &MxArray,
        i: usize,
        normed: Option<&MxArray>,
        fuse_next_attn: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let (mixed, gate) = self.mlp_mix(x, i, normed)?;
        let branch = self.moe_batch(&mixed, i)?;
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 batched MLP is missing its injection gate"))?;
        let next_layer = i + 1;
        let can_fuse_next = fuse_next_attn
            && super::super::cuda_qwen4_fast_enabled()
            && next_layer < self.config.num_hidden_layers
            && !self.config.ple_layer_ids.contains(&(next_layer + 1));
        let (out, next_normed) = if can_fuse_next {
            self.inject_for_hyper(
                x,
                &branch,
                &gate,
                &format!("layers.{next_layer}.attn_hyper_connection.hc_norm.weight"),
                &format!("blk.{next_layer}.hc_attn_norm.weight"),
                true,
            )?
        } else {
            (Self::inject(x, &branch, &gate)?, None)
        };
        if self.cuda_deferred_state() {
            // Submit every 6 layers on CUDA direct prefill to keep transient
            // activation footprint bounded without triggering per-layer WDDM
            // CUDA graph instantiation stalls.
            if (i + 1).is_multiple_of(6) {
                if let Some(ref n) = next_normed {
                    self.submit(&[&out, n], "qwen4::batch::mlp")?;
                } else {
                    self.submit(&[&out], "qwen4::batch::mlp")?;
                }
            }
        } else if !self.async_device_prefill() || (i >= 1 && (i - 1).is_multiple_of(3)) {
            self.submit(&[&out], "qwen4::batch::mlp")?;
        }
        Ok((out, next_normed))
    }

    /// Layer `i`'s MLP input from the residual stream `x` (the hyper mix)
    /// and its injection gate.
    pub(super) fn mlp_mix(
        &mut self,
        x: &MxArray,
        i: usize,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.hyper_with_norm(
            x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed,
        )
    }

    /// The residual stream after layer `i`'s MLP: `branch` injected into `x`.
    pub(super) fn mlp_inject(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: Option<MxArray>,
        i: usize,
    ) -> Result<MxArray> {
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 batched MLP is missing its injection gate"))?;
        let out = Self::inject(x, branch, &gate)?;
        // The next router readback (or final window output) fences this
        // graph. Slot-bank readers remain retained until their own fence.
        // Reference schedule: first two layers, then groups of three.
        if self.cuda_deferred_state() {
            if (i + 1).is_multiple_of(6) {
                self.submit(&[&out], "qwen4::batch::mlp")?;
            }
        } else if !self.async_device_prefill() || (i >= 1 && (i - 1).is_multiple_of(3)) {
            self.submit(&[&out], "qwen4::batch::mlp")?;
        }

        Ok(out)
    }

    pub(super) fn moe_batch(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let (selected, scores) = self.route_window(x, i)?;
        if self.device_routes.is_none()
            && let Some(out) = self.cuda_fused_moe(x, &selected, &scores, i)?
        {
            return Ok(out);
        }
        let sum = self.expert_rows_for(x, &selected, &scores, i)?;
        self.with_shared_expert(x, &sum, i)
    }

    /// The MoE output of layer `i` over the MLP input `x`: the routed `sum`
    /// plus the gated shared expert.
    pub(super) fn with_shared_expert(
        &mut self,
        x: &MxArray,
        sum: &MxArray,
        i: usize,
    ) -> Result<MxArray> {
        let (shared, gate) = self.shared_expert_parts(x, i)?;
        sum.astype(x.dtype()?)?
            .add(&math::sigmoid_mul(&gate, &shared)?)
    }

    /// Layer `i`'s router over the rows of `x`: each row's top experts
    /// `[1, T, top]` and their normalized scores `[1, T * top, 1]`.
    pub(super) fn route_window(&mut self, x: &MxArray, i: usize) -> Result<(MxArray, MxArray)> {
        let c = self.config.clone();
        let tokens = x.shape()?[1] as usize;
        let logits = self.linear(
            x,
            &format!("layers.{i}.mlp.gate.weight"),
            &format!("blk.{i}.ffn_gate_inp.weight"),
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
        Ok((selected, scores))
    }

    /// The routed experts' weighted sum over the rows of `x` for the routes
    /// `route_window` gave, before the shared expert.
    pub(super) fn expert_rows_for(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        i: usize,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let tokens = x.shape()?[1] as usize;
        let top = c.num_experts_per_tok;
        let resident = if self.device_routes.is_some() {
            Some(self.tentative_prefill_experts(x, selected, scores, i)?)
        } else {
            self.resident_experts(x, selected, scores, i)?
        };
        if let Some(sum) = resident {
            return Ok(sum);
        }
        let ids = selected.to_uint32()?;
        if let Some(out) = self.slotted_experts(x, &ids, scores, i)? {
            return Ok(out);
        }
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
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
            .sum(Some(&[2]), Some(false))
    }
}
