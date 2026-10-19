//! Stable, quantized expert slots for partial residency. Slot buffers are
//! mutated only after their last lazy reader has completed. The ordinary LRU
//! supplies bounded misses and releases imported chunks after the slot copy.
use super::*;
use crate::models::qwen4_exp::runtime_flags;
use std::time::{Duration, Instant};

/// The driver unwires a bound file-backed bank ~1.15 s after its last use and
/// re-wires resident pages at 12-17 ms/GiB, one command buffer at a time. A
/// forward pass that starts later than that after the previous one (the first
/// decode step after prefill, any slow step) wires 78 GiB layer by layer, and
/// layer 0 is idle long enough to unwire again before the pass ends: measured
/// 0.4-0.6 tokens/s until something breaks the cycle. One eval reading one
/// element of every mapped bank gives them all the same last use first.
const MAPPED_WIRE_GAP: Duration = Duration::from_millis(900);

/// One layer's routed experts addressed in place: the pack's sections as
/// no-copy banks, slot id == expert id, so nothing is ever copied, missed
/// or overwritten and no reader lease is needed.
pub(super) struct MappedExperts {
    banks: [Arc<Weight>; 3],
    pub(super) maps: [Arc<crate::array::Mmap>; 3],
    experts: usize,
}

impl MappedExperts {
    fn open(
        files: [&super::super::mapped_pack::LayerFile; 3],
        experts: usize,
    ) -> Result<(Self, u64)> {
        let mut bytes = 0u64;
        let mut parts = Vec::with_capacity(3);
        for file in files {
            let h = &file.header;
            if h.experts as usize != experts {
                return Err(err(format!(
                    "Mapped pack {} holds {} experts, the checkpoint {experts}",
                    h.name, h.experts
                )));
            }
            bytes += h.sections.iter().map(|s| s.length).sum::<u64>();
            let (map, mut arrays) = file.map_sections()?;
            let biases = if arrays.len() == 3 {
                arrays.pop()
            } else {
                None
            };
            let scales = arrays.pop();
            let values = arrays
                .pop()
                .ok_or_else(|| err(format!("Mapped pack {} has no values section", h.name)))?;
            parts.push((
                Arc::new(Weight {
                    values,
                    dense_bf16: None,
                    scales,
                    biases,
                    group: h.group,
                    bits: h.bits,
                    mode: h.mode.clone(),
                }),
                map,
            ));
        }
        let [(gate, gate_map), (up, up_map), (down, down_map)] = <[_; 3]>::try_from(parts)
            .map_err(|_| err("Mapped layer needs gate, up and down banks"))?;
        Ok((
            Self {
                banks: [gate, up, down],
                maps: [gate_map, up_map, down_map],
                experts,
            },
            bytes,
        ))
    }
}

/// The fused expert kernels' bank contract (K-quant gate/up, affine down).
fn fused_bank_modes(banks: &[Arc<Weight>; 3]) -> bool {
    matches!(banks[0].mode.as_str(), "q4k" | "q5k")
        && banks[1].mode == banks[0].mode
        && banks[2].mode == "affine"
        && matches!(banks[2].bits, 5 | 8)
        && banks
            .iter()
            .all(|w| w.group == 32 && w.scales.is_some() && w.biases.is_some())
}

pub(super) struct ExpertSlots {
    banks: [Arc<Weight>; 3],
    names: [String; 3],
    rows: [usize; 3],
    occupants: Vec<Option<u32>>,
    mapping: Vec<Option<usize>>,
    device_mapping: Option<MxArray>,
    ages: Vec<u64>,
    tick: u64,
    readers: Vec<MxArray>,
}

impl Weight {
    fn empty_slots(&self, capacity: usize) -> Result<Self> {
        let allocate = |a: &MxArray| {
            let shape = a.shape()?;
            MxArray::zeros(&[shape[0] * capacity as i64, shape[1]], Some(a.dtype()?))
        };
        Ok(Self {
            values: allocate(&self.values)?,
            dense_bf16: None,
            scales: self.scales.as_ref().map(allocate).transpose()?,
            biases: self.biases.as_ref().map(allocate).transpose()?,
            ..self.clone()
        })
    }
    fn arrays(&self) -> Vec<&MxArray> {
        [
            Some(&self.values),
            self.scales.as_ref(),
            self.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl Store {
    pub(super) fn auxiliary_cache_allowance(&self) -> Result<u64> {
        let auxiliary = self
            .tensors
            .iter()
            .filter(|(name, _)| super::residency::is_auxiliary(name))
            .try_fold(0u64, |bytes, (_, tensor)| {
                tensor.runtime_bytes().map(|n| bytes.saturating_add(n))
            })?;
        Ok(auxiliary
            .min(self.cache_limit / 8)
            .max(MAX_READ_BYTES * 6)
            .max(self.draft_bank_budget))
    }

    /// Map every routed expert of every layer from the pack when `auto`
    /// residency selects it (memory::mapped_residency_selected) and the live
    /// host holds the pack beside what stays anonymous (memory::mapped_plan),
    /// replacing the slot budget by one for dense banks and auxiliary rows
    /// only. Any failure leaves the slot path in charge. `pool_bytes` is the
    /// paged KV pool created before this decision.
    pub(in super::super) fn select_mapped_residency(&mut self, pool_bytes: u64) -> Result<()> {
        let Some(experts) = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
        else {
            return Ok(());
        };
        self.select_mapped_residency_with(
            experts as usize,
            pool_bytes,
            &mut super::super::memory::mapped_plan,
        )
    }

    pub(super) fn select_mapped_residency_with(
        &mut self,
        experts: usize,
        pool_bytes: u64,
        plan: &mut impl FnMut(&mut super::super::memory::Plan, u64, u64, u64) -> Result<()>,
    ) -> Result<()> {
        if !self.gguf
            || experts == 0
            || !crate::engine::persistence::compiled_forward_backend_available()
            || !self.slots.is_empty()
            || !self.banks.is_empty()
        {
            return Ok(());
        }
        let Some(pack) = &self.mapped else {
            return Ok(());
        };
        let layers: std::collections::BTreeSet<usize> = self
            .tensors
            .keys()
            .filter(|name| !super::residency::is_auxiliary(name))
            .filter_map(|name| super::routed_expert_tensor(name).map(|(layer, _)| layer))
            .collect();
        if layers.is_empty() {
            return Ok(());
        }
        let names = |layer: usize| {
            ["gate", "up", "down"].map(|part| format!("blk.{layer}.ffn_{part}_exps.weight"))
        };
        let complete = layers.iter().all(|&layer| {
            names(layer).iter().all(|name| {
                pack.file(name)
                    .is_some_and(|file| file.header.experts as usize == experts)
            })
        });
        if !super::super::memory::mapped_residency_selected(
            &self.plan.policy,
            super::super::memory::explicit_weight_budget(),
            complete,
            super::super::memory::mapped_pack_disabled(),
        ) {
            return Ok(());
        }
        let pack_bytes: u64 = layers
            .iter()
            .flat_map(|&layer| names(layer))
            .filter_map(|name| pack.file(&name))
            .flat_map(|file| file.header.sections.iter().map(|s| s.length))
            .sum();
        let dense = self.fixed_projection_bytes()?;
        // The Q8 draft banks are anonymous MLX memory built on the first MTP
        // owner: a mapped host must hold them beside the dense banks.
        let draft_bank = self.mtp_bank_estimate()?;
        let mut candidate = self.plan.clone();
        if let Err(e) = plan(
            &mut candidate,
            dense.saturating_add(draft_bank),
            pool_bytes,
            pack_bytes,
        ) {
            self.plan.mapped_bank_budget = candidate.mapped_bank_budget;
            self.plan.live_non_reclaimable = candidate.live_non_reclaimable;
            tracing::warn!("qwen4 residency: slot path, {e}");
            self.live.set_fallback(Fallback::LoadGate {
                pack: pack_bytes,
                budget: candidate.mapped_bank_budget,
            });
            return Ok(());
        }
        let mut mapped = HashMap::with_capacity(layers.len());
        let mut bytes = 0u64;
        for &layer in &layers {
            let [gate, up, down] = names(layer);
            let files = [&gate, &up, &down].map(|name| pack.file(name));
            let [Some(gate), Some(up), Some(down)] = files else {
                return Ok(());
            };
            match MappedExperts::open([gate, up, down], experts) {
                Ok((layer_banks, layer_bytes)) => {
                    bytes += layer_bytes;
                    mapped.insert(layer, layer_banks);
                }
                Err(e) => {
                    tracing::warn!("qwen4 mapped residency unavailable, using slots: {e}");
                    return Ok(());
                }
            }
        }
        self.plan = candidate;
        let files = (0..=layers.last().copied().unwrap_or(0))
            .map(|layer| mapped.get(&layer).map(|experts| experts.maps.clone()))
            .collect();
        match super::super::prefetch::Pool::new(files, super::super::prefetch::Advise) {
            Ok(pool) => {
                if super::super::prefetch::at_load() {
                    pool.request_all();
                }
                self.prefetch = Some(pool);
            }
            Err(e) => tracing::warn!("qwen4 mapped pack read-ahead unavailable: {e}"),
        }
        self.cache_limit = self.plan.budget;
        self.draft_bank_budget = draft_bank;
        self.slot_capacity = Some(experts);
        self.plan.mapped_bytes = bytes;
        self.plan.partial_expert_slots = experts;
        self.mapped_experts = mapped;
        self.live.store(&self.plan);
        tracing::info!(
            "qwen4 mapped residency: {} layers x {experts} experts, {:.1} GiB mapped within a {:.1} GiB bank budget ({:.1} GiB live host memory), {:.1} GiB budget for dense banks, rows and {:.2} GiB of draft banks",
            layers.len(),
            bytes as f64 / (1u64 << 30) as f64,
            self.plan.mapped_bank_budget as f64 / (1u64 << 30) as f64,
            self.plan.live_non_reclaimable as f64 / (1u64 << 30) as f64,
            self.cache_limit as f64 / (1u64 << 30) as f64,
            draft_bank as f64 / (1u64 << 30) as f64
        );
        Ok(())
    }

    /// Publish the residency the load settled on.
    pub(in super::super) fn publish_live(&self) {
        self.live.store(&self.plan);
    }

    /// The first turn of a measurement window opens: a trip whose switch
    /// failed at its pass moves the process to the slot path first, then the
    /// window's SSD count and slow streak begin. Returns the fallback applied.
    pub(in super::super) fn begin_turn(&mut self) -> Result<Option<Fallback>> {
        let applied = self.apply_fallback()?;
        self.tripwire
            .begin_turn(super::super::memory::process_disk_reads());
        Ok(applied)
    }

    /// A turn opens beside turns already open: residency is one per store,
    /// so an armed trip moves every open turn to slots here; the window they
    /// share keeps its baseline. Returns the fallback applied.
    pub(in super::super) fn apply_fallback(&mut self) -> Result<Option<Fallback>> {
        match self.tripwire.take() {
            Some(why) if self.plan.mapped => {
                self.leave_mapped(&why)?;
                Ok(Some(why))
            }
            _ => Ok(None),
        }
    }

    /// One main-network pass over `rows` rows took `elapsed`. A pass that
    /// trips the wire is the last mapped one: the store leaves the mapped
    /// path here, between two passes of the same request, and the request
    /// goes on over slot banks with its state intact (the pass's arrays hold
    /// their own leases on the mapping). Returns the fallback applied; a
    /// switch that fails leaves the trip armed for the next turn to open.
    pub(in super::super) fn observe_pass(
        &mut self,
        rows: usize,
        elapsed: Duration,
    ) -> Result<Option<Fallback>> {
        self.observe_pass_at(rows, elapsed, super::super::memory::process_disk_reads())
    }

    pub(in super::super) fn observe_pass_at(
        &mut self,
        rows: usize,
        elapsed: Duration,
        reads: u64,
    ) -> Result<Option<Fallback>> {
        if !self.plan.mapped {
            return Ok(None);
        }
        let Some(why) = self
            .tripwire
            .observe(rows, elapsed, reads, self.plan.mapped_bytes)
            .cloned()
        else {
            return Ok(None);
        };
        tracing::warn!(
            "qwen4 residency: mapped turn thrashes ({why}); the request goes on over slots"
        );
        match self.apply_fallback() {
            Ok(applied) => Ok(applied),
            Err(error) => {
                self.tripwire.rearm(why);
                Err(error)
            }
        }
    }

    #[cfg(test)]
    pub(in super::super) fn tripwire_mut(&mut self) -> &mut super::super::tripwire::Tripwire {
        &mut self.tripwire
    }

    #[cfg(test)]
    pub(super) fn trip(&mut self, why: Fallback) {
        self.tripwire.begin_turn(0);
        let mapped = self.plan.mapped_bytes;
        match why {
            Fallback::SsdReads { read, .. } => {
                self.tripwire.observe(1, Duration::ZERO, read, mapped);
            }
            other => panic!("test trips through SSD reads only, not {other}"),
        }
    }

    /// Drop the mapped banks and run the slot path from here on: the slot
    /// budget from the live host, banks built lazily by `expert_slots`, the
    /// dense banks and rows already held kept.
    fn leave_mapped(&mut self, why: &Fallback) -> Result<()> {
        self.leave_mapped_with(
            why,
            &mut super::super::memory::host_memory,
            &mut std::thread::sleep,
        )
    }

    /// The banks go first: their pages stay wired until the driver lets go
    /// of them, and a budget read while they are counts the pack against
    /// the slots for the life of the process.
    pub(super) fn leave_mapped_with(
        &mut self,
        why: &Fallback,
        host: &mut dyn FnMut() -> Result<super::super::memory::HostMemory>,
        sleep: &mut dyn FnMut(Duration),
    ) -> Result<()> {
        let experts = self.mapped_experts.values().next().map(|m| m.experts);
        self.prefetch = None;
        self.mapped_experts.clear();
        self.draft_bank_budget = 0;
        self.prefetch_pass = false;
        self.mapped_pass_start = None;
        self.slot_capacity = None;
        crate::array::memory::synchronize();
        let mut plan = self.plan.clone();
        super::super::memory::slot_fallback_plan_with(&mut plan, self.cache_bytes, host, sleep);
        self.plan = plan;
        self.cache_limit = self.plan.budget;
        if let (Some(experts), Some(top)) = (
            experts,
            self.metadata
                .get("qwen4exp.expert_used_count")
                .and_then(GgufMetaValue::as_u64),
        ) {
            self.plan.partial_expert_slots = self.expert_slot_capacity(experts, top as usize)?;
        }
        self.live.set_fallback(why.clone());
        self.live.store(&self.plan);
        tracing::warn!(
            "qwen4 residency: mapped -> slots ({why}); {:.1} GiB slot budget, {} slots",
            self.cache_limit as f64 / (1u64 << 30) as f64,
            self.plan.partial_expert_slots
        );
        Ok(())
    }

    /// Called at the start of every forward pass (layer 0's MoE). `main` is
    /// false for a pass that reads no mapped bank (the MTP head runs its own
    /// experts); such a pass must leave the stamp alone, or a run of drafts
    /// hides the gap the next main pass has to re-wire for.
    pub(in super::super) fn wire_mapped_banks(&mut self, main: bool) -> Result<()> {
        self.wire_mapped_banks_at(
            Instant::now(),
            main,
            super::super::prefetch::depth_from_env(),
        )
        .map(|_| ())
    }

    /// Returns whether the banks were touched. A stale pass over a pack that
    /// is not fully resident reads `depth` layers ahead instead: the touch
    /// binds every bank at once, so the driver would fault the whole pack in
    /// on the model thread at its own rate before layer 0 runs.
    pub(super) fn wire_mapped_banks_at(
        &mut self,
        now: Instant,
        main: bool,
        depth: usize,
    ) -> Result<bool> {
        if !self.plan.mapped || !main {
            return Ok(false);
        }
        let stale = self
            .mapped_pass_start
            .is_none_or(|start| now.saturating_duration_since(start) > MAPPED_WIRE_GAP);
        self.mapped_pass_start = Some(now);
        if !stale {
            return Ok(false);
        }
        self.prefetch_pass = false;
        if depth > 0
            && let Some(pool) = self.prefetch.as_mut()
            && !pool.sampled_resident()
        {
            tracing::info!("qwen4 mapped pack not resident: reading {depth} layers ahead");
            self.prefetch_depth = depth;
            pool.layer_started(0, depth);
            self.prefetch_pass = depth < self.last_mapped_layer();
            return Ok(false);
        }
        let mut groups: Vec<(DType, Vec<MxArray>)> = Vec::new();
        for mapped in self.mapped_experts.values() {
            for bank in &mapped.banks {
                for array in [
                    Some(&bank.values),
                    bank.scales.as_ref(),
                    bank.biases.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    let corner = array.slice(&[0, 0], &[1, 1])?;
                    let dtype = corner.dtype()?;
                    match groups.iter_mut().find(|(d, _)| *d == dtype) {
                        Some((_, list)) => list.push(corner),
                        None => groups.push((dtype, vec![corner])),
                    }
                }
            }
        }
        let touched = groups
            .iter()
            .map(|(_, list)| MxArray::concatenate_many(list.iter().collect(), Some(0)))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        MxArray::eval_arrays_with_context(
            &touched.iter().collect::<Vec<_>>(),
            "qwen4::mapped::wire",
        )?;
        let elapsed = started.elapsed();
        if elapsed > Duration::from_millis(50) {
            tracing::debug!("qwen4 mapped banks wired in {elapsed:.1?}");
        }
        self.mapped_pass_start = Some(now + elapsed);
        Ok(true)
    }

    /// Layer `layer` of a pass starts: in a read-ahead pass, request the
    /// layer `depth` ahead of it. The pass ends with the request for the last
    /// layer, not at the next layer-0 call: a pass over several rows reaches
    /// layer 0 once per row.
    pub(in super::super) fn prefetch_mapped_layer(&mut self, layer: usize, main: bool) {
        if main
            && self.prefetch_pass
            && layer > 0
            && let Some(pool) = self.prefetch.as_mut()
        {
            pool.layer_started(layer, self.prefetch_depth);
            if layer.saturating_add(self.prefetch_depth) >= self.last_mapped_layer() {
                self.prefetch_pass = false;
            }
        }
    }

    fn last_mapped_layer(&self) -> usize {
        self.mapped_experts.keys().max().copied().unwrap_or(0)
    }

    pub(in super::super) fn expert_slot_capacity(
        &mut self,
        experts: usize,
        top: usize,
    ) -> Result<usize> {
        if self.plan.mapped {
            return Ok(experts);
        }
        if !self.gguf
            || runtime_flags::is_zero(c"MLX_QWEN4_EXPERT_SLOTS")
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(0);
        }
        if let Some(n) = self.slot_capacity {
            return Ok(n);
        }
        let mut layer_bytes = HashMap::<&str, u64>::new();
        let routed = self
            .tensors
            .iter()
            .filter(|(name, _)| {
                name.contains("_exps.weight")
                    && !super::residency::is_auxiliary(name)
                    && !self.banks.contains_key(*name)
            })
            .try_fold(0u64, |n, (_, t)| {
                t.runtime_bytes().map(|b| n.saturating_add(b))
            })?;
        for (name, tensor) in &self.tensors {
            if name.contains("_exps.weight")
                && !super::residency::is_auxiliary(name)
                && !self.banks.contains_key(name)
            {
                let layer = match name.split_once(".ffn_") {
                    Some((layer, _)) => layer,
                    None => name.as_str(),
                };
                let bytes = layer_bytes.entry(layer).or_default();
                *bytes = bytes.saturating_add(tensor.runtime_bytes()?);
            }
        }
        let largest_layer = layer_bytes.values().copied().max().unwrap_or(0);
        let fixed = self.plan.hot_bytes.saturating_sub(routed);
        // Auxiliary checkpoints are attached before this bootstrap decision.
        // Reserve their actual inventory up to the existing auxiliary ceiling;
        // text-only loads can spend the remainder on warm experts. Always keep
        // bounded staging and sparse-row room inside the admitted weight cap.
        // Fully resident expert banks are included in `fixed`, and only the
        // remaining layers share slots. Never reserve a second copy of them.
        let reserve = self.auxiliary_cache_allowance()?;
        let bytes = self
            .cache_limit
            .saturating_sub(fixed)
            .saturating_sub(reserve);
        let capacity = if routed == 0 {
            0
        } else {
            let auxiliary_cap = bytes as u128 * experts as u128 / routed as u128;
            // Destination-bank preparation and bounded imports need room even
            // near the last layer. Size this allowance from the largest actual
            // layer and chosen slot count, not a fixed number of GiB. Solve
            // slots * (all layers + largest layer) <= available bank bytes.
            let staged_bytes = self
                .cache_limit
                .saturating_sub(fixed)
                .saturating_sub(MAX_READ_BYTES * 6);
            let staging_cap = staged_bytes as u128 * experts as u128
                / routed.saturating_add(largest_layer) as u128;
            auxiliary_cap.min(staging_cap).min(experts as u128) as usize
        };
        let capacity = if capacity < top { 0 } else { capacity };
        self.slot_capacity = Some(capacity);
        Ok(capacity)
    }

    pub(in super::super) fn expert_slots(
        &mut self,
        layer: usize,
        experts: usize,
        top: usize,
        ids: &[u32],
    ) -> Result<Option<([Arc<Weight>; 3], Vec<u32>, usize)>> {
        let capacity = self.expert_slot_capacity(experts, top)?;
        let wanted: HashSet<u32> = ids.iter().copied().collect();
        if capacity == 0 || wanted.len() > capacity {
            return Ok(None);
        }
        if ids.iter().any(|&e| e as usize >= experts) {
            return Err(err("Qwen4 expert id exceeds slot mapping"));
        }
        if self.plan.mapped {
            let mapped = self
                .mapped_experts
                .get(&layer)
                .ok_or_else(|| err("Qwen4 mapped layer has no expert banks"))?;
            self.slot_hits += wanted.len() as u64;
            return Ok(Some((mapped.banks.clone(), ids.to_vec(), mapped.experts)));
        }
        let mut slots = if let Some(slots) = self.slots.remove(&layer) {
            slots
        } else {
            let names =
                ["gate", "up", "down"].map(|part| format!("blk.{layer}.ffn_{part}_exps.weight"));
            let rows = names
                .each_ref()
                .map(|n| self.descriptor(n).map(|d| d.shape[1]));
            let [r0, r1, r2] = rows;
            let rows = [r0?, r1?, r2?];
            let prototypes = [
                self.read(&names[0], 0, rows[0])?,
                self.read(&names[1], 0, rows[1])?,
                self.read(&names[2], 0, rows[2])?,
            ];
            if prototypes.iter().any(|w| w.scales.is_none()) {
                return Ok(None);
            }
            let bytes = prototypes
                .iter()
                .try_fold(0u64, |n, w| w.bytes().map(|b| n + b * capacity as u64))?;
            // A single layer cannot bypass the same staging and system reserve.
            self.evict_to(self.cache_limit.saturating_sub(bytes + MAX_READ_BYTES * 3))?;
            super::super::memory::maintain_freelist(self.plan.physical_bytes);
            super::super::memory::admit(bytes + MAX_READ_BYTES * 3)?;
            let banks = [
                Arc::new(prototypes[0].empty_slots(capacity)?),
                Arc::new(prototypes[1].empty_slots(capacity)?),
                Arc::new(prototypes[2].empty_slots(capacity)?),
            ];
            self.slot_bytes += bytes;
            self.cache_bytes += bytes;
            self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
            ExpertSlots {
                banks,
                names,
                rows,
                occupants: vec![None; capacity],
                mapping: vec![None; experts],
                device_mapping: None,
                ages: vec![0; capacity],
                tick: 0,
                readers: Vec::new(),
            }
        };
        let result = (|| -> Result<_> {
            let mut missing: Vec<u32> = wanted
                .iter()
                .filter(|&&e| slots.mapping[e as usize].is_none())
                .copied()
                .collect();
            missing.sort_unstable();
            self.slot_hits += (wanted.len() - missing.len()) as u64;
            self.slot_misses += missing.len() as u64;
            if !missing.is_empty() {
                slots.device_mapping = None;
                MxArray::eval_arrays_with_context(
                    &slots.readers.iter().collect::<Vec<_>>(),
                    "qwen4::slots::complete_readers",
                )?;
                slots.readers.clear();
            }
            // All wanted slots, including newly loaded misses, remain protected
            // for this call. Duplicate requested ids resolve to one slot.
            for missing in missing.chunks(8) {
                let mut destinations = Vec::new();
                for bank in &slots.banks {
                    destinations.extend(bank.arrays().iter().map(|a| a.as_raw_ptr()));
                }
                let mut chosen = Vec::new();
                let mut requests = Vec::new();
                for &expert in missing {
                    let slot = slots
                        .occupants
                        .iter()
                        .enumerate()
                        .filter(|(s, e)| {
                            !chosen.contains(s) && e.is_none_or(|e| !wanted.contains(&e))
                        })
                        .min_by_key(|(s, e)| if e.is_none() { 0 } else { slots.ages[*s] + 1 })
                        .map(|(s, _)| s)
                        .ok_or_else(|| err("Qwen4 expert slot reservation exhausted"))?;
                    chosen.push(slot);
                    for part in 0..3 {
                        requests.push((
                            slots.names[part].clone(),
                            expert as usize * slots.rows[part],
                            slots.rows[part],
                        ));
                    }
                }
                let imported = self.read_expert_batch(&requests)?;
                let mut sources: Vec<_> = imported
                    .iter()
                    .flat_map(|w| w.arrays().into_iter().map(|a| a.as_raw_ptr()))
                    .collect();
                let slots_u32: Vec<_> = chosen.iter().map(|&s| s as u32).collect();
                let ok = {
                    unsafe {
                        mlx_sys::mlx_qwen4_copy_weight_rows(
                            destinations.as_mut_ptr(),
                            sources.as_mut_ptr(),
                            slots_u32.as_ptr(),
                            destinations.len(),
                            missing.len(),
                        )
                    }
                };
                if !ok {
                    return Err(err("Qwen4 expert slot transaction failed"));
                }
                for (&expert, &slot) in missing.iter().zip(&chosen) {
                    if let Some(old) = slots.occupants[slot] {
                        slots.mapping[old as usize] = None;
                    }
                    slots.occupants[slot] = Some(expert);
                    slots.mapping[expert as usize] = Some(slot);
                    for part in 0..3 {
                        let key = (
                            slots.names[part].clone(),
                            expert as usize * slots.rows[part],
                            slots.rows[part],
                        );
                        if let Some((w, freed)) = self.uncache(&key)? {
                            self.slot_upload_bytes += w.bytes()?;
                            self.cache_bytes -= freed;
                            self.recency.remove(&key);
                        }
                    }
                }
            }
            let local: Vec<u32> = ids
                .iter()
                .map(|&e| {
                    let slot = slots
                        .mapping
                        .get(e as usize)
                        .copied()
                        .flatten()
                        .filter(|&slot| slot < slots.ages.len())
                        .ok_or_else(|| err("Qwen4 reserved expert has no valid slot mapping"))?;
                    u32::try_from(slot)
                        .map_err(|_| err("Qwen4 expert slot exceeds device index range"))
                })
                .collect::<Result<_>>()?;
            for &slot in &local {
                let age = slots
                    .ages
                    .get_mut(slot as usize)
                    .ok_or_else(|| err("Qwen4 reserved expert age slot disappeared"))?;
                slots.tick += 1;
                *age = slots.tick;
            }
            Ok((slots.banks.clone(), local, capacity))
        })();
        // Retain the bank/accounting even on a failed source read. Already
        // committed slots stay valid, while unpublished misses remain absent.
        self.slots.insert(layer, slots);
        result.map(Some)
    }

    pub(in super::super) fn device_slots_ready(&self, layers: usize) -> bool {
        if self.plan.mapped {
            return (0..layers).all(|layer| self.mapped_experts.contains_key(&layer));
        }
        (0..layers).all(|layer| self.slots.contains_key(&layer))
    }

    pub(in super::super) fn note_prefill_routes(&mut self, layer: usize, ids: Vec<u32>) {
        self.wide_routes.insert(layer, ids);
    }

    /// Skip a tentative window if even the prior window's expert set no longer
    /// fits the unchanged banks. This is only an eligibility hint, never a
    /// prediction: every newly computed route is still validated at commit.
    pub(in super::super) fn prefill_device_slots_ready(&self, layers: usize) -> bool {
        if self.plan.mapped {
            return (0..layers).all(|layer| {
                self.mapped_experts
                    .get(&layer)
                    .is_some_and(|mapped| fused_bank_modes(&mapped.banks))
            });
        }
        (0..layers).all(|layer| {
            self.slots.get(&layer).is_some_and(|slots| {
                slots.occupants.first().is_some_and(Option::is_some)
                    && self.wide_routes.get(&layer).is_some_and(|ids| {
                        !ids.is_empty()
                            && ids.iter().all(|&id| {
                                slots.mapping.get(id as usize).is_some_and(Option::is_some)
                            })
                    })
                    && matches!(slots.banks[0].mode.as_str(), "q4k" | "q5k")
                    && slots.banks[1].mode == slots.banks[0].mode
                    && slots.banks[2].mode == "affine"
                    && matches!(slots.banks[2].bits, 5 | 8)
                    && slots
                        .banks
                        .iter()
                        .all(|w| w.group == 32 && w.scales.is_some() && w.biases.is_some())
            })
        })
    }

    /// Map absent experts to the initialized first slot while computing a
    /// tentative window. The original IDs, not these safe placeholders, decide
    /// commit. This keeps counting-sort permutations complete even on misses.
    pub(in super::super) fn prefill_device_expert_slots(
        &mut self,
        layer: usize,
        ids: &MxArray,
    ) -> Result<([Arc<Weight>; 3], MxArray, usize)> {
        if self.plan.mapped {
            let mapped = self
                .mapped_experts
                .get(&layer)
                .ok_or_else(|| err("Qwen4 tentative prefill has no mapped bank"))?;
            let local = ids.reshape(&[-1])?.astype(DType::Uint32)?;
            return Ok((mapped.banks.clone(), local, mapped.experts));
        }
        let count = self
            .slots
            .get(&layer)
            .filter(|s| s.occupants.first().is_some_and(Option::is_some))
            .ok_or_else(|| err("Qwen4 tentative prefill has no initialized bank"))?
            .occupants
            .len();
        let (banks, local) = self
            .device_expert_slots(layer, ids)?
            .ok_or_else(|| err("Qwen4 tentative prefill lost its bank"))?;
        let bound = MxArray::from_uint32(&[count as u32], &[])?;
        let zero = MxArray::from_uint32(&[0], &[])?;
        let safe = local.less(&bound)?.where_(&local, &zero)?;
        Ok((banks, safe, count))
    }

    /// The caller must validate every selected ID before publishing a result.
    /// Missing IDs map to the capacity sentinel, which the fused expert kernels
    /// handle without reading a bank. No bank or mapping may change during
    /// this tentative token; ordinary miss handling resumes after completion.
    pub(in super::super) fn device_expert_slots(
        &mut self,
        layer: usize,
        ids: &MxArray,
    ) -> Result<Option<([Arc<Weight>; 3], MxArray)>> {
        if self.plan.mapped {
            let Some(mapped) = self.mapped_experts.get(&layer) else {
                return Ok(None);
            };
            let local = ids.reshape(&[-1])?.astype(DType::Uint32)?;
            return Ok(Some((mapped.banks.clone(), local)));
        }
        let Some(slots) = self.slots.get_mut(&layer) else {
            return Ok(None);
        };
        if slots.device_mapping.is_none() {
            let mapping: Vec<_> = slots
                .mapping
                .iter()
                .map(|slot| slot.unwrap_or(slots.occupants.len()) as u32)
                .collect();
            slots.device_mapping = Some(MxArray::from_uint32(&mapping, &[mapping.len() as i64])?);
        }
        let local = slots
            .device_mapping
            .as_ref()
            .ok_or_else(|| err("Qwen4 device expert mapping was not initialized"))?
            .take(&ids.reshape(&[-1])?, 0)?;
        Ok(Some((slots.banks.clone(), local)))
    }

    /// Commit LRU ages only after the entire tentative token was a cache hit.
    /// A miss changes neither the bank contents nor their host bookkeeping.
    pub(in super::super) fn commit_device_routes(
        &mut self,
        layers: &[usize],
        ids: &[u32],
        top: usize,
    ) -> Result<bool> {
        if ids.len() != layers.len() * top || top == 0 {
            return Err(err("Qwen4 device route tape has an invalid shape"));
        }
        if self.plan.mapped {
            for (&layer, selected) in layers.iter().zip(ids.chunks_exact(top)) {
                let Some(mapped) = self.mapped_experts.get(&layer) else {
                    return Ok(false);
                };
                if selected.iter().any(|&e| e as usize >= mapped.experts) {
                    return Err(err(
                        "Qwen4 device route names an expert outside the mapped pack",
                    ));
                }
                self.slot_hits += selected.iter().collect::<HashSet<_>>().len() as u64;
            }
            return Ok(true);
        }
        for (&layer, selected) in layers.iter().zip(ids.chunks_exact(top)) {
            let Some(slots) = self.slots.get(&layer) else {
                return Ok(false);
            };
            if selected.iter().any(|&e| {
                slots
                    .mapping
                    .get(e as usize)
                    .copied()
                    .flatten()
                    .is_none_or(|slot| slot >= slots.ages.len())
            }) {
                return Ok(false);
            }
        }
        for (&layer, selected) in layers.iter().zip(ids.chunks_exact(top)) {
            let slots = self
                .slots
                .get_mut(&layer)
                .ok_or_else(|| err("Qwen4 device route layer disappeared during commit"))?;
            for &e in selected {
                let slot = slots
                    .mapping
                    .get(e as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| err("Qwen4 device route mapping disappeared during commit"))?;
                let age = slots
                    .ages
                    .get_mut(slot)
                    .ok_or_else(|| err("Qwen4 device route has an invalid age slot"))?;
                slots.tick += 1;
                *age = slots.tick;
            }
            self.slot_hits += selected.iter().collect::<HashSet<_>>().len() as u64;
        }
        Ok(true)
    }

    /// The tentative token has completed without mutating a bank. Complete
    /// every retained reader together, including independent earlier outputs,
    /// before releasing their leases. This prevents periodic per-layer waits
    /// on the eighth hit-only token and retains the ordinary miss protection.
    pub(in super::super) fn complete_device_readers(&mut self) -> Result<()> {
        if self.plan.mapped {
            return Ok(());
        }
        let readers: Vec<_> = self.slots.values().flat_map(|s| s.readers.iter()).collect();
        MxArray::eval_arrays_with_context(&readers, "qwen4::slots::device_readers")?;
        for slots in self.slots.values_mut() {
            slots.readers.clear();
        }
        self.deferred_reduction_layer = None;
        Ok(())
    }

    pub(in super::super) fn finish_expert_slots(
        &mut self,
        layer: usize,
        output: &MxArray,
    ) -> Result<()> {
        if self.plan.mapped {
            return Ok(());
        }
        if let Some(slots) = self.slots.get_mut(&layer) {
            slots.readers.push(output.clone());
            // Hit-only sequences also have a bounded pending graph. Retain all
            // readers since independent outputs need not depend on each other.
            if slots.readers.len() >= 8 {
                MxArray::eval_arrays_with_context(
                    &slots.readers.iter().collect::<Vec<_>>(),
                    "qwen4::slots::reader_window",
                )?;
                slots.readers.clear();
            }
        }
        Ok(())
    }

    pub(in super::super) fn complete_expert_window(
        &mut self,
        layer: usize,
        reduced: &MxArray,
    ) -> Result<()> {
        if self.plan.mapped {
            return MxArray::eval_arrays_with_context(&[reduced], "qwen4::mapped::complete_window");
        }
        if let Some(slots) = self.slots.get_mut(&layer) {
            // Complete every outstanding reader, including independent earlier
            // calls. Replacing the leases with only the newest output would
            // permit a slot overwrite before those earlier graphs execute.
            let mut ready = slots.readers.iter().collect::<Vec<_>>();
            ready.push(reduced);
            MxArray::eval_arrays_with_context(&ready, "qwen4::slots::complete_window")?;
            slots.readers.clear();
        }
        Ok(())
    }

    /// Submit one wide window while retaining only its compact reduction.
    /// The caller must pass an output depending on every new bank reader.
    /// Complete all earlier independent leases before replacing them; a miss
    /// still completes this lease before any slot buffer is overwritten.
    /// At most one deferred reduced window survives in the entire store, so
    /// retained scratch does not grow with the checkpoint's layer count.
    pub(in super::super) fn defer_expert_reduction(
        &mut self,
        layer: usize,
        reduced: &MxArray,
    ) -> Result<()> {
        if self.plan.mapped {
            MxArray::async_eval_arrays(&[reduced]);
            return Ok(());
        }
        if let Some(previous) = self.deferred_reduction_layer.take()
            && let Some(slots) = self.slots.get_mut(&previous)
        {
            MxArray::eval_arrays_with_context(
                &slots.readers.iter().collect::<Vec<_>>(),
                "qwen4::slots::previous_reduction",
            )?;
            slots.readers.clear();
        }
        if let Some(slots) = self.slots.get_mut(&layer) {
            MxArray::eval_arrays_with_context(
                &slots.readers.iter().collect::<Vec<_>>(),
                "qwen4::slots::earlier_windows",
            )?;
            slots.readers.clear();
            slots.readers.push(reduced.clone());
            MxArray::async_eval_arrays(&[reduced]);
            self.deferred_reduction_layer = Some(layer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_device_route_mapping_does_not_commit_earlier_layer_ages() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let weight = Arc::new(Weight {
            values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
            dense_bf16: None,
            scales: None,
            biases: None,
            group: 0,
            bits: 0,
            mode: String::new(),
        });
        for (layer, mapping) in [vec![Some(0)], vec![None, Some(2)]].into_iter().enumerate() {
            store.slots.insert(
                layer,
                ExpertSlots {
                    banks: [weight.clone(), weight.clone(), weight.clone()],
                    names: ["gate".into(), "up".into(), "down".into()],
                    rows: [1; 3],
                    occupants: vec![Some(layer as u32)],
                    mapping,
                    device_mapping: None,
                    ages: vec![0],
                    tick: 0,
                    readers: Vec::new(),
                },
            );
        }
        assert!(!store.commit_device_routes(&[0, 1], &[0, 1], 1).unwrap());
        assert_eq!(store.slot_hits, 0);
        for slots in store.slots.values() {
            assert_eq!(slots.tick, 0);
            assert_eq!(slots.ages, [0]);
        }
    }
    #[test]
    fn completed_expert_window_releases_large_reader_storage() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let bank = Arc::new(Weight {
            dense_bf16: None,
            values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
            scales: None,
            biases: None,
            group: 0,
            bits: 0,
            mode: String::new(),
        });
        store.slots.insert(
            0,
            ExpertSlots {
                banks: [bank.clone(), bank.clone(), bank],
                names: Default::default(),
                rows: [1; 3],
                occupants: vec![Some(0)],
                mapping: vec![Some(0)],
                device_mapping: None,
                ages: vec![0],
                tick: 0,
                readers: Vec::new(),
            },
        );
        crate::array::memory::synchronize();
        let before = crate::array::memory::get_active_memory();
        let reduced = {
            let large = MxArray::ones(&[1024, 1024], Some(DType::Float32)).unwrap();
            let reduced = large.sum(Some(&[0]), Some(false)).unwrap();
            store.finish_expert_slots(0, &large).unwrap();
            MxArray::eval_arrays(&[&large]).unwrap();
            assert!(unsafe { mlx_sys::mlx_array_is_available(large.as_raw_ptr()) });
            store.complete_expert_window(0, &reduced).unwrap();
            reduced
        };
        // Sibling tests allocate and free on the same process-wide counter
        // (a residency test holds fixture banks for seconds); what this
        // window retained is what is still there once they are done.
        let mut retained = f64::MAX;
        for _ in 0..1000 {
            crate::array::memory::synchronize();
            retained = crate::array::memory::get_active_memory() - before;
            if retained < 256. * 1024. {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            retained < 256. * 1024.,
            "completed window retained {retained} bytes"
        );
        assert!(store.slots[&0].readers.is_empty());
        assert!(reduced.to_float32().unwrap().iter().all(|&x| x == 1024.));
        let previous = store.slots[&0].banks.clone();
        store.slots.insert(
            1,
            ExpertSlots {
                banks: previous,
                names: Default::default(),
                rows: [1; 3],
                occupants: vec![Some(0)],
                mapping: vec![Some(0)],
                device_mapping: None,
                ages: vec![0],
                tick: 0,
                readers: Vec::new(),
            },
        );
        store.defer_expert_reduction(0, &reduced).unwrap();
        let independent = reduced.add_scalar(1.).unwrap();
        store.defer_expert_reduction(1, &independent).unwrap();
        assert!(store.slots[&0].readers.is_empty());
        assert_eq!(store.slots[&1].readers.len(), 1);
        assert_eq!(store.deferred_reduction_layer, Some(1));
        // Join independent leases from both layers, including an unevaluated
        // reader that does not depend on the newest token's output.
        let old = store.slots[&0].banks[0].values.add_scalar(7.).unwrap();
        store.finish_expert_slots(0, &old).unwrap();
        store.complete_device_readers().unwrap();
        assert!(store.slots.values().all(|s| s.readers.is_empty()));
        assert_eq!(store.deferred_reduction_layer, None);
        assert_eq!(&*old.to_float32().unwrap(), &[7.]);
        assert!(
            independent
                .to_float32()
                .unwrap()
                .iter()
                .all(|&x| x == 1025.)
        );
    }
    #[test]
    fn slot_budget_uses_attached_auxiliary_inventory_without_reading_weights() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        store.tensors.clear();
        store.gguf = true;
        let mut routed = 0u64;
        for layer in 0..48 {
            for part in ["gate", "up", "down"] {
                let t = Tensor {
                    path: PathBuf::new(),
                    shape: if part == "down" {
                        vec![512, 2560, 640]
                    } else {
                        vec![512, 640, 2560]
                    },
                    offset: 0,
                    bytes: 0,
                    encoding: Encoding::Gguf(if part == "down" {
                        GgufTensorType::Q5_1
                    } else {
                        GgufTensorType::Q4K
                    }),
                };
                routed += t.runtime_bytes().unwrap();
                store
                    .tensors
                    .insert(format!("blk.{layer}.ffn_{part}_exps.weight"), t);
            }
        }
        let fixed = 4 << 30;
        store.plan.hot_bytes = routed + fixed;
        let mut previous = 0;
        for gib in [16, 32, 48] {
            store.cache_limit = gib << 30;
            store.slot_capacity = None;
            store.tensors.remove("mtp.test.weight");
            store.tensors.remove("nextn.0.ffn_gate_exps.weight");
            let text = store.expert_slot_capacity(512, 10).unwrap();
            assert!(text > previous && text < 512);
            previous = text;
            assert!(routed * text as u64 / 512 + fixed + MAX_READ_BYTES * 6 <= store.cache_limit);
            assert!(
                (routed + routed / 48) * text as u64 / 512 + fixed + MAX_READ_BYTES * 6
                    <= store.cache_limit,
                "largest-layer staging must also fit the admitted weight budget"
            );
            store.tensors.insert(
                "mtp.test.weight".into(),
                Tensor {
                    path: PathBuf::new(),
                    shape: vec![65536, 65536],
                    offset: 0,
                    bytes: 8 << 30,
                    encoding: Encoding::Safe("BF16".into()),
                },
            );
            store.slot_capacity = None;
            let auxiliary = store.expert_slot_capacity(512, 10).unwrap();
            assert!(
                auxiliary < text,
                "attached auxiliary weights need their own cache allowance"
            );
            assert!(
                routed * auxiliary as u64 / 512 + fixed + (store.cache_limit / 8)
                    <= store.cache_limit
            );
            let extra = store.tensors["mtp.test.weight"].clone();
            store
                .tensors
                .insert("nextn.0.ffn_gate_exps.weight".into(), extra);
            store.slot_capacity = None;
            assert_eq!(
                store.expert_slot_capacity(512, 10).unwrap(),
                auxiliary,
                "auxiliary experts must not be counted as main-network banks"
            );
        }
        assert_eq!(store.bytes_read, 0);
        assert_eq!(store.slot_bytes, 0);
    }
    #[test]
    fn slot_replacement_completes_all_lazy_readers_and_retains_warm_codes() {
        let dir = std::env::temp_dir().join(format!("qwen4-slots-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let path = dir.join("experts.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        let string = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.bytes());
        };
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        for (part, name) in ["gate", "up", "down"].iter().enumerate() {
            string(&mut b, &format!("blk.0.ffn_{name}_exps.weight"));
            b.extend(3u32.to_le_bytes());
            for d in [32u64, 32, 5] {
                b.extend(d.to_le_bytes());
            }
            b.extend(7u32.to_le_bytes());
            b.extend((part as u64 * 5 * 32 * 24).to_le_bytes());
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for row in 0..3 * 5 * 32 {
            b.extend(half::f16::from_f32(0.015625).to_bits().to_le_bytes());
            b.extend(half::f16::from_f32(-0.25).to_bits().to_le_bytes());
            b.extend((row as u32 * 7919).to_le_bytes());
            b.extend([row as u8; 16]);
        }
        std::fs::write(&path, b).unwrap();
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        store.slot_capacity = Some(2);
        let x = MxArray::from_float32(
            &(0..64).map(|n| (n as f32 - 32.) / 64.).collect::<Vec<_>>(),
            &[2, 1, 32],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let mut expected = Vec::new();
        for ids in [[0u32, 1], [1, 0], [2, 3], [0, 4], [4, 0]] {
            let rows = ids
                .iter()
                .enumerate()
                .map(|(r, &id)| {
                    store
                        .read("blk.0.ffn_gate_exps.weight", id as usize * 32, 32)
                        .unwrap()
                        .linear(&x.slice_axis(0, r as i64, r as i64 + 1).unwrap())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            expected.push(
                MxArray::concatenate_many(rows.iter().collect(), Some(0))
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec(),
            );
        }
        let mut lazy = Vec::new();
        for (n, ids) in [[0u32, 1], [1, 0], [2, 3], [0, 4], [4, 0]]
            .iter()
            .enumerate()
        {
            let uploads = store.slot_upload_bytes;
            let (banks, local, count) = store.expert_slots(0, 5, 2, ids).unwrap().unwrap();
            let (_, device_local) = store
                .device_expert_slots(0, &MxArray::from_uint32(ids, &[2]).unwrap())
                .unwrap()
                .unwrap();
            assert_eq!(&*device_local.to_uint32().unwrap(), &local);
            assert!(store.commit_device_routes(&[0], ids, 2).unwrap());
            let absent = (0..5u32).find(|e| !ids.contains(e)).unwrap();
            let old_ages = store.slots[&0].ages.clone();
            let (_, missing) = store
                .device_expert_slots(0, &MxArray::from_uint32(&[absent], &[1]).unwrap())
                .unwrap()
                .unwrap();
            assert_eq!(&*missing.to_uint32().unwrap(), &[count as u32]);
            let (_, safe, safe_count) = store
                .prefill_device_expert_slots(
                    0,
                    &MxArray::from_uint32(&[ids[0], absent, ids[1]], &[3]).unwrap(),
                )
                .unwrap();
            assert_eq!(safe_count, count);
            assert_eq!(&*safe.to_uint32().unwrap(), &[local[0], 0, local[1]]);
            assert!(
                !store
                    .commit_device_routes(&[0], &[ids[0], absent, ids[1]], 3)
                    .unwrap()
            );
            assert!(!store.commit_device_routes(&[0], &[absent], 1).unwrap());
            assert_eq!(store.slots[&0].ages, old_ages);
            let output = banks[0]
                .gather_rows(
                    &x,
                    &MxArray::from_uint32(&local, &[2]).unwrap(),
                    count,
                    false,
                )
                .unwrap();
            if n == 1 || n == 3 {
                // Replaces earlier independent readers on a hit, then
                // protects an asynchronously submitted result across a miss.
                store.defer_expert_reduction(0, &output).unwrap();
                assert_eq!(store.slots[&0].readers.len(), 1);
            } else {
                store.finish_expert_slots(0, &output).unwrap();
            }
            if n == 1 || n == 4 {
                assert_eq!(
                    store.slot_upload_bytes, uploads,
                    "warm slots must not upload"
                );
            }
            lazy.push(output);
        }
        let reduced = lazy.last().unwrap().sum(Some(&[-1]), Some(false)).unwrap();
        store.complete_expert_window(0, &reduced).unwrap();
        assert!(store.slots[&0].readers.is_empty());
        store.expert_slots(0, 5, 2, &[1, 2]).unwrap().unwrap();
        for (output, want) in lazy.iter().zip(expected) {
            let got = output.astype(DType::Float32).unwrap().to_float32().unwrap();
            assert_eq!(&*got, &want, "reusing a slot changed a prior lazy output");
        }
        assert_eq!(store.slots[&0].occupants.len(), 2);
        assert!(store.cache_bytes >= store.slot_bytes);
        assert!(store.peak_cache_bytes <= store.cache_limit);
        let reads = store.bytes_read;
        assert!(store.expert_slots(0, 5, 2, &[5]).is_err());
        assert_eq!(
            reads, store.bytes_read,
            "bad ids must fail before reading weights"
        );
    }
}
