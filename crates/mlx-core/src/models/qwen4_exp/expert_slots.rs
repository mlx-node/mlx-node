//! Stable, quantized expert slots for partial residency. Slot buffers are
//! mutated only after their last lazy reader has completed. The ordinary LRU
//! supplies bounded misses and releases imported chunks after the slot copy.
use super::*;
use crate::array::alloc::{OutLayout, SendPtr};
use crate::array::purgeable::{PurgeableBank, PurgeableState};
use crate::utils::gguf_gpu::{
    GpuRepackKind, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available, gpu_repack_band,
};
use crate::utils::gguf_read::{ALIGN, READ_THREADS, Reader, Window, plan_reads};
use std::time::{Duration, Instant};

const DIRECT_READ_BYTES: usize = 8 << 20;
const DIRECT_STAGING_SLOTS: usize = 4;
const SLOT_RECLAIM_FALLBACK: Duration = Duration::from_secs(30);

/// The driver unwires a bound file-backed bank ~1.15 s after its last use and
/// re-wires resident pages at 12-17 ms/GiB, one command buffer at a time. A
/// forward pass that starts later than that after the previous one (the first
/// decode step after prefill, any slow step) wires 78 GiB layer by layer, and
/// layer 0 is idle long enough to unwire again before the pass ends: measured
/// 0.4-0.6 tokens/s until something breaks the cycle. One eval reading one
/// element of every mapped bank gives them all the same last use first.
const MAPPED_WIRE_GAP: Duration = Duration::from_millis(900);
/// Under the driver's ~1.15 s unwire timer and the 900 ms gap above.
const KEEP_WIRED_EVERY: Duration = Duration::from_millis(700);

/// One layer's routed experts addressed in place: the pack's sections as
/// no-copy banks, slot id == expert id, so nothing is ever copied, missed
/// or overwritten and no reader lease is needed.
pub(super) struct MappedExperts {
    banks: [Arc<Weight>; 3],
    pub(super) maps: [Arc<crate::array::Mmap>; 3],
    experts: usize,
}

/// Whether every probe of every mapped bank file is in the page cache: the
/// pack's own evidence behind the process-wide SSD count.
fn pack_resident(mapped: &HashMap<usize, MappedExperts>) -> bool {
    mapped
        .values()
        .flat_map(|experts| experts.maps.iter())
        .all(|map| super::super::prefetch::sampled_resident(map))
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

struct SlotBank {
    template: Weight,
    storage: Vec<PurgeableBank>,
}

impl SlotBank {
    fn set_purgeable(&self, state: PurgeableState) -> Result<bool> {
        let mut empty = false;
        for bank in &self.storage {
            empty |= bank.set_purgeable(state)? == PurgeableState::Empty;
        }
        Ok(empty)
    }
}

pub(super) struct ExpertSlots {
    banks: [Arc<Weight>; 3],
    storage: Option<[SlotBank; 3]>,
    direct: bool,
    suspended: bool,
    purge_requested: bool,
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
    #[cfg(not(target_os = "macos"))]
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
    #[cfg(target_os = "macos")]
    fn purgeable_slots(&self, capacity: usize) -> Result<(Self, SlotBank)> {
        let mut storage = Vec::with_capacity(3);
        let mut allocate = |a: &MxArray| -> Result<MxArray> {
            let shape = a.shape()?;
            let bank = PurgeableBank::new(&[shape[0] * capacity as i64, shape[1]], a.dtype()?)?;
            let array = bank.array().clone();
            storage.push(bank);
            Ok(array)
        };
        let template = Self {
            values: allocate(&self.values)?,
            dense_bf16: None,
            scales: self.scales.as_ref().map(&mut allocate).transpose()?,
            biases: self.biases.as_ref().map(&mut allocate).transpose()?,
            ..self.clone()
        };
        Ok((template.clone(), SlotBank { template, storage }))
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

impl Tensor {
    fn direct_kind(&self) -> Option<(GgufTensorType, GpuRepackKind)> {
        let Encoding::Gguf(ty) = &self.encoding else {
            return None;
        };
        let kind = match ty {
            GgufTensorType::Q8_0 => GpuRepackKind::Q8_0Bias,
            GgufTensorType::Q5_1 => GpuRepackKind::Q5_1,
            _ if ty.k_quant_format().is_some() => gpu_kind_for(*ty)?,
            _ => return None,
        };
        Some((*ty, kind))
    }

    fn direct_weight(&self, name: &str, rows: usize) -> Result<Weight> {
        let (ty, kind) = self
            .direct_kind()
            .ok_or_else(|| err(format!("{name} has no direct GGUF repack")))?;
        let info = GgufTensorInfo {
            name: name.into(),
            n_dims: 2,
            dims: vec![self.width()? as u64, rows as u64],
            tensor_type: ty,
            offset: 0,
        };
        let (dst, _) = RepackDst::alloc_for_kind(&info, kind, OutLayout::RowMajor)?;
        let (group, bits, mode) = if let Some(format) = ty.k_quant_format() {
            (
                format.group_size() as i32,
                format.bits() as i32,
                format.mlx_mode(),
            )
        } else {
            (32, if ty == GgufTensorType::Q5_1 { 5 } else { 8 }, "affine")
        };
        Ok(Weight {
            values: dst.weight,
            dense_bf16: None,
            scales: dst.scales,
            biases: dst.biases,
            group,
            bits,
            mode: mode.into(),
        })
    }
}

impl ExpertSlots {
    fn activate(&mut self) -> Result<()> {
        if !self.suspended {
            return Ok(());
        }
        let mut empty = std::mem::take(&mut self.purge_requested);
        if let Some(storage) = &self.storage {
            for bank in storage {
                empty |= bank.set_purgeable(PurgeableState::NonVolatile)?;
            }
        }
        if empty {
            if let Some(storage) = &self.storage {
                self.banks = storage
                    .each_ref()
                    .map(|bank| Arc::new(bank.template.clone()));
            }
            self.invalidate_all();
        }
        self.suspended = false;
        Ok(())
    }

    fn suspend(&mut self, force_empty: bool) -> Result<()> {
        let Some(storage) = &self.storage else {
            return Ok(());
        };
        for bank in storage {
            bank.set_purgeable(PurgeableState::Volatile)?;
            if force_empty {
                bank.set_purgeable(PurgeableState::Empty)?;
            }
        }
        self.purge_requested |= force_empty;
        self.suspended = true;
        Ok(())
    }

    fn invalidate_all(&mut self) {
        self.occupants.fill(None);
        self.mapping.fill(None);
        self.device_mapping = None;
        self.ages.fill(0);
        self.tick = 0;
    }

    fn invalidate_transaction(&mut self) {
        // A direct fill writes destinations in place. On any error, no bank
        // from this layer may publish a mixture of old and new source bytes.
        self.invalidate_all();
    }
}

struct DirectRequest {
    part: usize,
    slot: usize,
    rows: usize,
    k: usize,
    kind: GpuRepackKind,
}

struct DirectRun<'r, 's> {
    reader: &'r Reader<'s>,
    requests: &'r [DirectRequest],
    windows: &'r [Window],
    staging: &'r [(MxArray, SendPtr<u8>)],
    slots: &'r mut ExpertSlots,
}

impl DirectRun<'_, '_> {
    fn submit(&self, w: usize) {
        let (_, ptr) = &self.staging[w % self.staging.len()];
        let window = &self.windows[w];
        self.reader
            .submit(w as u64, *ptr, window.len, window.file_off);
    }

    fn complete(&self, _w: usize, outputs: &[MxArray]) -> Result<()> {
        let refs: Vec<&MxArray> = outputs.iter().collect();
        MxArray::eval_arrays_with_context(&refs, "qwen4::slots::direct_gguf_window")
    }

    fn all(&mut self) -> Result<()> {
        let n = self.windows.len();
        let slots = self.staging.len();
        if n == 0 {
            return Err(err("Qwen4 direct GGUF fill has no read windows"));
        }
        for w in 0..(slots - 1).min(n) {
            self.submit(w);
        }
        let mut pending: Vec<MxArray> = Vec::new();
        for w in 0..n {
            if w > 0 {
                self.complete(w - 1, &std::mem::take(&mut pending))?;
            }
            let next = w + slots - 1;
            if next < n {
                self.submit(next);
            }
            self.reader.wait(w as u64)?;
            pending = self.window(w)?;
            let refs: Vec<&MxArray> = pending.iter().collect();
            MxArray::async_eval_arrays(&refs);
        }
        self.complete(n - 1, &pending)
    }

    fn window(&mut self, w: usize) -> Result<Vec<MxArray>> {
        let (staging, _) = &self.staging[w % self.staging.len()];
        let mut outputs = Vec::new();
        for band in &self.windows[w].bands {
            let request = &self.requests[band.tensor];
            let current = Arc::clone(&self.slots.banks[request.part]);
            let destination_shape = current.values.shape()?.to_vec();
            let dst = RepackDst {
                weight: current.values.clone(),
                scales: current.scales.clone(),
                biases: current.biases.clone(),
            };
            let row0 = request
                .slot
                .checked_mul(request.rows)
                .and_then(|row| row.checked_add(band.row0))
                .and_then(|row| u32::try_from(row).ok())
                .ok_or_else(|| err("Qwen4 direct GGUF destination row exceeds u32"))?;
            let k = u32::try_from(request.k)
                .map_err(|_| err("Qwen4 direct GGUF row width exceeds u32"))?;
            let rows = u32::try_from(band.rows)
                .map_err(|_| err("Qwen4 direct GGUF band rows exceed u32"))?;
            let next = gpu_repack_band(
                staging,
                band.src_off as u64,
                request.kind,
                OutLayout::RowMajor,
                k,
                row0,
                rows,
                dst,
            )
            .map_err(|error| {
                err(format!(
                    "Qwen4 direct GGUF {} slot {} band row {} + {} (dst {destination_shape:?}, k {k}): {error}",
                    self.slots.names[request.part], request.slot, row0, rows
                ))
            })?;
            outputs.push(next.weight.clone());
            outputs.extend(next.scales.iter().cloned());
            outputs.extend(next.biases.iter().cloned());
            let mut weight = current.as_ref().clone();
            weight.values = next.weight;
            weight.scales = next.scales;
            weight.biases = next.biases;
            self.slots.banks[request.part] = Arc::new(weight);
        }
        Ok(outputs)
    }
}

impl Store {
    fn fill_direct_chunk(
        &mut self,
        slots: &mut ExpertSlots,
        experts: &[u32],
        chosen: &[usize],
    ) -> Result<()> {
        let source = self
            .gguf_source
            .clone()
            .ok_or_else(|| err("Qwen4 direct GGUF fill has no main source"))?;
        source.verify_path()?;
        let file = self.file(&source.path).map_err(|error| {
            err(format!(
                "Failed to open GGUF '{}': {error}",
                source.path.display()
            ))
        })?;
        source.verify(&file)?;
        let mut requests = Vec::with_capacity(experts.len() * 3);
        let mut infos = Vec::with_capacity(experts.len() * 3);
        let mut source_bytes = 0u64;
        let mut bank_bytes = 0u64;
        for (&expert, &slot) in experts.iter().zip(chosen) {
            for part in 0..3 {
                let tensor = self.descriptor(&slots.names[part])?.clone();
                let (ty, kind) = tensor
                    .direct_kind()
                    .ok_or_else(|| err("Qwen4 direct GGUF fill lost its repack route"))?;
                let rows = slots.rows[part];
                let start = expert as usize * rows;
                let (offset, bytes) = tensor.range(start, rows)?;
                source_bytes += bytes;
                bank_bytes += tensor.runtime_bytes()? / tensor.shape[0] as u64;
                infos.push(GgufTensorInfo {
                    name: slots.names[part].clone(),
                    n_dims: 2,
                    dims: vec![tensor.width()? as u64, rows as u64],
                    tensor_type: ty,
                    offset,
                });
                requests.push(DirectRequest {
                    part,
                    slot,
                    rows,
                    k: tensor.width()?,
                    kind,
                });
            }
        }
        let header = GgufFile {
            version: 3,
            tensor_count: infos.len() as u64,
            metadata: HashMap::new(),
            tensors: infos,
            alignment: 32,
            data_offset: 0,
        };
        let windows = plan_reads(&header, source.len, DIRECT_READ_BYTES, &|_| 1)?;
        let slot_len = windows.iter().map(|w| w.len).max().unwrap_or(0);
        // Keep the aligned backing allocations alive until the GPU stream drains.
        let mut _staging_owners = Vec::with_capacity(DIRECT_STAGING_SLOTS.min(windows.len()));
        let mut staging = Vec::with_capacity(DIRECT_STAGING_SLOTS.min(windows.len()));
        for _ in 0..DIRECT_STAGING_SLOTS.min(windows.len()) {
            let allocation = slot_len
                .checked_add(ALIGN as usize)
                .ok_or_else(|| err("Qwen4 direct GGUF staging size overflows usize"))?;
            let bank = PurgeableBank::new(&[allocation as i64], DType::Uint8)?;
            let base = bank.data_ptr();
            let padding = (ALIGN as usize - (base.0 as usize % ALIGN as usize)) % ALIGN as usize;
            let array = bank
                .array()
                .slice_axis(0, padding as i64, (padding + slot_len) as i64)?;
            let ptr = SendPtr(base.0.wrapping_add(padding));
            if !(ptr.0 as usize).is_multiple_of(ALIGN as usize) {
                return Err(err("Qwen4 direct GGUF staging is not 16 KiB aligned"));
            }
            _staging_owners.push(bank);
            staging.push((array, ptr));
        }
        let result = std::thread::scope(|scope| {
            let reader = Reader::start(
                scope,
                file.as_ref(),
                &source.path,
                READ_THREADS,
                DIRECT_READ_BYTES,
            );
            let mut run = DirectRun {
                reader: &reader,
                requests: &requests,
                windows: &windows,
                staging: &staging,
                slots,
            };
            run.all()
        });
        // A failed window may already have encoded earlier bands. Finish them
        // before the caller unpublishes every destination in this transaction.
        crate::array::memory::synchronize();
        result?;
        source.verify(&file)?;
        #[cfg(test)]
        self.direct_read_windows
            .extend(windows.iter().map(|w| (w.file_off, w.len)));
        self.bytes_read += source_bytes;
        self.slot_upload_bytes += bank_bytes;
        Ok(())
    }

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

    /// Map every routed expert of every layer from the pack when the pack is
    /// complete and the live host holds it beside what stays anonymous
    /// (memory::mapped_plan), replacing the slot budget by one for dense
    /// banks and auxiliary rows only. Any failure leaves the slot path in
    /// charge. `pool_bytes` is the paged KV pool created before this
    /// decision.
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
        if !complete {
            return Ok(());
        }
        let pack_bytes: u64 = layers
            .iter()
            .flat_map(|&layer| names(layer))
            .filter_map(|name| pack.file(&name))
            .flat_map(|file| file.header.sections.iter().map(|s| s.length))
            .sum();
        let dense = self.fixed_projection_bytes()?;
        // The Q8 draft banks are anonymous MLX memory built at load: a mapped
        // host must hold them beside the dense banks.
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
        match super::super::prefetch::Pool::new(
            files,
            super::super::prefetch::Pread,
            super::super::prefetch::WORKERS,
            Arc::clone(&self.rows_pending),
        ) {
            Ok(pool) => self.prefetch = Some(pool),
            Err(e) => tracing::warn!("qwen4 mapped pack stream unavailable: {e}"),
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

    /// The load settled on its residency and read the dense banks: build the
    /// MTP head's draft banks, then start the pack stream. A head the host
    /// cannot stage keeps its drafts on the row reader for the process; the
    /// load goes on.
    pub(in super::super) fn finish_load(&mut self) {
        self.finish_load_with(&mut super::super::memory::admit);
    }

    /// `finish_load` without the live host admission (see
    /// `prepare_fixture_mtp_expert_banks`).
    #[cfg(test)]
    pub(in super::super) fn finish_fixture_load(&mut self) {
        self.finish_load_with(&mut |_| Ok(()));
    }

    pub(super) fn finish_load_with(&mut self, admit: &mut impl FnMut(u64) -> Result<()>) {
        if let Err(error) = self.prepare_mtp_expert_banks_with(admit) {
            tracing::warn!(
                reason = %error.reason,
                "Qwen4 MTP expert banks unavailable; drafts read expert rows"
            );
        }
        self.start_pack_stream();
    }

    /// The load picked the mapped plan and read the dense banks: stream the
    /// pack into the page cache, so a request that arrives after the stream
    /// finds it resident. Nothing is wired by it and the live-memory gate's
    /// inputs do not move (file pages are reclaimable).
    pub(in super::super) fn start_pack_stream(&self) {
        if self.plan.mapped
            && let Some(pool) = &self.prefetch
            && !pool.sampled_resident()
        {
            pool.start();
        }
    }

    /// A turn opens over the mapped pack: a pack that is not fully resident
    /// gets a stream unless one runs already, and the turn's first pass
    /// streams instead of touching every bank.
    fn stream_for_turn(&mut self) {
        self.streaming_pass = false;
        if !self.plan.mapped {
            return;
        }
        let Some(pool) = &self.prefetch else {
            return;
        };
        if pool.streaming() {
            self.streaming_pass = true;
        } else if !pool.sampled_resident() {
            pool.start();
            self.streaming_pass = true;
        }
    }

    pub(in super::super) fn activate_expert_slots(&mut self) -> Result<()> {
        if self
            .slot_reclaim_since
            .is_some_and(|since| since.elapsed() >= SLOT_RECLAIM_FALLBACK)
        {
            self.purge_expert_slots()?;
        }
        for slots in self.slots.values_mut() {
            slots.activate()?;
        }
        self.slot_reclaim_since = None;
        Ok(())
    }

    /// Between turns, complete every bank reader before making slot storage
    /// reclaimable. Explicit Empty is the fallback when automatic reclaim is
    /// unavailable or a probe needs a guaranteed refill.
    pub(in super::super) fn reclaim_expert_slots(&mut self) -> Result<()> {
        self.suspend_expert_slots(false)
    }

    pub(in super::super) fn purge_expert_slots(&mut self) -> Result<()> {
        self.suspend_expert_slots(true)
    }

    fn suspend_expert_slots(&mut self, force_empty: bool) -> Result<()> {
        if self.plan.mapped {
            return Ok(());
        }
        self.complete_device_readers()?;
        crate::array::memory::synchronize();
        for slots in self.slots.values_mut() {
            slots.suspend(force_empty)?;
        }
        self.slot_reclaim_since = if force_empty {
            None
        } else if self.slots.values().any(|slots| slots.storage.is_some()) {
            Some(Instant::now())
        } else {
            None
        };
        Ok(())
    }

    /// The first turn of a measurement window opens: a trip whose switch
    /// failed at its pass moves the process to the slot path first, then the
    /// window's SSD count and slow streak begin. Returns the fallback applied.
    pub(in super::super) fn begin_turn(&mut self) -> Result<Option<Fallback>> {
        let applied = self.apply_fallback()?;
        self.activate_expert_slots()?;
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
            _ => {
                self.stream_for_turn();
                Ok(None)
            }
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
        let mapped = &self.mapped_experts;
        let Some(why) = self
            .tripwire
            .observe(rows, elapsed, reads, self.plan.mapped_bytes, || {
                pack_resident(mapped)
            })
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

    /// A layer sweep of a prefill group ended: the SSD rule alone is checked
    /// here (the slow-pass rules see the group's windows after it). Returns
    /// whether a trip is held, this one or an earlier one not yet applied.
    pub(in super::super) fn observe_sweep(&mut self) -> bool {
        if !self.plan.mapped {
            return self.tripwire.tripped().is_some();
        }
        let mapped = &self.mapped_experts;
        if let Some(why) = self.tripwire.observe_ssd(
            super::super::memory::process_disk_reads(),
            self.plan.mapped_bytes,
            || pack_resident(mapped),
        ) {
            tracing::warn!(
                "qwen4 residency: mapped turn thrashes ({why}); the group goes on over slots after this sweep"
            );
        }
        self.tripwire.tripped().is_some()
    }

    /// Apply a trip held by `observe_sweep` (or armed earlier) now that
    /// nothing lazy reads a mapped bank: the store leaves the mapped path and
    /// the request goes on over slots. Without a held trip nothing changes;
    /// a switch that fails leaves the trip armed for the next turn to open.
    pub(in super::super) fn apply_held_trip(&mut self) -> Result<Option<Fallback>> {
        let Some(why) = self.tripwire.take() else {
            return Ok(None);
        };
        if !self.plan.mapped {
            return Ok(None);
        }
        match self.leave_mapped(&why) {
            Ok(()) => Ok(Some(why)),
            Err(error) => {
                self.tripwire.rearm(why);
                Err(error)
            }
        }
    }

    /// A layer-major prefill group of more than one window starts: its
    /// bank touches run asynchronously and keep going after the stream ends,
    /// until `end_group`.
    pub(in super::super) fn begin_group(&mut self) {
        self.group_in_progress = true;
    }

    /// The group ended, committed or not: touches issued since
    /// `begin_group` and not taken by a commit are dropped.
    pub(in super::super) fn end_group(&mut self) {
        self.group_in_progress = false;
        self.pending_touch.clear();
    }

    /// The arrays of the asynchronous touches since the last take; a group
    /// commit evaluates them with its frontier.
    pub(in super::super) fn take_pending_touches(&mut self) -> Vec<MxArray> {
        std::mem::take(&mut self.pending_touch)
    }

    /// The turn opened over a pack that streams (or streamed) into the page
    /// cache: its first pass runs beside the stream.
    pub(in super::super) fn streaming_pass(&self) -> bool {
        self.streaming_pass
    }

    /// Whether the pack stream's threads exist.
    #[cfg(test)]
    pub(in super::super) fn pack_stream_alive(&self) -> bool {
        self.prefetch.is_some()
    }

    #[cfg(test)]
    pub(in super::super) fn group_in_progress(&self) -> bool {
        self.group_in_progress
    }

    #[cfg(test)]
    pub(in super::super) fn pending_touches(&self) -> usize {
        self.pending_touch.len()
    }

    #[cfg(test)]
    pub(in super::super) fn tripwire_mut(&mut self) -> &mut super::super::tripwire::Tripwire {
        &mut self.tripwire
    }

    /// Hold `why` as a sweep or pass would have tripped it; the store
    /// applies it where it applies a real trip.
    #[cfg(test)]
    pub(in super::super) fn trip(&mut self, why: Fallback) {
        self.tripwire.rearm(why);
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
        self.streaming_pass = false;
        self.last_touch = None;
        self.pending_touch.clear();
        self.mapped_pass_start = None;
        self.slot_capacity = None;
        self.slot_reclaim_since = None;
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
        self.wire_mapped_banks_at(Instant::now(), main).map(|_| ())
    }

    /// Returns whether the banks were touched. A stale pass while the pack
    /// streams does not touch: the touch binds every bank at once, so the
    /// driver would fault the whole pack in on the model thread at its own
    /// rate before layer 0 runs; the stream is ahead of the layers instead.
    pub(super) fn wire_mapped_banks_at(&mut self, now: Instant, main: bool) -> Result<bool> {
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
        if self.streaming_pass {
            if self.prefetch.as_ref().is_some_and(|pool| pool.streaming()) {
                self.last_touch = Some(now);
                return Ok(false);
            }
            self.streaming_pass = false;
        }
        let elapsed = self.touch_banks(usize::MAX)?;
        if elapsed > Duration::from_millis(50) {
            tracing::debug!("qwen4 mapped banks wired in {elapsed:.1?}");
        }
        self.mapped_pass_start = Some(now + elapsed);
        Ok(true)
    }

    /// Layer `layer` of a pass starts. While the pack streams or a prefill
    /// group runs, the banks of the layers already run are touched every
    /// `KEEP_WIRED_EVERY` so the driver keeps them wired through the long
    /// first window: it unwires a bank idle for ~1.15 s, and re-wiring all of
    /// them at the next window cost about a second with the GPU idle. Layers
    /// past `layer` are not touched: their pages may still be cold.
    pub(in super::super) fn keep_wired(&mut self, layer: usize, main: bool) -> Result<()> {
        self.keep_wired_at(Instant::now(), layer, main)
    }

    pub(super) fn keep_wired_at(&mut self, now: Instant, layer: usize, main: bool) -> Result<()> {
        if !main || !self.plan.mapped || !(self.streaming_pass || self.group_in_progress) {
            return Ok(());
        }
        if self
            .last_touch
            .is_some_and(|last| now.saturating_duration_since(last) < KEEP_WIRED_EVERY)
        {
            return Ok(());
        }
        self.last_touch = Some(now);
        let elapsed = self.touch_banks(layer)?;
        tracing::debug!("qwen4 mapped banks through layer {layer} kept wired in {elapsed:.1?}");
        if !self.prefetch.as_ref().is_some_and(|pool| pool.streaming()) {
            self.streaming_pass = false;
        }
        Ok(())
    }

    /// The mapped store's stream over the same files through `issue`.
    #[cfg(test)]
    pub(super) fn pool_with(&mut self, issue: impl super::super::prefetch::Issue) {
        let last = self.mapped_experts.keys().max().copied().unwrap_or(0);
        let files = (0..=last)
            .map(|layer| self.mapped_experts.get(&layer).map(|e| e.maps.clone()))
            .collect();
        self.prefetch = Some(
            super::super::prefetch::Pool::new(files, issue, 2, Arc::clone(&self.rows_pending))
                .unwrap(),
        );
    }

    /// One eval reading one element of every bank of the layers up to
    /// `through`, which binds them in a single command buffer. Inside a
    /// prefill group the eval would wait behind the queued sweeps (51-258 ms
    /// measured), so the read is submitted asynchronously and its arrays
    /// wait for the group commit.
    fn touch_banks(&mut self, through: usize) -> Result<Duration> {
        #[cfg(test)]
        {
            let mut layers: Vec<usize> = self
                .mapped_experts
                .keys()
                .copied()
                .filter(|&layer| layer <= through)
                .collect();
            layers.sort_unstable();
            self.touches.push(layers);
        }
        let mut groups: Vec<(DType, Vec<MxArray>)> = Vec::new();
        for (_, mapped) in self
            .mapped_experts
            .iter()
            .filter(|(layer, _)| **layer <= through)
        {
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
        if self.group_in_progress {
            #[cfg(test)]
            self.touches_async.push(true);
            MxArray::async_eval_arrays(&touched.iter().collect::<Vec<_>>());
            self.pending_touch.extend(touched);
        } else {
            #[cfg(test)]
            self.touches_async.push(false);
            MxArray::eval_arrays_with_context(
                &touched.iter().collect::<Vec<_>>(),
                "qwen4::mapped::wire",
            )?;
        }
        Ok(started.elapsed())
    }

    pub(in super::super) fn expert_slot_capacity(
        &mut self,
        experts: usize,
        top: usize,
    ) -> Result<usize> {
        if self.plan.mapped {
            return Ok(experts);
        }
        if !self.gguf || !crate::engine::persistence::compiled_forward_backend_available() {
            return Ok(0);
        }
        if let Some(n) = self.slot_capacity {
            return Ok(n);
        }
        let mut layer_bytes = HashMap::<&str, u64>::new();
        let routed = self.slot_routed_bytes()?;
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
        // Draft banks stay pinned beside every slot bank and are not in
        // `fixed`: the reserve holds the installed ones, or the ones the load
        // admits right after this count is cached, and the staging.
        let allowance = self.auxiliary_cache_allowance()?;
        let estimate = self.mtp_bank_estimate()?;
        let admissible = if !self.mtp_banks_declined && estimate <= allowance {
            estimate
        } else {
            0
        };
        let draft = self.mtp_bank_bytes()?.max(admissible);
        let reserve = allowance.max(draft.saturating_add(MAX_READ_BYTES * 6));
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

    /// Routed main-network expert bytes the slots serve.
    fn slot_routed_bytes(&self) -> Result<u64> {
        self.tensors
            .iter()
            .filter(|(name, _)| {
                name.contains("_exps.weight")
                    && !super::residency::is_auxiliary(name)
                    && !self.banks.contains_key(*name)
            })
            .try_fold(0u64, |n, (_, t)| {
                t.runtime_bytes().map(|b| n.saturating_add(b))
            })
    }

    pub(super) fn direct_layer(&self, names: &[String; 3]) -> bool {
        #[cfg(test)]
        if !self.direct_gguf {
            return false;
        }
        if self.plan.mapped || !gpu_repack_available() {
            return false;
        }
        let Some(source) = &self.gguf_source else {
            return false;
        };
        names.iter().all(|name| {
            self.descriptor(name).is_ok_and(|tensor| {
                tensor.path == source.path
                    && tensor.direct_kind().is_some_and(|(_, kind)| {
                        let rule = band_offset_rule(kind) as u64;
                        tensor.offset.is_multiple_of(rule)
                            && tensor.row_bytes().is_ok_and(|n| {
                                n <= DIRECT_READ_BYTES as u64 && n.is_multiple_of(rule)
                            })
                    })
            })
        })
    }

    /// Every layer's slot bank at the cached capacity, built or not.
    pub(super) fn planned_slot_bytes(&self) -> Result<u64> {
        let experts = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0);
        let planned = match self.slot_capacity {
            Some(capacity) if !self.plan.mapped && experts > 0 => {
                (self.slot_routed_bytes()? as u128 * capacity as u128 / experts as u128) as u64
            }
            _ => 0,
        };
        Ok(planned.max(self.slot_bytes))
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
            let direct = self.direct_layer(&names);
            let prototypes: Vec<Arc<Weight>> = if direct {
                names
                    .iter()
                    .zip(rows)
                    .map(|(name, rows)| {
                        self.descriptor(name)?
                            .direct_weight(name, rows)
                            .map(Arc::new)
                    })
                    .collect::<Result<_>>()?
            } else {
                vec![
                    self.read(&names[0], 0, rows[0])?,
                    self.read(&names[1], 0, rows[1])?,
                    self.read(&names[2], 0, rows[2])?,
                ]
            };
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
            #[cfg(target_os = "macos")]
            let (banks, storage) = {
                let mut banks = Vec::with_capacity(3);
                let mut storage = Vec::with_capacity(3);
                for prototype in &prototypes {
                    let (bank, owner) = prototype.purgeable_slots(capacity)?;
                    banks.push(Arc::new(bank));
                    storage.push(owner);
                }
                let banks: [Arc<Weight>; 3] = banks
                    .try_into()
                    .map_err(|_| err("Qwen4 slot bank count mismatch"))?;
                let storage: [SlotBank; 3] = storage
                    .try_into()
                    .map_err(|_| err("Qwen4 slot storage count mismatch"))?;
                (banks, Some(storage))
            };
            #[cfg(not(target_os = "macos"))]
            let (banks, storage) = {
                let banks = [
                    Arc::new(prototypes[0].empty_slots(capacity)?),
                    Arc::new(prototypes[1].empty_slots(capacity)?),
                    Arc::new(prototypes[2].empty_slots(capacity)?),
                ];
                (banks, None)
            };
            self.slot_bytes += bytes;
            self.cache_bytes += bytes;
            self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
            ExpertSlots {
                banks,
                storage,
                direct,
                suspended: false,
                purge_requested: false,
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
        slots.activate()?;
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
                if slots.direct {
                    if let Err(error) = self.fill_direct_chunk(&mut slots, missing, &chosen) {
                        slots.invalidate_transaction();
                        return Err(error);
                    }
                } else {
                    let mut destinations = Vec::new();
                    for bank in &slots.banks {
                        destinations.extend(bank.arrays().iter().map(|a| a.as_raw_ptr()));
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
        (0..layers).all(|layer| {
            self.slots.get(&layer).is_some_and(|slots| {
                !slots.suspended && slots.occupants.first().is_some_and(Option::is_some)
            })
        })
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
                !slots.suspended
                    && slots.occupants.first().is_some_and(Option::is_some)
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
        let Some(slots) = self.slots.get_mut(&layer).filter(|s| !s.suspended) else {
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
        let entries: Vec<_> = layers.iter().copied().zip(ids.chunks_exact(top)).collect();
        self.commit_device_route_entries(&entries)
    }

    /// `commit_device_routes` over entries of their own lengths: a prefill
    /// group's tape holds one entry per (layer, window) and its last window
    /// may be shorter.
    pub(in super::super) fn commit_device_route_entries(
        &mut self,
        entries: &[(usize, &[u32])],
    ) -> Result<bool> {
        if entries.iter().any(|(_, selected)| selected.is_empty()) {
            return Err(err("Qwen4 device route tape has an invalid shape"));
        }
        if self.plan.mapped {
            for &(layer, selected) in entries {
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
        for &(layer, selected) in entries {
            let Some(slots) = self.slots.get(&layer).filter(|s| !s.suspended) else {
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
        for &(layer, selected) in entries {
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
                    storage: None,
                    direct: false,
                    suspended: false,
                    purge_requested: false,
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
                storage: None,
                direct: false,
                suspended: false,
                purge_requested: false,
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
                storage: None,
                direct: false,
                suspended: false,
                purge_requested: false,
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
    const DRAFT_HEADS: [(&str, i64, i64); 2] = [
        ("mtp.layers.0.mlp.experts.gate_up_proj", 512 * 1280, 2560),
        ("mtp.layers.0.mlp.experts.down_proj", 512 * 2560, 640),
    ];
    const FIXED_BANK_BYTES: u64 = 5_245_160_960;
    const TEXT_ONLY_SLOTS_6_TO_70_GIB: [usize; 65] = [
        0, 11, 18, 25, 32, 39, 45, 52, 59, 66, 73, 80, 86, 93, 100, 107, 114, 120, 127, 134, 141,
        148, 154, 161, 168, 175, 182, 189, 195, 202, 209, 216, 223, 229, 236, 243, 250, 257, 264,
        270, 277, 284, 291, 298, 304, 311, 318, 325, 332, 339, 345, 352, 359, 366, 373, 379, 386,
        393, 400, 407, 413, 420, 427, 434, 441,
    ];

    /// The released UD-Q4_K_XL checkpoint's routed experts (or one GGUF type
    /// for all of them), its fixed bank bytes and the MTP head, on the slot
    /// path, with each layer's routed bytes.
    fn draft_bank_store(uniform: Option<GgufTensorType>) -> Option<(Store, Vec<u64>)> {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return None;
        }
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        store.tensors.clear();
        store.gguf = true;
        store
            .metadata
            .insert("qwen4exp.expert_count".into(), GgufMetaValue::Uint32(512));
        let mut layers = Vec::new();
        for layer in 0..48 {
            let (gate_up, down) = match (uniform, layer) {
                (Some(t), _) => (t, t),
                (None, 2) => (GgufTensorType::Q5K, GgufTensorType::Q8_0),
                (None, 4 | 30 | 46 | 47) => (GgufTensorType::Q4K, GgufTensorType::Q8_0),
                (None, _) => (GgufTensorType::Q4K, GgufTensorType::Q5_1),
            };
            let mut bytes = 0;
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
                    encoding: Encoding::Gguf(if part == "down" { down } else { gate_up }),
                };
                bytes += t.runtime_bytes().unwrap();
                store
                    .tensors
                    .insert(format!("blk.{layer}.ffn_{part}_exps.weight"), t);
            }
            layers.push(bytes);
        }
        store.plan.hot_bytes = layers.iter().sum::<u64>() + FIXED_BANK_BYTES;
        store.plan.mapped = false;
        store.draft_bank_budget = 0;
        for (name, rows, width) in DRAFT_HEADS {
            store.tensors.insert(
                name.into(),
                Tensor {
                    path: PathBuf::new(),
                    shape: vec![512, rows as usize / 512, width as usize],
                    offset: 0,
                    bytes: rows as u64 * width as u64 * 2,
                    encoding: Encoding::Safe("BF16".into()),
                },
            );
        }
        Some((store, layers))
    }

    /// The draft banks in the layout the admission takes, never evaluated.
    fn install_draft_banks(store: &mut Store) -> u64 {
        let specs = store
            .mtp_bank_admission()
            .unwrap()
            .expect("the admission takes the draft banks");
        for (name, _, rows, width, sides, _) in specs {
            let (rows, width, sides) = (rows as i64, width as i64, sides.unwrap());
            let lazy = |columns: i64, dtype| MxArray::zeros(&[rows, columns], Some(dtype)).unwrap();
            store.banks.insert(
                name.into(),
                Arc::new(Weight {
                    values: lazy(width / 4, DType::Uint32),
                    dense_bf16: None,
                    scales: Some(lazy(width / 32, sides)),
                    biases: Some(lazy(width / 32, sides)),
                    group: 32,
                    bits: 8,
                    mode: "affine".into(),
                }),
            );
        }
        let draft = store.mtp_bank_bytes().unwrap();
        store.bank_bytes += draft;
        store.cache_bytes += draft;
        draft
    }

    /// One pass's slot-bank builds at `capacity`: the eviction `expert_slots`
    /// runs before each layer's bank, with no LRU row to give way.
    fn build_slot_banks(store: &mut Store, layers: &[u64], capacity: usize) -> Result<()> {
        for &layer in layers {
            let bytes = layer * capacity as u64 / 512;
            store.evict_to(store.cache_limit.saturating_sub(bytes + MAX_READ_BYTES * 3))?;
            store.slot_bytes += bytes;
            store.cache_bytes += bytes;
        }
        Ok(())
    }

    /// A mapped MTP turn installs the Q8 draft banks; a trip then sizes the
    /// slots from the fallback budget while they stay pinned. Every layer's
    /// slot bank is built with only LRU rows evictable, so the last one needs
    /// the banks, all slots and its staging inside the budget.
    #[test]
    fn fallback_slots_fit_beside_installed_draft_banks() {
        let Some((mut store, layers)) = draft_bank_store(None) else {
            return;
        };
        let routed: u64 = layers.iter().sum();
        assert_eq!(routed, 78_538_342_400);
        let fixed = FIXED_BANK_BYTES;
        store.cache_limit = 36 << 30;
        let draft = store.mtp_bank_estimate().unwrap();
        assert_eq!(draft, 3_040_870_400, "the f32-sided gate_up bank");
        let slots = |store: &mut Store, gib: u64| {
            store.cache_limit = gib << 30;
            store.slot_capacity = None;
            store.expert_slot_capacity(512, 10).unwrap()
        };
        let budgets = [10u64, 16, 20, 26, 30, 36];
        let text = budgets.map(|gib| slots(&mut store, gib));
        assert_eq!(
            [text[1], text[3]],
            [63, 125],
            "no installed draft bank: the slot count is unchanged"
        );
        store.bank_bytes = fixed;
        store.cache_bytes = fixed;
        assert_eq!(install_draft_banks(&mut store), draft);
        let banked = budgets.map(|gib| slots(&mut store, gib));
        let table =
            format!("slots at {budgets:?} GiB: {banked:?} with the banks, {text:?} without");
        for (k, gib) in budgets.into_iter().enumerate() {
            assert!(
                store.bank_bytes + routed * banked[k] as u64 / 512 + MAX_READ_BYTES * 3
                    <= gib << 30,
                "the slots overrun {gib} GiB beside the draft banks; {table}"
            );
            if gib >= 26 {
                assert_eq!(banked[k], text[k], "{table}");
            }
        }
        assert_eq!(store.bytes_read, 0);
        assert_eq!(store.slot_bytes, 0);
    }

    /// `prepare_hot` caches the slot count before the load admits the draft
    /// banks, and the first turn's pass builds every layer's slot bank beside
    /// them. Every 1/16 GiB from 6 to 70 GiB, on an all-Q5_1 geometry
    /// and the released one: no layer runs out of staging, and the banks are
    /// still admitted at every whole budget from 22 GiB.
    #[test]
    fn draft_banks_admitted_after_the_cached_slot_count_leave_every_layer_staging() {
        for (uniform, at_22) in [(Some(GgufTensorType::Q5_1), 85), (None, 98)] {
            let Some((mut store, layers)) = draft_bank_store(uniform) else {
                return;
            };
            let mut admitted = Vec::new();
            for sixteenths in 6 * 16..=70 * 16u64 {
                let limit = sixteenths << 26;
                store.banks.retain(|name, _| !name.starts_with("mtp."));
                store.bank_bytes = FIXED_BANK_BYTES;
                store.cache_bytes = FIXED_BANK_BYTES;
                store.slot_bytes = 0;
                store.slot_capacity = None;
                store.mtp_banks_declined = false;
                store.cache_limit = limit;
                let capacity = store.expert_slot_capacity(512, 10).unwrap();
                let fits = store.mtp_bank_admission().unwrap().is_some();
                if fits {
                    install_draft_banks(&mut store);
                }
                let gib = limit as f64 / (1u64 << 30) as f64;
                if capacity > 0 {
                    build_slot_banks(&mut store, &layers, capacity).unwrap_or_else(|e| {
                        panic!("{uniform:?} at {gib} GiB, {capacity} slots, banks {fits}: {e}")
                    });
                }
                assert!(
                    store.cache_bytes + MAX_READ_BYTES * 3 <= limit,
                    "{uniform:?} at {gib} GiB"
                );
                if limit == 22 << 30 {
                    assert_eq!(
                        (capacity, fits),
                        (at_22, true),
                        "{uniform:?} at 22 GiB: (slots, draft banks admitted)"
                    );
                }
                if fits && limit % (1 << 30) == 0 {
                    admitted.push(limit >> 30);
                }
            }
            assert_eq!(admitted, (22..=70).collect::<Vec<_>>(), "{uniform:?}");
            assert_eq!(store.bytes_read, 0);
        }
    }

    /// The draft reserve costs slots only at budgets whose admission will take
    /// the banks; a decline gives them back, and a text-only load sizes
    /// exactly as before.
    #[test]
    fn slot_count_reserves_only_draft_banks_the_admission_takes() {
        let Some((mut store, _)) = draft_bank_store(None) else {
            return;
        };
        let slots = |store: &mut Store, gib: u64| {
            store.cache_limit = gib << 30;
            store.slot_capacity = None;
            store.expert_slot_capacity(512, 10).unwrap()
        };
        let mut text_only = draft_bank_store(None).unwrap().0;
        text_only
            .tensors
            .retain(|name, _| !name.starts_with("mtp."));
        assert_eq!(text_only.mtp_bank_estimate().unwrap(), 0);
        let text: Vec<usize> = (6..=70).map(|gib| slots(&mut text_only, gib)).collect();
        assert_eq!(text, TEXT_ONLY_SLOTS_6_TO_70_GIB, "no MTP head");
        let budgets = [21u64, 22, 23, 24, 26, 30];
        assert_eq!(
            budgets.map(|gib| slots(&mut store, gib)),
            [94, 98, 104, 111, 125, 149],
            "MTP head attached"
        );
        store.mtp_banks_declined = true;
        assert_eq!(
            budgets.map(|gib| slots(&mut store, gib)),
            [94, 100, 106, 112, 125, 149],
            "draft banks declined"
        );
    }

    /// A slot count cached with no room for the draft banks beside it: the
    /// admission declines them before reading a byte, the decline sticks, and
    /// every layer still builds. Half the layers already built: the same.
    #[test]
    fn draft_bank_admission_declines_beside_a_cached_count_without_staging() {
        let Some((mut store, layers)) = draft_bank_store(Some(GgufTensorType::Q5_1)) else {
            return;
        };
        let routed: u64 = layers.iter().sum();
        let draft = store.mtp_bank_estimate().unwrap();
        store.cache_limit = 22 << 30;
        assert!(draft <= store.auxiliary_cache_allowance().unwrap());
        // The count sized without the draft reserve.
        let capacity = 87;
        assert!(
            FIXED_BANK_BYTES + routed * capacity as u64 / 512 + draft + MAX_READ_BYTES * 3
                > store.cache_limit
        );
        for built in [0, 24] {
            store.bank_bytes = FIXED_BANK_BYTES;
            store.cache_bytes = FIXED_BANK_BYTES;
            store.slot_bytes = 0;
            store.slot_capacity = Some(capacity);
            store.mtp_banks_declined = false;
            build_slot_banks(&mut store, &layers[..built], capacity).unwrap();
            assert!(
                store.mtp_bank_admission().unwrap().is_none(),
                "{built} layers built: the draft banks fit beside {capacity} slots"
            );
            let admitted = store
                .prepare_mtp_expert_banks_with(&mut |_: u64| -> Result<()> {
                    panic!("the draft banks were admitted")
                })
                .unwrap();
            assert_eq!(admitted, 0);
            assert!(store.mtp_banks_declined);
            assert_eq!(store.mtp_bank_bytes().unwrap(), 0);
            assert_eq!(
                store
                    .prepare_mtp_expert_banks_with(&mut |_| Ok(()))
                    .unwrap(),
                0,
                "a declined build must not retry"
            );
            build_slot_banks(&mut store, &layers[built..], capacity).unwrap();
        }
        store.bank_bytes = FIXED_BANK_BYTES;
        store.cache_bytes = FIXED_BANK_BYTES;
        store.slot_bytes = 0;
        store.slot_capacity = Some(85);
        store.mtp_banks_declined = false;
        assert!(
            store.mtp_bank_admission().unwrap().is_some(),
            "85 slots leave the staging"
        );
        assert_eq!(store.bytes_read, 0);
    }

    /// The gate_up draft bank takes f32 sides (+0.2 GiB) only where they fit
    /// and keeps the f16 layout otherwise, never declining a bank the f16
    /// build would take: below a 22.66 GiB slot allowance, beside a mapped
    /// plan that charged the f16 layout, or with only the f16 staging left.
    /// The reserve never holds less than the build.
    /// MUTATIONS: only the f32 layout -> 22 GiB; the slot allowance beside a
    /// mapped charge -> the f16 charge; no f16 retry -> the f16 staging.
    #[test]
    fn draft_banks_take_f32_sides_only_where_they_fit() {
        let Some((mut store, _)) = draft_bank_store(None) else {
            return;
        };
        const F16_BANKS: u64 = 2_831_155_200;
        const F32_GATE_UP: u64 = 3_040_870_400;
        let layout = |store: &Store| {
            store.mtp_bank_admission().unwrap().map(|specs| {
                let bytes: u64 = specs.iter().map(|spec| spec.5).sum();
                (
                    bytes,
                    specs.iter().map(|spec| spec.4.unwrap()).collect::<Vec<_>>(),
                )
            })
        };
        let f16 = Some((F16_BANKS, vec![DType::Float16; 2]));
        let f32 = Some((F32_GATE_UP, vec![DType::Float32, DType::Float16]));
        store.bank_bytes = FIXED_BANK_BYTES;
        store.cache_bytes = FIXED_BANK_BYTES;
        // cache_limit / 8 holds the f32 layout from 22.66 GiB.
        for (sixteenths, want, built) in [
            (22 * 16, F16_BANKS, &f16),
            (22 * 16 + 10, F16_BANKS, &f16),
            (22 * 16 + 11, F32_GATE_UP, &f32),
            (23 * 16, F32_GATE_UP, &f32),
        ] {
            store.cache_limit = sixteenths << 26;
            let gib = sixteenths as f64 / 16.;
            assert_eq!(store.mtp_bank_estimate().unwrap(), want, "{gib} GiB");
            assert_eq!(&layout(&store), built, "{gib} GiB");
        }
        store.cache_limit = 40 << 30;
        for (charged, built) in [(F16_BANKS, &f16), (F32_GATE_UP, &f32)] {
            store.draft_bank_budget = charged;
            let why = format!("a mapped plan charged {charged}");
            assert_eq!(store.mtp_bank_estimate().unwrap(), charged, "{why}");
            assert_eq!(&layout(&store), built, "{why}");
        }
        store.draft_bank_budget = 0;
        // Pinned slot banks leave two copies of the f16 gate_up bank and the
        // staging reads: the reserve holds the f32 build, the admission
        // takes the f16 one.
        store.cache_limit = 30 << 30;
        store.slot_bytes =
            store.cache_limit - FIXED_BANK_BYTES - 2 * 1_887_436_800 - MAX_READ_BYTES * 3;
        assert_eq!(store.mtp_bank_estimate().unwrap(), F32_GATE_UP);
        assert_eq!(layout(&store), f16, "the f16 staging");
        store.slot_bytes += 1;
        assert_eq!(layout(&store), None);
        assert_eq!(store.bytes_read, 0);
    }
    /// The same fallback on the released checkpoint: a mapped load with the
    /// MTP head attached installs the draft banks (the load's call), a trip
    /// leaves mapped mode on a 128 GiB host snapshot that leaves a 16 GiB slot
    /// budget, and the next turn builds every layer's slot bank. Opt-in:
    /// `QWEN4_EXP_REAL_MODEL=<gguf> QWEN4_EXP_REAL_AUX=<MTP checkpoint dir>
    /// QWEN4_EXP_REAL_INPUTS=<inputs.json> cargo test -p mlx-core --lib
    /// real_fallback_slots_beside_draft_banks -- --ignored --nocapture`
    #[test]
    #[ignore = "real checkpoint: QWEN4_EXP_REAL_MODEL, QWEN4_EXP_REAL_INPUTS, QWEN4_EXP_REAL_AUX; --test-threads=1"]
    fn real_fallback_slots_beside_draft_banks() {
        use super::super::super::tests::{argmax, go_live, real_prompt};
        use super::super::super::{auxiliary, decoder::Decoder, gguf, memory::HostMemory};
        let model = std::env::var("QWEN4_EXP_REAL_MODEL").expect("QWEN4_EXP_REAL_MODEL");
        let aux = std::env::var("QWEN4_EXP_REAL_AUX").expect("QWEN4_EXP_REAL_AUX");
        let inputs = std::env::var("QWEN4_EXP_REAL_INPUTS").expect("QWEN4_EXP_REAL_INPUTS");
        let mut prompt = real_prompt(Path::new(&inputs), "6k");
        prompt.truncate(8);
        let mut store = Store::open(Path::new(&model)).unwrap();
        let mut config = gguf::config(&store).unwrap();
        auxiliary::attach(&mut store, &mut config, Path::new(&aux)).unwrap();
        let mut decoder = Decoder::new(config, store).unwrap();
        go_live(&mut decoder, 0);
        assert!(
            decoder.weights.plan.mapped,
            "the live host refused the pack"
        );
        let draft = decoder.weights.prepare_mtp_expert_banks().unwrap();
        assert_eq!(
            draft, 3_040_870_400,
            "the mapped plan admits the draft banks with f32 gate_up sides"
        );
        let turn = |decoder: &mut Decoder| -> Result<Vec<u32>> {
            decoder.reset();
            decoder.weights.begin_turn()?;
            let mut logits = decoder.prefill_chunk(&prompt, None, true)?;
            let mut text = Vec::new();
            for _ in 0..16 {
                let id = argmax(&logits);
                text.push(id);
                logits = decoder.step(id)?;
            }
            Ok(text)
        };
        let mapped = turn(&mut decoder).unwrap();
        assert!(
            decoder.weights.plan.mapped,
            "the mapped turn tripped by itself"
        );
        let host = HostMemory {
            physical: 128 << 30,
            free: 36 << 30,
            ..Default::default()
        };
        decoder
            .weights
            .leave_mapped_with(
                &Fallback::SsdReads { read: 0, mapped: 0 },
                &mut || Ok(host),
                &mut |_| {},
            )
            .unwrap();
        decoder.residency_changed();
        assert_eq!(decoder.weights.cache_limit, 16 << 30);
        let slots = decoder.weights.plan.partial_expert_slots;
        let held = decoder.weights.cache_bytes;
        eprintln!(
            "fallback: {slots} slots, 16 GiB budget, {:.3} GiB banks ({:.3} GiB draft), {:.3} GiB held",
            decoder.weights.bank_bytes as f64 / (1u64 << 30) as f64,
            decoder.weights.mtp_bank_bytes().unwrap() as f64 / (1u64 << 30) as f64,
            held as f64 / (1u64 << 30) as f64
        );
        let fallback = turn(&mut decoder)
            .unwrap_or_else(|e| panic!("the slot turn failed at {slots} slots: {e}"));
        assert_eq!(
            decoder.weights.slots.len(),
            48,
            "every layer built its slots"
        );
        assert!(decoder.weights.cache_bytes <= decoder.weights.cache_limit);
        assert_eq!(mapped, fallback, "the slot turn gives the mapped text");
        eprintln!(
            "slot bytes {:.3} GiB, cache {:.3} GiB",
            decoder.weights.slot_bytes as f64 / (1u64 << 30) as f64,
            decoder.weights.cache_bytes as f64 / (1u64 << 30) as f64
        );
    }
    /// The released checkpoint loaded on the slot path at a forced budget:
    /// `prepare_hot` caches the slot count, the load's call admits
    /// the draft banks, and a turn builds every layer's slot bank beside
    /// them. 22 GiB is a budget the product picks; at 22.25 GiB a count
    /// cached without the draft reserve overruns the last layer's staging.
    /// Opt-in, the same variables as `real_fallback_slots_beside_draft_banks`:
    /// `cargo test -p mlx-core --lib real_slot_load_admits_draft_banks_beside_the_cached_count
    /// -- --ignored --nocapture`
    #[test]
    #[ignore = "real checkpoint: QWEN4_EXP_REAL_MODEL, QWEN4_EXP_REAL_INPUTS, QWEN4_EXP_REAL_AUX; --test-threads=1"]
    fn real_slot_load_admits_draft_banks_beside_the_cached_count() {
        use super::super::super::tests::{argmax, real_prompt};
        use super::super::super::{auxiliary, decoder::Decoder, gguf};
        let model = std::env::var("QWEN4_EXP_REAL_MODEL").expect("QWEN4_EXP_REAL_MODEL");
        let aux = std::env::var("QWEN4_EXP_REAL_AUX").expect("QWEN4_EXP_REAL_AUX");
        let inputs = std::env::var("QWEN4_EXP_REAL_INPUTS").expect("QWEN4_EXP_REAL_INPUTS");
        let mut prompt = real_prompt(Path::new(&inputs), "6k");
        prompt.truncate(8);
        let mut first: Option<Vec<u32>> = None;
        for limit in [22u64 << 30, (89u64 << 30) / 4] {
            let gib = limit as f64 / (1u64 << 30) as f64;
            let mut store = Store::open(Path::new(&model)).unwrap();
            let mut config = gguf::config(&store).unwrap();
            auxiliary::attach(&mut store, &mut config, Path::new(&aux)).unwrap();
            let mut decoder = Decoder::new(config, store).unwrap();
            decoder.weights.set_cache_limit(limit);
            decoder.register_cache_limit();
            decoder.weights.prepare_fixture_hot().unwrap();
            decoder.weights.publish_live();
            decoder.residency_changed();
            assert!(!decoder.weights.plan.mapped);
            assert_eq!(decoder.weights.cache_limit, limit);
            let slots = decoder.weights.plan.partial_expert_slots;
            assert_eq!(decoder.weights.slot_capacity, Some(slots));
            let draft = decoder.weights.prepare_mtp_expert_banks().unwrap();
            eprintln!(
                "{gib} GiB: {slots} slots cached, draft banks {:.3} GiB, banks {:.3} GiB",
                draft as f64 / (1u64 << 30) as f64,
                decoder.weights.bank_bytes as f64 / (1u64 << 30) as f64
            );
            assert_eq!(draft, 2_831_155_200, "{gib} GiB admits the draft banks");
            decoder.reset();
            decoder.weights.begin_turn().unwrap();
            let mut logits = decoder
                .prefill_chunk(&prompt, None, true)
                .unwrap_or_else(|e| {
                    panic!("the slot turn failed at {slots} slots, {gib} GiB: {e}")
                });
            let mut text = Vec::new();
            for _ in 0..16 {
                let id = argmax(&logits);
                text.push(id);
                logits = decoder.step(id).unwrap();
            }
            assert_eq!(
                decoder.weights.slots.len(),
                48,
                "every layer built its slots"
            );
            assert!(decoder.weights.cache_bytes <= decoder.weights.cache_limit);
            match &first {
                None => first = Some(text),
                Some(first) => assert_eq!(&text, first, "{gib} GiB gives the 22 GiB text"),
            }
            eprintln!(
                "{gib} GiB: slot bytes {:.3} GiB, cache {:.3} GiB, peak {:.3} GiB",
                decoder.weights.slot_bytes as f64 / (1u64 << 30) as f64,
                decoder.weights.cache_bytes as f64 / (1u64 << 30) as f64,
                decoder.weights.peak_cache_bytes as f64 / (1u64 << 30) as f64
            );
            drop(decoder);
            crate::array::memory::synchronize_and_clear_cache();
        }
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
