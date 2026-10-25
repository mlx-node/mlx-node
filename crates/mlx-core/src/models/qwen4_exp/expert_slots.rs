//! Stable, quantized expert slots for partial residency. Slot buffers are
//! mutated only after their last lazy reader has completed. The ordinary LRU
//! supplies bounded misses and releases imported chunks after the slot copy.
use super::*;
use crate::array::alloc::{OutLayout, SendPtr};
use crate::array::purgeable::{PurgeableBank, PurgeableState};
use crate::utils::gguf_gpu::{
    GpuRepackKind, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available, gpu_repack_band,
};
use crate::utils::gguf_read::{
    ALIGN, PageCacheMode, READ_THREADS, Reader, Window, plan_reads, set_nocache,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DIRECT_READ_BYTES: usize = 8 << 20;
const DIRECT_STAGING_SLOTS: usize = 4;
pub(in super::super) const DIRECT_STAGING_BYTES: u64 =
    (DIRECT_READ_BYTES * DIRECT_STAGING_SLOTS) as u64;
const SLOT_RECLAIM_FALLBACK: Duration = Duration::from_secs(30);

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
    fn suspended_bank_bytes(&self) -> Result<u64> {
        if !self.suspended {
            return Ok(0);
        }
        let Some(storage) = &self.storage else {
            return Ok(0);
        };
        storage.iter().try_fold(0u64, |total, slot| {
            slot.storage.iter().try_fold(total, |total, bank| {
                total
                    .checked_add(bank.nbytes() as u64)
                    .ok_or_else(|| err("Qwen4 suspended bank size overflow"))
            })
        })
    }

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

    fn identity_ready(&self) -> bool {
        self.direct
            && !self.suspended
            && self.occupants.len() == self.mapping.len()
            && self
                .occupants
                .iter()
                .enumerate()
                .all(|(slot, expert)| *expert == Some(slot as u32))
            && self
                .mapping
                .iter()
                .enumerate()
                .all(|(expert, slot)| *slot == Some(expert))
    }

    fn publish_identity(&mut self) {
        for slot in 0..self.occupants.len() {
            self.occupants[slot] = Some(slot as u32);
            self.mapping[slot] = Some(slot);
        }
        self.device_mapping = None;
        self.ages.fill(0);
        self.tick = 0;
    }

    /// The fill worker mutates lazy output handles, never the store's slot
    /// bookkeeping or purge state. All handles share the same bank storage.
    fn fill_clone(&self) -> Self {
        Self {
            banks: self.banks.clone(),
            storage: None,
            direct: self.direct,
            suspended: false,
            purge_requested: false,
            names: self.names.clone(),
            rows: self.rows,
            occupants: self.occupants.clone(),
            mapping: self.mapping.clone(),
            device_mapping: None,
            ages: self.ages.clone(),
            tick: self.tick,
            readers: Vec::new(),
        }
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
    source: &'r DirectGgufSource,
    requests: &'r [DirectRequest],
    windows: &'r [Window],
    staging: &'r [(MxArray, SendPtr<u8>)],
    slots: &'r mut ExpertSlots,
    stop: Option<&'r AtomicBool>,
    rows: Option<&'r AtomicUsize>,
}

impl DirectRun<'_, '_> {
    fn admission(&self) -> Result<()> {
        while self
            .rows
            .is_some_and(|rows| rows.load(Ordering::Acquire) > 0)
        {
            if self.stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
                return Err(err("Qwen4 direct GGUF fill stopped"));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        if self.stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
            return Err(err("Qwen4 direct GGUF fill stopped"));
        }
        Ok(())
    }

    fn submit(&self, w: usize) -> Result<()> {
        self.admission()?;
        let (_, ptr) = &self.staging[w % self.staging.len()];
        let window = &self.windows[w];
        self.reader
            .submit(w as u64, *ptr, window.len, window.file_off);
        Ok(())
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
            self.submit(w)?;
        }
        let mut pending: Vec<MxArray> = Vec::new();
        for w in 0..n {
            if w > 0 {
                self.complete(w - 1, &std::mem::take(&mut pending))?;
            }
            let next = w + slots - 1;
            if next < n {
                self.submit(next)?;
            }
            let read = self.reader.wait(w as u64);
            if self.source.shards.len() == 1 {
                read?;
            } else {
                let window = &self.windows[w];
                let shard = self
                    .source
                    .shards
                    .iter()
                    .find(|shard| {
                        window.file_off >= shard.base && window.file_off < shard.base + shard.len
                    })
                    .or_else(|| self.source.shards.last())
                    .ok_or_else(|| err("Qwen4 direct GGUF window has no source shard"))?;
                read.map_err(|error| {
                    err(format!("GGUF shard '{}': {error}", shard.path.display()))
                })?;
            }
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

#[derive(Default)]
struct DirectFillStats {
    source_bytes: u64,
    bank_bytes: u64,
    slow_source: bool,
    #[cfg(test)]
    windows: Vec<(u64, usize)>,
}

struct DirectLayerJob {
    layer: usize,
    descriptors: [Tensor; 3],
    slots: ExpertSlots,
}

struct DirectFillMessage {
    layer: usize,
    stats: DirectFillStats,
    error: Option<String>,
}

/// One bounded full-residency fill. The worker owns only cloned array handles;
/// the store keeps every slot and publishes identity mappings as layers finish.
pub(super) struct DirectFill {
    receiver: mpsc::Receiver<DirectFillMessage>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl DirectFill {
    fn start(
        jobs: Vec<DirectLayerJob>,
        source: DirectGgufSource,
        rows: Arc<AtomicUsize>,
        live: Arc<super::super::live::Live>,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let total_source = jobs.iter().fold(0u64, |total, job| {
            total.saturating_add(job.descriptors.iter().map(|t| t.bytes).sum::<u64>())
        });
        let first_layer = jobs.first().map_or(0, |job| job.layer);
        live.direct_fill_in_progress.store(true, Ordering::Release);
        let worker_live = Arc::clone(&live);
        let spawned = std::thread::Builder::new()
            .name("mlx-qwen4-direct-fill".into())
            .spawn(move || {
                let run = || {
                    let files = match source.open_files() {
                        Ok(files) => files,
                        Err(error) => {
                            let _ = sender.send(DirectFillMessage {
                                layer: first_layer,
                                stats: DirectFillStats::default(),
                                error: Some(error.to_string()),
                            });
                            return;
                        }
                    };
                    let cache = match PageCacheMode::from_env() {
                        Ok(mode) => mode,
                        Err(error) => {
                            let _ = sender.send(DirectFillMessage {
                                layer: first_layer,
                                stats: DirectFillStats::default(),
                                error: Some(error.to_string()),
                            });
                            return;
                        }
                    };
                    if cache.nocache(total_source, crate::utils::host::physical_memory()) {
                        for (shard, file) in source.shards.iter().zip(&files) {
                            if let Err(error) = set_nocache(file) {
                                let _ = sender.send(DirectFillMessage {
                                    layer: first_layer,
                                    stats: DirectFillStats::default(),
                                    error: Some(format!(
                                        "Failed to set GGUF page-cache policy on '{}': {error}",
                                        shard.path.display()
                                    )),
                                });
                                return;
                            }
                        }
                    }
                    let file_refs: Vec<&File> = files.iter().collect();
                    let started = Instant::now();
                    let mut filled_source = 0u64;
                    let mut slow = false;
                    for mut job in jobs {
                        if worker_stop.load(Ordering::Acquire) {
                            return;
                        }
                        let layer = job.layer;
                        let experts: Vec<u32> = (0..job.slots.occupants.len() as u32).collect();
                        let chosen: Vec<usize> = (0..experts.len()).collect();
                        let result = fill_direct(
                            &source,
                            &file_refs,
                            &job.descriptors,
                            &mut job.slots,
                            &experts,
                            &chosen,
                            Some(&worker_stop),
                            Some(&rows),
                        );
                        let mut stats = DirectFillStats::default();
                        let mut error = None;
                        match result {
                            Ok(filled) => {
                                stats = filled;
                                filled_source = filled_source.saturating_add(stats.source_bytes);
                                if !slow && filled_source >= (1 << 30) {
                                    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
                                    let rate = filled_source as f64 / elapsed;
                                    if rate < 5_000_000_000.0 {
                                        slow = true;
                                        tracing::warn!(
                                            "qwen4 direct GGUF: slow source ({:.2} GB/s first GiB), estimated full fill {:.1} s; banks stay resident at idle",
                                            rate / 1_000_000_000.0,
                                            total_source as f64 / rate,
                                        );
                                    }
                                }
                                stats.slow_source = slow;
                                worker_live.direct_filled_layers.fetch_add(1, Ordering::AcqRel);
                            }
                            Err(failure) => {
                                stats.slow_source = slow;
                                if worker_stop.load(Ordering::Acquire) {
                                    return;
                                }
                                error = Some(failure.to_string());
                            }
                        }
                        let failed = error.is_some();
                        if sender
                            .send(DirectFillMessage {
                                layer,
                                stats,
                                error,
                            })
                            .is_err()
                        {
                            return;
                        }
                        if failed {
                            return;
                        }
                    }
                };
                run();
                worker_live
                    .direct_fill_in_progress
                    .store(false, Ordering::Release);
            });
        let worker = match spawned {
            Ok(worker) => worker,
            Err(error) => {
                live.direct_fill_in_progress.store(false, Ordering::Release);
                return Err(err(error));
            }
        };
        Ok(Self {
            receiver,
            stop,
            worker: Some(worker),
        })
    }
}

impl Drop for DirectFill {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Fill selected slots through the shared bounded reader/repack pipeline.
fn fill_direct(
    source: &DirectGgufSource,
    files: &[&File],
    descriptors: &[Tensor; 3],
    slots: &mut ExpertSlots,
    experts: &[u32],
    chosen: &[usize],
    stop: Option<&AtomicBool>,
    pending_rows: Option<&AtomicUsize>,
) -> Result<DirectFillStats> {
    if experts.len() != chosen.len() || experts.is_empty() {
        return Err(err("Qwen4 direct GGUF fill has an invalid expert batch"));
    }
    source.verify_path()?;
    source.verify(files)?;
    let mut requests = Vec::with_capacity(experts.len() * 3);
    let mut infos = Vec::with_capacity(experts.len() * 3);
    let mut stats = DirectFillStats::default();
    for (&expert, &slot) in experts.iter().zip(chosen) {
        for (part, tensor) in descriptors.iter().enumerate() {
            let (ty, kind) = tensor
                .direct_kind()
                .ok_or_else(|| err("Qwen4 direct GGUF fill lost its repack route"))?;
            let rows = slots.rows[part];
            let start = expert as usize * rows;
            let (offset, bytes) = tensor.range(start, rows)?;
            let shard = source
                .shards
                .iter()
                .find(|shard| shard.path == tensor.path)
                .ok_or_else(|| {
                    err(format!(
                        "Qwen4 direct GGUF tensor '{}' has no source shard",
                        slots.names[part]
                    ))
                })?;
            let offset = shard
                .base
                .checked_add(offset)
                .ok_or_else(|| err("Qwen4 direct GGUF logical tensor offset overflow"))?;
            stats.source_bytes = stats.source_bytes.saturating_add(bytes);
            stats.bank_bytes = stats
                .bank_bytes
                .saturating_add(tensor.runtime_bytes()? / tensor.shape[0] as u64);
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
    let read_at = DirectGgufReader::new(source, files)?;
    let result = std::thread::scope(|scope| {
        let reader = Reader::start(
            scope,
            &read_at,
            &source.shards[0].path,
            READ_THREADS,
            DIRECT_READ_BYTES,
        );
        let mut run = DirectRun {
            reader: &reader,
            source,
            requests: &requests,
            windows: &windows,
            staging: &staging,
            slots,
            stop,
            rows: pending_rows,
        };
        run.all()
    });
    // A failed window may already have encoded earlier bands. Finish them
    // before the caller unpublishes every destination in this transaction.
    crate::array::memory::synchronize();
    result?;
    source.verify(files)?;
    #[cfg(test)]
    stats
        .windows
        .extend(windows.iter().map(|w| (w.file_off, w.len)));
    Ok(stats)
}

impl Store {
    fn account_direct_fill(&mut self, stats: &DirectFillStats) {
        #[cfg(test)]
        self.direct_read_windows
            .extend(stats.windows.iter().copied());
        self.bytes_read += stats.source_bytes;
        self.slot_upload_bytes += stats.bank_bytes;
        self.direct_slow_source |= stats.slow_source;
    }

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
        let descriptors = slots
            .names
            .each_ref()
            .map(|name| self.descriptor(name).cloned());
        let [d0, d1, d2] = descriptors;
        let descriptors = [d0?, d1?, d2?];
        source.verify_path()?;
        let files = source
            .shards
            .iter()
            .map(|shard| {
                self.file(&shard.path).map_err(|error| {
                    err(format!(
                        "Failed to open GGUF '{}': {error}",
                        shard.path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let file_refs: Vec<&File> = files.iter().map(AsRef::as_ref).collect();
        let stats = fill_direct(
            &source,
            &file_refs,
            &descriptors,
            slots,
            experts,
            chosen,
            None,
            None,
        )?;
        self.account_direct_fill(&stats);
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

    fn full_residency(&self) -> bool {
        self.plan.direct
    }

    /// Select full or partial direct GGUF residency. Unsupported sources keep
    /// the existing bounded per-row checkpoint importer.
    pub(in super::super) fn select_residency(&mut self, pool_bytes: u64) -> Result<()> {
        let experts = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0) as usize;
        let direct = self.select_direct_residency_with(
            experts,
            pool_bytes,
            &mut super::super::memory::direct_plan,
            &mut super::super::memory::admit_direct,
        )?;
        if !direct {
            self.slot_capacity = Some(0);
            self.plan.partial_expert_slots = 0;
        }
        Ok(())
    }

    /// Returns true when direct eligibility was decided, including a gate
    /// refusal that must use partial slots. False keeps per-row imports.
    pub(in super::super) fn select_direct_residency_with(
        &mut self,
        experts: usize,
        pool_bytes: u64,
        plan: &mut impl FnMut(&mut super::super::memory::Plan, u64, u64, u64) -> Result<()>,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<bool> {
        if !self.gguf
            || experts == 0
            || !crate::engine::persistence::compiled_forward_backend_available()
            || !self.slots.is_empty()
            || !self.banks.is_empty()
            || self.plan.direct
            || self.gguf_source.is_none()
        {
            return Ok(false);
        }
        let layers: std::collections::BTreeSet<usize> = self
            .tensors
            .keys()
            .filter(|name| !super::residency::is_auxiliary(name))
            .filter_map(|name| super::routed_expert_tensor(name).map(|(layer, _)| layer))
            .collect();
        if layers.is_empty() {
            return Ok(false);
        }
        let names = |layer: usize| {
            ["gate", "up", "down"].map(|part| format!("blk.{layer}.ffn_{part}_exps.weight"))
        };
        if !layers.iter().all(|&layer| {
            let names = names(layer);
            self.direct_layer(&names)
                && names.iter().all(|name| {
                    self.descriptor(name)
                        .is_ok_and(|tensor| tensor.shape[0] == experts)
                })
        }) {
            return Ok(false);
        }
        let mut bank_bytes = 0u64;
        for name in layers.iter().flat_map(|&layer| names(layer)) {
            bank_bytes = bank_bytes
                .checked_add(self.descriptor(&name)?.runtime_bytes()?)
                .ok_or_else(|| err("Qwen4 direct expert bank size overflow"))?;
        }
        let dense = self.fixed_projection_bytes()?;
        let draft_bank = self.mtp_bank_estimate()?;
        let original = self.plan.clone();
        let original_cache_limit = self.cache_limit;
        let original_draft_budget = self.draft_bank_budget;
        let original_slot_capacity = self.slot_capacity;
        let original_slot_bank_budget = self.slot_bank_budget;
        let original_slot_bytes = self.slot_bytes;
        let original_cache_bytes = self.cache_bytes;
        let mut candidate = self.plan.clone();
        if let Err(error) = plan(
            &mut candidate,
            dense.saturating_add(draft_bank),
            pool_bytes,
            bank_bytes,
        ) {
            let bank_budget = candidate.bank_budget;
            self.plan = candidate;
            self.plan.direct = false;
            self.plan.direct_bytes = 0;
            self.plan.direct_layers = 0;
            self.plan.budget = self.plan.budget.saturating_add(bank_budget);
            self.cache_limit = self.plan.budget;
            self.draft_bank_budget = 0;
            self.slot_capacity = None;
            self.slot_bank_budget = Some(bank_budget);
            let top = self
                .metadata
                .get("qwen4exp.expert_used_count")
                .and_then(GgufMetaValue::as_u64)
                .unwrap_or(0) as usize;
            self.plan.partial_expert_slots = self.expert_slot_capacity(experts, top)?;
            self.live.set_fallback(Fallback::LoadGate {
                bank_bytes,
                budget: bank_budget,
            });
            self.live.store(&self.plan);
            tracing::warn!(
                "qwen4 residency: partial direct slots, {error}; {} slots within a {:.1} GiB bank budget",
                self.plan.partial_expert_slots,
                bank_budget as f64 / (1u64 << 30) as f64,
            );
            return Ok(true);
        }
        candidate.direct = true;
        candidate.direct_bytes = bank_bytes;
        candidate.direct_layers = layers.len();
        candidate.partial_expert_slots = experts;
        candidate.budget = candidate
            .budget
            .checked_add(bank_bytes)
            .ok_or_else(|| err("Qwen4 direct residency budget overflow"))?;
        self.plan = candidate;
        self.cache_limit = self.plan.budget;
        self.draft_bank_budget = draft_bank;
        self.slot_capacity = Some(experts);
        self.slot_bank_budget = None;
        let allocated = (|| -> Result<()> {
            admit(bank_bytes.saturating_add((DIRECT_READ_BYTES * DIRECT_STAGING_SLOTS) as u64))?;
            for &layer in &layers {
                let mut no_admit = |_| Ok(());
                let slots = self
                    .new_expert_slots(layer, experts, experts, &mut no_admit)?
                    .ok_or_else(|| err("Qwen4 direct slots require quantized scales"))?;
                self.slots.insert(layer, slots);
            }
            Ok(())
        })();
        if let Err(error) = allocated {
            self.slots.clear();
            self.slot_bytes = original_slot_bytes;
            self.cache_bytes = original_cache_bytes;
            self.plan = original;
            self.cache_limit = original_cache_limit;
            self.draft_bank_budget = original_draft_budget;
            self.slot_capacity = original_slot_capacity;
            self.slot_bank_budget = original_slot_bank_budget;
            return Err(error);
        }
        self.direct_slow_source = false;
        self.direct_error = None;
        self.direct_fill_pass = false;
        self.live.direct_filled_layers.store(0, Ordering::Release);
        self.live.store(&self.plan);
        tracing::info!(
            "qwen4 direct GGUF residency: {} layers x {experts} experts, {:.1} GiB anonymous banks within a {:.1} GiB bank budget ({:.1} GiB live host memory), {:.1} GiB total weight budget",
            layers.len(),
            bank_bytes as f64 / (1u64 << 30) as f64,
            self.plan.bank_budget as f64 / (1u64 << 30) as f64,
            self.plan.live_non_reclaimable as f64 / (1u64 << 30) as f64,
            self.cache_limit as f64 / (1u64 << 30) as f64,
        );
        Ok(true)
    }

    /// Publish the residency the load settled on.
    pub(in super::super) fn publish_live(&self) {
        self.live.store(&self.plan);
    }

    fn accept_direct_message(&mut self, message: DirectFillMessage) {
        self.direct_fill_remaining = self.direct_fill_remaining.saturating_sub(1);
        self.account_direct_fill(&message.stats);
        if let Some(error) = message.error {
            self.direct_error = Some(error);
            return;
        }
        let Some(slots) = self.slots.get_mut(&message.layer) else {
            self.direct_error = Some(format!(
                "Qwen4 direct fill completed unknown layer {}",
                message.layer
            ));
            return;
        };
        slots.publish_identity();
    }

    fn poll_direct_fills(&mut self) {
        loop {
            let Some(fill) = &self.direct_fill else {
                return;
            };
            match fill.receiver.try_recv() {
                Ok(message) => self.accept_direct_message(message),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.direct_fill = None;
                    return;
                }
            }
        }
    }

    /// Start or restart every unpublished direct layer. A previous failed
    /// worker is joined before its invalid layers are queued again.
    pub(in super::super) fn start_direct_fill(&mut self) -> Result<()> {
        if !self.plan.direct {
            return Ok(());
        }
        self.poll_direct_fills();
        if self.direct_fill.is_some()
            && self.direct_error.is_none()
            && self.direct_fill_remaining > 0
        {
            return Ok(());
        }
        self.direct_fill = None;
        self.direct_fill_remaining = 0;
        self.direct_error = None;
        let ready = self
            .slots
            .values()
            .filter(|slots| slots.identity_ready())
            .count();
        self.live
            .direct_filled_layers
            .store(ready as u64, Ordering::Release);
        let mut jobs = Vec::new();
        for (&layer, slots) in &self.slots {
            if slots.identity_ready() {
                continue;
            }
            let descriptors = slots
                .names
                .each_ref()
                .map(|name| self.descriptor(name).cloned());
            let [d0, d1, d2] = descriptors;
            jobs.push(DirectLayerJob {
                layer,
                descriptors: [d0?, d1?, d2?],
                slots: slots.fill_clone(),
            });
        }
        jobs.sort_by_key(|job| job.layer);
        if jobs.is_empty() {
            return Ok(());
        }
        let source = self
            .gguf_source
            .clone()
            .ok_or_else(|| err("Qwen4 direct GGUF fill has no main source"))?;
        self.direct_fill_remaining = jobs.len();
        self.direct_fill = Some(DirectFill::start(
            jobs,
            source,
            Arc::clone(&self.rows_pending),
            Arc::clone(&self.live),
        )?);
        Ok(())
    }

    fn check_activation_refusal(&self) -> Result<()> {
        if let Some(error) = &self.activation_error {
            return Err(err(format!(
                "Qwen4 expert bank activation refused: {error}"
            )));
        }
        Ok(())
    }

    fn wait_direct_layer(&mut self, layer: usize) -> Result<()> {
        self.check_activation_refusal()?;
        if self
            .slots
            .get(&layer)
            .is_some_and(ExpertSlots::identity_ready)
        {
            return Ok(());
        }
        if self.direct_fill.is_none() && self.direct_error.is_none() {
            self.start_direct_fill()?;
        }
        loop {
            if let Some(error) = &self.direct_error {
                return Err(err(format!(
                    "Qwen4 direct GGUF fill failed for layer {layer}: {error}"
                )));
            }
            if self
                .slots
                .get(&layer)
                .is_some_and(ExpertSlots::identity_ready)
            {
                return Ok(());
            }
            let received = {
                let fill = self.direct_fill.as_ref().ok_or_else(|| {
                    err(format!(
                        "Qwen4 direct GGUF layer {layer} has no fill schedule"
                    ))
                })?;
                fill.receiver.recv()
            };
            match received {
                Ok(message) => self.accept_direct_message(message),
                Err(_) => {
                    self.direct_fill = None;
                    if self.direct_error.is_none() {
                        self.direct_error = Some(
                            "Qwen4 direct GGUF fill ended before every layer completed".into(),
                        );
                    }
                }
            }
        }
    }

    pub(in super::super) fn direct_fill_pass(&self) -> bool {
        self.plan.direct && self.direct_fill_pass
    }

    #[cfg(test)]
    pub(in super::super) fn set_direct_fill_pass_for_test(&mut self, active: bool) {
        self.direct_fill_pass = active;
    }

    /// The load settled on its residency and read the dense banks: build the
    /// MTP head's draft banks, then start any direct expert fill. A head the
    /// host cannot stage keeps its drafts on the row reader for the process.
    pub(in super::super) fn finish_load(&mut self) -> Result<()> {
        if self.plan.direct || self.slot_bank_budget.is_some() {
            self.finish_load_with(&mut super::super::memory::admit_direct)
        } else {
            self.finish_load_with(&mut super::super::memory::admit)
        }
    }

    /// `finish_load` without the live host admission (see
    /// `prepare_fixture_mtp_expert_banks`).
    #[cfg(test)]
    pub(in super::super) fn finish_fixture_load(&mut self) -> Result<()> {
        self.finish_load_with(&mut |_| Ok(()))
    }

    pub(super) fn finish_load_with(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        if let Err(error) = self.prepare_mtp_expert_banks_with(admit) {
            tracing::warn!(
                reason = %error.reason,
                "Qwen4 MTP expert banks unavailable; drafts read expert rows"
            );
        }
        self.start_direct_fill()
    }

    pub(in super::super) fn activate_expert_slots(&mut self) -> Result<()> {
        self.activate_expert_slots_with(&mut super::super::memory::admit_direct)
    }

    pub(in super::super) fn activate_expert_slots_with(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        if self
            .slot_reclaim_since
            .is_some_and(|since| since.elapsed() >= SLOT_RECLAIM_FALLBACK)
        {
            self.purge_expert_slots()?;
        }
        let retrying_activation_refusal = self.activation_error.is_some();
        // Suspended banks are still reclaimable in the live host sample;
        // active banks are already counted there and must not be admitted again.
        let suspended_bytes = self.slots.values().try_fold(0u64, |total, slots| {
            total
                .checked_add(slots.suspended_bank_bytes()?)
                .ok_or_else(|| err("Qwen4 suspended bank total overflows u64"))
        })?;
        if suspended_bytes > 0 {
            let bytes = suspended_bytes
                .checked_add(DIRECT_STAGING_BYTES)
                .ok_or_else(|| err("Qwen4 activation admission overflows u64"))?;
            if let Err(error) = admit(bytes) {
                self.activation_error = Some(error.reason.clone());
                return Err(error);
            }
            self.activation_error = None;
        }
        for slots in self.slots.values_mut() {
            slots.activate()?;
        }
        self.slot_reclaim_since = None;
        // An activation retry must not erase an unrelated asynchronous worker error.
        if !retrying_activation_refusal || self.direct_error.is_none() {
            self.start_direct_fill()?;
        }
        self.direct_fill_pass =
            self.plan.direct && self.slots.values().any(|slots| !slots.identity_ready());
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
        if self.plan.direct && self.direct_slow_source {
            return Ok(());
        }
        if self.plan.direct && self.slots.values().any(|slots| !slots.identity_ready()) {
            // The fill worker still writes unpublished banks. Keep the whole
            // set non-volatile until a later idle boundary can drain it.
            return Ok(());
        }
        self.direct_fill_pass = false;
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

    /// Activate reclaimable banks and resume any unpublished direct layers
    /// before the first turn of an idle model.
    pub(in super::super) fn begin_turn(&mut self) -> Result<()> {
        self.activate_expert_slots()
    }

    #[cfg(test)]
    pub(in super::super) fn begin_turn_with(
        &mut self,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        self.activate_expert_slots_with(admit)
    }

    #[cfg(test)]
    pub(in super::super) fn slot_suspend_states_for_test(&self) -> Vec<(usize, bool, bool)> {
        let mut states: Vec<_> = self
            .slots
            .iter()
            .map(|(&layer, slots)| (layer, slots.suspended, slots.purge_requested))
            .collect();
        states.sort_unstable_by_key(|&(layer, ..)| layer);
        states
    }

    pub(in super::super) fn expert_slot_capacity(
        &mut self,
        experts: usize,
        top: usize,
    ) -> Result<usize> {
        if self.full_residency() {
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
        let capacity = if let Some(bank_budget) = self.slot_bank_budget {
            let gate_capacity = if routed == 0 {
                0
            } else {
                (bank_budget as u128 * experts as u128 / routed as u128).min(experts as u128)
                    as usize
            };
            capacity.min(gate_capacity)
        } else {
            capacity
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
        if !gpu_repack_available() {
            return false;
        }
        let Some(source) = &self.gguf_source else {
            return false;
        };
        names.iter().all(|name| {
            self.descriptor(name).is_ok_and(|tensor| {
                source.has_path(&tensor.path)
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
            Some(capacity) if experts > 0 => {
                (self.slot_routed_bytes()? as u128 * capacity as u128 / experts as u128) as u64
            }
            _ => 0,
        };
        Ok(planned.max(self.slot_bytes))
    }

    fn new_expert_slots(
        &mut self,
        layer: usize,
        experts: usize,
        capacity: usize,
        admit: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<Option<ExpertSlots>> {
        let names =
            ["gate", "up", "down"].map(|part| format!("blk.{layer}.ffn_{part}_exps.weight"));
        let rows = names
            .each_ref()
            .map(|n| self.descriptor(n).map(|d| d.shape[1]));
        let [r0, r1, r2] = rows;
        let rows = [r0?, r1?, r2?];
        let direct = self.direct_layer(&names);
        if !direct {
            // Unsupported layouts retain the decoder's bounded per-row CPU
            // importer rather than constructing another weight-pack format.
            return Ok(None);
        }
        let prototypes: Vec<Arc<Weight>> = names
            .iter()
            .zip(rows)
            .map(|(name, rows)| {
                self.descriptor(name)?
                    .direct_weight(name, rows)
                    .map(Arc::new)
            })
            .collect::<Result<_>>()?;
        if prototypes.iter().any(|w| w.scales.is_none()) {
            return Ok(None);
        }
        let bytes = prototypes
            .iter()
            .try_fold(0u64, |n, w| w.bytes().map(|b| n + b * capacity as u64))?;
        // A single layer cannot bypass the same staging and system reserve.
        self.evict_to(self.cache_limit.saturating_sub(bytes + MAX_READ_BYTES * 3))?;
        super::super::memory::maintain_freelist(self.plan.physical_bytes);
        admit(bytes + MAX_READ_BYTES * 3)?;
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
        Ok(Some(ExpertSlots {
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
        }))
    }

    pub(in super::super) fn expert_slots(
        &mut self,
        layer: usize,
        experts: usize,
        top: usize,
        ids: &[u32],
    ) -> Result<Option<([Arc<Weight>; 3], Vec<u32>, usize)>> {
        if !self.plan.direct {
            self.check_activation_refusal()?;
        }
        let capacity = self.expert_slot_capacity(experts, top)?;
        let wanted: HashSet<u32> = ids.iter().copied().collect();
        if capacity == 0 || wanted.len() > capacity {
            return Ok(None);
        }
        if ids.iter().any(|&e| e as usize >= experts) {
            return Err(err("Qwen4 expert id exceeds slot mapping"));
        }
        if self.plan.direct {
            self.wait_direct_layer(layer)?;
            let slots = self
                .slots
                .get(&layer)
                .filter(|slots| slots.identity_ready())
                .ok_or_else(|| err("Qwen4 direct layer has no filled expert banks"))?;
            self.slot_hits += wanted.len() as u64;
            return Ok(Some((slots.banks.clone(), ids.to_vec(), experts)));
        }
        let mut slots = if let Some(slots) = self.slots.remove(&layer) {
            slots
        } else {
            let direct_gate = self.slot_bank_budget.is_some();
            let mut admit = |bytes| {
                if direct_gate {
                    super::super::memory::admit_direct(bytes)
                } else {
                    super::super::memory::admit(bytes)
                }
            };
            match self.new_expert_slots(layer, experts, capacity, &mut admit)? {
                Some(slots) => slots,
                None => return Ok(None),
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
                for _ in missing {
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
                }
                if let Err(error) = self.fill_direct_chunk(&mut slots, missing, &chosen) {
                    slots.invalidate_transaction();
                    return Err(error);
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
        if self.plan.direct {
            return (0..layers).all(|layer| {
                self.slots
                    .get(&layer)
                    .is_some_and(ExpertSlots::identity_ready)
            });
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
        if self.plan.direct {
            return (0..layers).all(|layer| {
                self.slots.get(&layer).is_some_and(|slots| {
                    !slots.suspended && slots.direct && fused_bank_modes(&slots.banks)
                })
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
        if self.plan.direct {
            self.wait_direct_layer(layer)?;
            let slots = self
                .slots
                .get(&layer)
                .filter(|slots| slots.identity_ready())
                .ok_or_else(|| err("Qwen4 tentative prefill has no direct bank"))?;
            let local = ids.reshape(&[-1])?.astype(DType::Uint32)?;
            return Ok((slots.banks.clone(), local, slots.mapping.len()));
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
        if self.plan.direct {
            self.wait_direct_layer(layer)?;
            let Some(slots) = self
                .slots
                .get(&layer)
                .filter(|slots| slots.identity_ready())
            else {
                return Ok(None);
            };
            let local = ids.reshape(&[-1])?.astype(DType::Uint32)?;
            return Ok(Some((slots.banks.clone(), local)));
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
        if self.plan.direct {
            for &(layer, selected) in entries {
                let Some(slots) = self
                    .slots
                    .get(&layer)
                    .filter(|slots| slots.identity_ready())
                else {
                    return Ok(false);
                };
                if selected.iter().any(|&e| e as usize >= slots.mapping.len()) {
                    return Err(err(
                        "Qwen4 device route names an expert outside the direct banks",
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
        if self.full_residency() {
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
        if self.full_residency() {
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
        if self.full_residency() {
            return MxArray::eval_arrays_with_context(&[reduced], "qwen4::direct::complete_window");
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
        if self.full_residency() {
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

    /// A direct MTP load installs the Q8 draft banks; partial slots then fit
    /// beside them while they stay pinned. Every layer's
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

    /// The gate, not the old bootstrap budget, bounds partial banks on the
    /// released 48-layer geometry. No bank payload is allocated by this check.
    #[test]
    fn gate_sized_partial_capacity_tracks_the_full_bank_shortfall() {
        let Some((mut store, layers)) = draft_bank_store(None) else {
            return;
        };
        let routed: u64 = layers.iter().sum();
        assert_eq!(routed, 78_538_342_400);
        let anonymous = super::super::super::memory::direct_desired(
            FIXED_BANK_BYTES.saturating_add(3_040_870_400),
        );
        for (bank_budget, expected) in [(routed - 1, 511), (72 << 30, 503), (40 << 30, 279)] {
            store.cache_limit = anonymous.saturating_add(bank_budget);
            store.plan.budget = store.cache_limit;
            store.slot_bank_budget = Some(bank_budget);
            store.slot_capacity = None;
            store.mtp_banks_declined = false;
            let capacity = store.expert_slot_capacity(512, 10).unwrap();
            assert_eq!(capacity, expected, "bank budget {bank_budget}");
            let planned = store.planned_slot_bytes().unwrap();
            assert!(planned <= bank_budget);
            assert!(planned > bank_budget.saturating_sub(routed / 512));
            assert_eq!(store.plan.budget, anonymous + bank_budget);
        }
        assert_eq!(store.bytes_read, 0);
        assert_eq!(store.slot_bytes, 0);
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
    /// build would take: below a 22.66 GiB slot allowance, beside a direct
    /// plan that charged the f16 layout, or with only the f16 staging left.
    /// The reserve never holds less than the build.
    /// MUTATIONS: only the f32 layout -> 22 GiB; the slot allowance beside a
    /// direct charge -> the f16 charge; no f16 retry -> the f16 staging.
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
            let why = format!("a direct plan charged {charged}");
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
