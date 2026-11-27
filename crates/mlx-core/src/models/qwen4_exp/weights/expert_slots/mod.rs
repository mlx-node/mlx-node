//! Stable, quantized expert slots for partial residency. Slot buffers are
//! mutated only after their last lazy reader has completed. The ordinary LRU
//! supplies bounded misses and releases imported chunks after the slot copy.

#[cfg(not(target_os = "macos"))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(test)]
mod tests;

use super::*;
use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::utils::gguf_cpu_repack::{cpu_kquant_band_into, decode_affine_band_into};
use crate::utils::gguf_gpu::{
    GpuRepackKind, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available,
    gpu_repack_band, gpu_repack_supports_kind,
};
use crate::utils::gguf_read::{
    Batch, PageCacheMode, READ_THREADS, ReadQueue, Reader, ReaderPool, Span, plan_ranges,
    plan_reads, set_nocache,
};
#[cfg(not(target_os = "macos"))]
use cuda::{StagingOwner, allocate_slot_banks, staging_buffer};
#[cfg(target_os = "macos")]
use metal::{StagingOwner, allocate_slot_banks, staging_buffer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Whether this host supports GPU-resident direct expert banks (Metal and CUDA
/// both use GPU repack kernels for `Q4_K`, `Q5_K`, `Q5_1`, `Q8_0Bias`, falling
/// back to parallel CPU repack only if GPU repack is unavailable).
pub(in super::super) fn direct_residency_available() -> bool {
    crate::engine::persistence::compiled_forward_backend_available()
        || crate::accelerator::host_has_cuda()
}

const DIRECT_READ_BYTES: usize = 8 << 20;
const DIRECT_STAGING_SLOTS: usize = 4;
/// A slot fill's batch: the bytes of one staging buffer of its ring.
const FILL_BATCH_BYTES: usize = 64 << 20;
const FILL_RING: usize = 4;
/// The bank bytes of an expert `cap` that also holds the slots' staging: a
/// slot of every layer stages one more layer's share, so
/// `slots * (routed + largest) / experts <= cap`.
fn slot_bank_budget_for(cap: u64, routed: u64, largest_layer: u64) -> u64 {
    (u128::from(cap) * u128::from(routed) / u128::from(routed.saturating_add(largest_layer)).max(1))
        as u64
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
    direct: bool,
    names: [String; 3],
    rows: [usize; 3],
    occupants: Vec<Option<u32>>,
    mapping: Vec<Option<usize>>,
    device_mapping: Option<MxArray>,
    ages: Vec<u64>,
    tick: u64,
    readers: Vec<MxArray>,
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
            q8_symmetric_128: ty == GgufTensorType::Q8_0,
            qlinear: None,
        })
    }
}

impl ExpertSlots {
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
    /// bookkeeping. All handles share the same bank storage.
    fn fill_clone(&self) -> Self {
        Self {
            banks: self.banks.clone(),
            direct: self.direct,
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

    /// Independent test clone of an identity-ready direct bank set: shares the
    /// read-only quantized weight handles while resetting all mutable slot
    /// bookkeeping (`ages`, `tick`, `readers`, `device_mapping`).
    #[cfg(test)]
    pub(super) fn clone_identity_for_test(&self) -> Option<Self> {
        if !self.identity_ready() {
            return None;
        }
        Some(Self {
            banks: self.banks.clone(),
            direct: true,
            names: self.names.clone(),
            rows: self.rows,
            occupants: self.occupants.clone(),
            mapping: self.mapping.clone(),
            device_mapping: None,
            ages: vec![0; self.ages.len()],
            tick: 0,
            readers: Vec::new(),
        })
    }
}

struct DirectRequest {
    part: usize,
    slot: usize,
    rows: usize,
    k: usize,
    kind: GpuRepackKind,
}

struct DirectRun<'r> {
    reader: &'r ReadQueue,
    /// The ticket of batch 0; batch `b` reads under `first_ticket + b`.
    first_ticket: u64,
    source: &'r DirectGgufSource,
    requests: &'r [DirectRequest],
    batches: &'r [Batch],
    staging: &'r [(MxArray, SendPtr<u8>)],
    slots: &'r mut ExpertSlots,
    stop: Option<&'r AtomicBool>,
    rows: Option<&'r AtomicUsize>,
    #[cfg(test)]
    fail_batch: Option<usize>,
}

impl DirectRun<'_> {
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
        for read in &self.batches[w].reads {
            self.reader.submit(
                self.first_ticket + w as u64,
                SendPtr(ptr.0.wrapping_add(read.buf_off)),
                read.len,
                read.file_off,
            );
        }
        Ok(())
    }

    fn complete(&self, _w: usize, outputs: &[MxArray]) -> Result<()> {
        let refs: Vec<&MxArray> = outputs.iter().collect();
        MxArray::eval_arrays_with_context(&refs, "qwen4::slots::direct_gguf_window")
    }

    /// Every batch's reads, each repacked as soon as they ended; a staging
    /// buffer is read into again only after the repack of the batch before
    /// in it completed. The last batch's repack has completed on return.
    fn all(&mut self) -> Result<()> {
        self.submit_ahead()?;
        self.rest()
    }

    /// Queues one batch into each staging buffer, before any repack.
    fn submit_ahead(&self) -> Result<()> {
        let n = self.batches.len();
        let slots = self.staging.len();
        if n == 0 || slots == 0 {
            return Err(err("Qwen4 direct GGUF fill has no read windows"));
        }
        for w in 0..slots.min(n) {
            self.submit(w)?;
        }
        Ok(())
    }

    /// `all` after `submit_ahead`.
    fn rest(&mut self) -> Result<()> {
        let n = self.batches.len();
        let slots = self.staging.len();
        if n == 0 || slots == 0 {
            return Err(err("Qwen4 direct GGUF fill has no read windows"));
        }
        let mut pending: Vec<MxArray> = Vec::new();
        for w in 0..n {
            if w > 0 && !pending.is_empty() {
                self.complete(w - 1, &std::mem::take(&mut pending))?;
            }
            // Batch `next` reads into the buffer batch `w - 1` repacked
            // from; `submit_ahead` queued the batches before `slots`.
            let next = w + slots - 1;
            if next < n && next >= slots {
                self.submit(next)?;
            }
            self.wait(w)?;
            pending = self.batch(w)?;
            if !pending.is_empty() {
                let refs: Vec<&MxArray> = pending.iter().collect();
                MxArray::async_eval_arrays(&refs);
            }
        }
        if !pending.is_empty() {
            self.complete(n - 1, &pending)?;
        }
        Ok(())
    }

    fn wait(&self, w: usize) -> Result<()> {
        let read = self.reader.wait(self.first_ticket + w as u64);
        #[cfg(test)]
        if self.fail_batch == Some(w) {
            self.reader.stop();
            return Err(err(format!("injected read failure at batch {w}")));
        }
        if self.source.shards.len() == 1 {
            return read.map_err(Into::into);
        }
        let file_off = self.batches[w]
            .reads
            .first()
            .map_or(0, |read| read.file_off);
        let shard = self
            .source
            .shards
            .iter()
            .find(|shard| file_off >= shard.base && file_off < shard.base + shard.len)
            .or_else(|| self.source.shards.last())
            .ok_or_else(|| err("Qwen4 direct GGUF window has no source shard"))?;
        read.map_err(|error| err(format!("GGUF shard '{}': {error}", shard.path.display())))
    }

    fn batch(&mut self, w: usize) -> Result<Vec<MxArray>> {
        if !gpu_repack_available()
            || self.batches[w]
                .bands
                .iter()
                .any(|b| !gpu_repack_supports_kind(self.requests[b.tensor].kind))
        {
            return self.cpu_repack_batch(w);
        }
        let (staging, _) = &self.staging[w % self.staging.len()];
        let mut outputs = Vec::new();
        for band in &self.batches[w].bands {
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

    /// Parallel CPU repack fallback for non-Metal hosts (Windows CUDA Phase 0;
    /// Phase 2 hook point: once CUDA GPU repack kernels implement
    /// `gpu_repack_band`, `gpu_repack_available()` routes `batch()` there).
    fn cpu_repack_batch(&mut self, w: usize) -> Result<Vec<MxArray>> {
        const STRIPE_ROWS: usize = 64;
        struct StripeJob {
            name: String,
            kind: GpuRepackKind,
            k: usize,
            row0: usize,
            rows: usize,
            src: SendPtr<u8>,
            src_len: usize,
            dst: DstPtrs,
        }

        // Ensure no prior GPU kernel is still reading partial-slot banks
        // before overwriting their managed/unified storage from CPU threads.
        crate::array::memory::synchronize();

        let (_, staging_ptr) = &self.staging[w % self.staging.len()];
        let batch = &self.batches[w];
        let mut bank_ptrs: [Option<DstPtrs>; 3] = [None, None, None];
        let mut jobs = Vec::with_capacity(batch.bands.len());

        for band in &batch.bands {
            let request = &self.requests[band.tensor];
            let part = request.part;
            let dst = match bank_ptrs[part] {
                Some(ptrs) => ptrs,
                None => {
                    let ptrs = bank_dst_ptrs(&self.slots.banks[part], request.k)?;
                    bank_ptrs[part] = Some(ptrs);
                    ptrs
                }
            };
            let base_row0 = request
                .slot
                .checked_mul(request.rows)
                .and_then(|row| row.checked_add(band.row0))
                .ok_or_else(|| err("Qwen4 direct GGUF destination row overflow"))?;
            let ty = request.kind.gguf_type();
            let block_values = ty.block_size();
            if request.k == 0 || !request.k.is_multiple_of(block_values) {
                return Err(err(format!(
                    "Qwen4 direct GGUF {} row width {} is not a multiple of {}",
                    self.slots.names[part], request.k, block_values
                )));
            }
            let row_bytes = (request.k / block_values) * ty.type_size();
            if band.bytes != band.rows * row_bytes
                || band.src_off.saturating_add(band.bytes) > batch.len
            {
                return Err(err(format!(
                    "Qwen4 direct GGUF {} band byte length {} mismatches {} rows x {} B",
                    self.slots.names[part], band.bytes, band.rows, row_bytes
                )));
            }
            let mut done = 0usize;
            while done < band.rows {
                let take = STRIPE_ROWS.min(band.rows - done);
                let byte_off = band.src_off + done * row_bytes;
                let byte_len = take * row_bytes;
                jobs.push(StripeJob {
                    name: self.slots.names[part].clone(),
                    kind: request.kind,
                    k: request.k,
                    row0: base_row0 + done,
                    rows: take,
                    src: SendPtr(staging_ptr.0.wrapping_add(byte_off)),
                    src_len: byte_len,
                    dst,
                });
                done += take;
            }
        }

        let run_job = |job: &StripeJob| -> Result<()> {
            // SAFETY: `staging_ptr` stays alive and immutable while `batch`
            // runs; `byte_off + byte_len <= batch.len` was checked above.
            let src = unsafe { std::slice::from_raw_parts(job.src.0, job.src_len) };
            if job.kind.is_kquant() {
                let info = GgufTensorInfo {
                    name: job.name.clone(),
                    n_dims: 2,
                    dims: vec![job.k as u64, job.dst.rows as u64],
                    tensor_type: job.kind.gguf_type(),
                    offset: 0,
                };
                cpu_kquant_band_into(&info, src, job.row0, job.rows, &job.dst)
            } else if job.kind == GpuRepackKind::Q8_0Bias {
                let scales_dst = job
                    .dst
                    .scales
                    .ok_or_else(|| err("Qwen4 Q8_0 direct bank missing scales"))?;
                let biases_dst = job
                    .dst
                    .biases
                    .ok_or_else(|| err("Qwen4 Q8_0 direct bank missing biases"))?;
                let affine_dst = DstPtrs {
                    biases: None,
                    ..job.dst
                };
                decode_affine_band_into(
                    GgufTensorType::Q8_0,
                    src,
                    job.row0,
                    job.rows,
                    job.k,
                    &affine_dst,
                )?;
                let per_row = job.k / 32;
                let first = job.row0 * per_row;
                let count = job.rows * per_row;
                // SAFETY: `scales_dst` and `biases_dst` hold
                // `job.dst.rows * per_row` f16 values; `[first, first + count)`
                // belongs exclusively to this stripe.
                unsafe {
                    let scales =
                        std::slice::from_raw_parts(scales_dst.0.cast::<u16>().add(first), count);
                    let biases = std::slice::from_raw_parts_mut(
                        biases_dst.0.cast::<u16>().add(first),
                        count,
                    );
                    for (bias, &scale) in biases.iter_mut().zip(scales) {
                        *bias = half::f16::from_f32(half::f16::from_bits(scale).to_f32() * -128.0)
                            .to_bits();
                    }
                }
                Ok(())
            } else {
                decode_affine_band_into(
                    job.kind.gguf_type(),
                    src,
                    job.row0,
                    job.rows,
                    job.k,
                    &job.dst,
                )
            }
        };

        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(jobs.len());
        if workers <= 1 {
            for job in &jobs {
                run_job(job)?;
            }
        } else {
            let next = AtomicUsize::new(0);
            let failed = AtomicBool::new(false);
            let first_err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| {
                        while !failed.load(Ordering::Acquire) {
                            let idx = next.fetch_add(1, Ordering::AcqRel);
                            let Some(job) = jobs.get(idx) else {
                                break;
                            };
                            if let Err(error) = run_job(job) {
                                failed.store(true, Ordering::Release);
                                if let Ok(mut slot) = first_err.lock()
                                    && slot.is_none()
                                {
                                    *slot = Some(error);
                                }
                                break;
                            }
                        }
                    });
                }
            });
            if let Some(error) = first_err
                .into_inner()
                .map_err(|_| err("Qwen4 direct GGUF repack worker panicked"))?
            {
                return Err(error);
            }
        }
        Ok(Vec::new())
    }
}

#[derive(Default)]
struct DirectFillStats {
    source_bytes: u64,
    read_bytes: u64,
    bank_bytes: u64,
    /// Time this fill spent queueing its reads ahead of its lease.
    ahead_micros: u64,
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
    #[cfg(test)]
    live: Arc<super::super::live::Live>,
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
                    #[cfg(test)]
                    worker_live.fill_hold.wait();
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
                                worker_live
                                    .direct_filled_layers
                                    .fetch_add(1, Ordering::AcqRel);
                            }
                            Err(failure) => {
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
            #[cfg(test)]
            live,
        })
    }
}

impl Drop for DirectFill {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        #[cfg(test)]
        self.live.fill_hold.release();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The repack requests of `experts` going to `chosen` slots, and the source
/// range of each (its logical offset in `infos`).
fn direct_requests(
    source: &DirectGgufSource,
    descriptors: &[Tensor; 3],
    slots: &ExpertSlots,
    experts: &[u32],
    chosen: &[usize],
) -> Result<(Vec<DirectRequest>, Vec<GgufTensorInfo>, DirectFillStats)> {
    if experts.len() != chosen.len() || experts.is_empty() {
        return Err(err("Qwen4 direct GGUF fill has an invalid expert batch"));
    }
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
    Ok((requests, infos, stats))
}

/// Extract writable destination pointers for a settled expert bank allocated
/// via `MxArray::alloc_uninit` (non-Metal direct residency).
fn bank_dst_ptrs(bank: &Weight, k: usize) -> Result<DstPtrs> {
    let rows = bank.values.shape_at(0)? as usize;
    let ptr_of = |a: &MxArray, label: &str| -> Result<SendPtr<u8>> {
        // SAFETY: `a` is a live `MxArray` handle; `mlx_array_data_ptr` returns
        // its settled storage pointer or null.
        let p = unsafe { mlx_sys::mlx_array_data_ptr(a.as_raw_ptr()) };
        if p.is_null() {
            return Err(err(format!(
                "Qwen4 direct expert bank {label} has no settled storage"
            )));
        }
        Ok(SendPtr(p.cast_mut().cast::<u8>()))
    };
    Ok(DstPtrs {
        weight: ptr_of(&bank.values, "weight")?,
        scales: bank
            .scales
            .as_ref()
            .map(|a| ptr_of(a, "scales"))
            .transpose()?,
        biases: bank
            .biases
            .as_ref()
            .map(|a| ptr_of(a, "biases"))
            .transpose()?,
        rows,
        k,
        layout: OutLayout::RowMajor,
    })
}

/// The whole-store fill of selected slots through its own bounded reader
/// and staging.
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
    source.verify_path()?;
    source.verify(files)?;
    let (requests, infos, mut stats) =
        direct_requests(source, descriptors, slots, experts, chosen)?;
    let header = GgufFile {
        version: 3,
        tensor_count: infos.len() as u64,
        metadata: HashMap::new(),
        tensors: infos,
        alignment: 32,
        data_offset: 0,
    };
    let batches: Vec<Batch> = plan_reads(&header, source.len, DIRECT_READ_BYTES, &|_| 1)?
        .into_iter()
        .map(Batch::from)
        .collect();
    let slot_len = batches.iter().map(|b| b.len).max().unwrap_or(0);
    // Keep the aligned backing allocations alive until the GPU stream drains.
    let mut _staging_owners = Vec::with_capacity(DIRECT_STAGING_SLOTS.min(batches.len()));
    let mut staging = Vec::with_capacity(DIRECT_STAGING_SLOTS.min(batches.len()));
    for _ in 0..DIRECT_STAGING_SLOTS.min(batches.len()) {
        let (bank, array, ptr) = staging_buffer(slot_len)?;
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
            first_ticket: 0,
            source,
            requests: &requests,
            batches: &batches,
            staging: &staging,
            slots,
            stop,
            rows: pending_rows,
            #[cfg(test)]
            fail_batch: None,
        };
        run.all()
    });
    // A failed window may already have encoded earlier bands. Finish them
    // before the caller unpublishes every destination in this transaction.
    crate::array::memory::synchronize();
    result?;
    source.verify(files)?;
    stats.read_bytes = batches.iter().map(|b| b.len as u64).sum();
    #[cfg(test)]
    stats.windows.extend(
        batches
            .iter()
            .flat_map(|b| b.reads.iter().map(|r| (r.file_off, r.len))),
    );
    Ok(stats)
}

/// The slot fills' readers and staging ring: made by the first slot fill,
/// kept for the store's life. The readers drop first: a read they still run
/// writes a ring buffer.
pub(super) struct SlotFill {
    pool: ReaderPool,
    files: Vec<Arc<File>>,
    ring: Vec<(StagingOwner, MxArray, SendPtr<u8>)>,
    batch_bytes: usize,
}

impl SlotFill {
    fn start(source: &DirectGgufSource, files: Vec<Arc<File>>, batch_bytes: usize) -> Result<Self> {
        let read_at = OwnedDirectGgufReader::new(source.clone(), files.clone())?;
        let pool = ReaderPool::start(
            Arc::new(read_at),
            &source.shards[0].path,
            READ_THREADS,
            DIRECT_READ_BYTES,
        )
        .map_err(|error| err(format!("Qwen4 slot fill readers did not start: {error}")))?;
        Ok(Self {
            pool,
            files,
            ring: Vec::new(),
            batch_bytes,
        })
    }

    /// The ring buffers `batches` use, grown to them.
    fn staging(&mut self, batches: &[Batch]) -> Result<Vec<(MxArray, SendPtr<u8>)>> {
        if batches.iter().any(|b| b.len > self.batch_bytes) {
            return Err(err("Qwen4 slot fill batch exceeds its staging buffer"));
        }
        while self.ring.len() < FILL_RING.min(batches.len()) {
            self.ring.push(staging_buffer(self.batch_bytes)?);
        }
        Ok(self.ring[..FILL_RING.min(batches.len())]
            .iter()
            .map(|(_, array, ptr)| (array.clone(), *ptr))
            .collect())
    }

    /// Queues the reads of the batches the ring holds and returns the first
    /// batch's ticket for `run`. Writes no bank: a fill before it completed
    /// its repacks, so no GPU work reads the ring.
    fn begin(
        &mut self,
        source: &DirectGgufSource,
        requests: &[DirectRequest],
        batches: &[Batch],
        slots: &mut ExpertSlots,
    ) -> Result<u64> {
        let staging = self.staging(batches)?;
        let first_ticket = self.pool.tickets(batches.len());
        let queued = DirectRun {
            reader: &self.pool,
            first_ticket,
            source,
            requests,
            batches,
            staging: &staging,
            slots,
            stop: None,
            rows: None,
            #[cfg(test)]
            fail_batch: None,
        }
        .submit_ahead();
        if let Err(error) = queued {
            self.pool.recover();
            return Err(error);
        }
        Ok(first_ticket)
    }

    /// Reads `batches` into the ring and repacks each into `slots`, after
    /// the reads `begin` queued under `begun` when given; the last repack
    /// has completed on success. On an error no read is left running and no
    /// GPU work reads the ring.
    fn run(
        &mut self,
        source: &DirectGgufSource,
        requests: &[DirectRequest],
        batches: &[Batch],
        slots: &mut ExpertSlots,
        begun: Option<u64>,
        #[cfg(test)] fail_batch: Option<usize>,
    ) -> Result<()> {
        let staging = self.staging(batches)?;
        let first_ticket = begun.unwrap_or_else(|| self.pool.tickets(batches.len()));
        let mut run = DirectRun {
            reader: &self.pool,
            first_ticket,
            source,
            requests,
            batches,
            staging: &staging,
            slots,
            stop: None,
            rows: None,
            #[cfg(test)]
            fail_batch,
        };
        let result = if begun.is_some() {
            run.rest()
        } else {
            run.all()
        };
        if result.is_err() {
            // Bands of a failed batch may be encoded; reads of later batches
            // may still run. Both use the ring.
            crate::array::memory::synchronize();
            self.pool.recover();
        }
        result
    }
}

/// A slot fill `begin_expert_fill` queued: its victims chosen and its reads
/// running into the ring, no bank, mapping or age touched. The lease that
/// wants the same experts in the same victims finishes it.
pub(super) struct PendingFill {
    bank: String,
    experts: Vec<u32>,
    chosen: Vec<usize>,
    requests: Vec<DirectRequest>,
    batches: Vec<Batch>,
    stats: DirectFillStats,
    first_ticket: u64,
}

impl Store {
    fn account_direct_fill(&mut self, stats: &DirectFillStats) {
        #[cfg(test)]
        self.direct_read_windows
            .extend(stats.windows.iter().copied());
        self.count_bytes_read(stats.source_bytes);
        self.slot_upload_bytes += stats.bank_bytes;
    }

    /// Count `n` routed experts served from slots or banks, here and in the
    /// live counters `residencyInfo` reads.
    fn count_slot_hits(&mut self, n: u64) {
        self.slot_hits += n;
        self.live.slot_hits.fetch_add(n, Ordering::Relaxed);
    }

    fn count_slot_misses(&mut self, n: u64) {
        self.slot_misses += n;
        self.live.slot_misses.fetch_add(n, Ordering::Relaxed);
    }

    fn fill_direct_chunk(
        &mut self,
        slots: &mut ExpertSlots,
        experts: &[u32],
        chosen: &[usize],
    ) -> Result<()> {
        let started = Instant::now();
        let filled = self.fill_direct_slots(slots, experts, chosen);
        self.live.fill_calls.fetch_add(1, Ordering::Relaxed);
        self.live
            .fill_micros
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        let stats = filled?;
        self.live
            .fill_micros
            .fetch_add(stats.ahead_micros, Ordering::Relaxed);
        self.live
            .fill_read_bytes
            .fetch_add(stats.read_bytes, Ordering::Relaxed);
        self.account_direct_fill(&stats);
        Ok(())
    }

    /// The main source and the three descriptors of `slots`' banks.
    fn slot_source(&self, slots: &ExpertSlots) -> Result<(DirectGgufSource, [Tensor; 3])> {
        let source = self
            .gguf_source
            .clone()
            .ok_or_else(|| err("Qwen4 direct GGUF fill has no main source"))?;
        let descriptors = slots
            .names
            .each_ref()
            .map(|name| self.descriptor(name).cloned());
        let [d0, d1, d2] = descriptors;
        Ok((source, [d0?, d1?, d2?]))
    }

    /// The slot fills' readers and ring, started by the first slot fill.
    fn start_slot_fill(&mut self, source: &DirectGgufSource) -> Result<()> {
        if self.slot_fill.is_some() {
            return Ok(());
        }
        source.verify_path()?;
        let files = self.source_files(source)?;
        #[cfg(test)]
        let batch_bytes = self.fill_batch_bytes.unwrap_or(FILL_BATCH_BYTES);
        #[cfg(not(test))]
        let batch_bytes = FILL_BATCH_BYTES;
        let fill = SlotFill::start(source, files, batch_bytes)?;
        #[cfg(test)]
        {
            self.slot_fill_builds[0] += fill.pool.threads();
        }
        self.slot_fill = Some(fill);
        Ok(())
    }

    /// The repack requests, accounting and read batches of a slot fill of
    /// `experts` into `chosen`, after the shard check before it.
    fn plan_slot_fill(
        fill: &SlotFill,
        source: &DirectGgufSource,
        descriptors: &[Tensor; 3],
        slots: &ExpertSlots,
        experts: &[u32],
        chosen: &[usize],
    ) -> Result<(Vec<DirectRequest>, Vec<Batch>, DirectFillStats)> {
        let file_refs: Vec<&File> = fill.files.iter().map(AsRef::as_ref).collect();
        source.verify(&file_refs)?;
        let (requests, infos, stats) =
            direct_requests(source, descriptors, slots, experts, chosen)?;
        let spans = infos
            .iter()
            .map(|info| {
                Ok(Span {
                    off: info.offset,
                    bytes: info.data_size()?,
                    rows: info.dims[1] as usize,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let batches = plan_ranges(&spans, source.len, fill.batch_bytes)?;
        Ok((requests, batches, stats))
    }

    fn fill_direct_slots(
        &mut self,
        slots: &mut ExpertSlots,
        experts: &[u32],
        chosen: &[usize],
    ) -> Result<DirectFillStats> {
        let (source, descriptors) = self.slot_source(slots)?;
        #[cfg(test)]
        if self.old_slot_fill {
            source.verify_path()?;
            let files = self.source_files(&source)?;
            let file_refs: Vec<&File> = files.iter().map(AsRef::as_ref).collect();
            return fill_direct(
                &source,
                &file_refs,
                &descriptors,
                slots,
                experts,
                chosen,
                None,
                None,
            );
        }
        self.start_slot_fill(&source)?;
        let pending = self.pending_fill.take();
        let fill = self
            .slot_fill
            .as_mut()
            .ok_or_else(|| err("Qwen4 slot fill lost its readers"))?;
        let (requests, batches, mut stats, begun) = match pending {
            Some(ahead)
                if ahead.bank == slots.names[0]
                    && ahead.experts == experts
                    && ahead.chosen == chosen =>
            {
                (
                    ahead.requests,
                    ahead.batches,
                    ahead.stats,
                    Some(ahead.first_ticket),
                )
            }
            other => {
                if other.is_some() {
                    // Its reads still write the ring.
                    fill.pool.recover();
                }
                let (requests, batches, stats) =
                    Self::plan_slot_fill(fill, &source, &descriptors, slots, experts, chosen)?;
                (requests, batches, stats, None)
            }
        };
        #[cfg(test)]
        let staged = fill.ring.len();
        fill.run(
            &source,
            &requests,
            &batches,
            slots,
            begun,
            #[cfg(test)]
            self.fail_fill_batch,
        )?;
        let file_refs: Vec<&File> = fill.files.iter().map(AsRef::as_ref).collect();
        source.verify(&file_refs)?;
        let reads = batches.iter().flat_map(|b| &b.reads);
        stats.read_bytes = reads.clone().map(|r| r.len as u64).sum();
        #[cfg(test)]
        {
            self.slot_fill_builds[1] += fill.ring.len() - staged;
            stats.windows.extend(reads.map(|r| (r.file_off, r.len)));
        }
        Ok(stats)
    }

    /// The store's open descriptor of each shard of `source`.
    fn source_files(&mut self, source: &DirectGgufSource) -> Result<Vec<Arc<File>>> {
        source
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
            .collect()
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
    ///
    /// `anonymous_need` is what the plan holds beside the experts (the fixed
    /// weights, the draft banks, the row cache and the KV allowance); the
    /// host gate of a whole bank set reads it.
    pub(in super::super) fn select_residency(&mut self, anonymous_need: u64) -> Result<()> {
        let experts = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0) as usize;
        let direct = self.select_direct_residency_with(
            experts,
            anonymous_need,
            &mut super::super::memory::direct_gate,
            &mut super::super::memory::admit_direct,
        )?;
        if !direct {
            self.slot_capacity = Some(0);
            self.plan.partial_expert_slots = 0;
        }
        Ok(())
    }

    /// The store plan decides: whole experts take direct banks at identity
    /// capacity, a streamed store partial slots under its cap. A host that
    /// refuses the whole set lowers the allocation to partial slots of what it
    /// admits; the plan keeps counting the cap. Returns true when direct
    /// eligibility was decided; false keeps per-row imports.
    pub(in super::super) fn select_direct_residency_with(
        &mut self,
        experts: usize,
        anonymous_need: u64,
        gate: &mut impl FnMut(&mut super::super::memory::Plan, u64, u64) -> Result<()>,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<bool> {
        self.plan.anonymous_need = anonymous_need;
        self.plan.self_mlx_bytes = (crate::array::memory::get_active_memory()
            + crate::array::memory::get_cache_memory()) as u64;
        if !self.gguf
            || experts == 0
            || !direct_residency_available()
            || !self.slots.is_empty()
            || !self.banks.is_empty()
            || self.plan.direct
            || self.gguf_source.is_none()
        {
            return Ok(false);
        }
        let Some((layers, bank_bytes)) = self.direct_bank_set(experts)? else {
            return Ok(false);
        };
        let store = self
            .store_plan
            .ok_or_else(|| err("Qwen4 residency needs the store plan"))?;
        let draft_bank = self.mtp_bank_estimate()?;
        let largest = self.largest_routed_layer_bytes()?;
        let top = self
            .metadata
            .get("qwen4exp.expert_used_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0) as usize;
        let cap_budget = slot_bank_budget_for(store.expert_cap_bytes, bank_bytes, largest);
        let partial = |store: &mut Self, budget: u64| -> Result<()> {
            store.plan.direct = false;
            store.plan.partial_direct_gate = true;
            store.plan.direct_bytes = 0;
            store.plan.direct_layers = 0;
            store.draft_bank_budget = 0;
            store.slot_capacity = None;
            store.slot_bank_budget = Some(budget);
            store.plan.partial_expert_slots = store.expert_slot_capacity(experts, top)?;
            Ok(())
        };
        if !store.whole {
            let budget = cap_budget;
            if self.plan.physical_bytes.is_some() {
                let host = super::super::memory::host_memory()?;
                let gate = super::super::memory::bank_gate(
                    &host,
                    anonymous_need,
                    super::super::memory::DIRECT_MARGIN,
                );
                self.plan.host = Some(host);
                self.plan.bank_budget = gate.bank_budget;
                self.plan.live_non_reclaimable = gate.non_reclaimable;
            }
            partial(self, budget)?;
            self.live.set_fallback(Fallback::StorePlan {
                bank_bytes,
                cap: budget,
            });
            self.live.store(&self.plan);
            tracing::info!(
                "qwen4 residency: streamed experts, {} slots within a {:.1} GiB bank budget of the memory plan",
                self.plan.partial_expert_slots,
                budget as f64 / (1u64 << 30) as f64,
            );
            return Ok(true);
        }
        let original = self.plan.clone();
        let original_draft_budget = self.draft_bank_budget;
        let original_slot_capacity = self.slot_capacity;
        let original_slot_bank_budget = self.slot_bank_budget;
        let original_slot_bytes = self.slot_bytes;
        let original_cache_bytes = self.cache_bytes;
        let mut candidate = self.plan.clone();
        if let Err(error) = gate(&mut candidate, anonymous_need, bank_bytes) {
            let admitted = candidate.bank_budget;
            self.plan = candidate;
            let budget = cap_budget.min(admitted);
            partial(self, budget)?;
            self.live.set_fallback(Fallback::LoadGate {
                bank_bytes,
                budget: admitted,
            });
            self.live.store(&self.plan);
            tracing::warn!(
                "qwen4 residency: partial direct slots, {error}; {} slots within a {:.1} GiB bank budget",
                self.plan.partial_expert_slots,
                budget as f64 / (1u64 << 30) as f64,
            );
            return Ok(true);
        }
        candidate.direct = true;
        candidate.partial_direct_gate = false;
        candidate.direct_bytes = bank_bytes;
        candidate.direct_layers = layers.len();
        candidate.partial_expert_slots = experts;
        self.plan = candidate;
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
            self.draft_bank_budget = original_draft_budget;
            self.slot_capacity = original_slot_capacity;
            self.slot_bank_budget = original_slot_bank_budget;
            return Err(error);
        }
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

    /// The routed layers and the bytes of their whole bank set, when every
    /// layer takes direct GGUF banks of `experts` rows.
    fn direct_bank_set(
        &self,
        experts: usize,
    ) -> Result<Option<(std::collections::BTreeSet<usize>, u64)>> {
        let layers: std::collections::BTreeSet<usize> = self
            .tensors
            .keys()
            .filter(|name| !super::residency::is_auxiliary(name))
            .filter_map(|name| super::routed_expert_tensor(name).map(|(layer, _)| layer))
            .collect();
        if layers.is_empty() {
            return Ok(None);
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
            return Ok(None);
        }
        let mut bank_bytes = 0u64;
        for name in layers.iter().flat_map(|&layer| names(layer)) {
            bank_bytes = bank_bytes
                .checked_add(self.descriptor(&name)?.runtime_bytes()?)
                .ok_or_else(|| err("Qwen4 direct expert bank size overflow"))?;
        }
        Ok(Some((layers, bank_bytes)))
    }

    /// A store the host gate refused at load takes the whole bank set a load
    /// would have taken, when the same gate admits it now with this model's
    /// `own` bytes taken out of the host's non-reclaimable number. Run with
    /// no turn open. Layer by layer, a partial bank goes before its whole
    /// bank is allocated, and the direct fill starts. False when the store is
    /// not such a partial of a whole plan whose cap still holds every bank,
    /// or the gate refuses, or `admit` refuses the move's growth (the whole
    /// banks less the partial banks, plus a layer's staging) before any bank
    /// drops. On a refused allocation after that the layers it reached lose
    /// their banks and take new partial slots at their next lease.
    pub(in super::super) fn promote_whole_with(
        &mut self,
        own: u64,
        gate: &mut impl FnMut(&mut super::super::memory::Plan, u64, u64, u64) -> Result<()>,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<bool> {
        if self.plan.direct
            || !self.plan.partial_direct_gate
            || !matches!(self.live.fallback(), Some(Fallback::LoadGate { .. }))
            || self.gguf_source.is_none()
        {
            return Ok(false);
        }
        let Some(store) = self.store_plan.filter(|store| store.whole) else {
            return Ok(false);
        };
        let experts = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0) as usize;
        if experts == 0 {
            return Ok(false);
        }
        let Some((layers, bank_bytes)) = self.direct_bank_set(experts)? else {
            return Ok(false);
        };
        if store.expert_cap_bytes < bank_bytes {
            return Ok(false);
        }
        let mut candidate = self.plan.clone();
        candidate.self_mlx_bytes = (crate::array::memory::get_active_memory()
            + crate::array::memory::get_cache_memory()) as u64;
        if gate(&mut candidate, own, self.plan.anonymous_need, bank_bytes).is_err() {
            return Ok(false);
        }
        let mut partial = Vec::with_capacity(layers.len());
        for layer in &layers {
            partial.push(match self.slots.get(layer) {
                Some(old) => Self::bank_bytes_of(old)?,
                None => 0,
            });
        }
        // The last layer's admission below is the largest; a refusal there
        // would cost every warm slot the loop already dropped.
        super::super::memory::maintain_freelist(self.plan.physical_bytes);
        let growth = bank_bytes.saturating_sub(partial.iter().sum());
        if admit(growth.saturating_add(MAX_READ_BYTES * 3)).is_err() {
            self.live.promotion_refusals.fetch_add(1, Ordering::AcqRel);
            return Ok(false);
        }
        let draft_bank = self.mtp_bank_estimate()?;
        self.complete_device_readers()?;
        if let Some(fill) = self.slot_fill.as_mut()
            && self.pending_fill.take().is_some()
        {
            // Its reads still write the ring.
            fill.pool.recover();
        }
        self.pending_fill = None;
        self.preloaded.clear();
        let mut moved = Vec::with_capacity(layers.len());
        let mut allocated = Ok(());
        for (&layer, &old_bytes) in layers.iter().zip(&partial) {
            if self.slots.remove(&layer).is_some() {
                self.forget_bank_bytes(old_bytes);
            }
            let before = self.slot_bytes;
            match self.new_expert_slots(layer, experts, experts, admit) {
                Ok(Some(slots)) => {
                    self.slots.insert(layer, slots);
                    moved.push((layer, self.slot_bytes.saturating_sub(before)));
                }
                Ok(None) => {
                    allocated = Err(err("Qwen4 direct slots require quantized scales"));
                    break;
                }
                Err(error) => {
                    allocated = Err(error);
                    break;
                }
            }
        }
        if let Err(error) = allocated {
            for (layer, bytes) in moved {
                if self.slots.remove(&layer).is_some() {
                    self.forget_bank_bytes(bytes);
                }
            }
            return Err(error);
        }
        self.slot_fill = None;
        candidate.direct = true;
        candidate.partial_direct_gate = false;
        candidate.direct_bytes = bank_bytes;
        candidate.direct_layers = layers.len();
        candidate.partial_expert_slots = experts;
        self.plan = candidate;
        self.draft_bank_budget = draft_bank;
        self.slot_capacity = Some(experts);
        self.slot_bank_budget = None;
        self.direct_fill = None;
        self.direct_fill_remaining = 0;
        self.direct_error = None;
        self.direct_fill_pass = false;
        self.live.direct_filled_layers.store(0, Ordering::Release);
        self.live.store(&self.plan);
        self.live.clear_fallback();
        self.live.promotions.fetch_add(1, Ordering::AcqRel);
        tracing::info!(
            "qwen4 residency: host-gate partial store moved up to {} layers x {experts} experts, {:.1} GiB banks within a {:.1} GiB bank budget ({:.1} GiB live host memory beside this model's {:.1} GiB)",
            layers.len(),
            bank_bytes as f64 / (1u64 << 30) as f64,
            self.plan.bank_budget as f64 / (1u64 << 30) as f64,
            self.plan.live_non_reclaimable as f64 / (1u64 << 30) as f64,
            own as f64 / (1u64 << 30) as f64,
        );
        if let Err(error) = self.start_direct_fill() {
            tracing::warn!("qwen4 residency: direct fill starts at the next turn: {error}");
        }
        Ok(true)
    }

    fn bank_bytes_of(slots: &ExpertSlots) -> Result<u64> {
        slots
            .banks
            .iter()
            .try_fold(0u64, |n, w| w.bytes().map(|b| n.saturating_add(b)))
    }

    /// Takes the bytes of a dropped layer's banks out of the ledgers.
    fn forget_bank_bytes(&mut self, bytes: u64) {
        self.slot_bytes = self.slot_bytes.saturating_sub(bytes);
        self.cache_bytes = self.cache_bytes.saturating_sub(bytes);
    }

    /// The load's inventory from the descriptors, before any payload read.
    pub(in super::super) fn load_inventory(
        &self,
        c: &super::super::config::Config,
        vision: bool,
    ) -> Result<super::super::load_plan::LoadInventory> {
        let routed = self.routed_expert_bytes()?;
        let largest = self.largest_routed_expert_layer_bytes()?;
        let experts = c.num_experts.max(1) as u128;
        let minimum = (c.num_experts_per_tok as u128 * u128::from(routed.saturating_add(largest))
            / experts)
            .min(u128::from(routed)) as u64;
        let store_fixed_bytes = self
            .hot_bytes()?
            .saturating_sub(routed)
            .saturating_add(self.mtp_bank_estimate()?);
        let vision_bytes = if vision {
            self.tensors
                .iter()
                .filter(|(name, _)| name.starts_with("visual."))
                .try_fold(0u64, |n, (_, t)| {
                    Ok::<_, Error>(n.saturating_add((t.rows()? as u64) * (t.width()? as u64) * 2))
                })?
        } else {
            0
        };
        Ok(super::super::load_plan::LoadInventory {
            store_fixed_bytes,
            vision_bytes,
            experts: crate::engine::memory_plan::StreamedPart {
                all: routed,
                minimum,
            },
            rows: super::super::load_plan::row_cache_share(c, MAX_READ_BYTES * 6),
        })
    }

    /// Size the store from the memory plan, never from the host (Splash: the
    /// store starts at what the plan gives it and the host only gates its
    /// allocations): the weight budget holds the fixed weights with the draft
    /// banks, the expert cap and the row cache.
    pub(in super::super) fn set_store_plan(
        &mut self,
        store: crate::engine::memory_plan::StorePlan,
        store_fixed_bytes: u64,
    ) {
        self.store_plan = Some(store);
        self.cache_limit = store_fixed_bytes
            .saturating_add(store.expert_cap_bytes)
            .saturating_add(store.row_cache_bytes);
        self.plan.budget = self.cache_limit;
        self.slot_capacity = None;
    }

    /// A fixture's store plan: every expert whole beside a 2 GiB row cache.
    #[cfg(test)]
    pub(in super::super) fn plan_fixture_store(&mut self) -> Result<()> {
        let routed = self.routed_expert_bytes()?;
        let fixed = self
            .hot_bytes()?
            .saturating_sub(routed)
            .saturating_add(self.mtp_bank_estimate()?);
        self.set_store_plan(
            crate::engine::memory_plan::StorePlan {
                whole: true,
                expert_cap_bytes: routed,
                row_cache_bytes: 2 << 30,
            },
            fixed,
        );
        Ok(())
    }

    /// A fixture's streamed store: the load's store-plan branch, then
    /// `capacity` slots per layer in place of what the cap gives, and the
    /// fixed banks. Every slot starts empty.
    #[cfg(test)]
    pub(in super::super) fn stream_fixture_store(&mut self, capacity: usize) -> Result<()> {
        let routed = self.routed_expert_bytes()?;
        let fixed = self
            .hot_bytes()?
            .saturating_sub(routed)
            .saturating_add(self.mtp_bank_estimate()?);
        self.set_store_plan(
            crate::engine::memory_plan::StorePlan {
                whole: false,
                expert_cap_bytes: routed,
                row_cache_bytes: 2 << 30,
            },
            fixed,
        );
        let experts = self
            .metadata
            .get("qwen4exp.expert_count")
            .and_then(GgufMetaValue::as_u64)
            .unwrap_or(0) as usize;
        let streamed = self.select_direct_residency_with(
            experts,
            0,
            &mut |_, _, _| Err(err("a streamed store plan never asks the host gate")),
            &mut |_| Err(err("a streamed store allocates no full banks")),
        )?;
        if !streamed || self.plan.direct {
            return Err(err("Qwen4 fixture store did not take streamed experts"));
        }
        self.slot_capacity = Some(capacity);
        self.prepare_fixture_hot()
    }

    /// The expert each slot of `layer` holds; `None` before the layer's
    /// first fill.
    #[cfg(test)]
    pub(in super::super) fn slot_occupants(&self, layer: usize) -> Option<Vec<Option<u32>>> {
        self.slots.get(&layer).map(|slots| slots.occupants.clone())
    }

    /// The bytes of every slot bank, the slot count each layer's leases
    /// size new banks at, and whether the slot fills' readers are up.
    #[cfg(test)]
    pub(in super::super) fn slot_ledger(&self) -> (u64, Option<usize>, bool) {
        (
            self.slot_bytes,
            self.slot_capacity,
            self.slot_fill.is_some(),
        )
    }

    /// Each slot's age and the tick of `layer`; `None` before its first fill.
    #[cfg(test)]
    pub(in super::super) fn slot_ages(&self, layer: usize) -> Option<(Vec<u64>, u64)> {
        self.slots
            .get(&layer)
            .map(|slots| (slots.ages.clone(), slots.tick))
    }

    /// The victims of the fill read ahead of its lease.
    #[cfg(test)]
    pub(in super::super) fn pending_victims(&self) -> Option<Vec<usize>> {
        self.pending_fill.as_ref().map(|fill| fill.chosen.clone())
    }

    /// The read bytes of each batch of the fill read ahead of its lease.
    #[cfg(test)]
    pub(in super::super) fn pending_batch_bytes(&self) -> Vec<u64> {
        self.pending_fill.as_ref().map_or_else(Vec::new, |fill| {
            fill.batches
                .iter()
                .map(|batch| batch.reads.iter().map(|read| read.len as u64).sum())
                .collect()
        })
    }

    /// Bytes the slot fills' readers were handed so far.
    #[cfg(test)]
    pub(in super::super) fn slot_fill_submitted_bytes(&self) -> u64 {
        self.slot_fill
            .as_ref()
            .map_or(0, |fill| fill.pool.submitted_bytes())
    }

    /// The bits of every bank plane's rows of `slots` in `layer`.
    #[cfg(test)]
    pub(in super::super) fn slot_rows_bits(&self, layer: usize, slots: &[u32]) -> Vec<u32> {
        let Some(state) = self.slots.get(&layer) else {
            return Vec::new();
        };
        let capacity = state.occupants.len() as i64;
        let mut bits = Vec::new();
        for bank in &state.banks {
            for plane in [
                Some(&bank.values),
                bank.scales.as_ref(),
                bank.biases.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                let rows = plane.shape().unwrap()[0] / capacity;
                for &slot in slots {
                    let lo = slot as i64 * rows;
                    let part = plane.slice_axis(0, lo, lo + rows).unwrap();
                    if part.dtype().unwrap() == DType::Uint32 {
                        bits.extend(part.to_uint32().unwrap().iter());
                    } else {
                        let wide = part.astype(DType::Float32).unwrap();
                        bits.extend(wide.to_float32().unwrap().iter().map(|v| v.to_bits()));
                    }
                }
            }
        }
        bits
    }

    #[cfg(test)]
    pub(in super::super) fn slot_banks(&self, layer: usize) -> Option<[Arc<Weight>; 3]> {
        self.slots.get(&layer).map(|slots| slots.banks.clone())
    }

    /// Evict `expert` from `layer` as a fill would: its readers complete,
    /// its slot empties and the device map is rebuilt at its next use.
    /// Returns the emptied slot. Slot 0 must stay filled for a device pass.
    #[cfg(test)]
    pub(in super::super) fn evict_for_test(&mut self, layer: usize, expert: u32) -> Result<usize> {
        let slots = self
            .slots
            .get_mut(&layer)
            .ok_or_else(|| err("Qwen4 eviction of a layer with no slots"))?;
        let slot = slots
            .mapping
            .get(expert as usize)
            .copied()
            .flatten()
            .ok_or_else(|| err("Qwen4 eviction of an expert that holds no slot"))?;
        MxArray::eval_arrays_with_context(
            &slots.readers.iter().collect::<Vec<_>>(),
            "qwen4::slots::evict_for_test",
        )?;
        slots.readers.clear();
        slots.mapping[expert as usize] = None;
        slots.occupants[slot] = None;
        slots.ages[slot] = 0;
        slots.device_mapping = None;
        Ok(slot)
    }

    /// The store's planned bytes above what is live (Splash
    /// `hostHeldBackBytes`): the engine plan counts the store at its cap.
    pub(in super::super) fn held_back_bytes(&self) -> u64 {
        self.cache_limit.saturating_sub(self.cache_bytes)
    }

    /// The bytes the store's ledger holds: banks, slots and cached rows.
    pub(in super::super) fn ledgered_bytes(&self) -> u64 {
        self.cache_bytes
    }

    /// Give `bytes` of a streamed store's cap back before any slot exists;
    /// false when the experts are whole or a slot was allocated.
    pub(in super::super) fn lower_store_cap(&mut self, bytes: u64) -> Result<bool> {
        let Some(mut store) = self.store_plan else {
            return Ok(false);
        };
        if self.plan.direct || !self.slots.is_empty() || store.expert_cap_bytes < bytes {
            return Ok(false);
        }
        store.expert_cap_bytes -= bytes;
        self.store_plan = Some(store);
        self.cache_limit = self.cache_limit.saturating_sub(bytes);
        self.plan.budget = self.cache_limit;
        if let Some(budget) = self.slot_bank_budget {
            let routed = self.slot_routed_bytes()?;
            let largest = self.largest_routed_layer_bytes()?;
            self.slot_bank_budget = Some(budget.min(slot_bank_budget_for(
                store.expert_cap_bytes,
                routed,
                largest,
            )));
            self.slot_capacity = None;
            if let (Some(experts), Some(top)) = (
                self.metadata
                    .get("qwen4exp.expert_count")
                    .and_then(GgufMetaValue::as_u64),
                self.metadata
                    .get("qwen4exp.expert_used_count")
                    .and_then(GgufMetaValue::as_u64),
            ) {
                self.plan.partial_expert_slots =
                    self.expert_slot_capacity(experts as usize, top as usize)?;
            }
        }
        self.live.store(&self.plan);
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

    fn wait_direct_layer(&mut self, layer: usize) -> Result<()> {
        #[cfg(test)]
        self.live.fill_hold.release();
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

    /// The routed experts run on per-layer slots that can miss: a partial
    /// GGUF store whose slot count is settled above zero.
    pub(in super::super) fn slot_store(&self) -> bool {
        self.gguf
            && !self.plan.direct
            && self.slot_capacity.is_some_and(|slots| slots > 0)
            && direct_residency_available()
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

    /// Resumes any unpublished direct layers before a turn. The banks stay
    /// allocated and non-volatile until the model is released (Splash
    /// `ExpertStore` has no purgeable state and no timed purge).
    pub(in super::super) fn begin_turn(&mut self) -> Result<()> {
        self.start_direct_fill()?;
        self.direct_fill_pass =
            self.plan.direct && self.slots.values().any(|slots| !slots.identity_ready());
        Ok(())
    }

    pub(in super::super) fn expert_slot_capacity(
        &mut self,
        experts: usize,
        top: usize,
    ) -> Result<usize> {
        if self.full_residency() {
            return Ok(experts);
        }
        if !self.gguf || !direct_residency_available() {
            return Ok(0);
        }
        if let Some(n) = self.slot_capacity {
            return Ok(n);
        }
        let routed = self.slot_routed_bytes()?;
        let largest_layer = self.largest_routed_layer_bytes()?;
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

    /// The layer of a routed main-network expert tensor in either format the
    /// store reads: GGUF `blk.N.ffn_{gate,up,down}_exps.weight` and
    /// transformers `layers.N.mlp.experts.{gate_up_proj,down_proj}` (Splash
    /// `gguf::streamedExperts` makes both expert planes, never fixed bytes).
    fn routed_expert_layer<'a>(&self, name: &'a str) -> Option<&'a str> {
        if super::residency::is_auxiliary(name) || self.banks.contains_key(name) {
            return None;
        }
        if let Some((layer, rest)) = name.split_once(".ffn_") {
            return rest.ends_with("_exps.weight").then_some(layer);
        }
        let (layer, projection) = name.split_once(".mlp.experts.")?;
        matches!(projection, "gate_up_proj" | "down_proj").then_some(layer)
    }

    /// Routed main-network expert bytes in every format the store reads.
    fn routed_expert_bytes(&self) -> Result<u64> {
        self.tensors
            .iter()
            .filter(|(name, _)| self.routed_expert_layer(name).is_some())
            .try_fold(0u64, |n, (_, t)| {
                t.runtime_bytes().map(|b| n.saturating_add(b))
            })
    }

    /// Routed main-network expert bytes of the largest layer, in every
    /// format the store reads.
    fn largest_routed_expert_layer_bytes(&self) -> Result<u64> {
        let mut layer_bytes = HashMap::<&str, u64>::new();
        for (name, tensor) in &self.tensors {
            if let Some(layer) = self.routed_expert_layer(name) {
                let bytes = layer_bytes.entry(layer).or_default();
                *bytes = bytes.saturating_add(tensor.runtime_bytes()?);
            }
        }
        Ok(layer_bytes.values().copied().max().unwrap_or(0))
    }

    /// Routed main-network expert bytes of the largest layer the slots serve.
    fn largest_routed_layer_bytes(&self) -> Result<u64> {
        let mut layer_bytes = HashMap::<&str, u64>::new();
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
        Ok(layer_bytes.values().copied().max().unwrap_or(0))
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
        let gpu_repack = gpu_repack_available();
        if !gpu_repack && !crate::accelerator::host_has_cuda() {
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
                        (!gpu_repack || tensor.offset.is_multiple_of(rule))
                            && tensor.row_bytes().is_ok_and(|n| {
                                n <= DIRECT_READ_BYTES as u64
                                    && (!gpu_repack || n.is_multiple_of(rule))
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
        let banks = allocate_slot_banks(self, &names, &prototypes, capacity)?;
        self.slot_bytes += bytes;
        self.cache_bytes += bytes;
        self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
        Ok(Some(ExpertSlots {
            banks,
            direct,
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
            let banks = slots.banks.clone();
            self.count_slot_hits(wanted.len() as u64);
            return Ok(Some((banks, ids.to_vec(), experts)));
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
        let preloaded = self.preloaded.remove(&layer).unwrap_or_default();
        let result = (|| -> Result<_> {
            let missing = Self::missing_experts(&slots, &wanted);
            let resident = wanted
                .iter()
                .filter(|e| !missing.contains(e) && !preloaded.contains(e))
                .count();
            self.count_slot_hits(resident as u64);
            self.count_slot_misses(missing.len() as u64);
            #[cfg(test)]
            {
                let (leased, misses) = self.layer_leases.entry(layer).or_default();
                leased.extend(&wanted);
                *misses += missing.len() as u64;
                for &e in &missing {
                    *self.expert_fills.entry((layer, e)).or_default() += 1;
                }
            }
            self.load_experts(&mut slots, &wanted, &missing)?;
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

    /// The experts of `order` that hold a slot of `layer`, in `order`.
    pub(in super::super) fn experts_in_slots(&self, layer: usize, order: &[u32]) -> Vec<u32> {
        let Some(slots) = self.slots.get(&layer).filter(|_| !self.plan.direct) else {
            return Vec::new();
        };
        order
            .iter()
            .copied()
            .filter(|&e| slots.mapping.get(e as usize).copied().flatten().is_some())
            .collect()
    }

    /// The experts of `order` that hold a slot of `layer` become the newest,
    /// the last of `order` newest of all.
    pub(in super::super) fn touch_in_order(&mut self, layer: usize, order: &[u32]) {
        let Some(slots) = self.slots.get_mut(&layer) else {
            return;
        };
        for &e in order {
            let slot = slots.mapping.get(e as usize).copied().flatten();
            if let Some(age) = slot.and_then(|slot| slots.ages.get_mut(slot)) {
                slots.tick += 1;
                *age = slots.tick;
            }
        }
    }

    /// The experts of `wanted` that hold no slot, in id order.
    fn missing_experts(slots: &ExpertSlots, wanted: &HashSet<u32>) -> Vec<u32> {
        let mut missing: Vec<u32> = wanted
            .iter()
            .filter(|&&e| slots.mapping[e as usize].is_none())
            .copied()
            .collect();
        missing.sort_unstable();
        missing
    }

    /// Reads `missing` into the least recently used slots whose occupants
    /// are not `wanted`, in one fill. The layer's readers complete first and
    /// its device map is rebuilt at its next use: a kernel that read the old
    /// map would send a loaded expert to the sentinel or to the old
    /// occupant's slot. The fill waits for its own writes, which the GPU
    /// runs after the work queued before them.
    fn load_experts(
        &mut self,
        slots: &mut ExpertSlots,
        wanted: &HashSet<u32>,
        missing: &[u32],
    ) -> Result<()> {
        if missing.is_empty() {
            return Ok(());
        }
        slots.device_mapping = None;
        MxArray::eval_arrays_with_context(
            &slots.readers.iter().collect::<Vec<_>>(),
            "qwen4::slots::complete_readers",
        )?;
        slots.readers.clear();
        // All wanted slots, including newly loaded misses, remain protected
        // for this call. Duplicate requested ids resolve to one slot.
        #[cfg(test)]
        let chunk = if self.old_slot_fill { 8 } else { missing.len() };
        #[cfg(not(test))]
        let chunk = missing.len();
        for missing in missing.chunks(chunk) {
            let chosen = Self::victims(slots, wanted, missing)?;
            if let Err(error) = self.fill_direct_chunk(slots, missing, &chosen) {
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
        Ok(())
    }

    /// The slots `missing` take: for each, among the slots not taken yet
    /// whose occupant is not `wanted`, an empty one first, else the least
    /// recently used.
    fn victims(slots: &ExpertSlots, wanted: &HashSet<u32>, missing: &[u32]) -> Result<Vec<usize>> {
        let mut chosen = Vec::with_capacity(missing.len());
        for _ in missing {
            let slot = slots
                .occupants
                .iter()
                .enumerate()
                .filter(|(s, e)| !chosen.contains(s) && e.is_none_or(|e| !wanted.contains(&e)))
                .min_by_key(|(s, e)| if e.is_none() { 0 } else { slots.ages[*s] + 1 })
                .map(|(s, _)| s)
                .ok_or_else(|| err("Qwen4 expert slot reservation exhausted"))?;
            chosen.push(slot);
        }
        Ok(chosen)
    }

    /// Queues the reads of the experts of `ids` that hold no slot of
    /// `layer` for the victims a lease of `ids` would take now, and touches
    /// no bank, mapping or age: the lease of `ids` finishes the fill after
    /// the layer's readers completed, so the GPU computes the readers queued
    /// before it while the reads run. False when nothing is read ahead.
    pub(in super::super) fn begin_expert_fill(
        &mut self,
        layer: usize,
        experts: usize,
        top: usize,
        ids: &[u32],
    ) -> Result<bool> {
        #[cfg(test)]
        if self.old_slot_fill || self.no_fill_ahead {
            return Ok(false);
        }
        let capacity = self.expert_slot_capacity(experts, top)?;
        let wanted: HashSet<u32> = ids.iter().copied().collect();
        if self.plan.direct
            || capacity == 0
            || wanted.len() > capacity
            || ids.iter().any(|&e| e as usize >= experts)
        {
            return Ok(false);
        }
        let Some(slots) = self.slots.get(&layer) else {
            return Ok(false);
        };
        let missing = Self::missing_experts(slots, &wanted);
        if missing.is_empty() {
            return Ok(false);
        }
        let chosen = Self::victims(slots, &wanted, &missing)?;
        let started = Instant::now();
        let (source, descriptors) = self.slot_source(slots)?;
        self.start_slot_fill(&source)?;
        let fill = self
            .slot_fill
            .as_mut()
            .ok_or_else(|| err("Qwen4 slot fill lost its readers"))?;
        if self.pending_fill.take().is_some() {
            // Its reads still write the ring.
            fill.pool.recover();
        }
        let slots = self
            .slots
            .get_mut(&layer)
            .ok_or_else(|| err("Qwen4 expert slots disappeared"))?;
        let (requests, batches, mut stats) =
            Self::plan_slot_fill(fill, &source, &descriptors, slots, &missing, &chosen)?;
        #[cfg(test)]
        let staged = fill.ring.len();
        let first_ticket = fill.begin(&source, &requests, &batches, slots)?;
        #[cfg(test)]
        {
            self.slot_fill_builds[1] += fill.ring.len() - staged;
        }

        stats.ahead_micros = started.elapsed().as_micros() as u64;
        self.pending_fill = Some(PendingFill {
            bank: slots.names[0].clone(),
            experts: missing,
            chosen,
            requests,
            batches,
            stats,
            first_ticket,
        });
        Ok(true)
    }

    /// Loads the experts of `ids` that hold no slot into `layer`'s filled
    /// slots before a device pass runs it again, as `expert_slots` would,
    /// with every id of `ids` protected. A read counts as a miss and no age
    /// moves: the pass's commit (or a host route) counts and ticks the
    /// routed picks, and leaves the ones read here out of its hits until
    /// `end_preloads`. False when the layer cannot hold `ids` in slots.
    pub(in super::super) fn preload_expert_slots(
        &mut self,
        layer: usize,
        experts: usize,
        top: usize,
        ids: &[u32],
    ) -> Result<bool> {
        let capacity = self.expert_slot_capacity(experts, top)?;
        let wanted: HashSet<u32> = ids.iter().copied().collect();
        if self.plan.direct || capacity == 0 || wanted.len() > capacity {
            return Ok(false);
        }
        if ids.iter().any(|&e| e as usize >= experts) {
            return Err(err("Qwen4 expert id exceeds slot mapping"));
        }
        let Some(mut slots) = self.slots.remove(&layer) else {
            return Ok(false);
        };
        let missing = Self::missing_experts(&slots, &wanted);
        self.count_slot_misses(missing.len() as u64);
        let loaded = self.load_experts(&mut slots, &wanted, &missing);
        self.slots.insert(layer, slots);
        loaded?;
        self.preloaded.entry(layer).or_default().extend(missing);
        Ok(true)
    }

    /// The decode token that preloaded experts has ended: later routes count
    /// them as hits again.
    pub(in super::super) fn end_preloads(&mut self) {
        self.preloaded.clear();
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
            self.slots
                .get(&layer)
                .is_some_and(|slots| slots.occupants.first().is_some_and(Option::is_some))
        })
    }

    /// Every one of `layers` holds every expert in a slot: no route of a
    /// prefill window can miss.
    pub(in super::super) fn slots_hold_every_expert(&self, layers: usize) -> bool {
        (0..layers).all(|layer| {
            self.slots
                .get(&layer)
                .is_some_and(|slots| slots.mapping.iter().all(Option::is_some))
        })
    }

    /// A store that can miss never runs a tentative prefill window: a miss
    /// fails a group of windows, and a one-window miss runs the window twice.
    pub(in super::super) fn prefill_device_slots_ready(&self, layers: usize) -> bool {
        if self.plan.direct {
            return (0..layers).all(|layer| {
                self.slots
                    .get(&layer)
                    .is_some_and(|slots| slots.direct && fused_bank_modes(&slots.banks))
            });
        }
        self.slots_hold_every_expert(layers)
            && (0..layers).all(|layer| {
                self.slots
                    .get(&layer)
                    .is_some_and(|slots| fused_bank_modes(&slots.banks))
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
                self.count_slot_hits(selected.iter().collect::<HashSet<_>>().len() as u64);
            }
            return Ok(true);
        }
        if self.first_unmapped(entries).is_some() {
            return Ok(false);
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
            let preloaded = self.preloaded.remove(&layer).unwrap_or_default();
            let resident = selected
                .iter()
                .filter(|e| !preloaded.contains(e))
                .collect::<HashSet<_>>()
                .len();
            self.count_slot_hits(resident as u64);
        }
        Ok(true)
    }

    /// The layer of the first entry that names an expert with no slot; no
    /// age or count moves.
    pub(in super::super) fn first_unmapped(&self, entries: &[(usize, &[u32])]) -> Option<usize> {
        entries
            .iter()
            .find(|&&(layer, selected)| {
                let Some(slots) = self.slots.get(&layer) else {
                    return true;
                };
                if self.plan.direct {
                    return !slots.identity_ready();
                }
                selected.iter().any(|&e| {
                    slots
                        .mapping
                        .get(e as usize)
                        .copied()
                        .flatten()
                        .is_none_or(|slot| slot >= slots.ages.len())
                })
            })
            .map(|&(layer, _)| layer)
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
