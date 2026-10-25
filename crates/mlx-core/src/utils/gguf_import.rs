//! In-memory GGUF import: the file is read in aligned windows into a small
//! ring of staging buffers, and every band of rows is repacked (GPU kernels,
//! or the pure-Rust CPU fallback) straight into the final weight buffers. The
//! resulting map equals `load_gguf_tensors` with the plan's Tiled64 tensors
//! permuted like `tile_kquant_weights_for_disk`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::array::{DType, MxArray, clear_cache, synchronize};
use crate::stream::{DeviceType, Stream};
#[cfg(test)]
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufTensorInfo, GgufTensorType, convert_tensor_dtype,
    extract_clef_head_assets, fixup_gemma4_mmproj_layout, fixup_gemma4v_mmproj_layout,
    fixup_qwen35_linear_attn, fixup_qwen35_moe_shared_expert_gate, fixup_shapes,
    gguf_name_to_hf_for_metadata, load_gguf_tensor, remap_keys,
};
use crate::utils::gguf_cpu_repack::{
    copy_dense_band_into, cpu_kquant_band_into, decode_affine_band_into,
};
use crate::utils::gguf_gpu::{
    GpuRepackKind, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available, gpu_repack_band,
};
use crate::utils::gguf_plan::{ImportOptions, ImportPlan};
use crate::utils::gguf_read::{
    ALIGN, CHUNK_BYTES, PageCacheMode, READ_THREADS, ReadAt, Reader, SLOT_BYTES, SLOTS, Window,
    log_load_start, maybe_log_slow_disk, plan_reads, sampled_resident_fraction, set_nocache,
};
use crate::utils::gguf_source::{GgufFileHandle, GgufFileIdentity, GgufIdentity, GgufSourceSet};
use crate::utils::host::physical_memory;

pub(crate) const CPU_THREADS: usize = 8;
/// Staged source bytes the CPU fallback may have queued or running.
pub(crate) const CPU_INFLIGHT_BYTES: usize = 256 << 20;
/// Source bytes of one CPU job.
pub(crate) const CPU_BAND_BYTES: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepackBackend {
    Auto,
    #[cfg(test)]
    CpuOnly,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LoadStats {
    /// Bytes read from the file (whole aligned windows).
    pub bytes: u64,
    /// Time the model thread spent waiting on reads.
    pub read_secs: f64,
    pub preload_secs: f64,
    pub nocache: bool,
    #[cfg(test)]
    pub resident: Option<f64>,
    pub gpu_tensors: usize,
    pub cpu_tensors: usize,
}

pub(crate) struct RawTensors {
    pub map: HashMap<String, MxArray>,
    /// `(gguf name, shards, rows per shard)` for the plan's `ple_shard`.
    pub ple_shards: Option<(String, Vec<MxArray>, usize)>,
    pub stats: LoadStats,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadConfig {
    pub slot_bytes: usize,
    pub slots: usize,
    pub read_threads: usize,
    pub chunk_bytes: usize,
    pub cpu_threads: usize,
    pub cpu_inflight_bytes: usize,
    pub cpu_band_bytes: usize,
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self {
            slot_bytes: SLOT_BYTES,
            slots: SLOTS,
            read_threads: READ_THREADS,
            chunk_bytes: CHUNK_BYTES,
            cpu_threads: CPU_THREADS,
            cpu_inflight_bytes: CPU_INFLIGHT_BYTES,
            cpu_band_bytes: CPU_BAND_BYTES,
        }
    }
}

/// What a test can observe of one import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ImportEvent {
    /// The read of window `w` was issued into its slot.
    Submitted(usize),
    /// Window `w`'s GPU work and CPU jobs finished; its slot may be refilled.
    Completed(usize),
    Stopped,
    ReadersIdle,
    CpuJoined,
    GpuSynced,
    Freed,
    CacheCleared,
}

/// Test hooks and counters of one import; `Default` observes nothing.
#[derive(Default)]
pub(crate) struct ImportProbe {
    pub record: bool,
    pub events: Mutex<Vec<ImportEvent>>,
    /// Set right before the staging and destination buffers are released.
    pub freed: AtomicBool,
    pub max_alloc_bytes: AtomicUsize,
    pub max_cpu_inflight: AtomicUsize,
    pub gpu_error_at_window: Option<usize>,
    pub panic_at_window: Option<usize>,
    /// `(tensor name, byte offset in the tensor, bit)` flipped in the staged
    /// bytes after the read, before any repack.
    pub flip: Option<(String, usize, u8)>,
}

impl ImportProbe {
    fn event(&self, e: ImportEvent) {
        if self.record {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(e);
        }
    }

    fn alloc(&self, bytes: usize) {
        self.max_alloc_bytes.fetch_max(bytes, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Gpu(GpuRepackKind, OutLayout),
    CpuKQuant(OutLayout),
    CpuAffine,
    Dense(DType),
    Ple(DType, usize),
    /// Rare types with no band repacker: the tensor's bytes are gathered and
    /// handed to `load_gguf_tensor` on the model thread.
    Whole,
}

impl Route {
    fn row_multiple(self) -> usize {
        match self {
            Route::Gpu(_, OutLayout::Tiled64) | Route::CpuKQuant(OutLayout::Tiled64) => 64,
            _ => 1,
        }
    }
}

fn dense_dtype(t: GgufTensorType) -> Option<DType> {
    match t {
        GgufTensorType::F32 => Some(DType::Float32),
        GgufTensorType::F16 => Some(DType::Float16),
        GgufTensorType::BF16 => Some(DType::BFloat16),
        _ => None,
    }
}

fn route_for(t: &GgufTensorInfo, plan: &ImportPlan, gpu: bool) -> Result<Route> {
    let ty = t.tensor_type;
    if let Some((name, rows)) = &plan.ple_shard
        && *name == t.name
    {
        let dtype = dense_dtype(ty).filter(|_| t.dims.len() == 2 && *rows > 0);
        return dtype.map(|d| Route::Ple(d, *rows)).ok_or_else(|| {
            Error::from_reason(format!(
                "GGUF tensor '{}' ({}, dims {:?}) cannot be loaded as row shards",
                t.name,
                ty.name(),
                t.dims
            ))
        });
    }
    let tiled = plan.tiled_gguf_names.contains(&t.name);
    if ty.k_quant_format().is_some() && plan.import_k_quants {
        let layout = if tiled {
            OutLayout::Tiled64
        } else {
            OutLayout::RowMajor
        };
        return Ok(match gpu_kind_for(ty).filter(|_| gpu) {
            Some(kind) => Route::Gpu(kind, layout),
            None => Route::CpuKQuant(layout),
        });
    }
    if tiled {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' of type {} is planned Tiled64 but is not an imported K-quant",
            t.name,
            ty.name()
        )));
    }
    if let Some(kind) = gpu_kind_for(ty).filter(|_| gpu) {
        return Ok(Route::Gpu(kind, OutLayout::RowMajor));
    }
    Ok(match ty {
        GgufTensorType::Q4_0
        | GgufTensorType::Q4_1
        | GgufTensorType::Q5_0
        | GgufTensorType::Q5_1
        | GgufTensorType::Q8_0
        | GgufTensorType::MXFP4 => Route::CpuAffine,
        _ => match dense_dtype(ty) {
            Some(d) => Route::Dense(d),
            None => Route::Whole,
        },
    })
}

/// A tensor's destination while its bands arrive.
enum Dst {
    Empty,
    Consumed,
    Repack {
        dst: Option<RepackDst>,
        ptrs: DstPtrs,
    },
    Dense {
        array: MxArray,
        ptr: SendPtr<u8>,
    },
    Ple {
        shards: Vec<(MxArray, SendPtr<u8>)>,
        rows_per_shard: usize,
    },
    Whole(Vec<u8>),
    Done(Vec<(String, MxArray)>),
}

struct TensorState {
    route: Route,
    bands_left: usize,
    dst: Dst,
}

/// Owns the staging ring and every destination buffer of one file's import.
/// Declared before the reader / CPU worker scope, so on any exit (error,
/// panic, success) it drops only after the scope has stopped the readers,
/// waited for their issued reads and joined the CPU jobs ([`Quiesce`]); it
/// then waits for the GPU stream before releasing anything.
struct ImportGuard<'p> {
    slots: Vec<(MxArray, SendPtr<u8>)>,
    states: Vec<TensorState>,
    gpu: bool,
    consuming: bool,
    probe: &'p ImportProbe,
}

impl Drop for ImportGuard<'_> {
    fn drop(&mut self) {
        if self.gpu {
            Stream::default(DeviceType::Gpu).synchronize();
        } else if self.consuming {
            // Consumers use the default device even when source repacking
            // ran entirely on the CPU (including CPU-only builds).
            synchronize();
        }
        self.probe.event(ImportEvent::GpuSynced);
        self.probe.freed.store(true, Ordering::Release);
        self.slots.clear();
        self.states.clear();
        self.probe.event(ImportEvent::Freed);
        clear_cache();
        self.probe.event(ImportEvent::CacheCleared);
    }
}

/// First local of the scope, so it drops before the scope's threads are
/// joined: stop the readers, wait for every issued read, join the CPU jobs.
struct Quiesce<'a, 's> {
    reader: &'a Reader<'s>,
    pool: &'a CpuPool,
    probe: &'a ImportProbe,
}

impl Drop for Quiesce<'_, '_> {
    fn drop(&mut self) {
        self.reader.stop();
        self.probe.event(ImportEvent::Stopped);
        self.reader.wait_all_issued();
        self.probe.event(ImportEvent::ReadersIdle);
        self.pool.stop_and_join();
        self.probe.event(ImportEvent::CpuJoined);
    }
}

enum CpuWork {
    KQuant {
        info: GgufTensorInfo,
        src: SendPtr<u8>,
        len: usize,
        row0: usize,
        rows: usize,
        dst: DstPtrs,
    },
    Affine {
        ty: GgufTensorType,
        src: SendPtr<u8>,
        len: usize,
        row0: usize,
        rows: usize,
        dst: DstPtrs,
    },
    Copy {
        src: SendPtr<u8>,
        len: usize,
        dst: SendPtr<u8>,
        dst_off: usize,
    },
}

impl CpuWork {
    fn run(&self) -> Result<()> {
        // SAFETY (all arms): `src..src+len` lies in a staging slot whose read
        // finished before the job was queued and which is not refilled or
        // freed before the job's window completes or the pool is joined;
        // the destination rows are this job's alone.
        match self {
            CpuWork::KQuant {
                info,
                src,
                len,
                row0,
                rows,
                dst,
            } => {
                let blocks = unsafe { std::slice::from_raw_parts(src.0, *len) };
                cpu_kquant_band_into(info, blocks, *row0, *rows, dst)
            }
            CpuWork::Affine {
                ty,
                src,
                len,
                row0,
                rows,
                dst,
            } => {
                let blocks = unsafe { std::slice::from_raw_parts(src.0, *len) };
                decode_affine_band_into(*ty, blocks, *row0, *rows, dst.k, dst)
            }
            CpuWork::Copy {
                src,
                len,
                dst,
                dst_off,
            } => {
                let bytes = unsafe { std::slice::from_raw_parts(src.0, *len) };
                unsafe { copy_dense_band_into(bytes, *dst, *dst_off) };
                Ok(())
            }
        }
    }
}

struct CpuJob {
    window: usize,
    bytes: usize,
    work: CpuWork,
}

#[derive(Default)]
struct PoolState {
    queue: VecDeque<CpuJob>,
    pending: HashMap<usize, usize>,
    inflight: usize,
    running: usize,
    closed: bool,
    error: Option<String>,
}

struct PoolShared {
    state: Mutex<PoolState>,
    changed: Condvar,
}

impl PoolShared {
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'g>(&self, g: MutexGuard<'g, PoolState>) -> MutexGuard<'g, PoolState> {
        self.changed.wait(g).unwrap_or_else(PoisonError::into_inner)
    }
}

/// CPU fallback workers in the caller's thread scope. Jobs never call MLX.
struct CpuPool {
    shared: Arc<PoolShared>,
    cap: usize,
}

impl CpuPool {
    fn start<'s>(scope: &'s std::thread::Scope<'s, '_>, threads: usize, cap: usize) -> Self {
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState::default()),
            changed: Condvar::new(),
        });
        for _ in 0..threads.max(1) {
            let shared = Arc::clone(&shared);
            scope.spawn(move || cpu_worker(&shared));
        }
        Self {
            shared,
            cap: cap.max(1),
        }
    }

    fn error(state: &PoolState) -> Result<()> {
        match &state.error {
            Some(e) => Err(Error::from_reason(e.clone())),
            None => Ok(()),
        }
    }

    /// Queue `job`, first waiting until the in-flight source bytes leave room
    /// for it. Returns the in-flight bytes after queueing.
    fn submit(&self, job: CpuJob) -> Result<usize> {
        let mut state = self.shared.lock();
        while state.error.is_none() && state.inflight > 0 && state.inflight + job.bytes > self.cap {
            state = self.shared.wait(state);
        }
        Self::error(&state)?;
        state.inflight += job.bytes;
        *state.pending.entry(job.window).or_default() += 1;
        state.queue.push_back(job);
        let inflight = state.inflight;
        drop(state);
        self.shared.changed.notify_all();
        Ok(inflight)
    }

    fn wait_window(&self, window: usize) -> Result<()> {
        let mut state = self.shared.lock();
        while state.pending.get(&window).is_some_and(|&n| n > 0) {
            state = self.shared.wait(state);
        }
        state.pending.remove(&window);
        Self::error(&state)
    }

    /// Drop every queued job and wait for the running ones.
    fn stop_and_join(&self) {
        let mut state = self.shared.lock();
        while let Some(job) = state.queue.pop_front() {
            state.inflight -= job.bytes;
            if let Some(n) = state.pending.get_mut(&job.window) {
                *n -= 1;
            }
        }
        while state.running > 0 {
            state = self.shared.wait(state);
        }
        drop(state);
        self.shared.changed.notify_all();
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        self.stop_and_join();
        self.shared.lock().closed = true;
        self.shared.changed.notify_all();
    }
}

fn cpu_worker(shared: &PoolShared) {
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if let Some(job) = state.queue.pop_front() {
                    state.running += 1;
                    break job;
                }
                if state.closed {
                    return;
                }
                state = shared.wait(state);
            }
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.work.run()))
            .unwrap_or_else(|_| Err(Error::from_reason("GGUF CPU repack job panicked")));
        let mut state = shared.lock();
        state.running -= 1;
        state.inflight -= job.bytes;
        if let Some(n) = state.pending.get_mut(&job.window) {
            *n -= 1;
        }
        if let Err(e) = result
            && state.error.is_none()
        {
            state.error = Some(e.reason.clone());
        }
        drop(state);
        shared.changed.notify_all();
    }
}

/// A `Read + Seek` view of one tensor's gathered bytes at the absolute file
/// offset `base`, for `load_gguf_tensor`.
struct OffsetReader<'a> {
    base: u64,
    buf: &'a [u8],
    pos: usize,
}

impl Read for OffsetReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let rest = self.buf.get(self.pos..).unwrap_or_default();
        let n = rest.len().min(out.len());
        out[..n].copy_from_slice(&rest[..n]);
        self.pos += n;
        Ok(n)
    }
}

impl Seek for OffsetReader<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let abs = match pos {
            SeekFrom::Start(x) => Some(x),
            SeekFrom::Current(d) => (self.base + self.pos as u64).checked_add_signed(d),
            SeekFrom::End(d) => (self.base + self.buf.len() as u64).checked_add_signed(d),
        };
        let rel = abs
            .and_then(|a| a.checked_sub(self.base))
            .and_then(|r| usize::try_from(r).ok())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "seek outside the gathered tensor",
                )
            })?;
        self.pos = rel;
        Ok(self.base + rel as u64)
    }
}

fn alloc_dst(t: &GgufTensorInfo, route: Route, probe: &ImportProbe) -> Result<Dst> {
    Ok(match route {
        Route::Gpu(_, layout) | Route::CpuKQuant(layout) => {
            let (dst, ptrs) = match route {
                Route::Gpu(kind, _) => RepackDst::alloc_for_kind(t, kind, layout)?,
                _ => RepackDst::alloc(t, layout)?,
            };
            for a in std::iter::once(&dst.weight)
                .chain(dst.scales.as_ref())
                .chain(dst.biases.as_ref())
            {
                probe.alloc(a.nbytes());
            }
            Dst::Repack {
                dst: Some(dst),
                ptrs,
            }
        }
        Route::CpuAffine => alloc_dst(t, Route::CpuKQuant(OutLayout::RowMajor), probe)?,
        Route::Dense(dtype) => {
            let (array, ptr, bytes) = MxArray::alloc_uninit(&t.mlx_shape(), dtype)?;
            probe.alloc(bytes);
            Dst::Dense { array, ptr }
        }
        Route::Ple(dtype, rows_per_shard) => {
            let shape = t.mlx_shape();
            let (rows, cols) = (shape[0], shape[1]);
            let mut shards = Vec::new();
            let mut row0 = 0i64;
            while row0 < rows {
                let n = (rows - row0).min(rows_per_shard as i64);
                let (array, ptr, bytes) = MxArray::alloc_uninit(&[n, cols], dtype)?;
                probe.alloc(bytes);
                shards.push((array, ptr));
                row0 += n;
            }
            Dst::Ple {
                shards,
                rows_per_shard,
            }
        }
        Route::Whole => {
            let n = usize::try_from(t.data_size()?).map_err(|_| {
                Error::from_reason(format!("GGUF tensor '{}' does not fit in memory", t.name))
            })?;
            let mut buf = Vec::new();
            buf.try_reserve_exact(n).map_err(|e| {
                Error::from_reason(format!(
                    "Failed to allocate {n} bytes for GGUF tensor '{}': {e}",
                    t.name
                ))
            })?;
            buf.resize(n, 0);
            Dst::Whole(buf)
        }
    })
}

fn collect_tensor(
    t: &GgufTensorInfo,
    dst: Dst,
    map: &mut HashMap<String, MxArray>,
    ple_shards: &mut Option<(String, Vec<MxArray>, usize)>,
) -> Result<()> {
    match dst {
        Dst::Repack { dst: Some(dst), .. } => {
            let prefix = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            let RepackDst {
                weight,
                scales,
                biases,
            } = dst;
            map.insert(t.name.clone(), weight);
            if let Some(s) = scales {
                map.insert(format!("{prefix}.scales"), s);
            }
            if let Some(b) = biases {
                map.insert(format!("{prefix}.biases"), b);
            }
        }
        Dst::Dense { array, .. } => {
            map.insert(t.name.clone(), array);
        }
        Dst::Ple {
            shards,
            rows_per_shard,
        } => {
            *ple_shards = Some((
                t.name.clone(),
                shards.into_iter().map(|(a, _)| a).collect(),
                rows_per_shard,
            ));
        }
        Dst::Done(arrays) => map.extend(arrays),
        Dst::Consumed => {}
        Dst::Empty | Dst::Whole(_) | Dst::Repack { dst: None, .. } => {
            return Err(Error::from_reason(format!(
                "GGUF tensor '{}' was not fully read",
                t.name
            )));
        }
    }
    Ok(())
}

type TensorConsumer<'a> = dyn FnMut(HashMap<String, MxArray>) -> Result<()> + 'a;

/// Groups become available only after their last window's GPU and CPU work
/// completes. Moving their destinations out lets subsequent reads reuse the
/// source buffers after the model-specific consumer has finished with them.
struct ConsumeGroups<'a> {
    groups: Vec<Vec<usize>>,
    ready: Vec<Vec<usize>>,
    consume: &'a mut TensorConsumer<'a>,
}

impl ConsumeGroups<'_> {
    fn plan(&mut self, windows: &[Window], tensors: usize) -> Result<()> {
        let mut last = vec![0; tensors];
        for (w, window) in windows.iter().enumerate() {
            for band in &window.bands {
                last[band.tensor] = w;
            }
        }
        self.ready = vec![Vec::new(); windows.len()];
        let mut seen = HashSet::new();
        for (group, indices) in self.groups.iter().enumerate() {
            let mut end = 0;
            for &i in indices {
                if i >= tensors || !seen.insert(i) {
                    return Err(Error::from_reason("invalid GGUF consumer group"));
                }
                end = end.max(last[i]);
            }
            if !indices.is_empty() {
                self.ready[end].push(group);
            }
        }
        Ok(())
    }

    fn complete(&mut self, w: usize, header: &GgufFile, states: &mut [TensorState]) -> Result<()> {
        for group in std::mem::take(&mut self.ready[w]) {
            let mut map = HashMap::new();
            let mut ple = None;
            for &i in &self.groups[group] {
                if states[i].bands_left != 0 {
                    return Err(Error::from_reason(
                        "GGUF consumer received an incomplete tensor",
                    ));
                }
                let dst = std::mem::replace(&mut states[i].dst, Dst::Consumed);
                collect_tensor(&header.tensors[i], dst, &mut map, &mut ple)?;
            }
            if ple.is_some() {
                return Err(Error::from_reason(
                    "GGUF consumer cannot take row-sharded tensors",
                ));
            }
            (self.consume)(map)?;
        }
        Ok(())
    }
}

/// Read every tensor of `h` into its final arrays (model thread only).
pub(crate) fn read_gguf_tensors(
    h: &GgufFileHandle,
    plan: &ImportPlan,
    backend: RepackBackend,
    mode: PageCacheMode,
) -> Result<RawTensors> {
    read_gguf_tensors_with(
        &h.file,
        h,
        plan,
        backend,
        mode,
        &ReadConfig::default(),
        &ImportProbe::default(),
    )
}

/// [`read_gguf_tensors`] reading through `src` (the handle's file, or a test
/// double over it) with explicit ring / pool sizes.
pub(crate) fn read_gguf_tensors_with(
    src: &dyn ReadAt,
    h: &GgufFileHandle,
    plan: &ImportPlan,
    backend: RepackBackend,
    mode: PageCacheMode,
    cfg: &ReadConfig,
    probe: &ImportProbe,
) -> Result<RawTensors> {
    read_gguf_tensors_consuming(src, h, plan, backend, mode, cfg, probe, None)
}

fn read_gguf_tensors_consuming(
    src: &dyn ReadAt,
    h: &GgufFileHandle,
    plan: &ImportPlan,
    backend: RepackBackend,
    mode: PageCacheMode,
    cfg: &ReadConfig,
    probe: &ImportProbe,
    mut consumer: Option<&mut ConsumeGroups<'_>>,
) -> Result<RawTensors> {
    let started = Instant::now();
    let header: &GgufFile = &h.header;
    let gpu = backend == RepackBackend::Auto && gpu_repack_available();
    let mut routes = header
        .tensors
        .iter()
        .map(|t| route_for(t, plan, gpu))
        .collect::<Result<Vec<_>>>()?;
    let windows = plan_reads(header, h.len, cfg.slot_bytes, &|i| routes[i].row_multiple())?;

    let mut bands = vec![0usize; header.tensors.len()];
    for b in windows.iter().flat_map(|w| &w.bands) {
        bands[b.tensor] += 1;
        if let Route::Gpu(kind, layout) = routes[b.tensor]
            && !b.src_off.is_multiple_of(band_offset_rule(kind))
        {
            routes[b.tensor] = if kind.is_kquant() {
                Route::CpuKQuant(layout)
            } else {
                Route::CpuAffine
            };
        }
    }
    if let Some(i) = bands.iter().position(|&n| n == 0) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' has no data to read",
            header.tensors[i].name
        )));
    }
    if let Some(consumer) = consumer.as_mut() {
        consumer.plan(&windows, header.tensors.len())?;
    }

    let gpu_tensors = routes
        .iter()
        .filter(|r| matches!(r, Route::Gpu(..)))
        .count();
    let stats_base = LoadStats {
        bytes: windows.iter().map(|w| w.len as u64).sum(),
        gpu_tensors,
        cpu_tensors: routes.len() - gpu_tensors,
        ..LoadStats::default()
    };

    let mut guard = ImportGuard {
        slots: Vec::new(),
        states: routes
            .iter()
            .zip(&bands)
            .map(|(&route, &bands_left)| TensorState {
                route,
                bands_left,
                dst: Dst::Empty,
            })
            .collect(),
        gpu: gpu_tensors > 0,
        consuming: consumer.is_some(),
        probe,
    };
    let slot_len = windows.iter().map(|w| w.len).max().unwrap_or(0);
    for _ in 0..cfg.slots.max(1).min(windows.len()) {
        let (array, ptr, bytes) = MxArray::alloc_uninit(&[slot_len as i64], DType::Uint8)?;
        probe.alloc(bytes);
        guard.slots.push((array, ptr));
    }

    let mut nocache = mode.nocache(stats_base.bytes, physical_memory());
    if nocache
        && guard
            .slots
            .iter()
            .any(|(_, p)| !(p.0 as usize).is_multiple_of(ALIGN as usize))
    {
        tracing::warn!(
            "GGUF '{}': staging buffers are not {ALIGN}-byte aligned; reading through the page cache",
            h.path.display()
        );
        nocache = false;
    }
    if nocache && let Err(e) = set_nocache(&h.file) {
        tracing::warn!(
            "GGUF '{}': cannot bypass the page cache ({e}); reading through it",
            h.path.display()
        );
        nocache = false;
    }
    let resident = sampled_resident_fraction(&h.file, h.len);
    log_load_start(&h.path, stats_base.bytes, nocache, resident);

    let timing = std::thread::scope(|scope| {
        let reader = Reader::start(scope, src, &h.path, cfg.read_threads, cfg.chunk_bytes);
        let pool = CpuPool::start(scope, cfg.cpu_threads, cfg.cpu_inflight_bytes);
        let _quiesce = Quiesce {
            reader: &reader,
            pool: &pool,
            probe,
        };
        let mut run = Run {
            reader: &reader,
            pool: &pool,
            guard: &mut guard,
            windows: &windows,
            header,
            plan,
            cfg,
            probe,
        };
        run.all(consumer)
    })?;

    let mut map = HashMap::new();
    let mut ple_shards = None;
    for (t, state) in header.tensors.iter().zip(std::mem::take(&mut guard.states)) {
        collect_tensor(t, state.dst, &mut map, &mut ple_shards)?;
    }
    drop(guard);
    if let Some(secs) = timing.first_gib_secs {
        maybe_log_slow_disk(&h.path, stats_base.bytes, secs);
    }
    Ok(RawTensors {
        map,
        ple_shards,
        stats: LoadStats {
            read_secs: timing.read_secs,
            preload_secs: started.elapsed().as_secs_f64(),
            nocache,
            #[cfg(test)]
            resident,
            ..stats_base
        },
    })
}

struct Timing {
    read_secs: f64,
    first_gib_secs: Option<f64>,
}

struct Run<'r, 's, 'p> {
    reader: &'r Reader<'s>,
    pool: &'r CpuPool,
    guard: &'r mut ImportGuard<'p>,
    windows: &'r [Window],
    header: &'r GgufFile,
    plan: &'r ImportPlan,
    cfg: &'r ReadConfig,
    probe: &'r ImportProbe,
}

impl Run<'_, '_, '_> {
    fn submit(&self, w: usize) {
        let (_, ptr) = &self.guard.slots[w % self.guard.slots.len()];
        let window = &self.windows[w];
        self.probe.event(ImportEvent::Submitted(w));
        self.reader
            .submit(w as u64, *ptr, window.len, window.file_off);
    }

    fn complete(&self, w: usize, gpu_outputs: &[MxArray]) -> Result<()> {
        let refs: Vec<&MxArray> = gpu_outputs.iter().collect();
        MxArray::eval_arrays_with_context(&refs, "GGUF repack window")?;
        self.pool.wait_window(w)?;
        self.probe.event(ImportEvent::Completed(w));
        Ok(())
    }

    /// Window `w + slots - 1` reuses the slot of window `w - 1`, so it is
    /// issued only after window `w - 1` completed.
    fn all(&mut self, mut consumer: Option<&mut ConsumeGroups<'_>>) -> Result<Timing> {
        let n = self.windows.len();
        let slots = self.guard.slots.len();
        let mut timing = Timing {
            read_secs: 0.0,
            first_gib_secs: None,
        };
        if n == 0 {
            return Ok(timing);
        }
        let started = Instant::now();
        let mut read_bytes = 0u64;
        for w in 0..(slots - 1).min(n) {
            self.submit(w);
        }
        let mut pending: Vec<MxArray> = Vec::new();
        for w in 0..n {
            if w > 0 {
                self.complete(w - 1, &std::mem::take(&mut pending))?;
                if let Some(consumer) = consumer.as_mut() {
                    consumer.complete(w - 1, self.header, &mut self.guard.states)?;
                }
            }
            let next = w + slots - 1;
            if next < n {
                self.submit(next);
            }
            let wait_start = Instant::now();
            self.reader.wait(w as u64)?;
            timing.read_secs += wait_start.elapsed().as_secs_f64();
            read_bytes += self.windows[w].len as u64;
            if timing.first_gib_secs.is_none() && read_bytes >= 1 << 30 {
                timing.first_gib_secs = Some(started.elapsed().as_secs_f64());
            }
            pending = self.window(w)?;
            let refs: Vec<&MxArray> = pending.iter().collect();
            MxArray::async_eval_arrays(&refs);
        }
        self.complete(n - 1, &pending)?;
        drop(pending);
        if let Some(consumer) = consumer.as_mut() {
            consumer.complete(n - 1, self.header, &mut self.guard.states)?;
        }
        Ok(timing)
    }

    /// Route every band of window `w`; returns the GPU arrays it encoded.
    fn window(&mut self, w: usize) -> Result<Vec<MxArray>> {
        let (windows, header, plan) = (self.windows, self.header, self.plan);
        let slot = w % self.guard.slots.len();
        let (staging, base) = self.guard.slots[slot].clone();
        if let Some((name, off, bit)) = &self.probe.flip {
            for band in &windows[w].bands {
                let start = band.row0 * (band.bytes / band.rows);
                if header.tensors[band.tensor].name == *name
                    && (start..start + band.bytes).contains(off)
                {
                    // SAFETY: window `w`'s read has finished and nothing
                    // touches its slot until the window completes.
                    unsafe { *base.0.add(band.src_off + off - start) ^= 1u8 << (bit & 7) };
                }
            }
        }
        let mut outputs = Vec::new();
        for band in &windows[w].bands {
            if self.probe.panic_at_window == Some(w) {
                panic!("injected panic in GGUF window {w}");
            }
            let t = &header.tensors[band.tensor];
            let state = &mut self.guard.states[band.tensor];
            if matches!(state.dst, Dst::Empty) {
                state.dst = alloc_dst(t, state.route, self.probe)?;
            }
            let src = SendPtr(base.0.wrapping_add(band.src_off));
            let row_bytes = band.bytes / band.rows;
            match (state.route, &mut state.dst) {
                (Route::Gpu(kind, layout), Dst::Repack { dst, ptrs }) => {
                    let current = dst.take().ok_or_else(|| {
                        Error::from_reason(format!("GGUF tensor '{}' lost its arrays", t.name))
                    })?;
                    let next = gpu_repack_band(
                        &staging,
                        band.src_off as u64,
                        kind,
                        layout,
                        u32_of(ptrs.k, t)?,
                        u32_of(band.row0, t)?,
                        u32_of(band.rows, t)?,
                        current,
                    )?;
                    outputs.push(next.weight.clone());
                    outputs.extend(next.scales.iter().cloned());
                    outputs.extend(next.biases.iter().cloned());
                    *dst = Some(next);
                }
                (Route::CpuKQuant(layout), Dst::Repack { ptrs, .. }) => {
                    let unit = Route::CpuKQuant(layout).row_multiple();
                    let sub = (self.cfg.cpu_band_bytes / row_bytes / unit).max(1) * unit;
                    let ptrs = *ptrs;
                    let mut r = 0;
                    while r < band.rows {
                        let rows = sub.min(band.rows - r);
                        self.queue(
                            w,
                            rows * row_bytes,
                            CpuWork::KQuant {
                                info: t.clone(),
                                src: SendPtr(src.0.wrapping_add(r * row_bytes)),
                                len: rows * row_bytes,
                                row0: band.row0 + r,
                                rows,
                                dst: ptrs,
                            },
                        )?;
                        r += rows;
                    }
                }
                (Route::CpuAffine, Dst::Repack { ptrs, .. }) => {
                    let sub = (self.cfg.cpu_band_bytes / row_bytes).max(1);
                    let ptrs = *ptrs;
                    let mut r = 0;
                    while r < band.rows {
                        let rows = sub.min(band.rows - r);
                        self.queue(
                            w,
                            rows * row_bytes,
                            CpuWork::Affine {
                                ty: t.tensor_type,
                                src: SendPtr(src.0.wrapping_add(r * row_bytes)),
                                len: rows * row_bytes,
                                row0: band.row0 + r,
                                rows,
                                dst: ptrs,
                            },
                        )?;
                        r += rows;
                    }
                }
                (Route::Dense(_), Dst::Dense { ptr, .. }) => {
                    let (ptr, at) = (*ptr, band.row0 * row_bytes);
                    self.copy_bytes(w, src, band.bytes, ptr, at)?;
                }
                (
                    Route::Ple(..),
                    Dst::Ple {
                        shards,
                        rows_per_shard,
                    },
                ) => {
                    let rps = *rows_per_shard;
                    let targets: Vec<SendPtr<u8>> = shards.iter().map(|(_, p)| *p).collect();
                    let mut r = 0;
                    while r < band.rows {
                        let row = band.row0 + r;
                        let in_shard = row % rps;
                        let rows = (rps - in_shard).min(band.rows - r);
                        let target = *targets.get(row / rps).ok_or_else(|| {
                            Error::from_reason(format!(
                                "GGUF tensor '{}' row {row} has no shard",
                                t.name
                            ))
                        })?;
                        self.copy_bytes(
                            w,
                            SendPtr(src.0.wrapping_add(r * row_bytes)),
                            rows * row_bytes,
                            target,
                            in_shard * row_bytes,
                        )?;
                        r += rows;
                    }
                }
                (Route::Whole, Dst::Whole(buf)) => {
                    let at = band.row0 * row_bytes;
                    // SAFETY: window `w`'s read has finished and nothing
                    // writes its slot until the window completes.
                    let bytes = unsafe { std::slice::from_raw_parts(src.0, band.bytes) };
                    buf.get_mut(at..at + band.bytes)
                        .ok_or_else(|| {
                            Error::from_reason(format!(
                                "GGUF tensor '{}' band runs past its bytes",
                                t.name
                            ))
                        })?
                        .copy_from_slice(bytes);
                }
                _ => {
                    return Err(Error::from_reason(format!(
                        "GGUF tensor '{}' destination does not match its route",
                        t.name
                    )));
                }
            }
            let state = &mut self.guard.states[band.tensor];
            state.bands_left -= 1;
            if state.bands_left == 0
                && let Dst::Whole(buf) = &state.dst
            {
                let mut reader = OffsetReader {
                    base: header.data_offset + t.offset,
                    buf,
                    pos: 0,
                };
                let options = GgufLoadOptions {
                    verbose: false,
                    import_k_quants: plan.import_k_quants,
                };
                state.dst = Dst::Done(load_gguf_tensor(&mut reader, header, t, options)?);
            }
        }
        if self.probe.gpu_error_at_window == Some(w) {
            return Err(Error::from_reason(format!(
                "injected GPU repack failure in GGUF window {w}"
            )));
        }
        Ok(outputs)
    }

    fn queue(&self, w: usize, bytes: usize, work: CpuWork) -> Result<()> {
        let inflight = self.pool.submit(CpuJob {
            window: w,
            bytes,
            work,
        })?;
        self.probe
            .max_cpu_inflight
            .fetch_max(inflight, Ordering::Relaxed);
        Ok(())
    }

    /// Copy `len` staged bytes to `dst + at` in jobs of at most
    /// `cpu_band_bytes`.
    fn copy_bytes(
        &self,
        w: usize,
        src: SendPtr<u8>,
        len: usize,
        dst: SendPtr<u8>,
        at: usize,
    ) -> Result<()> {
        let step = self.cfg.cpu_band_bytes.max(1);
        let mut done = 0;
        while done < len {
            let n = step.min(len - done);
            self.queue(
                w,
                n,
                CpuWork::Copy {
                    src: SendPtr(src.0.wrapping_add(done)),
                    len: n,
                    dst,
                    dst_off: at + done,
                },
            )?;
            done += n;
        }
        Ok(())
    }
}

fn u32_of(v: usize, t: &GgufTensorInfo) -> Result<u32> {
    u32::try_from(v).map_err(|_| {
        Error::from_reason(format!(
            "GGUF tensor '{}': {v} exceeds the GPU repack's 32-bit range",
            t.name
        ))
    })
}

/// The tensor transform shared by `convert_gguf_to_safetensors` and the
/// loader's preload.
pub(crate) struct ImportedWeights {
    pub weights: HashMap<String, MxArray>,
    pub clef_head: Option<(HashMap<String, MxArray>, serde_json::Value)>,
    /// Keys whose array is a lazy view or reshuffle of the read buffers (not a
    /// dtype-pass output). The loader copies them into contiguous storage, as
    /// the safetensors round trip did.
    pub views: HashSet<String>,
}

fn is_available(a: &MxArray) -> bool {
    // SAFETY: the handle is a live array.
    unsafe { mlx_sys::mlx_array_is_available(a.as_raw_ptr()) }
}

/// The `--dtype` target for `key`, or `None` when the pass leaves it alone.
fn dtype_pass_target(
    key: &str,
    arr: &MxArray,
    plan: &ImportPlan,
    opts: &ImportOptions,
) -> Result<Option<DType>> {
    if opts.dtype.filter(|_| !plan.prism).is_none() {
        return Ok(None);
    }
    Ok(dtype_target_for(key, arr.dtype()?, plan, opts))
}

fn dtype_target_for(
    key: &str,
    dtype: DType,
    plan: &ImportPlan,
    opts: &ImportOptions,
) -> Option<DType> {
    let target = opts.dtype.filter(|_| !plan.prism)?;
    if plan.preserve_dtype_keys.contains(key)
        || key.ends_with(".scales")
        || key.ends_with(".biases")
        || dtype == DType::Uint32
    {
        return None;
    }
    Some(target)
}

/// `(gguf name, rows per shard)` for the one dense 2-D tensor whose
/// checkpoint key satisfies `is_target`, when it imports as `dtype` and its
/// imported bytes exceed `budget`; rows per shard = max(1, budget / imported
/// row bytes). Header only.
pub(crate) fn plan_dense_row_shards(
    header: &GgufFile,
    plan: &ImportPlan,
    opts: &ImportOptions,
    is_target: &dyn Fn(&str) -> bool,
    dtype: DType,
    budget: usize,
) -> Result<Option<(String, usize)>> {
    let mut hits = header.tensors.iter().filter_map(|t| {
        let key = gguf_name_to_hf_for_metadata(&t.name, &header.metadata)?;
        is_target(&key).then_some((t, key))
    });
    let Some((t, key)) = hits.next() else {
        return Ok(None);
    };
    if let Some((other, _)) = hits.next() {
        return Err(Error::from_reason(format!(
            "GGUF tensors '{}' and '{}' both map to checkpoint key '{key}'",
            t.name, other.name
        )));
    }
    let (Some(source), [cols, rows]) = (dense_dtype(t.tensor_type), t.dims.as_slice()) else {
        return Ok(None);
    };
    if dtype_target_for(&key, source, plan, opts).unwrap_or(source) != dtype {
        return Ok(None);
    }
    let row_bytes = usize::try_from(*cols)
        .ok()
        .and_then(|cols| cols.checked_mul(dtype.byte_size()))
        .filter(|&bytes| bytes > 0);
    let total = row_bytes.and_then(|row| usize::try_from(*rows).ok()?.checked_mul(row));
    match (row_bytes, total) {
        (Some(row_bytes), Some(total)) if total > budget => {
            Ok(Some((t.name.clone(), (budget / row_bytes).max(1))))
        }
        (Some(_), Some(_)) => Ok(None),
        _ => Err(Error::from_reason(format!(
            "GGUF tensor '{}' has dims {:?} that overflow a row-shard plan",
            t.name, t.dims
        ))),
    }
}

/// GGUF → HF names, layout fixups, the CLEF head split, and the optional
/// `--dtype` pass. Numeric ops run on the CPU device.
pub(crate) fn transform_imported(
    raw: HashMap<String, MxArray>,
    gguf: &GgufFile,
    plan: &ImportPlan,
    opts: &ImportOptions,
) -> Result<ImportedWeights> {
    transform_imported_part(raw, gguf, plan, opts, true, |_| Ok(()))
}

/// Run `before_dtype` after remapping, layout fixups and CLEF extraction, but
/// before conversion can change the source norm precision.
pub(crate) fn transform_imported_with_before_dtype<F>(
    raw: HashMap<String, MxArray>,
    gguf: &GgufFile,
    plan: &ImportPlan,
    opts: &ImportOptions,
    before_dtype: F,
) -> Result<ImportedWeights>
where
    F: FnOnce(&HashMap<String, MxArray>) -> Result<()>,
{
    transform_imported_part(raw, gguf, plan, opts, true, before_dtype)
}

fn transform_imported_part<F>(
    raw: HashMap<String, MxArray>,
    gguf: &GgufFile,
    plan: &ImportPlan,
    opts: &ImportOptions,
    extract_head: bool,
    before_dtype: F,
) -> Result<ImportedWeights>
where
    F: FnOnce(&HashMap<String, MxArray>) -> Result<()>,
{
    tracing::info!("Remapping tensor names to HuggingFace format...");
    let mut weights = remap_keys(raw, &gguf.metadata)?;

    // Unified Gemma4 mmproj tensors need name-aware CHW→HWC and position-table
    // permutations after remapping, before the generic shape fixups run. The
    // gemma4v SigLIP tower needs its own conv-kernel→Linear reshape.
    fixup_gemma4_mmproj_layout(&mut weights, &gguf.metadata)?;
    fixup_gemma4v_mmproj_layout(&mut weights, &gguf.metadata)?;

    fixup_shapes(&mut weights)?;

    // Sparse Qwen3.5 stores its shared-expert gate as a bare vector while the
    // MLX block is a `Linear(hidden, 1)`.
    fixup_qwen35_moe_shared_expert_gate(&mut weights, &gguf.metadata)?;

    fixup_qwen35_linear_attn(&mut weights, &gguf.metadata, plan.native_qwen35_layout)?;

    // After the GDN fixups (which only touch `linear_attn.*` keys) and before
    // the dtype pass, so head tensors keep their stored f32/f16 precision.
    let clef_head = if extract_head {
        extract_clef_head_assets(&mut weights, gguf)?
    } else {
        None
    };

    before_dtype(&weights)?;

    let mut views: HashSet<String> = weights
        .iter()
        .filter(|(_, a)| !is_available(a))
        .map(|(k, _)| k.clone())
        .collect();

    if let Some(target_dtype) = opts.dtype.filter(|_| !plan.prism) {
        tracing::info!("Converting to {target_dtype:?}...");
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            let Some(arr) = weights.get(&key) else {
                continue;
            };
            if dtype_pass_target(&key, arr, plan, opts)?.is_none() {
                continue;
            }
            if let Some(arr) = weights.remove(&key) {
                if arr.dtype()? != target_dtype {
                    views.remove(&key);
                }
                weights.insert(key, convert_tensor_dtype(arr, target_dtype)?);
            }
        }
    }
    Ok(ImportedWeights {
        weights,
        clef_head,
        views,
    })
}

/// Output bytes evaluated per batch while materializing the transform.
const EVAL_BATCH_BYTES: usize = 512 << 20;

fn eval_lazy<'a>(arrays: impl Iterator<Item = &'a MxArray>) -> Result<()> {
    let mut batch: Vec<&MxArray> = Vec::new();
    let mut bytes = 0usize;
    for a in arrays {
        if is_available(a) {
            continue;
        }
        bytes += a.nbytes();
        batch.push(a);
        if bytes >= EVAL_BATCH_BYTES {
            MxArray::eval_arrays_with_context(&batch, "GGUF import transform")?;
            batch.clear();
            bytes = 0;
        }
    }
    MxArray::eval_arrays_with_context(&batch, "GGUF import transform")
}

/// Copy `views` into contiguous storage, then evaluate every lazy array.
fn materialize(map: &mut HashMap<String, MxArray>, views: &HashSet<String>) -> Result<()> {
    for key in views {
        if let Some(a) = map.get_mut(key) {
            *a = a.deep_copy()?;
        }
    }
    eval_lazy(map.values())
}

/// One GGUF file of a preload: path, the header parsed at prepare, the
/// identity taken under the prepare lock, and its import plan.
pub(crate) type PreloadFile = (PathBuf, Arc<GgufFile>, GgufFileIdentity, ImportPlan);

pub(crate) struct PreloadInput {
    pub main: PreloadFile,
    pub companion: Option<PreloadFile>,
    /// Read the companion (gemma: media on; muse: draft present).
    pub load_companion: bool,
    pub opts: ImportOptions,
    /// Test hook: `(tensor name, byte offset, bit)` flipped in the main file's
    /// staged bytes.
    #[cfg(test)]
    pub flip: Option<(String, usize, u8)>,
}

impl PreloadInput {
    pub(crate) fn new(
        main: PreloadFile,
        companion: Option<PreloadFile>,
        load_companion: bool,
        opts: ImportOptions,
    ) -> Self {
        Self {
            main,
            companion,
            load_companion,
            opts,
            #[cfg(test)]
            flip: None,
        }
    }

    /// Every GGUF this load names, read or not: the companion is part of the
    /// checkpoint even when this load skips it.
    pub(crate) fn identity(&self) -> GgufIdentity {
        GgufIdentity::new(
            &self.main.2,
            self.companion.as_ref().map(|(_, _, identity, _)| identity),
        )
    }
}

pub(crate) struct GgufPreload {
    /// Before `expand_symmetric_affine_biases` (what the old cache's
    /// `model.safetensors` held).
    pub main: HashMap<String, MxArray>,
    /// gemma mmproj → vision map; muse draft → draft map. Empty when the
    /// companion was not read.
    pub companion: HashMap<String, MxArray>,
    pub clef_head: Option<(HashMap<String, MxArray>, serde_json::Value)>,
    /// Row shards of the plan's `ple_shard` tensor (absent from `main`).
    pub ple_shards: Option<(Vec<MxArray>, usize)>,
    #[cfg(test)]
    pub identity: GgufIdentity,
    pub stats: LoadStats,
}

fn add_stats(total: &mut LoadStats, part: &LoadStats) {
    total.bytes += part.bytes;
    total.read_secs += part.read_secs;
    total.gpu_tensors += part.gpu_tensors;
    total.cpu_tensors += part.cpu_tensors;
}

impl GgufPreload {
    /// GGUF's usual 2-D conv is reshaped to [C,K,1] by import. An unusual
    /// unsanitized 3-D conv makes sanitize's norm shift a whole-checkpoint
    /// decision; retain the ordinary path for that layout rather than deciding
    /// norms independently in layers that do not carry a conv (or final_norm).
    pub(crate) fn clef_layers_sanitize_independently(header: &GgufFile) -> bool {
        header.tensors.iter().all(|tensor| {
            let key = gguf_name_to_hf_for_metadata(&tensor.name, &header.metadata);
            !key.is_some_and(|key| key.contains("conv1d.weight"))
                || tensor.dims.len() <= 2
                || tensor.dims.first() == Some(&1)
        })
    }

    /// Model thread only.
    pub(crate) fn load(input: &PreloadInput) -> Result<Self> {
        Self::load_with(input, RepackBackend::Auto)
    }

    pub(crate) fn load_with(input: &PreloadInput, backend: RepackBackend) -> Result<Self> {
        Self::load_consuming(input, backend, None)
    }

    /// Stream complete backbone layers to CLEF's residency pass. Tables and
    /// the complete decision head still take the ordinary preload path.
    pub(crate) fn load_clef(
        input: &PreloadInput,
        consume: &mut TensorConsumer<'_>,
    ) -> Result<Self> {
        Self::load_consuming(input, RepackBackend::Auto, Some(consume))
    }

    fn load_consuming(
        input: &PreloadInput,
        backend: RepackBackend,
        mut consume: Option<&mut TensorConsumer<'_>>,
    ) -> Result<Self> {
        let started = Instant::now();
        let mode = PageCacheMode::from_env()?;
        let (main_path, main_header, main_identity, main_plan) = &input.main;
        let companion = input.companion.as_ref().filter(|_| input.load_companion);
        let set = GgufSourceSet::open(
            (main_path, main_header.clone(), main_identity),
            companion.map(|(path, header, identity, _)| (path.as_path(), header.clone(), identity)),
        )?;

        let main_probe = ImportProbe {
            #[cfg(test)]
            flip: input.flip.clone(),
            ..ImportProbe::default()
        };
        let streaming = consume.is_some();
        let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        if streaming {
            for (i, tensor) in main_header.tensors.iter().enumerate() {
                // Use the mapped key so MTP tensors and the decision head are
                // never confused with backbone layers, even in a shared blk.* namespace.
                if let Some(key) = gguf_name_to_hf_for_metadata(&tensor.name, &main_header.metadata)
                {
                    let key = crate::models::mtp_drafter::strip_wrapper_prefix(&key);
                    if let Some(rest) = key.strip_prefix("layers.")
                        && let Some((layer, _)) = rest.split_once('.')
                        && let Ok(layer) = layer.parse::<usize>()
                    {
                        groups.entry(layer).or_default().push(i);
                    }
                }
            }
        }
        let mut transform = |map| {
            let mut imported = transform_imported_part(
                map,
                main_header,
                main_plan,
                &input.opts,
                false,
                |_| Ok(()),
            )?;
            materialize(&mut imported.weights, &imported.views)?;
            match consume.as_deref_mut() {
                Some(consume) => consume(imported.weights),
                None => unreachable!("consumer groups require a callback"),
            }
        };
        let mut consumer = ConsumeGroups {
            groups: groups.into_values().collect(),
            ready: Vec::new(),
            consume: &mut transform,
        };
        let raw = read_gguf_tensors_consuming(
            &set.main.file,
            &set.main,
            main_plan,
            backend,
            mode,
            &ReadConfig::default(),
            &main_probe,
            streaming.then_some(&mut consumer),
        )?;
        let mut stats = raw.stats.clone();
        let ImportedWeights {
            weights: mut main,
            clef_head,
            views,
        } = transform_imported(raw.map, main_header, main_plan, &input.opts)?;
        materialize(&mut main, &views)?;

        let clef_head = match clef_head {
            Some((mut head, config)) => {
                let lazy: HashSet<String> = head
                    .iter()
                    .filter(|(_, a)| !is_available(a))
                    .map(|(k, _)| k.clone())
                    .collect();
                materialize(&mut head, &lazy)?;
                Some((head, config))
            }
            None => None,
        };

        let ple_shards = match raw.ple_shards {
            Some((name, shards, rows_per_shard)) => {
                let key = gguf_name_to_hf_for_metadata(&name, &main_header.metadata).ok_or_else(
                    || {
                        Error::from_reason(format!(
                            "GGUF row-sharded tensor '{name}' has no checkpoint name"
                        ))
                    },
                )?;
                let shards = shards
                    .into_iter()
                    .map(
                        |shard| match dtype_pass_target(&key, &shard, main_plan, &input.opts)? {
                            Some(target) => convert_tensor_dtype(shard, target),
                            None => Ok(shard),
                        },
                    )
                    .collect::<Result<Vec<_>>>()?;
                eval_lazy(shards.iter())?;
                Some((shards, rows_per_shard))
            }
            None => None,
        };

        let mut companion_map = HashMap::new();
        if let (Some(handle), Some((_, header, _, plan))) = (&set.companion, companion) {
            let raw = read_gguf_tensors(handle, plan, backend, mode)?;
            add_stats(&mut stats, &raw.stats);
            let imported = transform_imported(raw.map, header, plan, &input.opts)?;
            companion_map = imported.weights;
            materialize(&mut companion_map, &imported.views)?;
        }

        set.recheck()?;
        synchronize();
        clear_cache();
        stats.preload_secs = started.elapsed().as_secs_f64();
        Ok(Self {
            main,
            companion: companion_map,
            clef_head,
            ple_shards,
            #[cfg(test)]
            identity: input.identity(),
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    use std::fs;
    use std::path::PathBuf;

    use crate::array::memory::get_active_memory;
    use crate::models::quant_dispatch::kquant_tile_rows;
    use crate::utils::gguf::{
        GGUF_MAGIC, GGUF_VERSION_3, GgufMetaValue, GgufValueType, load_gguf_tensors, parse_gguf,
    };
    use crate::utils::gguf_plan::OutputRole;
    use crate::utils::gguf_source::{GgufSourceSet, file_identity};
    use crate::utils::gguf_test_fixtures::{
        IMPORT_FIXTURE_NAMES, ImportFixture, ImportFixtureFamily, write_import_fixture,
    };
    use GgufTensorType::*;
    use std::collections::BTreeMap;

    const MIB: usize = 1 << 20;

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    /// Block bytes for `ty`; dense types get finite values.
    fn payload(ty: GgufTensorType, elements: usize, seed: u64) -> Vec<u8> {
        let small = |bits: u16| bits & 0x3BFF;
        match ty {
            F32 => random_bytes(elements * 2, seed)
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|c| (f32::from(u16::from_le_bytes(*c)) / 65536.0).to_le_bytes())
                .collect(),
            F16 | BF16 => random_bytes(elements * 2, seed)
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|c| small(u16::from_le_bytes(*c)).to_le_bytes())
                .collect(),
            _ => random_bytes(elements / ty.block_size() * ty.type_size(), seed),
        }
    }

    type Spec = (&'static str, Vec<u64>, GgufTensorType);

    /// GGUF v3 bytes with tensor offsets aligned to `alignment`.
    fn build_gguf(
        meta: &[(&str, GgufMetaValue)],
        tensors: &[Spec],
        alignment: u64,
        seed: u64,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        for (key, value) in meta {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            match value {
                GgufMetaValue::Uint32(v) => {
                    buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufMetaValue::String(s) => {
                    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                _ => panic!("unsupported metadata value"),
            }
        }
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .enumerate()
            .map(|(i, (_, dims, ty))| {
                payload(*ty, dims.iter().product::<u64>() as usize, seed + i as u64)
            })
            .collect();
        let mut offsets = Vec::new();
        let mut at = 0u64;
        for d in &data {
            at = at.next_multiple_of(alignment);
            offsets.push(at);
            at += d.len() as u64;
        }
        for ((name, dims, ty), off) in tensors.iter().zip(&offsets) {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(*ty as u32).to_le_bytes());
            buf.extend_from_slice(&off.to_le_bytes());
        }
        buf.resize((buf.len() as u64).next_multiple_of(alignment) as usize, 0);
        let start = buf.len();
        for (d, off) in data.iter().zip(&offsets) {
            buf.resize(start + *off as usize, 0);
            buf.extend_from_slice(d);
        }
        buf
    }

    struct Fixture {
        dir: PathBuf,
        path: PathBuf,
        handle: GgufFileHandle,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn open_fixture(tag: &str, bytes: &[u8]) -> Fixture {
        let dir = std::env::temp_dir().join(format!("gguf-import-{tag}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        fs::write(&path, bytes).unwrap();
        let header = Arc::new(parse_gguf(&path).unwrap());
        let identity = file_identity(&path, &header, &fs::metadata(&path).unwrap()).unwrap();
        let handle = GgufSourceSet::open((&path, header, &identity), None)
            .unwrap()
            .main;
        Fixture { dir, path, handle }
    }

    fn plan(tiled: &[&str]) -> ImportPlan {
        ImportPlan {
            role: OutputRole::Primary,
            import_k_quants: true,
            native_qwen35_layout: false,
            prism: false,
            tie_word_embeddings: false,
            preserve_dtype_keys: HashSet::new(),
            tiled_gguf_names: tiled.iter().map(|s| s.to_string()).collect(),
            tiled_prefixes: BTreeSet::new(),
            config: None,
            clef_head_config: None,
            asset_copies: Vec::new(),
            strict_asset_copies: false,
            muse_secondary: false,
            ple_shard: None,
        }
    }

    fn small_slots(slots: usize) -> ReadConfig {
        ReadConfig {
            slot_bytes: MIB,
            slots,
            ..ReadConfig::default()
        }
    }

    fn require_gpu() -> bool {
        if cfg!(target_os = "macos") {
            assert!(gpu_repack_available(), "no GPU repack on macOS");
            true
        } else {
            false
        }
    }

    const GPU_SPECS: usize = 11;
    const TILED: [&str; 4] = [
        "blk.0.q4k.weight",
        "blk.0.q6k.weight",
        "blk.0.iq4xs.weight",
        "blk.0.q2k.weight",
    ];

    /// Every GPU type and every CPU type; `[k, rows]` dims.
    fn every_type_specs() -> Vec<Spec> {
        vec![
            ("blk.0.q4k.weight", vec![5120, 512], Q4K),
            ("blk.0.q5k.weight", vec![768, 130], Q5K),
            ("blk.0.q6k.weight", vec![1024, 128], Q6K),
            ("blk.0.iq4xs.weight", vec![768, 64], IQ4XS),
            ("blk.0.q3k.weight", vec![512, 96], Q3K),
            ("blk.0.iq4nl.weight", vec![256, 70], IQ4NL),
            ("blk.0.iq3s.weight", vec![512, 64], IQ3S),
            ("blk.0.q8_0.weight", vec![96, 33], Q8_0),
            ("blk.0.mxfp4.weight", vec![64, 40], MXFP4),
            ("blk.0.q2k.weight", vec![512, 128], Q2K),
            ("blk.0.iq2s.weight", vec![256, 20], IQ2S),
            ("blk.0.iq1m.weight", vec![256, 12], IQ1M),
            ("blk.0.q4_0.weight", vec![4096, 1200], Q4_0),
            ("blk.0.q4_1.weight", vec![64, 9], Q4_1),
            ("blk.0.q5_0.weight", vec![96, 5], Q5_0),
            ("blk.0.q5_1.weight", vec![64, 11], Q5_1),
            ("blk.0.f16.weight", vec![64, 3], F16),
            ("blk.0.f32.weight", vec![17], F32),
            ("blk.0.bf16.weight", vec![32, 4], BF16),
        ]
    }

    fn every_type_fixture() -> Fixture {
        let bytes = build_gguf(
            &[("general.architecture", GgufMetaValue::String("test".into()))],
            &every_type_specs(),
            32,
            100,
        );
        open_fixture("every", &bytes)
    }

    fn le_bytes(a: &MxArray) -> Vec<u8> {
        MxArray::eval_arrays_with_context(&[a], "gguf_import test").unwrap();
        match a.dtype().unwrap() {
            DType::Uint32 => a
                .to_uint32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            DType::Uint8 => a.to_uint8().unwrap(),
            DType::Int8 => a.to_int8().unwrap().into_iter().map(|v| v as u8).collect(),
            DType::Float16 | DType::BFloat16 => a
                .to_uint16_native()
                .unwrap()
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            DType::Float32 => a
                .to_float32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            other => panic!("unexpected dtype {other:?}"),
        }
    }

    /// `load_gguf_tensors` with the plan's Tiled64 tensors permuted like
    /// `tile_kquant_weights_for_disk`.
    fn oracle(fx: &Fixture, plan: &ImportPlan) -> HashMap<String, MxArray> {
        let header = &fx.handle.header;
        let options = GgufLoadOptions {
            verbose: false,
            import_k_quants: true,
        };
        let mut map = load_gguf_tensors(&fx.path, header, options).unwrap();
        for t in &header.tensors {
            if !plan.tiled_gguf_names.contains(&t.name) {
                continue;
            }
            let f = t.tensor_type.k_quant_format().unwrap();
            let prefix = t.name.strip_suffix(".weight").unwrap();
            for (key, unit) in [
                (t.name.clone(), f.bits()),
                (
                    format!("{prefix}.scales"),
                    f.super_ratio() * f.scale_bytes_per_group(),
                ),
                (format!("{prefix}.biases"), f.bias_entries_per_super_block()),
            ] {
                let tiled = kquant_tile_rows(&map[&key], unit as i64).unwrap();
                map.insert(key, tiled);
            }
        }
        map
    }

    fn assert_maps_equal(
        got: &HashMap<String, MxArray>,
        want: &HashMap<String, MxArray>,
        what: &str,
    ) {
        let mut gk: Vec<&String> = got.keys().collect();
        let mut wk: Vec<&String> = want.keys().collect();
        gk.sort();
        wk.sort();
        assert_eq!(gk, wk, "{what}: key sets");
        let mut bad = Vec::new();
        for (k, w) in want {
            let g = &got[k];
            let (gd, wd) = (g.dtype().unwrap(), w.dtype().unwrap());
            let (gs, ws) = (g.shape().unwrap().to_vec(), w.shape().unwrap().to_vec());
            let (gb, wb) = (le_bytes(g), le_bytes(w));
            let diff = gb.iter().zip(&wb).filter(|(a, b)| a != b).count();
            if gd != wd || gs != ws || gb.len() != wb.len() || diff > 0 {
                bad.push(format!(
                    "{k}: dtype {gd:?}/{wd:?} shape {gs:?}/{ws:?} bytes {}/{} mismatched {diff}",
                    gb.len(),
                    wb.len()
                ));
            }
        }
        bad.sort();
        assert!(
            bad.is_empty(),
            "{what}: {} tensors differ:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    fn read_with(
        fx: &Fixture,
        plan: &ImportPlan,
        backend: RepackBackend,
        cfg: &ReadConfig,
        probe: &ImportProbe,
    ) -> Result<RawTensors> {
        read_gguf_tensors_with(
            &fx.handle.file,
            &fx.handle,
            plan,
            backend,
            PageCacheMode::On,
            cfg,
            probe,
        )
    }

    #[test]
    fn consumed_groups_are_complete_before_reuse_and_equal_full_read() {
        require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&TILED);
        let want = oracle(&fx, &plan);
        for backend in [RepackBackend::Auto, RepackBackend::CpuOnly] {
            let probe = ImportProbe {
                record: true,
                ..ImportProbe::default()
            };
            let mut consumed = HashMap::new();
            let mut calls = 0;
            let mut consume = |map: HashMap<String, MxArray>| {
                // Reading every byte here checks the GPU/CPU completion
                // boundary, including tensors spanning multiple ring slots.
                for (key, array) in &map {
                    assert_eq!(le_bytes(array), le_bytes(&want[key]), "{key}");
                }
                assert!(!probe.freed.load(Ordering::Acquire));
                calls += 1;
                consumed.extend(map);
                Ok(())
            };
            let mut groups = ConsumeGroups {
                groups: vec![vec![0, 1, 2], vec![3, 4], vec![12, 13, 14, 15]],
                ready: Vec::new(),
                consume: &mut consume,
            };
            let raw = read_gguf_tensors_consuming(
                &fx.handle.file,
                &fx.handle,
                &plan,
                backend,
                PageCacheMode::On,
                &small_slots(2),
                &probe,
                Some(&mut groups),
            )
            .unwrap();
            assert!(consumed.keys().all(|key| !raw.map.contains_key(key)));
            consumed.extend(raw.map);
            assert_eq!(calls, 3);
            assert_maps_equal(&consumed, &want, "streamed groups");
            assert_eq!(
                guard_tail(&probe.events.lock().unwrap()),
                GUARD_ORDER.to_vec()
            );
        }
    }

    #[test]
    fn consumer_failure_quiesces_readers_and_gpu_before_freeing() {
        require_gpu();
        let fx = every_type_fixture();
        let probe = ImportProbe {
            record: true,
            ..ImportProbe::default()
        };
        let mut mock = MockRead::new(&fx.handle.file, &probe);
        mock.sleep_ms = 10;
        let mut fail = |_| Err(Error::from_reason("injected residency failure"));
        let mut groups = ConsumeGroups {
            groups: vec![vec![0]],
            ready: Vec::new(),
            consume: &mut fail,
        };
        let err = read_gguf_tensors_consuming(
            &mock,
            &fx.handle,
            &plan(&TILED),
            RepackBackend::Auto,
            PageCacheMode::On,
            &small_slots(SLOTS),
            &probe,
            Some(&mut groups),
        )
        .err()
        .expect("residency failure must abort preload");
        assert!(err.reason.contains("injected residency failure"));
        assert_eq!(mock.after_free.load(Ordering::SeqCst), 0);
        assert_eq!(
            guard_tail(&probe.events.lock().unwrap()),
            GUARD_ORDER.to_vec()
        );
    }

    #[test]
    fn staged_read_equals_load_gguf_tensors() {
        let gpu = require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&TILED);
        let raw = read_with(
            &fx,
            &plan,
            RepackBackend::Auto,
            &small_slots(SLOTS),
            &ImportProbe::default(),
        )
        .unwrap();
        assert_maps_equal(&raw.map, &oracle(&fx, &plan), "staged read");
        let n = every_type_specs().len();
        let want_gpu = if gpu { GPU_SPECS } else { 0 };
        assert_eq!(
            (raw.stats.gpu_tensors, raw.stats.cpu_tensors),
            (want_gpu, n - want_gpu),
            "(gpu, cpu) tensors"
        );
        assert!(raw.ple_shards.is_none());
        assert!(raw.stats.bytes >= fx.handle.len - fx.handle.header.data_offset);
    }

    #[test]
    fn q5_1_source_flip_changes_no_collateral_tensor() {
        if !require_gpu() {
            return;
        }
        let bytes = build_gguf(
            &[("general.architecture", GgufMetaValue::String("test".into()))],
            &[
                ("blk.0.q5_1.weight", vec![96, 5], Q5_1),
                ("blk.0.q8_0.weight", vec![96, 4], Q8_0),
            ],
            32,
            511,
        );
        let fx = open_fixture("q5-1-flip", &bytes);
        let plan = plan(&[]);
        let clean = read_with(
            &fx,
            &plan,
            RepackBackend::Auto,
            &small_slots(SLOTS),
            &ImportProbe::default(),
        )
        .unwrap();
        assert_eq!(clean.stats.gpu_tensors, 2, "Q5_1 and Q8_0 GPU routes");
        assert_maps_equal(&clean.map, &oracle(&fx, &plan), "Q5_1 flip baseline");
        let probe = ImportProbe {
            flip: Some(("blk.0.q5_1.weight".to_string(), 8, 0)),
            ..ImportProbe::default()
        };
        let flipped =
            read_with(&fx, &plan, RepackBackend::Auto, &small_slots(SLOTS), &probe).unwrap();
        let before = gate::manifest(&clean.map);
        let after = gate::manifest(&flipped.map);
        assert_eq!(
            before.keys().collect::<Vec<_>>(),
            after.keys().collect::<Vec<_>>()
        );
        let changed: Vec<&String> = before
            .keys()
            .filter(|key| before[*key] != after[*key])
            .collect();
        assert_eq!(
            changed,
            vec!["blk.0.q5_1.weight"],
            "the Q5_1 qs flip changed an unexpected tensor or companion"
        );
    }

    #[test]
    fn cpu_only_backend_equals_auto() {
        let gpu = require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&TILED);
        let cfg = small_slots(2);
        let auto = read_with(
            &fx,
            &plan,
            RepackBackend::Auto,
            &cfg,
            &ImportProbe::default(),
        )
        .unwrap();
        let cpu = read_with(
            &fx,
            &plan,
            RepackBackend::CpuOnly,
            &cfg,
            &ImportProbe::default(),
        )
        .unwrap();
        assert_maps_equal(&auto.map, &cpu.map, "Auto vs CpuOnly");
        assert_eq!(cpu.stats.gpu_tensors, 0);
        if gpu {
            assert!(auto.stats.gpu_tensors > 0, "Auto used no GPU kernel");
        }
    }

    #[test]
    fn misaligned_tensor_uses_cpu() {
        require_gpu();
        let specs: Vec<Spec> = vec![("a.weight", vec![1], F16), ("b.weight", vec![256, 64], Q4K)];
        let mut tag = String::new();
        let fx = loop {
            let bytes = build_gguf(
                &[
                    ("general.alignment", GgufMetaValue::Uint32(2)),
                    ("general.name", GgufMetaValue::String(tag.clone())),
                ],
                &specs,
                2,
                7,
            );
            let fx = open_fixture("misaligned", &bytes);
            let h = &fx.handle.header;
            if !(h.data_offset + h.tensors[1].offset).is_multiple_of(16) {
                break fx;
            }
            tag.push('x');
        };
        assert_eq!(fx.handle.header.tensors[1].offset, 2);
        let plan = plan(&[]);
        let raw = read_with(
            &fx,
            &plan,
            RepackBackend::Auto,
            &small_slots(SLOTS),
            &ImportProbe::default(),
        )
        .unwrap();
        assert_maps_equal(&raw.map, &oracle(&fx, &plan), "misaligned");
        assert_eq!(
            (raw.stats.gpu_tensors, raw.stats.cpu_tensors),
            (0, 2),
            "(gpu, cpu) tensors"
        );
    }

    #[test]
    fn cpu_inflight_bytes_bounded() {
        assert_eq!(ReadConfig::default().cpu_inflight_bytes, 256 * MIB);
        let bytes = build_gguf(
            &[],
            &[("blk.0.q4_0.weight", vec![4096, 32768], Q4_0)],
            32,
            9,
        );
        let fx = open_fixture("inflight", &bytes);
        let cfg = ReadConfig {
            slot_bytes: 16 * MIB,
            cpu_inflight_bytes: 4 * MIB,
            cpu_band_bytes: MIB,
            ..ReadConfig::default()
        };
        let probe = ImportProbe::default();
        let raw = read_with(&fx, &plan(&[]), RepackBackend::CpuOnly, &cfg, &probe).unwrap();
        let max = probe.max_cpu_inflight.load(Ordering::Relaxed);
        assert!(max > 0);
        assert!(
            max <= cfg.cpu_inflight_bytes,
            "max in-flight {max} > cap {}",
            cfg.cpu_inflight_bytes
        );
        assert_eq!(raw.stats.cpu_tensors, 1);
    }

    fn position(events: &[ImportEvent], e: &ImportEvent) -> Option<usize> {
        events.iter().position(|x| x == e)
    }

    #[test]
    fn slot_reuse_waits_for_gpu() {
        let gpu = require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&TILED);
        for slots in [1usize, 2] {
            let probe = ImportProbe {
                record: true,
                ..ImportProbe::default()
            };
            let raw =
                read_with(&fx, &plan, RepackBackend::Auto, &small_slots(slots), &probe).unwrap();
            if gpu {
                assert!(raw.stats.gpu_tensors > 0);
            }
            let events = probe.events.lock().unwrap().clone();
            let n = events
                .iter()
                .filter(|e| matches!(e, ImportEvent::Submitted(_)))
                .count();
            assert!(n > 3, "only {n} windows");
            let mut early = Vec::new();
            for v in slots..n {
                let refill = position(&events, &ImportEvent::Submitted(v)).unwrap();
                match position(&events, &ImportEvent::Completed(v - slots)) {
                    Some(done) if done < refill => {}
                    _ => early.push(v),
                }
            }
            assert_eq!(
                early.len(),
                0,
                "slots {slots}: windows refilled before their slot's window completed: {early:?}"
            );
        }
    }

    #[test]
    fn ple_dense_above_budget_is_sharded_at_alloc() {
        const NAME: &str = "per_layer_token_embd.weight";
        const KEY: &str = "embed_tokens_per_layer.weight";
        let (rows, cols, budget) = (512usize, 64usize, 9000usize);
        let rows_per_shard = (budget / (cols * 2)).max(1);
        let bytes = build_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".into()),
            )],
            &[
                ("blk.0.attn_norm.weight", vec![64], F32),
                (NAME, vec![cols as u64, rows as u64], BF16),
            ],
            32,
            21,
        );
        let fx = open_fixture("ple", &bytes);
        let mut plan = plan(&[]);
        plan.ple_shard = Some((NAME.to_string(), rows_per_shard));
        let cfg = ReadConfig {
            slot_bytes: 2 * ALIGN as usize,
            ..ReadConfig::default()
        };
        let probe = ImportProbe::default();
        let raw = read_with(&fx, &plan, RepackBackend::Auto, &cfg, &probe).unwrap();
        assert!(!raw.map.contains_key(NAME));
        let (name, shards, rps) = raw.ple_shards.expect("no PLE shards");
        assert_eq!(name, NAME);
        let max_alloc = probe.max_alloc_bytes.load(Ordering::Relaxed);
        assert!(
            max_alloc <= cfg.slot_bytes.max(budget),
            "largest allocation {max_alloc} B"
        );

        let whole =
            load_gguf_tensors(&fx.path, &fx.handle.header, GgufLoadOptions::default()).unwrap();
        let mut st = HashMap::from([(KEY.to_string(), whole[NAME].clone())]);
        crate::utils::safetensors::save_safetensors(
            fx.dir.join("model.safetensors"),
            &mut st,
            None,
        )
        .unwrap();
        let (want, want_rps) =
            crate::utils::safetensors::load_bf16_tensor_sharded(&fx.dir, KEY, budget).unwrap();
        assert_eq!(
            (rps as i64, shards.len()),
            (want_rps, want.len()),
            "(rows per shard, shards)"
        );
        for (i, (g, w)) in shards.iter().zip(&want).enumerate() {
            assert!(g.nbytes() <= budget, "shard {i}: {} B", g.nbytes());
            assert_eq!(
                g.shape().unwrap().to_vec(),
                w.shape().unwrap().to_vec(),
                "shard {i} shape"
            );
            let (gb, wb) = (le_bytes(g), le_bytes(w));
            assert_eq!(
                gb.iter().zip(&wb).filter(|(a, b)| a != b).count(),
                0,
                "shard {i} bytes"
            );
        }
    }

    /// Reads through the fixture file; counts reads that end after the
    /// import released its buffers and the event index at which each read
    /// ended.
    struct MockRead<'a> {
        file: &'a fs::File,
        probe: &'a ImportProbe,
        fail_at: Option<u64>,
        sleep_ms: u64,
        after_free: AtomicUsize,
        ended_at: Mutex<Vec<usize>>,
    }

    impl<'a> MockRead<'a> {
        fn new(file: &'a fs::File, probe: &'a ImportProbe) -> Self {
            Self {
                file,
                probe,
                fail_at: None,
                sleep_ms: 0,
                after_free: AtomicUsize::new(0),
                ended_at: Mutex::new(Vec::new()),
            }
        }
    }

    impl ReadAt for MockRead<'_> {
        fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
            if self.sleep_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.sleep_ms));
            }
            if self.fail_at == Some(off) {
                return Err(std::io::Error::other("injected read failure"));
            }
            let r = ReadAt::read_exact_at(self.file, buf, off);
            if self.probe.freed.load(Ordering::Acquire) {
                self.after_free.fetch_add(1, Ordering::SeqCst);
            }
            let at = self.probe.events.lock().unwrap().len();
            self.ended_at.lock().unwrap().push(at);
            r
        }
    }

    const GUARD_ORDER: [ImportEvent; 6] = [
        ImportEvent::Stopped,
        ImportEvent::ReadersIdle,
        ImportEvent::CpuJoined,
        ImportEvent::GpuSynced,
        ImportEvent::Freed,
        ImportEvent::CacheCleared,
    ];

    fn guard_tail(events: &[ImportEvent]) -> Vec<ImportEvent> {
        events
            .iter()
            .filter(|e| !matches!(e, ImportEvent::Submitted(_) | ImportEvent::Completed(_)))
            .cloned()
            .collect()
    }

    #[test]
    fn read_error_mid_import_frees_and_errs() {
        require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&[]);
        let cfg = small_slots(SLOTS);
        let windows = plan_reads(&fx.handle.header, fx.handle.len, cfg.slot_bytes, &|_| 1).unwrap();
        assert!(windows.len() > 4);
        let probe = ImportProbe {
            record: true,
            ..ImportProbe::default()
        };
        let mut mock = MockRead::new(&fx.handle.file, &probe);
        mock.fail_at = Some(windows[3].file_off);
        mock.sleep_ms = 30;
        clear_cache();
        let before = get_active_memory();
        let err = read_gguf_tensors_with(
            &mock,
            &fx.handle,
            &plan,
            RepackBackend::Auto,
            PageCacheMode::On,
            &cfg,
            &probe,
        )
        .err()
        .expect("read error must fail the import");
        let after = get_active_memory();
        let msg = err.reason.clone();
        assert!(
            msg.contains(&fx.handle.path.display().to_string()),
            "no path in: {msg}"
        );
        assert!(
            msg.contains(&format!("offset {}", windows[3].file_off)),
            "no offset in: {msg}"
        );
        assert_eq!(
            mock.after_free.load(Ordering::SeqCst),
            0,
            "reads ended after free"
        );
        assert_eq!(
            guard_tail(&probe.events.lock().unwrap()),
            GUARD_ORDER.to_vec()
        );
        assert!(
            after <= before + MIB as f64,
            "active memory {before} -> {after}"
        );
    }

    #[test]
    fn gpu_error_mid_import_frees_after_readers_stop() {
        require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&[]);
        let probe = ImportProbe {
            record: true,
            gpu_error_at_window: Some(2),
            ..ImportProbe::default()
        };
        let mut mock = MockRead::new(&fx.handle.file, &probe);
        mock.sleep_ms = 150;
        let err = read_gguf_tensors_with(
            &mock,
            &fx.handle,
            &plan,
            RepackBackend::Auto,
            PageCacheMode::On,
            &small_slots(SLOTS),
            &probe,
        )
        .err()
        .expect("GPU error must fail the import");
        assert!(
            err.reason.contains("injected GPU repack failure"),
            "{}",
            err.reason
        );
        let events = probe.events.lock().unwrap().clone();
        assert_eq!(guard_tail(&events), GUARD_ORDER.to_vec());
        let idle = position(&events, &ImportEvent::ReadersIdle).unwrap();
        let ended = mock.ended_at.lock().unwrap().clone();
        let late = ended.iter().filter(|&&at| at > idle).count();
        assert!(ended.len() > 3, "only {} reads", ended.len());
        assert_eq!(
            late, 0,
            "reads ended after ReadersIdle (event {idle}): {ended:?}"
        );
        assert_eq!(
            mock.after_free.load(Ordering::SeqCst),
            0,
            "reads ended after free"
        );
    }

    #[test]
    fn panic_mid_import_runs_guard() {
        require_gpu();
        let fx = every_type_fixture();
        let plan = plan(&TILED);
        let probe = ImportProbe {
            record: true,
            panic_at_window: Some(1),
            ..ImportProbe::default()
        };
        let mock = MockRead::new(&fx.handle.file, &probe);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_gguf_tensors_with(
                &mock,
                &fx.handle,
                &plan,
                RepackBackend::Auto,
                PageCacheMode::On,
                &small_slots(SLOTS),
                &probe,
            )
            .map(|_| ())
        }));
        assert!(r.is_err(), "the injected panic did not propagate");
        let events = probe
            .events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        assert_eq!(guard_tail(&events), GUARD_ORDER.to_vec());
        assert_eq!(
            mock.after_free.load(Ordering::SeqCst),
            0,
            "reads ended after free"
        );
    }

    fn import_golden_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/testdata/gguf_import_golden")
    }

    fn import_work_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gguf-preload-{tag}-{}", uuid::Uuid::new_v4()))
    }

    fn fixture_family(f: &ImportFixture) -> NativeGgufFamily {
        match f.family {
            ImportFixtureFamily::Qwen35 => NativeGgufFamily::Qwen35,
            ImportFixtureFamily::Qwen35Moe => NativeGgufFamily::Qwen35Moe,
            ImportFixtureFamily::Gemma4 => NativeGgufFamily::Gemma4,
            ImportFixtureFamily::MuseGlimmer => NativeGgufFamily::MuseGlimmer,
        }
    }

    fn fixture_input(f: &ImportFixture, load_companion: bool) -> PreloadInput {
        gate::preload_input(
            &f.main,
            f.companion.as_deref(),
            fixture_family(f),
            load_companion,
        )
        .unwrap()
    }

    fn golden_manifest(name: &str, stem: &str) -> BTreeMap<String, serde_json::Value> {
        let path = import_golden_root()
            .join(name)
            .join(format!("{stem}.manifest.json"));
        serde_json::from_slice(&fs::read(&path).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn assert_manifest(
        label: &str,
        got: &HashMap<String, MxArray>,
        want: &BTreeMap<String, serde_json::Value>,
    ) {
        let got = gate::manifest(got);
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "{label}: tensor names"
        );
        for (name, want) in want {
            assert_eq!(&got[name], want, "{label}: tensor {name}");
        }
    }

    #[test]
    fn streamed_backbone_transform_equals_full_preload() {
        let work = import_work_dir("streamed-backbone");
        let fx = write_import_fixture("qwen35-dense-untied", &work);
        let input = fixture_input(&fx, false);
        let full = GgufPreload::load(&input).unwrap();
        let mut streamed = HashMap::new();
        let mut calls = 0;
        let rest = GgufPreload::load_clef(&input, &mut |layer| {
            assert!(layer.keys().all(|key| key.starts_with("model.layers.")));
            calls += 1;
            streamed.extend(layer);
            Ok(())
        })
        .unwrap();
        assert!(calls > 0, "backbone was not streamed");
        assert!(
            rest.main
                .keys()
                .all(|key| !key.starts_with("model.layers."))
        );
        streamed.extend(rest.main);
        assert_maps_equal(&streamed, &full.main, "streamed transformed backbone");
        assert_eq!(rest.identity, full.identity);
        assert_eq!(rest.stats.bytes, full.stats.bytes);
        assert_eq!(rest.stats.gpu_tensors, full.stats.gpu_tensors);
        assert_eq!(rest.stats.cpu_tensors, full.stats.cpu_tensors);
        fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn streamed_preload_rechecks_source_after_residency() {
        let work = import_work_dir("streamed-recheck");
        let fx = write_import_fixture("qwen35-dense-untied", &work);
        let input = fixture_input(&fx, false);
        let mut changed = false;
        let err = GgufPreload::load_clef(&input, &mut |_| {
            if !changed {
                let file = fs::OpenOptions::new().write(true).open(&fx.main).unwrap();
                file.set_len(file.metadata().unwrap().len() + 1).unwrap();
                changed = true;
            }
            Ok(())
        })
        .err()
        .expect("a source changed during residency must be rejected");
        assert!(changed, "residency callback did not run");
        assert!(
            err.reason.contains("changed during the load"),
            "{}",
            err.reason
        );
        fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn preload_equals_base_fixture_golden() {
        let work = import_work_dir("golden");
        for name in IMPORT_FIXTURE_NAMES {
            let fx = write_import_fixture(name, &work.join(name));
            let input = fixture_input(&fx, true);
            let preload = GgufPreload::load(&input).unwrap();
            assert_manifest(name, &preload.main, &golden_manifest(name, "model"));
            match fx.companion {
                Some(_) => {
                    let stem = match fx.family {
                        ImportFixtureFamily::MuseGlimmer => "draft",
                        _ => "vision",
                    };
                    assert_manifest(
                        &format!("{name}/{stem}"),
                        &preload.companion,
                        &golden_manifest(name, stem),
                    );
                }
                None => assert!(preload.companion.is_empty(), "{name}: companion"),
            }
            assert!(preload.clef_head.is_none(), "{name}: CLEF head");
            assert!(preload.ple_shards.is_none(), "{name}: PLE shards");
            assert_eq!(
                preload.identity.files.len(),
                1 + usize::from(fx.companion.is_some()),
                "{name}: identity files"
            );
        }
        fs::remove_dir_all(&work).ok();
    }

    fn stream_device(a: &MxArray) -> i32 {
        // SAFETY: the handle is a live array.
        unsafe { mlx_sys::mlx_array_stream_device_type(a.as_raw_ptr()) }
    }

    #[test]
    fn transform_cpu_ops_on_cpu_stream() {
        if !require_gpu() {
            return;
        }
        let work = import_work_dir("cpu-ops");
        let fx = write_import_fixture("qwen35-dense-untied", &work);
        let input = fixture_input(&fx, false);
        let (path, header, identity, plan) = &input.main;
        let set = GgufSourceSet::open((path, header.clone(), identity), None).unwrap();
        let read = || {
            read_gguf_tensors(&set.main, plan, RepackBackend::Auto, PageCacheMode::On)
                .unwrap()
                .map
        };
        let a_log = "model.layers.0.linear_attn.A_log";

        for native in [true, false] {
            // Only A_log: the fixture's packed projections do not fit the
            // non-native re-interleave.
            let mut weights: HashMap<String, MxArray> = remap_keys(read(), &header.metadata)
                .unwrap()
                .into_iter()
                .filter(|(k, _)| k == a_log)
                .collect();
            assert_eq!(weights.len(), 1, "fixture has no {a_log}");
            fixup_qwen35_linear_attn(&mut weights, &header.metadata, native).unwrap();
            assert_eq!(
                stream_device(&weights[a_log]),
                0,
                "A_log (native layout {native}) is not on the CPU stream"
            );
        }

        let raw = read();
        let raw_dtypes: HashMap<String, DType> = remap_keys(raw, &header.metadata)
            .unwrap()
            .iter()
            .map(|(k, a)| (k.clone(), a.dtype().unwrap()))
            .collect();
        let imported = transform_imported(read(), header, plan, &input.opts).unwrap();
        let mut converted = 0;
        for (key, a) in &imported.weights {
            if a.dtype().unwrap() != raw_dtypes[key] {
                converted += 1;
                assert_eq!(
                    stream_device(a),
                    0,
                    "dtype pass on {key} is not on the CPU stream"
                );
            }
        }
        assert!(converted > 0, "the fixture exercised no dtype conversion");
        fs::remove_dir_all(&work).ok();
    }

    #[test]
    fn preload_skips_companion_without_media() {
        let work = import_work_dir("media");
        let fx = write_import_fixture("gemma4-mmproj", &work);
        let off = GgufPreload::load(&fixture_input(&fx, false)).unwrap();
        let on = GgufPreload::load(&fixture_input(&fx, true)).unwrap();
        assert!(off.companion.is_empty(), "companion read with media off");
        assert!(!on.companion.is_empty(), "companion missing with media on");

        let input = fixture_input(&fx, false);
        let (path, header, identity, plan) = &input.main;
        let set = GgufSourceSet::open((path, header.clone(), identity), None).unwrap();
        let main_bytes = read_gguf_tensors(&set.main, plan, RepackBackend::Auto, PageCacheMode::On)
            .unwrap()
            .stats
            .bytes;
        assert_eq!(off.stats.bytes, main_bytes, "bytes read with media off");
        assert!(
            on.stats.bytes > main_bytes,
            "media on read no companion bytes"
        );
        assert_eq!(off.identity.files.len(), 2, "identity keeps the companion");
        fs::remove_dir_all(&work).ok();
    }

    #[test]
    fn preload_flip_changes_one_tensor() {
        let work = import_work_dir("flip");
        let fx = write_import_fixture("qwen35-dense-untied", &work);
        let clean = GgufPreload::load(&fixture_input(&fx, false)).unwrap();
        let mut input = fixture_input(&fx, false);
        // Sign bit of the first F32 value: low mantissa bits vanish in the bf16 pass.
        input.flip = Some(("output_norm.weight".to_string(), 3, 7));
        let flipped = GgufPreload::load(&input).unwrap();
        let (a, b) = (gate::manifest(&clean.main), gate::manifest(&flipped.main));
        let differ: Vec<&String> = a.keys().filter(|k| a[*k] != b[*k]).collect();
        assert_eq!(
            differ,
            vec!["model.norm.weight"],
            "tensors changed by the flip"
        );
        fs::remove_dir_all(&work).ok();
    }
}

/// Gate-1 harness: per-tensor manifests of a preload in the real-golden JSON
/// format (`{name: {dtype, shape, sha256}}`).
#[cfg(test)]
pub(crate) mod gate {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::utils::gguf::parse_gguf;
    use crate::utils::gguf_plan::{OutputRole, annotate_gguf_header, plan_gguf_import};
    use crate::utils::gguf_source::file_identity;
    use crate::utils::gguf_test_fixtures::sha256_hex;

    fn st_dtype(d: DType) -> &'static str {
        match d {
            DType::Float32 => "F32",
            DType::Float16 => "F16",
            DType::BFloat16 => "BF16",
            DType::Int32 => "I32",
            DType::Uint32 => "U32",
            DType::Uint8 => "U8",
            DType::Int8 => "I8",
        }
    }

    pub(crate) fn tensor_bytes(a: &MxArray) -> Vec<u8> {
        MxArray::eval_arrays_with_context(&[a], "gguf gate manifest").unwrap();
        match a.dtype().unwrap() {
            DType::Uint32 => a
                .to_uint32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            DType::Int32 => a
                .to_int32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            DType::Uint8 => a.to_uint8().unwrap(),
            DType::Int8 => a.to_int8().unwrap().into_iter().map(|v| v as u8).collect(),
            DType::Float16 | DType::BFloat16 => a
                .to_uint16_native()
                .unwrap()
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            DType::Float32 => a
                .to_float32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        }
    }

    fn manifest_entry(a: &MxArray) -> serde_json::Value {
        let shape: Vec<i64> = a.shape().unwrap().to_vec();
        serde_json::json!({
            "dtype": st_dtype(a.dtype().unwrap()),
            "shape": shape,
            "sha256": sha256_hex(&tensor_bytes(a)),
        })
    }

    pub(crate) fn manifest(map: &HashMap<String, MxArray>) -> BTreeMap<String, serde_json::Value> {
        map.iter()
            .map(|(name, a)| (name.clone(), manifest_entry(a)))
            .collect()
    }

    /// Hash a weight map without retaining its tensors after each entry.
    pub(crate) fn manifest_owned(
        map: HashMap<String, MxArray>,
    ) -> BTreeMap<String, serde_json::Value> {
        map.into_iter()
            .map(|(name, a)| (name, manifest_entry(&a)))
            .collect()
    }

    /// The input native prepare would hand the loader for `main`
    /// (+ `companion`): same plan options as its converter calls.
    pub(crate) fn preload_input(
        main: &Path,
        companion: Option<&Path>,
        family: NativeGgufFamily,
        load_companion: bool,
    ) -> Result<PreloadInput> {
        let parent = main.parent().unwrap_or(Path::new("."));
        let base = ImportOptions {
            dtype: Some(DType::BFloat16),
            import_k_quants: true,
            ..ImportOptions::default()
        };
        let main_opts = ImportOptions {
            native_qwen35_layout: matches!(
                family,
                NativeGgufFamily::Qwen35 | NativeGgufFamily::Qwen35Moe | NativeGgufFamily::Clef
            ),
            config_source_dir: parent
                .join("config.json")
                .is_file()
                .then(|| parent.to_path_buf()),
            ..base.clone()
        };
        let companion_opts = ImportOptions {
            config_source_dir: Some(parent.to_path_buf()),
            ..base
        };
        let file = |path: &Path, opts: &ImportOptions, role: OutputRole| -> Result<PreloadFile> {
            let mut header = parse_gguf(path)?;
            annotate_gguf_header(&mut header);
            let plan = plan_gguf_import(&header, path, opts, role)?;
            let meta = fs::metadata(path)
                .map_err(|e| Error::from_reason(format!("{}: {e}", path.display())))?;
            let identity = file_identity(path, &header, &meta)?;
            Ok((identity.path.clone(), Arc::new(header), identity, plan))
        };
        let companion_role = OutputRole::Companion {
            filename: match family {
                NativeGgufFamily::MuseGlimmer => "draft.safetensors",
                _ => "vision.safetensors",
            }
            .to_string(),
        };
        Ok(PreloadInput::new(
            file(main, &main_opts, OutputRole::Primary)?,
            companion
                .map(|c| file(c, &companion_opts, companion_role))
                .transpose()?,
            load_companion,
            main_opts,
        ))
    }

    fn family_from_env() -> NativeGgufFamily {
        match std::env::var("MLX_GATE_FAMILY").as_deref() {
            Err(_) | Ok("qwen3_5") => NativeGgufFamily::Qwen35,
            Ok("qwen3_5_moe") => NativeGgufFamily::Qwen35Moe,
            Ok("clef") => NativeGgufFamily::Clef,
            Ok("gemma4") => NativeGgufFamily::Gemma4,
            Ok("muse_glimmer") => NativeGgufFamily::MuseGlimmer,
            Ok(other) => panic!("unknown MLX_GATE_FAMILY '{other}'"),
        }
    }

    fn compare(label: &str, got: &BTreeMap<String, serde_json::Value>, golden: &Path) -> usize {
        let want: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&fs::read(golden).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", golden.display()));
        let mut bad = 0;
        for name in want.keys().filter(|k| !got.contains_key(*k)) {
            eprintln!("{label}: MISSING {name}");
            bad += 1;
        }
        for (name, entry) in got {
            match want.get(name) {
                None => {
                    eprintln!("{label}: EXTRA {name}");
                    bad += 1;
                }
                Some(w) if w != entry => {
                    eprintln!("{label}: DIFF {name}: got {entry} want {w}");
                    bad += 1;
                }
                Some(_) => {}
            }
        }
        eprintln!(
            "{label}: {}/{} equal",
            want.len().saturating_sub(bad.min(want.len())),
            want.len()
        );
        bad
    }

    /// `MLX_GATE_GGUF` (+ `MLX_GATE_COMPANION`), `MLX_GATE_MANIFEST`
    /// (+ `MLX_GATE_COMPANION_MANIFEST`, `MLX_GATE_JOINT_HEAD_MANIFEST`),
    /// `MLX_GATE_FAMILY` (default qwen3_5), `MLX_GATE_FLIP=name:byte:bit`.
    #[test]
    #[ignore = "reads a real GGUF; set MLX_GATE_GGUF and MLX_GATE_MANIFEST"]
    fn preload_manifest_equals_golden() {
        let gguf = std::env::var_os("MLX_GATE_GGUF").expect("set MLX_GATE_GGUF");
        let golden = std::env::var_os("MLX_GATE_MANIFEST").expect("set MLX_GATE_MANIFEST");
        let companion = std::env::var_os("MLX_GATE_COMPANION");
        let companion_golden = std::env::var_os("MLX_GATE_COMPANION_MANIFEST");
        let mut input = preload_input(
            Path::new(&gguf),
            companion.as_deref().map(Path::new),
            family_from_env(),
            companion.is_some(),
        )
        .unwrap();
        if let Ok(spec) = std::env::var("MLX_GATE_FLIP") {
            let mut parts = spec.rsplitn(3, ':');
            let bit = parts.next().unwrap().parse().unwrap();
            let byte = parts.next().unwrap().parse().unwrap();
            input.flip = Some((parts.next().unwrap().to_string(), byte, bit));
        }
        let preload = GgufPreload::load(&input).unwrap();
        let s = &preload.stats;
        eprintln!(
            "PRELOAD_S {:.3} READ_WAIT_S {:.3} BYTES {} GPU {} CPU {} NOCACHE {} RESIDENT {:?}",
            s.preload_secs,
            s.read_secs,
            s.bytes,
            s.gpu_tensors,
            s.cpu_tensors,
            s.nocache,
            s.resident
        );
        let mut bad = compare("main", &manifest(&preload.main), Path::new(&golden));
        if let Some(path) = companion_golden {
            bad += compare("companion", &manifest(&preload.companion), Path::new(&path));
        }
        if let Some(path) = std::env::var_os("MLX_GATE_JOINT_HEAD_MANIFEST") {
            let head = &preload.clef_head.as_ref().expect("no CLEF head").0;
            bad += compare("joint_head", &manifest(head), Path::new(&path));
        }
        assert_eq!(bad, 0, "tensors differ from the golden");
    }

    /// Phase 1 G1: every listed real model, Auto vs CpuOnly vs baseline golden,
    /// followed by the four source bit-flip cases on the 27B model.
    #[test]
    #[ignore = "reads every real G1 GGUF; set MLX_GATE_LIST and MLX_GATE_GOLDEN"]
    fn gate1_whole_model() {
        super::gate1::run();
    }
}

#[cfg(test)]
mod gate1;
