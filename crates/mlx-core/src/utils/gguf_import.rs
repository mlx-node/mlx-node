#![allow(dead_code)]
//! In-memory GGUF import: the file is read in aligned windows into a small
//! ring of staging buffers, and every band of rows is repacked (GPU kernels,
//! or the pure-Rust CPU fallback) straight into the final weight buffers. The
//! resulting map equals `load_gguf_tensors` with the plan's Tiled64 tensors
//! permuted like `tile_kquant_weights_for_disk`.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::array::{DType, MxArray, clear_cache};
use crate::stream::{DeviceType, Stream};
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufTensorInfo, GgufTensorType, load_gguf_tensor,
};
use crate::utils::gguf_cpu_repack::{
    copy_dense_band_into, cpu_kquant_band_into, decode_affine_band_into,
};
use crate::utils::gguf_gpu::{
    GpuRepackKind, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available, gpu_repack_band,
};
use crate::utils::gguf_plan::ImportPlan;
use crate::utils::gguf_read::{
    ALIGN, CHUNK_BYTES, PageCacheMode, READ_THREADS, ReadAt, Reader, SLOT_BYTES, SLOTS, Window,
    log_load_start, maybe_log_slow_disk, plan_reads, sampled_resident_fraction, set_nocache,
};
use crate::utils::gguf_source::GgufFileHandle;
use crate::utils::host::physical_memory;

pub(crate) const CPU_THREADS: usize = 8;
/// Staged source bytes the CPU fallback may have queued or running.
pub(crate) const CPU_INFLIGHT_BYTES: usize = 256 << 20;
/// Source bytes of one CPU job.
pub(crate) const CPU_BAND_BYTES: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepackBackend {
    Auto,
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
    probe: &'p ImportProbe,
}

impl Drop for ImportGuard<'_> {
    fn drop(&mut self) {
        if self.gpu {
            Stream::default(DeviceType::Gpu).synchronize();
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
            let (dst, ptrs) = RepackDst::alloc(t, layout)?;
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
        run.all()
    })?;

    let mut map = HashMap::new();
    let mut ple_shards = None;
    for (t, state) in header.tensors.iter().zip(std::mem::take(&mut guard.states)) {
        match state.dst {
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
                ple_shards = Some((
                    t.name.clone(),
                    shards.into_iter().map(|(a, _)| a).collect(),
                    rows_per_shard,
                ));
            }
            Dst::Done(arrays) => map.extend(arrays),
            Dst::Empty | Dst::Whole(_) | Dst::Repack { dst: None, .. } => {
                return Err(Error::from_reason(format!(
                    "GGUF tensor '{}' was not fully read",
                    t.name
                )));
            }
        }
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
    fn all(&mut self) -> Result<Timing> {
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
        Ok(timing)
    }

    /// Route every band of window `w`; returns the GPU arrays it encoded.
    fn window(&mut self, w: usize) -> Result<Vec<MxArray>> {
        let (windows, header, plan) = (self.windows, self.header, self.plan);
        let slot = w % self.guard.slots.len();
        let (staging, base) = self.guard.slots[slot].clone();
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
    use GgufTensorType::*;

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

    const GPU_SPECS: usize = 9;
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
}
