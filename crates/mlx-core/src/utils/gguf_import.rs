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
